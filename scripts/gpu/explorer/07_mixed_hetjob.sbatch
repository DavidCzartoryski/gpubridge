#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-07-mixed-hetjob
#SBATCH --time=00:30:00
# Step 07, a heterogeneous job: component 0 on the gpu partition (NVIDIA, CUDA
# venv), component 1 on sharing (AMD, ROCm venv from setup-rocm). Submit with:
# scripts/gpu/explorer/submit.sh 07, which stays dry-run only until step 00
# confirms AMD GPUs in sharing.
# A REAL mixed-vendor job inside Explorer: NCCL on one node, RCCL on the other,
# joined by the Gloo bridge over the cluster network. Items 1 and 2 on GPUs,
# 3 with real RCCL ranks, 8 between nodes. The benchmark times each island's
# native all_reduce (NCCL, RCCL) in the same job as the bridged one (item 11).
# STEPS_07=training runs only the probe, the preflight and the training steps;
# STEPS_07=bench only the probe, the preflight and train_bench.py's steps.
# TRAIN_BENCH_BUCKETS="10 25 50 100" runs each train_bench step once per DDP bucket
# size (MB), each under its own name: train_bench_<model>_bucket<MB>.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

GPUS_NVIDIA="${GPUS_MIXED_NVIDIA:-1}"
GPUS_AMD="${GPUS_MIXED_AMD:-1}"
RDZV_PORT="${RDZV_PORT:-29500}"
PREFLIGHT_PORT="${PREFLIGHT_PORT:-29501}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
STEPS_07="${STEPS_07:-all}"  # all, training, or bench
case $STEPS_07 in all | training | bench) ;; *) die "STEPS_07 must be all, training or bench" ;; esac
TRAIN_BENCH_MODELS="${TRAIN_BENCH_MODELS:-gpt2-small resnet50}"
read -ra train_args <<<"${TRAIN_BENCH_ARGS:-}"
TRAIN_BENCH_BUCKETS="${TRAIN_BENCH_BUCKETS:-}"  # DDP bucket sizes (MB); empty: train_bench.py's
for mb in $TRAIN_BENCH_BUCKETS; do
    [[ $mb =~ ^[0-9]+([.][0-9]+)?$ ]] || die "TRAIN_BENCH_BUCKETS: '$mb' is not a size in MB"
done
if [ -n "$TRAIN_BENCH_BUCKETS" ] && [[ " ${train_args[*]-} " == *" --bucket-mb"* ]]; then
    die "set the bucket size in TRAIN_BENCH_BUCKETS or in TRAIN_BENCH_ARGS, not both"
fi
# Interfaces: empty lets NCCL choose, and Gloo on each node use the interface
# the preflight picks there (names may differ between the two partitions).
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && export GLOO_SOCKET_IFNAME
IFNAME_NVIDIA="" IFNAME_AMD=""  # filled in after the preflight

if [ "$DRY_RUN" = 1 ]; then
    HEAD="<gpu-node>"
else
    [ -n "${SLURM_JOB_NODELIST_HET_GROUP_1:-}" ] ||
        die "not a heterogeneous job; submit with scripts/gpu/explorer/submit.sh 07"
    HEAD=$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0:-}" | head -n 1)
fi

# fill ARG VENV ROLE SIDE NPROC: ARG with the placeholders replaced.
fill() {
    local arg=$1
    arg=${arg//@VENV@/$2}
    arg=${arg//@ROLE@/$3}
    arg=${arg//@SIDE@/$4}
    arg=${arg//@NPROC@/$5}
    printf '%s' "$arg"
}

# both ARGS...: one heterogeneous job step running ARGS on each component, in its
# own venv. @VENV@, @ROLE@, @SIDE@ and @NPROC@ become that component's venv,
# master|worker, nvidia|amd and GPU count. --kill-on-bad-exit stops one side
# when the other fails, so neither is left waiting for a rendezvous.
both() {
    local nvidia=() amd=() arg
    if [ -n "$IFNAME_NVIDIA" ]; then nvidia=(env "GLOO_SOCKET_IFNAME=$IFNAME_NVIDIA"); fi
    if [ -n "$IFNAME_AMD" ]; then amd=(env "GLOO_SOCKET_IFNAME=$IFNAME_AMD"); fi
    for arg in "$@"; do
        nvidia+=("$(fill "$arg" "$VENV" master nvidia "$GPUS_NVIDIA")")
        amd+=("$(fill "$arg" "$VENV_ROCM" worker amd "$GPUS_AMD")")
    done
    limited srun --het-group=0 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 "${nvidia[@]}" \
        : --het-group=1 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 "${amd[@]}"
}

# One torchrun per component; c10d rendezvous on the gpu node.
mixed_torchrun() {
    local tag=$1
    shift
    both @VENV@/bin/torchrun --nnodes=2 --nproc-per-node=@NPROC@ --rdzv-backend=c10d \
        --rdzv-endpoint="$HEAD:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-local}-$tag" "$@"
}

job_setup 07_mixed_hetjob   # on the gpu node, with the CUDA venv
# The AMD side's ROCm wheel needs its GPU-name table pointed out (common.sh); srun
# passes the variable to both components.
rocm_asic_ids "$VENV_ROCM/bin/python"
capture "$RESULTS/rocm-smi.txt" srun --het-group=1 --ntasks=1 rocm-smi
capture "$RESULTS/amd-smi-static.txt" srun --het-group=1 --ntasks=1 amd-smi static
step probe both @VENV@/bin/python "$KIT_DIR/probe.py" --out "$RESULTS/probe_@SIDE@"
step preflight both @VENV@/bin/python "$KIT_DIR/preflight.py" --role @ROLE@ \
    --master-addr "$HEAD" --port "$PREFLIGHT_PORT" --bandwidth-seconds "$PREFLIGHT_SECONDS" \
    --out "$RESULTS/preflight"
# Nothing below can pass without a usable GPU on each side and a network between
# them, so stop here instead of waiting out rendezvous timeouts. In simulation
# (the CPU test of this script) there is no GPU to find, so only the preflight counts.
blocking=()
for name in "${FAILED_STEPS[@]+"${FAILED_STEPS[@]}"}"; do
    [ "$name" = probe ] && [ -n "${GPUBRIDGE_VENDOR:-}${GPUBRIDGE_CPU_ONLY:-}" ] && continue
    blocking+=("$name")
done
if [ "${#blocking[@]}" -gt 0 ]; then
    finish || true
    die "stopping before torchrun: ${blocking[*]} failed; logs in $RESULTS"
fi
# Gloo on each node uses the interface the preflight chose there.
if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ "$USE_PREFLIGHT_IFNAME" = 1 ]; then
    IFNAME_NVIDIA=$(preflight_ifname "$RESULTS/preflight/preflight-master.json")
    IFNAME_AMD=$(preflight_ifname "$RESULTS/preflight/preflight-worker.json")
    log "GLOO_SOCKET_IFNAME: ${IFNAME_NVIDIA:-unset} on the gpu node," \
        "${IFNAME_AMD:-unset} on the AMD node, as the preflight chose"
fi
# Both components write into one directory on the shared filesystem, so the
# summary below covers every rank of the mixed job.
if [ "$STEPS_07" = all ]; then
    step check mixed_torchrun check "$KIT_DIR/check.py" --out "$RESULTS/check"
    step summarize python "$KIT_DIR/summarize.py" "$RESULTS/check"
    # Point-to-point between the NVIDIA and AMD ranks through backend="gpubridge" (item 32).
    step check_p2p mixed_torchrun check-p2p "$KIT_DIR/check.py" --out "$RESULTS/check_p2p" \
        --p2p-only
    step summarize_p2p python "$KIT_DIR/summarize.py" "$RESULTS/check_p2p"
    # Point-to-point latency and bandwidth between the V100 and the MI50 (bench_p2p.py);
    # again with pinned staging below.
    step bench_p2p mixed_torchrun bench-p2p "$KIT_DIR/bench_p2p.py" --out "$RESULTS/bench_p2p" \
        --max-bytes "$BENCH_MAX_BYTES"
    step bench mixed_torchrun bench "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" \
        --ops island,gpubridge,flat-gloo --phases --max-bytes "$BENCH_MAX_BYTES"
    # Pinned staging for the default bridge (item 25) on real NVIDIA + AMD leaders.
    export GPUBRIDGE_PINNED_STAGING=1
    step check_pinned mixed_torchrun check-pinned "$KIT_DIR/check.py" --out "$RESULTS/check_pinned"
    step summarize_pinned python "$KIT_DIR/summarize.py" "$RESULTS/check_pinned"
    step bench_pinned mixed_torchrun bench-pinned "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_pinned" --ops gpubridge --phases --max-bytes "$BENCH_MAX_BYTES"
    step bench_p2p_pinned mixed_torchrun bench-p2p-pinned "$KIT_DIR/bench_p2p.py" \
        --out "$RESULTS/bench_p2p_pinned" --max-bytes "$BENCH_MAX_BYTES"
    unset GPUBRIDGE_PINNED_STAGING
    # Bridge compression between real NVIDIA and AMD leaders (items 28 and 29). The kernel
    # benchmark runs on each side's GPU; the digests in the two summaries must match.
    step bench_compression_kernels both @VENV@/bin/python "$KIT_DIR/bench_compression.py" \
        --out "$RESULTS/bench_compression_@SIDE@" --max-bytes "$BENCH_MAX_BYTES"
    step bench_compressed mixed_torchrun bench-compressed "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_compressed" --ops gpubridge --compression none,bf16,fp8 --phases \
        --max-bytes "$BENCH_MAX_BYTES"
    # The UCX bridge between real NVIDIA and AMD leaders (items 26 and 27), opt-in
    # with UCX_BRIDGE=1. Each side builds it against its own UCX (CUDA- or ROCm-built).
    if [ "$UCX_BRIDGE" = 1 ]; then
        step ucx_build both @VENV@/bin/python -m gpubridge.ucx build
        step check_ucx mixed_torchrun check-ucx "$KIT_DIR/check.py" --out "$RESULTS/check_ucx" \
            --bridge ucx
        step summarize_ucx python "$KIT_DIR/summarize.py" "$RESULTS/check_ucx"
        step bench_ucx mixed_torchrun bench-ucx "$KIT_DIR/bench_all_reduce.py" \
            --out "$RESULTS/bench_ucx" --ops gpubridge --phases --max-bytes "$BENCH_MAX_BYTES" \
            --bridge ucx
    fi
    # Opt-in policies on the real mixed job (items 17 to 21).
    for policy in $OPT_IN_POLICIES; do
        step "check_$policy" mixed_torchrun "check-$policy" "$KIT_DIR/check.py" \
            --out "$RESULTS/check_$policy" --policy "$policy"
        step "summarize_$policy" python "$KIT_DIR/summarize.py" "$RESULTS/check_$policy"
    done
    step bench_candidates mixed_torchrun candidates "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_candidates" --ops gpubridge --max-bytes "$BENCH_MAX_BYTES" \
        --write-thresholds "$RESULTS/thresholds.json"
fi
# DDP with gpubridge.ddp_comm_hook on real NVIDIA + AMD ranks (item 22).
if [ "$STEPS_07" != bench ]; then
    step train_ddp mixed_torchrun train "$REPO_DIR/examples/train_synthetic.py" \
        --json "$RESULTS/train_ddp.json"
fi
# train_bench MODEL [MB]: one train_bench.py run on both sides; with MB, at that DDP
# bucket size, under its own step and JSON name.
train_bench() {
    local name="train_bench_$1${2:+_bucket$2}"
    step "$name" mixed_torchrun "${name//_/-}" "$KIT_DIR/train_bench.py" --model "$1" \
        --json "$RESULTS/$name.json" "${train_args[@]+"${train_args[@]}"}" ${2:+--bucket-mb "$2"}
}
# Training throughput through the same hook on real models (train_bench.py): samples/s,
# and the step split into compute, waiting for the slower GPU and exposed
# communication. Set against train_bench.sbatch's runs of each GPU alone.
for model in $TRAIN_BENCH_MODELS; do
    if [ -z "$TRAIN_BENCH_BUCKETS" ]; then
        train_bench "$model"
    fi
    for mb in $TRAIN_BENCH_BUCKETS; do
        train_bench "$model" "$mb"
    done
done
if [ "$STEPS_07" = bench ]; then  # no FSDP2 or HSDP
    finish
    exit
fi
# gpubridge as the torch.distributed backend (item 30): a plain FSDP2 script apart from
# backend="gpubridge" and `import gpubridge`. Exits non-zero if ranks disagree.
step train_fsdp2_backend mixed_torchrun fsdp2 "$REPO_DIR/examples/fsdp2_torch_backend.py" \
    --json "$RESULTS/train_fsdp2_backend.json"
# The same on exact data, against a single-process replay, bit for bit.
step verify_fsdp2_backend mixed_torchrun fsdp2-verify "$REPO_DIR/examples/fsdp2_torch_backend.py" \
    --verify
# HSDP through the backend (item 31): shard within each vendor (NVIDIA ranks first, as
# the hetjob orders them), replicate across vendors over the bridge. Needs the same GPU
# count on both sides. Its JSON has the median step time, to set against
# train_fsdp2_backend's; then bit for bit on exact data.
if [ "$GPUS_NVIDIA" = "$GPUS_AMD" ]; then
    step train_hsdp_backend mixed_torchrun hsdp "$REPO_DIR/examples/fsdp2_torch_backend.py" \
        --hsdp 2 --json "$RESULTS/train_hsdp_backend.json"
    step verify_hsdp_backend mixed_torchrun hsdp-verify \
        "$REPO_DIR/examples/fsdp2_torch_backend.py" --verify --hsdp 2
else
    log "skipping the HSDP steps: they need as many NVIDIA GPUs as AMD ($GPUS_NVIDIA vs $GPUS_AMD)"
fi
finish
