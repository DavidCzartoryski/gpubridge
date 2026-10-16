#!/usr/bin/env bash
# One machine's half of a REAL mixed NVIDIA + AMD job. Run it on both machines
# at about the same time; either machine can be the master:
#
#   NVIDIA machine:  ROLE=master MASTER_ADDR=100.64.0.1 scripts/gpu/mixed/node.sh
#   AMD machine:     ROLE=worker MASTER_ADDR=100.64.0.1 scripts/gpu/mixed/node.sh
#
# MASTER_ADDR is the master's address on a flat network between the machines:
# every TCP port open both ways. Tailscale is the default way to get one,
# WireGuard the alternative; see docs/GPU_VALIDATION_RUNBOOK.md. A preflight
# check runs first and stops with a clear message if the machines can't talk,
# before torchrun starts. Then the bridge check, the benchmarks, training through
# the backend, training throughput, and data-parallel and pipeline training on a
# LLaMA-style model; last, this machine's results are triaged (triage.txt) and
# packed. Add --dry-run to only print commands.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

# ---- Settings (override with environment variables) ---------------------------
ROLE="${ROLE:-}"                         # master | worker
MASTER_ADDR="${MASTER_ADDR:-}"           # master's flat-network IP (`tailscale ip -4`)
MASTER_PORT="${MASTER_PORT:-29500}"      # torchrun rendezvous
PREFLIGHT_PORT="${PREFLIGHT_PORT:-29501}"  # the preflight's own check
IFNAME="${IFNAME:-tailscale0}"           # flat-network interface: tailscale0 or wg0
LOCAL_ADDR="${LOCAL_ADDR:-}"             # this machine's IP on it; default: read from IFNAME
NPROC="${NPROC:-}"                       # GPUs to use here; default: all visible
RUN_ID="${RUN_ID:-gpubridge-mixed}"      # must be the same on both machines
TORCH_FLAVOR="${TORCH_FLAVOR:-}"         # cuda | rocm | system; default: from the hardware
export TORCH_VERSION="${TORCH_VERSION:-2.14.1}"  # same release on both machines
SKIP_SETUP="${SKIP_SETUP:-0}"
WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-256M}"  # the bridge crosses the internet: keep it modest
TRAIN_BENCH_MODELS="${TRAIN_BENCH_MODELS-gpt2-small resnet50}"  # train_bench.py's; "" = none
TRAIN_BENCH_ARGS="${TRAIN_BENCH_ARGS:-}"  # extra train_bench.py arguments, e.g. "--steps 50"
# Data-parallel and pipeline training (train_dp.py, train_pipeline.py), as in Explorer's
# step 07 with STEPS_07=dp and pipeline. Every GPU gets one rank; the pipeline puts one
# stage on each, so it needs at least as many microbatches and blocks as GPUs in all.
DP_MODEL="${DP_MODEL:-llama-125m}"       # llama.py's model, for both
DP_STEPS="${DP_STEPS:-500}"              # 1000 to compare loss curves (docs/COMPRESSION.md)
DP_GLOBAL_BATCH="${DP_GLOBAL_BATCH:-8}"  # sequences per step, over every GPU
DP_SEQ="${DP_SEQ:-512}"                  # tokens per sequence
DP_MICRO="${DP_MICRO:-1}"                # the one-GPU baseline's micro-batches
DP_RUNS="${DP_RUNS-ddp ddp-bf16 ddp-fp8}"  # step 07's run names; "" = none
DP_KERNELS_ON_AMD="${DP_KERNELS_ON_AMD:-auto}"  # torch where Triton can't compile them
DP_ARGS="${DP_ARGS:-}"                   # extra train_dp.py arguments
PP_RUNS="${PP_RUNS-gpipe 1f1b}"          # schedules; "" = none
PP_MICROBATCHES="${PP_MICROBATCHES:-4}"
PP_ARGS="${PP_ARGS:-}"                   # extra train_pipeline.py arguments, e.g. "--split 7,5"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
# -------------------------------------------------------------------------------

case $ROLE in
master) IS_HOST=1 ;;
worker) IS_HOST=0 ;;
*) die "set ROLE=master or ROLE=worker" ;;
esac
[ -n "$MASTER_ADDR" ] || die "set MASTER_ADDR to the master's flat-network IP"
if [ -z "$TORCH_FLAVOR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then TORCH_FLAVOR=cuda; else TORCH_FLAVOR=rocm; fi
fi
TARGET="${TARGET:-mixed-$ROLE-$(hostname -s)}"
RESULTS="$RESULTS_ROOT/$TARGET"
export RESULTS
VENV="${VENV:-$WORK_DIR/venv}"

if [ "$SKIP_SETUP" != 1 ]; then
    step setup "$KIT_DIR/setup_env.sh" --flavor "$TORCH_FLAVOR" --venv "$VENV"
fi
activate_venv "$VENV"
capture "$RESULTS/nvidia-smi.txt" nvidia-smi
capture "$RESULTS/rocm-smi.txt" rocm-smi

# Gloo (the bridge and the world group) must use the flat network. NCCL/RCCL
# only run inside each machine's island, so they keep their own defaults.
export GLOO_SOCKET_IFNAME="$IFNAME"
if [ -z "$LOCAL_ADDR" ]; then
    if [ "$DRY_RUN" = 1 ]; then
        LOCAL_ADDR="<$IFNAME-address>"
    elif command -v ip >/dev/null 2>&1; then
        LOCAL_ADDR=$(ip -4 -o addr show dev "$IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
    fi
    [ -n "$LOCAL_ADDR" ] || die "no IPv4 address on $IFNAME; is Tailscale/WireGuard up? Set LOCAL_ADDR."
fi
NPROC="${NPROC:-$(gpu_count)}"
log "$ROLE: $NPROC GPU(s), $TORCH_FLAVOR build, $LOCAL_ADDR on $IFNAME, master $MASTER_ADDR"

# Preflight is not a skippable step: if the machines can't talk, stop here.
if ! run python "$KIT_DIR/preflight.py" --role "$ROLE" --master-addr "$MASTER_ADDR" \
    --port "$PREFLIGHT_PORT" --ifname "$IFNAME" --out "$RESULTS"; then
    die "preflight failed; fix connectivity before starting torchrun (see message above)"
fi

mixed_torchrun() {
    local tag=$1
    shift
    limited torchrun --nnodes=2 --nproc-per-node="$NPROC" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv-id="$RUN_ID-$tag" \
        --rdzv-conf="is_host=$IS_HOST" --local-addr="$LOCAL_ADDR" "$@"
}

step check mixed_torchrun check "$KIT_DIR/check.py" --out "$RESULTS/check"
step summarize python "$KIT_DIR/summarize.py" --partial "$RESULTS/check"
step bench mixed_torchrun bench "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" \
    --max-bytes "$BENCH_MAX_BYTES" --phases
# Bridge compression: this machine's kernels (compare the digests with the other
# machine's summary), then the bridge step with and without compression.
step bench_compression_kernels limited python "$KIT_DIR/bench_compression.py" \
    --out "$RESULTS/bench_compression" --max-bytes "$BENCH_MAX_BYTES"
step bench_compressed mixed_torchrun bench-compressed "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_compressed" --max-bytes "$BENCH_MAX_BYTES" --phases \
    --compression none,bf16,fp8
# gpubridge as the torch.distributed backend (item 30): a plain FSDP2 script apart from
# backend="gpubridge" and `import gpubridge`.
step train_fsdp2_backend mixed_torchrun fsdp2 "$REPO_DIR/examples/fsdp2_torch_backend.py" \
    --json "$RESULTS/train_fsdp2_backend.json"
step verify_fsdp2_backend mixed_torchrun fsdp2-verify "$REPO_DIR/examples/fsdp2_torch_backend.py" \
    --verify
# Training throughput (train_bench.py, as in Explorer's step 07): DDP with
# gpubridge.ddp_comm_hook on each model, first on this machine's GPUs alone (one
# island, nothing crosses the bridge: the baseline), then on both machines' together.
# Both machines run their baseline at the same time; the faster one waits at the next
# rendezvous.
read -ra train_args <<<"$TRAIN_BENCH_ARGS"
for model in $TRAIN_BENCH_MODELS; do
    step "train_bench_alone_$model" local_torchrun --nproc-per-node="$NPROC" \
        "$KIT_DIR/train_bench.py" --model "$model" \
        --json "$RESULTS/train_bench_alone_$model.json" "${train_args[@]+"${train_args[@]}"}"
    step "train_bench_$model" mixed_torchrun "train-bench-$model" "$KIT_DIR/train_bench.py" \
        --model "$model" --json "$RESULTS/train_bench_$model.json" \
        "${train_args[@]+"${train_args[@]}"}"
done
# Data-parallel and pipeline training on llama.py's model: first one GPU of this machine
# alone with the same global batch and seed (the baseline both are set against), then both
# machines together.
read -ra dp_args <<<"$DP_ARGS"
read -ra pp_args <<<"$PP_ARGS"
dp_common=(--model "$DP_MODEL" --steps "$DP_STEPS" --global-batch "$DP_GLOBAL_BATCH"
           --seq-len "$DP_SEQ")
if [ -n "$DP_RUNS$PP_RUNS" ]; then
    step dp_single limited python "$KIT_DIR/train_dp.py" --mode single "${dp_common[@]}" \
        --micro "$DP_MICRO" --json "$RESULTS/dp_single.json" "${dp_args[@]+"${dp_args[@]}"}"
fi
for run in $DP_RUNS; do
    dp_run_options "$run"
    GPUBRIDGE_BRIDGE_COMPRESSION=$DP_COMPRESSION step "dp_$run" mixed_torchrun "dp-$run" \
        "$KIT_DIR/train_dp.py" --mode "$DP_MODE" "${dp_common[@]}" \
        --kernels-on-amd "$DP_KERNELS_ON_AMD" --json "$RESULTS/dp_$run.json" \
        "${DP_RUN_ARGS[@]+"${DP_RUN_ARGS[@]}"}" "${dp_args[@]+"${dp_args[@]}"}"
done
for schedule in $PP_RUNS; do
    case $schedule in gpipe | 1f1b) ;; *) die "PP_RUNS: unknown schedule '$schedule'" ;; esac
    step "pp_$schedule" mixed_torchrun "pp-$schedule" "$KIT_DIR/train_pipeline.py" \
        "${dp_common[@]}" --schedule "$schedule" --microbatches "$PP_MICROBATCHES" \
        --json "$RESULTS/pp_$schedule.json" "${pp_args[@]+"${pp_args[@]}"}"
done
# This machine's half only: triage the other machine's tarball the same way.
capture "$RESULTS/triage.txt" python "$KIT_DIR/triage.py" "$RESULTS"
package_results "$RESULTS"
finish
