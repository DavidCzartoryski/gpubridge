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
# Interfaces: empty lets NCCL/Gloo choose. Names may differ between the two
# partitions' nodes; set them only if the preflight's interface check says so.
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && export GLOO_SOCKET_IFNAME

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
    for arg in "$@"; do
        nvidia+=("$(fill "$arg" "$VENV" master nvidia "$GPUS_NVIDIA")")
        amd+=("$(fill "$arg" "$VENV_ROCM" worker amd "$GPUS_AMD")")
    done
    run srun --het-group=0 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 "${nvidia[@]}" \
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
capture "$RESULTS/rocm-smi.txt" srun --het-group=1 --ntasks=1 rocm-smi
capture "$RESULTS/amd-smi-static.txt" srun --het-group=1 --ntasks=1 amd-smi static
step probe both @VENV@/bin/python "$KIT_DIR/probe.py" --out "$RESULTS/probe_@SIDE@"
step preflight both @VENV@/bin/python "$KIT_DIR/preflight.py" --role @ROLE@ \
    --master-addr "$HEAD" --port "$PREFLIGHT_PORT" --out "$RESULTS/preflight"
# Nothing below can pass without a usable GPU on each side and a network between
# them, so stop here instead of waiting out rendezvous timeouts.
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    finish || true
    die "stopping before torchrun: ${FAILED_STEPS[*]} failed; logs in $RESULTS"
fi
# Both components write into one directory on the shared filesystem, so the
# summary below covers every rank of the mixed job.
step check mixed_torchrun check "$KIT_DIR/check.py" --out "$RESULTS/check"
step summarize python "$KIT_DIR/summarize.py" "$RESULTS/check"
step bench mixed_torchrun bench "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" \
    --ops island,gpubridge --phases --max-bytes "$BENCH_MAX_BYTES"
finish
