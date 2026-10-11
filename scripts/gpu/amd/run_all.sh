#!/usr/bin/env bash
# Everything for a rented AMD GPU machine in one command: set up, record the
# hardware, run every check, benchmark, package the results and print what to
# download. Steps that fail are logged and skipped, so one problem doesn't
# waste the session. Safe to rerun; setup skips what is already installed.
#
#   git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
#   bash scripts/gpu/amd/run_all.sh            # add --dry-run to only print commands
#
# Needs Linux with ROCm and internet. Expect about 20-30 minutes with 8 GPUs,
# most of it the PyTorch download (skip it with TORCH_FLAVOR=system on an image
# that already has a ROCm PyTorch). See docs/GPU_VALIDATION_RUNBOOK.md.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

# ---- Settings (override with environment variables) ---------------------------
TORCH_FLAVOR="${TORCH_FLAVOR:-rocm}"     # rocm: torch 2.14.1+rocm7.2 (6+ GB); system: reuse
export TORCH_VERSION="${TORCH_VERSION:-2.14.1}" ROCM_INDEX="${ROCM_INDEX:-rocm7.2}"
WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"
TARGET="${TARGET:-amd-$(hostname -s)-$(date +%Y%m%d-%H%M)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
BENCH_TRIALS="${BENCH_TRIALS:-10}"
SCALING_MAX_GPUS="${SCALING_MAX_GPUS:-4}"
SKIP_SETUP="${SKIP_SETUP:-0}"            # 1 = reuse an existing venv (e.g. on a rerun)
# Opt-in collective policies also checked (space-separated).
OPT_IN_POLICIES="${OPT_IN_POLICIES:-pipelined-reduce-bridge-broadcast sharded-bridge}"
# -------------------------------------------------------------------------------

RESULTS="$RESULTS_ROOT/$TARGET"
export RESULTS
VENV="${VENV:-$WORK_DIR/venv}"
log "results go to $RESULTS"

if [ "$SKIP_SETUP" != 1 ]; then
    step setup "$KIT_DIR/setup_env.sh" --flavor "$TORCH_FLAVOR" --venv "$VENV"
fi
activate_venv "$VENV"
capture "$RESULTS/rocm-smi.txt" rocm-smi
capture "$RESULTS/rocm-smi-detail.txt" rocm-smi --showproductname --showdriverversion \
    --showbus --showmeminfo vram
capture "$RESULTS/amd-smi-static.txt" amd-smi static
capture "$RESULTS/rocminfo.txt" rocminfo

GPUS=$(gpu_count)
log "$GPUS GPU(s) visible"
step probe python "$KIT_DIR/probe.py" --out "$RESULTS/probe"
step device_map python "$KIT_DIR/device_map.py" --out "$RESULTS/device_map"
# Item 12: one more process than GPUs. The extra rank's LOCAL_RANK has no GPU,
# so init() must fail on every rank, naming it. Real GPUs only: simulation
# has no devices to check.
if [ -z "${GPUBRIDGE_VENDOR:-}${GPUBRIDGE_CPU_ONLY:-}" ]; then
    step local_rank_check local_torchrun --nproc-per-node=$((GPUS + 1)) "$KIT_DIR/check.py" \
        --out "$RESULTS/local_rank_check" --expect-init-error "LOCAL_RANK=$GPUS asks for GPU $GPUS"
fi

# At NCCL_DEBUG=INFO (common.sh), RCCL logs its version to $RESULTS/nccl/check_1gpu/:
# proof that "nccl" runs RCCL (item 4).
step check_1gpu local_torchrun --nproc-per-node=1 "$KIT_DIR/check.py" \
    --out "$RESULTS/check_1gpu"
rccl_version() {
    run grep -h -m 1 -i -o "RCCL version[^,]*" "$RESULTS/nccl/check_1gpu/"*.log
}
step rccl_version rccl_version

checks=("$RESULTS/check_1gpu")
if [ "$GPUS" -ge 2 ]; then
    step check_island local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/check.py" \
        --out "$RESULTS/check_island"
    step check_split env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_split"
    checks+=("$RESULTS/check_island" "$RESULTS/check_split")
    step bench env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" --ops native,island,gpubridge \
        --phases --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS"
    # Opt-in policies (items 17 to 21), each under the same split-test check, then
    # every candidate timed side by side, writing a thresholds file for auto-tuned.
    for policy in $OPT_IN_POLICIES; do
        step "check_$policy" env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
            --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_$policy" \
            --policy "$policy"
        checks+=("$RESULTS/check_$policy")
    done
    step bench_candidates env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench_candidates" --ops gpubridge \
        --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS" \
        --write-thresholds "$RESULTS/thresholds.json"
    # DDP with gpubridge.ddp_comm_hook across two islands (item 22).
    step train_ddp_split env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$REPO_DIR/examples/train_synthetic.py" --json "$RESULTS/train_ddp_split.json"
    max=$((GPUS < SCALING_MAX_GPUS ? GPUS : SCALING_MAX_GPUS))
    for n in $(seq 1 "$max"); do
        step "train_n$n" local_torchrun --nproc-per-node="$n" \
            "$REPO_DIR/examples/train_synthetic.py" --json "$RESULTS/scaling/train-n$n.json"
    done
    step summarize_scaling python "$KIT_DIR/summarize.py" "$RESULTS/scaling"
else
    # One GPU: two ranks share it, each its own single-rank RCCL island.
    step check_split_shared_gpu env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node=2 "$KIT_DIR/check.py" --out "$RESULTS/check_split_shared_gpu"
    checks+=("$RESULTS/check_split_shared_gpu")
fi
step summarize python "$KIT_DIR/summarize.py" "${checks[@]}"

package_results "$RESULTS"
finish
