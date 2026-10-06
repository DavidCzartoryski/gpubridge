#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-03-single-island
#SBATCH --time=00:20:00
# Step 03, 2+ GPUs on 1 node (multigpu). Submit with: scripts/gpu/explorer/submit.sh 03
# One real NCCL island, no bridge: the native baseline. Also checks that each
# rank drives its own GPU (item 5 on NVIDIA) and benchmarks gpubridge against
# plain NCCL in a single island.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

GPUS="${GPUS_SINGLE_ISLAND:-2}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"

job_setup 03_single_island
step device_map python "$KIT_DIR/device_map.py" --out "$RESULTS"
step check local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/check.py" \
    --out "$RESULTS/check"
step summarize python "$KIT_DIR/summarize.py" "$RESULTS/check"
step bench local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench" --max-bytes "$BENCH_MAX_BYTES"
finish
