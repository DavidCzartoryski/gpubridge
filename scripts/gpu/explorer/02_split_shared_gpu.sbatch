#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-02-split-shared-gpu
#SBATCH --time=00:15:00
# Step 02, 1 GPU (free gpu partition). Submit with: scripts/gpu/explorer/submit.sh 02
# TEST ONLY split: 2 processes share one GPU, each its own single-rank NCCL island,
# joined by the Gloo bridge. Covers new_group(backend="nccl") on non-member ranks
# (item 3), device tensors through the bridge (item 6) and barrier on CUDA (item 10).
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

PROCS=2   # ranks sharing the GPU

job_setup 02_split_shared_gpu
export GPUBRIDGE_SPLIT_TEST=half
step check local_torchrun --nproc-per-node="$PROCS" "$KIT_DIR/check.py" \
    --out "$RESULTS/check" --large-numel $((16 * 1024 * 1024))
step summarize python "$KIT_DIR/summarize.py" "$RESULTS/check"
finish
