#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-01-probe
#SBATCH --time=00:10:00
# Step 01, 1 GPU (free gpu partition). Submit with: scripts/gpu/explorer/submit.sh 01
# Records nvidia-smi, gpubridge.probe() and whether this PyTorch build has
# kernels for the GPU. Proves the environment before spending time on anything else.
# Then checks that a rank whose LOCAL_RANK has no GPU fails init() on every rank.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

job_setup 01_probe
step probe python "$KIT_DIR/probe.py" --out "$RESULTS"
# Item 12: two processes on one GPU, without split-test mode. Rank 1's
# LOCAL_RANK has no GPU, so init() must fail on BOTH ranks, naming it.
step local_rank_check local_torchrun --nproc-per-node=2 "$KIT_DIR/check.py" \
    --out "$RESULTS/local_rank_check" --expect-init-error "LOCAL_RANK=1 asks for GPU 1"
step summarize python "$KIT_DIR/summarize.py" "$RESULTS/local_rank_check"
finish
