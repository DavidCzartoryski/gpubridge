#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-01-probe
#SBATCH --time=00:10:00
# Step 01, 1 GPU (free gpu partition). Submit with: scripts/gpu/explorer/submit.sh 01
# Records nvidia-smi, gpubridge.probe() and whether this PyTorch build has
# kernels for the GPU. Proves the environment before spending time on anything else.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

job_setup 01_probe
step probe python "$KIT_DIR/probe.py" --out "$RESULTS"
finish
