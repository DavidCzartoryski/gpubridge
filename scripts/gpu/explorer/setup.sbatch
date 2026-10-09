#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-setup
#SBATCH --time=01:00:00
# Setup on a compute node: installs never run on a login node. Submit with
#   scripts/gpu/explorer/submit.sh setup        CUDA venv, in a short CPU job
#   scripts/gpu/explorer/submit.sh setup-rocm   ROCm venv, in a job on an AMD node
# or run it inside an interactive job (see docs/GPU_VALIDATION_RUNBOOK.md):
#   srun -p short -N 1 -n 1 -c 4 --mem=16G -t 01:00:00 --pty bash
#   GPUBRIDGE_REPO=$PWD bash scripts/gpu/explorer/setup.sbatch
# Compute nodes reach the internet only through the proxy that the explorer
# module sets up, so SETUP_MODULES (default: explorer) is loaded before installing.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

FLAVOR="${TORCH_FLAVOR:-cuda}"
SETUP_MODULES="${SETUP_MODULES-explorer}"
[ "$FLAVOR" = rocm ] && VENV=$VENV_ROCM

if [ "$DRY_RUN" = 0 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    die "not in a Slurm job. Setup runs on a compute node: use submit.sh setup, or" \
        "start an interactive job first (srun -p short --pty bash)."
fi
log "creating the $FLAVOR venv at $VENV on $(hostname)"
load_modules "$SETUP_MODULES"
"$KIT_DIR/setup_env.sh" --flavor "$FLAVOR" --venv "$VENV"
