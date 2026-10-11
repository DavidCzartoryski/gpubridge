#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-amd-node
#SBATCH --time=00:30:00
# Rung 3 of the ladder on Explorer, one AMD node (sharing). Submit with:
# scripts/gpu/explorer/submit.sh amd, which stays dry-run only until step 00
# confirms AMD GPUs in sharing, like setup-rocm and 07. Needs the ROCm venv
# from submit.sh setup-rocm.
# Runs amd/run_all.sh on the node, without its setup or results tarball: probe,
# device mapping, an RCCL check and its version, the GPU-only tests, then an
# RCCL island, an AMD split test, benchmarks and scaling with 2+ GPUs, or a
# shared-GPU split test with 1. The single-vendor AMD result before step 07.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

load_modules "$MODULES"
export SKIP_SETUP=1 PACKAGE_RESULTS=0 VENV="$VENV_ROCM" RESULTS_ROOT TARGET=amd_node
log "job ${SLURM_JOB_ID:-local} on $(hostname): results in $RESULTS_ROOT/amd_node"
run bash "$KIT_DIR/amd/run_all.sh"
