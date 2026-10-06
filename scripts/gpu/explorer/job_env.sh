# shellcheck shell=bash
# Shared setup for the Explorer job scripts; sourced after common.sh.
# submit.sh exports the settings below. Defaults apply if you sbatch a step directly.

WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"          # venv and caches
VENV="${VENV:-$WORK_DIR/venv}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/explorer}"
MODULES="${MODULES:-}"                               # `module load` names, if any
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# job_setup STEP: point RESULTS at results/explorer/STEP, load modules, activate
# the venv and record the node's GPUs, interconnect topology and job details.
job_setup() {
    RESULTS="$RESULTS_ROOT/$1"
    export RESULTS
    run mkdir -p "$RESULTS"
    local module
    for module in $MODULES; do
        run module load "$module"
    done
    activate_venv "$VENV"
    capture "$RESULTS/nvidia-smi.txt" nvidia-smi
    capture "$RESULTS/nvidia-smi-topo.txt" nvidia-smi topo -m
    capture "$RESULTS/slurm-job.txt" scontrol show job "${SLURM_JOB_ID:-none}"
    log "job ${SLURM_JOB_ID:-local} on $(hostname): results in $RESULTS"
}
