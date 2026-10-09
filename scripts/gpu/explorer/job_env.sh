# shellcheck shell=bash
# Shared setup for the Explorer job scripts; sourced after common.sh.
# submit.sh exports the settings below. Defaults apply if you sbatch a step directly.

# Venvs and caches. Home is capped at 75 GB, so they live in the group's project space.
WORK_DIR="${WORK_DIR:-/projects/jon-bell-research-group/${USER:-$(id -un)}/gpubridge-gpu}"
VENV="${VENV:-$WORK_DIR/venv}"                       # CUDA build of torch
VENV_ROCM="${VENV_ROCM:-$WORK_DIR/venv-rocm}"        # ROCm build (step 07)
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/explorer}"
MODULES="${MODULES:-}"                               # `module load` names, if any
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# load_modules "NAMES": `module load` each space-separated name. Finds Lmod if
# this shell doesn't have the module command yet.
load_modules() {
    [ -n "$1" ] || return 0
    local init module
    set +u  # Lmod's shell code reads unset variables
    if [ "$DRY_RUN" = 0 ] && ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
            if [ -f "$init" ]; then
                # shellcheck disable=SC1090
                source "$init"
                break
            fi
        done
    fi
    for module in $1; do
        run module load "$module" || die "module load $module failed"
    done
    set -u
}

# job_setup STEP: point RESULTS at results/explorer/STEP, load modules, activate
# the venv and record the node's GPUs, interconnect topology and job details.
job_setup() {
    RESULTS="$RESULTS_ROOT/$1"
    export RESULTS
    run mkdir -p "$RESULTS"
    load_modules "$MODULES"
    activate_venv "$VENV"
    capture "$RESULTS/nvidia-smi.txt" nvidia-smi
    capture "$RESULTS/nvidia-smi-topo.txt" nvidia-smi topo -m
    capture "$RESULTS/slurm-job.txt" scontrol show job "${SLURM_JOB_ID:-none}"
    log "job ${SLURM_JOB_ID:-local} on $(hostname): results in $RESULTS"
}
