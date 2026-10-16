# shellcheck shell=bash
# Shared setup for the Explorer job scripts; sourced after common.sh.
# submit.sh exports the settings below. Defaults apply if you sbatch a step directly.

# Venvs and caches. Home is capped at 75 GB, so they live in the group's project
# space: PROJECT_DIR (e.g. /projects/<your-project>) from the environment or
# explorer/local.env (gitignored), or WORK_DIR directly.
load_env_file "${EXPLORER_LOCAL_ENV:-$KIT_DIR/explorer/local.env}"
PROJECT_DIR="${PROJECT_DIR:-}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR:+$PROJECT_DIR/${USER:-$(id -un)}/gpubridge-gpu}}"
[ -n "$WORK_DIR" ] || die "set PROJECT_DIR (e.g. /projects/<your-project>) or WORK_DIR," \
    "in the environment or in scripts/gpu/explorer/local.env; see submit.sh"
VENV="${VENV:-$WORK_DIR/venv}"                       # CUDA build of torch
# ROCm: the venv setup-rocm builds, and the one jobs run in (submit.sh explains both).
ROCM_SETUP_VENV="${ROCM_SETUP_VENV:-${VENV_ROCM:-$WORK_DIR/venv-rocm}}"
VENV_ROCM="${VENV_ROCM:-$WORK_DIR/venv-rocm63-triton38}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/explorer}"
MODULES="${MODULES:-}"                               # `module load` names, if any
# Opt-in collective policies each check-and-bench step also runs (off by default
# in gpubridge itself; selected here with --policy).
OPT_IN_POLICIES="${OPT_IN_POLICIES:-pipelined-reduce-bridge-broadcast sharded-bridge}"
# 1: steps 05 and 07 also build the UCX bridge on each node and run the check and
# the gpubridge benchmark over bridge="ucx" (docs/RDMA_TRANSPORT.md). Needs UCX
# there (e.g. MODULES="ucx") and a C++ compiler. Off by default.
UCX_BRIDGE="${UCX_BRIDGE:-0}"
# Triton's kernels persist here between jobs; each job builds and reads them in a
# node-local copy (triton_cache_begin in common.sh). "" = Triton's own default cache.
export TRITON_CACHE_STORE="${TRITON_CACHE_STORE-$WORK_DIR/triton-cache}"
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

# job_setup STEP: point RESULTS at results/explorer/STEP, record the job's start
# (job.jsonl), load modules, activate the venv and record the node's GPUs,
# interconnect topology and job details.
job_setup() {
    RESULTS=$(fresh_results "$RESULTS_ROOT/$1")
    export RESULTS
    run mkdir -p "$RESULTS"
    job_started
    load_modules "$MODULES"
    activate_venv "$VENV"
    require_kit_import
    triton_cache_begin
    capture "$RESULTS/nvidia-smi.txt" nvidia-smi
    capture "$RESULTS/nvidia-smi-topo.txt" nvidia-smi topo -m
    capture "$RESULTS/slurm-job.txt" scontrol show job "${SLURM_JOB_ID:-none}"
    log "job ${SLURM_JOB_ID:-local} on $(hostname): results in $RESULTS"
}
