#!/usr/bin/env bash
# Submit the Explorer (Slurm) validation steps. Every site-specific setting is
# below; override any of them with an environment variable of the same name.
#
#   scripts/gpu/explorer/submit.sh [--dry-run] STEP
#
#   00     GPU types in each partition (sinfo only; runs here, in seconds)
#   setup  create the CUDA venv in a short CPU job (never on the login node)
#   01     probe: 1 GPU                          free gpu partition
#   02     split test, 2 processes on 1 GPU      free gpu partition
#   free   01 and 02
#   03     single island, 2+ GPUs on 1 node      multigpu (access request)
#   04     split test, 4 GPUs on 1 node          multigpu
#   05     2 nodes: one island, then split by node   multigpu
#   06     scaling run, 1 to 4 GPUs              multigpu
#   multi  03 to 06
#   setup-rocm  create the ROCm venv in a job on an AMD node    dry-run only until
#   07     real mixed job: gpu + sharing, one heterogeneous job   step 00 shows AMD GPUs
#
# See docs/GPU_VALIDATION_RUNBOOK.md for what each step proves.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"
# Your own settings, e.g. PROJECT_DIR, can live in explorer/local.env (gitignored).
load_env_file "${EXPLORER_LOCAL_ENV:-$KIT_DIR/explorer/local.env}"

# ---- Site settings: VERIFY each against rc-docs.northeastern.edu -------------
ACCOUNT="${ACCOUNT:-}"                     # --account, if your allocation needs one
PARTITION_SETUP="${PARTITION_SETUP:-short}"  # setup job: any CPU partition
PARTITION_1GPU="${PARTITION_1GPU:-gpu}"    # 1-GPU jobs; docs: open to all, 1 GPU per job
PARTITION_MULTI="${PARTITION_MULTI:-multigpu}"  # 2+ GPUs and multi-node; needs access
PARTITION_AMD="${PARTITION_AMD:-sharing}"  # AMD GPU nodes, if step 00 finds any
GPU_TYPE_1GPU="${GPU_TYPE_1GPU:-}"         # --gres type, e.g. v100-pcie; empty = any GPU
GPU_TYPE_MULTI="${GPU_TYPE_MULTI:-}"       # --gres type for multigpu, e.g. v100-sxm2
GPU_TYPE_AMD="${GPU_TYPE_AMD:-}"           # --gres type of the AMD GPUs, from step 00 (required)
CPUS_PER_GPU="${CPUS_PER_GPU:-4}"
MEM_PER_GPU="${MEM_PER_GPU:-32G}"
CPUS_SETUP="${CPUS_SETUP:-4}"
MEM_SETUP="${MEM_SETUP:-16G}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"           # anything else, e.g. "--qos=... --reservation=..."
# ---- Job sizes and walltimes (short, so jobs schedule quickly) ---------------
GPUS_SINGLE_ISLAND="${GPUS_SINGLE_ISLAND:-2}"
GPUS_PER_NODE_MULTINODE="${GPUS_PER_NODE_MULTINODE:-2}"
# Step 07. With 1 GPU per island the per-island baselines only time the call
# itself; 2+ per island (multigpu access for NVIDIA) makes them real baselines.
PARTITION_MIXED_NVIDIA="${PARTITION_MIXED_NVIDIA:-$PARTITION_1GPU}"
GPUS_MIXED_NVIDIA="${GPUS_MIXED_NVIDIA:-1}"  # the gpu partition allows 1 GPU per job
GPUS_MIXED_AMD="${GPUS_MIXED_AMD:-1}"
TIME_SETUP="${TIME_SETUP:-01:00:00}"
TIME_01="${TIME_01:-00:10:00}"
TIME_02="${TIME_02:-00:15:00}"
TIME_03="${TIME_03:-00:20:00}"
TIME_04="${TIME_04:-00:30:00}"
TIME_05="${TIME_05:-00:30:00}"
TIME_06="${TIME_06:-00:30:00}"
TIME_07="${TIME_07:-00:30:00}"
# ---- Environment for the jobs -------------------------------------------------
# Venvs, caches and logs: ~7 GB, ~25 GB with the ROCm venv. Home is capped at 75 GB,
# so they go in your group's project space. Required: PROJECT_DIR or WORK_DIR.
PROJECT_DIR="${PROJECT_DIR:-}"             # e.g. /projects/<your-project>; or set it in local.env
WORK_DIR="${WORK_DIR:-${PROJECT_DIR:+$PROJECT_DIR/${USER:-$(id -un)}/gpubridge-gpu}}"
VENV="${VENV:-$WORK_DIR/venv}"             # CUDA build of torch: setup, steps 01 to 07
VENV_ROCM="${VENV_ROCM:-$WORK_DIR/venv-rocm}"  # ROCm build: setup-rocm, step 07
# setup-rocm's PyTorch: pick a ROCm build that supports the AMD GPU from step 00
# (see the runbook); older Instinct cards may not be supported by ROCm 7.2.
ROCM_TORCH_VERSION="${ROCM_TORCH_VERSION:-2.14.1}"
ROCM_INDEX="${ROCM_INDEX:-rocm7.2}"        # the wheel's local version tag, e.g. rocm6.4
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://download.pytorch.org/whl/$ROCM_INDEX}"
SETUP_MODULES="${SETUP_MODULES-explorer}"  # loaded before installing; explorer routes
                                           # compute nodes through the proxy ("" = none)
MODULES="${MODULES:-}"                     # `module load` names for steps 01 to 07; none needed
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"  # steps 05 and 07; empty = auto. VERIFY, e.g. ib0
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"  # steps 05 and 07; empty = auto
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/explorer}"
# ------------------------------------------------------------------------------

# setup-rocm and 07 only print their commands until step 00 shows AMD GPUs in
# $PARTITION_AMD. Set this to 1 in a reviewed change once the hardware is confirmed.
AMD_NODES_CONFIRMED=0

LOG_DIR="$WORK_DIR/logs"
export GPUBRIDGE_REPO="$REPO_DIR" WORK_DIR VENV VENV_ROCM SETUP_MODULES MODULES RESULTS_ROOT
export ROCM_TORCH_VERSION ROCM_INDEX ROCM_INDEX_URL
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export GPUS_SINGLE_ISLAND GPUS_PER_NODE_MULTINODE GPUS_MIXED_NVIDIA GPUS_MIXED_AMD

# site_options: append the account and EXTRA_SBATCH to REQUEST.
site_options() {
    [ -n "$ACCOUNT" ] && REQUEST+=(--account="$ACCOUNT")
    local extra=()
    read -ra extra <<<"$EXTRA_SBATCH"
    REQUEST+=("${extra[@]+"${extra[@]}"}")
}

# gpu_request PARTITION GPU_TYPE NODES GPUS TIME: sbatch options for one job, or
# one component of a heterogeneous job, in REQUEST.
gpu_request() {
    local partition=$1 gpu_type=$2 nodes=$3 gpus=$4 time=$5
    REQUEST=(
        --partition="$partition" --time="$time" --nodes="$nodes" --ntasks-per-node=1
        --gres="gpu:${gpu_type:+$gpu_type:}$gpus"
        --cpus-per-gpu="$CPUS_PER_GPU" --mem-per-gpu="$MEM_PER_GPU"
    )
    site_options
}

# sbatch_job NAME SCRIPT OPTIONS...: submit SCRIPT. A ":" in OPTIONS separates the
# components of a heterogeneous job; the name and log file belong to the first.
sbatch_job() {
    local name=$1 script=$2
    shift 2
    [ -n "$WORK_DIR" ] || die "set PROJECT_DIR to your group's project space, e.g." \
        "PROJECT_DIR=/projects/<your-project>, in the environment or in" \
        "scripts/gpu/explorer/local.env (gitignored); or set WORK_DIR. Venvs, caches and" \
        "logs go to \$PROJECT_DIR/\$USER/gpubridge-gpu, not home (capped at 75 GB)."
    run mkdir -p "$LOG_DIR"
    run sbatch --job-name="gpubridge-$name" --output="$LOG_DIR/%x-%j.out" --export=ALL \
        "$@" "$KIT_DIR/explorer/$script"
}

submit() {
    local script=$1 time=$2 nodes=$3 gpus=$4 partition=$5 gpu_type=$6
    gpu_request "$partition" "$gpu_type" "$nodes" "$gpus" "$time"
    sbatch_job "${script%.sbatch}" "$script" "${REQUEST[@]}"
}

amd_gate() {
    if [ "$DRY_RUN" = 1 ]; then
        [ "$AMD_NODES_CONFIRMED" = 1 ] || log "dry-run only until step 00 confirms AMD GPUs" \
            "in $PARTITION_AMD"
        AMD_TYPE="${GPU_TYPE_AMD:-amd-type-from-step-00}"
        return 0
    fi
    [ "$AMD_NODES_CONFIRMED" = 1 ] || die "the Explorer mixed-vendor path is dry-run only \
until step 00 shows AMD GPUs in the $PARTITION_AMD partition. Check \
$RESULTS_ROOT/00_partitions/gpu-types.txt, then set AMD_NODES_CONFIRMED=1 in $0."
    [ -n "$GPU_TYPE_AMD" ] || die "set GPU_TYPE_AMD to the AMD GPUs' --gres type from \
$RESULTS_ROOT/00_partitions/gpu-types.txt"
    AMD_TYPE=$GPU_TYPE_AMD
}

setup_job() {
    local name=$1
    shift
    export TORCH_FLAVOR=${name#setup-}
    sbatch_job "$name" setup.sbatch "$@"
    log "setup log: $LOG_DIR/gpubridge-$name-<job id>.out. Wait for the job to finish" \
        "(squeue -u \$USER) before submitting the steps that use its venv."
}

case "${ARGS[0]:-}" in
00) "$KIT_DIR/explorer/00_partitions.sh" "$PARTITION_AMD" "$PARTITION_1GPU" "$PARTITION_MULTI" ;;
setup)
    REQUEST=(--partition="$PARTITION_SETUP" --time="$TIME_SETUP" --nodes=1 --ntasks=1
             --cpus-per-task="$CPUS_SETUP" --mem="$MEM_SETUP")
    site_options
    setup_job setup-cuda "${REQUEST[@]}"
    ;;
setup-rocm)
    amd_gate
    gpu_request "$PARTITION_AMD" "$AMD_TYPE" 1 1 "$TIME_SETUP"
    setup_job setup-rocm "${REQUEST[@]}"
    ;;
01) submit 01_probe.sbatch "$TIME_01" 1 1 "$PARTITION_1GPU" "$GPU_TYPE_1GPU" ;;
02) submit 02_split_shared_gpu.sbatch "$TIME_02" 1 1 "$PARTITION_1GPU" "$GPU_TYPE_1GPU" ;;
03) submit 03_single_island.sbatch "$TIME_03" 1 "$GPUS_SINGLE_ISLAND" \
        "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
04) submit 04_split_4gpu.sbatch "$TIME_04" 1 4 "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
05) submit 05_multinode.sbatch "$TIME_05" 2 "$GPUS_PER_NODE_MULTINODE" \
        "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
06) submit 06_scaling.sbatch "$TIME_06" 1 4 "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
07)
    amd_gate
    gpu_request "$PARTITION_MIXED_NVIDIA" "$GPU_TYPE_1GPU" 1 "$GPUS_MIXED_NVIDIA" "$TIME_07"
    nvidia=("${REQUEST[@]}")
    gpu_request "$PARTITION_AMD" "$AMD_TYPE" 1 "$GPUS_MIXED_AMD" "$TIME_07"
    sbatch_job 07_mixed_hetjob 07_mixed_hetjob.sbatch "${nvidia[@]}" : "${REQUEST[@]}"
    ;;
free | multi)
    # DRY_RUN is exported, so the recursive calls inherit --dry-run.
    steps=(01 02)
    [ "${ARGS[0]}" = multi ] && steps=(03 04 05 06)
    for s in "${steps[@]}"; do "$0" "$s"; done
    ;;
*) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
