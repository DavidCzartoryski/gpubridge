#!/usr/bin/env bash
# Submit the Explorer (Slurm) validation steps. Every site-specific setting is
# below; override any of them with an environment variable of the same name.
#
#   scripts/gpu/explorer/submit.sh [--dry-run] STEP
#
#   setup  create the venv (runs here on the login node, which has internet)
#   01     probe: 1 GPU                          free gpu partition
#   02     split test, 2 processes on 1 GPU      free gpu partition
#   free   01 and 02
#   03     single island, 2+ GPUs on 1 node      multigpu (access request)
#   04     split test, 4 GPUs on 1 node          multigpu
#   05     2 nodes: one island, then split by node   multigpu
#   06     scaling run, 1 to 4 GPUs              multigpu
#   multi  03 to 06
#
# See docs/GPU_VALIDATION_RUNBOOK.md for what each step proves.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

# ---- Site settings: VERIFY each against rc-docs.northeastern.edu -------------
ACCOUNT="${ACCOUNT:-}"                     # --account, if your allocation needs one
PARTITION_1GPU="${PARTITION_1GPU:-gpu}"    # 1-GPU jobs; docs: open to all, 1 GPU per job
PARTITION_MULTI="${PARTITION_MULTI:-multigpu}"  # 2+ GPUs and multi-node; needs access
GPU_TYPE_1GPU="${GPU_TYPE_1GPU:-}"         # --gres type, e.g. v100-pcie; empty = any GPU
GPU_TYPE_MULTI="${GPU_TYPE_MULTI:-}"       # --gres type for multigpu, e.g. v100-sxm2
CPUS_PER_GPU="${CPUS_PER_GPU:-4}"
MEM_PER_GPU="${MEM_PER_GPU:-32G}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"           # anything else, e.g. "--qos=... --reservation=..."
# ---- Job sizes and walltimes (short, so jobs schedule quickly) ---------------
GPUS_SINGLE_ISLAND="${GPUS_SINGLE_ISLAND:-2}"
GPUS_PER_NODE_MULTINODE="${GPUS_PER_NODE_MULTINODE:-2}"
TIME_01="${TIME_01:-00:10:00}"
TIME_02="${TIME_02:-00:15:00}"
TIME_03="${TIME_03:-00:20:00}"
TIME_04="${TIME_04:-00:30:00}"
TIME_05="${TIME_05:-00:30:00}"
TIME_06="${TIME_06:-00:30:00}"
# ---- Environment for the jobs -------------------------------------------------
WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"  # venv and caches (~6 GB); VERIFY home quota
MODULES="${MODULES:-}"                     # `module load` names; none needed by default
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"  # step 05; empty = auto. VERIFY, e.g. ib0
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"  # step 05; empty = auto
# ------------------------------------------------------------------------------

LOG_DIR="$WORK_DIR/logs"
export GPUBRIDGE_REPO="$REPO_DIR" WORK_DIR MODULES NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export GPUS_SINGLE_ISLAND GPUS_PER_NODE_MULTINODE

submit() {
    local script=$1 time=$2 nodes=$3 gpus=$4 partition=$5 gpu_type=$6
    local args=(
        --job-name="gpubridge-${script%.sbatch}" --partition="$partition" --time="$time"
        --nodes="$nodes" --ntasks-per-node=1 --gres="gpu:${gpu_type:+$gpu_type:}$gpus"
        --cpus-per-gpu="$CPUS_PER_GPU" --mem-per-gpu="$MEM_PER_GPU"
        --output="$LOG_DIR/%x-%j.out" --export=ALL
    )
    [ -n "$ACCOUNT" ] && args+=(--account="$ACCOUNT")
    local extra=()
    read -ra extra <<<"$EXTRA_SBATCH"
    args+=("${extra[@]+"${extra[@]}"}")
    run mkdir -p "$LOG_DIR"
    run sbatch "${args[@]}" "$KIT_DIR/explorer/$script"
}

case "${ARGS[0]:-}" in
setup) run "$KIT_DIR/setup_env.sh" --flavor cuda --venv "$WORK_DIR/venv" ;;
01) submit 01_probe.sbatch "$TIME_01" 1 1 "$PARTITION_1GPU" "$GPU_TYPE_1GPU" ;;
02) submit 02_split_shared_gpu.sbatch "$TIME_02" 1 1 "$PARTITION_1GPU" "$GPU_TYPE_1GPU" ;;
03) submit 03_single_island.sbatch "$TIME_03" 1 "$GPUS_SINGLE_ISLAND" \
        "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
04) submit 04_split_4gpu.sbatch "$TIME_04" 1 4 "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
05) submit 05_multinode.sbatch "$TIME_05" 2 "$GPUS_PER_NODE_MULTINODE" \
        "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
06) submit 06_scaling.sbatch "$TIME_06" 1 4 "$PARTITION_MULTI" "$GPU_TYPE_MULTI" ;;
free | multi)
    # DRY_RUN is exported, so the recursive calls inherit --dry-run.
    steps=(01 02)
    [ "${ARGS[0]}" = multi ] && steps=(03 04 05 06)
    for s in "${steps[@]}"; do "$0" "$s"; done
    ;;
*) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
