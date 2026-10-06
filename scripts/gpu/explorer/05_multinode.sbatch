#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-05-multinode
#SBATCH --time=00:30:00
# Step 05, 2 nodes (multigpu). Submit with: scripts/gpu/explorer/submit.sh 05
# Part A: one NCCL island spanning both nodes (native multi-node baseline).
# Part B: TEST ONLY split by node, so the Gloo bridge crosses the network
# between nodes (item 8), plus a native vs bridged benchmark.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

GPUS_PER_NODE="${GPUS_PER_NODE_MULTINODE:-2}"
RDZV_PORT="${RDZV_PORT:-29500}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
# Interfaces: empty lets NCCL/Gloo choose. VERIFY the cluster's fast interface (e.g. ib0).
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && export GLOO_SOCKET_IFNAME

job_setup 05_multinode
NNODES="${SLURM_NNODES:-2}"
if [ "$DRY_RUN" = 1 ]; then
    HEAD="<first-node>"
else
    HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

# One torchrun per node via srun; c10d rendezvous on the first node.
multinode() {
    local tag=$1
    shift
    run srun --nodes="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
        --rdzv-backend=c10d --rdzv-endpoint="$HEAD:$RDZV_PORT" \
        --rdzv-id="${SLURM_JOB_ID:-local}-$tag" "$@"
}

step check_one_island multinode one "$KIT_DIR/check.py" --out "$RESULTS/check_one_island"
step summarize_one_island python "$KIT_DIR/summarize.py" "$RESULTS/check_one_island"
export GPUBRIDGE_SPLIT_TEST=node
step check_split_by_node multinode split "$KIT_DIR/check.py" \
    --out "$RESULTS/check_split_by_node"
step summarize_split_by_node python "$KIT_DIR/summarize.py" "$RESULTS/check_split_by_node"
step bench_split_by_node multinode bench "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_split_by_node" --max-bytes "$BENCH_MAX_BYTES"
finish
