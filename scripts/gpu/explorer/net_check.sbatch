#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-net-check
#SBATCH --time=00:10:00
# One node's network: what the matrix's runs (step 08) cross. net_report.sh records the
# InfiniBand devices and ports (link layer InfiniBand, or Ethernet for RoCE) and the
# interfaces; then one rank initializes NCCL (NVIDIA) or RCCL (AMD) with NCCL_DEBUG=INFO,
# which logs the network it would use between nodes ("Using network IB" or "Socket", and
# the device or interface), first as the library chooses (A1, B1), then with
# NCCL_IB_DISABLE=1 (A2, B2). bench_matrix.py summarizes it (matrix.md, "Network").
# Last, rdma_report.sh records what GPU-to-NIC RDMA would need on this node (peer-memory
# modules, dma-buf, UCX) into rdma/<side>/, after loading RDMA_MODULES (`module load`
# names, e.g. one that provides ucx_info; none by default), so the steps before it run
# in the same environment as without it.
# Submit with scripts/gpu/explorer/submit.sh net-nvidia or net-amd.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

NET_SIDE="${NET_SIDE:-nvidia}"
case $NET_SIDE in
nvidia) ;;
amd) VENV=$VENV_ROCM ;;
*) die "NET_SIDE must be nvidia or amd, not '$NET_SIDE'" ;;
esac
job_setup "net_check_$NET_SIDE"
[ "$NET_SIDE" = nvidia ] || rocm_asic_ids "$VENV/bin/python"
step network "$KIT_DIR/net_report.sh" "$RESULTS/network/$NET_SIDE"
# One rank, one all_reduce of 1 KB: enough for the library to pick its network.
init=("$KIT_DIR/bench_all_reduce.py" --ops native --min-bytes 1K --max-bytes 1K --warmup 1
      --trials 3)
step net_default local_torchrun --nproc-per-node=1 "${init[@]}" --out "$RESULTS/net_default"
step net_tcp env_local_torchrun NCCL_IB_DISABLE=1 --nproc-per-node=1 "${init[@]}" \
    --out "$RESULTS/net_tcp"
step matrix python "$KIT_DIR/bench_matrix.py" "$RESULTS" --out "$RESULTS"
load_modules "${RDMA_MODULES:-}"
step rdma "$KIT_DIR/rdma_report.sh" "$RESULTS/rdma/$NET_SIDE" "$VENV/bin/python"
finish
