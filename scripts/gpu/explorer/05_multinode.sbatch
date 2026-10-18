#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-05-multinode
#SBATCH --time=00:30:00
# Step 05, 2 nodes (multigpu). Submit with: scripts/gpu/explorer/submit.sh 05
# First the network preflight: TCP both ways between the nodes' interfaces, raw
# bandwidth, and the interface Gloo uses for the rest of the job (item 8).
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
PREFLIGHT_PORT="${PREFLIGHT_PORT:-29501}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
# Interfaces: empty lets NCCL choose, and Gloo use the one the preflight picks.
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && export GLOO_SOCKET_IFNAME

job_setup 05_multinode
NNODES="${SLURM_NNODES:-2}"
if [ "$DRY_RUN" = 1 ]; then
    HEAD="<first-node>"
else
    HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

# One torchrun per node via srun; c10d rendezvous on the first node. node_ifname.sh
# gives each node its own preflight choice when the two nodes' names differ.
multinode() {
    local tag=$1
    shift
    limited srun --nodes="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        bash "$KIT_DIR/node_ifname.sh" \
        torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
        --rdzv-backend=c10d --rdzv-endpoint="$HEAD:$RDZV_PORT" \
        --rdzv-id="${SLURM_JOB_ID:-local}-$tag" "$@"
}

# The first two nodes: srun task 0 is the master, on the rendezvous node.
step preflight limited srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    python "$KIT_DIR/preflight.py" --role auto --master-addr "$HEAD" --port "$PREFLIGHT_PORT" \
    --bandwidth-seconds "$PREFLIGHT_SECONDS" --out "$RESULTS/preflight"
# Nothing below can work if the nodes can't talk, so stop instead of waiting out
# rendezvous timeouts.
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    finish || true
    die "stopping before torchrun: the preflight failed; see $RESULTS/preflight.log"
fi
use_preflight_ifname "$RESULTS/preflight"

step check_one_island multinode one "$KIT_DIR/check.py" --out "$RESULTS/check_one_island"
step summarize_one_island python "$KIT_DIR/summarize.py" "$RESULTS/check_one_island"
export GPUBRIDGE_SPLIT_TEST=node
step check_split_by_node multinode split "$KIT_DIR/check.py" \
    --out "$RESULTS/check_split_by_node"
step summarize_split_by_node python "$KIT_DIR/summarize.py" "$RESULTS/check_split_by_node"
step bench_split_by_node multinode bench "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_split_by_node" --max-bytes "$BENCH_MAX_BYTES" \
    --ops native,island,gpubridge,flat-gloo --phases
# Pinned staging for the default bridge (item 25), with the bridge crossing the
# network: the same check and gpubridge benchmark with GPUBRIDGE_PINNED_STAGING=1.
export GPUBRIDGE_PINNED_STAGING=1
step check_pinned multinode check-pinned "$KIT_DIR/check.py" --out "$RESULTS/check_pinned"
step summarize_pinned python "$KIT_DIR/summarize.py" "$RESULTS/check_pinned"
step bench_pinned multinode bench-pinned "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_pinned" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge --phases
unset GPUBRIDGE_PINNED_STAGING
# The UCX bridge (docs/RDMA_TRANSPORT.md), opt-in with UCX_BRIDGE=1: built on
# both nodes, then the check and the gpubridge benchmark over bridge="ucx", next
# to the Gloo bridge's above (items 26 and 27).
if [ "$UCX_BRIDGE" = 1 ]; then
    step ucx_build limited srun --nodes=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
        python -m gpubridge.ucx build
    step check_ucx multinode check-ucx "$KIT_DIR/check.py" --out "$RESULTS/check_ucx" \
        --bridge ucx
    step summarize_ucx python "$KIT_DIR/summarize.py" "$RESULTS/check_ucx"
    step bench_ucx multinode bench-ucx "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_ucx" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge --phases \
        --bridge ucx
fi
# The verbs bridge (docs/VERBS_TRANSPORT.md), opt-in with VERBS_BRIDGE=1, same vendor
# on both nodes (items 38 to 40 and 42; the cross-vendor runs are step 07's): built on
# both nodes; each node's RDMA report; host-to-host verbs (one rank per node); the
# collectives on host tensors, then GPU tensors staged and direct (a direct path that
# can't run fails, never stages); the check and the benchmark over bridge="verbs", whose
# matched references are the Gloo bridge's runs above and UCX's (UCX_BRIDGE=1); then
# training over verbs and over Gloo. Observations go to verbs_observations/, and every
# verbs_kit record must show the payload on the RDMA port, not on TCP.
if [ "$VERBS_BRIDGE" = 1 ]; then
    one_per_node() {
        local tag=$1
        shift
        limited srun --nodes=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
            bash "$KIT_DIR/node_ifname.sh" \
            torchrun --nnodes=2 --nproc-per-node=1 --rdzv-backend=c10d \
            --rdzv-endpoint="$HEAD:$RDZV_PORT" --rdzv-id="${SLURM_JOB_ID:-local}-$tag" "$@"
    }
    step verbs_build limited srun --nodes=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
        python -m gpubridge.verbs build
    # One directory per node, named on that node (the single quotes are the inner bash's).
    # shellcheck disable=SC2016
    step verbs_rdma_report limited srun --nodes=2 --ntasks-per-node=1 \
        bash -c 'bash "$1/rdma_report.sh" "$2/verbs_rdma_report/$(hostname -s)"' _ \
        "$KIT_DIR" "$RESULTS"
    export GPUBRIDGE_VERBS_OBSERVE="$RESULTS/verbs_observations/rank-{rank}.jsonl"
    step verbs_endpoint one_per_node verbs-endpoint "$KIT_DIR/verbs_kit.py" endpoint \
        --out "$RESULTS/verbs_endpoint" --max-bytes "$BENCH_MAX_BYTES" --require-no-tcp
    step verbs_host multinode verbs-host "$KIT_DIR/verbs_kit.py" collectives --memory host \
        --out "$RESULTS/verbs_host" --max-bytes "$BENCH_MAX_BYTES" --require-no-tcp
    for path in staged direct; do
        export GPUBRIDGE_VERBS_GPU_PATH="$path"
        # Direct means nothing bounced: every message by rendezvous, written in place.
        if [ "$path" = direct ]; then export GPUBRIDGE_VERBS_RNDV_BYTES=1; fi
        step "verbs_gpu_$path" multinode "verbs-gpu-$path" "$KIT_DIR/verbs_kit.py" collectives \
            --memory gpu --expect-path "$path" --out "$RESULTS/verbs_gpu_$path" \
            --max-bytes "$BENCH_MAX_BYTES" --require-no-tcp
    done
    unset GPUBRIDGE_VERBS_GPU_PATH GPUBRIDGE_VERBS_RNDV_BYTES
    step check_verbs multinode check-verbs "$KIT_DIR/check.py" --out "$RESULTS/check_verbs" \
        --bridge verbs
    step summarize_verbs python "$KIT_DIR/summarize.py" "$RESULTS/check_verbs"
    step bench_verbs multinode bench-verbs "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_verbs" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge --phases \
        --bridge verbs
    for bridge in verbs gloo; do
        export GPUBRIDGE_BRIDGE="$bridge"
        step "train_$bridge" multinode "train-$bridge" "$KIT_DIR/train_bench.py" \
            --model "$VERBS_TRAIN_MODEL" --json "$RESULTS/train_${bridge}.json"
    done
    unset GPUBRIDGE_BRIDGE
    unset GPUBRIDGE_VERBS_OBSERVE
fi
# Opt-in policies with the bridge crossing the network (items 17 to 21): the
# check, and each policy's phases, so its bridge step can be read on its own.
for policy in $OPT_IN_POLICIES; do
    step "check_$policy" multinode "check-$policy" "$KIT_DIR/check.py" \
        --out "$RESULTS/check_$policy" --policy "$policy"
    step "summarize_$policy" python "$KIT_DIR/summarize.py" "$RESULTS/check_$policy"
    step "bench_phases_$policy" multinode "bench-$policy" "$KIT_DIR/bench_all_reduce.py" \
        --out "$RESULTS/bench_phases_$policy" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge \
        --policy "$policy" --phases
done
step bench_candidates multinode candidates "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_candidates" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge \
    --write-thresholds "$RESULTS/thresholds.json"
finish
