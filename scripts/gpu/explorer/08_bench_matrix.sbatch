#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-08-bench-matrix
#SBATCH --time=00:40:00
# Step 08, the comparison matrix: what gpubridge costs against native NCCL/RCCL and
# flat Gloo. One heterogeneous job of two components, one GPU each on its own node;
# MATRIX_PAIR (set by submit.sh) says which pair, and each pair is its own job on its
# own nodes, so bench_matrix.py lists each job's network next to the ratios:
#
#   nvidia-amd     a V100 (gpu) + an MI50 (sharing): D, every policy, and E   submit.sh 08
#                  (MULTITCP_BRIDGE=1 adds D over bridge="multitcp")
#   nvidia-nvidia  two V100s: A1, A2, C and E-nvidia               submit.sh 08-nvidia-pair
#   amd-amd        two MI50s: B1, B2, C-amd and E-amd                 submit.sh 08-amd-pair
#
# Each run is bench_all_reduce.py on both ranks, float32 from 1 KB to BENCH_MAX_BYTES
# (1 GB), timed with CUDA events, with --phases on every gpubridge run, into
# matrix/<run>/; bench_matrix.py then puts them side by side (matrix.md). Each node's
# network goes to network/<side>/ (net_report.sh), and NCCL/RCCL's own account of the
# transport it chose to nccl/matrix_<run>/ (NCCL_DEBUG=INFO).
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

MATRIX_PAIR="${MATRIX_PAIR:-nvidia-amd}"
RDZV_PORT="${RDZV_PORT:-29500}"
PREFLIGHT_PORT="${PREFLIGHT_PORT:-29501}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
# D's pipelined runs, one per chunk size (GPUBRIDGE_CHUNK_BYTES; gpubridge's default is 4M).
PIPELINE_CHUNKS="${PIPELINE_CHUNKS:-1M 4M 16M 64M}"
# 1: also D-multitcp, D-multitcp-pinned and D-multitcp-1stream, over bridge="multitcp".
MULTITCP_BRIDGE="${MULTITCP_BRIDGE:-0}"
case $MATRIX_PAIR in
nvidia-amd) VENV_0=$VENV VENV_1=$VENV_ROCM SIDE_0=nvidia SIDE_1=amd ;;
nvidia-nvidia) VENV_0=$VENV VENV_1=$VENV SIDE_0=nvidia-0 SIDE_1=nvidia-1 ;;
amd-amd) VENV_0=$VENV_ROCM VENV_1=$VENV_ROCM SIDE_0=amd-0 SIDE_1=amd-1 ;;
*) die "MATRIX_PAIR must be nvidia-amd, nvidia-nvidia or amd-amd, not '$MATRIX_PAIR'" ;;
esac
IFNAME_0="" IFNAME_1=""  # each node's interface for Gloo, filled in after the preflight

if [ "$DRY_RUN" = 1 ]; then
    HEAD="<node-0>"
else
    [ -n "${SLURM_JOB_NODELIST_HET_GROUP_1:-}" ] ||
        die "not a heterogeneous job; submit with scripts/gpu/explorer/submit.sh 08"
    HEAD=$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0:-}" | head -n 1)
    OTHER=$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1:-}" | head -n 1)
fi

# fill ARG GROUP: ARG with @VENV@, @SIDE@, @ROLE@ and @IFNAME@ replaced for component
# GROUP (0: master, on the rendezvous node; 1: worker).
fill() {
    local arg=$1 venv side ifname role
    if [ "$2" = 0 ]; then venv=$VENV_0 side=$SIDE_0 ifname=$IFNAME_0 role=master
    else venv=$VENV_1 side=$SIDE_1 ifname=$IFNAME_1 role=worker; fi
    arg=${arg//@VENV@/$venv}
    arg=${arg//@SIDE@/$side}
    arg=${arg//@ROLE@/$role}
    arg=${arg//@IFNAME@/$ifname}
    printf '%s' "$arg"
}

# both ARGS...: one heterogeneous job step running ARGS on each component, with its
# own venv, side name and interface. --kill-on-bad-exit stops one side when the
# other fails, so neither is left waiting for a rendezvous.
both() {
    local zero=() one=() arg
    if [ -n "$IFNAME_0" ]; then zero=(env "GLOO_SOCKET_IFNAME=$IFNAME_0"); fi
    if [ -n "$IFNAME_1" ]; then one=(env "GLOO_SOCKET_IFNAME=$IFNAME_1"); fi
    for arg in "$@"; do
        zero+=("$(fill "$arg" 0)")
        one+=("$(fill "$arg" 1)")
    done
    limited srun --het-group=0 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 "${zero[@]}" \
        : --het-group=1 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 "${one[@]}"
}

# matrix_run NAME [NAME=VALUE...] -- ARGS...: run NAME of the matrix: the benchmark on
# both ranks with those settings (@IFNAME@ is each node's Gloo interface) and ARGS.
matrix_run() {
    local name=$1 settings=() compression=""
    shift
    while [ $# -gt 0 ] && [ "$1" != -- ]; do
        settings+=("$1")
        case $1 in GPUBRIDGE_BRIDGE_COMPRESSION=*) compression=${1#*=} ;; esac
        shift
    done
    shift
    local run=(step "matrix_$name" both env "${settings[@]+"${settings[@]}"}" @VENV@/bin/torchrun
        --nnodes=2 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint="$HEAD:$RDZV_PORT"
        --rdzv-id="${SLURM_JOB_ID:-local}-$name" "$KIT_DIR/bench_all_reduce.py"
        --out "$RESULTS/matrix/$name" --max-bytes "$BENCH_MAX_BYTES" "$@")
    if [ -n "$compression" ]; then
        # So step sees that this run compresses, and checks Triton first (common.sh).
        GPUBRIDGE_BRIDGE_COMPRESSION=$compression "${run[@]}"
    else
        "${run[@]}"
    fi
}
# Triton's check (common.sh) before the first compressed run: on both nodes.
TRITON_CHECK_LAUNCH="both @VENV@/bin/python"

# The batch node runs this job's Python (bench_matrix.py) with its own side's venv.
[ "$MATRIX_PAIR" != amd-amd ] || VENV=$VENV_ROCM
job_setup "08_matrix_$MATRIX_PAIR"
# Two components of one partition can share a node, and then NCCL never touches the
# network: not what this job measures. After job_setup, so job.jsonl records why (without
# the node's name: the job's own log has it).
if [ "$DRY_RUN" = 0 ] && [ "$HEAD" = "$OTHER" ]; then
    log "both components are on $HEAD"
    die "both components are on one node; the matrix needs two"
fi
if [ "$MATRIX_PAIR" != nvidia-nvidia ]; then
    rocm_asic_ids "$VENV_ROCM/bin/python"  # the ROCm wheel's GPU-name table (common.sh)
fi
step network both "$KIT_DIR/net_report.sh" "$RESULTS/network/@SIDE@"
step probe both @VENV@/bin/python "$KIT_DIR/probe.py" --out "$RESULTS/probe_@SIDE@"
step preflight both @VENV@/bin/python "$KIT_DIR/preflight.py" --role @ROLE@ \
    --master-addr "$HEAD" --port "$PREFLIGHT_PORT" --bandwidth-seconds "$PREFLIGHT_SECONDS" \
    --out "$RESULTS/preflight"
# As in step 07: nothing below can pass without a GPU on each side and a network
# between them. In simulation (the CPU test) there is no GPU, so only the preflight counts.
blocking=()
for name in "${FAILED_STEPS[@]+"${FAILED_STEPS[@]}"}"; do
    [ "$name" = probe ] && [ -n "${GPUBRIDGE_VENDOR:-}${GPUBRIDGE_CPU_ONLY:-}" ] && continue
    [ "$name" = network ] && continue  # a record, not a requirement
    blocking+=("$name")
done
if [ "${#blocking[@]}" -gt 0 ]; then
    finish || true
    die "stopping before the benchmarks: ${blocking[*]} failed; logs in $RESULTS"
fi
if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ "$USE_PREFLIGHT_IFNAME" = 1 ]; then
    IFNAME_0=$(preflight_ifname "$RESULTS/preflight/preflight-master.json")
    IFNAME_1=$(preflight_ifname "$RESULTS/preflight/preflight-worker.json")
    log "GLOO_SOCKET_IFNAME: ${IFNAME_0:-unset} on $SIDE_0, ${IFNAME_1:-unset} on $SIDE_1," \
        "as the preflight chose"
elif [ -n "${GLOO_SOCKET_IFNAME:-}" ]; then
    IFNAME_0=$GLOO_SOCKET_IFNAME IFNAME_1=$GLOO_SOCKET_IFNAME
fi
# A2 and B2 put NCCL/RCCL on the bridge's wire: TCP on the interface Gloo uses, by
# its exact name ("=": NCCL otherwise matches it as a prefix).
tcp=(NCCL_IB_DISABLE=1)
if [ -n "$IFNAME_0" ] && [ -n "$IFNAME_1" ]; then
    tcp+=(NCCL_SOCKET_IFNAME==@IFNAME@)
elif [ "$MATRIX_PAIR" != nvidia-amd ]; then
    log "warning: no interface for Gloo on both nodes, so the TCP baseline runs on" \
        "the interface NCCL picks, which may not be the bridge's"
fi

case $MATRIX_PAIR in
nvidia-amd)
    matrix_run D-default -- --ops gpubridge --phases
    matrix_run D-pinned GPUBRIDGE_PINNED_STAGING=1 -- --ops gpubridge --phases
    # The multitcp bridge (bridge="multitcp", docs/RUST_TRANSPORT.md), opt-in with
    # MULTITCP_BRIDGE=1: each side builds the Rust library (cargo on the node), then it
    # runs with the default and the pinned staging, and over one connection instead of
    # several, to tell the connections' share from the chunked overlap's.
    if [ "$MULTITCP_BRIDGE" = 1 ]; then
        step multitcp_build both @VENV@/bin/python -m gpubridge.multitcp build
        matrix_run D-multitcp -- --ops gpubridge --phases --bridge multitcp
        matrix_run D-multitcp-pinned GPUBRIDGE_PINNED_STAGING=1 -- --ops gpubridge --phases \
            --bridge multitcp
        matrix_run D-multitcp-1stream GPUBRIDGE_MULTITCP_STREAMS=1 -- --ops gpubridge --phases \
            --bridge multitcp
    fi
    for chunk in $PIPELINE_CHUNKS; do
        matrix_run "D-pipelined-$chunk" "GPUBRIDGE_CHUNK_BYTES=$chunk" -- --ops gpubridge \
            --phases --policy pipelined-reduce-bridge-broadcast
    done
    matrix_run D-sharded -- --ops gpubridge --phases --policy sharded-bridge
    matrix_run D-bf16 GPUBRIDGE_BRIDGE_COMPRESSION=bf16 -- --ops gpubridge --phases
    matrix_run D-fp8 GPUBRIDGE_BRIDGE_COMPRESSION=fp8 -- --ops gpubridge --phases
    matrix_run E -- --ops flat-gloo
    # The default again, last: how far the same run moves within one job.
    matrix_run D-default-repeat -- --ops gpubridge --phases
    ;;
nvidia-nvidia)
    matrix_run A1 -- --ops native
    matrix_run A2 "${tcp[@]}" -- --ops native
    matrix_run C GPUBRIDGE_SPLIT_TEST=node -- --ops gpubridge --phases
    matrix_run E-nvidia -- --ops flat-gloo
    ;;
amd-amd)
    matrix_run B1 -- --ops native
    matrix_run B2 "${tcp[@]}" -- --ops native
    matrix_run C-amd GPUBRIDGE_SPLIT_TEST=node -- --ops gpubridge --phases
    matrix_run E-amd -- --ops flat-gloo
    ;;
esac
step matrix python "$KIT_DIR/bench_matrix.py" "$RESULTS" --out "$RESULTS"
finish
