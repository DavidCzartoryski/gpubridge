#!/usr/bin/env bash
# mixed/node.sh end to end on CPU, in two Linux containers on one Docker network
# standing in for two cloud machines on an overlay network: the master pretends to be
# the NVIDIA machine and the worker the AMD one (GPUBRIDGE_VENDOR), with NPROC ranks
# each. Every step node.sh runs on GPUs runs here, over a real network between two
# hosts: the preflight on eth0 (the overlay's interface), the bridge check, the
# benchmarks, training through the backend, training throughput, triage and the
# tarballs. What it can't show: GPUs, NCCL/RCCL, Tailscale or WireGuard themselves,
# and a real wide-area link.
#
#   scripts/gpu/mixed/containers.sh [--dry-run] [OUT]     OUT: results/mixed_containers
#
# Settings: NPROC (ranks per machine, default 2), CPUS (per container, default 2),
# BENCH_MAX_BYTES (default 1M), TRAIN_BENCH_MODELS (default "tiny-gpt tiny-resnet"),
# TRAIN_BENCH_ARGS, the data-parallel and pipeline runs' DP_* and PP_* (default: llama-tiny,
# 4 steps), KIT_TIMEOUT (default 600 s a step), IMAGE, SUBNET (a /24 for the network,
# default 10.231.0.0/24). Writes OUT/{master,worker}/ and their .tgz, each
# container's log, and triage.txt (both machines). Exits 1 if either machine failed.
# With Colima or Docker Desktop, the repo and OUT must be in a directory the VM shares
# (your home directory, by default); elsewhere the results stay inside the VM.
set -euo pipefail
DRY_RUN=0
[ "${1:-}" != --dry-run ] || { DRY_RUN=1; shift; }
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:-$REPO/results/mixed_containers}
IMAGE=${IMAGE:-gpubridge-mixed-cpu:dev}
SUBNET=${SUBNET:-10.231.0.0/24}
prefix=${SUBNET%.*}
MASTER_IP=$prefix.10
WORKER_IP=$prefix.11
NET=gpubridge-mixed-$$
export NPROC=${NPROC:-2} BENCH_MAX_BYTES=${BENCH_MAX_BYTES:-1M}
export TRAIN_BENCH_MODELS=${TRAIN_BENCH_MODELS:-tiny-gpt tiny-resnet}
export TRAIN_BENCH_ARGS=${TRAIN_BENCH_ARGS:-} KIT_TIMEOUT=${KIT_TIMEOUT:-600}
export DP_MODEL=${DP_MODEL:-llama-tiny} DP_STEPS=${DP_STEPS:-4} DP_GLOBAL_BATCH=${DP_GLOBAL_BATCH:-4}
export DP_SEQ=${DP_SEQ:-16} DP_ARGS=${DP_ARGS:---check-every 2 --probe-steps 1 --comm-trials 1}
export PP_ARGS=${PP_ARGS:---probe-trials 1}
CPUS=${CPUS:-2}

run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ %s\n' "$*" >&2
    else
        "$@"
    fi
}

run docker build -q -t "$IMAGE" -f "$REPO/scripts/gpu/mixed/Dockerfile" "$REPO" >/dev/null
run docker network create --subnet "$SUBNET" "$NET" >/dev/null
# Cleanup keeps the script's own exit status, and each command may fail (a container
# that never started) without skipping the next.
[ "$DRY_RUN" = 1 ] || trap 'status=$?
    docker rm -f "$NET-master" "$NET-worker" >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
    exit "$status"' EXIT
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# A git worktree's .git names the main repository's by its host path: mount that too,
# so the records keep the commit.
common=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
git_mount=()
[ -z "$common" ] || [ "$common" = "$REPO/.git" ] || git_mount=(-v "$common:$common:ro")

for role in master worker; do
    if [ "$role" = master ]; then vendor=nvidia ip=$MASTER_IP; else vendor=amd ip=$WORKER_IP; fi
    run docker run -d --name "$NET-$role" --hostname "$role" --network "$NET" --ip "$ip" \
        --cpus "$CPUS" -v "$REPO:/gpubridge:ro" "${git_mount[@]+"${git_mount[@]}"}" -v "$OUT:/out" \
        -e ROLE="$role" -e MASTER_ADDR="$MASTER_IP" -e IFNAME=eth0 -e SKIP_SETUP=1 \
        -e TORCH_FLAVOR=system -e VENV=/venv -e GPUBRIDGE_VENDOR="$vendor" -e TARGET="$role" \
        -e RESULTS_ROOT=/out -e RUN_ID="containers-$$" -e NPROC -e BENCH_MAX_BYTES \
        -e TRAIN_BENCH_MODELS -e TRAIN_BENCH_ARGS -e KIT_TIMEOUT -e DP_MODEL -e DP_STEPS \
        -e DP_GLOBAL_BATCH -e DP_SEQ -e DP_ARGS -e PP_ARGS \
        "$IMAGE" bash scripts/gpu/mixed/node.sh >/dev/null
done

status=0
for role in master worker; do
    code=$(run docker wait "$NET-$role")
    run docker logs "$NET-$role" >"$OUT/$role.log" 2>&1
    echo "$role: exit ${code:-?} (log: $OUT/$role.log)"
    [ "$DRY_RUN" = 1 ] || [ "$code" = 0 ] || status=1
done
for role in master worker; do
    run docker run --rm -v "$OUT:/out:ro" -v "$REPO:/gpubridge:ro" "$IMAGE" \
        python scripts/gpu/triage.py "/out/$role" --max-lines 0
done >"$OUT/triage.txt"
echo "triage of both machines: $OUT/triage.txt"
exit "$status"
