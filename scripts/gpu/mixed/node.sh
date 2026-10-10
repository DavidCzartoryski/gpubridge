#!/usr/bin/env bash
# One machine's half of a REAL mixed NVIDIA + AMD job. Run it on both machines
# at about the same time; either machine can be the master:
#
#   NVIDIA machine:  ROLE=master MASTER_ADDR=100.64.0.1 scripts/gpu/mixed/node.sh
#   AMD machine:     ROLE=worker MASTER_ADDR=100.64.0.1 scripts/gpu/mixed/node.sh
#
# MASTER_ADDR is the master's address on a flat network between the machines:
# every TCP port open both ways. Tailscale is the default way to get one,
# WireGuard the alternative; see docs/GPU_VALIDATION_RUNBOOK.md. A preflight
# check runs first and stops with a clear message if the machines can't talk,
# before torchrun starts. Add --dry-run to only print commands.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

# ---- Settings (override with environment variables) ---------------------------
ROLE="${ROLE:-}"                         # master | worker
MASTER_ADDR="${MASTER_ADDR:-}"           # master's flat-network IP (`tailscale ip -4`)
MASTER_PORT="${MASTER_PORT:-29500}"      # torchrun rendezvous
PREFLIGHT_PORT="${PREFLIGHT_PORT:-29501}"  # the preflight's own check
IFNAME="${IFNAME:-tailscale0}"           # flat-network interface: tailscale0 or wg0
LOCAL_ADDR="${LOCAL_ADDR:-}"             # this machine's IP on it; default: read from IFNAME
NPROC="${NPROC:-}"                       # GPUs to use here; default: all visible
RUN_ID="${RUN_ID:-gpubridge-mixed}"      # must be the same on both machines
TORCH_FLAVOR="${TORCH_FLAVOR:-}"         # cuda | rocm | system; default: from the hardware
export TORCH_VERSION="${TORCH_VERSION:-2.14.1}"  # same release on both machines
SKIP_SETUP="${SKIP_SETUP:-0}"
WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-256M}"  # the bridge crosses the internet: keep it modest
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
# -------------------------------------------------------------------------------

case $ROLE in
master) IS_HOST=1 ;;
worker) IS_HOST=0 ;;
*) die "set ROLE=master or ROLE=worker" ;;
esac
[ -n "$MASTER_ADDR" ] || die "set MASTER_ADDR to the master's flat-network IP"
if [ -z "$TORCH_FLAVOR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then TORCH_FLAVOR=cuda; else TORCH_FLAVOR=rocm; fi
fi
TARGET="${TARGET:-mixed-$ROLE-$(hostname -s)}"
RESULTS="$RESULTS_ROOT/$TARGET"
export RESULTS
VENV="${VENV:-$WORK_DIR/venv}"

if [ "$SKIP_SETUP" != 1 ]; then
    step setup "$KIT_DIR/setup_env.sh" --flavor "$TORCH_FLAVOR" --venv "$VENV"
fi
activate_venv "$VENV"
capture "$RESULTS/nvidia-smi.txt" nvidia-smi
capture "$RESULTS/rocm-smi.txt" rocm-smi

# Gloo (the bridge and the world group) must use the flat network. NCCL/RCCL
# only run inside each machine's island, so they keep their own defaults.
export GLOO_SOCKET_IFNAME="$IFNAME"
if [ -z "$LOCAL_ADDR" ]; then
    if [ "$DRY_RUN" = 1 ]; then
        LOCAL_ADDR="<$IFNAME-address>"
    elif command -v ip >/dev/null 2>&1; then
        LOCAL_ADDR=$(ip -4 -o addr show dev "$IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
    fi
    [ -n "$LOCAL_ADDR" ] || die "no IPv4 address on $IFNAME; is Tailscale/WireGuard up? Set LOCAL_ADDR."
fi
NPROC="${NPROC:-$(gpu_count)}"
log "$ROLE: $NPROC GPU(s), $TORCH_FLAVOR build, $LOCAL_ADDR on $IFNAME, master $MASTER_ADDR"

# Preflight is not a skippable step: if the machines can't talk, stop here.
if ! run python "$KIT_DIR/preflight.py" --role "$ROLE" --master-addr "$MASTER_ADDR" \
    --port "$PREFLIGHT_PORT" --ifname "$IFNAME" --out "$RESULTS"; then
    die "preflight failed; fix connectivity before starting torchrun (see message above)"
fi

mixed_torchrun() {
    local tag=$1
    shift
    limited torchrun --nnodes=2 --nproc-per-node="$NPROC" --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv-id="$RUN_ID-$tag" \
        --rdzv-conf="is_host=$IS_HOST" --local-addr="$LOCAL_ADDR" "$@"
}

step check mixed_torchrun check "$KIT_DIR/check.py" --out "$RESULTS/check"
step summarize python "$KIT_DIR/summarize.py" --partial "$RESULTS/check"
step bench mixed_torchrun bench "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" \
    --max-bytes "$BENCH_MAX_BYTES"
package_results "$RESULTS"
finish
