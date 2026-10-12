#!/usr/bin/env bash
# node_ifname.sh CMD...: run CMD with GLOO_SOCKET_IFNAME set to this node's own
# preflight choice. For job steps that srun starts on several nodes, whose chosen
# interface names can differ (e.g. nodes from two partitions). IFNAME_BY_NODE,
# from use_preflight_ifname in common.sh, holds one name per node in SLURM_NODEID
# order; the preflight's master is node 0. A GLOO_SOCKET_IFNAME already set wins.
# Without IFNAME_BY_NODE, CMD just runs.
set -euo pipefail
read -ra names <<<"${IFNAME_BY_NODE:-}"
name=${names[${SLURM_NODEID:-0}]:-}
if [ -n "$name" ] && [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
    export GLOO_SOCKET_IFNAME="$name"
    echo "[node_ifname.sh] node ${SLURM_NODEID:-0}: GLOO_SOCKET_IFNAME=$name" >&2
fi
exec "$@"
