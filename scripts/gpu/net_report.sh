#!/usr/bin/env bash
# This node's network, one file each in DIR, for bench_matrix.py: the InfiniBand
# devices (/sys/class/infiniband), each port's state, rate and link layer (ibstat,
# ibv_devinfo: InfiniBand, or Ethernet for RoCE) and the interfaces (ip -br addr).
# A tool that isn't installed leaves a one-line note instead. Never fails.
#
#   scripts/gpu/net_report.sh DIR
set -uo pipefail
dir=${1:?usage: net_report.sh DIR}
mkdir -p "$dir" || exit 0
if [ -d /sys/class/infiniband ]; then
    ls -1 /sys/class/infiniband >"$dir/sys_class_infiniband.txt" 2>/dev/null
else
    : >"$dir/sys_class_infiniband.txt"
fi
for tool in ibstat ibv_devinfo; do
    if command -v "$tool" >/dev/null 2>&1; then
        "$tool" >"$dir/$tool.txt" 2>&1
    else
        echo "$tool: not installed on this node" >"$dir/$tool.txt"
    fi
done
if command -v ip >/dev/null 2>&1; then
    ip -br addr >"$dir/ip_addr.txt" 2>&1
else
    echo "ip: not installed on this node" >"$dir/ip_addr.txt"
fi
printf 'network report: %s InfiniBand device(s) (%s)\n' \
    "$(wc -l <"$dir/sys_class_infiniband.txt" | tr -d ' ')" \
    "$(tr '\n' ' ' <"$dir/sys_class_infiniband.txt")"
exit 0
