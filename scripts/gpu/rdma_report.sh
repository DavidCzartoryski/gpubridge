#!/usr/bin/env bash
# What this node has for GPU-to-NIC RDMA, one file each in DIR, next to net_report.sh's
# ports: the kernel and its dma-buf and peer-to-peer options, the RDMA and GPU kernel
# modules loaded (and whether nvidia_peermem or nv_peer_mem exist at all), rdma-core
# and whether its libibverbs has ibv_reg_dmabuf_mr, the RDMA links, UCX (ucx_info -v
# and -d) and, for AMD, whether the HSA runtime exports dma-bufs. A tool or file that
# isn't there leaves a one-line note instead. Never fails.
#
#   scripts/gpu/rdma_report.sh DIR [PYTHON]    PYTHON: the venv's, to find its HSA runtime
set -uo pipefail
dir=${1:?usage: rdma_report.sh DIR [PYTHON]}
python=${2:-}
mkdir -p "$dir" || exit 0

note() { echo "$*: not found on this node"; }

{
    uname -r
    grep -h '^PRETTY_NAME=' /etc/os-release 2>/dev/null
    config=/boot/config-$(uname -r)
    if [ -r "$config" ]; then
        grep -E '^(# )?CONFIG_(DMA_SHARED_BUFFER|DMABUF_MOVE_NOTIFY|PCI_P2PDMA|HSA_AMD_P2P|INFINIBAND_USER_MEM|INFINIBAND_ON_DEMAND_PAGING)[= ]' "$config"
    else
        note "$config"
    fi
} >"$dir/kernel.txt" 2>&1

{
    echo "# loaded (/proc/modules)"
    grep -E '^(nvidia|nvidia_peermem|nv_peer_mem|amdgpu|amdkfd|amdp2p|ib_core|ib_uverbs|mlx5_core|mlx5_ib|rdma_ucm|rdma_cm) ' \
        /proc/modules 2>/dev/null | cut -d' ' -f1-3 || true
    echo "# installed for this kernel (modinfo)"
    modinfo=$(command -v modinfo || echo /usr/sbin/modinfo)
    for module in nvidia_peermem nv_peer_mem amdp2p nvidia amdgpu; do
        if [ -x "$modinfo" ] && "$modinfo" "$module" >/dev/null 2>&1; then
            printf '%s: version %s, license %s\n' "$module" \
                "$("$modinfo" -F version "$module" 2>/dev/null | head -1)" \
                "$("$modinfo" -F license "$module" 2>/dev/null | head -1)"
        else
            echo "$module: no module for this kernel"
        fi
    done
    [ -r /proc/driver/nvidia/version ] && head -1 /proc/driver/nvidia/version
} >"$dir/modules.txt" 2>&1

{
    if command -v rpm >/dev/null 2>&1; then
        rpm -q rdma-core libibverbs ucx 2>&1
    elif command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W rdma-core libibverbs1 libucx0 2>&1
    fi
    verbs=$(ldconfig -p 2>/dev/null | awk '/libibverbs\.so\.1 /{print $NF; exit}')
    if [ -n "$verbs" ] && command -v nm >/dev/null 2>&1; then
        if nm -D --defined-only "$verbs" 2>/dev/null | grep -q ' ibv_reg_dmabuf_mr'; then
            echo "$verbs: has ibv_reg_dmabuf_mr"
        else
            echo "$verbs: no ibv_reg_dmabuf_mr"
        fi
    else
        note "libibverbs.so.1 (or nm)"
    fi
} >"$dir/rdma_core.txt" 2>&1

if command -v rdma >/dev/null 2>&1; then
    rdma link show >"$dir/rdma_link.txt" 2>&1
else
    note rdma >"$dir/rdma_link.txt"
fi

{
    if command -v ucx_info >/dev/null 2>&1; then
        ucx_info -v
        ucx_info -d
    else
        note ucx_info
        ldconfig -p 2>/dev/null | grep -E 'libuc[pt]\.so' || note "libucp.so"
    fi
} >"$dir/ucx.txt" 2>&1

# AMD: ROCm exports a GPU buffer as a dma-buf through hsa_amd_portable_export_dmabuf
# (in torch's bundled HSA runtime for a pip-installed ROCm build).
{
    torch_lib=
    [ -z "$python" ] || torch_lib=$("$python" -c \
        'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)
    found=0
    for lib in ${torch_lib:+"$torch_lib"/libhsa-runtime64.so*} /opt/rocm*/lib/libhsa-runtime64.so*; do
        [ -f "$lib" ] || continue
        found=1
        where=system  # no paths in the record
        case $lib in "${torch_lib:-/nonexistent}"/*) where=torch ;; esac
        if ! command -v nm >/dev/null 2>&1; then
            note nm
        elif nm -D --defined-only "$lib" 2>/dev/null | grep -q ' hsa_amd_portable_export_dmabuf'; then
            echo "${lib##*/} ($where): has hsa_amd_portable_export_dmabuf"
        else
            echo "${lib##*/} ($where): no hsa_amd_portable_export_dmabuf"
        fi
    done
    [ "$found" = 1 ] || note libhsa-runtime64.so
} >"$dir/rocm_dmabuf.txt" 2>&1

loaded=$(grep -h -E -o '^(nvidia_peermem|nv_peer_mem|amdp2p) ' "$dir/modules.txt" | tr -d ' ' |
    tr '\n' ' ')
printf 'rdma report: %s\n' "${loaded:-no peer-memory module loaded}"
exit 0
