#!/usr/bin/env bash
# Step 00: record the GPU types in each partition, to see whether Explorer has
# AMD GPU nodes (in "sharing") for a mixed-vendor job of its own (step 07). It
# only queries Slurm, so it runs right here on the login node, in seconds.
#
#   scripts/gpu/explorer/submit.sh 00        (or: 00_partitions.sh [PARTITION...])
#
# The first partition is the one checked for AMD GPUs. Writes
# results/explorer/00_partitions/:
#   sinfo-<partition>.txt  nodes, CPUs, memory, features, GRES and state
#   gres-<partition>.txt   each node's full GRES string (the table truncates it)
#   gpu-types.txt          GPU types per partition, labelled amd, nvidia or unknown
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

[ "${#ARGS[@]}" -gt 0 ] || ARGS=(sharing gpu multigpu)
SINFO_FORMAT="${SINFO_FORMAT:-%20N %10c %10m %25f %10G %10t}"
RESULTS="${RESULTS_ROOT:-$REPO_DIR/results/explorer}/00_partitions"
AMD_PARTITION=${ARGS[0]}

# gpu_types PARTITION FILE: one line per GPU type in a "node gres" listing:
# partition, type, vendor, nodes, GPUs.
gpu_types() {
    awk -v partition="$1" '
    function vendor(type, t) {
        t = tolower(type)
        if (t ~ /amd|radeon|instinct|gfx|mi[0-9]/) return "amd"
        if (t ~ /nvidia|tesla|quadro|titan|rtx|gtx|gh200|l40|^[vahpkbtl][0-9]+/) return "nvidia"
        return "unknown"
    }
    {
        gres = $2
        gsub(/\([^)]*\)/, "", gres)  # socket bindings, e.g. (S:0-1)
        n = split(gres, items, ",")
        for (i = 1; i <= n; i++) {
            k = split(items[i], f, ":")
            if (f[1] != "gpu") continue
            type = (k >= 3) ? f[2] : "(untyped)"
            nodes[type]++
            gpus[type] += f[k]
        }
    }
    END {
        for (type in nodes)
            printf "%-12s %-20s %-8s %5d %5d\n", partition, type, vendor(type), nodes[type], gpus[type]
    }' "$2" | sort
}

if [ "$DRY_RUN" = 0 ] && ! command -v sinfo >/dev/null 2>&1; then
    die "sinfo not found: run step 00 on an Explorer login node"
fi
run mkdir -p "$RESULTS"
for partition in "${ARGS[@]}"; do
    capture "$RESULTS/sinfo-$partition.txt" sinfo -p "$partition" -o "$SINFO_FORMAT"
    capture "$RESULTS/gres-$partition.txt" sinfo -h -N -p "$partition" -o "%N %G"
done
[ "$DRY_RUN" = 1 ] && exit 0

types="$RESULTS/gpu-types.txt"
{
    printf "%-12s %-20s %-8s %5s %5s\n" partition gpu_type vendor nodes gpus
    for partition in "${ARGS[@]}"; do
        gpu_types "$partition" "$RESULTS/gres-$partition.txt"
    done
} >"$types"
amd=$(awk -v p="$AMD_PARTITION" '$1 == p && $3 == "amd" { printf "%s%s", sep, $2; sep = ", " }' \
    "$types")
if [ -n "$amd" ]; then
    verdict="AMD GPUs in $AMD_PARTITION: $amd. Set GPU_TYPE_AMD to one of them for setup-rocm and 07."
else
    verdict="No AMD GPU type found in $AMD_PARTITION; check its unknown types and $RESULTS/sinfo-$AMD_PARTITION.txt by hand."
fi
echo "$verdict" >>"$types"
cat "$types"
log "results in $RESULTS"
