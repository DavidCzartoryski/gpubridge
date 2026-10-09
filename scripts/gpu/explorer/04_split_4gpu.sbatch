#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-04-split-4gpu
#SBATCH --time=00:30:00
# Step 04, 4 GPUs on 1 node (multigpu). Submit with: scripts/gpu/explorer/submit.sh 04
# TEST ONLY split into two 2-GPU NCCL islands and the Gloo bridge: the full GPU
# code path except RCCL. Items 3, 6, 7, 9 and 11 (native vs bridged benchmark in
# the same job). Runs both split layouts: half (ranks 0-1 | 2-3) and alternate
# (0,2 | 1,3), so islands don't just follow GPU numbering.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

GPUS=4
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"

job_setup 04_split_4gpu
for mode in half alternate; do
    export GPUBRIDGE_SPLIT_TEST=$mode
    step "check_$mode" local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/check.py" \
        --out "$RESULTS/check_$mode"
    step "summarize_$mode" python "$KIT_DIR/summarize.py" "$RESULTS/check_$mode"
done
export GPUBRIDGE_SPLIT_TEST=half
# --ops island and --phases put the island step (a reduce onto the leader) next
# to a native island all_reduce of the same size (item 13).
step bench local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench" --max-bytes "$BENCH_MAX_BYTES" --ops native,island,gpubridge --phases
finish
