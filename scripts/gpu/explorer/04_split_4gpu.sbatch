#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-04-split-4gpu
#SBATCH --time=00:40:00
# Step 04, 4 GPUs on 1 node (multigpu). Submit with: scripts/gpu/explorer/submit.sh 04
# TEST ONLY split into two 2-GPU NCCL islands and the Gloo bridge: the full GPU
# code path except RCCL. Items 3, 6, 7, 9 and 11 (native vs bridged benchmark in
# the same job). Runs both split layouts: half (ranks 0-1 | 2-3) and alternate
# (0,2 | 1,3), so islands don't just follow GPU numbering. Then the opt-in
# policies (OPT_IN_POLICIES) and a thresholds file for auto-tuned.
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
# GPU-only correctness tests (tests/test_gpu.py): every policy against flat-gloo,
# fp32/fp16/bf16, odd and large sizes, async on non-default streams.
step gpu_tests limited python -m pytest -m gpu "$REPO_DIR/tests/test_gpu.py" -q \
    -p no:cacheprovider --junitxml="$RESULTS/gpu_tests.xml"
export GPUBRIDGE_SPLIT_TEST=half
# --ops island and --phases put the island step (a reduce onto the leader) next
# to a native island all_reduce of the same size (item 13).
step bench local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench" --max-bytes "$BENCH_MAX_BYTES" --ops native,island,gpubridge --phases
# Opt-in policies (items 17 to 21): the same check under each, then one benchmark
# that times every candidate side by side and writes a thresholds file, then the
# check again under auto-tuned with that file.
for policy in $OPT_IN_POLICIES; do
    step "check_$policy" local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/check.py" \
        --out "$RESULTS/check_$policy" --policy "$policy"
    step "summarize_$policy" python "$KIT_DIR/summarize.py" "$RESULTS/check_$policy"
done
step bench_candidates local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/bench_all_reduce.py" \
    --out "$RESULTS/bench_candidates" --max-bytes "$BENCH_MAX_BYTES" --ops gpubridge \
    --write-thresholds "$RESULTS/thresholds.json"
step check_auto_tuned env_local_torchrun GPUBRIDGE_THRESHOLDS="$RESULTS/thresholds.json" \
    --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_auto_tuned" \
    --policy auto-tuned
# DDP with gpubridge.ddp_comm_hook across two islands (item 22). It exits non-zero
# if the ranks' parameters diverge.
step train_ddp local_torchrun --nproc-per-node="$GPUS" "$REPO_DIR/examples/train_synthetic.py" \
    --json "$RESULTS/train_ddp.json"
finish
