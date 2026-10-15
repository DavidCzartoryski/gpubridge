#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-train-bench
#SBATCH --time=00:30:00
# Training throughput on one GPU alone: train_bench.py with one rank (DDP and
# gpubridge.ddp_comm_hook, nothing to sync), for each model in TRAIN_BENCH_MODELS. The
# baseline for step 07's train_bench steps: does a second GPU through gpubridge raise
# samples/s above this? probe.py first; its blas stage says whether matmuls run here.
# Against a step-07 run at --global-batch N (with or without --balance), run this with
# the same TRAIN_BENCH_ARGS: one rank takes all N samples.
# Submit with scripts/gpu/explorer/submit.sh train-nvidia or train-amd (VENV_ROCM picks
# the ROCm venv).
# With TRAIN_GPUS=N (2 or more; submit.sh train-multi, on multigpu), each model also
# runs on all N GPUs as one native island (train_bench_<model>_n<N>), then split into
# two islands of N/2 joined by the bridge (train_bench_<model>_split, a split test:
# GPUBRIDGE_SPLIT_TEST=half). Several GPUs per island, as on bigger hardware.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

TRAIN_SIDE="${TRAIN_SIDE:-nvidia}"
TRAIN_BENCH_MODELS="${TRAIN_BENCH_MODELS:-gpt2-small resnet50}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
case $TRAIN_GPUS in
1) job=train_bench_$TRAIN_SIDE ;;
[2-9] | [1-9][0-9]) job=train_bench_${TRAIN_SIDE}_n$TRAIN_GPUS ;;
*) die "TRAIN_GPUS must be a GPU count, not '$TRAIN_GPUS'" ;;
esac
read -ra train_args <<<"${TRAIN_BENCH_ARGS:-}"
case $TRAIN_SIDE in
nvidia) ;;
amd) VENV=$VENV_ROCM ;;
*) die "TRAIN_SIDE must be nvidia or amd, not '$TRAIN_SIDE'" ;;
esac
job_setup "$job"
[ "$TRAIN_SIDE" = nvidia ] || rocm_asic_ids "$VENV/bin/python"
step probe python "$KIT_DIR/probe.py" --out "$RESULTS/probe"
for model in $TRAIN_BENCH_MODELS; do
    step "train_bench_$model" local_torchrun --nproc-per-node=1 "$KIT_DIR/train_bench.py" \
        --model "$model" --json "$RESULTS/train_bench_$model.json" \
        "${train_args[@]+"${train_args[@]}"}"
    [ "$TRAIN_GPUS" -gt 1 ] || continue
    step "train_bench_${model}_n$TRAIN_GPUS" local_torchrun --nproc-per-node="$TRAIN_GPUS" \
        "$KIT_DIR/train_bench.py" --model "$model" \
        --json "$RESULTS/train_bench_${model}_n$TRAIN_GPUS.json" "${train_args[@]+"${train_args[@]}"}"
    step "train_bench_${model}_split" env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node="$TRAIN_GPUS" "$KIT_DIR/train_bench.py" --model "$model" \
        --json "$RESULTS/train_bench_${model}_split.json" "${train_args[@]+"${train_args[@]}"}"
done
finish
