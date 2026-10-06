#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-06-scaling
#SBATCH --time=00:30:00
# Step 06, 4 GPUs on 1 node (multigpu). Submit with: scripts/gpu/explorer/submit.sh 06
# Data-parallel training (examples/train_synthetic.py, gradients synced with
# gpubridge.all_reduce) on 1, 2, 3 and 4 GPUs: samples/sec and scaling
# efficiency. The table in summary.md is what the multigpu access request asks for.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

MAX_GPUS=4
TRAIN_ARGS=(--steps 100 --warmup-steps 10 --batch-size 256 --dim 2048 --layers 6)

job_setup 06_scaling
for n in $(seq 1 "$MAX_GPUS"); do
    step "train_n$n" local_torchrun --nproc-per-node="$n" \
        "$REPO_DIR/examples/train_synthetic.py" "${TRAIN_ARGS[@]}" \
        --json "$RESULTS/train-n$n.json"
done
step summarize python "$KIT_DIR/summarize.py" "$RESULTS"
finish
