#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-dp-baseline
#SBATCH --time=00:30:00
# The single-GPU baseline for step 07's data-parallel runs (STEPS_07=dp): train_dp.py
# --mode single, the same model, global batch, sequence length, steps and seed on one
# GPU alone (DP_MICRO micro-batches with accumulated gradients, if the batch won't fit).
# DP_SIDE=amd runs it on the AMD side instead (VENV_ROCM picks the ROCm venv).
# Submit with scripts/gpu/explorer/submit.sh dp-baseline or dp-baseline-amd.
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

DP_SIDE="${DP_SIDE:-nvidia}"
DP_MODEL="${DP_MODEL:-llama-125m}"
DP_STEPS="${DP_STEPS:-500}"
DP_GLOBAL_BATCH="${DP_GLOBAL_BATCH:-8}"
DP_SEQ="${DP_SEQ:-512}"
DP_MICRO="${DP_MICRO:-1}"
read -ra dp_args <<<"${DP_ARGS:-}"
case $DP_SIDE in
nvidia) ;;
amd) VENV=$VENV_ROCM ;;
*) die "DP_SIDE must be nvidia or amd, not '$DP_SIDE'" ;;
esac
job_setup "dp_baseline_$DP_SIDE"
[ "$DP_SIDE" = nvidia ] || rocm_asic_ids "$VENV/bin/python"
step probe python "$KIT_DIR/probe.py" --out "$RESULTS/probe"
step dp_single python "$KIT_DIR/train_dp.py" --mode single --model "$DP_MODEL" \
    --steps "$DP_STEPS" --global-batch "$DP_GLOBAL_BATCH" --seq-len "$DP_SEQ" \
    --micro "$DP_MICRO" --json "$RESULTS/dp_single.json" "${dp_args[@]+"${dp_args[@]}"}"
finish
