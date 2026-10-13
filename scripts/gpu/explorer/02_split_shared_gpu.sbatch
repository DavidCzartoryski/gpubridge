#!/usr/bin/env bash
#SBATCH --job-name=gpubridge-02-split-shared-gpu
#SBATCH --time=00:25:00
# Step 02, 1 GPU (free gpu partition). Submit with: scripts/gpu/explorer/submit.sh 02
# TEST ONLY split: 2 processes share one GPU, each its own single-rank NCCL island,
# joined by the Gloo bridge. Covers new_group(backend="nccl") on non-member ranks
# (item 3), device tensors through the bridge (item 6), barrier on CUDA (item 10)
# and the check that no island puts two ranks on one GPU (item 24).
set -euo pipefail
REPO="${GPUBRIDGE_REPO:?submit with scripts/gpu/explorer/submit.sh}"
# shellcheck source=../common.sh
source "$REPO/scripts/gpu/common.sh" "$@"
# shellcheck source=job_env.sh
source "$KIT_DIR/explorer/job_env.sh"

PROCS=2   # ranks sharing the GPU
# The deliberate failing run below fails within seconds when the check works.
SHARED_GPU_CHECK_SECONDS="${SHARED_GPU_CHECK_SECONDS:-60}"
SHARED_GPU_ERROR="would share a GPU"

job_setup 02_split_shared_gpu
export GPUBRIDGE_SPLIT_TEST=half
step check local_torchrun --nproc-per-node="$PROCS" "$KIT_DIR/check.py" \
    --out "$RESULTS/check" --large-numel $((16 * 1024 * 1024))
step summarize python "$KIT_DIR/summarize.py" "$RESULTS/check"
# Item 24: 4 ranks with alternate put ranks 0 and 2, and 1 and 3, on this GPU in
# one island each, which NCCL refuses. init() must fail on every rank first,
# naming them: the step passes only if that error shows. Real GPUs only:
# simulation has no GPUs to compare.
if [ -z "${GPUBRIDGE_VENDOR:-}${GPUBRIDGE_CPU_ONLY:-}" ]; then
    negative_step shared_gpu_check "$SHARED_GPU_CHECK_SECONDS" "$SHARED_GPU_ERROR" \
        env_local_torchrun GPUBRIDGE_SPLIT_TEST=alternate --nproc-per-node=4 \
        "$KIT_DIR/check.py" --out "$RESULTS/shared_gpu_check" \
        --expect-init-error "$SHARED_GPU_ERROR"
    step summarize_shared_gpu python "$KIT_DIR/summarize.py" "$RESULTS/shared_gpu_check"
fi
# GPU-only correctness tests (tests/test_gpu.py): every policy against flat-gloo,
# fp32/fp16/bf16, odd and large sizes, async on non-default streams.
step gpu_tests limited python -m pytest -m gpu "$REPO_DIR/tests/test_gpu.py" \
    "$REPO_DIR/tests/test_compression_kernels.py" \
    "$REPO_DIR/tests/test_torch_backend.py" -q -p no:cacheprovider \
    --junitxml="$RESULTS/gpu_tests.xml"
finish
