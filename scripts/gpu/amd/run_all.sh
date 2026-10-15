#!/usr/bin/env bash
# Everything for a rented AMD GPU machine in one command: set up, record the
# hardware, run every check, benchmark, package the results and print what to
# download. Steps that fail are logged and skipped, so one problem doesn't
# waste the session. Safe to rerun; setup skips what is already installed.
#
#   git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
#   bash scripts/gpu/amd/run_all.sh            # add --dry-run to only print commands
#
# Needs Linux with ROCm and internet. Expect about 20-30 minutes with 8 GPUs,
# most of it the PyTorch download (skip it with TORCH_FLAVOR=system on an image
# that already has a ROCm PyTorch). See docs/GPU_VALIDATION_RUNBOOK.md.
set -euo pipefail
# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh" "$@"

# ---- Settings (override with environment variables) ---------------------------
TORCH_FLAVOR="${TORCH_FLAVOR:-rocm}"     # rocm: torch 2.14.1+rocm7.2 (6+ GB); system: reuse
export TORCH_VERSION="${TORCH_VERSION:-2.14.1}" ROCM_INDEX="${ROCM_INDEX:-rocm7.2}"
WORK_DIR="${WORK_DIR:-$HOME/gpubridge-gpu}"
TARGET="${TARGET:-amd-$(hostname -s)-$(date +%Y%m%d-%H%M)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-1G}"
BENCH_TRIALS="${BENCH_TRIALS:-10}"
SCALING_MAX_GPUS="${SCALING_MAX_GPUS:-4}"
SKIP_SETUP="${SKIP_SETUP:-0}"            # 1 = reuse an existing venv (e.g. on a rerun)
PACKAGE_RESULTS="${PACKAGE_RESULTS:-1}"  # 0 = no tarball (e.g. on a cluster's shared filesystem)
# Opt-in collective policies also checked (space-separated).
OPT_IN_POLICIES="${OPT_IN_POLICIES:-pipelined-reduce-bridge-broadcast sharded-bridge}"
# Training throughput (train_bench.py): models, and extra flags such as "--batch-size 16".
TRAIN_BENCH_MODELS="${TRAIN_BENCH_MODELS:-gpt2-small resnet50}"
read -ra train_args <<<"${TRAIN_BENCH_ARGS:-}"
# -------------------------------------------------------------------------------

RESULTS="$RESULTS_ROOT/$TARGET"
export RESULTS
VENV="${VENV:-$WORK_DIR/venv}"
log "results go to $RESULTS"

if [ "$SKIP_SETUP" != 1 ]; then
    step setup "$KIT_DIR/setup_env.sh" --flavor "$TORCH_FLAVOR" --venv "$VENV"
fi
activate_venv "$VENV"
rocm_asic_ids python  # the ROCm wheel's GPU-name table (common.sh)
capture "$RESULTS/rocm-smi.txt" rocm-smi
capture "$RESULTS/rocm-smi-detail.txt" rocm-smi --showproductname --showdriverversion \
    --showbus --showmeminfo vram
capture "$RESULTS/amd-smi-static.txt" amd-smi static
capture "$RESULTS/rocminfo.txt" rocminfo

GPUS=$(gpu_count)
log "$GPUS GPU(s) visible"
step probe limited python "$KIT_DIR/probe.py" --out "$RESULTS/probe"
step device_map python "$KIT_DIR/device_map.py" --out "$RESULTS/device_map"
# Item 12: one more process than GPUs. The extra rank's LOCAL_RANK has no GPU,
# so init() must fail on every rank, naming it. Real GPUs only: simulation
# has no devices to check.
if [ -z "${GPUBRIDGE_VENDOR:-}${GPUBRIDGE_CPU_ONLY:-}" ]; then
    step local_rank_check local_torchrun --nproc-per-node=$((GPUS + 1)) "$KIT_DIR/check.py" \
        --out "$RESULTS/local_rank_check" --expect-init-error "LOCAL_RANK=$GPUS asks for GPU $GPUS"
fi

# At NCCL_DEBUG=INFO (common.sh), RCCL logs its version to $RESULTS/nccl/check_1gpu/:
# proof that "nccl" runs RCCL (item 4).
step check_1gpu local_torchrun --nproc-per-node=1 "$KIT_DIR/check.py" \
    --out "$RESULTS/check_1gpu"
rccl_version() {
    run grep -h -m 1 -i -o "RCCL version[^,]*" "$RESULTS/nccl/check_1gpu/"*.log
}
step rccl_version rccl_version
# GPU-only correctness tests (tests/test_gpu.py): every policy against flat-gloo,
# fp32/fp16/bf16, odd and large sizes, async on non-default streams. --durations=15:
# the slowest tests and their times end the log.
step gpu_tests limited python -m pytest -m gpu "$REPO_DIR/tests/test_gpu.py" \
    "$REPO_DIR/tests/test_compression_kernels.py" \
    "$REPO_DIR/tests/test_torch_backend.py" -q -p no:cacheprovider --durations=15 \
    --junitxml="$RESULTS/gpu_tests.xml"
# Bridge compression kernels (item 28): the same bits as the CPU reference, and their
# throughput against the GPU's memory bandwidth. One GPU is enough.
step bench_compression_kernels limited python "$KIT_DIR/bench_compression.py" \
    --out "$RESULTS/bench_compression" --max-bytes "$BENCH_MAX_BYTES"

checks=("$RESULTS/check_1gpu")
if [ "$GPUS" -ge 2 ]; then
    step check_island local_torchrun --nproc-per-node="$GPUS" "$KIT_DIR/check.py" \
        --out "$RESULTS/check_island"
    step check_split env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_split"
    checks+=("$RESULTS/check_island" "$RESULTS/check_split")
    step bench env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench" --ops native,island,gpubridge,flat-gloo \
        --phases --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS"
    # Pinned staging for the default bridge (item 25), against the pageable copies above.
    step check_pinned env_local_torchrun GPUBRIDGE_SPLIT_TEST=half GPUBRIDGE_PINNED_STAGING=1 \
        --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_pinned"
    checks+=("$RESULTS/check_pinned")
    step bench_pinned env_local_torchrun GPUBRIDGE_SPLIT_TEST=half GPUBRIDGE_PINNED_STAGING=1 \
        --nproc-per-node="$GPUS" "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench_pinned" \
        --ops gpubridge --phases --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS"
    # The bridge step with and without compression (item 29): where it starts paying off.
    step bench_compressed env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench_compressed" --ops gpubridge \
        --compression none,bf16,fp8 --phases --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS"
    # Opt-in policies (items 17 to 21), each under the same split-test check, then
    # every candidate timed side by side, writing a thresholds file for auto-tuned.
    for policy in $OPT_IN_POLICIES; do
        step "check_$policy" env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
            --nproc-per-node="$GPUS" "$KIT_DIR/check.py" --out "$RESULTS/check_$policy" \
            --policy "$policy"
        checks+=("$RESULTS/check_$policy")
    done
    step bench_candidates env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$KIT_DIR/bench_all_reduce.py" --out "$RESULTS/bench_candidates" --ops gpubridge \
        --max-bytes "$BENCH_MAX_BYTES" --trials "$BENCH_TRIALS" \
        --write-thresholds "$RESULTS/thresholds.json"
    # DDP with gpubridge.ddp_comm_hook across two islands (item 22).
    step train_ddp_split env_local_torchrun GPUBRIDGE_SPLIT_TEST=half --nproc-per-node="$GPUS" \
        "$REPO_DIR/examples/train_synthetic.py" --json "$RESULTS/train_ddp_split.json"
    # gpubridge as the torch.distributed backend (item 30), FSDP2, across two islands.
    step train_fsdp2_backend env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node="$GPUS" "$REPO_DIR/examples/fsdp2_torch_backend.py" \
        --json "$RESULTS/train_fsdp2_backend.json"
    step verify_fsdp2_backend env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node="$GPUS" "$REPO_DIR/examples/fsdp2_torch_backend.py" --verify
    max=$((GPUS < SCALING_MAX_GPUS ? GPUS : SCALING_MAX_GPUS))
    for n in $(seq 1 "$max"); do
        step "train_n$n" local_torchrun --nproc-per-node="$n" \
            "$REPO_DIR/examples/train_synthetic.py" --json "$RESULTS/scaling/train-n$n.json"
    done
    step summarize_scaling python "$KIT_DIR/summarize.py" "$RESULTS/scaling"
else
    # One GPU: two ranks share it, each its own single-rank RCCL island.
    step check_split_shared_gpu env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
        --nproc-per-node=2 "$KIT_DIR/check.py" --out "$RESULTS/check_split_shared_gpu"
    checks+=("$RESULTS/check_split_shared_gpu")
fi
# Training throughput (train_bench.py, DDP with gpubridge.ddp_comm_hook): each model on one
# GPU alone, then with 2+ GPUs on all of them as one RCCL island, and split into two
# islands joined by the bridge (the cost of the bridge on one node).
for model in $TRAIN_BENCH_MODELS; do
    step "train_bench_${model}_1gpu" local_torchrun --nproc-per-node=1 "$KIT_DIR/train_bench.py" \
        --model "$model" --json "$RESULTS/train_bench_${model}_1gpu.json" \
        "${train_args[@]+"${train_args[@]}"}"
    if [ "$GPUS" -ge 2 ]; then
        step "train_bench_${model}_island" local_torchrun --nproc-per-node="$GPUS" \
            "$KIT_DIR/train_bench.py" --model "$model" \
            --json "$RESULTS/train_bench_${model}_island.json" "${train_args[@]+"${train_args[@]}"}"
        step "train_bench_${model}_split" env_local_torchrun GPUBRIDGE_SPLIT_TEST=half \
            --nproc-per-node="$GPUS" "$KIT_DIR/train_bench.py" --model "$model" \
            --json "$RESULTS/train_bench_${model}_split.json" "${train_args[@]+"${train_args[@]}"}"
    fi
done
step summarize python "$KIT_DIR/summarize.py" "${checks[@]}"

[ "$PACKAGE_RESULTS" = 0 ] || package_results "$RESULTS"
finish
