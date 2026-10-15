# Pipeline training across vendors

**Status: works on CPU in simulation mode. Nothing here has run on GPUs yet;
the run that validates it is Explorer's step 07 with `STEPS_07=pipeline`
(below).**

A model's layers cut into stages on GPUs of different vendors: stage 0 on an
NVIDIA GPU, stage 1 on an AMD one, the activations going forward and their
gradients back between them through gpubridge. [`train_pipeline.py`](../scripts/gpu/train_pipeline.py)
does it with PyTorch's own pipelining (`torch.distributed.pipelining`) on a
plain `backend="gpubridge"` process group; nothing in gpubridge is specific to
pipelines.

## Design

**The model.** [`llama.py`](../scripts/gpu/llama.py): a LLaMA-style decoder
(RMSNorm, rotary embeddings, grouped-query attention, SwiGLU, untied output
layer), `llama-125m` (134M parameters) to `llama-1b` (1.03B), sized so a stage
of it, or the whole model with float32 AdamW state, fits a 32 GB V100 or MI50.
Every rank builds the whole model from one seed on the CPU, keeps its stage
and drops the rest, so every stage starts from the same weights as one GPU
would. Synthetic tokens follow a fixed rule with a little noise
(`llama.global_batch`), so the loss falls and two curves can be compared; a
small real dataset isn't wired in yet.

**Stages.** Stage *i* holds a contiguous range of the blocks; stage 0 also the
embedding, the last stage the final norm and output layer. The split is even
(`--split 7,5` sets it by hand, in stage order); automatic balancing is prompt
4's work. Stages go to the NVIDIA ranks first, then the AMD ones, each in rank
order, whatever ranks the launcher handed out (torch's c10d rendezvous doesn't
promise the NVIDIA node rank 0), and the pipeline's neighbours follow that
order; the record lists each stage's rank, vendor and blocks.

**Communication.** `PipelineStage` sends each microbatch's activations to the
next stage and receives its gradient back with `dist.isend`/`irecv` on the
default group. On `backend="gpubridge"` these are the backend's point-to-point
operations ([TORCH_BACKEND.md](TORCH_BACKEND.md#point-to-point-cpu-only)):
native NCCL/RCCL within an island, host-staged across the bridge between
islands, bit-exact either way. The stages' tensor shapes are given up front
(static mode), so torch infers nothing over the network; the activations that
cross a stage boundary are marked as needing gradients, which is how torch
knows to send their gradients back.

**Schedules.** GPipe (`--schedule gpipe`: every microbatch forward, then every
backward) and 1F1B (`1f1b`: one forward, one backward, alternating once the
pipeline is full), with `--microbatches` microbatches of the global batch per
step and the gradients averaged over them.

**What is recorded.** Each step's loss (the mean of the microbatches' losses
on the last stage) and time (the slowest stage's); tokens/s from the median
step; per stage its blocks, the forward and backward of one microbatch through
it alone, and the time one microbatch's activations take to arrive from the
previous stage, each pair of neighbouring stages timed on its own. From those, two estimates: each stage's idle time per step (the
step less its microbatches' compute) and GPipe's ideal bubble share,
(stages − 1) / (microbatches + stages − 1). The baseline is
[`train_dp.py`](../scripts/gpu/train_dp.py) `--mode single` on one GPU, with the
same model, global batch, seed and `--micro` equal to the microbatches, so the
two compute the same mean gradient each step.

## How to run

On CPU, two simulated islands:

```bash
GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=2 \
    scripts/gpu/train_pipeline.py --model llama-tiny --steps 20 --json /tmp/pp.json
```

On Explorer, a V100 (stage 0) and an MI50 (stage 1) in one heterogeneous job,
with the MI50 on the `venv-rocm63` venv (the build whose rocBLAS has gfx906
kernels), and the single-V100 baseline:

```bash
export VENV_ROCM=$WORK_DIR/venv-rocm63
STEPS_07=pipeline TIME_07=00:55:00 scripts/gpu/explorer/submit.sh 07     # gpipe, then 1f1b
DP_MICRO=4 scripts/gpu/explorer/submit.sh dp-baseline
python scripts/gpu/dp_report.py results/explorer/07_mixed_hetjob_pp \
    results/explorer/dp_baseline_nvidia --out results/explorer/pp-report
```

`PP_RUNS` picks the schedules, `PP_MICROBATCHES` the microbatches (4), `PP_ARGS`
more options for every run (`--split 7,5`); `DP_MODEL`, `DP_STEPS`,
`DP_GLOBAL_BATCH` and `DP_SEQ` the model and run (llama-125m, 500 steps, 8
sequences of 512 tokens), shared with the data-parallel runs and the baseline.
`dp_report.py` puts the runs against the baseline: tokens/s, the loss curves'
largest gap and the gap over the last 100 steps, and the curves as CSV and SVG.

## What's validated where

| What | Where | Status |
| --- | --- | --- |
| The pipeline through gpubridge gives the same losses as on a plain Gloo group, bit for bit | CPU, 2 stages (one per island), GPipe and 1F1B; 4 stages over 2 + 2 islands (hops within and across islands) | passes (`tests/test_gpu_kit.py`) |
| It tracks one process with the same global batch and microbatches | CPU, 4 stages, 8 steps: within 1e-4 | passes |
| Stages follow the stage order, not the ranks | CPU, stage 0 on rank 1: the same losses as stage 0 on rank 0, bit for bit | passes |
| The step 07 runs and their records | the fake cluster | passes (`tests/test_gpu_jobs.py`) |
| Loss curves, tokens/s, per-stage compute, idle time and hop time on a V100 + MI50, 500 steps | Explorer, `STEPS_07=pipeline` | not run |
| A two-stage pipeline on one V100, as a second baseline | - | not built |

A number in this file gets written only once it was measured, with where it
ran.
