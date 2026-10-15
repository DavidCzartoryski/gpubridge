# Pipeline training across vendors

**Status: trains on a V100 and an MI50 on Explorer**, GPipe and 1F1B, 1,000
steps each ([results](#results-a-v100-and-an-mi50)); bit for bit like plain
Gloo on CPU.

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

**Every rank on one PyTorch release.** `torch.distributed.pipelining`'s
schedules open with a handshake between neighbouring stages, and it isn't the
same in every release: 2.14.1's first messages don't match 2.9.1's. So
`train_pipeline.py` stops before building the stages if the ranks' releases
differ. 2.9's handshake also sends each stage's index as an int64 into the
neighbour's float32 buffer; NCCL moves the bytes regardless, but gpubridge's
point-to-point (and Gloo) refuse a payload that isn't the receiver's size and
dtype, so `train_pipeline.py` has 2.9 send a float32, as 2.14.1 does. The CPU
tests pass on 2.8.0, 2.9.1 and 2.14.1.

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
with the MI50 on the `venv-rocm63` venv (torch 2.9.1, the build whose rocBLAS
has gfx906 kernels) and the V100 on a CUDA venv of the same release, and the
single-V100 baseline:

```bash
export VENV_ROCM=$WORK_DIR/venv-rocm63
VENV=$WORK_DIR/venv-cu291 TORCH_VERSION=2.9.1 scripts/gpu/explorer/submit.sh setup   # once
VENV=$WORK_DIR/venv-cu291 STEPS_07=pipeline DP_STEPS=1000 TIME_07=00:55:00 \
    scripts/gpu/explorer/submit.sh 07                                     # gpipe, then 1f1b
DP_STEPS=1000 DP_MICRO=4 scripts/gpu/explorer/submit.sh dp-baseline
python scripts/gpu/dp_report.py results/explorer/07_mixed_hetjob_pp \
    results/explorer/dp_baseline_nvidia --out results/explorer/pp-report
```

`PP_RUNS` picks the schedules, `PP_MICROBATCHES` the microbatches (4), `PP_ARGS`
more options for every run (`--split 7,5`); `DP_MODEL`, `DP_STEPS`,
`DP_GLOBAL_BATCH` and `DP_SEQ` the model and run (llama-125m, 500 steps, 8
sequences of 512 tokens), shared with the data-parallel runs and the baseline.
`dp_report.py` puts the runs against the baseline: tokens/s, the loss curves'
largest gap and the gap over the last 100 steps, and the curves as CSV and SVG.

## Results: a V100 and an MI50

Explorer job 11109376, commit dc47b6a (clean on both ranks): stage 0 on a V100
(torch 2.9.1+cu126), stage 1 on an MI50 (2.9.1+rocm6.3), point-to-point
through `backend="gpubridge"`, host-staged over TCP on `internal.2137`
(preflight: 1164 to 1174 MB/s each way, 0.12 to 0.13 ms round trip). `llama-125m`, an
even split of 6 + 6 blocks (67.05M parameters each; stage 0 also has the
embedding, stage 1 the final norm and output layer), 4 microbatches of 2
sequences x 512 tokens a step, AdamW 3e-4, seed 0, 1,000 steps. The baselines
are `train_dp.py --mode single` on one V100 (2.14.1+cu126; jobs 11109239,
11109665, 11109872 and 11110316), the same model, data and seed, with the
global batch in 1, 2, 4 or 8 micro-batches: the same mean gradient, summed in
a different order. Medians of steps 6 to 1,000.

| Run | Tokens/s | Step (ms) | Against one V100, 4 micro-batches | Loss at step 1,000 | Mean gap to the V100 runs, last 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 alone, 1 micro-batch | 14,264 | 287.1 | 1.14x | 1.809 | 0.0044 to 0.0056 |
| V100 alone, 2 micro-batches | 13,518 | 303.0 | 1.08x | 1.802 | 0.0041 to 0.0047 |
| V100 alone, 4 micro-batches | 12,548 | 326.4 | 1.00x | 1.806 | 0.0042 to 0.0046 |
| V100 alone, 8 micro-batches | 10,841 | 377.8 | 0.86x | 1.800 | 0.0041 to 0.0056 |
| V100 + MI50, GPipe | 13,115 | 312.3 | 1.05x | 1.823 | 0.0203 to 0.0251 |
| V100 + MI50, 1F1B | 13,051 | 313.8 | 1.04x | 1.801 | 0.0036 to 0.0052 |

Per stage, from the same runs (GPipe; 1F1B within 0.5 ms of each):

| Stage | GPU | Blocks | Forward + backward of one microbatch, alone (ms) | Idle per step (est., ms) | One microbatch's activations arriving (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | V100 | 0 to 5, and the embedding | 37.0 | 164.5 | - |
| 1 | MI50 | 6 to 11, and the output layer | 61.8 | 65.2 | 8.0 |

![Loss over 1,000 steps: the four one-V100 runs, GPipe and 1F1B](img/pipeline-loss-v100-mi50.svg)

- **The pair is about as fast as the V100 alone, not faster.** It's 1.05x the
  V100 running the same 4 micro-batches (the run that computes the same
  gradient) and 0.92x the V100 with the batch in one piece. Each hop carries
  one microbatch's activations, 3 MB, against the 536 MB of gradients that
  data parallelism sends over the same link each step ([BENCHMARKS.md](BENCHMARKS.md)).
- **The MI50's stage is the slow one**: 61.8 ms a microbatch against the
  V100's 37.0, so the V100 idles about 164 ms of each 312 ms step. Not all of
  that is the GPU: both stages have 6 blocks, but stage 1 also runs the output
  layer (768 x 32,000) and the loss. The step is close to what GPipe's schedule predicts from the
  slowest stage, (microbatches + stages - 1) x 61.8 = 309 ms (an estimate, not
  a measurement). An uneven split (`--split 7,5` or `8,4`, not run) is the
  obvious next step; balancing is prompt 4's work.
- **1F1B tracks the one-V100 runs as closely as they track each other**: over
  the last 100 steps it's 0.0036 to 0.0052 from each, and they're 0.0041 to
  0.0056 apart.
- **GPipe ends 0.02 above them**, because it left the plateau later. Every
  run sits on a plateau near ln(32000) = 10.37 before dropping, and when the
  drop starts depends on float rounding. GPipe's loss first fell below 9 at
  step 341, the other five runs' between steps 172 and 210. A 500-step
  one-V100 run with the same settings as `micro 1` above was still at 8.2 at
  step 500 (job 11107533). Two of those identical runs part at step 5 and
  are 0.1 apart by step 157. So one GPipe run doesn't show a GPipe-specific
  difference; several seeds would.
- 500 steps, which the prompt asked for, end in the middle of the drop for
  some runs, where the curves can't be compared, hence 1,000.
- The first try failed in torch's stage setup, for the two reasons in
  [Design](#design): first 2.14.1 on the V100 against 2.9.1 on the MI50, then
  2.9.1's int64 handshake.

## What's validated where

| What | Where | Status |
| --- | --- | --- |
| The pipeline through gpubridge gives the same losses as on a plain Gloo group, bit for bit | CPU, 2 stages (one per island), GPipe and 1F1B; 4 stages over 2 + 2 islands (hops within and across islands) | passes (`tests/test_gpu_kit.py`) |
| It tracks one process with the same global batch and microbatches | CPU, 4 stages, 8 steps: within 1e-4 | passes |
| Stages follow the stage order, not the ranks | CPU, stage 0 on rank 1: the same losses as stage 0 on rank 0, bit for bit | passes |
| The step 07 runs and their records | the fake cluster | passes (`tests/test_gpu_jobs.py`) |
| Loss curves, tokens/s, per-stage compute, idle time and hop time on a V100 + MI50 | Explorer, `STEPS_07=pipeline`, 1,000 steps | ran: 1F1B within the one-V100 runs' spread, GPipe 0.02 above it ([results](#results-a-v100-and-an-mi50)) |
| A two-stage pipeline on one V100, as a second baseline | - | not built |

A number in this file gets written only once it was measured, with where it
ran.
