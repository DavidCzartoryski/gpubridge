"""Pipeline training across vendors: llama.py's model cut into contiguous stages, one per
rank, trained with torch.distributed.pipelining through backend="gpubridge", against one
GPU alone with the same global batch and seed (train_dp.py --mode single).

Launched by torchrun, one process per stage. Stages go to the NVIDIA ranks first, then
the AMD ones, each in rank order, whatever ranks the launcher handed out, so stage 0 is
an NVIDIA GPU in a mixed job. Stage 0 holds the embedding and the first blocks, the last
stage the last blocks, the final norm and the output layer; the split is even by
default (``--split 6,6`` gives the blocks per stage, in stage order). Activations go forward
and their gradients back with the backend's point-to-point sends and receives: native
within an island, across the bridge between islands. Schedules: GPipe (``--schedule
gpipe``) and 1F1B (``1f1b``), ``--microbatches`` microbatches of the global batch per
step, gradients averaged over them (torch's ``scale_grads``). ``--backend gloo`` runs the
same pipeline on a plain Gloo group, the reference gpubridge's point-to-point must match
bit for bit (CPU only).

Every step trains on llama.global_batch(step), the same tokens train_dp.py uses, so the
loss curves of the two compare directly. Recorded (rank 0 writes ``--json``): every step's
loss (the mean of the microbatches' losses on the last stage) and its time (the slowest
stage's); tokens/s from the median step after ``--warmup-steps``; each stage's blocks and
its compute alone, forward and backward of one microbatch with no communication; the
time to send one microbatch's activations from each stage to the next; and estimates
from those: each stage's idle time per step (the step less its microbatches' compute)
and GPipe's ideal bubble share, (stages - 1) / (microbatches + stages - 1).

On CPU, in simulation mode:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=2 \\
        scripts/gpu/train_pipeline.py --model llama-tiny --steps 20 --json /tmp/pp.json
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import llama
import torch
import torch.distributed as dist
import torch.nn.functional as F
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json
from torch import nn

import gpubridge  # noqa: F401  registers backend="gpubridge"

SCHEDULES = ("gpipe", "1f1b")


def even_split(blocks: int, stages: int) -> list[int]:
    """Blocks per stage, as even as whole blocks allow (the first stages take the rest)."""
    if blocks < stages:
        raise ValueError(f"{blocks} blocks can't give {stages} stages one each")
    base, extra = divmod(blocks, stages)
    return [base + (i < extra) for i in range(stages)]


class Stage(nn.Module):
    """One stage of the model: its contiguous blocks, plus the embedding (first stage) or
    the final norm and output layer (last)."""

    def __init__(self, model: llama.Llama, start: int, end: int, first: bool, last: bool):
        super().__init__()
        self.embed = model.embed if first else None
        self.blocks = nn.ModuleList(model.blocks[start:end])
        self.norm = model.norm if last else None
        self.head = model.head if last else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        if self.head is not None:
            x = self.head(self.norm(x))
        return x


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1))


def gather(value: Any) -> list[Any]:
    everyone: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, value)
    return everyone


def ms(seconds: float) -> float:
    return round(seconds * 1e3, 3)


def stage_compute(stage: Stage, example: torch.Tensor, target: torch.Tensor | None,
                  device: torch.device, trials: int) -> list[float]:
    """Seconds for forward and backward of one microbatch through this stage alone: no
    communication, no optimizer step (gradients are cleared after)."""
    out = []
    for i in range(trials + 1):
        x = example.clone().requires_grad_(example.is_floating_point())
        sync(device)
        start = time.perf_counter()
        y = stage(x)
        (loss_fn(y, target) if target is not None else y.float().sum()).backward()
        sync(device)
        if i:  # the first is a warmup
            out.append(time.perf_counter() - start)
    stage.zero_grad(set_to_none=True)
    return out


def stage_order(vendors: list[str]) -> list[int]:
    """The ranks in stage order: NVIDIA ranks first, then the rest, each in rank order."""
    return sorted(range(len(vendors)), key=lambda r: (vendors[r] != "nvidia", r))


def hop_time(stage: int, order: list[int], activation: torch.Tensor, device: torch.device,
             trials: int) -> list[float]:
    """Seconds for one microbatch's activations to arrive from the previous stage (on the
    receiving stage; empty on stage 0). Each pair of neighbours is timed on its own, every
    rank stepping through the same barriers, so no other hop runs at the same time."""
    out = []
    for sender in range(len(order) - 1):
        for i in range(trials + 1):
            dist.barrier()
            if stage == sender:
                dist.send(activation, dst=order[sender + 1])
            elif stage == sender + 1:
                sync(device)
                start = time.perf_counter()
                dist.recv(activation, src=order[sender])
                sync(device)
                if i:  # the first is a warmup
                    out.append(time.perf_counter() - start)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=llama.MODELS, default="llama-125m")
    parser.add_argument("--schedule", choices=SCHEDULES, default="gpipe")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=8, help="sequences per step")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="steps left out of the median step time")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", help="blocks per stage, e.g. 6,6 (default: even)")
    parser.add_argument("--backend", choices=("gpubridge", "gloo"), default="gpubridge",
                        help="gloo: the same pipeline on a plain Gloo group (CPU reference)")
    parser.add_argument("--probe-trials", type=int, default=3,
                        help="trials of each stage's compute alone and of each hop")
    parser.add_argument("--json", type=Path, help="rank 0 writes the result here")
    parser.add_argument("--timeout", type=float, default=default_timeout(1800),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 1800")
    args = parser.parse_args()
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

    config = llama.MODELS[args.model]
    if args.global_batch % args.microbatches:
        raise SystemExit("--global-batch must split into --microbatches equal microbatches")
    if not 0 < args.seq_len <= config.context:
        raise SystemExit(f"--seq-len must be 1 to {config.context}")
    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=args.timeout))
    dist.barrier()  # gpubridge's setup needs every rank before any point-to-point
    rank, world = dist.get_rank(), dist.get_world_size()
    topology = gpubridge.get_topology() if args.backend == "gpubridge" else None
    device = topology.device if topology is not None else torch.device("cpu")
    vendor = topology.vendor if topology is not None else "cpu"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    split = ([int(n) for n in args.split.split(",")] if args.split
             else even_split(config.blocks, world))
    if len(split) != world or sum(split) != config.blocks or min(split) < 1:
        raise SystemExit(f"--split {args.split}: need {world} positive counts adding up to "
                         f"{config.blocks} blocks")
    order = stage_order(gather(vendor))
    me = order.index(rank)  # this rank's stage
    first, last = me == 0, me == world - 1
    start = sum(split[:me])
    model = llama.build(args.model, seed=args.seed)  # every rank the same weights, on CPU
    stage = Stage(model, start, start + split[me], first, last).to(device)
    del model
    parameters = sum(p.numel() for p in stage.parameters())

    micro = args.global_batch // args.microbatches
    tokens = torch.zeros(micro, args.seq_len, dtype=torch.int64, device=device)
    hidden = torch.zeros(micro, args.seq_len, config.width, device=device)
    logits = torch.zeros(micro, args.seq_len, config.vocab, device=device)
    example = tokens if first else hidden
    x0, y0 = llama.global_batch(args.model, -1, micro, args.seq_len, args.seed)
    if first:
        example = x0.to(device)
    compute = stage_compute(stage, example, y0.to(device) if last else None,
                            device, args.probe_trials)
    hops = hop_time(me, order, hidden.clone(), device, args.probe_trials)

    # Static shapes, so nothing is inferred over the network. torch derives each gradient's
    # metadata from these: activations crossing a stage boundary require gradients (a later
    # stage's input, an earlier stage's output), or no gradient is sent back for them.
    meta_input = example.to("meta").requires_grad_(not first)
    meta_output = (logits if last else hidden).to("meta").requires_grad_(True)
    pipe_stage = PipelineStage(stage, me, world, device, input_args=(meta_input,),
                               output_args=(meta_output,))
    # Stage i's neighbours are the ranks holding stages i - 1 and i + 1 (torch's default
    # puts stage i on rank i).
    pipe_stage.stage_index_to_group_rank = dict(enumerate(order))
    schedule = (ScheduleGPipe if args.schedule == "gpipe" else Schedule1F1B)(
        pipe_stage, n_microbatches=args.microbatches, loss_fn=loss_fn)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr)

    step_s, last_losses = [], []
    for step in range(args.steps):
        x, y = llama.global_batch(args.model, step, args.global_batch, args.seq_len, args.seed)
        sync(device)
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if first:
            schedule.step(x.to(device))
        elif last:
            losses: list[torch.Tensor] = []
            schedule.step(target=y.to(device), losses=losses)
        else:
            schedule.step()
        optimizer.step()
        sync(device)
        step_s.append(time.perf_counter() - begin)
        if last:
            last_losses.append(float(torch.stack(losses).mean()))

    everyone = gather({"step_s": step_s, "losses": last_losses, "compute": compute,
                       "hops": hops, "vendor": vendor, "blocks": [start, start + split[rank]],
                       "parameters": parameters,
                       "device": torch.cuda.get_device_name(device) if device.type == "cuda"
                       else str(device)})
    environments = gather(environment()) if args.json else []
    losses_all = everyone[order[-1]]["losses"]  # the last stage's
    steps = [max(times) for times in zip(*(e["step_s"] for e in everyone))]
    median_step = statistics.median(steps[args.warmup_steps:] or steps)
    ok = all(math.isfinite(v) for v in losses_all)
    if rank == 0:
        stages = []  # in stage order
        for s, r in enumerate(order):
            e = everyone[r]
            per_micro = statistics.median(e["compute"])
            stages.append({
                "stage": s, "rank": r, "vendor": e["vendor"], "device": e["device"],
                "blocks": e["blocks"], "parameters": e["parameters"],
                "compute_ms_per_microbatch": ms(per_micro),
                "idle_ms_per_step_est": ms(max(0.0, median_step - args.microbatches * per_micro)),
                "hop_in_ms": ms(statistics.median(e["hops"])) if e["hops"] else None,
                "share": f"{e['blocks'][1] - e['blocks'][0]} blocks"})
        record = {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/train_pipeline.py",
            "argv": sys.argv, "run": run_info(topology) if topology is not None else None,
            "mode": f"pipeline-{args.schedule}", "backend": args.backend, "world_size": world,
            "compression": os.environ.get("GPUBRIDGE_BRIDGE_COMPRESSION") or "none",
            "model": {"name": args.model, "config": asdict(config),
                      "parameters": llama.parameters(args.model)},
            "global_batch": args.global_batch, "seq_len": args.seq_len,
            "microbatches": args.microbatches, "steps": args.steps,
            "warmup_steps": args.warmup_steps, "seed": args.seed, "lr": args.lr,
            "optimizer": f"AdamW(lr={args.lr})", "split": split,
            "ranks": stages, "losses": losses_all,  # full precision: compared bit for bit
            "step_ms": [ms(s) for s in steps], "median_step_ms": ms(median_step),
            "tokens_per_s": round(args.global_batch * args.seq_len / median_step, 1),
            "loss_first": round(losses_all[0], 4), "loss_last": round(losses_all[-1], 4),
            "ideal_bubble_fraction": round((world - 1) / (args.microbatches + world - 1), 4),
            "estimates": "idle_ms_per_step_est: the median step less the stage's microbatches "
                         "of compute alone; hop_in_ms: one microbatch's activations from the "
                         "previous stage, timed alone",
            "environment": environments[0] if environments else None,
            "environments": environments,
        }
        if topology is not None and topology.split_test:
            record["warning"] = "SPLIT TEST: not a mixed-vendor result."
        print(f"{args.model}, {args.schedule}, {world} stage(s) {split}, {args.backend}: loss "
              f"{losses_all[0]:.3f} -> {losses_all[-1]:.3f}, median step {ms(median_step)} ms, "
              f"{record['tokens_per_s']} tokens/s; compute per microbatch "
              + ", ".join(f"{s['vendor']} {s['compute_ms_per_microbatch']}" for s in stages),
              flush=True)
        if args.json:
            write_json(args.json, record)
    del schedule, pipe_stage, optimizer, stage
    gc.collect()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit("a loss is not finite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
