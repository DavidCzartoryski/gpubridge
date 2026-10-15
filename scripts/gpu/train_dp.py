"""Data-parallel training across vendors on the LLaMA-style model (llama.py), against one
GPU alone with the same global batch and seed.

Launched by torchrun, one process per GPU; ``--mode single`` is one process, the
baseline. Modes:

- ``ddp``: DistributedDataParallel with ``gpubridge.ddp_comm_hook`` (each gradient
  bucket averaged by an async gpubridge.all_reduce that overlaps backward).
- ``fsdp2``: ``fully_shard`` on every block and the model, through
  ``backend="gpubridge"``: parameters sharded across every rank.
- ``hsdp``: the same on a 2-D mesh, sharded within each of ``--islands`` vendor
  islands and replicated across them (docs/TORCH_BACKEND.md).
- ``single``: one GPU, no process group, the whole global batch (in ``--micro``
  micro-batches with accumulated gradients, if given).

Every step trains on llama.global_batch(step): the same rows for any world size. Each
rank takes a contiguous share of them: equal shares by default; with ``--shares speed``,
shares proportional to each rank's measured speed (forward, backward and an optimizer
step on a few batches before training, timed on its own GPU), so the faster GPU takes
more rows; or the rows per rank as given (``--shares 5,3``). Each
rank's loss is then scaled by world x share / global batch before backward, so the
average every mode takes is still the whole batch's mean gradient (sample-weighted
averaging).

Bridge compression is gpubridge's own: ``GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8``, with
error feedback on the residual it keeps per buffer (DDP's buckets keep their buffers
across steps). ``--kernels-on-amd torch`` makes this script's AMD ranks use gpubridge's
PyTorch codec instead of its Triton kernels (``GPUBRIDGE_COMPRESSION_KERNELS=torch``,
the same bits), for ROCm builds whose Triton can't compile them (gfx906 on
torch 2.9.1+rocm6.3).

Recorded (rank 0 writes ``--json``): every step's loss over the whole global batch and its
time; every ``--check-every`` steps, whether every rank holds the same parameter bits
(``ddp``: replicas; ``hsdp``: the replicate groups; ``fsdp2`` has one copy of each shard,
so nothing to compare) and the largest gap between ranks' parameter sums; and at the end
tokens/s from the median step after ``--warmup-steps``. For ``ddp`` also the step split,
as train_bench.py does it: each rank's compute alone (forward, backward and an AdamW step
with learning rate 0, so no parameter moves), the gradient buckets all_reduced with
nothing else running, the communication backward didn't hide, and the share it did.

On CPU, in simulation mode:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=2 \\
        scripts/gpu/train_dp.py --model llama-tiny --mode ddp --steps 20 --json /tmp/dp.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel

import gpubridge

MODES = ("ddp", "fsdp2", "hsdp", "single")


def shares_for(weights: list[float], total: int) -> list[int]:
    """``total`` rows split in proportion to ``weights``, at least one each (largest
    remainder, ties to the lower rank)."""
    if total < len(weights):
        raise ValueError(f"{total} rows can't give {len(weights)} ranks one each")
    spare = total - len(weights)
    exact = [spare * w / sum(weights) for w in weights]
    out = [1 + int(e) for e in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - int(exact[i])), i))
    for i in order[:total - sum(out)]:
        out[i] += 1
    return out


def rows(shares: list[int], rank: int) -> slice:
    start = sum(shares[:rank])
    return slice(start, start + shares[rank])


def digest_and_sum(tensors) -> tuple[str, float, bool]:
    """sha256 of the tensors' bits, their total, and whether every value is finite."""
    digest, total, finite = hashlib.sha256(), 0.0, True
    for t in tensors:
        values = t.detach().cpu().contiguous()
        digest.update(values.view(torch.uint8).numpy().tobytes())
        total += float(values.double().sum())
        finite = finite and bool(torch.isfinite(values).all())
    return digest.hexdigest(), total, finite


def gather(value: Any) -> list[Any]:
    everyone: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, value)
    return everyone


def ms(seconds: float) -> float:
    return round(seconds * 1e3, 3)


def compute_alone(model, data, device, steps: int) -> list[float]:
    """Seconds per step of forward, backward and an AdamW step at learning rate 0 on this
    rank's own GPU: what one step costs without any gradient sync. No parameter moves."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    times = []
    for i in range(steps + 1):
        x, y = data
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        llama.loss_of(model, x, y).backward()
        optimizer.step()
        sync(device)
        if i:  # the first is a warmup
            times.append(time.perf_counter() - start)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return times


def comm_alone(sizes: list[tuple[int, torch.dtype]], device, trials: int) -> list[float]:
    """Seconds to average every DDP bucket (async, as the hook does) with nothing else."""
    buffers = [torch.ones(n, dtype=dtype, device=device) for n, dtype in sizes]
    out = []
    for i in range(trials + 1):
        sync(device)
        dist.barrier()
        start = time.perf_counter()
        works = [gpubridge.all_reduce(b, op=ReduceOp.AVG, async_op=True) for b in buffers]
        for work in works:
            if work is not None:
                work.wait()
        sync(device)
        if i:
            out.append(time.perf_counter() - start)
    return out


def recording_hook(sizes: list[tuple[int, torch.dtype]]):
    """gpubridge.ddp_comm_hook, noting each bucket's size and dtype (the last step's)."""

    def hook(state, bucket):
        if bucket.index() == 0:
            sizes.clear()
        buffer = bucket.buffer()
        sizes.append((buffer.numel(), buffer.dtype))
        return gpubridge.ddp_comm_hook(state, bucket)

    hook.__annotations__ = {"state": object, "bucket": dist.GradBucket,
                            "return": torch.futures.Future[torch.Tensor]}
    return hook


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)


def setup(args) -> tuple[Any, torch.device, int, int, str]:
    """(topology or None, device, rank, world, vendor) for the mode."""
    timeout = timedelta(seconds=args.timeout)
    if args.mode == "single":
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        vendor = ("amd" if torch.version.hip else "nvidia") if device.type == "cuda" else "cpu"
        return None, device, 0, 1, vendor
    if args.mode == "ddp":
        topology = gpubridge.init(timeout=timeout)
    else:  # torch.distributed's own init; the backend builds gpubridge's topology at the
        dist.init_process_group(backend="gpubridge", timeout=timeout)  # first collective
        dist.barrier()
        topology = gpubridge.get_topology()
    if topology.device.type == "cuda":
        torch.cuda.set_device(topology.device)
    return topology, topology.device, topology.rank, topology.world_size, topology.vendor


def shard(model, args, device, world: int):
    from torch.distributed.device_mesh import init_device_mesh

    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:  # releases before fully_shard moved there
        from torch.distributed._composable.fsdp import fully_shard
    if args.mode == "hsdp":
        if world % args.islands:
            raise SystemExit(f"--islands {args.islands}: {world} ranks don't split evenly")
        mesh = init_device_mesh(device.type, (args.islands, world // args.islands),
                                mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh(device.type, (world,))
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=llama.MODELS, default="llama-125m")
    parser.add_argument("--mode", choices=MODES, default="ddp")
    parser.add_argument("--global-batch", type=int, default=8, help="sequences per step, all ranks")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="steps left out of the median step time")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bucket-mb", type=float, default=25.0, help="DDP's bucket_cap_mb")
    parser.add_argument("--shares", default="equal",
                        help="equal, speed, or rows per rank in rank order, e.g. 5,3")
    parser.add_argument("--islands", type=int, default=2, help="hsdp: vendor islands")
    parser.add_argument("--micro", type=int, default=1, help="single: micro-batches per step")
    parser.add_argument("--check-every", type=int, default=50,
                        help="steps between parameter checks across ranks")
    parser.add_argument("--probe-steps", type=int, default=3,
                        help="steps timed per rank for --shares speed and the ddp step split")
    parser.add_argument("--comm-trials", type=int, default=5)
    parser.add_argument("--kernels-on-amd", choices=("auto", "torch"), default="auto",
                        help="compression kernels on AMD ranks (see the docstring)")
    parser.add_argument("--json", type=Path, help="rank 0 writes the result here")
    parser.add_argument("--timeout", type=float, default=default_timeout(1800),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 1800")
    args = parser.parse_args()
    if not 0 < args.seq_len <= llama.MODELS[args.model].context:
        raise SystemExit(f"--seq-len must be 1 to {llama.MODELS[args.model].context}")
    if args.mode == "single" and args.global_batch % args.micro:
        raise SystemExit("--global-batch must split into --micro equal micro-batches")

    topology, device, rank, world, vendor = setup(args)
    if vendor == "amd" and args.kernels_on_amd == "torch":
        os.environ["GPUBRIDGE_COMPRESSION_KERNELS"] = "torch"
    split_test = bool(topology is not None and getattr(topology, "split_test", False))
    if split_test and rank == 0:
        print("SPLIT TEST: not a mixed-vendor result.", file=sys.stderr, flush=True)
    model = llama.build(args.model, seed=args.seed, device=device)
    parameters = sum(p.numel() for p in model.parameters())
    distributed = args.mode != "single"

    # Shares: equal, proportional to each rank's speed on an equal share, or as given.
    shares = shares_for([1.0] * world, args.global_batch)
    if args.shares not in ("equal", "speed"):
        given = [int(n) for n in args.shares.split(",")]
        if len(given) != world or sum(given) != args.global_batch or min(given) < 1:
            raise SystemExit(f"--shares {args.shares}: need {world} positive counts that add "
                             f"up to --global-batch {args.global_batch}")
        shares = given
    probe: list[float] = []
    if distributed and (args.shares == "speed" or args.mode == "ddp"):
        x, y = llama.global_batch(args.model, -1, args.global_batch, args.seq_len, args.seed)
        mine = rows(shares, rank)
        probe = compute_alone(model, (x[mine].to(device), y[mine].to(device)), device,
                              args.probe_steps)
    if distributed and args.shares == "speed":
        speeds = [s / statistics.median(t) for s, t in zip(shares, gather(probe), strict=True)]
        shares = shares_for(speeds, args.global_batch)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sizes: list[tuple[int, torch.dtype]] = []
    if args.mode == "ddp":
        trained = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            bucket_cap_mb=args.bucket_mb, broadcast_buffers=False)
        trained.register_comm_hook(None, recording_hook(sizes))
    elif args.mode in ("fsdp2", "hsdp"):
        trained = shard(model, args, device, world)
    else:
        trained = model
    optimizer = torch.optim.AdamW(trained.parameters(), lr=args.lr)
    weight = world * shares[rank] / args.global_batch
    mine = rows(shares, rank)

    losses, step_s, checks = [], [], []
    for step in range(args.steps):
        x, y = llama.global_batch(args.model, step, args.global_batch, args.seq_len, args.seed)
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "single":
            total = torch.zeros((), device=device)
            for xs, ys in zip(x.chunk(args.micro), y.chunk(args.micro), strict=True):
                loss = llama.loss_of(trained, xs.to(device), ys.to(device))
                (loss / args.micro).backward()
                total += loss.detach() / args.micro
        else:
            loss = llama.loss_of(trained, x[mine].to(device), y[mine].to(device))
            (loss * weight).backward()
        optimizer.step()
        if distributed:
            # The whole batch's loss: each share's mean weighted by its rows, gathered as
            # Python floats so bridge compression never touches the record.
            total = sum(gather(float(loss.detach()) * shares[rank] / args.global_batch))
        losses.append(float(total))  # waits for the step's GPU work
        sync(device)
        step_s.append(time.perf_counter() - start)
        last = step == args.steps - 1
        if distributed and args.mode != "fsdp2" and ((step + 1) % args.check_every == 0 or last):
            if args.mode == "hsdp":
                tensors = [p.full_tensor() for p in trained.parameters()]
            else:
                tensors = list(model.parameters())
            everyone = gather(digest_and_sum(tensors))
            del tensors
            checks.append({"step": step + 1, "identical": len({e[0] for e in everyone}) == 1,
                           "sum_spread": max(e[1] for e in everyone) - min(e[1] for e in everyone),
                           "finite": all(e[2] for e in everyone)})

    split: dict[str, Any] = {}
    if args.mode == "ddp":
        alone = comm_alone(sizes, device, args.comm_trials)
        everyone = gather({"compute": probe, "alone": alone})
        step = statistics.median(step_s[args.warmup_steps:] or step_s)
        computes = [statistics.median(e["compute"]) for e in everyone]
        slowest = max(computes)
        exposed = max(0.0, step - slowest)
        alone_s = statistics.median(max(t) for t in zip(*(e["alone"] for e in everyone)))
        hidden = max(0.0, alone_s - exposed)
        split = {"slowest_compute_ms": ms(slowest), "exposed_comm_ms": ms(exposed),
                 "comm_alone_ms": ms(alone_s), "hidden_comm_ms": ms(hidden),
                 "hidden_fraction": round(hidden / alone_s, 4) if alone_s else None,
                 "buckets": len(sizes),
                 "gradient_bytes": sum(n * torch.tensor([], dtype=d).element_size()
                                       for n, d in sizes),
                 "compute_ms": [ms(c) for c in computes],
                 "waiting_ms": [ms(slowest - c) for c in computes]}
    me = {"vendor": vendor, "device": device_name(device), "share": shares[rank],
          "compression_kernels": os.environ.get("GPUBRIDGE_COMPRESSION_KERNELS") or "auto"}
    who = gather(me) if distributed else [me]
    environments = (gather(environment()) if distributed else [environment()]) if args.json \
        else []
    median_step = statistics.median(step_s[args.warmup_steps:] or step_s)
    ok = all(c["finite"] for c in checks) and all(math.isfinite(v) for v in losses)
    if rank == 0:
        record = {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/train_dp.py", "argv": sys.argv,
            "run": run_info(topology) if topology is not None else None,
            "mode": args.mode, "world_size": world,
            "compression": os.environ.get("GPUBRIDGE_BRIDGE_COMPRESSION") or "none",
            "model": {"name": args.model, "config": asdict(llama.MODELS[args.model]),
                      "parameters": parameters},
            "global_batch": args.global_batch, "seq_len": args.seq_len, "steps": args.steps,
            "warmup_steps": args.warmup_steps, "seed": args.seed, "lr": args.lr,
            "optimizer": f"AdamW(lr={args.lr})", "bucket_mb": args.bucket_mb,
            "shares_rule": args.shares, "micro_batches": args.micro,
            "ranks": [{"rank": r, **w} for r, w in enumerate(who)],
            "losses": [round(v, 6) for v in losses],
            "step_ms": [ms(s) for s in step_s],
            "median_step_ms": ms(median_step),
            "tokens_per_s": round(args.global_batch * args.seq_len / median_step, 1),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "param_checks": checks, **split,
            "environment": environments[0] if environments else None,
            "environments": environments,
        }
        if split_test:
            record["warning"] = "SPLIT TEST: not a mixed-vendor result."
        print(f"{args.model}, {args.mode}, {world} rank(s), compression "
              f"{record['compression']}: loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
              f"median step {ms(median_step)} ms, {record['tokens_per_s']} tokens/s, shares "
              f"{shares}" + (f"; hidden {split['hidden_fraction']} of comm" if split else ""),
              flush=True)
        for c in checks:
            if not c["identical"]:
                print(f"note: step {c['step']}: the ranks' parameters differ (sums "
                      f"{c['sum_spread']:.3g} apart)", flush=True)
        if args.json:
            write_json(args.json, record)
    del trained, optimizer, model
    gc.collect()  # FSDP2 holds process groups in reference cycles: free them first
    if args.mode == "ddp":
        gpubridge.destroy()
    elif distributed:
        dist.destroy_process_group()
    if not ok:
        raise SystemExit("a loss or parameter is not finite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
