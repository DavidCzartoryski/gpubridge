"""Data-parallel training on synthetic data, with gradients synced by gpubridge.

A small MLP learns a fixed random classification task. Each rank trains on its
own batches. The model is wrapped in DistributedDataParallel on gpubridge's
Gloo world group, with gpubridge.ddp_comm_hook registered: DDP buckets the
gradients and overlaps their sync with backward, and each bucket is averaged
with an async gpubridge.all_reduce. So the same script trains on NVIDIA, AMD
or both at once. It reports throughput (samples/sec across all ranks) and
checks that every rank ends with identical parameters.

On CPU, in simulation mode:

    GPUBRIDGE_VENDOR=nvidia torchrun --nproc-per-node=2 examples/train_synthetic.py

On GPUs, the same command without GPUBRIDGE_VENDOR. scripts/gpu uses it for
the scaling run (1 to 4 GPUs) with --json to record the result.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import gpubridge


def make_model(dim: int, layers: int, classes: int) -> nn.Module:
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks += [nn.Linear(dim, dim), nn.ReLU()]
    return nn.Sequential(*blocks, nn.Linear(dim, classes))


def wait(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=50, help="timed steps")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256, help="per rank")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--json", type=Path, help="rank 0 writes a result record here")
    parser.add_argument("--timeout", type=float,
                        default=float(os.environ.get("KIT_TIMEOUT") or 600),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 600")
    args = parser.parse_args()

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
    device, rank, world = topology.device, topology.rank, topology.world_size

    # Same seed everywhere; DDP also broadcasts rank 0's parameters when it wraps
    # the model. The default group is gpubridge's Gloo world group, which spans
    # both vendors; the hook sends every gradient bucket through gpubridge.
    torch.manual_seed(0)
    model = make_model(args.dim, args.layers, args.classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None)
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # The task: labels come from a fixed random projection. Each rank sees its own samples.
    teacher = torch.randn(args.dim, args.classes, generator=torch.Generator().manual_seed(1))
    data = torch.Generator().manual_seed(100 + rank)
    batches = []
    for _ in range(8):
        x = torch.randn(args.batch_size, args.dim, generator=data)
        batches.append((x.to(device), (x @ teacher).argmax(dim=1).to(device)))

    losses, step_times = [], []
    for step in range(args.warmup_steps + args.steps):
        x, y = batches[step % len(batches)]
        wait(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        loss = loss_fn(model(x), y)
        loss.backward()  # DDP waits for every bucket's all_reduce before returning
        optimizer.step()
        wait(device)
        if step >= args.warmup_steps:
            step_times.append(time.perf_counter() - start)
        losses.append(loss.item())

    # Slowest rank sets the pace; every rank should hold identical parameters.
    elapsed = torch.tensor([sum(step_times)], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    checksum = float(sum(p.detach().double().sum() for p in model.parameters()))
    checksums: list[float | None] = [None] * world
    dist.all_gather_object(checksums, checksum)
    in_sync = all(c == checksums[0] for c in checksums)
    samples_per_sec = world * args.batch_size * args.steps / float(elapsed)

    if rank == 0:
        result = {
            "script": "examples/train_synthetic.py",
            "world_size": world,
            "run": {"kind": topology.run_kind, "split_test": topology.split_test,
                    "islands": [list(i.ranks) for i in topology.layout.islands]},
            "device": str(device),
            "grad_sync": "DistributedDataParallel + gpubridge.ddp_comm_hook",
            "policy": topology.policy_name,
            "batch_size_per_rank": args.batch_size,
            "model": {"dim": args.dim, "layers": args.layers, "classes": args.classes,
                      "parameters": sum(p.numel() for p in model.parameters())},
            "steps": args.steps,
            "samples_per_sec": round(samples_per_sec, 2),
            "step_ms_median": round(statistics.median(step_times) * 1e3, 3),
            "loss_first": round(losses[0], 4),
            "loss_last": round(losses[-1], 4),
            "params_in_sync": in_sync,
        }
        if topology.split_test:
            result["warning"] = "SPLIT TEST: not a mixed-vendor result."
        print(f"{world} rank(s), {topology.run_kind}: {samples_per_sec:.1f} samples/sec, "
              f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, params in sync: {in_sync}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2))
    gpubridge.destroy()
    if not in_sync:
        raise SystemExit("parameters diverged across ranks")


if __name__ == "__main__":
    main()
