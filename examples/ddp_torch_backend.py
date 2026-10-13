"""DistributedDataParallel with gpubridge as the torch.distributed backend.

A plain DDP training script: the only gpubridge in it is the backend name.
Installed gpubridge registers that name when torch is imported (from a source
checkout that isn't installed, add ``import gpubridge``). On CPU, in simulation
mode, with two islands:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        examples/ddp_torch_backend.py

On a mixed NVIDIA + AMD cluster, the same command without the two variables.
It exits non-zero if the ranks' parameters end up different.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json", type=Path, help="rank 0 writes the losses here")
    args = parser.parse_args()

    dist.init_process_group(backend="gpubridge")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available() and not os.environ.get("GPUBRIDGE_VENDOR"):
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)  # the same initial model everywhere (DDP also broadcasts it)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8)).to(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda"
                                  else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    teacher = torch.randn(32, 8, generator=torch.Generator().manual_seed(1)).to(device)
    data = torch.Generator().manual_seed(100 + rank)  # every rank its own batches
    losses = []
    for _ in range(args.steps):
        x = torch.randn(64, 32, generator=data).to(device)
        loss = nn.functional.cross_entropy(ddp(x), (x @ teacher).argmax(dim=1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    # Every rank must hold the same parameters: compare the largest and smallest
    # of each parameter's values across ranks.
    same = True
    for p in model.parameters():
        hi, lo = p.detach().clone(), p.detach().clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        same &= bool(torch.equal(hi, lo))
    if rank == 0:
        print(f"{world} ranks, loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
              f"parameters {'identical' if same else 'DIFFER'} on every rank")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({"world": world, "losses": losses,
                                             "identical": same}, indent=2))
    dist.destroy_process_group()
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
