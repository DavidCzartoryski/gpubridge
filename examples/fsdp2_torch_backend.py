"""FSDP2 (``fully_shard``) with gpubridge as the torch.distributed backend.

A plain FSDP2 training script: the only gpubridge in it is the backend name.
Installed gpubridge registers that name when torch is imported (from a source
checkout that isn't installed, add ``import gpubridge``). Parameters are
sharded across every rank of every island; FSDP2's all-gathers and
reduce-scatters cross the bridge. On CPU, in simulation mode, with two islands:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        examples/fsdp2_torch_backend.py

On a mixed NVIDIA + AMD cluster, the same command without the two variables.
It exits non-zero unless the loss went down and every rank assembled the same
parameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:  # releases before fully_shard moved there
    from torch.distributed._composable.fsdp import fully_shard


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
    mesh = init_device_mesh(device.type, (world,))

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                          nn.Linear(64, 8)).to(device)
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    teacher = torch.randn(32, 8, generator=torch.Generator().manual_seed(1)).to(device)
    data = torch.Generator().manual_seed(100 + rank)
    losses = []
    for _ in range(args.steps):
        x = torch.randn(64, 32, generator=data).to(device)
        loss = nn.functional.cross_entropy(model(x), (x @ teacher).argmax(dim=1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    # Each rank assembles the full parameters from every rank's shard (an
    # all-gather); they must agree everywhere.
    same = True
    for p in model.parameters():
        full = p.full_tensor().detach()
        hi, lo = full.clone(), full.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        same &= bool(torch.equal(hi, lo))
    ok = same and losses[-1] < losses[0]
    if rank == 0:
        print(f"{world} ranks, loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
              f"parameters {'identical' if same else 'DIFFER'} on every rank")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({"world": world, "losses": losses,
                                             "identical": same, "ok": ok}, indent=2))
    del model, opt  # FSDP2 holds process groups in reference cycles: free them before
    gc.collect()    # the groups are destroyed, not at interpreter exit
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
