"""FSDP2 (``fully_shard``) with gpubridge as the torch.distributed backend.

A plain FSDP2 training script apart from the backend name and ``import
gpubridge``, which registers that name. (On torch 2.14 and later, installed
gpubridge also registers it when the name is first asked for, so the import is
optional there.) Parameters are sharded across every rank of every island;
FSDP2's all-gathers and reduce-scatters cross the bridge. On CPU, in simulation
mode, with two islands:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        examples/fsdp2_torch_backend.py

On a mixed NVIDIA + AMD cluster, the same command without the two variables.
It exits non-zero unless the loss went down and every rank assembled the same
parameters. With ``--hsdp 2`` it runs HSDP instead: parameters sharded within
each vendor island, replicated across the two (docs/TORCH_BACKEND.md,
"Subgroups and HSDP"). With ``--verify`` it trains a float64 model on integer-valued data
(learning rate 4, so every update is a whole-number sum, exact in any order)
and rank 0 replays the same steps in one process: the parameters must match
bit for bit. A collective that read or wrote a tensor at the wrong moment on a
GPU would show there, even when every rank agrees.
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

import gpubridge  # noqa: F401  registers backend="gpubridge"

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:  # releases before fully_shard moved there
    from torch.distributed._composable.fsdp import fully_shard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json", type=Path, help="rank 0 writes the losses here")
    parser.add_argument("--verify", action="store_true",
                        help="exact data; compare with a single-process replay on rank 0")
    parser.add_argument("--hsdp", type=int, metavar="ISLANDS",
                        help="HSDP on a 2-D mesh: shard within each of ISLANDS vendor islands "
                             "(contiguous, equal blocks of ranks), replicate across them")
    args = parser.parse_args()
    if args.verify:
        return verify(args.hsdp)

    dist.init_process_group(backend="gpubridge")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available() and not os.environ.get("GPUBRIDGE_VENDOR"):
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    mesh = make_mesh(device, world, args.hsdp)

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


def make_mesh(device: torch.device, world: int, islands: int | None):
    """1-D over every rank (FSDP2), or with ``islands`` 2-D (HSDP): row i is island i's
    ranks, the shard group; each column, one rank per island, is a replicate group.
    Shard groups are single-vendor, so all-gathers and reduce-scatters stay on
    NCCL/RCCL; only the replicate all_reduce, once per step, crosses the bridge."""
    if not islands:
        return init_device_mesh(device.type, (world,))
    if world % islands:
        raise SystemExit(f"--hsdp {islands}: {world} ranks don't split into {islands} islands")
    return init_device_mesh(device.type, (islands, world // islands),
                            mesh_dim_names=("replicate", "shard"))


def verify(islands: int | None = None) -> int:
    """float64, integer-valued data and lr=4: every rank's averaged update is a whole-number
    sum, so FSDP2 through the backend must equal one process on the whole batch exactly."""
    dist.init_process_group(backend="gpubridge")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available() and not os.environ.get("GPUBRIDGE_VENDOR"):
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    mesh = make_mesh(device, world, islands)

    def make() -> nn.Module:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).double()
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randint(-3, 4, p.shape).double())
        return model

    def batch(r: int, step: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(100 * step + r)
        return torch.randint(-2, 3, (4, 8), generator=g).double()

    model = make().to(device)
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=4.0)
    for step in range(3):
        model(batch(rank, step).to(device)).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().cpu() for p in model.parameters()]
    ok = True
    if rank == 0:
        reference = make()
        ref_opt = torch.optim.SGD(reference.parameters(), lr=4.0)
        for step in range(3):
            (sum(reference(batch(r, step)).sum() for r in range(world)) / world).backward()
            ref_opt.step()
            ref_opt.zero_grad()
        ok = all(torch.equal(a, b.detach()) for a, b in zip(full, reference.parameters(),
                                                            strict=True))
        print(f"verify: {world} ranks, parameters {'equal' if ok else 'DIFFER from'} "
              "a single-process replay, bit for bit")
    flag = torch.tensor([1.0 if ok else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)  # every rank exits with rank 0's verdict
    del model, opt
    gc.collect()
    dist.destroy_process_group()
    return 0 if flag.item() == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
