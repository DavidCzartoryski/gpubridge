"""One script for every node, launched with torchrun.

On a real mixed cluster, run the same command on NVIDIA and AMD nodes; each
process detects its vendor from its PyTorch build. To try it on one machine,
pretend to be two nodes of different vendors, in two terminals:

    GPUBRIDGE_VENDOR=nvidia torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 \
        --master-addr=127.0.0.1 --master-port=29500 examples/all_reduce_torchrun.py
    GPUBRIDGE_VENDOR=amd torchrun --nnodes=2 --nproc-per-node=2 --node-rank=1 \
        --master-addr=127.0.0.1 --master-port=29500 examples/all_reduce_torchrun.py
"""

from __future__ import annotations

import torch

import gpubridge


def main() -> None:
    topology = gpubridge.init()  # env:// rendezvous from torchrun

    # Each rank contributes its rank number; the same code runs on every vendor.
    tensor = torch.full((4,), float(topology.rank), device=topology.device)
    gpubridge.all_reduce(tensor)

    role = "leader" if topology.is_leader else "member"
    print(f"rank {topology.rank} ({topology.vendor}, {role}): {tensor.tolist()}", flush=True)
    gpubridge.destroy()


if __name__ == "__main__":
    main()
