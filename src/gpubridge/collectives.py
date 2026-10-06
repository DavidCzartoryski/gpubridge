"""Cluster-wide collectives built from native island collectives and the bridge transport."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge.topology import Topology, get_topology


def all_reduce(tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
    """Sum ``tensor`` across every rank in the cluster, in place.

    Mirrors ``torch.distributed.all_reduce`` on the world group. Every rank must
    call it with a tensor of the same shape and dtype. In a mixed cluster it
    runs reduce-bridge-broadcast:

    1. all_reduce within each island on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    Args:
        tensor: Input and output. Must be contiguous and on this rank's device
            (CPU in simulation and CPU-only mode).
        op: Only ``ReduceOp.SUM`` is supported in v1.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: for any op other than SUM, or a tensor on the wrong device
            or not contiguous.
    """
    topology = get_topology()
    if op != ReduceOp.SUM:
        raise ValueError(f"gpubridge.all_reduce only supports ReduceOp.SUM, got {op}")
    _check_tensor(tensor, topology)

    dist.all_reduce(tensor, group=topology.island_group)
    if not topology.layout.needs_bridge:
        return
    if topology.is_leader:
        assert topology.bridge is not None
        staged = tensor.cpu()  # the same tensor in simulation mode
        topology.bridge.all_reduce(staged)
        if staged is not tensor:
            tensor.copy_(staged)
    dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)


def broadcast(tensor: torch.Tensor, src: int) -> None:
    """Copy ``tensor`` from global rank ``src`` to every rank, in place.

    Mirrors ``torch.distributed.broadcast`` on the world group. In a mixed
    cluster the data travels in three hops:

    1. From ``src`` to the rest of its island on the native backend.
    2. From that island's leader to the other leaders over the bridge.
    3. From each of those leaders to the rest of its island.

    Args:
        tensor: Data to send on ``src``, overwritten everywhere else. Must be
            contiguous and on this rank's device (CPU in simulation mode).
        src: Global rank that holds the data.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: if ``src`` is not a rank, or the tensor is on the wrong
            device or not contiguous.
    """
    topology = get_topology()
    if not 0 <= src < topology.world_size:
        raise ValueError(f"src must be a rank in [0, {topology.world_size}), got {src}")
    _check_tensor(tensor, topology)

    src_island = topology.layout.island_of(src)
    in_src_island = topology.island == src_island
    if in_src_island:
        dist.broadcast(tensor, src=src, group=topology.island_group)
    if not topology.layout.needs_bridge:
        return
    if topology.is_leader:
        assert topology.bridge is not None
        staged = tensor.cpu()  # the same tensor in simulation mode
        topology.bridge.broadcast(staged, src=src_island.leader)
        if not in_src_island and staged is not tensor:
            tensor.copy_(staged)
    if not in_src_island:
        dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)


def barrier() -> None:
    """Block until every rank in the cluster reaches this call.

    On GPUs this first waits for the rank's queued device work, so collectives
    issued before the barrier have finished locally when it returns.

    Raises:
        RuntimeError: if gpubridge is not initialized.
    """
    topology = get_topology()
    if not topology.simulated:
        torch.cuda.synchronize(topology.device)
    dist.barrier()


def _check_tensor(tensor: torch.Tensor, topology: Topology) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    if tensor.device != topology.device:
        hint = ""
        if topology.simulated:
            hint = " (tensors stay on CPU in simulation and CPU-only mode)"
        raise ValueError(
            f"tensor is on {tensor.device} but this rank communicates on {topology.device}{hint}"
        )
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
