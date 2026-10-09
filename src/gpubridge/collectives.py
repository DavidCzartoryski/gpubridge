"""Cluster-wide collectives: validate the call, then hand it to the active collective policy."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import observe
from gpubridge.topology import Topology, get_topology

#: Dtypes every built-in path supports: Gloo (world group, bridge, CPU islands)
#: and NCCL/RCCL (GPU islands). Gloo all_reduce and broadcast were checked for
#: each one on torch 2.3.1, 2.6.0, 2.9.1 and 2.14.1, including the CUDA and
#: ROCm builds of 2.14.1 (rerun with scripts/gloo_dtypes.py). Not included:
#: - bool, because a SUM of bools is a logical OR on both NCCL and Gloo;
#: - int16, uint16, uint32 and float8, which Gloo rejects;
#: - complex64, whose Gloo broadcast fails before torch 2.9.
SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
)


def all_reduce(tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
    """Sum ``tensor`` across every rank in the cluster, in place.

    Mirrors ``torch.distributed.all_reduce`` on the world group. Every rank must
    call it with a tensor of the same shape and dtype. The active collective
    policy decides how the data travels; in a mixed cluster the default runs
    reduce-bridge-broadcast:

    1. reduce within each island onto its leader, on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    Args:
        tensor: Input and output. Must be contiguous, on this rank's device
            (CPU in simulation and CPU-only mode), and of a dtype in
            :data:`SUPPORTED_DTYPES`.
        op: Only ``ReduceOp.SUM`` is supported in v1.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: for any op other than SUM, an unsupported dtype, or a tensor
            on the wrong device or not contiguous.
    """
    topology = get_topology()
    if op != ReduceOp.SUM:
        raise ValueError(f"gpubridge.all_reduce only supports ReduceOp.SUM, got {op}")
    _check_tensor(tensor, topology)
    policy = topology.policy.select_policy(tensor.nbytes)
    with observe.collective(topology, "all_reduce", tensor, policy=policy.name):
        policy.all_reduce(tensor, topology)


def broadcast(tensor: torch.Tensor, src: int) -> None:
    """Copy ``tensor`` from global rank ``src`` to every rank, in place.

    Mirrors ``torch.distributed.broadcast`` on the world group. The active
    collective policy decides how the data travels; in a mixed cluster the
    default takes three hops:

    1. From ``src`` to the rest of its island on the native backend.
    2. From that island's leader to the other leaders over the bridge.
    3. From each of those leaders to the rest of its island.

    Args:
        tensor: Data to send on ``src``, overwritten everywhere else. Must be
            contiguous, on this rank's device (CPU in simulation mode), and of a
            dtype in :data:`SUPPORTED_DTYPES`.
        src: Global rank that holds the data.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        ValueError: if ``src`` is not a rank, or for an unsupported dtype, or a
            tensor on the wrong device or not contiguous.
    """
    topology = get_topology()
    if not 0 <= src < topology.world_size:
        raise ValueError(f"src must be a rank in [0, {topology.world_size}), got {src}")
    _check_tensor(tensor, topology)
    policy = topology.policy.select_policy(tensor.nbytes)
    with observe.collective(topology, "broadcast", tensor, src=src, policy=policy.name):
        policy.broadcast(tensor, src, topology)


def barrier() -> None:
    """Block until every rank in the cluster reaches this call.

    On GPUs this first waits for the rank's queued device work, so collectives
    issued before the barrier have finished locally when it returns.

    Raises:
        RuntimeError: if gpubridge is not initialized.
    """
    topology = get_topology()
    with observe.collective(topology, "barrier"):
        if not topology.simulated:
            torch.cuda.synchronize(topology.device)
        dist.barrier()


def _check_tensor(tensor: torch.Tensor, topology: Topology) -> None:
    """Reject bad calls before any communication, identically on every rank."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        names = ", ".join(str(dtype).removeprefix("torch.") for dtype in SUPPORTED_DTYPES)
        raise ValueError(
            f"gpubridge doesn't support {tensor.dtype}: a dtype must work on both the "
            f"island backend and Gloo. Supported: {names}."
        )
    if tensor.device != topology.device:
        hint = ""
        if topology.simulated:
            hint = " (tensors stay on CPU in simulation and CPU-only mode)"
        raise ValueError(
            f"tensor is on {tensor.device} but this rank communicates on {topology.device}{hint}"
        )
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
