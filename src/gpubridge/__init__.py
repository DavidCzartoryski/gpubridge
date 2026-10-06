"""gpubridge: write collective communication once, run it on mixed NVIDIA and AMD clusters.

Ranks are grouped into one island per vendor, each using its native backend
(NCCL on NVIDIA, RCCL on AMD). Island leaders are linked by a CPU Gloo bridge,
and cluster-wide collectives are composed from island collectives and the bridge.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import topology as _topology
from gpubridge.collectives import all_reduce, barrier, broadcast
from gpubridge.config import CPU_BACKEND, resolve_config
from gpubridge.detect import detect_vendor
from gpubridge.topology import Island, Layout, PeerInfo, Topology, get_topology, is_initialized

__version__ = "0.1.0"

__all__ = [
    "Island",
    "Layout",
    "PeerInfo",
    "ReduceOp",
    "Topology",
    "all_reduce",
    "barrier",
    "broadcast",
    "destroy",
    "get_topology",
    "init",
    "is_initialized",
]


def init(
    init_method: str | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    timeout: timedelta | None = None,
) -> Topology:
    """Join the cluster and build the island and bridge groups.

    Call once on every rank. The arguments mirror
    ``torch.distributed.init_process_group``. With none given, the ``env://``
    variables set by torchrun are used (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE).

    gpubridge owns the default process group: it is a Gloo group over all
    ranks, used for discovery and :func:`barrier`.

    Args:
        init_method: Rendezvous URL, e.g. ``env://``, ``tcp://host:port`` or ``file:///path``.
        rank: This process's global rank. Read from the environment if omitted.
        world_size: Total number of ranks. Read from the environment if omitted.
        timeout: Timeout for every group gpubridge creates. None keeps PyTorch's default.

    Returns:
        This rank's :class:`Topology`.

    Raises:
        RuntimeError: if gpubridge or torch.distributed is already initialized,
            if no vendor can be detected, or if ranks disagree on simulation mode.
    """
    if is_initialized():
        raise RuntimeError("gpubridge is already initialized")
    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized; gpubridge.init() creates the "
            "default process group itself"
        )
    vendor = detect_vendor()
    config = resolve_config()
    if not config.simulated:
        torch.cuda.set_device(config.device)

    kwargs: dict[str, Any] = {}
    if rank is not None:
        kwargs["rank"] = rank
    if world_size is not None:
        kwargs["world_size"] = world_size
    if timeout is not None:
        kwargs["timeout"] = timeout
    dist.init_process_group(backend=CPU_BACKEND, init_method=init_method, **kwargs)
    try:
        topology = _topology.build_topology(vendor, config, timeout=timeout)
    except BaseException:
        dist.destroy_process_group()
        raise
    _topology._set_topology(topology)
    return topology


def destroy() -> None:
    """Tear down every process group created by :func:`init`.

    Does nothing if gpubridge is not initialized.
    """
    if not is_initialized():
        return
    _topology._set_topology(None)
    dist.destroy_process_group()
