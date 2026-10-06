"""gpubridge: write collective communication once, run it on mixed NVIDIA and AMD clusters.

Ranks are grouped into one island per vendor, each using its native backend
(NCCL on NVIDIA, RCCL on AMD). Island leaders are linked by a CPU bridge (Gloo by
default, swappable via :mod:`gpubridge.transport`), and cluster-wide collectives
are composed from island collectives and the bridge by a collective policy
(:mod:`gpubridge.policies`).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import topology as _topology
from gpubridge.collectives import all_reduce, barrier, broadcast
from gpubridge.config import CPU_BACKEND
from gpubridge.detect import Probe, probe
from gpubridge.policies import (
    DEFAULT_POLICY,
    CollectivePolicy,
    policy_name,
    register_policy,
    resolve_policy,
)
from gpubridge.split_test import SplitTestWarning
from gpubridge.topology import Island, Layout, PeerInfo, Topology, get_topology, is_initialized
from gpubridge.transport import (
    DEFAULT_TRANSPORT,
    BridgeTransport,
    GlooTransport,
    register_transport,
    resolve_transport,
)

__version__ = "0.1.0"

__all__ = [
    "BridgeTransport",
    "CollectivePolicy",
    "GlooTransport",
    "Island",
    "Layout",
    "PeerInfo",
    "Probe",
    "ReduceOp",
    "SplitTestWarning",
    "Topology",
    "all_reduce",
    "barrier",
    "broadcast",
    "destroy",
    "get_topology",
    "init",
    "is_initialized",
    "probe",
    "register_policy",
    "register_transport",
]


def init(
    init_method: str | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    timeout: timedelta | None = None,
    bridge: str | type[BridgeTransport] = DEFAULT_TRANSPORT,
    policy: str | type[CollectivePolicy] = DEFAULT_POLICY,
) -> Topology:
    """Join the cluster and build the island and bridge groups.

    Call once on every rank. The arguments mirror
    ``torch.distributed.init_process_group``. With none given, the ``env://``
    variables set by torchrun are used (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE).

    gpubridge owns the default process group: it is a Gloo group over all
    ranks, used for discovery and :func:`barrier`. During discovery every rank
    shares its :func:`probe` results and any problems that would stop it from
    joining, so a rank without a GPU or a vendor fails ``init()`` on every rank
    with one message instead of leaving the others waiting.

    Args:
        init_method: Rendezvous URL, e.g. ``env://``, ``tcp://host:port`` or ``file:///path``.
        rank: This process's global rank. Read from the environment if omitted.
        world_size: Total number of ranks. Read from the environment if omitted.
        timeout: Timeout for every group gpubridge creates. None keeps PyTorch's default.
        bridge: Transport between island leaders: a registered name or a
            :class:`BridgeTransport` subclass. Must be the same on every rank.
        policy: How collectives are put together: ``"auto"``, a registered
            name (``"reduce-bridge-broadcast"``, ``"native-only"``,
            ``"flat-gloo"``) or a :class:`CollectivePolicy` subclass. Must be the
            same on every rank. ``"auto"`` picks ``native-only`` for one island
            and ``reduce-bridge-broadcast`` otherwise.

    Returns:
        This rank's :class:`Topology`.

    Raises:
        RuntimeError: if gpubridge or torch.distributed is already initialized,
            if any rank reported a problem (no vendor, no GPU, no NCCL/RCCL
            backend), if ranks disagree on CPU mode, the bridge transport or the
            collective policy, or if the policy doesn't apply to the cluster.
        ValueError: for an unknown bridge or policy name, or invalid gpubridge env vars.
    """
    if is_initialized():
        raise RuntimeError("gpubridge is already initialized")
    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized; gpubridge.init() creates the "
            "default process group itself"
        )
    transport = resolve_transport(bridge)
    chosen = resolve_policy(policy)
    local, config = _topology.describe_local(transport.name, policy_name(chosen))
    if not config.simulated and not local.problems:
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
        topology = _topology.build_topology(
            local, config, transport=transport, policy=chosen, timeout=timeout
        )
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
    topology = get_topology()
    _topology._set_topology(None)
    try:
        topology.policy.close()
        if topology.bridge is not None:
            topology.bridge.close()
    finally:
        dist.destroy_process_group()
