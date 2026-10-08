"""Collective policies: how a cluster-wide collective is put together.

A policy decides which process groups carry an all_reduce or broadcast, and in
which order. It never changes what the collective computes. ``collectives.py``
validates each call and hands it to the active policy, which ``init()`` picks
once and checks is the same on every rank. Policies are registered by name,
the same way ``transport.py`` registers bridge transports.

Built in:

- ``reduce-bridge-broadcast``: island collective, island leaders over the
  bridge, island broadcast. The default for clusters with two or more islands.
- ``native-only``: one native collective on the single island. The default for
  single-vendor clusters.
- ``flat-gloo``: every rank stages its tensor to CPU and uses the world Gloo
  group. Slow but obviously correct: the reference in tests, and a fallback for
  debugging that never touches NCCL/RCCL communicators.

``auto`` (the default) picks ``native-only`` for one island and
``reduce-bridge-broadcast`` otherwise.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist

from gpubridge.observe import phase

if TYPE_CHECKING:
    from gpubridge.topology import Layout, Topology

AUTO = "auto"


def _stage_to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` to CPU for the bridge or Gloo.

    Returns the tensor itself when it is already on CPU (simulation mode), so
    the copy back is skipped there. Tests replace this with ``clone`` to run the
    copy-out-and-back path that GPUs take.
    """
    return tensor.cpu()


class CollectivePolicy(abc.ABC):
    """How all_reduce and broadcast travel between ranks. One instance per rank.

    ``init()`` refuses a job whose ranks asked for different policies, and the
    public collectives require the same shape and dtype on every rank. So every
    rank always takes the same path, which is what keeps a policy from hanging.
    """

    name: ClassVar[str]

    @classmethod
    @abc.abstractmethod
    def applies_to(cls, layout: Layout) -> bool:
        """Whether this policy can run on a cluster with this island layout."""

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        """Sum an already-validated tensor across every rank, in place."""

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        """Copy an already-validated tensor from global rank ``src`` to every rank, in place."""

    def setup(self, topology: Topology) -> None:  # noqa: B027 - optional hook
        """Called once by ``init()`` after every group exists, e.g. to allocate staging buffers."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Called by ``destroy()``. Release anything :meth:`setup` acquired."""

    def select_policy(self, nbytes: int) -> CollectivePolicy:
        """Return the policy that carries a collective of ``nbytes`` bytes.

        The hook for choosing by message size, e.g. one path for small messages
        and a pipelined one for large messages. The default uses this policy for
        every size. An override may depend only on ``nbytes`` and on state that
        is identical on every rank, so that every rank picks the same path.
        """
        return self


class ReduceBridgeBroadcast(CollectivePolicy):
    """Reduce in each island, combine island sums over the bridge, broadcast back.

    all_reduce:

    1. all_reduce within each island on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    broadcast: from ``src`` to its island, then from that island's leader to the
    other leaders over the bridge, then from each leader to its island.
    """

    name = "reduce-bridge-broadcast"

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return layout.needs_bridge

    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        with phase("island-reduce"):
            dist.all_reduce(tensor, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            with phase("bridge"):
                staged = _stage_to_host(tensor)
                topology.bridge.all_reduce(staged)
                if staged is not tensor:
                    tensor.copy_(staged)
        with phase("island-broadcast"):
            dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        src_island = topology.layout.island_of(src)
        in_src_island = topology.island == src_island
        if in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=src, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            with phase("bridge"):
                staged = _stage_to_host(tensor)
                topology.bridge.broadcast(staged, src=src_island.leader)
                if not in_src_island and staged is not tensor:
                    tensor.copy_(staged)
        if not in_src_island:
            with phase("island-broadcast"):
                dist.broadcast(tensor, src=topology.island.leader, group=topology.island_group)


class NativeOnly(CollectivePolicy):
    """One native collective on the only island: NCCL, RCCL, or Gloo on CPU."""

    name = "native-only"

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return not layout.needs_bridge

    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        dist.all_reduce(tensor, group=topology.island_group)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        dist.broadcast(tensor, src=src, group=topology.island_group)


class FlatGloo(CollectivePolicy):
    """Every rank copies its tensor to CPU and runs the collective on the world Gloo group.

    Slow, but it has no islands, no bridge and no leaders, so it is the
    reference other policies are tested against, and a fallback for debugging.
    """

    name = "flat-gloo"

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return True

    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        staged = _stage_to_host(tensor)
        dist.all_reduce(staged, group=dist.group.WORLD)
        if staged is not tensor:
            tensor.copy_(staged)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        staged = _stage_to_host(tensor)
        dist.broadcast(staged, src=src, group=dist.group.WORLD)
        if staged is not tensor:
            tensor.copy_(staged)


# ---- Planned, not implemented -------------------------------------------------
#
# PipelinedReduceBridgeBroadcast (docs/PRIOR_ART.md, Triton-distributed idea 1):
# reduce-bridge-broadcast with the leaders' bridge step split into chunks, so
# the GPU->CPU copy of chunk k+1, the bridge all_reduce of chunk k and the
# CPU->GPU copy of chunk k-1 overlap. It needs no API change:
#   - setup(): allocate pinned staging buffers and a side stream once;
#   - all_reduce() / broadcast(): the chunked pipeline;
#   - close(): free the buffers.
# Size-based choice (idea 2) is select_policy(nbytes) on a policy that returns,
# say, FlatGloo below a threshold and the pipelined policy above it. Thresholds
# come from scripts/gpu/bench_all_reduce.py runs on real GPUs (--policy).
# -------------------------------------------------------------------------------

DEFAULT_POLICY = AUTO

_POLICIES: dict[str, type[CollectivePolicy]] = {
    cls.name: cls for cls in (ReduceBridgeBroadcast, NativeOnly, FlatGloo)
}


def register_policy(policy: type[CollectivePolicy]) -> type[CollectivePolicy]:
    """Make ``policy`` selectable by name in ``gpubridge.init(policy=...)``.

    Usable as a class decorator. Every rank must register the same policies.

    Raises:
        TypeError: if ``policy`` is not a :class:`CollectivePolicy` subclass.
        ValueError: if the name is ``auto`` or another class already uses it.
    """
    if not (isinstance(policy, type) and issubclass(policy, CollectivePolicy)):
        raise TypeError(f"expected a CollectivePolicy subclass, got {policy!r}")
    if policy.name == AUTO:
        raise ValueError(f"{AUTO!r} is reserved for automatic selection")
    existing = _POLICIES.get(policy.name)
    if existing is not None and existing is not policy:
        raise ValueError(f"a different policy is already registered as {policy.name!r}")
    _POLICIES[policy.name] = policy
    return policy


def resolve_policy(policy: str | type[CollectivePolicy]) -> str | type[CollectivePolicy]:
    """Check a requested policy: ``"auto"``, a registered name, or a policy class.

    Returns ``"auto"`` or a policy class.

    Raises:
        ValueError: for an unknown name.
        TypeError: for anything that is neither a name nor a policy class.
    """
    if policy == AUTO:
        return AUTO
    if isinstance(policy, str):
        try:
            return _POLICIES[policy]
        except KeyError:
            known = ", ".join([AUTO, *sorted(_POLICIES)])
            raise ValueError(f"unknown collective policy {policy!r}; known: {known}") from None
    if isinstance(policy, type) and issubclass(policy, CollectivePolicy):
        return policy
    raise TypeError(f"policy must be a policy name or CollectivePolicy subclass, got {policy!r}")


def policy_name(policy: str | type[CollectivePolicy]) -> str:
    """The name a rank reports during discovery for a resolved policy."""
    return policy if isinstance(policy, str) else policy.name


def choose_policy(policy: str | type[CollectivePolicy], layout: Layout) -> type[CollectivePolicy]:
    """Turn a resolved policy into the class to run on ``layout``.

    ``auto`` picks ``native-only`` for one island and ``reduce-bridge-broadcast``
    otherwise. Every rank sees the same layout, so every rank chooses the same.

    Raises:
        RuntimeError: if the policy doesn't apply to this layout.
    """
    if policy == AUTO:
        return ReduceBridgeBroadcast if layout.needs_bridge else NativeOnly
    assert not isinstance(policy, str)
    if not policy.applies_to(layout):
        islands = len(layout.islands)
        usable = ", ".join([AUTO, *sorted(n for n, c in _POLICIES.items() if c.applies_to(layout))])
        raise RuntimeError(
            f"Collective policy {policy.name!r} doesn't apply to this cluster "
            f"({islands} island{'s' if islands != 1 else ''}). Policies that do: {usable}."
        )
    return policy
