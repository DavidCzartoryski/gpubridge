"""Collective policies: how a cluster-wide collective is put together.

A policy decides which process groups carry an all_reduce or broadcast, and in
which order. It never changes what the collective computes. ``collectives.py``
validates each call and hands it to the active policy, which ``init()`` picks
once and checks is the same on every rank. Policies are registered by name,
the same way ``transport.py`` registers bridge transports.

Built in:

- ``reduce-bridge-broadcast``: island reduce onto each leader, island leaders
  over the bridge, island broadcast. The default for clusters with two or
  more islands.
- ``native-only``: one native collective on the single island. The default for
  single-vendor clusters.
- ``flat-gloo``: every rank stages its tensor to CPU and uses the world Gloo
  group. Slow but obviously correct: the reference in tests, and a fallback for
  debugging that never touches NCCL/RCCL communicators.

``auto`` (the default) picks ``native-only`` for one island and
``reduce-bridge-broadcast`` otherwise.

What each built-in policy carries:

==========================  =====================  ==========================
collective                  reductions             policies
==========================  =====================  ==========================
all_reduce                  SUM, AVG, MAX, MIN     all three
broadcast                   (none)                 all three
all_gather_into_tensor      (none)                 all three
reduce_scatter_tensor       SUM, AVG               all three
==========================  =====================  ==========================

AVG is always a SUM followed by a division in ``collectives.py``, so policies
never see it. MAX and MIN over the bridge also need the bridge transport to
list them (the default Gloo transport does). A custom policy that doesn't
override ``all_gather_into_tensor`` or ``reduce_scatter_tensor`` simply
doesn't offer them: the call fails on every rank before any communication.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge.config import GPU_BACKEND
from gpubridge.observe import phase

if TYPE_CHECKING:
    from gpubridge.topology import Layout, Topology
    from gpubridge.transport import BridgeTransport

AUTO = "auto"


def _stage_to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` to CPU for the bridge or Gloo.

    Returns the tensor itself when it is already on CPU (simulation mode), so
    the copy back is skipped there. Tests replace this with ``clone`` to run the
    copy-out-and-back path that GPUs take.
    """
    return tensor.cpu()


def _host_like(tensor: torch.Tensor) -> torch.Tensor:
    """A CPU buffer to receive ``tensor``'s data into.

    On CPU it is whatever :func:`_stage_to_host` returns (the tensor itself,
    or a copy in tests); on GPUs a new buffer, since its old contents aren't
    needed.
    """
    if tensor.device.type == "cpu":
        return _stage_to_host(tensor)
    return torch.empty(tensor.shape, dtype=tensor.dtype)


def _bridge_all_reduce(bridge: BridgeTransport, tensor: torch.Tensor, op: ReduceOp.RedOpType
                       ) -> None:
    # SUM goes without op, so transports written before reductions existed still work.
    if op == ReduceOp.SUM:
        bridge.all_reduce(tensor)
    else:
        bridge.all_reduce(tensor, op=op)


def _native(topology: Topology) -> bool:
    """Whether islands run NCCL/RCCL (GPUs), which have the tensor collectives built in."""
    return topology.config.island_backend == GPU_BACKEND


def _all_gather(output: torch.Tensor, input: torch.Tensor, group, native: bool) -> None:  # noqa: A002
    """all_gather ``input`` into ``output`` (rows in group-rank order) on ``group``."""
    if native:
        dist.all_gather_into_tensor(output, input, group=group)
    else:  # Gloo: the list form works on every supported torch release
        size = dist.get_world_size(group)
        dist.all_gather(list(output.view(-1).chunk(size)), input.reshape(-1), group=group)


def _reduce_scatter(output: torch.Tensor, input: torch.Tensor, group, rank: int,  # noqa: A002
                    native: bool) -> None:
    """SUM ``input`` over ``group`` and keep slice ``rank`` (a group rank) in ``output``."""
    if native:
        dist.reduce_scatter_tensor(output, input, group=group)
    else:  # Gloo: all_reduce a copy, then slice it; the input stays unmodified
        work = input.clone()
        dist.all_reduce(work, group=group)
        size = dist.get_world_size(group)
        output.view(-1).copy_(work.view(size, -1)[rank])


class CollectivePolicy(abc.ABC):
    """How all_reduce and broadcast travel between ranks. One instance per rank.

    ``init()`` refuses a job whose ranks asked for different policies, and the
    public collectives require the same shape and dtype on every rank. So every
    rank always takes the same path, which is what keeps a policy from hanging.
    """

    name: ClassVar[str]

    #: Reductions :meth:`all_reduce` can carry: always ``"sum"``, plus ``"max"``
    #: and ``"min"`` if listed. Only a policy that lists one is called with it,
    #: as ``all_reduce(tensor, topology, op=ReduceOp.MAX)``; SUM always comes
    #: without ``op``, so policies written before reductions existed still work.
    reduce_ops: ClassVar[frozenset[str]] = frozenset({"sum"})

    @classmethod
    @abc.abstractmethod
    def applies_to(cls, layout: Layout) -> bool:
        """Whether this policy can run on a cluster with this island layout."""

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor, topology: Topology) -> None:
        """Sum an already-validated tensor across every rank, in place.

        A policy that lists ``"max"`` or ``"min"`` in :attr:`reduce_ops` also
        takes ``op=ReduceOp.MAX`` / ``ReduceOp.MIN``.
        """

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        """Copy an already-validated tensor from global rank ``src`` to every rank, in place."""

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        """Gather every rank's ``input`` into ``output``, ordered by global rank.

        Optional. Policies that don't override it don't offer the collective;
        ``gpubridge.all_gather_into_tensor`` checks that before communicating.
        """
        raise NotImplementedError(
            f"collective policy {self.name!r} doesn't implement all_gather_into_tensor")

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        """SUM every rank's ``input`` and leave this rank's slice in ``output``.

        Optional, like :meth:`all_gather_into_tensor`. Must not modify ``input``.
        """
        raise NotImplementedError(
            f"collective policy {self.name!r} doesn't implement reduce_scatter_tensor")

    def supported_reduce_ops(self, topology: Topology) -> frozenset[str]:
        """The reductions this policy can carry on this cluster.

        Depends only on state that is identical on every rank, so a call it
        rejects is rejected everywhere. The default is :attr:`reduce_ops`.
        """
        return self.reduce_ops

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

    1. reduce within each island onto its leader, on the native backend.
    2. Island leaders copy the island sum to CPU and all_reduce it over the bridge.
    3. Each leader copies the total back to its device and broadcasts it to its island.

    Step 1 is a reduce, not an all_reduce, because only the leader uses the
    island sum: the other ranks' copies would be overwritten by step 3 anyway.
    A ring reduce moves about half the bytes of a ring all_reduce. The other
    ranks' tensors hold unspecified values between steps 1 and 3. MAX and MIN
    take the same three steps.

    broadcast: from ``src`` to its island, then from that island's leader to the
    other leaders over the bridge, then from each leader to its island.

    all_gather_into_tensor: island gather onto the leader, then each leader in
    turn broadcasts its island's rows over the bridge, then each leader
    broadcasts the assembled output to its island. Bridge broadcasts are pure
    copies, so the result is exact for any data.

    reduce_scatter_tensor: island reduce onto the leader, bridge all_reduce of
    the whole input, then each leader scatters its island members' slices.
    """

    name = "reduce-bridge-broadcast"
    reduce_ops = frozenset({"sum", "max", "min"})

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return layout.needs_bridge

    def supported_reduce_ops(self, topology: Topology) -> frozenset[str]:
        assert topology.transport is not None
        return self.reduce_ops & topology.transport.reduce_ops

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        leader = topology.island.leader
        with phase("island-reduce"):
            dist.reduce(tensor, dst=leader, op=op, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            with phase("bridge"):
                staged = _stage_to_host(tensor)
                _bridge_all_reduce(topology.bridge, staged, op)
                if staged is not tensor:
                    tensor.copy_(staged)
        with phase("island-broadcast"):
            dist.broadcast(tensor, src=leader, group=topology.island_group)

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

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        island, world, n = topology.island, topology.world_size, input.numel()
        rows: torch.Tensor | None = None
        with phase("island-gather"):
            if topology.is_leader:
                rows = torch.empty((len(island.ranks), n), dtype=input.dtype, device=input.device)
            dist.gather(input.reshape(-1), list(rows.unbind(0)) if rows is not None else None,
                        dst=island.leader, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None and rows is not None
            with phase("bridge"):
                assembled = _host_like(output).view(world, n)
                mine = _stage_to_host(rows)
                for other in topology.layout.islands:  # same order on every leader
                    staged = mine if other == island else torch.empty(
                        (len(other.ranks), n), dtype=input.dtype)
                    topology.bridge.broadcast(staged, src=other.leader)
                    assembled[list(other.ranks)] = staged
                if assembled.data_ptr() != output.data_ptr():
                    output.view(world, n).copy_(assembled)
        with phase("island-broadcast"):
            dist.broadcast(output, src=island.leader, group=topology.island_group)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        island, world = topology.island, topology.world_size
        work = input.clone()  # the input stays unmodified
        with phase("island-reduce"):
            dist.reduce(work, dst=island.leader, group=topology.island_group)
        if topology.is_leader:
            assert topology.bridge is not None
            with phase("bridge"):
                staged = _stage_to_host(work)
                topology.bridge.all_reduce(staged)
                if staged is not work:
                    work.copy_(staged)
        with phase("island-scatter"):
            slices = work.view(world, -1)
            dist.scatter(output.view(-1),
                         [slices[r] for r in island.ranks] if topology.is_leader else None,
                         src=island.leader, group=topology.island_group)


class NativeOnly(CollectivePolicy):
    """One native collective on the only island: NCCL, RCCL, or Gloo on CPU."""

    name = "native-only"
    reduce_ops = frozenset({"sum", "max", "min"})

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return not layout.needs_bridge

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        dist.all_reduce(tensor, op=op, group=topology.island_group)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        dist.broadcast(tensor, src=src, group=topology.island_group)

    # The only island holds every rank in ascending order, so group ranks are
    # global ranks and native output order is global order.
    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        _all_gather(output, input, topology.island_group, _native(topology))

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        _reduce_scatter(output, input, topology.island_group, topology.rank, _native(topology))


class FlatGloo(CollectivePolicy):
    """Every rank copies its tensor to CPU and runs the collective on the world Gloo group.

    Slow, but it has no islands, no bridge and no leaders, so it is the
    reference other policies are tested against, and a fallback for debugging.
    """

    name = "flat-gloo"
    reduce_ops = frozenset({"sum", "max", "min"})

    @classmethod
    def applies_to(cls, layout: Layout) -> bool:
        return True

    def all_reduce(self, tensor: torch.Tensor, topology: Topology,
                   op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        staged = _stage_to_host(tensor)
        dist.all_reduce(staged, op=op, group=dist.group.WORLD)
        if staged is not tensor:
            tensor.copy_(staged)

    def broadcast(self, tensor: torch.Tensor, src: int, topology: Topology) -> None:
        staged = _stage_to_host(tensor)
        dist.broadcast(staged, src=src, group=dist.group.WORLD)
        if staged is not tensor:
            tensor.copy_(staged)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                               topology: Topology) -> None:
        received = _host_like(output)
        _all_gather(received, _stage_to_host(input), dist.group.WORLD, native=False)
        if received.data_ptr() != output.data_ptr():
            output.copy_(received)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor,  # noqa: A002
                              topology: Topology) -> None:
        received = _host_like(output)
        _reduce_scatter(received, _stage_to_host(input), dist.group.WORLD, topology.rank,
                        native=False)
        if received.data_ptr() != output.data_ptr():
            output.copy_(received)


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


def implements(policy: CollectivePolicy | type[CollectivePolicy], fn: str) -> bool:
    """Whether ``policy`` overrides the optional collective ``fn``.

    ``fn`` is a method name, e.g. ``"reduce_scatter_tensor"``.
    """
    cls = policy if isinstance(policy, type) else type(policy)
    return getattr(cls, fn) is not getattr(CollectivePolicy, fn)


def policies_implementing(fn: str, layout: Layout) -> list[str]:
    """Registered policies that implement ``fn`` and apply to ``layout``, by name."""
    return sorted(name for name, cls in _POLICIES.items()
                  if cls.applies_to(layout) and implements(cls, fn))


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
