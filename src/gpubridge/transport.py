"""Bridge transports: how island leaders exchange data across vendors.

By default the bridge moves CPU tensors between island leaders, so anything
that can sum and broadcast a CPU tensor among a fixed set of processes can
carry it (a transport with ``device_tensors`` also takes GPU tensors). Gloo is
the default; ``bridge="ucx"`` (:mod:`gpubridge.ucx.transport`) is the other
built-in one. Another transport (MPI, an RDMA path) plugs in by subclassing
:class:`BridgeTransport` and registering it, without touching the
reduce-bridge-broadcast logic in ``collectives.py``.
"""

from __future__ import annotations

import abc
import contextvars
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from gpubridge.config import CPU_BACKEND

_scope: contextvars.ContextVar[Callable[[Sequence[int]], str] | None] = contextvars.ContextVar(
    "gpubridge_subgroup_scope", default=None)
"""Set while a subgroup's members build its topology (topology._members_only)."""


def scoped_name(ranks: Sequence[int]) -> str | None:
    """Inside a subgroup's setup, a name for a new group or link over ``ranks``: the same
    on every member, unique in the job. None otherwise.

    A transport whose setup keys anything by a per-process count (store keys, say)
    must use this when it isn't None: a subgroup's setup runs on its members only,
    so such counts would differ between ranks."""
    namer = _scope.get()
    return namer(ranks) if namer is not None else None


class BridgeTransport(abc.ABC):
    """Moves CPU tensors between island leaders. One instance per leader.

    Subclasses set :attr:`name`, which every rank reports during discovery so
    that ``init()`` can refuse a job whose ranks picked different transports.
    """

    name: ClassVar[str]

    #: Reductions :meth:`all_reduce` can do: always ``"sum"``, plus ``"max"`` and
    #: ``"min"`` if listed. Only a transport that lists one is called with it, as
    #: ``all_reduce(tensor, op=ReduceOp.MAX)``; SUM always comes without ``op``.
    reduce_ops: ClassVar[frozenset[str]] = frozenset({"sum"})

    #: True if :meth:`all_reduce` and :meth:`broadcast` also take tensors in GPU
    #: memory and order them against the caller's stream themselves. The default
    #: policy then hands GPU tensors straight to the transport instead of copying
    #: them to host memory first. Transports without it only ever see host tensors.
    device_tensors: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> BridgeTransport | None:
        """Set up the bridge between ``ranks`` (the island leaders).

        Collective over every rank that builds the topology, in the same order
        relative to other group creation, including ranks not in ``ranks``: the
        whole job for the world's topology; for a subgroup's (torch.distributed
        backend), its members only, with :func:`scoped_name` set and
        ``dist.new_group`` synchronizing only each new group's ranks.

        Returns:
            A transport on members of ``ranks``, None everywhere else.
        """

    @classmethod
    def create_groups(
        cls, groups: Sequence[Sequence[int]], *, timeout: timedelta | None = None
    ) -> list[BridgeTransport | None]:
        """Set up one bridge per entry of ``groups``, for policies that use several links.

        Collective like :meth:`create`: every rank that builds the topology
        calls it with the same ``groups`` in the same order. The default calls
        :meth:`create` once per group, in order, so every transport supports
        it; one that can set up several links more cheaply at once may
        override it. The leader bridge ``init()`` creates is unaffected.

        Returns:
            One entry per group: a transport on that group's members, None elsewhere.
        """
        return [cls.create(ranks, timeout=timeout) for ranks in groups]

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Sum a contiguous CPU tensor across all bridge members, in place.

        Transports that list ``"max"`` or ``"min"`` in :attr:`reduce_ops` also
        take ``op=ReduceOp.MAX`` / ``ReduceOp.MIN``.
        """

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        """Copy a contiguous CPU tensor from bridge member ``src`` (a global rank), in place."""

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release resources. Called by :func:`gpubridge.destroy`; the default does nothing."""


class GlooTransport(BridgeTransport):
    """The default bridge: a Gloo process group of the island leaders."""

    name = "gloo"
    reduce_ops = frozenset({"sum", "max", "min"})

    def __init__(self, group: ProcessGroup) -> None:
        self.group = group

    @classmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> GlooTransport | None:
        group = dist.new_group(list(ranks), timeout=timeout, backend=CPU_BACKEND)
        return cls(group) if dist.get_rank() in ranks else None

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        dist.all_reduce(tensor, op=op, group=self.group)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        dist.broadcast(tensor, src=src, group=self.group)


DEFAULT_TRANSPORT = GlooTransport.name

_TRANSPORTS: dict[str, type[BridgeTransport]] = {GlooTransport.name: GlooTransport}


def register_transport(transport: type[BridgeTransport]) -> type[BridgeTransport]:
    """Make ``transport`` selectable by name in ``gpubridge.init(bridge=...)``.

    Usable as a class decorator. Every rank must register the same transports.

    Raises:
        TypeError: if ``transport`` is not a :class:`BridgeTransport` subclass.
        ValueError: if a different class is already registered under its name.
    """
    if not (isinstance(transport, type) and issubclass(transport, BridgeTransport)):
        raise TypeError(f"expected a BridgeTransport subclass, got {transport!r}")
    existing = _TRANSPORTS.get(transport.name)
    if existing is not None and existing is not transport:
        raise ValueError(f"a different transport is already registered as {transport.name!r}")
    _TRANSPORTS[transport.name] = transport
    return transport


def resolve_transport(bridge: str | type[BridgeTransport]) -> type[BridgeTransport]:
    """Turn a transport name or class into a class.

    Raises:
        ValueError: for an unknown name.
        TypeError: for anything that is neither a name nor a transport class.
    """
    if isinstance(bridge, str):
        try:
            return _TRANSPORTS[bridge]
        except KeyError:
            known = ", ".join(sorted(_TRANSPORTS))
            raise ValueError(f"unknown bridge transport {bridge!r}; known: {known}") from None
    if isinstance(bridge, type) and issubclass(bridge, BridgeTransport):
        return bridge
    raise TypeError(f"bridge must be a transport name or BridgeTransport subclass, got {bridge!r}")
