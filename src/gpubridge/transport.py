"""Bridge transports: how island leaders exchange data across vendors.

The bridge only ever moves CPU tensors between island leaders, so anything
that can sum and broadcast a CPU tensor among a fixed set of processes can
carry it. Gloo is the default. Another transport (MPI, UCX, an RDMA path)
plugs in by subclassing :class:`BridgeTransport` and registering it, without
touching the reduce-bridge-broadcast logic in ``collectives.py``.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from gpubridge.config import CPU_BACKEND


class BridgeTransport(abc.ABC):
    """Moves CPU tensors between island leaders. One instance per leader.

    Subclasses set :attr:`name`, which every rank reports during discovery so
    that ``init()`` can refuse a job whose ranks picked different transports.
    """

    name: ClassVar[str]

    @classmethod
    @abc.abstractmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> BridgeTransport | None:
        """Set up the bridge between ``ranks`` (the island leaders).

        Collective over the whole world: every rank calls it, in the same order
        relative to other group creation, including ranks not in ``ranks``.

        Returns:
            A transport on members of ``ranks``, None everywhere else.
        """

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Sum a contiguous CPU tensor across all bridge members, in place."""

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        """Copy a contiguous CPU tensor from bridge member ``src`` (a global rank), in place."""

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release resources. Called by :func:`gpubridge.destroy`; the default does nothing."""


class GlooTransport(BridgeTransport):
    """The default bridge: a Gloo process group of the island leaders."""

    name = "gloo"

    def __init__(self, group: ProcessGroup) -> None:
        self.group = group

    @classmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> GlooTransport | None:
        group = dist.new_group(list(ranks), timeout=timeout, backend=CPU_BACKEND)
        return cls(group) if dist.get_rank() in ranks else None

    def all_reduce(self, tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, group=self.group)

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
