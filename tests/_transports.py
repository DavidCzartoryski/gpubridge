"""Test bridge transports: a non-Gloo one (the bridge can be swapped) and a counting one."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge.transport import BridgeTransport, GlooTransport, register_transport

#: Path of the FileStore the leaders share; set by each test before spawning.
STORE_ENV = "GPUBRIDGE_TEST_BRIDGE_STORE"


@register_transport
class StoreTransport(BridgeTransport):
    """Bridges island leaders through a c10d FileStore instead of a process group.

    Each call writes this leader's bytes under a key unique to the call and reads
    the others'. Every leader sums in rank order, so all get bit-identical results.
    """

    name = "test-store"

    def __init__(self, store: dist.Store, ranks: Sequence[int]) -> None:
        self.store = store
        self.ranks = tuple(ranks)
        self.calls: Counter[str] = Counter()

    @classmethod
    def create(
        cls, ranks: Sequence[int], *, timeout: timedelta | None = None
    ) -> StoreTransport | None:
        if dist.get_rank() not in ranks:
            return None
        # One file per group, so several links (the sharded policy) don't share keys.
        path = f"{os.environ[STORE_ENV]}-{'-'.join(map(str, sorted(ranks)))}"
        return cls(dist.FileStore(path, len(ranks)), ranks)

    def _key(self, op: str, suffix: str = "") -> str:
        # Leaders make the same bridge calls in the same order, so the counter agrees.
        return f"{op}/{sum(self.calls.values())}/{suffix}"

    def all_reduce(self, tensor: torch.Tensor) -> None:
        key = self._key("all_reduce")
        self.calls["all_reduce"] += 1
        self.store.set(f"{key}{dist.get_rank()}", tensor.numpy().tobytes())
        total = torch.zeros_like(tensor)
        for rank in self.ranks:
            total += _from_bytes(self.store.get(f"{key}{rank}"), tensor)
        tensor.copy_(total)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        key = self._key("broadcast")
        self.calls["broadcast"] += 1
        if dist.get_rank() == src:
            self.store.set(key, tensor.numpy().tobytes())
        else:
            tensor.copy_(_from_bytes(self.store.get(key), tensor))


def _from_bytes(data: bytes, like: torch.Tensor) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=like.dtype).reshape(like.shape)


#: (call, numel) of every bridge call this process made through CountingGloo.
BRIDGE_CALLS: list[tuple[str, int]] = []


@register_transport
class CountingGloo(GlooTransport):
    """The Gloo bridge, noting every message's size and every create_groups call."""

    name = "test-counting-gloo"
    groups: list[list[list[int]]] = []  # noqa: RUF012 - per process, on purpose

    @classmethod
    def create_groups(cls, groups, *, timeout=None):
        cls.groups.append([list(ranks) for ranks in groups])
        return super().create_groups(groups, timeout=timeout)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        BRIDGE_CALLS.append(("all_reduce", tensor.numel()))
        super().all_reduce(tensor, op)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        BRIDGE_CALLS.append(("broadcast", tensor.numel()))
        super().broadcast(tensor, src)
