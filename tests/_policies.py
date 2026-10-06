"""A test-only policy that exercises the hooks: select_policy, setup and close."""

from __future__ import annotations

from collections import Counter

from gpubridge.policies import FlatGloo, ReduceBridgeBroadcast, register_policy

#: Below this many bytes, SizeSelecting sends a collective through FlatGloo.
SMALL = 1024


class _CountingFlatGloo(FlatGloo):
    def __init__(self, calls: Counter) -> None:
        self.calls = calls

    def all_reduce(self, tensor, topology):
        self.calls["small"] += 1
        super().all_reduce(tensor, topology)

    def broadcast(self, tensor, src, topology):
        self.calls["small"] += 1
        super().broadcast(tensor, src, topology)


@register_policy
class SizeSelecting(ReduceBridgeBroadcast):
    """FlatGloo below SMALL bytes, reduce-bridge-broadcast above.

    A size-based policy in miniature, built only from the public hooks.
    """

    name = "test-size-selecting"

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.small = _CountingFlatGloo(self.calls)
        self.closed = False

    def setup(self, topology) -> None:
        self.calls["setup"] += 1
        self.world_size = topology.world_size

    def close(self) -> None:
        self.closed = True

    def select_policy(self, nbytes):
        return self.small if nbytes < SMALL else self

    def all_reduce(self, tensor, topology):
        self.calls["large"] += 1
        super().all_reduce(tensor, topology)

    def broadcast(self, tensor, src, topology):
        self.calls["large"] += 1
        super().broadcast(tensor, src, topology)
