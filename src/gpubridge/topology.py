"""Discovery, island and bridge groups, and leader election."""

from __future__ import annotations

import socket
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from gpubridge.config import CPU_BACKEND, CPU_ONLY_ENV, VENDOR_ENV, Config


@dataclass(frozen=True)
class Island:
    """All ranks of one vendor. They share a process group on the native backend."""

    vendor: str
    ranks: tuple[int, ...]
    """Global ranks in ascending order."""

    @property
    def leader(self) -> int:
        """The island's member of the bridge: its lowest global rank."""
        return self.ranks[0]


@dataclass(frozen=True)
class Layout:
    """How the cluster splits into islands. Depends only on each rank's vendor."""

    vendors: tuple[str, ...]
    """Vendor of each global rank."""
    islands: tuple[Island, ...]
    """Ordered by leader rank."""

    @property
    def world_size(self) -> int:
        return len(self.vendors)

    @property
    def bridge_ranks(self) -> tuple[int, ...]:
        """Global ranks of the island leaders, in ascending order."""
        return tuple(island.leader for island in self.islands)

    @property
    def needs_bridge(self) -> bool:
        """False for a single-vendor cluster, where collectives skip the bridge."""
        return len(self.islands) > 1

    def island_of(self, rank: int) -> Island:
        """Return the island containing global rank ``rank``."""
        vendor = self.vendors[rank]
        return next(island for island in self.islands if island.vendor == vendor)


def plan_topology(vendors: Sequence[str]) -> Layout:
    """Group ranks into islands by vendor and elect each island's leader.

    This is a pure function of the vendor map, so every rank that sees the
    same map computes the same islands, leaders and bridge.

    Args:
        vendors: The vendor of each global rank, indexed by rank.

    Raises:
        ValueError: if ``vendors`` is empty.
    """
    if not vendors:
        raise ValueError("cannot plan a topology for zero ranks")
    ranks_by_vendor: dict[str, list[int]] = {}
    for rank, vendor in enumerate(vendors):
        ranks_by_vendor.setdefault(vendor, []).append(rank)
    islands = sorted(
        (Island(vendor, tuple(ranks)) for vendor, ranks in ranks_by_vendor.items()),
        key=lambda island: island.leader,
    )
    return Layout(vendors=tuple(vendors), islands=tuple(islands))


@dataclass(frozen=True)
class PeerInfo:
    """What each rank reports about itself during discovery."""

    vendor: str
    simulated: bool
    torch_version: str
    hostname: str


def validate_peers(peers: Sequence[PeerInfo], *, warn: bool = True) -> None:
    """Fail if some ranks run on CPU and others on GPUs; warn if PyTorch releases differ.

    Raises:
        RuntimeError: if some ranks are in simulation or CPU-only mode and others are not.
    """
    on_cpu = [rank for rank, peer in enumerate(peers) if peer.simulated]
    if on_cpu and len(on_cpu) != len(peers):
        on_gpu = [rank for rank, peer in enumerate(peers) if not peer.simulated]
        raise RuntimeError(
            f"Ranks disagree on where to run: ranks {on_cpu} run on CPU ({VENDOR_ENV} or "
            f"{CPU_ONLY_ENV} is set) but ranks {on_gpu} run on GPUs. Use the same mode "
            "on every rank."
        )
    # CUDA and ROCm builds of one release differ only in the local version
    # suffix, e.g. 2.5.1+cu124 and 2.5.1+rocm6.2.
    releases = sorted({peer.torch_version.split("+")[0] for peer in peers}, key=_release_key)
    if warn and len(releases) > 1:
        warnings.warn(
            f"Ranks run different PyTorch releases ({', '.join(releases)}). Mixed releases "
            "passed gpubridge's CPU-only tests (2.9.1 to 2.14.1) but are untested on GPUs; "
            "prefer the same release on every rank.",
            stacklevel=2,
        )


def _release_key(release: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that orders 2.9.1 before 2.14.1."""
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in release.split("."))


@dataclass(frozen=True)
class Topology:
    """This rank's view of the cluster, built by :func:`gpubridge.init`."""

    rank: int
    layout: Layout
    peers: tuple[PeerInfo, ...]
    """What every rank reported during discovery, indexed by global rank."""
    config: Config
    island_group: ProcessGroup
    """This rank's island, on the native backend (Gloo when running on CPU)."""
    bridge_group: ProcessGroup | None
    """The Gloo group of island leaders. None on non-leaders and in single-vendor clusters."""

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def vendor(self) -> str:
        return self.layout.vendors[self.rank]

    @property
    def island(self) -> Island:
        return self.layout.island_of(self.rank)

    @property
    def is_leader(self) -> bool:
        return self.island.leader == self.rank

    @property
    def simulated(self) -> bool:
        return self.config.simulated

    @property
    def device(self) -> torch.device:
        """Where this rank's tensors must live: its GPU, or CPU in simulation and CPU-only mode."""
        return self.config.device


def build_topology(vendor: str, config: Config, *, timeout: timedelta | None = None) -> Topology:
    """Discover every rank's vendor, then create the island and bridge groups.

    Collective: every rank must call this after the global Gloo group exists.

    Args:
        vendor: This rank's vendor.
        config: This rank's resolved configuration.
        timeout: Timeout for the new groups' collectives. None keeps PyTorch's default.
    """
    rank = dist.get_rank()
    local = PeerInfo(vendor, config.simulated, torch.__version__, socket.gethostname())
    gathered: list[Any] = [None] * dist.get_world_size()
    # Exchange plain dicts so ranks only need to agree on pickling builtins.
    dist.all_gather_object(gathered, asdict(local))
    peers = tuple(PeerInfo(**info) for info in gathered)
    validate_peers(peers, warn=rank == 0)
    layout = plan_topology([peer.vendor for peer in peers])

    # new_group is collective over the whole world: every rank creates every
    # group, in the same order, including groups it is not a member of.
    island_group: ProcessGroup | None = None
    for island in layout.islands:
        group = dist.new_group(list(island.ranks), timeout=timeout, backend=config.island_backend)
        if rank in island.ranks:
            island_group = group
    bridge_group: ProcessGroup | None = None
    if layout.needs_bridge:
        group = dist.new_group(list(layout.bridge_ranks), timeout=timeout, backend=CPU_BACKEND)
        if rank in layout.bridge_ranks:
            bridge_group = group
    assert island_group is not None

    return Topology(
        rank=rank,
        layout=layout,
        peers=peers,
        config=config,
        island_group=island_group,
        bridge_group=bridge_group,
    )


_current: Topology | None = None


def get_topology() -> Topology:
    """Return this rank's topology.

    Raises:
        RuntimeError: if :func:`gpubridge.init` has not been called.
    """
    if _current is None:
        raise RuntimeError("gpubridge is not initialized; call gpubridge.init() first")
    return _current


def is_initialized() -> bool:
    """Return True between :func:`gpubridge.init` and :func:`gpubridge.destroy`."""
    return _current is not None


def _set_topology(topology: Topology | None) -> None:
    global _current
    _current = topology
