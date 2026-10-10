"""Discovery, island and bridge groups, and leader election."""

from __future__ import annotations

import socket
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from gpubridge.config import (
    CPU_ONLY_ENV,
    GPU_BACKEND,
    SPLIT_TEST_ENV,
    VENDOR_ENV,
    Config,
    resolve_config,
    split_test_mode,
)
from gpubridge.detect import Probe, detect_vendor, probe
from gpubridge.policies import AUTO, CollectivePolicy, choose_policy
from gpubridge.split_test import SplitTestWarning, banner, split_labels
from gpubridge.transport import BridgeTransport


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

    vendor: str | None
    """None if this rank could not work out its vendor; see :attr:`problems`."""
    simulated: bool
    hostname: str
    bridge: str
    """Name of the bridge transport this rank was asked to use."""
    probe: Probe
    """The rank's build, GPUs and backends, from :func:`gpubridge.probe`."""
    problems: tuple[str, ...] = ()
    """Why this rank cannot take part. A problem on any rank fails init() on every rank."""
    split_test: str = ""
    """This rank's ``GPUBRIDGE_SPLIT_TEST`` mode, or empty when off. TEST ONLY."""
    policy: str = AUTO
    """Name of the collective policy this rank was asked to use (``auto`` by default)."""

    @property
    def torch_version(self) -> str:
        return self.probe.torch_version

    def to_wire(self) -> dict[str, Any]:
        """Plain builtins for all_gather_object, so ranks only need to agree on pickling those."""
        return asdict(self)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> PeerInfo:
        data = dict(data)
        probe_data = dict(data.pop("probe"))
        probe_data["errors"] = tuple(probe_data.get("errors", ()))
        data["problems"] = tuple(data.get("problems", ()))
        return cls(probe=Probe(**probe_data), **data)


def describe_local(bridge: str, policy: str = AUTO) -> tuple[PeerInfo, Config]:
    """Work out this rank's vendor, configuration and capabilities.

    Anything that would stop this rank from joining (no detectable vendor, no
    GPU, no NCCL/RCCL backend) is recorded in :attr:`PeerInfo.problems` rather
    than raised. Discovery then reports every rank's problems at once, on every
    rank, instead of one rank exiting and the others hanging until a timeout.

    Raises:
        ValueError: for invalid ``GPUBRIDGE_VENDOR``, ``GPUBRIDGE_CPU_ONLY`` or
            ``GPUBRIDGE_SPLIT_TEST`` values, which are configuration mistakes rather
            than properties of the machine.
    """
    split = split_test_mode() or ""
    local_probe = probe()
    problems: list[str] = []
    vendor: str | None
    try:
        vendor = detect_vendor()
    except RuntimeError as exc:
        vendor = None
        problems.append(str(exc))
    try:
        config = resolve_config()
    except RuntimeError as exc:
        problems.append(str(exc))
        # Never used for communication: init() fails during discovery.
        config = Config(simulated=False, island_backend=GPU_BACKEND, device=torch.device("cpu"))
    if not config.simulated and not local_probe.nccl_available:
        problems.append(
            "This PyTorch build has no NCCL backend (RCCL on ROCm builds), so it cannot "
            "form a GPU island."
        )
    local = PeerInfo(
        vendor=vendor,
        simulated=config.simulated,
        hostname=socket.gethostname(),
        bridge=bridge,
        probe=local_probe,
        problems=tuple(problems),
        split_test=split,
        policy=policy,
    )
    return local, config


def validate_peers(peers: Sequence[PeerInfo], *, warn: bool = True) -> None:
    """Check that every rank can join and that all ranks agree on how to run.

    Warns if PyTorch releases differ.

    Raises:
        RuntimeError: if any rank reported problems, if some ranks run on CPU and
            others on GPUs, or if ranks picked different bridge transports,
            collective policies or split-test modes.
    """
    failing = [(rank, peer) for rank, peer in enumerate(peers) if peer.problems]
    if failing:
        lines = [
            f"  rank {rank} ({peer.hostname}): {problem}"
            for rank, peer in failing
            for problem in peer.problems
        ]
        raise RuntimeError(
            f"gpubridge cannot start: {len(failing)} of {len(peers)} ranks reported "
            "problems.\n" + "\n".join(lines)
        )
    on_cpu = [rank for rank, peer in enumerate(peers) if peer.simulated]
    if on_cpu and len(on_cpu) != len(peers):
        on_gpu = [rank for rank, peer in enumerate(peers) if not peer.simulated]
        raise RuntimeError(
            f"Ranks disagree on where to run: ranks {on_cpu} run on CPU ({VENDOR_ENV} or "
            f"{CPU_ONLY_ENV} is set) but ranks {on_gpu} run on GPUs. Use the same mode "
            "on every rank."
        )
    bridges: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        bridges.setdefault(peer.bridge, []).append(rank)
    if len(bridges) > 1:
        detail = "; ".join(f"{name!r} on ranks {ranks}" for name, ranks in bridges.items())
        raise RuntimeError(
            f"Ranks picked different bridge transports: {detail}. Pass the same bridge "
            "to gpubridge.init() on every rank."
        )
    policies: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        policies.setdefault(peer.policy, []).append(rank)
    if len(policies) > 1:
        detail = "; ".join(f"{name!r} on ranks {ranks}" for name, ranks in policies.items())
        raise RuntimeError(
            f"Ranks picked different collective policies: {detail}. Pass the same policy "
            "to gpubridge.init() on every rank."
        )
    splits: dict[str, list[int]] = {}
    for rank, peer in enumerate(peers):
        splits.setdefault(peer.split_test or "off", []).append(rank)
    if len(splits) > 1:
        detail = "; ".join(f"{mode} on ranks {ranks}" for mode, ranks in splits.items())
        raise RuntimeError(
            f"Ranks disagree on {SPLIT_TEST_ENV}: {detail}. Set it to the same value on "
            "every rank, or on none."
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
    bridge: BridgeTransport | None
    """The transport linking island leaders. None on non-leaders and in single-vendor clusters."""
    policy: CollectivePolicy
    """The collective policy every rank runs (``auto`` already resolved)."""

    @property
    def policy_name(self) -> str:
        """Name of the active collective policy, e.g. ``"reduce-bridge-broadcast"``."""
        return self.policy.name

    @property
    def bridge_group(self) -> ProcessGroup | None:
        """The bridge's process group, if its transport has one (the default Gloo one does)."""
        return getattr(self.bridge, "group", None)

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def vendor(self) -> str:
        """This rank's island label. Differs from :attr:`detected_vendor` only in a split test."""
        return self.layout.vendors[self.rank]

    @property
    def detected_vendor(self) -> str:
        """The vendor this rank detected (or was told by ``GPUBRIDGE_VENDOR``)."""
        return cast(str, self.peers[self.rank].vendor)

    @property
    def split_test(self) -> str | None:
        """The ``GPUBRIDGE_SPLIT_TEST`` mode of this run, or None. TEST ONLY."""
        return self.peers[self.rank].split_test or None

    @property
    def run_kind(self) -> str:
        """What kind of run this is, for results and logs.

        ``"split-test"`` (islands faked from one vendor), ``"cpu"`` (simulation or
        CPU-only mode), ``"gpu-mixed"`` (real GPUs of both vendors) or
        ``"gpu-single-vendor"``.
        """
        if self.split_test:
            return "split-test"
        if self.config.simulated:
            return "cpu"
        return "gpu-mixed" if self.layout.needs_bridge else "gpu-single-vendor"

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


def build_topology(
    local: PeerInfo,
    config: Config,
    *,
    transport: type[BridgeTransport],
    policy: str | type[CollectivePolicy] = AUTO,
    timeout: timedelta | None = None,
) -> Topology:
    """Exchange every rank's :class:`PeerInfo`, then create the island groups and the bridge.

    Collective: every rank must call this after the global Gloo group exists.

    Args:
        local: What this rank reports, from :func:`describe_local`.
        config: This rank's resolved configuration.
        transport: The bridge transport class; its name must match ``local.bridge``.
        policy: ``"auto"`` or the collective policy class; its name must match
            ``local.policy``.
        timeout: Timeout for the new groups' collectives. None keeps PyTorch's default.

    Raises:
        RuntimeError: from :func:`validate_peers`, or if the policy doesn't apply
            to the layout, on every rank alike.
    """
    rank = dist.get_rank()
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local.to_wire())
    peers = tuple(PeerInfo.from_wire(info) for info in gathered)
    validate_peers(peers, warn=rank == 0)
    if not config.simulated:
        # Only now that every rank's device index has passed discovery. Called
        # earlier, an index with no GPU behind it crashed one rank on its own.
        torch.cuda.set_device(config.device)
    # validate_peers has ruled out ranks without a vendor.
    vendors = [cast(str, peer.vendor) for peer in peers]
    labels = vendors
    if local.split_test:
        labels = split_labels(vendors, [peer.hostname for peer in peers], local.split_test)
        warnings.warn(banner(local.split_test, vendors, labels), SplitTestWarning, stacklevel=3)
    layout = plan_topology(labels)
    # Every rank sees the same layout, so every rank picks (or rejects) the same
    # policy, before any group is created.
    policy_cls = choose_policy(policy, layout)

    # new_group is collective over the whole world: every rank creates every
    # group, in the same order, including groups it is not a member of.
    island_group: ProcessGroup | None = None
    for island in layout.islands:
        group = dist.new_group(list(island.ranks), timeout=timeout, backend=config.island_backend)
        if rank in island.ranks:
            island_group = group
    bridge: BridgeTransport | None = None
    if layout.needs_bridge:
        bridge = transport.create(layout.bridge_ranks, timeout=timeout)
        if (bridge is not None) != (rank in layout.bridge_ranks):
            raise RuntimeError(
                f"{transport.__name__}.create() must return a transport exactly on the "
                f"bridge ranks {list(layout.bridge_ranks)}; rank {rank} got {bridge!r}"
            )
    assert island_group is not None

    topology = Topology(
        rank=rank,
        layout=layout,
        peers=peers,
        config=config,
        island_group=island_group,
        bridge=bridge,
        policy=policy_cls(),
    )
    topology.policy.setup(topology)
    return topology


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
