"""TEST ONLY: split one single-vendor job into two islands.

With ``GPUBRIDGE_SPLIT_TEST`` set, some ranks of an all-NVIDIA (or all-AMD) job
are labeled as the other vendor. They keep their real GPU, their real
NCCL/RCCL and their true build in :class:`~gpubridge.detect.Probe`; only the
island layout changes. That runs the whole GPU code path (two native islands,
the bridge, reduce-bridge-broadcast on device tensors, ``new_group`` on ranks
outside the group) on a single-vendor cluster. Only RCCL talking to NCCL goes
untested.

The environment variable is the only switch. Every rank warns with
:class:`SplitTestWarning`, and :attr:`gpubridge.Topology.split_test` and
:attr:`gpubridge.Topology.run_kind` mark the run, so its results cannot pass
for a real mixed-vendor run.
"""

from __future__ import annotations

from collections.abc import Sequence

from gpubridge.config import SPLIT_TEST_ENV, SPLIT_TEST_MODES


class SplitTestWarning(UserWarning):
    """Issued on every rank of a split-test run."""


def split_labels(vendors: Sequence[str], hostnames: Sequence[str], mode: str) -> list[str]:
    """Return the island label of each rank for split-test ``mode``.

    - ``half``: the second half of the ranks (by global rank) becomes the other vendor.
    - ``alternate``: odd ranks become the other vendor.
    - ``node``: ranks on every second host (hosts ordered by their lowest rank)
      become the other vendor, so islands follow machines.

    Raises:
        RuntimeError: if the job has fewer than two ranks, already has ranks of
            both vendors, or ``node`` mode runs on a single host.
        ValueError: for an unknown mode.
    """
    if mode not in SPLIT_TEST_MODES:
        raise ValueError(f"unknown split-test mode {mode!r}")
    world = len(vendors)
    if world < 2:
        raise RuntimeError(f"{SPLIT_TEST_ENV} needs at least 2 ranks, got {world}")
    real = sorted(set(vendors))
    if len(real) != 1:
        raise RuntimeError(
            f"{SPLIT_TEST_ENV} only splits a single-vendor job, but this job already "
            f"has {' and '.join(real)} ranks. Unset it for a real mixed-vendor run."
        )
    vendor = real[0]
    other = "amd" if vendor == "nvidia" else "nvidia"
    if mode == "half":
        flipped = [rank >= (world + 1) // 2 for rank in range(world)]
    elif mode == "alternate":
        flipped = [rank % 2 == 1 for rank in range(world)]
    else:
        hosts = list(dict.fromkeys(hostnames))  # ordered by each host's lowest rank
        if len(hosts) < 2:
            raise RuntimeError(
                f"{SPLIT_TEST_ENV}=node splits islands by machine but every rank runs "
                f"on {hosts[0]}. Use half or alternate on a single machine."
            )
        flipped = [hosts.index(host) % 2 == 1 for host in hostnames]
    return [other if flip else vendor for flip in flipped]


def banner(mode: str, vendors: Sequence[str], labels: Sequence[str]) -> str:
    """The warning text shown on every rank of a split-test run."""
    relabeled = [rank for rank, (real, label) in enumerate(zip(vendors, labels, strict=True))
                 if real != label]
    real = vendors[0]
    return (
        f"{SPLIT_TEST_ENV}={mode}: TEST-ONLY split of a single-vendor job. Ranks "
        f"{relabeled} run {real} builds but are labeled {labels[relabeled[0]]!r} to "
        "exercise two islands and the bridge. This is NOT a mixed-vendor run; do not "
        "report its results as one."
    )
