"""Run simulated ranks as CPU processes for multi-process tests."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

import gpubridge
from gpubridge.config import VENDOR_ENV

# Fail a deadlocked test instead of hanging the suite.
TIMEOUT = timedelta(seconds=60)


def run_ranks(
    vendors: Sequence[str], fn: Callable[[gpubridge.Topology], None], tmp_path: Path
) -> None:
    """Run ``fn(topology)`` on one simulated process per entry in ``vendors``.

    ``fn`` must be picklable: a module-level function or a ``functools.partial``
    of one. A failure on any rank is re-raised in the calling process.
    """
    init_file = tmp_path / "rendezvous"
    mp.spawn(_rank_main, args=(tuple(vendors), fn, str(init_file)), nprocs=len(vendors))


def _rank_main(
    rank: int,
    vendors: tuple[str, ...],
    fn: Callable[[gpubridge.Topology], None],
    init_file: str,
) -> None:
    os.environ[VENDOR_ENV] = vendors[rank]
    torch.set_num_threads(1)
    topology = gpubridge.init(
        init_method=f"file://{init_file}", rank=rank, world_size=len(vendors), timeout=TIMEOUT
    )
    try:
        fn(topology)
    finally:
        gpubridge.destroy()


def rank_tensor(shape: tuple[int, ...], dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Return a tensor that differs per rank and per element.

    Values are small integers, so sums are exact even in float16 and do not
    depend on the order the bridge adds them in.
    """
    base = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return (base + rank + 1).to(dtype)


def expected_sum(shape: tuple[int, ...], dtype: torch.dtype, world_size: int) -> torch.Tensor:
    """Return the sum of :func:`rank_tensor` over ranks ``0..world_size-1``."""
    total = sum(rank_tensor(shape, torch.float64, rank) for rank in range(world_size))
    return torch.as_tensor(total).to(dtype)
