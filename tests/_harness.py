"""Run simulated ranks as CPU processes for multi-process tests."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

import gpubridge
from gpubridge.config import (
    CHUNK_BYTES_ENV,
    CPU_ONLY_ENV,
    SPLIT_TEST_ENV,
    THRESHOLDS_ENV,
    VENDOR_ENV,
)

# Fail a deadlocked test instead of hanging the suite.
TIMEOUT = timedelta(seconds=60)


def run_ranks(
    vendors: Sequence[str],
    fn: Callable[[gpubridge.Topology], None],
    tmp_path: Path,
    *,
    fake_builds: bool = False,
    init_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run ``fn(topology)`` on one CPU process per entry in ``vendors``.

    By default each process fakes its vendor with ``GPUBRIDGE_VENDOR``. With
    ``fake_builds`` it instead patches ``torch.version`` to look like a CUDA or
    ROCm build and sets ``GPUBRIDGE_CPU_ONLY``, so the vendor comes from build
    detection. ``init_kwargs`` are passed to ``gpubridge.init`` on every rank.

    ``fn`` must be picklable: a module-level function or a ``functools.partial``
    of one. A failure on any rank is re-raised in the calling process.
    """
    init_file = tmp_path / "rendezvous"
    mp.spawn(
        _rank_main,
        args=(tuple(vendors), fn, str(init_file), fake_builds, init_kwargs or {}),
        nprocs=len(vendors),
    )


def _rank_main(
    rank: int,
    vendors: tuple[str, ...],
    fn: Callable[[gpubridge.Topology], None],
    init_file: str,
    fake_builds: bool,
    init_kwargs: dict[str, Any],
) -> None:
    if fake_builds:
        os.environ[CPU_ONLY_ENV] = "1"
        torch.version.cuda = "13.0" if vendors[rank] == "nvidia" else None
        torch.version.hip = "7.2.26015" if vendors[rank] == "amd" else None
    else:
        os.environ[VENDOR_ENV] = vendors[rank]
    torch.set_num_threads(1)
    topology = gpubridge.init(
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(vendors),
        timeout=TIMEOUT,
        **init_kwargs,
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


def run_failing_init(setups: Sequence[dict[str, Any]], tmp_path: Path, patterns: Sequence[str]):
    """Check that ``gpubridge.init`` raises RuntimeError on every rank, without hanging.

    Each entry of ``setups`` configures one rank before it calls init:

    - ``env``: environment variables to set (gpubridge's own are cleared first),
    - ``cuda`` / ``hip``: values for ``torch.version.cuda`` / ``.hip``, to fake a build,
    - ``gpu``: what ``torch.cuda.is_available()`` should return,
    - ``gpus``: what ``torch.cuda.device_count()`` should return,
    - ``init_kwargs``: extra arguments for ``gpubridge.init``.

    Every rank's error message must match every regex in ``patterns``, and
    nothing may stay initialized afterwards.
    """
    init_file = tmp_path / "rendezvous"
    mp.spawn(
        _failing_rank_main,
        args=(tuple(setups), str(init_file), tuple(patterns)),
        nprocs=len(setups),
    )


def _failing_rank_main(
    rank: int, setups: tuple[dict[str, Any], ...], init_file: str, patterns: tuple[str, ...]
) -> None:
    setup = setups[rank]
    for name in (VENDOR_ENV, CPU_ONLY_ENV, SPLIT_TEST_ENV, CHUNK_BYTES_ENV, THRESHOLDS_ENV):
        os.environ.pop(name, None)
    os.environ.update(setup.get("env", {}))
    for attr in ("cuda", "hip"):
        if attr in setup:
            setattr(torch.version, attr, setup[attr])
    if "gpu" in setup:
        torch.cuda.is_available = lambda: setup["gpu"]
    if "gpus" in setup:
        torch.cuda.device_count = lambda: setup["gpus"]
    torch.set_num_threads(1)
    try:
        gpubridge.init(
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=len(setups),
            timeout=TIMEOUT,
            **setup.get("init_kwargs", {}),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        gpubridge.destroy()
        raise AssertionError(f"rank {rank}: gpubridge.init() unexpectedly succeeded")
    for pattern in patterns:
        assert re.search(pattern, message), f"rank {rank}: {pattern!r} not found in:\n{message}"
    assert not gpubridge.is_initialized()
    assert not torch.distributed.is_initialized()
