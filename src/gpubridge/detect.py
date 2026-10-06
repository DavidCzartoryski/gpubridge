"""Vendor detection: which GPU vendor this process belongs to."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.distributed as dist

from gpubridge.config import CPU_ONLY_ENV, VENDOR_ENV, VENDORS, Vendor

T = TypeVar("T")


def vendor_override() -> Vendor | None:
    """Return the vendor forced by ``GPUBRIDGE_VENDOR``, or None if it is unset or empty.

    Raises:
        ValueError: if the variable holds anything other than ``nvidia`` or ``amd``.
    """
    value = os.environ.get(VENDOR_ENV, "").strip().lower()
    if not value:
        return None
    if value not in VENDORS:
        raise ValueError(
            f"{VENDOR_ENV}={os.environ[VENDOR_ENV]!r} is not a known vendor; "
            f"expected one of: {', '.join(VENDORS)}"
        )
    return cast(Vendor, value)


def detect_hardware_vendor() -> Vendor:
    """Return the vendor of the installed PyTorch build.

    A PyTorch install is either a ROCm build (``torch.version.hip`` is set) or a
    CUDA build (``torch.version.cuda`` is set), never both.

    Raises:
        RuntimeError: if this is a CPU-only build of PyTorch.
    """
    if getattr(torch.version, "hip", None):
        return "amd"
    if getattr(torch.version, "cuda", None):
        return "nvidia"
    raise RuntimeError(
        "This PyTorch build has neither CUDA nor ROCm support, so the vendor cannot be "
        f"detected ({CPU_ONLY_ENV} still needs a CUDA or ROCm build). To develop without "
        f"GPUs, set {VENDOR_ENV}=nvidia or {VENDOR_ENV}=amd to run in CPU simulation mode."
    )


def detect_vendor() -> Vendor:
    """Return this process's vendor: the ``GPUBRIDGE_VENDOR`` override if set, else the build's."""
    override = vendor_override()
    return override if override is not None else detect_hardware_vendor()


@dataclass(frozen=True)
class Probe:
    """What this process can do, gathered by :func:`probe` without raising."""

    build_vendor: str | None
    """``"nvidia"`` or ``"amd"`` from the PyTorch build; None on a CPU-only build."""
    torch_version: str
    cuda_version: str | None
    hip_version: str | None
    gpu_count: int
    """GPUs visible to this process; 0 if there are none or the query failed."""
    nccl_available: bool
    """Whether the build has the NCCL backend, which runs RCCL on ROCm builds."""
    gloo_available: bool
    errors: tuple[str, ...] = ()
    """Checks that failed while probing, as ``"<check>: <error>"``."""


def probe() -> Probe:
    """Report this process's PyTorch build, GPUs and backends. Never raises.

    Every check runs on its own. If one fails (a broken driver can make even
    ``torch.cuda.device_count()`` raise), its field gets a safe default and the
    failure is recorded in :attr:`Probe.errors` instead of propagating.
    """
    errors: list[str] = []

    def check(name: str, fn: Callable[[], T], default: T) -> T:
        try:
            return fn()
        except Exception as exc:  # a probe must never take the process down
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return default

    hip = check("torch.version.hip", lambda: getattr(torch.version, "hip", None), None)
    cuda = check("torch.version.cuda", lambda: getattr(torch.version, "cuda", None), None)
    return Probe(
        build_vendor="amd" if hip else "nvidia" if cuda else None,
        torch_version=check("torch.__version__", lambda: str(torch.__version__), "unknown"),
        cuda_version=str(cuda) if cuda else None,
        hip_version=str(hip) if hip else None,
        gpu_count=check("torch.cuda.device_count", lambda: int(torch.cuda.device_count()), 0),
        nccl_available=check("dist.is_nccl_available", lambda: bool(dist.is_nccl_available()),
                             False),
        gloo_available=check("dist.is_gloo_available", lambda: bool(dist.is_gloo_available()),
                             False),
        errors=tuple(errors),
    )
