"""Vendor detection: which GPU vendor this process belongs to."""

from __future__ import annotations

import os
from typing import cast

import torch

from gpubridge.config import CPU_ONLY_ENV, VENDOR_ENV, VENDORS, Vendor


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
