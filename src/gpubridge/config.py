"""Runtime configuration: simulation mode, backend and device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch

Vendor = Literal["nvidia", "amd"]
VENDORS: tuple[Vendor, ...] = ("nvidia", "amd")

#: Set to ``nvidia`` or ``amd`` to fake that vendor on CPU (simulation mode).
VENDOR_ENV = "GPUBRIDGE_VENDOR"

#: Backend of the global discovery group and the cross-vendor bridge. Both run on CPU.
CPU_BACKEND = "gloo"
#: Backend of islands on real GPUs. On ROCm builds of PyTorch, "nccl" runs RCCL.
GPU_BACKEND = "nccl"


def is_simulated() -> bool:
    """Return True when ``GPUBRIDGE_VENDOR`` is set, which turns on CPU simulation mode."""
    return bool(os.environ.get(VENDOR_ENV, "").strip())


@dataclass(frozen=True)
class Config:
    """How this process communicates, resolved once by :func:`gpubridge.init`."""

    simulated: bool
    island_backend: str
    device: torch.device


def resolve_config() -> Config:
    """Pick the island backend and the device this rank communicates on.

    In simulation mode islands use Gloo and tensors stay on CPU. Otherwise
    islands use the native backend on ``cuda:$LOCAL_RANK``, or on the current
    CUDA device when ``LOCAL_RANK`` is unset. ROCm builds expose AMD GPUs
    through the same ``torch.cuda`` API.

    Raises:
        RuntimeError: if not simulating and no GPU is visible to this process.
    """
    if is_simulated():
        return Config(simulated=True, island_backend=CPU_BACKEND, device=torch.device("cpu"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is visible to this process. To develop without GPUs, set "
            f"{VENDOR_ENV}=nvidia or {VENDOR_ENV}=amd to run in CPU simulation mode."
        )
    local_rank = os.environ.get("LOCAL_RANK")
    index = int(local_rank) if local_rank is not None else torch.cuda.current_device()
    return Config(simulated=False, island_backend=GPU_BACKEND, device=torch.device("cuda", index))
