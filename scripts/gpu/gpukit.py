"""Shared helpers for the GPU validation kit.

Every script in scripts/gpu writes JSON records with the same layout: the kit
version, the environment (hardware, driver, library versions, relevant env
vars), the kind of run, and pass/fail per stage with full error text. The
``run`` block always says whether the run was a split test, so split-test
results can't be mistaken for a real mixed-vendor run.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import gpubridge

KIT_VERSION = 1
REPO = Path(__file__).resolve().parents[2]

#: Environment variables worth recording; none of these carry secrets.
ENV_PREFIXES = (
    "GPUBRIDGE_", "NCCL_", "RCCL_", "GLOO_", "MASTER_", "TORCHELASTIC_", "SLURM_JOB",
    "SLURM_NODELIST", "SLURM_NNODES", "SLURM_GPUS", "CUDA_VISIBLE", "HIP_VISIBLE",
    "ROCR_VISIBLE", "LOCAL_RANK", "RANK", "WORLD_SIZE", "GROUP_RANK", "OMP_NUM_THREADS",
)


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as exc:  # recording must never take a run down
        return default if default is not None else f"error: {type(exc).__name__}: {exc}"


def _git_commit() -> str | None:
    result = _safe(lambda: subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=5,
    ))
    if isinstance(result, subprocess.CompletedProcess) and result.returncode == 0:
        return result.stdout.strip()
    return None


def _nccl_version() -> str | None:
    """NCCL's version on CUDA builds, RCCL's on ROCm builds."""
    if not dist.is_nccl_available():
        return None
    version = torch.cuda.nccl.version()
    return ".".join(map(str, version)) if isinstance(version, tuple) else str(version)


def device_info(index: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", None)  # ROCm, e.g. "gfx942:sramecc+:xnack-"
    if arch:
        target = arch.split(":")[0]
    else:
        target = f"sm_{props.major}{props.minor}"
    arch_list = _safe(torch.cuda.get_arch_list, [])
    return {
        "index": index,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "target": target,
        # False means this PyTorch build has no kernels for this GPU: pick another wheel.
        "target_in_build": target in arch_list or target.replace("sm_", "compute_") in arch_list,
        "total_memory_gb": round(props.total_memory / 2**30, 1),
        "pci_bus_id": getattr(props, "pci_bus_id", None),
        "pci_device_id": getattr(props, "pci_device_id", None),
        "uuid": str(getattr(props, "uuid", "")) or None,
    }


def environment() -> dict[str, Any]:
    """Hardware, versions and relevant env vars of this process."""
    cuda_ok = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda_ok else 0
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "gpubridge": gpubridge.__version__,
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "nccl_version": _safe(_nccl_version),
        "cuda_available": cuda_ok,
        "device_count": count,
        "arch_list": _safe(torch.cuda.get_arch_list, []) if cuda_ok else [],
        "devices": [_safe(lambda i=i: device_info(i)) for i in range(count)],
        "probe": asdict(gpubridge.probe()),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
    }


def run_info(topology: gpubridge.Topology) -> dict[str, Any]:
    """How this run was set up. ``split_test`` and ``real_mixed_vendor`` are never ambiguous."""
    info: dict[str, Any] = {
        "kind": topology.run_kind,
        "split_test": topology.split_test,
        "real_mixed_vendor": topology.run_kind == "gpu-mixed",
        "world_size": topology.world_size,
        "rank": topology.rank,
        "vendor_label": topology.vendor,
        "detected_vendor": topology.detected_vendor,
        "device": str(topology.device),
        "islands": [
            {"vendor": island.vendor, "ranks": list(island.ranks), "leader": island.leader}
            for island in topology.layout.islands
        ],
        "bridge": topology.peers[topology.rank].bridge,
    }
    if topology.split_test:
        info["warning"] = (
            "SPLIT TEST: islands were faked from a single-vendor job with "
            "GPUBRIDGE_SPLIT_TEST. This is NOT a mixed-vendor result."
        )
    return info


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


class Record:
    """One JSON record, rewritten after every stage so a crash or hang shows where it stopped."""

    def __init__(self, path: Path, script: str, *, hang_dump_seconds: float = 600) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stacks = open(path.with_suffix(".stacks.txt"), "w")  # noqa: SIM115
        self.hang_dump_seconds = hang_dump_seconds
        self.data: dict[str, Any] = {
            "kit_version": KIT_VERSION,
            "script": script,
            "argv": sys.argv,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": environment(),
            "run": None,
            "stages": [],
            "ok": None,
        }
        self.failed = False
        self.write()

    def write(self) -> None:
        write_json(self.path, self.data)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a stage. Set ``entry["ok"] = False`` for a failed check; exceptions propagate."""
        entry: dict[str, Any] = {"name": name, "ok": None}
        self.data["stages"].append(entry)
        self.write()  # ok=None marks the stage a rank is stuck in
        faulthandler.dump_traceback_later(self.hang_dump_seconds, file=self.stacks)
        start = time.monotonic()
        try:
            yield entry
        except BaseException:
            entry["ok"] = False
            entry["error"] = traceback.format_exc()
            raise
        else:
            if entry["ok"] is None:
                entry["ok"] = True
        finally:
            faulthandler.cancel_dump_traceback_later()
            entry["seconds"] = round(time.monotonic() - start, 4)
            self.failed = self.failed or not entry["ok"]
            self.write()

    def finish(self) -> int:
        """Set the overall result and return a process exit code."""
        self.data["ok"] = not self.failed
        self.data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.write()
        return 0 if self.data["ok"] else 1


def rank_from_env() -> int:
    return int(os.environ.get("RANK", "0"))


def sync(device: torch.device) -> None:
    """Wait for queued GPU work, so timings and checks see finished results."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
