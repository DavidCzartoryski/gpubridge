"""Shared helpers for the GPU validation kit.

Every script in scripts/gpu writes JSON records with the same layout: the kit
version, the environment (hardware, driver, library versions, relevant env
vars), the kind of run, and pass/fail per stage with full error text. The
``run`` block always says whether the run was a split test, so split-test
results can't be mistaken for a real mixed-vendor run.

Records hold only allowlisted environment variables (``recorded_env``), and
everything written goes through ``redact``: no project path, home directory,
user name or URL credentials.
"""

from __future__ import annotations

import contextlib
import faulthandler
import getpass
import json
import os
import platform
import re
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

#: The environment variables a record may hold: an allowlist, by exact name or
#: prefix. Nothing else from the environment is recorded.
ENV_PREFIXES = ("GPUBRIDGE_", "NCCL_", "RCCL_", "GLOO_", "TORCH_")
ENV_NAMES = frozenset({
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS", "KIT_TIMEOUT", "STEP_TIMEOUT",
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_NODELIST",
    "SLURM_NNODES", "SLURM_JOB_NUM_NODES", "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID",
    "SLURM_NTASKS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_HET_SIZE",
})
#: Never recorded, even under an allowed prefix: names that may hold a
#: credential or say who ran the job.
SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "USER",
                "HOME", "ACCOUNT")


def recorded_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted environment variables, values redacted."""
    environ = os.environ if environ is None else environ
    return {name: redact(value) for name, value in sorted(environ.items())
            if (name in ENV_NAMES or name.startswith(ENV_PREFIXES))
            and not any(word in name.upper() for word in SECRET_WORDS)}


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # no user name for this uid, in some containers
        return None


def redact(text: str) -> str:
    """``text`` without project paths, the home directory, the user name as a path
    segment, or credentials in URLs; placeholders take their place."""
    end = r"(?=/|$|\s|['\":,])"  # a whole path component, not a prefix of one
    text = re.sub(r"(\w+://)[^/@\s]+@", r"\1***@", text)
    text = re.sub(r"/projects/[^/\s'\":,]+", "$PROJECT_DIR", text)
    project = os.environ.get("PROJECT_DIR", "").rstrip("/")
    if len(project) > 1:
        text = re.sub(re.escape(project) + end, "$PROJECT_DIR", text)
    home = os.path.expanduser("~").rstrip("/")
    if len(home) > 1:
        text = re.sub(re.escape(home) + end, "~", text)
    user = _user()
    if user:
        text = re.sub(f"/{re.escape(user)}{end}", "/$USER", text)
    return text


def scrub(data: Any) -> Any:
    """``data`` with every string in it redacted."""
    if isinstance(data, str):
        return redact(data)
    if isinstance(data, dict):
        return {key: scrub(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [scrub(value) for value in data]
    return data


def default_timeout(fallback: float) -> float:
    """Seconds for gpubridge.init's timeout: KIT_TIMEOUT, which the kit's shell
    scripts set (common.sh), or ``fallback`` when a tool runs on its own."""
    raw = os.environ.get("KIT_TIMEOUT", "").strip()
    return float(raw) if raw else fallback


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as exc:  # recording must never take a run down
        return default if default is not None else f"error: {type(exc).__name__}: {exc}"


def _command(*args: str) -> str | None:
    """A command's stdout, stripped, or None if it is missing or fails."""
    result = _safe(lambda: subprocess.run(args, capture_output=True, text=True, timeout=10))
    if isinstance(result, subprocess.CompletedProcess) and result.returncode == 0:
        return result.stdout.strip()
    return None


def _git_commit() -> str | None:
    return _command("git", "-C", str(REPO), "rev-parse", "--short", "HEAD")


def _git_dirty() -> bool | None:
    """Whether tracked files differ from the commit (new results files don't count)."""
    status = _command("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def driver_version() -> str | None:
    """The GPU kernel driver: NVIDIA's from nvidia-smi, AMD's amdgpu module version."""
    if torch.version.cuda:
        out = _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        return out.splitlines()[0].strip() if out else None
    if torch.version.hip:
        return _read("/sys/module/amdgpu/version") or _command(
            "rocm-smi", "--showdriverversion", "--csv")
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
    """Hardware, versions and allowlisted env vars of this process, redacted."""
    cuda_ok = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda_ok else 0
    return scrub({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "gpubridge": gpubridge.__version__,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "rocm_version": _read("/opt/rocm/.info/version") if torch.version.hip else None,
        # NCCL on CUDA builds, RCCL on ROCm builds; nccl_version is that library's.
        "collective_library": "rccl" if torch.version.hip else "nccl" if torch.version.cuda
        else None,
        "nccl_version": _safe(_nccl_version),
        "driver_version": _safe(driver_version),
        # Unset means the library picks.
        "interfaces": {name: os.environ.get(name) or "auto"
                       for name in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME")},
        "cuda_available": cuda_ok,
        "device_count": count,
        "arch_list": _safe(torch.cuda.get_arch_list, []) if cuda_ok else [],
        "devices": [_safe(lambda i=i: device_info(i)) for i in range(count)],
        "probe": asdict(gpubridge.probe()),
        "env": recorded_env(),
    })


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
        "policy": topology.policy_name,
    }
    if topology.split_test:
        info["warning"] = (
            "SPLIT TEST: islands were faked from a single-vendor job with "
            "GPUBRIDGE_SPLIT_TEST. This is NOT a mixed-vendor result."
        )
    return info


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically, so a crash never leaves half a file; every string redacted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(scrub(json.loads(json.dumps(data, default=str))), indent=2))
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
