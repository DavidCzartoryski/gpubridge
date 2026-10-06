"""Per-rank probe for the mixed-build experiments, launched by torchrun.

Same flow as examples/all_reduce_torchrun.py, plus checks: it runs gpubridge's
public API in CPU-only mode (GPUBRIDGE_CPU_ONLY=1, vendor from the PyTorch
build) and writes a JSON record for this rank after every stage, so a crash or
hang still shows the last stage reached. Run by scripts/mixed_build_rendezvous.py.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import platform
import socket
import sys
import time
import traceback
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import gpubridge


class Recorder:
    """Accumulates this rank's results and rewrites its JSON file after each stage."""

    def __init__(self, out_dir: Path, rank: int, hang_dump_seconds: float) -> None:
        self.path = out_dir / f"rank{rank}.json"
        self.stacks = open(out_dir / f"rank{rank}.stacks.txt", "w")  # noqa: SIM115
        self.hang_dump_seconds = hang_dump_seconds
        self.data: dict[str, Any] = {"rank": rank, "stages": []}
        self.failed = False

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a stage. Checks inside set ``entry["ok"] = False``; exceptions are re-raised."""
        entry: dict[str, Any] = {"name": name, "ok": None}
        self.data["stages"].append(entry)
        self.write()  # an entry with ok=None marks the stage this rank is stuck in
        # If the stage hangs, dump every thread's stack so we can see where.
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
            entry["seconds"] = round(time.monotonic() - start, 3)
            self.failed = self.failed or not entry["ok"]
            self.write()


def build_vendor() -> str:
    """The vendor of this PyTorch build, read directly rather than through gpubridge."""
    if torch.version.hip:
        return "amd"
    if torch.version.cuda:
        return "nvidia"
    return "cpu"


def rank_tensor(shape: tuple[int, ...], dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Integer-valued, so sums are exact in float16 and independent of summation order."""
    base = torch.arange(int(torch.Size(shape).numel()), dtype=torch.float64).reshape(shape)
    return (base + rank + 1).to(dtype)


CASES = [((3, 4), torch.float32), ((5,), torch.float16)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600, help="gpubridge group timeout (s)")
    parser.add_argument("--hang-dump", type=float, default=300, help="dump stacks after (s)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ["RANK"])
    rec = Recorder(args.out, rank, args.hang_dump)
    rec.data["process"] = {
        "env": os.environ.get("PROBE_ENV"),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "node_rank": int(os.environ.get("GROUP_RANK", -1)),
        "world_size_env": int(os.environ["WORLD_SIZE"]),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "torch_version_hip": torch.version.hip,
        "build_vendor": build_vendor(),
        "cuda_available": torch.cuda.is_available(),
        "gloo_available": dist.is_gloo_available(),
        "master": f"{os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
        "agent_store": os.environ.get("TORCHELASTIC_USE_AGENT_STORE"),
    }
    rec.write()

    try:
        with rec.stage("init"):
            topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
        rec.data["topology"] = {
            "vendor": topology.vendor,
            "is_leader": topology.is_leader,
            "simulated": topology.simulated,
            "islands": [
                {"vendor": i.vendor, "ranks": list(i.ranks), "leader": i.leader}
                for i in topology.layout.islands
            ],
            "peers": [
                {"vendor": p.vendor, "torch": p.torch_version, "simulated": p.simulated}
                for p in topology.peers
            ],
        }
        world = topology.world_size

        with rec.stage("all_reduce") as entry:
            entry["cases"] = []
            for shape, dtype in CASES:
                tensor = rank_tensor(shape, dtype, rank)
                gpubridge.all_reduce(tensor)
                expected = sum(rank_tensor(shape, torch.float64, r) for r in range(world))
                ok = torch.equal(tensor, torch.as_tensor(expected).to(dtype))
                entry["cases"].append({"shape": list(shape), "dtype": str(dtype), "ok": ok})
                if not ok:
                    entry["ok"] = False
                    entry["cases"][-1]["got"] = tensor.tolist()

        with rec.stage("broadcast") as entry:
            entry["bad"] = []
            for src in range(world):
                for shape, dtype in CASES:
                    tensor = rank_tensor(shape, dtype, rank)
                    gpubridge.broadcast(tensor, src)
                    if not torch.equal(tensor, rank_tensor(shape, dtype, src)):
                        entry["ok"] = False
                        entry["bad"].append({"src": src, "dtype": str(dtype)})
            entry["sources"] = world

        with rec.stage("barrier"):
            gpubridge.barrier()

        with rec.stage("destroy"):
            gpubridge.destroy()
    except BaseException:
        traceback.print_exc()
        return 1
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
