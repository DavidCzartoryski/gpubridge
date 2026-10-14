"""How long the compression kernels take to build and to load, by Triton cache location.

Needs one GPU, and a few minutes. For each directory (by default one next to Triton's
own default, ~/.triton; one in node-local storage, $TMPDIR or /tmp; one under
$WORK_DIR if set), it starts a fresh process with TRITON_CACHE_DIR in a new, empty
subdirectory there and launches each of a fixed set of kernel variants once: each
first launch builds the variant (Triton's compile, the cache write, the module load).
A second fresh process then launches them again against the now-filled cache: each
first launch there is a load from the cache. A third launch in that process is a
plain launch, for scale. Prints the median of each per location, with its
filesystem, writes OUT (JSON) and removes the subdirectories it made::

    python scripts/gpu/time_triton_cache.py --out results/explorer/triton_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
#: (mode, dtype, numel, error feedback): distinct specializations, as the GPU tests make.
VARIANTS = [(mode, dtype, n, ef) for mode in ("fp8", "bf16")
            for dtype in ("float32", "float64", "float16", "bfloat16")
            for n, ef in ((255, True), (1 << 16, False))
            if not (mode == "bf16" and dtype in ("float16", "bfloat16"))]


def filesystem(path: Path, mounts: str | None = None) -> str:
    """The type of the filesystem ``path`` is on, from /proc/mounts (its longest prefix)."""
    if mounts is None:
        try:
            mounts = Path("/proc/mounts").read_text()
        except OSError:
            return "?"
    resolved, best, kind = str(path.resolve()), "", "?"
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            point = fields[1]
            inside = resolved == point or resolved.startswith(point.rstrip("/") + "/")
            if inside and len(point) > len(best):
                best, kind = point, fields[2]
    return kind


def child() -> None:
    """In a fresh process with TRITON_CACHE_DIR set: each variant's first launch, then
    another launch, in seconds, as JSON."""
    import torch

    from gpubridge import _compression_kernels as K
    from gpubridge import compression as C

    device = torch.device("cuda")
    torch.ones(1, device=device).sum().item()  # the CUDA context, outside the timings
    first, again = [], []
    for mode, dtype, n, ef in VARIANTS:
        x = torch.randn(n, device=device).to(getattr(torch, dtype))
        residual = torch.zeros(n, device=device) if ef else None
        payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8, device=device)
        for times in (first, again):
            torch.cuda.synchronize()
            start = time.perf_counter()
            K.quantize(mode, x, residual, payload)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    print(json.dumps({"first": first, "again": again}))


def measure(where: Path) -> dict:
    cache = Path(tempfile.mkdtemp(prefix="gpubridge-triton-timing-", dir=where))
    env = {**os.environ, "TRITON_CACHE_DIR": str(cache)}
    runs = []
    try:
        for _ in ("cold", "warm"):
            result = subprocess.run([sys.executable, str(HERE), "--child"], env=env,
                                    capture_output=True, text=True, timeout=1800)
            if result.returncode:
                raise RuntimeError(f"timing process failed in {where}: "
                                   f"{result.stderr.strip()[-2000:]}")
            runs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    cold, warm = runs
    return {
        "dir": str(where), "filesystem": filesystem(where), "variants": len(VARIANTS),
        "build_s": [round(t, 4) for t in cold["first"]],
        "load_s": [round(t, 4) for t in warm["first"]],
        "launch_s": [round(t, 5) for t in warm["again"]],
        "median_build_s": round(statistics.median(cold["first"]), 3),
        "median_load_s": round(statistics.median(warm["first"]), 3),
        "median_launch_s": round(statistics.median(warm["again"]), 5),
    }


def default_dirs() -> list[Path]:
    dirs = [Path.home() / ".triton", Path(os.environ.get("TMPDIR") or "/tmp")]
    if os.environ.get("WORK_DIR"):
        dirs.append(Path(os.environ["WORK_DIR"]))
    return dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dirs", nargs="+", type=Path, help="where to put the caches")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        child()
        return 0
    import torch

    if not torch.cuda.is_available():
        print("time_triton_cache.py needs a GPU", file=sys.stderr)
        return 2
    rows = []
    for where in args.dirs or default_dirs():
        where.mkdir(parents=True, exist_ok=True)
        row = measure(where)
        rows.append(row)
        print(f"{row['dir']} ({row['filesystem']}): per variant, build {row['median_build_s']} s,"
              f" load from the cache {row['median_load_s']} s, launch {row['median_launch_s']} s"
              f" ({row['variants']} variants)", flush=True)
    record = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "triton": __import__("triton").__version__, "locations": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
