"""What ``import gpubridge`` costs, one source tree against another, in fresh processes.

    git archive 38b1e20 src | tar -x -C /tmp/before     # any earlier tree
    python scripts/bench_import.py --before /tmp/before/src --runs 30
    python scripts/bench_import.py --before /tmp/before/src --runs 20 --no-bytecode

Each run is a new interpreter that imports torch, then gpubridge, under ``python -X
importtime``. The before arm imports gpubridge from ``--before`` (first on
PYTHONPATH), the after arm from ``--after``, by default this checkout's ``src``. The
arms alternate, so drift on the host spreads over both.

Bytecode decides much of the cost, so both arms get the same: by default both trees
are compiled first (``compileall``); with ``--no-bytecode`` each arm imports a copy of
its tree without ``__pycache__`` under ``PYTHONDONTWRITEBYTECODE=1``, so every run
compiles from source, as an install without bytecode does.

Prints JSON: per arm, the gpubridge modules loaded, the median (and quartiles) of the
package's cumulative ``-X importtime``, torch's own import time and the process's peak
RSS; and the after-minus-before medians.
"""

from __future__ import annotations

import argparse
import compileall
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE = (
    "import resource, sys, time\n"
    "t0 = time.perf_counter(); import torch, torch.distributed; t1 = time.perf_counter()\n"
    "import gpubridge\n"
    "print(t1 - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,\n"
    "      len([m for m in sys.modules if m.startswith('gpubridge')]), gpubridge.__file__)\n"
)
#: ru_maxrss is in bytes on macOS and in KiB on Linux.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def run_once(src: Path, bytecode: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(src)
    if not bytecode:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-X", "importtime", "-c", PROBE],
                            capture_output=True, text=True, env=env, check=True)
    cumulative = re.search(r"^import time:\s+\d+ \|\s+(\d+) \| gpubridge$", result.stderr, re.M)
    torch_s, rss, modules, path = result.stdout.split()
    if not Path(path).resolve().is_relative_to(src.resolve()):
        raise RuntimeError(f"gpubridge came from {path}, not {src}")
    return {"gpubridge_ms": int(cumulative[1]) / 1000, "torch_ms": float(torch_s) * 1000,
            "rss_mib": int(rss) * RSS_UNIT / 2**20, "modules": int(modules)}


def summary(rows: list[dict]) -> dict:
    times = [r["gpubridge_ms"] for r in rows]
    quartiles = statistics.quantiles(times, n=4) if len(times) > 1 else times * 3
    return {"runs": len(rows), "modules": sorted({r["modules"] for r in rows}),
            "gpubridge_ms_median": round(statistics.median(times), 3),
            "gpubridge_ms_quartiles": [round(quartiles[0], 3), round(quartiles[2], 3)],
            "torch_ms_median": round(statistics.median(r["torch_ms"] for r in rows), 1),
            "rss_mib_median": round(statistics.median(r["rss_mib"] for r in rows), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=Path, required=True, help="a src directory")
    parser.add_argument("--after", type=Path, default=REPO / "src", help="a src directory")
    parser.add_argument("--runs", type=int, default=30, help="runs per arm")
    parser.add_argument("--no-bytecode", action="store_true",
                        help="compile from source in every run")
    args = parser.parse_args()
    trees = {"before": args.before, "after": args.after}
    with tempfile.TemporaryDirectory(prefix="gpubridge-import-") as tmp:
        for arm, src in list(trees.items()):
            if args.no_bytecode:
                trees[arm] = Path(tmp) / arm / "src"
                shutil.copytree(src / "gpubridge", trees[arm] / "gpubridge",
                                ignore=shutil.ignore_patterns("__pycache__", "target", "*.pyc"))
            elif not compileall.compile_dir(src / "gpubridge", quiet=1):
                raise RuntimeError(f"compiling {src} failed")
        rows: dict[str, list[dict]] = {"before": [], "after": []}
        for i in range(args.runs):
            for arm in ("before", "after") if i % 2 == 0 else ("after", "before"):
                rows[arm].append(run_once(trees[arm], bytecode=not args.no_bytecode))
    out = {"bytecode": not args.no_bytecode, "python": sys.version.split()[0],
           **{arm: summary(r) for arm, r in rows.items()}}
    out["after_minus_before"] = {
        key: round(out["after"][key] - out["before"][key], 3)
        for key in ("gpubridge_ms_median", "torch_ms_median", "rss_mib_median")}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
