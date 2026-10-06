"""all_reduce latency and bandwidth: native single-island vs gpubridge, launched by torchrun.

For each tensor size (1 KB to 1 GB by default) it times:

- ``gpubridge``: gpubridge.all_reduce. Bridged when the job has two islands
  (a split test or a real mixed job); a plain native all_reduce otherwise.
- ``native``: torch.distributed.all_reduce on one group of all ranks using the
  island backend (NCCL/RCCL on GPUs, Gloo on CPU). Only possible when every
  rank has the same detected vendor, which includes split tests, so a split
  test measures native and bridged in the same job.

Each size gets warmup runs, then timed trials until --trials or --max-seconds.
A trial's latency is the slowest rank's. Bandwidth follows nccl-tests:
algbw = bytes / time and busbw = algbw * 2(n-1)/n.

Rank 0 writes OUT/bench.csv, OUT/bench.json and OUT/summary.md. Runs on CPU in
simulation mode too (use a smaller --max-bytes):

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        scripts/gpu/bench_all_reduce.py --out /tmp/bench --max-bytes 4M
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from gpukit import KIT_VERSION, environment, run_info, sync, write_json

import gpubridge

UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
FIELDS = ["op", "run_kind", "split_test", "islands", "world_size", "bytes", "dtype", "trials",
          "median_us", "p90_us", "min_us", "algbw_GBps", "busbw_GBps"]


def parse_size(text: str) -> int:
    text = text.strip().upper().removesuffix("B")
    if text and text[-1] in UNITS:
        return int(float(text[:-1]) * UNITS[text[-1]])
    return int(text)


def human(nbytes: int) -> str:
    for unit in ("G", "M", "K"):
        if nbytes >= UNITS[unit] and nbytes % UNITS[unit] == 0:
            return f"{nbytes // UNITS[unit]}{unit}B"
    return f"{nbytes}B"


def sizes(lo: int, hi: int, factor: int) -> list[int]:
    out, size = [], lo
    while size <= hi:
        out.append(size)
        size *= factor
    return out


def time_op(op, tensor, device, *, warmup: int, trials: int, max_seconds: float) -> list[float]:
    """Slowest-rank latency in seconds for each timed trial."""
    for _ in range(warmup):
        op(tensor)
    sync(device)
    latencies: list[float] = []
    spent = 0.0
    while len(latencies) < trials:
        dist.barrier()
        start = time.perf_counter()
        op(tensor)
        sync(device)
        elapsed = torch.tensor([time.perf_counter() - start], dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)  # world group is Gloo, on CPU
        latencies.append(float(elapsed))
        spent += float(elapsed)
        # Every rank sees the same maxima, so every rank stops at the same trial.
        if spent > max_seconds and len(latencies) >= 3:
            break
    return latencies


def row(op: str, info: dict[str, Any], nbytes: int, dtype: str, lat: list[float]) -> dict:
    world = info["world_size"]
    median = statistics.median(lat)
    p90 = sorted(lat)[max(0, int(round(0.9 * len(lat))) - 1)]
    algbw = nbytes / median / 1e9
    return {
        "op": op, "run_kind": info["kind"], "split_test": info["split_test"] or "",
        "islands": len(info["islands"]), "world_size": world, "bytes": nbytes, "dtype": dtype,
        "trials": len(lat), "median_us": round(median * 1e6, 2), "p90_us": round(p90 * 1e6, 2),
        "min_us": round(min(lat) * 1e6, 2), "algbw_GBps": round(algbw, 4),
        "busbw_GBps": round(algbw * 2 * (world - 1) / world, 4) if world > 1 else None,
    }


def summary_table(rows: list[dict], info: dict[str, Any]) -> str:
    lines = []
    if info["split_test"]:
        lines += ["> **SPLIT TEST - not a mixed-vendor result.** Islands were faked from a "
                  "single-vendor job with `GPUBRIDGE_SPLIT_TEST`.", ""]
    lines += [f"Run kind `{info['kind']}`, {info['world_size']} ranks, "
              f"{len(info['islands'])} island(s). Latency is the median of the slowest rank.",
              "", "| size | native median (us) | native busbw (GB/s) | gpubridge median (us) "
              "| gpubridge busbw (GB/s) | gpubridge / native |",
              "| --- | --- | --- | --- | --- | --- |"]
    by_size: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_size.setdefault(r["bytes"], {})[r["op"]] = r
    for nbytes, ops in sorted(by_size.items()):
        native, bridge = ops.get("native"), ops.get("gpubridge")
        ratio = (f"{bridge['median_us'] / native['median_us']:.2f}x"
                 if native and bridge and native["median_us"] else "-")

        def cell(r, key):
            return "-" if r is None or r[key] is None else f"{r[key]}"
        lines.append(f"| {human(nbytes)} | {cell(native, 'median_us')} | "
                     f"{cell(native, 'busbw_GBps')} | {cell(bridge, 'median_us')} | "
                     f"{cell(bridge, 'busbw_GBps')} | {ratio} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-bytes", type=parse_size, default=parse_size("1K"))
    parser.add_argument("--max-bytes", type=parse_size, default=parse_size("1G"))
    parser.add_argument("--factor", type=int, default=4, help="size multiplier per step")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=20.0,
                        help="stop timing a size after this much measured time (min 3 trials)")
    parser.add_argument("--ops", default="native,gpubridge", help="which to measure")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
    info = run_info(topology)
    device, dtype = topology.device, DTYPES[args.dtype]
    if topology.split_test and topology.rank == 0:
        print(info["warning"], file=sys.stderr, flush=True)

    ops: dict[str, Any] = {}
    wanted = args.ops.split(",")
    if "native" in wanted:
        detected = {peer.vendor for peer in topology.peers}
        if len(detected) == 1:
            # Collective: every rank creates it. Same backend as the islands.
            group = dist.new_group(list(range(topology.world_size)),
                                   backend=topology.config.island_backend,
                                   timeout=timedelta(seconds=args.timeout))
            ops["native"] = lambda t: dist.all_reduce(t, group=group)
        elif topology.rank == 0:
            print("native: skipped, ranks have different real vendors", file=sys.stderr)
    if "gpubridge" in wanted:
        ops["gpubridge"] = gpubridge.all_reduce

    element = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for nbytes in sizes(args.min_bytes, args.max_bytes, args.factor):
        tensor = torch.ones(max(1, nbytes // element), dtype=dtype, device=device)
        for name, op in ops.items():
            lat = time_op(op, tensor, device, warmup=args.warmup, trials=args.trials,
                          max_seconds=args.max_seconds)
            rows.append(row(name, info, tensor.numel() * element, args.dtype, lat))
            if topology.rank == 0:
                r = rows[-1]
                print(f"{name:>9} {human(r['bytes']):>6}: median {r['median_us']:>12} us  "
                      f"busbw {r['busbw_GBps']} GB/s  ({r['trials']} trials)", flush=True)
        del tensor

    if topology.rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "bench.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_json(args.out / "bench.json", {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/bench_all_reduce.py",
            "argv": sys.argv, "run": info, "environment": environment(), "rows": rows,
        })
        table = summary_table(rows, info)
        (args.out / "summary.md").write_text(f"# {args.out.name}: all_reduce benchmark\n\n{table}")
        print(table)
    gpubridge.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
