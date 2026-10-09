"""all_reduce latency and bandwidth: native single-island vs gpubridge, launched by torchrun.

For each tensor size (1 KB to 1 GB by default) it times:

- ``gpubridge``: gpubridge.all_reduce. Bridged when the job has two islands
  (a split test or a real mixed job); a plain native all_reduce otherwise.
- ``native``: torch.distributed.all_reduce on one group of all ranks using the
  island backend (NCCL/RCCL on GPUs, Gloo on CPU). Only possible when every
  rank has the same detected vendor, which includes split tests, so a split
  test measures native and bridged in the same job.
- ``island`` (opt in with --ops): every island all_reduces on its own native
  group at the same time (NCCL on NVIDIA, RCCL on AMD): one row per island,
  ``island:<vendor>``. These are the per-island baselines for a real mixed
  job, where ``native`` is impossible.

Each size gets warmup runs, then timed trials until --trials or --max-seconds.
A trial's latency is the slowest rank's (for ``island``, the slowest rank of
that island). Bandwidth follows nccl-tests: algbw = bytes / time and
busbw = algbw * 2(n-1)/n, with n the ranks taking part.

--phases attaches a gpubridge observer and adds the median time of each phase
of the gpubridge op (island-reduce, bridge, island-broadcast for
reduce-bridge-broadcast), slowest rank, to bench.json and summary.md. With
--ops island,gpubridge that puts the island step of reduce-bridge-broadcast (a
reduce onto the leader) next to a native island all_reduce of the same size.

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
FIELDS = ["op", "policy", "run_kind", "split_test", "islands", "world_size", "bytes", "dtype",
          "trials",
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


def time_op(op, tensor, device, *, warmup: int, trials: int,
            max_seconds: float) -> list[list[float]]:
    """Every rank's latency in seconds, for each timed trial."""
    for _ in range(warmup):
        op(tensor)
    sync(device)
    world = dist.get_world_size()
    per_trial: list[list[float]] = []
    spent = 0.0
    while len(per_trial) < trials:
        dist.barrier()
        start = time.perf_counter()
        op(tensor)
        sync(device)
        elapsed = torch.tensor([time.perf_counter() - start], dtype=torch.float64)
        gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(world)]
        dist.all_gather(gathered, elapsed)  # world group is Gloo, on CPU
        per_trial.append([float(t) for t in gathered])
        spent += max(per_trial[-1])
        # Every rank sees the same values, so every rank stops at the same trial.
        if spent > max_seconds and len(per_trial) >= 3:
            break
    return per_trial


def slowest(per_trial: list[list[float]], ranks=None) -> list[float]:
    """Each trial's latency: the slowest of ``ranks`` (default: every rank)."""
    return [max(t if ranks is None else [t[r] for r in ranks]) for t in per_trial]


def row(op: str, info: dict[str, Any], nbytes: int, dtype: str, lat: list[float],
        world: int | None = None) -> dict:
    """One result row. ``world`` is the ranks taking part (default: all)."""
    world = world or info["world_size"]
    median = statistics.median(lat)
    p90 = sorted(lat)[max(0, int(round(0.9 * len(lat))) - 1)]
    algbw = nbytes / median / 1e9
    return {
        "op": op, "policy": info["policy"] if op == "gpubridge" else "",
        "run_kind": info["kind"], "split_test": info["split_test"] or "",
        "islands": len(info["islands"]), "world_size": world, "bytes": nbytes, "dtype": dtype,
        "trials": len(lat), "median_us": round(median * 1e6, 2), "p90_us": round(p90 * 1e6, 2),
        "min_us": round(min(lat) * 1e6, 2), "algbw_GBps": round(algbw, 4),
        "busbw_GBps": round(algbw * 2 * (world - 1) / world, 4) if world > 1 else None,
    }


class PhaseTimes(gpubridge.CollectiveObserver):
    """Keeps every record; reads phase times once the collective's marks are ready."""

    def __init__(self) -> None:
        self.records: list[gpubridge.CollectiveRecord] = []

    def on_collective(self, record: gpubridge.CollectiveRecord) -> None:
        self.records.append(record)


def phase_medians(records: list[gpubridge.CollectiveRecord]) -> dict[str, float]:
    """Median milliseconds per phase name over ``records`` on this rank."""
    by_phase: dict[str, list[float]] = {}
    for record in records:
        for p in record.phases:
            by_phase.setdefault(p.name, []).append(p.elapsed_ms())
    return {name: statistics.median(times) for name, times in by_phase.items()}


def slowest_phases(mine: dict[str, float]) -> dict[str, float]:
    """Each phase's time on the slowest rank that ran it. Collective over the world group."""
    everyone: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, mine)
    # In call order: rank 0 always leads its island, so it ran every phase.
    names = list(dict.fromkeys(name for medians in everyone for name in medians))
    return {name: round(max(m[name] for m in everyone if name in m), 4) for name in names}


def summary_table(rows: list[dict], info: dict[str, Any]) -> str:
    lines = []
    if info["split_test"]:
        lines += ["> **SPLIT TEST - not a mixed-vendor result.** Islands were faked from a "
                  "single-vendor job with `GPUBRIDGE_SPLIT_TEST`.", ""]
    lines += [f"Run kind `{info['kind']}`, gpubridge policy `{info['policy']}`, "
              f"{info['world_size']} ranks, "
              f"{len(info['islands'])} island(s). Latency is the median of the slowest rank.",
              ""]
    present = {r["op"] for r in rows}
    islands = sorted(op for op in present if op.startswith("island:"))
    ops = [op for op in ("native", *islands, "gpubridge") if op in present]
    ratios = []
    if "gpubridge" in present and "native" in present:
        ratios.append("gpubridge / native")
    if "gpubridge" in present and islands:
        ratios.append("gpubridge / slowest island")
    header = ["size"] + [f"{op} {what}" for op in ops
                         for what in ("median (us)", "busbw (GB/s)")] + ratios
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    by_size: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_size.setdefault(r["bytes"], {})[r["op"]] = r

    def cell(r, key):
        return "-" if r is None or r[key] is None else f"{r[key]}"

    def ratio(bridge, base_us):
        return f"{bridge['median_us'] / base_us:.2f}x" if bridge and base_us else "-"

    for nbytes, by_op in sorted(by_size.items()):
        cells = [human(nbytes)]
        for op in ops:
            cells += [cell(by_op.get(op), "median_us"), cell(by_op.get(op), "busbw_GBps")]
        bridge = by_op.get("gpubridge")
        if "gpubridge / native" in ratios:
            cells.append(ratio(bridge, by_op.get("native", {}).get("median_us")))
        if "gpubridge / slowest island" in ratios:
            island_us = [by_op[op]["median_us"] for op in islands if op in by_op]
            cells.append(ratio(bridge, max(island_us, default=None)))
        lines.append("| " + " | ".join(cells) + " |")
    phased = [r for r in rows if r.get("phases_ms")]
    if phased:
        names = list(dict.fromkeys(name for r in phased for name in r["phases_ms"]))
        lines += ["", f"Phases of the gpubridge op ({phased[0]['policy']}), median ms on the "
                  "slowest rank that ran each phase:", "",
                  "| size | " + " | ".join(names) + " |", "|" + " --- |" * (len(names) + 1)]
        for r in phased:
            lines.append(f"| {human(r['bytes'])} | "
                         + " | ".join(str(r["phases_ms"].get(n, "-")) for n in names) + " |")
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
    parser.add_argument("--ops", default="native,gpubridge",
                        help="which to measure: native, island, gpubridge")
    parser.add_argument("--policy", default="auto",
                        help="collective policy for the gpubridge op, e.g. flat-gloo")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--phases", action="store_true",
                        help="also record the median time of each phase of the gpubridge op")
    args = parser.parse_args()

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout), policy=args.policy)
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
    if "island" in wanted:
        ops["island"] = lambda t: dist.all_reduce(t, group=topology.island_group)
    if "gpubridge" in wanted:
        ops["gpubridge"] = gpubridge.all_reduce

    phases = PhaseTimes() if args.phases else None
    if phases is not None:
        gpubridge.add_observer(phases)
    element = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for nbytes in sizes(args.min_bytes, args.max_bytes, args.factor):
        tensor = torch.ones(max(1, nbytes // element), dtype=dtype, device=device)
        for name, op in ops.items():
            if phases is not None:
                phases.records.clear()
            per_trial = time_op(op, tensor, device, warmup=args.warmup, trials=args.trials,
                                max_seconds=args.max_seconds)
            if name == "island":
                new = [row(f"island:{island.vendor}", info, tensor.numel() * element,
                           args.dtype, slowest(per_trial, island.ranks), world=len(island.ranks))
                       for island in topology.layout.islands]
            else:
                new = [row(name, info, tensor.numel() * element, args.dtype, slowest(per_trial))]
            if name == "gpubridge" and phases is not None:
                # time_op synchronized the device, so every mark is ready. Only the
                # timed trials count, not the warmup.
                timed = phases.records[len(phases.records) - len(per_trial):]
                new[0]["phases_ms"] = slowest_phases(phase_medians(timed))
            rows += new
            if topology.rank == 0:
                for r in new:
                    print(f"{r['op']:>14} {human(r['bytes']):>6}: median {r['median_us']:>12} us  "
                          f"busbw {r['busbw_GBps']} GB/s  ({r['trials']} trials)", flush=True)
        del tensor

    if topology.rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "bench.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
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
