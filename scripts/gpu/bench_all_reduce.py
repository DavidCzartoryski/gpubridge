"""all_reduce latency and bandwidth: native NCCL/RCCL, gpubridge and flat-gloo side by side.

Launched by torchrun. For each tensor size (1 KB to 1 GB by default, x4 per
step) it times, by default:

- ``native``: torch.distributed.all_reduce on one group of all ranks using the
  island backend (NCCL/RCCL on GPUs, Gloo on CPU): the baseline. Only possible
  when every rank has the same detected vendor (split tests included) and no
  two ranks share a GPU. Otherwise, as in a real mixed job, the baseline is
  ``island`` instead, labeled "island-native (fewer ranks, best case)": each
  island all_reduces over fewer ranks and crosses no bridge, so it is a lower
  bound rather than a like-for-like comparison.
- ``gpubridge``: gpubridge.all_reduce. Bridged when the job has two islands
  (a split test or a real mixed job); a plain native all_reduce otherwise.
- ``flat-gloo``: the flat-gloo policy on the same ranks (stage to CPU, Gloo
  over every rank, copy back): what gpubridge improves on.
- ``island`` (also opt in with --ops): every island all_reduces on its own
  native group at the same time (NCCL on NVIDIA, RCCL on AMD): one row per
  island, ``island:<vendor>``.

Each size gets --warmup untimed runs, then timed trials until --trials or
--max-seconds. On GPUs a trial is timed with CUDA events recorded on the
current stream around the call (the host timer is kept as ``host_median_us``);
on CPU with the host timer. A trial's latency is the slowest rank's (for
``island``, the slowest rank of that island). Bandwidth follows nccl-tests:
algbw = bytes / time and busbw = algbw * 2(n-1)/n, with n the ranks taking
part. Every row names the baseline of its run (``baseline``), and summary.md
puts every op side by side, with gpubridge and flat-gloo as multiples of it.

--phases attaches a gpubridge observer and adds the median time of each phase
of the gpubridge op (island-reduce, bridge, island-broadcast for
reduce-bridge-broadcast), slowest rank, to bench.json and summary.md. With
--ops island,gpubridge that puts the island step of reduce-bridge-broadcast (a
reduce onto the leader) next to a native island all_reduce of the same size.

--candidates a,b,c times gpubridge's all_reduce under each of those collective
policies in the same job (rows ``policy:<name>``; policies that don't fit the
cluster are skipped). --write-thresholds PATH also times the candidates
(default: flat-gloo, reduce-bridge-broadcast and the pipelined and sharded
policies), picks
the fastest per size and writes a thresholds file for the ``auto-tuned``
policy: point GPUBRIDGE_THRESHOLDS at it on every rank. Measure on the cluster
and layout you will train on; thresholds don't carry over.

--compression none,bf16,fp8 also times the active policy (if it bridges:
reduce-bridge-broadcast, pipelined or sharded) with each bridge compression
mode, on its own instance (rows ``compression:<mode>``; GPUBRIDGE_BRIDGE_COMPRESSION
needn't be set). With --phases each row gets its compress, bridge and decompress
times, so the bridge step's own time shows with and without compression. Each
row also says, for one random tensor per size, the largest error against the
uncompressed result (relative to its largest magnitude) and whether every rank
ended with the same bits; summary.md says from which size each mode is faster.

Rank 0 writes OUT/bench.csv, OUT/bench.json and OUT/summary.md. Runs on CPU in
simulation mode too (use a smaller --max-bytes):

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        scripts/gpu/bench_all_reduce.py --out /tmp/bench --max-bytes 4M
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json

import gpubridge
from gpubridge import observe
from gpubridge.config import chunk_bytes_setting
from gpubridge.policies import (
    THRESHOLDS_FORMAT,
    FlatGloo,
    PipelinedReduceBridgeBroadcast,
    ReduceBridgeBroadcast,
    choose_policy,
    resolve_policy,
)

DEFAULT_CANDIDATES = ("flat-gloo,reduce-bridge-broadcast,pipelined-reduce-bridge-broadcast,"
                      "sharded-bridge")

UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
FIELDS = ["op", "policy", "baseline", "run_kind", "split_test", "islands", "world_size", "bytes",
          "dtype", "trials", "timer",
          "median_us", "p90_us", "min_us", "algbw_GBps", "busbw_GBps", "host_median_us",
          "max_rel_error", "identical"]
COMPRESSION_MODES = ("none", "bf16", "fp8")
#: The baseline when there is no ``native`` row: each island on its own.
ISLAND_BASELINE = "island-native (fewer ranks, best case)"


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
            max_seconds: float) -> tuple[list[list[float]], list[list[float]]]:
    """Every rank's latency in seconds for each timed trial: (timed, host).

    ``timed`` comes from CUDA events on the current stream on GPUs, from the
    host timer on CPU; ``host`` is always the host timer.
    """
    for _ in range(warmup):
        op(tensor)
    sync(device)
    world = dist.get_world_size()
    events = device.type == "cuda"
    timed: list[list[float]] = []
    host: list[list[float]] = []
    spent = 0.0
    while len(timed) < trials:
        dist.barrier()
        begin = end = None
        if events:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(device))
        start = time.perf_counter()
        op(tensor)
        if end is not None:
            end.record(torch.cuda.current_stream(device))
        sync(device)
        on_host = time.perf_counter() - start
        timed_s = begin.elapsed_time(end) / 1e3 if begin is not None else on_host
        mine = torch.tensor([timed_s, on_host], dtype=torch.float64)
        gathered = [torch.zeros(2, dtype=torch.float64) for _ in range(world)]
        dist.all_gather(gathered, mine)  # world group is Gloo, on CPU
        timed.append([float(t[0]) for t in gathered])
        host.append([float(t[1]) for t in gathered])
        spent += max(timed[-1])
        # Every rank sees the same values, so every rank stops at the same trial.
        if spent > max_seconds and len(timed) >= 3:
            break
    return timed, host


def slowest(per_trial: list[list[float]], ranks=None) -> list[float]:
    """Each trial's latency: the slowest of ``ranks`` (default: every rank)."""
    return [max(t if ranks is None else [t[r] for r in ranks]) for t in per_trial]


def row(op: str, info: dict[str, Any], nbytes: int, dtype: str, lat: list[float],
        world: int | None = None, *, host: list[float] | None = None,
        timer: str = "host") -> dict:
    """One result row. ``world`` is the ranks taking part (default: all).

    ``lat`` are the trials' latencies from ``timer``; ``host``, if given, the
    same trials on the host timer.
    """
    world = world or info["world_size"]
    median = statistics.median(lat)
    p90 = sorted(lat)[max(0, int(round(0.9 * len(lat))) - 1)]
    algbw = nbytes / median / 1e9
    if op == "gpubridge" or op.startswith("compression:"):
        policy = info["policy"]
    elif op.startswith("policy:"):
        policy = op.removeprefix("policy:")
    else:
        policy = "flat-gloo" if op == "flat-gloo" else ""
    return {
        "op": op, "policy": policy,
        "run_kind": info["kind"], "split_test": info["split_test"] or "",
        "islands": len(info["islands"]), "world_size": world, "bytes": nbytes, "dtype": dtype,
        "trials": len(lat), "timer": timer,
        "median_us": round(median * 1e6, 2), "p90_us": round(p90 * 1e6, 2),
        "min_us": round(min(lat) * 1e6, 2), "algbw_GBps": round(algbw, 4),
        "busbw_GBps": round(algbw * 2 * (world - 1) / world, 4) if world > 1 else None,
        "host_median_us": round(statistics.median(host if host else lat) * 1e6, 2),
    }


def native_possible(topology: gpubridge.Topology) -> str | None:
    """Why one native group over every rank can't be formed here, or None if it can."""
    if len({peer.vendor for peer in topology.peers}) > 1:
        return "ranks have different real vendors (NCCL and RCCL can't share a group)"
    gpus = [(peer.hostname, peer.gpu) for peer in topology.peers if peer.gpu]
    if len(set(gpus)) < len(gpus):
        return "ranks share a GPU, which NCCL and RCCL refuse within one group"
    return None


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


def baseline_of(ops) -> str:
    """What the gpubridge and flat-gloo rows of a run with these ops are compared with."""
    return "native" if "native" in ops else ISLAND_BASELINE if "island" in ops else "none"


def summary_table(rows: list[dict], info: dict[str, Any]) -> str:
    lines = []
    if info["split_test"]:
        lines += ["> **SPLIT TEST - not a mixed-vendor result.** Islands were faked from a "
                  "single-vendor job with `GPUBRIDGE_SPLIT_TEST`.", ""]
    timers = sorted({r.get("timer", "host") for r in rows})
    timing = ("CUDA events on the current stream" if timers == ["cuda-events"]
              else "the host timer" if timers == ["host"] else ", ".join(timers))
    if info.get("bridge") == "ucx" and info.get("bridge_path"):
        staging = f"by UCX ({info['bridge_path']})"
    else:
        staging = ("pinned (`GPUBRIDGE_PINNED_STAGING`)" if info.get("pinned_staging")
                   else "pageable")
    lines += [f"Run kind `{info['kind']}`, gpubridge policy `{info['policy']}`, "
              f"bridge `{info.get('bridge', 'gloo')}`, "
              f"bridge staging {staging}, {info['world_size']} ranks, "
              f"{len(info['islands'])} island(s). Latency is the median of the slowest rank, "
              f"timed with {timing}.",
              ""]
    present = {r["op"] for r in rows}
    islands = sorted(op for op in present if op.startswith("island:"))
    candidates = list(dict.fromkeys(r["op"] for r in rows if r["op"].startswith("policy:")))
    ops = [op for op in ("native", *islands, "gpubridge", "flat-gloo", *candidates)
           if op in present]
    base = "native" if "native" in present else ISLAND_BASELINE if islands else None
    ratios = [f"{op} / {base}" for op in ("gpubridge", "flat-gloo") if base and op in present]
    if base == ISLAND_BASELINE:
        lines += [f"Baseline: **{ISLAND_BASELINE}**, the slowest island's own native "
                  "all_reduce at each size. One native group over every rank wasn't measured "
                  "here, and each island reduces over fewer ranks with no bridge, so this is a "
                  "best case for any all_reduce across the islands, not a like-for-like "
                  "comparison.", ""]
    header = ["size"] + [f"{op} {what}" for op in ops
                         for what in ("median (us)", "busbw (GB/s)")] + ratios
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    by_size: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_size.setdefault(r["bytes"], {})[r["op"]] = r

    def cell(r, key):
        return "-" if r is None or r[key] is None else f"{r[key]}"

    def ratio(measured, base_us):
        return f"{measured['median_us'] / base_us:.2f}x" if measured and base_us else "-"

    for nbytes, by_op in sorted(by_size.items()):
        cells = [human(nbytes)]
        for op in ops:
            cells += [cell(by_op.get(op), "median_us"), cell(by_op.get(op), "busbw_GBps")]
        if base == "native":
            base_us = by_op.get("native", {}).get("median_us")
        else:
            base_us = max((by_op[op]["median_us"] for op in islands if op in by_op), default=None)
        for label in ratios:
            cells.append(ratio(by_op.get(label.split(" / ")[0]), base_us))
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


def candidate_ops(names: str, topology: gpubridge.Topology) -> dict[str, Any]:
    """One timed op per candidate policy that fits this cluster, run directly on its instance.

    Every rank builds the same instances in the same order (the layout is the
    same everywhere), so their setup() may create groups.
    """
    ops: dict[str, Any] = {}
    chunk = chunk_bytes_setting(PipelinedReduceBridgeBroadcast.DEFAULT_CHUNK_BYTES)
    for name in names.split(","):
        cls = resolve_policy(name)
        if isinstance(cls, str) or not cls.applies_to(topology.layout):
            continue
        cls = choose_policy(cls, topology.layout)
        policy = (cls(chunk_bytes=chunk) if issubclass(cls, PipelinedReduceBridgeBroadcast)
                  else cls())
        policy.setup(topology)
        ops[f"policy:{name}"] = lambda t, p=policy: p.all_reduce(t, topology)
    return ops


def compression_ops(modes: str, topology: gpubridge.Topology, policy: str) -> dict[str, Any]:
    """One timed op per bridge compression mode, on an instance of the active policy.

    Calls go through ``observe.collective`` like gpubridge.all_reduce, so --phases
    sees their compress, bridge and decompress steps. Every rank builds the same
    instances in the same order.
    """
    cls = choose_policy(resolve_policy(policy), topology.layout)
    if not issubclass(cls, ReduceBridgeBroadcast) or not topology.layout.needs_bridge:
        if topology.rank == 0:
            print(f"--compression: {cls.name} on this layout has no bridge step to compress; "
                  "skipped", file=sys.stderr)
        return {}
    chunk = chunk_bytes_setting(PipelinedReduceBridgeBroadcast.DEFAULT_CHUNK_BYTES)
    ops: dict[str, Any] = {}
    for mode in modes.split(","):
        if mode not in COMPRESSION_MODES:
            raise SystemExit(f"--compression: unknown mode {mode!r}; use {COMPRESSION_MODES}")
        compression = "" if mode == "none" else mode
        instance = (cls(chunk_bytes=chunk, compression=compression)
                    if issubclass(cls, PipelinedReduceBridgeBroadcast) else cls(compression=compression))
        instance.setup(topology)

        def run(t, p=instance):
            with observe.collective(topology, "all_reduce", t, policy=p.name, reduce_op="sum"):
                p.all_reduce(t, topology)

        run.instance = instance
        ops[f"compression:{mode}"] = run
    return ops


def compression_quality(op, uncompressed, numel: int, dtype, device, topology) -> dict[str, Any]:
    """Largest error against the uncompressed result, for one random tensor, and whether
    every rank ended with the same bits."""
    g = torch.Generator().manual_seed(1000 + topology.rank)
    x = torch.randn(numel, generator=g).to(dtype).to(device)
    want, got = x.clone(), x.clone()
    uncompressed(want)
    op(got)
    sync(device)
    scale = want.double().abs().max().item() or 1.0
    error = (got.double() - want.double()).abs().max().item() / scale
    digests: list[Any] = [None] * topology.world_size
    dist.all_gather_object(digests, hashlib.sha256(got.cpu().view(torch.uint8).numpy().tobytes())
                           .hexdigest())
    errors: list[Any] = [None] * topology.world_size
    dist.all_gather_object(errors, error)
    return {"max_rel_error": float(f"{max(errors):.3g}"), "identical": len(set(digests)) == 1}


def compression_table(rows: list[dict]) -> str:
    """Median time and bridge phase per mode and size, and where each mode starts to win."""
    by: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r["op"].startswith("compression:"):
            by.setdefault(r["bytes"], {})[r["op"].removeprefix("compression:")] = r
    if not by:
        return ""
    modes = [m for m in COMPRESSION_MODES if any(m in cells for cells in by.values())]
    lines = ["", "## Bridge compression", "",
             "Median time per all_reduce (bridge phase, slowest rank, in ms, with --phases); "
             "max relative error against uncompressed; whether every rank had the same bits.", "",
             "| size | " + " | ".join(modes) + " |", "|" + " --- |" * (len(modes) + 1)]
    for nbytes in sorted(by):
        cells = []
        for m in modes:
            r = by[nbytes].get(m)
            if r is None:
                cells.append("-")
                continue
            bridge = r.get("phases_ms", {}).get("bridge")
            cell = f"{r['median_us']} us"
            if bridge is not None:
                cell += f" (bridge {bridge})"
            if m != "none":
                cell += f", err {r.get('max_rel_error')}, {'same' if r.get('identical') else 'DIFFER'}"
            cells.append(cell)
        lines.append(f"| {human(nbytes)} | " + " | ".join(cells) + " |")
    for m in modes:
        if m == "none":
            continue
        faster = [n for n in sorted(by) if m in by[n] and "none" in by[n]
                  and by[n][m]["median_us"] < by[n]["none"]["median_us"]]
        sizes = sorted(n for n in by if m in by[n] and "none" in by[n])
        start = next((n for n in sizes if all(k in faster for k in sizes if k >= n)), None)
        lines.append("")
        lines.append(f"{m}: faster than uncompressed from {human(start)} up." if start is not None
                     else f"{m}: not faster than uncompressed at any measured size.")
    return "\n".join(lines) + "\n"


def derive_thresholds(rows: list[dict], info: dict[str, Any], chunk_bytes: int) -> dict:
    """The fastest candidate per size, merged into rules for the auto-tuned policy."""
    by_size: dict[int, list[dict]] = {}
    for r in rows:
        if r["op"].startswith("policy:"):
            by_size.setdefault(r["bytes"], []).append(r)
    winners = [(nbytes, min(cands, key=lambda r: r["median_us"])["policy"])
               for nbytes, cands in sorted(by_size.items())]
    rules: list[dict[str, Any]] = []
    for nbytes, policy in winners:
        if rules and rules[-1]["policy"] == policy:
            rules[-1]["max_bytes"] = nbytes
        else:
            rules.append({"max_bytes": nbytes, "policy": policy})
    if rules:
        rules[-1]["max_bytes"] = None  # the largest measured winner covers everything above
    return {
        "gpubridge_thresholds": THRESHOLDS_FORMAT,
        "rules": rules,
        "chunk_bytes": chunk_bytes,
        "measured": {"run_kind": info["kind"], "split_test": info["split_test"],
                     "islands": info["islands"], "world_size": info["world_size"],
                     "torch": torch.__version__, "gpubridge": gpubridge.__version__,
                     "date": time.strftime("%Y-%m-%d"),
                     "median_us": {f"{r['policy']}@{r['bytes']}": r["median_us"]
                                   for cands in by_size.values() for r in cands}},
    }


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
    parser.add_argument("--ops", default="native,gpubridge,flat-gloo",
                        help="which to measure: native, island, gpubridge, flat-gloo. native "
                             "falls back to island where one native group is impossible")
    parser.add_argument("--policy", default="auto",
                        help="collective policy for the gpubridge op, e.g. flat-gloo")
    parser.add_argument("--bridge", default="gloo",
                        help="bridge transport for the gpubridge op: gloo, or ucx "
                             "(docs/RDMA_TRANSPORT.md)")
    parser.add_argument("--timeout", type=float, default=default_timeout(1800),
                        help="group timeout (s); default KIT_TIMEOUT, else 1800")
    parser.add_argument("--phases", action="store_true",
                        help="also record the median time of each phase of the gpubridge op")
    parser.add_argument("--candidates", default=None,
                        help="also time gpubridge's all_reduce under these policies (comma list)")
    parser.add_argument("--write-thresholds", type=Path, default=None, metavar="PATH",
                        help="time the candidates and write an auto-tuned thresholds file")
    parser.add_argument("--compression", default=None, metavar="MODES",
                        help="also time the active policy with bridge compression: a comma "
                             "list of none, bf16, fp8 (rows compression:<mode>)")
    args = parser.parse_args()
    if args.write_thresholds and not args.candidates:
        args.candidates = DEFAULT_CANDIDATES

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout), policy=args.policy,
                              bridge=args.bridge)
    info = run_info(topology)
    device, dtype = topology.device, DTYPES[args.dtype]
    if topology.split_test and topology.rank == 0:
        print(info["warning"], file=sys.stderr, flush=True)

    ops: dict[str, Any] = {}
    wanted = args.ops.split(",")
    if "native" in wanted:
        why_not = native_possible(topology)
        if why_not is None:
            # Collective: every rank creates it. Same backend as the islands.
            group = dist.new_group(list(range(topology.world_size)),
                                   backend=topology.config.island_backend,
                                   timeout=timedelta(seconds=args.timeout))
            ops["native"] = lambda t: dist.all_reduce(t, group=group)
        else:
            if topology.rank == 0:
                print(f"native: not possible here ({why_not}); each island's native "
                      "all_reduce is the baseline instead", file=sys.stderr)
            if "island" not in wanted:
                wanted.append("island")
    if "island" in wanted:
        ops["island"] = lambda t: dist.all_reduce(t, group=topology.island_group)
    if "gpubridge" in wanted:
        ops["gpubridge"] = gpubridge.all_reduce
    if "flat-gloo" in wanted:
        flat = FlatGloo()
        flat.setup(topology)
        ops["flat-gloo"] = lambda t: flat.all_reduce(t, topology)
    if args.candidates:
        ops.update(candidate_ops(args.candidates, topology))
    squeezed = compression_ops(args.compression, topology, args.policy) if args.compression else {}
    ops.update(squeezed)
    uncompressed = None
    if squeezed:
        off = compression_ops("none", topology, args.policy)["compression:none"]
        uncompressed = lambda t: off.instance.all_reduce(t, topology)  # noqa: E731

    baseline = baseline_of(ops)
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
            per_trial, on_host = time_op(op, tensor, device, warmup=args.warmup,
                                         trials=args.trials, max_seconds=args.max_seconds)
            timer = "cuda-events" if device.type == "cuda" else "host"
            nbytes_real = tensor.numel() * element
            if name == "island":
                new = [row(f"island:{island.vendor}", info, nbytes_real, args.dtype,
                           slowest(per_trial, island.ranks), world=len(island.ranks),
                           host=slowest(on_host, island.ranks), timer=timer)
                       for island in topology.layout.islands]
            else:
                new = [row(name, info, nbytes_real, args.dtype, slowest(per_trial),
                           host=slowest(on_host), timer=timer)]
            for r in new:
                r["baseline"] = baseline
            if (name == "gpubridge" or name in squeezed) and phases is not None:
                # time_op synchronized the device, so every mark is ready. Only the
                # timed trials count, not the warmup.
                timed = phases.records[len(phases.records) - len(per_trial):]
                new[0]["phases_ms"] = slowest_phases(phase_medians(timed))
            if name in squeezed and uncompressed is not None:
                new[0].update(compression_quality(op, uncompressed, tensor.numel(), dtype,
                                                  device, topology))
            rows += new
            if topology.rank == 0:
                for r in new:
                    print(f"{r['op']:>14} {human(r['bytes']):>6}: median {r['median_us']:>12} us  "
                          f"busbw {r['busbw_GBps']} GB/s  ({r['trials']} trials)", flush=True)
        del tensor

    # Every leader's own path and GPUDirect RDMA flush (a mixed job's two sides may
    # differ), not just rank 0's.
    paths: list[tuple[str | None, str | None]] = [(None, None)] * topology.world_size
    dist.all_gather_object(paths, (getattr(topology.bridge, "last_path", None),
                                   getattr(topology.bridge, "flush_method", None)))
    info["bridge_path"] = "+".join(sorted({p for p, _ in paths if p})) or None
    info["bridge_flush"] = "; ".join(sorted({f for _, f in paths if f})) or None
    if topology.rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "bench.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        write_json(args.out / "bench.json", {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/bench_all_reduce.py",
            "argv": sys.argv, "run": info, "environment": environment(), "baseline": baseline,
            "rows": rows,
        })
        table = summary_table(rows, info) + compression_table(rows)
        (args.out / "summary.md").write_text(f"# {args.out.name}: all_reduce benchmark\n\n{table}")
        print(table)
        if args.write_thresholds:
            chunk = chunk_bytes_setting(PipelinedReduceBridgeBroadcast.DEFAULT_CHUNK_BYTES)
            thresholds = derive_thresholds(rows, info, chunk)
            write_json(args.write_thresholds, thresholds)
            print(f"thresholds -> {args.write_thresholds}: {thresholds['rules']}")
    gpubridge.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
