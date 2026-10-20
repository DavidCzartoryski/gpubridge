"""Staged verbs all_reduce on one node: one arm of an A/B, every raw sample kept.

docs/verbs-perf-20261019.md, "M1", is the plan this runs: its arms, rotation,
acceptance rules and owner commands. ``run`` is one arm in one job, launched by
torchrun; ``compare`` reads the runs of two arms and applies the predeclared rules.

    GPUBRIDGE_SPLIT_TEST=half GPUBRIDGE_VERBS_GPU_PATH=staged \\
        torchrun --nproc-per-node=4 scripts/gpu/verbs_staged_ab.py run \\
        --op gpubridge --bridge verbs --arm before --repeat 0 --out OUT
    python scripts/gpu/verbs_staged_ab.py compare OUT --before before --after after

Per size of the fixed table (fp32 integer values, so every sum is exact), ``run``
does untimed warmup calls, then timed calls. Each timed call starts from the same
input after a barrier and is timed with CUDA events on the current stream (on CPU,
the host timer) and on the host; its result is then checked against the exact sum,
outside the timed region. Each rank keeps every sample, its process CPU around each
call (all threads: the bridge's progress threads too) and over the whole timed
region, its max RSS, and on a bridge leader the verbs telemetry's change over the
timed calls (sends, write windows and their nanoseconds, copies, completions,
staged phase times). Rank 0 writes ``OUT/<arm>-r<repeat>.json`` with every rank's
data, the run's full wall time and its provenance (git commit, imported sources'
hash, the verbs library's file hash, allowlisted environment).

``--op native`` times torch.distributed.all_reduce on one group of every rank on
the island backend (NCCL), in the same job layout, as the baseline. It refuses to
run where that is not possible (mixed vendors, shared GPUs) rather than time
something else.

Numbers from CPU runs (the tests) are software checks only, never GPU or NIC evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

#: (bytes, warmup calls, timed calls): the predeclared table. ``smoke`` is for CPU tests.
TABLES = {
    "full": [(64 * 2**10, 20, 300), (2**20, 10, 100), (4 * 2**20, 10, 50),
             (25 * 2**20, 5, 30), (150 * 2**20, 3, 10)],
    "smoke": [(4 * 2**10, 2, 5), (2**20, 2, 5)],
}
#: Verbs telemetry counts whose change over the timed calls is recorded: top level,
#: then under ``direct``.
COUNTS = ("sends", "recvs", "bytes_sent", "bytes_received", "writes_posted",
          "bytes_written", "copied_bytes", "control_sent", "completions", "notifications",
          "write_windows_closed", "write_windows_retired")
DIRECT_COUNTS = ("write_windows", "write_window_bytes", "write_window_register_ns",
                 "write_windows_released", "write_window_release_ns")
#: Counts that must be exactly equal between arms of one transport (same messages). Not
#: copied_bytes or control_sent: whether an eager message is copied once or twice, and
#: when credits go back, depend on timing, so they are recorded and reported only.
WIRE = ("sends", "recvs", "bytes_sent", "bytes_received", "writes_posted", "bytes_written",
        "write_windows", "write_window_bytes")
LABEL_CPU = "CPU run: software check only, not GPU or NIC evidence"


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def verbs_counts(bridge) -> dict[str, float] | None:
    """The recorded verbs telemetry counts of this rank's bridge, or None (no verbs
    bridge here)."""
    if bridge is None or getattr(bridge, "name", None) != "verbs":
        return None
    t = bridge.telemetry()
    out = {k: t.get(k, 0) for k in COUNTS}
    out.update({k: (t.get("direct") or {}).get(k, 0) for k in DIRECT_COUNTS})
    for k, v in (t.get("staged_phases") or {}).items():
        out[f"staged_{k}"] = v
    for k, v in (t.get("paths") or {}).items():
        out[f"path_{k}"] = v
    return out


def provenance(topology) -> dict[str, Any]:
    import gpubridge
    from gpukit import environment

    package = Path(gpubridge.__file__).parent
    out: dict[str, Any] = {"environment": environment(),
                           "gpubridge_py_sha256": sha256_files(package.rglob("*.py"))}
    bridge = topology.bridge
    if bridge is not None and getattr(bridge, "name", None) == "verbs":
        from gpubridge import verbs

        lib = Path(bridge.worker.endpoint._lib._name)
        out["verbs_library"] = lib.name
        out["verbs_library_sha256"] = hashlib.sha256(lib.read_bytes()).hexdigest()
        out["verbs_crate_sha256"] = sha256_files(verbs._sources())
        t = bridge.telemetry()
        out["verbs"] = {k: t.get(k) for k in ("provider", "mock", "library", "device", "port",
                                              "link_layer", "progress", "rndv_bytes",
                                              "chunk_bytes", "gpu_path_setting")}
    return out


def time_size(op, topology, nbytes: int, warmup: int, calls: int) -> dict[str, Any]:
    """This rank's samples for one size (module docs)."""
    import torch
    import torch.distributed as dist
    from gpukit import sync

    device, world = topology.device, topology.world_size
    numel = nbytes // 4
    pattern = torch.arange(numel, dtype=torch.float32) % 1021
    base = (pattern + topology.rank).to(device)
    want = (pattern * world + sum(range(world))).to(device)
    tensor = base.clone()
    events = device.type == "cuda"
    w0 = time.perf_counter()
    for _ in range(warmup):
        tensor.copy_(base)
        op(tensor)
    sync(device)
    warm_s = time.perf_counter() - w0
    before, rss0 = verbs_counts(topology.bridge), maxrss_bytes()
    timed, host, cpu, wrong = [], [], [], 0
    region_cpu0, t0 = cpu_s(), time.perf_counter()
    for _ in range(calls):
        tensor.copy_(base)
        sync(device)
        dist.barrier()
        begin = end = None
        if events:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(device))
        c0, h0 = cpu_s(), time.perf_counter()
        op(tensor)
        if end is not None:
            end.record(torch.cuda.current_stream(device))
        sync(device)
        h1, c1 = time.perf_counter(), cpu_s()
        host.append(h1 - h0)
        cpu.append(c1 - c0)
        timed.append(begin.elapsed_time(end) / 1e3 if begin is not None else h1 - h0)
        wrong += not torch.equal(tensor, want)
    region_s, region_cpu = time.perf_counter() - t0, cpu_s() - region_cpu0
    after = verbs_counts(topology.bridge)
    delta = None if before is None else {k: after[k] - before[k] for k in after}
    return {"bytes": nbytes, "payload_bytes": numel * 4, "warmup": warmup, "calls": calls,
            "timer": "cuda_events" if events else "host", "timed_s": timed, "host_s": host,
            "cpu_s": cpu, "mismatches": wrong, "warmup_wall_s": warm_s,
            "timed_wall_s": region_s, "timed_cpu_s": region_cpu, "maxrss_bytes": maxrss_bytes(),
            "rss_growth_bytes": maxrss_bytes() - rss0, "verbs_delta": delta}


def run(args) -> int:
    start = time.time()
    t_start = time.perf_counter()
    import torch.distributed as dist
    from bench_all_reduce import native_possible
    from gpukit import default_timeout, run_info, write_json

    import gpubridge

    timeout = timedelta(seconds=default_timeout(1800))
    topology = gpubridge.init(timeout=timeout, policy=args.policy, bridge=args.bridge)
    if args.op == "native":
        why_not = native_possible(topology)
        if why_not is not None:
            if topology.rank == 0:
                print(f"UNSUPPORTED: native all_reduce is not possible here: {why_not}",
                      file=sys.stderr)
            return 3
        group = dist.new_group(list(range(topology.world_size)),
                               backend=topology.config.island_backend, timeout=timeout)
        op = lambda t: dist.all_reduce(t, group=group)  # noqa: E731
    else:
        op = gpubridge.all_reduce
    sizes = []
    for nbytes, warmup, calls in TABLES[args.table]:
        mine = time_size(op, topology, nbytes, warmup, calls)
        mine["rank"] = topology.rank
        mine["last_path"] = (topology.bridge.telemetry().get("path")
                             if verbs_counts(topology.bridge) is not None else None)
        ranks: list[Any] = [None] * topology.world_size
        dist.all_gather_object(ranks, mine)
        sizes.append(ranks)
    prov = provenance(topology)
    provs: list[Any] = [None] * topology.world_size
    dist.all_gather_object(provs, prov)
    info = run_info(topology)
    dist.barrier()
    if topology.rank == 0:
        record = {"arm": args.arm, "repeat": args.repeat, "op": args.op, "bridge": args.bridge,
                  "policy": args.policy, "table": args.table, "started": start,
                  "wall_s": time.perf_counter() - t_start, "run": info,
                  "provenance": provs, "sizes": sizes}
        if topology.device.type != "cuda":
            record["label"] = LABEL_CPU
        write_json(Path(args.out) / f"{args.arm}-r{args.repeat}.json", record)
    gpubridge.destroy()
    return 0


# ---- compare ------------------------------------------------------------------------

def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def summarize(record: dict) -> dict[int, dict[str, Any]]:
    """Per size: the slowest rank's latency per call, then p50, p99, throughput, mean CPU
    per call and max RSS growth over the ranks, mismatches over every rank, and the
    leaders' summed verbs deltas."""
    out = {}
    for ranks in record["sizes"]:
        slowest = [max(r["timed_s"][i] for r in ranks) for i in range(ranks[0]["calls"])]
        p50 = statistics.median(slowest)
        deltas = [r["verbs_delta"] for r in ranks if r["verbs_delta"] is not None]
        counts = {k: sum(d[k] for d in deltas) for k in deltas[0]} if deltas else {}
        out[ranks[0]["bytes"]] = {
            "p50_s": p50, "p99_s": quantile(slowest, 0.99),
            "throughput_GBps": ranks[0]["payload_bytes"] / p50 / 1e9,
            "cpu_per_call_s": max(statistics.fmean(r["cpu_s"]) for r in ranks),
            "rss_growth_bytes": max(r["rss_growth_bytes"] for r in ranks),
            "mismatches": sum(r["mismatches"] for r in ranks), "counts": counts}
    return out


def spread(values: list[float]) -> float:
    m = statistics.median(values)
    return (max(values) - min(values)) / m if m else 0.0


def compare(args) -> int:
    """The predeclared rules (docs/verbs-perf-20261019.md, "Acceptance"), per size."""
    runs: dict[str, dict[int, dict]] = {args.before: {}, args.after: {}}
    for path in sorted(Path(args.dir).glob("*-r*.json")):
        record = json.loads(path.read_text())
        if record["arm"] in runs:
            runs[record["arm"]][record["repeat"]] = summarize(record)
    pairs = sorted(set(runs[args.before]) & set(runs[args.after]))
    if not pairs:
        print("UNSUPPORTED: no repeat has both arms", file=sys.stderr)
        return 3
    rows, failed = [], False
    for nbytes in runs[args.before][pairs[0]]:
        b = [runs[args.before][p][nbytes] for p in pairs]
        a = [runs[args.after][p][nbytes] for p in pairs]
        row: dict[str, Any] = {"bytes": nbytes, "pairs": len(pairs)}
        if any(x["mismatches"] for x in a + b):
            row["verdict"] = "FAIL (wrong result)"
            failed = True
        for metric in ("p50_s", "p99_s", "cpu_per_call_s"):
            vb, va = [x[metric] for x in b], [x[metric] for x in a]
            s = max(spread(vb), spread(va))
            change = statistics.median(va) / statistics.median(vb) - 1
            better = sum(y < x for x, y in zip(vb, va, strict=True))
            row[metric] = {"before": statistics.median(vb), "after": statistics.median(va),
                           "change": change, "spread": s, "pairs_better": better}
            if change > s:
                row.setdefault("regressions", []).append(metric)
        if args.wire:
            differ = sorted({k for x, y in zip(b, a, strict=True) for k in WIRE
                             if x["counts"].get(k) != y["counts"].get(k)})
            if differ:
                row.setdefault("regressions", []).append(f"wire counts differ: {differ}")
        p50 = row["p50_s"]
        if row.get("regressions") and "verdict" not in row:
            row["verdict"] = "REGRESSION"
            failed = True
        elif -p50["change"] > p50["spread"] and p50["pairs_better"] >= max(1, len(pairs) - 1):
            row.setdefault("verdict", "win")
        else:
            row.setdefault("verdict", "no change (inside spread)")
        rows.append(row)
    Path(args.dir, f"compare-{args.before}-{args.after}.json").write_text(
        json.dumps({"before": args.before, "after": args.after, "pairs": pairs, "rows": rows},
                   indent=1))
    print("| bytes | p50 before | p50 after | change | spread | pairs better | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        p = r["p50_s"]
        print(f"| {r['bytes']} | {p['before'] * 1e6:.1f} us | {p['after'] * 1e6:.1f} us | "
              f"{p['change']:+.1%} | {p['spread']:.1%} | {p['pairs_better']}/{r['pairs']} | "
              f"{r['verdict']}{' ' + str(r['regressions']) if r.get('regressions') else ''} |")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="one arm, under torchrun")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--op", choices=("gpubridge", "native"), required=True)
    p.add_argument("--bridge", choices=("verbs", "gloo"), required=True)
    p.add_argument("--policy", default="auto")
    p.add_argument("--table", choices=sorted(TABLES), default="full")
    c = sub.add_parser("compare", help="two arms' runs against the predeclared rules")
    c.add_argument("dir")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--wire", action="store_true",
                   help="the arms must put the same messages on the wire (equal counts)")
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
