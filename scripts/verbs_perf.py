"""Implementation cost of ``bridge="verbs"`` on the MOCK test fabric (docs/VERBS_PERF.md).

CPU/MOCK, not a GPU or RDMA speedup: the test-only library carries RC queue pairs over
Unix sockets between processes on this machine, so these numbers are what the bridge's
own code costs per call (Python, Python/Rust crossings, control messages, copies,
registrations), never a NIC's bandwidth or latency.

    python scripts/verbs_perf.py run --series host --policy economy --out run.json
    python scripts/verbs_perf.py ab --arm before=/tmp/base --arm after=. --pairs 5 \\
        --series host --policy latency,balanced,economy --out results/verbs-perf/x
    python scripts/verbs_perf.py compare results/verbs-perf/x
    python scripts/verbs_perf.py profile --series host --policy economy --size 4096
    PYTHONPATH=TREE/src python scripts/verbs_perf.py python-cost --out cost.json

``--src`` (``run``) or each ``--arm`` names a source tree whose ``src/`` is imported, so
both arms of a comparison run this one harness on the same workload. Series, sizes,
repeats and warmup are fixed here and in docs/VERBS_PERF.md, section 1; they are not
options.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Self

LABEL = "CPU/MOCK, not a GPU or RDMA speedup"
KiB, MiB = 1024, 1024 * 1024
#: (ranks, [(bytes, warmup, timed calls)]) per series; docs/VERBS_PERF.md, section 1.
SERIES = {
    "host": (2, [(4 * KiB, 20, 500), (64 * KiB, 20, 300), (1 * MiB, 10, 100),
                 (16 * MiB, 3, 20)]),
    "ring": (3, [(64 * KiB, 20, 300), (1 * MiB, 10, 100)]),
    "observe": (2, [(4 * KiB, 20, 500), (1 * MiB, 10, 100)]),
    "pingpong": (2, [(8, 100, 2000)]),
    "idle": (2, []),
}
POLICIES = ("latency", "balanced", "economy")
IDLE_S = 2.0
COUNTED_CALLS = 50
TELEMETRY_COUNTS = ("copied_bytes", "control_sent", "completions", "notifications", "sends",
                    "recvs", "chunks_sent", "writes_posted", "batches_posted")
# Settings that would change the workload; a run refuses to start with any of them set.
FIXED_ENVS = ("GPUBRIDGE_VERBS_CHUNK_BYTES", "GPUBRIDGE_VERBS_RNDV_BYTES",
              "GPUBRIDGE_VERBS_SLOT_BYTES", "GPUBRIDGE_VERBS_CREDITS",
              "GPUBRIDGE_VERBS_SEND_SLOTS", "GPUBRIDGE_VERBS_PROGRESS_CPU",
              "GPUBRIDGE_VERBS_OBSERVE", "GPUBRIDGE_VERBS_UNEXPECTED_BYTES")


def maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def machine() -> dict:
    model = ""
    if sys.platform == "darwin":
        model = subprocess.run(["sysctl", "-n", "hw.model", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, check=False).stdout.split("\n")
        model = " / ".join(m for m in model if m)
    return {"model": model or platform.processor(), "machine": platform.machine(),
            "cpus": os.cpu_count(), "system": platform.platform(),
            "python": platform.python_version()}


def load() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip()


# ---- one rank -------------------------------------------------------------------------

class Counter:
    """Counts calls into the library's exported functions, and the time spent in them,
    while installed."""

    def __init__(self, lib) -> None:
        self.lib, self.counts, self.seconds, self.saved = lib, {}, {}, {}

    def __enter__(self) -> Self:
        # Every function the binding declared (ctypes caches each on the library object).
        for name in [n for n in vars(self.lib) if n.startswith("gb_verbs_")]:
            fn = getattr(self.lib, name)
            self.saved[name] = fn

            def wrapper(*args, _fn=fn, _name=name):
                self.counts[_name] = self.counts.get(_name, 0) + 1
                t0 = time.perf_counter()
                try:
                    return _fn(*args)
                finally:
                    self.seconds[_name] = self.seconds.get(_name, 0.0) + \
                        time.perf_counter() - t0

            setattr(self.lib, name, wrapper)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self.saved.items():
            setattr(self.lib, name, fn)


def _telemetry_counts(t: dict) -> dict:
    direct = t.get("direct") or {}
    reg = direct.get("registered") or {}
    out = {k: t.get(k, 0) for k in TELEMETRY_COUNTS}
    out["registrations"] = sum(int(v or 0) for v in reg.values())
    out["host_cache_misses"] = (direct.get("host_cache") or {}).get("misses", 0)
    out["host_cache_hits"] = (direct.get("host_cache") or {}).get("hits", 0)
    return out


def _delta(after: dict, before: dict, calls: int) -> dict:
    return {k: (after[k] - before[k]) / calls for k in after}


def _bridge_size(bridge, rank: int, world: int, nbytes: int, warmup: int, calls: int,
                 profile: bool) -> dict:
    import torch

    numel = nbytes // 4
    base = (torch.arange(numel, dtype=torch.float32) % 1021) + rank
    want = sum((torch.arange(numel, dtype=torch.float32) % 1021) + r for r in range(world))
    tensor = base.clone()
    mismatches = 0

    def one() -> tuple[float, float]:
        nonlocal mismatches
        tensor.copy_(base)
        c0, t0 = cpu_s(), time.perf_counter()
        bridge.all_reduce(tensor)
        t1, c1 = time.perf_counter(), cpu_s()
        mismatches += not torch.equal(tensor, want)
        return t1 - t0, c1 - c0

    for _ in range(warmup):
        one()
    tele0, rss0 = _telemetry_counts(bridge.telemetry()), maxrss_bytes()
    profiler = cProfile.Profile() if profile else None
    lat, cpu = [], []
    loop_cpu0, loop_wall0 = cpu_s(), time.perf_counter()
    for _ in range(calls):
        if profiler:
            profiler.enable()
        wall, used = one()
        if profiler:
            profiler.disable()
        lat.append(wall)
        cpu.append(used)
    loop_cpu, loop_wall = cpu_s() - loop_cpu0, time.perf_counter() - loop_wall0
    tele1, rss1 = _telemetry_counts(bridge.telemetry()), maxrss_bytes()
    with Counter(bridge.worker.endpoint._lib) as counter:
        counted = [one()[0] for _ in range(COUNTED_CALLS)]
    out = {"bytes": nbytes, "warmup": warmup, "calls": calls, "mismatches": mismatches,
           "latency_s": lat, "p50_us": 1e6 * statistics.median(lat),
           "p99_us": 1e6 * quantile(lat, 0.99),
           "throughput_mb_s": nbytes / statistics.median(lat) / 1e6,
           "cpu_per_op_us": 1e6 * sum(cpu) / calls,
           "loop_cpu_per_wall": loop_cpu / loop_wall,
           "maxrss_bytes": rss1, "rss_growth_bytes": rss1 - rss0,
           "per_call": _delta(tele1, tele0, calls),
           "crossings_per_call": {k: v / COUNTED_CALLS for k, v in sorted(counter.counts.items())},
           # The counted pass (wrapped, so slightly slower): where a call's time goes.
           "breakdown_us": {"call": 1e6 * sum(counted) / COUNTED_CALLS,
                            **{k: 1e6 * v / COUNTED_CALLS
                               for k, v in sorted(counter.seconds.items())}}}
    out["crossings_total_per_call"] = sum(v for k, v in out["crossings_per_call"].items()
                                          if k != "gb_verbs_telemetry")
    if profiler:
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).sort_stats("tottime").print_stats(25)
        out["profile"] = text.getvalue()
    return out


def _pingpong(rank: int, warmup: int, calls: int) -> dict:
    import torch
    import torch.distributed as dist

    from gpubridge import verbs
    from gpubridge.verbs.transport import endpoint_settings

    store = dist.distributed_c10d._get_default_store()
    peer = 1 - rank
    ep = verbs.Endpoint(2, rank, endpoint_settings())
    try:
        store.set(f"perf/addr/{rank}", ep.local_info())
        store.wait([f"perf/addr/{peer}"])
        ep.connect(peer, store.get(f"perf/addr/{peer}").decode())
        store.set(f"perf/ready/{rank}", "1")
        store.wait([f"perf/ready/{peer}"])
        buf = torch.zeros(8, dtype=torch.uint8)
        lat, cpu = [], []
        for i in range(warmup + calls):
            c0, t0 = cpu_s(), time.perf_counter()
            deadline = time.monotonic() + 60
            if rank == 0:
                ep.wait([ep.send(1, buf.data_ptr(), 8, 2 * i)], deadline)
                ep.wait([ep.recv(1, buf.data_ptr(), 8, 2 * i + 1)], deadline)
            else:
                ep.wait([ep.recv(0, buf.data_ptr(), 8, 2 * i)], deadline)
                ep.wait([ep.send(0, buf.data_ptr(), 8, 2 * i + 1)], deadline)
            if i >= warmup:
                lat.append(time.perf_counter() - t0)
                cpu.append(cpu_s() - c0)
        return {"bytes": 8, "warmup": warmup, "calls": calls, "mismatches": 0,
                "latency_s": lat, "p50_us": 1e6 * statistics.median(lat),
                "p99_us": 1e6 * quantile(lat, 0.99), "throughput_mb_s": None,
                "cpu_per_op_us": 1e6 * sum(cpu) / calls, "maxrss_bytes": maxrss_bytes(),
                "rss_growth_bytes": None}
    finally:
        ep.close(timeout=10.0)


def _rank_main(rank: int, world: int, series: str, init: str, out_dir: str,
               profile_size: int | None) -> None:
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    result: dict = {"rank": rank, "sizes": []}
    try:
        if series == "pingpong":
            _, plan = SERIES[series]
            result["sizes"].append(_pingpong(rank, plan[0][1], plan[0][2]))
        else:
            from gpubridge.verbs.transport import VerbsTransport

            bridge = VerbsTransport.create(list(range(world)), timeout=timedelta(seconds=120))
            try:
                result["telemetry"] = {k: v for k, v in bridge.telemetry().items()
                                       if k in ("provider", "mock", "progress", "library",
                                                "chunk_bytes", "rndv_bytes", "slot_bytes",
                                                "credits")}
                dist.barrier()
                c0, t0 = cpu_s(), time.perf_counter()
                time.sleep(IDLE_S)
                result["idle_cpu_per_s"] = (cpu_s() - c0) / (time.perf_counter() - t0)
                for nbytes, warmup, calls in SERIES[series][1]:
                    if profile_size is not None and nbytes != profile_size:
                        continue
                    dist.barrier()
                    result["sizes"].append(_bridge_size(
                        bridge, rank, world, nbytes, warmup, calls,
                        profile=profile_size is not None and rank == 0))
                dist.barrier()
            finally:
                bridge.close()
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def run_child(args) -> int:
    """Runs one series in fresh rank processes; this process imports gpubridge from the
    tree its PYTHONPATH names (set by :func:`run`)."""
    import torch.multiprocessing as mp

    world = SERIES[args.series][0]
    fabric = tempfile.mkdtemp(prefix="gbvp-", dir="/tmp")
    work = tempfile.mkdtemp(prefix="gbvp-out-")
    os.environ["GPUBRIDGE_VERBS_TEST_FABRIC"] = fabric
    os.environ["GPUBRIDGE_VERBS_PROGRESS"] = args.policy
    if args.series == "observe":
        os.environ["GPUBRIDGE_VERBS_OBSERVE"] = str(Path(work, "obs-{rank}.jsonl"))
    try:
        mp.spawn(_rank_main, args=(world, args.series, str(Path(work, "init")), work,
                                   args.profile_size), nprocs=world)
        ranks = [json.loads(Path(work, f"rank{r}.json").read_text()) for r in range(world)]
        observed = 0
        for f in Path(work).glob("obs-*.jsonl"):
            observed += sum(1 for _ in f.open())
        Path(args.child_out).write_text(json.dumps({"ranks": ranks, "observations": observed}))
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    return 0


def run(series: str, policy: str, src: Path, out: Path | None, *, arm: str = "",
        profile_size: int | None = None) -> dict:
    """One series under one policy, from source tree ``src``; the record (written to
    ``out`` when given)."""
    for env in FIXED_ENVS:
        if os.environ.get(env, "").strip():
            raise SystemExit(f"{env} is set; unset it so every arm runs the same workload")
    src = src.resolve()
    rev = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, check=False,
                         text=True).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "src"],
                                capture_output=True, text=True, check=False).stdout.strip()) if rev else None
    env = {**os.environ, "PYTHONPATH": str(src / "src")}
    load0 = load()
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        command = [sys.executable, __file__, "_child", "--series", series, "--policy", policy,
                   "--child-out", tmp.name]
        if profile_size is not None:
            command += ["--profile-size", str(profile_size)]
        started = time.time()
        subprocess.run(command, env=env, check=True)
        child = json.loads(Path(tmp.name).read_text())
    record = {"label": LABEL, "series": series, "policy": policy, "arm": arm,
              "src": str(src), "src_rev": rev, "src_dirty": dirty,
              "harness": str(Path(__file__).resolve()), "started": started,
              "seconds": time.time() - started, "load_before": load0, "load_after": load(),
              "machine": machine(), "ranks_n": SERIES[series][0], **child}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1))
    return record


def ab(args) -> int:
    """Interleaved arms: A B B A A B ... per series and policy, ``--pairs`` pairs."""
    arms = [a.split("=", 1) for a in args.arm]
    if len(arms) != 2:
        raise SystemExit("ab needs exactly two --arm NAME=TREE")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for series in args.series.split(","):
        for policy in args.policy.split(","):
            for pair in range(args.pairs):
                order = arms if pair % 2 == 0 else arms[::-1]
                for name, tree in order:
                    path = out / f"{series}-{policy}-pair{pair}-{name}.json"
                    if path.exists():
                        continue  # resumable: a finished run is never redone
                    record = run(series, policy, Path(tree), path, arm=name)
                    print(f"{series} {policy} pair {pair} {name}: "
                          + ", ".join(f"{s['bytes']}B p50 {s['p50_us']:.1f}us"
                                      for s in record["ranks"][0]["sizes"]), flush=True)
    return 0


# ---- Python cost alone -----------------------------------------------------------------

class Instant:
    """An endpoint whose operations complete at once: what is left of a call is the
    bridge's Python (and torch's reduction), with no fabric, no native code and no peer."""

    failed = None
    _handle = 1

    def __init__(self) -> None:
        self.ops = 0

    def rndv_bytes(self) -> int:
        return 256 * KiB

    def _op(self, *_args) -> int:
        self.ops += 1
        return self.ops

    send = recv = send_reg = recv_reg = _op

    def wait(self, ids, deadline) -> None:
        pass

    def register_host(self, pointer, nbytes, owner):
        from gpubridge.verbs import Registration

        return Registration(owner, pointer, nbytes, "host")

    def release(self, registration) -> None:
        pass

    def invalidate_host(self, owner) -> int:
        return 0

    def telemetry(self) -> dict:
        # What the library reports, at its real size, so a caller that reads it per call
        # pays for parsing it here too.
        return json.loads(self._telemetry)

    _telemetry = json.dumps({"mock": True, "provider": "INSTANT (no fabric)",
                             "peers": [{"index": i, "state": "connected"} for i in range(2)],
                             **{f"counter_{i}": i for i in range(60)}})


def python_cost(args) -> int:
    """Python per 2-leader host all_reduce against :class:`Instant` (rank 0's side only),
    with and without an observation emitter; median of 5 runs of 2000 calls."""
    import torch

    from gpubridge.verbs import observe
    from gpubridge.verbs.gpu import Router
    from gpubridge.verbs.transport import VerbsTransport

    torch.set_num_threads(1)
    out = {"label": LABEL + "; no fabric: Python and torch only", "machine": machine(),
           "load": load(), "src": str(Path(sys.modules["gpubridge"].__file__).parents[1]),
           "runs": []}
    work = tempfile.mkdtemp(prefix="gbvp-py-")
    try:
        for observed in (False, True):
            for nbytes in (4 * KiB, 1 * MiB):
                emitter = observe.Emitter(str(Path(work, "obs.jsonl"))) if observed else None
                tele = {"device": "mock0", "port": 1, "gid_index": 0,
                        "link_layer": "InfiniBand"}
                base = ({"schema": observe.SCHEMA, "host": "h", "job": "j", "rank": 0,
                         "bridge": "verbs", "nic": observe.nic_object(tele)} if observed else {})
                bridge = VerbsTransport(Router(Instant()), [0, 1], 0, timedelta(seconds=60),
                                        4 * MiB, emitter=emitter, base_record=base,
                                        peer_nics={1: {"device": "mock0", "port": 1,
                                                       "link_layer": "InfiniBand"}})
                tensor = torch.ones(nbytes // 4)
                for _ in range(200):
                    bridge.all_reduce(tensor)
                runs = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(2000):
                        bridge.all_reduce(tensor)
                    runs.append(1e6 * (time.perf_counter() - t0) / 2000)
                if emitter:
                    emitter.close()
                out["runs"].append({"bytes": nbytes, "observe": observed, "us_per_call": runs,
                                    "median_us": statistics.median(runs)})
                print(f"{nbytes} B observe={observed}: {statistics.median(runs):.2f} us/call "
                      f"(runs {min(runs):.2f}-{max(runs):.2f})", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# ---- summaries ------------------------------------------------------------------------

METRICS = ("p50_us", "p99_us", "throughput_mb_s", "cpu_per_op_us", "copied_bytes",
           "rss_growth_bytes", "maxrss_bytes")
HIGHER_IS_BETTER = {"throughput_mb_s"}
#: Deterministic counts: any increase is a regression. (Copies are not here: an eager
#: message that arrives before its receive is posted is copied once more, so copied
#: bytes depend on timing and are judged like the timed metrics.)
COUNTED = ("control_sent", "registrations", "crossings_total_per_call")


def _values(records: list[dict], size_index: int, metric: str) -> list[float]:
    out = []
    for r in records:
        s = r["ranks"][0]["sizes"][size_index]
        if metric == "crossings_total_per_call" and "crossings_per_call" in s:
            # Every crossing, telemetry reads included (the raw record's own total left
            # them out).
            value = sum(s["crossings_per_call"].values())
        else:
            value = s.get(metric, (s.get("per_call") or {}).get(metric))
        if value is not None:
            out.append(float(value))
    return out


def compare(args) -> int:
    """Per series, policy, size and metric: medians, spread, pairs and the rule-2/3
    verdict of docs/VERBS_PERF.md (before = the first arm name in sorted file order
    unless --before is given)."""
    runs: dict = {}
    for path in sorted(Path(args.dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        r = json.loads(path.read_text())
        pair = int(path.stem.split("-pair")[1].split("-")[0])
        runs.setdefault((r["series"], r["policy"]), {}).setdefault(r["arm"], {})[pair] = r
    rows = []
    for (series, policy), arms in sorted(runs.items()):
        names = sorted(arms)
        before = args.before if args.before in names else names[0]
        after = next(n for n in names if n != before)
        pairs = sorted(set(arms[before]) & set(arms[after]))
        b = [arms[before][p] for p in pairs]
        a = [arms[after][p] for p in pairs]
        if not b or not b[0]["ranks"][0]["sizes"]:
            idle_b = [r["ranks"][0].get("idle_cpu_per_s") for r in b]
            idle_a = [r["ranks"][0].get("idle_cpu_per_s") for r in a]
            rows.append(_row(series, policy, None, "idle_cpu_per_s", idle_b, idle_a))
            continue
        for i, size in enumerate(b[0]["ranks"][0]["sizes"]):
            for metric in METRICS + COUNTED:
                rows.append(_row(series, policy, size["bytes"], metric,
                                 _values(b, i, metric), _values(a, i, metric)))
            rows.append(_row(series, policy, size["bytes"], "mismatches",
                             _values(b, i, "mismatches"), _values(a, i, "mismatches")))
        idle_b = [r["ranks"][0].get("idle_cpu_per_s") for r in b if "idle_cpu_per_s" in
                  r["ranks"][0]]
        idle_a = [r["ranks"][0].get("idle_cpu_per_s") for r in a if "idle_cpu_per_s" in
                  r["ranks"][0]]
        if idle_b and idle_a:
            rows.append(_row(series, policy, None, "idle_cpu_per_s", idle_b, idle_a))
    summary = {"label": LABEL, "dir": str(args.dir), "rows": [r for r in rows if r]}
    Path(args.dir, "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"# {LABEL}")
    print("| series | policy | bytes | metric | before | after | change | spread | "
          "pairs better | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in summary["rows"]:
        print(f"| {r['series']} | {r['policy']} | {r['bytes'] or ''} | {r['metric']} | "
              f"{_fmt(r['before'])} | {_fmt(r['after'])} | {_pct(r['change'])} | "
              f"{_pct(r['spread'])} | {r['pairs_better']}/{r['pairs']} | {r['verdict']} |")
    return 0


def _row(series, policy, nbytes, metric, before, after) -> dict | None:
    if not before or not after:
        return None
    mb, ma = statistics.median(before), statistics.median(after)

    def spread(v):
        m = statistics.median(v)
        return (max(v) - min(v)) / abs(m) if m else (0.0 if max(v) == min(v) else float("inf"))

    s = max(spread(before), spread(after))
    change = (ma - mb) / abs(mb) if mb else (0.0 if ma == mb else float("inf"))
    up = metric in HIGHER_IS_BETTER
    better = sum((x > y) if up else (x < y) for y, x in zip(before, after, strict=False))
    worse_by = -change if up else change
    if metric == "mismatches":
        verdict = "FAIL (wrong result)" if ma or mb else "ok"
    elif metric in COUNTED:
        verdict = "regression" if worse_by > 0 else ("better" if worse_by < 0 else "same")
    elif metric in ("maxrss_bytes", "rss_growth_bytes") and abs(ma - mb) < 2 * MiB:
        verdict = "no change (< 2 MiB)"
    elif worse_by < -s and better >= max(1, len(before) - 1):
        verdict = "win"
    elif worse_by > s:
        verdict = "REGRESSION"
    else:
        verdict = "no change (inside spread)"
    return {"series": series, "policy": policy, "bytes": nbytes, "metric": metric,
            "before": mb, "after": ma, "change": change, "spread": s,
            "pairs_better": better, "pairs": min(len(before), len(after)), "verdict": verdict}


def _fmt(v: float) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _pct(v: float) -> str:
    return f"{100 * v:+.1f}%" if v not in (float("inf"),) else "inf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--series", choices=SERIES, required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ab")
    p.add_argument("--arm", action="append", required=True, help="NAME=SOURCE_TREE, twice")
    p.add_argument("--series", required=True, help="comma-separated")
    p.add_argument("--policy", default=",".join(POLICIES), help="comma-separated")
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("dir")
    p.add_argument("--before", default="before")
    p = sub.add_parser("profile")
    p.add_argument("--series", choices=SERIES, required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path)
    p = sub.add_parser("python-cost", help="Python per call against an instant endpoint")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("_child")
    p.add_argument("--series", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--child-out", required=True)
    p.add_argument("--profile-size", type=int, default=None)
    args = parser.parse_args()
    if args.cmd == "_child":
        return run_child(args)
    if args.cmd == "run":
        record = run(args.series, args.policy, args.src, args.out)
        print(json.dumps({s["bytes"]: [round(s["p50_us"], 1), round(s["p99_us"], 1)]
                          for s in record["ranks"][0]["sizes"]}))
        return 0
    if args.cmd == "ab":
        return ab(args)
    if args.cmd == "compare":
        return compare(args)
    if args.cmd == "python-cost":
        return python_cost(args)
    record = run(args.series, args.policy, args.src, args.out, profile_size=args.size)
    size = record["ranks"][0]["sizes"][0]
    print(f"# {LABEL}\n{args.series} {args.policy} {size['bytes']} B: p50 "
          f"{size['p50_us']:.1f} us, crossings/call {size['crossings_per_call']}, "
          f"per call {size['per_call']}")
    print(size["profile"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
