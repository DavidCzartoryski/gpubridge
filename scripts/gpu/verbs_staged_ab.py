"""Staged verbs all_reduce on one node: one arm of an A/B, every raw sample kept.

docs/verbs-perf-20261019.md, "M1", is the plan this runs: its arms, rotation,
acceptance rules and owner commands. ``run`` is one arm in one job, launched by
torchrun; ``compare`` reads the runs of two arms and applies the predeclared rules.

    GPUBRIDGE_SPLIT_TEST=half GPUBRIDGE_VERBS_GPU_PATH=staged \\
        torchrun --nproc-per-node=4 scripts/gpu/verbs_staged_ab.py run \\
        --op gpubridge --bridge verbs --arm before --repeat 0 --out OUT
    python scripts/gpu/verbs_staged_ab.py compare OUT --before before --after after \\
        --before-repeats 3 --after-repeats 3 [--differ KEY ...] [--wire]
    python scripts/gpu/verbs_staged_ab.py trigger OUT --arm P0x2 --repeats 3 \\
        --compare compare-P0-D3.json

Per size of the fixed table (fp32 integer values, so every sum is exact), ``run``
does untimed warmup calls, then timed calls. Each timed call starts from the same
input after a barrier and is timed with CUDA events on the current stream (on CPU,
the host timer) and on the host; its result is then checked against the exact sum,
outside the timed region. Each rank keeps every sample, its process CPU around each
call (all threads: the bridge's progress threads too) and over the whole timed
region, its max RSS, and on a bridge leader the verbs telemetry's change over the
timed calls (sends, write windows and their nanoseconds, copies, completions,
staged phase times), and the last call's path and, staged, whether it overlapped its
copies (GPUBRIDGE_VERBS_STAGED_OVERLAP). Rank 0 writes ``OUT/<arm>-r<repeat>.json``
with every rank's data, the run's full wall time and its provenance (git commit,
imported sources' hash, the verbs library's file hash, allowlisted environment).

``--op native`` times torch.distributed.all_reduce on one group of every rank on
the island backend (NCCL), in the same job layout, as the baseline. It refuses to
run where that is not possible (mixed vendors, shared GPUs) rather than time
something else.

``compare`` judges only a complete, matched pair of arms. Each arm must have exactly
its predeclared repeats 0 to N-1, every one a whole record (the table's sizes, every
rank, every call's samples); with ``--wire``, every island leader's verbs telemetry
with every counted message. Every run of both arms is checked for wrong results, the
unpaired repeats of the longer arm too. The arms' settings, topology, table, code
and host (``workload``) must be equal, repeat to repeat and arm to arm, except the
keys named by ``--differ`` (a key, or a group: ``--differ code`` allows every
``code.`` key). Exit codes: 0 judged and no failure; 1 a wrong result or a
REGRESSION; 3 UNSUPPORTED (``--wire`` with no verbs telemetry in an arm); 4
INCOMPLETE (a repeat missing, extra, truncated or unreadable, or telemetry missing); 5
REFUSED (the arms are not the predeclared comparison). Every outcome writes
``compare-<before>-<after>.json`` with its status and reasons.

``trigger`` decides the plan's first next-change row (section 5): for each leader of
the pure bridge arm (1+1 on verbs) at 25 and 150 MiB, each repeat's
``(staged_to_host_s / staged_calls) / p50``, their median, and whether it is at least
0.30; then whether the named compare JSON, judged, says ``win`` at either size. It
prints to stdout only: the shares, the compare's status, the reasons, and last a line
that is exactly ``TRIGGER MET`` or ``TRIGGER NOT MET`` (exit 0) or ``INCOMPLETE``
(exit 4). INCOMPLETE decides nothing: a repeat missing, extra, unreadable or not whole,
an arm-named file with no record of the arm, repeats that differ in workload or are not
1+1, staged phases not from exactly the recorded leaders or not covering every timed
call, other than one share per repeat for each leader and size, or a compare JSON
missing, of other arms than its name, not judged (only ``judged`` and ``REGRESSION``
are) or lacking either size.

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
#: compare's exit codes besides 0 and 1 (module docs).
UNSUPPORTED, INCOMPLETE, REFUSED = 3, 4, 5
#: Section 5's copy overlap trigger: the sizes it looks at and the copy out's share.
TRIGGER_SIZES, TRIGGER_SHARE = (25 * 2**20, 150 * 2**20), 0.30
#: The compare statuses whose rows were judged: the trigger reads only these.
JUDGED = ("judged", "REGRESSION")
#: Recorded environment variables that name the code under test (under ``code.``)...
ENV_CODE = ("GPUBRIDGE_VERBS_LIB", "GPUBRIDGE_VERBS_LIB_DIR", "PYTHONPATH")
#: ... and the settings compared between arms (under ``env.``), by prefix or name. The
#: rest of the recorded environment (ranks, ports, Slurm ids) differs from run to run.
ENV_SETTINGS = ("GPUBRIDGE_", "NCCL_", "RCCL_", "GLOO_", "UCX_", "TORCH_",
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "OMP_NUM_THREADS")


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


def verbs_counts(bridge) -> dict[str, float | None] | None:
    """The recorded verbs telemetry counts of this rank's bridge, or None (no verbs
    bridge here). A count the telemetry lacks is None, never 0."""
    if bridge is None or getattr(bridge, "name", None) != "verbs":
        return None
    t = bridge.telemetry()
    out = {k: t.get(k) for k in COUNTS}
    out.update({k: (t.get("direct") or {}).get(k) for k in DIRECT_COUNTS})
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
                                              "chunk_bytes", "gpu_path_setting",
                                              "staged_overlap_setting")}
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
    delta = None if before is None else {
        k: None if after[k] is None or before.get(k) is None else after[k] - before[k]
        for k in after}
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
        last = topology.bridge.telemetry() if verbs_counts(topology.bridge) is not None else {}
        mine["last_path"] = last.get("path")
        mine["last_staging"] = (last.get("last_gpu") or {}).get("staging")
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
    leaders' summed verbs deltas (None where a leader's telemetry lacks the count)."""
    out = {}
    for ranks in record["sizes"]:
        slowest = [max(r["timed_s"][i] for r in ranks) for i in range(ranks[0]["calls"])]
        p50 = statistics.median(slowest)
        deltas = [r["verbs_delta"] for r in ranks if r["verbs_delta"] is not None]
        counts = {k: None if any(d.get(k) is None for d in deltas)
                  else sum(d[k] for d in deltas)
                  for k in dict.fromkeys(k for d in deltas for k in d)}
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


def leaders(record: dict) -> int:
    return len(record["run"]["islands"])


def problems(record: dict, wire: bool) -> list[str]:
    """Why this record is not a whole run of its table (with ``wire``: or lacks a
    leader's counted messages); empty when it is."""
    try:
        workload(record)
        world, out = record["run"]["world_size"], []
        table = [tuple(row) for row in TABLES[record["table"]]]
        got = [(ranks[0]["bytes"], ranks[0]["warmup"], ranks[0]["calls"])
               for ranks in record["sizes"]]
        if got != table:
            out.append(f"sizes {got} are not table {record['table']!r}'s {table}")
        for ranks in record["sizes"]:
            nbytes, calls = ranks[0]["bytes"], ranks[0]["calls"]
            if len(ranks) != world:
                out.append(f"{nbytes} bytes: {len(ranks)} ranks of {world}")
            for r in ranks:
                counts = {len(r["timed_s"]), len(r["host_s"]), len(r["cpu_s"])}
                if counts != {calls} or (r["bytes"], r["calls"]) != (nbytes, calls):
                    out.append(f"{nbytes} bytes, rank {r['rank']}: samples {sorted(counts)} "
                               f"for {calls} calls")
                if r["payload_bytes"] != nbytes or type(r["mismatches"]) is not int:
                    out.append(f"{nbytes} bytes, rank {r['rank']}: payload "
                               f"{r['payload_bytes']}, mismatches {r['mismatches']!r}")
            if wire:
                deltas = [r["verbs_delta"] for r in ranks if r["verbs_delta"] is not None]
                if len(deltas) != leaders(record):
                    out.append(f"{nbytes} bytes: verbs telemetry from {len(deltas)} of "
                               f"{leaders(record)} leaders")
                for d in deltas:
                    missing = [k for k in WIRE if not isinstance(d.get(k), int | float)]
                    if missing:
                        out.append(f"{nbytes} bytes: a leader's telemetry lacks {missing}")
        return out
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        return [f"malformed record ({type(e).__name__}: {e})"]


def rank_entries(record: dict):
    """Every rank's entry of every size of a record, whole or not, skipping what isn't one."""
    sizes = record.get("sizes")
    for ranks in sizes if isinstance(sizes, list) else []:
        yield from (r for r in ranks if isinstance(r, dict)) if isinstance(ranks, list) else ()


def wrong_results(record: dict) -> int:
    """Mismatched calls over every rank and size of a record, whole or not. Only integer
    counts are summed: any other value is a malformed record, which problems() reports."""
    return sum(r["mismatches"] for r in rank_entries(record)
               if type(r.get("mismatches")) is int)


def read_records(directory, arms) -> tuple[list[tuple[str, dict]], list[str]]:
    """The run records in a directory, and why the files named for one of ``arms``
    (``<arm>-r<k>.json``) that hold no record could not be read."""
    records, reasons = [], []
    for path in sorted(Path(directory).glob("*-r*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            record, why = None, f"{type(e).__name__}: {e}"
        else:
            why = f"not a JSON object but {type(record).__name__}"
        if isinstance(record, dict):
            records.append((path.name, record))
        elif any(path.name.startswith(f"{arm}-r") for arm in arms):
            reasons.append(f"{path.name}: unreadable ({why})")
    return records, reasons


def workload(record: dict) -> dict[str, Any]:
    """What a comparison holds equal, flattened to ``group.key``: the operation, bridge,
    policy and table, the sizes, the topology, the verbs settings, the code (commit,
    source and library hashes) and the host. JSON values, as the record has them."""
    run, provs = record["run"], record["provenance"]
    env = provs[0]["environment"]
    verbs = [p for p in provs if "verbs" in p]
    out: dict[str, Any] = {
        "op": record["op"], "bridge.requested": record["bridge"],
        "bridge.rank0": run["bridge"], "policy.requested": record["policy"],
        "policy.resolved": run["policy"], "table": record["table"],
        "sizes": [[r["bytes"], r["payload_bytes"], r["warmup"], r["calls"], r["timer"]]
                  for r in (ranks[0] for ranks in record["sizes"])],
        "topology.kind": run["kind"], "topology.split_test": run["split_test"],
        "topology.world_size": run["world_size"], "topology.islands": run["islands"],
        "topology.pinned_staging": run["pinned_staging"],
        "code.git_commit": env["git_commit"], "code.git_dirty": env["git_dirty"],
        "code.gpubridge": env["gpubridge"], "code.import": env["import"],
        "code.gpubridge_py_sha256": [p["gpubridge_py_sha256"] for p in provs],
        "code.verbs.library_sha256": [p["verbs_library_sha256"] for p in verbs],
        "code.verbs.crate_sha256": [p["verbs_crate_sha256"] for p in verbs],
        "verbs.leaders": len(verbs)}
    for key in ("hostname", "platform", "python", "torch", "torch_cuda", "torch_hip",
                "nccl_version", "driver_version", "devices"):
        out[f"host.{key}"] = env.get(key)
    for key in sorted({k for p in verbs for k in p["verbs"]}):
        name = "code.verbs.library" if key == "library" else f"verbs.{key}"
        out[name] = [p["verbs"].get(key) for p in verbs]
    for name, value in env["env"].items():
        if name in ENV_CODE:
            out[f"code.env.{name}"] = value
        elif name.startswith(ENV_SETTINGS):
            out[f"env.{name}"] = value
    return out


def differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def allowed(key: str, differ: list[str]) -> bool:
    return any(key == d or key.startswith(d + ".") for d in differ)


def compare(args) -> int:
    """The predeclared rules (docs/verbs-perf-20261019.md, "Acceptance"), per size, on a
    complete and matched pair of arms only (module docs)."""
    declared = {args.before: args.before_repeats, args.after: args.after_repeats}
    records: dict[str, dict[int | str, tuple[str, dict]]] = {arm: {} for arm in declared}
    status, wrong = None, []
    found, reasons = read_records(args.dir, declared)
    for name, record in found:
        arm, repeat = record.get("arm"), record.get("repeat")
        if isinstance(arm, str) and arm in records:
            key = repeat if type(repeat) is int else repr(repeat)
            if key in records[arm]:
                reasons.append(f"{name}: a second record of {arm} repeat {key}")
            records[arm][key] = (name, record)
        elif any(name.startswith(f"{a}-r") for a in declared):
            reasons.append(f"{name}: unreadable (no record of a compared arm: arm {arm!r})")
    for arm, n in declared.items():
        got, want = set(records[arm]), set(range(n))
        if want - got:
            reasons.append(f"{arm}: repeats {sorted(want - got)} of the predeclared {n} "
                           "missing (a failed run counts as failed, not as absent)")
        if got - want:
            reasons.append(f"{arm}: repeats {sorted(got - want, key=str)} outside the "
                           f"predeclared {n}")
        for name, record in records[arm].values():
            reasons += [f"{name}: {p}" for p in problems(record, args.wire)]
            if wrong_results(record):
                wrong.append(f"{name}: {wrong_results(record)} wrong results")
        if args.wire and records[arm] and not any(
                r.get("verbs_delta") is not None
                for _, record in records[arm].values() for r in rank_entries(record)):
            status = "UNSUPPORTED"
            reasons.append(f"{arm}: no verbs telemetry, which --wire compares")
    if status is None and reasons:
        status = "INCOMPLETE"
    if status is None:
        works = {arm: {k: workload(rec) for k, (_, rec) in runs.items()}
                 for arm, runs in records.items()}
        for arm, by_repeat in works.items():
            first = by_repeat[0]
            for k, w in sorted(by_repeat.items()):
                if differ := differences(first, w):
                    reasons.append(f"{arm} repeat {k} differs from repeat 0 in {differ}")
        unexpected = [k for k in differences(works[args.before][0], works[args.after][0])
                      if not allowed(k, args.differ)]
        for k in unexpected:
            reasons.append(f"{k}: {works[args.before][0].get(k)!r} in {args.before}, "
                           f"{works[args.after][0].get(k)!r} in {args.after}, "
                           "not declared with --differ")
        if reasons:
            status = "REFUSED"
    rows: list[dict[str, Any]] = []
    pairs = list(range(min(declared.values())))
    if status is None:
        try:
            runs = {arm: {k: summarize(rec) for k, (_, rec) in recs.items()}
                    for arm, recs in records.items()}
            rows = judge(runs[args.before], runs[args.after], pairs, args.wire)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError,
                ArithmeticError) as e:  # values no whole record holds
            status, rows = "INCOMPLETE", []
            reasons.append(f"malformed record ({type(e).__name__}: {e})")
    if wrong:
        status = "FAIL (wrong result)"
        reasons = wrong + reasons
    elif status is None:
        status = ("REGRESSION" if any(r["verdict"] == "REGRESSION" for r in rows)
                  else "judged")
    Path(args.dir, f"compare-{args.before}-{args.after}.json").write_text(
        json.dumps({"before": args.before, "after": args.after, "declared": declared,
                    "differ": args.differ, "wire": args.wire, "status": status,
                    "reasons": reasons, "pairs": pairs, "rows": rows}, indent=1))
    if not rows:
        print(f"{status}: {args.before} against {args.after} not judged", file=sys.stderr)
        for reason in reasons:
            print(f"  {reason}", file=sys.stderr)
        return {"UNSUPPORTED": UNSUPPORTED, "INCOMPLETE": INCOMPLETE,
                "REFUSED": REFUSED}.get(status, 1)
    print("| bytes | p50 before | p50 after | change | spread | pairs better | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        p = r["p50_s"]
        print(f"| {r['bytes']} | {p['before'] * 1e6:.1f} us | {p['after'] * 1e6:.1f} us | "
              f"{p['change']:+.1%} | {p['spread']:.1%} | {p['pairs_better']}/{r['pairs']} | "
              f"{r['verdict']}{' ' + str(r['regressions']) if r.get('regressions') else ''} |")
    for reason in reasons:
        print(reason, file=sys.stderr)
    return 0 if status == "judged" else 1


def trigger(args) -> int:
    """Section 5's copy overlap row on the pure bridge arm and the named compare (module
    docs). Everything goes to stdout, the reasons before the verdict, so the verdict is
    the last line of the file the owner command writes."""
    found, reasons = read_records(args.dir, [args.arm])
    runs: dict[int | str, tuple[str, dict]] = {}
    for name, record in found:
        if record.get("arm") == args.arm:
            repeat = record.get("repeat")
            key = repeat if type(repeat) is int else repr(repeat)
            if key in runs:
                reasons.append(f"{name}: a second record of {args.arm} repeat {key}")
            runs[key] = (name, record)
        elif name.startswith(f"{args.arm}-r"):
            reasons.append(f"{name}: unreadable (no record of {args.arm}: arm "
                           f"{record.get('arm')!r})")
    if set(runs) != set(range(args.repeats)):
        reasons.append(f"{args.arm}: repeats {sorted(runs, key=str)}, not the predeclared "
                       f"0 to {args.repeats - 1}")
    shares: dict[tuple[int, int], list[float]] = {}
    first: tuple[str, dict[str, Any], list[int]] | None = None  # name, workload, leaders
    for key in sorted(runs, key=str):
        name, record = runs[key]
        if bad := problems(record, False):
            reasons += [f"{name}: {p}" for p in bad]
            continue
        try:
            islands = record["run"]["islands"]
            if (record["op"], record["bridge"], record["run"]["world_size"],
                    [len(i["ranks"]) for i in islands]) != ("gpubridge", "verbs", 2, [1, 1]):
                reasons.append(f"{name}: not the pure bridge (op {record['op']!r}, bridge "
                               f"{record['bridge']!r}, islands "
                               f"{[i['ranks'] for i in islands]}), which is 1+1 on verbs")
                continue
            want = sorted(i["leader"] for i in islands)
            if first is None:
                first = (name, workload(record), want)
            elif differ := differences(first[1], workload(record)):
                reasons.append(f"{name} differs from {first[0]} in {differ}")
            p50 = summarize(record)
            for nbytes in TRIGGER_SIZES:
                if nbytes not in p50:
                    reasons.append(f"{name}: no {nbytes}-byte row in table {record['table']!r}")
                    continue
                ranks, = (ranks for ranks in record["sizes"] if ranks[0]["bytes"] == nbytes)
                leads = [r for r in ranks if r["verbs_delta"] is not None]
                if sorted(r["rank"] for r in leads) != want:
                    reasons.append(f"{name}: {nbytes} bytes: staged phases from ranks "
                                   f"{sorted(r['rank'] for r in leads)}, the leaders are {want}")
                    continue
                for r in leads:
                    calls, to_host = (r["verbs_delta"].get(k)
                                      for k in ("staged_calls", "staged_to_host_s"))
                    if type(calls) is not int or calls != r["calls"] or \
                            not isinstance(to_host, int | float):
                        reasons.append(f"{name}: {nbytes} bytes, rank {r['rank']}: staged "
                                       f"phases missing or partial (staged_calls {calls!r} "
                                       f"of {r['calls']} calls, staged_to_host_s {to_host!r})")
                        continue
                    share = to_host / calls / p50[nbytes]["p50_s"]
                    shares.setdefault((nbytes, r["rank"]), []).append(share)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError,
                ArithmeticError) as e:  # values no whole record holds
            reasons.append(f"{name}: malformed record ({type(e).__name__}: {e})")
    if first is not None:
        for nbytes in TRIGGER_SIZES:
            for rank in first[2]:
                if len(shares.get((nbytes, rank), [])) != args.repeats:
                    reasons.append(f"{args.arm} {nbytes} bytes, leader rank {rank}: "
                                   f"{len(shares.get((nbytes, rank), []))} shares, not one "
                                   f"per predeclared repeat ({args.repeats})")
    for (nbytes, rank), values in sorted(shares.items()):
        print(f"{args.arm} {nbytes} bytes, leader rank {rank}: (staged_to_host_s / "
              f"staged_calls) / p50 per repeat {' '.join(f'{v:.3f}' for v in values)}, "
              f"median {statistics.median(values):.3f}")
    wins: list[int] = []
    try:
        verdict = json.loads(Path(args.dir, args.compare).read_text())
        status, before, after = verdict["status"], verdict["before"], verdict["after"]
        if args.compare != f"compare-{before}-{after}.json":
            reasons.append(f"{args.compare}: holds the compare of {before!r} against "
                           f"{after!r}")
        if status not in JUDGED:
            reasons.append(f"{args.compare}: {status}, not a judged comparison")
        else:
            rows = {row["bytes"]: row["verdict"] for row in verdict["rows"]}
            if missing := [nbytes for nbytes in TRIGGER_SIZES if nbytes not in rows]:
                reasons.append(f"{args.compare}: no row at {missing} bytes")
            wins = [nbytes for nbytes in TRIGGER_SIZES if rows.get(nbytes) == "win"]
            print(f"{args.compare}: {status}, win at {wins} bytes")
    except (OSError, ValueError, KeyError, TypeError) as e:
        reasons.append(f"{args.compare}: unreadable ({type(e).__name__}: {e})")
    if reasons:
        for reason in reasons:
            print(f"not decided: {reason}")
        print("INCOMPLETE")
        return INCOMPLETE
    met = [f"{args.arm} leader rank {rank} at {nbytes} bytes, median {median:.3f} >= "
           f"{TRIGGER_SHARE}" for (nbytes, rank), values in sorted(shares.items())
           if (median := statistics.median(values)) >= TRIGGER_SHARE]
    met += [f"{args.compare} says win at {nbytes} bytes" for nbytes in wins]
    for reason in met:
        print(f"met: {reason}")
    print("TRIGGER MET" if met else "TRIGGER NOT MET")
    return 0


def judge(before: dict[int, dict], after: dict[int, dict], pairs: list[int],
          wire: bool) -> list[dict[str, Any]]:
    """Per size, the paired repeats' timing and wire rules, and wrong results in any
    repeat of either arm."""
    rows = []
    for nbytes in before[0]:
        b = [before[p][nbytes] for p in pairs]
        a = [after[p][nbytes] for p in pairs]
        row: dict[str, Any] = {"bytes": nbytes, "pairs": len(pairs)}
        if any(x[nbytes]["mismatches"] for x in [*before.values(), *after.values()]):
            row["verdict"] = "FAIL (wrong result)"
        for metric in ("p50_s", "p99_s", "cpu_per_call_s"):
            vb, va = [x[metric] for x in b], [x[metric] for x in a]
            s = max(spread(vb), spread(va))
            change = statistics.median(va) / statistics.median(vb) - 1
            better = sum(y < x for x, y in zip(vb, va, strict=True))
            row[metric] = {"before": statistics.median(vb), "after": statistics.median(va),
                           "change": change, "spread": s, "pairs_better": better}
            if change > s:
                row.setdefault("regressions", []).append(metric)
        if wire:
            differ = sorted({k for x, y in zip(b, a, strict=True) for k in WIRE
                             if x["counts"][k] != y["counts"][k]})
            if differ:
                row.setdefault("regressions", []).append(f"wire counts differ: {differ}")
        p50 = row["p50_s"]
        if row.get("regressions") and "verdict" not in row:
            row["verdict"] = "REGRESSION"
        elif -p50["change"] > p50["spread"] and p50["pairs_better"] >= max(1, len(pairs) - 1):
            row.setdefault("verdict", "win")
        else:
            row.setdefault("verdict", "no change (inside spread)")
        rows.append(row)
    return rows


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
    c.add_argument("--before-repeats", type=int, required=True,
                   help="the before arm's predeclared repeats: exactly 0 to N-1")
    c.add_argument("--after-repeats", type=int, required=True,
                   help="the after arm's predeclared repeats: exactly 0 to N-1")
    c.add_argument("--differ", action="append", default=[], metavar="KEY",
                   help="a workload key or group the arms differ in by design (repeatable)")
    c.add_argument("--wire", action="store_true",
                   help="the arms must put the same messages on the wire (equal counts)")
    t = sub.add_parser("trigger", help="section 5's copy overlap trigger (candidate A)")
    t.add_argument("dir")
    t.add_argument("--arm", required=True, help="the pure bridge arm (P0x2)")
    t.add_argument("--repeats", type=int, required=True,
                   help="its predeclared repeats: exactly 0 to N-1")
    t.add_argument("--compare", required=True,
                   help="the D3 compare JSON in DIR (compare-P0-D3.json)")
    args = parser.parse_args()
    if args.command == "compare" and min(args.before_repeats, args.after_repeats) < 1:
        parser.error("--before-repeats and --after-repeats must be at least 1")
    if args.command == "trigger" and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return {"run": run, "compare": compare, "trigger": trigger}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
