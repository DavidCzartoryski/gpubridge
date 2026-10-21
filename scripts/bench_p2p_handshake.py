"""The cost of the cross-island P2P handshake, one ``p2p.py`` against another, on CPU.

    git show b16113f:src/gpubridge/p2p.py > /tmp/p2p_before.py
    python scripts/bench_p2p_handshake.py --before /tmp/p2p_before.py --store tcp
    python scripts/bench_p2p_handshake.py --before /tmp/p2p_before.py --store file

Each arm runs two processes, rank 0 and rank 1, each with its own client of one Store (a
TCPStore that rank 0 serves, or one FileStore file), wrapped in the ``PrefixStore`` the
torch backend gives its engine, and an ``Engine`` from the ``p2p.py`` under test (the
after arm: this checkout's, or ``--after``). The engines' link completes every transfer at
once, without Gloo, so what is measured is the Store handshake, the lane and completion
threads and the health watcher. Phases, in this order:

- ``pingpong``: rank 0 sends to rank 1 and rank 1 sends back, ``--iterations`` times after
  ``--warmup``; one-way time is half of rank 0's round trip.
- ``idle``: both engines idle for ``--idle`` seconds.
- ``missing``: rank 0 starts ``--missing`` operations with a rank that never comes and,
  while they wait, times one ready exchange with rank 1; then ``--hold`` seconds of the
  wait alone; then ``close()``, which ends them.

For each phase and rank: Store calls by method (counted on the engine's Store), the
process's user+system CPU time, wall time and threads. Each process's stderr goes to a
file whose size is the log cost (the Store logs its timeouts there). Arms alternate
``--repeats`` times. Prints JSON with each ``p2p.py``'s SHA-256 and git blob id.

CPU processes on one machine: this is not GPU, RDMA or cross-node evidence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAG, READY_TAG, MISSING_TAG = 7, 50, 100


class _InstantLink:
    def post(self, tensor, peer, tag, sending):
        return self

    def wait(self, timeout):
        pass

    def close(self):
        pass


class _Counted:
    """The engine's Store, counting each method's calls."""

    def __init__(self, store):
        self.store, self.calls = store, collections.Counter()

    def __getattr__(self, name):
        method = getattr(self.store, name)

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return method(*args, **kwargs)

        return counted


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _load(path: str):
    spec = importlib.util.spec_from_file_location("p2p_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its imports (gpubridge.work ...) are this checkout's
    return module


def worker(rank: int, args: argparse.Namespace, p2p_path: str, log: str, out: str) -> None:
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(fd, 2)  # the C++ Store logs to fd 2 as well
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    p2p = _load(p2p_path)
    timeout = timedelta(seconds=60)
    if args.store == "tcp":
        base = dist.TCPStore("127.0.0.1", args.port, 2, rank == 0, timeout,
                             wait_for_workers=False)
        sync = dist.TCPStore("127.0.0.1", args.port, 2, False, timeout)
    else:
        base = dist.FileStore(args.path, 2)
        sync = dist.FileStore(args.path + ".sync", 2)
    store = _Counted(dist.PrefixStore("gpubridge-p2p", base))
    engine = p2p.Engine(_InstantLink(), torch.device("cpu"), rank, store, timeout)
    peer, tensor = 1 - rank, torch.ones(256)  # 1 KiB
    phases, step = {}, [0]

    def barrier() -> None:
        step[0] += 1
        sync.set(f"barrier/{step[0]}/{rank}", "1")
        sync.wait([f"barrier/{step[0]}/{peer}"], timeout)

    def measure(name, fn):
        barrier()
        calls, cpu, wall = collections.Counter(store.calls), _cpu(), time.perf_counter()
        extra = fn() or {}
        wall = time.perf_counter() - wall
        phases[name] = {"store_calls": dict(store.calls - calls),
                        "store_calls_total": sum((store.calls - calls).values()),
                        "cpu_s": _cpu() - cpu, "wall_s": wall,
                        "threads": threading.active_count(), **extra}

    def exchange(n):
        times = []
        for _ in range(n):
            start = time.perf_counter()
            if rank == 0:
                engine.submit(tensor, peer, TAG, True).wait()
                engine.submit(tensor, peer, TAG, False).wait()
            else:
                engine.submit(tensor, peer, TAG, False).wait()
                engine.submit(tensor, peer, TAG, True).wait()
            times.append(time.perf_counter() - start)
        return times

    exchange(args.warmup)

    def pingpong():
        times = exchange(args.iterations)
        if rank:
            return {}
        return {"one_way_us_median": statistics.median(times) / 2 * 1e6,
                "one_way_us_p90": statistics.quantiles(times, n=10)[8] / 2 * 1e6}

    measure("pingpong", pingpong)
    measure("idle", lambda: time.sleep(args.idle))

    def missing():
        waiting = []
        if rank == 0:  # rank 2 doesn't exist
            waiting = [engine.submit(tensor, 2, MISSING_TAG + i, True)
                       for i in range(args.missing)]
        time.sleep(0.2)  # the lanes are waiting
        start = time.perf_counter()
        engine.submit(tensor, peer, READY_TAG, rank == 0).wait(timedelta(seconds=30))
        ready_s = time.perf_counter() - start
        time.sleep(args.hold)
        start = time.perf_counter()
        engine.close(timeout=1.0)
        close_s = time.perf_counter() - start
        ended = sum(work._done.wait(1) for work in waiting)
        return {"ready_exchange_s": ready_s, "close_s": close_s,
                "waiting_ended_by_close": ended, "waiting": len(waiting)}

    measure("missing", missing)
    barrier()
    Path(out).write_text(json.dumps(phases))


def run_arm(args: argparse.Namespace, p2p_path: str, tmp: Path, label: str) -> dict:
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        args.port = s.getsockname()[1]
    args.path = str(tmp / f"{label}-store")
    for suffix in ("", ".sync"):
        Path(args.path + suffix).unlink(missing_ok=True)
    logs = [str(tmp / f"{label}-rank{r}.log") for r in (0, 1)]
    outs = [str(tmp / f"{label}-rank{r}.json") for r in (0, 1)]
    procs = [mp.get_context("spawn").Process(target=worker, args=(r, args, p2p_path,
                                                                  logs[r], outs[r]))
             for r in (0, 1)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(300)
        if proc.exitcode != 0:
            raise RuntimeError(f"{label}: a rank exited with {proc.exitcode}; see {logs}")
    ranks = [json.loads(Path(o).read_text()) for o in outs]
    return {"ranks": ranks, "stderr_bytes": [Path(log).stat().st_size for log in logs]}


def summarize(runs: list[dict]) -> dict:
    """Medians over repeats: rank 0's latency, both ranks' Store calls and CPU summed."""
    def med(fn):
        return round(statistics.median(fn(r) for r in runs), 6)

    out = {"repeats": len(runs),
           "stderr_bytes": med(lambda r: sum(r["stderr_bytes"]))}
    for phase in ("pingpong", "idle", "missing"):
        both = lambda r, key, phase=phase: sum(rank[phase][key] for rank in r["ranks"])  # noqa: E731
        out[phase] = {"store_calls": med(lambda r: both(r, "store_calls_total")),
                      "cpu_s": med(lambda r: both(r, "cpu_s")),
                      "wall_s_rank0": med(lambda r: r["ranks"][0][phase]["wall_s"])}
    out["pingpong"]["one_way_us_median"] = med(
        lambda r: r["ranks"][0]["pingpong"]["one_way_us_median"])
    out["pingpong"]["one_way_us_p90"] = med(lambda r: r["ranks"][0]["pingpong"]["one_way_us_p90"])
    for key in ("ready_exchange_s", "close_s"):
        out["missing"][key + "_rank0"] = med(lambda r, key=key: r["ranks"][0]["missing"][key])
    out["missing"]["waiting_ended_by_close"] = [r["ranks"][0]["missing"]["waiting_ended_by_close"]
                                                for r in runs]
    return out


def identity(path: Path) -> dict:
    data = path.read_bytes()
    blob = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True,
                          cwd=REPO).stdout.strip()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=Path, required=True, help="a p2p.py")
    parser.add_argument("--after", type=Path, default=REPO / "src" / "gpubridge" / "p2p.py")
    parser.add_argument("--store", choices=("tcp", "file"), default="tcp")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--idle", type=float, default=2.0)
    parser.add_argument("--missing", type=int, default=8)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    import torch

    arms = {"before": args.before.resolve(), "after": args.after.resolve()}
    runs: dict[str, list] = {"before": [], "after": []}
    with tempfile.TemporaryDirectory(prefix="gpubridge-p2p-bench-") as tmp:
        for i in range(args.repeats):
            for label in ("before", "after") if i % 2 == 0 else ("after", "before"):
                runs[label].append(run_arm(args, str(arms[label]), Path(tmp), f"{label}{i}"))
    print(json.dumps({
        "store": args.store, "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(),
        "settings": {k: getattr(args, k) for k in ("iterations", "warmup", "idle", "missing",
                                                   "hold", "repeats")},
        "sources": {label: identity(path) for label, path in arms.items()},
        "summary": {label: summarize(r) for label, r in runs.items()},
        "runs": runs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
