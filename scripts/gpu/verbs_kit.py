"""Hardware checks for the verbs bridge (docs/VERBS_TRANSPORT.md), launched by torchrun.

PREPARED, NOT RUN: nothing here has run on RDMA hardware or GPUs. Every mode writes
OUT/rank<N>.json with what a reviewer needs to judge the run, never a remote key:

- the endpoint the library chose (device, port, link layer, GID index and type, MTU,
  firmware) and the peers' (telemetry), the provider (it must not say MOCK);
- completion errors, failure state, and per call the path (``host``/``direct``/
  ``staged``) and why;
- checksums of what arrived next to the sender's, or exact agreement with a Gloo
  reference on integer-valued data;
- latency and bandwidth, and the process's CPU time over the same window;
- NIC counter windows: every ``/sys/class/infiniband/*/ports/*/counters`` and
  ``hw_counters`` file before and after (raw values; the standard IB data counters
  count 4-octet words), and ``/proc/net/dev`` bytes of every interface: context only.
  A port is shared with every other process and job on the node, and on RoCE its
  Ethernet netdev also carries traffic that is not TCP, so neither proves which
  transport carried this job's payload;
- no-silent-TCP evidence (``tcp_evidence``), positive and attributable to this
  process, judged ``pass``, ``fail`` or ``unknown``:

  1. the bridge's own telemetry over the window: a real provider (``libibverbs``,
     not MOCK), its sent plus received bytes at least the payload, completions
     polled from its own queue pairs and no completion errors; each call's path
     (``host``/``direct``/``staged``) is in the call records;
  2. this process's RC queue pairs as the kernel lists them (``rdma resource show
     qp``, by pid), where the ``rdma`` tool exists (recorded; not required);
  3. this process's own TCP sockets to other hosts (``ss -tinp``, by pid: Gloo's,
     the store's) idle over the window: they moved no more than a small fraction of
     the payload. Without ``ss`` this is ``unknown``, never a pass.

  Gloo reference collectives run before the window, not inside it. A run passes
  only when every part is shown; ``--require-no-tcp`` fails a bridge leader's step
  unless its evidence is ``pass``. Ranks that aren't bridge leaders carry no bridge
  traffic and record ``not applicable``; a job with no leader at all (one island: an
  all-NVIDIA node without ``GPUBRIDGE_SPLIT_TEST``, say) never runs the bridge, and
  ``--require-no-tcp`` fails every rank of it.

Modes:

``endpoint``: host-to-host verbs without collectives, between ranks 0 and 1 (one per
node): a ping-pong latency series, a one-way bandwidth sweep, SHA-256 of every
buffer on both sides. Uses ``gpubridge.verbs.Endpoint`` directly, its handshake
through torchrun's store.

``collectives``: ``gpubridge.init(bridge=...)`` then all_reduce and broadcast at each
size on ``--memory host`` or ``gpu`` tensors, checked exactly against a Gloo
all_reduce of the same integer-valued data. With ``--expect-path direct|staged`` every
GPU call must have taken that path (``GPUBRIDGE_VERBS_GPU_PATH=direct`` makes a
direct path that can't run fail loudly rather than stage). ``direct`` means nothing
bounced: a call with any receive through a host bounce buffer (below
``GPUBRIDGE_VERBS_RNDV_BYTES``) is ``direct+bounce``, which fails ``--expect-path
direct``; every call records its bounced bytes, and the run their total
(``bounced_bytes``). The direct step therefore runs with ``GPUBRIDGE_VERBS_RNDV_BYTES=1``,
so every message goes by rendezvous. ``--bridge gloo`` or ``ucx`` runs the same sizes
as matched references (their records have no verbs telemetry).

    torchrun --nnodes=2 --nproc-per-node=1 ... scripts/gpu/verbs_kit.py endpoint --out OUT
    GPUBRIDGE_VERBS_GPU_PATH=direct torchrun ... scripts/gpu/verbs_kit.py collectives \\
        --memory gpu --expect-path direct --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

SYSFS = Path("/sys/class/infiniband")
#: This process's TCP sockets to other hosts are "idle" over a window when they moved no
#: more than this fraction of the payload, plus a little for the store's and Gloo's own
#: control traffic (barriers, keep-alives).
TCP_FRACTION = 0.05
TCP_SLACK = 1 << 20
_LOOPBACK = ("127.", "[::1]", "::1", "[::ffff:127.")


def rank_from_env() -> int:
    return int(os.environ.get("RANK", "0"))


def read_counters(root: Path = SYSFS) -> dict:
    """Every port counter and hardware counter, raw, by device/port/name."""
    out: dict = {}
    for dev in sorted(root.glob("*")) if root.exists() else []:
        for port in sorted((dev / "ports").glob("*")):
            for group in ("counters", "hw_counters"):
                for f in sorted((port / group).glob("*")):
                    try:
                        out[f"{dev.name}/{port.name}/{group}/{f.name}"] = int(f.read_text())
                    except (OSError, ValueError):
                        pass
    return out


def read_netdev(path: Path = Path("/proc/net/dev")) -> dict:
    """rx/tx bytes of every interface but loopback."""
    out = {}
    try:
        lines = path.read_text().splitlines()[2:]
    except OSError:
        return out
    for line in lines:
        name, _, rest = line.partition(":")
        fields = rest.split()
        if name.strip() != "lo" and len(fields) >= 9:
            out[name.strip()] = {"rx_bytes": int(fields[0]), "tx_bytes": int(fields[8])}
    return out


def window(before: dict, after: dict) -> dict:
    """after - before per key (None where it went backwards: a reset or wrap)."""
    out = {}
    for key, value in after.items():
        if isinstance(value, dict):
            out[key] = window(before.get(key, {}), value)
        elif key in before:
            out[key] = value - before[key] if value >= before[key] else None
    return out


def parse_ss(text: str, pid: int) -> dict:
    """Bytes sent and received by each of ``pid``'s TCP sockets to another host, from
    ``ss -tinpH`` (a socket line, then its indented info line), by "local peer"."""
    out: dict = {}
    current = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            fields = line.split()
            current = None
            if len(fields) >= 5 and re.search(rf"\bpid={pid}\b", line) \
                    and not fields[4].startswith(_LOOPBACK):
                current = f"{fields[3]} {fields[4]}"
                out[current] = {"sent": 0, "received": 0}
        elif current is not None:
            sent = re.search(r"\bbytes_acked:(\d+)", line) or \
                re.search(r"\bbytes_sent:(\d+)", line)
            got = re.search(r"\bbytes_received:(\d+)", line)
            out[current]["sent"] += int(sent.group(1)) if sent else 0
            out[current]["received"] += int(got.group(1)) if got else 0
    return out


def read_process_tcp(pid: int | None = None, run=subprocess.run) -> dict | None:
    """:func:`parse_ss` of this process, or None where ``ss`` can't say."""
    try:
        done = run(["ss", "-tinpH"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_ss(done.stdout, os.getpid() if pid is None else pid)


def read_qps(pid: int | None = None, run=subprocess.run) -> list | None:
    """This process's queue pairs as the kernel lists them (``rdma -j resource show
    qp``), or None where the ``rdma`` tool can't say."""
    pid = os.getpid() if pid is None else pid
    try:
        done = run(["rdma", "-j", "resource", "show", "qp"], capture_output=True, text=True,
                   timeout=10, check=False)
        entries = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    return [{k: e.get(k) for k in ("ifname", "port", "lqpn", "type", "state", "pid", "comm")}
            for e in entries if isinstance(e, dict) and e.get("pid") == pid]


def tcp_moved(before: dict | None, after: dict | None) -> int | None:
    """Bytes this process's TCP sockets to other hosts moved between two snapshots (a
    socket opened in between counts in full), or None if either is unknown."""
    if before is None or after is None:
        return None
    total = 0
    for key, now in after.items():
        then = before.get(key, {"sent": 0, "received": 0})
        total += max(0, now["sent"] - then["sent"]) + max(0, now["received"] - then["received"])
    return total


def tcp_evidence(payload: int, tele_before: dict | None, tele_after: dict | None,
                 tcp: int | None, qps: list | None, netdev: dict | None = None,
                 counters: dict | None = None, device: str | None = None) -> dict:
    """Whether this process's payload went by verbs and not over TCP; see the module.
    ``verdict`` is ``pass`` only with every positive part shown, ``fail`` when a part
    contradicts it, else ``unknown``; ``no_silent_tcp`` is ``verdict == "pass"``."""
    fails, unknown = [], []
    moved = completions = errors = None
    if tele_after is None:
        unknown.append("no bridge telemetry on this rank")
    else:
        before = tele_before or {}
        provider = str(tele_after.get("provider") or "")
        if tele_after.get("mock") or "MOCK" in provider.upper():
            fails.append(f"the bridge ran on a MOCK provider ({provider}): not RDMA")
        elif "libibverbs" not in provider:
            fails.append(f"the provider is {provider or 'unreported'}, not libibverbs")

        def grew(key: str) -> int:
            return int(tele_after.get(key) or 0) - int(before.get(key) or 0)

        moved = grew("bytes_sent") + grew("bytes_received")
        completions, errors = grew("completions"), grew("completion_errors")
        if moved < payload:
            fails.append(f"the bridge's own counters moved {moved} bytes, less than the "
                         f"payload {payload}")
        if completions <= 0:
            fails.append("no completions were polled from the bridge's queue pairs")
        if errors:
            fails.append(f"{errors} completion errors")
    limit = TCP_FRACTION * payload + TCP_SLACK
    if tcp is None:
        unknown.append("this process's TCP sockets could not be read (no ss): TCP not "
                       "shown idle")
    elif tcp > limit:
        fails.append(f"this process's TCP sockets to other hosts moved {tcp} bytes "
                     f"(limit {int(limit)})")
    if qps is not None and tele_after is not None and not any(
            q.get("type") == "RC" for q in qps):
        fails.append("the kernel lists no RC queue pair for this process")
    words = sum(v or 0 for k, v in (counters or {}).items()
                if device and k.startswith(f"{device}/")
                and k.endswith(("/counters/port_xmit_data", "/counters/port_rcv_data")))
    interfaces = max((max(v.get("rx_bytes") or 0, v.get("tx_bytes") or 0)
                      for v in (netdev or {}).values()), default=0)
    verdict = "fail" if fails else "unknown" if unknown else "pass"
    return {"verdict": verdict, "no_silent_tcp": verdict == "pass", "failures": fails,
            "unknown": unknown, "payload_bytes": payload, "bridge_bytes": moved,
            "bridge_completions": completions, "bridge_completion_errors": errors,
            "process_tcp_bytes": tcp, "process_tcp_limit": int(limit),
            "qps": qps if qps is not None else "unavailable (no rdma tool)",
            "context_only": {
                "rdma_port_data_bytes": 4 * words, "max_interface_bytes": interfaces,
                "why": "port and netdev counters are shared by every process and job on the "
                       "node, and a RoCE netdev carries non-TCP traffic: never decisive"}}


class Window:
    """Counters, netdev bytes, CPU time and wall time over a measured region."""

    def __init__(self, telemetry=None) -> None:
        #: The bridge's telemetry (a callable), read at both ends; None off the bridge.
        self.telemetry = telemetry

    def __enter__(self) -> Window:
        self.tele0 = self.telemetry() if self.telemetry else None
        self.tcp0 = read_process_tcp()
        self.counters0, self.netdev0 = read_counters(), read_netdev()
        self.cpu0, self.wall0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.wall = time.monotonic() - self.wall0
        self.cpu = (cpu.ru_utime - self.cpu0.ru_utime) + (cpu.ru_stime - self.cpu0.ru_stime)
        self.counters = window(self.counters0, read_counters())
        self.netdev = window(self.netdev0, read_netdev())
        self.tcp = tcp_moved(self.tcp0, read_process_tcp())
        self.tele1 = self.telemetry() if self.telemetry else None
        self.qps = read_qps() if self.telemetry else None

    def record(self, payload: int, device: str | None) -> dict:
        return {"wall_s": self.wall, "cpu_s": self.cpu,
                "cpu_per_wall": self.cpu / self.wall if self.wall else None,
                "nic_counters": self.counters, "netdev": self.netdev,
                "tcp_evidence": tcp_evidence(payload, self.tele0, self.tele1, self.tcp,
                                             self.qps, self.netdev, self.counters, device)}


def sizes(max_bytes: int) -> list[int]:
    out, n = [], 8
    while n <= max_bytes:
        out.append(n)
        n *= 4
    return out


def sha(buffer) -> str:
    return hashlib.sha256(memoryview(buffer).cast("B")).hexdigest()


def endpoint_mode(args, store, rank: int, world: int) -> dict:
    """Host-to-host verbs between ranks 0 and 1; see the module docs."""
    import torch

    from gpubridge import verbs
    from gpubridge.verbs.transport import endpoint_settings

    if world < 2:
        raise SystemExit("endpoint mode needs ranks 0 and 1")
    if rank > 1:
        return {"rank": rank, "skipped": "only ranks 0 and 1 take part"}
    peer = 1 - rank
    ep = verbs.Endpoint(2, rank, endpoint_settings())
    try:
        store.set(f"verbs_kit/addr/{rank}", ep.local_info())
        store.wait([f"verbs_kit/addr/{peer}"], timedelta(seconds=args.timeout))
        ep.connect(peer, store.get(f"verbs_kit/addr/{peer}").decode())
        store.set(f"verbs_kit/ready/{rank}", "1")
        store.wait([f"verbs_kit/ready/{peer}"], timedelta(seconds=args.timeout))
        tele = ep.telemetry()
        deadline = lambda: time.monotonic() + args.timeout  # noqa: E731
        out = {"rank": rank, "host": socket.gethostname(), "endpoint": tele, "latency": {},
               "bandwidth": [], "checksums": []}
        # Ping-pong latency, 8 bytes.
        small = torch.zeros(8, dtype=torch.uint8)
        times = []
        for i in range(args.iters):
            t0 = time.monotonic()
            if rank == 0:
                ep.wait([ep.send(1, small.data_ptr(), 8, 2 * i)], deadline())
                ep.wait([ep.recv(1, small.data_ptr(), 8, 2 * i + 1)], deadline())
            else:
                ep.wait([ep.recv(0, small.data_ptr(), 8, 2 * i)], deadline())
                ep.wait([ep.send(0, small.data_ptr(), 8, 2 * i + 1)], deadline())
            times.append(time.monotonic() - t0)
        times.sort()
        out["latency"] = {"round_trips": len(times), "median_us": 1e6 * times[len(times) // 2],
                          "p99_us": 1e6 * times[int(0.99 * (len(times) - 1))]}
        moved = 0
        with Window(ep.telemetry) as w:
            for n in sizes(args.max_bytes):
                data = torch.randint(0, 256, (n,), dtype=torch.uint8,
                                     generator=torch.Generator().manual_seed(n))
                buf = data if rank == 0 else torch.empty(n, dtype=torch.uint8)
                reps = max(1, min(args.iters, (64 << 20) // n))
                t0 = time.monotonic()
                for r in range(reps):
                    tag = (1 << 32) + n * 1000 + r
                    if rank == 0:
                        ep.wait([ep.send(1, buf.data_ptr(), n, tag)], deadline())
                    else:
                        ep.wait([ep.recv(0, buf.data_ptr(), n, tag)], deadline())
                elapsed = time.monotonic() - t0
                moved += n * reps
                out["bandwidth"].append({"bytes": n, "messages": reps, "seconds": elapsed,
                                         "gbit_s": 8 * n * reps / elapsed / 1e9})
                out["checksums"].append({"bytes": n, "sha256": sha(buf.numpy())})
        out["window"] = w.record(moved, tele.get("device"))
        out["telemetry_after"] = ep.telemetry()
        return out
    finally:
        ep.close(timeout=10.0)


def collectives_mode(args, rank: int, world: int) -> dict:
    """gpubridge collectives over --bridge on --memory tensors; see the module docs."""
    import torch
    import torch.distributed as dist

    import gpubridge

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout), bridge=args.bridge,
                              policy=args.policy)
    device = topology.device if args.memory == "gpu" else torch.device("cpu")
    if args.memory == "gpu" and device.type != "cuda":
        raise SystemExit("--memory gpu needs a GPU on every rank")
    reference = dist.new_group(backend="gloo")
    bridge = topology.bridge
    # How many ranks lead a bridge: none in a job of one island.
    leaders = torch.tensor([int(bridge is not None)])
    dist.all_reduce(leaders, group=reference)
    out = {"rank": rank, "host": socket.gethostname(), "bridge": args.bridge,
           "memory": args.memory, "leader": bridge is not None,
           "leaders": int(leaders.item()), "calls": [],
           "mismatches": 0, "path_violations": 0, "bounced_bytes": 0}
    tele0 = bridge.telemetry() if bridge is not None and hasattr(bridge, "telemetry") else None

    def path_of(call: dict) -> None:
        """The call's path and bounced bytes from the bridge; counts a violation when it
        isn't --expect-path (direct+bounce isn't direct)."""
        if bridge is None or not hasattr(bridge, "telemetry"):
            return
        t = bridge.telemetry()
        gpu = t.get("last_gpu") or {}
        call.update(path=t.get("path"), path_reason=t.get("path_reason"), last_gpu=gpu or None,
                    bounced_bytes=gpu.get("bounced_bytes", 0))
        out["bounced_bytes"] += call["bounced_bytes"]
        if args.expect_path and args.memory == "gpu" and t.get("path") != args.expect_path:
            out["path_violations"] += 1

    # Integer-valued, so every reduction order gives the same bits. The Gloo references
    # run before the measured window: inside it, their TCP traffic would be counted
    # against the bridge.
    wants = {}
    for n in sizes(args.max_bytes):
        wants[n] = (torch.arange(max(1, n // 4), dtype=torch.float32) % 1024) + rank
        dist.all_reduce(wants[n], group=reference)
    moved = 0
    leader_telemetry = bridge.telemetry if bridge is not None and hasattr(
        bridge, "telemetry") else None
    with Window(leader_telemetry) as w:
        for n in sizes(args.max_bytes):
            numel = max(1, n // 4)
            base = (torch.arange(numel, dtype=torch.float32) % 1024) + rank
            tensor = base.clone().to(device)  # a copy on CPU too: the call works in place
            want = wants[n]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.monotonic()
            gpubridge.all_reduce(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t0
            got = tensor.cpu()
            ok = bool(torch.equal(got, want))
            out["mismatches"] += not ok
            call = {"op": "all_reduce", "bytes": numel * 4, "seconds": elapsed, "exact": ok,
                    "sha256": sha(got.numpy())}
            path_of(call)
            out["calls"].append(call)
            moved += numel * 4
            src = topology.world_size - 1
            tensor = (base + 1000 * rank).to(device)  # a new tensor
            gpubridge.broadcast(tensor, src)
            want = (torch.arange(numel, dtype=torch.float32) % 1024) + src + 1000 * src
            ok = bool(torch.equal(tensor.cpu(), want))
            out["mismatches"] += not ok
            call = {"op": "broadcast", "bytes": numel * 4, "exact": ok}
            path_of(call)
            out["calls"].append(call)
    device_name = (tele0 or {}).get("device")
    out["window"] = w.record(moved, device_name)
    if bridge is not None and hasattr(bridge, "telemetry"):
        out["telemetry_before"], out["telemetry_after"] = tele0, bridge.telemetry()
    gpubridge.destroy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["endpoint", "collectives"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=lambda s: int(float(s.rstrip("Mm")) * 2**20)
                        if s[-1] in "Mm" else int(s), default=256 << 20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--bridge", default="verbs", help="verbs, or gloo/ucx as references")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--memory", choices=["host", "gpu"], default="host")
    parser.add_argument("--expect-path", choices=["direct", "staged"], default=None)
    parser.add_argument("--require-no-tcp", action="store_true",
                        help="fail unless the window shows the payload on the RDMA port and not "
                             "on any interface's TCP/IP (verbs runs on hardware)")
    args = parser.parse_args()
    rank, world = rank_from_env(), int(os.environ.get("WORLD_SIZE", "1"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    started = time.time()
    status, error, result = "ok", None, {}
    try:
        if args.mode == "endpoint":
            import torch.distributed as dist

            store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]) + 7,
                                  world, rank == 0, timedelta(seconds=args.timeout))
            result = endpoint_mode(args, store, rank, world)
        else:
            result = collectives_mode(args, rank, world)
        if result.get("mismatches") or result.get("path_violations"):
            status = "failed"
        window_record = result.get("window") or {}
        evidence = window_record.get("tcp_evidence") or {}
        if result.get("leaders") == 0 and args.bridge == "verbs":
            # No rank ran the bridge: nothing shows it carried the payload.
            why = ("no rank is a bridge leader, so the verbs bridge carried nothing (a job of "
                   "one island, like an all-NVIDIA node without GPUBRIDGE_SPLIT_TEST)")
            if args.require_no_tcp:
                status, error = "failed", why
                evidence.update(verdict="fail (no bridge leader in the job)", no_silent_tcp=False)
            else:
                evidence.update(verdict="not applicable (no bridge leader in the job)",
                                no_silent_tcp=None)
        elif result.get("leader") is False and evidence:
            # Not a bridge leader: no bridge traffic here to judge.
            evidence.update(verdict="not applicable (not a bridge leader)", no_silent_tcp=None)
        elif args.require_no_tcp and args.bridge == "verbs" and "window" in result \
                and evidence.get("verdict") != "pass":
            status, error = "failed", (
                f"no-silent-TCP evidence is {evidence.get('verdict')}: "
                + "; ".join(evidence.get("failures", []) + evidence.get("unknown", [])))
    except Exception as exc:  # noqa: BLE001 - recorded, then the exit code says so
        status, error = "error", f"{type(exc).__name__}: {exc}"
    record = {"kit": "verbs_kit", "mode": args.mode, "args": vars(args), "status": status,
              "error": error, "started": started, "finished": time.time(), **result}
    text = json.dumps(record, indent=1, default=str)
    if "rkey" in text or "lkey" in text:
        raise SystemExit("refusing to write a record with a key in it")
    (Path(args.out) / f"rank{rank}.json").write_text(text)
    print(f"verbs_kit {args.mode} rank {rank}: {status}{': ' + error if error else ''}",
          file=sys.stderr)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
