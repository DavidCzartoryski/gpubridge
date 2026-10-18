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
  count 4-octet words), next to the bytes the bridge reports moving;
- no-silent-TCP evidence: ``/proc/net/dev`` bytes of every interface (IPoIB too) over
  the same window. A verbs run passes it when the RDMA port's data counters moved at
  least the payload while no interface's TCP/IP bytes moved more than a small fraction
  of it.

Modes:

``endpoint``: host-to-host verbs without collectives, between ranks 0 and 1 (one per
node): a ping-pong latency series, a one-way bandwidth sweep, SHA-256 of every
buffer on both sides. Uses ``gpubridge.verbs.Endpoint`` directly, its handshake
through torchrun's store.

``collectives``: ``gpubridge.init(bridge=...)`` then all_reduce and broadcast at each
size on ``--memory host`` or ``gpu`` tensors, checked exactly against a Gloo
all_reduce of the same integer-valued data. With ``--expect-path direct|staged`` every
GPU call must have taken that path (``GPUBRIDGE_VERBS_GPU_PATH=direct`` makes a
direct path that can't run fail loudly rather than stage). ``--bridge gloo`` or ``ucx``
runs the same sizes as matched references (their records have no verbs telemetry).

    torchrun --nnodes=2 --nproc-per-node=1 ... scripts/gpu/verbs_kit.py endpoint --out OUT
    GPUBRIDGE_VERBS_GPU_PATH=direct torchrun ... scripts/gpu/verbs_kit.py collectives \\
        --memory gpu --expect-path direct --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

SYSFS = Path("/sys/class/infiniband")
#: A verbs run "moved no payload over TCP" when no interface's bytes grew by more than
#: this fraction of the payload (plus a little for the store and Gloo's own traffic).
TCP_FRACTION = 0.05
TCP_SLACK = 4 << 20


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


def tcp_evidence(netdev: dict, counters: dict, payload: int, device: str | None) -> dict:
    """Whether the payload went over the RDMA port and not over TCP/IP; see the module."""
    tcp = max((max(v.get("rx_bytes") or 0, v.get("tx_bytes") or 0) for v in netdev.values()),
              default=0)
    words = sum(v or 0 for k, v in counters.items()
                if device and k.startswith(f"{device}/")
                and k.endswith(("/counters/port_xmit_data", "/counters/port_rcv_data")))
    rdma_bytes = 4 * words  # the standard IB data counters count 4-octet words
    return {"payload_bytes": payload, "max_interface_bytes": tcp,
            "rdma_port_data_bytes": rdma_bytes,
            "tcp_small": tcp <= TCP_FRACTION * payload + TCP_SLACK,
            "rdma_moved_payload": rdma_bytes >= payload,
            "no_silent_tcp": tcp <= TCP_FRACTION * payload + TCP_SLACK and rdma_bytes >= payload}


class Window:
    """Counters, netdev bytes, CPU time and wall time over a measured region."""

    def __enter__(self) -> Window:
        self.counters0, self.netdev0 = read_counters(), read_netdev()
        self.cpu0, self.wall0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.wall = time.monotonic() - self.wall0
        self.cpu = (cpu.ru_utime - self.cpu0.ru_utime) + (cpu.ru_stime - self.cpu0.ru_stime)
        self.counters = window(self.counters0, read_counters())
        self.netdev = window(self.netdev0, read_netdev())

    def record(self, payload: int, device: str | None) -> dict:
        return {"wall_s": self.wall, "cpu_s": self.cpu,
                "cpu_per_wall": self.cpu / self.wall if self.wall else None,
                "nic_counters": self.counters, "netdev": self.netdev,
                "tcp_evidence": tcp_evidence(self.netdev, self.counters, payload, device)}


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
        with Window() as w:
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
    out = {"rank": rank, "host": socket.gethostname(), "bridge": args.bridge,
           "memory": args.memory, "leader": bridge is not None, "calls": [],
           "mismatches": 0, "path_violations": 0}
    tele0 = bridge.telemetry() if bridge is not None and hasattr(bridge, "telemetry") else None
    moved = 0
    with Window() as w:
        for n in sizes(args.max_bytes):
            numel = max(1, n // 4)
            # Integer-valued, so every reduction order gives the same bits.
            base = (torch.arange(numel, dtype=torch.float32) % 1024) + rank
            tensor = base.clone().to(device)  # a copy on CPU too: the call works in place
            want = base.clone()
            dist.all_reduce(want, group=reference)
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
            if bridge is not None and hasattr(bridge, "telemetry"):
                t = bridge.telemetry()
                call.update(path=t.get("path"), path_reason=t.get("path_reason"),
                            last_gpu=t.get("last_gpu"))
                if args.expect_path and args.memory == "gpu" and t.get("path") != args.expect_path:
                    out["path_violations"] += 1
            out["calls"].append(call)
            moved += numel * 4
            src = topology.world_size - 1
            tensor = (base + 1000 * rank).to(device)  # a new tensor
            gpubridge.broadcast(tensor, src)
            want = (torch.arange(numel, dtype=torch.float32) % 1024) + src + 1000 * src
            ok = bool(torch.equal(tensor.cpu(), want))
            out["mismatches"] += not ok
            out["calls"].append({"op": "broadcast", "bytes": numel * 4, "exact": ok})
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
        evidence = (result.get("window") or {}).get("tcp_evidence") or {}
        if args.require_no_tcp and args.bridge == "verbs" and "window" in result \
                and not evidence.get("no_silent_tcp"):
            status, error = "failed", f"no evidence the payload avoided TCP: {evidence}"
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
