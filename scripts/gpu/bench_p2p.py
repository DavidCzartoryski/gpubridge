"""Point-to-point latency and bandwidth through ``backend="gpubridge"``, for each route a pair of ranks can take.

Launched by torchrun; ``dist.init_process_group("gpubridge")`` first. For each size (1 KB to
1 GB by default, x4 per step), rank 0 and a peer play ping-pong: rank 0 sends a float32
tensor, the peer receives it and sends it back. One-way latency is half the round trip,
and bandwidth is the size over that. The routes:

- ``bridge``: rank 0 and the lowest rank of another island: gpubridge's point-to-point
  engine over the job's bridge transport, with this job's staging settings
  (``GPUBRIDGE_PINNED_STAGING``, ``GPUBRIDGE_P2P_CHUNK_BYTES``, ``GPUBRIDGE_BRIDGE``).
- ``native``: rank 0 and the next rank of its own island, if it has one: the island's
  backend directly (NCCL or RCCL on GPUs, Gloo on CPU), the baseline.

Other ranks wait at a barrier. Each size starts with a dedicated untimed exchange that
checks the payload bit for bit both ways. Timing is the host clock, with the device
synchronized before and after each round trip; each size gets --warmup untimed round
trips, then --trials timed ones. To compare staging settings, run it once per setting:
each row records the staging and the transport.

Rank 0 writes OUT/bench_p2p.json, OUT/bench_p2p.csv and OUT/summary.md. Runs on CPU in
simulation mode too (use a smaller --max-bytes):

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=4 \\
        scripts/gpu/bench_p2p.py --out /tmp/p2p --max-bytes 4M
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
from bench_all_reduce import human, parse_size, sizes
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json

import gpubridge
from gpubridge.p2p import chunk_bytes_setting

TAG = 7
PATTERN_CHUNK = 1 << 20


def routes(topology: gpubridge.Topology) -> dict[str, int]:
    """Each route's peer for rank 0: ``bridge`` across islands, ``native`` within rank 0's."""
    layout = topology.layout
    own = layout.island_of(0)
    out: dict[str, int] = {}
    others = [island for island in layout.islands if island != own]
    if others:
        out["bridge"] = min(others[0].ranks)
    if len(own.ranks) > 1:
        out["native"] = sorted(own.ranks)[1]
    return out


def _pattern_chunk(offset: int, numel: int) -> torch.Tensor:
    """Exact residues even beyond float32's consecutive-integer range."""
    start = offset % 251
    return torch.arange(start, start + numel, dtype=torch.int64).remainder_(251).float()


def pattern(numel: int, device: torch.device) -> torch.Tensor:
    """Integer-valued float32s; bounded CPU temporaries, distinct indices mod 251."""
    result = torch.empty(numel, dtype=torch.float32, device=device)
    for start in range(0, numel, PATTERN_CHUNK):
        count = min(PATTERN_CHUNK, numel - start)
        result[start:start + count].copy_(_pattern_chunk(start, count))
    return result


def ping_pong(peer: int, numel: int, device: torch.device, *, warmup: int,
              trials: int) -> tuple[list[float], bool]:
    """Rank 0's side: seconds per round trip, and whether the payload came back intact."""
    sent = pattern(numel, device)
    back = torch.empty_like(sent)
    # Complete both peers' verification before starting any measured exchange,
    # including --warmup 0. The peer sends this reply only after its own check.
    back.fill_(-1)
    dist.send(sent, peer, tag=TAG)
    dist.recv(back, peer, tag=TAG)
    sync(device)
    ok = bool(torch.equal(back, sent))
    times: list[float] = []
    for i in range(warmup + trials):
        sync(device)
        start = time.perf_counter()
        dist.send(sent, peer, tag=TAG)
        dist.recv(back, peer, tag=TAG)
        sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return times, ok


def echo(numel: int, device: torch.device, *, rounds: int) -> bool:
    """Untimed verification, then ``rounds`` replies without verification work."""
    buf = torch.empty(numel, dtype=torch.float32, device=device)
    dist.recv(buf, 0, tag=TAG)
    ok = bool(torch.equal(buf, pattern(numel, device)))
    dist.send(buf, 0, tag=TAG)
    for _ in range(rounds):
        dist.recv(buf, 0, tag=TAG)
        dist.send(buf, 0, tag=TAG)
    return ok


def summary(rows: list[dict[str, Any]], staging: dict[str, Any]) -> str:
    lines = ["Ping-pong between rank 0 and each route's peer; one-way = half the median round trip.",
             "Payload verification is a separate untimed exchange before warmups and trials.",
             f"Staging: {staging}.", "",
             "| Route | Peer | Size | One-way (us) | Bandwidth (GB/s) | Round trip p90 (us) | Payload |",
             "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['route']} | {r['peer']} | {human(r['bytes'])} | {r['one_way_us']} | "
                     f"{r['bandwidth_GBps']} | {r['rtt_p90_us']} | {'ok' if r['payload_ok'] else 'WRONG'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-bytes", type=parse_size, default=parse_size("1K"))
    parser.add_argument("--max-bytes", type=parse_size, default=parse_size("1G"))
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=default_timeout(600),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 600")
    args = parser.parse_args()
    if args.factor <= 1:
        parser.error("--factor must be greater than 1")
    if args.min_bytes <= 0 or args.max_bytes < args.min_bytes:
        parser.error("size bounds require 0 < --min-bytes <= --max-bytes")
    if args.trials <= 0 or args.warmup < 0:
        parser.error("--trials must be positive and --warmup nonnegative")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    dist.init_process_group("gpubridge", timeout=timedelta(seconds=args.timeout))
    dist.barrier()  # builds gpubridge's topology; point-to-point needs it first
    topology = gpubridge.get_topology()
    rank, device = topology.rank, topology.device
    if topology.world_size < 2:
        raise SystemExit("bench_p2p.py needs at least two ranks")
    peers = routes(topology)
    staging = {"pinned_staging": topology.pinned_staging, "p2p_chunk_bytes": chunk_bytes_setting(),
               "transport": run_info(topology)["bridge"]}
    rows: list[dict[str, Any]] = []
    for route, peer in peers.items():
        for nbytes in sizes(args.min_bytes, args.max_bytes, args.factor):
            numel = max(1, nbytes // 4)
            times: list[float] = []
            ok = True
            if rank == 0:
                times, ok = ping_pong(peer, numel, device, warmup=args.warmup, trials=args.trials)
            elif rank == peer:
                ok = echo(numel, device, rounds=args.warmup + args.trials)
            both = [None] * topology.world_size
            dist.all_gather_object(both, ok)
            if rank == 0:
                rtt = statistics.median(times)
                p90 = sorted(times)[min(len(times) - 1, int(0.9 * len(times)))]
                rows.append({"route": route, "peer": peer, "bytes": numel * 4, "dtype": "float32",
                             "trials": len(times), "rtt_median_us": round(rtt * 1e6, 2),
                             "rtt_p90_us": round(p90 * 1e6, 2),
                             "one_way_us": round(rtt / 2 * 1e6, 2),
                             "bandwidth_GBps": round(numel * 4 / (rtt / 2) / 1e9, 4),
                             "payload_ok": bool(both[0] and both[peer])})
                print(f"{route:7s} {human(numel * 4):>6s}: one-way {rows[-1]['one_way_us']:>12} us  "
                      f"{rows[-1]['bandwidth_GBps']} GB/s", flush=True)
            dist.barrier()
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        record = {"kit_version": KIT_VERSION, "script": "scripts/gpu/bench_p2p.py", "argv": sys.argv,
                  "run": run_info(topology), "environment": environment(), "staging": staging,
                  "routes": peers, "rows": rows}
        write_json(args.out / "bench_p2p.json", record)
        with open(args.out / "bench_p2p.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["route"])
            writer.writeheader()
            writer.writerows(rows)
        (args.out / "summary.md").write_text(f"# {args.out.name}: point-to-point benchmark\n\n"
                                             + summary(rows, staging))
    ok = all(r["payload_ok"] for r in rows) if rank == 0 else True
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
