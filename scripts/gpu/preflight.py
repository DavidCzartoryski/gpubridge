"""Connectivity preflight for a job spanning two machines. Fails fast with a clear message.

Run it on both machines at about the same time, before torchrun:

    python scripts/gpu/preflight.py --role master --master-addr 100.64.0.1 --port 29501
    python scripts/gpu/preflight.py --role worker --master-addr 100.64.0.1 --port 29501

Checks, in order:

1. The master address resolves.
2. The network interface in GLOO_SOCKET_IFNAME (or --ifname) exists.
3. Worker only: a TCP connection to master:port opens (the master listens there).
4. Both: a real 2-rank Gloo group forms and runs a barrier and an all_reduce.
   Gloo connects ranks on random ports in both directions, so this catches
   firewalls that let the rendezvous port through but block everything else.

Exit code 0 on success. Writes OUT/preflight-<role>.json when --out is given.
"""

from __future__ import annotations

import argparse
import errno
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

HINT_FLAT_NETWORK = (
    "gpubridge needs a flat network between the machines: every TCP port open in "
    "both directions. Put both machines on one Tailscale tailnet (or a WireGuard "
    "tunnel) and use the tailnet address as --master-addr; see "
    "docs/GPU_VALIDATION_RUNBOOK.md."
)


class PreflightError(RuntimeError):
    pass


def check_resolve(addr: str) -> str:
    try:
        return socket.getaddrinfo(addr, None, socket.AF_INET)[0][4][0]
    except socket.gaierror as exc:
        raise PreflightError(
            f"cannot resolve master address {addr!r}: {exc}. Use the master's IP "
            "(for Tailscale: `tailscale ip -4` on the master)."
        ) from exc


def check_interface(name: str | None) -> dict[str, Any]:
    if not name:
        return {"ifname": None, "note": "GLOO_SOCKET_IFNAME unset; Gloo picks an interface"}
    names = [n for _, n in socket.if_nameindex()]
    if name not in names:
        raise PreflightError(
            f"network interface {name!r} (GLOO_SOCKET_IFNAME) does not exist here; "
            f"interfaces: {', '.join(names)}. For Tailscale it is usually tailscale0, "
            "for WireGuard wg0."
        )
    return {"ifname": name, "interfaces": names}


def check_tcp(addr: str, port: int, timeout: float) -> float:
    """Keep trying to connect until the master is listening or the deadline passes."""
    deadline = time.monotonic() + timeout
    last: OSError | None = None
    while time.monotonic() < deadline:
        start = time.monotonic()
        try:
            with socket.create_connection((addr, port), timeout=5):
                return round(time.monotonic() - start, 4)
        except OSError as exc:
            last = exc
            time.sleep(1)
    refused = getattr(last, "errno", None) == errno.ECONNREFUSED
    if isinstance(last, ConnectionRefusedError) or refused:
        why = ("connection refused: the master isn't running its preflight yet, or a "
               "firewall rejects the port")
    elif isinstance(last, socket.timeout | TimeoutError):
        why = "timed out: a firewall is probably dropping the packets, or the address is wrong"
    else:
        why = f"{type(last).__name__}: {last}"
    raise PreflightError(f"could not reach {addr}:{port} within {timeout:.0f}s ({why}). "
                         + HINT_FLAT_NETWORK)


def check_gloo(addr: str, port: int, rank: int, timeout: float) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    start = time.monotonic()
    try:
        dist.init_process_group("gloo", init_method=f"tcp://{addr}:{port}", rank=rank,
                                world_size=2, timeout=timedelta(seconds=timeout))
        dist.barrier()
        value = torch.tensor([rank + 1.0])
        dist.all_reduce(value)
        if float(value) != 3.0:
            raise PreflightError(f"Gloo all_reduce returned {float(value)}, expected 3.0")
        dist.destroy_process_group()
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError(
            f"Gloo could not connect the two machines: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0] if str(exc) else ''}. The rendezvous port may be "
            "open while Gloo's own (random) ports are blocked. " + HINT_FLAT_NETWORK
        ) from exc
    return {"seconds": round(time.monotonic() - start, 3), "torch": torch.__version__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--role", choices=["master", "worker"], required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--port", type=int, required=True,
                        help="port for the preflight's own store (not torchrun's)")
    parser.add_argument("--ifname", default=os.environ.get("GLOO_SOCKET_IFNAME"))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    record: dict[str, Any] = {"script": "scripts/gpu/preflight.py", "role": args.role,
                              "master_addr": args.master_addr, "port": args.port,
                              "hostname": socket.gethostname(), "stages": []}

    def stage(name: str, fn) -> Any:
        try:
            result = fn()
            record["stages"].append({"name": name, "ok": True, "result": result})
            print(f"preflight [{args.role}] {name}: ok", flush=True)
            return result
        except PreflightError as exc:
            record["stages"].append({"name": name, "ok": False, "error": str(exc)})
            raise

    try:
        stage("resolve", lambda: check_resolve(args.master_addr))
        stage("interface", lambda: check_interface(args.ifname))
        if args.role == "worker":
            stage("tcp", lambda: check_tcp(args.master_addr, args.port, args.timeout))
        stage("gloo", lambda: check_gloo(args.master_addr, args.port,
                                         0 if args.role == "master" else 1, args.timeout))
        record["ok"] = True
    except PreflightError as exc:
        record["ok"] = False
        print(f"\npreflight [{args.role}] FAILED: {exc}\n", file=sys.stderr, flush=True)
    if args.out:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpukit import write_json
        write_json(args.out / f"preflight-{args.role}.json", record)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
