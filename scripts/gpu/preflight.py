"""Network preflight between two machines: reachability, the interface for Gloo, raw bandwidth.

Run it on both machines (the two island leaders) at about the same time, before torchrun:

    python scripts/gpu/preflight.py --role master --master-addr 100.64.0.1 --port 29501
    python scripts/gpu/preflight.py --role worker --master-addr 100.64.0.1 --port 29501

Under srun with one task per machine, ``--role auto`` makes task 0 the master.

Checks, in order:

1. ``resolve``: the master address resolves.
2. ``interface``: GLOO_SOCKET_IFNAME (or --ifname), if set, exists here.
3. ``tcp`` (worker only): a TCP connection to master:port opens (the master listens there).
4. ``store``: both sides meet in a TCP store on master:port, used for the rest.
5. ``interfaces``: each side lists its candidate interfaces: those with an IPv4
   address, minus loopback, link-local addresses and container bridges, or
   only GLOO_SOCKET_IFNAME's when it is set.
6. ``reachability``: pairs of interfaces, one per side, that the kernels route
   between (for each address on one side, the interface the other side would
   send from). For each pair, a TCP connection from the worker's interface to
   the master's, then one the other way, since a Gloo pair connects in
   whichever direction its addresses decide. Over the first connection: the
   round-trip time, then raw TCP bandwidth one way and then the other,
   ``--bandwidth-seconds`` each, over one stream (Gloo uses one connection per
   pair of ranks).
7. ``choose``: the pair that connects both ways and whose slower direction is
   fastest. Each side's interface in it is that side's GLOO_SOCKET_IFNAME (the
   names may differ between the machines).
8. ``gloo``: a real 2-rank Gloo group on the chosen interfaces runs a barrier and
   an all_reduce. Gloo connects on random ports in both directions, so this
   catches firewalls that let the rendezvous port through but block the rest.

Exit code 0 on success. Writes OUT/preflight-<role>.json when --out is given,
with ``chosen_ifname``, the chosen pair's ``bandwidth`` from this side (MB/s in
and out, round-trip time) and every pair's numbers. The Explorer jobs export
``chosen_ifname`` as GLOO_SOCKET_IFNAME for the rest of the job.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import select
import socket
import statistics
import struct
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
#: Interfaces that only reach containers on this machine.
BRIDGE_PREFIXES = ("docker", "br-", "veth", "virbr", "cni", "flannel", "vxlan")
#: SIOCGIFADDR, the ioctl that reads an interface's IPv4 address.
SIOCGIFADDR = {"linux": 0x8915, "darwin": 0xC0206921}
CHUNK = 4 * 2**20
WARMUP = 0.2  # of --bandwidth-seconds, not counted
RTT_ROUNDS = 20


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
        return {"ifname": None, "note": "GLOO_SOCKET_IFNAME unset; the preflight chooses one"}
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


# ---- interfaces ----------------------------------------------------------------------

def interface_addresses() -> dict[str, str]:
    """The IPv4 address of every interface that has one."""
    request = SIOCGIFADDR.get(sys.platform)
    if request is None:
        return {}
    import fcntl

    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                raw = fcntl.ioctl(sock.fileno(), request, struct.pack("256s", name.encode()[:15]))
            except OSError:
                continue  # no IPv4 address
            found[name] = socket.inet_ntoa(raw[20:24])
    return found


def candidates(addresses: dict[str, str], ifname: str | None, loopback: bool
               ) -> list[tuple[str, str]]:
    """(name, address) of the interfaces to try, in name order.

    ``ifname``, if given, is the only one. Loopback counts only when both
    sides are on this machine (``loopback``, the master address is local).
    """
    if ifname:
        if ifname not in addresses:
            raise PreflightError(f"network interface {ifname!r} (GLOO_SOCKET_IFNAME) has no "
                                 "IPv4 address here, so Gloo can't use it.")
        return [(ifname, addresses[ifname])]
    keep = []
    for name, addr in sorted(addresses.items()):
        if addr.startswith("127."):
            if loopback:
                keep.append((name, addr))
        elif not name.startswith(BRIDGE_PREFIXES) and not addr.startswith("169.254."):
            keep.append((name, addr))
    if not keep:
        raise PreflightError(f"no candidate interface here (IPv4 addresses: {addresses}).")
    return keep


def route_source(addr: str) -> str | None:
    """The local address the kernel would send from to reach ``addr``, if it has a route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect((addr, 9))  # UDP: picks a route, sends nothing
        except OSError:
            return None
        return sock.getsockname()[0]


def routed_pairs(mine: list[tuple[str, str]], theirs: list[tuple[str, str]], role: str
                 ) -> list[tuple[str, str]]:
    """(master interface, worker interface) pairs this side would route between.

    For each of the other side's addresses, the interface this side's kernel
    would send from to reach it, if that is one of this side's candidates.
    """
    by_address = {addr: name for name, addr in mine}
    pairs = []
    for name, addr in theirs:
        local = by_address.get(route_source(addr) or "")
        if local is not None:
            pairs.append((local, name) if role == "master" else (name, local))
    return pairs


def choose(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The pair that connects both ways whose slower direction is fastest; the first on a tie."""
    working = [pair for pair in pairs if pair["connects"]]
    if not working:
        tried = "; ".join(describe(pair) for pair in pairs) or "no pair of interfaces routes"
        raise PreflightError("no pair of interfaces connects the two machines both ways: "
                             f"{tried}. " + HINT_FLAT_NETWORK)
    return max(working, key=lambda pair: min(pair["MBps_master_to_worker"],
                                             pair["MBps_worker_to_master"]))


def describe(pair: dict[str, Any]) -> str:
    names = (f"master {pair['master']} {pair['master_address']} <-> "
             f"worker {pair['worker']} {pair['worker_address']}")
    if not pair["connects"]:
        return f"{names}: {pair['error']}"
    return (f"{names}: master to worker {pair['MBps_master_to_worker']:.0f} MB/s, "
            f"worker to master {pair['MBps_worker_to_master']:.0f} MB/s, "
            f"rtt {pair['rtt_ms']:.3f} ms")


# ---- raw TCP measurements ------------------------------------------------------------

def _send_for(conn: socket.socket, seconds: float) -> None:
    """Send zeros for ``seconds``, then close this direction."""
    data = memoryview(bytes(CHUNK))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        conn.sendall(data)
    conn.shutdown(socket.SHUT_WR)


def _receive_rate(conn: socket.socket, seconds: float) -> float:
    """Read until the other side closes its direction; MB/s after a warmup.

    The first fifth of the time doesn't count, so TCP's slow start and buffer
    autotuning don't drag the number down.
    """
    buf = memoryview(bytearray(CHUNK))
    start = warm = None
    counted = 0
    while n := conn.recv_into(buf):
        now = time.monotonic()
        if start is None:
            start = now
        elif warm is None and now - start >= seconds * WARMUP:
            warm = now
        elif warm is not None:
            counted += n
    end = time.monotonic()
    if warm is None or end <= warm:
        return 0.0
    return counted / (end - warm) / 1e6


def _rtt_client(conn: socket.socket) -> float:
    times = []
    for _ in range(RTT_ROUNDS):
        start = time.monotonic()
        conn.sendall(b"x")
        if not conn.recv(1):
            raise ConnectionError("closed during the round-trip test")
        times.append(time.monotonic() - start)
    return statistics.median(times) * 1e3


def _rtt_server(conn: socket.socket) -> None:
    for _ in range(RTT_ROUNDS):
        if not conn.recv(1):
            raise ConnectionError("closed during the round-trip test")
        conn.sendall(b"x")


def _accept(store, key: str, addr: str, connect_timeout: float) -> socket.socket | str:
    """Listen on ``addr`` for the other side's connection: the socket, or why not."""
    try:
        server = socket.create_server((addr, 0), backlog=1)
    except OSError as exc:
        store.set(f"{key}/port", "0")
        store.get(f"{key}/dialed")
        return f"cannot listen on {addr}: {exc}"
    with server:
        store.set(f"{key}/port", str(server.getsockname()[1]))
        deadline = time.monotonic() + connect_timeout + 10
        while time.monotonic() < deadline:
            if select.select([server], [], [], 0.2)[0]:
                conn, _ = server.accept()
                store.get(f"{key}/dialed")
                return conn
            if store.check([f"{key}/dialed"]):
                break  # the other side gave up
        why = store.get(f"{key}/dialed").decode()
        return why if why != "ok" else "the connection never arrived"


def _dial(store, key: str, local: str, remote: str, connect_timeout: float
          ) -> socket.socket | str:
    """Connect from ``local`` to the other side's listener on ``remote``: the socket, or why not."""
    port = int(store.get(f"{key}/port"))
    if port == 0:
        store.set(f"{key}/dialed", "the other side could not listen")
        return f"cannot listen on {remote}"
    try:
        conn = socket.create_connection((remote, port), timeout=connect_timeout,
                                        source_address=(local, 0))
    except OSError as exc:
        why = "timed out" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
        why = f"{local} could not connect to {remote}: {why}"
        store.set(f"{key}/dialed", why)
        return why
    store.set(f"{key}/dialed", "ok")
    return conn


def _measure_pair(store, key: str, role: str, pair: dict[str, Any], connect_timeout: float,
                  seconds: float) -> None:
    """Connect both ways between one pair of interfaces; measure over the first connection.

    First the worker connects to the master's listener (round trips, then data
    worker to master, then master to worker), then the master connects to the
    worker's: a Gloo pair connects in whichever direction its addresses decide.
    Fills in ``pair`` on both sides alike.
    """
    master, worker = pair["master_address"], pair["worker_address"]
    if role == "master":
        conn = _accept(store, f"{key}/to-master", master, connect_timeout)
        if isinstance(conn, socket.socket):
            try:
                with conn:
                    conn.settimeout(seconds + 60)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    _rtt_server(conn)
                    store.set(f"{key}/w2m", str(round(_receive_rate(conn, seconds), 1)))
                    _send_for(conn, seconds)
                    conn.recv(1)  # the worker closes once it has read everything
            except OSError as exc:
                conn = f"the connection to {master} broke: {type(exc).__name__}: {exc}"
        back = _dial(store, f"{key}/to-worker", master, worker, connect_timeout)
    else:
        conn = _dial(store, f"{key}/to-master", worker, master, connect_timeout)
        if isinstance(conn, socket.socket):
            try:
                with conn:
                    conn.settimeout(seconds + 60)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    store.set(f"{key}/rtt", str(round(_rtt_client(conn), 3)))
                    _send_for(conn, seconds)
                    store.set(f"{key}/m2w", str(round(_receive_rate(conn, seconds), 1)))
            except OSError as exc:
                conn = f"the connection to {master} broke: {type(exc).__name__}: {exc}"
        back = _accept(store, f"{key}/to-worker", worker, connect_timeout)
    errors = [why for why in (conn, back) if isinstance(why, str)]
    if isinstance(back, socket.socket):
        back.close()
    store.set(f"{key}/{role}-errors", json.dumps(errors))
    other = "worker" if role == "master" else "master"
    errors = sorted(set(errors) | set(json.loads(store.get(f"{key}/{other}-errors"))))
    pair["connects"] = not errors
    if errors:
        pair["error"] = "; ".join(errors)
    if store.check([f"{key}/w2m", f"{key}/m2w", f"{key}/rtt"]):
        pair["MBps_worker_to_master"] = float(store.get(f"{key}/w2m"))
        pair["MBps_master_to_worker"] = float(store.get(f"{key}/m2w"))
        pair["rtt_ms"] = float(store.get(f"{key}/rtt"))


def measure(store, role: str, mine: list[tuple[str, str]], connect_timeout: float,
            seconds: float) -> list[dict[str, Any]]:
    """Every routed pair of interfaces, one at a time, the same on both sides."""
    other = "worker" if role == "master" else "master"
    store.set(f"preflight/{role}/candidates", json.dumps(mine))
    theirs = [tuple(c) for c in json.loads(store.get(f"preflight/{other}/candidates"))]
    store.set(f"preflight/{role}/pairs", json.dumps(routed_pairs(mine, theirs, role)))
    found = {tuple(p) for p in json.loads(store.get(f"preflight/{other}/pairs"))}
    found |= set(routed_pairs(mine, theirs, role))
    masters = dict(mine if role == "master" else theirs)
    workers = dict(theirs if role == "master" else mine)
    pairs = [{"master": m, "master_address": masters[m], "worker": w,
              "worker_address": workers[w]} for m, w in sorted(found)]
    for i, pair in enumerate(pairs):
        _measure_pair(store, f"preflight/pair/{i}", role, pair, connect_timeout, seconds)
    return pairs


def check_gloo(store, rank: int, ifname: str, timeout: float) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = ifname
    start = time.monotonic()
    try:
        dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                                timeout=timedelta(seconds=timeout))
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
            f"Gloo could not connect the two machines on {ifname}: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0] if str(exc) else ''}. The rendezvous port may be "
            "open while Gloo's own (random) ports are blocked. " + HINT_FLAT_NETWORK
        ) from exc
    return {"seconds": round(time.monotonic() - start, 3), "torch": torch.__version__,
            "GLOO_SOCKET_IFNAME": ifname}


def open_store(addr: str, port: int, role: str, timeout: float):
    import torch.distributed as dist

    try:
        return dist.TCPStore(addr, port, 2, role == "master", timedelta(seconds=timeout),
                             wait_for_workers=False)
    except Exception as exc:
        raise PreflightError(f"could not open the preflight's store on {addr}:{port}: "
                             f"{type(exc).__name__}: {exc}. " + HINT_FLAT_NETWORK) from exc


def resolve_role(role: str) -> str:
    if role != "auto":
        return role
    procid = os.environ.get("SLURM_PROCID")
    if procid is None:
        raise SystemExit("--role auto needs SLURM_PROCID (run it under srun)")
    return "master" if procid == "0" else "worker"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--role", choices=["master", "worker", "auto"], required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--port", type=int, required=True,
                        help="port for the preflight's own store (not torchrun's)")
    parser.add_argument("--ifname", default=os.environ.get("GLOO_SOCKET_IFNAME") or None)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--connect-timeout", type=float, default=3,
                        help="seconds to wait for each connection between two interfaces")
    parser.add_argument("--bandwidth-seconds", type=float, default=2,
                        help="seconds of data each way per pair of interfaces")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    role = resolve_role(args.role)

    record: dict[str, Any] = {"script": "scripts/gpu/preflight.py", "role": role,
                              "master_addr": args.master_addr, "port": args.port,
                              "hostname": socket.gethostname(), "stages": []}

    def stage(name: str, fn, shown=None) -> Any:
        """Run one check; record ``shown(result)`` (or the result) in its stage."""
        try:
            result = fn()
            record["stages"].append({"name": name, "ok": True,
                                     "result": shown(result) if shown else result})
            print(f"preflight [{role}] {name}: ok", flush=True)
            return result
        except PreflightError as exc:
            record["stages"].append({"name": name, "ok": False, "error": str(exc)})
            raise
        except Exception as exc:  # e.g. the other side died and the store timed out
            error = f"{type(exc).__name__}: {exc}"
            record["stages"].append({"name": name, "ok": False, "error": error})
            raise PreflightError(f"{name}: {error}") from exc

    try:
        master_ip = stage("resolve", lambda: check_resolve(args.master_addr))
        stage("interface", lambda: check_interface(args.ifname))
        if role == "worker":
            stage("tcp", lambda: check_tcp(args.master_addr, args.port, args.timeout))
        store = stage("store", lambda: open_store(args.master_addr, args.port, role,
                                                   args.timeout),
                      shown=lambda _: f"{args.master_addr}:{args.port}")
        mine = stage("interfaces", lambda: candidates(
            interface_addresses(), args.ifname, master_ip.startswith("127.")))
        pairs = stage("reachability", lambda: measure(
            store, role, mine, args.connect_timeout, args.bandwidth_seconds),
            shown=lambda found: [describe(pair) for pair in found])
        record["pairs"] = pairs
        for pair in pairs:
            print(f"preflight [{role}]   {describe(pair)}", flush=True)
        chosen = stage("choose", lambda: choose(pairs), shown=describe)
        other = "worker" if role == "master" else "master"
        record["chosen_ifname"] = chosen[role]
        record["chosen_address"] = chosen[f"{role}_address"]
        record["peer_ifname"] = chosen[other]
        record["bandwidth"] = {
            "MBps_in": chosen[f"MBps_{other}_to_{role}"],
            "MBps_out": chosen[f"MBps_{role}_to_{other}"],
            "rtt_ms": chosen["rtt_ms"],
            "seconds_each_way": args.bandwidth_seconds,
            "streams": 1,
        }
        print(f"preflight [{role}] GLOO_SOCKET_IFNAME={chosen[role]} (the {other} uses "
              f"{chosen[other]})", flush=True)
        stage("gloo", lambda: check_gloo(store, 0 if role == "master" else 1,
                                         chosen[role], args.timeout))
        record["ok"] = True
    except PreflightError as exc:
        record["ok"] = False
        print(f"\npreflight [{role}] FAILED: {exc}\n", file=sys.stderr, flush=True)
    if args.out:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpukit import write_json
        write_json(args.out / f"preflight-{role}.json", record)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
