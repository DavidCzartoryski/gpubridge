"""The comparison matrix: gpubridge against native NCCL/RCCL and flat Gloo, run by run.

    python scripts/gpu/bench_matrix.py RESULTS [RESULTS ...] [--out DIR] [--pending RUNS=WHY]

explorer/08_bench_matrix.sbatch writes one directory per run, ``<job>/matrix/<run>/``,
holding bench_all_reduce.py's bench.json: 2 ranks, one GPU each on two nodes,
all_reduce of float32 from 1 KB to 1 GB, CUDA-event timing, and the phases of
every gpubridge run. The runs:

- ``A1``: V100 + V100, native NCCL on the transport NCCL picks (InfiniBand if any).
- ``A2``: the same over TCP: NCCL_IB_DISABLE=1, on the interface the bridge uses.
- ``B1``, ``B2``: the same with two MI50s and RCCL.
- ``C``: V100 + V100 through gpubridge's bridge, as two NVIDIA islands
  (GPUBRIDGE_SPLIT_TEST=node): gpubridge's own cost, with no vendor mixing.
  ``C-amd``: the same with two MI50s.
- ``D-<variant>``: V100 + MI50 through gpubridge: ``default``, ``pinned``,
  ``pipelined-<chunk>``, ``sharded``, ``bf16``, ``fp8``, and ``default-repeat``
  (the default again at the end of the job, for the run-to-run spread).
- ``E``: V100 + MI50, one flat Gloo group of both ranks (the flat-gloo policy).
  ``E-nvidia`` and ``E-amd``: the same on the V100 and MI50 pairs.

Each pair is its own job on its own nodes: A1, A2 and C on two V100 nodes, B1,
B2 and C-amd on two MI50 nodes, D and E on a V100 node and an MI50 node. So a
ratio between runs of different jobs also compares their nodes' networks:
matrix.md lists each job's preflight (interfaces, MB/s each way, round trip)
and marks every such ratio.

Writes matrix.md, matrix.json and matrix.csv to --out (default: the first
RESULTS). For each size range (under 256 KB, 256 KB to under 64 MB, 64 MB and
up), every run's median latency (slowest rank), bus bandwidth, and its time as
a multiple of A2's (TCP on the bridge's interface) and A1's (the best native)
at the same size; for runs with a bridge, the bridge phase next to what the
job's preflight predicts (predicted_ms). Then every gpubridge run's phases,
what each run was (from its records, not its name), and the network each job
saw: InfiniBand devices and link layers on each node, and the transport
NCCL/RCCL chose in each run (``NET/IB`` or ``NET/Socket``, with how many
channel connections).

--pending A1,A2=reason lists runs that aren't in the results, and why.

Safe to publish: nodes are named by vendor, never by host, and addresses are
dropped (interfaces keep their names); paths are redacted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation_tables import predicted_bridge_ms

#: Which op of a run's bench.json is the run, by the run name's letter.
OP_OF = {"A": "native", "B": "native", "C": "gpubridge", "D": "gpubridge", "E": "flat-gloo"}
#: Run names in table order; anything else sorts after them, by name.
ORDER = ("A1", "A2", "B1", "B2", "C", "C-amd", "D-default", "D-pinned", "D-pipelined", "D-sharded",
         "D-bf16", "D-fp8", "D-default-repeat", "E", "E-nvidia", "E-amd")
#: Size ranges: (title, smallest, end) in bytes, the end excluded; together they cover
#: every size.
RANGES = (("Small: under 256 KB (latency)", 0, 256 * 2**10),
          ("Medium: 256 KB to under 64 MB", 256 * 2**10, 64 * 2**20),
          ("Large: 64 MB and up (bandwidth)", 64 * 2**20, 2**62))
#: Bytes per element of the dtypes bench_all_reduce.py measures.
ELEMENT = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
GUIDE = """\
How to read it:

- **A1 against A2 is the network**: the same NCCL on the same two GPUs in one job,
  on the transport NCCL picks (the Network section says which) and then over TCP on
  the bridge's interface.
- **A2 against C is gpubridge's own cost**: the same job, the same two V100s and the
  same interface; C takes gpubridge's path (each GPU an island, the sum staged
  through host memory and all_reduced by Gloo over one connection) instead of
  NCCL's socket transport, which may open several (see its channel connections).
  B2 against C-amd is the same on two MI50s.
- **C against D is the vendor mixing, across jobs**: the same bridge with an MI50 and
  RCCL on one side, but on other nodes. A ratio marked † divides by a run of
  another job, so check the two jobs' networks (Jobs) before putting a difference
  down to the vendor.
- **D's variants against D-default** are what each policy buys on this pair, and
  **D-default against E** what gpubridge does better than one flat Gloo group.
  With one GPU per island, `sharded-bridge` has one link (k = 1), so it runs
  reduce-bridge-broadcast's steps.
- **Measured / predicted** compares the bridge phase with what the job's preflight
  (one TCP stream, its slower direction, and the round trip) predicts: a ring
  all_reduce, the whole tensor each way at once; with compression, each leader's
  payload broadcast in turn, one way at a time. The bridge phase also includes
  the copies between GPU and host, which the prediction doesn't.
"""


@dataclass
class Run:
    name: str
    job: str
    bench: dict[str, Any]
    preflight: dict[str, Any] | None
    """The job's preflight-master.json; ``worker`` holds preflight-worker.json."""
    worker: dict[str, Any] | None = None
    nccl: dict[str, list[str]] = field(default_factory=dict)
    """NCCL/RCCL's network lines in this run's step, per node side."""

    @property
    def op(self) -> str:
        return OP_OF.get(self.name[:1], "gpubridge")

    def rows(self) -> dict[int, dict[str, Any]]:
        """This run's op's rows, by size."""
        return {r["bytes"]: r for r in self.bench.get("rows", []) if r["op"] == self.op}


def parse_size(text: str) -> int:
    """``4M`` and the like in bytes (0 if it isn't one)."""
    m = re.fullmatch(r"(\d+)([KMG]?)B?", text.strip().upper())
    return int(m[1]) * {"": 1, "K": 2**10, "M": 2**20, "G": 2**30}[m[2]] if m else 0


def order_key(name: str) -> tuple[int, int, str]:
    """ORDER's position of ``name``, or of the entry it extends (D-pipelined-1M); then
    a pipelined run's chunk size, so 1M comes before 16M."""
    if name in ORDER:
        return ORDER.index(name), 0, name
    for i, prefix in enumerate(ORDER):
        if name.startswith(prefix + "-"):
            return i, parse_size(name[len(prefix) + 1:]), name
    return len(ORDER), 0, name


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_runs(roots: list[Path]) -> list[Run]:
    runs = []
    for root in roots:
        for path in sorted(root.rglob("matrix/*/bench.json")):
            bench = _load(path)
            if bench is None:
                continue
            job_dir = path.parent.parent.parent
            preflight = _load(job_dir / "preflight" / "preflight-master.json")
            run = Run(path.parent.name, job_dir.name, bench,
                      preflight if preflight and "bandwidth" in preflight else None,
                      _load(job_dir / "preflight" / "preflight-worker.json"))
            run.nccl = nccl_lines(job_dir / "nccl" / f"matrix_{run.name}")
            runs.append(run)
    runs.sort(key=lambda r: (order_key(r.name), r.job))
    return runs


# ---- what each run was, from its records ------------------------------------------------

def _env(bench: dict[str, Any]) -> dict[str, str]:
    return (bench.get("environment") or {}).get("env") or {}


def described(run: Run) -> str:
    """What the run measured, from its own records."""
    info, env = run.bench.get("run", {}), _env(run.bench)
    vendors = "+".join(i["vendor"] for i in info.get("islands", [])) or "?"
    if run.op == "native":
        library = (run.bench.get("environment") or {}).get("collective_library") or "native"
        parts = [f"native {library.upper()}"]
        for name in ("NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_NET"):
            if env.get(name):
                parts.append(f"{name}={env[name]}")
        return ", ".join(parts)
    if run.op == "flat-gloo":
        return f"flat-gloo over {info.get('world_size')} ranks ({vendors} islands)"
    parts = [f"gpubridge {info.get('policy')}", f"islands {vendors}"]
    if info.get("split_test"):
        parts.append(f"split test ({info['split_test']})")
    if info.get("pinned_staging"):
        parts.append("pinned staging")
    if "pipelined" in str(info.get("policy")):
        parts.append(f"chunk {env.get('GPUBRIDGE_CHUNK_BYTES') or '4M (default)'}")
    if env.get("GPUBRIDGE_BRIDGE_COMPRESSION"):
        parts.append(f"{env['GPUBRIDGE_BRIDGE_COMPRESSION']} compression")
    return ", ".join(parts)


# ---- the network --------------------------------------------------------------------------

_ADDRESS = re.compile(r"(\b\d{1,3}(?:\.\d{1,3}){3}\b|\[[0-9a-f]*:[0-9a-f:]+\]"
                      r"|\b[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}\b)", re.I)
#: NCCL/RCCL INFO lines about the network: the plugin's choice, the interfaces, the
#: transport between ranks, and the settings that steer them.
_NET = re.compile(r"NCCL INFO (NET/\S+ : .*|Using network \S+|Bootstrap ?: Using .*"
                  r"|Channel \d+/\d+ : .* via NET/\S+"
                  r"|NCCL_(?:IB_DISABLE|SOCKET_IFNAME|NET|IB_HCA) set by environment to \S+)")


def no_addresses(text: str) -> str:
    """``text`` with IPv4 and IPv6 addresses (and the %scope or <n> after them) dropped."""
    return re.sub(r":?" + _ADDRESS.pattern + r"(%[\w.-]+)?(<\d+>)?", "", text,
                  flags=re.I).strip()


def nccl_lines(directory: Path) -> dict[str, list[str]]:
    """NCCL/RCCL's lines about the network, per file (one per rank), addresses dropped.

    Keyed by the library: RCCL's logs give the ROCr and HIP runtime versions.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.log")) if directory.is_dir() else []:
        text = path.read_text(errors="replace")
        side = "rccl" if re.search(r"ROCr version|Hipruntime version|RCCL version", text) \
            else "nccl"
        lines, channels = [], {}
        for match in _NET.finditer(text):
            line = no_addresses(match[1])
            # Channels: one line per channel and direction, counted by transport.
            if via := re.match(r"Channel .* via (NET/\S+)", line):
                channels[via[1]] = channels.get(via[1], 0) + 1
                continue
            if line not in lines:
                lines.append(line)
        lines += [f"{count} channel connections via {net} (one rank's log)"
                  for net, count in channels.items()]
        found.setdefault(side, [])
        found[side] += [line for line in lines if line not in found[side]]
    return found


def infiniband(side_dir: Path) -> dict[str, Any]:
    """What network/<side>/ holds: InfiniBand devices (/sys/class/infiniband), and each
    port's link layer, state and rate (ibstat, else ibv_devinfo)."""
    listing = side_dir / "sys_class_infiniband.txt"
    devices = listing.read_text().split() if listing.exists() else []
    ports: dict[tuple[str, str], dict[str, str]] = {}

    def port(device: str, number: str) -> dict[str, str]:
        return ports.setdefault((device, number), {"device": device, "port": number})

    tools = {}
    for name in ("ibstat", "ibv_devinfo"):
        path = side_dir / f"{name}.txt"
        text = path.read_text(errors="replace") if path.exists() else ""
        tools[name] = bool(text) and "not installed" not in text
    text = (side_dir / "ibstat.txt").read_text(errors="replace") if tools["ibstat"] else ""
    device = number = None
    for line in text.splitlines():
        if m := re.match(r"CA '([^']+)'", line):
            device, number = m[1], None
        elif m := re.match(r"\s*Port (\d+):", line):
            number = m[1]
        elif device and number and (m := re.match(r"\s*(State|Rate|Link layer):\s*(.+)", line)):
            key = {"State": "state", "Rate": "rate", "Link layer": "link_layer"}[m[1]]
            port(device, number)[key] = m[2].strip()
    text = (side_dir / "ibv_devinfo.txt").read_text(errors="replace") \
        if tools["ibv_devinfo"] else ""
    device = number = None
    for line in text.splitlines():
        if m := re.match(r"\s*hca_id:\s*(\S+)", line):
            device, number = m[1], None
        elif m := re.match(r"\s*port:\s*(\d+)", line):
            number = m[1]
        elif device and number and (m := re.match(r"\s*(state|link_layer):\s*(.+)", line)):
            port(device, number).setdefault(m[1], m[2].strip())
    return {"devices": devices, "ports": list(ports.values()), "tools": tools}


def network(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for side_dir in sorted(root.rglob("network/*")):
            if not side_dir.is_dir():
                continue
            job_dir = side_dir.parent.parent
            ib = infiniband(side_dir)
            # net_check.sbatch's one-rank initializations, net_default and net_tcp.
            nccl = {step.name: nccl_lines(step)
                    for step in sorted((job_dir / "nccl").glob("net_*")) if step.is_dir()}
            rows.append({"job": job_dir.name, "side": side_dir.name, **ib,
                         "interfaces": interfaces(side_dir / "ip_addr.txt"), "nccl": nccl})
    return rows


def interfaces(path: Path) -> list[str]:
    """Interface names and states from ``ip -br addr``, without addresses."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] != "lo":
            out.append(f"{parts[0]} {parts[1]}")
    return out


# ---- the matrix -----------------------------------------------------------------------------

def human(nbytes: int) -> str:
    for unit, size in (("G", 2**30), ("M", 2**20), ("K", 2**10)):
        if nbytes >= size and nbytes % size == 0:
            return f"{nbytes // size}{unit}B"
    return f"{nbytes}B"


def latency(us: float | None) -> str:
    if us is None:
        return "-"
    if us >= 1e6:
        return f"{us / 1e6:.3g} s"
    return f"{us / 1e3:.3g} ms" if us >= 1e3 else f"{us:.3g} us"


def compression(run: Run) -> str:
    return _env(run.bench).get("GPUBRIDGE_BRIDGE_COMPRESSION", "").strip().lower()


def predicted_ms(run: Run, row: dict[str, Any], mbps: float, rtt_ms: float) -> float:
    """The bridge phase the preflight's numbers predict for one benchmark row.

    Uncompressed, the leaders' Gloo all_reduce is a ring
    (validation_tables.predicted_bridge_ms: the whole tensor each way at once, for
    two islands). Compressed, every leader broadcasts its payload
    (gpubridge.compression.payload_nbytes) to the others in turn
    (BridgeCompressor.all_reduce), so the payloads cross one after another: for p
    leaders, p broadcasts of the payload to p - 1 others, and p - 1 round trips.
    """
    islands = row.get("islands") or 2
    mode = compression(run)
    if not mode:
        return predicted_bridge_ms(row["bytes"], islands, mbps, rtt_ms)
    from gpubridge.compression import payload_nbytes  # torch, only when compressed

    payload = payload_nbytes(mode, row["bytes"] // ELEMENT.get(row.get("dtype", ""), 4))
    return (islands - 1) * rtt_ms + islands * (islands - 1) * payload / (mbps * 1e6) * 1e3


def matrix_rows(runs: list[Run]) -> list[dict[str, Any]]:
    by_name = {run.name: run for run in reversed(runs)}  # the first of a name wins
    baselines = {name: by_name[name].rows() if name in by_name else {} for name in ("A1", "A2")}
    jobs = {name: by_name[name].job for name in ("A1", "A2") if name in by_name}
    rows = []
    for run in runs:
        for nbytes, r in sorted(run.rows().items()):
            row = {"run": run.name, "job": run.job, "what": described(run), "bytes": nbytes,
                   "median_us": r["median_us"], "p90_us": r.get("p90_us"),
                   "busbw_GBps": r.get("busbw_GBps"), "trials": r.get("trials"),
                   "timer": r.get("timer")}
            for base in ("A2", "A1"):
                b = baselines[base].get(nbytes)
                row[f"x_{base}"] = round(r["median_us"] / b["median_us"], 2) if b else None
                # The baseline ran on other nodes: the ratio also compares two networks.
                row[f"x_{base}_other_job"] = bool(b) and jobs[base] != run.job
            phases = r.get("phases_ms") or {}
            row["phases_ms"] = phases
            bridge = phases.get("bridge")
            if bridge is not None and run.preflight is not None:
                bandwidth = run.preflight["bandwidth"]
                mbps = min(bandwidth["MBps_in"], bandwidth["MBps_out"])
                predicted = predicted_ms(run, r, mbps, bandwidth["rtt_ms"])
                row.update(predicted_bridge_ms=round(predicted, 3), bridge_ms=bridge,
                           measured_over_predicted=round(bridge / predicted, 2))
            rows.append(row)
    return rows


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return "-" if value is None else str(value).replace("|", "\\|")

    return (["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
            + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def ratio(value: float | None, other_job: bool = False) -> str | None:
    """A multiple as text; † marks one whose baseline ran in another job."""
    return None if value is None else f"{value:.2f}x" + (" †" if other_job else "")


def job_rows(runs: list[Run]) -> list[dict[str, Any]]:
    """Each job's pair and preflight: the network its runs crossed."""
    seen: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.job in seen:
            continue
        islands = run.bench.get("run", {}).get("islands", [])
        row: dict[str, Any] = {"job": run.job, "vendors": "+".join(sorted(
            {i["vendor"] for i in islands})) or "?"}
        if run.preflight is not None:
            bandwidth = run.preflight["bandwidth"]
            row.update(master=run.preflight.get("chosen_ifname"),
                       worker=(run.worker or {}).get("chosen_ifname")
                       or run.preflight.get("peer_ifname"),
                       MBps_master_to_worker=bandwidth["MBps_out"],
                       MBps_worker_to_master=bandwidth["MBps_in"], rtt_ms=bandwidth["rtt_ms"])
        seen[run.job] = row
    return list(seen.values())


def markdown(runs: list[Run], rows: list[dict[str, Any]], net: list[dict[str, Any]],
             pending: dict[str, str]) -> str:
    lines = ["# all_reduce comparison matrix", "",
             "Made by `scripts/gpu/bench_matrix.py`. 2 ranks, one GPU each on two nodes; "
             "float32 all_reduce; the median latency of the slowest rank, timed with CUDA "
             "events. Bus bandwidth as nccl-tests computes it (with 2 ranks, bytes / time). "
             "x A2 and x A1: the run's time over that baseline's at the same size (lower is "
             "faster); † marks one whose baseline ran in another job, on other nodes.", ""]
    present = {run.name for run in runs}
    missing = {name: why for name, why in pending.items() if name not in present}
    if missing:
        lines += ["Not in these results: " + "; ".join(f"**{name}** ({why})"
                                                    for name, why in missing.items()) + ".", ""]
    if not {"A1", "A2"} & present:
        lines += ["Without A1 and A2, the x A2 and x A1 columns are empty.", ""]
    lines += [GUIDE, "## Jobs", "",
              "Each job's nodes and the network between them, from its preflight (one TCP "
              "stream from Python: a lower bound on fast links).", ""]
    lines += table(["Job", "Islands", "Interface (master, worker)", "MB/s master to worker",
                    "MB/s worker to master", "Round trip (ms)"],
                   [[j["job"], j["vendors"], f"{j.get('master') or '-'}, {j.get('worker') or '-'}",
                     j.get("MBps_master_to_worker"), j.get("MBps_worker_to_master"),
                     j.get("rtt_ms")] for j in job_rows(runs)])
    lines += ["", "## Runs", ""]
    lines += table(["Run", "Job", "What it ran", "Sizes", "Timer"],
                   [[run.name, run.job, described(run), len(run.rows()),
                     ", ".join(sorted({r.get("timer", "?") for r in run.rows().values()}))]
                    for run in runs])
    for title, lo, hi in RANGES:
        chosen = [r for r in rows if lo <= r["bytes"] < hi]
        if not chosen:
            continue
        chosen.sort(key=lambda r: (r["bytes"], order_key(r["run"]), r["job"]))
        lines += ["", f"## {title}", ""]
        lines += table(
            ["Size", "Run", "Latency", "Bus bandwidth (GB/s)", "x A2", "x A1",
             "Bridge predicted (ms)", "Bridge measured (ms)", "Measured / predicted"],
            [[human(r["bytes"]), r["run"], latency(r["median_us"]), r["busbw_GBps"],
              ratio(r["x_A2"], r["x_A2_other_job"]), ratio(r["x_A1"], r["x_A1_other_job"]),
              r.get("predicted_bridge_ms"),
              r.get("bridge_ms"), ratio(r.get("measured_over_predicted"))] for r in chosen])
    phased = [r for r in rows if r["phases_ms"]]
    if phased:
        names = list(dict.fromkeys(name for r in phased for name in r["phases_ms"]))
        lines += ["", "## Phases of the gpubridge runs", "",
                  "Median ms of each phase on the slowest rank that ran it.", ""]
        lines += table(["Run", "Size", *names],
                       [[r["run"], human(r["bytes"]), *(r["phases_ms"].get(n) for n in names)]
                        for r in sorted(phased, key=lambda r: (order_key(r["run"]),
                                                               r["bytes"]))])
    if net or any(run.nccl for run in runs):
        lines += ["", "## Network", ""]
    if net:
        lines += table(
            ["Job", "Node", "InfiniBand devices", "Ports (device:port link layer, state, rate)",
             "Interfaces (ip -br addr)"],
            [[n["job"], n["side"], ", ".join(n["devices"]) or "none",
              "; ".join(f"{p['device']}:{p['port']} {p.get('link_layer', '?')}, "
                        f"{p.get('state', '?')}" + (f", {p['rate']} Gb/s" if p.get("rate") else "")
                        for p in n["ports"])
              or ("none" if any(n["tools"].values()) else "ibstat and ibv_devinfo not installed"),
              ", ".join(n["interfaces"]) or "-"] for n in net])
        lines.append("")
    chose = [(f"{n['side']} node, {step}", side, line) for n in net
             for step, found in n.get("nccl", {}).items() for side, said in found.items()
             for line in said]
    chose += [(run.name, side, line) for run in runs for side, found in run.nccl.items()
              for line in found]
    if chose:
        lines += ["What NCCL and RCCL said about the network (NCCL_DEBUG=INFO). `net_default` "
                  "and `net_tcp` are one rank initializing the library on its own (as chosen, "
                  "then with NCCL_IB_DISABLE=1); in a run, `via NET/...` is the transport "
                  "between its two ranks:", ""]
        lines += table(["Run", "Library", "Line"],
                       [[where, side.upper(), line] for where, side, line in chose])
    from gpukit import redact  # here, not at the top: reading results needs no GPU

    return redact("\n".join(lines) + "\n")


def parse_pending(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        names, _, why = item.partition("=")
        for name in names.split(","):
            if name.strip():
                out[name.strip()] = why.strip() or "not run"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", type=Path, nargs="+", help="results directories")
    parser.add_argument("--out", type=Path, help="where to write (default: the first path)")
    parser.add_argument("--pending", action="append", default=[], metavar="RUNS=WHY",
                        help="runs not in the results and why, e.g. A1,A2='waiting on access'")
    args = parser.parse_args(argv)
    runs = find_runs(args.paths)
    net = network(args.paths)
    if not runs and not net:
        print(f"no matrix runs (matrix/<run>/bench.json) or network records under "
              f"{', '.join(map(str, args.paths))}", file=sys.stderr)
        return 1
    rows = matrix_rows(runs)
    pending = parse_pending(args.pending)
    text = markdown(runs, rows, net, pending)
    out = args.out or args.paths[0]
    out.mkdir(parents=True, exist_ok=True)
    from gpukit import write_json

    write_json(out / "matrix.json", {
        "script": "scripts/gpu/bench_matrix.py", "rows": rows, "network": net,
        "jobs": job_rows(runs),
        "nccl": {run.name: run.nccl for run in runs},
        "runs": [{"run": run.name, "job": run.job, "what": described(run),
                  "environment": {k: (run.bench.get("environment") or {}).get(k)
                                  for k in ("torch", "torch_cuda", "torch_hip",
                                            "collective_library", "nccl_version",
                                            "driver_version", "git_commit", "git_dirty")},
                  "preflight": (run.preflight or {}).get("bandwidth")} for run in runs],
        "pending": pending,
    })
    with open(out / "matrix.csv", "w", newline="") as fh:
        fields = ["run", "job", "what", "bytes", "median_us", "p90_us", "busbw_GBps", "x_A2",
                  "x_A1", "predicted_bridge_ms", "bridge_ms", "measured_over_predicted",
                  "trials", "timer"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "matrix.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
