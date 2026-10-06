"""Run the mixed-build rendezvous experiments inside the scripts/Dockerfile image.

Each experiment starts two torchrun "nodes" on localhost, each from a different
venv (a CUDA build or a ROCm build of PyTorch), joined by a c10d rendezvous.
Every rank runs scripts/rendezvous_probe.py in CPU-only mode. Results go to
results/mixed_build/: one directory of logs per experiment, plus summary.json
and summary.md.

    python scripts/mixed_build_rendezvous.py                 # every experiment
    python scripts/mixed_build_rendezvous.py --group 2.14.1  # one group
    python scripts/mixed_build_rendezvous.py --env-report    # versions and sizes only
"""

from __future__ import annotations

import argparse
import contextlib
import html.parser
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "scripts" / "rendezvous_probe.py"
RESULTS = REPO / "results" / "mixed_build"
ENVS_DIR = Path("/envs")

# Must match scripts/Dockerfile.
ENVS = {
    "cu-2.14.1": ("2.14.1+cu130", "https://download.pytorch.org/whl/cu130"),
    "rocm-2.14.1": ("2.14.1+rocm7.2", "https://download.pytorch.org/whl/rocm7.2"),
    "cu-2.13.0": ("2.13.0+cu130", "https://download.pytorch.org/whl/cu130"),
    "cu-2.9.1": ("2.9.1+cu128", "https://download.pytorch.org/whl/cu128"),
    "rocm-2.9.1": ("2.9.1+rocm6.4", "https://download.pytorch.org/whl/rocm6.4"),
}


@dataclass(frozen=True)
class Node:
    env: str
    nproc: int


@dataclass(frozen=True)
class Experiment:
    id: str
    group: str
    title: str
    host: Node
    """Started first, and hosts the rendezvous store (is_host=1)."""
    guest: Node


def _exp(id: str, group: str, title: str, host: tuple[str, int], guest: tuple[str, int]):
    return Experiment(id, group, title, Node(*host), Node(*guest))


EXPERIMENTS = [
    _exp("2.14.1-1+1", "2.14.1", "matched 2.14.1, 1 CUDA + 1 ROCm",
         ("cu-2.14.1", 1), ("rocm-2.14.1", 1)),
    _exp("2.14.1-2+2", "2.14.1", "matched 2.14.1, 2 CUDA + 2 ROCm, ROCm hosts store",
         ("rocm-2.14.1", 2), ("cu-2.14.1", 2)),
    _exp("2.14.1-3+1", "2.14.1", "matched 2.14.1, 3 CUDA + 1 ROCm",
         ("cu-2.14.1", 3), ("rocm-2.14.1", 1)),
    _exp("2.9.1-1+1", "2.9.1", "matched 2.9.1, 1 CUDA + 1 ROCm",
         ("cu-2.9.1", 1), ("rocm-2.9.1", 1)),
    _exp("2.9.1-2+2", "2.9.1", "matched 2.9.1, 2 CUDA + 2 ROCm, ROCm hosts store",
         ("rocm-2.9.1", 2), ("cu-2.9.1", 2)),
    _exp("2.9.1-3+1", "2.9.1", "matched 2.9.1, 3 CUDA + 1 ROCm",
         ("cu-2.9.1", 3), ("rocm-2.9.1", 1)),
    _exp("minor-1+1-cu-hosts", "mismatch", "CUDA 2.13.0 + ROCm 2.14.1, CUDA hosts store",
         ("cu-2.13.0", 1), ("rocm-2.14.1", 1)),
    _exp("minor-1+1-rocm-hosts", "mismatch", "CUDA 2.13.0 + ROCm 2.14.1, ROCm hosts store",
         ("rocm-2.14.1", 1), ("cu-2.13.0", 1)),
    _exp("minor-2+2", "mismatch", "CUDA 2.13.0 + ROCm 2.14.1, 2 + 2",
         ("cu-2.13.0", 2), ("rocm-2.14.1", 2)),
    _exp("wide-cu2.14-rocm2.9", "mismatch", "CUDA 2.14.1 + ROCm 2.9.1",
         ("cu-2.14.1", 1), ("rocm-2.9.1", 1)),
    _exp("wide-cu2.9-rocm2.14", "mismatch", "CUDA 2.9.1 + ROCm 2.14.1",
         ("cu-2.9.1", 1), ("rocm-2.14.1", 1)),
    _exp("control-cu2.13-cu2.14", "control", "CUDA 2.13.0 + CUDA 2.14.1 (same vendor)",
         ("cu-2.13.0", 1), ("cu-2.14.1", 1)),
]

STAGES = ["init", "all_reduce", "broadcast", "barrier", "destroy"]


def env_python(env: str) -> Path:
    return ENVS_DIR / env / "bin" / "python"


def wait_for_port(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Wait until the host's rendezvous store accepts connections, or the host exits."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def launch(exp: Experiment, node: Node, is_host: bool, port: int, out: Path, args) -> tuple:
    role = "host" if is_host else "guest"
    cmd = [
        str(ENVS_DIR / node.env / "bin" / "torchrun"),
        "--nnodes=2",
        f"--nproc-per-node={node.nproc}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=127.0.0.1:{port}",
        f"--rdzv-id={exp.id}",
        f"--rdzv-conf=is_host={int(is_host)},join_timeout={int(args.timeout)}",
        "--max-restarts=0",
        str(PROBE),
        "--out", str(out / "ranks"),
        "--timeout", str(int(args.timeout * 0.6)),
        "--hang-dump", str(int(args.timeout * 0.5)),
    ]
    env = {k: v for k, v in os.environ.items() if k != "GPUBRIDGE_VENDOR"}
    env |= {"GPUBRIDGE_CPU_ONLY": "1", "OMP_NUM_THREADS": "1", "PROBE_ENV": node.env}
    log = open(out / f"{role}-{node.env}.log", "w")  # noqa: SIM115
    log.write(" ".join(cmd) + "\n\n")
    log.flush()
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=REPO
    )
    return proc, log


def kill(proc: subprocess.Popen) -> bool:
    """Kill a torchrun agent and its workers if still running. Return True if it was."""
    if proc.poll() is not None:
        return False
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    return True


def run_once(exp: Experiment, port: int, out: Path, args) -> dict:
    """Run one attempt of an experiment and return its raw outcome."""
    if out.exists():
        shutil.rmtree(out)
    (out / "ranks").mkdir(parents=True)
    start = time.monotonic()
    host, host_log = launch(exp, exp.host, True, port, out, args)
    guest = guest_log = None
    timed_out = False
    killed = []
    try:
        if not wait_for_port(port, host, args.timeout):
            timed_out = host.poll() is None
        else:
            guest, guest_log = launch(exp, exp.guest, False, port, out, args)
        procs = [p for p in (host, guest) if p is not None]
        deadline = start + args.timeout
        first_exit = None
        while any(p.poll() is None for p in procs):
            now = time.monotonic()
            if first_exit is None and any(p.poll() is not None for p in procs):
                first_exit = now
            # Once one node has exited, give the other a grace period, not the full timeout.
            if now > deadline or (first_exit is not None and now - first_exit > args.grace):
                timed_out = timed_out or now > deadline
                break
            time.sleep(1)
    finally:
        for role, p in (("host", host), ("guest", guest)):
            if p is not None and kill(p):
                killed.append(role)
        for log in (host_log, guest_log):
            if log is not None:
                log.close()
    return {
        "seconds": round(time.monotonic() - start, 1),
        "timed_out": timed_out,
        "host_exit": host.returncode,
        "guest_exit": guest.returncode if guest is not None else None,
        "guest_started": guest is not None,
        "killed": killed,
    }


def evaluate(exp: Experiment, raw: dict, out: Path) -> dict:
    """Turn per-rank records into pass/fail per check."""
    world = exp.host.nproc + exp.guest.nproc
    ranks = {}
    for path in sorted((out / "ranks").glob("rank*.json")):
        record = json.loads(path.read_text())
        ranks[record["rank"]] = record

    def stage_ok(name: str) -> bool:
        return len(ranks) == world and all(
            any(s["name"] == name and s["ok"] for s in r["stages"]) for r in ranks.values()
        )

    builds = {r: rec["process"]["build_vendor"] for r, rec in ranks.items()}
    expected_counts: dict[str, int] = {}
    for node in (exp.host, exp.guest):
        vendor = "amd" if node.env.startswith("rocm") else "nvidia"
        expected_counts[vendor] = expected_counts.get(vendor, 0) + node.nproc
    maps = [tuple(p["vendor"] for p in rec.get("topology", {}).get("peers", []))
            for rec in ranks.values()]
    build_map = tuple(builds.get(r) for r in range(world))
    vendor_map_ok = (
        stage_ok("init")
        and len(set(maps)) == 1
        and maps[0] == build_map
        and {v: build_map.count(v) for v in set(build_map)} == expected_counts
    )
    checks = {
        "rendezvous": len(ranks) == world,
        "init": stage_ok("init"),
        "vendor_map": vendor_map_ok,
        "all_reduce": stage_ok("all_reduce"),
        "broadcast": stage_ok("broadcast"),
        # Every rank finished destroy() and both nodes exited without being killed.
        "shutdown": stage_ok("destroy") and raw["guest_started"] and not raw["killed"],
    }
    first_failure = next((name for name, ok in checks.items() if not ok), None)
    if first_failure is None:
        verdict = "PASS"
    elif raw["timed_out"]:
        verdict = f"TIMEOUT ({first_failure})"
    else:
        verdict = f"FAIL ({first_failure})"

    errors = {}
    for r, rec in ranks.items():
        for s in rec["stages"]:
            if s["ok"] is False and s.get("error"):
                errors[f"rank{r}:{s['name']}"] = s["error"]
            elif s["ok"] is None:
                errors[f"rank{r}:{s['name']}"] = "stuck in this stage"
    processes = {r: rec["process"] for r, rec in sorted(ranks.items())}
    return {
        "verdict": verdict,
        "checks": checks,
        "vendor_map": list(build_map),
        "stage_seconds": {
            r: {s["name"]: s.get("seconds") for s in rec["stages"]} for r, rec in ranks.items()
        },
        "processes": processes,
        "errors": errors,
    }


def run_experiment(exp: Experiment, port: int, args) -> dict:
    out = RESULTS / exp.id
    attempts = []
    timeout = args.timeout
    for attempt in (1, 2):
        print(f"[{exp.id}] attempt {attempt}, timeout {timeout:.0f}s ...", flush=True)
        args_for_attempt = argparse.Namespace(**{**vars(args), "timeout": timeout})
        raw = run_once(exp, port + attempt - 1, out, args_for_attempt)
        result = evaluate(exp, raw, out)
        attempts.append({"timeout": timeout, **raw, "verdict": result["verdict"]})
        print(f"[{exp.id}] {result['verdict']} in {raw['seconds']}s", flush=True)
        # Emulation is slow: a timeout gets one rerun with double the limit.
        if not raw["timed_out"]:
            break
        timeout *= 2
    return {"experiment": asdict(exp), "attempts": attempts, **result}


def env_report() -> dict:
    """Versions, installed size and approximate download size of each venv."""
    report = {}
    for env, (version, index) in ENVS.items():
        python = env_python(env)
        if not python.exists():
            continue
        probe = (
            "import json, platform, importlib.metadata as m, torch;"
            "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,"
            "'hip': torch.version.hip, 'cuda_available': torch.cuda.is_available(),"
            "'python': platform.python_version(),"
            "'dists': sorted((d.metadata['Name'], d.version) for d in m.distributions())}))"
        )
        info = json.loads(subprocess.check_output([str(python), "-c", probe], text=True))
        size = subprocess.check_output(["du", "-sb", str(ENVS_DIR / env)], text=True).split()[0]
        downloads = {name: wheel_size(index, name, ver) for name, ver in info.pop("dists")
                     if name != "gpubridge"}
        report[env] = {
            **info,
            "expected_torch": version,
            "index": index,
            "installed_bytes": int(size),
            "download_bytes": sum(v for v in downloads.values() if v),
            "largest_downloads": dict(sorted(downloads.items(), key=lambda kv: -(kv[1] or 0))[:6]),
        }
    return report


# download-r2.pytorch.org answers 403 to urllib's default User-Agent.
HEADERS = {"User-Agent": "gpubridge-mixed-build/0.1"}


class _Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links += [v for k, v in attrs if k == "href" and v]


def wheel_size(index: str, name: str, version: str) -> int | None:
    """Size of the wheel uv would have downloaded for ``name==version`` from ``index``."""
    project = name.lower().replace("_", "-").replace(".", "-")
    try:
        request = urllib.request.Request(f"{index}/{project}/", headers=HEADERS)
        page = urllib.request.urlopen(request, timeout=60).read().decode()
    except OSError:
        return None
    parser = _Links()
    parser.feed(page)
    prefix = f"{name.replace('-', '_').lower()}-{version.lower()}-"
    for href in parser.links:
        url = urllib.parse.urljoin(f"{index}/{project}/", href.split("#")[0])
        filename = urllib.parse.unquote(url.rsplit("/", 1)[-1]).lower()
        compatible_python = any(t in filename for t in ("cp312", "py3-none", "abi3", "py2.py3"))
        compatible_platform = "x86_64" in filename or filename.endswith("-any.whl")
        if filename.startswith(prefix) and compatible_python and compatible_platform:
            request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return int(response.headers["Content-Length"])
            except (OSError, TypeError, ValueError):
                return None
    return None


def write_summary(new_results: list[dict]) -> None:
    """Merge results into summary.json (keyed by experiment) and regenerate summary.md."""
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for result in new_results:
        summary[result["experiment"]["id"]] = result
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    order = [e.id for e in EXPERIMENTS]
    lines = [
        "| Experiment | Host node (store) | Guest node | Result | Rdzv | Init | Vendor map "
        "| all_reduce | broadcast | Shutdown | Wall time | Attempts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for exp_id in sorted(summary, key=lambda i: order.index(i) if i in order else len(order)):
        r = summary[exp_id]
        e = r["experiment"]
        mark = {True: "ok", False: "FAIL"}
        last = r["attempts"][-1]
        lines.append(
            f"| {exp_id} | {e['host']['nproc']}x {e['host']['env']} "
            f"| {e['guest']['nproc']}x {e['guest']['env']} | {r['verdict']} | "
            + " | ".join(mark[r["checks"][c]] for c in
                         ("rendezvous", "init", "vendor_map", "all_reduce", "broadcast",
                          "shutdown"))
            + f" | {last['seconds']}s | {len(r['attempts'])} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", action="append", help="run only these groups")
    parser.add_argument("--only", action="append", help="run only these experiment ids")
    parser.add_argument("--timeout", type=float, default=900, help="per-attempt limit (s)")
    parser.add_argument("--grace", type=float, default=180,
                        help="how long to wait for the other node after one exits (s)")
    parser.add_argument("--port", type=int, default=29400)
    parser.add_argument("--env-report", action="store_true", help="only report the venvs")
    parser.add_argument("--results", type=Path, default=RESULTS, help="output directory")
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)

    report = env_report()
    (RESULTS / "envs.json").write_text(json.dumps(report, indent=2))
    for env, info in report.items():
        print(f"{env}: torch {info['torch']} cuda={info['cuda']} hip={info['hip']} "
              f"cuda_available={info['cuda_available']} python {info['python']} "
              f"download {info['download_bytes'] / 1e9:.2f} GB "
              f"installed {info['installed_bytes'] / 1e9:.2f} GB", flush=True)
    if args.env_report:
        return 0

    selected = [
        e for e in EXPERIMENTS
        if (not args.group or e.group in args.group) and (not args.only or e.id in args.only)
    ]
    results = []
    for i, exp in enumerate(selected):
        missing = [n.env for n in (exp.host, exp.guest) if not env_python(n.env).exists()]
        if missing:
            print(f"[{exp.id}] skipped: venv(s) not in the image: {', '.join(missing)}")
            continue
        results.append(run_experiment(exp, args.port + 10 * i, args))
        write_summary(results)
    print((RESULTS / "summary.md").read_text() if results else "nothing ran")
    return 0 if all(r["verdict"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
