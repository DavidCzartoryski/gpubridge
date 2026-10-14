"""A short plain-text summary of a kit results directory or tarball, to paste into a review.

    python scripts/gpu/triage.py results/explorer
    python scripts/gpu/triage.py kaggle-t4x2-20261014-0930.tgz
    python scripts/gpu/triage.py results/explorer --max-lines 0     # everything

Every directory under the path with a steps.jsonl is one job (as for
validation_tables.py): an Explorer step directory, a mixed or AMD target, or a
Kaggle run. For each job it prints:

- what it ran on: hosts (numbered, never named), GPU models, driver, torch,
  CUDA or ROCm, NCCL or RCCL, the gpubridge commit;
- each step that didn't pass, with its exit code and the last lines of its log,
  timed-out steps, and skipped steps with their reasons;
- all_reduce benchmarks: gpubridge's bus bandwidth as a share of the baseline's
  (native, or the slowest island) at a few sizes;
- the bridge phase against what the preflight's bandwidth predicts;
- the compression kernels' check against the CPU reference.

Then whether the compression digests agree across runs, and one verdict line.
It stays under ``--max-lines`` (80) by cutting benchmark detail first, then log
tails. Results from CPU simulation say so on their first line. Hosts are numbered
and paths are left out, as in validation_tables.py, so the output can be pasted
anywhere. A tarball is unpacked into a temporary directory; nothing in it is run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from validation_tables import (
    Hosts,
    Job,
    bench_rows,
    find_jobs,
    human,
    prediction_rows,
    run_row,
)

TAIL = 3
"""Log lines kept per failed step."""
WIDTH = 150
"""Characters kept per printed log line."""
ERRORISH = ("error", "Error", "ERROR", "Traceback", "FAILED", "failed", "assert", "Timeout",
            "timed out", "ran out of time", "Killed", "CANCELLED", "No such", "not found")


def unpack(tarball: Path, into: Path) -> Path:
    """Extract ``tarball`` into ``into`` without following anything outside it."""
    with tarfile.open(tarball) as tar:
        try:
            tar.extractall(into, filter="data")
        except TypeError:  # Python without extraction filters: refuse unsafe members ourselves
            for member in tar.getmembers():
                target = (into / member.name).resolve()
                if (member.issym() or member.islnk() or member.isdev()
                        or not str(target).startswith(str(into.resolve()))):
                    raise ValueError(f"unsafe member in {tarball.name}: {member.name}") from None
            tar.extractall(into)  # noqa: S202 - every member checked above
    return into


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def tail(log: Path, n: int = TAIL) -> list[str]:
    """The last ``n`` lines of ``log`` that look like an error, else its last ``n`` lines."""
    try:
        lines = [line.rstrip() for line in log.read_text(errors="replace").splitlines()
                 if line.strip()]
    except OSError:
        return ["(no log)"]
    errors = [line for line in lines if any(word in line for word in ERRORISH)]
    picked = (errors or lines)[-n:]
    return [line[:WIDTH] + ("..." if len(line) > WIDTH else "") for line in picked]


def step_lines(job: Job) -> tuple[list[tuple[str, str | None]], dict[str, int]]:
    """(a line per step that didn't pass, with its log tail or None; counts)."""
    out: list[tuple[str, str | None]] = []
    counts = {"passed": 0, "failed": 0, "timed out": 0, "skipped": 0}
    for step in job.steps:
        name, code = step.get("step", "?"), step.get("exit_code", 1)
        if code == 0:
            counts["passed"] += 1
            continue
        timed_out = step.get("timed_out") in (True, "true")
        counts["timed out" if timed_out else "failed"] += 1
        why = step.get("result") or ("ran out of time" if timed_out else f"exit {code}")
        out.append((f"  {'TIMEOUT' if timed_out else 'FAIL'} {name}: {why}, "
                    f"{step.get('seconds', '?')} s",
                    "    log: " + " | ".join(tail(job.path / step.get("log", f"{name}.log")))))
    for skip in _jsonl(job.path / "skipped.jsonl"):
        counts["skipped"] += 1
        out.append((f"  SKIP {skip.get('step', '?')}: {skip.get('reason', '')}", None))
    return out, counts


def features_line(job: Job) -> str | None:
    """Kaggle: what each GPU could run, from features.json."""
    data = _load(job.path / "features.json")
    if not data:
        return None
    gpus = data.get("gpus") or []
    caps = sorted({f"{g.get('name', '?')} (sm {g.get('capability', '?')})" for g in gpus})
    missing = sorted({name for g in gpus for name, c in (g.get("checks") or {}).items()
                      if isinstance(c, dict) and not c.get("ok")})
    return (f"  features: {', '.join(caps) or 'no GPU'}; "
            + (f"not available: {', '.join(missing)}" if missing else "all checked features ok"))


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def busbw_lines(job: Job) -> list[str]:
    """gpubridge's bus bandwidth as a share of the baseline's, per benchmark, at a few sizes."""
    out = []
    by_bench: dict[str, list[dict[str, Any]]] = {}
    for row in bench_rows([job]):
        by_bench.setdefault(row["bench"], []).append(row)
    for name, rows in by_bench.items():
        bench = job.benches[name]
        busbw = {(r["bytes"], r["op"]): r.get("busbw_GBps") for r in bench.get("rows", [])}
        cells = []
        for row in rows:
            share = row["gpubridge_over_baseline"]  # time ratio; bandwidth share is its inverse
            gb = busbw.get((row["bytes"], "gpubridge"))
            cells.append(f"{human(row['bytes'])} {gb if gb is not None else '-'} GB/s"
                         + (f" ({100 / share:.0f}%)" if share else ""))
        first = rows[0]
        out.append(f"  benchmark {name or '.'}/: {first['bridge']} bridge, {first['staging']} "
                   f"staging, {first['timer']} timer; gpubridge busbw (% of {first['baseline']}): "
                   + "; ".join(cells))
    return out


def prediction_lines(job: Job) -> list[str]:
    rows = prediction_rows([job])
    if not rows:
        return []
    shown = rows if len(rows) <= 3 else [rows[0], rows[len(rows) // 2], rows[-1]]
    return ["  bridge vs predicted: " + "; ".join(
        f"{human(r['bytes'])} {r['measured_ms']} ms vs {r['predicted_ms']} ms "
        f"({r['measured_over_predicted']}x)" for r in shown)]


def compression_checks(job: Job) -> list[dict[str, Any]]:
    found = []
    for path in sorted(job.path.rglob("compression.json")):
        data = _load(path) or {}
        for check in data.get("checks", []):
            if isinstance(check, dict):
                found.append({**check, "job": job.name})
    return found


def compression_line(checks: list[dict[str, Any]]) -> str | None:
    if not checks:
        return None
    bad = [f"{c.get('mode')} {c.get('dtype')}" for c in checks if not c.get("matches_reference")]
    return (f"  compression kernels: {len(checks) - len(bad)} of {len(checks)} match the CPU "
            "reference" + (f"; MISMATCH: {', '.join(bad)}" if bad else ""))


def digest_line(checks: list[dict[str, Any]]) -> str | None:
    """Whether every run's kernels gave the same bytes for the same mode and dtype."""
    by_case: dict[tuple[str, str], set[str]] = {}
    jobs = {c["job"] for c in checks}
    for c in checks:
        by_case.setdefault((c.get("mode"), c.get("dtype")), set()).add(
            f"{c.get('payload_sha256')}/{c.get('result_sha256')}")
    if len(jobs) < 2:
        return None
    differ = [f"{mode} {dtype}" for (mode, dtype), digests in sorted(by_case.items())
              if len(digests) > 1]
    return (f"compression digests across {len(jobs)} runs: "
            + (f"DIFFER for {', '.join(differ)}" if differ else "all agree"))


def job_block(job: Job, hosts: Hosts) -> tuple[list[str], list[tuple[str, str | None]],
                                               list[str], dict[str, int]]:
    """(header lines, step lines with their log tails, benchmark lines, counts) for one job."""
    run = run_row(job, hosts)
    kinds = f" [{', '.join(run['run_kinds'])}]" if run["run_kinds"] else ""
    lines = [f"== {job.name}{kinds}" + (" (CPU SIMULATION)" if run["simulated"] else ""),
             f"  on: {run['hosts']}; GPUs {run['gpus']}; driver {run['driver']}; torch "
             f"{run['torch']}; {run['toolkit']}; {run['collectives']}; gpubridge {run['gpubridge']}"]
    features = features_line(job)
    if features:
        lines.append(features)
    steps_, counts = step_lines(job)
    lines.append("  steps: " + ", ".join(f"{n} {what}" for what, n in counts.items() if n)
                 if any(counts.values()) else "  steps: none recorded")
    compression = compression_line(compression_checks(job))
    if compression:
        steps_.append((compression, None))  # always shown, after the steps
    return lines, steps_, busbw_lines(job) + prediction_lines(job), counts


def summarize(root: Path, label: str, max_lines: int = 80) -> tuple[str, bool]:
    """The text, and whether everything that ran passed."""
    jobs = find_jobs([root])
    if not jobs:
        return f"triage: no steps.jsonl under {label}: not a kit results directory?", False
    hosts = Hosts()
    blocks = [job_block(job, hosts) for job in jobs]
    totals = {k: sum(b[3][k] for b in blocks) for k in ("passed", "failed", "timed out", "skipped")}
    simulated = any(job.simulated for job in jobs)
    checks = [c for job in jobs for c in compression_checks(job)]
    ok = totals["failed"] == 0 and totals["timed out"] == 0
    header = [f"gpubridge results triage: {label}, {len(jobs)} job(s)"
              + ("; CPU SIMULATION, says nothing about GPUs" if simulated else "")]
    footer = [line for line in (digest_line(checks),) if line] + [
        "verdict: " + ("every step that ran passed" if ok else
                       f"{totals['failed']} failed, {totals['timed out']} timed out")
        + (f", {totals['skipped']} skipped (reasons above)" if totals["skipped"] else "")]

    def render(with_bench: bool, tails_kept: int) -> list[str]:
        out = list(header)
        for lines, steps_, bench, _ in blocks:
            out += lines
            for line, log in steps_:
                out.append(line)
                if log is not None:
                    out.append(log if tails_kept >= TAIL else _cut(log, tails_kept))
            if with_bench:
                out += bench
        return out + footer

    for with_bench, kept in ((True, TAIL), (False, TAIL), (False, 1)):
        text = render(with_bench, kept)
        if not max_lines or len(text) <= max_lines:
            return "\n".join(text), ok
    cut = text[:max_lines - len(footer) - 1]
    return "\n".join(cut + [f"... {len(text) - len(cut) - len(footer)} more lines "
                            "(--max-lines 0 shows all)"] + footer), ok


def _cut(tail_line: str, keep: int) -> str:
    """A log-tail line with only its last ``keep`` log lines."""
    head, _, rest = tail_line.partition(": ")
    return f"{head}: " + " | ".join(rest.split(" | ")[-keep:])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", type=Path, help="a results directory, or a .tgz / .tar.gz of one")
    p.add_argument("--max-lines", type=int, default=80, help="0 for no limit")
    args = p.parse_args(argv)
    path = args.results
    if not path.exists():
        print(f"triage: {path} doesn't exist", file=sys.stderr)
        return 2
    if path.is_file():
        with tempfile.TemporaryDirectory() as tmp:
            text, ok = summarize(unpack(path, Path(tmp)), path.name, args.max_lines)
    else:
        text, ok = summarize(path, path.name, args.max_lines)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
