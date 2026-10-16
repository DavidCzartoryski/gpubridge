"""A short plain-text summary of a kit results directory or tarball, to paste into a review.

    python scripts/gpu/triage.py results/explorer
    python scripts/gpu/triage.py kaggle-t4x2-20261014-0930.tgz
    python scripts/gpu/triage.py results/explorer --max-lines 0     # everything

Every directory under the path with a steps.jsonl is one job (as for
validation_tables.py): an Explorer step directory, a mixed or AMD target, or a
Kaggle run. Problems come first: an Explorer job that started but never
finished (job.jsonl: its last start has no end), with the error that stopped
it if it recorded one; each step that failed, timed out or never finished (a
log with no steps.jsonl line: the job was killed), with the last error lines
of its log; and the skipped steps with their reasons. Then, per job:

- what it ran on: hosts (numbered), GPU models, driver, torch, CUDA or ROCm,
  NCCL or RCCL, the gpubridge commit;
- all_reduce benchmarks: gpubridge's bus bandwidth against the baseline's, at a
  few sizes;
- the bridge phase against what the preflight's bandwidth predicts;
- the compression kernels' check against the CPU reference, with the kernels
  that ran.

Then whether the compression digests agree across every compression record (a
mixed job has one per vendor), and one verdict line. A step run again counts
as its last attempt.

The output is redacted: host names become ``host N``, IPv4 addresses ``<ip>``,
and paths under /projects, /home, /scratch, /work and /Users lose their owner.
Still read the error lines before pasting them anywhere public. It stays under
``--max-lines`` (80) by dropping, in turn, benchmark detail, log tails and
passing jobs' detail. Needs only Python 3.10 or later: no torch. A tarball is
unpacked into a temporary directory; nothing in it is run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from validation_tables import (
    Hosts,
    Job,
    _environments,
    bench_rows,
    find_jobs,
    human,
    prediction_rows,
    run_row,
)

TAIL = 3
"""Error lines kept per failed step."""
WIDTH = 160
"""Characters per printed line."""
ERROR = re.compile(r"\b\w*(Error|Exception)\b[:(]|FAILED|Traceback|ran out of time|Killed|"
                   r"CANCELLED|NCCL WARN|timed out|not found|No such file|^FAIL \w+:")
#: torchrun's failure footer: says that a rank failed, not why.
BOILERPLATE = re.compile(r"error_file:|traceback\s*:|To enable traceback|Root Cause|Failures:|"
                         r"^[=\-]+$|<NO_OTHER_FAILURES>|exitcode\s*:|local_rank:|host\s*:|time\s*:|"
                         r"rank\s*:|\[\d+\]:$|ChildFailedError")
NCCL_TAIL = "==> NCCL/RCCL log"
"""common.sh appends NCCL's logs after this line in a failed step's log."""


# ---- redaction --------------------------------------------------------------------------------

class Redact:
    """Host names (from the records, full and short) to ``host N``, IPv4 addresses, owners of
    paths. Applied to every printed line."""

    PROJECT = re.compile(r"/projects/[^/\s:'\"]+(/[^/\s:'\"]+)?")  # the group, then the user
    HOME = re.compile(r"/(home|scratch|work|Users)/[^/\s:'\"]+")
    IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def __init__(self, hosts: Hosts) -> None:
        self.hosts = hosts

    def __call__(self, text: str) -> str:
        forms: dict[str, str] = {}
        for name, label in self.hosts.names.items():
            for form in (name, name.split(".")[0]) if name else ():
                if len(form) >= 3:
                    forms.setdefault(form, label)
        if forms:
            # One pass, longest first: a full name wins over its short form (which would
            # leave "host 1.cluster.example.edu"), and no label is replaced again.
            ordered = sorted(forms, key=lambda f: (-len(f), f))
            pattern = r"(?<![\w-])(?:" + "|".join(map(re.escape, ordered)) + r")(?![\w-])"
            text = re.sub(pattern, lambda m: forms[m.group()], text)
        text = self.HOME.sub("~", self.PROJECT.sub("$PROJECT_DIR", text))
        return self.IPV4.sub("<ip>", text)


def one_line(text: Any, width: int = WIDTH) -> str:
    """``text`` on one line (its indent kept), at most ``width`` characters."""
    text = str(text)
    flat = " " * (len(text) - len(text.lstrip(" "))) + " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 3] + "..."


# ---- reading ------------------------------------------------------------------------------------

def unpack(tarball: Path, into: Path) -> Path:
    """Extract ``tarball`` into ``into``: regular files and directories only, all inside it."""
    with tarfile.open(tarball) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(into, filter="data")
            return into
        root = into.resolve()  # a Python without extraction filters: do it by hand
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(root):
                raise ValueError(f"unsafe member in {tarball.name}: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            target.write_bytes(source.read() if source is not None else b"")
    return into


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def inside(job: Path, name: str) -> Path | None:
    """``job/name``, if it stays inside the job's directory (steps.jsonl names the log)."""
    path = (job / name).resolve()
    return path if path.is_relative_to(job.resolve()) else None


def tail(log: Path | None, n: int = TAIL) -> list[str]:
    """The error lines at the end of a step's log (before any NCCL log tail): the last ``n``
    that look like an error, else its last ``n`` lines."""
    if log is None:
        return ["(log outside the results directory)"]
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return ["(no log)"]
    text = text.split(NCCL_TAIL)[0]
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not BOILERPLATE.search(line.strip())]
    errors = [line for line in lines if ERROR.search(line)]
    return (errors or lines)[-n:] or ["(empty log)"]


# ---- one job ------------------------------------------------------------------------------------

class Report:
    def __init__(self, job: Job, hosts: Hosts) -> None:
        self.job = job
        self.run = run_row(job, hosts)
        self.counts = {"passed": 0, "failed": 0, "timed out": 0, "unfinished": 0, "skipped": 0}
        self.problems: list[tuple[str, list[str]]] = []  # (line, log tail)
        self.unfinished_job = False
        self._steps()
        self._job_events()

    def _job_events(self) -> None:
        """job.jsonl (Explorer jobs): whether the job's last start reached its end."""
        events = _jsonl(self.job.path / "job.jsonl")
        starts = [i for i, event in enumerate(events) if event.get("event") == "started"]
        if not starts:
            return
        start = events[starts[-1]]
        mine = [e for e in events[starts[-1] + 1:] if e.get("job_id") == start.get("job_id")]
        kinds = [str(e.get("event")) for e in mine]
        end = kinds.index("finished") if "finished" in kinds else None
        errors = [(i, str(e.get("message", ""))) for i, e in enumerate(mine)
                  if e.get("event") == "error"]
        job = f"job {start.get('job_id', '?')}"
        lines = []
        if end is None:
            self.unfinished_job = True
            codes = [e.get("code") for e in mine if e.get("event") == "exit"]
            if errors:
                why = f"it stopped with an error: {errors[-1][1]}"
            elif codes and isinstance(codes[-1], int) and codes[-1] > 128:
                why = f"it ended on signal {codes[-1] - 128}: killed, or out of Slurm time"
            elif codes:
                why = (f"it exited with code {codes[-1]} before its end: a command failed "
                       "(the job log says which)")
            else:
                why = "killed, out of Slurm time, or still running"
            lines.append(f"  UNFINISHED {job}, started {start.get('time', '?')}: no end "
                         f"recorded; {why}. Its job log has the rest")
        else:
            lines += [f"  ERROR in {job}, which went on to its end: {message}"
                      for i, message in errors if i < end]
            stops = [message for i, message in errors if i > end]
            if stops:
                lines.append(f"  STOPPED {job} early: {stops[-1]}")
        self.problems[:0] = [(line, []) for line in lines]

    def _steps(self) -> None:
        last: dict[str, dict[str, Any]] = {}
        earlier_failures: dict[str, int] = {}
        for step in self.job.steps:  # a step run again counts as its last attempt
            name = str(step.get("step", "?"))
            if name in last and last[name].get("exit_code") != 0:
                earlier_failures[name] = earlier_failures.get(name, 0) + 1
            last[name] = step
        for name, step in last.items():
            code = step.get("exit_code", 1)
            again = (f" ({earlier_failures[name]} earlier attempt(s) failed)"
                     if name in earlier_failures else "")
            note = f"; note: {step['note']}" if step.get("note") else ""
            if code == 0:
                self.counts["passed"] += 1
                if note:
                    self.problems.append((f"  PASS {name}{note}", []))
                continue
            timed_out = step.get("timed_out") in (True, "true")
            self.counts["timed out" if timed_out else "failed"] += 1
            why = ("ran out of time" if timed_out else step.get("result")
                   or (f"not run: {step['not_run']}" if step.get("not_run") else f"exit {code}"))
            log = inside(self.job.path, str(step.get("log", f"{name}.log")))
            self.problems.append((f"  {'TIMEOUT' if timed_out else 'FAIL'} {name}: {why}, "
                                  f"{step.get('seconds', '?')} s{again}{note}", tail(log)))
        recorded = {f"{name}.log" for name in last} | {
            str(s.get("log")) for s in last.values() if s.get("log")}
        for log in sorted(self.job.path.glob("*.log")):
            if log.name not in recorded:  # a step that started but never recorded an end
                self.counts["unfinished"] += 1
                self.problems.append((f"  UNFINISHED {log.stem}: a log but no steps.jsonl line "
                                      "(the job was killed, or ran out of Slurm time)", tail(log)))
        for skip in _jsonl(self.job.path / "skipped.jsonl"):
            self.counts["skipped"] += 1
            self.problems.append((f"  SKIP {skip.get('step', '?')}: {skip.get('reason', '')}", []))

    @property
    def ok(self) -> bool:
        return not (self.counts["failed"] or self.counts["timed out"] or self.counts["unfinished"]
                    or self.unfinished_job)

    def header(self) -> list[str]:
        r = self.run
        kinds = f" [{', '.join(r['run_kinds'])}]" if r["run_kinds"] else ""
        lines = [f"== {self.job.name}{kinds}" + (" (CPU SIMULATION)" if r["simulated"] else ""),
                 f"  on: {r['hosts']}; GPUs {r['gpus']}; driver {r['driver']}; torch {r['torch']}; "
                 f"{r['toolkit']}; {r['collectives']}; gpubridge {r['gpubridge']}",
                 "  steps: " + (", ".join(f"{n} {what}" for what, n in self.counts.items() if n)
                                or "none recorded")]
        features = self._features()
        return lines + ([features] if features else [])

    def _features(self) -> str | None:
        data = _load(self.job.path / "features.json")
        if not data:
            return None
        gpus = data.get("gpus") or []
        if not gpus:
            return "  features: no GPU recorded"
        caps = sorted({f"{g.get('name', '?')} (sm {g.get('capability', '?')})" for g in gpus})
        missing = sorted({name for g in gpus for name, c in (g.get("checks") or {}).items()
                          if isinstance(c, dict) and not c.get("ok")})
        return (f"  features: {', '.join(caps)}; "
                + (f"not available: {', '.join(missing)}" if missing else "all checked features ok"))

    def compression(self) -> list[str]:
        out = []
        for path in sorted(self.job.path.rglob("compression.json")):
            data = _load(path) or {}
            checks = [c for c in data.get("checks", []) if isinstance(c, dict)]
            if not checks:
                continue
            bad = [f"{c.get('mode')} {c.get('dtype')}" for c in checks
                   if not c.get("matches_reference")]
            where = path.parent.relative_to(self.job.path).as_posix()
            out.append(f"  compression {where}/ ({data.get('impl', '?')} kernels on "
                       f"{data.get('device', '?')}): {len(checks) - len(bad)} of {len(checks)} "
                       "match the CPU reference" + (f"; MISMATCH: {', '.join(bad)}" if bad else ""))
        return out

    def bench(self) -> list[str]:
        out = []
        by_dir: dict[str, list[dict[str, Any]]] = {}
        for row in bench_rows([self.job]):
            by_dir.setdefault(row["bench"], []).append(row)
        for name, rows in by_dir.items():
            raw = {(r["bytes"], r["op"]): r for r in self.job.benches[name].get("rows", [])}
            first, cells = rows[0], []
            for row in rows:
                ours = raw.get((row["bytes"], "gpubridge"), {})
                theirs = raw.get((row["bytes"], "native"), {})
                if first["baseline"] == "native" and ours.get("busbw_GBps") and theirs.get(
                        "busbw_GBps"):
                    share = f" ({100 * ours['busbw_GBps'] / theirs['busbw_GBps']:.0f}%)"
                elif row["gpubridge_us"] and row["baseline_us"]:
                    share = f" ({100 * row['baseline_us'] / row['gpubridge_us']:.0f}% speed)"
                else:
                    share = ""
                cells.append(f"{human(row['bytes'])} {ours.get('busbw_GBps', '-')} GB/s{share}")
            out.append(f"  benchmark {name or '.'}/ ({first['bridge']} bridge, {first['staging']}, "
                       f"{first['timer']}): gpubridge busbw, % of {first['baseline']}: "
                       + "; ".join(cells))
        by_bench: dict[str, list[dict[str, Any]]] = {}
        for row in prediction_rows([self.job]):
            by_bench.setdefault(row["bench"], []).append(row)
        for name, rows in by_bench.items():
            shown = rows if len(rows) <= 3 else [rows[0], rows[len(rows) // 2], rows[-1]]
            out.append(f"  bridge vs predicted, {name or '.'}/: " + "; ".join(
                f"{human(r['bytes'])} {r['measured_ms']} ms vs {r['predicted_ms']} ms "
                f"({r['measured_over_predicted']}x)" for r in shown))
        return out


def digest_line(jobs: list[Job]) -> tuple[str | None, bool]:
    """Whether every compression record's kernels gave the same bytes per mode and dtype."""
    by_case: dict[tuple[Any, Any], set[str]] = {}
    records = 0
    for job in jobs:
        for path in job.path.rglob("compression.json"):
            checks = [c for c in (_load(path) or {}).get("checks", []) if isinstance(c, dict)]
            records += bool(checks)
            for c in checks:
                by_case.setdefault((c.get("mode"), c.get("dtype")), set()).add(
                    f"{c.get('payload_sha256')}/{c.get('result_sha256')}")
    if records < 2:
        return None, True
    differ = [f"{mode} {dtype}" for (mode, dtype), digests in sorted(by_case.items(), key=str)
              if len(digests) > 1]
    return (f"compression digests across {records} records: "
            + (f"DIFFER for {', '.join(differ)}" if differ else "all agree")), not differ


# ---- the whole summary ------------------------------------------------------------------------

def summarize(root: Path, label: str, max_lines: int = 80) -> tuple[str, bool]:
    """The text ("" without any job), and whether everything that ran passed (and the
    compression digests agree)."""
    jobs = find_jobs([root])
    if not jobs:
        return "", False
    hosts = Hosts()
    reports = [Report(job, hosts) for job in jobs]
    for job in jobs:  # every host name the records know, so that redaction catches them all
        for record in job.records:
            for env in _environments(record):
                hosts(env.get("hostname"))
    redact = Redact(hosts)
    totals = {k: sum(r.counts[k] for r in reports) for k in reports[0].counts}
    digests, agree = digest_line(jobs)
    bad = [f"{totals[k]} {k}" for k in ("failed", "timed out", "unfinished") if totals[k]]
    unfinished_jobs = sum(r.unfinished_job for r in reports)
    if unfinished_jobs:
        bad.append(f"{unfinished_jobs} job(s) unfinished")
    ok = not bad and agree
    simulated = any(job.simulated for job in jobs)
    head = [f"gpubridge results triage: {label}, {len(jobs)} job(s)"
            + ("; CPU SIMULATION, says nothing about GPUs" if simulated else "")]
    verdict = "verdict: " + (", ".join(bad) if bad else "every step that ran passed")
    if not agree:
        verdict += "; compression digests DIFFER"
    if totals["skipped"]:
        verdict += f"; {totals['skipped']} skipped (reasons above)"
    foot = ([digests] if digests else []) + [verdict]

    def render(bench: bool, tails: bool, passing_detail: bool) -> list[str]:
        out = list(head)
        for r in sorted(reports, key=lambda r: r.ok):  # jobs with problems first
            if r.ok and not passing_detail and not r.problems:
                out.append(f"== {r.job.name}: all {r.counts['passed']} steps passed")
                continue
            out += r.header()
            for line, log in r.problems:
                out.append(line)
                if tails and log:
                    out += ["    | " + entry for entry in log]
            out += r.compression()
            if bench:
                out += r.bench()
        return [one_line(redact(line)) for line in out + foot]

    stages = ((True, True, True), (False, True, True), (False, True, False), (False, False, False))
    for stage in stages:
        text = render(*stage)
        if not max_lines or len(text) <= max_lines:
            return "\n".join(text), ok
    cut = text[:max(1, max_lines - len(foot) - 1)]
    more = len(text) - len(cut) - len(foot)
    return "\n".join(cut + [f"... {more} more lines (--max-lines 0 shows all)"]
                     + text[-len(foot):]), ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", type=Path, help="a results directory, or a .tgz / .tar.gz of one")
    p.add_argument("--max-lines", type=int, default=80,
                   help="0 for no limit (at least 10 otherwise)")
    args = p.parse_args(argv)
    if args.max_lines and args.max_lines < 10:
        p.error("--max-lines is 0 (no limit) or at least 10")
    path = args.results
    if not path.exists():
        print(f"triage: {path} doesn't exist", file=sys.stderr)
        return 2
    try:
        if path.is_file():
            with tempfile.TemporaryDirectory() as tmp:
                text, ok = summarize(unpack(path, Path(tmp)), path.name, args.max_lines)
        else:
            text, ok = summarize(path, path.name, args.max_lines)
    except (tarfile.TarError, ValueError, OSError) as error:
        print(f"triage: can't read {path.name}: {error}", file=sys.stderr)
        return 2
    if not text:
        print(f"triage: no steps.jsonl under {path.name}: not a kit results directory?",
              file=sys.stderr)
        return 1
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
