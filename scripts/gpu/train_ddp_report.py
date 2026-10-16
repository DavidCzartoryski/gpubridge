"""One row per train_ddp run (examples/train_synthetic.py's record): whether the ranks'
parameters diverged, where it began, and a total (a markdown table).

    python scripts/gpu/train_ddp_report.py results/explorer/ddp-a results/explorer/ddp-b

Every ``train_ddp.json``, ``train_ddp_<i>.json`` (step 07's TRAIN_DDP_REPEATS) and
``train_ddp_split.json`` under the paths is a run. Per run: its folder and job (the
record's SLURM_JOB_ID, else the folder's slurm-job.txt or job.jsonl), the repeat, bridge
compression and each rank's compression kernels (as the record's environments hold
them), whether the parameters diverged (``params_in_sync``, from the bits), the first
step at which the synced gradients and the parameters differed between ranks
(``first_grad_mismatch``, ``first_param_mismatch``; steps count the warm-up), whether
the float64 checksums agreed (``params_checksums_equal``), the largest difference and
its tensor, and a verdict:

- in sync: every rank's parameters the same, bit for bit;
- gradients differ after all_reduce: the synced gradients already differed, which
  points at the sync (the bridge, compression);
- parameters differ, gradients equal: the optimizer step turned the same gradients
  into different parameters;
- old checksum only: the bits agree, but the float64 checksums, the check before the
  bit-for-bit one, differ (they can for identical parameters summed on two devices);
- no per-step check: a record from before that check, judged by its checksums alone.

A train_ddp step in a folder's steps.jsonl that left no record (it stopped before
writing one), or a repeat the job planned (job.jsonl) that never ran, is listed as "no
record"; so is a record that can't be read. Then the totals: k of n diverged, one line
per bridge compression, kernels and kind of run (a split test is not a mixed-vendor
run), with the older checksum-only records counted apart. Needs only Python 3.10 or
later: no torch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Step and record names: train_ddp, its repeats train_ddp_<i>, and the AMD split test's.
NAME = re.compile(r"train_ddp(?:_(split|\d+))?")
OFF = ("", "0", "false", "no", "off", "none")


def job_id(record: dict, folder: Path) -> str:
    """The job the record came from: its own SLURM_JOB_ID (rank 0's), else the folder's
    slurm-job.txt (scontrol show job) or the last start in its job.jsonl."""
    env = (record.get("environment") or {}).get("env") or {}
    if env.get("SLURM_JOB_ID"):
        return str(env["SLURM_JOB_ID"])
    scontrol = folder / "slurm-job.txt"
    if scontrol.is_file():
        found = re.search(r"\bJobId=(\d+)", scontrol.read_text(errors="replace"))
        if found:
            return found.group(1)
    events = folder / "job.jsonl"
    if events.is_file():
        starts = [e for e in _lines(events) if e.get("event") == "started"]
        if starts and starts[-1].get("job_id") not in (None, "local"):
            return str(starts[-1]["job_id"])
    return "-"


def _lines(path: Path) -> list[dict]:
    """A JSON-lines file's objects; lines that aren't one (cut short) are skipped."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def compression(record: dict) -> tuple[str, str]:
    """(bridge compression, compression kernels) from the ranks' recorded environments:
    the kernels per rank, joined by "/" where the ranks differ."""
    envs = [(e or {}).get("env") or {} for e in record.get("environments") or []]
    if not envs:
        return "-", "-"
    mode = str(envs[0].get("GPUBRIDGE_BRIDGE_COMPRESSION", "")).strip().lower()
    kernels = [str(e.get("GPUBRIDGE_COMPRESSION_KERNELS", "")).strip().lower() or "auto"
               for e in envs]
    return ("none" if mode in OFF else mode), (
        kernels[0] if len(set(kernels)) == 1 else "/".join(kernels))


def _first(record: dict, key: str) -> dict | None:
    found = record.get(key)
    return found if isinstance(found, dict) else None


def _at(first: dict | None) -> float:
    step = (first or {}).get("step")
    return step if isinstance(step, int) else float("inf")


def verdict(record: dict) -> str:
    """What differed first: the synced gradients (the sync), or the parameters alone."""
    if "first_param_mismatch" not in record:
        return "no per-step check"
    grads, params = _first(record, "first_grad_mismatch"), _first(record, "first_param_mismatch")
    if grads and not params and record.get("params_in_sync") is not False:
        return "gradients differed, parameters stayed in sync"
    if grads and _at(grads) <= _at(params):
        return "gradients differ after all_reduce"
    if params or record.get("params_in_sync") is False:
        return "parameters differ, gradients equal"
    if record.get("params_checksums_equal") is False:
        return "old checksum only"
    return "in sync"


def _step(first: dict | None) -> str:
    return "-" if not isinstance(first, dict) else str(first.get("step", "?"))


def _largest(record: dict) -> str:
    """The parameters' largest difference from rank 0 and its tensor, at the first step
    they differed; the gradients' if only those differed."""
    first = _first(record, "first_param_mismatch") or _first(record, "first_grad_mismatch")
    if not first:
        return "-"
    try:
        diff = f"{float(first.get('max_abs_diff')):.3g}"
    except (TypeError, ValueError):
        return "?"
    if first.get("tensor") is None:  # every difference a signed zero or a NaN
        return f"{diff} (signed zeros or NaNs)"
    return f"{diff} in {first['tensor']}"


def _run(record: dict) -> str:
    """The kind of run the record says it was, "split test" added for one."""
    run = record.get("run") if isinstance(record.get("run"), dict) else {}
    kind = str(run.get("kind") or "-")
    return f"{kind}, split test" if run.get("split_test") else kind


def _yes_no(value: object) -> str:
    return "-" if value is None else "yes" if value else "no"


def _which(name: str) -> str:
    """The repeat a step or record name holds: its number, "split", or "" for train_ddp."""
    found = NAME.fullmatch(name)
    return (found.group(1) or "") if found else ""


def _order(name: str) -> tuple[int, str]:
    which = _which(name)
    return (int(which), "") if which.isdigit() else (0, which)


def _planned(folder: Path) -> int:
    """The repeats the job's job.jsonl says it planned (step 07's TRAIN_DDP_REPEATS)."""
    events = folder / "job.jsonl"
    planned = [e.get("planned") for e in _lines(events) if e.get("event") == "train_ddp_repeats"
               ] if events.is_file() else []
    return planned[-1] if planned and isinstance(planned[-1], int) else 0


def _label(path: Path, folder: Path) -> str:
    """A folder as the table names it: from the argument's own name down."""
    if path.is_file():
        return path.parent.name
    return path.name if folder == path else f"{path.name}/{folder.relative_to(path).as_posix()}"


def rows(paths: list[Path]) -> list[dict]:
    """One row per train_ddp record under ``paths``, and per train_ddp step that left
    none: folder by folder in the order given, each in repeat order."""
    folders: dict[Path, tuple[str, dict[str, Path | str]]] = {}
    for path in paths:
        if path.is_file():
            files, logs = [path], []
        else:
            files = [f for f in path.rglob("train_ddp*.json") if NAME.fullmatch(f.stem)]
            logs = list(path.rglob("steps.jsonl"))
        for folder in sorted({f.parent for f in files} | {s.parent for s in logs}):
            # resolved: one folder named twice (relative, absolute) counts once
            folders.setdefault(folder.resolve(), (_label(path, folder), {}))
        for f in files:
            folders[f.parent.resolve()][1][f.stem] = f
        for log in logs:
            runs = folders[log.parent.resolve()][1]
            for entry in _lines(log):
                name = str(entry.get("step", ""))
                if NAME.fullmatch(name) and not isinstance(runs.get(name), Path):
                    code = entry.get("exit_code")  # the last run's
                    runs[name] = f"step exit {code if isinstance(code, int) else '?'}"
            for i in range(1, _planned(log.parent) + 1):
                runs.setdefault(f"train_ddp_{i}", "never ran")
    out = []
    for folder, (label, runs) in folders.items():
        for name in sorted(runs, key=_order):
            row = {"folder": label, "repeat": _which(name) or "-", "record": None}
            got, record = runs[name], None
            if isinstance(got, Path):
                try:
                    record = json.loads(got.read_text())
                except ValueError:  # cut short as it was written
                    pass
                if not isinstance(record, dict):
                    record = None
            if record is None:
                why = "unreadable" if isinstance(got, Path) else got
                out.append({**row, "job": job_id({}, folder), "verdict": f"no record ({why})"})
                continue
            if record.get("script") != "examples/train_synthetic.py":
                continue
            mode, kernels = compression(record)
            in_sync = record.get("params_in_sync")
            out.append({**row, "job": job_id(record, folder), "record": record,
                        "run": _run(record), "compression": mode, "kernels": kernels,
                        "diverged": _yes_no(None if in_sync is None else not in_sync),
                        "first_grad_mismatch": _step(record.get("first_grad_mismatch")),
                        "first_param_mismatch": _step(record.get("first_param_mismatch")),
                        "checksums_equal": _yes_no(record.get("params_checksums_equal")),
                        "largest": _largest(record), "verdict": verdict(record)})
    return out


def table(found: list[dict]) -> str:
    out = ["| Folder | Job | Repeat | Run | Compression | Kernels | Diverged | First grad "
           "mismatch (step) | First param mismatch (step) | Checksums equal | Largest "
           "difference | Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in found:
        cells = [r["folder"], r["job"], r["repeat"], r.get("run", "-"), r.get("compression", "-"),
                 r.get("kernels", "-"), r.get("diverged", "-"), r.get("first_grad_mismatch", "-"),
                 r.get("first_param_mismatch", "-"), r.get("checksums_equal", "-"),
                 r.get("largest", "-"), r["verdict"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def total(found: list[dict]) -> str:
    """k of n diverged per bridge compression, kernels and run, the older checksum-only
    records apart; then how many runs left no record."""
    groups: dict[tuple[str, str, str, bool], list[bool]] = {}
    for r in found:
        record = r["record"]
        if record is None or not isinstance(record.get("params_in_sync"), bool):
            continue
        key = (r["compression"], r["kernels"], r["run"], "first_param_mismatch" not in record)
        groups.setdefault(key, []).append(record["params_in_sync"] is False)
    lines = [f"bridge {mode}, kernels {kernels}, {run}"
             f"{' (float64 checksums only, before the per-step check)' if older else ''}: "
             f"{sum(diverged)} of {len(diverged)} diverged"
             for (mode, kernels, run, older), diverged in groups.items()]
    unjudged = sum(1 for r in found if r["record"] is not None
                   and not isinstance(r["record"].get("params_in_sync"), bool))
    if unjudged:
        lines.append(f"{unjudged} record(s) without params_in_sync")
    missing = sum(1 for r in found if r["record"] is None)
    if missing:
        lines.append(f"{missing} train_ddp run(s) left no record")
    return "\n".join(lines) or "no judged records"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="result directories or JSON files")
    args = parser.parse_args(argv)
    found = rows(args.paths)
    if not found:
        print("no train_ddp records found", file=sys.stderr)
        return 1
    print(table(found))
    print()
    print(total(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
