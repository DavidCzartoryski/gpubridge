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
writing one) is listed as "no record". Then the total: k of n diverged. Needs only
Python 3.10 or later: no torch.
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


def verdict(record: dict) -> str:
    if "first_param_mismatch" not in record:
        return "no per-step check"
    if record.get("first_grad_mismatch"):
        return "gradients differ after all_reduce"
    if record.get("first_param_mismatch") or record.get("params_in_sync") is False:
        return "parameters differ, gradients equal"
    if record.get("params_checksums_equal") is False:
        return "old checksum only"
    return "in sync"


def _step(first: dict | None) -> str:
    return str(first["step"]) if first else "-"


def _largest(record: dict) -> str:
    """The parameters' largest difference from rank 0 and its tensor, at the first step
    they differed; the gradients' if only those differed."""
    first = record.get("first_param_mismatch") or record.get("first_grad_mismatch")
    if not first:
        return "-"
    return f"{first['max_abs_diff']:.3g} in {first['tensor']}"


def _yes_no(value: object) -> str:
    return "-" if value is None else "yes" if value else "no"


def _which(name: str) -> str:
    """The repeat a step or record name holds: its number, "split", or "" for train_ddp."""
    found = NAME.fullmatch(name)
    return (found.group(1) or "") if found else ""


def _order(name: str) -> tuple[int, str]:
    which = _which(name)
    return (int(which), "") if which.isdigit() else (0, which)


def _label(path: Path, folder: Path) -> str:
    """A folder as the table names it: from the argument's own name down."""
    if path.is_file():
        return path.parent.name
    return path.name if folder == path else f"{path.name}/{folder.relative_to(path).as_posix()}"


def rows(paths: list[Path]) -> list[dict]:
    """One row per train_ddp record under ``paths``, and per train_ddp step that left
    none: folder by folder in the order given, each in repeat order."""
    folders: dict[Path, tuple[str, dict[str, Path | int]]] = {}
    for path in paths:
        if path.is_file():
            files, logs = [path], []
        else:
            files = [f for f in path.rglob("train_ddp*.json") if NAME.fullmatch(f.stem)]
            logs = list(path.rglob("steps.jsonl"))
        for folder in sorted({f.parent for f in files} | {s.parent for s in logs}):
            folders.setdefault(folder, (_label(path, folder), {}))
        for f in files:
            folders[f.parent][1][f.stem] = f
        for log in logs:
            runs = folders[log.parent][1]
            for entry in _lines(log):
                name = str(entry.get("step", ""))
                if NAME.fullmatch(name) and not isinstance(runs.get(name), Path):
                    runs[name] = int(entry.get("exit_code", 1))  # the last run's
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
            if record is None:
                why = "unreadable" if isinstance(got, Path) else f"step exit {got}"
                out.append({**row, "job": job_id({}, folder), "verdict": f"no record ({why})"})
                continue
            if record.get("script") != "examples/train_synthetic.py":
                continue
            mode, kernels = compression(record)
            in_sync = record.get("params_in_sync")
            out.append({**row, "job": job_id(record, folder), "record": record,
                        "compression": mode, "kernels": kernels,
                        "diverged": _yes_no(None if in_sync is None else not in_sync),
                        "first_grad_mismatch": _step(record.get("first_grad_mismatch")),
                        "first_param_mismatch": _step(record.get("first_param_mismatch")),
                        "checksums_equal": _yes_no(record.get("params_checksums_equal")),
                        "largest": _largest(record), "verdict": verdict(record)})
    return out


def table(found: list[dict]) -> str:
    out = ["| Folder | Job | Repeat | Compression | Kernels | Diverged | First grad mismatch "
           "(step) | First param mismatch (step) | Checksums equal | Largest difference | "
           "Verdict |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in found:
        cells = [r["folder"], r["job"], r["repeat"], r.get("compression", "-"),
                 r.get("kernels", "-"), r.get("diverged", "-"), r.get("first_grad_mismatch", "-"),
                 r.get("first_param_mismatch", "-"), r.get("checksums_equal", "-"),
                 r.get("largest", "-"), r["verdict"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def total(found: list[dict]) -> str:
    records = [r["record"] for r in found if r["record"] is not None
               and "params_in_sync" in r["record"]]
    diverged = sum(1 for record in records if not record["params_in_sync"])
    text = f"{diverged} of {len(records)} diverged"
    older = sum(1 for record in records if "first_param_mismatch" not in record)
    if older:
        text += f" ({older} older record(s) judged by their float64 checksums alone)"
    missing = sum(1 for r in found if r["record"] is None)
    if missing:
        text += f"; {missing} train_ddp step(s) left no record"
    return text


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
