"""Merge the per-rank JSON records of a kit step into summary.json and summary.md.

    python scripts/gpu/summarize.py [--partial] results/explorer/04_split_4gpu [more dirs...]

A directory with rank<N>.json files (from check.py) gets a per-stage pass/fail
table. A directory with train-n<N>.json files (from examples/train_synthetic.py)
gets a scaling table: samples/sec and efficiency against 1 GPU. Split-test runs
are flagged at the top of every summary. Exits non-zero if anything failed.

--partial is for one machine of a multi-machine job: ranks without a record
here are listed as being on another machine instead of failing the summary.
Copy every machine's rank files into one directory for the full verdict.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from gpukit import write_json

SPLIT_BANNER = (
    "> **SPLIT TEST - not a mixed-vendor result.** Islands were faked from a "
    "single-vendor job with `GPUBRIDGE_SPLIT_TEST`."
)


def summarize_checks(directory: Path, partial: bool = False) -> tuple[dict[str, Any], str]:
    records = {}
    for path in directory.glob("rank*.json"):
        match = re.fullmatch(r"rank(\d+)\.json", path.name)
        if match:
            records[int(match.group(1))] = json.loads(path.read_text())
    expected = None
    for record in records.values():
        if record.get("run"):
            expected = record["run"]["world_size"]
    missing = sorted(set(range(expected)) - set(records)) if expected else []
    stage_names: list[str] = []
    for record in records.values():
        for stage in record["stages"]:
            if stage["name"] not in stage_names:
                stage_names.append(stage["name"])

    runs = [r["run"] for r in records.values() if r.get("run")]
    split = next((run["split_test"] for run in runs if run["split_test"]), None)
    kind = runs[0]["kind"] if runs else "unknown"

    errors = {}
    for rank, record in sorted(records.items()):
        for stage in record["stages"]:
            if stage["ok"] is False:
                errors[f"rank {rank} / {stage['name']}"] = stage.get("error") or json.dumps(
                    {k: v for k, v in stage.items() if k not in ("name", "ok")}, default=str)
            elif stage["ok"] is None:
                errors[f"rank {rank} / {stage['name']}"] = "stuck in this stage (no result)"

    # Two ranks on one host driving the same GPU is only expected in a split test.
    shared_gpus = []
    by_gpu: dict[tuple[str, str], list[int]] = {}
    for rank, record in records.items():
        device = next((s for s in record["stages"] if s["name"] == "device"), {})
        if device.get("pci_bus_id"):
            host = record["environment"]["hostname"]
            by_gpu.setdefault((host, device["pci_bus_id"]), []).append(rank)
    for (host, bus), ranks in by_gpu.items():
        if len(ranks) > 1:
            shared_gpus.append({"host": host, "pci_bus_id": bus, "ranks": sorted(ranks)})

    ok = (bool(records) and (partial or not missing) and not errors
          and all(r.get("ok") for r in records.values())
          and (not shared_gpus or bool(split)))
    first = records[min(records)] if records else {}
    summary = {
        "directory": str(directory),
        "ok": ok,
        "run_kind": kind,
        "policy": runs[0].get("policy") if runs else None,
        "split_test": split,
        "real_mixed_vendor": kind == "gpu-mixed",
        "ranks_reported": sorted(records),
        "ranks_missing": missing,
        "partial": partial,
        "islands": runs[0]["islands"] if runs else None,
        "shared_gpus": shared_gpus,
        "environment": {k: first.get("environment", {}).get(k) for k in
                        ("torch", "torch_cuda", "torch_hip", "nccl_version", "gpubridge",
                         "git_commit")},
        "stages": {
            name: {
                rank: next((s["ok"] for s in r["stages"] if s["name"] == name), None)
                for rank, r in sorted(records.items())
            }
            for name in stage_names
        },
        "errors": errors,
    }

    lines = [f"# {directory.name}: {'PASS' if ok else 'FAIL'}", ""]
    if split:
        lines += [SPLIT_BANNER, ""]
    policy = runs[0].get("policy", "unknown") if runs else "unknown"
    lines += [f"Run kind: `{kind}`; policy: `{policy}`; split test: `{split or 'off'}`; "
              "ranks reported "
              f"{len(records)}{f' of {expected}' if expected else ''}.", ""]
    if missing and partial:
        lines += [f"Ranks {missing} ran on another machine (partial summary).", ""]
    elif missing:
        lines += [f"**Missing ranks:** {missing} (crashed before writing a record?)", ""]
    if shared_gpus and not split:
        lines += [f"**Ranks share a GPU outside a split test:** {shared_gpus}", ""]
    header = "| Stage | " + " | ".join(f"rank {r}" for r in sorted(records)) + " |"
    lines += [header, "|" + " --- |" * (len(records) + 1)]
    mark = {True: "ok", False: "FAIL", None: "-"}
    for name, by_rank in summary["stages"].items():
        lines.append(f"| {name} | " + " | ".join(mark[v] for v in by_rank.values()) + " |")
    if errors:
        lines += ["", "## Errors", ""]
        for where, text in errors.items():
            lines += [f"### {where}", "", "```", text.strip(), "```", ""]
    return summary, "\n".join(lines) + "\n"


def summarize_scaling(directory: Path) -> tuple[dict[str, Any], str]:
    runs = {}
    for path in directory.glob("train-n*.json"):
        record = json.loads(path.read_text())
        runs[record["world_size"]] = record
    base = runs.get(1, {}).get("samples_per_sec")
    rows = []
    for n, record in sorted(runs.items()):
        sps = record.get("samples_per_sec")
        efficiency = sps / (n * base) if base and sps else None
        rows.append({"gpus": n, "samples_per_sec": sps, "efficiency": efficiency,
                     "step_ms_median": record.get("step_ms_median"),
                     "params_in_sync": record.get("params_in_sync"),
                     "run_kind": record.get("run", {}).get("kind"),
                     "split_test": record.get("run", {}).get("split_test")})
    split = next((row["split_test"] for row in rows if row["split_test"]), None)
    ok = bool(rows) and all(row["params_in_sync"] for row in rows)
    summary = {"directory": str(directory), "ok": ok, "split_test": split, "rows": rows}
    lines = [f"# {directory.name}: scaling", ""]
    if split:
        lines += [SPLIT_BANNER, ""]
    lines += ["Weak scaling: fixed batch per GPU. Efficiency = samples/sec on N GPUs / "
              "(N x samples/sec on 1 GPU).", "",
              "| GPUs | samples/sec | efficiency | median step (ms) | params in sync |",
              "| --- | --- | --- | --- | --- |"]
    for row in rows:
        eff = f"{row['efficiency']:.2f}" if row["efficiency"] is not None else "-"
        sps = f"{row['samples_per_sec']:.1f}" if row["samples_per_sec"] else "-"
        lines.append(f"| {row['gpus']} | {sps} | {eff} | {row['step_ms_median']} | "
                     f"{row['params_in_sync']} |")
    return summary, "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    partial = "--partial" in argv
    all_ok = True
    for arg in [a for a in argv if a != "--partial"]:
        directory = Path(arg)
        if any(directory.glob("train-n*.json")):
            summary, markdown = summarize_scaling(directory)
        else:
            summary, markdown = summarize_checks(directory, partial)
        write_json(directory / "summary.json", summary)
        (directory / "summary.md").write_text(markdown)
        print(markdown)
        all_ok = all_ok and summary["ok"]
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
