"""Side by side: train_dp.py's runs against the single-GPU baseline (a markdown table, the
loss curves as CSV and as an SVG plot).

    python scripts/gpu/dp_report.py results/explorer/07_mixed_hetjob_dp \\
        results/explorer/dp_baseline_nvidia --out results/explorer/dp_report

Every ``dp_*.json`` under the paths is a run; the one with ``"mode": "single"`` is the
baseline (``--baseline`` picks another). Per run: mode, compression, batch shares,
tokens/s from the median step, DDP's exposed communication and the share of it backward
hid, the last loss, and against the baseline the largest gap between the loss curves over
every step and the mean gap over the last 100 steps; then whether every parameter check
found the ranks' parameters identical. Needs only Python 3.10 or later: no torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2",
          "#17becf")


def load(paths: list[Path]) -> dict[str, dict]:
    runs = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(path.rglob("dp_*.json"))
        for f in files:
            record = json.loads(f.read_text())
            if record.get("script") == "scripts/gpu/train_dp.py":
                name = f.stem.removeprefix("dp_")
                runs[name if name not in runs else f"{f.parent.name}/{name}"] = record
    return runs


def gaps(curve: list[float], baseline: list[float], tail: int = 100) -> tuple[float, float]:
    """(largest gap over the steps both have, mean gap over the last ``tail`` of them)."""
    n = min(len(curve), len(baseline))
    diffs = [abs(a - b) for a, b in zip(curve[:n], baseline[:n])]
    last = diffs[-tail:] or [0.0]
    return max(diffs, default=0.0), sum(last) / len(last)


def describe(record: dict) -> tuple[str, str]:
    """(GPUs, shares) of a run, e.g. ("V100 + MI50", "4 + 4")."""
    ranks = record.get("ranks", [])
    names = [r["device"].replace("Tesla ", "").replace("AMD Instinct ", "").split("-")[0]
             for r in ranks]
    return " + ".join(names), " + ".join(str(r["share"]) for r in ranks)


def table(runs: dict[str, dict], baseline: str | None) -> str:
    base = runs[baseline]["losses"] if baseline else None
    out = ["| Run | Mode | Compression | GPUs | Shares | Tokens/s | Step (ms) | Exposed comm "
           "(ms) | Hidden | Last loss | Max gap | Gap, last 100 | Params identical |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in runs.items():
        gpus, shares = describe(r)
        biggest, tail = gaps(r["losses"], base) if base is not None and name != baseline \
            else (None, None)
        checks = r.get("param_checks", [])
        identical = ("-" if not checks else "yes" if all(c["identical"] for c in checks) else
                     f"no (sums up to {max(c['sum_spread'] for c in checks):.3g} apart)")
        hidden = r.get("hidden_fraction")
        cells = [name + (" (baseline)" if name == baseline else ""), r["mode"],
                 r["compression"], gpus, shares, f"{r['tokens_per_s']:.0f}",
                 f"{r['median_step_ms']:.1f}",
                 f"{r['exposed_comm_ms']:.1f}" if "exposed_comm_ms" in r else "-",
                 f"{hidden:.0%}" if hidden is not None else "-", f"{r['loss_last']:.4f}",
                 f"{biggest:.4f}" if biggest is not None else "-",
                 f"{tail:.4f}" if tail is not None else "-", identical]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def svg(runs: dict[str, dict], width: int = 720, height: int = 400) -> str:
    """The loss curves, one polyline per run, with a legend."""
    left, right, top, bottom = 50, 170, 15, 35
    steps = max(len(r["losses"]) for r in runs.values())
    values = [v for r in runs.values() for v in r["losses"]]
    lo, hi = min(values), max(values)
    hi = hi if hi > lo else lo + 1

    def x(i: int) -> float:
        return left + (width - left - right) * i / max(1, steps - 1)

    def y(v: float) -> float:
        return top + (height - top - bottom) * (hi - v) / (hi - lo)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             'font-family="sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
             f'y2="{height - bottom}" stroke="black"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
             f'<text x="{(width - right + left) / 2}" y="{height - 8}" text-anchor="middle">'
             'step</text>']
    for v in (lo, (lo + hi) / 2, hi):
        parts.append(f'<text x="{left - 5}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    parts.append(f'<text x="{left + 4}" y="{top + 10}">loss</text>')
    for i, (name, r) in enumerate(runs.items()):
        color = COLORS[i % len(COLORS)]
        points = " ".join(f"{x(j):.1f},{y(v):.1f}" for j, v in enumerate(r["losses"]))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.2" '
                     f'points="{points}"/>')
        ly = top + 14 * i + 10
        parts.append(f'<line x1="{width - right + 10}" y1="{ly}" x2="{width - right + 30}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{width - right + 35}" y="{ly + 4}">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def csv(runs: dict[str, dict]) -> str:
    names = list(runs)
    steps = max(len(r["losses"]) for r in runs.values())
    lines = ["step," + ",".join(names)]
    for i in range(steps):
        lines.append(f"{i + 1}," + ",".join(
            str(runs[n]["losses"][i]) if i < len(runs[n]["losses"]) else "" for n in names))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="result directories or JSON files")
    parser.add_argument("--baseline", help="the run to compare against (default: mode single)")
    parser.add_argument("--out", type=Path, help="write dp.md, dp_losses.csv and dp_losses.svg")
    args = parser.parse_args(argv)
    runs = load(args.paths)
    if not runs:
        print("no train_dp.py records found", file=sys.stderr)
        return 1
    baseline = args.baseline or next((n for n, r in runs.items() if r["mode"] == "single"), None)
    if baseline is not None and baseline not in runs:
        print(f"no run named {baseline!r}; runs: {', '.join(runs)}", file=sys.stderr)
        return 1
    text = table(runs, baseline) + "\n"
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "dp.md").write_text(text)
        (args.out / "dp_losses.csv").write_text(csv(runs))
        (args.out / "dp_losses.svg").write_text(svg(runs))
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
