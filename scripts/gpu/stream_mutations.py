"""Show that the GPU stream-ordering tests catch what they guard. Needs one GPU.

tests/test_gpu.py's MUTATIONS each undo one stream-ordering fix in every rank:

- ``worker-ignores-caller``: the async worker's stream no longer waits for the
  caller's stream;
- ``staging-ignores-caller``: the pinned copy streams no longer wait for it either;
- ``wait-skips-stream``: ``Work.wait()`` blocks the host but orders no stream;
- ``slots-forget-copies-back``: the pipelined bridge reuses a pinned slot before its
  last copy back has run.

For each, this runs the GPU test that guards the fix with GPUBRIDGE_TEST_MUTATION set:
it must fail. Then it runs those tests without a mutation: they must pass. Writes
OUT/mutations.json and exits 0 only if every mutation was caught and the unmutated
tests passed; 2 if the tests didn't run (no GPU)::

    python scripts/gpu/stream_mutations.py --out results/stream_mutations [--mutation NAME]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "test_gpu.py"
SUMMARY = re.compile(r"\b(\d+) (passed|failed|skipped|errors?|deselected)\b")


def mutations() -> dict[str, str]:
    """Each mutation's name and the test that must fail with it, from tests/test_gpu.py."""
    spec = importlib.util.spec_from_file_location("gpu_tests_for_mutations", TESTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: test for name, (_, test) in module.MUTATIONS.items()}


def command(tests: list[str]) -> list[str]:
    return [sys.executable, "-m", "pytest", "-m", "gpu", str(TESTS), "-q", "-p",
            "no:cacheprovider", "-k", " or ".join(tests)]


def run(tests: list[str], mutation: str | None, timeout: float) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "GPUBRIDGE_TEST_MUTATION"}
    if mutation:
        env["GPUBRIDGE_TEST_MUTATION"] = mutation
    start = time.monotonic()
    try:
        result = subprocess.run(command(tests), cwd=REPO, env=env, capture_output=True,
                                text=True, timeout=timeout)
        output, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        output, code = f"{exc.stdout or ''}{exc.stderr or ''}\nran past {timeout:g} s", None
    counts = {kind.rstrip("s"): int(n) for n, kind in SUMMARY.findall(output)}
    # The first lines that say what came out wrong (Mismatches) or why the run failed.
    evidence = [line.strip() for line in output.splitlines()
                if re.search(r"elements wrong|mismatch|differ|Error|FAILED", line)]
    wrong = [line for line in evidence if "elements wrong" in line]
    return {"tests": tests, "mutation": mutation, "exit_code": code, "counts": counts,
            "seconds": round(time.monotonic() - start, 1), "wrong_values": bool(wrong),
            "evidence": (wrong or evidence)[:8]}


def main(argv: list[str] | None = None) -> int:
    known = mutations()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "results" / "stream_mutations")
    parser.add_argument("--mutation", action="append", choices=sorted(known),
                        help="only these (default: all)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per pytest run")
    parser.add_argument("--dry-run", action="store_true", help="print the runs, run nothing")
    args = parser.parse_args(argv)
    chosen = args.mutation or list(known)
    plan = [(name, [known[name]]) for name in chosen]
    plan.append((None, sorted({known[name] for name in chosen})))
    if args.dry_run:
        for mutation, tests in plan:
            prefix = f"GPUBRIDGE_TEST_MUTATION={mutation} " if mutation else ""
            print(f"+ {prefix}{' '.join(command(tests))}")
        return 0

    rows = []
    for mutation, tests in plan:
        row = run(tests, mutation, args.timeout)
        ran = row["counts"].get("passed", 0) + row["counts"].get("failed", 0)
        wrong = row["wrong_values"]
        if row["exit_code"] is None:
            row["verdict"] = "timed out"
        elif row["counts"].get("error"):
            row["verdict"] = "error (collection or setup; see the evidence)"
        elif not ran:
            row["verdict"] = "did not run (no GPU?)"
        elif mutation and row["counts"].get("failed"):
            # Caught only if the test failed on wrong values, not by crashing.
            row["verdict"] = "caught" if wrong else "failed, but not on wrong values"
        elif mutation:
            row["verdict"] = "NOT CAUGHT"
        else:
            row["verdict"] = "passed" if not row["counts"].get("failed") else "FAILED"
        rows.append(row)
        print(f"{mutation or 'no mutation':<26} {', '.join(tests)}: {row['verdict']}"
              f" ({row['seconds']} s)", flush=True)
        for line in row["evidence"][:2] if mutation else []:
            print(f"    | {line[:150]}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mutations.json").write_text(json.dumps(rows, indent=2))
    if any(row["verdict"].startswith("did not run") for row in rows):
        print("the GPU tests didn't run: this needs a GPU", file=sys.stderr)
        return 2
    good = all(row["verdict"] in ("caught", "passed") for row in rows)
    print("every mutation was caught, and the tests pass without one" if good else
          "some mutation went uncaught, or the tests fail without one (above)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
