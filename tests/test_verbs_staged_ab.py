"""scripts/gpu/verbs_staged_ab.py on CPU: one arm's run end to end on the MOCK test fabric,
and compare's predeclared rules on made-up records. Software checks only: the CPU runs
time the host path on a MOCK fabric, not the staged GPU path or a NIC."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile

import pytest
from test_gpu_kit import KIT, torchrun
from test_verbs import UNAVAILABLE

SCRIPT = KIT / "verbs_staged_ab.py"
needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"verbs test library unavailable: {UNAVAILABLE}")


def _run(out, arm: str, repeat: int, op: str, bridge: str, fabric: str):
    result = torchrun(4, SCRIPT, "run", "--out", str(out), "--arm", arm,
                      "--repeat", str(repeat), "--op", op, "--bridge", bridge,
                      "--table", "smoke", GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half", GPUBRIDGE_VERBS_TEST_FABRIC=fabric)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((out / f"{arm}-r{repeat}.json").read_text())


@needs_library
def test_a_run_keeps_every_sample_the_leaders_counts_and_its_provenance(tmp_path):
    fabric = tempfile.mkdtemp(prefix="gbab-", dir="/tmp")
    try:
        records = [_run(tmp_path, arm, 0, "gpubridge", "verbs", fabric)
                   for arm in ("before", "after")]
        native = _run(tmp_path, "native", 0, "native", "gloo", fabric)
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    for record in records:
        assert record["label"].startswith("CPU run")
        assert record["table"] == "smoke" and record["wall_s"] > 0
        assert len(record["sizes"]) == 2
        for ranks in record["sizes"]:
            assert len(ranks) == 4
            for r in ranks:
                assert r["mismatches"] == 0
                assert len(r["timed_s"]) == len(r["host_s"]) == len(r["cpu_s"]) == r["calls"]
                assert r["payload_bytes"] == r["bytes"]
                assert r["warmup_wall_s"] > 0 and r["timed_wall_s"] > 0
            leaders = [r for r in ranks if r["verbs_delta"] is not None]
            assert len(leaders) == 2  # one per island
            for r in leaders:
                d = r["verbs_delta"]
                # Smoke sizes are under one 4 MiB chunk: one send and one receive per call.
                assert d["sends"] == d["recvs"] == r["calls"], d
                # CPU tensors take the host path: no staged call, nothing timed.
                assert r["last_path"] == "host"
                assert d["staged_calls"] == d["staged_post_s"] == d["staged_wait_s"] == 0
        leaders = [p for p in record["provenance"] if "verbs_library_sha256" in p]
        assert len(leaders) == 2 and leaders[0]["verbs"]["mock"] is True
        assert len({p["gpubridge_py_sha256"] for p in record["provenance"]}) == 1
    assert all(r["verbs_delta"] is None for ranks in native["sizes"] for r in ranks)
    assert all(r["mismatches"] == 0 for ranks in native["sizes"] for r in ranks)
    # The same code in both arms puts the same messages on the wire: compare --wire finds
    # no count differing (timing verdicts on a loaded CPU are noise and not asserted).
    result = subprocess.run([sys.executable, str(SCRIPT), "compare", str(tmp_path),
                             "--before", "before", "--after", "after", "--wire"],
                            capture_output=True, text=True, timeout=60)
    assert "wire counts differ" not in result.stdout, result.stdout
    assert "FAIL" not in result.stdout, result.stdout


def _record(arm: str, repeat: int, p50: float, *, wrong: int = 0, sends: int = 10) -> dict:
    ranks = [{"bytes": 1024, "payload_bytes": 1024, "calls": 3, "timed_s": [p50] * 3,
              "cpu_s": [p50 / 2] * 3, "rss_growth_bytes": 0, "mismatches": wrong,
              "verbs_delta": {"sends": sends} if rank in (0, 2) else None}
             for rank in range(4)]
    return {"arm": arm, "repeat": repeat, "sizes": [ranks]}


@pytest.mark.parametrize("after, wrong, sends, verdict, code", [
    ([1.0, 1.0, 1.0, 1.0, 1.0], 0, 10, "no change (inside spread)", 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5], 0, 10, "win", 0),
    ([0.5, 0.5, 0.5, 0.5, 0.5], 0, 11, "REGRESSION", 1),  # different messages on the wire
    ([2.0, 2.0, 2.0, 2.0, 2.0], 0, 10, "REGRESSION", 1),
    ([0.5, 0.5, 0.5, 0.5, 0.5], 1, 10, "FAIL (wrong result)", 1),
])
def test_compare_applies_the_predeclared_rules(tmp_path, after, wrong, sends, verdict, code):
    before = [1.0, 1.02, 0.99, 1.01, 1.0]  # a 3% spread
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        (tmp_path / f"before-r{i}.json").write_text(json.dumps(_record("before", i, b)))
        (tmp_path / f"after-r{i}.json").write_text(
            json.dumps(_record("after", i, a, wrong=wrong, sends=sends)))
    result = subprocess.run([sys.executable, str(SCRIPT), "compare", str(tmp_path),
                             "--before", "before", "--after", "after", "--wire"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == code, result.stdout + result.stderr
    row, = json.loads((tmp_path / "compare-before-after.json").read_text())["rows"]
    assert row["verdict"] == verdict, row
