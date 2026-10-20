"""scripts/gpu/verbs_staged_ab.py on CPU: one arm's run end to end on the MOCK test fabric,
and compare's predeclared rules on made-up records. Software checks only: the CPU runs
time the host path on a MOCK fabric, not the staged GPU path or a NIC."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest
from test_gpu_kit import KIT, REPO, torchrun
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
    result = _compare(tmp_path, "before", "after", 1, 1, "--wire")
    assert result.returncode in (0, 1), result.stdout + result.stderr  # judged
    assert "wire counts differ" not in result.stdout, result.stdout
    assert "FAIL" not in result.stdout, result.stdout


def _compare(out, before: str, after: str, nb: int, na: int, *extra: str):
    return subprocess.run([sys.executable, str(SCRIPT), "compare", str(out), "--before", before,
                           "--after", after, "--before-repeats", str(nb),
                           "--after-repeats", str(na), *extra],
                          capture_output=True, text=True, timeout=60)


def _verdict(out, before: str, after: str) -> dict:
    return json.loads((out / f"compare-{before}-{after}.json").read_text())


def _record(arm: str, repeat: int, p50: float, *, wrong: int = 0, sends: int = 10,
            world: int = 4, policy: str = "reduce-bridge-broadcast", chunk: int = 4 << 20,
            telemetry: bool = True, code: str = "a" * 64) -> dict:
    """A whole record of the smoke table as ``run`` writes it, its two leaders ranks 0 and
    world // 2, every timed call ``p50`` seconds."""
    lead = (0, world // 2)
    wire = {k: sends for k in ("sends", "recvs", "bytes_sent", "bytes_received",
                               "writes_posted", "bytes_written", "write_windows",
                               "write_window_bytes")}
    sizes = [[{"rank": rank, "bytes": nbytes, "payload_bytes": nbytes, "warmup": warmup,
               "calls": calls, "timer": "host", "timed_s": [p50] * calls,
               "host_s": [p50] * calls, "cpu_s": [p50 / 2] * calls, "rss_growth_bytes": 0,
               "mismatches": wrong if rank == 1 else 0,
               "verbs_delta": dict(wire) if telemetry and rank in lead else None}
              for rank in range(world)] for nbytes, warmup, calls in [(4096, 2, 5),
                                                                      (2**20, 2, 5)]]
    environment = {"hostname": "node1", "platform": "Linux", "python": "3.12", "torch": "2.9",
                   "git_commit": "c0ffee", "git_dirty": False, "gpubridge": "0.1",
                   "import": {"gpubridge_git_commit": code[:7]},
                   "env": {"GPUBRIDGE_SPLIT_TEST": "half", "MASTER_PORT": str(29500 + repeat),
                           "GPUBRIDGE_VERBS_CHUNK_BYTES": str(chunk)}}
    provenance = [{"environment": environment, "gpubridge_py_sha256": code}
                  | ({"verbs_library_sha256": code, "verbs_crate_sha256": code,
                      "verbs": {"chunk_bytes": chunk, "progress": "economy"}}
                     if rank in lead else {}) for rank in range(world)]
    islands = [{"vendor": "nvidia", "ranks": list(range(i, i + world // 2)), "leader": i}
               for i in lead]
    run = {"kind": "gpu-split", "split_test": True, "world_size": world, "islands": islands,
           "bridge": "verbs", "policy": policy, "pinned_staging": True}
    return {"arm": arm, "repeat": repeat, "op": "gpubridge", "bridge": "verbs",
            "policy": "auto", "table": "smoke", "run": run, "provenance": provenance,
            "sizes": sizes}


def _write(out, arm: str, repeats, p50=1.0, **kwargs) -> None:
    for i in repeats:
        (out / f"{arm}-r{i}.json").write_text(json.dumps(_record(arm, i, p50, **kwargs)))


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
    result = _compare(tmp_path, "before", "after", 5, 5, "--wire")
    assert result.returncode == code, result.stdout + result.stderr
    rows = _verdict(tmp_path, "before", "after")["rows"]
    assert len(rows) == 2 and {row["verdict"] for row in rows} == {verdict}, rows


def test_compare_needs_every_predeclared_repeat_and_checks_every_run(tmp_path):
    # The review's case: B0's last repeat is wrong and P0's is missing. Pairing the
    # repeats both arms have used to drop the wrong one and pass.
    _write(tmp_path, "B0", [0, 1])
    _write(tmp_path, "B0", [2], wrong=1)
    _write(tmp_path, "P0", [0, 1])
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 1, result.stdout + result.stderr
    verdict = _verdict(tmp_path, "B0", "P0")
    assert verdict["status"] == "FAIL (wrong result)" and verdict["rows"] == []
    assert any("B0-r2.json: 2 wrong results" in r for r in verdict["reasons"]), verdict
    assert any("P0: repeats [2]" in r for r in verdict["reasons"]), verdict
    # Missing, and nothing wrong: INCOMPLETE, never a verdict on the repeats that are there.
    (tmp_path / "B0-r2.json").write_text(json.dumps(_record("B0", 2, 1.0)))
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 4 and "INCOMPLETE" in result.stderr, result.stderr
    # A repeat outside the predeclared set, or a truncated one, is INCOMPLETE too.
    _write(tmp_path, "P0", [2, 3])
    assert _compare(tmp_path, "B0", "P0", 3, 3).returncode == 4
    (tmp_path / "P0-r3.json").unlink()
    assert _compare(tmp_path, "B0", "P0", 3, 3).returncode == 0
    truncated = _record("P0", 2, 1.0)
    truncated["sizes"][1][3]["timed_s"].pop()
    (tmp_path / "P0-r2.json").write_text(json.dumps(truncated))
    result = _compare(tmp_path, "B0", "P0", 3, 3)
    assert result.returncode == 4 and "samples [4, 5] for 5 calls" in result.stderr
    # The longer arm's unpaired repeat (P0 has 3, D1 2) is still checked for correctness.
    _write(tmp_path, "P0", [2], wrong=1)
    _write(tmp_path, "D1", [0, 1])
    result = _compare(tmp_path, "P0", "D1", 3, 2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _verdict(tmp_path, "P0", "D1")["status"] == "FAIL (wrong result)"
    # So is an incomplete one: its wrong result fails the comparison, not only INCOMPLETE.
    _write(tmp_path, "P0", [2])
    broken = _record("D1", 1, 1.0, wrong=1)
    broken["sizes"][0][3]["timed_s"].pop()
    (tmp_path / "D1-r1.json").write_text(json.dumps(broken))
    result = _compare(tmp_path, "P0", "D1", 3, 2)
    assert result.returncode == 1, result.stdout + result.stderr
    verdict = _verdict(tmp_path, "P0", "D1")
    assert verdict["status"] == "FAIL (wrong result)" and verdict["rows"] == [], verdict
    assert any("D1-r1.json: 2 wrong results" in r for r in verdict["reasons"]), verdict
    assert any("samples [4, 5] for 5 calls" in r for r in verdict["reasons"]), verdict


@pytest.mark.parametrize("change, differ", [
    ({"world": 2}, ["topology", "verbs", "code"]),  # 1+1 against 2+2: never comparable
    ({"policy": "native-only"}, ["policy"]),
    ({"chunk": 16 << 20}, ["env.GPUBRIDGE_VERBS_CHUNK_BYTES", "verbs.chunk_bytes"]),
    ({"code": "b" * 64}, ["code"]),
])
def test_compare_refuses_arms_that_differ_in_anything_not_declared(tmp_path, change, differ):
    _write(tmp_path, "B0", range(3))
    _write(tmp_path, "P0", range(3), p50=0.5, **change)
    for wire in ([], ["--wire"]):
        result = _compare(tmp_path, "B0", "P0", 3, 3, *wire)
        assert result.returncode == 5 and "REFUSED" in result.stderr, result.stderr
        verdict = _verdict(tmp_path, "B0", "P0")
        assert verdict["status"] == "REFUSED" and verdict["rows"] == [], verdict
    if "world" in change:
        return
    declared = [arg for key in differ for arg in ("--differ", key)]
    result = _compare(tmp_path, "B0", "P0", 3, 3, *declared)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {row["verdict"] for row in _verdict(tmp_path, "B0", "P0")["rows"]} == {"win"}
    # Every repeat of one arm must match its repeat 0, whatever --differ allows.
    _write(tmp_path, "P0", [1], p50=0.5)
    assert _compare(tmp_path, "B0", "P0", 3, 3, *declared).returncode == 5


def test_compare_wire_needs_every_leaders_counted_messages(tmp_path):
    _write(tmp_path, "B0", range(3), telemetry=False)
    _write(tmp_path, "P0", range(3), telemetry=False)
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 3 and "UNSUPPORTED" in result.stderr, result.stderr
    _write(tmp_path, "B0", range(3))
    _write(tmp_path, "P0", range(3))
    assert _compare(tmp_path, "B0", "P0", 3, 3, "--wire").returncode == 0
    one = _record("P0", 1, 1.0)
    one["sizes"][0][2]["verbs_delta"] = None  # one leader's telemetry missing
    (tmp_path / "P0-r1.json").write_text(json.dumps(one))
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 4 and "from 1 of 2 leaders" in result.stderr, result.stderr
    counter = _record("P0", 1, 1.0)
    del counter["sizes"][1][0]["verbs_delta"]["write_windows"]
    (tmp_path / "P0-r1.json").write_text(json.dumps(counter))
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 4 and "lacks ['write_windows']" in result.stderr, result.stderr


DOC = REPO / "docs" / "verbs-perf-20261019.md"


def _section(heading: str) -> str:
    text = DOC.read_text()
    start = text.index(f"\n## {heading}")
    return text[start:text.find("\n## ", start + 1)]


@pytest.mark.docs
def test_the_copy_trigger_uses_only_measured_phases():
    # Call latency minus the host phases also holds the island broadcast and framework
    # work, and the host's enqueue can overlap GPU work: it is no measure of the copy back.
    trigger, = [line for line in _section("5.").splitlines()
                if line.startswith("|") and "candidate A" in line]
    assert "minus" not in trigger and "copy back" not in trigger, trigger
    assert "staged_to_host_s / staged_calls" in trigger and "P0x2" in trigger, trigger
    assert "H2D" in _section("2.") and "UNAVAILABLE" in _section("2."), _section("2.")


@pytest.mark.docs
def test_the_setup_makes_every_directory_the_owner_commands_write_to(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("no bash")
    section = _section("4.")
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
    setup = section.split("### Setup", 1)[1]
    setup = re.search(r"```bash\n(.*?)```", setup, flags=re.DOTALL).group(1)
    # Stand-ins for the node: a library build, nvidia-smi, two checkouts.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text('#!/bin/sh\nmkdir -p "$GPUBRIDGE_VERBS_LIB_DIR" && '
                                    'touch "$GPUBRIDGE_VERBS_LIB_DIR/libgpubridge_verbs-x.so"\n')
    (bin_dir / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "OUT": str(tmp_path / "out")}
    for name in ("BASE", "PR0"):
        tree = tmp_path / name
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "--allow-empty", "-m", "t"], check=True)
        env[name] = str(tree)
    script = "set -e\n" + re.sub(r"^export BASE=.*$", "", setup, flags=re.MULTILINE)
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    written = {m for block in blocks for m in re.findall(r'"\$OUT/([^"]+)"', block)}
    assert {"m1/$ARM-r$R.time", "m2/N-r$R.time"} <= written, written
    for path in written:
        parent = (tmp_path / "out" / path).parent
        assert parent.is_dir(), f"{path}: {parent} does not exist after the setup"
