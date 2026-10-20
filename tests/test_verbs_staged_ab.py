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
from types import SimpleNamespace

import pytest
from test_gpu_kit import KIT, REPO, torchrun
from test_verbs import UNAVAILABLE

import verbs_staged_ab as ab  # on the path once test_gpu_kit is imported

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
                assert r["last_path"] == "host" and r["last_staging"] is None
                assert d["staged_calls"] == d["staged_post_s"] == d["staged_wait_s"] == 0
        leaders = [p for p in record["provenance"] if "verbs_library_sha256" in p]
        assert len(leaders) == 2 and leaders[0]["verbs"]["mock"] is True
        assert all(p["verbs"]["staged_overlap_setting"] is False for p in leaders)
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
            telemetry: bool = True, code: str = "a" * 64, table: str = "smoke",
            staged=None) -> dict:
    """A whole record of a table as ``run`` writes it, its two leaders ranks 0 and
    world // 2, every timed call ``p50`` seconds; ``staged(nbytes, rank, calls)`` gives
    a leader's staged phase deltas."""
    lead = (0, world // 2)
    wire = {k: sends for k in ("sends", "recvs", "bytes_sent", "bytes_received",
                               "writes_posted", "bytes_written", "write_windows",
                               "write_window_bytes")}
    sizes = [[{"rank": rank, "bytes": nbytes, "payload_bytes": nbytes, "warmup": warmup,
               "calls": calls, "timer": "host", "timed_s": [p50] * calls,
               "host_s": [p50] * calls, "cpu_s": [p50 / 2] * calls, "rss_growth_bytes": 0,
               "mismatches": wrong if rank == 1 else 0,
               "verbs_delta": wire | (staged(nbytes, rank, calls) if staged else {})
               if telemetry and rank in lead else None}
              for rank in range(world)] for nbytes, warmup, calls in ab.TABLES[table]]
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
            "policy": "auto", "table": table, "run": run, "provenance": provenance,
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


@pytest.mark.parametrize("text", ["", '{"arm": "P0", "rep', "[]", "null", '"P0"', "3", "{}"])
def test_compare_reports_an_unreadable_run_incomplete_and_writes_its_verdict(tmp_path, text):
    _write(tmp_path, "B0", range(3))
    _write(tmp_path, "P0", range(3))
    (tmp_path / "D1-r0.json").write_text("")  # another arm's: not this comparison's
    assert _compare(tmp_path, "B0", "P0", 3, 3, "--wire").returncode == 0
    (tmp_path / "P0-r2.json").write_text(text)
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 4 and "Traceback" not in result.stderr, result.stderr
    verdict = _verdict(tmp_path, "B0", "P0")
    assert verdict["status"] == "INCOMPLETE" and verdict["rows"] == [], verdict
    assert any(r.startswith("P0-r2.json: unreadable (") for r in verdict["reasons"]), verdict
    assert any("P0: repeats [2]" in r for r in verdict["reasons"]), verdict


@pytest.mark.parametrize("mismatches", ["3", True, 1.0, None, [1]])
def test_compare_counts_only_integer_mismatches_and_reports_the_rest(tmp_path, mismatches):
    _write(tmp_path, "B0", range(3))
    _write(tmp_path, "P0", range(3))
    record = _record("P0", 2, 1.0)
    record["sizes"][0][1]["mismatches"] = mismatches
    (tmp_path / "P0-r2.json").write_text(json.dumps(record))
    result = _compare(tmp_path, "B0", "P0", 3, 3)
    assert result.returncode == 4 and "Traceback" not in result.stderr, result.stderr
    verdict = _verdict(tmp_path, "B0", "P0")
    assert verdict["status"] == "INCOMPLETE", verdict
    assert any(f"mismatches {mismatches!r}" in r for r in verdict["reasons"]), verdict


class _Bridge:
    """A verbs bridge whose telemetry lacks the WIRE count ``writes_posted`` and the
    direct count ``write_window_release_ns``, and counts one send per telemetry read."""

    name = "verbs"

    def __init__(self):
        self.reads = 0

    def telemetry(self) -> dict:
        self.reads += 1
        wire = {k: self.reads for k in ab.COUNTS if k != "writes_posted"}
        direct = {k: self.reads for k in ab.DIRECT_COUNTS if k != "write_window_release_ns"}
        return {**wire, "direct": direct, "staged_phases": {"calls": 0}}


def test_a_count_missing_from_the_telemetry_is_recorded_missing_never_zero(tmp_path):
    # Through run's own code: verbs_counts and time_size's delta on a one-rank group.
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0,
                            world_size=1)
    try:
        topology = SimpleNamespace(device=torch.device("cpu"), world_size=1, rank=0,
                                   bridge=_Bridge())
        assert ab.verbs_counts(topology.bridge)["writes_posted"] is None
        size = ab.time_size(lambda t: None, topology, 4096, 1, 5)
    finally:
        dist.destroy_process_group()
    delta = size["verbs_delta"]
    assert delta["writes_posted"] is None and delta["write_window_release_ns"] is None
    assert delta["sends"] == 1 and delta["write_windows"] == 1, delta
    # That delta as both leaders' at 4 KiB: compare --wire finds the WIRE count missing.
    _write(tmp_path, "B0", range(3))
    _write(tmp_path, "P0", range(3))
    record = _record("P0", 1, 1.0)
    for leader in (0, 2):
        record["sizes"][0][leader]["verbs_delta"] = delta
    (tmp_path / "P0-r1.json").write_text(json.dumps(record))
    result = _compare(tmp_path, "B0", "P0", 3, 3, "--wire")
    assert result.returncode == 4 and "lacks ['writes_posted']" in result.stderr, result.stderr
    assert _verdict(tmp_path, "B0", "P0")["status"] == "INCOMPLETE"
    # Without --wire it is not gated, and a summed count with a missing part is missing.
    assert ab.summarize(record)[4096]["counts"]["writes_posted"] is None
    assert _compare(tmp_path, "B0", "P0", 3, 3).returncode == 0


def _trigger(out, *extra: str):
    """Section 5's owner command: stdout and stderr into one trigger.txt (``2>&1``); the
    file's text as ``stdout``."""
    path = out / "trigger.txt"
    with path.open("w") as f:
        code = subprocess.run([sys.executable, str(SCRIPT), "trigger", str(out), "--arm",
                               "P0x2", "--repeats", "3", "--compare", "compare-P0-D3.json",
                               *extra], stdout=f, stderr=subprocess.STDOUT,
                              timeout=60).returncode
    return SimpleNamespace(returncode=code, stdout=path.read_text())


def _pure_bridge(out, shares, *, d3_p50=1.0):
    """P0x2's 3 repeats of the full table, every call 1 s, leader rank 1's copy out
    ``shares[k]`` of the call at 25 MiB in repeat k (0.1 everywhere else); then P0 and
    D3, and section 5's compare P0 D3 (D3 every call ``d3_p50`` s)."""
    for k, share in enumerate(shares):
        def staged(nbytes, rank, calls, share=share):
            s = share if (nbytes, rank) == (25 * 2**20, 1) else 0.1
            return {"staged_calls": calls, "staged_to_host_s": s * calls}
        record = _record("P0x2", k, 1.0, world=2, table="full", staged=staged)
        (out / f"P0x2-r{k}.json").write_text(json.dumps(record))
    _write(out, "P0", range(3), table="full")
    _write(out, "D3", range(2), p50=d3_p50, table="full", policy="pipelined")
    assert _compare(out, "P0", "D3", 3, 2, "--differ", "policy").returncode == 0


def _edit(out, name: str, change) -> None:
    record = json.loads((out / name).read_text())
    change(record)
    (out / name).write_text(json.dumps(record))


def _incomplete(out, reason: str) -> None:
    result = _trigger(out)
    lines = result.stdout.splitlines()
    assert result.returncode == 4 and lines[-1] == "INCOMPLETE", result.stdout
    assert f"not decided: {reason}" in result.stdout and "TRIGGER" not in result.stdout, \
        result.stdout


@pytest.mark.parametrize("shares, d3_p50, verdict, met", [
    ([0.31, 0.29, 0.35], 1.0, "TRIGGER MET",
     ["met: P0x2 leader rank 1 at 26214400 bytes, median 0.310 >= 0.3"]),
    ([0.29, 0.31, 0.2], 1.0, "TRIGGER NOT MET", []),  # one repeat over, the median not
    ([0.1, 0.1, 0.1], 0.5, "TRIGGER MET", ["met: compare-P0-D3.json says win at 26214400 bytes",
                                           "met: compare-P0-D3.json says win at 157286400 bytes"]),
])
def test_trigger_decides_section_5s_copy_overlap_row(tmp_path, shares, d3_p50, verdict, met):
    _pure_bridge(tmp_path, shares, d3_p50=d3_p50)
    result = _trigger(tmp_path)
    assert result.returncode == 0, result.stdout
    lines = result.stdout.splitlines()
    # The reasons, then the verdict alone on the file's last line.
    assert lines[-1] == verdict and lines[-1 - len(met):-1] == met, lines
    assert not any(line.startswith("met: ") for line in lines[:-1 - len(met)]), lines
    per_repeat = " ".join(f"{v:.3f}" for v in shares)
    assert (f"P0x2 26214400 bytes, leader rank 1: (staged_to_host_s / staged_calls) / p50 "
            f"per repeat {per_repeat}, median {sorted(shares)[1]:.3f}") in lines, lines
    # Every leader at both sizes, then the D3 compare's status.
    assert sum(line.startswith("P0x2 ") for line in lines) == 4, lines
    wins = "[26214400, 157286400]" if d3_p50 < 1 else "[]"
    assert f"compare-P0-D3.json: judged, win at {wins} bytes" in lines, lines


def test_trigger_decides_nothing_on_an_incomplete_record(tmp_path):
    _pure_bridge(tmp_path, [0.5, 0.5, 0.5])
    assert _trigger(tmp_path).returncode == 0
    for name, broken in [("P0x2-r2.json", "["), ("compare-P0-D3.json", "")]:
        good = (tmp_path / name).read_text()
        (tmp_path / name).write_text(broken)
        _incomplete(tmp_path, f"{name}: unreadable")
        (tmp_path / name).write_text(good)
    _edit(tmp_path, "P0x2-r1.json",
          lambda r: r["sizes"][3][1]["verbs_delta"].pop("staged_calls"))
    _incomplete(tmp_path, "P0x2-r1.json: 26214400 bytes, rank 1: staged phases missing")
    (tmp_path / "P0x2-r1.json").unlink()
    _incomplete(tmp_path, "P0x2: repeats [0, 2], not the predeclared 0 to 2")


def test_trigger_reads_only_a_judged_compare_of_its_arms(tmp_path):
    _pure_bridge(tmp_path, [0.5, 0.5, 0.5])  # met by the share alone
    compare = tmp_path / "compare-P0-D3.json"
    good = compare.read_text()
    _edit(tmp_path, compare.name, lambda v: v.update(status="REGRESSION"))
    assert _trigger(tmp_path).stdout.splitlines()[-1] == "TRIGGER MET"  # judged rows
    for status in ("INCOMPLETE", "UNSUPPORTED", "REFUSED", "FAIL (wrong result)"):
        compare.write_text(good)
        _edit(tmp_path, compare.name, lambda v, status=status: v.update(status=status))
        _incomplete(tmp_path, f"compare-P0-D3.json: {status}, not a judged comparison")
    compare.write_text(good)
    _edit(tmp_path, compare.name, lambda v: v.update(after="D1"))
    _incomplete(tmp_path, "compare-P0-D3.json: holds the compare of 'P0' against 'D1'")
    compare.write_text(good)
    _edit(tmp_path, compare.name, lambda v: v.update(rows=v["rows"][:-1]))
    _incomplete(tmp_path, "compare-P0-D3.json: no row at [157286400] bytes")
    # The real compare with a D3 repeat gone: INCOMPLETE, and so is the trigger.
    compare.write_text(good)
    (tmp_path / "D3-r1.json").unlink()
    assert _compare(tmp_path, "P0", "D3", 3, 2, "--differ", "policy").returncode == 4
    _incomplete(tmp_path, "compare-P0-D3.json: INCOMPLETE, not a judged comparison")


def test_trigger_needs_three_shares_of_the_same_two_leaders(tmp_path):
    _pure_bridge(tmp_path, [0.5, 0.5, 0.5])
    good = {k: (tmp_path / f"P0x2-r{k}.json").read_text() for k in range(3)}

    def restore():
        for k, text in good.items():
            (tmp_path / f"P0x2-r{k}.json").write_text(text)

    # A leader that is not the recorded one in one repeat: no median from fewer repeats.
    _edit(tmp_path, "P0x2-r1.json",
          lambda r: [ranks[1].update(rank=99) for ranks in r["sizes"]])
    _incomplete(tmp_path, "P0x2-r1.json: 26214400 bytes: staged phases from ranks [0, 99], "
                          "the leaders are [0, 1]")
    assert ("not decided: P0x2 26214400 bytes, leader rank 1: 2 shares, not one per "
            "predeclared repeat (3)") in _trigger(tmp_path).stdout
    restore()
    # Staged phases over fewer calls than were timed: the share would be inflated.
    for k in range(3):
        _edit(tmp_path, f"P0x2-r{k}.json", lambda r: [
            x["verbs_delta"].update(staged_calls=1) for ranks in r["sizes"] for x in ranks])
    _incomplete(tmp_path, "P0x2-r0.json: 26214400 bytes, rank 0: staged phases missing or "
                          "partial (staged_calls 1 of 30 calls")
    restore()
    # The repeats' workloads differ (another chunk size in one).
    (tmp_path / "P0x2-r2.json").write_text(json.dumps(_record(
        "P0x2", 2, 1.0, world=2, table="full", chunk=1 << 20,
        staged=lambda n, rank, calls: {"staged_calls": calls, "staged_to_host_s": calls})))
    _incomplete(tmp_path, "P0x2-r2.json differs from P0x2-r0.json in ")
    # Not the pure bridge: 2+2 ranks.
    for k in range(3):
        (tmp_path / f"P0x2-r{k}.json").write_text(json.dumps(_record(
            "P0x2", k, 1.0, table="full",
            staged=lambda n, rank, calls: {"staged_calls": calls, "staged_to_host_s": calls})))
    _incomplete(tmp_path, "P0x2-r0.json: not the pure bridge (op 'gpubridge', bridge 'verbs', "
                          "islands [[0, 1], [2, 3]]), which is 1+1 on verbs")


def test_trigger_rejects_a_file_named_for_the_arm_with_no_record_of_it(tmp_path):
    _pure_bridge(tmp_path, [0.5, 0.5, 0.5])
    for text, arm in [("{}", None), (json.dumps({"arm": "P0", "repeat": 3}), "'P0'")]:
        (tmp_path / "P0x2-r3.json").write_text(text)
        _incomplete(tmp_path, f"P0x2-r3.json: unreadable (no record of P0x2: arm {arm})")


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
def test_section_5s_rows_are_mechanical():
    section = _section("5.")
    rows = [line for line in section.splitlines() if line.startswith("| ")][-4:]
    assert "write_window_release_ns) / 1e9" in rows[1], rows[1]
    assert "first row met picks the next change" in section
    command, = re.findall(r'python "\$H" trigger [^\n]*\\\n[^\n]*', section)
    assert "--arm P0x2 --repeats 3 --compare compare-P0-D3.json" in command, command
    assert ab.TRIGGER_SIZES == (25 * 2**20, 150 * 2**20) and ab.TRIGGER_SHARE == 0.30
    assert "at 25 or 150 MiB" in rows[0] and "at least 0.30" in rows[0], rows[0]
    # What the trigger tests check of the file: its verdict alone on the last line.
    assert ("last a line that is exactly `TRIGGER MET`, `TRIGGER NOT MET` or `INCOMPLETE`"
            in " ".join(section.split())), section


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
