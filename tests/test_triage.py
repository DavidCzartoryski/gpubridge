"""scripts/gpu/triage.py on fake results trees: an Explorer step directory with a failure,
a timeout and benchmarks, a Kaggle run with skips, a CPU simulation, and a tarball."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "gpu"))

import triage  # noqa: E402

MiB = 2**20


def env(host: str, gpu: str = "NVIDIA A100-SXM4-40GB", device: str = "cuda:0") -> dict:
    return {"environment": {"hostname": host, "devices": [{"name": gpu}],
                            "driver_version": "550.54", "torch": "2.14.1", "torch_cuda": "12.6",
                            "nccl_version": "2.21.5", "git_commit": "abc1234", "git_dirty": False},
            "run": {"device": device, "kind": "split-test"}}


def steps(path: Path, *lines: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "steps.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


def explorer_tree(root: Path, *, digest: str = "aa", device: str = "cuda:0") -> Path:
    job = root / "explorer" / "04_split_4gpu"
    steps(job,
          {"step": "probe", "exit_code": 0, "seconds": 4, "log": "probe.log", "timed_out": False},
          {"step": "check", "exit_code": 1, "seconds": 31, "log": "check.log", "timed_out": False},
          {"step": "bench", "exit_code": 124, "seconds": 1200, "log": "bench.log", "timed_out": True},
          {"step": "bench_compression", "exit_code": 0, "seconds": 40, "log": "bc.log",
           "timed_out": False})
    (job / "check.log").write_text("starting\n" + "ok line\n" * 50 + "Traceback (most recent call last):\n"
                                   "  File check.py, line 9\nRuntimeError: NCCL error: unhandled "
                                   "system error\nstep check FAILED (exit 1)\n")
    (job / "bench.log").write_text("size 1M\nstep bench ran out of time (STEP_TIMEOUT=1200s)\n")
    (job / "check").mkdir(exist_ok=True)
    (job / "check" / "rank0.json").write_text(json.dumps({**env("node-a12", device=device),
                                                          "script": "scripts/gpu/check.py"}))
    rows = []
    for nbytes in (MiB, 64 * MiB):
        for op, us, bw in (("native", 100.0, 20.0), ("gpubridge", 400.0, 5.0)):
            rows.append({"bytes": nbytes, "op": op, "median_us": us * nbytes / MiB,
                         "busbw_GBps": bw, "timer": "cuda", "islands": 2, "policy": "rbb",
                         "phases_ms": {"bridge": 0.3 * nbytes / MiB} if op == "gpubridge" else {}})
    (job / "bench").mkdir(exist_ok=True)
    (job / "bench" / "bench.json").write_text(json.dumps(
        {**env("node-b07", device=device), "baseline": "native", "rows": rows,
         "run": {"kind": "split-test", "bridge": "gloo", "pinned_staging": False}}))
    (job / "preflight-node-a12.json").write_text(json.dumps(
        {"bandwidth": {"MBps_in": 1000.0, "MBps_out": 900.0, "rtt_ms": 0.05}}))
    (job / "compression.json").write_text(json.dumps({**env("node-a12", device=device), "checks": [
        {"mode": "bf16", "dtype": "float32", "matches_reference": True,
         "payload_sha256": digest, "result_sha256": digest},
        {"mode": "fp8", "dtype": "bfloat16", "matches_reference": False,
         "payload_sha256": "bb", "result_sha256": "bb"}]}))
    return job


def kaggle_tree(root: Path) -> Path:
    job = root / "kaggle-t4x2"
    steps(job, {"step": "probe", "exit_code": 0, "seconds": 3, "log": "probe.log", "timed_out": False},
          {"step": "gpu_tests_without_triton", "exit_code": 0, "seconds": 90,
           "log": "gpu_tests_without_triton.log", "timed_out": False})
    (job / "skipped.jsonl").write_text(json.dumps(
        {"step": "gpu_tests", "reason": "compute capability 7.5: triton_float32 failed"}) + "\n")
    (job / "features.json").write_text(json.dumps({**env("kaggle-host", gpu="Tesla T4"), "gpus": [
        {"index": i, "name": "Tesla T4", "capability": "7.5",
         "checks": {"cuda_events": {"ok": True}, "triton_float32": {"ok": False, "error": "x"}}}
        for i in range(2)]}))
    return job


def test_a_results_tree_becomes_a_short_paste(tmp_path, capsys):
    explorer_tree(tmp_path)
    kaggle_tree(tmp_path)
    assert triage.main([str(tmp_path)]) == 1  # something failed
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) <= 80
    assert lines[0].startswith("gpubridge results triage:") and "SIMULATION" not in lines[0]
    assert "== explorer/04_split_4gpu [split-test]" in out
    assert "GPUs NVIDIA A100-SXM4-40GB" in out and "CUDA 12.6" in out and "NCCL 2.21.5" in out
    assert "gpubridge abc1234" in out and "driver 550.54" in out
    assert "FAIL check: exit 1, 31 s" in out
    assert "RuntimeError: NCCL error: unhandled system error" in out  # the log's error, not "ok line"
    assert "ok line" not in out
    assert "TIMEOUT bench: ran out of time, 1200 s" in out
    assert "SKIP gpu_tests: compute capability 7.5: triton_float32 failed" in out
    assert "features: Tesla T4 (sm 7.5); not available: triton_float32" in out
    assert "gpubridge busbw (% of native): 1M 5.0 GB/s (25%)" in out
    assert "bridge vs predicted: 1M 0.3 ms vs " in out
    assert "compression kernels: 1 of 2 match the CPU reference; MISMATCH: fp8 bfloat16" in out
    assert lines[-1] == "verdict: 1 failed, 1 timed out, 1 skipped (reasons above)"
    # Hosts are numbered, never named; no path is printed.
    assert "node-a12" not in out and "node-b07" not in out and "host 1" in out
    assert str(tmp_path) not in out


def test_digests_are_compared_across_runs(tmp_path, capsys):
    explorer_tree(tmp_path / "nvidia", digest="aa")
    explorer_tree(tmp_path / "amd", digest="cc")
    triage.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "compression digests across 2 runs: DIFFER for bf16 float32" in out
    explorer_tree(tmp_path / "amd", digest="aa")
    triage.main([str(tmp_path)])
    assert "compression digests across 2 runs: DIFFER for fp8 bfloat16" not in capsys.readouterr().out


def test_cpu_simulation_says_so_and_a_clean_run_passes(tmp_path, capsys):
    job = tmp_path / "sim"
    steps(job, {"step": "probe", "exit_code": 0, "seconds": 2, "log": "probe.log", "timed_out": False})
    (job / "rank0.json").write_text(json.dumps({**env("laptop", device="cpu"),
                                                "script": "scripts/gpu/check.py"}))
    assert triage.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "CPU SIMULATION, says nothing about GPUs" in out.splitlines()[0]
    assert out.splitlines()[-1] == "verdict: every step that ran passed"


def test_a_tarball_reads_like_its_directory_and_unsafe_members_are_refused(tmp_path, capsys):
    kaggle_tree(tmp_path / "src")
    tarball = tmp_path / "kaggle.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(tmp_path / "src" / "kaggle-t4x2", arcname="kaggle-t4x2")
    assert triage.main([str(tarball)]) == 0
    from_tar = capsys.readouterr().out
    triage.main([str(tmp_path / "src")])
    from_dir = capsys.readouterr().out
    assert from_tar.splitlines()[1:] == from_dir.splitlines()[1:]  # only the label differs
    evil = tmp_path / "evil.tgz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(Exception):  # noqa: B017 - tarfile's filter error or ours
        triage.main([str(evil)])
    assert not (tmp_path / "outside.txt").exists()


def test_long_output_is_cut_to_the_limit_keeping_the_verdict(tmp_path, capsys):
    for i in range(12):
        explorer_tree(tmp_path / f"run{i}")
    triage.main([str(tmp_path), "--max-lines", "40"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 40
    assert lines[-1].startswith("verdict: 12 failed, 12 timed out")
    assert any(line.startswith("... ") and "more lines" in line for line in lines)


def test_not_a_results_directory(tmp_path, capsys):
    assert triage.main([str(tmp_path)]) == 1
    assert "no steps.jsonl" in capsys.readouterr().out
    assert triage.main([str(tmp_path / "missing")]) == 2
