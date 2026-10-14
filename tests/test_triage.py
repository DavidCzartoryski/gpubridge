"""scripts/gpu/triage.py on fake results trees: an Explorer step directory with a failure,
a timeout and benchmarks, a Kaggle run with skips, a mixed job's two compression records,
a CPU simulation, killed and rerun steps, redaction, and tarballs."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import tempfile
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


def ok(name: str) -> dict:
    return {"step": name, "exit_code": 0, "seconds": 3, "log": f"{name}.log", "timed_out": False}


TORCHRUN_FAILURE = """\
Traceback (most recent call last):
  File "/projects/secret-lab/alice/gpubridge/scripts/gpu/check.py", line 9, in <module>
RuntimeError: NCCL error: unhandled system error, connecting to 10.99.1.8 on gpu-node-07
E1014 torch/distributed/elastic/multiprocessing/api.py] failed (exitcode: 1) local_rank: 0
Root Cause (first observed failure):
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================

==> NCCL/RCCL log nccl/check/gpu-node-07.123.log (last 40 lines) <==
gpu-node-07:123:150 [0] NCCL INFO Bootstrap: Using ib0:10.99.1.7<0>
"""


def explorer_tree(root: Path, *, digest: str = "aa", device: str = "cuda:0",
                  job: str = "04_split_4gpu") -> Path:
    path = root / "explorer" / job
    steps(path, ok("probe"),
          {"step": "check", "exit_code": 1, "seconds": 31, "log": "check.log", "timed_out": False},
          {"step": "bench", "exit_code": 124, "seconds": 1200, "log": "bench.log", "timed_out": True},
          ok("bench_compression"))
    (path / "check.log").write_text("starting\n" + "ok line\n" * 50 + TORCHRUN_FAILURE)
    (path / "bench.log").write_text("size 1M\nstep bench ran out of time (STEP_TIMEOUT=1200s)\n")
    (path / "check").mkdir(exist_ok=True)
    (path / "check" / "rank0.json").write_text(json.dumps({**env("gpu-node-07", device=device),
                                                           "script": "scripts/gpu/check.py"}))
    rows = []
    for nbytes in (MiB, 64 * MiB):
        for op, us, bw in (("native", 100.0, 20.0), ("gpubridge", 400.0, 5.0)):
            rows.append({"bytes": nbytes, "op": op, "median_us": us * nbytes / MiB,
                         "busbw_GBps": bw, "timer": "cuda-events", "islands": 2, "policy": "rbb",
                         "phases_ms": {"bridge": 0.3 * nbytes / MiB} if op == "gpubridge" else {}})
    (path / "bench").mkdir(exist_ok=True)
    (path / "bench" / "bench.json").write_text(json.dumps(
        {**env("gpu-node-08", device=device), "baseline": "native", "rows": rows,
         "run": {"kind": "split-test", "bridge": "gloo", "pinned_staging": False}}))
    (path / "preflight-gpu-node-07.json").write_text(json.dumps(
        {"bandwidth": {"MBps_in": 1000.0, "MBps_out": 900.0, "rtt_ms": 0.05}}))
    (path / "compression.json").write_text(json.dumps({
        **env("gpu-node-07", device=device), "impl": "triton", "device": "cuda", "checks": [
            {"mode": "bf16", "dtype": "float32", "matches_reference": True,
             "payload_sha256": digest, "result_sha256": digest},
            {"mode": "fp8", "dtype": "bfloat16", "matches_reference": False,
             "payload_sha256": "bb", "result_sha256": "bb"}]}))
    return path


def kaggle_tree(root: Path) -> Path:
    path = root / "kaggle-t4x2"
    steps(path, ok("probe"), ok("gpu_tests_without_triton"),
          {**ok("bench_compression_kernels"), "note": "float32 inputs only: bfloat16 failed here"})
    (path / "skipped.jsonl").write_text(json.dumps(
        {"step": "gpu_tests", "reason": "compute capability 7.5: triton_float32 failed:\n"
         "CUDA error: no kernel image is available\nfor execution on the device\n"}) + "\n")
    (path / "features.json").write_text(json.dumps({**env("kaggle-0123", gpu="Tesla T4"), "gpus": [
        {"index": i, "name": "Tesla T4", "capability": "7.5",
         "checks": {"cuda_events": {"ok": True}, "triton_float32": {"ok": False, "error": "x"}}}
        for i in range(2)]}))
    return path


def run(path: Path, *args: str, capsys) -> tuple[int, str]:
    code = triage.main([str(path), *args])
    return code, capsys.readouterr().out


def test_a_results_tree_becomes_a_short_redacted_paste(tmp_path, capsys):
    explorer_tree(tmp_path)
    kaggle_tree(tmp_path)
    code, out = run(tmp_path, capsys=capsys)
    lines = out.splitlines()
    assert code == 1 and len(lines) <= 80
    assert lines[0].startswith("gpubridge results triage:") and "SIMULATION" not in lines[0]
    assert "== explorer/04_split_4gpu [split-test]" in out
    assert "GPUs NVIDIA A100-SXM4-40GB" in out and "CUDA 12.6" in out and "NCCL 2.21.5" in out
    assert "gpubridge abc1234" in out and "driver 550.54" in out
    assert "  FAIL check: exit 1, 31 s" in out
    assert "RuntimeError: NCCL error: unhandled system error" in out  # the error, not the footer
    assert "ok line" not in out and "error_file" not in out and "NCCL INFO" not in out
    assert "  TIMEOUT bench: ran out of time, 1200 s" in out
    assert "  SKIP gpu_tests: compute capability 7.5: triton_float32 failed: CUDA error" in out
    assert "  PASS bench_compression_kernels; note: float32 inputs only" in out
    assert "features: Tesla T4 (sm 7.5); not available: triton_float32" in out
    assert "gpubridge busbw, % of native: 1M 5.0 GB/s (25%); 64M 5.0 GB/s (25%)" in out
    assert "bridge vs predicted, bench/: 1M 0.3 ms vs " in out
    assert ("compression ./ (triton kernels on cuda): 1 of 2 match the CPU reference; "
            "MISMATCH: fp8 bfloat16") in out
    assert lines[-1] == "verdict: 1 failed, 1 timed out; 1 skipped (reasons above)"
    # Redacted: hosts numbered, no IP, no project or user name.
    for secret in ("gpu-node-07", "gpu-node-08", "kaggle-0123", "10.99.1.8", "secret-lab", "alice",
                   str(tmp_path)):
        assert secret not in out, secret
    assert "host 1" in out and "<ip>" in out


def test_redaction_of_paths_hosts_and_addresses():
    hosts = triage.Hosts()
    hosts("c2187.explorer.example.edu")
    redact = triage.Redact(hosts)
    assert redact("/projects/lab/alice/gpubridge/x.py on c2187 (c2187.explorer.example.edu) "
                  "and /home/bob/.cache, 10.0.0.12") == (
        "$PROJECT_DIR/gpubridge/x.py on host 1 (host 1) and ~/.cache, <ip>")
    assert redact("c21870 and xc2187") == "c21870 and xc2187"  # whole names only


def test_problems_come_first_and_the_limit_counts_printed_lines(tmp_path, capsys):
    for i in range(30):
        steps(tmp_path / f"ok{i:02d}", ok("probe"), ok("check"))
    explorer_tree(tmp_path, job="zz_broken")
    kaggle_tree(tmp_path)  # a skip reason over several lines
    code, out = run(tmp_path, "--max-lines", "40", capsys=capsys)
    lines = out.splitlines()
    assert code == 1 and len(lines) <= 40
    assert any("FAIL check" in line for line in lines)  # the only failure survives the cut
    assert lines[-1].startswith("verdict: 1 failed, 1 timed out")
    assert "== ok00: all 2 steps passed" in out or "more lines" in out


def test_digests_are_compared_across_every_compression_record(tmp_path, capsys):
    path = explorer_tree(tmp_path, digest="aa")
    other = path / "bench_compression_amd"  # a mixed job: one record per vendor, in one job
    other.mkdir()
    (other / "compression.json").write_text(json.dumps({"impl": "triton", "device": "cuda", "checks": [
        {"mode": "bf16", "dtype": "float32", "matches_reference": True,
         "payload_sha256": "cc", "result_sha256": "cc"}]}))
    code, out = run(tmp_path, capsys=capsys)
    assert "compression digests across 2 records: DIFFER for bf16 float32" in out
    assert out.splitlines()[-1].endswith("compression digests DIFFER")
    (other / "compression.json").write_text(json.dumps({"checks": [
        {"mode": "bf16", "dtype": "float32", "matches_reference": True,
         "payload_sha256": "aa", "result_sha256": "aa"}]}))
    code, out = run(tmp_path, capsys=capsys)
    assert "compression digests across 2 records: all agree" in out


def test_killed_rerun_and_negative_steps(tmp_path, capsys):
    path = tmp_path / "02_split_shared_gpu"
    steps(path,
          {"step": "bench", "exit_code": 1, "seconds": 9, "log": "bench.log", "timed_out": False},
          ok("bench"),  # run again: the last attempt counts
          {**ok("shared_gpu_check"), "result": "expected failure confirmed"},
          {"step": "shared_gpu_check2", "exit_code": 1, "seconds": 60, "log": "x.log",
           "timed_out": False, "result": "check did not fire"},
          {"step": "outside", "exit_code": 1, "seconds": 1, "log": "../../../../etc/hosts",
           "timed_out": False})
    (path / "steps.jsonl").write_text((path / "steps.jsonl").read_text() + '{"step": "cut sh')
    (path / "gpu_tests.log").write_text("collecting ...\nKilled\n")  # no steps.jsonl line
    code, out = run(tmp_path, capsys=capsys)
    assert code == 1
    assert "FAIL bench" not in out  # passed on its second attempt
    assert "FAIL shared_gpu_check2: check did not fire" in out
    assert "shared_gpu_check:" not in out  # an expected failure, confirmed: a pass
    assert "UNFINISHED gpu_tests: a log but no steps.jsonl line" in out and "| Killed" in out
    assert "(log outside the results directory)" in out and "localhost" not in out


def test_cpu_simulation_says_so_and_a_clean_run_passes(tmp_path, capsys):
    path = tmp_path / "sim"
    steps(path, ok("probe"))
    (path / "rank0.json").write_text(json.dumps({**env("laptop", device="cpu"),
                                                 "script": "scripts/gpu/check.py"}))
    code, out = run(tmp_path, capsys=capsys)
    assert code == 0
    assert "CPU SIMULATION, says nothing about GPUs" in out.splitlines()[0]
    assert out.splitlines()[-1] == "verdict: every step that ran passed"


def _tarball(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _file(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _link(name: str, target: str, kind: bytes) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = kind, target
    return info, None


def test_a_tarball_reads_like_its_directory(tmp_path, capsys):
    kaggle_tree(tmp_path / "src")
    tarball = tmp_path / "kaggle.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(tmp_path / "src" / "kaggle-t4x2", arcname="kaggle-t4x2")
    _, from_tar = run(tarball, capsys=capsys)
    _, from_dir = run(tmp_path / "src", capsys=capsys)
    assert from_tar.splitlines()[1:] == from_dir.splitlines()[1:]  # only the label differs


@pytest.mark.parametrize("filters", [True, False], ids=["data filter", "no filter (old Python)"])
@pytest.mark.parametrize("member", [
    lambda: _file("../escape-triage-test.txt"),
    lambda: _file("/tmp/escape-triage-test-abs.txt"),
    lambda: _link("link", "/etc/hosts", tarfile.SYMTYPE),
    lambda: _link("hard", "/etc/hosts", tarfile.LNKTYPE),
], ids=["dotdot", "absolute", "symlink", "hardlink"])
def test_unsafe_tarballs_are_refused(tmp_path, monkeypatch, capsys, filters, member):
    if not filters:
        monkeypatch.delattr(tarfile, "data_filter", raising=False)
    evil = _tarball(tmp_path / "evil.tgz", [_file("job/steps.jsonl", b""), member()])
    escaped = [Path(tempfile.gettempdir()) / "escape-triage-test.txt",
               Path("/tmp/escape-triage-test-abs.txt")]
    code = triage.main([str(evil)])
    assert not any(p.exists() for p in escaped)
    if filters and member()[0].name.startswith("/"):
        return  # the data filter makes an absolute name relative: extracted inside, harmless
    assert code == 2 and "can't read evil.tgz" in capsys.readouterr().err


def test_not_a_results_directory(tmp_path, capsys):
    assert triage.main([str(tmp_path)]) == 1
    assert "no steps.jsonl" in capsys.readouterr().err
    assert triage.main([str(tmp_path / "missing")]) == 2
    with pytest.raises(SystemExit):
        triage.main([str(tmp_path), "--max-lines", "3"])


def test_triage_needs_no_torch(tmp_path):
    # On a laptop or a login node: plain python3, no torch or gpubridge installed.
    kaggle_tree(tmp_path)
    code = ("import sys; sys.modules['torch'] = None; sys.modules['gpubridge'] = None; "
            f"sys.path.insert(0, {str(REPO / 'scripts' / 'gpu')!r}); import triage; "
            f"sys.exit(triage.main([{str(tmp_path)!r}]))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "verdict:" in done.stdout, done.stderr
