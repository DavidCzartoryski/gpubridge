"""The Kaggle runner (scripts/gpu/kaggle), on CPU: its plan, hard timeouts, the T4
feature decisions, packaging, and that its results read like any kit job's.
Nothing here needs a GPU or Kaggle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "scripts" / "gpu" / "kaggle"
sys.path.insert(0, str(KAGGLE))
sys.path.insert(0, str(REPO / "scripts" / "gpu"))

import kaggle_gpubridge as kg  # noqa: E402
import make_notebook  # noqa: E402
import validation_tables  # noqa: E402


def runner(tmp_path: Path, *argv: str) -> kg.Runner:
    args = kg.parse(["--work", str(tmp_path / "work"), "--scratch", str(tmp_path / "scratch"),
                     "--target", "t", *argv])
    return kg.Runner(args)


def test_the_dry_run_plans_every_step_in_order_with_its_limit(tmp_path, capsys):
    assert kg.main(["--dry-run", "--work", str(tmp_path), "--target", "t"]) == 0
    out = capsys.readouterr().out
    names = [line.split("step ", 1)[1].split(" ")[0].rstrip(":") for line in out.splitlines()
             if "] step " in line]
    assert names == ["clone", "setup", "setup_triton", "versions", "features", "probe", "probe_gpu0",
                     "probe_gpu1", "gpu_tests", "check_split", "bench", "bench_compression_kernels",
                     "verify_fsdp2_backend", "tables", "package"]
    assert "(limit 1800 s)" in out and "GPUBRIDGE_SPLIT_TEST=half" in out
    assert "--ops native,gpubridge,flat-gloo" in out and "--verify" in out
    assert not (tmp_path / "results").exists()  # a dry run writes nothing


def test_an_unknown_step_is_refused():
    with pytest.raises(SystemExit):
        kg.parse(["--only", "setup,nope"])


def test_a_step_past_its_limit_has_its_whole_process_group_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "KILL_GRACE", 5)
    r = runner(tmp_path)
    pid_file = tmp_path / "child.pid"
    start = time.monotonic()
    code = r.run("slow", ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"], timeout=1)
    assert code == 124 and time.monotonic() - start < 20
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):  # the background child went with its group
        os.kill(child, 0)
    (line,) = (r.results / "steps.jsonl").read_text().splitlines()
    entry = json.loads(line)
    assert entry["step"] == "slow" and entry["exit_code"] == 124 and entry["timed_out"] is True
    assert "ran out of time" in (r.results / "slow.log").read_text()
    assert r.failed == ["slow"]


def test_a_step_that_fails_is_recorded_and_the_next_one_still_runs(tmp_path):
    r = runner(tmp_path)
    assert r.run("bad", ["bash", "-c", "echo boom; exit 3"]) == 3
    assert r.run("good", ["bash", "-c", "echo fine"]) == 0
    steps = [json.loads(x) for x in (r.results / "steps.jsonl").read_text().splitlines()]
    assert [(s["step"], s["exit_code"], s["timed_out"]) for s in steps] == [
        ("bad", 3, False), ("good", 0, False)]
    assert "boom" in (r.results / "bad.log").read_text()


def write_features(r: kg.Runner, **ok: bool) -> None:
    r.results.mkdir(parents=True, exist_ok=True)
    found = {"cuda_events": True, "bfloat16": True, "triton_float32": True, "triton_bfloat16": True, **ok}
    gpus = [{"index": i, "name": "Tesla T4", "capability": "7.5",
             "checks": {k: ({"ok": True} if v else {"ok": False, "error": f"{k} failed on sm_75"})
                        for k, v in found.items()}} for i in range(2)]
    (r.results / "features.json").write_text(json.dumps({"gpus": gpus, "ok_features": found}))


def test_without_triton_the_kernel_steps_run_on_pytorch_under_other_names(tmp_path, capsys):
    r = runner(tmp_path, "--dry-run")
    write_features(r, triton_float32=False, triton_bfloat16=False)
    r.step_gpu_tests()
    r.step_bench_compression_kernels()
    out = capsys.readouterr().out
    assert "step gpu_tests_without_triton" in out and "test_compression_kernels.py" not in out.split(
        "step gpu_tests_without_triton")[1].split("\n")[0]
    assert "--durations=15" in out.split("step gpu_tests_without_triton")[1].split("\n")[0]
    assert "step bench_compression_torch_kernels" in out
    assert out.count("GPUBRIDGE_COMPRESSION_KERNELS=torch") == 2
    assert "compute capability 7.5: triton_float32 failed on sm_75" in out
    assert r.skipped == ["gpu_tests", "bench_compression_kernels"]


def test_without_bfloat16_the_compression_bench_takes_float32_only(tmp_path, capsys):
    r = runner(tmp_path, "--dry-run")
    write_features(r, triton_bfloat16=False)
    r.step_bench_compression_kernels()
    out = capsys.readouterr().out
    assert "step bench_compression_kernels" in out and "--dtypes float32" in out
    assert "float32 inputs only: bfloat16 failed here" in out


def test_without_cuda_events_or_a_second_gpu_the_steps_that_need_them_are_skipped(tmp_path, capsys):
    r = runner(tmp_path, "--dry-run")
    write_features(r, cuda_events=False)
    r.step_bench()
    r1 = runner(tmp_path, "--dry-run", "--gpus", "1")
    for step in (r1.step_check_split, r1.step_bench, r1.step_verify_fsdp2_backend):
        step()
    out = capsys.readouterr().out
    assert "step bench SKIPPED: CUDA events failed here" in out
    assert r1.skipped == ["check_split", "bench", "verify_fsdp2_backend"]
    assert "needs 2 GPUs; 1 visible" in out


def test_skips_go_to_their_own_file_and_never_count_as_a_pass(tmp_path):
    r = runner(tmp_path)
    write_features(r, triton_float32=False)
    r.run("gpu_tests_without_triton", ["true"])
    r.skip("bench", "needs 2 GPUs; 1 visible")
    r.run("bench_compression_torch_kernels", ["true"])
    assert [json.loads(x)["step"] for x in (r.results / "steps.jsonl").read_text().splitlines()] == [
        "gpu_tests_without_triton", "bench_compression_torch_kernels"]
    assert json.loads((r.results / "skipped.jsonl").read_text())["reason"] == "needs 2 GPUs; 1 visible"
    assert validation_tables.main([str(r.results)]) == 0
    tables = json.loads((r.results / "validation_tables.json").read_text())
    items = {i["item"]: i for i in tables["items"]}
    # Neither the skipped bench nor the PyTorch-kernel runs stand in for items 11 and 28.
    assert items[11]["evidence"] == [] and items[11]["verdict"] == "not run"
    assert items[28]["evidence"] == [] and items[28]["verdict"] == "not run"


def test_package_packs_the_results_with_a_summary_from_the_files(tmp_path, capsys):
    r = runner(tmp_path)
    r.run("probe", ["true"])
    r.run("bench", ["false"])
    r.skip("check_split", "needs 2 GPUs; 1 visible")
    later = runner(tmp_path)  # the notebook runs each step in a new Runner
    tarball = later.step_package()
    assert tarball == tmp_path / "work" / "t.tgz"
    with tarfile.open(tarball) as tar:
        names = set(tar.getnames())
        summary = json.load(tar.extractfile("t/kaggle_summary.json"))
    assert {"t/steps.jsonl", "t/skipped.jsonl", "t/probe.log", "t/bench.log"} <= names
    assert summary == {"passed": ["probe"], "failed": ["bench"], "timed_out": [],
                       "skipped": ["check_split: needs 2 GPUs; 1 visible"]}
    assert "results: " in capsys.readouterr().out


def test_the_feature_check_runs_and_writes_a_kit_record_without_a_gpu(tmp_path):
    r = runner(tmp_path)
    r.py, r.kit = Path(sys.executable), REPO / "scripts" / "gpu"
    r.step_features()
    assert r.failed == [], (r.results / "features.log").read_text()
    record = json.loads((r.results / "features.json").read_text())
    assert record["ok"] is True and record["environment"] and record["gpus"] == []
    assert r.features() == {"cuda_events": True, "bfloat16": True, "triton_float32": True,
                            "triton_bfloat16": True}  # nothing checked: nothing skipped


@pytest.mark.docs
def test_the_notebook_holds_this_script_and_runs_every_step():
    assert make_notebook.main(["--check"]) == 0, "run python scripts/gpu/kaggle/make_notebook.py"
    nb = json.loads(make_notebook.NOTEBOOK.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code[0] == "%%writefile kaggle_gpubridge.py\n" + (KAGGLE / "kaggle_gpubridge.py").read_text()
    ran = [s for c in code if c.startswith("k.main(") for s in c.split('"--only", "')[1].split('"')[0].split(",")]
    assert ran == list(kg.STEPS)


def test_the_script_runs_as_a_program():
    out = subprocess.run([sys.executable, str(KAGGLE / "kaggle_gpubridge.py"), "--dry-run", "--only",
                          "versions"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "step versions" in out.stdout
