"""Every GPU job script, run end to end on CPU against a fake cluster.

tests/fake_cluster has stand-ins for sbatch, srun, scontrol, Lmod's module and
uv. scripts/gpu/explorer/submit.sh submits through the fake sbatch, which runs
the job script right here with made-up SLURM_* variables; the fake srun starts
one process per pretend node. gpubridge runs in simulation mode
(GPUBRIDGE_VENDOR), so every step must pass except the few that need a real
GPU, which each test names. A script bug (a wrong path, flag or argument)
fails here instead of in the first GPU allocation.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from test_gpu_kit import KIT, REPO, _free_port, clean_env, steps

FAKE_CLUSTER = REPO / "tests" / "fake_cluster"
EXPLORER = KIT / "explorer"
#: A short run for the scaling job, instead of its 2048-wide model.
SMALL_TRAINING = "--steps 3 --warmup-steps 1 --batch-size 8 --dim 32 --layers 2"

#: Every script in scripts/gpu, and the test that runs it end to end on CPU.
END_TO_END = {
    "common.sh": "sourced by every shell script below",
    "gpukit.py": "imported by check, bench, probe and device_map",
    "check.py": "every job below, and test_gpu_kit.py",
    "bench_all_reduce.py": "test_step_03, 04, 05, 07, the node pair",
    "summarize.py": "every job below",
    "probe.py": "test_step_01 and 07 (fails on CPU: no GPU)",
    "device_map.py": "test_step_03 (fails on CPU: no GPU)",
    "preflight.py": "test_step_07, the node pair, test_gpu_kit.py",
    "setup_env.sh": "test_setup_job_creates_the_cuda_venv",
    "explorer/submit.sh": "every test_step_*, and test_gpu_kit.py for 00",
    "explorer/job_env.sh": "sourced by every Explorer job",
    "explorer/00_partitions.sh": "test_gpu_kit.py: test_step_00_*",
    "explorer/setup.sbatch": "test_setup_job_*",
    "explorer/01_probe.sbatch": "test_step_01_probe",
    "explorer/02_split_shared_gpu.sbatch": "test_step_02_split_on_one_gpu",
    "explorer/03_single_island.sbatch": "test_step_03_single_island",
    "explorer/04_split_4gpu.sbatch": "test_step_04_split_4gpu",
    "explorer/05_multinode.sbatch": "test_step_05_two_nodes",
    "explorer/06_scaling.sbatch": "test_step_06_scaling",
    "explorer/07_mixed_hetjob.sbatch": "test_step_07_mixed_hetjob",
    "explorer/amd_node.sbatch": "test_amd_node_job",
    "mixed/node.sh": "test_mixed_node_script_on_two_pretend_machines",
    "amd/run_all.sh": "test_gpu_kit.py: test_amd_script_keeps_going_*",
}


def cluster_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return clean_env(**{
        "PATH": f"{FAKE_CLUSTER}{os.pathsep}{os.environ['PATH']}",
        "GPUBRIDGE_REPO": str(REPO),
        "VENV": sys.prefix,
        "VENV_ROCM": sys.prefix,
        "WORK_DIR": str(tmp_path / "work"),
        "RESULTS_ROOT": str(tmp_path / "results"),
        "FAKE_SLURM_LOG": str(tmp_path / "slurm.log"),
        "GPUBRIDGE_VENDOR": "nvidia",
        "BENCH_MAX_BYTES": "16K",
        "KIT_TIMEOUT": "120",
        "RDZV_PORT": str(_free_port()),
        "PREFLIGHT_PORT": str(_free_port()),
        **extra,
    })


def submit(tmp_path: Path, step: str, **extra: str) -> int:
    """Run submit.sh STEP through the fake sbatch; return the job's exit code."""
    result = subprocess.run(["bash", str(EXPLORER / "submit.sh"), step],
                            env=cluster_env(tmp_path, **extra), capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, result.stderr
    return job_exit(tmp_path)


def sbatch(tmp_path: Path, *args: str, **extra: str) -> int:
    """The fake sbatch itself, for jobs submit.sh won't submit yet (07, setup-rocm)."""
    result = subprocess.run([str(FAKE_CLUSTER / "sbatch"),
                             f"--output={tmp_path}/work/logs/%x-%j.out", *args],
                            env=cluster_env(tmp_path, **extra), capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, result.stderr
    return job_exit(tmp_path)


def job_exit(tmp_path: Path) -> int:
    lines = (tmp_path / "slurm.log").read_text().splitlines()
    codes = [int(line.split()[-1]) for line in lines if line.startswith("job ")]
    assert len(codes) == 1, lines
    return codes[0]


def job_output(tmp_path: Path) -> str:
    return "\n".join(p.read_text()[-4000:] for p in (tmp_path / "work" / "logs").glob("*.out"))


def failed(results: Path) -> set[str]:
    return {name for name, code in steps(results).items() if code}


def summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def make_venv(path: Path) -> Path:
    """A venv that sees this interpreter's packages (torch, gpubridge) without installing."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True)
    site = next(path.glob("lib/python*/site-packages"))
    parents = [p for p in sys.path if p.endswith("site-packages")]
    (site / "_parent.pth").write_text(
        "".join(f"import site; site.addsitedir({p!r})\n" for p in parents))
    return path


def test_every_gpu_script_runs_end_to_end_somewhere():
    scripts = {p.relative_to(KIT).as_posix() for p in KIT.rglob("*")
               if p.suffix in {".sh", ".sbatch", ".py"} and "__pycache__" not in p.parts}
    assert scripts == set(END_TO_END)


def test_every_job_script_sets_a_walltime():
    for script in EXPLORER.glob("*.sbatch"):
        assert "\n#SBATCH --time=" in script.read_text(), script.name


# ---- setup ---------------------------------------------------------------------

def test_setup_job_creates_the_cuda_venv(tmp_path):
    venv = make_venv(tmp_path / "work" / "venv")
    code = submit(tmp_path, "setup", VENV=str(venv), HOME=str(tmp_path / "home"))
    assert code == 0, job_output(tmp_path)
    log = (tmp_path / "slurm.log").read_text()
    assert "--partition=short" in log and "--time=01:00:00" in log
    assert "module load explorer" in log  # the proxy, before installing
    assert ("--index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 numpy"
            in log)
    assert f"uv pip install --python {venv}/bin/python -e {REPO}[test]" in log
    assert "environment ready" in job_output(tmp_path)


def test_setup_job_for_rocm_stops_with_a_clear_message_without_an_amd_gpu(tmp_path):
    venv = make_venv(tmp_path / "work" / "venv-rocm")
    code = sbatch(tmp_path, "--job-name=gpubridge-setup-rocm", str(EXPLORER / "setup.sbatch"),
                  TORCH_FLAVOR="rocm", VENV_ROCM=str(venv), HOME=str(tmp_path / "home"))
    assert code == 1
    assert "torch==2.14.1+rocm7.2" in (tmp_path / "slurm.log").read_text()
    assert "the ROCm build can't run this node's GPU" in job_output(tmp_path)


# ---- steps 01 to 07 ------------------------------------------------------------

def test_step_01_probe(tmp_path):
    code = submit(tmp_path, "01")
    out = tmp_path / "results" / "01_probe"
    assert failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert code == 1
    assert "--time=00:10:00" in (tmp_path / "slurm.log").read_text()


def test_step_02_split_on_one_gpu(tmp_path):
    code = submit(tmp_path, "02")
    out = tmp_path / "results" / "02_split_shared_gpu"
    assert code == 0 and failed(out) == set(), job_output(tmp_path)
    assert summary(out / "check")["split_test"] == "half"


def test_step_03_single_island(tmp_path):
    code = submit(tmp_path, "03")
    out = tmp_path / "results" / "03_single_island"
    assert failed(out) == {"device_map"}, job_output(tmp_path)  # no GPU here
    assert code == 1
    assert summary(out / "check")["ok"]
    assert (out / "bench" / "bench.csv").exists()


def test_step_04_split_4gpu(tmp_path):
    code = submit(tmp_path, "04")
    out = tmp_path / "results" / "04_split_4gpu"
    assert code == 0 and failed(out) == set(), job_output(tmp_path)
    assert (out / "thresholds.json").exists() and (out / "train_ddp.json").exists()


def test_step_05_two_nodes(tmp_path):
    code = submit(tmp_path, "05")
    out = tmp_path / "results" / "05_multinode"
    assert code == 0 and failed(out) == set(), job_output(tmp_path)
    assert summary(out / "check_one_island")["ok"]
    assert summary(out / "check_split_by_node")["split_test"] == "node"
    log = (tmp_path / "slurm.log").read_text()
    assert "--nodes=2" in log and "srun --nodes=2 --ntasks-per-node=1" in log
    assert (out / "thresholds.json").exists()


def test_step_06_scaling(tmp_path):
    code = submit(tmp_path, "06", SCALING_TRAIN_ARGS=SMALL_TRAINING)
    out = tmp_path / "results" / "06_scaling"
    assert code == 0 and failed(out) == set(), job_output(tmp_path)
    assert {f"train-n{n}.json" for n in range(1, 5)} <= {p.name for p in out.iterdir()}


def test_step_07_mixed_hetjob(tmp_path):
    # submit.sh won't submit 07 until the AMD nodes are confirmed, so the
    # heterogeneous job goes to the fake sbatch with the options submit.sh uses.
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd",
                  SCALING_TRAIN_ARGS=SMALL_TRAINING)
    out = tmp_path / "results" / "07_mixed_hetjob"
    assert failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert code == 1
    check = summary(out / "check")
    assert check["ok"] and check["split_test"] is None
    assert (out / "train_ddp.json").exists() and (out / "thresholds.json").exists()


def test_amd_node_job(tmp_path):
    # Rung 3 on Explorer: amd/run_all.sh on one AMD node, without setup or tarball.
    code = sbatch(tmp_path, "--job-name=gpubridge-amd_node", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "amd_node.sbatch"), GPUBRIDGE_VENDOR="amd", GPU_COUNT="1")
    out = tmp_path / "results" / "amd_node"
    assert failed(out) == {"probe", "device_map", "rccl_version"}, job_output(tmp_path)
    assert code == 1
    assert {"check_1gpu", "gpu_tests", "check_split_shared_gpu", "summarize"} <= set(steps(out))
    assert not (tmp_path / "results" / "amd_node.tgz").exists()


def loopback() -> str:
    names = {name for _, name in socket.if_nameindex()}
    return next(name for name in ("lo", "lo0") if name in names)


def test_mixed_node_script_on_two_pretend_machines(tmp_path):
    common = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(_free_port()),
              "PREFLIGHT_PORT": str(_free_port()), "IFNAME": loopback(),
              "LOCAL_ADDR": "127.0.0.1", "NPROC": "1", "SKIP_SETUP": "1",
              "TORCH_FLAVOR": "system", "RUN_ID": f"test-{os.getpid()}",
              "BENCH_MAX_BYTES": "16K", "RESULTS_ROOT": str(tmp_path / "results"),
              "VENV": sys.prefix, "KIT_TIMEOUT": "120"}
    procs = {
        role: subprocess.Popen(["bash", str(KIT / "mixed" / "node.sh")],
                               env=clean_env(ROLE=role, TARGET=role, GPUBRIDGE_VENDOR=vendor,
                                             **common),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for role, vendor in (("master", "nvidia"), ("worker", "amd"))
    }
    output = {role: proc.communicate(timeout=600)[0] for role, proc in procs.items()}
    for role, proc in procs.items():
        assert proc.returncode == 0, output[role][-3000:]
        out = tmp_path / "results" / role
        assert failed(out) == set()
        assert (out / f"preflight-{role}.json").exists()
        assert (tmp_path / "results" / f"{role}.tgz").exists()


# ---- limits and NCCL logs --------------------------------------------------------

def test_a_failed_step_keeps_the_end_of_its_nccl_logs(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(f"""set -euo pipefail
source {KIT}/common.sh
RESULTS={tmp_path}/results
# What NCCL writes at NCCL_DEBUG=INFO, to the file NCCL_DEBUG_FILE names; then exit $1.
fake_nccl() {{
    local file=${{NCCL_DEBUG_FILE//%h/node0}}
    file=${{file//%p/123}}
    {{ echo "NCCL INFO NCCL version 2.27.5"; seq 1 50 | sed 's/^/NCCL INFO line /'; }} >"$file"
    return "$1"
}}
step ok fake_nccl 0
step broken fake_nccl 3
step quiet false
step slow limited sleep 3
finish || true
""")
    result = subprocess.run(["bash", str(script)], env=clean_env(STEP_TIMEOUT="1"),
                            capture_output=True, text=True, timeout=120)
    results = tmp_path / "results"
    records = {r["step"]: r for r in map(json.loads,
                                         (results / "steps.jsonl").read_text().splitlines())}
    assert {name: r["exit_code"] for name, r in records.items() if name != "slow"} == {
        "ok": 0, "broken": 3, "quiet": 1}, result.stderr
    broken = (results / "broken.log").read_text()
    assert "==> NCCL/RCCL log" in broken and "NCCL INFO line 50" in broken
    assert "NCCL INFO line 10\n" not in broken  # only the last 40 lines
    assert "NCCL/RCCL log" not in (results / "ok.log").read_text()
    assert (results / "nccl" / "ok" / "node0.123.log").exists()  # kept for reading later
    assert "no NCCL/RCCL log" in (results / "quiet.log").read_text()
    assert not any(r["timed_out"] for name, r in records.items() if name != "slow")
    if shutil.which("timeout"):  # coreutils; not on macOS
        assert records["slow"]["exit_code"] == 124 and records["slow"]["timed_out"]
        assert "ran out of time" in (results / "slow.log").read_text()
    else:
        assert records["slow"]["exit_code"] == 0


def test_kit_tools_default_to_the_kit_timeout(monkeypatch):
    # common.sh exports KIT_TIMEOUT; each tool hands it to gpubridge.init.
    import gpukit  # scripts/gpu is on sys.path (test_gpu_kit)

    monkeypatch.setenv("KIT_TIMEOUT", "42")
    assert gpukit.default_timeout(600) == 42
    monkeypatch.delenv("KIT_TIMEOUT")
    assert gpukit.default_timeout(600) == 600
    out = subprocess.run(["bash", "-c", f"source {KIT}/common.sh && echo $KIT_TIMEOUT "
                          "$TORCH_NCCL_ASYNC_ERROR_HANDLING $STEP_TIMEOUT $NCCL_DEBUG"],
                         env=clean_env(), capture_output=True, text=True, check=True).stdout
    assert out.split() == ["300", "3", "1200", "INFO"]
