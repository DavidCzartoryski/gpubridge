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
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

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
    "bench_compression.py": "test_step_04, 07, the node pair, test_amd_node_job",
    "summarize.py": "every job below",
    "probe.py": "test_step_01 and 07 (fails on CPU: no GPU)",
    "device_map.py": "test_step_03 (fails on CPU: no GPU)",
    "preflight.py": "test_step_05, 07, the node pair, test_gpu_kit.py",
    "node_ifname.sh": "test_step_05, test_each_node_uses_its_own_preflight_choice",
    "validation_tables.py": "test_step_05, test_gpu_kit.py",
    "triage.py": "test_triage.py: Explorer, Kaggle and CPU-simulation results trees, and tarballs",
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
    "kaggle/kaggle_gpubridge.py": "test_kaggle_runner.py: the plan, hard timeouts, feature decisions, "
                                  "the feature check and packaging (the GPU steps need GPUs)",
    "kaggle/make_notebook.py": "test_kaggle_runner.py: the notebook matches the script",
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
        "PREFLIGHT_SECONDS": "0.3",
        **extra,
    })


def submit(tmp_path: Path, step: str, **extra: str) -> int:
    """Run submit.sh STEP through the fake sbatch; return the job's exit code."""
    return submit_output(tmp_path, step, **extra)[0]


def submit_output(tmp_path: Path, step: str, **extra: str) -> tuple[int, str]:
    """submit(), and what submit.sh itself printed on stderr."""
    result = subprocess.run(["bash", str(EXPLORER / "submit.sh"), step],
                            env=cluster_env(tmp_path, **extra), capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, result.stderr
    return job_exit(tmp_path), result.stderr


def job_id(tmp_path: Path) -> str:
    """The fake sbatch's job id (one job)."""
    return re.findall(r"^job (\d+) exit", (tmp_path / "slurm.log").read_text(), re.M)[0]


def events(results: Path) -> list[dict]:
    """job.jsonl: the job's start, errors and end."""
    return [json.loads(line) for line in (results / "job.jsonl").read_text().splitlines()]


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
    code, stderr = submit_output(tmp_path, "setup", VENV=str(venv), HOME=str(tmp_path / "home"))
    assert code == 0, job_output(tmp_path)
    # The log's path, with the job's id in it (sbatch --parsable).
    log_path = tmp_path / "work" / "logs" / f"gpubridge-setup-cuda-{job_id(tmp_path)}.out"
    assert f"setup log: {log_path}." in stderr and log_path.exists()
    assert "<job id>" not in stderr
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
    code, stderr = submit_output(tmp_path, "01")
    out = tmp_path / "results" / "01_probe"
    assert failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert code == 1
    assert "--time=00:10:00" in (tmp_path / "slurm.log").read_text()
    assert f"job {job_id(tmp_path)}, log {tmp_path}/work/logs/gpubridge-01_probe-" in stderr
    # The job's own log says why: the step's last lines, with probe's FAIL lines, after
    # its FAILED line.
    log = job_output(tmp_path)
    after = log[log.index("step probe FAILED (exit 1)"):]
    assert "the last 20 lines of probe.log, before any torchrun failure report:" in after
    assert "    | FAIL gpus_visible: device_count 0: no GPU visible to PyTorch" in after
    # Each launch's limit stays inside the job's: 10 minutes less 2.
    record = json.loads((out / "probe.json").read_text())
    assert record["environment"]["env"]["STEP_TIMEOUT"] == "480"
    assert [(e["event"], e.get("failed", e.get("code"))) for e in events(out)] == [
        ("started", None), ("finished", 1), ("exit", 1)]


def test_a_job_that_stops_in_its_setup_is_one_triage_lists(tmp_path):
    # No venv: job_setup stops the job before its first step, so steps.jsonl stays empty.
    code = submit(tmp_path, "01", VENV=str(tmp_path / "no-venv"))
    out = tmp_path / "results" / "01_probe"
    assert code == 1 and (out / "steps.jsonl").read_text() == ""
    assert [e["event"] for e in events(out)] == ["started", "error", "exit"]
    assert "no venv at" in events(out)[1]["message"]
    triage = subprocess.run([sys.executable, str(KIT / "triage.py"), str(tmp_path / "results")],
                            capture_output=True, text=True, timeout=60)
    assert triage.returncode == 1
    assert f"UNFINISHED job {job_id(tmp_path)}, started" in triage.stdout
    assert "it stopped with an error: no venv at" in triage.stdout
    assert triage.stdout.splitlines()[-1] == "verdict: 1 job(s) unfinished"


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
    # The pinned-staging A/B: same check and benchmark with GPUBRIDGE_PINNED_STAGING=1.
    assert summary(out / "check_pinned")["ok"]
    for path in (out / "check_pinned").glob("rank*.json"):
        assert json.loads(path.read_text())["run"]["pinned_staging"] is True
    assert "bridge staging pinned" in (out / "bench_pinned" / "summary.md").read_text()
    assert "bridge staging pageable" in (out / "bench_split_by_node" / "summary.md").read_text()
    log = (tmp_path / "slurm.log").read_text()
    assert "--nodes=2" in log and "srun --nodes=2 --ntasks-per-node=1" in log
    assert (out / "thresholds.json").exists()
    # The preflight ran first on both pretend nodes; Gloo then used its choice.
    records = [json.loads((out / "preflight" / f"preflight-{role}.json").read_text())
               for role in ("master", "worker")]
    assert all(record["ok"] for record in records)
    chosen = records[0]["chosen_ifname"]
    assert chosen and records[1]["chosen_ifname"] == chosen  # one machine: the same name
    logs = "".join(p.read_text() for p in (tmp_path / "work" / "logs").glob("*.out"))
    assert f"GLOO_SOCKET_IFNAME={chosen}, as the preflight chose" in logs
    # Every torchrun goes through node_ifname.sh, for nodes whose names differ (below).
    torchruns = [line for line in log.splitlines()
                 if line.startswith("srun ") and "torchrun" in line]
    assert torchruns and all("node_ifname.sh torchrun" in line for line in torchruns)
    # The validation tables from this job: all simulation, so no item counts.
    tables = subprocess.run([sys.executable, str(KIT / "validation_tables.py"), str(out)],
                            capture_output=True, text=True, timeout=120)
    assert tables.returncode == 0, tables.stderr
    data = json.loads((out / "validation_tables.json").read_text())
    assert [run["simulated"] for run in data["runs"]] == [True]
    verdicts = {row["item"]: row["verdict"] for row in data["items"]}
    assert verdicts[8] == "simulation only" and verdicts[1] == "not run"
    assert data["network"][0]["master"].endswith(chosen)
    assert {row["policy"] for row in data["predicted"]} >= {"sharded-bridge"}


def test_a_two_node_rendezvous_never_looks_up_the_host_name(tmp_path):
    # Steps 05 and 07 start one torchrun per pretend node with a c10d rendezvous
    # on 127.0.0.1. Unless told its address, torchrun uses socket.getfqdn() for
    # the node hosting the rendezvous's store; on macOS that can name an address
    # that doesn't resolve (the reverse name of ::1), and the other node waits
    # until it times out. Here getfqdn() names one that never resolves, on any
    # machine, and the rendezvous must still complete through the fake srun.
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import socket\nsocket.getfqdn = lambda name='': 'unresolvable.invalid'\n")
    worker = tmp_path / "worker.py"
    # One write per line: the two workers share a pipe and may run unbuffered, where
    # print() writes each piece separately and two lines can interleave.
    worker.write_text("import os\nos.write(1, f\"{os.environ['RANK']} "
                      "{os.environ['MASTER_ADDR']}\\n\".encode())\n")
    env = cluster_env(tmp_path, PYTHONPATH=str(site))
    result = subprocess.run(
        ["srun", "--nodes=2", "--ntasks-per-node=1", "--kill-on-bad-exit=1", sys.executable,
         "-m", "torch.distributed.run", "--nnodes=2", "--nproc-per-node=1",
         "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{_free_port()}",
         "--rdzv-id=fqdn", "--rdzv-conf=join_timeout=60", str(worker)],
        env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-4000:]
    assert sorted(result.stdout.split("\n")[:-1]) == ["0 127.0.0.1", "1 127.0.0.1"]


@pytest.mark.parametrize(("names", "extra", "seen"), [
    (("ib0", "eth1"), {}, ["node 0: ib0", "node 1: eth1"]),  # e.g. two partitions
    (("ib0", "ib0"), {}, ["node 0: ib0", "node 1: ib0"]),
    (("ib0", "eth1"), {"GLOO_SOCKET_IFNAME": "bond0"}, ["node 0: bond0", "node 1: bond0"]),
    (("ib0", "eth1"), {"USE_PREFLIGHT_IFNAME": "0"}, ["node 0: unset", "node 1: unset"]),
])
def test_each_node_uses_its_own_preflight_choice(names, extra, seen, tmp_path):
    # What step 05 does after its preflight, with records from two nodes that
    # chose different names: each node's job steps, started by srun as
    # multinode() starts torchrun, see their own node's choice.
    records = tmp_path / "preflight"
    records.mkdir()
    for role, name in zip(("master", "worker"), names):
        (records / f"preflight-{role}.json").write_text(
            json.dumps({"ok": True, "chosen_ifname": name}))
    script = tmp_path / "job.sh"
    script.write_text(f"""set -euo pipefail
source {KIT}/common.sh
use_preflight_ifname {records}
srun --nodes=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash {KIT}/node_ifname.sh \\
    bash -c 'echo "node $SLURM_NODEID: ${{GLOO_SOCKET_IFNAME:-unset}}"'
""")
    env = cluster_env(tmp_path, **extra)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"  # python, as in a job
    if "GLOO_SOCKET_IFNAME" not in extra:
        env.pop("GLOO_SOCKET_IFNAME", None)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert sorted(result.stdout.splitlines()) == seen
    if names == ("ib0", "eth1") and not extra:
        assert "ib0 on node 0 (the master), eth1 on node 1" in result.stderr


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
    # Every job step after the preflight runs with each side's chosen interface.
    chosen = json.loads((out / "preflight" / "preflight-master.json").read_text())["chosen_ifname"]
    later = (tmp_path / "slurm.log").read_text().split("preflight.py", 1)[1]
    assert f"env GLOO_SOCKET_IFNAME={chosen}" in later


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


def test_a_negative_step_passes_only_on_the_expected_error(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(f"""set -euo pipefail
source {KIT}/common.sh
RESULTS={tmp_path}/results
ERROR="would share a GPU"
negative_step fired 60 "$ERROR" \\
    echo "rank 0: init() failed: ranks of one island would share a GPU"
negative_step quiet 60 "$ERROR" echo "rank 0: init() failed: no GPU"
negative_step other_failure 60 "$ERROR" bash -c 'echo "would share a GPU"; exit 1'
negative_step hangs 1 "$ERROR" limited sleep 3
step plain true
echo "STEP_TIMEOUT=$STEP_TIMEOUT after"
finish || true
""")
    result = subprocess.run(["bash", str(script)], env=clean_env(), capture_output=True,
                            text=True, timeout=120)
    results = tmp_path / "results"
    records = {r["step"]: r for r in map(json.loads,
                                         (results / "steps.jsonl").read_text().splitlines())}
    assert records["fired"]["exit_code"] == 0, result.stderr
    assert records["fired"]["result"] == "expected failure confirmed"
    assert "step fired: expected failure confirmed" in (results / "fired.log").read_text()
    # Exit 0 without the error, or the error but a failed command, or a hang: no pass.
    for name in ("quiet", "other_failure", "hangs"):
        assert records[name]["exit_code"] != 0, name
        assert records[name]["result"] == "check did not fire", name
    assert "result" not in records["plain"]
    if shutil.which("timeout"):  # coreutils; not on macOS
        assert records["hangs"]["exit_code"] == 124 and records["hangs"]["timed_out"]
        assert "STEP_TIMEOUT=1s" in (results / "hangs.log").read_text()
    assert "STEP_TIMEOUT=1200 after" in result.stdout  # only the step had the short limit


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
