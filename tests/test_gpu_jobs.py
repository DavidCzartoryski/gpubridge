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
    "import_check.py": "every job below (job_setup, run_all.sh), and test_gpu_kit.py",
    "triton_check.py": "test_step_02, 04, 07, test_amd_node_job (before the kernel steps), "
                       "and test_gpu_kit.py",
    "check.py": "every job below, and test_gpu_kit.py",
    "bench_all_reduce.py": "test_step_03, 04, 05, 07, the node pair",
    "bench_compression.py": "test_step_04, 07, the node pair, test_amd_node_job",
    "bench_matrix.py": "test_step_08_*, test_net_check_job, test_gpu_kit.py",
    "train_bench.py": "test_step_07_*, test_train_bench_job, test_gpu_kit.py",
    "train_dp.py": "test_step_07_data_parallel, test_dp_baseline_job, test_gpu_kit.py",
    "llama.py": "imported by train_dp.py: the same tests",
    "dp_report.py": "test_step_07_data_parallel, test_gpu_kit.py",
    "train_ddp_report.py": "test_step_07_ddp_repeats, test_gpu_kit.py",
    "train_pipeline.py": "test_step_07_pipeline, test_gpu_kit.py",
    "net_report.sh": "test_step_08_*, test_net_check_job",
    "rdma_report.sh": "test_net_check_job, test_rdma_report_keeps_paths_out_and_reads_all_of_nm",
    "stream_mutations.py": "test_gpu_kit.py: its plan, each mutation, and its refusal without a "
                           "GPU (the mutated tests themselves need one)",
    "time_triton_cache.py": "test_gpu_kit.py: filesystems and its refusal without a GPU (the "
                            "timings need one)",
    "diag_fp8_residual.py": "test_compression_kernels.py: on CPU through Triton's interpreter, "
                            "and with a stand-in device that fuses the residual's multiply-add",
    "summarize.py": "every job below",
    "probe.py": "test_step_01 and 07 (fails on CPU: no GPU)",
    "device_map.py": "test_step_03 (fails on CPU: no GPU)",
    "preflight.py": "test_step_05, 07, the node pair, test_gpu_kit.py",
    "node_ifname.sh": "test_step_05, test_each_node_uses_its_own_preflight_choice",
    "validation_tables.py": "test_step_05, test_gpu_kit.py",
    "triage.py": "test_triage.py: Explorer, Kaggle and CPU-simulation results trees, and tarballs",
    "setup_env.sh": "test_setup_job_creates_the_cuda_venv",
    "bench_p2p.py": "test_gpu_kit.py: test_bench_p2p_*, test_step_02 and 07",
    "rocm_triton_overlay.sh": "test_gpu_kit.py: test_rocm_triton_overlay_*",
    "megatron_pp.py": "test_gpu_kit.py: its tokens, weights and step 07's dry run (Megatron-Core's "
                      "pipeline needs CUDA, so no run on CPU)",
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
    "explorer/07_mixed_hetjob.sbatch": "test_step_07_mixed_hetjob, test_step_07_training_only, "
                                       "test_step_07_ddp_repeats, "
                                       "test_step_07_multi, test_step_07_data_parallel, "
                                       "test_step_07_pipeline, "
                                       "test_step_07_pipeline_compares_splits, "
                                       "test_step_07_refuses_bad_pipeline_splits, "
                                       "test_step_07_trains_on_a_same_vendor_pair, "
                                       "test_step_07_multi_trains_data_parallel_and_pipelined, "
                                       "test_step_07_megatron_needs_megatron_path",
    "explorer/08_bench_matrix.sbatch": "test_step_08_*",
    "explorer/net_check.sbatch": "test_net_check_job",
    "explorer/train_bench.sbatch": "test_train_bench_job, test_train_multi_job",
    "explorer/dp_baseline.sbatch": "test_dp_baseline_job",
    "explorer/amd_node.sbatch": "test_amd_node_job",
    "mixed/node.sh": "test_mixed_node_script_on_two_pretend_machines, "
                     "test_mixed_node_script_leaves_out_what_is_set_empty, mixed/containers.sh",
    "mixed/containers.sh": "test_mixed_containers_dry_run (the containers need Docker: run it "
                           "by hand, see the runbook)",
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
        # train_bench.py's smallest model, briefly; one chunk size for step 08.
        "TRAIN_BENCH_MODELS": "tiny-gpt",
        "TRAIN_BENCH_ARGS": "--steps 2 --warmup-steps 1 --comm-trials 1",
        "PIPELINE_CHUNKS": "4K",
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
    # Triton's kernels: a node-local cache for the job, kept in WORK_DIR's store.
    assert "Triton kernel cache: " in log and f"from {tmp_path}/work/triton-cache" in log
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
    assert summary(out / "check_p2p")["ok"]


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
    assert summary(out / "check_p2p")["ok"]
    assert (out / "train_ddp.json").exists() and (out / "thresholds.json").exists()
    # Triton's check runs on both sides, once, before the first step that uses the kernels.
    order = list(steps(out))
    assert order.index("triton_check") == order.index("bench_compression_kernels") - 1, order
    log = "\n".join(p.read_text() for p in (tmp_path / "work" / "logs").glob("*.out"))
    check = [line for line in log.splitlines() if "step triton_check:" in line]
    assert len(check) == 1 and "step triton_check: both @VENV@/bin/python" in check[0]
    bench = json.loads((out / "train_bench_tiny-gpt.json").read_text())
    assert bench["world_size"] == 2 and [r["vendor"] for r in bench["ranks"]] == ["nvidia", "amd"]
    # Every job step after the preflight runs with each side's chosen interface.
    chosen = json.loads((out / "preflight" / "preflight-master.json").read_text())["chosen_ifname"]
    later = (tmp_path / "slurm.log").read_text().split("preflight.py", 1)[1]
    assert f"env GLOO_SOCKET_IFNAME={chosen}" in later


def test_step_07_training_only(tmp_path):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="training")
    out = tmp_path / "results" / "07_mixed_hetjob"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)
    assert list(steps(out)) == ["probe", "preflight", "train_ddp", "train_bench_tiny-gpt",
                                "train_fsdp2_backend", "verify_fsdp2_backend",
                                "train_hsdp_backend", "verify_hsdp_backend"]


def test_step_07_ddp_repeats(tmp_path):
    # STEPS_07=ddp: train_ddp alone, TRAIN_DDP_REPEATS times, each its own step and record.
    # Bridge compression and its kernels come from the job's environment (submit.sh's
    # sbatch --export=ALL passes it on) and reach every rank of every repeat.
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="ddp",
                  TRAIN_DDP_REPEATS="3", GPUBRIDGE_BRIDGE_COMPRESSION="fp8",
                  GPUBRIDGE_COMPRESSION_KERNELS="torch")
    out = tmp_path / "results" / "07_mixed_hetjob"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert list(steps(out)) == ["probe", "preflight", "train_ddp_1", "train_ddp_2",
                                "train_ddp_3"]
    assert not (out / "train_ddp.json").exists()
    for i in (1, 2, 3):
        record = json.loads((out / f"train_ddp_{i}.json").read_text())
        assert record["world_size"] == 2 and record["params_in_sync"] is True
        envs = [e["env"] for e in record["environments"]]
        assert [e["GPUBRIDGE_VENDOR"] for e in envs] == ["nvidia", "amd"]
        assert [e["GPUBRIDGE_BRIDGE_COMPRESSION"] for e in envs] == ["fp8", "fp8"]
        assert [e["GPUBRIDGE_COMPRESSION_KERNELS"] for e in envs] == ["torch", "torch"]
    # train_ddp_report.py: a row per repeat, with this job's id, then the total.
    report = subprocess.run([sys.executable, str(KIT / "train_ddp_report.py"),
                             str(tmp_path / "results")], capture_output=True, text=True,
                            timeout=60)
    assert report.returncode == 0, report.stderr
    rows = report.stdout.splitlines()[2:5]
    for i, row in enumerate(rows, start=1):
        assert row.startswith(f"| results/07_mixed_hetjob | {job_id(tmp_path)} | {i} | "), row
        assert row.endswith("| fp8 | torch | no | - | - | yes | - | in sync |"), row
    assert report.stdout.splitlines()[-1].endswith(": 0 of 3 diverged")
    events = [json.loads(line) for line in (out / "job.jsonl").read_text().splitlines()]
    assert {"event": "train_ddp_repeats", "planned": 3}.items() <= next(
        e for e in events if e["event"] == "train_ddp_repeats").items()



def test_step_07_bench_only_sweeps_bucket_sizes(tmp_path):
    # STEPS_07=bench: train_bench.py's steps alone, once per TRAIN_BENCH_BUCKETS size.
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="bench",
                  TRAIN_BENCH_BUCKETS="0.01 25")
    out = tmp_path / "results" / "07_mixed_hetjob"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert list(steps(out)) == ["probe", "preflight", "train_bench_tiny-gpt_bucket0.01",
                                "train_bench_tiny-gpt_bucket25"]
    small, large = (json.loads((out / f"train_bench_tiny-gpt_bucket{mb}.json").read_text())
                    for mb in ("0.01", "25"))
    assert (small["bucket_mb"], large["bucket_mb"]) == (0.01, 25.0)
    assert small["buckets"] > large["buckets"] >= 1
    for record in (small, large):
        assert record["world_size"] == 2 and record["params_identical"]
        assert [r["vendor"] for r in record["ranks"]] == ["nvidia", "amd"]

# ---- step 08, the network check and training throughput ------------------------------

def test_step_07_multi(tmp_path):
    # With multigpu access: an NVIDIA island of several GPUs and the AMD side, training only.
    code = submit(tmp_path, "07-multi", AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  GPU_TYPE_MULTI="v100-sxm2", GPUS_MIXED_NVIDIA_MULTI="2", STEPS_07="training",
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd")
    out = tmp_path / "results" / "07_mixed_hetjob"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    record = json.loads((out / "train_bench_tiny-gpt.json").read_text())
    assert record["world_size"] == 3 and record["params_identical"]
    log = (tmp_path / "slurm.log").read_text()
    assert "--partition=multigpu" in log and "--gres=gpu:v100-sxm2:2" in log
    assert "--gres=gpu:mi50:1" in log


#: A short data-parallel run of the tiny model, for the fake cluster.
SMALL_DP = {"DP_MODEL": "llama-tiny", "DP_STEPS": "4", "DP_GLOBAL_BATCH": "4", "DP_SEQ": "16",
            "DP_ARGS": "--check-every 2 --probe-steps 1 --comm-trials 1"}


def test_step_07_data_parallel(tmp_path):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="dp",
                  DP_RUNS="ddp ddp-fp8 fsdp2 ddp-speed ddp-bucket-5", **SMALL_DP)
    out = tmp_path / "results" / "07_mixed_hetjob_dp"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    # Triton's check before the first compressed run only, on the NVIDIA side (the AMD
    # side runs the PyTorch kernels: DP_KERNELS_ON_AMD=torch).
    assert list(steps(out)) == ["probe", "preflight", "dp_ddp", "triton_check", "dp_ddp-fp8",
                                "dp_fsdp2", "dp_ddp-speed", "dp_ddp-bucket-5"]
    log = "\n".join(p.read_text() for p in (tmp_path / "work" / "logs").glob("*.out"))
    check = [line for line in log.splitlines() if "step triton_check:" in line]
    assert check and "--het-group=0" in check[0] and "--het-group=1" not in check[0]
    runs = {p.stem: json.loads(p.read_text()) for p in out.glob("dp_*.json")}
    assert runs["dp_ddp-fp8"]["compression"] == "fp8" and runs["dp_ddp"]["compression"] == "none"
    assert [r["compression_kernels"] for r in runs["dp_ddp-fp8"]["ranks"]] == ["auto", "torch"]
    assert runs["dp_fsdp2"]["mode"] == "fsdp2" and runs["dp_ddp-bucket-5"]["bucket_mb"] == 5.0
    assert runs["dp_ddp-speed"]["shares_rule"] == "speed"
    assert {r["world_size"] for r in runs.values()} == {2}


def test_step_07_pipeline(tmp_path):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="pipeline",
                  PP_MICROBATCHES="2", PP_ARGS="--probe-trials 1", **SMALL_DP)
    out = tmp_path / "results" / "07_mixed_hetjob_pp"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert list(steps(out)) == ["probe", "preflight", "pp_gpipe", "pp_1f1b"]
    runs = {p.stem: json.loads(p.read_text()) for p in out.glob("pp_*.json")}
    assert runs["pp_gpipe"]["mode"] == "pipeline-gpipe" and runs["pp_1f1b"]["mode"] == "pipeline-1f1b"
    assert [r["vendor"] for r in runs["pp_gpipe"]["ranks"]] == ["nvidia", "amd"]
    assert runs["pp_gpipe"]["microbatches"] == 2 and len(runs["pp_gpipe"]["losses"]) == 4


def test_step_07_pipeline_compares_splits(tmp_path):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", STEPS_07="pipeline",
                  PP_RUNS="1f1b", PP_SPLITS="even 3,1 auto", PP_MICROBATCHES="2",
                  PP_ARGS="--probe-trials 1", **SMALL_DP)
    out = tmp_path / "results" / "07_mixed_hetjob_pp"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert list(steps(out)) == ["probe", "preflight", "pp_1f1b_even", "pp_1f1b_3-1",
                                "pp_1f1b_auto"]
    runs = {p.stem: json.loads(p.read_text()) for p in out.glob("pp_*.json")}
    assert runs["pp_1f1b_even"]["split"] == [2, 2] and runs["pp_1f1b_3-1"]["split"] == [3, 1]
    assert runs["pp_1f1b_auto"]["split_rule"] == "auto" and runs["pp_1f1b_auto"]["balance"]


@pytest.mark.parametrize("env, message", [
    ({"PP_SPLITS": "even 7:5"}, "PP_SPLITS: '7:5' is not a split"),
    ({"PP_SPLITS": "auto", "PP_ARGS": "--split 3,1"}, "in PP_SPLITS or in PP_ARGS, not both"),
])
def test_step_07_refuses_bad_pipeline_splits(tmp_path, env, message):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"), STEPS_07="pipeline", **env)
    assert code != 0 and message in job_output(tmp_path)
@pytest.mark.parametrize("mode, out_dir, record", [("dp", "07_mixed_hetjob_dp", "dp_ddp.json"),
                                                   ("pipeline", "07_mixed_hetjob_pp",
                                                    "pp_1f1b.json")])
def test_step_07_multi_trains_data_parallel_and_pipelined(tmp_path, mode, out_dir, record):
    # With multigpu access: train_dp.py and train_pipeline.py with an NVIDIA island of two
    # GPUs and the AMD side, three ranks (the pipeline in three stages).
    code = submit(tmp_path, "07-multi", AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  GPU_TYPE_MULTI="v100-sxm2", GPUS_MIXED_NVIDIA_MULTI="2", STEPS_07=mode,
                  DP_RUNS="ddp", PP_RUNS="1f1b", PP_MICROBATCHES="4", PP_ARGS="--probe-trials 1",
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd", **SMALL_DP)
    out = tmp_path / "results" / out_dir
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert json.loads((out / record).read_text())["world_size"] == 3


def test_step_07_megatron_needs_megatron_path(tmp_path):
    code = sbatch(tmp_path, "--job-name=gpubridge-07_mixed_hetjob", "--nodes=1",
                  "--gres=gpu:1", ":", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "07_mixed_hetjob.sbatch"), STEPS_07="megatron")
    assert code != 0 and "STEPS_07=megatron needs MEGATRON_PATH" in job_output(tmp_path)


def test_dp_baseline_job(tmp_path):
    code = submit(tmp_path, "dp-baseline", DP_MICRO="2", **SMALL_DP)
    out = tmp_path / "results" / "dp_baseline_nvidia"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    record = json.loads((out / "dp_single.json").read_text())
    assert record["mode"] == "single" and record["micro_batches"] == 2
    assert len(record["losses"]) == 4
    log = (tmp_path / "slurm.log").read_text()
    assert "--partition=gpu" in log and "--time=00:30:00" in log


def test_step_08_matrix_on_the_mixed_pair(tmp_path):
    code = submit(tmp_path, "08", AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd")
    out = tmp_path / "results" / "08_matrix_nvidia-amd"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    runs = sorted(p.parent.name for p in (out / "matrix").glob("*/bench.json"))
    assert runs == ["D-bf16", "D-default", "D-default-repeat", "D-fp8", "D-pinned",
                    "D-pipelined-4K", "D-sharded", "E"]
    # Each run's compression reaches step, so Triton's check comes before the first compressed run.
    order = list(steps(out))
    first = min(order.index(f"matrix_{name}") for name in ("D-bf16", "D-fp8"))
    assert "triton_check" in order[:first] and order.count("triton_check") == 1, order
    log = "\n".join(p.read_text() for p in (tmp_path / "work" / "logs").glob("*.out"))
    assert "step triton_check: both @VENV@/bin/python" in log  # on both sides

    def bench(name):
        return json.loads((out / "matrix" / name / "bench.json").read_text())

    assert bench("D-pinned")["environment"]["env"]["GPUBRIDGE_PINNED_STAGING"] == "1"
    assert bench("D-pipelined-4K")["run"]["policy"] == "pipelined-reduce-bridge-broadcast"
    assert bench("D-fp8")["environment"]["env"]["GPUBRIDGE_BRIDGE_COMPRESSION"] == "fp8"
    assert bench("D-bf16")["environment"]["env"]["GPUBRIDGE_BRIDGE_COMPRESSION"] == "bf16"
    assert bench("D-pipelined-4K")["environment"]["env"]["GPUBRIDGE_CHUNK_BYTES"] == "4K"
    assert {r["op"] for r in bench("E")["rows"]} == {"flat-gloo"}
    assert all(r.get("phases_ms") for r in bench("D-default")["rows"])
    assert {p.name for p in (out / "network").iterdir()} == {"nvidia", "amd"}
    md = (out / "matrix.md").read_text()
    assert "| D-fp8 |" in md and "Measured / predicted" in md
    assert "Without A1 and A2" in md


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo to build multitcp")
def test_step_08_with_the_multitcp_bridge(tmp_path):
    code = submit(tmp_path, "08", AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  MULTITCP_BRIDGE="1", GPUBRIDGE_MULTITCP_LIB_DIR=str(tmp_path / "lib"),
                  FAKE_SLURM_HET_ENV_0="GPUBRIDGE_VENDOR=nvidia",
                  FAKE_SLURM_HET_ENV_1="GPUBRIDGE_VENDOR=amd")
    out = tmp_path / "results" / "08_matrix_nvidia-amd"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    runs = {p.parent.name: json.loads(p.read_text())
            for p in (out / "matrix").glob("D-multitcp*/bench.json")}
    assert sorted(runs) == ["D-multitcp", "D-multitcp-1stream", "D-multitcp-pinned"]
    assert runs["D-multitcp-pinned"]["environment"]["env"]["GPUBRIDGE_PINNED_STAGING"] == "1"
    assert runs["D-multitcp-1stream"]["environment"]["env"]["GPUBRIDGE_MULTITCP_STREAMS"] == "1"
    assert "| D-multitcp |" in (out / "matrix.md").read_text()


@pytest.mark.parametrize(("target", "vendor", "names"),
                         [("08-nvidia-pair", "nvidia", ["A1", "A2", "C", "E-nvidia"]),
                          ("08-amd-pair", "amd", ["B1", "B2", "C-amd", "E-amd"])])
def test_step_08_matrix_on_a_same_vendor_pair(target, vendor, names, tmp_path):
    code = submit(tmp_path, target, AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  SAME_PARTITION_PAIRS_OK="1", GPUBRIDGE_VENDOR=vendor)
    pair = f"{vendor}-{vendor}"
    out = tmp_path / "results" / f"08_matrix_{pair}"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)
    assert sorted(p.parent.name for p in (out / "matrix").glob("*/bench.json")) == sorted(names)
    native, tcp, bridged = (json.loads((out / "matrix" / n / "bench.json").read_text())
                            for n in names[:3])
    assert {r["op"] for r in native["rows"]} == {"native"}
    chosen = json.loads((out / "preflight" / "preflight-master.json").read_text())["chosen_ifname"]
    assert tcp["environment"]["env"]["NCCL_IB_DISABLE"] == "1"
    assert tcp["environment"]["env"]["NCCL_SOCKET_IFNAME"] == f"={chosen}"  # exact name
    assert bridged["run"]["split_test"] == "node" and len(bridged["run"]["islands"]) == 2
    md = (out / "matrix.md").read_text()
    if vendor == "nvidia":
        assert re.search(r"\| 1KB \| A2 \| [^|]+ \| [^|]+ \| 1\.00x \|", md), md


def test_step_08_refuses_two_components_on_one_node(tmp_path):
    # Slurm put both 1-GPU components on one node: NCCL would never use the network.
    env = cluster_env(tmp_path, SLURM_JOB_ID="4242", MATRIX_PAIR="nvidia-nvidia",
                      SLURM_JOB_NODELIST_HET_GROUP_0="gpunode1",
                      SLURM_JOB_NODELIST_HET_GROUP_1="gpunode1")
    result = subprocess.run(["bash", str(EXPLORER / "08_bench_matrix.sbatch")], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1
    assert "both components are on gpunode1" in result.stderr
    assert "both components are on one node; the matrix needs two" in result.stderr
    out = tmp_path / "results" / "08_matrix_nvidia-nvidia"
    assert [e["event"] for e in events(out)] == ["started", "error", "exit"]
    assert "gpunode1" not in (out / "job.jsonl").read_text()  # triage pastes it as is
    assert not (out / "matrix").exists()


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
@pytest.mark.parametrize("mode, out_dir, record", [("dp", "dp", "dp_ddp.json"),
                                                   ("pipeline", "pp", "pp_1f1b.json")])
def test_step_07_trains_on_a_same_vendor_pair(vendor, mode, out_dir, record, tmp_path):
    # Two GPUs of one type: one island, so the runs never touch the bridge.
    code = submit(tmp_path, f"07-{vendor}-pair", AMD_NODES_CONFIRMED="1", GPU_TYPE_AMD="mi50",
                  SAME_PARTITION_PAIRS_OK="1", GPUBRIDGE_VENDOR=vendor, STEPS_07=mode,
                  DP_RUNS="ddp", PP_RUNS="1f1b", PP_MICROBATCHES="2",
                  PP_ARGS="--probe-trials 1", **SMALL_DP)
    out = tmp_path / "results" / f"07_pair_{vendor}_{out_dir}"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    run = json.loads((out / record).read_text())
    assert run["world_size"] == 2 and len(run["run"]["islands"]) == 1
    gres = "gpu:mi50:1" if vendor == "amd" else "gpu:1"
    assert (tmp_path / "slurm.log").read_text().count(f"--gres={gres}") == 2


def test_step_07_pairs_only_train(tmp_path):
    code = submit(tmp_path, "07-nvidia-pair", SAME_PARTITION_PAIRS_OK="1",
                  GPUBRIDGE_VENDOR="nvidia", STEPS_07="all")
    assert code != 0 and "a same-vendor pair runs STEPS_07=dp or pipeline" in job_output(tmp_path)


def test_same_partition_pairs_wait_for_research_computing(tmp_path):
    for target in ("08-nvidia-pair", "08-amd-pair", "07-nvidia-pair", "07-amd-pair"):
        result = subprocess.run(["bash", str(EXPLORER / "submit.sh"), target],
                                env=cluster_env(tmp_path, AMD_NODES_CONFIRMED="1",
                                                GPU_TYPE_AMD="mi50"),
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 1, result.stderr
        assert "waiting on Research Computing" in result.stderr
    assert not (tmp_path / "slurm.log").exists()  # nothing reached sbatch


def test_net_check_job(tmp_path):
    code = submit(tmp_path, "net-nvidia")
    out = tmp_path / "results" / "net_check_nvidia"
    assert code == 0 and failed(out) == set(), job_output(tmp_path)
    assert list(steps(out)) == ["network", "net_default", "net_tcp", "matrix", "rdma"]
    side = out / "network" / "nvidia"
    assert {p.name for p in side.iterdir()} == {"sys_class_infiniband.txt", "ibstat.txt",
                                                "ibv_devinfo.txt", "ip_addr.txt"}
    rdma = out / "rdma" / "nvidia"
    assert {p.name for p in rdma.iterdir()} == {"kernel.txt", "modules.txt", "rdma_core.txt",
                                                "rdma_link.txt", "ucx.txt", "rocm_dmabuf.txt"}
    assert str(tmp_path) not in "".join(p.read_text() for p in rdma.iterdir())
    assert "## Network" in (out / "matrix.md").read_text()
    log = (tmp_path / "slurm.log").read_text()
    assert "--partition=gpu" in log and "--time=00:10:00" in log


def test_rdma_report_keeps_paths_out_and_reads_all_of_nm(tmp_path):
    # Stand-ins: an nm whose symbol comes first in a long listing (grep -q would stop
    # reading and nm die of SIGPIPE, which pipefail reports as no match), and a ucx_info
    # and ldconfig that print site paths, which the records must not keep.
    bin_dir, lib = tmp_path / "bin", tmp_path / "site" / "lib"
    (lib / "ucx").mkdir(parents=True)
    bin_dir.mkdir()
    for name in ("ucx/libuct_cma.so", "ucx/libuct_ib.so", "libibverbs.so.1"):
        (lib / name).touch()
    tools = {
        # Versioned, as newer binutils print rdma-core's symbols; an unversioned near miss.
        "nm": "echo '0000 T ibv_reg_dmabuf_mr@@IBVERBS_1.12'; echo '0000 T ibv_reg_dmabuf_mr_x'; "
              "seq 1 200000 | sed 's/^/0000 T sym_/'",
        "ucx_info": f"""if [ "$1" = -v ]; then echo '# Library path: {lib}/libucs.so.0'
echo '# Configured with: --prefix={tmp_path}/site'; else echo '# Memory domain: tcp'; fi""",
        "ldconfig": f"echo '\tlibibverbs.so.1 (libc6,x86-64) => {lib}/libibverbs.so.1'",
    }
    for name, body in tools.items():
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    out = tmp_path / "out"
    result = subprocess.run(["bash", str(KIT / "rdma_report.sh"), str(out)],
                            env=clean_env(PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "libibverbs.so.1: has ibv_reg_dmabuf_mr" in (out / "rdma_core.txt").read_text()
    ucx = (out / "ucx.txt").read_text()
    assert "# transport modules installed: cma ib" in ucx
    assert "# Library path: .../libucs.so.0" in ucx
    assert not [p.name for p in out.iterdir() if str(tmp_path) in p.read_text()]


def test_net_check_job_survives_a_module_that_wont_load(tmp_path):
    code = submit(tmp_path, "net-nvidia", RDMA_MODULES="no-such-module",
                  FAKE_MODULES_MISSING="no-such-module")
    out = tmp_path / "results" / "net_check_nvidia"
    assert code == 1 and failed(out) == {"rdma"}, job_output(tmp_path)
    assert "finished" in [e["event"] for e in events(out)]  # finish ran: the job's records


def test_train_bench_job(tmp_path):
    code = submit(tmp_path, "train-nvidia")
    out = tmp_path / "results" / "train_bench_nvidia"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    record = json.loads((out / "train_bench_tiny-gpt.json").read_text())
    assert record["world_size"] == 1 and record["samples_per_s"] > 0
    assert record["ranks"][0]["waiting_ms"] == 0.0


def test_train_multi_job(tmp_path):
    # With multigpu access: 1 GPU, then 4 as one island, then two bridged halves.
    code = submit(tmp_path, "train-multi", GPU_TYPE_MULTI="v100-sxm2")
    out = tmp_path / "results" / "train_bench_nvidia_n4"
    assert code == 1 and failed(out) == {"probe"}, job_output(tmp_path)  # no GPU here
    assert list(steps(out)) == ["probe", "train_bench_tiny-gpt", "train_bench_tiny-gpt_n4",
                                "train_bench_tiny-gpt_split"]
    one, island, split = (json.loads((out / f"train_bench_tiny-gpt{suffix}.json").read_text())
                          for suffix in ("", "_n4", "_split"))
    assert (one["world_size"], island["world_size"], split["world_size"]) == (1, 4, 4)
    assert split["params_identical"] and island["params_identical"]
    log = (tmp_path / "slurm.log").read_text()
    assert "--partition=multigpu" in log and "--gres=gpu:v100-sxm2:4" in log
    assert "--time=01:00:00" in log


def test_train_multi_needs_an_even_gpu_count(tmp_path):
    result = subprocess.run(["bash", str(EXPLORER / "submit.sh"), "train-multi"],
                            env=cluster_env(tmp_path, GPUS_TRAIN_MULTI="3"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "must be even" in result.stderr
    assert not (tmp_path / "slurm.log").exists()


def test_amd_node_job(tmp_path):
    # Rung 3 on Explorer: amd/run_all.sh on one AMD node, without setup or tarball.
    code = sbatch(tmp_path, "--job-name=gpubridge-amd_node", "--nodes=1", "--gres=gpu:mi100:1",
                  str(EXPLORER / "amd_node.sbatch"), GPUBRIDGE_VENDOR="amd", GPU_COUNT="1")
    out = tmp_path / "results" / "amd_node"
    assert failed(out) == {"probe", "device_map", "rccl_version"}, job_output(tmp_path)
    assert code == 1
    assert {"check_1gpu", "gpu_tests", "check_split_shared_gpu", "summarize"} <= set(steps(out))
    assert not (tmp_path / "results" / "amd_node.tgz").exists()
    # job.jsonl, as every Explorer job writes it: started, finished (3 failed), exit 1.
    events = [json.loads(line) for line in (out / "job.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["started", "finished", "exit"]
    assert events[1]["failed"] == 3 and events[2]["code"] == 1


def loopback() -> str:
    names = {name for _, name in socket.if_nameindex()}
    return next(name for name in ("lo", "lo0") if name in names)


def test_mixed_node_script_on_two_pretend_machines(tmp_path):
    common = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(_free_port()),
              "PREFLIGHT_PORT": str(_free_port()), "IFNAME": loopback(),
              "LOCAL_ADDR": "127.0.0.1", "NPROC": "1", "SKIP_SETUP": "1",
              "TORCH_FLAVOR": "system", "RUN_ID": f"test-{os.getpid()}",
              "BENCH_MAX_BYTES": "16K", "RESULTS_ROOT": str(tmp_path / "results"),
              "VENV": sys.prefix, "KIT_TIMEOUT": "120", "TRAIN_BENCH_MODELS": "tiny-gpt",
              "TRAIN_BENCH_ARGS": "--steps 2 --warmup-steps 1 --comm-trials 1",
              **SMALL_DP, "DP_RUNS": "ddp ddp-fp8", "PP_RUNS": "1f1b", "PP_MICROBATCHES": "2",
              "PP_ARGS": "--probe-trials 1"}
    # The worker's Triton check fails (TRITON_CHECK_LAUNCH=false), the master's passes: each
    # machine decides alone, so the worker skips only its own kernel benchmark and every
    # mixed step, compressed ones included, still runs on both, in step.
    procs = {
        role: subprocess.Popen(["bash", str(KIT / "mixed" / "node.sh")],
                               env=clean_env(ROLE=role, TARGET=role, GPUBRIDGE_VENDOR=vendor,
                                             **common, **extra),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for role, vendor, extra in (("master", "nvidia", {}),
                                    ("worker", "amd", {"TRITON_CHECK_LAUNCH": "false"}))
    }
    output = {role: proc.communicate(timeout=600)[0] for role, proc in procs.items()}
    for role, proc in procs.items():
        out = tmp_path / "results" / role
        if role == "master":
            assert proc.returncode == 0, output[role][-3000:]
            assert failed(out) == set()
            assert "verdict: every step that ran passed" in (out / "triage.txt").read_text()
        else:
            assert proc.returncode == 1, output[role][-3000:]
            assert failed(out) == {"triton_check", "bench_compression_kernels"}
            records = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
            skipped = {r["step"] for r in records if "not_run" in r}
            assert skipped == {"bench_compression_kernels"}
        assert (out / f"preflight-{role}.json").exists()
        assert (tmp_path / "results" / f"{role}.tgz").exists()
        alone = json.loads((out / "train_bench_alone_tiny-gpt.json").read_text())
        assert alone["world_size"] == 1
        baseline = json.loads((out / "dp_single.json").read_text())
        assert baseline["mode"] == "single" and len(baseline["losses"]) == 4
    # The two machines together: rank 0 writes the record. torchrun numbers the
    # machines by address, then process id, not by role: here (one address) rank 0
    # is on whichever machine started its torchrun first.
    def together(name: str) -> dict:
        records = [tmp_path / "results" / role / name for role in procs]
        written = [path for path in records if path.exists()]
        assert len(written) == 1, records
        return json.loads(written[0].read_text())

    both = together("train_bench_tiny-gpt.json")
    assert both["world_size"] == 2 and both["params_identical"]
    for run, compression in (("ddp", "none"), ("ddp-fp8", "fp8")):
        dp = together(f"dp_{run}.json")
        assert dp["world_size"] == 2 and dp["compression"] == compression
        assert dp["param_checks"] and all(c["identical"] for c in dp["param_checks"])
    pp = together("pp_1f1b.json")
    assert pp["world_size"] == 2 and len(pp["losses"]) == 4


def test_mixed_node_script_leaves_out_what_is_set_empty(tmp_path):
    result = subprocess.run(["bash", str(KIT / "mixed" / "node.sh"), "--dry-run"],
                            env=clean_env(ROLE="master", MASTER_ADDR="127.0.0.1",
                                          LOCAL_ADDR="127.0.0.1", IFNAME=loopback(), NPROC="1",
                                          SKIP_SETUP="1", TORCH_FLAVOR="system",
                                          VENV=sys.prefix, RESULTS_ROOT=str(tmp_path),
                                          TRAIN_BENCH_MODELS="", DP_RUNS="", PP_RUNS=""),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    printed = result.stdout + result.stderr
    assert "bench_all_reduce.py" in printed
    for script in ("train_bench.py", "train_dp.py", "train_pipeline.py"):
        assert script not in printed, script


def test_mixed_containers_dry_run(tmp_path):
    result = subprocess.run(["bash", str(KIT / "mixed" / "containers.sh"), "--dry-run",
                             str(tmp_path / "out")], env=clean_env(SUBNET="10.9.8.0/24"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    runs = [line for line in result.stderr.splitlines() if line.startswith("+ docker run -d")]
    assert len(runs) == 2
    for line, role, vendor, ip in zip(runs, ("master", "worker"), ("nvidia", "amd"),
                                      ("10.9.8.10", "10.9.8.11")):
        for part in (f"--ip {ip}", f"ROLE={role}", f"GPUBRIDGE_VENDOR={vendor}",
                     "MASTER_ADDR=10.9.8.10", "IFNAME=eth0", "-e DP_RUNS", "-e PP_RUNS",
                     "-e PP_MICROBATCHES", "scripts/gpu/mixed/node.sh"):
            assert part in line, (part, line)
    assert "triage.py /out/master" in result.stderr and "triage.py /out/worker" in result.stderr


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
