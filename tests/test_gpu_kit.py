"""The GPU validation kit (scripts/gpu, examples/train_synthetic.py), tested without GPUs.

Shell scripts: shellcheck and --dry-run output. Python tools: run end to end
under torchrun on CPU in simulation mode, including split-test runs.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "scripts" / "gpu"
sys.path.insert(0, str(KIT))

import bench_all_reduce  # noqa: E402
import summarize  # noqa: E402

SHELL_SCRIPTS = sorted(
    [*KIT.glob("*.sh"), *KIT.glob("*/*.sh"), *KIT.glob("explorer/*.sbatch")]
)
FAKE_PROJECT = "/projects/example-lab"
OWN_SETTINGS = ("PROJECT_DIR", "WORK_DIR", "VENV", "VENV_ROCM", "ROCM_INDEX", "ROCM_INDEX_URL",
                "ROCM_TORCH_VERSION", "TORCH_VERSION", "TORCH_INDEX_URL")


def clean_env(**extra: str) -> dict[str, str]:
    # A developer's own settings, including an explorer/local.env, must not leak in.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GPUBRIDGE_") and k not in OWN_SETTINGS}
    env.update({"OMP_NUM_THREADS": "1", "EXPLORER_LOCAL_ENV": os.devnull, **extra})
    return env


def torchrun(nproc: int, script: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    # Not --standalone: it hangs on hosts whose own name doesn't resolve (macOS).
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
           "--master-addr=127.0.0.1", f"--master-port={_free_port()}",
           f"--nproc-per-node={nproc}", str(script), *args]
    return subprocess.run(cmd, env=clean_env(**env), capture_output=True, text=True,
                          timeout=300)


def dry_run(script: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    env = {"GPUBRIDGE_REPO": str(REPO), "PROJECT_DIR": FAKE_PROJECT, **env}
    return subprocess.run(["bash", str(script), "--dry-run", *args], env=clean_env(**env),
                          capture_output=True, text=True, timeout=60)


# ---- shell scripts ---------------------------------------------------------------

@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_is_clean():
    result = subprocess.run(["shellcheck", *map(str, SHELL_SCRIPTS)], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_every_shell_script_is_covered():
    names = {p.name for p in SHELL_SCRIPTS}
    assert {"common.sh", "setup_env.sh", "submit.sh", "run_all.sh", "node.sh",
            "00_partitions.sh", "setup.sbatch", "01_probe.sbatch", "06_scaling.sbatch",
            "07_mixed_hetjob.sbatch"} <= names


@pytest.mark.parametrize("flavor", ["cuda", "rocm", "system"])
def test_setup_dry_run(flavor, tmp_path):
    result = dry_run(KIT / "setup_env.sh", "--flavor", flavor, "--venv", str(tmp_path / "v"))
    assert result.returncode == 0, result.stderr
    assert "uv pip install" in result.stdout
    if flavor == "cuda":
        assert "torch==2.14.1+cu126" in result.stdout
    if flavor == "rocm":
        assert "torch==2.14.1+rocm7.2" in result.stdout
    if flavor == "system":
        assert "--system-site-packages" in result.stdout
    assert not (tmp_path / "v").exists(), "dry-run must not create anything"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("01", ["--partition=gpu", "--gres=gpu:1", "01_probe.sbatch"]),
        ("02", ["--partition=gpu", "--gres=gpu:1", "02_split_shared_gpu.sbatch"]),
        ("03", ["--partition=multigpu", "--gres=gpu:2", "03_single_island.sbatch"]),
        ("04", ["--partition=multigpu", "--gres=gpu:4", "04_split_4gpu.sbatch"]),
        ("05", ["--partition=multigpu", "--nodes=2", "05_multinode.sbatch"]),
        ("06", ["--partition=multigpu", "--gres=gpu:4", "06_scaling.sbatch"]),
    ],
)
def test_submit_dry_run(step, expected):
    result = dry_run(KIT / "explorer" / "submit.sh", step)
    assert result.returncode == 0, result.stderr
    for text in expected:
        assert text in result.stdout, (text, result.stdout)
    assert "--account" not in result.stdout


def fake_command(directory: Path, name: str, body: str) -> str:
    """Put an executable NAME on a PATH that starts with DIRECTORY; returns that PATH."""
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return f"{directory}{os.pathsep}{os.environ['PATH']}"


def test_setup_refuses_to_run_on_a_slurm_login_node(tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "exit 0\n")
    result = subprocess.run(["bash", str(KIT / "setup_env.sh"), "--flavor", "cuda", "--venv",
                             str(tmp_path / "v")], env=clean_env(PATH=path),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "login node" in result.stderr and "submit.sh setup" in result.stderr
    assert not (tmp_path / "v").exists()


def test_submit_setup_is_a_short_cpu_job_not_a_login_node_install():
    result = dry_run(KIT / "explorer" / "submit.sh", "setup", USER="alice")
    assert result.returncode == 0, result.stderr
    sbatch = [line for line in result.stdout.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, result.stdout
    for text in ["--partition=short", "--time=01:00:00", "--cpus-per-task=4", "--mem=16G",
                 "explorer/setup.sbatch"]:
        assert text in sbatch[0], text
    assert "--gres" not in sbatch[0]
    assert "uv pip install" not in result.stdout, "nothing may be installed on the login node"


@pytest.mark.parametrize(("flavor", "venv", "wheel"), [
    ("cuda", "/gpubridge-gpu/venv ", "torch==2.14.1+cu126"),
    ("rocm", "/gpubridge-gpu/venv-rocm ", "torch==2.14.1+rocm7.2"),
])
def test_setup_job_loads_the_explorer_module_before_installing(flavor, venv, wheel):
    result = dry_run(KIT / "explorer" / "setup.sbatch", USER="alice", TORCH_FLAVOR=flavor)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.index("+ module load explorer") < out.index("+ uv venv") < out.index(wheel)
    assert f"{FAKE_PROJECT}/alice{venv}" in out


def test_setup_rocm_uses_the_configured_build_and_checks_the_gpu():
    result = dry_run(KIT / "explorer" / "setup.sbatch", TORCH_FLAVOR="rocm",
                     ROCM_TORCH_VERSION="2.9.1", ROCM_INDEX="rocm6.4")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1+rocm6.4" in out
    assert out.index("torch==2.9.1+rocm6.4") < out.index("probe.py")
    assert "/setup_rocm" in out
    url = "https://mirror.example/whl/rocm6.4"
    out = dry_run(KIT / "explorer" / "setup.sbatch", TORCH_FLAVOR="rocm", ROCM_INDEX="rocm6.4",
                  ROCM_INDEX_URL=url).stdout
    assert f"--index-url {url} torch==2.14.1+rocm6.4" in out
    assert "probe.py" not in dry_run(KIT / "explorer" / "setup.sbatch").stdout  # CUDA: CPU job


def test_setup_env_takes_an_index_url(tmp_path):
    out = dry_run(KIT / "setup_env.sh", "--flavor", "cuda", "--venv", str(tmp_path / "v"),
                  TORCH_INDEX_URL="https://mirror.example/whl/cu126").stdout
    assert "--index-url https://mirror.example/whl/cu126 torch==2.14.1+cu126" in out


def test_setup_job_needs_a_slurm_job_and_can_skip_the_module():
    result = subprocess.run(["bash", str(KIT / "explorer" / "setup.sbatch")],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PROJECT_DIR=FAKE_PROJECT),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "not in a Slurm job" in result.stderr
    assert "module load" not in dry_run(KIT / "explorer" / "setup.sbatch",
                                        SETUP_MODULES="").stdout


def test_work_dir_comes_from_project_dir():
    work = f"{FAKE_PROJECT}/alice/gpubridge-gpu"
    out = dry_run(KIT / "explorer" / "submit.sh", "01", USER="alice").stdout
    assert f"--output={work}/logs/%x-%j.out" in out
    out = dry_run(KIT / "explorer" / "01_probe.sbatch", USER="alice").stdout
    assert f"source {work}/venv/bin/activate" in out
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="",
                  WORK_DIR="/scratch/w").stdout
    assert "--output=/scratch/w/logs/%x-%j.out" in out and "/projects/" not in out


@pytest.mark.parametrize(("script", "args"), [("submit.sh", ["01"]), ("submit.sh", ["setup"]),
                                              ("01_probe.sbatch", []), ("setup.sbatch", [])])
def test_explorer_scripts_need_a_project_dir(script, args):
    result = dry_run(KIT / "explorer" / script, *args, PROJECT_DIR="")
    assert result.returncode == 1
    assert "set PROJECT_DIR" in result.stderr and "local.env" in result.stderr
    assert "/projects/<your-project>" in result.stderr
    assert "+ sbatch" not in result.stdout


def test_project_dir_from_local_env_and_the_environment_wins(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f'# my settings\nPROJECT_DIR="/projects/from-file"\n'
                     f"touch {tmp_path}/ran\nACCOUNT=$(touch {tmp_path}/ran2)\n")
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="", USER="alice",
                  EXPLORER_LOCAL_ENV=str(local)).stdout
    assert "--output=/projects/from-file/alice/gpubridge-gpu/logs/" in out
    assert not (tmp_path / "ran").exists() and not (tmp_path / "ran2").exists(), "parsed, not run"
    out = dry_run(KIT / "explorer" / "submit.sh", "01", PROJECT_DIR="/projects/from-env",
                  USER="alice", EXPLORER_LOCAL_ENV=str(local)).stdout
    assert "--output=/projects/from-env/alice/gpubridge-gpu/logs/" in out


def test_local_env_is_gitignored():
    result = subprocess.run(["git", "check-ignore", "-q", "scripts/gpu/explorer/local.env"],
                            cwd=REPO)
    assert result.returncode == 0


FAKE_SINFO = r"""
part="" nodes=0 fmt=""
while [ $# -gt 0 ]; do
    case $1 in
    -p) part=$2; shift 2 ;;
    -o) fmt=$2; shift 2 ;;
    -N) nodes=1; shift ;;
    *) shift ;;
    esac
done
if [ "$nodes" = 0 ]; then echo "format=[$fmt] partition=$part"; exit 0; fi
case $part in
sharing) printf '%s\n' "${SHARING_GRES[@]}" ;;
gpu) printf 'd1001 gpu:v100-sxm2:4(S:0-1)\nd1002 gpu:v100-pcie:2\nd1003 gpu:2\n' ;;
*) echo "sinfo: error: invalid partition specified: $part" >&2; exit 1 ;;
esac
"""


def partitions_step(tmp_path, sharing: list[str]) -> tuple[subprocess.CompletedProcess, Path]:
    gres = " ".join(f"'{line}'" for line in sharing)
    path = fake_command(tmp_path / "bin", "sinfo", f"SHARING_GRES=({gres})\n" + FAKE_SINFO)
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), "00"],
                            env=clean_env(PATH=path, RESULTS_ROOT=str(tmp_path / "results")),
                            capture_output=True, text=True, timeout=60)
    return result, tmp_path / "results" / "00_partitions"


def test_step_00_records_sinfo_and_finds_amd_gpus_in_sharing(tmp_path):
    result, out = partitions_step(tmp_path, [
        "c0101 gpu:mi100:4(S:0-1)", "c0102 gpu:mi100:4(S:0,1)", "c0103 gpu:a100:2,gpu:t4:1",
        "c0104 (null)",
    ])
    assert result.returncode == 0, result.stderr
    fmt = "format=[%20N %10c %10m %25f %10G %10t]"
    assert (out / "sinfo-sharing.txt").read_text() == f"{fmt} partition=sharing\n"
    assert (out / "sinfo-gpu.txt").read_text() == f"{fmt} partition=gpu\n"
    rows = [line.split() for line in (out / "gpu-types.txt").read_text().splitlines()]
    assert ["sharing", "mi100", "amd", "2", "8"] in rows
    assert ["sharing", "a100", "nvidia", "1", "2"] in rows
    assert ["sharing", "t4", "nvidia", "1", "1"] in rows
    assert ["gpu", "v100-sxm2", "nvidia", "1", "4"] in rows
    assert ["gpu", "(untyped)", "unknown", "1", "2"] in rows
    assert "AMD GPUs in sharing: mi100." in result.stdout


def test_step_00_says_when_sharing_has_no_amd_gpus(tmp_path):
    result, out = partitions_step(tmp_path, ["c0103 gpu:a100:2"])
    assert result.returncode == 0, result.stderr
    assert "No AMD GPU type found in sharing" in (out / "gpu-types.txt").read_text()


def test_step_00_dry_run_prints_the_sinfo_commands():
    out = dry_run(KIT / "explorer" / "submit.sh", "00").stdout
    for partition in ("sharing", "gpu", "multigpu"):
        assert f"+ sinfo -p {partition} -o %20N\\ %10c" in out, out


@pytest.mark.parametrize("step", ["setup-rocm", "07"])
def test_explorer_mixed_path_is_dry_run_only_until_amd_gpus_are_confirmed(step, tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo SUBMITTED\n")
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), step],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          GPU_TYPE_AMD="mi100", WORK_DIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "dry-run only until step 00" in result.stderr
    assert "SUBMITTED" not in result.stdout


def test_submit_07_is_one_heterogeneous_job_on_gpu_and_sharing():
    out = dry_run(KIT / "explorer" / "submit.sh", "07", GPU_TYPE_AMD="mi100").stdout
    sbatch = [line for line in out.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, out
    nvidia, amd = sbatch[0].split(" : ")
    assert "--partition=gpu " in nvidia and "--gres=gpu:1 " in nvidia
    assert "--job-name=gpubridge-07_mixed_hetjob" in nvidia
    assert "--partition=sharing " in amd and "--gres=gpu:mi100:1 " in amd
    assert amd.rstrip().endswith("07_mixed_hetjob.sbatch")


def test_submit_07_can_put_the_nvidia_side_on_multigpu():
    out = dry_run(KIT / "explorer" / "submit.sh", "07", PARTITION_MIXED_NVIDIA="multigpu",
                  GPUS_MIXED_NVIDIA="2", GPUS_MIXED_AMD="2").stdout
    nvidia, amd = next(line for line in out.splitlines() if line.startswith("+ sbatch ")).split(
        " : ")
    assert "--partition=multigpu " in nvidia and "--gres=gpu:2 " in nvidia
    assert "--partition=sharing " in amd and ":2 " in amd


def test_mixed_hetjob_runs_each_side_in_its_own_venv():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    sruns = [line for line in out.splitlines() if line.startswith("+ srun --het-group=0")]
    # probe, preflight, check, bench, then a check per opt-in policy and the candidates bench
    assert len(sruns) >= 6
    assert any("--policy pipelined-reduce-bridge-broadcast" in line for line in sruns[4:])
    assert "--write-thresholds" in sruns[-1]
    for line in sruns:
        nvidia, amd = line.split(" : ")
        assert "/w/venv/bin/" in nvidia and "/w/venv-rocm/" not in nvidia
        assert amd.startswith("--het-group=1 ") and "/w/venv-rocm/bin/" in amd
    assert "--role master" in sruns[1] and "--role worker" in sruns[1]
    assert "probe_nvidia" in sruns[0] and "probe_amd" in sruns[0]
    assert out.index("preflight.py") < out.index("torchrun")
    assert "check.py" in sruns[2] and "bench_all_reduce.py" in sruns[3]
    nvidia_bench, amd_bench = sruns[3].replace("\\", "").split(" : ")  # undo printf %q
    assert "--ops island,gpubridge" in nvidia_bench and "--ops island,gpubridge" in amd_bench
    assert "srun --het-group=1 --ntasks=1 rocm-smi" in out


def test_submit_uses_site_settings_from_the_environment():
    result = dry_run(KIT / "explorer" / "submit.sh", "05", ACCOUNT="lab",
                     PARTITION_MULTI="bigq", GPU_TYPE_MULTI="h100", TIME_05="00:05:00")
    for text in ["--account=lab", "--partition=bigq", "--gres=gpu:h100:2", "--time=00:05:00"]:
        assert text in result.stdout


def test_submit_free_and_multi_submit_every_step():
    def submitted(group):
        out = dry_run(KIT / "explorer" / "submit.sh", group).stdout
        return [line.split()[-1].rsplit("/", 1)[-1] for line in out.splitlines()
                if line.startswith("+ sbatch ")]
    assert submitted("free") == ["01_probe.sbatch", "02_split_shared_gpu.sbatch"]
    assert submitted("multi") == ["03_single_island.sbatch", "04_split_4gpu.sbatch",
                                  "05_multinode.sbatch", "06_scaling.sbatch"]


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ("01_probe.sbatch", ["probe.py", "--nproc-per-node=2",
                             "--expect-init-error LOCAL_RANK=1\\ asks\\ for\\ GPU\\ 1"]),
        ("02_split_shared_gpu.sbatch", ["--nproc-per-node=2", "check.py"]),
        ("03_single_island.sbatch", ["device_map.py", "check.py", "bench_all_reduce.py"]),
        ("04_split_4gpu.sbatch", ["check_half", "check_alternate", "bench_all_reduce.py",
                                  "--policy pipelined-reduce-bridge-broadcast",
                                  "--write-thresholds", "--policy auto-tuned"]),
        ("05_multinode.sbatch", ["--policy pipelined-reduce-bridge-broadcast",
                                 "--write-thresholds"]),
        ("05_multinode.sbatch", ["srun", "--rdzv-backend=c10d", "check_split_by_node"]),
        ("06_scaling.sbatch", ["--nproc-per-node=4", "train_synthetic.py", "summarize.py"]),
        ("07_mixed_hetjob.sbatch", ["--het-group=1", "--rdzv-backend=c10d", "summarize.py"]),
    ],
)
def test_explorer_jobs_dry_run(job, expected):
    result = dry_run(KIT / "explorer" / job)
    assert result.returncode == 0, result.stderr
    for text in expected:
        assert text in result.stdout, (text, result.stdout)


def test_amd_dry_run_with_several_gpus():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="8").stdout
    for text in ["rocm-smi", "probe.py", "--nproc-per-node=9", "LOCAL_RANK=8\\ asks",
                 "NCCL_DEBUG=INFO", "--nproc-per-node=8",
                 "GPUBRIDGE_SPLIT_TEST=half", "bench_all_reduce.py", "train_n4", "tar -czf"]:
        assert text in out, text
    assert "train_n5" not in out


def test_amd_dry_run_with_one_gpu_shares_it():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="1").stdout
    assert "check_split_shared_gpu" in out and "--nproc-per-node=2" in out
    assert "bench_all_reduce.py" not in out


@pytest.mark.parametrize(("role", "is_host"), [("master", "1"), ("worker", "0")])
def test_mixed_node_dry_run(role, is_host):
    result = dry_run(KIT / "mixed" / "node.sh", ROLE=role, MASTER_ADDR="100.64.0.1",
                     GPU_COUNT="2")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.index("preflight.py") < out.index("torchrun"), "preflight must come first"
    assert f"--rdzv-conf=is_host={is_host}" in out
    assert "--rdzv-endpoint=100.64.0.1:29500" in out
    assert "--local-addr=" in out


def test_mixed_node_needs_a_role_and_master():
    assert "ROLE=master or ROLE=worker" in dry_run(KIT / "mixed" / "node.sh").stderr
    assert "MASTER_ADDR" in dry_run(KIT / "mixed" / "node.sh", ROLE="worker").stderr


# ---- Python tools end to end on CPU ------------------------------------------------

def read_ranks(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("rank*.json"))]


def test_check_in_a_split_test_is_flagged_everywhere(tmp_path):
    out = tmp_path / "check"
    result = torchrun(4, KIT / "check.py", "--out", str(out),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "NOT a mixed-vendor run" in result.stderr
    records = read_ranks(out)
    assert len(records) == 4
    for record in records:
        assert record["ok"]
        assert record["run"]["kind"] == "split-test"
        assert record["run"]["split_test"] == "half"
        assert record["run"]["real_mixed_vendor"] is False
        assert "NOT a mixed-vendor result" in record["run"]["warning"]
        assert [s["name"] for s in record["stages"]] == [
            "init", "device", "dtypes", "busy_gpu", "broadcast", "reductions", "all_gather",
            "reduce_scatter", "async", "reduction_agreement", "barrier", "destroy"]
        agreement = next(s for s in record["stages"] if s["name"] == "reduction_agreement")
        assert agreement["all_product_agree"] is True  # Gloo against Gloo on CPU
        assert agreement["backend"] == "gloo"
    assert summarize.main([str(out)]) == 0
    assert "SPLIT TEST - not a mixed-vendor result" in (out / "summary.md").read_text()
    assert json.loads((out / "summary.json").read_text())["split_test"] == "half"


@pytest.mark.parametrize(
    ("pattern", "ok"),
    [("doesn't apply to this cluster", True), ("LOCAL_RANK", False)],
)
def test_check_can_expect_init_to_fail_on_every_rank(pattern, ok, tmp_path):
    # native-only on a split test's two islands fails init() on every rank.
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), "--policy", "native-only",
                      "--expect-init-error", pattern,
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert (result.returncode == 0) == ok, result.stderr[-3000:]
    records = read_ranks(out)
    assert len(records) == 2
    for record in records:
        assert record["ok"] is ok
        (stage,) = record["stages"]
        assert stage["name"] == "expected_init_error"
        assert "doesn't apply" in stage["message"]
    assert summarize.main([str(out)]) == (0 if ok else 1)


def test_check_expecting_an_init_error_fails_when_init_succeeds(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), "--expect-init-error", "LOCAL_RANK",
                      GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 1
    for record in read_ranks(out):
        assert "init() succeeded" in record["stages"][0]["error"]


def test_check_without_split_is_not_flagged(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 0, result.stderr[-3000:]
    for record in read_ranks(out):
        assert record["run"]["split_test"] is None
        assert "warning" not in record["run"]
    summarize.main([str(out)])
    assert "SPLIT TEST" not in (out / "summary.md").read_text()


def test_bench_measures_native_and_bridged_in_one_split_run(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "16K",
                      "--warmup", "1", "--trials", "3",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    assert {r["op"] for r in rows} == {"native", "gpubridge"}
    assert {r["policy"] for r in rows if r["op"] == "gpubridge"} == {"reduce-bridge-broadcast"}
    assert {r["bytes"] for r in rows} == {1024, 4096, 16384}
    assert all(r["split_test"] == "half" and r["islands"] == 2 for r in rows)
    assert (out / "bench.csv").read_text().startswith("op,policy,run_kind,split_test,")
    assert "SPLIT TEST" in (out / "summary.md").read_text()


def test_bench_times_each_island_natively_in_the_same_job(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "4K",
                      "--warmup", "1", "--trials", "3", "--ops", "island,gpubridge",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    assert {(r["op"], r["world_size"]) for r in rows} == {
        ("island:nvidia", 2), ("island:amd", 2), ("gpubridge", 4)}
    assert {r["bytes"] for r in rows} == {1024, 4096}
    summary = (out / "summary.md").read_text()
    assert "island:amd median (us)" in summary and "island:nvidia median (us)" in summary
    assert "gpubridge / slowest island" in summary and "gpubridge / native" not in summary


def test_bench_reports_the_phases_of_the_gpubridge_op(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "4K",
                      "--warmup", "1", "--trials", "3", "--ops", "island,gpubridge", "--phases",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    bridged = [r for r in rows if r["op"] == "gpubridge"]
    assert len(bridged) == 2
    for r in bridged:
        assert list(r["phases_ms"]) == ["island-reduce", "bridge", "island-broadcast"]
        assert all(ms > 0 for ms in r["phases_ms"].values())
    assert all("phases_ms" not in r for r in rows if r["op"] != "gpubridge")
    assert "Phases of the gpubridge op (reduce-bridge-broadcast)" in (
        out / "summary.md").read_text()
    assert "phases_ms" not in (out / "bench.csv").read_text()


def test_bench_size_helpers():
    assert bench_all_reduce.parse_size("1K") == 1024
    assert bench_all_reduce.parse_size("1GB") == 2**30
    steps = bench_all_reduce.sizes(1024, 2**30, 4)
    assert steps[0] == 1024 and steps[-1] == 2**30 and len(steps) == 11
    info = {"kind": "gpu-mixed", "split_test": None, "islands": [1, 2], "world_size": 4,
            "policy": "reduce-bridge-broadcast"}
    r = bench_all_reduce.row("gpubridge", info, 2**20, "float32", [0.001, 0.002, 0.003])
    assert r["median_us"] == 2000.0
    assert r["policy"] == "reduce-bridge-broadcast"
    assert r["busbw_GBps"] == pytest.approx(r["algbw_GBps"] * 1.5, rel=1e-3)
    island = bench_all_reduce.row("island:amd", info, 2**20, "float32", [0.001], world=2)
    assert island["world_size"] == 2 and island["busbw_GBps"] == island["algbw_GBps"]
    trials = [[0.1, 0.4, 0.2, 0.3], [0.5, 0.1, 0.1, 0.2]]
    assert bench_all_reduce.slowest(trials) == [0.4, 0.5]
    assert bench_all_reduce.slowest(trials, (2, 3)) == [0.3, 0.2]


def test_training_demo_keeps_ranks_in_sync_and_learns(tmp_path):
    record = tmp_path / "train-n2.json"
    result = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "30",
                      "--dim", "128", "--batch-size", "32", "--json", str(record),
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(record.read_text())
    assert data["params_in_sync"] is True
    assert data["loss_last"] < data["loss_first"]
    assert data["run"]["split_test"] == "half" and "warning" in data


def test_scaling_summary_computes_efficiency(tmp_path):
    for n, sps in [(1, 100.0), (2, 180.0), (4, 300.0)]:
        (tmp_path / f"train-n{n}.json").write_text(json.dumps({
            "world_size": n, "samples_per_sec": sps, "step_ms_median": 1.0,
            "params_in_sync": True, "run": {"kind": "gpu-single-vendor", "split_test": None},
        }))
    assert summarize.main([str(tmp_path)]) == 0
    rows = json.loads((tmp_path / "summary.json").read_text())["rows"]
    assert [round(r["efficiency"], 2) for r in rows] == [1.0, 0.9, 0.75]


def _fake_rank(rank: int, world: int, bus: str, split=None) -> dict:
    return {
        "ok": True, "environment": {"hostname": "node0"},
        "run": {"kind": "split-test" if split else "gpu-single-vendor", "split_test": split,
                "world_size": world, "islands": []},
        "stages": [{"name": "device", "ok": True, "pci_bus_id": bus}],
    }


def test_summary_fails_on_missing_ranks_unless_partial(tmp_path):
    (tmp_path / "rank0.json").write_text(json.dumps(_fake_rank(0, 2, "0000:01:00.0")))
    assert summarize.main([str(tmp_path)]) == 1
    assert summarize.main(["--partial", str(tmp_path)]) == 0


def test_summary_flags_a_shared_gpu_outside_a_split_test(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(_fake_rank(rank, 2, "0000:01:00.0")))
    assert summarize.main([str(tmp_path)]) == 1
    for rank in (0, 1):
        (tmp_path / f"rank{rank}.json").write_text(
            json.dumps(_fake_rank(rank, 2, "0000:01:00.0", split="half")))
    assert summarize.main([str(tmp_path)]) == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_preflight_passes_on_loopback(tmp_path):
    port = str(_free_port())
    cmd = [sys.executable, str(KIT / "preflight.py"), "--master-addr", "127.0.0.1",
           "--port", port, "--timeout", "30", "--out", str(tmp_path)]
    master = subprocess.Popen([*cmd, "--role", "master"], env=clean_env(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    worker = subprocess.run([*cmd, "--role", "worker"], env=clean_env(),
                            capture_output=True, text=True, timeout=60)
    master.wait(timeout=60)
    assert worker.returncode == 0, worker.stderr
    assert master.returncode == 0
    record = json.loads((tmp_path / "preflight-worker.json").read_text())
    assert [s["name"] for s in record["stages"]] == ["resolve", "interface", "tcp", "gloo"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--master-addr", "127.0.0.1", "--port", "PORT", "--timeout", "2"],
         "could not reach 127.0.0.1"),
        (["--master-addr", "no-such-host.invalid", "--port", "1"], "cannot resolve"),
        (["--master-addr", "127.0.0.1", "--port", "1", "--ifname", "no-such-if0"],
         "does not exist here"),
    ],
)
def test_preflight_fails_fast_with_a_clear_message(args, message):
    args = [str(_free_port()) if a == "PORT" else a for a in args]
    result = subprocess.run([sys.executable, str(KIT / "preflight.py"), "--role", "worker",
                             *args], env=clean_env(), capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 1
    assert message in result.stderr


def test_probe_reports_a_missing_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(tmp_path)],
                            env=clean_env(), capture_output=True, text=True, timeout=120)
    record = json.loads((tmp_path / "probe.json").read_text())
    if record["environment"]["device_count"] == 0:
        assert result.returncode == 1
        assert record["stages"][0]["name"] == "gpus_visible"
        assert record["stages"][0]["ok"] is False


# ---- whole scripts, for real, in simulation mode -------------------------------------

def run_script(script: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=600,
                          env=clean_env(GPUBRIDGE_REPO=str(REPO), VENV=str(Path(sys.prefix)),
                                        PROJECT_DIR=FAKE_PROJECT, **env))


def steps(results: Path) -> dict[str, int]:
    lines = (results / "steps.jsonl").read_text().splitlines()
    return {json.loads(line)["step"]: json.loads(line)["exit_code"] for line in lines}


def test_explorer_split_job_runs_end_to_end_in_simulation(tmp_path):
    result = run_script(KIT / "explorer" / "04_split_4gpu.sbatch", GPUBRIDGE_VENDOR="nvidia",
                        RESULTS_ROOT=str(tmp_path), BENCH_MAX_BYTES="16K")
    assert result.returncode == 0, result.stderr[-3000:]
    out = tmp_path / "04_split_4gpu"
    assert set(steps(out).values()) == {0}
    for mode in ("half", "alternate"):
        summary = json.loads((out / f"check_{mode}" / "summary.json").read_text())
        assert summary["ok"] and summary["split_test"] == mode
    assert (out / "bench" / "bench.csv").exists()


def test_amd_script_keeps_going_after_failures_and_packages_results(tmp_path):
    # No GPU here: probe, device_map and rccl_version must fail, everything else must run.
    result = run_script(KIT / "amd" / "run_all.sh", GPUBRIDGE_VENDOR="amd", SKIP_SETUP="1",
                        GPU_COUNT="2", SCALING_MAX_GPUS="2", BENCH_MAX_BYTES="16K",
                        BENCH_TRIALS="3", RESULTS_ROOT=str(tmp_path), TARGET="sim")
    assert result.returncode == 1  # some steps failed, so the run reports failure
    codes = steps(tmp_path / "sim")
    assert {name for name, code in codes.items() if code} == {"probe", "device_map",
                                                               "rccl_version"}
    assert {"check_1gpu", "check_island", "check_split", "bench", "train_n2",
            "summarize"} <= {name for name, code in codes.items() if code == 0}
    assert (tmp_path / "sim.tgz").exists()
    assert "Download it before stopping this machine" in result.stderr


def test_check_and_bench_take_a_policy(tmp_path):
    check = torchrun(4, KIT / "check.py", "--out", str(tmp_path / "check"), "--policy",
                     "flat-gloo", GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert check.returncode == 0, check.stderr[-3000:]
    for record in read_ranks(tmp_path / "check"):
        assert record["ok"] and record["run"]["policy"] == "flat-gloo"
        dtypes = next(s for s in record["stages"] if s["name"] == "dtypes")
        assert [c["dtype"] for c in dtypes["cases"]] == [
            "float16", "bfloat16", "float32", "float64", "int8", "uint8", "int32", "int64"]
    summarize.main([str(tmp_path / "check")])
    assert "policy: `flat-gloo`" in (tmp_path / "check" / "summary.md").read_text()

    bench = torchrun(2, KIT / "bench_all_reduce.py", "--out", str(tmp_path / "bench"),
                     "--max-bytes", "4K", "--warmup", "1", "--trials", "2", "--policy",
                     "flat-gloo", GPUBRIDGE_VENDOR="amd")
    assert bench.returncode == 0, bench.stderr[-3000:]
    rows = json.loads((tmp_path / "bench" / "bench.json").read_text())["rows"]
    assert {r["policy"] for r in rows if r["op"] == "gpubridge"} == {"flat-gloo"}
