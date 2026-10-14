"""The GPU validation kit (scripts/gpu, examples/train_synthetic.py), tested without GPUs.

Shell scripts: shellcheck and --dry-run output. Python tools: run end to end
under torchrun on CPU in simulation mode, including split-test runs.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "scripts" / "gpu"
sys.path.insert(0, str(KIT))

import bench_all_reduce  # noqa: E402
import preflight  # noqa: E402
import summarize  # noqa: E402
import validation_tables  # noqa: E402

SHELL_SCRIPTS = sorted(
    [*KIT.glob("*.sh"), *KIT.glob("*/*.sh"), *KIT.glob("explorer/*.sbatch")]
)
#: Stand-ins for sbatch, srun, scontrol, module and uv (tests/test_gpu_jobs.py).
FAKE_CLUSTER_SCRIPTS = sorted((REPO / "tests" / "fake_cluster").iterdir())
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
    result = subprocess.run(["shellcheck", *map(str, SHELL_SCRIPTS + FAKE_CLUSTER_SCRIPTS)],
                            cwd=REPO,
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


@pytest.mark.parametrize(
    ("step", "env", "cap"),
    [
        ("01", {}, "480s (its Slurm limit, 00:10:00, less 120s), not 1200s"),
        ("02", {}, None),                                    # 25 minutes: 1200 s fits
        ("03", {}, "1080s (its Slurm limit, 00:20:00, less 120s), not 1200s"),
        ("05", {"STEP_TIMEOUT": "0", "TIME_05": "1-00"}, "86280s"),  # 0 = no limit: capped
        ("06", {"TIME_06": "01:30"}, "60s"),                 # never below a minute
        ("04", {"STEP_TIMEOUT": "400"}, None),               # already short enough
        ("04", {"TIME_04": "0"}, None),                      # Slurm's "no limit"
        ("04", {"TIME_04": "UNLIMITED"}, None),
        ("04", {"TIME_04": "infinite"}, None),
    ],
)
def test_each_jobs_step_timeout_fits_inside_its_slurm_limit(step, env, cap):
    result = dry_run(KIT / "explorer" / "submit.sh", step, **env)
    assert result.returncode == 0, result.stderr
    if cap is None:
        assert "STEP_TIMEOUT for this job" not in result.stderr
    else:
        assert f"STEP_TIMEOUT for this job: {cap}" in result.stderr
    # Only a cap at or below KIT_TIMEOUT (300 s) warns.
    assert ("is not above KIT_TIMEOUT" in result.stderr) is (cap == "60s")


def test_job_events_are_redacted_and_record_the_exit_code(tmp_path):
    # job.jsonl, as job_setup starts it: a die message loses the project, home and user,
    # and a command failing under set -e still leaves its exit code.
    script = f"""
source {KIT}/common.sh
RESULTS={tmp_path}
PROJECT_DIR=/projects/example-lab USER=someuser HOME=/home/someuser
job_started
(die "no venv at /projects/example-lab/someuser/venv; see /home/someuser/x")
false
"""
    result = subprocess.run(["bash", "-e", "-c", script], env=clean_env(), capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 1
    lines = (tmp_path / "job.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    # A subshell has no EXIT trap, so only the script's own exit (false's) is recorded.
    assert [e["event"] for e in events] == ["started", "error", "exit"]
    assert events[1]["message"] == "no venv at $PROJECT_DIR/$USER/venv; see ~/x"
    assert events[2]["code"] == 1
    assert "someuser" not in "".join(lines) and "example-lab" not in "".join(lines)


def test_submit_rejects_a_time_limit_it_cant_read():
    result = dry_run(KIT / "explorer" / "submit.sh", "01", TIME_01="ten minutes")
    assert result.returncode == 1 and "can't read the time limit 'ten minutes'" in result.stderr


@pytest.mark.parametrize(("env", "python"), [({}, "3.12"),
                                             ({"PYTHON_VERSION": "3.12.15"}, "3.12"),
                                             ({"VENV_PYTHON": "3.11"}, "3.11")])
def test_setup_venv_python_ignores_python_version(env, python, tmp_path):
    # Python's Docker images set PYTHON_VERSION to their own full version.
    result = dry_run(KIT / "setup_env.sh", "--flavor", "cuda", "--venv", str(tmp_path / "v"),
                     **env)
    assert result.returncode == 0, result.stderr
    assert f"uv venv --python {python} " in result.stdout


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


@pytest.mark.parametrize("path", ["scripts/gpu/explorer/local.env",
                                  "results/explorer/01_probe/probe.log",
                                  "results/explorer/02_split_shared_gpu.tgz"])
def test_local_settings_and_raw_explorer_results_are_gitignored(path):
    # Raw results hold unredacted logs: a `git add -A` on the cluster must not take them.
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    assert result.returncode == 0


def test_bench_compression_fp_fusion_switch_without_triton_kernels(tmp_path):
    # On CPU the benchmark times the PyTorch kernels, which have no such option.
    result = subprocess.run(
        [sys.executable, str(KIT / "bench_compression.py"), "--out", str(tmp_path),
         "--min-bytes", "4K", "--max-bytes", "4K", "--modes", "fp8", "--dtypes", "float32",
         "--warmup", "0", "--trials", "1", "--fp-fusion"],
        env=clean_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--fp-fusion: these are the PyTorch kernels" in result.stderr
    record = json.loads((tmp_path / "compression.json").read_text())
    assert record["fp_fusion"] is None and record["impl"] == "torch (CPU)"


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


@pytest.mark.parametrize("step", ["setup-rocm", "amd", "07"])
def test_explorer_mixed_path_is_dry_run_only_until_amd_gpus_are_confirmed(step, tmp_path):
    path = fake_command(tmp_path / "bin", "sbatch", "echo SUBMITTED\n")
    result = subprocess.run(["bash", str(KIT / "explorer" / "submit.sh"), step],
                            env=clean_env(GPUBRIDGE_REPO=str(REPO), PATH=path,
                                          GPU_TYPE_AMD="mi100", WORK_DIR=str(tmp_path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "dry-run only until step 00" in result.stderr
    assert "SUBMITTED" not in result.stdout


def test_submit_amd_is_one_job_on_an_amd_node():
    out = dry_run(KIT / "explorer" / "submit.sh", "amd", GPU_TYPE_AMD="mi100").stdout
    sbatch = [line for line in out.splitlines() if line.startswith("+ sbatch ")]
    assert len(sbatch) == 1, out
    assert "--partition=sharing " in sbatch[0] and "--gres=gpu:mi100:1 " in sbatch[0]
    assert "--time=00:30:00" in sbatch[0] and sbatch[0].rstrip().endswith("amd_node.sbatch")


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


def test_mixed_hetjob_skips_hsdp_unless_both_sides_have_as_many_gpus():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w",
                     GPUS_MIXED_NVIDIA="2", GPUS_MIXED_AMD="1")
    assert result.returncode == 0, result.stderr
    assert "--hsdp" not in result.stdout
    assert "skipping the HSDP steps: they need as many NVIDIA GPUs as AMD (2 vs 1)" in result.stderr


def test_mixed_hetjob_runs_each_side_in_its_own_venv():
    result = dry_run(KIT / "explorer" / "07_mixed_hetjob.sbatch", WORK_DIR="/w")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    launches = [line for line in out.splitlines() if "srun --het-group=0" in line]
    # Every launch runs under coreutils timeout (STEP_TIMEOUT, common.sh).
    assert all(line.startswith("+ timeout --kill-after=30 1200 srun ") for line in launches)
    sruns = [line[line.index("srun "):] for line in launches]
    # probe, preflight, check, bench, a check per opt-in policy, the candidates bench,
    # DDP training, then FSDP2 on the torch.distributed backend (trained, then verified),
    # then HSDP the same way (one GPU per side by default: as many on each side)
    assert len(sruns) >= 11
    assert any("--policy pipelined-reduce-bridge-broadcast" in line for line in sruns[4:])
    assert any("--policy sharded-bridge" in line for line in sruns[4:])
    assert "--write-thresholds" in sruns[-6]
    assert "examples/train_synthetic.py" in sruns[-5] and "train_ddp.json" in sruns[-5]
    assert "examples/fsdp2_torch_backend.py" in sruns[-4] and "--verify" in sruns[-3]
    assert "--hsdp 2 --json" in sruns[-2] and "--verify --hsdp 2" in sruns[-1]
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
        ("02_split_shared_gpu.sbatch", ["--nproc-per-node=2", "check.py",
                                        "GPUBRIDGE_SPLIT_TEST=alternate", "--nproc-per-node=4",
                                        "--expect-init-error would\\ share\\ a\\ GPU",
                                        # its own limit, not STEP_TIMEOUT
                                        "timeout --kill-after=30 60 env "
                                        "GPUBRIDGE_SPLIT_TEST=alternate",
                                        # the slowest tests end gpu_tests' log
                                        "pytest -m gpu", "--durations=15"]),
        ("03_single_island.sbatch", ["device_map.py", "check.py", "bench_all_reduce.py"]),
        ("04_split_4gpu.sbatch", ["check_half", "check_alternate", "bench_all_reduce.py",
                                  "--durations=15",
                                  "--policy pipelined-reduce-bridge-broadcast",
                                  "--policy sharded-bridge", "--write-thresholds",
                                  "--policy auto-tuned", "train_synthetic.py",
                                  "train_ddp.json", "[check_pinned]",
                                  "env GPUBRIDGE_PINNED_STAGING=1 torchrun",
                                  "bench_pinned --max-bytes"]),
        ("05_multinode.sbatch", ["--policy pipelined-reduce-bridge-broadcast",
                                 "--policy sharded-bridge", "bench_phases_sharded-bridge",
                                 "--write-thresholds"]),
        ("05_multinode.sbatch", ["srun", "--rdzv-backend=c10d", "check_split_by_node",
                                 "--kill-on-bad-exit=1 bash KIT/node_ifname.sh torchrun"]),
        ("05_multinode.sbatch", ["preflight.py --role auto", "--bandwidth-seconds 2"]),
        ("05_multinode.sbatch", ["[check_pinned]", "[bench_pinned]", "check-pinned"]),
        ("05_multinode.sbatch", ["UCX_BRIDGE=1", "[ucx_build]", "python -m gpubridge.ucx build",
                                 "--bridge ucx", "[bench_ucx]"]),
        ("07_mixed_hetjob.sbatch", ["[check_pinned]", "[bench_pinned]"]),
        ("06_scaling.sbatch", ["--nproc-per-node=4", "train_synthetic.py", "summarize.py"]),
        ("07_mixed_hetjob.sbatch", ["--het-group=1", "--rdzv-backend=c10d", "summarize.py"]),
    ],
)
def test_explorer_jobs_dry_run(job, expected):
    env = {"UCX_BRIDGE": "1"} if "UCX_BRIDGE=1" in expected else {}
    expected = [text for text in expected if text != "UCX_BRIDGE=1"]
    result = dry_run(KIT / "explorer" / job, **env)
    assert result.returncode == 0, result.stderr
    for text in expected:
        text = text.replace("KIT/", f"{KIT}/")
        assert text in result.stdout, (text, result.stdout)


def test_amd_dry_run_with_several_gpus():
    out = dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="8").stdout
    # common.sh exports NCCL_DEBUG=INFO; RCCL's version comes from the step's NCCL logs.
    for text in ["rocm-smi", "probe.py", "--nproc-per-node=9", "LOCAL_RANK=8\\ asks",
                 "nccl/check_1gpu/", "--nproc-per-node=8",
                 "GPUBRIDGE_SPLIT_TEST=half", "bench_all_reduce.py", "train_n4", "tar -czf",
                 "--policy pipelined-reduce-bridge-broadcast", "--policy sharded-bridge",
                 "train_ddp_split.json", "GPUBRIDGE_PINNED_STAGING=1", "bench_pinned",
                 "--durations=15"]:
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
            "reduce_scatter", "async", "reduction_agreement", "in_flight", "barrier",
            "destroy"]
        agreement = next(s for s in record["stages"] if s["name"] == "reduction_agreement")
        assert agreement["all_product_agree"] is True  # Gloo against Gloo on CPU
        assert agreement["backend"] == "gloo"
    # Islands {0, 1} and {2, 3}; rank 3 is late, so its leader waits in island-reduce.
    phases = {}
    for record in records:
        stage = next(s for s in record["stages"] if s["name"] == "in_flight")
        assert stage["late_rank"] == 3
        flights = stage["in_flight"]
        phases[record["run"]["rank"]] = flights[0]["phase"] if flights else None
        assert len({f["seq"] for f in flights}) <= 1
    assert phases == {0: "bridge", 1: "island-broadcast", 2: "island-reduce", 3: None}
    assert summarize.main([str(out)]) == 0
    assert "SPLIT TEST - not a mixed-vendor result" in (out / "summary.md").read_text()
    assert json.loads((out / "summary.json").read_text())["split_test"] == "half"


@pytest.mark.parametrize("delayed", [0, 2])  # island A's leader, island B's leader
def test_in_flight_stage_does_not_need_ranks_to_arrive_together(delayed, tmp_path):
    # One rank reaches the in_flight stage 1.5 s after the others, as after a slow
    # previous stage on a loaded machine. Every rank must still record its view
    # while the late rank is missing.
    wrapper = tmp_path / "check_skewed.py"
    wrapper.write_text(f"""import sys, time
sys.path.insert(0, {str(KIT)!r})
sys.argv[0] = {str(KIT / "check.py")!r}
import check
stage = check.in_flight_while_late

def skewed(rank, *args, **kwargs):
    if rank == {delayed}:
        time.sleep(1.5)
    return stage(rank, *args, **kwargs)

check.in_flight_while_late = skewed
sys.exit(check.main())
""")
    out = tmp_path / "check"
    result = torchrun(4, wrapper, "--out", str(out), GPUBRIDGE_VENDOR="nvidia",
                      GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    phases = {}
    for record in read_ranks(out):
        flights = next(s for s in record["stages"] if s["name"] == "in_flight")["in_flight"]
        phases[record["run"]["rank"]] = flights[0]["phase"] if flights else None
    assert phases == {0: "bridge", 1: "island-broadcast", 2: "island-reduce", 3: None}


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
    for rank in range(2):  # in the log too, where negative_step looks (common.sh)
        assert f"rank {rank}: init() failed: " in result.stderr
    assert result.stderr.count("doesn't apply to this cluster") >= 2
    assert summarize.main([str(out)]) == (0 if ok else 1)


def test_check_expecting_an_init_error_fails_when_init_succeeds(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), "--expect-init-error", "LOCAL_RANK",
                      GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 1
    for record in read_ranks(out):
        assert "init() succeeded" in record["stages"][0]["error"]
    assert "rank 1: init() succeeded; this run expected it to fail" in result.stderr
    assert "LOCAL_RANK" not in result.stderr  # nothing a negative_step could mistake for it


def test_check_without_split_is_not_flagged(tmp_path):
    out = tmp_path / "check"
    result = torchrun(2, KIT / "check.py", "--out", str(out), GPUBRIDGE_VENDOR="amd")
    assert result.returncode == 0, result.stderr[-3000:]
    for record in read_ranks(out):
        assert record["run"]["split_test"] is None
        assert "warning" not in record["run"]
    # One island: rank 0 waits inside the all_reduce, the late rank 1 shows nothing.
    flights = [next(s for s in record["stages"] if s["name"] == "in_flight")["in_flight"]
               for record in read_ranks(out)]
    assert [f["op"] for f in flights[0]] == ["all_reduce"] and flights[1] == []
    summarize.main([str(out)])
    assert "SPLIT TEST" not in (out / "summary.md").read_text()


def test_bench_measures_native_and_bridged_in_one_split_run(tmp_path):
    out = tmp_path / "bench"
    result = torchrun(4, KIT / "bench_all_reduce.py", "--out", str(out), "--max-bytes", "16K",
                      "--warmup", "1", "--trials", "3",
                      GPUBRIDGE_VENDOR="nvidia", GPUBRIDGE_SPLIT_TEST="half")
    assert result.returncode == 0, result.stderr[-3000:]
    rows = json.loads((out / "bench.json").read_text())["rows"]
    # By default: the native baseline, gpubridge and flat-gloo, side by side.
    assert {r["op"] for r in rows} == {"native", "gpubridge", "flat-gloo"}
    assert {r["policy"] for r in rows if r["op"] == "gpubridge"} == {"reduce-bridge-broadcast"}
    assert {r["policy"] for r in rows if r["op"] == "flat-gloo"} == {"flat-gloo"}
    assert {r["bytes"] for r in rows} == {1024, 4096, 16384}
    assert all(r["split_test"] == "half" and r["islands"] == 2 for r in rows)
    assert all(r["timer"] == "host" and r["host_median_us"] > 0 for r in rows)  # CPU
    assert all(r["baseline"] == "native" for r in rows)
    assert (out / "bench.csv").read_text().startswith("op,policy,baseline,run_kind,split_test,")
    summary = (out / "summary.md").read_text()
    assert "SPLIT TEST" in summary and "timed with the host timer" in summary
    assert "| gpubridge / native | flat-gloo / native |" in summary
    assert "fewer ranks" not in summary


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
    assert {r["baseline"] for r in rows} == {"island-native (fewer ranks, best case)"}
    summary = (out / "summary.md").read_text()
    assert "island:amd median (us)" in summary and "island:nvidia median (us)" in summary
    assert "| gpubridge / island-native (fewer ranks, best case) |" in summary
    assert "gpubridge / native" not in summary
    assert "not a like-for-like comparison" in summary
    assert "flat-gloo" not in summary  # only what --ops asked for


@pytest.mark.parametrize(("ops", "baseline"), [
    (["native", "gpubridge", "flat-gloo"], "native"),
    (["island", "gpubridge", "flat-gloo"], "island-native (fewer ranks, best case)"),  # fallback
    (["gpubridge"], "none"),
])
def test_every_bench_row_names_its_baseline(ops, baseline):
    assert bench_all_reduce.baseline_of(ops) == baseline


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
    timed = bench_all_reduce.row("native", info, 2**20, "float32", [0.001, 0.002, 0.003],
                                 host=[0.004, 0.005, 0.006], timer="cuda-events")
    assert timed["timer"] == "cuda-events" and timed["median_us"] == 2000.0
    assert timed["host_median_us"] == 5000.0 and timed["policy"] == ""
    assert bench_all_reduce.row("flat-gloo", info, 2**20, "float32", [0.001])["policy"] == (
        "flat-gloo")


def test_bench_times_gpu_trials_with_cuda_events(tmp_path, monkeypatch):
    # The GPU branch of time_op, on one CPU rank with fake events: the timed
    # latency is the events' elapsed time, the host one the wall clock.
    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self, stream=None):
            pass

        def elapsed_time(self, end):
            return 2.5  # ms

    import torch.distributed as dist
    monkeypatch.setattr(bench_all_reduce.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(bench_all_reduce.torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(bench_all_reduce, "sync", lambda device: None)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        calls = []
        timed, host = bench_all_reduce.time_op(calls.append, "tensor", torch.device("cuda", 0),
                                               warmup=2, trials=3, max_seconds=60)
    finally:
        dist.destroy_process_group()
    assert len(calls) == 5  # 2 warmup, 3 timed
    assert timed == [[0.0025]] * 3
    assert len(host) == 3 and all(0 < t[0] < 1 for t in host)


def _bench_topology(peers):
    return SimpleNamespace(peers=[SimpleNamespace(vendor=v, hostname=h, gpu=g) for v, h, g in peers])


def test_bench_native_baseline_needs_one_vendor_and_a_gpu_per_rank():
    two_gpus = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n1", "pci 0000:02:00")]
    assert bench_all_reduce.native_possible(_bench_topology(two_gpus)) is None
    shared = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n1", "pci 0000:01:00")]
    assert "share a GPU" in bench_all_reduce.native_possible(_bench_topology(shared))
    other_host = [("nvidia", "n1", "pci 0000:01:00"), ("nvidia", "n2", "pci 0000:01:00")]
    assert bench_all_reduce.native_possible(_bench_topology(other_host)) is None
    mixed = [("nvidia", "n1", "pci 0000:01:00"), ("amd", "n2", "pci 0000:c1:00")]
    assert "different real vendors" in bench_all_reduce.native_possible(_bench_topology(mixed))
    cpu = [("nvidia", "n1", ""), ("nvidia", "n1", "")]
    assert bench_all_reduce.native_possible(_bench_topology(cpu)) is None


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
    assert "ddp_comm_hook" in data["grad_sync"] and data["policy"] == "reduce-bridge-broadcast"
    # Every rank's versions and settings, as the kit records them.
    assert len({env["pid"] for env in data["environments"]}) == 2
    assert data["environment"] == data["environments"][0]
    for env in data["environments"]:
        assert env["torch"] == torch.__version__ and env["git_commit"]
        assert env["env"]["GPUBRIDGE_VENDOR"] == "nvidia"
    assert data["timeout_s"] > 0 and data["warmup_steps"] == 5


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


def test_environment_names_versions_driver_and_interfaces(monkeypatch):
    import gpukit

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ib0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    env = gpukit.environment()
    assert env["interfaces"] == {"GLOO_SOCKET_IFNAME": "ib0", "NCCL_SOCKET_IFNAME": "auto"}
    assert env["git_commit"] and env["git_dirty"] in (True, False)
    for key in ("torch", "torch_cuda", "torch_hip", "rocm_version", "collective_library",
                "nccl_version", "driver_version", "devices", "hostname"):
        assert key in env
    if not torch.version.cuda and not torch.version.hip:  # a CPU build of PyTorch
        assert env["collective_library"] is None and env["driver_version"] is None


#: The arch list of PyTorch's cu126 wheels.
CU126 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def fake_device(monkeypatch, *, name: str, capability: tuple[int, int], arch_list: list[str],
                gcn: str | None = None, hip: str | None = None) -> dict:
    """gpukit.device_info for one made-up GPU. CUDA builds of PyTorch fill gcnArchName
    with the product name, so that is the default."""
    import gpukit

    props = SimpleNamespace(name=name, major=capability[0], minor=capability[1],
                            gcnArchName=name if gcn is None else gcn, total_memory=32 * 2**30)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))
    return gpukit.device_info(0)


@pytest.mark.parametrize(
    ("gpu", "target", "match"),
    [
        # A CUDA device whose gcnArchName is its product name: the target comes from
        # the compute capability instead.
        ({"name": "NVIDIA A100-SXM4-40GB", "capability": (8, 0), "arch_list": CU126},
         "sm_80", "exact"),
        # A V100 on the cu126 build: its gcnArchName, "Tesla V100-SXM2-32GB", is not
        # its target.
        ({"name": "Tesla V100-SXM2-32GB", "capability": (7, 0), "arch_list": CU126},
         "sm_70", "exact"),
        # ROCm: gcnArchName without its feature flags.
        ({"name": "AMD Instinct MI50", "capability": (9, 0), "gcn": "gfx906:sramecc+:xnack-",
          "hip": "6.4.43482", "arch_list": ["gfx900", "gfx906", "gfx908", "gfx90a"]},
         "gfx906", "exact"),
        # Only PTX for an earlier capability: the driver compiles it at load time.
        ({"name": "NVIDIA H100 80GB HBM3", "capability": (9, 0),
          "arch_list": ["sm_70", "sm_80", "compute_80"]}, "sm_90", "ptx"),
        # A cubin for an earlier minor of the same major runs unchanged.
        ({"name": "NVIDIA L40S", "capability": (8, 9), "arch_list": CU126},
         "sm_89", "compatible"),
        # A genuine miss: no sm_12x cubin and no PTX, so the probe must still fail.
        ({"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "capability": (12, 0),
          "arch_list": CU126}, "sm_120", None),
    ],
)
def test_device_target_and_the_rule_that_covers_it(monkeypatch, gpu, target, match):
    info = fake_device(monkeypatch, **gpu)
    assert (info["target"], info["target_match"]) == (target, match)
    assert info["target_in_build"] is (match is not None)


@pytest.mark.parametrize(
    ("target", "capability", "arch_list", "match"),
    [
        ("sm_80", (8, 0), ["sm_86"], None),            # a later minor's cubin doesn't run
        ("sm_80", (8, 0), ["sm_70"], None),            # nor an earlier major's
        ("sm_80", (8, 0), ["compute_90"], None),       # PTX above the capability doesn't
        ("sm_90", (9, 0), ["sm_90a", "sm_100f"], None),  # arch- and family-specific cubins
        ("sm_75", (7, 5), ["sm_70", "compute_70"], "compatible"),  # the cubin is preferred
        ("sm_120", (12, 0), ["compute_90"], "ptx"),
        ("gfx90a", None, ["gfx90a:xnack-"], "exact"),  # ROCm list entries with flags
        ("gfx942", None, ["gfx90a", "sm_90"], None),
        ("sm_70", (7, 0), [], None),
    ],
)
def test_target_match_rules(target, capability, arch_list, match):
    import gpukit

    assert gpukit.target_match(target, capability, arch_list) == match


def run_probe(monkeypatch, tmp_path, devices: list, *, nccl: bool = True) -> int:
    """probe.py's main() on a made-up environment."""
    import gpukit
    import probe

    env = {"device_count": len(devices), "devices": devices, "arch_list": CU126,
           "probe": {"nccl_available": nccl, "gloo_available": True}, "torch": "2.14.1+cu126",
           "torch_cuda": "12.6", "torch_hip": None, "nccl_version": "2.21.5"}
    monkeypatch.setattr(gpukit, "environment", lambda: env)
    monkeypatch.setattr(sys, "argv", ["probe.py", "--out", str(tmp_path)])
    return probe.main()


def test_probe_passes_a_gpu_its_build_covers(monkeypatch, tmp_path, capsys):
    v100 = fake_device(monkeypatch, name="Tesla V100-SXM2-32GB", capability=(7, 0),
                       arch_list=CU126)
    assert run_probe(monkeypatch, tmp_path, [v100]) == 0
    out = capsys.readouterr()
    assert "GPUs: Tesla V100-SXM2-32GB (sm_70, exact)" in out.out and "FAIL" not in out.err
    record = json.loads((tmp_path / "probe.json").read_text())
    assert record["ok"] and record["environment"]["devices"][0]["target_match"] == "exact"


def test_probe_names_each_failed_check_and_its_values(monkeypatch, tmp_path, capsys):
    blackwell = fake_device(monkeypatch, name="NVIDIA RTX PRO 6000", capability=(12, 0),
                            arch_list=CU126)
    unreadable = "error: RuntimeError: CUDA error: unknown error"
    assert run_probe(monkeypatch, tmp_path, [blackwell, unreadable], nccl=False) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"FAIL device_info: device 1: {unreadable}",
        f"FAIL target_in_build: device 0 target sm_120 not in arch_list {CU126} (nor an sm_12x "
        "cubin at or below 12.0, nor PTX at or below 12.0)",
        "FAIL backends: nccl_available False, gloo_available True",
    ]


def test_probe_without_a_gpu_says_so(monkeypatch, tmp_path, capsys):
    assert run_probe(monkeypatch, tmp_path, []) == 1
    assert capsys.readouterr().err.splitlines() == [
        "FAIL gpus_visible: device_count 0: no GPU visible to PyTorch (check the job's GPU "
        "request and driver)"]


#: Must never reach a record: credentials, who ran it, where the project lives.
SECRETS = {
    "GPUBRIDGE_API_TOKEN": "tok-0a1b2c3d", "NCCL_SECRET_KEY": "key-4e5f6a7b",
    "GLOO_PASSWORD": "pw-8c9d0e1f", "AWS_SECRET_ACCESS_KEY": "aws-2a3b4c5d",
    "SLURM_JOB_USER": "someuser", "SLURM_JOB_ACCOUNT": "example-lab-account",
    "TORCH_INDEX_URL": "https://bob:hunter2@pypi.example.org/simple",
    "GPUBRIDGE_REPO_DIR": f"{FAKE_PROJECT}/gpubridge",
}


def test_records_hold_only_allowlisted_env_vars_and_no_secrets(tmp_path):
    env = {**SECRETS, "GPUBRIDGE_VENDOR": "nvidia", "SLURM_JOB_ID": "4242"}
    probe = subprocess.run([sys.executable, str(KIT / "probe.py"), "--out", str(tmp_path)],
                           env=clean_env(**env), capture_output=True, text=True, timeout=120)
    train = torchrun(2, REPO / "examples" / "train_synthetic.py", "--steps", "2",
                     "--warmup-steps", "1", "--dim", "16", "--batch-size", "4",
                     "--json", str(tmp_path / "train.json"), **env)
    assert train.returncode == 0, train.stderr[-3000:]
    for name in ("probe.json", "train.json"):
        text = (tmp_path / name).read_text()
        for secret in ("tok-0a1b2c3d", "key-4e5f6a7b", "pw-8c9d0e1f", "aws-2a3b4c5d",
                       "someuser", "example-lab", "hunter2", FAKE_PROJECT, str(Path.home())):
            assert secret not in text, (name, secret, probe.stderr[-2000:])
        for env_record in [json.loads(text)["environment"]]:
            recorded = env_record["env"]
            assert recorded["GPUBRIDGE_VENDOR"] == "nvidia" and recorded["SLURM_JOB_ID"] == "4242"
            assert recorded["TORCH_INDEX_URL"] == "https://***@pypi.example.org/simple"
            assert recorded["GPUBRIDGE_REPO_DIR"] == "$PROJECT_DIR/gpubridge"
            assert not set(recorded) & {"GPUBRIDGE_API_TOKEN", "NCCL_SECRET_KEY",
                                        "GLOO_PASSWORD", "AWS_SECRET_ACCESS_KEY",
                                        "SLURM_JOB_USER", "SLURM_JOB_ACCOUNT", "HOME", "USER",
                                        "PATH"}


def test_redact_replaces_paths_users_and_url_credentials(monkeypatch):
    import gpukit

    monkeypatch.setenv("HOME", "/home/someuser")
    monkeypatch.setattr(gpukit, "_user", lambda: "someuser")
    assert gpukit.redact("/projects/example-lab/gpubridge/results/x.json") == (
        "$PROJECT_DIR/gpubridge/results/x.json")
    assert gpukit.redact("/home/someuser/venv/bin/python") == "~/venv/bin/python"
    assert gpukit.redact("/home/someusers/x") == "/home/someusers/x"
    assert gpukit.redact("/scratch/someuser/out /scratch/someusers") == (
        "/scratch/$USER/out /scratch/someusers")
    assert gpukit.redact("git+https://u:t@host/r@abc") == "git+https://***@host/r@abc"
    assert gpukit.scrub({"argv": ["--out", "/projects/x/r"], "n": 1}) == {
        "argv": ["--out", "$PROJECT_DIR/r"], "n": 1}


# ---- validation_tables.py ------------------------------------------------------------

def test_predicted_bridge_time_is_a_round_trip_plus_the_ring_traffic():
    # Two islands: the whole tensor each way. 100 MB at 100 MB/s is a second.
    assert validation_tables.predicted_bridge_ms(10**8, 2, 100.0, 0.5) == pytest.approx(1000.5)
    # Four islands send 2(p-1)/p = 1.5 times the tensor, and wait p-1 = 3 round trips.
    assert validation_tables.predicted_bridge_ms(10**8, 4, 100.0, 0.0) == pytest.approx(1500)
    assert validation_tables.predicted_bridge_ms(10**8, 4, 100.0, 0.5) == pytest.approx(1501.5)


def test_verdict_counts_only_gpu_runs():
    evidence = validation_tables.Evidence
    assert validation_tables.verdict([]) == "not run"
    assert validation_tables.verdict([evidence("a", True, True)]) == "simulation only"
    assert validation_tables.verdict(
        [evidence("a", True, False), evidence("b", True, False), evidence("c", False, True)]
    ) == "pass (2)"
    assert validation_tables.verdict(
        [evidence("a", True, False), evidence("b", False, False)]) == "FAIL (1 of 2)"


def test_tables_cover_every_hardware_validation_item():
    text = (REPO / "HARDWARE_VALIDATION.md").read_text().split("## Mixed build results")[0]
    numbers = {int(m) for m in re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)}
    assert numbers == set(validation_tables.ITEMS)


def test_every_step_the_tables_look_for_is_one_a_job_runs():
    runs = [dry_run(KIT / "explorer" / job, UCX_BRIDGE="1") for job in sorted(
        p.name for p in (KIT / "explorer").glob("0*.sbatch"))]
    runs += [dry_run(KIT / "amd" / "run_all.sh", GPU_COUNT="4"),
             dry_run(KIT / "mixed" / "node.sh", ROLE="master", MASTER_ADDR="10.0.0.1",
                     GPU_COUNT="2")]
    names = set(re.findall(r"^\+ \[([^\]]+)\]", "".join(r.stdout for r in runs),
                           flags=re.MULTILINE))
    for _, tests in validation_tables.ITEMS.values():
        for test in tests:
            if isinstance(test, validation_tables.Step):
                assert any(re.fullmatch(test.pattern, n) for n in names), test.pattern


GPU_ENV = {"hostname": "gpu-a", "devices": [{"name": "NVIDIA H200"}], "driver_version": "570.86",
           "torch": "2.8.0+cu128", "torch_cuda": "12.8", "torch_hip": None,
           "collective_library": "nccl", "nccl_version": "2.26.2", "git_commit": "abc1234",
           "git_dirty": False, "interfaces": {"GLOO_SOCKET_IFNAME": "ib0"}, "env": {}}


def _job(root: Path, name: str, steps: list[tuple[str, int]], env: dict, device: str) -> Path:
    """A job's results as the kit leaves them: steps.jsonl and the steps' records."""
    job = root / name
    job.mkdir(parents=True)
    (job / "steps.jsonl").write_text("".join(
        json.dumps({"step": step, "exit_code": code,
                    **({"result": "expected failure confirmed"} if step == "shared_gpu_check"
                       else {})}) + "\n" for step, code in steps))
    for rank, host in enumerate(("gpu-a", "gpu-b")):
        (job / "check_split_by_node").mkdir(exist_ok=True)
        (job / "check_split_by_node" / f"rank{rank}.json").write_text(json.dumps({
            "script": "scripts/gpu/check.py", "environment": {**env, "hostname": host},
            "run": {"kind": "split-test", "device": device},
            "stages": [{"name": "init", "ok": True}, {"name": "barrier", "ok": True}]}))
    for role, host, ifname, out, back in (("master", "gpu-a", "ib0", 1000.0, 800.0),
                                          ("worker", "gpu-b", "ib1", 800.0, 1000.0)):
        (job / "preflight").mkdir(exist_ok=True)
        (job / "preflight" / f"preflight-{role}.json").write_text(json.dumps({
            "role": role, "hostname": host, "chosen_ifname": ifname, "environment": env,
            "bandwidth": {"MBps_out": out, "MBps_in": back, "rtt_ms": 0.5}}))
    rows = []
    for size in (2**20, 64 * 2**20, 256 * 2**20):
        ms = size / 2**20  # 1 ms per MiB, so ratios are easy to check
        rows += [{"op": "native", "bytes": size, "median_us": ms * 100, "islands": 2},
                 {"op": "gpubridge", "bytes": size, "median_us": ms * 1500, "islands": 2,
                  "policy": "reduce-bridge-broadcast", "phases_ms": {"bridge": ms}},
                 {"op": "flat-gloo", "bytes": size, "median_us": ms * 3000, "islands": 2}]
    (job / "bench_phases_x").mkdir()
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps({
        "environment": env, "run": {"kind": "split-test", "device": device},
        "baseline": "native", "rows": rows}))
    return job


def test_tables_from_a_results_tree(tmp_path):
    passed = [("preflight", 0), ("check_split_by_node", 0), ("bench_phases_x", 0),
              ("shared_gpu_check", 0)]
    _job(tmp_path, "explorer/05_multinode", passed, GPU_ENV, "cuda:0")
    # A simulation run whose steps all failed must not change any verdict.
    _job(tmp_path, "sim/05_multinode", [(name, 1) for name, _ in passed],
         {**GPU_ENV, "env": {"GPUBRIDGE_VENDOR": "nvidia"}}, "cpu")
    assert validation_tables.main([str(tmp_path)]) == 0
    tables = json.loads((tmp_path / "validation_tables.json").read_text())
    md = (tmp_path / "validation_tables.md").read_text()

    real, sim = tables["runs"]
    assert not real["simulated"] and sim["simulated"]
    assert real["hosts"] == "host 1, host 2" and real["gpus"] == "NVIDIA H200"
    assert (real["driver"], real["toolkit"], real["collectives"], real["gpubridge"],
            real["gloo_ifname"]) == ("570.86", "CUDA 12.8", "NCCL 2.26.2", "abc1234", "ib0")
    assert real["negative_steps"] == {"shared_gpu_check": "expected failure confirmed"}

    verdicts = {row["item"]: row["verdict"] for row in tables["items"]}
    assert verdicts[8] == "pass (2)"  # the preflight and check_split_by_node
    assert verdicts[10] == "pass (1)" and verdicts[24] == "pass (1)"
    assert verdicts[1] == "not run"  # needs a real mixed run, not a split test
    assert verdicts[22] == "not run"

    network = tables["network"][0]
    assert (network["master"], network["worker"]) == ("host 1 ib0", "host 2 ib1")
    assert (network["MBps_master_to_worker"], network["MBps_worker_to_master"]) == (1000, 800)

    bench = [r for r in tables["benchmarks"] if not r["simulated"]]
    assert [r["bytes"] for r in bench] == [2**20, 64 * 2**20, 256 * 2**20]
    assert {(r["baseline"], r["gpubridge_over_baseline"], r["flat-gloo_over_baseline"])
            for r in bench} == {("native", 15.0, 30.0)}
    assert {(r["bridge"], r["staging"]) for r in tables["benchmarks"]} == {("gloo", "pageable")}

    predicted = next(r for r in tables["predicted"]
                     if r["bytes"] == 64 * 2**20 and not r["simulated"])
    assert predicted["MBps"] == 800  # the slower direction
    assert predicted["predicted_ms"] == pytest.approx(0.5 + 64 * 2**20 / 800e6 * 1e3, abs=1e-3)
    assert predicted["measured_ms"] == 64

    assert "never count towards an item" in md
    assert "| sim/05_multinode (simulation) |" in md
    assert "explorer/05_multinode/shared_gpu_check: expected failure confirmed" in md
    assert "## Predicted vs measured bridge time" in md
    # Safe to publish: no hostname, no path.
    out = md + (tmp_path / "validation_tables.json").read_text()
    assert "gpu-a" not in out and "gpu-b" not in out and str(tmp_path) not in out


def test_tables_name_the_island_baseline(tmp_path):
    job = _job(tmp_path, "amd/target", [("bench", 0)], GPU_ENV, "cuda:0")
    bench = json.loads((job / "bench_phases_x" / "bench.json").read_text())
    del bench["baseline"]  # a bench.json from before it named its baseline
    bench["rows"] = [{**r, "op": "island:nvidia"} if r["op"] == "native" else r
                     for r in bench["rows"]]
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps(bench))
    rows = validation_tables.build([tmp_path])["benchmarks"]
    assert {r["baseline"] for r in rows} == {"island-native (fewer ranks, best case)"}


def test_tables_need_job_results(tmp_path, capsys):
    assert validation_tables.main([str(tmp_path)]) == 1
    assert "no job results" in capsys.readouterr().err


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_preflight_pair(tmp_path: Path, *extra: str, env=None) -> dict[str, dict]:
    """Both sides of a preflight on this machine; their records by role."""
    cmd = [sys.executable, str(KIT / "preflight.py"), "--master-addr", "127.0.0.1",
           "--port", str(_free_port()), "--timeout", "30", "--bandwidth-seconds", "0.3",
           "--out", str(tmp_path), *extra]
    master = subprocess.Popen([*cmd, "--role", "master"], env=env or clean_env(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    worker = subprocess.run([*cmd, "--role", "worker"], env=env or clean_env(),
                            capture_output=True, text=True, timeout=120)
    master_err = master.communicate(timeout=120)[1]
    assert worker.returncode == 0, worker.stderr
    assert master.returncode == 0, master_err
    return {role: json.loads((tmp_path / f"preflight-{role}.json").read_text())
            for role in ("master", "worker")}


def test_preflight_passes_on_loopback(tmp_path):
    records = run_preflight_pair(tmp_path)
    worker, master = records["worker"], records["master"]
    assert [s["name"] for s in worker["stages"]] == [
        "resolve", "interface", "tcp", "store", "interfaces", "reachability", "choose", "gloo"]
    assert "tcp" not in [s["name"] for s in master["stages"]]
    # Both sides agree on one pair, and Gloo ran on it.
    assert master["chosen_ifname"] == worker["peer_ifname"]
    assert worker["chosen_ifname"] == master["peer_ifname"]
    assert master["pairs"] == worker["pairs"] and master["pairs"]
    gloo = worker["stages"][-1]["result"]
    assert gloo["GLOO_SOCKET_IFNAME"] == worker["chosen_ifname"]
    for record in records.values():
        bandwidth = record["bandwidth"]
        assert bandwidth["MBps_in"] > 0 and bandwidth["MBps_out"] > 0
        assert bandwidth["rtt_ms"] > 0 and bandwidth["streams"] == 1
        assert "one stream driven from Python; may understate fast links" in bandwidth["note"]
    # The other side's out is this side's in.
    assert master["bandwidth"]["MBps_out"] == worker["bandwidth"]["MBps_in"]
    for record in records.values():
        assert record["environment"]["hostname"] and "--out" in record["argv"]


def test_preflight_measures_only_the_interface_it_is_given(tmp_path):
    names = {name for _, name in socket.if_nameindex()}
    lo = next(name for name in ("lo", "lo0") if name in names)
    records = run_preflight_pair(tmp_path, env=clean_env(GLOO_SOCKET_IFNAME=lo))
    for record in records.values():
        assert record["chosen_ifname"] == lo
        assert [(p["master"], p["worker"]) for p in record["pairs"]] == [(lo, lo)]


def test_preflight_without_a_worker_fails_cleanly(tmp_path):
    # The master alone: the store's wait for the worker times out, and the
    # preflight reports it and writes its record instead of a traceback.
    result = subprocess.run([sys.executable, str(KIT / "preflight.py"), "--role", "master",
                             "--master-addr", "127.0.0.1", "--port", str(_free_port()),
                             "--timeout", "3", "--out", str(tmp_path)],
                            env=clean_env(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "FAILED: reachability:" in result.stderr and "Traceback" not in result.stderr
    record = json.loads((tmp_path / "preflight-master.json").read_text())
    assert record["ok"] is False
    assert record["stages"][-1]["name"] == "reachability" and not record["stages"][-1]["ok"]


def test_preflight_candidates_skip_loopback_and_container_bridges():
    found = {"lo": "127.0.0.1", "eno1": "10.1.0.5", "ib0": "10.2.0.5",
             "docker0": "172.17.0.1", "veth12": "10.3.0.1", "usb0": "169.254.3.1"}
    assert preflight.candidates(found, None, loopback=False) == [
        ("eno1", "10.1.0.5"), ("ib0", "10.2.0.5")]
    assert ("lo", "127.0.0.1") in preflight.candidates(found, None, loopback=True)
    assert preflight.candidates(found, "ib0", loopback=False) == [("ib0", "10.2.0.5")]
    with pytest.raises(preflight.PreflightError, match="no IPv4 address"):
        preflight.candidates(found, "wg0", loopback=False)
    with pytest.raises(preflight.PreflightError, match="no candidate interface"):
        preflight.candidates({"lo": "127.0.0.1"}, None, loopback=False)


def _pair(master, worker, m2w=None, w2m=None, error=None):
    pair = {"master": master, "master_address": "10.0.0.1", "worker": worker,
            "worker_address": "10.0.0.2", "connects": error is None}
    if error is None:
        pair.update(MBps_master_to_worker=m2w, MBps_worker_to_master=w2m, rtt_ms=0.1)
    else:
        pair["error"] = error
    return pair


def test_preflight_chooses_the_pair_whose_slower_direction_is_fastest():
    pairs = [_pair("eno1", "eno1", 117, 118), _pair("ib0", "ib0", 3000, 900),
             _pair("ib1", "ib1", 1200, 1100), _pair("ib2", "ib2", error="timed out")]
    assert preflight.choose(pairs)["master"] == "ib1"
    with pytest.raises(preflight.PreflightError, match="ib2 10.0.0.1 <-> worker ib2"):
        preflight.choose([_pair("ib2", "ib2", error="timed out")])


def test_preflight_pairs_follow_the_routes():
    # On one machine every local address routes from itself.
    mine = [("lo0", "127.0.0.1")]
    assert preflight.routed_pairs(mine, [("lo", "127.0.0.1")], "master") == [("lo0", "lo")]
    assert preflight.routed_pairs(mine, [("lo", "127.0.0.1")], "worker") == [("lo", "lo0")]
    assert preflight.routed_pairs([("eno1", "10.9.9.9")], [("lo", "127.0.0.1")], "master") == []


def test_preflight_role_auto_follows_srun(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert preflight.resolve_role("auto") == "master"
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert preflight.resolve_role("auto") == "worker"
    assert preflight.resolve_role("master") == "master"


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


def test_ucx_steps_are_off_unless_asked_for():
    out = dry_run(KIT / "explorer" / "05_multinode.sbatch").stdout
    assert "--bridge ucx" not in out and "ucx_build" not in out


def test_tables_name_the_bridge_and_its_path(tmp_path):
    job = _job(tmp_path, "explorer/05_multinode", [("bench_ucx", 0)], GPU_ENV, "cuda:0")
    bench = json.loads((job / "bench_phases_x" / "bench.json").read_text())
    bench["run"].update({"bridge": "ucx", "bridge_path": "direct"})
    (job / "bench_phases_x" / "bench.json").write_text(json.dumps(bench))
    tables = validation_tables.build([tmp_path])
    assert {(r["bridge"], r["staging"]) for r in tables["benchmarks"]} == {("ucx", "ucx direct")}
    assert {(r["bridge"], r["staging"]) for r in tables["predicted"]} == {("ucx", "ucx direct")}
