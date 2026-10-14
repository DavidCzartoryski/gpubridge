"""gpubridge's GPU validation kit on a free Kaggle notebook: "GPU T4 x2", internet on.

    python kaggle_gpubridge.py --tag v0.4.0-alpha                  # every step
    python kaggle_gpubridge.py --tag v0.4.0-alpha --only bench     # one step (after setup)
    python kaggle_gpubridge.py --dry-run                           # print the plan

kaggle_gpubridge.ipynb, next to this file, runs the same steps one cell at a time
(its first cell holds this file). In order:

1. ``clone`` the gpubridge tag under test, and ``setup``: a venv with the CUDA build
   of torch (which brings Triton) and that gpubridge, both outside /kaggle/working
   so they aren't saved as notebook output;
2. ``versions``: nvidia-smi, and the torch, CUDA, NCCL and Triton versions;
3. ``features``: what this GPU can run. T4s are compute capability 7.5: CUDA
   events, bfloat16 (no native bfloat16 math before 8.0) and Triton kernels are
   each tried on every GPU. A step that needs something that fails is skipped, or
   run without it, and steps.jsonl says why;
4. the checks, each under a hard timeout: ``gpu_tests`` (pytest -m gpu), ``probe``
   on each GPU, ``check_split`` (the NVIDIA split test: two islands on 2 GPUs, the
   bridge on device tensors), ``bench`` (bench_all_reduce.py with CUDA events:
   native NCCL, gpubridge, flat-gloo, a small size sweep),
   ``bench_compression_kernels`` (output digests against the CPU reference: the
   bit-identical check) and ``verify_fsdp2_backend`` (the FSDP2 example with
   --verify, across the two islands);
5. ``tables``: validation_tables.py over the results, and ``package``: one tarball
   in /kaggle/working to download.

The checks run the cloned tag's own kit (scripts/gpu), so any tag that has it
works. A step that fails or times out is logged and the rest still run, as in
scripts/gpu/amd/run_all.sh. steps.jsonl has the kit's usual format, so
validation_tables.py and summarize.py read it. Steps run with ``GPU_COUNT=2``
or fewer: one GPU still runs everything except the split test, the benchmark and
the FSDP2 run, which are skipped with the reason. See README.md here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

REPO_URL = "https://github.com/DavidCzartoryski/gpubridge"
DEFAULT_TAG = "v0.4.0-alpha"
STEPS = ("clone", "setup", "versions", "features", "probe", "gpu_tests", "check_split", "bench",
         "bench_compression_kernels", "verify_fsdp2_backend", "tables", "package")
"""Every step, in order."""
TIMEOUTS = {"clone": 300, "setup": 2400, "versions": 120, "features": 600, "probe": 300,
            "gpu_tests": 1800, "check_split": 900, "bench": 1200, "bench_compression_kernels": 900,
            "verify_fsdp2_backend": 900, "tables": 300, "package": 300}
"""Hard wall-clock limits, in seconds: the step's whole process group is killed after
this (``timed_out`` in steps.jsonl, exit code 124). Each is above KIT_TIMEOUT, so a
stuck collective fails with its own error first."""
KIT_TIMEOUT = 300
"""Seconds every kit tool passes to gpubridge.init: bounds the rendezvous and each collective."""
KILL_GRACE = 30
"""Seconds between SIGTERM and SIGKILL when a step runs out of time."""

# What features.json says a step needs, and what to do without it.
FEATURES_PY = r'''
"""The T4 feature check: run on each visible GPU, write OUT/features.json (a kit record)."""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, sys.argv[2])  # the cloned tag's scripts/gpu, for gpukit
import torch
from gpukit import Record

out = Path(sys.argv[1])
rec = Record(out / "features.json", "scripts/gpu/kaggle/kaggle_gpubridge.py (features)")
gpus = []
rec.data["gpus"] = gpus


def tried(entry, fn):
    try:
        detail = fn()
        entry.update({"ok": True, **(detail or {})})
    except Exception as exc:  # noqa: BLE001 - a feature that fails is the finding
        entry.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=3)})


for i in range(torch.cuda.device_count()):
    dev = torch.device("cuda", i)
    gpu = {"index": i, "name": torch.cuda.get_device_name(i),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(i))), "checks": {}}
    gpus.append(gpu)
    checks = gpu["checks"]

    def events():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x = torch.ones(1 << 22, device=dev)
        start.record()
        x.mul_(2)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        assert ms >= 0, ms
        return {"elapsed_ms": round(ms, 4)}

    def bf16():
        native = torch.cuda.is_bf16_supported(including_emulation=False) if hasattr(
            torch.cuda, "is_bf16_supported") else None
        a = torch.arange(256, dtype=torch.float32).bfloat16()
        got = (a.to(dev) * 3 + 1).cpu()
        assert torch.equal(got, a * 3 + 1), "bfloat16 arithmetic on the GPU differs from CPU"
        return {"native": native}

    def triton_kernel(dtype):
        def run():
            import triton
            import triton.language as tl

            @triton.jit
            def axpy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                tl.store(y_ptr + offs, x * 2 + 1, mask=mask)

            x = torch.arange(1000, dtype=torch.float32).to(dtype).to(dev)
            y = torch.empty_like(x)
            axpy[(triton.cdiv(1000, 256),)](x, y, 1000, BLOCK=256)
            assert torch.equal(y.cpu(), x.cpu() * 2 + 1), "Triton kernel gave wrong values"
            return {"triton": triton.__version__}
        return run

    for name, fn in (("cuda_events", events), ("bfloat16", bf16),
                     ("triton_float32", triton_kernel(torch.float32)),
                     ("triton_bfloat16", triton_kernel(torch.bfloat16))):
        checks[name] = entry = {}
        tried(entry, fn)  # a feature that fails is recorded, not fatal: the check ran
        rec.write()
rec.data["ok_features"] = {name: all(g["checks"][name]["ok"] for g in gpus)
                           for name in ("cuda_events", "bfloat16", "triton_float32",
                                        "triton_bfloat16")} if gpus else {}
rec.finish()
print(json.dumps({"gpus": [(g["name"], g["capability"]) for g in gpus],
                  "features": rec.data["ok_features"]}, indent=1))
'''


def log(msg: str) -> None:
    print(f"[gpubridge-kaggle {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = Path(args.work)
        self.scratch = Path(args.scratch)
        self.repo = self.scratch / f"gpubridge-{args.tag}"
        self.venv = self.scratch / "venv"
        self.py = self.venv / "bin" / "python"
        self.kit = self.repo / "scripts" / "gpu"
        self.results = self.work / "results" / args.target
        self.dry = args.dry_run
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.skipped: list[str] = []

    # ---- steps.jsonl, as common.sh's step writes it --------------------------------------

    def record(self, name: str, entry: dict) -> None:
        if self.dry:
            return
        self.results.mkdir(parents=True, exist_ok=True)
        with open(self.results / "steps.jsonl", "a") as fh:
            fh.write(json.dumps({"step": name, **entry}) + "\n")

    def skip(self, name: str, reason: str) -> None:
        """A step not run, in skipped.jsonl: steps.jsonl's lines are steps that ran, and
        validation_tables.py counts one that exited 0 as a pass."""
        log(f"step {name} SKIPPED: {reason}")
        self.skipped.append(name)
        if not self.dry:
            self.results.mkdir(parents=True, exist_ok=True)
            with open(self.results / "skipped.jsonl", "a") as fh:
                fh.write(json.dumps({"step": name, "reason": reason}) + "\n")

    def env(self, name: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"KIT_TIMEOUT": str(KIT_TIMEOUT), "STEP_TIMEOUT": str(TIMEOUTS.get(name, 0)),
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3", "NCCL_DEBUG": "INFO",
                    "PYTHONUNBUFFERED": "1", "RESULTS": str(self.results)})
        nccl = self.results / "nccl" / name
        if not self.dry:
            nccl.mkdir(parents=True, exist_ok=True)
        env["NCCL_DEBUG_FILE"] = str(nccl / "%h.%p.log")
        if self.venv.exists():
            env["PATH"] = f"{self.venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(self.venv)
        env.update(extra or {})
        return env

    def run(self, name: str, cmd: list[str], *, extra_env: dict[str, str] | None = None,
            timeout: float | None = None, cwd: Path | None = None, note: str | None = None) -> int:
        """Run one step: stream its output to the notebook and RESULTS/NAME.log, kill its
        whole process group after its timeout, append its line to steps.jsonl."""
        limit = TIMEOUTS.get(name, 600) if timeout is None else timeout
        shown = " ".join(shlex.quote(c) for c in cmd)
        env_shown = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items())
        log(f"step {name} (limit {limit:.0f} s): {env_shown + ' ' if env_shown else ''}{shown}")
        if self.dry:
            self.passed.append(name)
            return 0
        self.results.mkdir(parents=True, exist_ok=True)
        log_path = self.results / f"{name}.log"
        start = time.monotonic()
        timed_out = False
        with open(log_path, "w") as fh:
            if note:
                fh.write(f"# {note}\n")
            fh.write(f"$ {env_shown} {shown}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env=self.env(name, extra_env), cwd=cwd, start_new_session=True)

            def copy() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    fh.write(line)
                    fh.flush()
                    sys.stdout.write(line)
            reader = threading.Thread(target=copy, daemon=True)
            reader.start()
            try:
                code = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc)
                code = 124
            reader.join(timeout=10)
            if timed_out:
                msg = f"step {name} ran out of time ({limit:.0f} s); its processes were killed"
                fh.write(f"# {msg}\n")
                log(msg)
        nccl = self.results / "nccl" / name
        if nccl.is_dir() and not any(nccl.iterdir()):
            nccl.rmdir()
        self.record(name, {"exit_code": code, "seconds": int(time.monotonic() - start),
                           "log": f"{name}.log", "timed_out": timed_out,
                           **({"note": note} if note else {})})
        (self.passed if code == 0 else self.failed).append(name)
        if code:
            log(f"step {name} FAILED (exit {code}); continuing. Log: {log_path}")
        return code

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        for sig, wait in ((signal.SIGTERM, KILL_GRACE), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def torchrun(self, nproc: int, script: Path, *args: str) -> list[str]:
        return [str(self.py), "-m", "torch.distributed.run", "--nnodes=1",
                "--master-addr=127.0.0.1", f"--master-port={free_port()}",
                f"--nproc-per-node={nproc}", str(script), *args]

    # ---- what the GPUs can do ------------------------------------------------------------

    def gpus(self) -> int:
        if self.args.gpus is not None:
            return self.args.gpus
        if self.dry or not self.py.exists():
            return 2
        out = subprocess.run([str(self.py), "-c", "import torch; print(torch.cuda.device_count())"],
                             capture_output=True, text=True, timeout=120)
        return int(out.stdout.strip() or 0)

    def features(self) -> dict[str, bool]:
        """features.json's verdicts; every feature assumed present until it is checked."""
        path = self.results / "features.json"
        assumed = {"cuda_events": True, "bfloat16": True, "triton_float32": True,
                   "triton_bfloat16": True}
        if not path.exists():
            return assumed
        found = json.loads(path.read_text()).get("ok_features") or {}
        return {**assumed, **found}

    def feature_errors(self, name: str) -> str:
        path = self.results / "features.json"
        if not path.exists():
            return ""
        gpus = json.loads(path.read_text()).get("gpus", [])
        errors = {g["checks"][name].get("error") for g in gpus if not g["checks"][name]["ok"]}
        caps = sorted({g["capability"] for g in gpus})
        return f" (compute capability {', '.join(caps)}: {'; '.join(sorted(e for e in errors if e))})"

    # ---- the steps -----------------------------------------------------------------------

    def step_clone(self) -> None:
        if self.repo.exists() and not self.dry:
            log(f"{self.repo} exists; reusing it")
            return
        self.run("clone", ["git", "clone", "--depth", "1", "--branch", self.args.tag,
                           self.args.repo_url, str(self.repo)])

    def step_setup(self) -> None:
        flavor = self.args.flavor
        extra = {"TORCH_VERSION": self.args.torch_version, "CUDA_INDEX": self.args.cuda_index,
                 "VENV": str(self.venv), "UV_CACHE_DIR": str(self.scratch / "uv-cache")}
        if flavor == "system":
            extra["SYSTEM_PYTHON"] = shutil.which("python3") or sys.executable
        self.run("setup", ["bash", str(self.kit / "setup_env.sh"), "--flavor", flavor,
                           "--venv", str(self.venv)], extra_env=extra)
        # PyTorch's Linux CUDA wheels depend on Triton; install it only if that didn't happen.
        py = shlex.quote(str(self.py))
        self.run("setup_triton", ["bash", "-c", f"{py} -c 'import triton; print(\"triton\", "
                                  f"triton.__version__)' || uv pip install --python {py} triton"],
                 timeout=600)

    def step_versions(self) -> None:
        self.run("versions", ["bash", "-c",
                 "nvidia-smi; nvidia-smi --query-gpu=index,name,compute_cap,driver_version,"
                 "memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv; "
                 f"{shlex.quote(str(self.py))} -c \"import torch, gpubridge; "
                 "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
                 "'nccl', '.'.join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None, "
                 "'gpubridge', gpubridge.__version__); "
                 "import importlib.util as u; t = u.find_spec('triton'); "
                 "print('triton', __import__('triton').__version__ if t else 'not installed')\""])

    def step_features(self) -> None:
        script = self.scratch / "features_check.py"  # a file: Triton's jit reads kernel source
        if not self.dry:
            self.scratch.mkdir(parents=True, exist_ok=True)
            script.write_text(FEATURES_PY)
        self.run("features", [str(self.py), str(script), str(self.results), str(self.kit)])

    def step_probe(self) -> None:
        self.run("probe", [str(self.py), str(self.kit / "probe.py"), "--out", str(self.results / "probe")])
        for i in range(self.gpus()):
            self.run(f"probe_gpu{i}", [str(self.py), str(self.kit / "probe.py"), "--out",
                                       str(self.results / f"probe_gpu{i}")],
                     extra_env={"CUDA_VISIBLE_DEVICES": str(i)}, timeout=TIMEOUTS["probe"])

    def step_gpu_tests(self) -> None:
        tests = [self.repo / "tests" / t for t in
                 ("test_gpu.py", "test_compression_kernels.py", "test_torch_backend.py")]
        ok = self.features()
        name, extra, note = "gpu_tests", {}, None
        if not ok["triton_float32"]:
            # Under its own name, so it never counts for the kernels' item (28).
            name = "gpu_tests_without_triton"
            tests = [t for t in tests if t.name != "test_compression_kernels.py"]
            extra["GPUBRIDGE_COMPRESSION_KERNELS"] = "torch"
            note = ("test_compression_kernels.py not run and compression on PyTorch kernels: "
                    "Triton can't run here" + self.feature_errors("triton_float32"))
            self.skip("gpu_tests", note)
        elif not ok["triton_bfloat16"]:
            note = ("Triton bfloat16 failed here" + self.feature_errors("triton_bfloat16") +
                    "; bfloat16 kernel tests are expected to fail")
        if note:
            log(note)
        self.run(name, [str(self.py), "-m", "pytest", "-m", "gpu", *map(str, tests), "-q",
                               "-p", "no:cacheprovider", "--durations=15",
                               f"--junitxml={self.results / 'gpu_tests.xml'}"],
                 extra_env=extra, cwd=self.repo, note=note)

    def two_gpus(self, name: str) -> bool:
        if self.gpus() >= 2:
            return True
        self.skip(name, f"needs 2 GPUs; {self.gpus()} visible")
        return False

    def step_check_split(self) -> None:
        if self.two_gpus("check_split"):
            self.run("check_split", self.torchrun(2, self.kit / "check.py", "--out",
                                                  str(self.results / "check_split")),
                     extra_env={"GPUBRIDGE_SPLIT_TEST": "half"})

    def step_bench(self) -> None:
        if not self.two_gpus("bench"):
            return
        if not self.features()["cuda_events"]:
            self.skip("bench", "CUDA events failed here" + self.feature_errors("cuda_events"))
            return
        self.run("bench", self.torchrun(2, self.kit / "bench_all_reduce.py", "--out",
                                        str(self.results / "bench"), "--ops", "native,gpubridge,flat-gloo",
                                        "--phases", "--min-bytes", "1K", "--max-bytes",
                                        self.args.max_bytes, "--trials", str(self.args.trials)),
                 extra_env={"GPUBRIDGE_SPLIT_TEST": "half"})

    def step_bench_compression_kernels(self) -> None:
        ok = self.features()
        name = "bench_compression_kernels"
        args = ["--out", str(self.results / "bench_compression"), "--max-bytes", "64M"]
        extra, note = {}, None
        if not ok["triton_float32"]:
            # The PyTorch kernels on the GPU, under another name: not the Triton kernels' item.
            name = "bench_compression_torch_kernels"
            extra["GPUBRIDGE_COMPRESSION_KERNELS"] = "torch"
            note = ("PyTorch kernels instead of Triton (the digests still compare GPU with CPU): "
                    "Triton can't run here" + self.feature_errors("triton_float32"))
            self.skip("bench_compression_kernels", note)
        elif not ok["triton_bfloat16"] or not ok["bfloat16"]:
            args += ["--dtypes", "float32"]
            failed = "triton_bfloat16" if not ok["triton_bfloat16"] else "bfloat16"
            note = "float32 inputs only: bfloat16 failed here" + self.feature_errors(failed)
        if note:
            log(note)
        self.run(name, [str(self.py), str(self.kit / "bench_compression.py"), *args],
                 extra_env=extra, note=note)

    def step_verify_fsdp2_backend(self) -> None:
        if self.two_gpus("verify_fsdp2_backend"):
            self.run("verify_fsdp2_backend",
                     self.torchrun(2, self.repo / "examples" / "fsdp2_torch_backend.py", "--verify"),
                     extra_env={"GPUBRIDGE_SPLIT_TEST": "half"})

    def step_tables(self) -> None:
        self.run("tables", [str(self.py), str(self.kit / "validation_tables.py"), str(self.results),
                            "--out", str(self.results)])

    def step_package(self) -> Path:
        tarball = self.work / f"{self.args.target}.tgz"
        log(f"step package: {tarball}")
        if self.dry:
            return tarball
        start = time.monotonic()
        (self.results / "kaggle_summary.json").write_text(json.dumps(self.summary(), indent=1) + "\n")
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(self.results, arcname=self.results.name)
        self.record("package", {"exit_code": 0, "seconds": int(time.monotonic() - start),
                                "log": None, "timed_out": False})
        self.passed.append("package")
        size = tarball.stat().st_size / 1e6
        log(f"results: {tarball} ({size:.1f} MB). In Kaggle: the notebook's Output panel "
            "(right sidebar, /kaggle/working), or the link below.")
        return tarball

    def summary(self) -> dict[str, list[str]]:
        """Every step so far, from steps.jsonl and skipped.jsonl: the notebook runs each step
        in a new Runner, so this can't come from one Runner's lists."""
        def lines(name: str) -> list[dict]:
            path = self.results / name
            return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
        ran = lines("steps.jsonl")
        return {"passed": [s["step"] for s in ran if s["exit_code"] == 0],
                "failed": [s["step"] for s in ran if s["exit_code"] != 0],
                "timed_out": [s["step"] for s in ran if s.get("timed_out")],
                "skipped": [f"{s['step']}: {s['reason']}" for s in lines("skipped.jsonl")]}

    def main(self, only: list[str]) -> int:
        log(f"gpubridge {self.args.tag}; results go to {self.results}")
        for name in only:
            getattr(self, f"step_{name}")()
        log(f"{'planned' if self.dry else 'passed'}: {' '.join(self.passed) or 'none'}")
        if self.skipped:
            log(f"skipped (reason in steps.jsonl): {' '.join(self.skipped)}")
        if self.failed:
            log(f"FAILED: {' '.join(self.failed)}")
            return 1
        return 0


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    on_kaggle = Path("/kaggle/working").is_dir()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default=DEFAULT_TAG, help="the gpubridge tag (or branch) to test")
    p.add_argument("--repo-url", default=REPO_URL)
    p.add_argument("--work", default="/kaggle/working" if on_kaggle else str(Path.home() / "gpubridge-kaggle"),
                   help="results and the tarball go here (Kaggle saves /kaggle/working)")
    p.add_argument("--scratch", default="/tmp/gpubridge-kaggle",
                   help="the clone, the venv and the download cache (not saved)")
    p.add_argument("--target", default=f"kaggle-t4x2-{stamp}", help="name of the results directory")
    p.add_argument("--flavor", choices=("cuda", "system"), default="cuda",
                   help="cuda: install the CUDA build of torch; system: reuse the image's torch (>= 2.8)")
    p.add_argument("--torch-version", default="2.14.1")
    p.add_argument("--cuda-index", default="cu126", help="PyTorch wheel index; cu126 runs on T4 (sm_75)")
    p.add_argument("--max-bytes", default="16M", help="largest all_reduce in the benchmark sweep")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--gpus", type=int, default=None, help="GPU count to plan for (default: what torch sees)")
    p.add_argument("--only", default=",".join(STEPS), help="comma-separated steps, in this order: "
                   + ", ".join(STEPS))
    p.add_argument("--dry-run", action="store_true", help="print each step's command, run nothing")
    args = p.parse_args(argv)
    args.only_steps = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in args.only_steps if s not in STEPS]
    if unknown:
        p.error(f"unknown step(s) {unknown}; steps are {', '.join(STEPS)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    return Runner(args).main(args.only_steps)


if __name__ == "__main__":
    sys.exit(main())
