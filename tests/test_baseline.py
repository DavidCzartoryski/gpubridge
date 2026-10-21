"""The CPU baseline's tools: its identity record and the import benchmark."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def test_the_identity_record_names_what_ran_and_leaks_no_home_path(tmp_path):
    out = tmp_path / "identity.json"
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "baseline_identity.py"),
                             str(out)], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    record = json.loads(text)
    assert re.fullmatch(r"[0-9a-f]{40}", record["git"]["commit"] or "")
    assert set(record["git"]["trees"]) >= {"src", "src/gpubridge/verbs/crate"}
    # The same release (a CPU wheel's metadata may or may not carry "+cpu").
    assert record["packages"]["torch"].split("+")[0] == torch.__version__.split("+")[0]
    assert record["python"]["version"] == ".".join(map(str, sys.version_info[:3]))
    assert record["kit_environment"]["probe"]["torch_version"] == torch.__version__
    # Built libraries: a name, a size and a hash each, or None where none was built.
    for library in record["libraries"].values():
        assert library is None or re.fullmatch(r"[0-9a-f]{64}", library["sha256"])
    assert os.path.expanduser("~") not in text  # redacted like every kit record


def test_the_baseline_script_parses():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "baseline.sh")]).returncode == 0


def test_the_import_benchmark_compares_two_trees(tmp_path):
    # This checkout against a copy of itself, one run each, without bytecode: both arms
    # load the same modules, and the copy really is what the before arm imported.
    copy = tmp_path / "src"
    shutil.copytree(REPO / "src" / "gpubridge", copy / "gpubridge",
                    ignore=shutil.ignore_patterns("__pycache__", "target"))
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "bench_import.py"),
                             "--before", str(copy), "--runs", "1", "--no-bytecode"],
                            cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["bytecode"] is False
    assert out["before"]["modules"] == out["after"]["modules"]
    assert out["before"]["runs"] == out["after"]["runs"] == 1
    assert out["after"]["gpubridge_ms_median"] > 0
