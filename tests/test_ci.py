"""CI's split of the suite across jobs (GPUBRIDGE_TEST_SHARD, tests/conftest.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = ["tests/test_release.py", "tests/test_detect.py"]  # quick to collect


def collected(shard: str | None) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "GPUBRIDGE_TEST_SHARD"}
    if shard is not None:
        env["GPUBRIDGE_TEST_SHARD"] = shard
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider", *FILES],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    return [line for line in out.stdout.splitlines() if "::" in line]


def test_the_shards_run_every_test_exactly_once():
    everything = collected(None)
    shards = [collected(f"{k}/3") for k in (1, 2, 3)]
    assert len(everything) > 6
    assert sorted(sum(shards, [])) == sorted(everything)
    assert all(abs(len(s) - len(everything) / 3) < 1 for s in shards)


@pytest.mark.parametrize("shard", ["0/3", "4/3", "2", "a/b"])
def test_a_bad_shard_is_an_error(shard):
    env = {**os.environ, "GPUBRIDGE_TEST_SHARD": shard}
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider", FILES[0]],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "GPUBRIDGE_TEST_SHARD" in out.stdout + out.stderr
