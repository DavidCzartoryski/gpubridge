"""Session-wide test settings."""

import os

import pytest

from gpubridge.config import BRIDGE_COMPRESSION_ENV

#: CI splits the suite across parallel jobs: GPUBRIDGE_TEST_SHARD=K/N keeps every Nth
#: collected test, starting with the Kth (1-based), so N jobs run each test exactly once.
SHARD_ENV = "GPUBRIDGE_TEST_SHARD"


def pytest_configure(config):
    # CI's docs job runs only these (pytest -m docs) on every pull request, docs-only
    # ones included, so a test that reads a committed doc or notebook must carry it.
    config.addinivalue_line(
        "markers", "docs: reads a committed doc or notebook (HARDWARE_VALIDATION.md, the "
        "Kaggle notebook)")


@pytest.fixture(autouse=True)
def _uncompressed_by_default(monkeypatch):
    """Every test runs with bridge compression off unless it turns it on itself (on the
    ranks it spawns), so the bit-exact tests always take the uncompressed path."""
    monkeypatch.delenv(BRIDGE_COMPRESSION_ENV, raising=False)


def pytest_collection_modifyitems(config, items):
    shard = os.environ.get(SHARD_ENV)
    if not shard:
        return
    try:
        k, n = (int(part) for part in shard.split("/"))
    except ValueError:
        raise pytest.UsageError(f"{SHARD_ENV}={shard!r}: expected K/N, e.g. 1/3") from None
    if not 1 <= k <= n:
        raise pytest.UsageError(f"{SHARD_ENV}={shard!r}: K must be 1 to N")
    # Round-robin in collection order: each file's tests, slow ones included, spread
    # over every shard, so the shards take about the same time without a timings file.
    config.hook.pytest_deselected(items=[t for i, t in enumerate(items) if i % n != k - 1])
    items[:] = items[k - 1::n]
