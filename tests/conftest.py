"""Session-wide test settings."""

import pytest

from gpubridge.config import BRIDGE_COMPRESSION_ENV


@pytest.fixture(autouse=True)
def _uncompressed_by_default(monkeypatch):
    """Every test runs with bridge compression off unless it turns it on itself (on the
    ranks it spawns), so the bit-exact tests always take the uncompressed path."""
    monkeypatch.delenv(BRIDGE_COMPRESSION_ENV, raising=False)
