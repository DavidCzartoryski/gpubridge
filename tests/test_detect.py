import pytest
import torch

from gpubridge.config import VENDOR_ENV, is_simulated, resolve_config
from gpubridge.detect import detect_hardware_vendor, detect_vendor, vendor_override


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(VENDOR_ENV, raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("nvidia", "nvidia"), ("amd", "amd"), ("NVIDIA", "nvidia"), (" amd\n", "amd")],
)
def test_override_selects_vendor(monkeypatch, value, expected):
    monkeypatch.setenv(VENDOR_ENV, value)
    assert detect_vendor() == expected
    assert is_simulated()


def test_override_rejects_unknown_vendor(monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "intel")
    with pytest.raises(ValueError, match="intel"):
        detect_vendor()


def test_empty_override_counts_as_unset(monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "  ")
    assert vendor_override() is None
    assert not is_simulated()


@pytest.mark.parametrize(
    ("hip", "cuda", "expected"),
    [("6.2.41133", None, "amd"), (None, "12.4", "nvidia")],
)
def test_detects_vendor_from_build(monkeypatch, hip, cuda, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    assert detect_hardware_vendor() == expected
    assert detect_vendor() == expected


def test_override_wins_over_build(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setenv(VENDOR_ENV, "amd")
    assert detect_vendor() == "amd"


def test_cpu_only_build_points_at_simulation_mode(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match=VENDOR_ENV):
        detect_vendor()


def test_simulation_uses_gloo_on_cpu(monkeypatch):
    monkeypatch.setenv(VENDOR_ENV, "nvidia")
    config = resolve_config()
    assert config.simulated
    assert config.island_backend == "gloo"
    assert config.device == torch.device("cpu")


def test_real_mode_uses_nccl_on_local_rank(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LOCAL_RANK", "3")
    config = resolve_config()
    assert not config.simulated
    assert config.island_backend == "nccl"
    assert config.device == torch.device("cuda", 3)


def test_real_mode_without_gpu_points_at_simulation_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=VENDOR_ENV):
        resolve_config()
