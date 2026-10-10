import pytest
import torch

from gpubridge.config import CPU_ONLY_ENV, VENDOR_ENV, is_cpu_only, is_simulated, resolve_config
from gpubridge.detect import detect_hardware_vendor, detect_vendor, vendor_override


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(VENDOR_ENV, raising=False)
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
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
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "3")
    config = resolve_config()
    assert not config.simulated
    assert config.island_backend == "nccl"
    assert config.device == torch.device("cuda", 3)


@pytest.mark.parametrize(
    ("local_rank", "count", "match"),
    [("2", 2, r"LOCAL_RANK=2 asks for GPU 2, but this process sees only 2 GPUs\."),
     ("1", 1, r"sees only 1 GPU\."),
     ("-1", 4, r"LOCAL_RANK=-1 asks for GPU -1"),
     ("one", 4, r"LOCAL_RANK='one' is not an integer")],
)
def test_a_local_rank_without_a_gpu_is_a_clear_error(monkeypatch, local_rank, count, match):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    with pytest.raises(RuntimeError, match=match):
        resolve_config()


def test_without_local_rank_the_current_device_is_checked_too(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert resolve_config().device == torch.device("cuda", 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="The current CUDA device asks for GPU 1"):
        resolve_config()


def test_real_mode_without_gpu_points_at_simulation_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=VENDOR_ENV):
        resolve_config()


@pytest.mark.parametrize(
    ("hip", "cuda", "expected"),
    [("7.2.26015", None, "amd"), (None, "13.0", "nvidia")],
)
def test_cpu_only_keeps_the_build_vendor(monkeypatch, hip, cuda, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(CPU_ONLY_ENV, "1")
    assert detect_vendor() == expected
    config = resolve_config()
    assert config.simulated
    assert config.island_backend == "gloo"
    assert config.device == torch.device("cpu")


def test_cpu_only_on_a_cpu_build_cannot_detect_a_vendor(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setenv(CPU_ONLY_ENV, "1")
    with pytest.raises(RuntimeError, match=CPU_ONLY_ENV):
        detect_vendor()


def test_vendor_override_wins_over_cpu_only(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setenv(CPU_ONLY_ENV, "1")
    monkeypatch.setenv(VENDOR_ENV, "amd")
    assert detect_vendor() == "amd"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" YES ", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("", False)],
)
def test_cpu_only_flag_values(monkeypatch, value, expected):
    monkeypatch.setenv(CPU_ONLY_ENV, value)
    assert is_cpu_only() is expected
    assert is_simulated() is expected


def test_cpu_only_rejects_non_boolean(monkeypatch):
    monkeypatch.setenv(CPU_ONLY_ENV, "maybe")
    with pytest.raises(ValueError, match="maybe"):
        is_cpu_only()
