"""The Triton compression kernels against the plain-PyTorch reference, bit for bit.

Without a GPU, Triton's interpreter runs the kernels on CPU tensors: this
module sets TRITON_INTERPRET=1 before Triton is first imported, so it must
not have been imported already (gpubridge itself imports it only on demand).
Tests marked ``gpu`` run the compiled kernels on a CUDA or ROCm GPU and are
skipped without one. CI sets GPUBRIDGE_REQUIRE_TRITON=1, so a missing Triton
fails there instead of skipping.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

GPU = torch.cuda.is_available()
if not GPU and "triton" not in sys.modules:
    os.environ.setdefault("TRITON_INTERPRET", "1")

try:
    import triton  # noqa: F401
except ImportError as exc:
    if os.environ.get("GPUBRIDGE_REQUIRE_TRITON") == "1":
        raise RuntimeError("GPUBRIDGE_REQUIRE_TRITON=1 but Triton can't be imported") from exc
    pytest.skip(f"Triton isn't installed ({exc})", allow_module_level=True)

from gpubridge import _compression_kernels as K  # noqa: E402
from gpubridge import compression as C  # noqa: E402

FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
SIZES = (1, 7, 255, 256, 257, 1000, 3000)
#: The interpreter computes masked-out lanes too (inf - inf there is harmless).
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _awkward(n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Many magnitudes per block, plus every special value the codec handles."""
    g = torch.Generator().manual_seed(seed)
    scale = 2.0 ** torch.randint(-40, 40, (n,), generator=g)
    x = torch.randn(n, generator=g, dtype=torch.float64) * scale
    specials = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-35, -1e-31,
                2.0**-100, 3.4e38, -3.0e38, 448.0, 2.0**-6, 2.0**-9]
    for i, value in enumerate(specials[:n]):
        x[(i * 37) % n] = value
    return x.to(dtype)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit for bit, NaN payloads aside."""
    if a.dtype.is_floating_point:
        return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(0.0),
                                                                 b.nan_to_num(0.0))
    return torch.equal(a, b)


def _quantize_both(mode, x, ef, device="cpu"):
    n = x.numel()
    residual = (torch.randn(n, generator=torch.Generator().manual_seed(5)) * 1e-3
                if ef else None)
    ref_p = torch.zeros(C.payload_nbytes(mode, n), dtype=torch.uint8)
    ref_r = residual.clone() if ef else None
    C.quantize_reference(mode, x, ref_r, ref_p)
    tri_p = torch.zeros_like(ref_p, device=device)
    tri_r = residual.to(device) if ef else None
    K.quantize(mode, x.to(device), tri_r, tri_p)
    return (ref_p, ref_r), (tri_p.cpu(), tri_r.cpu() if ef else None)


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("ef", [False, True], ids=["no-feedback", "feedback"])
def test_quantize_matches_the_reference(mode, dtype, n, ef):
    if not C.applies(mode, dtype):
        pytest.skip(f"{mode} doesn't compress {dtype}")
    (ref_p, ref_r), (tri_p, tri_r) = _quantize_both(mode, _awkward(n, dtype, n), ef)
    assert torch.equal(ref_p, tri_p)
    if ef:
        assert torch.equal(ref_r.view(torch.int32), tri_r.view(torch.int32))


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("out_dtype", FLOATS)
@pytest.mark.parametrize("members", [1, 2, 3])
@pytest.mark.parametrize("n", SIZES)
def test_dequantize_accumulate_matches_the_reference(mode, out_dtype, members, n):
    rows = []
    for i in range(members):
        payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
        C.quantize_reference(mode, _awkward(n, torch.float32, 100 + i), None, payload)
        rows.append(payload)
    payloads = torch.stack(rows)
    ref = torch.empty(n, dtype=out_dtype)
    tri = torch.empty(n, dtype=out_dtype)
    C.dequantize_accumulate_reference(mode, payloads, ref)
    K.dequantize_accumulate(mode, payloads, tri)
    assert _same(ref, tri)


@pytest.mark.parametrize("mode", C.MODES)
def test_cancellation_near_tiny_gives_the_same_zeros(mode):
    # Two members whose values nearly cancel: the exact difference is below 2^-100,
    # where a GPU that flushes subnormals and one that doesn't could disagree. Both
    # implementations zero it explicitly.
    n = 512
    # Both exact in the mode's format; their sum, 2^-102 or 2^-106, is below 2^-100.
    a = torch.full((n,), (1.125 if mode == "fp8" else 1 + 2.0**-7) * 2.0**-99)
    b = torch.full((n,), -(2.0**-99))
    rows = []
    for v in (a, b):
        payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
        C.quantize_reference(mode, v, None, payload)
        rows.append(payload)
    ref, tri = torch.empty(n), torch.empty(n)
    C.dequantize_accumulate_reference(mode, torch.stack(rows), ref)
    K.dequantize_accumulate(mode, torch.stack(rows), tri)
    assert torch.equal(ref, tri) and torch.equal(ref, torch.zeros(n))


def test_empty_tensors_do_nothing():
    for mode in C.MODES:
        K.quantize(mode, torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.uint8))
        K.dequantize_accumulate(mode, torch.empty((2, 0), dtype=torch.uint8), torch.empty(0))


def test_dispatch_uses_triton_only_when_asked_or_on_gpus(monkeypatch):
    monkeypatch.delenv(C.KERNELS_ENV, raising=False)
    assert C._triton_kernels(torch.empty(1)) is None  # CPU tensors: the reference
    monkeypatch.setenv(C.KERNELS_ENV, "triton")
    assert C._triton_kernels(torch.empty(1)) is K
    monkeypatch.setenv(C.KERNELS_ENV, "torch")
    assert C._triton_kernels(torch.empty(1)) is None
    monkeypatch.setenv(C.KERNELS_ENV, "cuda")
    with pytest.raises(ValueError, match="GPUBRIDGE_COMPRESSION_KERNELS"):
        C._triton_kernels(torch.empty(1))


# ---- on a GPU: the compiled kernels, against the reference on CPU ------------------------------

gpu = pytest.mark.gpu
needs_gpu = pytest.mark.skipif(not GPU, reason="needs a CUDA or ROCm GPU")


@gpu
@needs_gpu
@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("ef", [False, True], ids=["no-feedback", "feedback"])
def test_gpu_quantize_matches_the_cpu_reference(mode, dtype, ef):
    if not C.applies(mode, dtype):
        pytest.skip(f"{mode} doesn't compress {dtype}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in SIZES + (1 << 20,):
            (ref_p, ref_r), (tri_p, tri_r) = _quantize_both(mode, _awkward(n, dtype, n), ef,
                                                            device="cuda")
            assert torch.equal(ref_p, tri_p), n
            if ef:
                assert torch.equal(ref_r.view(torch.int32), tri_r.view(torch.int32)), n


@gpu
@needs_gpu
@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("out_dtype", FLOATS)
def test_gpu_dequantize_accumulate_matches_the_cpu_reference(mode, out_dtype):
    for n in SIZES + (1 << 20,):
        rows = []
        for i in range(3):
            payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
            C.quantize_reference(mode, _awkward(n, torch.float32, 100 + i), None, payload)
            rows.append(payload)
        payloads = torch.stack(rows)
        ref = torch.empty(n, dtype=out_dtype)
        C.dequantize_accumulate_reference(mode, payloads, ref)
        tri = torch.empty(n, dtype=out_dtype, device="cuda")
        K.dequantize_accumulate(mode, payloads.cuda(), tri)
        assert _same(ref, tri.cpu()), n
