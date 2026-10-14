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
                2.0**-100, 3.4e38, -3.0e38, 448.0, 2.0**-6, 2.0**-9,
                # float32's largest, and the float32 one unit above fp8's largest sent value,
                # 448 * 2^119: both saturate.
                3.4028234663852886e38, 448 * 2.0**119 + 2.0**104]
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


FLT_MAX = torch.finfo(torch.float32).max


def _edge(case: str, dtype: torch.dtype) -> torch.Tensor:
    """Inputs at the codec's edges: float32's largest, a block of float32 subnormals, and
    a block mixing 1e-4 with 3e38 (whose small values are subnormal after scaling)."""
    if case == "near-max":
        big = [FLT_MAX, -FLT_MAX, 3.4e38, -3.4e38, 3.0e38, 448 * 2.0**119,
               448 * 2.0**119 + 2.0**104,
               float(torch.finfo(torch.bfloat16).max), 3.3961e38, 1.0, -2.5]
        if dtype == torch.float64:
            big += [4e38, -1e39, 1e300]
        x = torch.tensor(big * 30, dtype=torch.float64)
    elif case == "subnormal-block":
        x = torch.tensor([1e-40, -1e-42, 1.4e-45, -(2.0**-140), 2.0**-127] * 60,
                         dtype=torch.float64)
    else:
        x = torch.tensor([3e38, -3e38] + [1e-4, -1e-4, 2.5e-4, 7e-5] * 63 + [1e-4, 3e-4],
                         dtype=torch.float64)
    return x.to(dtype)


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("case", ["near-max", "subnormal-block", "mixed-block"])
def test_kernels_match_the_reference_at_the_codecs_edges(mode, dtype, case):
    if not C.applies(mode, dtype):
        pytest.skip(f"{mode} doesn't compress {dtype}")
    x = _edge(case, dtype)
    (ref_p, ref_r), (tri_p, tri_r) = _quantize_both(mode, x, True)
    assert torch.equal(ref_p, tri_p)
    assert torch.equal(ref_r.view(torch.int32), tri_r.view(torch.int32))
    assert ref_r.isfinite().all()
    n = x.numel()
    ref, tri = torch.empty(n), torch.empty(n)
    C.dequantize_accumulate_reference(mode, ref_p[None], ref)
    K.dequantize_accumulate(mode, tri_p[None], tri)
    assert _same(ref, tri)
    assert ref[x.isfinite()].isfinite().all()  # a finite value arrives finite


@pytest.mark.parametrize("mode", C.MODES)
def test_kernels_error_feedback_loop_matches_the_reference_and_stays_finite(mode):
    g = torch.Generator().manual_seed(3)
    x = torch.cat([_edge("near-max", torch.float32), _edge("mixed-block", torch.float32),
                   torch.randn(700, generator=g) * 2.0 ** torch.randint(-30, 30, (700,),
                                                                         generator=g)])
    n = x.numel()
    ref_r, tri_r = torch.zeros(n), torch.zeros(n)
    for _ in range(30):
        ref_p = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
        tri_p = torch.empty_like(ref_p)
        C.quantize_reference(mode, x, ref_r, ref_p)
        K.quantize(mode, x, tri_r, tri_p)
        assert torch.equal(ref_p, tri_p)
        assert torch.equal(ref_r.view(torch.int32), tri_r.view(torch.int32))
        assert ref_r.isfinite().all()


#: Compiles the kernels for NVIDIA targets without a GPU (Triton's own LLVM and ptxas)
#: and counts fused multiply-adds in their PTX. Prints JSON, or "SKIP: why".
_PTX_COUNT = r"""
import json, os, re, sys
os.environ.pop("TRITON_INTERPRET", None)
try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
except ImportError as exc:
    print(f"SKIP: this Triton can't compile offline ({exc})"); sys.exit(0)
from gpubridge import _compression_kernels as K

Q = {"res_ptr": "*fp32", "scale_ptr": "*fp32", "code_ptr": "*u8", "n": "i32",
     "BLOCK_SIZE": "constexpr", "EF": "constexpr"}
KERNELS = {
    "quantize_fp8 float32": (K._quantize_fp8, {"x_ptr": "*fp32", **Q}, {"BLOCK_SIZE": 256, "EF": True}),
    "quantize_fp8 float64": (K._quantize_fp8, {"x_ptr": "*fp64", **Q}, {"BLOCK_SIZE": 256, "EF": True}),
    "quantize_bf16": (K._quantize_bf16, {"x_ptr": "*fp32", "res_ptr": "*fp32", "code_ptr": "*i16",
                      "n": "i32", "BLOCK_SIZE": "constexpr", "EF": "constexpr"},
                      {"BLOCK_SIZE": 1024, "EF": True}),
    "dequantize_fp8": (K._dequantize_fp8, {"scale_ptr": "*fp32", "code_ptr": "*u8",
                       "out_ptr": "*fp32", "n": "i32", "members": "i32", "scale_stride": "i32",
                       "code_stride": "i32", "BLOCK_SIZE": "constexpr", "OUT_BF16": "constexpr"},
                       {"BLOCK_SIZE": 256, "OUT_BF16": False}),
    "dequantize_bf16": (K._dequantize_bf16, {"code_ptr": "*i16", "out_ptr": "*fp32", "n": "i32",
                        "members": "i32", "code_stride": "i32", "BLOCK_SIZE": "constexpr",
                        "OUT_BF16": "constexpr"}, {"BLOCK_SIZE": 1024, "OUT_BF16": False}),
}
counts = {}
for arch, fusions in ((70, (False, True)), (90, (False,))):
    for name, (fn, signature, constexprs) in KERNELS.items():
        for fuse in fusions:
            options = {**K.LAUNCH_OPTIONS} if not fuse else {**K.LAUNCH_OPTIONS, "enable_fp_fusion": True}
            try:
                kernel = triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs),
                                        target=GPUTarget("cuda", arch, 32), options=options)
            except Exception as exc:
                if "ptxas" in str(exc).lower():
                    print(f"SKIP: no ptxas for an offline compile ({exc})"); sys.exit(0)
                raise
            ptx = kernel.asm["ptx"]
            counts[f"sm_{arch} {name}{' fusion on' if fuse else ''}"] = len(
                re.findall(r"\bfma\.rn(?:\.ftz)?\.f32\b", ptx))
print(json.dumps(counts))
"""


def test_the_kernels_compile_without_fused_multiply_adds():
    # As gpubridge launches them (LAUNCH_OPTIONS): no fma.rn.f32 in any kernel's PTX, so
    # every product is rounded before it is added, as on CPU. With fusion on, the same
    # kernels do have them: the count would see a regression.
    import json
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    result = subprocess.run([sys.executable, "-c", _PTX_COUNT], env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    if result.stdout.startswith("SKIP:"):
        pytest.skip(result.stdout.strip())
    counts = json.loads(result.stdout.strip().splitlines()[-1])
    assert {name: c for name, c in counts.items() if "fusion on" not in name and c} == {}
    assert counts["sm_70 quantize_fp8 float32 fusion on"] > 0
    assert counts["sm_70 dequantize_fp8 fusion on"] > 0


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
    # Payloads and residuals, bit for bit. Not v, the value before quantizing: for a NaN
    # input (index 74 at n=255, _awkward's NaN at 2 * 37) v is NaN on both, but with the
    # input's own bits on CPU (0x7fc00000) and the GPU's canonical NaN on the device,
    # which scripts/gpu/diag_fp8_residual.py's probe reports apart. A NaN is never sent:
    # its code is 0x7f and its residual 0, so neither can differ.
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


# ---- scripts/gpu/diag_fp8_residual.py, on CPU through the interpreter ---------------------------

def _diagnostic():
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
    import diag_fp8_residual
    return diag_fp8_residual


def test_the_fp8_residual_diagnostic_reruns_the_gpu_test_case():
    diag = _diagnostic()
    for n in SIZES:
        for dtype in FLOATS:
            assert _same(diag.awkward(n, dtype, n), _awkward(n, dtype, n))
        expected = torch.randn(n, generator=torch.Generator().manual_seed(5)) * 1e-3
        assert torch.equal(diag.prior_residual(n), expected)


#: The script's own run on a GPU is the real thing; these check it through the interpreter.
interpreter_only = pytest.mark.skipif(GPU, reason="on a GPU, run the script itself")


@interpreter_only
def test_the_fp8_residual_diagnostic_finds_nothing_where_the_kernels_match(capsys):
    assert _diagnostic().main(["--n", "255", "--dtype", "float32", "float16"]) == 0
    out = capsys.readouterr().out
    assert out.count("residual: 0 of 255 differ in their bits") == 2
    assert "all payloads and residuals match the reference" in out


@interpreter_only
def test_the_fp8_residual_diagnostic_shows_where_a_device_differs(monkeypatch, capsys):
    # A stand-in for what the first GPU run found: the codec before its scale exponent
    # was capped, with v - decoded * scale fused into one rounding. At x = 3.4e38 its
    # code decoded to 256 * 2^120 = 2^128, beyond float32: the reference's residual was
    # -inf, the fused one finite. The fixed reference sends 448 * 2^119 instead.
    diag = _diagnostic()

    def old_codec_fused(mode, x, residual, payload):
        prior = residual.clone()
        with monkeypatch.context() as patch:
            patch.setattr(C, "MAX_SCALE_EXP", 127)  # no cap
            C.quantize_reference(mode, x, residual, payload)
        v, finite = C._prepare(x, prior)
        scales, codes = diag.split(payload, x.numel())
        exact = (v.double() - C._decode_e4m3(codes).double()
                 * diag.blocks_of(scales, x.numel()).double())
        residual.copy_(torch.where(finite, exact.float(), 0.0))

    monkeypatch.setattr(diag.K, "quantize", old_codec_fused)
    assert diag.main(["--n", "255", "--dtype", "float32", "--limit", "3"]) == 1
    out = capsys.readouterr().out
    assert "payload: DIFFERS (codes differ at 4, scales differ, 1 block(s))" in out
    assert "scales differ in 1 block(s): block 0: reference 2^119, device 2^120" in out
    # 3.4e38 and float32's largest: the two whose code decoded beyond float32.
    assert "residual: 2 of 255 differ in their bits; indices [41, 226]" in out
    block = out[out.index("index 41"):out.index("index 78")]
    for label, text in [("code (reference)", "0x7e"), ("code (device)", "0x78"),
                        ("sent = decoded * scale", "2.97747071e+38"),
                        ("sent on the device (probe)", "inf"),
                        ("residual, reference", "4.22529242e+37"),
                        ("residual, device ", "-2.82371707e+35")]:
        line = next(line for line in block.splitlines() if line.strip().startswith(label))
        assert text in line, (label, line)
    assert "the device's residual is neither the reference's nor the fused value" in block
    assert "some payloads or residuals differ" in out
