"""Where the fp8 kernel's error-feedback residual differs from the CPU reference.

Reruns the fp8, error-feedback case of
``tests/test_compression_kernels.py::test_gpu_quantize_matches_the_cpu_reference``
on one device: the same inputs (``_awkward(n, dtype, n)``), the same prior
residual (``randn(n)`` from seed 5, times 1e-3), and
``gpubridge._compression_kernels.quantize`` against
``gpubridge.compression.quantize_reference`` on CPU. For the first values
whose new residual, code or scale differs, bit for bit, it prints what each
step computed: the input, the prior residual, v (the input in float32 plus
the prior residual, magnitudes below 2^-100 zeroed), the block's largest
magnitude, the scale exponent k, the code, its decoded value, what was sent,
both residuals (value and bits), which of those are subnormal in float32, and
what a fused multiply-add would give for ``v - decoded * scale`` (one
rounding instead of two).

Then, for the same inputs:

- the kernel compiled with ``enable_fp_fusion=False`` (no multiply-add
  contraction), against the reference;
- a probe kernel built from the kernel's own helpers, which stores v and
  ``decoded * scale`` as the device computes them;
- what the kernel's compiled code contains: fused multiply-adds, flush-to-zero
  instructions and the target.

The kernel itself is unchanged. One device, well under a minute::

    python scripts/gpu/diag_fp8_residual.py [--dtype float32 float64] [--n 255] [--limit 10]

Without a GPU it runs on CPU through Triton's interpreter (TRITON_INTERPRET=1,
set here before Triton is first imported), where every residual should match.
Exits 1 if any payload or residual differs from the reference, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import torch

GPU = torch.cuda.is_available()
if not GPU and "triton" not in sys.modules:
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from gpubridge import _compression_kernels as K  # noqa: E402
from gpubridge import compression as C  # noqa: E402
from gpubridge._compression_kernels import _decode_e4m3, _prepare  # noqa: E402

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
          "float64": torch.float64}
#: The smallest normal float32, 2^-126, and the largest finite one.
MIN_NORMAL = 2.0**-126
FLT_MAX = torch.finfo(torch.float32).max


def awkward(n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """The test's inputs: tests/test_compression_kernels.py's ``_awkward``, kept equal by a test."""
    g = torch.Generator().manual_seed(seed)
    scale = 2.0 ** torch.randint(-40, 40, (n,), generator=g)
    x = torch.randn(n, generator=g, dtype=torch.float64) * scale
    specials = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-35, -1e-31,
                2.0**-100, 3.4e38, -3.0e38, 448.0, 2.0**-6, 2.0**-9]
    for i, value in enumerate(specials[:n]):
        x[(i * 37) % n] = value
    return x.to(dtype)


def prior_residual(n: int) -> torch.Tensor:
    """The test's residual before the call: float32, from seed 5."""
    return torch.randn(n, generator=torch.Generator().manual_seed(5)) * 1e-3


@triton.jit
def _probe(x_ptr, res_ptr, scale_ptr, code_ptr, v_ptr, sent_ptr, n, BLOCK_SIZE: tl.constexpr):
    """v as the kernel prepares it, and decoded * scale from its payload, both stored."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    v, finite = _prepare(x_ptr, res_ptr, offs, mask, True)
    code = tl.load(code_ptr + offs, mask=mask, other=0).to(tl.int32)
    sent = _decode_e4m3(code) * tl.load(scale_ptr + pid)
    tl.store(v_ptr + offs, v, mask=mask)
    tl.store(sent_ptr + offs, sent, mask=mask)


def launch(x: torch.Tensor, residual: torch.Tensor, payload: torch.Tensor, **options):
    """K.quantize's fp8 branch, with Triton launch options; returns the compiled kernel
    (None under the interpreter)."""
    n = x.numel()
    nb = triton.cdiv(n, C.BLOCK)
    scales = payload[:4 * nb].view(torch.float32)
    codes = payload[4 * nb:4 * nb + n]
    compiled = K._quantize_fp8[(nb,)](x.reshape(-1), residual, scales, codes, n,
                                      BLOCK_SIZE=C.BLOCK, EF=True, **options)
    payload[4 * nb + n:].zero_()
    return compiled


def code_facts(compiled) -> str:
    """Counts of the instructions that matter here, from the compiled kernel."""
    asm = getattr(compiled, "asm", None) or {}
    if "ptx" in asm:
        ptx = asm["ptx"]
        target = re.search(r"^\.target\s+(\S+)", ptx, re.M)
        counts = {name: len(re.findall(pattern, ptx)) for name, pattern in (
            ("fma.rn.f32", r"\bfma\.rn(?:\.ftz)?(?:\.sat)?\.f32\b"),
            ("mul.rn.f32", r"\bmul\.rn(?:\.ftz)?\.f32\b"),
            ("mul.f32 (no rounding mode: ptxas may fuse it)", r"\bmul(?:\.ftz)?\.f32\b"),
            ("sub/add.rn.f32", r"\b(?:sub|add)\.rn(?:\.ftz)?\.f32\b"),
            ("sub/add.f32 (no rounding mode)", r"\b(?:sub|add)(?:\.ftz)?\.f32\b"),
            ("any .ftz", r"\.ftz\b"))}
        return (f"PTX target {target.group(1) if target else '?'}; "
                + ", ".join(f"{name}: {count}" for name, count in counts.items()))
    if "amdgcn" in asm:
        gcn = asm["amdgcn"]
        fma = len(re.findall(r"\bv_(?:fma|fmac|mad|mac)\w*_f32\b", gcn))
        return f"AMDGCN: fma/mad f32 instructions: {fma}"
    return "not available (Triton's interpreter, or no PTX or AMDGCN in the compiled kernel)"


def blocks_of(t: torch.Tensor, n: int) -> torch.Tensor:
    """A per-block value repeated for each of the n values."""
    return t.repeat_interleave(C.BLOCK)[:n]


def split(payload: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    nb = C._blocks(n)
    return payload[:4 * nb].view(torch.float32), payload[4 * nb:4 * nb + n].to(torch.int32)


def bits32(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def hex32(value: torch.Tensor) -> str:
    return f"0x{int(value.view(torch.int32)) & 0xFFFFFFFF:08x}"


def show32(value: torch.Tensor) -> str:
    return f"{float(value):<16.9g} {hex32(value)}"


def show_input(value: torch.Tensor) -> str:
    size = value.element_size()
    bits = int(value.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[size]))
    digits = 17 if size == 8 else 9
    return (f"{float(value):<16.{digits}g} 0x{bits & (2**(8 * size) - 1):0{2 * size}x}"
            f" ({str(value.dtype)[6:]})")


def kind32(value: float) -> str:
    """Where a value falls in float32: zero, subnormal, normal, inf, nan or beyond."""
    if value != value:
        return "nan"
    if abs(value) == float("inf"):
        return "inf"
    if value == 0:
        return "zero"
    if abs(value) < MIN_NORMAL:
        return "SUBNORMAL"
    return "normal" if abs(value) <= FLT_MAX else "beyond float32's range"


def diagnose(dtype: torch.dtype, n: int, limit: int, device: str) -> bool:
    """Prints one case; returns whether the device matched the reference everywhere."""
    print(f"\n== fp8 with error feedback, {str(dtype)[6:]}, n={n} ==")
    x = awkward(n, dtype, n)
    prior = prior_residual(n)
    nb = C._blocks(n)

    def on(t: torch.Tensor) -> torch.Tensor:  # a copy: on CPU, .to() would return t itself
        return t.to(device, copy=True)

    # The reference on CPU, and its intermediates (the same helpers it uses).
    ref_p = torch.zeros(C.payload_nbytes("fp8", n), dtype=torch.uint8)
    ref_r = prior.clone()
    C.quantize_reference("fp8", x, ref_r, ref_p)
    v, finite = C._prepare(x, prior)
    padded = torch.zeros(nb * C.BLOCK)
    padded[:n] = torch.where(finite, v, 0.0)
    amax = blocks_of(padded.abs().view(nb, C.BLOCK).amax(dim=1), n)
    ref_scales, ref_codes = split(ref_p, n)
    scale = blocks_of(ref_scales, n)
    k = ((bits32(scale) >> 23) & 0xFF) - 127
    q = torch.where(finite, v, 0.0) * C._pow2(-k)
    decoded = C._decode_e4m3(ref_codes)
    sent = decoded * scale  # rounded to float32, as the reference does
    fused = torch.where(finite, (v.double() - decoded.double() * scale.double()).float(), 0.0)
    # In float64 the difference is exact here: sent is v rounded to 4 significant
    # bits, or 0, so v - decoded * scale needs at most 25 bits. It is then rounded once.

    # The kernel, exactly as the test calls it.
    dev_p = torch.zeros_like(ref_p, device=device)
    dev_r = on(prior)
    K.quantize("fp8", on(x), dev_r, dev_p)
    dev_p, dev_r = dev_p.cpu(), dev_r.cpu()
    dev_scales, dev_codes = split(dev_p, n)

    # The kernel without multiply-add contraction (a launch option, the same source).
    nofuse_p = torch.zeros_like(ref_p, device=device)
    nofuse_r = on(prior)
    try:
        nofuse = launch(on(x), nofuse_r, nofuse_p, enable_fp_fusion=False)
        nofuse_r = nofuse_r.cpu()
    except Exception as exc:  # an older Triton without the option
        print(f"enable_fp_fusion=False: launch failed: {type(exc).__name__}: {exc}")
        nofuse = nofuse_r = None

    # The compiled kernel's code, from a default launch (cached: the same binary).
    again_r = on(prior)
    compiled = launch(on(x), again_r, torch.zeros_like(ref_p, device=device))

    # The probe: v and decoded * scale on the device.
    dev_v = torch.empty(n, device=device)
    dev_sent = torch.empty(n, device=device)
    scales_d, codes_d = on(dev_scales), on(dev_p[4 * nb:4 * nb + n])
    _probe[(nb,)](on(x).reshape(-1), on(prior), scales_d, codes_d, dev_v,
                  dev_sent, n, BLOCK_SIZE=C.BLOCK)
    dev_v, dev_sent = dev_v.cpu(), dev_sent.cpu()

    same_codes = torch.equal(ref_codes, dev_codes)
    same_scales = torch.equal(bits32(ref_scales), bits32(dev_scales))
    same_payload = torch.equal(ref_p, dev_p)
    r_diff = (bits32(ref_r) != bits32(dev_r)).nonzero().flatten()
    dev_scale = blocks_of(dev_scales, n)
    # Shown below: every value whose residual, code or scale differs.
    diff = ((bits32(ref_r) != bits32(dev_r)) | (ref_codes != dev_codes)
            | (bits32(scale) != bits32(dev_scale))).nonzero().flatten()
    codes = "identical" if same_codes else f"differ at {int((ref_codes != dev_codes).sum())}"
    print(f"payload: {'identical' if same_payload else 'DIFFERS'} (codes {codes},"
          f" scales {'identical' if same_scales else 'differ'}, {nb} block(s))")
    print(f"residual: {len(r_diff)} of {n} differ in their bits"
          + (f"; indices {r_diff[:limit].tolist()}" if len(r_diff) else ""))
    if not torch.equal(bits32(again_r), bits32(dev_r)):
        print("note: a second default launch gave a different residual from the first")
    if nofuse_r is not None:
        nd = int((bits32(ref_r) != bits32(nofuse_r)).sum())
        print(f"enable_fp_fusion=False: {nd} of {n} residuals differ from the reference")
    vd = (bits32(v) != bits32(dev_v)).nonzero().flatten()
    print(f"probe, v on the device: {len(vd)} of {n} differ from the reference's v"
          + (f" (indices {vd[:limit].tolist()})" if len(vd) else ""))
    sd = (bits32(sent) != bits32(dev_sent)) & ~(sent.isnan() & dev_sent.isnan())
    print(f"probe, decoded * scale on the device: {int(sd.sum())} of {n} differ from the"
          " reference's sent")
    print(f"compiled kernel: {code_facts(compiled)}")
    if nofuse_r is not None:
        print(f"compiled kernel, enable_fp_fusion=False: {code_facts(nofuse)}")
    if len(r_diff):
        as_fused = int((bits32(dev_r[r_diff]) == bits32(fused[r_diff])).sum())
        print(f"of the {len(r_diff)} differing residuals, the device's equals the fused value"
              f" (v - decoded * scale, rounded once) at {as_fused}")
        inexact = finite & (sent.double() != decoded.double() * scale.double())
        print(f"decoded * scale isn't exact in float32 (overflow or underflow) at"
              f" {int(inexact.sum())} of {n} values,"
              f" {int(inexact[r_diff].sum())} of them among the differing residuals")

    for i in diff[:limit].tolist():
        b = i // C.BLOCK
        print(f"\n  index {i} (block {b}), finite: {bool(finite[i])}")
        rows = [
            ("x (input)", show_input(x[i]), float(x[i])),
            ("prior residual", show32(prior[i]), float(prior[i])),
            ("v = float32(x) + prior", show32(v[i]), float(v[i])),
            ("v on the device (probe)", show32(dev_v[i]), float(dev_v[i])),
            ("block amax", show32(amax[i]), float(amax[i])),
            ("scale exponent k", f"{int(k[i])} (scale 2^{int(k[i])} = {float(scale[i]):.9g})",
             None),
            ("scale on the device", show32(dev_scale[i]), float(dev_scale[i])),
            ("q = v * 2^-k", show32(q[i]), float(q[i])),
            ("code (reference)", f"0x{int(ref_codes[i]):02x}", None),
            ("code (device)", f"0x{int(dev_codes[i]):02x}", None),
            ("decoded code", show32(decoded[i]), float(decoded[i])),
            ("sent = decoded * scale", show32(sent[i]), float(sent[i])),
            ("sent on the device (probe)", show32(dev_sent[i]), float(dev_sent[i])),
            ("fused: v - decoded * scale", show32(fused[i]), float(fused[i])),
            ("residual, reference", show32(ref_r[i]), float(ref_r[i])),
            ("residual, device", show32(dev_r[i]), float(dev_r[i])),
        ]
        if nofuse_r is not None:
            rows.append(("residual, device, no fusion", show32(nofuse_r[i]), float(nofuse_r[i])))
        print(f"    {'':<30} {'value':<16} {'bits':<10}  in float32")
        for label, text, value in rows:
            print(f"    {label:<30} {text}" + ("" if value is None else f"  {kind32(value)}"))
        if bits32(dev_r[i]) == bits32(ref_r[i]):
            match = "the reference's"
        elif bits32(dev_r[i]) == bits32(fused[i]):
            match = "the fused value"
        else:
            match = "neither the reference's nor the fused value"
        print(f"    the device's residual is {match}")
    return same_payload and not len(diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", nargs="+", choices=DTYPES, default=["float32", "float64"])
    parser.add_argument("--n", nargs="+", type=int, default=[255],
                        help="sizes; the test runs 1 7 255 256 257 1000 3000 1048576")
    parser.add_argument("--limit", type=int, default=10, help="differing indices shown per case")
    args = parser.parse_args(argv)
    if any(n < 1 for n in args.n):
        parser.error("--n: sizes must be at least 1")
    device = "cuda" if GPU else "cpu"
    if GPU:
        props = torch.cuda.get_device_properties(0)
        where = f"{props.name}, capability {props.major}.{props.minor}"
    else:
        where = "CPU, Triton's interpreter"
    fusion = os.environ.get("TRITON_DEFAULT_FP_FUSION")
    print(f"fp8 residual diagnostic on {where}; torch {torch.__version__}"
          f" (CUDA {torch.version.cuda}, HIP {torch.version.hip}), triton {triton.__version__}"
          + (f", TRITON_DEFAULT_FP_FUSION={fusion}" if fusion is not None else ""))
    ok = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # as the test does: inf - inf in masked-out lanes
        for name in args.dtype:
            for n in args.n:
                ok = diagnose(DTYPES[name], n, args.limit, device) and ok
    print("\nall payloads and residuals match the reference" if ok else
          "\nsome payloads or residuals differ from the reference (above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
