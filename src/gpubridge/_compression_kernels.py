"""Triton kernels for bridge compression: one source for CUDA and ROCm builds.

Imported only when a GPU tensor is compressed (or ``GPUBRIDGE_COMPRESSION_KERNELS=triton``),
so gpubridge never needs Triton otherwise. Each kernel computes exactly what
its reference in :mod:`gpubridge.compression` does, bit for bit: the same
integer arithmetic on float32 bit patterns, power-of-two scales, and float32
additions in the same order. No vendor fp8 type is used, since NVIDIA and AMD
GPUs implement different fp8 variants.

On CPU, Triton's interpreter runs them (``TRITON_INTERPRET=1``, set before
Triton is first imported); the tests compare them with the reference that way.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpubridge.compression import BLOCK, payload_nbytes

#: Values per program for bf16, which has no scale blocks.
BF16_BLOCK = 1024


@triton.jit
def _pow2(exponent):
    """2.0**exponent as float32, for exponents in [-126, 127]."""
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _prepare(x_ptr, res_ptr, offs, mask, EF: tl.constexpr):
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if EF:
        v = v + tl.load(res_ptr + offs, mask=mask, other=0.0)
    finite = (v == v) & (tl.abs(v) != float("inf"))
    tiny = finite & (tl.abs(v) < 7.888609052210118e-31)  # 2^-100
    return tl.where(tiny, 0.0, v), finite


@triton.jit
def _encode_e4m3(q):
    sign = tl.where(q < 0, 0x80, 0)
    a = tl.abs(q)
    bits = a.to(tl.int32, bitcast=True)
    exp = bits >> 23
    mant = bits & 0x7FFFFF
    keep = mant >> 20
    rem = mant & 0xFFFFF
    up = (rem > 0x80000) | ((rem == 0x80000) & ((keep & 1) == 1))
    m = keep + up.to(tl.int32)
    normal = ((exp - 120 + (m >> 3)) << 3) | (m & 7)
    y = a * 512.0
    whole = y.to(tl.int32)
    frac = y - whole.to(tl.float32)
    up = (frac > 0.5) | ((frac == 0.5) & ((whole & 1) == 1))
    sub = whole + up.to(tl.int32)
    return tl.where(exp >= 121, normal, sub) | sign


@triton.jit
def _decode_e4m3(code):
    exp = (code >> 3) & 0xF
    mant = code & 7
    normal = (mant + 8).to(tl.float32) * _pow2(exp - 10)
    sub = mant.to(tl.float32) * 0.001953125  # 2^-9
    value = tl.where(exp == 0, sub, normal)
    value = tl.where((code & 0x80) != 0, -value, value)
    return tl.where((code & 0x7F) == 0x7F, float("nan"), value)


@triton.jit
def _bf16_bits(v):
    """bfloat16 bits (int32, 0..0xFFFF) of float32 values, rounded to nearest even; NaN
    becomes 0x7FC0. Integer arithmetic, so every GPU and the interpreter agree."""
    bits = v.to(tl.int32, bitcast=True)
    code = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return tl.where(v != v, 0x7FC0, code)


@triton.jit
def _store(out_ptr, offs, acc, mask, OUT_BF16: tl.constexpr):
    if OUT_BF16:  # out_ptr is the output viewed as int16
        tl.store(out_ptr + offs, _bf16_bits(acc).to(tl.int16), mask=mask)
    else:
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _quantize_fp8(x_ptr, res_ptr, scale_ptr, code_ptr, n, BLOCK_SIZE: tl.constexpr,
                  EF: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    v, finite = _prepare(x_ptr, res_ptr, offs, mask, EF)
    kept = tl.where(finite & mask, v, 0.0)
    amax = tl.max(tl.abs(kept), axis=0)
    bits = amax.to(tl.int32, bitcast=True)
    k = ((bits >> 23) & 0xFF) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    k = tl.where(amax == 0, 0, k)
    scale = _pow2(k)
    code = tl.where(finite, _encode_e4m3(kept * _pow2(-k)), 0x7F)
    tl.store(code_ptr + offs, code.to(tl.uint8), mask=mask)
    tl.store(scale_ptr + pid, scale)
    if EF:
        sent = _decode_e4m3(code) * scale
        tl.store(res_ptr + offs, tl.where(finite, v - sent, 0.0), mask=mask)


@triton.jit
def _quantize_bf16(x_ptr, res_ptr, code_ptr, n, BLOCK_SIZE: tl.constexpr, EF: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    v, finite = _prepare(x_ptr, res_ptr, offs, mask, EF)
    code = _bf16_bits(v)
    tl.store(code_ptr + offs, code.to(tl.int16), mask=mask)
    if EF:
        sent = (code << 16).to(tl.float32, bitcast=True)
        tl.store(res_ptr + offs, tl.where(finite, v - sent, 0.0), mask=mask)


@triton.jit
def _dequantize_fp8(scale_ptr, code_ptr, out_ptr, n, members, scale_stride, code_stride,
                    BLOCK_SIZE: tl.constexpr, OUT_BF16: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(members):
        code = tl.load(code_ptr + i * code_stride + offs, mask=mask, other=0).to(tl.int32)
        scale = tl.load(scale_ptr + i * scale_stride + pid)
        acc = acc + _decode_e4m3(code) * scale
        acc = tl.where(tl.abs(acc) < 7.888609052210118e-31, 0.0, acc)
    _store(out_ptr, offs, acc, mask, OUT_BF16)


@triton.jit
def _dequantize_bf16(code_ptr, out_ptr, n, members, code_stride, BLOCK_SIZE: tl.constexpr,
                     OUT_BF16: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(members):
        code = tl.load(code_ptr + i * code_stride + offs, mask=mask, other=0).to(tl.int32)
        acc = acc + ((code & 0xFFFF) << 16).to(tl.float32, bitcast=True)
        acc = tl.where(tl.abs(acc) < 7.888609052210118e-31, 0.0, acc)
    _store(out_ptr, offs, acc, mask, OUT_BF16)


def quantize(mode: str, x: torch.Tensor, residual: torch.Tensor | None,
             payload: torch.Tensor) -> None:
    """Same contract as :func:`gpubridge.compression.quantize_reference`."""
    n = x.numel()
    if n == 0:
        return
    flat = x.reshape(-1)
    ef = residual is not None
    res = residual if ef else flat  # never read or written without error feedback
    if mode == "bf16":
        codes = payload[:2 * n].view(torch.int16)
        _quantize_bf16[(triton.cdiv(n, BF16_BLOCK),)](flat, res, codes, n,
                                                      BLOCK_SIZE=BF16_BLOCK, EF=ef)
        payload[2 * n:].zero_()
    elif mode == "fp8":
        nb = triton.cdiv(n, BLOCK)
        scales = payload[:4 * nb].view(torch.float32)
        codes = payload[4 * nb:4 * nb + n]
        _quantize_fp8[(nb,)](flat, res, scales, codes, n, BLOCK_SIZE=BLOCK, EF=ef)
        payload[4 * nb + n:].zero_()
    else:
        raise ValueError(f"unknown compression mode {mode!r}")


def dequantize_accumulate(mode: str, payloads: torch.Tensor, out: torch.Tensor) -> None:
    """Same contract as :func:`gpubridge.compression.dequantize_accumulate_reference`."""
    n = out.numel()
    if n == 0:
        return
    members, row = payloads.shape
    assert row == payload_nbytes(mode, n) and payloads.is_contiguous()
    flat = out.view(-1)
    out_bf16 = out.dtype == torch.bfloat16
    if out_bf16:  # rounded with integer arithmetic, written as bits
        flat = flat.view(torch.int16)
    if mode == "bf16":
        codes = payloads.view(torch.int16)  # (members, row / 2)
        _dequantize_bf16[(triton.cdiv(n, BF16_BLOCK),)](codes, flat, n, members, row // 2,
                                                        BLOCK_SIZE=BF16_BLOCK, OUT_BF16=out_bf16)
    elif mode == "fp8":
        nb = triton.cdiv(n, BLOCK)
        scales = payloads.view(torch.float32)  # (members, row / 4): the first nb are scales
        codes = payloads[:, 4 * nb:]
        _dequantize_fp8[(nb,)](scales, codes, flat, n, members, row // 4, row,
                               BLOCK_SIZE=BLOCK, OUT_BF16=out_bf16)
    else:
        raise ValueError(f"unknown compression mode {mode!r}")
