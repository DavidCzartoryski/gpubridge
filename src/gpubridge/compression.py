"""Opt-in bridge compression: fewer bytes between island leaders, at a cost in precision.

``GPUBRIDGE_BRIDGE_COMPRESSION=bf16`` or ``fp8`` (off by default, the same on
every rank: discovery checks) makes the bridged policies compress the bridge
step of floating-point SUM and AVG reductions. Island steps stay in full
precision, and every other op and dtype runs uncompressed. Design, numerics
and error bounds: docs/COMPRESSION.md.

How a compressed bridge step runs, on each bridge member (an island leader, or
a rank of a sharded link):

1. **compress** (its own phase): quantize this island's sum, plus the
   residual this rank kept from the last call on the same buffer (error
   feedback), into a byte payload, on the tensor's device.
2. **bridge**: every member broadcasts its payload in turn, so each ends up
   with every island's payload. Payloads are uint8 tensors, since Gloo has no
   fp8 dtype. The copies to and from host memory, if any, happen here too.
3. **decompress** (its own phase): dequantize every payload and add them in
   fp32, in member order, into the output.

Every member dequantizes the same bytes and adds them in the same order, and
every operation in that step is exact or correctly rounded (power-of-two
scales, codes computed with integer arithmetic). So every rank of every island
still ends with the same bits, whichever vendor's GPU (or the PyTorch
fallback) ran the kernels. Nothing is re-quantized after the exchange, so the
error doesn't compound with the number of islands.

Formats:

- ``bf16``: each value rounded to bfloat16 (round to nearest even). Applies to
  float32 and float64, where it halves or quarters the bytes.
- ``fp8``: OCP FP8 E4M3 codes with one power-of-two scale per block of
  :data:`BLOCK` values, from the block's largest magnitude. Applies to float16,
  bfloat16, float32 and float64. The payload is the scales (float32) followed
  by the codes.

Kernels are Triton (:mod:`gpubridge._compression_kernels`, one source for
CUDA and ROCm builds), used for GPU tensors when Triton is installed. The
functions here are the plain-PyTorch reference they are tested against, and
what runs on CPU (simulation mode) or where Triton is missing.
"""

from __future__ import annotations

import os
import threading
import warnings
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from gpubridge.observe import phase

if TYPE_CHECKING:
    from gpubridge.transport import BridgeTransport

#: Values per fp8 scale.
BLOCK = 256
#: The largest finite E4M3 value.
FP8_MAX = 448.0
#: Magnitudes below this (2^-100) count as zero, before quantizing and after every
#: addition when dequantizing. It keeps every value the kernels handle a normal
#: float32, so GPUs that flush subnormals to zero and those that don't agree.
TINY = 2.0**-100

#: What each mode compresses: the floating-point dtypes it makes smaller.
APPLIES_TO: dict[str, frozenset[torch.dtype]] = {
    "bf16": frozenset({torch.float32, torch.float64}),
    "fp8": frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64}),
}
MODES = tuple(APPLIES_TO)

#: Local, testing and debugging only: ``triton`` or ``torch`` forces one
#: implementation of the kernels (``auto``, the default, uses Triton for GPU
#: tensors when it is installed). Both give the same bits, so ranks may differ.
KERNELS_ENV = "GPUBRIDGE_COMPRESSION_KERNELS"
_KERNEL_CHOICES = ("auto", "triton", "torch")


def applies(mode: str | None, dtype: torch.dtype) -> bool:
    """Whether ``mode`` compresses a SUM of ``dtype`` (AVG is a SUM to policies)."""
    return bool(mode) and dtype in APPLIES_TO[mode]  # type: ignore[index]


def payload_nbytes(mode: str, numel: int) -> int:
    """Bytes one member sends for ``numel`` values: a multiple of 4, so rows of a
    gathered buffer can be viewed as float32 (scales) or int16 (bf16 codes)."""
    if mode == "bf16":
        return _round4(2 * numel)
    return 4 * _blocks(numel) + _round4(numel)


def _blocks(numel: int) -> int:
    return -(-numel // BLOCK)


def _round4(n: int) -> int:
    return -(-n // 4) * 4


# ---- the reference codec, in plain PyTorch --------------------------------------------------
#
# Integer arithmetic on float32 bit patterns, so it gives the same bits on any
# device and matches the Triton kernels bit for bit.

def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int32)


def _float(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """2.0**exponent as float32, for exponents in the normal range [-126, 127]."""
    return _float((exponent + 127) << 23)


def _encode_e4m3(q: torch.Tensor) -> torch.Tensor:
    """E4M3 codes (int32 0..255) of finite float32 values with |q| <= 448, rounded to
    nearest even."""
    sign = torch.where(q < 0, 0x80, 0)
    a = q.abs()
    bits = _bits(a)
    exp, mant = bits >> 23, bits & 0x7FFFFF
    # Normal codes (|q| >= 2^-6): keep 3 mantissa bits, round half to even.
    keep, rem = mant >> 20, mant & 0xFFFFF
    up = (rem > 0x80000) | ((rem == 0x80000) & ((keep & 1) == 1))
    m = keep + up.to(torch.int32)
    normal = ((exp - 120 + (m >> 3)) << 3) | (m & 7)
    # Subnormal codes: the multiple of 2^-9, rounded half to even (8 is the
    # smallest normal code, 2^-6, so it needs no special case).
    y = a * 512.0
    whole = y.to(torch.int32)
    frac = y - whole.to(torch.float32)
    up = (frac > 0.5) | ((frac == 0.5) & ((whole & 1) == 1))
    sub = whole + up.to(torch.int32)
    return torch.where(exp >= 121, normal, sub) | sign


def _decode_e4m3(code: torch.Tensor) -> torch.Tensor:
    """float32 values of E4M3 codes (int32 0..255); 0x7F and 0xFF are NaN."""
    exp, mant = (code >> 3) & 0xF, code & 7
    normal = (mant + 8).to(torch.float32) * _pow2(exp - 10)
    sub = mant.to(torch.float32) * 2.0**-9
    value = torch.where(exp == 0, sub, normal)
    value = torch.where((code & 0x80) != 0, -value, value)
    return torch.where((code & 0x7F) == 0x7F, float("nan"), value)


def _prepare(x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """x + residual in float32, tiny values zeroed; and which values are finite."""
    v = x.reshape(-1).to(torch.float32)
    if residual is not None:
        v = v + residual
    finite = torch.isfinite(v)
    return torch.where(finite & (v.abs() < TINY), 0.0, v), finite


def _bf16_bits(v: torch.Tensor) -> torch.Tensor:
    """bfloat16 bits (int32, 0..0xFFFF) of float32 values, rounded to nearest even; NaN
    becomes 0x7FC0."""
    bits = _bits(v)
    code = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return torch.where(torch.isnan(v), 0x7FC0, code)


def quantize_reference(mode: str, x: torch.Tensor, residual: torch.Tensor | None,
                       payload: torch.Tensor) -> None:
    """Compress ``x`` (plus ``residual``) into ``payload``; leave the new residual.

    ``payload`` is a uint8 tensor of :func:`payload_nbytes` bytes, padding
    included (zeroed). ``residual``
    (float32, ``x.numel()`` values, or None for no error feedback) ends holding
    what this payload leaves out of each finite value; 0 for values that aren't
    finite, which are sent as NaN (fp8) or as themselves (bf16).
    """
    n = x.numel()
    v, finite = _prepare(x, residual)
    if mode == "bf16":
        code = _bf16_bits(v)
        payload[:2 * n].view(torch.int16).copy_(code.to(torch.int16))
        payload[2 * n:].zero_()  # padding: the same bytes on the wire every time
        sent = _float(code << 16)
    elif mode == "fp8":
        nb = _blocks(n)
        padded = torch.zeros(nb * BLOCK, dtype=torch.float32, device=v.device)
        padded[:n] = torch.where(finite, v, 0.0)
        amax = padded.abs().view(nb, BLOCK).amax(dim=1)
        bits = _bits(amax)
        # The smallest power of two with amax / scale <= 448 (= 1.75 * 2^8).
        k = ((bits >> 23) & 0xFF) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
        k = torch.where(amax == 0, 0, k)
        scale = _pow2(k)
        q = (padded.view(nb, BLOCK) * _pow2(-k)[:, None]).view(-1)[:n]
        code = torch.where(finite, _encode_e4m3(q), 0x7F)
        payload[:4 * nb].view(torch.float32).copy_(scale)
        payload[4 * nb:4 * nb + n].copy_(code.to(torch.uint8))
        payload[4 * nb + n:].zero_()
        sent = (_decode_e4m3(code).view(-1) * scale.repeat_interleave(BLOCK)[:n])
    else:
        raise ValueError(f"unknown compression mode {mode!r}")
    if residual is not None:
        residual.copy_(torch.where(finite, v - sent, 0.0))


def _dequantize_row(mode: str, row: torch.Tensor, n: int) -> torch.Tensor:
    if mode == "bf16":
        code = row[:2 * n].view(torch.int16).to(torch.int32) & 0xFFFF
        return _float(code << 16)
    nb = _blocks(n)
    scale = row[:4 * nb].view(torch.float32)
    code = row[4 * nb:4 * nb + n].to(torch.int32)
    return _decode_e4m3(code) * scale.repeat_interleave(BLOCK)[:n]


def dequantize_accumulate_reference(mode: str, payloads: torch.Tensor, out: torch.Tensor) -> None:
    """Sum the members' payloads (rows of ``payloads``) into ``out``, in row order.

    Dequantized values are added in float32, values below :data:`TINY` zeroed
    after every addition, then rounded once to ``out``'s dtype (to bfloat16 with
    integer arithmetic, as the kernels do).
    """
    n = out.numel()
    acc = torch.zeros(n, dtype=torch.float32, device=out.device)
    for row in payloads:
        acc = acc + _dequantize_row(mode, row, n)
        acc = torch.where(acc.abs() < TINY, 0.0, acc)
    if out.dtype == torch.bfloat16:
        out.view(-1).view(torch.int16).copy_(_bf16_bits(acc).to(torch.int16))
    else:
        out.view(-1).copy_(acc.to(out.dtype))


# ---- dispatch ----------------------------------------------------------------------------

_warned = False


def kernels_setting() -> str:
    """``GPUBRIDGE_COMPRESSION_KERNELS``: ``auto`` (default), ``triton`` or ``torch``."""
    value = os.environ.get(KERNELS_ENV, "").strip().lower() or "auto"
    if value not in _KERNEL_CHOICES:
        raise ValueError(f"{KERNELS_ENV}={os.environ[KERNELS_ENV]!r} is not valid; use one of "
                         f"{', '.join(_KERNEL_CHOICES)}")
    return value


def _triton_kernels(tensor: torch.Tensor):
    """The Triton kernel module to use for ``tensor``, or None for the PyTorch reference."""
    global _warned
    choice = kernels_setting()
    if choice == "torch" or (choice == "auto" and tensor.device.type != "cuda"):
        return None
    try:
        from gpubridge import _compression_kernels
    except ImportError as exc:
        if choice == "triton":
            raise RuntimeError(f"{KERNELS_ENV}=triton, but Triton can't be imported: {exc}") from exc
        if not _warned:
            _warned = True
            warnings.warn("gpubridge bridge compression: Triton isn't installed, so it runs "
                          "slower PyTorch kernels (same results).", RuntimeWarning, stacklevel=3)
        return None
    return _compression_kernels


def quantize(mode: str, x: torch.Tensor, residual: torch.Tensor | None,
             payload: torch.Tensor) -> None:
    """:func:`quantize_reference`, on Triton for GPU tensors when it is available."""
    kernels = _triton_kernels(x)
    if kernels is None:
        quantize_reference(mode, x, residual, payload)
    else:
        kernels.quantize(mode, x, residual, payload)


def dequantize_accumulate(mode: str, payloads: torch.Tensor, out: torch.Tensor) -> None:
    """:func:`dequantize_accumulate_reference`, on Triton for GPU tensors when available."""
    kernels = _triton_kernels(out)
    if kernels is None:
        dequantize_accumulate_reference(mode, payloads, out)
    else:
        kernels.dequantize_accumulate(mode, payloads, out)


# ---- error feedback state and the bridge exchange -----------------------------------------

Key = tuple


def buffer_key(tensor: torch.Tensor, part: int = 0) -> Key:
    """Error feedback's key for ``tensor``: its device, address, size and dtype, and which
    ``part`` of it (a sharded link's segment)."""
    return (str(tensor.device), tensor.data_ptr(), tensor.numel(), tensor.dtype, part)


_compressors: weakref.WeakSet[BridgeCompressor] = weakref.WeakSet()


def reset_error_feedback() -> None:
    """Forget every residual on this rank.

    Error feedback keeps one residual per buffer, found by the buffer's
    address, size and dtype. Training loops and DDP reuse their gradient
    buffers, so a residual reaches the next step on the same data; call this
    after reallocating them, so a new buffer at an old address doesn't
    inherit an old residual. ``destroy()`` clears residuals too.
    """
    for compressor in list(_compressors):
        compressor.reset()


class BridgeCompressor:
    """Compresses one bridge member's side of a bridge step. One per policy instance."""

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown bridge compression mode {mode!r}; use one of "
                             f"{', '.join(MODES)}")
        self.mode = mode
        self._residuals: dict[Key, torch.Tensor] = {}
        self._lock = threading.Lock()  # async collectives run on gpubridge's worker thread
        _compressors.add(self)

    def applies(self, dtype: torch.dtype) -> bool:
        return applies(self.mode, dtype)

    def residual(self, key: Key, like: torch.Tensor) -> torch.Tensor:
        """The float32 residual kept for ``key`` (zeros the first time, or if the last one
        was for a different size or device: one buffer used two ways)."""
        with self._lock:
            found = self._residuals.get(key)
            if found is None or found.numel() != like.numel() or found.device != like.device:
                found = self._residuals[key] = torch.zeros(like.numel(), dtype=torch.float32,
                                                           device=like.device)
            return found

    def reset(self) -> None:
        with self._lock:
            self._residuals.clear()

    def compress(self, segment: torch.Tensor, key: Key, members: Sequence[int],
                 me: int) -> torch.Tensor:
        """Step 1: a (members, payload) uint8 buffer on ``segment``'s device, with this
        member's row filled in."""
        gathered = torch.empty((len(members), payload_nbytes(self.mode, segment.numel())),
                               dtype=torch.uint8, device=segment.device)
        with phase("compress"):
            quantize(self.mode, segment, self.residual(key, segment), gathered[me])
        return gathered

    def decompress(self, gathered: torch.Tensor, segment: torch.Tensor) -> None:
        """Step 3: the sum of every member's row, into ``segment``."""
        with phase("decompress"):
            dequantize_accumulate(self.mode, gathered, segment)

    def all_reduce(self, segment: torch.Tensor, key: Key, bridge: BridgeTransport,
                   members: Sequence[int], *, rank: int, direct: bool = False) -> None:
        """Replace ``segment`` (this member's island sum) with every member's, compressed.

        ``members``: the bridge's global ranks in ascending order, the same on
        every member. ``direct``: the transport takes device tensors, so no
        host copies.
        """
        me = list(members).index(rank)
        gathered = self.compress(segment, key, members, me)
        with phase("bridge"):
            rows = gathered if direct else host_rows(gathered, me)
            for i, src in enumerate(members):
                bridge.broadcast(rows[i], src=src)
            if rows is not gathered:
                for i in range(len(members)):
                    if i != me:
                        gathered[i].copy_(rows[i])
        self.decompress(gathered, segment)


def _host_rows(gathered: torch.Tensor, me: int) -> torch.Tensor:
    """Host memory for the exchange: ``gathered`` itself on CPU; on a GPU, a host buffer
    holding this member's row."""
    if gathered.device.type == "cpu":
        return gathered
    rows = torch.empty(gathered.shape, dtype=gathered.dtype)
    rows[me].copy_(gathered[me])
    return rows


#: Tests swap this for one that always copies, to run the GPU path's copies on CPU.
host_rows: Callable[[torch.Tensor, int], torch.Tensor] = _host_rows
