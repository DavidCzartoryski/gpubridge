"""Opt-in bridge compression (GPUBRIDGE_BRIDGE_COMPRESSION, docs/COMPRESSION.md).

The codec itself: its fp8 codes against PyTorch's own float8_e4m3fn, its stated
error bounds, non-finite and tiny values, and error feedback over many steps.
Then whole jobs in simulation mode: every bridged policy compressed, compared
bit for bit with a local model of the compressed exchange (island sums,
quantized once each, dequantized and added in leader order) and against the
exact sum within the stated bounds; every rank must end with the same bits.
The Triton kernels are compared with this reference in test_compression_kernels.py.
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from functools import partial

import pytest
import torch
from _harness import rank_tensor, run_failing_init, run_ranks
from _staging import LazyStaging
from torch.distributed import ReduceOp

import gpubridge
from gpubridge import compression as C
from gpubridge.config import BRIDGE_COMPRESSION_ENV, CHUNK_BYTES_ENV, bridge_compression_setting
from gpubridge.policies import FlatGloo

BRIDGED = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge")
#: (vendors, island labels or None): 2, 3 and 4 islands. More than two islands
#: need relabelled ranks, as gpubridge makes one island per vendor.
LAYOUTS = {
    "2+2": (["nvidia", "nvidia", "amd", "amd"], None),
    "1+3": (["amd", "nvidia", "nvidia", "nvidia"], None),
    "three-islands": (["nvidia"] * 5, ["nvidia", "amd", "third", "nvidia", "amd"]),
    "four-islands": (["nvidia"] * 4, ["nvidia", "amd", "third", "fourth"]),
}
HAS_TRITON = importlib.util.find_spec("triton") is not None


# ---- the codec ------------------------------------------------------------------------------

def test_fp8_codes_match_pytorchs_float8_e4m3fn():
    torch.manual_seed(0)
    q = torch.cat([torch.linspace(-448, 448, 100001), torch.randn(50000) * 10,
                   torch.randn(50000) * 2.0**-7,
                   torch.tensor([0.0, 2.0**-6, 2.0**-9, 2.0**-10, 3 * 2.0**-11, 448.0])])
    mine = C._decode_e4m3(C._encode_e4m3(q))
    assert torch.equal(mine, q.to(torch.float8_e4m3fn).to(torch.float32))
    codes = torch.arange(256, dtype=torch.int32)
    theirs = codes.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    decoded = C._decode_e4m3(codes)
    assert torch.equal(decoded.isnan(), theirs.isnan())
    assert torch.equal(decoded.nan_to_num(), theirs.nan_to_num())


def test_what_each_mode_compresses():
    for mode in C.MODES:
        for dtype in (torch.int8, torch.uint8, torch.int32, torch.int64):
            assert not C.applies(mode, dtype)
        assert C.applies(mode, torch.float32) and C.applies(mode, torch.float64)
    # bf16 doesn't shrink 16-bit floats, so it leaves them alone; fp8 halves them.
    assert not C.applies("bf16", torch.float16) and not C.applies("bf16", torch.bfloat16)
    assert C.applies("fp8", torch.float16) and C.applies("fp8", torch.bfloat16)
    assert not C.applies("", torch.float32) and not C.applies(None, torch.float32)
    # Bytes on the bridge per member: scales then codes (fp8), codes (bf16).
    assert C.payload_nbytes("fp8", 1000) == 4 * 4 + 1000
    assert C.payload_nbytes("fp8", 1001) == 4 * 4 + 1004
    assert C.payload_nbytes("bf16", 1001) == 2004


def test_the_setting(monkeypatch):
    assert bridge_compression_setting() == ""
    for value, want in [("fp8", "fp8"), (" BF16 ", "bf16"), ("off", ""), ("0", ""),
                        ("none", "")]:
        monkeypatch.setenv(BRIDGE_COMPRESSION_ENV, value)
        assert bridge_compression_setting() == want
    monkeypatch.setenv(BRIDGE_COMPRESSION_ENV, "fp4")
    with pytest.raises(ValueError, match="GPUBRIDGE_BRIDGE_COMPRESSION='fp4' is not valid"):
        bridge_compression_setting()


def _spread(n: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    """Values over many orders of magnitude, a different spread in every block."""
    g = torch.Generator().manual_seed(seed)
    span = 10 if dtype == torch.float16 else 30  # float16 overflows past 65504
    scale = 2.0 ** torch.randint(-span, span, (C._blocks(n),), generator=g).repeat_interleave(
        C.BLOCK)[:n]
    return (torch.randn(n, generator=g) * scale).to(dtype)


def _bound(mode: str, v: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
    """The stated per-value error of one quantization (docs/COMPRESSION.md):
    fp8 2^-4 |v| + 2^-10 s with s the block's scale, bf16 2^-8 |v|."""
    v = v.to(torch.float32).abs()
    if mode == "bf16":
        return v * 2.0**-8
    n = v.numel()
    scale = payload[:4 * C._blocks(n)].view(torch.float32).repeat_interleave(C.BLOCK)[:n]
    return v * 2.0**-4 + scale * 2.0**-10


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 255, 256, 1000])
def test_one_quantization_stays_within_its_stated_bound(mode, dtype, n):
    if not C.applies(mode, dtype):
        pytest.skip(f"{mode} doesn't compress {dtype}")
    x = _spread(n, n, dtype)
    payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
    residual = torch.zeros(n)
    C.quantize_reference(mode, x, residual, payload)
    out = torch.empty(n, dtype=torch.float32)
    C.dequantize_accumulate_reference(mode, payload[None], out)
    error = (out - x.to(torch.float32)).abs()
    assert (error <= _bound(mode, x, payload)).all()
    # The residual is exactly what the payload left out.
    assert torch.equal(residual, x.to(torch.float32) - out)
    if mode == "fp8":
        nb = C._blocks(n)
        scale = payload[:4 * nb].view(torch.float32)
        amax = torch.zeros(nb * C.BLOCK)
        amax[:n] = x.to(torch.float32).abs()
        amax = amax.view(nb, C.BLOCK).amax(dim=1)
        # Powers of two, as small as fits: no code saturates, and none wastes range.
        assert torch.equal(scale, 2.0 ** torch.log2(scale).round())
        assert ((amax / scale > 224) & (amax / scale <= 448)).all()


@pytest.mark.parametrize("mode", C.MODES)
def test_a_block_of_only_tiny_values_sends_zeros(mode):
    # Below 2^-100 counts as zero before quantizing: such a block's scale exponent
    # would otherwise fall outside float32's normal range, and its bf16 codes would
    # be subnormals that GPUs flushing to zero read differently.
    x = torch.full((300,), 1e-35)
    payload = torch.empty(C.payload_nbytes(mode, 300), dtype=torch.uint8)
    C.quantize_reference(mode, x, None, payload)
    if mode == "fp8":
        scales = payload[:8].view(torch.float32)
        assert torch.equal(scales, torch.ones(2))  # zero blocks keep scale 1
        assert not payload[8:].any()
    else:
        assert not payload.any()


@pytest.mark.parametrize("mode", C.MODES)
def test_non_finite_values_stay_non_finite_and_tiny_ones_become_zero(mode):
    x = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 1e-35, -2.0])
    payload = torch.empty(C.payload_nbytes(mode, 6), dtype=torch.uint8)
    residual = torch.zeros(6)
    C.quantize_reference(mode, x, residual, payload)
    out = torch.empty(6)
    C.dequantize_accumulate_reference(mode, payload[None], out)
    assert out[0] == 1.0 and out[5] == -2.0 and out[4] == 0.0
    assert not out[1:4].isfinite().any()  # AMP's overflow check still sees them
    if mode == "bf16":
        assert out[2] == float("inf") and out[3] == -float("inf")
    assert torch.equal(residual[1:4], torch.zeros(3))  # never poisons the next step


FLT_MAX = torch.finfo(torch.float32).max
#: The largest value each mode sends: 448 * 2^119 in fp8, bfloat16's largest in bf16.
LARGEST = {"fp8": 448 * 2.0**119, "bf16": float(torch.finfo(torch.bfloat16).max)}


def near_the_max() -> torch.Tensor:
    """float32 values at and near float32's largest, on both sides of each mode's
    largest sent value."""
    up = lambda value: torch.nextafter(torch.tensor(value), torch.tensor(math.inf)).item()  # noqa: E731
    values = [FLT_MAX, -FLT_MAX, torch.nextafter(torch.tensor(FLT_MAX), torch.tensor(0.0)).item(),
              3.4e38, -3.4e38, 3.0e38, LARGEST["fp8"], up(LARGEST["fp8"]), -up(LARGEST["fp8"]),
              LARGEST["bf16"], up(LARGEST["bf16"]), 3.3961e38, 1.0, -2.5]
    return torch.tensor(values, dtype=torch.float32)


def _quantize_once(mode, x, residual):
    """The reference's payload, its dequantized values and what it was given (v)."""
    n = x.numel()
    v, finite = C._prepare(x, residual)
    payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
    C.quantize_reference(mode, x, residual, payload)
    sent = torch.empty(n)
    C.dequantize_accumulate_reference(mode, payload[None], sent)
    return payload, sent, v, finite


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_values_near_float32s_max_arrive_finite_and_leave_finite_residuals(mode, dtype):
    x = near_the_max().to(dtype)
    if dtype == torch.float64:  # beyond float32's range: counts as float32's largest
        x = torch.cat([x, torch.tensor([4e38, -1e39, 1e300], dtype=dtype)])
    residual = torch.zeros(x.numel())
    _, sent, v, finite = _quantize_once(mode, x, residual)
    assert finite.all() and sent.isfinite().all() and residual.isfinite().all()
    assert v.abs().max() <= FLT_MAX
    # What crossed plus what stayed behind is exactly v: nothing is lost but the part of a
    # float64 value beyond float32's range.
    assert torch.equal(sent.double() + residual.double(), v.double())
    largest = LARGEST[mode]
    over = v.abs() > largest
    assert over.any()
    assert torch.equal(sent[over], torch.copysign(torch.tensor(largest), v[over]))
    assert (sent.abs() <= largest).all()
    # The stated error beyond the largest value (docs/COMPRESSION.md): up to 2^-3 |v| in
    # fp8, within bf16's usual 2^-8 |v|.
    error = (v.double() - sent.double()).abs()[over]
    assert (error <= v.double().abs()[over] * (2.0**-3 if mode == "fp8" else 2.0**-8)).all()
    if dtype == torch.float64:
        assert torch.equal(v[-3:], torch.tensor([FLT_MAX, -FLT_MAX, FLT_MAX]))


def test_the_fp8_scale_exponent_is_capped_so_448_times_it_fits():
    assert C.MAX_SCALE_EXP == 119
    assert math.isfinite(torch.tensor(448 * 2.0**119, dtype=torch.float32).item())
    assert math.isinf(torch.tensor(448 * 2.0**120, dtype=torch.float32).item())
    x = torch.tensor([3.4e38] + [1.0] * 255)
    residual = torch.zeros(256)
    payload, sent, _, _ = _quantize_once("fp8", x, residual)
    assert payload[:4].view(torch.float32).item() == 2.0**119
    # 448: saturated. The uncapped exponent, 120, sent code 256: 256 * 2^120 = 2^128,
    # inf in float32, and a residual of -inf.
    assert payload[4].item() == 0x7E and sent[0].item() == 448 * 2.0**119
    assert residual[0].item() == torch.tensor(3.4e38).item() - 448 * 2.0**119
    # 1.0 is far below this block's scale: code 0, all of it kept for the next step.
    assert torch.equal(sent[1:], torch.zeros(255)) and torch.equal(residual[1:], torch.ones(255))


def test_a_block_of_float32_subnormals_sends_zeros_and_keeps_no_residual():
    # Every value below 2^-100 counts as zero, so no subnormal reaches a kernel.
    x = torch.tensor([1e-40, -1e-42, 1.4e-45, -2.0**-140, 2.0**-127] * 60)
    for mode in C.MODES:
        residual = torch.zeros(300)
        payload, sent, _, _ = _quantize_once(mode, x, residual)
        assert torch.equal(sent, torch.zeros(300)) and torch.equal(residual, torch.zeros(300))
        if mode == "fp8":
            assert torch.equal(payload[:8].view(torch.float32), torch.ones(2))


def test_a_block_mixing_1e_4_and_3e38():
    # The small values are subnormal after scaling by 2^-119, so they send code 0 and keep
    # all of themselves as residual; 3e38 is beyond 448 * 2^119 and saturates.
    x = torch.tensor([3e38, -3e38] + [1e-4, -1e-4, 2.5e-4, 7e-5] * 63 + [1e-4, 3e-4])
    residual = torch.zeros(256)
    payload, sent, v, _ = _quantize_once("fp8", x, residual)
    scale = payload[:4].view(torch.float32).item()
    assert scale == 2.0**119
    small = slice(2, None)
    q = v[small] * 2.0**-119
    assert (q.abs() < 2.0**-126).all() and (q != 0).all()  # float32 subnormals
    assert torch.equal(sent[small], torch.zeros(254))
    assert torch.equal(residual[small], v[small])
    assert torch.equal(sent[:2], torch.tensor([LARGEST["fp8"], -LARGEST["fp8"]]))
    assert torch.equal(residual[:2], v[:2] - sent[:2])
    assert residual.isfinite().all()


@pytest.mark.parametrize("mode", C.MODES)
def test_error_feedback_stays_finite_step_after_step_at_float32s_max(mode):
    # The same values every step, the largest ones included: v = x + residual saturates
    # at float32's largest, so no residual grows past FLT_MAX - largest, and every step's
    # v is exactly what was sent plus what stayed behind. (A value just above the largest
    # sent one gains its excess every step: error feedback carrying it, up to that bound.)
    torch.manual_seed(0)
    x = torch.cat([near_the_max(), torch.tensor([3e38] + [1e-4] * 100),
                   torch.randn(300) * 2.0 ** torch.randint(-30, 30, (300,))])
    residual = torch.zeros(x.numel())
    for _ in range(50):
        _, sent, v, finite = _quantize_once(mode, x, residual)
        assert finite.all() and sent.isfinite().all() and residual.isfinite().all()
        assert torch.equal(sent.double() + residual.double(), v.double())
        assert (residual.abs() <= FLT_MAX - LARGEST[mode]).all()
    at_max = x.abs() == FLT_MAX
    assert torch.equal(residual[at_max].abs(), torch.full((2,), FLT_MAX - LARGEST[mode]))


def test_dequantize_adds_members_in_order_in_float32():
    n = 300
    rows = []
    for seed in range(3):
        payload = torch.empty(C.payload_nbytes("fp8", n), dtype=torch.uint8)
        C.quantize_reference("fp8", _spread(n, seed), None, payload)
        rows.append(payload)
    out = torch.empty(n, dtype=torch.bfloat16)
    C.dequantize_accumulate_reference("fp8", torch.stack(rows), out)
    acc = torch.zeros(n)
    for row in rows:
        acc = acc + C._dequantize_row("fp8", row, n)
    assert torch.equal(out, acc.to(torch.bfloat16))  # one rounding, at the end
    # Order matters in float32: (2^24 + 1) - 2^24 is 0, (-2^24 + 1) + 2^24 is 1.
    order = []
    for value in (2.0**24, 1.0, -2.0**24):
        payload = torch.empty(C.payload_nbytes("fp8", 1), dtype=torch.uint8)
        C.quantize_reference("fp8", torch.tensor([value]), None, payload)
        order.append(payload)
    out = torch.empty(1)
    C.dequantize_accumulate_reference("fp8", torch.stack(order), out)
    assert out.item() == 0.0
    C.dequantize_accumulate_reference("fp8", torch.stack(order[::-1]), out)
    assert out.item() == 1.0


def _steps(mode: str, steps: int, feedback: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Two islands sending the same values every step. Returns the worst accumulated
    error over the steps, and one step's stated bound."""
    n = 512
    x = [_spread(n, 1) * 0.37, _spread(n, 2) * 0.21]
    residuals = [torch.zeros(n), torch.zeros(n)] if feedback else [None, None]
    total = torch.zeros(n, dtype=torch.float64)
    worst = torch.zeros(n, dtype=torch.float64)
    bound = torch.zeros(n)
    for step in range(1, steps + 1):
        rows = []
        for xi, ri in zip(x, residuals, strict=True):
            payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
            sent = xi + (ri if ri is not None else 0)
            C.quantize_reference(mode, xi, ri, payload)
            rows.append(payload)
            if step == 1:
                bound = bound + _bound(mode, sent, payload)
        out = torch.empty(n)
        C.dequantize_accumulate_reference(mode, torch.stack(rows), out)
        total += out.to(torch.float64)
        exact = step * (x[0].to(torch.float64) + x[1].to(torch.float64))
        worst = torch.maximum(worst, (total - exact).abs())
    return worst, bound


@pytest.mark.parametrize("mode", C.MODES)
def test_error_feedback_keeps_the_accumulated_error_bounded(mode):
    worst, bound = _steps(mode, 500, feedback=True)
    # Sum over steps of what was sent = sum of the values - the last residual, so the
    # accumulated error never exceeds about one step's (2x for the residual's own
    # quantization and float32 rounding along the way).
    assert (worst <= 2 * bound.to(torch.float64) + 1e-30).all()
    # Without error feedback the same error repeats every step and grows with them.
    worst_without, _ = _steps(mode, 500, feedback=False)
    assert worst_without.max() > 100 * worst.max()


def test_a_compressor_keeps_one_residual_per_buffer_and_part():
    compressor = C.BridgeCompressor("fp8")
    a, b = torch.zeros(10), torch.zeros(10)
    ra = compressor.residual(C.buffer_key(a), a)
    assert compressor.residual(C.buffer_key(a), a) is ra
    assert compressor.residual(C.buffer_key(b), b) is not ra
    assert compressor.residual(C.buffer_key(a, 1), a) is not ra  # a sharded segment
    C.reset_error_feedback()
    assert compressor.residual(C.buffer_key(a), a) is not ra
    with pytest.raises(ValueError, match="unknown bridge compression mode"):
        C.BridgeCompressor("fp4")


class _ThreadBridge:
    """A bridge among threads of one process: broadcast through shared memory."""

    def __init__(self, members: int) -> None:
        self.barrier = threading.Barrier(members)
        self.box: dict[int, torch.Tensor] = {}

    def member(self, rank: int):
        bridge = self

        class Member:
            def broadcast(self, tensor, src):
                if rank == src:
                    bridge.box[src] = tensor.clone()
                bridge.barrier.wait()
                if rank != src:
                    tensor.copy_(bridge.box[src])
                bridge.barrier.wait()

        return Member()


@pytest.mark.parametrize("members", [2, 3, 5])
@pytest.mark.parametrize("mode", C.MODES)
def test_more_islands_quantize_once_each_so_the_error_does_not_compound(members, mode):
    # Every island's sum is quantized once and dequantized by every member, so the
    # result's error is the sum of one quantization per island, however many there are.
    n = 700
    sums = [_spread(n, 10 + i) for i in range(members)]
    outs = [s.clone() for s in sums]
    bridge = _ThreadBridge(members)
    ranks = list(range(members))

    def run(i):
        compressor = C.BridgeCompressor(mode)
        compressor.all_reduce(outs[i], C.buffer_key(outs[i]), bridge.member(i), ranks, rank=i)

    threads = [threading.Thread(target=run, args=(i,)) for i in ranks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(torch.equal(out, outs[0]) for out in outs)  # the same bits everywhere
    exact = sum(s.to(torch.float64) for s in sums)
    bound = torch.zeros(n, dtype=torch.float64)
    for s in sums:
        payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
        C.quantize_reference(mode, s, None, payload)
        bound += _bound(mode, s, payload).to(torch.float64)
    bound += 2.0**-23 * (members - 1) * sum(s.abs().to(torch.float64) for s in sums)
    assert ((outs[0].to(torch.float64) - exact).abs() <= bound).all()


# ---- whole jobs in simulation mode ------------------------------------------------------------

def _values(numel: int, dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Per-rank values whose island sums are exact in ``dtype`` whatever the order, so the
    model's island sums match the island reduce bit for bit. Every block of 256 has its
    own power-of-two magnitude, so the scales differ."""
    g = torch.Generator().manual_seed(1000 + rank)
    if dtype == torch.bfloat16:
        v = torch.randint(-32, 33, (numel,), generator=g).to(torch.float64)
    elif dtype == torch.float16:
        v = torch.randint(-256, 257, (numel,), generator=g).to(torch.float64) / 4
    else:
        v = torch.randint(-2**20, 2**20 + 1, (numel,), generator=g).to(torch.float64) / 256
    if dtype in (torch.float32, torch.float64):
        g2 = torch.Generator().manual_seed(7)  # the same magnitudes on every rank
        exps = torch.randint(-12, 12, (C._blocks(numel),), generator=g2).repeat_interleave(
            C.BLOCK)[:numel]
        v = v * 2.0**exps
    return v.to(dtype)


def _model(mode: str, sums: list[torch.Tensor], dtype: torch.dtype, parts: int = 1):
    """The compressed exchange, locally: each island's sum (leader order) quantized once,
    split into ``parts`` padded segments as the sharded bridge does; dequantized and added
    in order. Returns the result and the stated error bound."""
    n = sums[0].numel()
    seg = -(-n // parts)
    padded = [torch.zeros(seg * parts, dtype=dtype) for _ in sums]
    for p, s in zip(padded, sums, strict=True):
        p[:n] = s
    out = torch.empty(seg * parts, dtype=dtype)
    bound = torch.zeros(seg * parts, dtype=torch.float64)
    for j in range(parts):
        rows = []
        for p in padded:
            payload = torch.empty(C.payload_nbytes(mode, seg), dtype=torch.uint8)
            C.quantize_reference(mode, p[j * seg:(j + 1) * seg], torch.zeros(seg), payload)
            rows.append(payload)
            bound[j * seg:(j + 1) * seg] += _bound(mode, p[j * seg:(j + 1) * seg],
                                                    payload).to(torch.float64)
        C.dequantize_accumulate_reference(mode, torch.stack(rows), out[j * seg:(j + 1) * seg])
    magnitude = sum(s.to(torch.float64).abs() for s in sums)
    bound = bound[:n] + 2.0**-23 * (len(sums) - 1) * magnitude
    unit = {torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8}.get(dtype, 2.0**-24)
    return out[:n], bound + unit * (magnitude + bound)


def _island_sums(topology, numel: int, dtype: torch.dtype) -> list[torch.Tensor]:
    return [sum(_values(numel, torch.float64, r) if dtype == torch.float64
                else _values(numel, dtype, r).to(torch.float64) for r in island.ranks)
            .to(dtype) for island in topology.layout.islands]


def _parts(topology) -> int:
    if topology.policy_name != "sharded-bridge":
        return 1
    return min(len(island.ranks) for island in topology.layout.islands)


def _gathered_equal(tensor: torch.Tensor) -> bool:
    out = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(out, tensor)
    return all(torch.equal(o, out[0]) for o in out)


SIZES = (1, 255, 257, 1000)


def _check_member_order(topology, mode):
    """With three or more islands, a sum whose float32 rounding depends on the order:
    every member must add the islands in the same (leader) order."""
    islands = topology.layout.islands
    if len(islands) < 3 or mode != "fp8":
        return
    leaders = {island.leader: value for island, value in zip(islands, (2.0**24, 1.0, -2.0**24))}
    tensor = torch.zeros(16)
    tensor[0] = leaders.get(topology.rank, 0.0)
    C.reset_error_feedback()
    gpubridge.all_reduce(tensor)
    assert _gathered_equal(tensor) and tensor[0].item() == 0.0, tensor[0].item()


def _check_compressed(topology, mode):
    assert topology.bridge_compression == mode
    _check_member_order(topology, mode)
    rank, world = topology.rank, topology.world_size
    parts = _parts(topology)
    for numel in SIZES:
        for dtype in sorted(C.APPLIES_TO[mode], key=str):
            want, bound = _model(mode, _island_sums(topology, numel, dtype), dtype, parts)
            exact = sum(_values(numel, dtype, r).to(torch.float64) for r in range(world))
            for op in (ReduceOp.SUM, ReduceOp.AVG):
                # A new buffer may reuse a freed one's address: start without a residual.
                C.reset_error_feedback()
                tensor = _values(numel, dtype, rank)
                gpubridge.all_reduce(tensor, op=op)
                assert _gathered_equal(tensor), (numel, dtype, op)
                if op == ReduceOp.AVG:
                    assert torch.equal(tensor, want.clone().div_(world)), (numel, dtype)
                    continue
                assert torch.equal(tensor, want), (numel, dtype)
                assert ((tensor.to(torch.float64) - exact).abs() <= bound).all(), (numel, dtype)


def _turn_on(mode, labels, extra, rank):
    os.environ[BRIDGE_COMPRESSION_ENV] = mode
    os.environ.update(extra)
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda vendors: original(labels)


@pytest.mark.parametrize("policy", BRIDGED)
@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_bridged_policy_compresses_the_same_bits_on_every_rank(layout, mode, policy,
                                                                     tmp_path):
    vendors, labels = LAYOUTS[layout]
    run_ranks(vendors, partial(_check_compressed, mode=mode), tmp_path,
              init_kwargs={"policy": policy},
              before_init=partial(_turn_on, mode, labels, {CHUNK_BYTES_ENV: "64"}))


def _check_everything_else_exact(topology, mode):
    rank, world = topology.rank, topology.world_size
    reference = FlatGloo()
    for dtype in (torch.int8, torch.uint8, torch.int32, torch.int64, torch.float16,
                  torch.bfloat16, torch.float32, torch.float64):
        ops = [ReduceOp.MAX, ReduceOp.MIN]
        if not C.applies(mode, dtype):
            ops.append(ReduceOp.SUM)
        for op in ops:
            tensor = rank_tensor((5, 7), dtype, rank)  # sums exact in bfloat16
            want = tensor.clone()
            gpubridge.all_reduce(tensor, op=op)
            reference.all_reduce(want, topology, op=op)
            assert torch.equal(tensor, want), (dtype, op)
        for src in range(world):
            tensor = rank_tensor((5, 7), dtype, rank)  # sums exact in bfloat16
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            reference.broadcast(want, src, topology)
            assert torch.equal(tensor, want), (dtype, src)
    rows = rank_tensor((world, 5), torch.float32, rank)
    got, want = torch.empty((world, 5)), torch.empty((world, 5))
    gpubridge.all_gather_into_tensor(got, rows[rank].contiguous())
    reference.all_gather_into_tensor(want, rows[rank].contiguous(), topology)
    assert torch.equal(got, want)


@pytest.mark.parametrize("policy", BRIDGED)
@pytest.mark.parametrize("mode", C.MODES)
def test_other_ops_and_dtypes_run_uncompressed(mode, policy, tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], partial(_check_everything_else_exact, mode=mode),
              tmp_path, init_kwargs={"policy": policy},
              before_init=partial(_turn_on, mode, None, {}))


def _check_reduce_scatter(topology):
    rank, world = topology.rank, topology.world_size
    rows = _values(world * 300, torch.float32, rank).view(world, 300)
    got = torch.empty(300)
    gpubridge.reduce_scatter_tensor(got, rows)
    want, _ = _model("fp8", _island_sums(topology, world * 300, torch.float32), torch.float32)
    assert torch.equal(got, want.view(world, 300)[rank])


def test_reduce_scatter_compresses_its_bridge_step(tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check_reduce_scatter, tmp_path,
              before_init=partial(_turn_on, "fp8", None, {}))


class _Phases(gpubridge.CollectiveObserver):
    def __init__(self) -> None:
        self.records: list = []

    def on_collective(self, record) -> None:
        self.records.append(record)


def _check_phases(topology):
    seen: list[list] = []
    original = C.quantize, C.dequantize_accumulate

    def quantize(*args):
        seen.append([f.phase for f in gpubridge.in_flight()])
        original[0](*args)

    def dequantize(*args):
        seen.append([f.phase for f in gpubridge.in_flight()])
        original[1](*args)

    C.quantize, C.dequantize_accumulate = quantize, dequantize
    observer = _Phases()
    gpubridge.add_observer(observer)
    gpubridge.all_reduce(_values(1000, torch.float32, topology.rank))
    gpubridge.all_reduce(rank_tensor((1000,), torch.int32, topology.rank))
    member = (topology.is_leader if topology.policy_name != "sharded-bridge"
              else topology.policy.index < topology.policy.k)
    names = [[p.name for p in r.phases] for r in observer.records]
    if topology.policy_name == "sharded-bridge":
        island = ["island-reduce-scatter", "island-all-gather"]
    else:
        island = ["island-reduce", "island-broadcast"]
    if member:
        # Kernel time is never inside "bridge", so a slow kernel never looks like a slow link.
        assert names[0] == [island[0], "compress", "bridge", "decompress", island[1]], names
        assert names[1] == [island[0], "bridge", island[1]], names  # int32: uncompressed
        assert seen == [["compress"], ["decompress"]]
    else:
        assert names == [island, island] and seen == []


@pytest.mark.parametrize("policy", BRIDGED)
def test_compress_and_decompress_are_phases_of_their_own(policy, tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check_phases, tmp_path,
              init_kwargs={"policy": policy}, before_init=partial(_turn_on, "fp8", None, {}))


def _check_feedback(topology, mode):
    rank, world, steps = topology.rank, topology.world_size, 40
    x = _values(1000, torch.float32, rank) / 3  # not exact in fp8 or bf16
    exact = sum(_values(1000, torch.float32, r).to(torch.float64) / 3 for r in range(world))
    sums = [sum(_values(1000, torch.float32, r).to(torch.float64) / 3 for r in island.ranks)
            .to(torch.float32) for island in topology.layout.islands]
    _, bound = _model(mode, sums, torch.float32)  # one step's stated bound
    buffer = torch.empty(1000)  # one buffer, as training reuses its gradients'
    total = torch.zeros(1000, dtype=torch.float64)
    for _ in range(steps):
        buffer.copy_(x)
        gpubridge.all_reduce(buffer)
        total += buffer.to(torch.float64)
    # The accumulated error stays about one step's bound instead of growing with the
    # steps. Only float32 rounding (island sums, the final add) isn't fed back.
    rounding = steps * 2.0**-22 * sum(_values(1000, torch.float32, r).abs().to(torch.float64)
                                      for r in range(world))
    assert ((total - steps * exact).abs() <= 2 * bound + rounding).all()


@pytest.mark.parametrize("mode", C.MODES)
def test_error_feedback_carries_over_between_calls_on_one_buffer(mode, tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], partial(_check_feedback, mode=mode), tmp_path,
              before_init=partial(_turn_on, mode, None, {}))


def _always_copy(gathered, me):
    return gathered.clone()


def _copying(mode, rank):
    _turn_on(mode, None, {CHUNK_BYTES_ENV: "48"}, rank)
    C.host_rows = _always_copy  # the host copies a GPU leader makes
    gpubridge.staging.make_staging = lambda device, chunk, slots: LazyStaging(chunk, slots)


@pytest.mark.parametrize("policy", BRIDGED)
def test_the_gpu_paths_copies_on_cpu(policy, tmp_path):
    # Host rows for the default and sharded bridge, lazily run copies for the
    # pipelined one: a missing wait would leave stale bytes and break the model.
    run_ranks(["nvidia", "nvidia", "amd", "amd"], partial(_check_compressed, mode="fp8"),
              tmp_path, init_kwargs={"policy": policy}, before_init=partial(_copying, "fp8"))


def _check_lazy_queues_drained(topology):
    _check_compressed(topology, "fp8")
    if topology.is_leader:
        for staging in (topology.policy.staging, topology.policy.recv_staging):
            assert isinstance(staging, LazyStaging)
            assert not staging.out.queue and not staging.back.queue, "a copy never ran"


def test_the_compressed_pipeline_waits_for_every_copy(tmp_path):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check_lazy_queues_drained, tmp_path,
              init_kwargs={"policy": "pipelined-reduce-bridge-broadcast"},
              before_init=partial(_copying, "fp8"))


def _check_off(topology):
    assert topology.bridge_compression == ""
    assert topology.policy.compressor is None


def test_compression_is_off_by_default(tmp_path):
    run_ranks(["nvidia", "amd"], _check_off, tmp_path)


def _check_single_island(topology):
    assert topology.bridge_compression == "fp8" and topology.policy_name == "native-only"
    tensor = rank_tensor((300,), torch.float32, topology.rank)
    want = tensor.clone()
    gpubridge.all_reduce(tensor)
    FlatGloo().all_reduce(want, topology)
    assert torch.equal(tensor, want)  # no bridge, nothing to compress


def test_a_single_island_has_no_bridge_to_compress(tmp_path):
    run_ranks(["nvidia", "nvidia"], _check_single_island, tmp_path,
              before_init=partial(_turn_on, "fp8", None, {}))


@pytest.mark.parametrize(("settings", "pattern"), [
    (["fp8", "", "fp8", "fp8"], r"Ranks disagree on GPUBRIDGE_BRIDGE_COMPRESSION: fp8 on ranks "
                                r"\[0, 2, 3\]; off on ranks \[1\]"),
    (["fp8", "bf16", "fp8", "fp8"], "Ranks disagree on GPUBRIDGE_BRIDGE_COMPRESSION"),
    (["fp4", "fp8", "fp8", "fp8"], r"rank 0 .*GPUBRIDGE_BRIDGE_COMPRESSION='fp4' is not valid"),
])
def test_init_refuses_ranks_that_disagree_or_a_bad_value(settings, pattern, tmp_path):
    vendors = ["nvidia", "nvidia", "amd", "amd"]
    setups = [{"env": {"GPUBRIDGE_VENDOR": vendor,
                       **({BRIDGE_COMPRESSION_ENV: value} if value else {})}}
              for vendor, value in zip(vendors, settings, strict=True)]
    run_failing_init(setups, tmp_path, [pattern])


def _kernels_per_rank(rank):
    _turn_on("fp8", None, {}, rank)
    if rank == 0:  # one leader on Triton (interpreted on CPU), the other on PyTorch
        os.environ["TRITON_INTERPRET"] = "1"
        os.environ[C.KERNELS_ENV] = "triton"


def _check_mixed_kernels(topology):
    _check_compressed(topology, "fp8")
    if topology.rank == 0:
        assert C._triton_kernels(torch.empty(1)) is not None


@pytest.mark.skipif(not HAS_TRITON, reason="Triton isn't installed")
def test_triton_and_pytorch_kernels_on_different_leaders_agree_bit_for_bit(tmp_path):
    # As NVIDIA and AMD leaders, or a leader without Triton, would in a mixed job.
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _check_mixed_kernels, tmp_path,
              before_init=_kernels_per_rank)


def test_bound_helpers_are_tight_enough_to_mean_something():
    # The bound is close to the real worst case: a bound 10x too loose would hide bugs.
    x = _spread(4096, 3)
    for mode in C.MODES:
        payload = torch.empty(C.payload_nbytes(mode, x.numel()), dtype=torch.uint8)
        C.quantize_reference(mode, x, None, payload)
        out = torch.empty_like(x)
        C.dequantize_accumulate_reference(mode, payload[None], out)
        ratio = ((out - x).abs() / _bound(mode, x, payload)).max().item()
        assert 0.3 < ratio <= 1.0, (mode, ratio)
        assert math.isfinite(ratio)
