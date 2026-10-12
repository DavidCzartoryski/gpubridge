"""Opt-in pinned staging for reduce-bridge-broadcast (GPUBRIDGE_PINNED_STAGING).

The protocol runs on lazy copies (tests/_staging.py), so a missing wait shows
up as a wrong answer on CPU. The GPU side is in tests/test_gpu.py.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
from _harness import rank_tensor, run_failing_init, run_ranks
from _staging import EagerCopyBack, LazyPinned
from torch.distributed import ReduceOp

import gpubridge
from gpubridge import staging
from gpubridge.config import PINNED_STAGING_ENV, VENDOR_ENV, pinned_staging_setting
from gpubridge.policies import FlatGloo, ReduceBridgeBroadcast

# ---- the protocol ---------------------------------------------------------------------


def test_the_bridge_reads_the_buffer_only_after_the_copy_out():
    pinned = LazyPinned()
    tensor = torch.arange(6, dtype=torch.float32)
    host = pinned.to_host(tensor)
    assert host.tolist() == tensor.tolist()  # the lazy copy ran: to_host waited for it
    assert pinned.log[-3:] == [("copy", "copy1", None), ("wait", "copy1"), ("ran", "copy1")]


def _two_collectives(pinned: LazyPinned, second_stream: str, second_numel: int = 4):
    """A collective whose copy back stays queued, then another on ``second_stream``."""
    first = torch.full((4,), 1.0)
    host = pinned.to_host(first)
    host.add_(10.0)  # what the bridge brings back
    pinned.to_device(first, host)  # queued, not run: the collective returns
    pinned.stream = second_stream
    second = torch.full((second_numel,), 2.0)
    host = pinned.to_host(second)
    host.add_(20.0)
    pinned.to_device(second, host)
    return first, second


@pytest.mark.parametrize("stream", ["caller", "other"])
def test_the_next_collective_waits_for_the_last_copy_back(stream):
    pinned = LazyPinned()
    first, second = _two_collectives(pinned, stream)
    pinned.run_streams()
    assert first.tolist() == [11.0] * 4 and second.tolist() == [22.0] * 4
    # The second copy out waited, on the device, for the first copy back.
    assert ("copy", "copy3", "copy2") in pinned.log


def test_without_that_wait_the_first_result_is_overwritten():
    # Proof that the test above can fail: the second collective's copy out,
    # on another stream, overwrites the buffer before the first copy back ran.
    pinned = LazyPinned()
    copy = pinned._copy
    pinned._copy = lambda dst, src, after: copy(dst, src, None)
    first, second = _two_collectives(pinned, "other")
    pinned.run_streams()
    assert first.tolist() == [22.0] * 4 and second.tolist() == [22.0] * 4


def test_a_larger_tensor_gets_a_new_buffer_after_the_last_copy_back():
    pinned = LazyPinned()
    first = torch.full((4,), 1.0)
    host = pinned.to_host(first)
    host.add_(10.0)
    pinned.to_device(first, host)
    pinned.to_host(torch.full((8,), 2.0))
    # The host waited for the copy back before dropping the old buffer.
    assert first.tolist() == [11.0] * 4
    assert [entry for entry in pinned.log if entry[0] == "allocate"] == [
        ("allocate", 16), ("allocate", 32)]


def test_a_smaller_tensor_reuses_the_buffer():
    pinned = LazyPinned()
    first, second = _two_collectives(pinned, "caller", second_numel=2)
    pinned.run_streams()
    assert first.tolist() == [11.0] * 4 and second.tolist() == [22.0] * 2
    assert sum(entry[0] == "allocate" for entry in pinned.log) == 1


def test_close_waits_for_the_last_copy_back():
    pinned = LazyPinned()
    tensor = torch.full((4,), 1.0)
    host = pinned.to_host(tensor)
    host.add_(10.0)
    pinned.to_device(tensor, host)
    pinned.close()
    assert tensor.tolist() == [11.0] * 4 and pinned.buffer is None


def test_cpu_devices_need_no_staging():
    assert staging.make_bridge_staging(torch.device("cpu")) is None


# ---- reduce-bridge-broadcast with it --------------------------------------------------

def _lazy_staging(rank: int) -> None:
    staging.make_bridge_staging = lambda device: EagerCopyBack()


def _check_pinned(topology, on: bool):
    rank, world = topology.rank, topology.world_size
    policy = topology.policy
    assert isinstance(policy, ReduceBridgeBroadcast) and topology.pinned_staging is on
    pinned = policy.staging_buffer
    if not on or not topology.is_leader:
        assert pinned is None  # off by default, and only leaders cross the bridge
    else:
        assert isinstance(pinned, EagerCopyBack)
    reference = FlatGloo()
    for dtype, shape in [(torch.float32, (37,)), (torch.float16, (5, 7)), (torch.int64, (9,)),
                         (torch.float64, (1,)), (torch.float32, (1000,))]:
        for op in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN):
            tensor = rank_tensor(shape, dtype, rank)
            want = tensor.clone()
            gpubridge.all_reduce(tensor, op=op)
            reference.all_reduce(want, topology, op=op)
            assert torch.equal(tensor, want), (dtype, op)
        for src in range(world):
            tensor = rank_tensor(shape, dtype, rank)
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            reference.broadcast(want, src, topology)
            assert torch.equal(tensor, want), (dtype, src)
    rows = rank_tensor((world, 3), torch.float32, rank)
    got, want = torch.empty(3), torch.empty(3)
    gpubridge.reduce_scatter_tensor(got, rows)
    reference.reduce_scatter_tensor(want, rows, topology)
    assert torch.equal(got, want)
    if pinned is not None:
        copies = [entry for entry in pinned.log if entry[0] == "copy"]
        assert copies and pinned.buffer is not None
    gpubridge.destroy()
    assert policy.staging_buffer is None  # close() released it


VENDORS = {"2+2": ["nvidia", "nvidia", "amd", "amd"], "1+3": ["amd", "nvidia", "nvidia", "nvidia"],
           "interleaved-2+3": ["nvidia", "amd", "nvidia", "amd", "amd"]}


@pytest.mark.parametrize("vendors", VENDORS.values(), ids=VENDORS.keys())
def test_pinned_staging_gives_flat_gloos_answer(vendors, tmp_path, monkeypatch):
    monkeypatch.setenv(PINNED_STAGING_ENV, "1")
    run_ranks(vendors, partial(_check_pinned, on=True), tmp_path, before_init=_lazy_staging)


def test_pinned_staging_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(PINNED_STAGING_ENV, raising=False)
    run_ranks(VENDORS["2+2"], partial(_check_pinned, on=False), tmp_path,
              before_init=_lazy_staging)


def test_ranks_that_disagree_on_pinned_staging_fail_init_on_every_rank(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", PINNED_STAGING_ENV: "1"}},
         {"env": {VENDOR_ENV: "amd"}}],
        tmp_path,
        [rf"Ranks disagree on {PINNED_STAGING_ENV}: on for ranks \[0\], off for ranks \[1\]"],
    )


def test_a_bad_pinned_staging_value_fails_init_on_every_rank(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", PINNED_STAGING_ENV: "maybe"}},
         {"env": {VENDOR_ENV: "amd", PINNED_STAGING_ENV: "1"}}],
        tmp_path,
        [r"1 of 2 ranks reported problems",
         rf"rank 0 \([^)]*\): {PINNED_STAGING_ENV}='maybe' is not a boolean"],
    )


@pytest.mark.parametrize(("raw", "expected"), [("", False), ("0", False), ("off", False),
                                               ("1", True), (" Yes ", True), ("on", True)])
def test_the_setting_parses(raw, expected, monkeypatch):
    monkeypatch.setenv(PINNED_STAGING_ENV, raw)
    assert pinned_staging_setting() is expected
