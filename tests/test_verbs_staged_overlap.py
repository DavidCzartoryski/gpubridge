"""GPUBRIDGE_VERBS_STAGED_OVERLAP: the staged two-leader all_reduce chunk by chunk.

No GPU or RDMA device here, so CPU tensors stand in for GPU tensors and lazy copies
(tests/_staging.py: a copy runs only when something waits for it, or for a later copy
on its stream) for the caller's stream:

- against a fake wire that plays the peer and logs every post, wait and copy in one
  order: each chunk's copy out is waited for before its send is posted (and the send
  sees the copied bytes), each copy back is queued after its chunk was reduced, the
  first copy out waits on the device for the last copy back, which stays the last copy
  back queued when a call fails part way; the messages are the same with it on and off;
  with it off the call does exactly what it did before;
- on the MOCK test fabric, end to end: exact results with it on, off and on one leader
  only, under every progress policy, with equal wire counts (scripts/gpu/verbs_staged_ab.py's
  --wire rule) and the same observations.

SOFTWARE-VERIFIED only: not a GPU, NIC or speed result.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import shutil
import tempfile
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _harness import rank_tensor, run_ranks
from test_gpu_kit import KIT
from test_verbs import LAYOUTS
from test_verbs_gpu import LazyRegistered, Records, _stand_in, needs_library
from torch.distributed import ReduceOp

from gpubridge import verbs
from gpubridge.verbs import Registration, observe
from gpubridge.verbs import transport as vt
from gpubridge.verbs.gpu import HostBuffer, Router

CHUNK = 64  # bytes: 16 float32 elements per chunk
ELEMS = CHUNK // 4
AB = KIT / "verbs_staged_ab.py"


# ---- a fake wire that plays the peer ------------------------------------------------

class Wire:
    """The endpoint for member 0 of two. Logs every post and wait into ``log`` (the
    staging's, so copies and posts share one order), keeps the bytes each send would
    read at the moment it is posted, and on a wait writes the peer's chunk into each
    completed receive."""

    def __init__(self, peer: torch.Tensor | None) -> None:
        self.peer = peer  # the other leader's tensor, or None: receives get nothing
        self.log: list = []
        self.ops: dict[int, tuple] = {}
        self.sent: dict[int, bytes] = {}
        self.next_op = 0
        self.failed = None
        self._handle = 1
        self.waits = 0
        self.fail_wait = None  # the n-th wait (1-based) raises
        self.fail_send = None  # the n-th send post (0-based chunk index) raises
        self.aborted: list[str] = []

    def rndv_bytes(self):
        return 1024

    def register_host(self, pointer, nbytes, owner):
        return Registration(100 + owner, pointer, nbytes, "host")

    def release(self, reg):
        pass

    def invalidate_host(self, owner):
        return 0

    def telemetry(self):
        return {"state": "ok"}

    def _post(self, kind, reg, pointer, nbytes, tag):
        if self.failed is not None:
            raise verbs.TransportError(f"the verbs bridge already failed ({self.failed})")
        if kind == "send" and self.fail_send == tag & 0xFFFFF:
            raise verbs.TransportError("posting failed: queue pair in state Err")
        self.next_op += 1
        self.ops[self.next_op] = (kind, reg.handle, pointer - reg.addr, nbytes, tag, pointer)
        self.log.append((kind, tag & 0xFFFFF))
        if kind == "send":
            self.sent[self.next_op] = ctypes.string_at(pointer, nbytes)
        return self.next_op

    def send_reg(self, member, reg, pointer, nbytes, tag):
        return self._post("send", reg, pointer, nbytes, tag)

    def recv_reg(self, member, reg, pointer, nbytes, tag):
        return self._post("recv", reg, pointer, nbytes, tag)

    def wait(self, ids, deadline):
        self.waits += 1
        if self.waits == self.fail_wait:
            raise verbs.TransportError("verbs bridge failed: a send failed with retry exhausted")
        self.log.append(("wait_ops", tuple(self.ops[i][4] & 0xFFFFF for i in ids)))
        for i in ids:
            kind, _, offset, nbytes, tag, pointer = self.ops[i]
            if kind == "recv" and self.peer is not None:
                src = self.peer.view(torch.uint8)[offset:offset + nbytes]
                ctypes.memmove(pointer, src.data_ptr(), nbytes)

    def abort(self, why="aborted"):
        self.aborted.append(why)
        self.failed = why

    def operation_outcome(self, op):
        return {"status": "cancelled", "error": "aborted"}


class Staged(vt.VerbsTransport):
    """The transport over a Wire, every tensor standing in for a GPU tensor."""

    def _is_device(self, tensor):
        return True

    def _device_index(self, tensor):
        return 0

    def _support_for(self, index):
        return vt.Support(False, "standing in: no GPU in this test")

    def _staging_for(self, tensor):
        if 0 not in self._staging:
            self._staging[0] = LazyRegistered(
                HostBuffer(self.worker, lambda n: torch.zeros(n, dtype=torch.uint8)))
            self.worker.endpoint.log = self._staging[0].log
        return self._staging[0]


def _transport(monkeypatch, overlap: str | None, peer=None, members=(0, 1)) -> Staged:
    monkeypatch.setenv(vt.GPU_PATH_ENV, "auto")
    if overlap is None:
        monkeypatch.delenv(vt.STAGED_OVERLAP_ENV, raising=False)
    else:
        monkeypatch.setenv(vt.STAGED_OVERLAP_ENV, overlap)
    t = Staged(Router(Wire(peer)), list(members), 0, timedelta(seconds=5), CHUNK)
    t.emitter = Records()
    return t


def _mine(numel: int, k: int = 0) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.float32) + 1 + 100 * k


def _theirs(numel: int, k: int = 0) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.float32) * 3 + 7 + 100 * k


def _wire(t: Staged) -> dict[str, list]:
    """Each direction's messages in posting order, with each registration named by the
    buffer it covers: (buffer, offset, bytes, tag)."""
    names = {t._staging[0].host.registration.handle: "staging",
             t._host_scratch.registration.handle: "scratch"}
    out: dict[str, list] = {"send": [], "recv": []}
    for kind, handle, offset, nbytes, tag, _ in t.worker.endpoint.ops.values():
        out[kind].append((names[handle], offset, nbytes, tag))
    return out


def _copies(log) -> list[tuple]:
    return [e for e in log if e[0] == "copy"]


# ---- default off: unchanged ---------------------------------------------------------

def test_off_by_default_and_a_bad_value_fails_setup(monkeypatch):
    monkeypatch.delenv(vt.STAGED_OVERLAP_ENV, raising=False)
    assert vt.staged_overlap_setting() is False
    monkeypatch.setenv(vt.STAGED_OVERLAP_ENV, "0")
    assert vt.staged_overlap_setting() is False
    monkeypatch.setenv(vt.STAGED_OVERLAP_ENV, "1")
    assert vt.staged_overlap_setting() is True
    for bad in ("yes", "2", "on"):
        monkeypatch.setenv(vt.STAGED_OVERLAP_ENV, bad)
        with pytest.raises(ValueError, match=vt.STAGED_OVERLAP_ENV):
            vt.staged_overlap_setting()
        # Setup checks it before it opens a device or touches the store.
        with pytest.raises(ValueError, match=vt.STAGED_OVERLAP_ENV):
            vt.VerbsTransport._connect(None, "k", [0, 1], 0, timedelta(seconds=1), 0.0)


@pytest.mark.parametrize("overlap", [None, "0"])
def test_off_the_staged_call_is_exactly_what_it_was(monkeypatch, overlap):
    """The whole-tensor protocol, step for step, as before this option existed: copy
    out, the host waits for it, every receive, every send, a wait per chunk, the copy
    back queued; the next call's copy out waits on the device for that copy back."""
    numel = 3 * ELEMS + ELEMS // 2  # 4 chunks, the last half full
    t = _transport(monkeypatch, overlap, peer=_theirs(numel))
    tensor = _mine(numel)
    t.all_reduce(tensor)
    staging = t._staging[0]
    staging.stream = "other"  # the next call on another stream (an async worker's)
    t.worker.endpoint.peer = _theirs(numel, 1)
    second = _mine(numel, 1)
    t.all_reduce(second)
    staging.run_streams()
    chunks = [0, 1, 2, 3]
    call = [*(("recv", i) for i in chunks), *(("send", i) for i in chunks),
            *(("wait_ops", (i, i)) for i in chunks)]
    assert staging.log == [
        ("copy", "copy1", None), ("wait", "copy1"), ("ran", "copy1"),
        *call, ("copy", "copy2", None),
        ("copy", "copy3", "copy2"), ("wait", "copy3"), ("ran", "copy2"), ("ran", "copy3"),
        *call, ("copy", "copy4", None), ("ran", "copy4")]
    assert torch.equal(tensor, _mine(numel) + _theirs(numel))
    assert torch.equal(second, _mine(numel, 1) + _theirs(numel, 1))
    telemetry = t.telemetry()
    assert telemetry["staged_overlap_setting"] is False
    assert telemetry["last_gpu"]["staging"] == "whole"
    assert telemetry["staged_phases"]["calls"] == 2
    assert telemetry["staged_phases"]["overlapped_calls"] == 0


# ---- on: the order per chunk --------------------------------------------------------

def test_on_each_chunk_is_copied_sent_reduced_and_copied_back_in_order(monkeypatch):
    numel = 3 * ELEMS + ELEMS // 2
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    tensor = _mine(numel)
    staging = t._staging_for(tensor)
    # A GPU may run a copy back as soon as it is queued: keep what it would read then.
    queued = []
    to_device = staging.chunk_to_device
    monkeypatch.setattr(staging, "chunk_to_device",
                        lambda chunk, host: queued.append(host.clone()) or to_device(chunk, host))
    t.all_reduce(tensor)
    log = staging.log
    want = _mine(numel) + _theirs(numel)
    assert len(queued) == 4
    for i, host in enumerate(queued):  # each chunk was already reduced
        assert torch.equal(host, want[i * ELEMS:(i + 1) * ELEMS]), i
    # Every copy out queued at once, the first after the last copy back (none yet).
    assert log[:4] == [("copy", "copy1", None), ("copy", "copy2", None),
                       ("copy", "copy3", None), ("copy", "copy4", None)]
    # Per chunk: its receive, then the host waits for its copy out, then its send.
    posting = [("recv", 0), ("wait", "copy1"), ("ran", "copy1"), ("send", 0),
               ("recv", 1), ("wait", "copy2"), ("ran", "copy2"), ("send", 1),
               ("recv", 2), ("wait", "copy3"), ("ran", "copy3"), ("send", 2),
               ("recv", 3), ("wait", "copy4"), ("ran", "copy4"), ("send", 3)]
    assert log[4:20] == posting
    # Per chunk: both completed, reduced, then its copy back queued.
    assert log[20:] == [("wait_ops", (0, 0)), ("copy", "copy5", None),
                        ("wait_ops", (1, 1)), ("copy", "copy6", None),
                        ("wait_ops", (2, 2)), ("copy", "copy7", None),
                        ("wait_ops", (3, 3)), ("copy", "copy8", None)]
    assert staging.copied_back.label == "copy8"
    # Each send saw its chunk's copied bytes when it was posted, never stale ones.
    sends = [op for op, v in t.worker.endpoint.ops.items() if v[0] == "send"]
    for i, op in enumerate(sends):
        want = _mine(numel)[i * ELEMS:(i + 1) * ELEMS]
        assert t.worker.endpoint.sent[op] == want.numpy().tobytes(), i
    # The copies back are still queued: the caller's stream gets to them later.
    assert not torch.equal(tensor, _mine(numel) + _theirs(numel))
    staging.run_streams()
    assert torch.equal(tensor, _mine(numel) + _theirs(numel))
    tel = t.telemetry()
    assert tel["last_gpu"]["staging"] == "overlapped" and tel["path"] == "staged"
    assert tel["staged_overlap_setting"] is True
    assert tel["staged_phases"]["overlapped_calls"] == tel["staged_phases"]["calls"] == 1
    # One observation per direction, as for a whole call.
    assert [(r["op"], r["bytes"], r["status"], r["path"]) for r in t.emitter.records] == [
        ("recv", numel * 4, "ok", "staged"), ("send", numel * 4, "ok", "staged")]


def test_on_the_next_call_waits_for_the_last_copy_back_on_any_stream(monkeypatch):
    numel = 2 * ELEMS + 3
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    first = _mine(numel)
    t.all_reduce(first)
    staging = t._staging[0]
    last_back = staging.copied_back.label
    staging.stream = "other"
    t.worker.endpoint.peer = _theirs(numel, 1)
    second = _mine(numel, 1)
    before = len(staging.log)
    t.all_reduce(second)
    copies = _copies(staging.log[before:])
    assert copies[0] == ("copy", "copy7", last_back)  # first copy out: after the copy back
    assert all(after is None for _, _, after in copies[1:])
    # The first call's copies back ran before the second call's first copy out.
    ran = [e[1] for e in staging.log if e[0] == "ran"]
    assert ran.index(last_back) < ran.index("copy7")
    staging.run_streams()
    assert torch.equal(first, _mine(numel) + _theirs(numel))
    assert torch.equal(second, _mine(numel, 1) + _theirs(numel, 1))


def test_on_a_bigger_tensor_replaces_the_buffer_only_after_the_last_copy_back(monkeypatch):
    t = _transport(monkeypatch, "1", peer=_theirs(4 * ELEMS))
    small = _mine(2 * ELEMS)
    t.worker.endpoint.peer = _theirs(2 * ELEMS)
    t.all_reduce(small)
    staging = t._staging[0]
    last_back = staging.copied_back.label
    t.worker.endpoint.peer = _theirs(4 * ELEMS)
    big = _mine(4 * ELEMS)
    old = staging.buffer.data_ptr()
    before = len(staging.log)
    t.all_reduce(big)
    after = staging.log[before:]
    # The host waits for the last copy back before anything else, then a new buffer.
    assert after[0] == ("wait", last_back)
    assert _copies(after)[0] == ("copy", "copy5", None)  # the new buffer: nothing to wait on
    assert staging.buffer.data_ptr() != old
    staging.run_streams()
    assert torch.equal(small, _mine(2 * ELEMS) + _theirs(2 * ELEMS))
    assert torch.equal(big, _mine(4 * ELEMS) + _theirs(4 * ELEMS))


@pytest.mark.parametrize("numel", [1, ELEMS - 1, ELEMS, ELEMS + 1, 3 * ELEMS + ELEMS // 2])
@pytest.mark.parametrize("op", [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN])
def test_on_and_off_put_the_same_messages_on_the_wire_with_the_same_result(monkeypatch,
                                                                            numel, op):
    results, wires, logs = {}, {}, {}
    for overlap in ("0", "1"):
        t = _transport(monkeypatch, overlap, peer=_theirs(numel) * (-1 if op != ReduceOp.SUM
                                                                    else 1))
        tensor = _mine(numel)
        t.all_reduce(tensor, op)
        t._staging[0].run_streams()
        results[overlap], wires[overlap] = tensor, _wire(t)
        logs[overlap] = list(t._staging[0].log)
        staging = t.telemetry()["last_gpu"]["staging"]
        assert staging == ("whole" if overlap == "0" else
                           "overlapped" if numel > ELEMS else "whole (one chunk)")
    assert torch.equal(results["0"], results["1"])
    peer = _theirs(numel) * (-1 if op != ReduceOp.SUM else 1)
    want = {ReduceOp.SUM: _mine(numel) + peer, ReduceOp.MAX: torch.maximum(_mine(numel), peer),
            ReduceOp.MIN: torch.minimum(_mine(numel), peer)}[op]
    assert torch.equal(results["1"], want)
    # The same buffers, offsets, sizes and tags, each direction in the same order.
    assert wires["0"] == wires["1"]
    assert len(wires["1"]["send"]) == max(1, -(-numel // ELEMS))
    if numel <= ELEMS:  # one chunk: the call itself is the same, step for step
        assert logs["0"] == logs["1"]


def test_on_broadcasts_more_leaders_and_one_chunk_run_whole_and_say_why(monkeypatch):
    t = _transport(monkeypatch, "1", peer=_theirs(3 * ELEMS))
    t.broadcast(_mine(3 * ELEMS), 0)
    assert t.telemetry()["last_gpu"]["staging"] == "whole (not an all_reduce)"
    t.all_reduce(_mine(ELEMS))
    assert t.telemetry()["last_gpu"]["staging"] == "whole (one chunk)"
    three = _transport(monkeypatch, "1", peer=None, members=(0, 1, 2))
    three.all_reduce(_mine(3 * ELEMS))
    assert three.telemetry()["last_gpu"]["staging"] == "whole (more than two leaders)"
    for x in (t, three):
        assert x.staged_phases["overlapped_calls"] == 0


# ---- on: failures ---------------------------------------------------------------------

def test_on_a_failed_wait_part_way_raises_keeps_the_last_copy_back_and_refuses_later_calls(
        monkeypatch):
    numel = 3 * ELEMS + ELEMS // 2
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    wire = t.worker.endpoint
    wire.fail_wait = 3  # chunks 0 and 1 complete, chunk 2's wait fails
    tensor = _mine(numel)
    with pytest.raises(verbs.TransportError, match="retry exhausted"):
        t.all_reduce(tensor)
    staging = t._staging[0]
    assert wire.aborted, "the worker is stopped before the error propagates"
    # Chunks 0 and 1 were reduced and their copies back queued; the last of them is the
    # one the next copy out would wait for.
    assert [c[1] for c in _copies(staging.log)] == [f"copy{i}" for i in range(1, 7)]
    assert staging.copied_back.label == "copy6"
    staging.run_streams()
    reduced = 2 * ELEMS
    assert torch.equal(tensor[:reduced], (_mine(numel) + _theirs(numel))[:reduced])
    assert torch.equal(tensor[reduced:], _mine(numel)[reduced:])  # the rest untouched
    # The memory the library may still hold is kept; failed calls aren't timed.
    assert t._keep and t._keep[-1][0].data_ptr() == staging.buffer.data_ptr()
    assert t.staged_phases["calls"] == t.staged_phases["overlapped_calls"] == 0
    assert len(t.emitter.records) == 2
    assert all(r["status"] != "ok" for r in t.emitter.records), t.emitter.records
    # The transport refuses later calls; its copy out still waits for the last copy back.
    staging.stream = "other"
    before = len(staging.log)
    again = _mine(numel)
    with pytest.raises(verbs.TransportError, match="already failed"):
        t.all_reduce(again)
    assert _copies(staging.log[before:])[0] == ("copy", "copy7", "copy6")
    assert staging.copied_back.label == "copy6"
    assert t.telemetry()["staged_phases"]["calls"] == 0


def test_on_a_failed_post_queues_no_copy_back_and_keeps_the_previous_one(monkeypatch):
    numel = 3 * ELEMS
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    first = _mine(numel)
    t.all_reduce(first)
    staging = t._staging[0]
    previous = staging.copied_back.label
    t.worker.endpoint.fail_send = 1  # chunk 1's send can't be posted
    second = _mine(numel, 1)
    with pytest.raises(verbs.TransportError, match="posting failed"):
        t.all_reduce(second)
    assert staging.copied_back.label == previous
    staging.run_streams()
    assert torch.equal(first, _mine(numel) + _theirs(numel))
    assert torch.equal(second, _mine(numel, 1))  # nothing came back into it
    send = next(r for r in t.emitter.records[-2:] if r["op"] == "send")
    assert send["status"] == "error" and "queue pair in state Err" in send["error"]


def _log_aborts(t: Staged, log: list) -> None:
    wire = t.worker.endpoint
    abort = wire.abort
    wire.abort = lambda why="aborted": (log.append(("abort",)), abort(why))[1]


def test_on_a_failed_wait_part_way_drains_its_copies_before_the_error_reaches_the_caller(
        monkeypatch):
    # An async collective: its worker's stream, whose copies the caller's stream is not
    # ordered after when the call fails. Nothing here waits for a copy after the call.
    numel = 3 * ELEMS + ELEMS // 2
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    tensor = _mine(numel)
    staging = t._staging_for(tensor)
    staging.stream = "other"
    _log_aborts(t, staging.log)
    t.worker.endpoint.fail_wait = 3  # chunks 0 and 1 complete, chunk 2's wait fails
    with pytest.raises(verbs.TransportError, match="retry exhausted"):
        t.all_reduce(tensor)
    # The caller reuses its tensor at once: no copy of the failed call writes it later.
    reset = tensor.clone()
    tensor.zero_()
    staging.run_streams()
    assert torch.equal(tensor, torch.zeros(numel))
    # When the error reached the caller, chunks 0 and 1 were already back, the rest
    # untouched, and nothing was still queued: aborted, then the stream drained.
    reduced = 2 * ELEMS
    assert torch.equal(reset[:reduced], (_mine(numel) + _theirs(numel))[:reduced])
    assert torch.equal(reset[reduced:], _mine(numel)[reduced:])
    log = staging.log
    drain = log.index(("drain", "other"))
    assert log.index(("abort",)) < drain
    assert all(e[0] == "ran" for e in log[drain + 1:]), log[drain:]


def test_on_a_copy_out_that_cant_be_queued_aborts_and_drains_those_queued(monkeypatch):
    numel = 3 * ELEMS
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    first = _mine(numel)
    t.all_reduce(first)  # its copies back are still queued
    staging = t._staging[0]
    previous = staging.copied_back.label
    _log_aborts(t, staging.log)
    copy = staging._copy

    def second_copy_out_fails(dst, src, after):
        if staging.copies == 7:  # copy7 is the second call's first copy out
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return copy(dst, src, after)

    monkeypatch.setattr(staging, "_copy", second_copy_out_fails)
    second = _mine(numel, 1)
    before = len(staging.log)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        t.all_reduce(second)
    # The worker is aborted, so the transport is no longer healthy, and the copy out
    # already queued (after the previous copies back) was drained; nothing was posted.
    assert t.worker.failed is not None and t.telemetry()["failed"]
    assert not staging.streams["caller"].queue and not staging.streams["other"].queue
    log = staging.log[before:]
    assert not any(e[0] in ("send", "recv") for e in log), log
    assert _copies(log) == [("copy", "copy7", previous)]
    assert log.index(("abort",)) < log.index(("drain", "caller"))
    assert staging.copied_back.label == previous
    assert t._keep and t._keep[-1][0].data_ptr() == staging.buffer.data_ptr()
    assert torch.equal(first, _mine(numel) + _theirs(numel))
    assert torch.equal(second, _mine(numel, 1))
    # A later call, on another stream, is refused, and leaves nothing queued either.
    monkeypatch.setattr(staging, "_copy", copy)
    staging.stream = "other"
    again = _mine(numel, 2)
    with pytest.raises(verbs.TransportError, match="already failed"):
        t.all_reduce(again)
    assert not staging.streams["caller"].queue and not staging.streams["other"].queue
    assert torch.equal(again, _mine(numel, 2))
    assert torch.equal(first, _mine(numel) + _theirs(numel))


def test_on_a_drain_that_fails_too_the_calls_own_error_reaches_the_caller(monkeypatch):
    # On a GPU a sticky error fails the stream wait as well: the caller must still see
    # why the call failed, not the wait's error with the call's only as its context.
    numel = 3 * ELEMS + ELEMS // 2
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    staging = t._staging_for(_mine(numel))
    _log_aborts(t, staging.log)

    def sticky():
        staging.log.append(("drain failed",))
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(staging, "_drain", sticky)
    t.worker.endpoint.fail_wait = 3  # chunk 2's wait fails
    with pytest.raises(verbs.TransportError, match="retry exhausted") as info:
        t.all_reduce(_mine(numel))
    assert isinstance(info.value.__cause__, RuntimeError), info.value.__cause__
    assert "illegal memory access" in str(info.value.__cause__)
    log = staging.log
    assert log.index(("abort",)) < log.index(("drain failed",))
    # Its copies may still be pending: the buffer stays kept, and later calls are refused.
    assert t._keep and t._keep[-1][0].data_ptr() == staging.buffer.data_ptr()
    assert t.worker.failed is not None and t.telemetry()["failed"]
    with pytest.raises(verbs.TransportError, match="already failed"):
        t.all_reduce(_mine(numel, 1))


def test_on_the_phases_partition_the_call_like_a_whole_call(monkeypatch):
    numel = 4 * ELEMS
    t = _transport(monkeypatch, "1", peer=_theirs(numel))
    then = dict(t.staged_phases)
    start = time.perf_counter()
    t.all_reduce(_mine(numel))
    wall = time.perf_counter() - start
    now = t.staged_phases
    grew = {k: now[k] - then[k] for k in now}
    assert grew["calls"] == grew["overlapped_calls"] == 1
    assert all(v >= 0 for v in grew.values()), grew
    assert grew["to_host_s"] > 0 and grew["to_device_s"] > 0 and grew["bridge_s"] > 0
    assert grew["post_s"] + grew["wait_s"] <= grew["bridge_s"] + 1e-9, grew
    assert grew["to_host_s"] + grew["bridge_s"] + grew["to_device_s"] <= wall + 1e-9


# ---- end to end on the MOCK fabric ----------------------------------------------------

def _ab():
    spec = importlib.util.spec_from_file_location("verbs_staged_ab", AB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verbs_counts():
    return _ab().verbs_counts


def _stand_in_overlap(env, overlap, rank):
    on = overlap == "on" or (overlap == "mixed" and rank < 2)  # 2+2: ranks 0 and 2 lead
    if on:
        os.environ[vt.STAGED_OVERLAP_ENV] = "1"
    else:
        os.environ.pop(vt.STAGED_OVERLAP_ENV, None)
    _stand_in(env, rank)


E2E_CHUNK = 65536  # float32 elements in 256 KiB
CASES = [(1, ReduceOp.SUM), (E2E_CHUNK - 1, ReduceOp.SUM), (E2E_CHUNK, ReduceOp.SUM),
         (E2E_CHUNK + 1, ReduceOp.SUM), (3 * E2E_CHUNK + E2E_CHUNK // 2, ReduceOp.SUM),
         (3 * E2E_CHUNK + E2E_CHUNK // 2, ReduceOp.MAX)]


def _check_overlap(out_dir, topology):
    bridge = topology.bridge
    counts = _verbs_counts()
    sizes = []
    for numel, op in CASES:
        # As the A/B harness does: the counts before a barrier every rank passes, so no
        # message of this call can arrive (and be counted) before them.
        before = counts(bridge)
        dist.barrier()
        if bridge is None:
            continue
        tensor = rank_tensor((numel,), torch.float32, topology.rank)
        inputs = [rank_tensor((numel,), torch.float32, m) for m in bridge.members]
        want = sum(inputs) if op == ReduceOp.SUM else torch.stack(inputs).max(0).values
        c0, h0 = time.process_time(), time.perf_counter()
        bridge.all_reduce(tensor, op)
        bridge._staging[0].run_streams()  # the device gets to the queued copies back
        h1, c1 = time.perf_counter(), time.process_time()
        after = counts(bridge)
        staging = bridge.telemetry()["last_gpu"]["staging"]
        if bridge.staged_overlap:
            assert staging == ("overlapped" if numel > E2E_CHUNK else "whole (one chunk)")
        else:
            assert staging == "whole"
        sizes.append({"bytes": numel * 4, "payload_bytes": numel * 4, "calls": 1,
                      "timed_s": [h1 - h0], "host_s": [h1 - h0], "cpu_s": [c1 - c0],
                      "rss_growth_bytes": 0, "mismatches": int(not torch.equal(tensor, want)),
                      "verbs_delta": {k: after[k] - before[k] for k in after},
                      "rank": topology.rank, "staging": staging})
    if bridge is not None:
        src = bridge.members[0]
        tensor = rank_tensor((3 * E2E_CHUNK,), torch.float32, topology.rank)
        bridge.broadcast(tensor, src)
        bridge._staging[0].run_streams()
        assert torch.equal(tensor, rank_tensor((3 * E2E_CHUNK,), torch.float32, src))
        assert all(s["mismatches"] == 0 for s in sizes), sizes
        t = bridge.telemetry()
        overlapped = 3 if bridge.staged_overlap else 0
        assert t["staged_phases"]["overlapped_calls"] == overlapped, t["staged_phases"]
        assert t["staged_phases"]["calls"] == len(CASES) + 1
        assert "key" not in json.dumps(t).lower()
        Path(out_dir, f"rank-{topology.rank}.json").write_text(json.dumps(sizes))
    dist.barrier()


def _arm(tmp_path, overlap: str, progress: str) -> tuple[dict, list[list[dict]]]:
    fabric = tempfile.mkdtemp(prefix="gbov-", dir="/tmp")
    out = tmp_path / overlap
    obs = out / "obs"
    out.mkdir()
    (tmp_path / f"r-{overlap}").mkdir()
    env = {verbs.TEST_FABRIC_ENV: fabric, observe.OBSERVE_ENV: str(obs / "rank-{rank}.jsonl"),
           observe.JOB_ENV: "job-overlap", vt.RNDV_BYTES_ENV: "64K",
           vt.CHUNK_BYTES_ENV: "256K", vt.PROGRESS_ENV: progress}
    try:
        run_ranks(LAYOUTS["2+2"][0], partial(_check_overlap, str(out)), tmp_path / f"r-{overlap}",
                  init_kwargs={"bridge": "verbs"},
                  before_init=partial(_stand_in_overlap, env, overlap))
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    leaders = sorted(out.glob("rank-*.json"))
    assert len(leaders) == 2, leaders
    per_rank = [json.loads(p.read_text()) for p in leaders]
    observations = {}
    for path in sorted(obs.glob("rank-*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        observations[path.name] = [(r["op"], r["bytes"], r["status"], r["path"])
                                   for r in records]
    sizes = [list(entries) for entries in zip(*per_rank, strict=True)]
    return observations, sizes


@needs_library
@pytest.mark.parametrize("progress", ["latency", "balanced", "economy"])
def test_end_to_end_exact_with_equal_wire_counts_on_off_and_mixed(tmp_path, progress):
    arms = {overlap: _arm(tmp_path, overlap, progress) for overlap in ("off", "on", "mixed")}
    wire = ("sends", "recvs", "bytes_sent", "bytes_received", "writes_posted",
            "bytes_written", "write_windows", "write_window_bytes")
    off_obs, off_sizes = arms["off"]
    for overlap, (obs, sizes) in arms.items():
        # The same observation records (one per call and direction), all ok and staged.
        assert obs == off_obs, overlap
        assert len(obs) == 2 and all(len(v) == 2 * len(CASES) + 1 for v in obs.values())
        assert all(status == "ok" and path == "staged"
                   for v in obs.values() for _, _, status, path in v)
        for case, (ranks, base) in enumerate(zip(sizes, off_sizes, strict=True)):
            for mine, theirs in zip(ranks, base, strict=True):
                assert mine["mismatches"] == 0, (overlap, case)
                assert {k: mine["verbs_delta"][k] for k in wire} == \
                    {k: theirs["verbs_delta"][k] for k in wire}, (overlap, case)
    # The 3.5-chunk calls make a write window per chunk on each leader.
    big = off_sizes[4]
    assert [r["verbs_delta"]["write_windows"] for r in big] == [4, 4]
    # The A/B harness's own wire rule agrees (its timing verdicts on a loaded CPU with
    # one repeat are noise, and not asserted). These are the leaders' entries for this
    # test's sizes, not whole harness records, so the rule is applied without compare's
    # completeness and matching checks.
    ab = _ab()
    off = {0: ab.summarize({"sizes": off_sizes})}
    for after in ("on", "mixed"):
        rows = ab.judge(off, {0: ab.summarize({"sizes": arms[after][1]})}, [0], wire=True)
        assert rows and not any("wire counts differ" in str(r.get("regressions"))
                                for r in rows), rows
        assert not any(r["verdict"].startswith("FAIL") for r in rows), rows
