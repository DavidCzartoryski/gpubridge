"""The UCX bridge's flush of GPUDirect RDMA writes (gpubridge.ucx.gdr).

On the direct path the NIC writes received data into GPU memory, and the CPU
can see the receive complete before kernels can see the data
(docs/RDMA_TRANSPORT.md, section 4). Here a fake UCX worker models exactly
that: a receive into "GPU memory" completes, but its bytes land only when the
transport flushes. Every collective must still give the right answer, so a
missing or late flush (after the reduction that reads the data, or after
returning to a caller whose kernels read it next) fails. The CUDA driver and
HIP runtime are fakes too: nothing here needs a GPU or the UCX library.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
from torch.distributed import ReduceOp

from gpubridge import ucx
from gpubridge.ucx import gdr
from gpubridge.ucx import transport as ucx_transport
from gpubridge.ucx.transport import UcxTransport

DEADLINE = 10.0


class _Network:
    """Messages in flight between fake workers, by (destination, tag)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.messages: dict[tuple[int, int], bytes] = {}


class _Worker:
    """A fake ucx.Worker for one member. A receive into host memory is written when it
    completes; into GPU memory, only at the next flush (:meth:`land`)."""

    def __init__(self, network: _Network, member: int) -> None:
        self.network, self.member = network, member
        self.posted: dict[int, tuple] = {}
        self.unflushed: list[tuple[int, bytes]] = []
        self.flushes = 0
        self._ids = 0

    def memory_types(self) -> set[str]:
        return {"host", "cuda"}

    def _id(self, op: tuple) -> int:
        self._ids += 1
        self.posted[self._ids] = op
        return self._ids

    def send(self, member, pointer, nbytes, tag, memory=ucx.HOST):
        with self.network.cond:
            self.network.messages[(member, tag)] = ctypes.string_at(pointer, nbytes)
            self.network.cond.notify_all()
        return self._id(("send",))

    def recv(self, pointer, nbytes, tag, memory=ucx.HOST):
        return self._id(("recv", pointer, nbytes, tag, memory))

    def wait(self, ids, deadline):
        for i in ids:
            op = self.posted.pop(i)
            if op[0] != "recv":
                continue
            _, pointer, nbytes, tag, memory = op
            key = (self.member, tag)
            with self.network.cond:
                assert self.network.cond.wait_for(
                    lambda key=key: key in self.network.messages, DEADLINE)
                data = self.network.messages.pop(key)
            assert len(data) == nbytes
            if memory == ucx.HOST:
                ctypes.memmove(pointer, data, nbytes)
            else:  # complete as far as the CPU knows, not yet visible to the GPU
                self.unflushed.append((pointer, data))

    def land(self) -> None:
        self.flushes += 1
        for pointer, data in self.unflushed:
            ctypes.memmove(pointer, data, len(data))
        self.unflushed.clear()


def _members(n: int, monkeypatch, *, direct: bool = True, chunk_bytes: int = 16):
    network = _Network()
    workers = [_Worker(network, m) for m in range(n)]
    members = [UcxTransport(w, list(range(n)), m, timedelta(seconds=DEADLINE), chunk_bytes)
               for m, w in enumerate(workers)]
    if direct:  # CPU tensors, treated as GPU memory UCX takes directly
        monkeypatch.setattr(ucx_transport, "choose_path",
                            lambda device, hip, types: ("direct", ucx.CUDA))
        for member, worker in zip(members, workers, strict=True):
            member._ready = lambda tensor: None
            member._flush_for = lambda device, w=worker: gdr.Flush("fake", w.land)
    return members, workers


def _run(members, fn) -> None:
    errors: list[BaseException] = []

    def body(index: int) -> None:
        try:
            fn(index, members[index])
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=body, args=(i,)) for i in range(len(members))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(DEADLINE * 2)
    assert not errors, errors


def _data(member: int, numel: int) -> torch.Tensor:
    return ((torch.arange(numel) * 5 + member * 3) % 11 - 5).float()


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize(("op", "combine"), [
    (ReduceOp.SUM, lambda t: t.sum(0)), (ReduceOp.MAX, lambda t: t.max(0).values),
    (ReduceOp.MIN, lambda t: t.min(0).values)])
@pytest.mark.parametrize("numel", [1, 5, 37])
def test_all_reduce_flushes_every_receive_before_reading_it(n, op, combine, numel, monkeypatch):
    members, workers = _members(n, monkeypatch)
    results = [_data(m, numel) for m in range(n)]
    want = combine(torch.stack([_data(m, numel) for m in range(n)]))
    _run(members, lambda i, member: member.all_reduce(results[i], op))
    for result in results:
        assert torch.equal(result, want)
    steps = 1 if n == 2 else 2 * (n - 1)  # one exchange, or a ring's two passes
    assert [w.flushes for w in workers] == [steps] * n
    assert all(not w.unflushed for w in workers)  # nothing left for the caller to read stale
    assert {m.last_path for m in members} == {"direct"}
    assert {m.flush_method for m in members} == {"fake"}


@pytest.mark.parametrize(("n", "root"), [(2, 0), (2, 1), (3, 2), (4, 1)])
def test_broadcast_flushes_on_every_receiver_before_returning(n, root, monkeypatch):
    members, workers = _members(n, monkeypatch)
    tensors = [_data(m, 21) for m in range(n)]
    _run(members, lambda i, member: member.broadcast(tensors[i], root))
    for tensor in tensors:
        assert torch.equal(tensor, _data(root, 21))
    assert [w.flushes for w in workers] == [0 if m == root else 1 for m in range(n)]


def test_the_host_path_never_flushes(monkeypatch):
    members, workers = _members(3, monkeypatch, direct=False)
    tensors = [_data(m, 9) for m in range(3)]
    _run(members, lambda i, member: member.all_reduce(tensors[i]))
    assert torch.equal(tensors[0], torch.stack([_data(m, 9) for m in range(3)]).sum(0))
    assert [w.flushes for w in workers] == [0, 0, 0]
    assert {m.last_path for m in members} == {"host"}
    assert {m.flush_method for m in members} == {None}


def test_a_gpu_that_cant_be_flushed_takes_the_staged_path(monkeypatch):
    (member, *_), _ = _members(2, monkeypatch)
    member._flush_for = lambda device: gdr.Flush("unavailable: no way", available=False)
    member._start(torch.ones(4))
    assert (member.last_path, member._memory) == ("staged", ucx.HOST)
    assert member.flush_method == "unavailable: no way"


# ---- choosing the flush ---------------------------------------------------------------

class _Cuda:
    """A fake libcuda.so.1: device ``index`` is CUdevice ``index + 10``."""

    def __init__(self, ordering=0, options=gdr.OPTION_HOST, attribute_status=0,
                 current=None, flush_status=0):
        self.ordering, self.options = ordering, options
        self.attribute_status, self.current, self.flush_status = (attribute_status, current,
                                                                  flush_status)
        self.flushed: list[tuple[int, int]] = []

    def cuDeviceGet(self, ref, index):
        ref._obj.value = index + 10
        return 0

    def cuDeviceGetAttribute(self, ref, attribute, device):
        ref._obj.value = {gdr.CU_WRITES_ORDERING: self.ordering,
                          gdr.CU_FLUSH_WRITES_OPTIONS: self.options}[attribute]
        return self.attribute_status

    def cuCtxGetDevice(self, ref):
        if self.current is None:
            return 201  # CUDA_ERROR_INVALID_CONTEXT
        ref._obj.value = self.current
        return 0

    def cuFlushGPUDirectRDMAWrites(self, target, scope):
        self.flushed.append((target, scope))
        return self.flush_status


class _Selected:
    """Counts how often the device was made current."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self):
        self.count += 1
        return nullcontext()


@pytest.mark.parametrize("ordering", [100, 200])  # OWNER, ALL_DEVICES (Hopper and later)
def test_cuda_gpus_that_order_rdma_writes_need_no_flush(ordering):
    select = _Selected()
    flush = gdr.cuda(1, select, _Cuda(ordering=ordering, options=0))
    assert flush.available and flush.run is None and flush.method.startswith("not needed")
    flush()
    assert select.count == 0


def test_other_cuda_gpus_flush_with_the_driver_on_their_own_context():
    select, driver = _Selected(), _Cuda(current=11)
    flush = gdr.cuda(1, select, driver)
    assert flush.available and flush.method == "cuFlushGPUDirectRDMAWrites"
    flush()
    flush()
    assert driver.flushed == [(gdr.TARGET_CURRENT, gdr.TO_OWNER)] * 2 and select.count == 2


@pytest.mark.parametrize(("driver", "reason"), [
    (_Cuda(options=2), "neither orders"),             # MemOps only, no host flush
    (_Cuda(attribute_status=1), "CUresult 1"),        # a driver older than CUDA 11.3
])
def test_cuda_gpus_with_no_way_to_flush_are_unavailable(driver, reason):
    flush = gdr.cuda(0, _Selected(), driver)
    assert not flush.available and reason in flush.method


def test_without_libcuda_the_flush_is_unavailable(monkeypatch):
    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(gdr.ctypes, "CDLL", missing)
    flush = gdr.cuda(0, _Selected())
    assert not flush.available and "libcuda.so.1" in flush.method


@pytest.mark.parametrize(("driver", "error"), [
    (_Cuda(current=12), "context is not current"),   # another GPU's context
    (_Cuda(current=None), "context is not current"),  # none at all
    (_Cuda(current=11, flush_status=999), "CUresult 999"),
])
def test_a_cuda_flush_that_fails_raises(driver, error):
    flush = gdr.cuda(1, _Selected(), driver)
    with pytest.raises(RuntimeError, match=error):
        flush()


class _Hip:
    """A fake libamdhip64.so with an HDP flush register at ``register``."""

    def __init__(self, register=0xDEAD000, status=0):
        self.register, self.status = register, status
        self.attributes: list[tuple[int, int]] = []

    def hipDeviceGetAttribute(self, ref, attribute, index):
        self.attributes.append((attribute, index))
        ref._obj.value = self.register
        return self.status


class _HipWithFlush(_Hip):
    def __init__(self, api_status=0):
        super().__init__()
        self.api_status, self.flushed = api_status, []

    def hipDeviceFlushGPUDirectRDMAWrites(self, target, scope):
        self.flushed.append((target, scope))
        return self.api_status


def test_rocm_uses_hips_flush_where_the_runtime_has_it():
    select, hip = _Selected(), _HipWithFlush()
    flush = gdr.rocm(2, select, hip)
    assert flush.available and flush.method == "hipDeviceFlushGPUDirectRDMAWrites"
    flush()
    # Once at setup (does it work?), then once per flush, each on the GPU's device.
    assert hip.flushed == [(gdr.TARGET_CURRENT, gdr.TO_OWNER)] * 2 and select.count == 2
    assert hip.attributes == []


def test_rocm_without_hips_flush_writes_the_hdp_register():
    written: list[int] = []
    hip = _Hip()
    flush = gdr.rocm(2, _Selected(), hip, hdp_flush=written.append)
    assert flush.available and flush.method == "HDP flush register"
    assert hip.attributes == [(gdr.HIP_HDP_MEM_FLUSH_CNTL, 2)]
    flush()
    assert written == [0xDEAD000]


@pytest.mark.parametrize(("hip", "reason"), [
    (_HipWithFlush(api_status=801), "hipError_t 801"),  # hipErrorNotSupported
    (_Hip(register=0), "no HDP flush register"),
    (_Hip(status=101), "no HDP flush register"),
])
def test_rocm_gpus_with_no_way_to_flush_are_unavailable(hip, reason):
    flush = gdr.rocm(0, _Selected(), hip, hdp_flush=lambda address: None)
    assert not flush.available and reason in flush.method


def test_rocm_never_loads_a_second_hip_runtime(monkeypatch):
    monkeypatch.setattr(gdr, "loaded_library", lambda name: None)
    flush = gdr.rocm(0, _Selected())
    assert not flush.available and "isn't loaded" in flush.method


def test_the_loaded_hip_runtime_is_found_in_the_process_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n"
        "7f02-7f03 rw-p 00000000 00:00 0 \n"
        "7f04-7f05 r-xp 00000000 08:01 2 /venv/torch/lib/libamdhip64.so.7\n")
    assert gdr.loaded_library("libamdhip64.so", maps) == "/venv/torch/lib/libamdhip64.so.7"
    assert gdr.loaded_library("libcuda.so", maps) is None
    assert gdr.loaded_library("libamdhip64.so", tmp_path / "missing") is None


def test_a_gpu_that_cant_be_flushed_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(gdr, "cuda", lambda index, select: gdr.Flush("unavailable: why",
                                                                     available=False))
    with caplog.at_level(logging.WARNING, logger="gpubridge"):
        assert not gdr.for_device(3, hip=False).available
    assert "GPU 3's tensors take the staged path (unavailable: why)" in caplog.text
