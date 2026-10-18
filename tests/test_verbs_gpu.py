"""GPU tensors on the verbs bridge, and its transport observations, without a GPU.

docs/VERBS_TRANSPORT.md, "GPU memory". There is no GPU or RDMA device here, so:

- the router's ordering (stream wait before device memory is touched, flush after
  receives into it, bounce copies for small receives) runs against a recording fake
  endpoint;
- the choice between direct and staged, with its reasons, runs against a fake
  endpoint whose driver checks and registrations a test sets;
- the staged path runs end to end on the MOCK test fabric with CPU tensors standing in
  for GPU tensors and lazy copies standing in for the caller's stream (a copy happens
  only when waited for, so a bridge that read the staging buffer early would move
  zeros), against the expected sums;
- observations are written there and checked against the detector's schema.

SOFTWARE-VERIFIED only: every GPU path is HARDWARE BLOCKED (docs: "Status").
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _harness import rank_tensor, run_ranks
from _staging import LazyPinned
from test_verbs import LAYOUTS, UNAVAILABLE

from gpubridge import verbs
from gpubridge.ucx import gdr
from gpubridge.verbs import Registration, observe
from gpubridge.verbs import transport as vt
from gpubridge.verbs.gpu import DeviceSpan, HostBuffer, RegisteredStaging, Router

needs_library = pytest.mark.skipif(UNAVAILABLE is not None,
                                   reason=f"verbs test library unavailable: {UNAVAILABLE}")


# ---- a recording fake endpoint --------------------------------------------------------

class FakeEndpoint:
    """Records every call; registrations and driver answers are what the test sets."""

    def __init__(self, rndv: int = 1024) -> None:
        self.rndv = rndv
        self.calls: list = []
        self.next_op = 0
        self.failed = None
        self._handle = 1
        self.driver: str | Exception = "FAKE driver"
        self.caps = {"driver": "FAKE", "dmabuf": "ok", "peer_memory": "ok (nvidia_peermem)"}
        self.registrations: list = []  # what register_device returns, in order
        self.released: list = []

    def rndv_bytes(self):
        return self.rndv

    def _op(self, *call):
        self.calls.append(call)
        self.next_op += 1
        return self.next_op

    def send(self, member, pointer, nbytes, tag):
        return self._op("send", member, nbytes)

    def recv(self, member, pointer, nbytes, tag):
        return self._op("recv", member, nbytes)

    def send_reg(self, member, reg, pointer, nbytes, tag):
        return self._op("send_reg", member, reg.method, pointer - reg.addr, nbytes)

    def recv_reg(self, member, reg, pointer, nbytes, tag):
        return self._op("recv_reg", member, reg.method, pointer - reg.addr, nbytes)

    def wait(self, ids, deadline):
        self.calls.append(("wait", tuple(ids)))

    def load_gpu(self, vendor):
        self.calls.append(("load_gpu", vendor))
        if isinstance(self.driver, Exception):
            raise self.driver
        return self.driver

    def gpu_capabilities(self, device):
        return dict(self.caps)

    def register_device(self, device, pointer, nbytes):
        self.calls.append(("register_device", device, nbytes))
        reg = self.registrations.pop(0)
        if isinstance(reg, Registration):
            # Cover exactly the memory asked for (the library rounds out; the router
            # counts offsets from the registration's address either way).
            reg = Registration(reg.handle, pointer, nbytes, reg.method)
        return reg

    def register_host(self, pointer, nbytes, owner):
        return Registration(100 + owner, pointer, nbytes, "host")

    def release(self, reg):
        self.released.append(reg.handle)

    def invalidate_host(self, owner):
        return 0

    def telemetry(self):
        return {"state": "ok"}


def test_the_router_waits_for_the_stream_flushes_after_writes_and_bounces_small_receives():
    endpoint = FakeEndpoint(rndv=1024)
    router = Router(endpoint)
    order = []
    router.sync = lambda: order.append("sync")
    router.flush = lambda: order.append("flush")
    device = torch.zeros(4096, dtype=torch.uint8)  # stands in for GPU memory
    plain = torch.zeros(64, dtype=torch.uint8)
    host = HostBuffer(router, lambda n: torch.zeros(n, dtype=torch.uint8))
    staged = host.get(2048)
    reg = Registration(7, device.data_ptr(), device.numel(), "dmabuf")
    call = router.begin([DeviceSpan(reg, device)])
    base = device.data_ptr()
    a = router.send(1, base, 2048, tag=1)                  # device: in place, after a sync
    b = router.recv(1, base + 2048, 1024, tag=2)           # device, rendezvous size: written
    c = router.recv(1, base + 3072, 100, tag=3)            # device, small: bounced
    d = router.send(1, staged.data_ptr(), 2048, tag=4)     # registered host: in place
    e = router.send(1, plain.data_ptr(), 64, tag=5)        # plain: copied by the library
    assert order == ["sync"], "one stream wait before the first device message"
    kinds = [c[0] for c in endpoint.calls]
    assert kinds == ["send_reg", "recv_reg", "recv", "send_reg", "send"]
    assert endpoint.calls[0][2:] == ("dmabuf", 0, 2048)
    assert endpoint.calls[1][2:] == ("dmabuf", 2048, 1024)
    assert endpoint.calls[3][2] == "host"
    # The bounce copy happens only once its receive is waited for, then the flush for
    # the written receive comes before the wait returns.
    router.wait([a, d, e], deadline=0)
    assert order == ["sync"]
    router.wait([b, c], deadline=0)
    assert order == ["sync", "flush"]
    assert call.bounces == {} and call.written == set()
    # After a wait, the next device message waits for the stream again.
    router.send(1, base, 16, tag=6)
    assert order == ["sync", "flush", "sync"]
    finished = router.end()
    assert (finished.syncs, finished.flushes) == (2, 1)
    assert (finished.bounced_ops, finished.bounced_bytes) == (1, 100)
    assert (router.in_place_ops, router.plain_ops, router.bounced_ops) == (4, 1, 1)


class Planner(vt.VerbsTransport):
    """The transport over a fake endpoint, with CPU tensors standing in for GPU ones."""

    flush = gdr.Flush("FAKE flush")

    def _device_index(self, tensor):
        return 0

    def _flush_for(self, index):
        return self.flush


def _planner(endpoint, monkeypatch, path="auto"):
    monkeypatch.setenv(vt.GPU_PATH_ENV, path)
    return Planner(Router(endpoint), [0, 1], 0, timedelta(seconds=5), 4096)


def test_direct_only_when_every_check_passes_and_registration_succeeds(monkeypatch):
    tensor = torch.ones(512)
    ok = Registration(1, 0, 0, "dmabuf")
    endpoint = FakeEndpoint()
    endpoint.registrations = [ok, Registration(2, 0, 0, "dmabuf")]
    plan = _planner(endpoint, monkeypatch)._plan(tensor, scratch_numel=512)
    assert (plan.path, plan.method, plan.flush) == ("direct", "dmabuf", "FAKE flush")
    assert plan.reason == "registered by dmabuf"
    assert len(plan.spans) == 2 and plan.scratch.numel() == 512

    # Each failing check gives staged, with its reason.
    def staged_reason(setup, path="auto") -> str:
        endpoint = FakeEndpoint()
        endpoint.registrations = [Registration(1, 0, 0, "dmabuf"), Registration(2, 0, 0, "x")]
        planner = _planner(endpoint, monkeypatch, path)
        setup(endpoint, planner)
        plan = planner._plan(tensor, scratch_numel=512)
        assert plan.path == "staged"
        return plan.reason

    def rndv_off(e, p):
        p.worker.rndv = 0

    def no_driver(e, p):
        e.driver = verbs.TransportError("unsupported: can't load libcuda.so.1")

    def no_paths(e, p):
        e.caps = {"dmabuf": "the driver reports no dma-buf export",
                  "peer_memory": "neither nvidia_peermem nor nv_peer_mem is loaded"}

    def no_flush(e, p):
        p.flush = gdr.Flush("unavailable: neither ordered nor flushable", available=False)

    def refused(e, p):
        e.registrations = ["dma-buf: ibv_reg_dmabuf_mr failed; peer memory: not loaded"]

    def scratch_refused(e, p):
        e.registrations = [Registration(1, 0, 0, "dmabuf"), "dma-buf: out of BAR space"]

    assert "rendezvous is off" in staged_reason(rndv_off)
    assert "can't load libcuda.so.1" in staged_reason(no_driver)
    assert staged_reason(no_paths) == ("dma-buf: the driver reports no dma-buf export; peer "
                                       "memory: neither nvidia_peermem nor nv_peer_mem is "
                                       "loaded")
    assert "neither ordered nor flushable" in staged_reason(no_flush)
    assert "registering the tensor failed: dma-buf: ibv_reg_dmabuf_mr" in staged_reason(refused)
    assert "registering the scratch buffer failed" in staged_reason(scratch_refused)
    assert staged_reason(lambda e, p: None, path="staged") == f"{vt.GPU_PATH_ENV}=staged"

    # The tensor's registration is given back when the scratch's fails.
    endpoint = FakeEndpoint()
    endpoint.registrations = [Registration(5, 0, 0, "dmabuf"), "no"]
    _planner(endpoint, monkeypatch)._plan(tensor, scratch_numel=512)
    assert endpoint.released == [5]

    # Staging off: the reason fails the call instead.
    endpoint = FakeEndpoint()
    endpoint.registrations = ["dma-buf: refused; peer memory: refused"]
    with pytest.raises(RuntimeError, match="staging is off .*=direct.*dma-buf: refused"):
        _planner(endpoint, monkeypatch, "direct")._plan(tensor, scratch_numel=0)
    # An empty tensor has nothing to register.
    plan = _planner(FakeEndpoint(), monkeypatch)._plan(torch.ones(0), scratch_numel=0)
    assert (plan.path, plan.reason) == ("staged", "an empty tensor: nothing to register")


class Records:
    """Stands in for the observation emitter."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def emit(self, record: dict) -> None:
        observe.check_record(record)
        self.records.append(record)

    def stats(self) -> dict:
        return {"emitted": len(self.records)}


def test_direct_is_reported_only_when_nothing_bounced_else_the_split(monkeypatch):
    endpoint = FakeEndpoint(rndv=1024)
    planner = _planner(endpoint, monkeypatch)
    planner.worker.bounce = lambda n: torch.zeros(n, dtype=torch.uint8)  # no pinned memory here
    planner.emitter = Records()
    tensor = torch.zeros(4096, dtype=torch.uint8)
    base = tensor.data_ptr()

    def call(*sizes):
        endpoint.registrations = [Registration(1, 0, 0, "dmabuf")]
        plan = planner._plan(tensor, scratch_numel=0)
        assert plan.path == "direct"
        plan.sync = lambda: None  # no CUDA stream here

        def run(_):
            router, at, ids = planner.worker, base, []
            for n in sizes:
                ids.append(router.recv(1, at, n, tag=len(ids)))
                at += n
            router.wait(ids, 0)

        planner.emitter.records.clear()
        planner._run("direct", plan, tensor, run)
        return planner.last, planner.emitter.records

    # Every receive at the rendezvous size or above: written in place, "direct".
    last, records = call(2048, 1024)
    assert (last["path"], last["path_reason"]) == ("direct", "registered by dmabuf")
    assert (last["gpu"]["bounced_ops"], last["gpu"]["bounced_bytes"]) == (0, 0)
    assert [r["path"] for r in records] == ["direct"]
    assert "bounced_bytes" not in records[0]
    # One below it came through a bounce buffer: "direct+bounce", and how much.
    last, records = call(2048, 100, 28)
    assert last["path"] == "direct+bounce"
    assert last["path_reason"] == (
        "registered by dmabuf; 128 bytes in 2 receives below the rendezvous size "
        f"({vt.RNDV_BYTES_ENV}=1024) came through a pinned host bounce buffer")
    assert (last["gpu"]["bounced_ops"], last["gpu"]["bounced_bytes"]) == (2, 128)
    assert [(r["path"], r["bytes"], r["bounced_bytes"]) for r in records] == [
        ("direct+bounce", 2176, 128)]
    assert planner.paths == {"host": 0, "direct": 1, "direct+bounce": 1, "staged": 0}
    t = planner.telemetry()
    assert t["path"] == "direct+bounce" and t["last_gpu"]["bounced_bytes"] == 128


def test_close_still_closes_the_endpoint_when_a_buffer_is_still_in_use(monkeypatch):
    endpoint = FakeEndpoint()
    endpoint.invalidate_host = lambda owner: 1  # still in use: HostBuffer.drop() raises
    closed = []
    endpoint.close = lambda timeout: closed.append(timeout) or True
    planner = _planner(endpoint, monkeypatch)
    planner._host_scratch.get(64)
    planner.close()
    assert closed, "the endpoint was closed anyway"
    assert planner._host_scratch.buffer is not None, "the buffer was kept until then"


def test_driver_support_is_checked_once_per_gpu(monkeypatch):
    endpoint = FakeEndpoint()
    endpoint.registrations = [Registration(i, 0, 0, "peer_memory") for i in range(4)]
    planner = _planner(endpoint, monkeypatch)
    planner._plan(torch.ones(8), 0)
    planner._plan(torch.ones(8), 0)
    assert [c for c in endpoint.calls if c[0] == "load_gpu"] == [("load_gpu", "cuda")]
    assert planner.telemetry()["gpus"]["0"]["direct"] is True


def test_settings(monkeypatch):
    monkeypatch.setenv(vt.GPU_PATH_ENV, "sideways")
    with pytest.raises(ValueError, match=vt.GPU_PATH_ENV):
        vt.gpu_path_setting()
    monkeypatch.setenv(vt.DMABUF_ENV, "yes")
    with pytest.raises(ValueError, match=vt.DMABUF_ENV):
        vt.endpoint_settings()
    monkeypatch.setenv(vt.DMABUF_ENV, "0")
    monkeypatch.setenv(vt.RNDV_BYTES_ENV, "0")
    settings = vt.endpoint_settings()
    assert (settings["dmabuf"], settings["rndv_bytes"]) == (0, 0)


# ---- observations ----------------------------------------------------------------------

def test_records_with_key_like_names_are_refused():
    observe.check_record({"nic": {"device": "mlx5_0", "pci_bus_id": "0000:3b:00.0"}})
    for bad in ({"rkey": 1}, {"nic": {"lkey": 2}}, {"a": [{"remote_key": 3}]}, {"Key": 4},
                {"mkey": 5}, {"x_qkey": 6}):
        with pytest.raises(ValueError, match="looks like a key"):
            observe.check_record(bad)


def test_nic_identity_comes_from_sysfs_and_is_null_where_absent(tmp_path):
    pci = tmp_path / "devices" / "0000:3b:00.0"
    pci.mkdir(parents=True)
    dev = tmp_path / "ib" / "mlx5_0"
    (dev / "ports" / "1").mkdir(parents=True)
    (dev / "device").symlink_to(pci)
    (dev / "ports" / "1" / "rate").write_text("100 Gb/sec (4X EDR)\n")
    nic = observe.nic_object({"device": "mlx5_0", "port": 1, "gid_index": None,
                              "link_layer": "Ethernet (RoCE)"}, root=tmp_path / "ib")
    assert nic == {"device": "mlx5_0", "port": 1, "gid_index": None, "link_layer": "Ethernet",
                   "pci_bus_id": "0000:3b:00.0", "rate_gbps": 100.0}
    nic = observe.nic_object({"device": "mockfab0", "port": 1, "link_layer": "InfiniBand"},
                             root=tmp_path / "ib")
    assert (nic["pci_bus_id"], nic["rate_gbps"], nic["link_layer"]) == (None, None, "InfiniBand")
    assert observe.peer_nic("gbverbs 2;ll=eth;dev=mlx5_1;port=2") == {
        "device": "mlx5_1", "port": 2, "link_layer": "Ethernet"}


def test_the_job_comes_from_the_environment_or_the_store(monkeypatch):
    for env in (observe.JOB_ENV, *observe.JOB_FALLBACK_ENVS):
        monkeypatch.delenv(env, raising=False)
    store = dist.HashStore()
    first = observe.job_id(store)
    assert first != "unknown" and observe.job_id(store) == first  # agreed, not per rank
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert observe.job_id(store) == "123"
    monkeypatch.setenv(observe.JOB_ENV, "run-7")
    assert observe.job_id(store) == "run-7"


def test_an_emitter_that_cant_write_drops_and_counts_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    emitter = observe.Emitter(str(blocker / "obs.jsonl"), limit=10)  # parent is a file
    emitter._thread.join(5)
    emitter.emit({"schema": observe.SCHEMA})
    stats = emitter.stats()
    assert stats["error"] and stats["dropped"] >= 1 and stats["emitted"] == 0
    emitter.close()
    full = observe.Emitter(str(tmp_path / "ok.jsonl"), limit=1)
    full._stop = True  # nothing more is taken: the bound holds
    full.emit({"a": 1})
    assert full.stats()["dropped"] == 1
    full.close()


# ---- the staged path end to end, CPU tensors standing in ------------------------------

class LazyRegistered(RegisteredStaging, LazyPinned):
    """RegisteredStaging's buffer and registration, LazyPinned's lazy copies: a copy runs
    only when waited for, as on a stream the host hasn't synchronised with."""

    def __init__(self, host: HostBuffer) -> None:
        LazyPinned.__init__(self)
        self.host = host


def _stand_in(env, rank):
    """In each rank: GPU tensors are CPU tensors, the direct path is unavailable (with a
    reason) and staging copies are lazy."""
    os.environ.update(env)
    stagings = {}

    def staging_for(self, tensor):
        if 0 not in stagings:
            stagings[0] = LazyRegistered(HostBuffer(self.worker,
                                                    lambda n: torch.zeros(n, dtype=torch.uint8)))
        self._staging = stagings
        return stagings[0]

    vt.VerbsTransport._is_device = lambda self, tensor: True
    vt.VerbsTransport._device_index = lambda self, tensor: 0
    vt.VerbsTransport._support_for = lambda self, index: vt.Support(
        False, "standing in: no GPU in this test")
    vt.VerbsTransport._staging_for = staging_for


def _check_staged(obs_dir, topology):
    bridge = topology.bridge
    rank = topology.rank
    if bridge is not None:
        members = len(bridge.members)
        # Big enough for the rendezvous (writes into the registered staging buffer) and
        # several chunks; small enough to be quick.
        for numel in (1, 1000, 300_000):
            tensor = rank_tensor((numel,), torch.float32, rank)
            want = sum(rank_tensor((numel,), torch.float32, bridge.members[i])
                       for i in range(members))
            bridge.all_reduce(tensor)
            bridge._staging[0].run_streams()  # the device gets to the queued copy back
            assert torch.equal(tensor, want), numel
            t = bridge.telemetry()
            assert (t["path"], t["path_reason"]) == ("staged", "standing in: no GPU in this test")
            src = bridge.members[0]
            tensor = rank_tensor((numel,), torch.float32, rank)
            bridge.broadcast(tensor, src)
            bridge._staging[0].run_streams()
            assert torch.equal(tensor, rank_tensor((numel,), torch.float32, src)), numel
        log = bridge._staging[0].log
        # Every copy out was waited for before the bridge used the buffer.
        copies_out = [e[1] for e in log if e[0] == "copy"][::2]
        waited = {e[1] for e in log if e[0] == "wait"}
        assert set(copies_out) <= waited, log
        t = bridge.telemetry()
        assert t["paths"]["staged"] == 6
        assert t["direct"]["registered"]["host"] >= 1, t["direct"]
        assert t["in_place_ops"] > 0 and t["bytes_written"] > 0, t
        assert "key" not in json.dumps(t).lower()
    dist.barrier()


@needs_library
def test_the_staged_path_with_cpu_tensors_standing_in_and_its_observations(tmp_path):
    fabric = tempfile.mkdtemp(prefix="gbvf-", dir="/tmp")
    obs = tmp_path / "obs"
    env = {verbs.TEST_FABRIC_ENV: fabric, observe.OBSERVE_ENV: str(obs / "rank-{rank}.jsonl"),
           observe.JOB_ENV: "job-a3", vt.RNDV_BYTES_ENV: "64K", vt.CHUNK_BYTES_ENV: "256K"}
    try:
        run_ranks(LAYOUTS["2+2"][0], partial(_check_staged, str(obs)), tmp_path,
                  init_kwargs={"bridge": "verbs"}, before_init=partial(_stand_in, env))
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    files = sorted(Path(obs).glob("rank-*.jsonl"))
    assert len(files) == 2, files  # the two leaders
    for path in files:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        # 6 calls (3 all_reduce, 3 broadcast), one peer: a record per call and direction
        # that moved anything (the broadcast root only sends, the others only receive).
        assert len(records) == 9, records
        for r in records:
            observe.check_record(r)
            assert r["schema"] == observe.SCHEMA
            assert r["job"] == "job-a3" and r["host"] and r["bridge"] == "verbs"
            assert "pci_bus_id" in r["nic"] and r["nic"]["device"] == "mockfab0"
            assert r["nic"]["link_layer"] == "InfiniBand"
            assert r["peer_nic"]["device"] == "mockfab0"
            assert r["op"] in ("send", "recv") and r["status"] == "ok"
            assert r["path"] == "staged" and r["bytes"] > 0 and r["duration_s"] >= 0
            assert r["t_wall"] > 1.7e9 and isinstance(r["rank"], int)
            assert r["peer"] != r["rank"]
