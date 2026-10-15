"""P2P routing, matching, concurrency and failure on CPU processes."""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpubridge import topology
from gpubridge import p2p
from gpubridge.config import VENDOR_ENV
from gpubridge.torch_backend import GpubridgeProcessGroup


def _main(rank, vendors, labels, fn, rendezvous, seconds):
    os.environ[VENDOR_ENV] = vendors[rank]
    torch.set_num_threads(1)
    if labels:
        original = topology.plan_topology
        topology.plan_topology = lambda vendors: original(labels)
    dist.init_process_group("gpubridge", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=len(vendors), timeout=timedelta(seconds=seconds))
    try:
        fn(rank, len(vendors))
    finally:
        dist.destroy_process_group()


def run(vendors, fn, tmp_path, *, labels=None, seconds=15):
    mp.spawn(_main, nprocs=len(vendors), args=(vendors, labels, fn,
                                             str(tmp_path / "rendezvous"), seconds))


DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
          torch.float16, torch.bfloat16, torch.float32, torch.float64,
          torch.complex64, torch.complex128)


def _payload(n, dtype, src, seq=0):
    return ((torch.arange(n) % 97) + src * 3 + seq).to(dtype)


def _all_pairs(rank, world):
    dist.barrier()
    for src in range(world):
        for dst in range(world):
            if src == dst:
                continue
            for i, dtype in enumerate(DTYPES):
                for n in (0, 1, 31, 4097):
                    t = _payload(n, dtype, src) if rank == src else torch.empty(n, dtype=dtype)
                    if rank == src:
                        dist.send(t, dst, tag=i)
                    elif rank == dst:
                        assert dist.recv(t, src, tag=i) == src
                        assert torch.equal(t, _payload(n, dtype, src)), (src, dst, dtype, n)
    # Gloo native path and bridge path both ran, including non-leader ranks.
    dist.barrier()


@pytest.mark.parametrize(("vendors", "labels"), [
    (["nvidia", "amd"], None),
    (["nvidia", "nvidia", "amd", "amd"], None),
    (["nvidia", "amd", "nvidia", "nvidia"], None),
    (["nvidia"] * 4, ["nvidia", "amd", "third", "nvidia"]),
], ids=["1+1", "2+2", "uneven", "three-islands"])
def test_every_pair_sizes_and_dtypes(vendors, labels, tmp_path):
    run(vendors, _all_pairs, tmp_path, labels=labels)


def _tags_and_batches(rank, world):
    dist.barrier()
    peer = 1 - rank
    # Different tags may be received in the opposite order. A single worker
    # waiting for the first receive would deadlock this exchange.
    if rank == 0:
        sends = [dist.isend(_payload(4097, torch.int64, rank, tag), peer, tag=tag)
                 for tag in (2**31 - 1, 12, 3)]
        for work in sends:
            work.wait()
    else:
        for tag in (3, 12, 2**31 - 1):
            t = torch.empty(4097, dtype=torch.int64)
            dist.recv(t, peer, tag=tag)
            assert torch.equal(t, _payload(4097, torch.int64, 0, tag))
    # Same tag, same pair: each message must match in program order.
    receive = [torch.empty(257, dtype=torch.int64) for _ in range(12)]
    ops = []
    for seq in range(12):
        ops.extend([dist.P2POp(dist.irecv, receive[seq], peer, tag=99),
                    dist.P2POp(dist.isend, _payload(257, torch.int64, rank, seq), peer, tag=99)])
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    for seq, t in enumerate(receive):
        assert torch.equal(t, _payload(257, torch.int64, peer, seq)), seq
    dist.barrier()


def test_out_of_order_tags_and_full_duplex_batch_fifo(tmp_path):
    run(["nvidia", "amd"], _tags_and_batches, tmp_path)


def _concurrent(rank, world):
    dist.barrier()
    receives, works = [], []
    for peer in reversed(range(world)):
        if peer == rank:
            continue
        incoming = torch.empty(2**18, dtype=torch.int32)
        receives.append((peer, incoming))
        works.append(dist.irecv(incoming, peer, tag=41))
        works.append(dist.isend(_payload(2**18, torch.int32, rank), peer, tag=41))
    for work in reversed(works):
        work.wait()
    for peer, incoming in receives:
        assert torch.equal(incoming, _payload(2**18, torch.int32, peer))
    dist.barrier()


def test_all_pairs_in_flight_native_and_bridged(tmp_path):
    run(["nvidia", "nvidia", "amd", "amd"], _concurrent, tmp_path)


def _slow_receiver(rank, world):
    dist.barrier()
    if rank == 0:
        slow = dist.isend(torch.ones(11), 1, tag=7)
        fast = dist.isend(torch.full((9,), 2.), 2, tag=8)
        fast.wait()
        assert not slow.is_completed(), "the slow receive must still be waiting"
        slow.wait()
    elif rank == 1:
        time.sleep(0.6)
        t = torch.empty(11)
        dist.recv(t, 0, tag=7)
        assert torch.equal(t, torch.ones(11))
    else:
        t = torch.empty(9)
        dist.recv(t, 0, tag=8)
        assert torch.equal(t, torch.full((9,), 2.))
    dist.barrier()


def test_a_slow_peer_does_not_block_other_pairs(tmp_path):
    run(["nvidia", "amd", "amd"], _slow_receiver, tmp_path)


def _subgroup(rank, world):
    dist.barrier()
    group = dist.new_group([1, 3])
    if rank in (1, 3):
        dist.barrier(group=group)
        t = torch.full((17,), 42) if rank == 3 else torch.empty(17, dtype=torch.int64)
        if rank == 3:
            dist.send(t, 1, group=group, tag=5)
        else:
            dist.recv(t, 3, group=group, tag=5)
            assert torch.equal(t, torch.full((17,), 42))
        dist.barrier(group=group)
    dist.barrier()


def test_subgroup_peers_are_global_ranks(tmp_path):
    run(["nvidia", "nvidia", "amd", "amd"], _subgroup, tmp_path)


def _refusals(rank, world):
    with pytest.raises(RuntimeError, match="call dist.barrier"):
        dist.isend(torch.ones(1), 1 - rank)
    dist.barrier()
    with pytest.raises(ValueError, match="contiguous"):
        dist.isend(torch.ones(2, 3).T, 1 - rank)
    with pytest.raises(ValueError, match="tag"):
        dist.isend(torch.ones(1), 1 - rank, tag=-1)
    with pytest.raises(NotImplementedError, match="any-source"):
        dist.irecv(torch.empty(1))
    dist.barrier()


def test_refusals_do_not_start_communication(tmp_path):
    run(["nvidia", "amd"], _refusals, tmp_path)


def _mismatch(rank, world, dtype):
    dist.barrier()
    t = torch.ones(4 if rank == 0 or dtype else 3,
                   dtype=torch.int64 if rank == 0 or not dtype else torch.float64)
    work = dist.isend(t, 1) if rank == 0 else dist.irecv(t, 0)
    with pytest.raises(RuntimeError, match="payload mismatch"):
        work.wait()
    with pytest.raises(RuntimeError, match="payload mismatch"):
        dist.barrier()


@pytest.mark.parametrize("dtype", [False, True], ids=["size", "dtype"])
def test_payload_mismatch_errors_on_both_ranks_without_gloo_abort(dtype, tmp_path):
    run(["nvidia", "amd"], partial(_mismatch, dtype=dtype), tmp_path)


def _missing(rank, world):
    dist.barrier()
    if rank == 1:
        # This is a graceful process exit with no posted receive. The sender's
        # metadata deadline must fail, and teardown must not wait for a peer.
        return
    start = time.monotonic()
    work = dist.isend(torch.ones(13), 1)
    with pytest.raises(RuntimeError, match="timed out"):
        work.wait()
    assert time.monotonic() - start < 5


def test_missing_peer_times_out_and_teardown_is_bounded(tmp_path):
    run(["nvidia", "amd"], _missing, tmp_path, seconds=1)


def _dead_peer(rank, world):
    dist.barrier()
    if rank == 1:
        os._exit(0)  # no cleanup: simulate a peer disappearing during the job
    if rank == 0:
        work = dist.isend(torch.ones(13), 1)
    else:
        time.sleep(0.35)  # its own deadline is later: it must see rank 0's failure
        work = dist.irecv(torch.empty(13), 0, tag=19)
    with pytest.raises((RuntimeError, TimeoutError), match="rank 1.*timed out"):
        work.get_future().wait()
    with pytest.raises(RuntimeError, match="timed out"):
        dist.barrier()


def test_dead_peer_failure_reaches_other_pending_pairs(tmp_path):
    run(["nvidia", "amd", "amd"], _dead_peer, tmp_path, seconds=1)


class _BlockedLink:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def post(self, tensor, peer, tag, sending):
        self.entered.set()
        return self

    def wait(self, timeout):
        self.release.wait(2)

    def close(self):
        pass


def test_teardown_fails_handle_without_waiting_for_inflight_transport(monkeypatch):
    link = _BlockedLink()
    engine = p2p.Engine(link, torch.device("cpu"), 0, dist.HashStore(), timedelta(seconds=5))
    monkeypatch.setattr(engine, "match", lambda *args: None)
    work = engine.submit(torch.ones(3), 1, 0, True)
    assert link.entered.wait(2)
    start = time.monotonic()
    engine.close(timeout=0.02)
    assert time.monotonic() - start < 0.2
    with pytest.raises(RuntimeError, match="destroyed"):
        work.wait()
    link.release.set()


def test_torch_work_wait_and_future_both_report_transfer_failure(monkeypatch):
    link = _BlockedLink()

    def fail(timeout):
        raise ValueError("injected transfer failure")

    monkeypatch.setattr(link, "wait", fail)
    engine = p2p.Engine(link, torch.device("cpu"), 0, dist.HashStore(), timedelta(seconds=5))
    monkeypatch.setattr(engine, "match", lambda *args: None)
    work = p2p.TorchWork(engine.submit(torch.ones(3), 1, 0, True), 0)
    try:
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            work.wait()
        with pytest.raises(ValueError, match="injected transfer failure"):
            work.get_future().wait()
        assert work.is_completed() and not work.is_success()
    finally:
        engine.close()


def test_tag_range_is_checked_without_a_process_group():
    pg = GpubridgeProcessGroup(0, 2, "unit", None)
    with pytest.raises(ValueError, match="tag"):
        pg.send([torch.empty(1)], 1, 2**31)
