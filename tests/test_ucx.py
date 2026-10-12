"""The UCX bridge transport, ``bridge="ucx"`` (docs/RDMA_TRANSPORT.md).

On CPU, between processes on one machine, over UCX's TCP transport: the same
code that would run over InfiniBand or RoCE.
Every bridge policy runs on it and must match ``flat-gloo`` bit for bit. The
transport tests need the UCX library and skip where it can't be built
(macOS, no UCX, no compiler); the pure-Python parts run everywhere.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _harness import rank_tensor, run_failing_init, run_ranks
from test_ops import FLOATS, OP_NAMES, expected_reduce, reference_all_reduce
from torch.distributed import ReduceOp

import gpubridge
from gpubridge import ucx
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import VENDOR_ENV
from gpubridge.policies import FlatGloo
from gpubridge.ucx import transport as ucx_transport
from gpubridge.ucx.transport import CHUNK_BYTES_ENV, UcxTransport

try:
    ucx.build()
    UNAVAILABLE = None
except ucx.Unavailable as exc:
    UNAVAILABLE = str(exc).splitlines()[0]
needs_ucx = pytest.mark.skipif(UNAVAILABLE is not None,
                               reason=f"UCX bridge unavailable: {UNAVAILABLE}")

#: Over TCP between the processes, as between two hosts.
TCP = {"UCX_TLS": "tcp,self", "UCX_WARN_UNUSED_ENV_VARS": "n"}

#: (vendors, island labels if patched): 2, 3 and 4 leaders.
LAYOUTS = {
    "2+2": (["nvidia", "nvidia", "amd", "amd"], None),
    "1+3": (["amd", "nvidia", "nvidia", "nvidia"], None),
    "three-islands-2+2+1": (["nvidia"] * 5, ["nvidia", "amd", "third", "nvidia", "amd"]),
    "four-islands": (["nvidia"] * 4, ["nvidia", "amd", "third", "fourth"]),
}
POLICIES = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge")
#: Element counts for every dtype: one element, fewer than leaders, odd. Small
#: enough that every sum stays exact in float16, bfloat16 and int8 (see
#: rank_tensor); with 16-byte chunks most are still several messages.
SIZES = (1, 3, 7, 13)


# ---- pure Python ----------------------------------------------------------------------

def test_tags_keep_collective_step_sender_and_chunk_apart():
    fields = [(seq, step, sender, chunk) for seq in (1, 2, 2**24 - 1) for step in (0, 1, 4095)
              for sender in (0, 1, 255) for chunk in (0, 1, 2**20 - 1)]
    tags = {ucx_transport.tag(*f) for f in fields}
    assert len(tags) == len(fields) and all(0 <= t < 2**64 for t in tags)


@pytest.mark.parametrize("numel", [0, 1, 3, 7, 10])
@pytest.mark.parametrize("parts", [2, 3, 4])
def test_segments_cover_the_tensor_once(numel, parts):
    segments = ucx_transport._segments(numel, parts)
    assert len(segments) == parts
    assert [start for start, _ in segments] == [sum(n for _, n in segments[:i])
                                               for i in range(parts)]
    assert sum(n for _, n in segments) == numel
    assert max(n for _, n in segments) - min(n for _, n in segments) <= 1


def test_chunks_cover_the_buffer_once():
    assert ucx_transport._pieces(0, 8) == [(0, 0)]
    assert ucx_transport._pieces(20, 8) == [(0, 8), (8, 8), (16, 4)]
    assert ucx_transport._pieces(16, 8) == [(0, 8), (8, 8)]


def test_chunk_size_setting(monkeypatch):
    monkeypatch.delenv(CHUNK_BYTES_ENV, raising=False)
    assert ucx_transport.chunk_bytes_setting() == 4 * 2**20
    monkeypatch.setenv(CHUNK_BYTES_ENV, "64K")
    assert ucx_transport.chunk_bytes_setting() == 65536
    monkeypatch.setenv(CHUNK_BYTES_ENV, "12")
    with pytest.raises(ValueError, match="multiple of 8"):
        ucx_transport.chunk_bytes_setting()


def test_ucx_is_a_known_bridge():
    assert gpubridge.resolve_transport("ucx") is UcxTransport


@pytest.mark.skipif(UNAVAILABLE is None, reason="UCX is available here")
def test_without_ucx_the_bridge_says_why():
    with pytest.raises(ucx.Unavailable):
        ucx.load()


# ---- every policy on it, bit for bit --------------------------------------------------

def _label_islands(labels, env, rank):
    os.environ.update(env)
    if labels:
        original = gpubridge.topology.plan_topology
        gpubridge.topology.plan_topology = lambda vendors: original(labels)


def _check_matches_flat_gloo(topology, policy):
    gpubridge.policies._stage_to_host = torch.Tensor.clone  # run the copy back too
    assert topology.policy_name == policy
    if topology.is_leader:
        assert isinstance(topology.bridge, UcxTransport)
    rank, world = topology.rank, topology.world_size
    for numel in SIZES:
        for dtype in SUPPORTED_DTYPES:
            ops = [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN]
            if dtype in FLOATS:
                ops.append(ReduceOp.AVG)
            for op in ops:
                tensor = rank_tensor((numel,), dtype, rank)
                want = tensor.clone()
                gpubridge.all_reduce(tensor, op=op)
                reference_all_reduce(want, op, topology)
                assert torch.equal(tensor, want), (numel, dtype, OP_NAMES[op])
                assert torch.equal(tensor, expected_reduce((numel,), dtype, world, op))
        for src in range(world):
            tensor = rank_tensor((numel,), torch.float32, rank)
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            FlatGloo().broadcast(want, src, topology)
            assert torch.equal(tensor, want), (numel, src)
    rows = rank_tensor((world, 5), torch.float32, rank)
    got, want = torch.empty(5), torch.empty(5)
    gpubridge.reduce_scatter_tensor(got, rows)
    FlatGloo().reduce_scatter_tensor(want, rows, topology)
    assert torch.equal(got, want)
    a = rank_tensor((3, 7), torch.bfloat16, rank)
    work = gpubridge.all_reduce(a, async_op=True)
    b = rank_tensor((4,), torch.int64, rank)
    gpubridge.broadcast(b, src=world - 1)
    assert work.wait() is True
    assert torch.equal(a, expected_reduce((3, 7), torch.bfloat16, world, ReduceOp.SUM))
    assert torch.equal(b, rank_tensor((4,), torch.int64, world - 1))


@needs_ucx
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("vendors", "labels"), list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_every_policy_on_the_ucx_bridge_matches_flat_gloo(vendors, labels, policy, tmp_path):
    # 16-byte chunks: most messages are several chunks, all posted at once.
    env = {**TCP, CHUNK_BYTES_ENV: "16"}
    run_ranks(vendors, partial(_check_matches_flat_gloo, policy=policy), tmp_path,
              before_init=partial(_label_islands, labels, env),
              init_kwargs={"bridge": "ucx", "policy": policy})


@needs_ucx
def test_the_ucx_bridge_with_ucxs_own_choice_of_transport(tmp_path):
    # No UCX_TLS: UCX picks among the transports that can report a failed peer
    # (not shared memory, which can't), as it would on a cluster.
    env = {"UCX_WARN_UNUSED_ENV_VARS": "n"}
    run_ranks(LAYOUTS["2+2"][0],
              partial(_check_matches_flat_gloo, policy="reduce-bridge-broadcast"), tmp_path,
              before_init=partial(_label_islands, None, env),
              init_kwargs={"bridge": "ucx", "policy": "reduce-bridge-broadcast"})


def _check_large(topology):
    # float32 and int64 hold these sums exactly; odd sizes leave a short last chunk.
    rank, world = topology.rank, topology.world_size
    for dtype, numel in ((torch.float32, 2**20 + 3), (torch.int64, 3 * 2**19 + 1),
                         (torch.float32, 4099)):
        for op in (ReduceOp.SUM, ReduceOp.MAX):
            tensor = rank_tensor((numel,), dtype, rank)
            gpubridge.all_reduce(tensor, op=op)
            assert torch.equal(tensor, expected_reduce((numel,), dtype, world, op)), (dtype, op)
        tensor = rank_tensor((numel,), dtype, rank)
        gpubridge.broadcast(tensor, world - 1)
        assert torch.equal(tensor, rank_tensor((numel,), dtype, world - 1))


@needs_ucx
@pytest.mark.parametrize("layout", ["2+2", "three-islands-2+2+1"])
def test_large_tensors_in_many_chunks(layout, tmp_path):
    vendors, labels = LAYOUTS[layout]
    run_ranks(vendors, _check_large, tmp_path,
              before_init=partial(_label_islands, labels, {**TCP, CHUNK_BYTES_ENV: "1M"}),
              init_kwargs={"bridge": "ucx"})


# ---- failures -------------------------------------------------------------------------

@needs_ucx
def test_a_leader_whose_ucx_fails_to_start_fails_init_on_every_rank(tmp_path):
    good = {VENDOR_ENV: "nvidia", **TCP}
    run_failing_init(
        [{"env": good, "init_kwargs": {"bridge": "ucx"}},
         {"env": {**good, VENDOR_ENV: "amd", "UCX_TLS": "no-such-transport"},
          "init_kwargs": {"bridge": "ucx"}},
         {"env": good, "init_kwargs": {"bridge": "ucx"}}],
        tmp_path,
        [r"The UCX bridge between ranks \[0, 1\] could not be set up",
         r'bridge="gloo"', r"rank 1: UcxError: UCX initialization failed"],
    )


@needs_ucx
def test_a_bad_setting_on_one_leader_fails_init_on_every_rank(tmp_path):
    good = {VENDOR_ENV: "nvidia", **TCP}
    run_failing_init(
        [{"env": {**good, CHUNK_BYTES_ENV: "12"}, "init_kwargs": {"bridge": "ucx"}},
         {"env": {**good, VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": "ucx"}}],
        tmp_path,
        [r"could not be set up", rf"rank 0: ValueError: {CHUNK_BYTES_ENV}='12' must be"],
    )


def _check_deadline(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        bridge.timeout = 1.0
        tensor = torch.ones(10)
        start = time.monotonic()
        with pytest.raises(ucx.UcxError, match=r"timed out with \d+ UCX operation"):
            bridge.all_reduce(tensor)  # the other leader never joins
        assert time.monotonic() - start < 10
        with pytest.raises(ucx.UcxError, match="already failed"):
            bridge.all_reduce(tensor)
    dist.barrier()  # on the world Gloo group, not the bridge


@needs_ucx
def test_a_leader_that_never_joins_trips_the_deadline(tmp_path):
    run_ranks(["nvidia", "amd"], _check_deadline, tmp_path,
              before_init=partial(_label_islands, None, TCP), init_kwargs={"bridge": "ucx"})


def _check_teardown(topology):
    bridge = topology.bridge
    if bridge is not None and bridge.index == 0:
        never = torch.empty(4)
        bridge.worker.recv(never.data_ptr(), 16, ucx_transport.tag(99, 0, 1, 0))
    dist.barrier()
    start = time.monotonic()
    gpubridge.destroy()  # closes the bridge with a receive still posted
    assert time.monotonic() - start < 15
    gpubridge.init(init_method=f"file://{os.environ['GB_RDZV']}-again", rank=topology.rank,
                   world_size=topology.world_size, bridge="ucx",
                   timeout=timedelta(seconds=60))
    tensor = rank_tensor((5,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)  # a new bridge works after the old one closed
    assert torch.equal(tensor, expected_reduce((5,), torch.float32, topology.world_size,
                                               ReduceOp.SUM))


@needs_ucx
def test_teardown_with_an_operation_outstanding(tmp_path):
    env = {**TCP, "GB_RDZV": str(tmp_path / "rdzv")}
    run_ranks(["nvidia", "amd"], _check_teardown, tmp_path,
              before_init=partial(_label_islands, None, env), init_kwargs={"bridge": "ucx"})
