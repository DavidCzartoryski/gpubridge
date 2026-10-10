"""The sharded bridge: k ranks per island share the bridge step, one segment each.

test_policies.py and test_ops.py already run it in their layouts (2+2, 3+1,
interleaved, split test). These tests add the layouts where it differs from
reduce-bridge-broadcast: islands of different sizes above one (one island
takes the reduce_scatter path, the other the k reduces), three islands, and
sizes that leave padding.
"""

import json
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _harness import expected_sum, rank_tensor, run_ranks
from _transports import BRIDGE_CALLS, STORE_ENV, CountingGloo, StoreTransport
from test_ops import FLOATS, OP_NAMES, Keep, expected_reduce, gather, reference_all_reduce

import gpubridge
import gpubridge.topology
from gpubridge import ReduceOp
from gpubridge.collectives import SUPPORTED_DTYPES
from gpubridge.config import THRESHOLDS_ENV
from gpubridge.policies import (
    AUTO,
    FlatGloo,
    ReduceBridgeBroadcast,
    ShardedBridge,
    choose_policy,
)
from gpubridge.topology import plan_topology

SHARDED = ShardedBridge.name

#: (vendors, island labels if patched, k)
LAYOUTS = {
    "3+2": (["nvidia"] * 3 + ["amd"] * 2, None, 2),
    "2+4-interleaved": (["nvidia", "amd", "nvidia", "amd", "amd", "amd"], None, 2),
    "3+3": (["amd"] * 3 + ["nvidia"] * 3, None, 3),
    # gpubridge knows two vendors; a third island needs patched labels.
    "three-islands-2+3+2": (["nvidia"] * 7, ["nvidia", "amd", "third", "amd", "nvidia", "amd",
                                             "third"], 2),
}
#: Element counts: even splits, padding, and fewer elements than links. Small
#: enough that every sum stays exact in bfloat16 and int8 with seven ranks.
SIZES = (1, 2, 5, 6, 7, 13)


def _label_islands(labels, rank):
    original = gpubridge.topology.plan_topology
    gpubridge.topology.plan_topology = lambda vendors: original(labels)


def _run(vendors, labels, fn, tmp_path, **kwargs):
    before = partial(_label_islands, labels) if labels else None
    run_ranks(vendors, fn, tmp_path, before_init=before,
              init_kwargs={"policy": SHARDED, **kwargs.pop("init_kwargs", {})}, **kwargs)


def _check_matches_flat_gloo(topology, k):
    # On CPU, staging returns the tensor itself, so the copy back that GPUs need
    # would never run. Force a real copy so it does.
    gpubridge.policies._stage_to_host = torch.Tensor.clone
    assert topology.policy_name == SHARDED and topology.policy.k == k
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
    # A shaped tensor, async, with a sync call behind it.
    a = rank_tensor((3, 7), torch.bfloat16, rank)
    work = gpubridge.all_reduce(a, async_op=True)
    b = rank_tensor((4,), torch.int64, rank)
    gpubridge.broadcast(b, src=world - 1)
    assert work.wait() is True
    assert torch.equal(a, expected_sum((3, 7), torch.bfloat16, world))
    assert torch.equal(b, rank_tensor((4,), torch.int64, world - 1))


@pytest.mark.parametrize(("vendors", "labels", "k"), list(LAYOUTS.values()), ids=list(LAYOUTS))
def test_sharded_bridge_matches_flat_gloo(vendors, labels, k, tmp_path):
    _run(vendors, labels, partial(_check_matches_flat_gloo, k=k), tmp_path)


def test_auto_never_picks_the_sharded_bridge():
    for vendors, labels, _ in LAYOUTS.values():
        assert choose_policy(AUTO, plan_topology(labels or vendors)) is ReduceBridgeBroadcast


# ---- each link carries one segment ------------------------------------------------------

def _check_bridge_load(topology, k, numel):
    gpubridge.policies._stage_to_host = torch.Tensor.clone
    policy = topology.policy
    index = topology.island.ranks.index(topology.rank)
    # Links 1..k-1 came from one create_groups call on every rank: rank j of each island.
    islands = topology.layout.islands
    want_groups = [sorted(island.ranks[j] for island in islands) for j in range(1, k)]
    assert CountingGloo.groups == ([want_groups] if k > 1 else [])
    assert (policy.link is not None) == (index < k)
    if index == 0:
        assert policy.link is topology.bridge
    BRIDGE_CALLS.clear()
    tensor = rank_tensor((numel,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((numel,), torch.float32, topology.world_size))
    segment = -(-numel // k)
    assert BRIDGE_CALLS == ([("all_reduce", segment)] if index < k else [])
    # Every link moved the same number of bytes: 1/k of the (padded) tensor.
    loads = [load for load in gather(list(BRIDGE_CALLS)) if load]
    assert len(loads) == k * len(islands)


@pytest.mark.parametrize(("vendors", "labels", "k"), [
    (["nvidia", "nvidia", "amd", "amd"], None, 2),
    (["nvidia"] * 3 + ["amd"], None, 1),
    *[LAYOUTS[name] for name in ("3+2", "three-islands-2+3+2")],
], ids=["2+2", "3+1", "3+2", "three-islands"])
def test_each_link_carries_one_segment(vendors, labels, k, tmp_path):
    _run(vendors, labels, partial(_check_bridge_load, k=k, numel=1001), tmp_path,
         init_kwargs={"bridge": CountingGloo})


def _check_store_links(topology):
    gpubridge.policies._stage_to_host = torch.Tensor.clone
    link = topology.policy.link
    for numel in SIZES:
        tensor = rank_tensor((numel,), torch.float32, topology.rank)
        gpubridge.all_reduce(tensor)
        assert torch.equal(tensor, expected_sum((numel,), torch.float32, topology.world_size))
    last = topology.world_size - 1
    tensor = rank_tensor((SIZES[-1],), torch.float32, topology.rank)
    gpubridge.broadcast(tensor, src=last)
    assert torch.equal(tensor, rank_tensor((SIZES[-1],), torch.float32, last))
    # Every island rank is on a link here (k = 2 = both island sizes).
    assert isinstance(link, StoreTransport)
    assert link.calls == {"all_reduce": len(SIZES), "broadcast": 1}


def test_a_transport_with_only_create_gets_several_links(tmp_path, monkeypatch):
    # StoreTransport implements create(), not create_groups(): the default makes the links.
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    _run(["nvidia", "amd", "nvidia", "amd"], None, _check_store_links, tmp_path,
         init_kwargs={"bridge": StoreTransport})


# ---- observers see which island step each rank took ----------------------------------

def _check_phases(topology):
    keep = Keep()
    gpubridge.add_observer(keep)
    tensor = rank_tensor((9,), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    phases = [p.name for p in keep.records[-1].phases]
    index = topology.island.ranks.index(topology.rank)
    on_link = ["bridge"] if index < 2 else []
    if len(topology.island.ranks) == 2:
        assert phases == ["island-reduce-scatter", *on_link, "island-all-gather"]
    else:
        assert phases == ["island-reduce", *on_link, "island-broadcast"]


def test_observers_see_each_islands_steps(tmp_path):
    _run(*LAYOUTS["3+2"][:2], _check_phases, tmp_path)


# ---- auto-tuned can route sizes to it ------------------------------------------------

def _check_auto_tuned_rule(topology):
    sub = [policy.name for _, policy in topology.policy.rules]
    assert sub == ["flat-gloo", SHARDED]
    for numel in (3, 1001):  # 12 bytes and 4004 bytes: one per rule
        tensor = rank_tensor((numel,), torch.float32, topology.rank)
        want = tensor.clone()
        gpubridge.all_reduce(tensor)
        FlatGloo().all_reduce(want, topology)
        assert torch.equal(tensor, want)
    assert dist.get_world_size() == topology.world_size


def test_auto_tuned_can_route_large_sizes_to_it(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"gpubridge_thresholds": 1, "rules": [
        {"max_bytes": 64, "policy": "flat-gloo"}, {"max_bytes": None, "policy": SHARDED}]}))
    monkeypatch.setenv(THRESHOLDS_ENV, str(path))
    run_ranks(["nvidia", "amd", "nvidia", "amd", "amd"], _check_auto_tuned_rule, tmp_path,
              init_kwargs={"policy": "auto-tuned"})
