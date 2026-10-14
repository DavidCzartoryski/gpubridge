"""The small-message fast path: what a collective works out once instead of on every
call, and that every call still makes the same checks."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from _harness import run_ranks

import gpubridge
from gpubridge import ReduceOp, collectives
from gpubridge.topology import Layout

POLICIES = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast", "sharded-bridge",
            "flat-gloo", "auto-tuned")  # flat-gloo never needs the island: the control


def _island_looked_up_once(topology: gpubridge.Topology) -> None:
    looked_up = []
    real = Layout.island_of

    def counting(self: Layout, rank: int):
        looked_up.append(rank)
        return real(self, rank)

    Layout.island_of = counting
    try:
        t = torch.ones(4)
        for _ in range(10):
            gpubridge.all_reduce(t)
            gpubridge.all_reduce(t, async_op=True).wait()
    finally:
        Layout.island_of = real
    assert torch.equal(t, torch.full((4,), 4.0**20))
    assert len(looked_up) <= 1, looked_up  # at most the first call, never again


@pytest.mark.parametrize("policy", POLICIES)
def test_a_rank_looks_up_its_island_once(tmp_path, policy):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _island_looked_up_once, tmp_path,
              init_kwargs={"policy": policy})


def _support_asked_once_and_checked_every_call(topology: gpubridge.Topology) -> None:
    policy = topology.policy.select_policy(8)  # the one that carries it (auto-tuned picks one)
    policy.__dict__.pop("_gpubridge_reduce_ops", None)
    cls, asked = type(policy), []
    real = cls.supported_reduce_ops

    def only_sum(self, topology: gpubridge.Topology) -> frozenset[str]:
        asked.append(topology)
        return frozenset({"sum"})

    cls.supported_reduce_ops = only_sum
    try:
        t = torch.ones(2)
        for async_op in (False, True, False, True):
            work = gpubridge.all_reduce(t, async_op=async_op)
            if work is not None:
                work.wait()
            with pytest.raises(NotImplementedError, match="supports only SUM"):
                gpubridge.all_reduce(t, op=ReduceOp.MAX, async_op=async_op)
    finally:
        cls.supported_reduce_ops = real
    assert asked == [topology]
    assert torch.equal(t, torch.full((2,), 4.0**4))


@pytest.mark.parametrize("policy", ("reduce-bridge-broadcast", "auto-tuned"))
def test_the_policy_is_asked_its_reduce_ops_once_and_every_call_is_still_checked(tmp_path, policy):
    run_ranks(["nvidia", "nvidia", "amd", "amd"], _support_asked_once_and_checked_every_call,
              tmp_path, init_kwargs={"policy": policy})


class _Asked:
    def __init__(self) -> None:
        self.asked = 0

    def supported_reduce_ops(self, topology: object) -> frozenset[str]:
        self.asked += 1
        return frozenset({str(id(topology))})


class _Topology:  # weak-referenceable, as gpubridge's is
    pass


def test_a_new_topology_is_never_given_an_old_answer():
    policy, first, second = _Asked(), _Topology(), _Topology()
    ask = partial(collectives._supported_reduce_ops, policy)
    assert ask(first) == ask(first) == {str(id(first))} and policy.asked == 1
    assert ask(second) == {str(id(second))} and policy.asked == 2  # after destroy() and init()


def test_a_policy_with_slots_is_asked_every_call():
    class Slotted:
        __slots__ = ("asked",)

        def __init__(self) -> None:
            self.asked = 0

        def supported_reduce_ops(self, topology: object) -> frozenset[str]:
            self.asked += 1
            return frozenset({"sum"})

    policy, topology = Slotted(), _Topology()
    for _ in range(3):
        assert collectives._supported_reduce_ops(policy, topology) == {"sum"}
    assert policy.asked == 3


def test_reduce_ops_are_named_by_lookup_and_by_comparison():
    for op, name in ((ReduceOp.SUM, "sum"), (ReduceOp.AVG, "avg"), (ReduceOp.MAX, "max"),
                     (ReduceOp.MIN, "min")):
        assert collectives._reduce_op_name(op, "all_reduce") == name
        # torch.distributed's ReduceOp objects (the torch backend passes these)
        assert collectives._reduce_op_name(torch.distributed.ReduceOp(op), "all_reduce") == name
    with pytest.raises(ValueError, match="PRODUCT is not supported yet"):
        collectives._reduce_op_name(ReduceOp.PRODUCT, "all_reduce")


def test_the_per_call_benchmark_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_per_call.py"
    out = tmp_path / "per_call.json"
    done = subprocess.run([sys.executable, str(script), "--stub-comm", "--policies",
                           "reduce-bridge-broadcast", "--numels", "1", "--calls", "5", "--warmup", "1",
                           "--rounds", "1", "--out", str(out)], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    rows = json.loads(out.read_text())["rows"]
    paths = {(r["mode"], r["path"], r["observer"]) for r in rows}
    assert ("sync", "gpubridge", "observer") in paths and ("async", "bare, same calls", "-") in paths
    assert all(r["overhead_us"] is not None for r in rows if r["path"] == "gpubridge")
