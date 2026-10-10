"""Correctness on real GPUs: every policy against flat-gloo, every op, fp32/fp16/bf16.

Marked ``gpu`` and skipped without a GPU. On a GPU node, run
``pytest -m gpu tests/test_gpu.py`` (the Explorer jobs and the AMD script do).
No launcher is needed: each test spawns its ranks. With 4+ GPUs, 4 ranks on
separate GPUs; with fewer, 2 ranks sharing GPU 0, which only split-test mode
allows.

The two-island cases use split-test mode (``GPUBRIDGE_SPLIT_TEST=half``): two
real NCCL (or RCCL) islands joined by the real Gloo bridge, on device tensors.
Only NCCL talking to RCCL goes untested here; the mixed-vendor job does that.

Data are integer-valued floats in [-8, 8], so sums and averages over the
ranks are exact in fp16 and bf16 whatever the reduction order. Every policy
must therefore match the analytic result, and flat-gloo, bit for bit.

``test_gpu_test_logic_in_simulation`` (not marked ``gpu``) runs the same checks
on CPU in simulation mode, so a bug in these tests shows up in CI instead of
on a GPU node.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

import gpubridge
from gpubridge.policies import implements

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")

TIMEOUT = timedelta(seconds=180)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
REFERENCE = "flat-gloo"
BRIDGE_POLICIES = ("reduce-bridge-broadcast", "pipelined-reduce-bridge-broadcast",
                   "sharded-bridge", "auto-tuned")
#: Odd sizes: 1 element, a few, not a multiple of any chunk or shard, and ~1M.
ODD_NUMELS = (1, 7, 4099, 1_000_003)
#: Large tensors: enough that a copy racing the island reduce would see partial sums.
LARGE_NUMEL = int(os.environ.get("GPU_TEST_LARGE_NUMEL", 64 * 1024 * 1024))
#: A pipelined chunk that doesn't divide any test size, so the last chunk is partial.
ODD_CHUNK_BYTES = str(1024 * 1024 + 8)
#: GPUBRIDGE_* settings a developer's shell might carry; each job sets its own.
OWN_ENV = ("GPUBRIDGE_VENDOR", "GPUBRIDGE_CPU_ONLY", "GPUBRIDGE_SPLIT_TEST",
           "GPUBRIDGE_CHUNK_BYTES", "GPUBRIDGE_THRESHOLDS", "GPUBRIDGE_POLICY")


# ---- data and expected results ---------------------------------------------------

def rank_values(rank: int, numel: int) -> torch.Tensor:
    """Rank ``rank``'s data as int64, every value in [-8, 8]."""
    base = torch.arange(numel, dtype=torch.int64)
    return (base * 7 + rank * 3) % 17 - 8


def rank_data(rank: int, numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return rank_values(rank, numel).to(dtype).to(device)


def expected_reduce(op: str, world: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    stack = torch.stack([rank_values(r, numel) for r in range(world)])
    if op == "max":
        return stack.max(0).values.to(dtype)
    if op == "min":
        return stack.min(0).values.to(dtype)
    total = stack.sum(0).to(dtype)
    return total / world if op == "avg" else total


def digest(tensor: torch.Tensor) -> str:
    """The exact bytes of ``tensor``, for comparing policies bit for bit."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
                          .tobytes()).hexdigest()


class Mismatches:
    """Collects every failed case on a rank, so one run reports all of them."""

    def __init__(self, rank: int) -> None:
        self.rank, self.failed = rank, []

    def equal(self, case: str, got: torch.Tensor, want: torch.Tensor) -> None:
        if got.dtype != want.dtype or got.shape != want.shape:
            self.failed.append(f"{case}: got {got.dtype} {tuple(got.shape)}, "
                               f"want {want.dtype} {tuple(want.shape)}")
        elif not torch.equal(got.cpu(), want.cpu()):
            wrong = int((got.cpu() != want.cpu()).sum())
            self.failed.append(f"{case}: {wrong} of {got.numel()} elements wrong")

    def check(self, case: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.failed.append(f"{case}: {detail}")

    def raise_if_any(self) -> None:
        if self.failed:
            raise AssertionError(f"rank {self.rank}: {len(self.failed)} failed case(s):\n  "
                                 + "\n  ".join(self.failed[:50]))


# ---- running ranks ---------------------------------------------------------------

def gpu_ranks() -> int:
    """Ranks for a two-island job: 4 on separate GPUs if there are 4, else 2."""
    return 4 if torch.cuda.device_count() >= 4 else 2


def spawn(fn: Callable[[int, int, Path], None], world: int, tmp_path: Path,
          env: dict[str, str]) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mp.spawn(_rank_main, args=(world, fn, tmp_path, env), nprocs=world)


def _rank_main(rank: int, world: int, fn: Callable[[int, int, Path], None], tmp_path: Path,
               env: dict[str, str]) -> None:
    for name in OWN_ENV:
        os.environ.pop(name, None)
    os.environ.update(env)
    count = torch.cuda.device_count()
    os.environ["LOCAL_RANK"] = str(rank % count if count else 0)
    torch.set_num_threads(1)
    fn(rank, world, tmp_path)


def init(tmp_path: Path, tag: str, rank: int, world: int, policy: str) -> gpubridge.Topology:
    return gpubridge.init(init_method=f"file://{tmp_path}/rendezvous-{tag}", rank=rank,
                          world_size=world, timeout=TIMEOUT, policy=policy)


# ---- the checks ------------------------------------------------------------------

def run_ops(topology: gpubridge.Topology, numels: tuple[int, ...], bad: Mismatches,
            digests: dict[str, str]) -> None:
    """Every op on every dtype and size, sync and async; record a digest per result."""
    rank, world, device = topology.rank, topology.world_size, topology.device
    name = topology.policy.name
    for dtype in DTYPES:
        for numel in numels:
            ops = topology.policy.select_policy(numel * dtype.itemsize).supported_reduce_ops(
                topology)
            ops = sorted(ops | ({"avg"} if "sum" in ops else set()))
            for op in ops:
                want = expected_reduce(op, world, numel, dtype)
                for mode in ("sync", "async"):
                    case = f"{name} all_reduce {op} {dtype} n={numel} {mode}"
                    tensor = rank_data(rank, numel, dtype, device)
                    work = gpubridge.all_reduce(tensor, op=getattr(gpubridge.ReduceOp, op.upper()),
                                                async_op=mode == "async")
                    if work is not None:
                        work.wait()
                    bad.equal(case, tensor, want)
                    digests[case.split(" ", 1)[1]] = digest(tensor)
            for src in range(world):
                for mode in ("sync", "async"):
                    case = f"{name} broadcast src={src} {dtype} n={numel} {mode}"
                    tensor = rank_data(rank, numel, dtype, device)
                    work = gpubridge.broadcast(tensor, src, async_op=mode == "async")
                    if work is not None:
                        work.wait()
                    bad.equal(case, tensor, rank_values(src, numel).to(dtype))
                    digests[case.split(" ", 1)[1]] = digest(tensor)
            policy = topology.policy.select_policy(numel * world * dtype.itemsize)
            if implements(policy, "all_gather_into_tensor"):
                case = f"{name} all_gather_into_tensor {dtype} n={numel}"
                output = torch.full((world * numel,), 99, dtype=dtype, device=device)
                gpubridge.all_gather_into_tensor(output, rank_data(rank, numel, dtype, device))
                bad.equal(case, output, torch.cat([rank_values(r, numel)
                                                   for r in range(world)]).to(dtype))
                digests[case.split(" ", 1)[1]] = digest(output)
            if implements(policy, "reduce_scatter_tensor"):
                for op in ("sum", "avg"):
                    case = f"{name} reduce_scatter_tensor {op} {dtype} n={numel}"
                    data = rank_data(rank, world * numel, dtype, device)
                    before = data.clone()
                    output = torch.full((numel,), 99, dtype=dtype, device=device)
                    gpubridge.reduce_scatter_tensor(output, data,
                                                    op=getattr(gpubridge.ReduceOp, op.upper()))
                    want = expected_reduce(op, world, world * numel, dtype)
                    bad.equal(case, output, want[rank * numel:(rank + 1) * numel])
                    bad.equal(f"{case} input unchanged", data, before)
                    digests[case.split(" ", 1)[1]] = digest(output)
    work = gpubridge.barrier(async_op=True)
    work.wait()


def check_refuses_non_contiguous(topology: gpubridge.Topology, bad: Mismatches) -> None:
    """Non-contiguous tensors are refused with a clear error, never reduced wrongly."""
    device = topology.device
    strided = torch.zeros(8, 2, device=device).t()  # 2 x 8, column-major view
    for fn in (lambda: gpubridge.all_reduce(strided),
               lambda: gpubridge.broadcast(strided, 0),
               lambda: gpubridge.all_reduce(strided, async_op=True)):
        try:
            fn()
        except ValueError as error:
            bad.check("non-contiguous", "contiguous" in str(error), f"unclear error: {error}")
        else:
            bad.check("non-contiguous", False, "accepted a non-contiguous tensor")
    fixed = strided.contiguous() + topology.rank
    gpubridge.all_reduce(fixed)
    want = torch.full((2, 8), float(sum(range(topology.world_size))), device=device)
    bad.equal("non-contiguous after .contiguous()", fixed, want)


def check_large(topology: gpubridge.Topology, numel: int, bad: Mismatches) -> None:
    """A large all_reduce and broadcast, checked on the device."""
    rank, world, device = topology.rank, topology.world_size, topology.device
    name = topology.policy.name
    tensor = rank_data(rank, numel, torch.float32, device)
    gpubridge.all_reduce(tensor)
    want = expected_reduce("sum", world, numel, torch.float32).to(device)
    bad.equal(f"{name} large all_reduce n={numel}", tensor, want)
    del want
    tensor = rank_data(rank, numel, torch.float32, device)
    gpubridge.broadcast(tensor, world - 1)
    bad.equal(f"{name} large broadcast n={numel}", tensor,
              rank_data(world - 1, numel, torch.float32, device))


def compare_policies(rank: int, world: int, tmp_path: Path, *, numels: tuple[int, ...],
                     large: int) -> None:
    """Each bridge policy against flat-gloo, plus the analytic result for every case."""
    bad = Mismatches(rank)
    digests: dict[str, dict[str, str]] = {}
    runs = [(policy, policy, {}) for policy in (REFERENCE, *BRIDGE_POLICIES)]
    runs.append(("pipelined-odd-chunk", "pipelined-reduce-bridge-broadcast",
                 {"GPUBRIDGE_CHUNK_BYTES": ODD_CHUNK_BYTES}))
    for tag, policy, env in runs:
        os.environ.update(env)
        topology = init(tmp_path, tag, rank, world, policy)
        try:
            assert topology.split_test == "half" and len(topology.layout.islands) == 2
            digests[tag] = {}
            run_ops(topology, numels, bad, digests[tag])
            check_large(topology, large, bad)
            if tag == REFERENCE:
                check_refuses_non_contiguous(topology, bad)
        finally:
            gpubridge.destroy()
            for name in env:
                os.environ.pop(name)
    reference = digests.pop(REFERENCE)
    for tag, cases in digests.items():
        bad.check(f"{tag} ran every case", set(cases) == set(reference),
                  f"{len(cases)} cases, flat-gloo ran {len(reference)}")
        for case, value in cases.items():
            bad.check(f"{tag} vs {REFERENCE}: {case}", reference.get(case) == value,
                      "differs from flat-gloo" if case in reference else "flat-gloo didn't run it")
    bad.raise_if_any()


def native_only(rank: int, world: int, tmp_path: Path, *, numels: tuple[int, ...]) -> None:
    """One island (no split test): native-only against flat-gloo."""
    bad = Mismatches(rank)
    digests = {}
    for policy in (REFERENCE, "native-only"):
        topology = init(tmp_path, policy, rank, world, policy)
        try:
            digests[policy] = {}
            run_ops(topology, numels, bad, digests[policy])
        finally:
            gpubridge.destroy()
    bad.check("native-only ran every case", set(digests["native-only"]) == set(digests[REFERENCE]))
    for case, value in digests["native-only"].items():
        bad.check(f"native-only vs {REFERENCE}: {case}", digests[REFERENCE].get(case) == value,
                  "differs from flat-gloo")
    bad.raise_if_any()


def busy(device: torch.device, iters: int) -> torch.Tensor:
    """Queue ``iters`` matmuls on the current stream, so whatever comes next waits."""
    a = torch.randn(2048, 2048, device=device)
    for _ in range(iters):
        a = torch.tanh(a @ a)
    return a


def streams(rank: int, world: int, tmp_path: Path, *, numel: int, iters: int) -> None:
    """Bridge copies and the async worker, with the caller on non-default streams.

    The input is written at the end of a queue of matmuls on a side stream, and
    is large enough that the island reduce takes a while, so a device-to-host
    copy that didn't wait for the input or for the island reduce would stage
    stale or partial data and the sums would be wrong.
    """
    bad = Mismatches(rank)
    for policy in BRIDGE_POLICIES:
        topology = init(tmp_path, f"streams-{policy}", rank, world, policy)
        try:
            device = topology.device
            side, other = torch.cuda.Stream(device), torch.cuda.Stream(device)
            want = expected_reduce("sum", world, numel, torch.float32).to(device)

            # Sync, on a side stream: the bridge copy runs on the caller's stream.
            with torch.cuda.stream(side):
                tensor = torch.empty(numel, device=device)
                busy(device, iters)
                tensor.copy_(rank_data(rank, numel, torch.float32, device))
                gpubridge.all_reduce(tensor)
                got = tensor.clone()
            side.synchronize()
            bad.equal(f"{policy} sync on a side stream", got, want)

            # Async from a side stream, waited on another: the worker's stream
            # must wait for the side stream, and wait() must order the other one.
            with torch.cuda.stream(side):
                tensor = torch.empty(numel, device=device)
                busy(device, iters)
                tensor.copy_(rank_data(rank, numel, torch.float32, device))
                work = gpubridge.all_reduce(tensor, async_op=True)
            with torch.cuda.stream(other):
                work.wait()
                got = tensor.clone()
            other.synchronize()
            bad.equal(f"{policy} async, side stream to another", got, want)

            # Several async collectives in flight, waited in order on the default stream.
            tensors, works = [], []
            with torch.cuda.stream(side):
                busy(device, iters)
                for i in range(4):
                    tensor = rank_data(rank + i, 4099, torch.float16, device)
                    tensors.append(tensor)
                    works.append(gpubridge.all_reduce(tensor, async_op=True))
            for i, (tensor, work) in enumerate(zip(tensors, works, strict=True)):
                work.wait()
                want_i = torch.stack([rank_values(r + i, 4099) for r in range(world)]).sum(0)
                bad.equal(f"{policy} async #{i} in flight", tensor, want_i.to(torch.float16))

            # Async broadcast from the last rank, on a side stream.
            with torch.cuda.stream(side):
                tensor = rank_data(rank, numel, torch.bfloat16, device)
                gpubridge.broadcast(tensor, world - 1, async_op=True).wait()
                got = tensor.clone()
            side.synchronize()
            bad.equal(f"{policy} async broadcast on a side stream", got,
                      rank_data(world - 1, numel, torch.bfloat16, device))
        finally:
            gpubridge.destroy()
    bad.raise_if_any()


# ---- tests -----------------------------------------------------------------------

SPLIT = {"GPUBRIDGE_SPLIT_TEST": "half"}


@pytest.mark.gpu
@requires_gpu
def test_every_bridge_policy_matches_flat_gloo_on_gpus(tmp_path):
    spawn(partial(compare_policies, numels=ODD_NUMELS, large=LARGE_NUMEL), gpu_ranks(),
          tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
def test_native_only_matches_flat_gloo_on_one_island(tmp_path):
    spawn(partial(native_only, numels=ODD_NUMELS), min(4, torch.cuda.device_count()),
          tmp_path, {})


@pytest.mark.gpu
@requires_gpu
def test_bridge_copies_wait_for_the_callers_stream_and_the_island_reduce(tmp_path):
    spawn(partial(streams, numel=16 * 1024 * 1024, iters=40), gpu_ranks(), tmp_path, SPLIT)


def test_gpu_test_logic_in_simulation(tmp_path):
    """The checks above, on CPU in simulation mode (no streams), so CI runs them."""
    sim = {"GPUBRIDGE_VENDOR": "nvidia"}
    spawn(partial(compare_policies, numels=(1, 7, 4099), large=100_003), 4, tmp_path / "split",
          {**sim, **SPLIT})
    spawn(partial(native_only, numels=(1, 7)), 2, tmp_path / "one", sim)
