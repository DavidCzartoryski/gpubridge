"""Correctness on real GPUs: every policy against flat-gloo, every op, fp32/fp16/bf16.

Marked ``gpu`` and skipped without a GPU. On a GPU node, run
``pytest -m gpu tests/test_gpu.py`` (the Explorer jobs and the AMD script do).
No launcher is needed: each test spawns its ranks. With 4+ GPUs, 4 ranks on
separate GPUs; with 2 or 3, 2 ranks on separate GPUs; with 1, 2 ranks sharing
it, which only split-test mode allows (each is its own island).

The two-island cases use split-test mode (``GPUBRIDGE_SPLIT_TEST=half``): two
real NCCL (or RCCL) islands joined by the real Gloo bridge, on device tensors.
Only NCCL talking to RCCL goes untested here; the mixed-vendor job does that.

Bit for bit, although NCCL, RCCL and Gloo add in different orders: data are
integer-valued floats in [-8, 8], so with W ranks every partial sum, in any
order or tree, is an integer of magnitude at most 8W. The tests use W = 1, 2
or 4, and every integer up to 256 in magnitude is exact in bf16 (fp16 and fp32
hold more), so every addition is exact. AVG is a SUM followed by one division
by W in ``collectives.py`` (policies, NCCL and RCCL never see AVG), and W is a
power of two, so the quotient is exact too; so is NCCL's own AVG (scale by 1/W,
then sum), should a policy ever use it. MAX and MIN are exact anyway. So every
policy must match the analytic result, and flat-gloo, bit for bit, and any
difference is a bug, not rounding. ``check_world`` refuses world sizes for
which that argument fails, and ``test_gpu_test_data_is_exact_in_any_order``
checks its premises on CPU.

``test_gpu_test_logic_in_simulation`` (not marked ``gpu``) runs the same checks
on CPU in simulation mode, so a bug in these tests shows up in CI instead of
on a GPU node.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
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
#: Pipelined chunk for the slot-reuse test: its tensors are two chunks, fewer than the slots.
SLOT_CHUNK_BYTES = 1024 * 1024
#: Largest integer magnitude below which every integer is exact in bf16 (8 significant bits).
BF16_EXACT = 256
#: GPUBRIDGE_* settings a developer's shell might carry; each job sets its own.
OWN_ENV = ("GPUBRIDGE_VENDOR", "GPUBRIDGE_CPU_ONLY", "GPUBRIDGE_SPLIT_TEST",
           "GPUBRIDGE_CHUNK_BYTES", "GPUBRIDGE_THRESHOLDS", "GPUBRIDGE_POLICY")


# ---- data and expected results ---------------------------------------------------

def rank_values(rank: int, numel: int) -> torch.Tensor:
    """Rank ``rank``'s data as int64, every value in [-8, 8]."""
    base = torch.arange(numel, dtype=torch.int64)
    return (base * 7 + rank * 3) % 17 - 8


def check_world(world: int) -> None:
    """Refuse a world size for which the bit-for-bit argument (module docstring) fails."""
    if world < 1 or world & (world - 1) or 8 * world > BF16_EXACT:
        raise ValueError(f"{world} ranks: the bit-for-bit checks need a power of two "
                         f"with 8 * ranks <= {BF16_EXACT}")


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


def island_ranks() -> int:
    """Ranks for a one-island job, one per GPU: 4, 2 or 1 (a power of two, see check_world)."""
    count = torch.cuda.device_count()
    return 4 if count >= 4 else 2 if count >= 2 else 1


def spawn(fn: Callable[[int, int, Path], None], world: int, tmp_path: Path,
          env: dict[str, str]) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mp.spawn(_rank_main, args=(world, fn, tmp_path, env), nprocs=world)


def _rank_main(rank: int, world: int, fn: Callable[[int, int, Path], None], tmp_path: Path,
               env: dict[str, str]) -> None:
    check_world(world)
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


# Streams on GPUs; on CPU (the simulation self-test) a stand-in that does nothing.

def new_stream(device: torch.device) -> torch.cuda.Stream | None:
    return torch.cuda.Stream(device) if device.type == "cuda" else None


def on(stream: torch.cuda.Stream | None) -> contextlib.AbstractContextManager:
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def streams(rank: int, world: int, tmp_path: Path, *, numel: int, iters: int) -> None:
    """Bridge copies and the async worker, with the caller on non-default streams.

    The input is written at the end of a queue of matmuls on a side stream, and
    is large enough that the island reduce takes a while, so a device-to-host
    copy that didn't wait for the input or for the island reduce would stage
    stale or partial data and the sums would be wrong. The pipelined policy
    copies into pinned slots with ``non_blocking=True`` and those copies wait
    for the same queue, so a bridge step that read a slot before the host had
    waited for its copy's event would send stale bytes and fail here too.
    """
    bad = Mismatches(rank)
    for policy in BRIDGE_POLICIES:
        topology = init(tmp_path, f"streams-{policy}", rank, world, policy)
        try:
            device = topology.device
            side, other = new_stream(device), new_stream(device)
            want = expected_reduce("sum", world, numel, torch.float32).to(device)

            # Sync, on a side stream: the bridge copy runs on the caller's stream.
            with on(side):
                tensor = torch.empty(numel, device=device)
                busy(device, iters)
                tensor.copy_(rank_data(rank, numel, torch.float32, device))
                gpubridge.all_reduce(tensor)
                got = tensor.clone()
            sync(device)
            bad.equal(f"{policy} sync on a side stream", got, want)

            # Async from a side stream, waited on another: the worker's stream
            # must wait for the side stream, and wait() must order the other one.
            with on(side):
                tensor = torch.empty(numel, device=device)
                busy(device, iters)
                tensor.copy_(rank_data(rank, numel, torch.float32, device))
                work = gpubridge.all_reduce(tensor, async_op=True)
            with on(other):
                work.wait()
                got = tensor.clone()
            sync(device)
            bad.equal(f"{policy} async, side stream to another", got, want)

            # Several async collectives in flight, waited in order on the default stream.
            tensors, works = [], []
            with on(side):
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
            with on(side):
                tensor = rank_data(rank, numel, torch.bfloat16, device)
                gpubridge.broadcast(tensor, world - 1, async_op=True).wait()
                got = tensor.clone()
            sync(device)
            bad.equal(f"{policy} async broadcast on a side stream", got,
                      rank_data(world - 1, numel, torch.bfloat16, device))
        finally:
            gpubridge.destroy()
    bad.raise_if_any()


def slot_reuse(rank: int, world: int, tmp_path: Path, *, iters: int) -> None:
    """The pipelined bridge's pinned slots, reused by the next collective.

    A pipelined collective returns before its copies back out of the pinned
    slots have run: only the caller's stream waits for them. Here the receiving
    island holds them back behind a queue of matmuls, and the next broadcast
    starts at once, on the same stream or another one. If its bridge step
    wrote into a slot before that slot's last copy back had run, the first
    result would hold the second one's data.
    """
    bad = Mismatches(rank)
    os.environ["GPUBRIDGE_CHUNK_BYTES"] = str(SLOT_CHUNK_BYTES)
    topology = init(tmp_path, "slot-reuse", rank, world, "pipelined-reduce-bridge-broadcast")
    try:
        device = topology.device
        receiving = topology.island != topology.layout.island_of(0)
        numel = 2 * SLOT_CHUNK_BYTES // 4  # two float32 chunks: no reuse within one call
        side, other = new_stream(device), new_stream(device)
        for case, then in (("the same stream", side), ("another stream", other)):
            with on(side):
                if receiving:
                    busy(device, iters)  # the copies back wait for this queue
                first = rank_data(rank, numel, torch.float32, device)
                gpubridge.broadcast(first, 0)
            with on(then):
                second = rank_data(rank + 1, numel, torch.float32, device)
                gpubridge.broadcast(second, 0)
            sync(device)
            bad.equal(f"broadcast, then another on {case}: first", first,
                      rank_values(0, numel).float())
            bad.equal(f"broadcast, then another on {case}: second", second,
                      rank_values(1, numel).float())
    finally:
        gpubridge.destroy()
        os.environ.pop("GPUBRIDGE_CHUNK_BYTES")
    bad.raise_if_any()


def completion(rank: int, world: int, tmp_path: Path, *, numel: int) -> None:
    """``Work.is_completed()`` turns True only once the GPU work is done, as with NCCL.

    The worker thread finishes as soon as it has queued the island broadcast,
    which on a large tensor still runs for a while. A result read on the
    current stream right after ``is_completed()`` turns True, without
    ``wait()``, must be complete.
    """
    bad = Mismatches(rank)
    for policy in BRIDGE_POLICIES:
        topology = init(tmp_path, f"completion-{policy}", rank, world, policy)
        try:
            device = topology.device
            want = expected_reduce("sum", world, numel, torch.float32).to(device)
            tensor = rank_data(rank, numel, torch.float32, device)
            sync(device)
            work = gpubridge.all_reduce(tensor, async_op=True)
            while not work.is_completed():
                time.sleep(0.0002)
            got = tensor.clone()  # no wait(): is_completed() said the result is ready
            sync(device)
            bad.equal(f"{policy} read right after is_completed()", got, want)
            work.wait()
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
    spawn(partial(native_only, numels=ODD_NUMELS), island_ranks(), tmp_path, {})


@pytest.mark.gpu
@requires_gpu
def test_bridge_copies_wait_for_the_callers_stream_and_the_island_reduce(tmp_path):
    spawn(partial(streams, numel=16 * 1024 * 1024, iters=40), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
def test_pinned_slots_are_not_reused_before_their_copies_back(tmp_path):
    spawn(partial(slot_reuse, iters=80), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
def test_is_completed_means_the_result_is_ready(tmp_path):
    spawn(partial(completion, numel=LARGE_NUMEL), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_gpu_test_data_is_exact_in_any_order(dtype):
    """The premises of the bit-for-bit checks, on CPU, for every world size they allow."""
    values = torch.stack([rank_values(rank, 4099) for rank in range(8)])
    assert values.min() >= -8 and values.max() <= 8
    for world in (1, 2, 4):
        check_world(world)
        # Every partial sum of up to ``world`` values in [-8, 8], and its average.
        sums = torch.arange(-8 * world, 8 * world + 1)
        assert torch.equal(sums.to(dtype).to(torch.int64), sums)
        averages = sums.double() / world
        assert torch.equal(averages.to(dtype).double(), averages)
        # So adding in any order gives the same bits: forwards, backwards, as a tree.
        stack = values[:world].to(dtype)
        exact = values[:world].sum(0)
        forwards, backwards = stack[0].clone(), stack[-1].clone()
        for rank in range(1, world):
            forwards += stack[rank]
            backwards += stack[world - 1 - rank]
        tree = stack
        while len(tree) > 1:
            tree = tree[0::2] + tree[1::2]
        for got in (forwards, backwards, tree[0]):
            assert torch.equal(got.to(torch.int64), exact)
    for world in (3, 6, 64):
        with pytest.raises(ValueError, match="power of two"):
            check_world(world)


def test_gpu_test_logic_in_simulation(tmp_path):
    """The checks above, on CPU in simulation mode (no streams), so CI runs them."""
    sim = {"GPUBRIDGE_VENDOR": "nvidia"}
    split = {**sim, **SPLIT}
    spawn(partial(compare_policies, numels=(1, 7, 4099), large=100_003), 4, tmp_path / "split",
          split)
    spawn(partial(native_only, numels=(1, 7)), 2, tmp_path / "one", sim)

    def each(fn: Callable[[int, int, Path], None], world: int, name: str) -> None:
        spawn(fn, world, tmp_path / name, split)

    each(partial(streams, numel=10_007, iters=0), 4, "streams")
    each(partial(slot_reuse, iters=0), 4, "slots")
    each(partial(completion, numel=100_003), 2, "completion")
