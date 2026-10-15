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
import sys
import time
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
from gpukit import p2p_checks, queue_gpu_delay  # noqa: E402

import gpubridge  # noqa: E402
from gpubridge import ucx
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
           "GPUBRIDGE_CHUNK_BYTES", "GPUBRIDGE_THRESHOLDS", "GPUBRIDGE_POLICY",
           "GPUBRIDGE_PINNED_STAGING")
#: The default bridge with opt-in pinned staging.
PINNED = {"GPUBRIDGE_PINNED_STAGING": "1"}
#: An ``env`` key that init() passes as the bridge transport instead of setting it.
BRIDGE = "bridge"


def _ucx_available() -> bool:
    try:
        ucx.build()
    except ucx.Unavailable:
        return False
    return True


UCX = _ucx_available()
#: Every bridge policy, reduce-bridge-broadcast with pinned staging, and, where
#: the UCX bridge builds, the default and pipelined policies over it: (tag, policy, env).
VARIANTS = (*((policy, policy, {}) for policy in BRIDGE_POLICIES),
            ("reduce-bridge-broadcast-pinned", "reduce-bridge-broadcast", PINNED),
            *((("ucx", "reduce-bridge-broadcast", {BRIDGE: "ucx"}),
               ("ucx-pipelined", "pipelined-reduce-bridge-broadcast", {BRIDGE: "ucx"}))
              if UCX else ()))


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
    apply_mutation(os.environ.get(MUTATION_ENV))
    fn(rank, world, tmp_path)


def init(tmp_path: Path, tag: str, rank: int, world: int, policy: str,
         env: dict[str, str] | None = None) -> gpubridge.Topology:
    """gpubridge.init, with ``env`` set while it reads its settings (and ``env[BRIDGE]``,
    if given, as the bridge transport)."""
    settings = {k: v for k, v in (env or {}).items() if k != BRIDGE}
    os.environ.update(settings)
    try:
        return gpubridge.init(init_method=f"file://{tmp_path}/rendezvous-{tag}", rank=rank,
                              world_size=world, timeout=TIMEOUT, policy=policy,
                              bridge=(env or {}).get(BRIDGE, "gloo"))
    finally:
        for name in settings:
            os.environ.pop(name)


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
    runs = [(REFERENCE, REFERENCE, {}), *VARIANTS]
    runs.append(("pipelined-odd-chunk", "pipelined-reduce-bridge-broadcast",
                 {"GPUBRIDGE_CHUNK_BYTES": ODD_CHUNK_BYTES}))
    for tag, policy, env in runs:
        topology = init(tmp_path, tag, rank, world, policy, env)
        try:
            assert topology.split_test == "half" and len(topology.layout.islands) == 2
            digests[tag] = {}
            run_ops(topology, numels, bad, digests[tag])
            check_large(topology, large, bad)
            if tag == REFERENCE:
                check_refuses_non_contiguous(topology, bad)
        finally:
            gpubridge.destroy()
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


#: Fills a tensor before its real data is copied in: no rank's data (values in [-8, 8])
#: or sum of them, and exact in every dtype tested.
POISON = 1000.0


def poisoned(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.full((numel,), POISON, dtype=dtype, device=device)


def busy(device: torch.device, seconds: float) -> None:
    """Queue about ``seconds`` of GPU work on the current stream, so whatever comes next
    waits. No matmuls (gpukit.queue_gpu_delay): some ROCm builds have no BLAS kernels
    for the GPU, and gpubridge needs none."""
    queue_gpu_delay(device, seconds)


# ---- mutations: each undoes one stream-ordering fix ----------------------------------
#
# GPUBRIDGE_TEST_MUTATION=<name> applies one in every rank, so that the GPU test that
# guards the fix can be shown to fail without it (scripts/gpu/stream_mutations.py runs
# each). Test code only: gpubridge has no such switch.

MUTATION_ENV = "GPUBRIDGE_TEST_MUTATION"


def _worker_ignores_caller() -> None:
    from gpubridge import work

    run_one = work._Worker._run_one

    def mutated(self, item, fn, ready, tensors):
        if self.stream is not None:  # an event that is already done: waits for nothing
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return run_one(self, item, fn, ready, tensors)

    work._Worker._run_one = mutated


def _staging_ignores_caller() -> None:
    from gpubridge import staging

    def begin(self, tensor):  # the copy streams no longer wait for the caller's
        tensor.record_stream(self.out_stream)
        tensor.record_stream(self.back_stream)

    staging.CudaStaging.begin = begin


def _wait_skips_stream() -> None:
    from gpubridge import work

    wait = work.Work.wait

    def mutated(self, timeout=None):  # blocks the host, but orders no stream
        event, self._event = self._event, None
        try:
            return wait(self, timeout)
        finally:
            self._event = event

    work.Work.wait = mutated


def _slots_forget_copies_back() -> None:
    from gpubridge import policies

    pipeline = policies.PipelinedReduceBridgeBroadcast._pipeline

    def mutated(self, *args, **kwargs):  # each collective forgets the last one's copies back
        self.copied_back = [None] * self.SLOTS
        return pipeline(self, *args, **kwargs)

    policies.PipelinedReduceBridgeBroadcast._pipeline = mutated


STREAMS_TEST = "test_bridge_copies_wait_for_the_callers_stream_and_the_island_reduce"
SLOTS_TEST = "test_pinned_slots_are_not_reused_before_their_copies_back"
#: Each mutation, and the GPU test that must fail with it.
MUTATIONS = {
    "worker-ignores-caller": (_worker_ignores_caller, STREAMS_TEST),
    "staging-ignores-caller": (_staging_ignores_caller, STREAMS_TEST),
    "wait-skips-stream": (_wait_skips_stream, STREAMS_TEST),
    "slots-forget-copies-back": (_slots_forget_copies_back, SLOTS_TEST),
}


def apply_mutation(name: str | None) -> None:
    if not name:
        return
    if name not in MUTATIONS:
        raise ValueError(f"{MUTATION_ENV}={name!r}: not one of {', '.join(MUTATIONS)}")
    MUTATIONS[name][0]()


# Streams on GPUs; on CPU (the simulation self-test) a stand-in that does nothing.

def new_stream(device: torch.device) -> torch.cuda.Stream | None:
    return torch.cuda.Stream(device) if device.type == "cuda" else None


def on(stream: torch.cuda.Stream | None) -> contextlib.AbstractContextManager:
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def streams(rank: int, world: int, tmp_path: Path, *, numel: int, seconds: float) -> None:
    """Bridge copies and the async worker, with the caller on non-default streams.

    The input is written at the end of queued GPU work on a side stream, and
    is large enough that the island reduce takes a while, so a device-to-host
    copy that didn't wait for the input or for the island reduce would stage
    stale or partial data and the sums would be wrong. The pipelined policy
    copies into pinned slots with ``non_blocking=True``, as does the default
    bridge with pinned staging, and those copies wait for the same queue, so a
    bridge step that read host memory before the host had waited for its
    copy's event would send stale bytes and fail here too.
    """
    bad = Mismatches(rank)
    for tag, policy, env in VARIANTS:
        topology = init(tmp_path, f"streams-{tag}", rank, world, policy, env)
        try:
            device = topology.device
            side, other = new_stream(device), new_stream(device)
            want = expected_reduce("sum", world, numel, torch.float32).to(device)

            # The input lands on the GPU before the delay and is copied into a
            # poisoned tensor behind it, device to device: so the host never waits
            # for the delay, and a copy that didn't wait for it reads the poison.
            # (Copying from host memory after the delay would make the host wait.)
            source = rank_data(rank, numel, torch.float32, device)
            sources = [rank_data(rank + i, 4099, torch.float16, device) for i in range(4)]
            sync(device)

            # Sync, on a side stream: the bridge copy runs on the caller's stream.
            with on(side):
                tensor = poisoned(numel, torch.float32, device)
                busy(device, seconds)
                tensor.copy_(source)
                gpubridge.all_reduce(tensor)
                got = tensor.clone()
            sync(device)
            bad.equal(f"{tag} sync on a side stream", got, want)

            # Async from a side stream, waited on another: the worker's stream
            # must wait for the side stream, and wait() must order the other one.
            with on(side):
                tensor = poisoned(numel, torch.float32, device)
                busy(device, seconds)
                tensor.copy_(source)
                work = gpubridge.all_reduce(tensor, async_op=True)
            with on(other):
                work.wait()
                got = tensor.clone()
            sync(device)
            bad.equal(f"{tag} async, side stream to another", got, want)

            # Several async collectives in flight, waited in order on the default stream.
            tensors, works = [], []
            with on(side):
                tensors = [poisoned(4099, torch.float16, device) for _ in range(4)]
                busy(device, seconds)
                for tensor, source_i in zip(tensors, sources, strict=True):
                    tensor.copy_(source_i)
                    works.append(gpubridge.all_reduce(tensor, async_op=True))
            for i, (tensor, work) in enumerate(zip(tensors, works, strict=True)):
                work.wait()
                want_i = torch.stack([rank_values(r + i, 4099) for r in range(world)]).sum(0)
                bad.equal(f"{tag} async #{i} in flight", tensor, want_i.to(torch.float16))

            # Async broadcast from the last rank, on a side stream.
            with on(side):
                tensor = rank_data(rank, numel, torch.bfloat16, device)
                gpubridge.broadcast(tensor, world - 1, async_op=True).wait()
                got = tensor.clone()
            sync(device)
            bad.equal(f"{tag} async broadcast on a side stream", got,
                      rank_data(world - 1, numel, torch.bfloat16, device))
        finally:
            gpubridge.destroy()
    bad.raise_if_any()


def slot_reuse(rank: int, world: int, tmp_path: Path, *, seconds: float) -> None:
    """The pipelined bridge's pinned slots, reused by the next collective.

    A pipelined collective returns before its copies back out of the pinned
    slots have run: only the caller's stream waits for them. Here the receiving
    island holds them back behind queued GPU work, and the next broadcast
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
            # Both tensors on the GPU first: copying from host memory behind the
            # delay would make the host wait it out, and nothing would be held back.
            first = rank_data(rank, numel, torch.float32, device)
            second = rank_data(rank + 1, numel, torch.float32, device)
            sync(device)
            with on(side):
                if receiving:
                    busy(device, seconds)  # the copies back wait for this queue
                gpubridge.broadcast(first, 0)
            with on(then):
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
    for tag, policy, env in VARIANTS:
        topology = init(tmp_path, f"completion-{tag}", rank, world, policy, env)
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
            bad.equal(f"{tag} read right after is_completed()", got, want)
            work.wait()
        finally:
            gpubridge.destroy()
    bad.raise_if_any()


def ucx_device_path(rank: int, world: int, tmp_path: Path, *, numel: int) -> None:
    """bridge="ucx" says which path a GPU tensor took, and the result is right either way.

    ``direct`` where this rank's UCX supports the GPU's memory type (CUDA or
    ROCm), ``staged`` through pinned host memory where it doesn't. On CPU (the
    simulation self-test) tensors are on the host.
    """
    bad = Mismatches(rank)
    topology = init(tmp_path, "ucx-path", rank, world, "reduce-bridge-broadcast",
                    {BRIDGE: "ucx"})
    try:
        device = topology.device
        tensor = rank_data(rank, numel, torch.float32, device)
        gpubridge.all_reduce(tensor)
        sync(device)
        bad.equal("ucx all_reduce", tensor,
                  expected_reduce("sum", world, numel, torch.float32).to(device))
        bridge = topology.bridge
        if bridge is not None:
            kind = "rocm" if torch.version.hip else "cuda"
            want = ("host" if device.type == "cpu"
                    else "direct" if kind in bridge.memory_types else "staged")
            bad.check("ucx path", bridge.last_path == want,
                      f"{bridge.last_path}, expected {want} (UCX memory types "
                      f"{sorted(bridge.memory_types)})")
    finally:
        gpubridge.destroy()
    bad.raise_if_any()


def ucx_fresh_data(rank: int, world: int, tmp_path: Path, *, numel: int, iters: int) -> None:
    """What UCX writes into GPU memory is what the next kernel reads.

    On the direct path the NIC writes received data into GPU memory, and the
    CPU can see a receive complete before kernels can see that data; the
    transport flushes after every receive (docs/RDMA_TRANSPORT.md, section 4).
    Each iteration sends new values, so a reduction or copy that read a
    buffer's earlier contents would give an earlier iteration's numbers.
    Kernels check every result right after its collective returns, with no
    host synchronization in between; the counts are read back at the end. The
    NIC writes GPU memory only when UCX uses a network transport for it: across
    nodes, or on one node with ``GPUBRIDGE_TEST_UCX_TLS`` naming one (e.g.
    ``rc,cuda_copy``). Otherwise UCX copies with CUDA or ROCm IPC, and this
    checks the ordering only.
    """
    bad = Mismatches(rank)
    topology = init(tmp_path, "ucx-fresh", rank, world, "reduce-bridge-broadcast",
                    {BRIDGE: "ucx"})
    try:
        device = topology.device
        total = expected_reduce("sum", world, numel, torch.float32).to(device)
        root = rank_values(0, numel).float().to(device)
        wrong = torch.zeros(2, iters, dtype=torch.int64, device=device)
        for it in range(iters):  # every value stays below 2**24: exact in float32
            tensor = rank_data(rank, numel, torch.float32, device) + it
            gpubridge.all_reduce(tensor)
            wrong[0, it] = (tensor != total + world * it).sum()
            tensor = rank_data(rank, numel, torch.float32, device) + it
            gpubridge.broadcast(tensor, 0)
            wrong[1, it] = (tensor != root + it).sum()
        sync(device)
        for row, op in enumerate(("all_reduce", "broadcast")):
            stale = [(it, n) for it, n in enumerate(wrong[row].tolist()) if n]
            bad.check(f"ucx {op}, read right after it returned", not stale,
                      f"(iteration, wrong elements): {stale[:10]}")
        bridge = topology.bridge
        if bridge is not None:
            print(f"rank {rank}: UCX path {bridge.last_path}, flush {bridge.flush_method}",
                  flush=True)
            # Direct: how it flushes. Host: nothing to flush. Staged: why, if UCX had
            # the memory type but no flush was possible.
            if bridge.last_path in ("direct", "host"):
                bad.check("ucx flush recorded",
                          (bridge.flush_method is None) == (bridge.last_path == "host"),
                          f"path {bridge.last_path}, flush {bridge.flush_method}")
    finally:
        gpubridge.destroy()
    bad.raise_if_any()


def pinned_reuse(rank: int, world: int, tmp_path: Path, *, numel: int) -> None:
    """The default bridge's pinned buffer (``GPUBRIDGE_PINNED_STAGING``), reused at once.

    A collective returns once its copy back out of the pinned buffer is queued,
    not done. The next broadcast starts right away, on the same stream or
    another, at the same size (the buffer is reused: its copy out must wait for
    that copy back on the device) or twice the size (the buffer is replaced:
    the host must wait for the copy back first). The tensors are large, so the
    copy back takes long enough for a copy out on another stream, on the other
    copy engine, to overlap it if nothing ordered them.
    """
    bad = Mismatches(rank)
    topology = init(tmp_path, "pinned-reuse", rank, world, "reduce-bridge-broadcast", PINNED)
    try:
        device = topology.device
        side, other = new_stream(device), new_stream(device)
        for case, then, scale in (("the same stream", side, 1), ("another stream", other, 1),
                                  ("another stream, twice the size", other, 2)):
            with on(side):
                first = rank_data(rank, numel, torch.float32, device)
                gpubridge.broadcast(first, 0)
            with on(then):
                second = rank_data(rank + 1, scale * numel, torch.float32, device)
                gpubridge.broadcast(second, 0)
            sync(device)
            bad.equal(f"broadcast, then another on {case}: first", first,
                      rank_values(0, numel).float())
            bad.equal(f"broadcast, then another on {case}: second", second,
                      rank_values(1, scale * numel).float())
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
    spawn(partial(streams, numel=16 * 1024 * 1024, seconds=0.1), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
def test_pinned_slots_are_not_reused_before_their_copies_back(tmp_path):
    spawn(partial(slot_reuse, seconds=0.2), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not UCX, reason="the UCX bridge can't be built here")
def test_the_ucx_bridge_takes_gpu_tensors_directly_or_says_it_staged_them(tmp_path):
    spawn(partial(ucx_device_path, numel=LARGE_NUMEL // 4), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not UCX, reason="the UCX bridge can't be built here")
def test_data_ucx_writes_into_gpu_memory_is_what_the_next_kernel_reads(tmp_path):
    tls = os.environ.get("GPUBRIDGE_TEST_UCX_TLS")
    spawn(partial(ucx_fresh_data, numel=LARGE_NUMEL // 4, iters=30), gpu_ranks(), tmp_path,
          {**SPLIT, **({"UCX_TLS": tls} if tls else {})})


@pytest.mark.gpu
@requires_gpu
def test_the_pinned_bridge_buffer_is_not_reused_before_its_copy_back(tmp_path):
    spawn(partial(pinned_reuse, numel=LARGE_NUMEL // 4), gpu_ranks(), tmp_path, SPLIT)


@pytest.mark.gpu
@requires_gpu
def test_is_completed_means_the_result_is_ready(tmp_path):
    spawn(partial(completion, numel=LARGE_NUMEL), gpu_ranks(), tmp_path, SPLIT)


def point_to_point(rank: int, world: int, tmp_path: Path) -> None:
    """Send and receive through ``backend="gpubridge"`` across the two islands (and
    within them with 4 ranks): six dtypes and odd sizes bit for bit, the send's input
    written on a busy producer stream and dropped before the send completes, the
    received tensor read on another stream right after wait(), and tags matched out of
    order (gpukit.p2p_checks, which check.py's ``p2p`` stage runs too)."""
    import torch.distributed as dist
    dist.init_process_group("gpubridge", init_method=f"file://{tmp_path}/rendezvous-p2p",
                            rank=rank, world_size=world, timeout=TIMEOUT)
    try:
        dist.barrier()
        topology = gpubridge.get_topology()
        seconds = 0.02 if topology.device.type == "cuda" else 0.0
        cases = p2p_checks(topology, busy_seconds=seconds)
        assert cases and all(case["ok"] for case in cases)
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@requires_gpu
def test_point_to_point_across_islands_keeps_stream_order_and_storage(tmp_path):
    spawn(point_to_point, gpu_ranks(), tmp_path, SPLIT)


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

    each(partial(streams, numel=10_007, seconds=0), 4, "streams")
    each(partial(slot_reuse, seconds=0), 4, "slots")
    each(partial(completion, numel=100_003), 2, "completion")
    each(partial(pinned_reuse, numel=10_007), 4, "pinned")
    each(point_to_point, 2, "p2p")
    each(point_to_point, 4, "p2p-4")
    if UCX:
        each(partial(ucx_device_path, numel=10_007), 4, "ucx-path")
        each(partial(ucx_fresh_data, numel=10_007, iters=3), 4, "ucx-fresh")
