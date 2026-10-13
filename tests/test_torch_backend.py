"""gpubridge as a torch.distributed backend: init_process_group(backend="gpubridge").

Whole jobs in simulation mode: DDP and FSDP2 trained through the backend
match the same training through Gloo bit for bit (integer-valued data, so
every sum is exact in any order); every torch.distributed collective the
backend carries, coalesced and bucketed paths included; and every unsupported
call refused on every rank. docs/TORCH_BACKEND.md has the design.
"""

from __future__ import annotations

import gc
import importlib.metadata
import os
import subprocess
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from _harness import rank_tensor

import gpubridge
from gpubridge import torch_backend
from gpubridge.config import BRIDGE_ENV, POLICY_ENV, VENDOR_ENV

TWO_TWO = ["nvidia", "nvidia", "amd", "amd"]
TIMEOUT = timedelta(seconds=60)
REPO = Path(__file__).resolve().parents[1]


def run_backend(vendors, fn, tmp_path: Path, *, backend: str = "gpubridge",
                env: dict[int, dict[str, str]] | None = None) -> None:
    """Run ``fn(rank, world)`` on one process per vendor, after
    ``init_process_group(backend)``; destroy the group after."""
    mp.spawn(_main, args=(tuple(vendors), fn, str(tmp_path / "rendezvous"), backend, env or {}),
             nprocs=len(vendors))


def _main(rank, vendors, fn, init_file, backend, env):
    os.environ[VENDOR_ENV] = vendors[rank]
    os.environ.update(env.get(rank, env.get(-1, {})))
    torch.set_num_threads(1)
    dist.init_process_group(backend=backend, init_method=f"file://{init_file}", rank=rank,
                            world_size=len(vendors), timeout=TIMEOUT)
    try:
        fn(rank, len(vendors))
    finally:
        # FSDP2 modules keep process groups alive in reference cycles. Destroyed later,
        # at interpreter exit, a Gloo group can abort the process ("terminate called
        # without an active exception"), with either backend. Collect them first.
        gc.collect()
        dist.destroy_process_group()
        gc.collect()
    assert not gpubridge.is_initialized()  # shutdown() released it


# ---- training -----------------------------------------------------------------------------

#: The 4 ranks' gradients are averaged, so a learning rate of 4 makes each update
#: their sum: parameters stay whole numbers (about 1e5 after 3 steps), and float64
#: adds them exactly in any order. Different summation orders (Gloo's, gpubridge's
#: islands and bridge) must then give the same bits.
LR = 4.0
STEPS = 3


def _integer_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randint(-3, 4, p.shape).double())
    return model


def _batch(rank: int, step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 * step + rank)
    return torch.randint(-2, 3, (4, 8), generator=g).double()


def _ddp(rank, world, out, bucket_mb, static):
    model = _integer_model()
    ddp = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_mb, static_graph=static)
    opt = torch.optim.SGD(ddp.parameters(), lr=LR)
    for step in range(STEPS):
        ddp(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    if rank == 0:
        torch.save([p.detach().clone() for p in model.parameters()], out)


@pytest.mark.parametrize(("bucket_mb", "static"), [(0.0001, False), (25, False), (0.0001, True)],
                         ids=["many-buckets", "one-bucket", "static-graph"])
def test_ddp_trains_bit_for_bit_like_gloo(bucket_mb, static, tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_ddp, out=str(out), bucket_mb=bucket_mb, static=static),
                    tmp_path / backend, backend=backend)
        results[backend] = torch.load(out)
    assert all(torch.equal(a, b) for a, b in zip(results["gloo"], results["gpubridge"],
                                                strict=True))
    # And the model really moved: the comparison isn't between two untouched models.
    assert not all(torch.equal(a, b) for a, b in zip(results["gloo"], _integer_model().parameters(),
                                                    strict=True))


def _fully_shard():
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:  # older releases
        from torch.distributed._composable.fsdp import fully_shard
    return fully_shard


def _fsdp2(rank, world, out):
    from torch.distributed.device_mesh import init_device_mesh

    fully_shard = _fully_shard()
    mesh = init_device_mesh("cpu", (world,))
    model = _integer_model()
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        model(_batch(rank, step)).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().clone() for p in model.parameters()]
    if rank == 0:
        torch.save(full, out)


def _fsdp2_float32(rank, world, out):
    """float32 gradients take FSDP2's AVG reduce-scatter (float64 takes SUM)."""
    from torch.distributed.device_mesh import init_device_mesh

    fully_shard = _fully_shard()
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    fully_shard(model[0], mesh=mesh)
    fully_shard(model[2], mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(STEPS):
        model(_batch(rank, step).float()).sum().backward()
        opt.step()
        opt.zero_grad()
    full = [p.full_tensor().detach().clone() for p in model.parameters()]
    if rank == 0:
        torch.save(full, out)


def test_fsdp2_in_float32_takes_the_avg_path_and_matches_gloo_closely(tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_fsdp2_float32, out=str(out)), tmp_path / backend,
                    backend=backend)
        results[backend] = torch.load(out)
    # Not bit for bit: float32 sums in a different order round differently.
    for a, b in zip(results["gloo"], results["gpubridge"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_fsdp2_trains_bit_for_bit_like_gloo_and_a_single_process(tmp_path):
    results = {}
    for backend in ("gloo", "gpubridge"):
        out = tmp_path / f"{backend}.pt"
        (tmp_path / backend).mkdir()
        run_backend(TWO_TWO, partial(_fsdp2, out=str(out)), tmp_path / backend, backend=backend)
        results[backend] = torch.load(out)
    assert all(torch.equal(a, b) for a, b in zip(results["gloo"], results["gpubridge"],
                                                strict=True))
    # One process on the whole batch: the mean gradient over the 4 ranks' batches.
    model = _integer_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(STEPS):
        loss = sum(model(_batch(r, step)).sum() for r in range(4)) / 4
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert all(torch.equal(a, b.detach()) for a, b in zip(results["gpubridge"],
                                                          model.parameters(), strict=True))


# ---- every collective the backend carries -----------------------------------------------------

def _collectives(rank, world):
    expected = sum(rank_tensor((5, 7), torch.float32, r) for r in range(world))
    for op, want in [(dist.ReduceOp.SUM, expected), (dist.ReduceOp.AVG, expected / world),
                     (dist.ReduceOp.MAX, rank_tensor((5, 7), torch.float32, world - 1)),
                     (dist.ReduceOp.MIN, rank_tensor((5, 7), torch.float32, 0))]:
        t = rank_tensor((5, 7), torch.float32, rank)
        dist.all_reduce(t, op=op)
        assert torch.equal(t, want), op
    t = rank_tensor((5, 7), torch.int64, rank)
    work = dist.all_reduce(t, async_op=True)
    work.wait()
    assert torch.equal(t, expected.to(torch.int64))
    t = rank_tensor((5, 7), torch.float32, rank)
    future = dist.all_reduce(t, async_op=True).get_future()
    assert torch.equal(future.wait()[0], expected)
    for src in range(world):
        t = rank_tensor((9,), torch.float32, rank)
        dist.broadcast(t, src=src)
        assert torch.equal(t, rank_tensor((9,), torch.float32, src))
    out = torch.empty(world * 6)
    dist.all_gather_into_tensor(out, rank_tensor((6,), torch.float32, rank))
    assert torch.equal(out, torch.cat([rank_tensor((6,), torch.float32, r) for r in range(world)]))
    outs = [torch.empty(3, 2) for _ in range(world)]
    dist.all_gather(outs, rank_tensor((3, 2), torch.float32, rank))
    assert all(torch.equal(o, rank_tensor((3, 2), torch.float32, r)) for r, o in enumerate(outs))
    rows = rank_tensor((world, 6), torch.float32, rank)
    got = torch.empty(6)
    dist.reduce_scatter_tensor(got, rows)
    assert torch.equal(got, sum(rank_tensor((world, 6), torch.float32, r)
                                for r in range(world))[rank])
    got = torch.empty(6)
    dist.reduce_scatter(got, list(rank_tensor((world, 6), torch.float32, rank).unbind(0)))
    assert torch.equal(got, sum(rank_tensor((world, 6), torch.float32, r)
                                for r in range(world))[rank])
    dist.barrier()
    # Coalesced: the deprecated all_reduce_coalesced, and the coalescing manager.
    ts = [rank_tensor((4,), torch.float32, rank), rank_tensor((3,), torch.float32, rank)]
    dist.all_reduce_coalesced(ts)
    assert torch.equal(ts[1], sum(rank_tensor((3,), torch.float32, r) for r in range(world)))
    from torch.distributed.distributed_c10d import _coalescing_manager

    for device in (None, torch.device("cpu")):
        ts = [rank_tensor((4,), torch.float32, rank), rank_tensor((5,), torch.float32, rank)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            for t in ts:
                dist.all_reduce(t)
        cm.wait()
        assert torch.equal(ts[0], sum(rank_tensor((4,), torch.float32, r) for r in range(world)))
        assert torch.equal(ts[1], sum(rank_tensor((5,), torch.float32, r) for r in range(world)))
        outs = [torch.empty(world * 2), torch.empty(world * 3)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            dist.all_gather_into_tensor(outs[0], rank_tensor((2,), torch.float32, rank))
            dist.all_gather_into_tensor(outs[1], rank_tensor((3,), torch.float32, rank))
        cm.wait()
        assert torch.equal(outs[1], torch.cat([rank_tensor((3,), torch.float32, r)
                                               for r in range(world)]))
        got = [torch.empty(2), torch.empty(3)]
        with _coalescing_manager(device=device, async_ops=True) as cm:
            dist.reduce_scatter_tensor(got[0], rank_tensor((world * 2,), torch.float32, rank))
            dist.reduce_scatter_tensor(got[1], rank_tensor((world * 3,), torch.float32, rank))
        cm.wait()
        whole = sum(rank_tensor((world * 3,), torch.float32, r) for r in range(world))
        assert torch.equal(got[1], whole.view(world, 3)[rank])
    # gpubridge's own API works on the same topology, and its observers see the calls.
    seen = []

    class Seen(gpubridge.CollectiveObserver):
        def on_collective(self, record):
            seen.append((record.op, record.policy))

    gpubridge.add_observer(Seen())
    t = rank_tensor((3,), torch.float32, rank)
    gpubridge.all_reduce(t)
    dist.all_reduce(rank_tensor((3,), torch.float32, rank))
    assert seen == [("all_reduce", "reduce-bridge-broadcast")] * 2, seen
    assert gpubridge.get_topology().layout.needs_bridge


@pytest.mark.parametrize("vendors", [TWO_TWO, ["amd", "nvidia", "nvidia"]], ids=["2+2", "1+2"])
def test_every_collective_the_backend_carries(vendors, tmp_path):
    run_backend(vendors, _collectives, tmp_path)


def _policy_and_bridge(rank, world, policy):
    dist.barrier()
    assert gpubridge.get_topology().policy_name == policy
    t = rank_tensor((50,), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, sum(rank_tensor((50,), torch.float32, r) for r in range(world)))


@pytest.mark.parametrize("policy", ["pipelined-reduce-bridge-broadcast", "sharded-bridge",
                                    "flat-gloo"])
def test_the_policy_comes_from_gpubridge_policy(policy, tmp_path):
    run_backend(TWO_TWO, partial(_policy_and_bridge, policy=policy), tmp_path,
                env={-1: {POLICY_ENV: policy, "GPUBRIDGE_CHUNK_BYTES": "64"}})


# ---- refused, on every rank ---------------------------------------------------------------

def _refused(rank, world):
    def expect(fn, pattern):
        with pytest.raises((NotImplementedError, ValueError, RuntimeError), match=pattern):
            fn()

    dist.barrier()  # set up first, so the refusals below are the backend's
    expect(lambda: dist.all_reduce(torch.ones(3), op=dist.ReduceOp.PRODUCT),
           "gpubridge backend, allreduce: .*supports ReduceOp.SUM, AVG, MAX and MIN")
    expect(lambda: dist.all_to_all_single(torch.empty(world), torch.ones(world)),
           "the gpubridge torch.distributed backend doesn't support all_to_all")
    expect(lambda: dist.gather(torch.ones(2), [torch.empty(2)] * world if rank == 0 else None),
           "doesn't support gather")
    expect(lambda: dist.scatter(torch.empty(2), [torch.ones(2)] * world if rank == 0 else None),
           "doesn't support scatter")
    expect(lambda: dist.reduce(torch.ones(2), dst=0), "doesn't support reduce")
    peer = rank ^ 1
    expect(lambda: dist.send(torch.ones(2), dst=peer) if rank % 2 == 0
           else dist.recv(torch.empty(2), src=peer), "doesn't support (send|recv)")
    expect(lambda: dist.all_gather([torch.empty(2) for _ in range(world)], torch.ones(3)),
           "allgather with tensors of different sizes")
    expect(lambda: dist.reduce_scatter_tensor(torch.empty(2), torch.ones(5)),
           "gpubridge backend, reduce_scatter_tensor: .*times as many elements")
    # A subgroup can't be gpubridge's: refused where it is created.
    expect(lambda: dist.new_group([0, 1], use_local_synchronization=True) if rank < 2
           else (_ for _ in ()).throw(RuntimeError("runs on the world group only")),
           "runs on the world group only")
    # The world again is fine: it shares gpubridge's topology.
    again = dist.new_group(list(range(world)))
    t = rank_tensor((4,), torch.float32, rank)
    dist.all_reduce(t, group=again)
    assert torch.equal(t, sum(rank_tensor((4,), torch.float32, r) for r in range(world)))
    # Still healthy after every refusal.
    t = rank_tensor((4,), torch.float32, rank)
    dist.all_reduce(t)
    assert torch.equal(t, sum(rank_tensor((4,), torch.float32, r) for r in range(world)))


def test_unsupported_calls_fail_clearly_on_every_rank(tmp_path):
    run_backend(TWO_TWO, _refused, tmp_path)


def _first_collective_fails(rank, world):
    with pytest.raises(RuntimeError, match="Ranks picked different collective policies"):
        dist.barrier()


def test_a_setting_that_differs_fails_the_first_collective_on_every_rank(tmp_path):
    run_backend(TWO_TWO, _first_collective_fails, tmp_path,
                env={0: {POLICY_ENV: "flat-gloo"}, -1: {}})


def _two_jobs_in_a_row(rank, world):
    t = rank_tensor((3,), torch.float32, rank)
    dist.all_reduce(t)
    with pytest.raises(RuntimeError, match="already initialized"):
        gpubridge.init()


def test_gpubridge_init_and_the_backend_exclude_each_other(tmp_path):
    run_backend(TWO_TWO, _two_jobs_in_a_row, tmp_path)


def _backend_group_after_init(topology):
    group = dist.new_group(list(range(topology.world_size)), backend="gpubridge")
    with pytest.raises(RuntimeError, match="use either gpubridge.init\\(\\) or the gpubridge"):
        dist.all_reduce(torch.ones(2), group=group)


def test_a_backend_group_after_gpubridge_init_is_refused(tmp_path):
    from _harness import run_ranks

    run_ranks(TWO_TWO, _backend_group_after_init, tmp_path)


def test_bridge_from_gpubridge_bridge_is_checked(tmp_path):
    run_backend(["nvidia", "amd"], _first_collective_unknown_bridge, tmp_path,
                env={-1: {BRIDGE_ENV: "no-such-bridge"}})


def _first_collective_unknown_bridge(rank, world):
    with pytest.raises(ValueError, match="unknown bridge transport 'no-such-bridge'"):
        dist.barrier()


# ---- registration and pure logic -----------------------------------------------------------

def test_the_backend_is_registered_by_import_gpubridge():
    assert "gpubridge" in dist.Backend.backend_list
    torch_backend.register()  # again: nothing happens
    assert dist.Backend.backend_list.count("gpubridge") == 1


def _entry_point_installed() -> bool:
    return any(ep.name == "gpubridge" for ep in importlib.metadata.entry_points(
        group="torch.backends"))


@pytest.mark.skipif(not _entry_point_installed(), reason="gpubridge isn't installed with its "
                    "entry points (e.g. run from a source tree)")
def test_import_torch_registers_it_without_import_gpubridge():
    code = ("import sys, torch, torch.distributed as dist; "
            "print('gpubridge' in dist.Backend.backend_list, 'gpubridge.topology' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["True", "False"]  # registered, and gpubridge not yet imported


def test_the_autoload_hook_never_raises(monkeypatch):
    import gpubridge_backend

    monkeypatch.setitem(sys.modules, "torch.distributed", None)  # importing it now fails
    gpubridge_backend.register()  # must not raise: it would break `import torch`


def test_cpu_tensors_go_to_gloo_only_on_gpu_jobs():
    gpu = SimpleNamespace(device=torch.device("cuda", 0))
    cpu = SimpleNamespace(device=torch.device("cpu"))
    assert torch_backend._on_gloo(gpu, [torch.ones(1)])
    assert not torch_backend._on_gloo(cpu, [torch.ones(1)])  # simulation: CPU is gpubridge's
    assert not torch_backend._on_gloo(gpu, [torch.ones(1), torch.ones(1, device="meta")])


# ---- on GPUs (written; skipped without one) ------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_gpu_ddp_on_two_gpus_split_into_two_islands(tmp_path):
    run_backend(["nvidia", "nvidia"], _gpu_ddp, tmp_path, env={-1: {
        VENDOR_ENV: "", "GPUBRIDGE_SPLIT_TEST": "half"}})


def _gpu_ddp(rank, world):
    torch.cuda.set_device(rank)
    model = _integer_model().cuda()
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ddp(_batch(rank, 0).cuda()).sum().backward()
    for p in model.parameters():
        g = p.grad.clone()
        dist.all_reduce(g, op=dist.ReduceOp.MAX)
        assert torch.equal(g, p.grad)  # identical on every rank
    flags = torch.ones(1)  # a CPU tensor on a GPU job: Gloo carries it
    dist.all_reduce(flags)
    assert flags.item() == world
