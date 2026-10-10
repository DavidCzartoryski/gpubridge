"""Per-rank correctness check for the GPU validation kit, launched by torchrun.

Stages (see docs/GPU_VALIDATION_RUNBOOK.md for which HARDWARE_VALIDATION.md
item each one closes):

1. init: gpubridge.init(), discovery, islands, bridge, new_group on non-members (item 3)
2. device: which GPU this rank drives, by index and PCI bus id (item 5)
3. dtypes: all_reduce of integer-valued data in every dtype gpubridge supports
   (gpubridge.collectives.SUPPORTED_DTYPES), checked for exact results (item 7)
4. busy_gpu: a large tensor reduced right behind queued GPU work, so the
   bridge's device-to-host copy must wait for it (items 6 and 9)
5. broadcast: from every rank
6. reductions: all_reduce with MAX, MIN and AVG in every dtype (item 15)
7. all_gather: all_gather_into_tensor, rows in global rank order (item 15)
8. reduce_scatter: reduce_scatter_tensor with SUM and AVG (item 15)
9. async: an async all_reduce queued right behind GPU work, read after wait()
   on the caller's stream with no host sync in between (item 16)
10. reduction_agreement: informational. Whether the island backend and Gloo
    agree on PRODUCT in every dtype, and on MAX/MIN with NaN (item 14)
11. in_flight: the last rank reaches an all_reduce late; meanwhile every rank
    records what gpubridge.in_flight() shows from another thread (item 23)
12. barrier: includes torch.cuda.synchronize on GPUs (item 10)
13. destroy

Writes OUT/rank<N>.json. Run scripts/gpu/summarize.py OUT afterwards.

    torchrun --nproc-per-node=4 scripts/gpu/check.py --out results/explorer/04_split_4gpu

--policy picks the collective policy (default auto), so runs can compare them.

--expect-init-error REGEX turns the run into a negative test: every rank must
fail gpubridge.init() with a matching message, and nothing else runs (item 12).
Launch one more process than the node has GPUs and expect "LOCAL_RANK":

    torchrun --nproc-per-node=2 scripts/gpu/check.py --out OUT --expect-init-error LOCAL_RANK
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
from gpukit import Record, default_timeout, rank_from_env, run_info, sync

import gpubridge
from gpubridge.collectives import SUPPORTED_DTYPES

DTYPES = {str(dtype).removeprefix("torch."): dtype for dtype in SUPPORTED_DTYPES}


def _values64(shape: tuple[int, ...], dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Small integers that differ per rank and element, kept small enough that
    every partial sum is exact: at most 16 per element (bfloat16 stays exact up
    to 16 ranks) and at most 3 for 8-bit types (int8 stays exact up to 42 ranks)."""
    index = torch.arange(int(torch.Size(shape).numel()), dtype=torch.float64).reshape(shape)
    if dtype in (torch.int8, torch.uint8):
        return index % 2 + rank % 2 + 1
    return index % 8 + rank % 8 + 1


def rank_values(shape: tuple[int, ...], dtype: torch.dtype, rank: int, device) -> torch.Tensor:
    return _values64(shape, dtype, rank).to(dtype).to(device)


def expected_sum(shape, dtype, world, device) -> torch.Tensor:
    total = sum(_values64(shape, dtype, r) for r in range(world))
    return torch.as_tensor(total).to(dtype).to(device)


def expected_reduce(shape, dtype, world, device, op: str) -> torch.Tensor:
    """The exact answer for rank_values reduced with op ("sum", "avg", "max", "min")."""
    if op in ("sum", "avg"):
        total = expected_sum(shape, dtype, world, device)
        return total.div(world) if op == "avg" else total
    stacked = torch.stack([_values64(shape, dtype, r) for r in range(world)])
    picked = stacked.amax(0) if op == "max" else stacked.amin(0)
    return picked.to(dtype).to(device)


OPS = {"sum": gpubridge.ReduceOp.SUM, "avg": gpubridge.ReduceOp.AVG,
       "max": gpubridge.ReduceOp.MAX, "min": gpubridge.ReduceOp.MIN}


def mismatch(tensor: torch.Tensor, expected: torch.Tensor) -> dict | None:
    if torch.equal(tensor, expected):
        return None
    diff = (tensor.double() - expected.double()).abs()
    return {"max_abs_error": float(diff.max()), "wrong_elements": int((diff > 0).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for rank JSON")
    parser.add_argument("--timeout", type=float, default=default_timeout(600),
                        help="group timeout (s); default KIT_TIMEOUT, else 600")
    parser.add_argument("--dtypes", default=",".join(DTYPES), help="comma-separated")
    parser.add_argument("--policy", default="auto", help="collective policy for gpubridge.init")
    parser.add_argument("--large-numel", type=int, default=None,
                        help="elements in the busy-GPU tensor (default: 64M on GPU, 1M on CPU)")
    parser.add_argument("--busy-iters", type=int, default=20,
                        help="matmuls queued before the busy-GPU all_reduce")
    parser.add_argument("--in-flight-wait", type=float, default=1.0,
                        help="seconds before each rank looks at in_flight(); the late "
                             "rank waits twice as long")
    parser.add_argument("--expect-init-error", metavar="REGEX",
                        help="pass only if gpubridge.init() fails on this rank with a message "
                             "matching REGEX; runs nothing else")
    args = parser.parse_args()

    rank = rank_from_env()
    rec = Record(args.out / f"rank{rank}.json", "scripts/gpu/check.py",
                 hang_dump_seconds=args.timeout * 0.8)
    if args.expect_init_error:
        return expect_init_error(rec, args)
    try:
        with rec.stage("init"):
            topology = gpubridge.init(timeout=timedelta(seconds=args.timeout),
                                      policy=args.policy)
        rec.data["run"] = run_info(topology)
        rec.write()
        if topology.split_test and rank == 0:
            print(rec.data["run"]["warning"], file=sys.stderr, flush=True)
        device = topology.device
        world = topology.world_size

        with rec.stage("device") as entry:
            entry["device"] = str(device)
            if device.type == "cuda":
                index = torch.cuda.current_device()
                entry["current_device"] = index
                entry["matches_topology"] = index == device.index
                props = torch.cuda.get_device_properties(index)
                entry["name"] = props.name
                entry["pci_bus_id"] = getattr(props, "pci_bus_id", None)
                entry["uuid"] = str(getattr(props, "uuid", "")) or None
                entry["ok"] = index == device.index

        with rec.stage("dtypes") as entry:
            entry["cases"] = []
            for name in args.dtypes.split(","):
                dtype = DTYPES[name]
                tensor = rank_values((1024, 3), dtype, rank, device)
                gpubridge.all_reduce(tensor)
                expected = expected_sum((1024, 3), dtype, world, device)
                ok = bool(torch.equal(tensor, expected))
                entry["cases"].append({"dtype": name, "ok": ok})
                if not ok:
                    entry["ok"] = False
                    diff = (tensor.double() - expected.double()).abs()
                    entry["cases"][-1]["max_abs_error"] = float(diff.max())

        with rec.stage("busy_gpu") as entry:
            numel = args.large_numel or (64 * 2**20 if device.type == "cuda" else 2**20)
            entry["numel"] = numel
            entry["megabytes"] = round(numel * 4 / 2**20, 1)
            tensor = torch.full((numel,), float(rank + 1), device=device)
            if device.type == "cuda":
                # Queue real work ahead of the all_reduce and make the tensor depend
                # on it, so the bridge's .cpu() copy must wait for the GPU.
                work = torch.randn(2048, 2048, device=device)
                for _ in range(args.busy_iters):
                    work = torch.tanh(work @ work)
                tensor.add_(work.sum() * 0)
            start = time.perf_counter()
            gpubridge.all_reduce(tensor)
            sync(device)
            entry["all_reduce_seconds"] = round(time.perf_counter() - start, 4)
            expected = world * (world + 1) / 2
            entry["ok"] = bool(torch.all(tensor == expected))
            if not entry["ok"]:
                entry["wrong_elements"] = int((tensor != expected).sum())

        with rec.stage("broadcast") as entry:
            entry["bad_sources"] = []
            for src in range(world):
                tensor = rank_values((257,), torch.float32, rank, device)
                gpubridge.broadcast(tensor, src)
                if not torch.equal(tensor, rank_values((257,), torch.float32, src, device)):
                    entry["bad_sources"].append(src)
            entry["ok"] = not entry["bad_sources"]

        with rec.stage("reductions") as entry:
            entry["cases"] = []
            for name in args.dtypes.split(","):
                dtype = DTYPES[name]
                for op in ("max", "min") + (("avg",) if dtype.is_floating_point else ()):
                    tensor = rank_values((1024, 3), dtype, rank, device)
                    gpubridge.all_reduce(tensor, op=OPS[op])
                    bad = mismatch(tensor, expected_reduce((1024, 3), dtype, world, device, op))
                    entry["cases"].append({"dtype": name, "op": op, "ok": bad is None,
                                           **(bad or {})})
            entry["ok"] = all(case["ok"] for case in entry["cases"])

        with rec.stage("all_gather") as entry:
            entry["cases"] = []
            for name in args.dtypes.split(","):
                dtype = DTYPES[name]
                piece = rank_values((257, 2), dtype, rank, device)
                output = torch.empty((world * 257, 2), dtype=dtype, device=device)
                gpubridge.all_gather_into_tensor(output, piece)
                want = torch.cat([rank_values((257, 2), dtype, r, device) for r in range(world)])
                bad = mismatch(output, want)
                entry["cases"].append({"dtype": name, "ok": bad is None, **(bad or {})})
            entry["ok"] = all(case["ok"] for case in entry["cases"])

        with rec.stage("reduce_scatter") as entry:
            entry["cases"] = []
            for name in args.dtypes.split(","):
                dtype = DTYPES[name]
                for op in ("sum",) + (("avg",) if dtype.is_floating_point else ()):
                    full = rank_values((world * 129,), dtype, rank, device)
                    output = torch.empty(129, dtype=dtype, device=device)
                    gpubridge.reduce_scatter_tensor(output, full, op=OPS[op])
                    want = expected_reduce((world * 129,), dtype, world, device, op)
                    bad = mismatch(output, want[rank * 129:(rank + 1) * 129])
                    entry["cases"].append({"dtype": name, "op": op, "ok": bad is None,
                                           **(bad or {})})
            entry["ok"] = all(case["ok"] for case in entry["cases"])

        with rec.stage("async") as entry:
            numel = 4 * 2**20 if device.type == "cuda" else 2**16
            tensor = torch.full((numel,), float(rank + 1), device=device)
            if device.type == "cuda":
                # Queue real work on this stream and make the tensor depend on it: the
                # worker's stream must wait for it before the all_reduce reads the tensor.
                work_matrix = torch.randn(2048, 2048, device=device)
                for _ in range(args.busy_iters):
                    work_matrix = torch.tanh(work_matrix @ work_matrix)
                tensor.add_(work_matrix.sum() * 0)
            work = gpubridge.all_reduce(tensor, async_op=True)
            work.wait()  # orders this stream after the collective; no host sync
            correct = tensor == world * (world + 1) / 2  # queued after wait() on this stream
            entry["ok"] = bool(correct.all())
            entry["is_completed"] = work.is_completed()
            if not entry["ok"]:
                entry["wrong_elements"] = int((~correct).sum())

        with rec.stage("reduction_agreement") as entry:
            entry.update(reduction_agreement(topology, rank, device))

        with rec.stage("in_flight") as entry:
            entry.update(in_flight_while_late(rank, world, device, args.in_flight_wait))

        with rec.stage("barrier"):
            gpubridge.barrier()

        with rec.stage("destroy"):
            gpubridge.destroy()
    except BaseException:
        traceback.print_exc()
        rec.failed = True
    return rec.finish()


def in_flight_while_late(rank: int, world: int, device, wait: float) -> dict:
    """The last rank reaches an all_reduce ``2 * wait`` seconds late; after ``wait``
    seconds every rank records ``gpubridge.in_flight()`` from another thread.

    Fails only if the late rank shows a collective or an entry isn't this
    all_reduce. Which phase each rank shows is informational: on GPUs a rank's
    host can leave a collective once its device work is queued (item 23).
    """
    late = world - 1
    seen: list[list[gpubridge.InFlight]] = []

    def look() -> None:
        time.sleep(wait)
        seen.append(gpubridge.in_flight())

    watcher = threading.Thread(target=look, daemon=True)
    watcher.start()
    if rank == late:
        time.sleep(2 * wait)
    tensor = torch.ones(16, device=device)
    gpubridge.all_reduce(tensor)
    sync(device)
    watcher.join()
    flights = [{"seq": f.seq, "op": f.op, "policy": f.policy, "phase": f.phase,
                "phases_done": list(f.phases_done), "thread": f.thread,
                "seconds": round(f.seconds, 3)} for f in (seen[0] if seen else [])]
    ok = bool(seen) and all(f["op"] == "all_reduce" for f in flights)
    return {"late_rank": late, "wait_seconds": wait, "in_flight": flights,
            "ok": ok and (rank != late or not flights)}


def reduction_agreement(topology: gpubridge.Topology, rank: int, device) -> dict:
    """Informational (item 14): does the island backend agree with Gloo where gpubridge can't
    yet promise it does? PRODUCT in every dtype, including 8-bit overflow, and MAX/MIN with
    a NaN on one rank. Uses raw torch.distributed calls, not gpubridge."""
    import torch.distributed as dist

    groups = {}
    for island in topology.layout.islands:  # collective: every rank, in the same order
        groups[island.ranks] = dist.new_group(list(island.ranks), backend="gloo")
    gloo = groups[topology.island.ranks]

    def both(tensor: torch.Tensor, op) -> tuple[torch.Tensor, torch.Tensor]:
        native = tensor.clone()
        dist.all_reduce(native, op=op, group=topology.island_group)
        on_cpu = tensor.cpu()
        dist.all_reduce(on_cpu, op=op, group=gloo)
        return native.cpu(), on_cpu

    product, nan = {}, {}
    for name, dtype in DTYPES.items():
        if dtype in (torch.int8, torch.uint8):
            base = torch.tensor([3.0, 7.0, 2.0, 1.0])  # 7 ** ranks overflows quickly
        else:
            base = torch.tensor([1.0, 2.0, -1.0, 0.5]) if dtype.is_floating_point else (
                torch.tensor([1.0, 2.0, -1.0, 3.0]))
        tensor = (base + (rank % 2)).to(dtype).to(device)
        native, on_cpu = both(tensor, dist.ReduceOp.PRODUCT)
        product[name] = bool(torch.equal(native, on_cpu))
        if dtype.is_floating_point:
            values = torch.tensor([1.0, -2.0, 3.0]).to(dtype)
            if rank == topology.island.leader:
                values[0] = float("nan")
            for op_name, op in (("max", dist.ReduceOp.MAX), ("min", dist.ReduceOp.MIN)):
                native, on_cpu = both(values.to(device), op)
                nan[f"{name}/{op_name}"] = {
                    "agree": bool(torch.equal(native.isnan(), on_cpu.isnan())
                                  and torch.equal(native.nan_to_num(), on_cpu.nan_to_num())),
                    "native_first": str(native[0].item()), "gloo_first": str(on_cpu[0].item())}
    return {"ok": True, "backend": topology.config.island_backend,
            "product_agree": product, "all_product_agree": all(product.values()),
            "nan_max_min": nan}


def expect_init_error(rec: Record, args: argparse.Namespace) -> int:
    """Negative test: init() must fail on this rank, with a message matching the regex."""
    with rec.stage("expected_init_error") as entry:
        entry["pattern"] = args.expect_init_error
        try:
            gpubridge.init(timeout=timedelta(seconds=args.timeout), policy=args.policy)
        except RuntimeError as exc:
            entry["message"] = str(exc)
            entry["ok"] = re.search(args.expect_init_error, str(exc)) is not None
            if not entry["ok"]:
                entry["error"] = f"init() failed, but not with {args.expect_init_error!r}"
        else:
            gpubridge.destroy()
            entry["ok"] = False
            entry["error"] = "init() succeeded; this run expected it to fail"
    return rec.finish()


if __name__ == "__main__":
    sys.exit(main())
