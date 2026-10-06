"""Per-rank correctness check for the GPU validation kit, launched by torchrun.

Stages (see docs/GPU_VALIDATION_RUNBOOK.md for which HARDWARE_VALIDATION.md
item each one closes):

1. init: gpubridge.init(), discovery, islands, bridge, new_group on non-members (item 3)
2. device: which GPU this rank drives, by index and PCI bus id (item 5)
3. dtypes: all_reduce of integer-valued data in float32, float16, bfloat16 and
   int64, checked for exact results (item 7)
4. busy_gpu: a large tensor reduced right behind queued GPU work, so the
   bridge's device-to-host copy must wait for it (items 6 and 9)
5. broadcast: from every rank
6. barrier: includes torch.cuda.synchronize on GPUs (item 10)
7. destroy

Writes OUT/rank<N>.json. Run scripts/gpu/summarize.py OUT afterwards.

    torchrun --nproc-per-node=4 scripts/gpu/check.py --out results/explorer/04_split_4gpu
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
from gpukit import Record, rank_from_env, run_info, sync

import gpubridge

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
}


def rank_values(shape: tuple[int, ...], dtype: torch.dtype, rank: int, device) -> torch.Tensor:
    """Small integers that differ per rank and element, so sums are exact in every dtype."""
    numel = int(torch.Size(shape).numel())
    base = (torch.arange(numel, dtype=torch.float64) % 64).reshape(shape)
    return (base + rank + 1).to(dtype).to(device)


def expected_sum(shape, dtype, world, device) -> torch.Tensor:
    total = sum(rank_values(shape, torch.float64, r, "cpu") for r in range(world))
    return torch.as_tensor(total).to(dtype).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for rank JSON")
    parser.add_argument("--timeout", type=float, default=600, help="group timeout (s)")
    parser.add_argument("--dtypes", default=",".join(DTYPES), help="comma-separated")
    parser.add_argument("--large-numel", type=int, default=None,
                        help="elements in the busy-GPU tensor (default: 64M on GPU, 1M on CPU)")
    parser.add_argument("--busy-iters", type=int, default=20,
                        help="matmuls queued before the busy-GPU all_reduce")
    args = parser.parse_args()

    rank = rank_from_env()
    rec = Record(args.out / f"rank{rank}.json", "scripts/gpu/check.py",
                 hang_dump_seconds=args.timeout * 0.8)
    try:
        with rec.stage("init"):
            topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
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

        with rec.stage("barrier"):
            gpubridge.barrier()

        with rec.stage("destroy"):
            gpubridge.destroy()
    except BaseException:
        traceback.print_exc()
        rec.failed = True
    return rec.finish()


if __name__ == "__main__":
    sys.exit(main())
