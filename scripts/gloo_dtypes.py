"""Check which dtypes Gloo all_reduce (SUM) and broadcast handle correctly in this torch.

Backs gpubridge.collectives.SUPPORTED_DTYPES. Two CPU processes contribute small
integers, so a correct SUM is exact; broadcast is checked from rank 1. Prints a
table and writes JSON:

    python scripts/gloo_dtypes.py /tmp/gloo-dtypes.json

Run it in each torch version you care about (e.g. the oldest supported one).
"""

import json
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CANDIDATES = ["float16", "bfloat16", "float32", "float64", "int8", "uint8", "int32", "int64"]
EXTRAS = ["bool", "int16", "complex64", "float8_e4m3fn", "uint16", "uint32"]


def values(dt, rank):
    if dt == torch.bool:
        return torch.tensor([True, rank == 0, False, rank == 1])
    return (torch.arange(6) % 3 + rank + 1).to(dt)


def same(a, b):
    if a.dtype in (getattr(torch, "float8_e4m3fn", None),):
        a, b = a.float(), b.float()
    return bool(torch.equal(a, b))


def first_line(exc):
    return (str(exc).splitlines() or [type(exc).__name__])[0][:120]


def worker(rank, init_file, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    results = {}
    for name in CANDIDATES + EXTRAS:
        dt = getattr(torch, name, None)
        if dt is None:
            results[name] = {"all_reduce": "n/a (not in this torch)", "broadcast": "n/a"}
            continue
        entry = {}
        try:
            t = values(dt, rank)
            if dt == torch.bool:
                expected = values(dt, 0) | values(dt, 1)  # report what SUM does
            else:
                expected = (values(torch.float64, 0) + values(torch.float64, 1)).to(dt)
            dist.all_reduce(t)
            entry["all_reduce"] = "ok" if same(t, expected) else f"WRONG {t.float().tolist()}"
            if dt == torch.bool and entry["all_reduce"] == "ok":
                entry["all_reduce"] = "ok (logical OR)"
        except Exception as exc:
            entry["all_reduce"] = "ERROR " + first_line(exc)
        try:
            t = values(dt, rank)
            dist.broadcast(t, src=1)
            entry["broadcast"] = "ok" if same(t, values(dt, 1)) else "WRONG"
        except Exception as exc:
            entry["broadcast"] = "ERROR " + first_line(exc)
        results[name] = entry
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        with open(out, "w") as fh:
            json.dump({"torch": torch.__version__, "results": results}, fh)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gloo-dtypes.json"
    mp.spawn(worker, args=(tempfile.mktemp(), out), nprocs=2)
    with open(out) as fh:
        data = json.load(fh)
    print(f"torch {data['torch']}")
    for name, entry in data["results"].items():
        print(f"  {name:14} all_reduce: {entry['all_reduce']:28} broadcast: {entry['broadcast']}")
