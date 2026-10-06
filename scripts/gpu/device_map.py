"""Check that the visible-devices variable selects the GPU you think it does (item 5).

Lists every visible GPU's PCI bus id, then starts a child process with only
the last GPU visible (CUDA_VISIBLE_DEVICES on CUDA builds; HIP_VISIBLE_DEVICES
and ROCR_VISIBLE_DEVICES on ROCm builds) and checks that the child's cuda:0 is
that same GPU. Writes OUT/device_map.json. Needs 2 or more GPUs to be useful.

    python scripts/gpu/device_map.py --out results/amd/device_map
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from gpukit import Record

CHILD = (
    "import json, torch; p = torch.cuda.get_device_properties(0) "
    "if torch.cuda.device_count() else None; "
    "print(json.dumps({'count': torch.cuda.device_count(), "
    "'bus': getattr(p, 'pci_bus_id', None) if p else None, "
    "'name': p.name if p else None}))"
)


def bus_id(index: int):
    return getattr(torch.cuda.get_device_properties(index), "pci_bus_id", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rec = Record(args.out / "device_map.json", "scripts/gpu/device_map.py")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    variables = (["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] if torch.version.hip
                 else ["CUDA_VISIBLE_DEVICES"])
    with rec.stage("list") as entry:
        entry["devices"] = [{"index": i, "pci_bus_id": bus_id(i)} for i in range(count)]
        entry["ok"] = count > 0
    if count >= 2:
        target = count - 1
        for variable in variables:
            with rec.stage(f"select_with_{variable}") as entry:
                env = {k: v for k, v in os.environ.items() if k not in variables}
                env[variable] = str(target)
                result = subprocess.run([sys.executable, "-c", CHILD], env=env,
                                        capture_output=True, text=True, timeout=120)
                entry["variable"] = f"{variable}={target}"
                entry["stderr_tail"] = result.stderr[-2000:]
                child = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout else {}
                entry["child"] = child
                entry["expected_bus"] = bus_id(target)
                entry["ok"] = child.get("count") == 1 and child.get("bus") == bus_id(target)
    else:
        rec.data["note"] = "fewer than 2 GPUs: nothing to select between"
    return rec.finish()


if __name__ == "__main__":
    sys.exit(main())
