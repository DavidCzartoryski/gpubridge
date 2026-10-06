"""Single-process environment probe for the GPU validation kit.

Records gpubridge.probe(), every visible GPU and the PyTorch build, and checks
that this build can run on these GPUs. Writes OUT/probe.json.

    python scripts/gpu/probe.py --out results/explorer/01_probe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpukit import Record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="don't fail without a GPU (for testing the kit on CPU)")
    args = parser.parse_args()

    rec = Record(args.out / "probe.json", "scripts/gpu/probe.py")
    env = rec.data["environment"]
    with rec.stage("gpus_visible") as entry:
        entry["device_count"] = env["device_count"]
        entry["ok"] = env["device_count"] > 0 or args.allow_no_gpu
        if not entry["ok"]:
            entry["error"] = "no GPU visible to PyTorch (check the job's GPU request and driver)"
    with rec.stage("build_supports_gpus") as entry:
        unsupported = [d for d in env["devices"]
                       if isinstance(d, dict) and not d["target_in_build"]]
        entry["arch_list"] = env["arch_list"]
        entry["unsupported"] = unsupported
        entry["ok"] = not unsupported
        if unsupported:
            entry["error"] = (
                f"this PyTorch build has no kernels for {[d['target'] for d in unsupported]}; "
                f"it supports {env['arch_list']}. Pick a different wheel index (e.g. cu126 "
                "for V100)."
            )
    with rec.stage("backends") as entry:
        probe = env["probe"]
        entry["nccl_available"] = probe["nccl_available"]
        entry["gloo_available"] = probe["gloo_available"]
        entry["ok"] = probe["gloo_available"] and (probe["nccl_available"] or args.allow_no_gpu)
    devices = ", ".join(f"{d['name']} ({d['target']})" for d in env["devices"]
                        if isinstance(d, dict)) or "none"
    print(f"torch {env['torch']} (cuda {env['torch_cuda']}, hip {env['torch_hip']}, "
          f"nccl/rccl {env['nccl_version']}); GPUs: {devices}")
    return rec.finish()


if __name__ == "__main__":
    sys.exit(main())
