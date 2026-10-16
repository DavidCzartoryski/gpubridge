"""Single-process environment probe for the GPU validation kit.

Records gpubridge.probe(), every visible GPU and the PyTorch build, and checks
that this build can run on these GPUs, that ``import gpubridge`` found
this checkout's src (import_check.py). It also records whether an installed
Triton compiles gpubridge's compression kernels for the GPU, without failing on it:
only steps that use them check (triton_check.py). Writes OUT/probe.json. If a check
fails, it prints one line per failure on stderr, naming the check and its
values, and exits 1. A failed import check also stops the job (common.sh).

    python scripts/gpu/probe.py --out results/explorer/01_probe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gpukit import Record, blas_check, triton_check
from import_check import message as import_problem


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
        # A device whose properties couldn't be read: its entry is the error.
        unreadable = {i: d for i, d in enumerate(env["devices"]) if not isinstance(d, dict)}
        entry["arch_list"] = env["arch_list"]
        entry["unsupported"] = unsupported
        if unreadable:
            entry["unreadable"] = unreadable
        entry["ok"] = not unsupported and not unreadable
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
    with rec.stage("kit_import") as entry:
        entry.update(env["import"] if isinstance(env["import"], dict) else {})
        problem = (import_problem(env["import"]) if isinstance(env["import"], dict)
                   else f"could not tell where gpubridge came from: {env['import']}")
        entry["ok"] = problem is None
        if problem:
            entry["error"] = problem
    with rec.stage("triton_kernels") as entry:
        # Informational: only steps that use the kernels check them (triton_check.py).
        entry["kernels_setting"] = os.environ.get("GPUBRIDGE_COMPRESSION_KERNELS", "auto")
        entry.update(triton_check(0) if env["device_count"] else {"compiles": None})
        if entry["compiles"] is False:
            print(f"note: Triton {entry['triton']} can't compile gpubridge's compression "
                  f"kernels here ({entry['error']}); steps that use them won't run",
                  file=sys.stderr)
        entry["ok"] = True
    with rec.stage("blas") as entry:
        # Informational: gpubridge never needs BLAS, so this never fails the probe.
        if env["device_count"]:
            entry.update(blas_check(0))
            if not entry["blas_available"]:
                print(f"note: no BLAS on device 0 (gpubridge doesn't need it): "
                      f"{entry['error']}", file=sys.stderr)
        entry["ok"] = True
    devices = ", ".join(f"{d['name']} ({d['target']}, {d['target_match'] or 'not in the build'})"
                        for d in env["devices"] if isinstance(d, dict)) or "none"
    print(f"torch {env['torch']} (cuda {env['torch_cuda']}, hip {env['torch_hip']}, "
          f"nccl/rccl {env['nccl_version']}); GPUs: {devices}")
    code = rec.finish()
    for line in failures(rec.data["stages"]):
        print(line, file=sys.stderr)
    return code


def failures(stages: list[dict]) -> list[str]:
    """One line per failed check, naming it and its values."""
    lines = []
    for stage in stages:
        if stage.get("ok") is not False:
            continue
        name = stage["name"]
        if name == "gpus_visible":
            lines.append(f"FAIL gpus_visible: device_count {stage['device_count']}: "
                         f"{stage['error']}")
        elif name == "build_supports_gpus":
            for index, error in stage.get("unreadable", {}).items():
                lines.append(f"FAIL device_info: device {index}: {error}")
            for d in stage["unsupported"]:
                cap = d["capability"]
                rules = (f" (nor an sm_{cap.split('.')[0]}x cubin at or below {cap}, nor PTX at "
                         f"or below {cap})" if d["target"].startswith("sm_") else "")
                lines.append(f"FAIL target_in_build: device {d['index']} target {d['target']} "
                             f"not in arch_list {stage['arch_list']}{rules}")
        elif name == "backends":
            lines.append(f"FAIL backends: nccl_available {stage['nccl_available']}, "
                         f"gloo_available {stage['gloo_available']}")
        else:
            lines.append(f"FAIL {name}: {stage.get('error', 'failed')}")
    return lines


if __name__ == "__main__":
    sys.exit(main())
