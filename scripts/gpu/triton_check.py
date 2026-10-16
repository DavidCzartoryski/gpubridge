"""Whether this process's GPU can run gpubridge's Triton compression kernels.

gpubridge compresses GPU tensors with Triton kernels when Triton is installed
(GPUBRIDGE_COMPRESSION_KERNELS=auto, the default) or forced (=triton). A Triton that
can't compile them for the GPU, as venv-rocm63's 3.5.1 can't for an MI50's gfx906,
fails every step that uses them. common.sh runs this (as step triton_check) before
the first such step; if it fails, those steps are recorded as not run.

    python scripts/gpu/triton_check.py     # exit 0, or 1 with the reason
"""

from __future__ import annotations

import os
import sys

import torch

from gpukit import triton_check

FAILED = "gpubridge-kit: Triton can't run gpubridge's compression kernels here"


def verdict(result: dict | None, forced: str) -> str | None:
    """None if the kernels can run (or won't be needed); else why not.

    ``result`` is gpukit.triton_check's, or None without a GPU (the kernels then
    run as PyTorch ops on CPU tensors).
    """
    if result is None:
        return None
    if result["compiles"] is None:  # no Triton: auto uses the PyTorch kernels
        return (None if forced != "triton" else
                f"{FAILED}: GPUBRIDGE_COMPRESSION_KERNELS=triton but Triton isn't installed")
    if result["compiles"]:
        return None
    return (f"{FAILED}: Triton {result['triton']} can't compile them for this GPU "
            f"({result.get('error')}). On an MI50, run in the overlay venv "
            "(scripts/gpu/rocm_triton_overlay.sh; the default VENV_ROCM on Explorer).")


def main() -> int:
    forced = os.environ.get("GPUBRIDGE_COMPRESSION_KERNELS", "auto").strip().lower()
    result = triton_check(0) if torch.cuda.is_available() else None
    problem = verdict(result, forced)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    if result is None:
        print("no GPU: compression runs PyTorch ops on CPU tensors")
    elif result["compiles"]:
        print(f"Triton {result['triton']} compiles and runs the compression kernels "
              f"on {torch.cuda.get_device_name(0)}")
    else:
        print("Triton not installed: compression uses the PyTorch kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
