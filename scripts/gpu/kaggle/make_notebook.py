"""Write kaggle_gpubridge.ipynb from kaggle_gpubridge.py, so the two can't drift apart.

    python scripts/gpu/kaggle/make_notebook.py           # rewrite the notebook
    python scripts/gpu/kaggle/make_notebook.py --check   # exit 1 if it is out of date

The notebook's first code cell writes kaggle_gpubridge.py, byte for byte; each later
cell runs one or two of its steps, so a failure shows up next to its step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "kaggle_gpubridge.py"
NOTEBOOK = HERE / "kaggle_gpubridge.ipynb"

INTRO = """\
# gpubridge GPU validation on Kaggle (GPU T4 x2)

Runs gpubridge's GPU validation kit on this notebook's two T4 GPUs and packs the
results into one tarball in `/kaggle/working`. Before you run it: **Accelerator: GPU T4 x2**
and **Internet: on** (see `scripts/gpu/kaggle/README.md`).

The first code cell writes `kaggle_gpubridge.py` (the same file as in the repository);
the cells after it run its steps in order. Each step has a hard time limit. A step that
fails is logged and the rest still run. Expect about 30 to 45 minutes, most of it the
PyTorch download.

Two T4s are one NVIDIA machine, so the two islands come from gpubridge's split test
(`GPUBRIDGE_SPLIT_TEST=half`): real NCCL and a real bridge on device tensors, but no AMD
GPU. These results never count as a mixed NVIDIA + AMD run."""

SETTINGS = """\
import sys, time
sys.path.insert(0, ".")
import kaggle_gpubridge as k

TAG = "v0.4.0-alpha"  # the gpubridge tag (or branch) to test
TARGET = "kaggle-t4x2-" + time.strftime("%Y%m%d-%H%M")
ARGS = ["--tag", TAG, "--target", TARGET]
print(TAG, TARGET)"""

STEP_CELLS = [
    ("Clone the tag and install torch (CUDA build, with Triton) and gpubridge into a venv in /tmp.",
     "clone,setup"),
    ("nvidia-smi, and the torch, CUDA, NCCL and Triton versions.", "versions"),
    ("What these GPUs can run (CUDA events, bfloat16, Triton): later steps skip or adapt.",
     "features"),
    ("The kit's probe: once with both GPUs, then on each GPU alone.", "probe"),
    ("The GPU-only tests: `pytest -m gpu`.", "gpu_tests"),
    ("The NVIDIA split test: two islands on 2 GPUs, the bridge on device tensors.", "check_split"),
    ("all_reduce with CUDA events: native NCCL, gpubridge and flat-gloo, 1 KB to 16 MB.", "bench"),
    ("Compression kernels: GPU output digests against the CPU reference (the bit-identical check).",
     "bench_compression_kernels"),
    ("The FSDP2 example on the torch.distributed backend, with `--verify`.", "verify_fsdp2_backend"),
    ("validation_tables.py, then one tarball to download.", "tables,package"),
]

DOWNLOAD = """\
from IPython.display import FileLink, display
display(FileLink(f"{TARGET}.tgz"))
print(open(f"results/{TARGET}/kaggle_summary.json").read())"""


def cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    out = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def notebook() -> dict:
    cells = [cell("markdown", INTRO),
             cell("code", "%%writefile kaggle_gpubridge.py\n" + SCRIPT.read_text()),
             cell("code", SETTINGS)]
    for text, steps in STEP_CELLS:
        cells += [cell("markdown", text), cell("code", f'k.main(ARGS + ["--only", "{steps}"])')]
    cells += [cell("markdown", "Download the results (also in the Output panel)."),
              cell("code", DOWNLOAD)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                    "isGpuEnabled": True}}}


def render() -> str:
    return json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if not NOTEBOOK.exists() or NOTEBOOK.read_text() != text:
            print(f"{NOTEBOOK.name} is out of date: run python {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    NOTEBOOK.write_text(text)
    print(f"wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
