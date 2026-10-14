# The GPU validation kit on Kaggle (GPU T4 x2)

A free Kaggle notebook with two NVIDIA T4s can run most of gpubridge's GPU
checks on real hardware: the GPU-only tests, the kit's probe, the NVIDIA split
test (two islands and the bridge on device tensors), the all_reduce benchmark
with CUDA events, the compression kernels' bit-identical check and the FSDP2
example. Two T4s are one vendor, so this is not a mixed NVIDIA + AMD run: the
split test gives real NCCL islands and a real bridge, and the results say
`split-test` wherever they appear.

- `kaggle_gpubridge.ipynb`: the notebook to upload. Its first cell holds
  `kaggle_gpubridge.py`; each later cell runs one step.
- `kaggle_gpubridge.py`: the same steps as one script
  (`python kaggle_gpubridge.py --tag v0.4.0-alpha`, `--dry-run` to see the plan),
  for a Kaggle terminal or any machine with two NVIDIA GPUs.
- `make_notebook.py`: rebuilds the notebook from the script. A test fails if
  they differ.

## The clicks

Labels are as of October 2026; Kaggle renames things now and then.

1. **Sign in** at kaggle.com. Notebooks get internet access only on a
   phone-verified account: your profile picture → **Settings** → **Phone
   verification**.
2. **Get the notebook:** download `kaggle_gpubridge.ipynb` from this folder
   on GitHub (**Download raw file**).
3. **Create a notebook:** **Create** → **New Notebook**. Then **File** →
   **Import Notebook**, and choose the downloaded file.
4. **Session options**, in the right-hand sidebar (open it with the arrow at
   the top right if it is hidden):
   - **Accelerator:** **GPU T4 x2**. The session restarts with the GPUs.
   - **Internet:** **On**.
5. **Optional:** in the settings cell, set `TAG` to the gpubridge tag to test
   (default `v0.4.0-alpha`).
6. **Run:** **Run All** in the toolbar, or Shift+Enter cell by cell. It takes
   about 30 to 45 minutes, most of it the PyTorch download. Each step prints as
   it runs, and a failed step doesn't stop the rest.
7. **Download:** the last cell shows a link to `kaggle-t4x2-<date>.tgz`; click
   it. The file is also in the sidebar's **Output** panel (`/kaggle/working`),
   with a download button beside it.
8. **Stop the session** (the power icon at the top) when you're done, to save
   GPU quota.

To keep a run's output as a saved version instead: **Save Version** → **Save
& Run All (Commit)**. The tarball then appears in that version's **Output**
tab.

## What it runs, and its limits

| Step | What | Time limit |
| --- | --- | --- |
| `clone`, `setup` | the tag, and a venv in /tmp with torch 2.14.1+cu126 (it brings Triton) and that gpubridge | 5, 40 min |
| `versions` | `nvidia-smi`; the torch, CUDA, NCCL and Triton versions | 2 min |
| `features` | on each GPU: CUDA events, bfloat16, a Triton float32 and bfloat16 kernel | 10 min |
| `probe`, `probe_gpu0`, `probe_gpu1` | the kit's probe with both GPUs, then each alone | 5 min each |
| `gpu_tests` | `pytest -m gpu` on test_gpu.py, test_compression_kernels.py, test_torch_backend.py | 30 min |
| `check_split` | `check.py` with `GPUBRIDGE_SPLIT_TEST=half` on 2 GPUs | 15 min |
| `bench` | `bench_all_reduce.py`: native NCCL, gpubridge, flat-gloo, 1 KB to 16 MB | 20 min |
| `bench_compression_kernels` | `bench_compression.py`: kernel throughput, and digests against the CPU reference | 15 min |
| `verify_fsdp2_backend` | `examples/fsdp2_torch_backend.py --verify` across the two islands | 15 min |
| `tables`, `package` | `validation_tables.py`, then the tarball | 5 min each |

- **Hard limits.** A step that runs past its limit has its whole process
  group killed. steps.jsonl records it with `"timed_out": true`, and the next
  step runs. Kit tools also pass a 300 s timeout to gpubridge.init, so a stuck
  collective fails with its own error first.
- **T4s are compute capability 7.5.** The `features` step tries what that
  could break, and later steps adapt with the reason recorded (in
  `skipped.jsonl`, the step's log and `kaggle_summary.json`):
  - **CUDA events:** without them there's no `bench`.
  - **bfloat16:** a T4 has no native bfloat16 math, and PyTorch emulates it.
    If the emulated arithmetic or the Triton bfloat16 kernel fails, the
    compression benchmark runs on float32 inputs only.
  - **Triton:** if no Triton kernel runs, the compression kernel tests aren't
    run, and the compression benchmark and the GPU tests run on the PyTorch
    kernels under other step names (`bench_compression_torch_kernels`,
    `gpu_tests_without_triton`). That way they never count as evidence for
    the Triton kernels in HARDWARE_VALIDATION.md.
  - The compression kernels' fp8 is integer-encoded, so it needs no fp8
    hardware.
- **Where things go.** The clone, venv and download cache live in
  `/tmp/gpubridge-kaggle`, so they aren't saved as notebook output. The
  results go to `/kaggle/working/results/<target>/`, packed into
  `/kaggle/working/<target>.tgz`.
- **What's in the tarball.** Each step's log, the kit's JSON records (with an
  allowlist of environment variables, as everywhere in the kit), NCCL's INFO
  logs, `steps.jsonl`, `skipped.jsonl` and `validation_tables.md`/`.json`.
  The tables number hosts instead of naming them and show no paths, so they're
  safe to publish; the raw records keep the container's hostname.
