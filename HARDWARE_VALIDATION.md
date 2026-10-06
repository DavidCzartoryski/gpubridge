# What still needs real-hardware validation

The library itself is tested in CPU simulation mode. Simulation runs the same
orchestration code, but it cannot answer the questions here, which depend on
real CUDA and ROCm builds of PyTorch running side by side. Items 1 and 2 are now
answered for the CPU side by running real builds without GPUs; see
[Mixed build results](#mixed-build-results) below.

## Blocking: can the job start at all?

1. **CUDA-PyTorch and ROCm-PyTorch processes rendezvous in one job.** Both
   builds must connect to one c10d store (TCPStore through `torchrun` or
   `tcp://`) and form one Gloo default group. This is expected to work because
   the store and Gloo run on CPU and are compiled into both builds, but nobody
   has checked it. Test it with a launcher that starts one `torchrun` per node,
   each with its own PyTorch build.

   **Status: works on CPU.** CUDA-build and ROCm-build processes rendezvoused
   through torchrun's c10d backend and completed discovery, all_reduce and
   broadcast in every run, with either build hosting the store. Still open:
   separate hosts on a real network, and the same with GPUs attached.
2. **Version matching between builds.** Use the same PyTorch release on both
   sides, e.g. `2.x.y+cu12x` and `2.x.y+rocm6.x`, and the same Python minor
   version, since discovery pickles objects with `all_gather_object`.
   `init()` warns when releases differ but does not block. Things to find out:
   - Which release pairs actually work.
   - Whether a minor-version mismatch breaks the store or Gloo protocol.
   - Whether ROCm wheels lag behind CUDA wheels for the release you want.

   **Status: looser than feared on CPU.** Mixed releases from 2.9.1 to 2.14.1
   worked, in both directions, with Python 3.12.12 on every rank. CUDA and ROCm
   wheels exist for every release from 2.6 to 2.14 (Python 3.12, Linux x86_64),
   with different CUDA/ROCm versions per release. Still open: other Python
   versions, releases outside 2.9.1 to 2.14.1, and the GPU side.
3. **`new_group(backend="nccl")` on ranks outside the group.** Every rank
   creates every island group, so ROCm ranks register the NVIDIA island and the
   other way round. That should only touch the store, without initializing the
   backend. Confirm nothing tries to load NCCL on AMD ranks or RCCL on NVIDIA
   ranks.

## Correctness on GPUs

4. **`"nccl"` on ROCm really runs RCCL.** Check with `NCCL_DEBUG=INFO`; RCCL
   logs its own version.
5. **Device selection on AMD nodes.** Ranks use `cuda:$LOCAL_RANK`. Confirm this
   maps to the intended GPU under `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`
   and the cluster's scheduler.
6. **Stream ordering in the bridge step.** A leader's `tensor.cpu()` must see
   the finished island all_reduce, and the island broadcast must see the
   leader's `copy_` back to the GPU. PyTorch's synchronous collectives should
   guarantee both. Check it on both vendors with large tensors and a busy GPU.
7. **float16 and other dtypes over Gloo between real hosts**, compared bit for
   bit with a single-vendor NCCL/RCCL all_reduce on integer-valued data.

## Operational

8. **Gloo networking.** Leaders on different nodes reach each other on the right
   interface; set `GLOO_SOCKET_IFNAME` on nodes with several NICs.
9. **Timeouts.** While leaders work over the bridge, the other ranks wait inside
   an island broadcast. With large tensors the bridge step can be slow enough to
   trip the NCCL/RCCL watchdog. Check whether the `timeout` passed to `init()` is
   enough or whether the watchdog needs its own setting.
10. **`barrier()` on ROCm**, which calls `torch.cuda.synchronize`.

## Not correctness, but worth measuring early

11. Bridge throughput and latency compared with a single-vendor all_reduce, to
    set a baseline before any performance work.

## Mixed build results

Run on 2026-10-06 with [`scripts/mixed_build_rendezvous.sh`](scripts/mixed_build_rendezvous.sh),
which reproduces everything below with one command. Raw results:
[`results/mixed_build/summary.md`](results/mixed_build/summary.md) (one row per run),
[`summary.json`](results/mixed_build/summary.json) (per-rank detail) and
[`envs.json`](results/mixed_build/envs.json) (builds and sizes).

### Setup

- No GPUs. A linux/amd64 Docker image (emulated on an Apple Silicon Mac) holds
  one venv per PyTorch build, installed from the official wheel indexes. Every
  venv uses Python 3.12.12.
- `GPUBRIDGE_CPU_ONLY=1` on every rank: the vendor comes from the real build
  (`torch.version.cuda` or `torch.version.hip`), islands use Gloo, tensors stay
  on CPU. `GPUBRIDGE_VENDOR` is not set.
- Each run starts two `torchrun` "nodes" on localhost, one per venv, with
  `--nnodes=2` and a c10d rendezvous. `--rdzv-conf is_host=...` chooses which
  node hosts the rendezvous store; that node also ends up as node 0, which
  hosts the store the workers use.
- Each rank runs [`scripts/rendezvous_probe.py`](scripts/rendezvous_probe.py)
  and checks, in order: rendezvous, `gpubridge.init()` (Gloo world group,
  `all_gather_object` discovery, island and bridge groups), that every rank got
  the same vendor map and that it matches each rank's own build, `all_reduce`
  SUM (exact, float32 and float16), `broadcast` from every rank, and a clean
  `destroy()` and exit.
- Negative control: with the bridge `all_reduce` removed from `collectives.py`,
  the same harness reports `FAIL (all_reduce)` with broadcast still passing,
  so the checks do catch wrong results.

| venv | torch | CUDA / HIP | Index | Download | Installed |
| --- | --- | --- | --- | --- | --- |
| cu-2.14.1 | 2.14.1+cu130 | 13.0 | `whl/cu130` | 3.0 GB | 5.6 GB |
| rocm-2.14.1 | 2.14.1+rocm7.2 | 7.2.53211 | `whl/rocm7.2` | 6.6 GB | 15.7 GB |
| cu-2.13.0 | 2.13.0+cu130 | 13.0 | `whl/cu130` | 2.7 GB | 4.8 GB |
| cu-2.9.1 | 2.9.1+cu128 | 12.8 | `whl/cu128` | 4.1 GB | 7.0 GB |
| rocm-2.9.1 | 2.9.1+rocm6.4 | 6.4.43484 | `whl/rocm6.4` | 4.8 GB | 15.4 GB |

ROCm builds are the large ones: the 2.14.1 ROCm torch wheel alone is 6.2 GB.

### Compatibility

Layouts are CUDA ranks + ROCm ranks; the build in brackets hosted the store.

| CUDA build | ROCm build | Runs | Result | Notes |
| --- | --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release |
| 2.9.1+cu128 | 2.9.1+rocm6.4 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release, older CUDA and ROCm generation |
| 2.13.0+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 1+1 (ROCm), 2+2 (CUDA) | **PASS** 3/3 | One minor release apart; `init()` warns |
| 2.14.1+cu130 | 2.9.1+rocm6.4 | 1+1 (CUDA 2.14.1) | **PASS** | Five minor releases apart, newer side hosts; warns |
| 2.9.1+cu128 | 2.14.1+rocm7.2 | 1+1 (CUDA 2.9.1) | **PASS** | Five minor releases apart, older side hosts; warns |
| 2.13.0+cu130 | (2.14.1+cu130, CUDA) | 1+1 | **PASS** | Control: one vendor, no bridge; warns |

Every run passed on its first attempt, with no timeouts and so no reruns.
Each took 8 to 9 seconds end to end, emulation included.

### What this means for the design

- **The bridge design stands.** The parts of a mixed job that both builds must
  share all run on CPU: the c10d store, Gloo, and pickled builtins in
  `all_gather_object`. All of them interoperate between CUDA and ROCm builds,
  and across releases from 2.9.1 to 2.14.1. No workaround was needed.
- **Version matching is a recommendation, not a requirement, on the CPU side.**
  `init()` keeps warning when releases differ, because the GPU side and other
  Python versions are untested.

### What this does not show

- Anything on GPUs: NCCL, RCCL, device selection, stream ordering (items 4 to 7, 10).
- Separate hosts on a real network. Gloo only ran over loopback (item 8).
- Item 3 (`new_group(backend="nccl")` on ranks outside the group). In CPU-only
  mode islands use Gloo, and on a machine without GPUs a member rank cannot
  create an NCCL group at all: it raises `ValueError: ProcessGroupNCCL is only
  supported with GPUs, no GPUs found!` (checked on 2.14.1+cu130).
- Python versions other than 3.12, releases outside 2.9.1 to 2.14.1, and
  non-x86 hosts.

### Setup problems worth knowing about

- Installing the 6.2 GB ROCm wheel straight from the index with uv failed once
  with "Failed to read from zip file". Downloading it with `curl` and checking
  the index's sha256 worked, and uv then installed the file without trouble.
  The image now does that ([`scripts/fetch_wheel.sh`](scripts/fetch_wheel.sh)).
  Whether the first download was cut off or uv mishandled the large file was
  not pinned down.
- ROCm builds of torch need `libatomic.so.1`, which slim Debian images lack
  (package `libatomic1`).
- `download-r2.pytorch.org` answers 403 to Python's default `urllib`
  User-Agent; set one when scripting against it.
