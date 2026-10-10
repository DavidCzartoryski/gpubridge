# What still needs real-hardware validation

The library itself is tested in CPU simulation mode. Simulation runs the same
orchestration code, but it cannot answer the questions here, which depend on
real CUDA and ROCm builds of PyTorch running side by side. Items 1 and 2 are now
answered for the CPU side by running real builds without GPUs; see
[Mixed build results](#mixed-build-results) below.

**On GPUs:** [docs/GPU_VALIDATION_RUNBOOK.md](docs/GPU_VALIDATION_RUNBOOK.md)
maps every item below to a scripted step: Explorer (Slurm) for NVIDIA, a rented
AMD machine, and a real mixed job between two cloud machines. If Explorer's
`sharing` partition turns out to have AMD GPUs, the mixed job can instead run
inside Explorer as one heterogeneous Slurm job.

## Blocking: can the job start at all?

1. **CUDA-PyTorch and ROCm-PyTorch processes rendezvous in one job.** Both
   builds must connect to one c10d store (TCPStore through `torchrun` or
   `tcp://`) and form one Gloo default group. This is expected to work because
   the store and Gloo run on CPU and are compiled into both builds, but nobody
   has checked it. Test it with a launcher that starts one `torchrun` per node,
   each with its own PyTorch build.

   **Status: works on CPU.** CUDA-build and ROCm-build processes rendezvoused
   through torchrun's c10d backend and completed discovery, all_reduce and
   broadcast in every run, with either build hosting the store. Rechecked on
   `main` after discovery started sending each rank's full probe record; see
   [Recheck on current main](#recheck-on-current-main-nested-probe-record).
   Still open: separate hosts on a real network, and the same with GPUs attached.
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

## Added with later features

Each item below came with a feature that is tested on CPU but makes a claim
only real GPUs or NICs can check. Each names the kit step that checks it.

12. **A rank whose `LOCAL_RANK` has no GPU fails `init()` on every rank.**
    `resolve_config` compares `LOCAL_RANK` with `torch.cuda.device_count()` and
    discovery reports a mismatch as that rank's problem; `set_device` now waits
    until discovery succeeds.
    *Status: works on CPU, with a faked device count
    (`test_an_out_of_range_local_rank_fails_init_on_every_rank`).* Still open:
    that the count PyTorch reports honours `CUDA_VISIBLE_DEVICES`,
    `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` and Slurm's GPU binding, so
    the check fires on real nodes. *Step:* `check.py --expect-init-error`,
    launched with one more process than the node has GPUs (Explorer step 01,
    `amd/run_all.sh`).
13. **The island step of reduce-bridge-broadcast is a reduce onto the leader**,
    not an all_reduce: only the leader needs the island sum, and a ring reduce
    moves about half the bytes.
    *Status: works on CPU: every layout still matches `flat-gloo` exactly, and
    a test counts one island reduce and one island broadcast per all_reduce.*
    Still open: the time saved on NCCL and RCCL, by tensor size. Small tensors
    may see none, because both ops are latency-bound there. *Step:*
    `bench_all_reduce.py --ops native,island,gpubridge --phases` puts the
    `island-reduce` phase next to a native island all_reduce (`island:<vendor>`)
    of the same size (Explorer steps 04, 05 and 07, `amd/run_all.sh`).
14. **Where NCCL/RCCL and Gloo agree on reductions.** gpubridge mixes the island
    backend and Gloo in one reduction, so a result is only well defined where
    both agree. PRODUCT is left out until they are shown to agree in every
    supported dtype, including 8-bit overflow. MAX and MIN with NaN in the data
    follow whatever the backends do.
    *Status: not verifiable on CPU, where islands are Gloo too.* *Step:*
    `check.py` stage `reduction_agreement` (informational) compares a raw
    island-backend all_reduce with a Gloo one: PRODUCT in every dtype, MAX/MIN
    with a NaN on one rank. Read `product_agree` and `nan_max_min` in the rank
    records.
15. **MAX, MIN, AVG, `all_gather_into_tensor` and `reduce_scatter_tensor` on
    NCCL/RCCL islands.** Native-only uses NCCL's own all_gather_into_tensor
    and reduce_scatter_tensor; reduce-bridge-broadcast uses island gather and
    scatter, which torch builds from NCCL send/recv.
    *Status: works on CPU, bit for bit against `flat-gloo` in every layout and
    policy (`tests/test_ops.py`).* Still open: the same on GPUs. *Step:*
    `check.py` stages `reductions`, `all_gather` and `reduce_scatter`.
16. **Async collectives keep their stream semantics.** The worker's stream
    waits on an event recorded on the caller's stream; `wait()` makes the
    caller's stream wait on an event recorded after the collective.
    *Status: ordering, results and failure handling work on CPU
    (`tests/test_ops.py`), where there are no streams.* Still open: the
    events and `record_stream` on real GPUs. *Step:* `check.py` stage `async`
    queues the all_reduce right behind GPU work and checks the result on the
    caller's stream after `wait()`, with no host sync in between.
17. **The pipelined bridge is correct on GPUs.** `pipelined-reduce-bridge-broadcast`
    (opt-in) copies chunks through pinned host memory on two side streams,
    ordered by CUDA events, waiting on the host only for the chunk the bridge
    needs next.
    *Status: works on CPU, bit for bit against `flat-gloo`, including with a
    staging layer that runs each copy only when something waits for its event,
    so a missing wait corrupts the result (`tests/test_pipelined.py` shows it
    does).* Still open: the real events, pinned buffers and `record_stream` on
    GPUs. *Step:* `check.py --policy pipelined-reduce-bridge-broadcast` (its
    `busy_gpu` stage queues work ahead of the collective), in Explorer steps
    04, 05 and 07 and `amd/run_all.sh` (`OPT_IN_POLICIES`).
18. **The overlap pays off.** For large tensors the pipelined bridge should
    beat reduce-bridge-broadcast without hurting small ones; the best chunk
    size depends on the machine.
    *Status: not measurable on CPU.* *Step:* `bench_all_reduce.py
    --candidates` (or `--write-thresholds`), rerun with a few
    `GPUBRIDGE_CHUNK_BYTES` values.
19. **Thresholds measured on GPUs drive `auto-tuned`.**
    *Status: on CPU, a file written by `--write-thresholds` loads, and
    `auto-tuned` routes each size to the rule's policy.* Still open: the
    thresholds themselves, per cluster. *Step:* the `bench_candidates` and
    `check_auto_tuned` steps write `thresholds.json` next to the results and
    rerun the check under `auto-tuned` with it.
20. **The sharded bridge is correct on GPUs.** `sharded-bridge` (opt-in) runs
    NCCL/RCCL `reduce_scatter_tensor` and `all_gather_into_tensor` on islands
    of exactly k ranks, k NCCL/RCCL reduces and broadcasts on larger ones, and
    k Gloo bridge groups at once, one per island rank below k.
    *Status: works on CPU, bit for bit against `flat-gloo` for every
    supported dtype and SUM/AVG/MAX/MIN, in 2+2, 3+1, 3+2, 2+4, 3+3,
    interleaved and three-island layouts, with sizes that need padding and
    sizes smaller than k (`tests/test_sharded.py`).* Still open: the native
    collectives on device tensors, and k bridge groups on separate hosts.
    *Step:* `check.py --policy sharded-bridge` in Explorer steps 04, 05 and 07
    and `amd/run_all.sh` (`OPT_IN_POLICIES`).
21. **More bridge links mean more cross-vendor bandwidth.** Each link carries
    1/k of the bytes, and each of the k ranks copies its own segment to the
    host, so with the bridge as the bottleneck, k links should beat one.
    *Status: not shown on CPU.* In CPU simulation (8 ranks on one 10-core
    machine), `sharded-bridge` is 1.3x to 1.8x slower than
    reduce-bridge-broadcast. Every rank shares one CPU and memory, so the
    bridge phase barely changes with 4 links (10.8 ms against 10.9 ms at
    64 MB). The island steps are Gloo there, and on CPU islands the
    reduce_scatter is an all_reduce plus a slice, which makes them slower.
    GPU islands use NCCL/RCCL for those steps.
    *Step:* `bench_candidates` in step 05, where the bridge crosses the
    network, and in step 07: compare the `policy:sharded-bridge` and
    `policy:reduce-bridge-broadcast` columns. Step 05's
    `bench_phases_sharded-bridge` times the bridge phase alone, next to the
    `bench_split_by_node` phases of reduce-bridge-broadcast.

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

### Recheck on current main (nested probe record)

Run on 2026-10-06 against `main` at `3a815c0` (after #3). Discovery there sends
each rank's full `probe()` record nested inside its `PeerInfo`, which the
earlier runs above did not. Same setup, 2.14.1 pair only, with the wheels
installed offline from the host cache:

```
GPUBRIDGE_ENVS=2.14.1 scripts/mixed_build_rendezvous.sh --group 2.14.1 \
    --results /gpubridge/results/mixed_build_recheck
```

Raw results:
[`results/mixed_build_recheck/summary.md`](results/mixed_build_recheck/summary.md)
and [`summary.json`](results/mixed_build_recheck/summary.json).

| CUDA build | ROCm build | Layout (store host) | Result |
| --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 2+2 (ROCm) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 3+1 (CUDA) | **PASS** |

Every check passed (rendezvous, init, vendor map, all_reduce, broadcast,
shutdown), each run on its first attempt, in 8 seconds.

**The nested record crosses builds intact.** Every rank received every peer's
probe record, and all ranks held identical tables. A CUDA rank saw the ROCm
rank as:

```
build_vendor="amd", torch_version="2.14.1+rocm7.2", cuda_version=None,
hip_version="7.2.53211", gpu_count=0, nccl_available=True,
gloo_available=True, errors=()
```

The ROCm ranks saw the mirror image (`build_vendor="nvidia"`,
`cuda_version="13.0"`). No rank reported problems.

`nccl_available` is True on both builds even without GPUs, because it reports
whether the backend is compiled into the build. `gpu_count` is what shows the
missing hardware.

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
  Whether the first download was cut off or uv mishandled the large file was
  not pinned down. ROCm wheels are now always fetched that way
  ([`scripts/fetch_wheel.sh`](scripts/fetch_wheel.sh)).
- The wheels first lived in a BuildKit cache mount, but Docker Desktop's
  BuildKit garbage collection caps cache mounts at about 2.76 GiB, so the 6+ GB
  ROCm wheels were evicted. They now live in a permanent cache on the host
  (`~/.cache/gpubridge/wheels`, filled by
  [`scripts/fetch_wheels.sh`](scripts/fetch_wheels.sh)), and the image
  installs offline from it.
- ROCm builds of torch need `libatomic.so.1`, which slim Debian images lack
  (package `libatomic1`).
- Recent macOS ships a BSD `sha256sum` that has no `--status` flag. An early
  version of `fetch_wheel.sh` used it, mistook the usage error for a checksum
  mismatch, and deleted a complete 6.2 GB wheel. The script now compares
  digests as text, preferring `shasum`.
- `download-r2.pytorch.org` answers 403 to Python's default `urllib`
  User-Agent; set one when scripting against it.
