# Changelog

All notable changes to gpubridge. Versions follow the release tags; every
release so far is an alpha. Numbers in parentheses are pull requests.

**GPU runs so far**, on Explorer:

- **NVIDIA V100 (sm_70), commit af51b20**, torch 2.14.1+cu126, NCCL 2.29.3: the
  bridge on device tensors and the shared-GPU check passed; two fp8
  error-feedback residual tests failed, from FMA contraction plus a codec
  overflow, fixed in #50. Step 01's probe failed: it took the build target
  from `gcnArchName`, which holds the product name on CUDA builds, fixed in
  #49.
- **NVIDIA V100 (sm_70), commit 14ed60c**, step 01 rerun after #49: driver
  545.23.08, torch 2.14.1+cu126, NCCL 2.29.3. The probe passed (an exact
  target match), and so did the job's other 2 steps.
- **NVIDIA H200 (sm_90), commit 4a478f4** (#50's branch before it merged):
  driver 570.86.15, torch 2.14.1+cu126, CUDA 12.6, NCCL 2.29.3. Step 02
  passed; `gpu_tests` 26 passed, 8 skipped, in 162 s.
- **NVIDIA V100 (SXM2), commit df80abb**, step 02 rerun after #50: passed;
  `gpu_tests` 26 passed, 8 skipped, in 782 s.
- **NVIDIA A100-SXM4-80GB (sm_80), commit df80abb** with one local change
  (`AMD_NODES_CONFIRMED=1` in `submit.sh`, no library code; the records say
  df80abb+dirty): driver 570.86.15, torch 2.14.1+cu126, NCCL 2.29.3. Step 02
  passed; `gpu_tests` 26 passed, 8 skipped, in 437 s.
- **NVIDIA Tesla T4 (sm_75), commit df80abb** with the same local line:
  driver 570.86.15. Step 02 passed; `gpu_tests` 26 passed, 8 skipped, in
  205 s.
- **NVIDIA L40S (sm_89), commit df80abb** with the same local line, `sharing`
  partition: driver 545.23.08. Step 02 passed; `gpu_tests` 26 passed, 8
  skipped, in 186 s. The probe found the build covers it with sm_86 cubins
  (`target_match` "compatible").
- **AMD MI50 (gfx906), commit df80abb** (from the clone's history; the
  run's records were overwritten), `sharing` partition, torch
  2.14.1+rocm7.2, HIP 7.2.53211, RCCL 2.27.7: `setup-rocm` succeeded, and the
  probe found gfx906 in the build's arch list (an exact match). The
  single-node AMD step: `gpu_tests` 24 passed, 2 failed, 8 skipped;
  `check_1gpu` and `check_split_shared_gpu` aborted (exit -6); the
  compression kernels matched the CPU reference in 3 of 3 cases. Every
  failure was the wheel's rocBLAS, which has no kernels for gfx906, in a
  matmul used as busy work, which #52 replaces. `diag_fp8_residual.py` found
  every fp8 payload and residual matching the CPU reference, for float32 and
  float64, with fusion on and off.
- **Compression kernels at df80abb** (`bench_compression.py`; the records
  say df80abb+dirty) on a V100, an MI50 and an H200: the kernels matched the
  CPU reference in 3 of 3 cases on each. Their speed is in
  [docs/COMPRESSION.md](docs/COMPRESSION.md). The float32 outputs differ
  between vendors only in NaN bits, where the input is NaN: `0x7fffffff` on
  the NVIDIA GPUs, `0x7fc00000` on the MI50, as in the CPU reference. The
  check compares where the NaNs are, not their bits.
- **AMD MI50 (gfx906), commit 4014234** (after #52), `sharing` partition:
  driver 5.14.0-611.5.1.el9_7 (as `rocm-smi` reports it), torch
  2.14.1+rocm7.2, HIP 7.2.53211, RCCL 2.27.7. The single-node AMD step passed
  all 9 steps, in each of two runs: `gpu_tests` 26 passed, 8 skipped (in
  170 s and 178 s); `check_1gpu` and `check_split_shared_gpu` passed; the
  compression kernels matched the CPU reference in 3 of 3 cases. The wheel's
  rocBLAS still has no gfx906 kernels (the probe's `blas` stage says so);
  nothing in the checks needs it now.
- **Stream mutations at 4014234** (`stream_mutations.py`, #52), on a V100
  (sm_70; torch 2.14.1+cu126, NCCL 2.29.3) and on the MI50 (gfx906; torch
  2.14.1+rocm7.2, RCCL 2.27.7), with the same results on both:
  `worker-ignores-caller`, `staging-ignores-caller` and
  `slots-forget-copies-back` were caught (the guarding test failed on wrong
  values), and both guarding tests passed without a mutation.
  `wait-skips-stream` was **not caught**: its test passed while `Work.wait()`
  ordered no stream, so no GPU test yet shows that this ordering is needed.
  The script doesn't record the driver.
- **Triton's kernel cache at 4014234** (`time_triton_cache.py`, #52), on a
  V100, torch 2.14.1+cu126, Triton 3.8.0, the median of 12 kernel variants:
  a build took 4.14 s with the cache in the home directory (NFS), 0.32 s in
  project space (NFS) and 0.13 s in node-local `/tmp` (XFS); a load from the
  cache took 0.04 s, 0.009 s and 0.001 s. The home directory was timed first,
  and each location's first build was its slowest (9.6 s, 2.7 s and 2.3 s).
- **The first mixed-vendor job: an NVIDIA V100 (sm_70) and an AMD MI50
  (gfx906), commit 4014234**, step 07: one heterogeneous Slurm job across the
  `gpu` and `sharing` partitions, one GPU and one rank each. V100: driver
  545.23.08, torch 2.14.1+cu126, NCCL 2.29.3. MI50: driver
  5.14.0-611.5.1.el9_7, torch 2.14.1+rocm7.2, RCCL 2.27.7. The Gloo bridge
  ran over TCP between the two nodes; in both runs the preflight measured
  about 1.15 GB/s each way and a 0.05 to 0.07 ms round trip. The second run
  (`TIME_07=01:00:00`) passed 15 of 20 steps:
  - the bridge check on device tensors passed every stage on both ranks,
    with the default policy, with pinned staging, and with the pipelined and
    sharded policies;
  - a 1 GB all_reduce took 1.69 s through the bridge (0.64 GB/s bus
    bandwidth) and 1.28 s with pinned staging (0.84 GB/s), against 1.73 s
    for one flat Gloo group of both ranks;
  - bridge compression: `bf16` was faster than uncompressed from 1 MB up and
    `fp8` from 256 KB up, with every rank holding the same bits; the kernels
    matched the CPU reference on both GPUs;
  - the 5 training steps failed: DDP with `gpubridge.ddp_comm_hook`, FSDP2
    and HSDP through the backend, and the bit-for-bit checks of the last
    two. In each, the MI50's rank aborted (exit -6) in rocBLAS, which has no
    gfx906 kernels in this wheel, and the V100's rank then lost its
    connection.
  - The first run, at the same commit with a 30-minute limit, passed 10
    steps. In 4, Slurm couldn't create the job step (`srun: error: Unable to
    create step ... Unexpected message received`) while it still listed the
    previous step as running, and `summarize_pinned` then had no records to
    summarize; all 5 passed in the second run. `train_ddp` failed
    differently: the MI50's rank timed out (Gloo, 300 s) in DDP's parameter
    check while the V100's printed nothing, cause unknown. The job reached
    its limit during `train_fsdp2_backend`.

Everything else is **validated on CPU only**:
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) lists what still needs real
hardware.

## [Unreleased]

### Added

- A Kaggle runner for the GPU validation kit, for a free notebook with two
  T4s (`scripts/gpu/kaggle/`): a notebook and the same steps as one script,
  each step under a hard time limit. It checks what compute capability 7.5
  can run (CUDA events, bfloat16, Triton) and skips or adapts the rest with
  the reason. The results are the kit's usual records and validation tables,
  in one tarball (#43).
- `scripts/bench_per_call.py`: the per-call time of small all_reduces through
  each policy, sync and async, with and without an observer, against the same
  torch.distributed calls made directly. `--stub-comm` leaves only gpubridge's
  own work; `--profile` runs cProfile on rank 0 (#44).
- `scripts/gpu/triage.py`: a short plain-text summary of an Explorer or Kaggle
  results directory or tarball, under 80 lines, to paste into a review. It shows
  each failed, timed-out or skipped step and why, the GPUs and versions, the
  key benchmark numbers and the compression check (#45).
- `docs/EXPLORER_FIRST_SESSION.md`: a first Explorer session in order, from a
  fresh login: the clone in the project space, `local.env`, step 00, the setup
  job, the first rungs of the ladder, how to follow the jobs, where results
  land, what to do when a step fails and what to paste back (#46).
- `scripts/gpu/diag_fp8_residual.py`: reruns the fp8 error-feedback case of
  the GPU kernel test on one device and prints, for each value whose residual,
  code or scale differs from the CPU reference, every intermediate value with
  its bits and whether it is subnormal in float32. It also runs the kernel
  with multiply-add contraction on (gpubridge's own launches have it off since
  #50), a probe of what the device computes, and counts the compiled code's
  fused multiply-adds and flush-to-zero instructions (#48).
- `scripts/gpu/stream_mutations.py`: undoes each stream-ordering fix in turn
  (`GPUBRIDGE_TEST_MUTATION` in `tests/test_gpu.py`) and checks that the GPU
  test guarding it fails, and that the tests pass without one (#52).
- `scripts/gpu/time_triton_cache.py`: how long the compression kernels take
  to build and to load from Triton's cache, by cache location (#52).
- `probe.py` records whether BLAS works (`blas_available`, with the error),
  from a matmul in a child process, without failing the probe (#52).
- `scripts/gpu/bench_compression.py --fp-fusion`: launches the compression
  kernels with multiply-add contraction on, so a run each way times what
  gpubridge's default (off) costs (#50).
- Step 08, a comparison matrix of what gpubridge costs
  (`scripts/gpu/explorer/08_bench_matrix.sbatch`, `scripts/gpu/bench_matrix.py`):
  one heterogeneous job of two 1-GPU nodes runs the all_reduce benchmark
  (1 KB to 1 GB, CUDA events, phases) for native NCCL or RCCL on the
  transport each picks and over TCP on the bridge's interface (A1, A2, B1,
  B2), gpubridge between two islands of one vendor (C), gpubridge between a
  V100 and an MI50 under every policy, chunk size and compression mode (D),
  and one flat Gloo group (E). `matrix.md` has a table per size range with
  each run as a multiple of the two native baselines (marked where the
  baseline ran in another job, on other nodes) and the bridge phase against
  the preflight's prediction, each job's preflight, each node's InfiniBand
  ports, and the transport NCCL and RCCL chose. `submit.sh 08` submits the
  V100 + MI50 job;
  `08-nvidia-pair` and `08-amd-pair`, two GPUs of one type in one partition,
  are refused until `SAME_PARTITION_PAIRS_OK=1`.
- `submit.sh net-nvidia` and `net-amd` (`net_check.sbatch`,
  `scripts/gpu/net_report.sh`): one node's InfiniBand devices and ports, its
  interfaces, and the network NCCL or RCCL initializes there, as chosen and
  with `NCCL_IB_DISABLE=1`.
- `scripts/gpu/train_bench.py`: training throughput through DDP and
  `gpubridge.ddp_comm_hook` on GPT-2 small and ResNet-50 (built from
  `torch.nn`, synthetic data), with each step split into compute, waiting
  for the slower GPU and exposed communication. Step 07 runs it across both
  vendors (`STEPS_07=training` runs only its training steps), and
  `submit.sh train-nvidia` or `train-amd` on one GPU alone.

### Changed

- A rank's island and whether it leads, and a policy's supported reduce ops,
  are worked out once instead of on every collective. Every call still checks
  its op. On CPU, gpubridge's own work per small reduce-bridge-broadcast call
  went from 4.0 to 3.4 µs; the README has the per-call numbers (#44).
- For policy authors: `CollectivePolicy.supported_reduce_ops()` is asked once
  per policy instance and topology, at the first collective, and its answer
  must not change after that (#44).
- The README opens with a pitch, badges, a one-line status, an architecture
  diagram, a quickstart and a feature table with each feature's status. New
  diagrams for all_reduce and HSDP (docs/img, light and dark). New
  CONTRIBUTING.md, SECURITY.md and issue templates (#42).
- The GPU kit says why a step failed where you look first: the job's own log
  has the step's last 20 lines (before torchrun's failure report) after its
  `FAILED` line, and a failed `probe.py` ends with one line per failed check,
  naming it and its values (#49).
- `setup_env.sh`: the venv's Python is `VENV_PYTHON` (default 3.12). It was
  `PYTHON_VERSION`, which Python's Docker images set to their own full
  version (#49).
- `results/explorer/` is gitignored: its step logs, stack dumps and NCCL logs
  are not redacted (#50).
- Explorer jobs build Triton's kernels in a node-local cache, filled from
  `$WORK_DIR/triton-cache` at the start and saved back at the end, instead of
  Triton's default `~/.triton/cache` in the home directory
  (`TRITON_CACHE_STORE`; a `TRITON_CACHE_DIR` you set wins) (#52).
- The GPU kit's `gpu_tests` step (Explorer 02 and 04, `amd/run_all.sh`, the
  Kaggle runner) runs pytest with `--durations=15`, so the slowest tests and
  their times end its log (#51).
- `AMD_NODES_CONFIRMED` is a setting, read from `explorer/local.env` or the
  environment, instead of a line to edit in `submit.sh`, which left runs
  git_dirty (#51).
- `submit.sh` warns before it submits a job asking the `gpu` partition for
  more than one GPU: that partition's QOS allows one per job in total, so
  such a job sits PENDING with `QOSMaxGRESPerJob`. EXPLORER_FIRST_SESSION.md
  describes that limit and multigpu's access request (#51).

### Fixed

- The stream-ordering checks (`check.py`'s `busy_gpu` and `async` stages,
  `tests/test_gpu.py`) queued matmuls as busy work, and failed or aborted
  where rocBLAS has no kernels for the GPU (gfx906 on the ROCm 7.2 wheel).
  They queue a calibrated spin kernel instead (`gpukit.queue_gpu_delay`); no
  BLAS anywhere in the checks. `check.py --busy-iters` is now
  `--busy-seconds` (#52).
- The AMD driver was recorded as `rocm-smi`'s whole CSV output (`name,
  value "Driver version", "5.14.0-..."`). Records hold the version now, and
  triage reads older ones the same way (#52).
- ROCm runs printed `(null): No such file or directory` once per GPU per
  process: the ROCm wheel's bundled libdrm couldn't find its GPU-name table.
  The kit now points `AMDGPU_ASIC_ID_TABLE_PATHS` at the copy the wheel ships
  (#51).
- Bridge compression: a finite value near float32's maximum could arrive as
  inf, and leave a residual of -inf for the next step. In `fp8` a block above
  448 · 2^119 got the scale 2^120, so a code could decode to 2^128; in `bf16`
  a value from about 3.3961e38 (halfway past bfloat16's largest) rounded to
  inf. Both now saturate: `fp8` caps the scale exponent at 119 and its codes
  at 448, `bf16` its codes at bfloat16's largest, and a finite float64 beyond
  float32's range, or a value its residual pushes past it, counts as
  float32's largest. Every finite value is now sent as a finite float32 and
  leaves a finite residual. Payloads change only for blocks holding such
  values (#50).
- The Triton compression kernels are compiled without multiply-add
  contraction (`enable_fp_fusion=False`), so they round every product as the
  PyTorch reference does, whatever the compiler. On the V100, a fused
  `v - decoded * scale` gave a finite residual where the reference's was
  -inf (#50).
- `probe.py` took a CUDA GPU's build target from `gcnArchName`, which CUDA
  builds of PyTorch fill with the product name, so its build check could fail
  for a GPU the build covers. The target is `gcnArchName` without its feature
  flags on ROCm, and `sm_XY` from the compute capability on CUDA. A GPU counts
  as covered by its own target, a cubin for an earlier minor of its major, or
  PTX at or below its capability; `probe.json` records which in
  `target_match` (#49).
- `submit.sh setup` printed `<job id>` in its log's path. Every submission now
  prints its job id and log (#49).
- Step 01's probe had no limit of its own, and the kit's `STEP_TIMEOUT` (20
  minutes) was longer than the job's 10-minute Slurm limit. `submit.sh` now
  caps each job's `STEP_TIMEOUT` at its Slurm limit less `STEP_MARGIN` (120 s),
  and the probe runs under it (#49).
- `triage.py` couldn't see a job that stopped before or during its first step.
  Explorer steps 01 to 07 now write `job.jsonl` (their start, an error that
  stopped them, their end and exit code), and triage lists such a job as
  unfinished, with its error or exit code (#49).

## [0.4.0-alpha] - 2026-10-13

CPU-validated. New in this alpha: gpubridge as an experimental
`torch.distributed` backend (DDP, FSDP2 and HSDP through subgroups), opt-in
bridge compression and an opt-in UCX bridge transport. None of it has run on a
GPU or on RDMA hardware. Not on PyPI.

### Added

- Subgroups on the `torch.distributed` backend, **experimental, tested in
  simulation only**: `new_group` and multi-dimensional `DeviceMesh` groups get a
  topology of their own over their members, built at their first collective
  (native-only for one vendor, bridged for mixed members). HSDP with shard
  groups set to the vendor islands keeps all-gathers and reduce-scatters inside
  each vendor and sends one replicate all_reduce per step over the bridge.
  `examples/fsdp2_torch_backend.py --hsdp ISLANDS`. Design:
  docs/TORCH_BACKEND.md, "Subgroups and HSDP". Subgroup setup patches torch's
  private `distributed_c10d._process_group_name`. On a torch where that
  function or `dist.new_group`'s use of it has changed, creating a subgroup
  fails with an error naming the torch version, so the job doesn't risk a
  hang (#39, #40).

- gpubridge as a `torch.distributed` backend,
  `dist.init_process_group(backend="gpubridge")`, **experimental and not
  validated on GPUs.** DDP and FSDP2 scripts run their collectives through
  gpubridge across vendor islands with the backend name and `import
  gpubridge`, which registers it. On torch 2.14 and later, a lazy
  `torch.distributed.backends` entry point registers it without the import;
  nothing hooks `import torch`. gpubridge's groups are built at the first
  collective, with the usual discovery checks. `GPUBRIDGE_POLICY` and
  `GPUBRIDGE_BRIDGE` pick the policy and transport. CPU tensors in GPU jobs go
  to a Gloo group. Point-to-point, all-to-all, reduce,
  GPU-tensor gather and scatter, and other reductions are refused before any
  communication. New examples `ddp_torch_backend.py` and
  `fsdp2_torch_backend.py` (with a `--verify` mode); the GPU kit's
  `train_fsdp2_backend` and `verify_fsdp2_backend` steps. Design:
  docs/TORCH_BACKEND.md (#38).

- Opt-in bridge compression, `GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8`, **off by
  default, changes results (no longer bit-exact against the uncompressed sum)
  and not validated on GPUs.** The bridged policies (default, pipelined,
  sharded) compress the bridge step of floating-point SUM and AVG, over Gloo
  or UCX. Island steps and every other op and dtype stay exact. Leaders
  quantize on the GPU with Triton kernels (one source for CUDA and ROCm;
  PyTorch fallback without Triton), exchange the payloads, then dequantize and
  add them in float32. `fp8` is OCP E4M3 with a power-of-two scale per 256
  values, travelling as uint8 bytes plus a float32 scale tensor. Every rank
  still ends with the same bits, whichever GPU ran the kernels. Error feedback
  per buffer carries each step's rounding error into the next. New phases
  `compress` and `decompress` keep kernel time out of `bridge`. The GPU kit's
  `bench_compression.py` times the kernels against memory bandwidth, and
  `bench_all_reduce.py --compression` compares bridge time with and without
  compression. Design and error bounds: docs/COMPRESSION.md (#37).

- An opt-in UCX bridge transport, `init(bridge="ucx")`, **not validated on
  GPUs or RDMA hardware** (tested on CPU over UCX's TCP transport only); the
  Gloo bridge stays the default. It carries the leaders' bridge over
  InfiniBand or RoCE where UCX finds them. Leaders hand GPU tensors to UCX
  directly where its build has their memory type, and flush the NIC's writes
  into the GPU after every receive, since UCX doesn't. Elsewhere, GPU tensors
  are staged through pinned host memory. It is a small C++ library over UCX,
  built on first use (`python -m gpubridge.ucx build`). Design and open
  questions: docs/RDMA_TRANSPORT.md (#35).

### Fixed

- The two-node job tests (Explorer steps 05 and 07) no longer hang on machines
  whose own name doesn't resolve, such as macOS, where `socket.getfqdn()` can
  return the reverse name of `::1`. The test harness's fake `srun` gives each
  pretend node torchrun's `--local-addr=127.0.0.1`; the job scripts are
  unchanged (#36).

## [0.3.0-alpha] - 2026-10-12

CPU-validated, with the GPU validation kit ready for a first GPU run. Not on
PyPI.

### Breaking

- Requires PyTorch 2.8 or later. Tearing down a Gloo group could deadlock on
  older releases; the fix landed upstream in 2.8 (#22).
- `init()` refuses a layout that puts two ranks of one island on the same GPU,
  on every rank and with the same message, since NCCL and RCCL refuse it when
  the island first communicates. A rank whose GPU reports neither a UUID nor a
  PCI address leaves its island unchecked, with a warning from rank 0 (#25,
  #29).
- `Work.is_completed()` turns true only once the collective's GPU work is
  done, as with NCCL, instead of as soon as the worker thread had queued it
  (#24).

### Added

- `gpubridge.in_flight()`: the collective each thread is inside, with its
  phase, for debugging hangs (#21).
- `GPUBRIDGE_PINNED_STAGING=1` (opt-in, off by default, **not validated on
  GPUs**): the default policy's leaders stage the bridge step through one
  pinned host buffer instead of pageable `.cpu()` copies, ordered with CUDA
  events. Every rank must agree (#32).
- GPU-only correctness tests, `pytest -m gpu tests/test_gpu.py`: every policy
  against `flat-gloo` bit for bit, every op and dtype, stream ordering,
  pinned buffer reuse, `is_completed()` (#24, #32).
- A release workflow: every pull request and version tag builds the sdist and
  wheel and checks them with twine, and a tag must match the package version.
  Publishing to PyPI (trusted publishing) stays off until enabled (#34).

### Fixed

- The pipelined policy could reuse a pinned host slot in the next collective
  before the previous collective's copy back out of it had run. It now keeps
  each slot's last copy-back event across collectives (#24).

### GPU validation kit

- Every job script runs end to end on CPU against a fake Slurm cluster, with
  hard time limits and the last NCCL/RCCL log lines on failure (#23); those
  tests run in the Python 3.12 CI job only (#26).
- A ladder of runs from one GPU to a real mixed job, a checked split test on
  one NVIDIA node, and an Explorer AMD-node job (#25).
- A network preflight before multi-node runs: every routed pair of interfaces
  tested both ways, raw TCP bandwidth (one stream, a lower bound), and a
  `GLOO_SOCKET_IFNAME` chosen per node (#27, #29).
- `bench_all_reduce.py` times with CUDA events and puts native NCCL/RCCL,
  gpubridge and `flat-gloo` side by side; every row names its baseline (#28,
  #29).
- Step 02's deliberate failing run passes only on the duplicate-GPU error,
  within its own time limit (#29).
- Every record carries the versions, driver, GPU model, commit and interfaces
  of the run, with an allowlist of environment variables and no paths or
  user names; `validation_tables.py` builds the HARDWARE_VALIDATION.md tables,
  safe to publish, including predicted against measured bridge time (#30,
  #31).
- `check.py`'s `in_flight` stage orders its ranks through the store instead of
  sleeping (#31).
- `check_pinned` and `bench_pinned` compare pinned staging with pageable
  copies in the same allocation (#32).

## [0.2.0-alpha] - 2026-10-10

CPU-validated, opt-in policies not validated on GPUs.

### Added

- `ReduceOp.AVG`, `MAX` and `MIN` for `all_reduce`, plus
  `all_gather_into_tensor`, `reduce_scatter_tensor` and `async_op=True` on
  every collective (#15).
- `gpubridge.ddp_comm_hook` for `DistributedDataParallel` (comm hook only;
  **not validated on GPUs**) (#19).
- Opt-in collective policies, never chosen by `auto` (**not validated on
  GPUs**): `pipelined-reduce-bridge-broadcast` (#17), `sharded-bridge` (#18)
  and `auto-tuned` (#17).
- `BridgeTransport.create_groups` for policies with several bridge links (#18).
- Collective observers, `gpubridge.observe`: a record of every collective and
  the phases its policy marks (#9).
- GPU validation kit for Slurm clusters: setup on compute nodes, a
  `PROJECT_DIR` setting, a configurable ROCm build, per-island NCCL/RCCL
  baselines and a mixed heterogeneous-job path (#11, #12).
- Docs: `docs/ROADMAP.md` (bridge-only compression, an RDMA bridge transport,
  a `torch.distributed` backend for FSDP2; design only) and the 2026
  mixed-vendor systems in `docs/PRIOR_ART.md` (#10, #16, #20).

### Changed

- The default policy's island step is a reduce onto the island leader instead
  of an island all_reduce (#14).

### Fixed

- `init()` fails on every rank, with the same message, when `LOCAL_RANK` names
  a GPU the process can't see, instead of hanging (#13).

## [0.1.0-alpha] - 2026-10-07

CPU-validated, GPU validation pending.

### Added

- `gpubridge.init()`, `all_reduce` (SUM), `broadcast` and `barrier` over one
  job that mixes NVIDIA and AMD ranks: an NCCL or RCCL island per vendor,
  island leaders joined by a CPU Gloo bridge.
- Collective policies `auto`, `reduce-bridge-broadcast`, `native-only` and
  `flat-gloo`, with hooks for size-based selection, and an up-front dtype
  check (#7).
- A swappable bridge transport, Gloo by default; discovery shares every rank's
  probe results, so a bad setup fails `init()` on every rank (#3).
- Simulation mode (`GPUBRIDGE_VENDOR`), CPU-only mode, and a test-only
  split-test mode flagged in every log and result (#5).
- A GPU validation kit in `scripts/gpu` with a runbook (#5).
- A CPU-only matrix showing CUDA and ROCm builds of PyTorch can share one job
  (#1, #4), and prior-art notes (#2, #6).

[Unreleased]: https://github.com/DavidCzartoryski/gpubridge/compare/v0.4.0-alpha...main
[0.4.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.4.0-alpha
[0.3.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.3.0-alpha
[0.2.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.1.0-alpha
