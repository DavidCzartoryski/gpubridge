# Changelog

All notable changes to gpubridge. Versions follow the release tags; every
release so far is an alpha that is **validated on CPU only**. Nothing has run
on a GPU yet: [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) lists what
still needs real hardware. Numbers in parentheses are pull requests.

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

[0.4.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.4.0-alpha
[0.3.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.3.0-alpha
[0.2.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.1.0-alpha
