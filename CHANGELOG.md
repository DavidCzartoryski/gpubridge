# Changelog

All notable changes to gpubridge. Versions follow the release tags; every
release so far is an alpha that is **validated on CPU only**. Nothing has run
on a GPU yet: [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) lists what
still needs real hardware. Numbers in parentheses are pull requests.

## [Unreleased]

### Added

- An opt-in UCX bridge transport, `init(bridge="ucx")`, **not validated on
  GPUs or RDMA hardware** (tested on CPU over UCX's TCP transport only); the
  Gloo bridge stays the default. It carries the leaders' bridge over
  InfiniBand or RoCE where UCX finds them. Leaders hand GPU tensors to UCX
  directly where its build has their memory type, and flush the NIC's writes
  into the GPU after every receive, since UCX doesn't. Elsewhere, GPU tensors
  are staged through pinned host memory. It is a small C++ library over UCX,
  built on first use (`python -m gpubridge.ucx build`). Design and open
  questions: docs/RDMA_TRANSPORT.md (#35).

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

[Unreleased]: https://github.com/DavidCzartoryski/gpubridge/compare/v0.3.0-alpha...main
[0.3.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.3.0-alpha
[0.2.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/DavidCzartoryski/gpubridge/releases/tag/v0.1.0-alpha
