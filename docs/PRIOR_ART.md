# Prior art: Modular MAX `comm`, RAJA, Triton, Triton-distributed, torchcomms and three 2026 papers

Notes comparing gpubridge with five projects that also promise "write it once,
run it on NVIDIA or AMD", and with three 2026 papers that put both vendors in
one training job ([below](#three-2026-papers-mixed-vendor-collectives-over-rdma)).
Each repo was read at a fixed commit (on 2026-10-06, torchcomms on 2026-10-07),
and every code link points at that commit:

- Modular [`85356f6`](https://github.com/modular/modular/tree/85356f6562ed57bab8762fde38448ba5b50b69c9)
- RAJA [`09626e8`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53)
  (`develop`; RAJA has no `main`)
- Triton [`4a5147d`](https://github.com/triton-lang/triton/tree/4a5147d919defd3064388d4cdd202fb5e9403b61)
- Triton-distributed [`7908e4e`](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a)
  (`main` as of 2026-09-18)
- torchcomms [`a61f9ad`](https://github.com/meta-pytorch/torchcomms/tree/a61f9adc18384160079d533b53cc5bcd663acf85)
  (`main` as of 2026-10-07)

Statements marked *(reading)* are conclusions from reading the code, not from
running it.

| Project | Compute or communication | Vendors supported | Vendor chosen | Mixed vendors in one job |
| --- | --- | --- | --- | --- |
| Modular MAX `comm` | communication (vendor libraries or its own kernels) | NVIDIA, AMD | compile time, per target | no (its README says so) |
| RAJA | compute (loops and kernels) | CUDA, HIP, SYCL, OpenMP, CPU | compile time, by policy type | no |
| Triton | compute (kernel language and compiler) | NVIDIA (compute capability 8.0+), AMD (ROCm 6.2+) | run time, one driver per process; compiled per GPU architecture | no: exactly one active driver per process |
| Triton-distributed | communication inside kernels, overlapped with compute | NVIDIA (NVSHMEM), AMD (rocSHMEM or MORI, single node), MetaX | run time, per process (`nvidia-smi` / `rocm-smi` on PATH) | no: one SHMEM library over one NCCL group |
| torchcomms (Meta) | communication (collectives, plus one-sided windows) | NVIDIA (NCCL, NCCLX), AMD (RCCL, RCCLX), Intel XPU, CPU (Gloo) | build time (a CUDA or a ROCm build), then a backend per communicator | no: one vendor library per communicator; listed as a design goal |
| HetCCL, SNU et al. ([2601.22585](https://arxiv.org/abs/2601.22585)) | communication (orchestrates NCCL and RCCL) | NVIDIA, AMD | per node | **yes**: NCCL/RCCL inside each vendor, GPU-to-GPU RDMA between vendors |
| HetCCL, PKU/BAAI et al. ([2605.31000](https://arxiv.org/abs/2605.31000)) | communication (wraps vendor libraries) | 8 claimed; evaluated NVIDIA A800 and three unnamed vendors | per vendor group | **yes**: vendor-library collectives inside each group, RDMA between groups |
| Joint Training on AMD and NVIDIA GPUs ([2602.18007](https://arxiv.org/abs/2602.18007)) | communication for Megatron/DeepSpeed training | AMD (MI325X), NVIDIA (H200) | per node | **yes**, for pipeline-parallel traffic only |
| **gpubridge** | communication (no kernels) | NVIDIA, AMD | run time, per process (from the PyTorch build) | **yes**: one island per vendor, joined by a CPU bridge |

## Modular MAX: `comm` and `comm/vendor/ccl`

### What it does

[`max/kernels/src/comm/`](https://github.com/modular/modular/tree/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm)
is MAX's collectives package for tensor-parallel inference. Its own docstring
scopes it to "distributed inference across multiple GPUs within a node"
([`comm/__init__.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/__init__.mojo#L20-L21)).
It has two implementations behind one call shape:

- **Native Mojo kernels.** `allreduce`, `allgather`, `reducescatter`,
  `broadcast` and more, using peer-to-peer (P2P) GPU memory access, with a
  slower fallback when P2P is off
  ([`allreduce.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/allreduce.mojo#L1793-L1807)).
- **Vendor libraries.** [`comm/vendor/ccl.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo)
  binds NCCL or RCCL through `dlopen`.

**Choosing the path.** Native versus vendor is a build-time define,
`MODULAR_USE_VENDOR_CCL`, checked in the graph op
([`distributed.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/graph_compiler/builtin_kernels/distributed.mojo#L258-L262)).
Serving sets it from the `MAX_SERVE_USE_VENDOR_CCL` env var, which defaults to `"false"`
([`pipeline_runtime_config.py`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/python/max/pipelines/lib/pipeline_runtime_config.py#L293-L299)).

**Detecting the vendor and loading its library.**
- NCCL versus RCCL is decided at compile time from the default accelerator:
  `comptime if default_accelerator().is_amd_gpu()`
  ([`ccl.mojo` L114-119](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo#L114-L119)).
  The vendor README still calls this a runtime choice and names an older
  function, so it lags the code.
- The loader tries a fixed list of names and paths in order: `librccl.so`,
  `librccl.so.1`, `/opt/rocm/lib/...`, `libnccl.so`, `libnccl.so.2`,
  `/usr/lib/x86_64-linux-gnu/...`
  ([L98-111](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo#L98-L111)).
- The library handle is cached once per process. There is no env var to
  override the path.

**Availability probes.**
- `is_allreduce_available()`, `is_allgather_available()` and
  `is_broadcast_available()` resolve a symbol without calling it
  ([L452-498](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo#L452-L498)).
- Their docstrings say they return `False` when the library is absent, but
  they load it through `_find_dylib`, whose `abort_on_failure` defaults to `True`
  ([ffi](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/Mojo/stdlib/std/ffi/__init__.mojo#L947-L973)).
  **Confirmed by reproduction** on 2026-10-06:
  - A five-line Mojo program that calls `is_allreduce_available()` on a machine
    without NCCL aborts the whole process, with `ABORT: ... Failed to load NCCL
    from libnccl.so ...` and exit code 133 (SIGTRAP), and never returns.
  - It aborted on macOS 27 arm64 and Debian 13 aarch64, with the shipped
    nightly package (`max 26.7.0.dev2026100605`) and with `main`'s `ccl.mojo`
    compiled from source.
  - A patch that checks the library through the raising `_try_find_dylib`
    first returns `False` without NCCL, and still returns `True` with a stub
    `libnccl.so`. A draft issue with the patch is pending.
- The graph op does not probe; it calls the vendor path directly.

**How collectives are structured.**
- A `Communicators` struct holds one `ncclComm_t` per GPU, created in one
  process with `ncclCommInitAll` over device IDs `0..n-1`, for at most
  `MAX_GPUS = 8`
  ([L325-339](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo#L325-L339),
  [`sync.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/sync.mojo#L132)).
- Each collective is called once per device, with all ranks' buffers and a
  `DeviceContext`. The only reduction op is `ncclSum`. Supported dtypes are
  f32, bf16 and f16; anything else raises `"vendor_ccl: dtype not supported"`
  ([L306-320](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/ccl.mojo#L306-L320)).
- The native package keeps the same signatures so either path can be swapped
  in. It even ships no-op `group_start`/`group_end` "(enables vendor_ccl drop
  in replacement)"
  ([`sync.mojo`](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/sync.mojo#L36-L60)).

**Mixing NVIDIA and AMD: not supported.**
- The vendor README says "Mixed‑vendor hosts are not explicitly supported"
  ([README L73-74](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/vendor/README.md#L73-L74)).
- The design assumes one vendor throughout. One library is chosen per compile
  and cached per process. `DeviceContext.target` is a compile-time constant.
  Communicators come from a single-process `ncclCommInitAll`.
- The native allreduce carries `# TODO: check all devices have the same GPU
  sm_version`
  ([`allreduce.mojo` L1773](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/max/kernels/src/comm/allreduce.mojo#L1773-L1774)).
- GitHub code search finds no `ncclGetUniqueId` or `ncclCommInitRank`, which a
  communicator spanning several processes or hosts would need.

### Overlap with gpubridge

- Both pick NCCL or RCCL for the user, so one call works on either vendor.
- Both started with a deliberately narrow v1: SUM only, a few dtypes, and
  all_reduce, broadcast and allgather-style ops. gpubridge has since added
  AVG, MAX and MIN, `all_gather_into_tensor` and `reduce_scatter_tensor`.
- Both keep a vendor-library path and a second path behind one API: Modular's
  own kernels; gpubridge's simulation and CPU-only modes.

### Where gpubridge is different

- **Scope.** Modular `comm` is one process driving up to 8 GPUs on one node,
  for inference. gpubridge is many processes across nodes, through
  `torch.distributed`, for training.
- **The core problem.** gpubridge exists to put both vendors in one job,
  which Modular explicitly does not support. In Modular every GPU in the
  program shares one vendor. In gpubridge each process has one vendor, and
  islands of different vendors are joined by a CPU bridge.
- **Where the choice is made.** Modular chooses the vendor at compile time per
  target. gpubridge chooses at run time from the PyTorch build each process
  imports, which is why two builds can share a job.
- **Kernels.** gpubridge writes none, while kernels are much of Modular's value.

### Ideas worth adopting

1. **Probes that never crash, plus one capability report.** Modular's probes
   have the right shape but abort the process when the library is missing.
   *Status: implemented as `gpubridge.probe()` and discovery-time problem
   reports, which also cover a `LOCAL_RANK` with no visible GPU behind it.*
   - Add `detect.probe()`, which never raises. It returns this rank's build
     vendor, `dist.is_nccl_available()`, `dist.is_gloo_available()`, whether a
     GPU is visible, and the torch version.
   - Send it in `PeerInfo` during discovery (`topology.py`). Then `init()` can
     fail with one message naming every rank that lacks its island backend,
     instead of some ranks hanging in `new_group`.
   - Monitoring or profiling tools also get a per-rank capability snapshot for
     free.
2. **A bridge transport with a fixed signature.** Modular keeps the vendor and
   native paths call-compatible so a flag can switch them.
   *Status: implemented in `transport.py`, with Gloo as the default.*
   - Wrap what collectives need from the bridge (all_reduce and broadcast on
     CPU tensors) in a small `BridgeTransport` in a new `transport.py`.
     Gloo is the first implementation; `topology.bridge_group` becomes a
     transport object.
   - MPI, UCX, or an RDMA path can then replace Gloo later without touching
     the reduce-bridge-broadcast logic in `collectives.py`.
3. **Check dtype and op support up front.** Modular rejects unsupported dtypes
   by name before calling the library. Before this, gpubridge left dtype errors
   to torch, which could fail on the bridge after the island step had already
   run.
   *Status: implemented as `gpubridge.collectives.SUPPORTED_DTYPES`, checked in
   `_check_tensor`. Every dtype in it passes Gloo all_reduce and broadcast on
   torch 2.3.1, 2.6.0, 2.9.1 and 2.14.1 (including the 2.14.1 CUDA and ROCm
   builds); the NCCL/RCCL side is checked by `scripts/gpu/check.py` on GPUs.*
   - Keep one table of dtypes that both the island backend and Gloo support.
   - Check it in `collectives._check_tensor`, so every rank raises the same
     error before any communication.

## RAJA

### What it does

RAJA is LLNL's C++ performance-portability layer. It covers loops and kernels
on one node, not communication. You write the loop body once as a lambda, and
a policy type picks the backend: `RAJA::forall<Policy>(range, body)`.

**How a policy becomes a backend.**
- The template form builds the policy value and that backend's default
  resource (its stream or queue), then forwards to a value-based interface
  ([`forall.hpp` L578-591](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/pattern/forall.hpp#L578-L591)).
- That interface calls `forall_impl(resource, policy, ...)`. Each backend
  provides its own overload, chosen by the policy type: for example
  `seq_exec` with `resources::Host`
  ([sequential](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/sequential/forall.hpp#L55))
  and `hip_exec` with `resources::Hip`
  ([hip](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/hip/forall.hpp#L505)).
- Every policy also carries a tag from one enum:
  `Policy {sequential, simd, openmp, target_openmp, cuda, hip, sycl}`
  ([`PolicyBase.hpp`](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/PolicyBase.hpp#L34-L45)).
- Reducers and `launch` follow the same pattern.
- There is also an experimental `dynamic_forall`, which picks a policy at run
  time from a list
  ([L670](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/pattern/forall.hpp#L670)).

**How a backend is added.**
- It gets an enum entry, a CMake option (`RAJA_ENABLE_HIP`), and an "active"
  macro that is defined only when that compiler is in use:
  `RAJA_ENABLE_HIP && __HIPCC__`
  ([`config.hpp.in`](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/config.hpp.in#L286-L305)).
- It gets its own directory that mirrors the others. For example,
  [`policy/hip/`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/hip)
  matches [`policy/cuda/`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/cuda)
  file for file: policy, forall, reduce, scan, sort, kernel, launch, and so on.
- [`policy/sycl/`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53/include/RAJA/policy/sycl)
  implements a subset. So a backend can start small and grow.

**Tests.** They are generated once per enabled backend from shared templates,
with per-backend policy lists kept in one header
([`test/functional/forall`](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/test/functional/forall/CMakeLists.txt#L10-L29),
[`RAJA_test-forall-execpol.hpp`](https://github.com/llnl/RAJA/blob/09626e855db5005357eb1dac7cc4db0546cf0f53/test/include/RAJA_test-forall-execpol.hpp)).

### Overlap with gpubridge

- **The same promise.** Code is written once against an abstract operation,
  and a backend-specific layer underneath decides how it runs on CUDA or HIP.
- **The same separation.** What the code does is kept apart from which
  backend runs it. In gpubridge, user code calls `all_reduce(tensor)`, and the
  island backend, bridge and device come from `Config` and `Topology`.

### Where gpubridge is different

- **Compute versus communication.** RAJA decides how a loop runs on one
  device. gpubridge decides how data moves between processes.
- **Heterogeneity.** In RAJA, backends are chosen at compile time, and only
  one GPU backend is active per translation unit *(reading: the
  `RAJA_*_ACTIVE` macros depend on the compiler in use)*. So even RAJA does
  not run CUDA and HIP code in one binary. gpubridge never needs that, because
  each process already has one vendor. Its heterogeneity sits between
  processes, and the bridge is what handles it.
- **Language and timing.** RAJA is a header-only C++20 template library that
  dispatches at compile time. gpubridge is Python and decides at run time.

### Ideas worth adopting

1. **Collective policies: separate what a collective computes from how it is
   composed.** `collectives.py` used to hard-code reduce-bridge-broadcast.
   *Status: implemented in `policies.py`, with `reduce-bridge-broadcast`,
   `native-only`, `flat-gloo` and `auto` (the default, which keeps the
   earlier behaviour). The "later" variants below now exist as opt-in
   policies, `pipelined-reduce-bridge-broadcast`, `sharded-bridge` and
   `auto-tuned`, tested on CPU and not validated on GPUs. There is no
   per-call override: a mismatch between ranks there would hang with no cheap
   way to detect it.*
   - Make the composition a policy object chosen at `init()`, with an
     optional per-call override:
     - `ReduceBridgeBroadcast`: the current behaviour.
     - `NativeOnly`: single-vendor clusters, which `collectives.py` already
       special-cases.
     - `FlatGloo`: every rank on the world Gloo group. It is slow but
       obviously correct, which makes it a reference for tests and a fallback
       for debugging.
     - Later, chunked or overlapped variants.
   - The public `all_reduce(tensor)` stays the same, the way the loop body
     stays the same in RAJA. Put policies in a new `policies.py`, keyed by
     name like RAJA's `Policy` enum.
2. **A vendor registry, so a new vendor is one entry.** Vendor knowledge is
   currently spread out: `Vendor` and `GPU_BACKEND` live in `config.py`, and
   the `torch.version` checks live in `detect.py`.
   *Status: not implemented yet.*
   - Collect each vendor's details into one `VendorSpec` table in a new
     `vendors.py`. Each entry gives its name, how to detect it from the build,
     its island backend, and how to map a local rank to a device.
   - Adding a third vendor then means adding an entry, not editing three
     modules. A RAJA backend is added the same way: one mirrored directory
     plus one enum value.
   - Like SYCL in RAJA, a new vendor could start with a subset of operations.
3. **One test matrix, generated per mode.** gpubridge already runs the same
   checks in several modes: simulated vendors, faked builds under
   `GPUBRIDGE_CPU_ONLY`, real mixed builds in `scripts/`, and eventually real
   GPUs.
   *Status: not implemented yet. The GPU validation kit (`scripts/gpu/`) runs
   one check script in every mode, but the pytest suite is not yet
   parametrized over modes.*
   - List the modes once in `tests/_harness.py` and parametrize the collective
     tests over them.
   - Enable the GPU modes automatically when hardware is present, so a
     self-hosted GPU runner later needs no new tests.

## Triton

### What it does

Triton is "a language and compiler for writing highly efficient custom
Deep-Learning primitives" ([`README.md`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/README.md#L25)): kernels written in
Python, compiled through MLIR and LLVM. It is MIT-licensed
([`setup.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/setup.py#L612)). The README lists Linux, "NVIDIA GPUs (Compute
Capability 8.0+)", "AMD GPUs (ROCm 6.2+)" and CPUs as under development
([`README.md` L292-302](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/README.md#L292-L302)).

**Below compute capability 8.0 (e.g. V100).**
- Not supported, but not blocked either: the NVIDIA backend only checks
  `target.backend == 'cuda'`
  ([`compiler.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/third_party/nvidia/backend/compiler.py#L186-L188)).
- Matrix multiplies on compute capability 7.x fall back from tensor cores to
  the FMA path, with a deprecation remark
  ([`AccelerateMatmul.cpp`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L503-L510)).
- The Volta tensor-core layout is rejected as "deprecated and no longer
  supported" ([`Dialect.cpp`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/lib/Dialect/TritonGPU/IR/Dialect.cpp#L1574-L1577)).
- *(reading)* So kernels may still compile and run on a V100, slowly and
  untested.

**Communication: none shipped.**
- Core Triton has no collectives and no NVSHMEM or NCCL/RCCL bindings.
  Searches for allreduce, nvshmem, rocshmem, multimem, nccl and IPC calls
  found nothing in the language, `tl.extra`, the tutorials or `third_party/`.
- It does have building blocks:
  - atomics, loads and stores that take `scope="sys"`
    ([`core.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/language/core.py#L2598-L2619));
  - raw pointers, which may point at another GPU's memory. A test signals a
    peer GPU with a system-scope atomic and then reads its buffer
    ([`test_symmetric_memory.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/test/gsan/test_symmetric_memory.py#L74-L85)).
- Two adjacent pieces live in the same repo:
  - an experimental, CUDA-only data-race sanitizer (GSan) with its own
    symmetric-memory rendezvous
    ([`gsan/README.md`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/experimental/gsan/README.md#L5-L12));
  - the separate `triton_kernels` package, which uses PyTorch's symmetric
    memory to dispatch mixture-of-experts tokens within one node
    ([`mesh.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton_kernels/triton_kernels/distributed_details/mesh.py#L127-L131),
    [`distributed.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton_kernels/triton_kernels/distributed.py#L135-L171)).

**How a backend is found, chosen and compiled for.**
- **Discovery.** Backends are discovered through Python entry points in the
  `triton.backends` group
  ([`backends/__init__.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/backends/__init__.py#L38-L66)).
- **What a backend is.** One compiler class (`BaseBackend`: `supports_target`,
  `hash`, `parse_options`, `add_stages`, ...,
  [`compiler.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/backends/compiler.py#L23-L73)) plus one
  driver class (`DriverBase`: `is_active`, `get_current_target`,
  `get_active_torch_device`, ...,
  [`driver.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/backends/driver.py#L114-L147)).
- **Packaging.** Both the NVIDIA and AMD backends ship in every wheel
  ([`setup.py` L401](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/setup.py#L401)). Out-of-tree backends come in
  through `TRITON_PLUGIN_DIRS` at build time
  ([`setup.py` L113-126](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/setup.py#L113-L126)), or as any package that
  registers the entry point.
- **One driver per process.** Exactly one driver may be active, or
  `_create_driver()` raises "There should only be one", unless
  `TRITON_DEFAULT_BACKEND` names one
  ([`runtime/driver.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/runtime/driver.py#L8-L21)).
  The AMD driver is active when
  `torch.cuda.is_available() and (torch.version.hip is not None)`
  ([AMD `driver.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/third_party/amd/backend/driver.py#L360-L366)).
- **Compilation.** Kernels compile on first launch for a
  `GPUTarget(backend, arch, warp_size)`
  ([`compiler.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/backends/compiler.py#L8-L14)). The
  pipeline is `ttir -> ttgir -> llir -> ptx -> cubin` on NVIDIA
  ([L651-668](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/third_party/nvidia/backend/compiler.py#L651-L668)) and
  `... -> llir -> amdgcn -> hsaco` on AMD
  ([L829-844](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/third_party/amd/backend/compiler.py#L829-L844)).
- **Cache.** The on-disk cache key includes the backend's hash, which contains
  the GPU architecture
  ([`cache.py`](https://github.com/triton-lang/triton/blob/4a5147d919defd3064388d4cdd202fb5e9403b61/python/triton/runtime/cache.py#L321-L323)).

### Overlap with gpubridge

- **The same split.** One codebase covers both vendors, and the vendor is
  chosen per process at run time from the environment. In Triton that's the
  active driver; in gpubridge it's the PyTorch build.
- **The same invariant.** One vendor per process, which is exactly how
  gpubridge's islands work.
- **The same registry shape.** Triton's backend table (a name mapped to a
  compiler, a driver, an `is_active()` check and a target) is close to the
  `VendorSpec` registry proposed above under RAJA.

### Where gpubridge is different

- **Compute versus communication.** Triton writes kernels for one GPU;
  gpubridge writes none and moves data between processes.
- **Two vendors in one job.** Triton never needs this; it is gpubridge's
  purpose.
- **A caveat for gpubridge users who run Triton kernels.** *(reading)* On a
  host with both vendors' drivers installed, a ROCm-PyTorch process can see
  both the HIP and CUDA drivers as active, and Triton raises "There should
  only be one" unless `TRITON_DEFAULT_BACKEND=amd` is set. A CUDA-PyTorch
  process is unaffected, because the HIP check needs `torch.version.hip`.

### Ideas worth adopting

1. **Model the vendor registry on Triton's backend table.** This extends the
   RAJA idea above with Triton's details.
   *Status: not implemented yet.*
   - In a new `vendors.py`, each `VendorSpec` carries: a name, an `is_active()`
     check against the PyTorch build, the island backend, the device for a
     local rank, and a target description.
   - Discover third-party vendors through a `gpubridge.vendors` entry point,
     so adding one (e.g. Intel XPU) is a package, not a patch to `detect.py`
     and `config.py`.
   - Keep Triton's strict rule: exactly one active vendor per process, and a
     clear error otherwise.
2. **Handle `TRITON_DEFAULT_BACKEND` on hosts with both vendors.** After
   detection gpubridge knows each rank's vendor.
   *Status: not implemented yet; it only matters once both vendors share a
   host.*
   - Document the caveat above.
   - Optionally set the variable when it is unset, before user code launches
     Triton kernels. This would go in `detect.py` / `config.py`, opt-in to
     avoid surprises.

## Triton-distributed

### What it does

Triton-distributed is ByteDance Seed's extension of Triton for "programming
overlapping kernels on distributed AI systems"
([arXiv 2504.19442](https://arxiv.org/abs/2504.19442)).
- It adds "communication primitives compliant with the OpenSHMEM standard" to
  the compiler.
- On NVIDIA these are lowered to NVSHMEM. On AMD they go to rocSHMEM or AMD's
  MORI SHMEM, picked with `TRITON_DIST_SHMEM_BACKEND`
  ([`utils.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/utils.py#L83-L94)).
- MIT-licensed. It is now an out-of-tree plugin built against Triton v3.7.1,
  with two small hook patches
  ([`build_triton.sh`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/scripts/build_triton.sh#L2-L38)).

**How the vendor is chosen.**
- Per process, at run time: `is_cuda()` is
  `bool(shutil.which("nvidia-smi"))`, and `is_hip()` checks for `rocm-smi`
  ([`utils.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/utils.py#L52-L70)).
- `setup.py` picks its pip dependencies the same way: NVSHMEM wheels only
  when `nvidia-smi` exists
  ([`setup.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/setup.py#L473-L481)).
- The compiler plugin always builds both the NVIDIA and AMD lowerings: "There
  are no per-pass / per-backend build toggles"
  ([`plugin/CMakeLists.txt`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/plugin/CMakeLists.txt#L149-L157)).

**How SHMEM is loaded.**
- NVSHMEM comes from its pip wheel. Its device bitcode is linked into every
  kernel that uses it, and a hook initializes each loaded module
  ([`jit.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/jit.py#L110-L227)).
- rocSHMEM is built from source in a single-node configuration
  ([`build_rshm_ipc_single.sh`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/shmem/rocshmem_bind/scripts/build_rshm_ipc_single.sh#L31)).

**Bootstrap: through PyTorch.**
- `initialize_distributed` calls `init_process_group` with Gloo and NCCL,
  creates an NCCL group over all ranks, and initializes SHMEM over that group
  ([`utils.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/utils.py#L339-L370)).
- The NVSHMEM unique ID is shared with `broadcast_object_list`
  ([L221-237](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/utils.py#L221-L237)).
- Symmetric-heap buffers are wrapped as torch tensors.

**How it overlaps compute and communication.**
- **AllGather + GEMM, NVIDIA**
  ([`allgather_gemm.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/allgather_gemm.py#L622-L720)).
  - Copy engines pull each peer's shard on high-priority side streams.
  - `cuStreamWriteValue` raises a per-rank "ready" flag
    ([`common_ops.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/common_ops.py#L364-L406)).
  - A persistent GEMM starts on the local shard, then waits for each peer's
    flag (`dl.wait` + `consume_token`) before using that shard.
- **GEMM + ReduceScatter**
  ([`gemm_reduce_scatter.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/gemm_reduce_scatter.py#L320-L331)).
  GEMM tiles bump a per-destination counter, the last tile notifies, and the
  scatter and reduce run on a separate stream.
- **AllReduce, NVIDIA, single node.**
  - One-shot (push everything, reduce locally) or two-shot (reduce-scatter,
    then all-gather), chosen by message size
    ([`allreduce.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/allreduce.py#L1112-L1127)).
    For example, with NVLink multimem: two-shot above 64 KB.
  - Inputs larger than the workspace run chunk by chunk
    ([L1180-1207](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/allreduce.py#L1180-L1207)).
- **AMD.**
  - Data moves in 1024-row chunks with `hipMemcpyAsync(... DeviceToDeviceNoCU)`,
    one stream per peer.
  - The ready flags are 4-byte memcpys, "Because driver API(waitValue/writeValue)
    on AMD will affect the perf of gemm"
    ([`amd/allgather_gemm.py`](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/amd/allgather_gemm.py#L337-L357)).

**Results (paper).**
- Speedups of "1.09× to 44.97×" over PyTorch + NCCL/RCCL.
- The largest numbers are against a Python-loop GroupGEMM that the paper
  itself calls "a weak baseline".
- For the fused AllGather + GEMM and GEMM + ReduceScatter kernels:
  - 1.28-1.42× over PyTorch + NCCL on H800, within and across nodes;
  - 1.09-1.16× over PyTorch + RCCL on MI308X;
  - about 95-96% of FLUX across nodes.

**Mixed vendors: not supported.**
- Searches of the code, issues and PRs for heterogeneous, mixed-vendor,
  cross-vendor and interoperability turned up nothing about putting NVIDIA
  and AMD in one job. The paper's "heterogeneous communication" means NVLink
  plus InfiniBand.
- By design, each process runs one SHMEM library over one NCCL world group,
  with one vendor's bitcode linked into its kernels. *(reading)* NVSHMEM and
  rocSHMEM have separate bootstraps and no shared transport.

**Maturity.**
- AMD is single-node only. A maintainer in
  [#108](https://github.com/ByteDance-Seed/Triton-distributed/issues/108):
  "rocshmem does not post the inter-node communication features, and we are
  waiting for it too."
- Only pre-releases exist.
- [#132](https://github.com/ByteDance-Seed/Triton-distributed/issues/132)
  (initializing NVSHMEM on a tensor-parallel subgroup fails) is open.

### Overlap with gpubridge

- **Bootstrap.** Both start through `torch.distributed` and exchange setup data
  over a process group: gpubridge gathers `PeerInfo`, Triton-distributed
  broadcasts the SHMEM unique ID.
- **Vendor choice.** Both support NVIDIA and AMD from one codebase, choosing
  per process at run time.
- **The copy path.** Both care about moving data between GPU memory and the
  network. Triton-distributed optimizes it; gpubridge so far only gets it
  right.

### Where gpubridge is different

- **Level.** Triton-distributed is GPU-initiated communication inside
  kernels: one vendor, symmetric memory, GPU to GPU. gpubridge composes
  host-initiated collectives across vendors, and its bridge goes through CPU
  memory.
- **Mixed vendors.** Triton-distributed has none; that is gpubridge's purpose.
- **Who could host whom.** *(reading)*
  - NVSHMEM's init uses group-relative rank and size, so it could likely run
    on a gpubridge NVIDIA island's process group, giving that island fused
    kernels.
  - Caveats: only one SHMEM world per process; the rocSHMEM and MORI inits
    assume global rank 0 is in the group; issue #132 is open.
  - The reverse doesn't apply, because Triton-distributed has no notion of a
    second vendor.

### Ideas worth adopting

Each idea below can be measured with `scripts/gpu/bench_all_reduce.py`. It
reports the bridged/native latency ratio from 1 KB to 1 GB, in one split-test
job.

1. **A chunked, pipelined bridge**, from the chunked all-reduce, per-chunk
   flags and copy-engine streams.
   *Status: implemented as the opt-in policy
   `pipelined-reduce-bridge-broadcast` (chunk size `GPUBRIDGE_CHUNK_BYTES`),
   with its device code in `staging.py`. Tested on CPU, including a staging
   layer that runs each copy only when its event is waited for. Not validated
   on GPUs (HARDWARE_VALIDATION.md items 17 and 18).*
   - Split the leaders' bridge step into chunks. On a side stream, copy chunk
     k+1 from GPU to pinned CPU memory while Gloo all-reduces chunk k and
     chunk k-1 is copied back.
   - This belongs in a collective policy (`PipelinedReduceBridgeBroadcast` in
     the proposed `policies.py`), not in `BridgeTransport`. The transport only
     sees CPU tensors, while the GPU-to-CPU staging happens in
     `collectives.py`. Allocate the pinned staging buffers once per size.
   - Success: the bridged/native ratio drops for large tensors without hurting
     small ones.
2. **Choose the algorithm by size.** Triton-distributed picks one-shot or
   two-shot by byte count, and falls back to NCCL for small inputs.
   *Status: implemented as the opt-in policy `auto-tuned`. It reads rules
   from `GPUBRIDGE_THRESHOLDS`, a file `bench_all_reduce.py
   --write-thresholds` writes after timing the candidate policies side by
   side; without one it does what `auto` does. The thresholds themselves still
   need real GPU benchmark data (item 19).*
   - For gpubridge: small tensors take one Gloo all_reduce across all ranks
     (the `FlatGloo` policy), which avoids three serialized steps. Large
     tensors take the pipelined bridge.
   - Thresholds come from each cluster's benchmark CSV.
   - Where: a `select_policy(nbytes)` in `policies.py`.
3. **Order copies on the GPU, not with host syncs.** Triton-distributed orders
   copies with stream wait/write-value on CUDA, and tiny memcpys on AMD,
   instead of blocking the host.
   *Status: implemented in the pipelined policy (`CudaStaging`): copies are
   non-blocking, into pinned memory, on two side streams, ordered by events;
   the host waits only for the chunk the bridge needs next. Not validated on
   GPUs.*
   - In gpubridge, the leader's `tensor.cpu()` blocks the host. Replace it
     with non-blocking copies into pinned memory, ordered by events, and block
     only where Gloo needs the data. This pairs with idea 1.
   - The AMD lesson, that the signalling mechanism is vendor-specific, puts
     that choice in the proposed `VendorSpec`.
4. **Share the bridge step out, as the two-shot all-reduce shares the
   reduction.** Two-shot gives each rank one shard to reduce, then all-gathers.
   gpubridge's bridge runs the same split one level up: each island
   reduce-scatters onto k ranks, and rank j of every island all-reduces shard
   j over its own bridge link.
   *Status: implemented as the opt-in policy `sharded-bridge`, with extra
   links from `BridgeTransport.create_groups`. Tested on CPU, bit for bit
   against `flat-gloo` in uneven and three-island layouts. Slower than
   reduce-bridge-broadcast in CPU simulation; not validated on GPUs
   (HARDWARE_VALIDATION.md items 20 and 21).*
   - Success: with the bridge crossing the network (Explorer step 05), the
     bridge phase shrinks as k grows, and the whole all_reduce beats
     reduce-bridge-broadcast for large tensors.

## torchcomms

### What it does

torchcomms is Meta's "new experimental communications API for PyTorch"
([README](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/README.md#L9-L12)), announced on the
[PyTorch blog](https://pytorch.org/blog/torchcomms/) on 2025-10-22.
- BSD-3-licensed ([LICENSE](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/LICENSE#L1)), version 0.3.0
  ([`version.txt`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/version.txt#L1)). The blog warns the API "may undergo
  breaking changes as it matures".
- Needs Python 3.10+ and PyTorch 2.8+ ([README](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/README.md#L19-L24)). The
  wheel pins the exact torch release it was built against
  ([`setup.py`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/setup.py#L122-L124)).
- It sits on top of c10d. Rendezvous goes through a c10d `TCPStore` built from
  `MASTER_ADDR` / `MASTER_PORT`
  ([`StoreManager.cpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/utils/StoreManager.cpp#L11-L34)),
  and a wrapper exposes a communicator as a `c10d::Backend`, so DeviceMesh and
  FSDP2 can use it
  ([`BackendWrapper.hpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/BackendWrapper.hpp#L49),
  [`device_mesh.py`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/device_mesh.py#L59-L75)).
- The blog's long-term plan is to "deprecate the old c10d::Backend interface
  and adopt torchcomms as the underlying implementation for PyTorch
  Distributed".

**API.** `new_comm(backend, device, name)` creates a communicator eagerly,
bound to one device and one backend
([`TorchComm.hpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/TorchComm.hpp#L353-L374)). Every op
takes `async_op`, ranks are relative to the communicator, and `split` makes
sub-communicators. There are also batched send/recv and experimental one-sided
windows.

**Backends**, each a directory under `comms/torchcomms/`:
- `nccl`, and `ncclx` (Meta's NCCL fork with its CTran transport), for NVIDIA;
- `rccl` and `rcclx` for AMD. `rccl` compiles against `ATen/hip` and puts its
  tensors on HIP devices
  ([`TorchCommRCCL.cpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/rccl/TorchCommRCCL.cpp#L13-L31));
- `gloo` for CPU. GPU tensors are copied to the host, reduced over Gloo and
  copied back
  ([`TorchCommGloo.cpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/gloo/TorchCommGloo.cpp#L724-L767));
- `xccl` for Intel XPU.

**How the vendor is chosen.**
- At build time. `setup.py` turns on NCCL and NCCLX, and turns off RCCL,
  unless the installed torch is a ROCm build
  ([`setup.py`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/setup.py#L79-L85)).
- The stable wheels on download.pytorch.org are CUDA-only. ROCm wheels exist
  only in the nightly index (checked 2026-10-07).
- Then per communicator, at run time: the backend name passed to `new_comm`.

**Mixed vendors: a goal, not implemented.**
- The blog lists "Heterogeneous Hardware Support" as a project goal: "we're
  designing for heterogeneous systems from the ground up—enabling mixed
  deployments that span multiple hardware generations and vendors within a
  single training job."
- Its "native multi-vendor GPU support from Day 1" refers to the new RCCL
  backend: the same API on either vendor, not both vendors in one job.
- *(reading)* Each communicator wraps one vendor library. The NCCL and RCCL
  backends each share their own `ncclUniqueId` through the store and call
  their own init
  ([NCCL](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/nccl/TorchCommNCCLBootstrap.cpp#L102-L128),
  [RCCL](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/rccl/TorchCommRCCLBootstrap.cpp#L95-L121)).
  No backend combines two others. The only mention of composite backends is a
  comment on how they would rank abort reports
  ([`TorchComm.hpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/TorchComm.hpp#L274-L277)).
- UniFlow, in the same repo, is a "Unified Transport for Heterogeneous LLM
  Systems" ([`ONBOARDING.md`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/uniflow/docs/ONBOARDING.md#L6-L8)), but
  it moves data point to point, with no collectives, and also picks its
  vendor at build time
  ([`AMD_BUILD.md`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/uniflow/amd/AMD_BUILD.md#L5-L9)).

**Hooks and fault tolerance.**
- Pre-, post-, abort-, reconfigure- and graph-replay hooks on every
  communicator ([`TorchComm.hpp`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/TorchComm.hpp#L299-L317)).
- Planned: a fault-tolerant backend built on CTran (blog).

### Overlap with gpubridge

- **Layer.** Both sit above NCCL and RCCL, rendezvous through c10d stores,
  and use Gloo to move GPU data through the host.
- **Hooks.** torchcomms has per-communicator pre/post hooks. gpubridge has
  collective observers (`gpubridge.observe`), which give each cluster-wide
  collective a record with matching sequence numbers on every rank.
- **Goal.** Both name one job across vendors as a goal.

### Where gpubridge is different

- **Mixed vendors work today.** gpubridge runs each vendor's library in its
  own island and joins the islands with a CPU bridge, using stock PyTorch
  builds (2.3 or later). In torchcomms, each communicator is still one
  vendor's library.
- **No custom build.** gpubridge is pure Python over `torch.distributed`.
  torchcomms is a C++ extension pinned to one torch release, and its ROCm
  wheels are nightly-only.
- **Scope.** torchcomms is a whole communications stack: new collective
  semantics, one-sided windows, fault tolerance, scaling to 100,000+ GPUs.
  gpubridge offers five cluster-wide collectives (all_reduce, broadcast,
  all_gather_into_tensor, reduce_scatter_tensor, barrier), each with
  `async_op`, and the cross-vendor composition.
- **Who could host whom.** *(reading)*
  - gpubridge's islands could run on torchcomms `nccl` and `rccl`
    communicators.
  - gpubridge could register as a torchcomms backend from Python
    ([`register_backend`](https://github.com/meta-pytorch/torchcomms/blob/a61f9adc18384160079d533b53cc5bcd663acf85/comms/torchcomms/TorchCommBackendPy.cpp#L282-L300)),
    so DeviceMesh and FSDP2 code could run on a mixed cluster.
  - If torchcomms adds mixed-vendor jobs, a host bridge like gpubridge's is
    one way to build them.

### Ideas worth adopting

1. **Per-collective hooks.**
   *Status: implemented as collective observers (`gpubridge.observe`).*
   Unlike torchcomms' hooks, which see one communicator, an observer sees the
   whole cross-vendor collective, with its phases: island-reduce, bridge,
   island-broadcast.
2. **Become a torchcomms backend**, so code written for DeviceMesh or FSDP2
   runs on a mixed cluster unchanged.
   *Status: not implemented. Worth revisiting when torchcomms' API stabilizes
   and its ROCm wheels leave the nightly index. A plain `torch.distributed`
   backend, which reaches the same FSDP2 and DeviceMesh code, is sketched in
   [ROADMAP.md](ROADMAP.md), section 3.*

## Three 2026 papers: mixed-vendor collectives over RDMA

Three papers from 2026 put NVIDIA and AMD (or other vendors) in one training
job. Read on arXiv on 2026-10-09 (version 1 of each). Two of them are called
HetCCL but come from different groups with different designs; the later one
doesn't cite the earlier. Numbers below are as the papers report them, not
reproduced. None of the three has released code that could be found on
2026-10-09.

### HetCCL (Seoul National University, Samsung Research, Moreh)

"HetCCL: Accelerating LLM Training with Heterogeneous GPUs", Heehoon Kim,
Jaehwan Lee, Taejeoung Kim, Jongwon Park et al., [arXiv 2601.22585](https://arxiv.org/abs/2601.22585)
(30 Jan 2026).

**What it does.**
- An orchestration layer over NCCL and RCCL: "HetCCL acts as an orchestration
  layer that invokes pure NCCL and RCCL for vendor-local collectives, while
  handling cross-vendor coordination in a separate layer" (§4.1).
- **Between vendors, GPU memory goes straight to the NIC.** Memory from
  `cudaMalloc` / `hipMalloc` is registered with `ibv_reg_mr` and sent with IB
  Verbs, relying on NVIDIA GPUDirect RDMA and AMD DirectGMA. No driver
  changes. It falls back to host staging when RDMA isn't available.
- One vendor per node. Applications run unchanged: `LD_PRELOAD` swaps in
  HetCCL's NCCL/RCCL symbols.

**Results (paper).** 2 nodes of 4x V100-PCIe and 2 nodes of 4x Radeon Pro
W7800, ConnectX-6 InfiniBand:
- up to 1.48x faster than NVIDIA-only training and 2.97x faster than AMD-only;
- up to 97% efficiency (about 90% on average), defined as mixed throughput
  over the sum of the two single-vendor throughputs.

**Availability.** The paper says the code is released, but its links are
omitted for review, and no code could be found.

### HetCCL (Peking University, BAAI, Infrawaves, ICT-CAS)

"HetCCL: Enabling Collective Communication For Mixed-Vendor Heterogeneous
Clusters", Yuejie Wang, Tao Chang, Yuanyuan Zhao, Yulong Ao et al.,
[arXiv 2605.31000](https://arxiv.org/abs/2605.31000) (29 May 2026).

**What it does.**
- **Device-centric transport.** "We choose the common RDMA APIs (i.e., verbs)
  as the bridge for cross-vendor device data transport" (§2.1). The sender
  copies into a registered RDMA buffer in GPU memory. A CPU proxy thread
  drives IB Verbs, in 4 MB chunks, with copies and transfers pipelined.
- **Hierarchical collectives.** Each vendor's own library runs inside its
  group, with a ring between groups. Reductions use the vendor library inside
  a "border communicator".
- A C++ PyTorch backend plugin, so training code runs unchanged.

**Results (paper).**
- Hardware: NVIDIA A800 and three unnamed vendors. AMD is not among the
  evaluated hardware.
- Against Gloo: the abstract reports 17-19x Gloo's bandwidth; §6.1.1 says
  "> 6x" for point to point.
- Training step time drops 9.1% (Llama3-3B) and 16.9% (Llama3-8B) against
  Gloo.

**Availability.** "publicly available", link omitted for anonymity; no code
could be found.

### Joint Training on AMD and NVIDIA GPUs (Zettabyte AI)

Jon Hu, Thomas Jia, Jing Zhu, Zhendong Yu, [arXiv 2602.18007](https://arxiv.org/abs/2602.18007)
(20 Feb 2026).

**What it does.** Two designs for Megatron, with heterogeneity only in the
pipeline-parallel groups ("In this paper, heterogeneity is introduced only in
the PP groups", §4.1):
- **CPU forwarding.** Gloo carries only the cross-vendor pipeline-parallel
  traffic, while data- and tensor-parallel groups stay on NCCL or RCCL. This
  is close to gpubridge's design, applied to pipeline parallelism.
- **Device-Direct.** A CPU proxy controls transfers. The GPU copies into a
  chunk buffer, and GPUDirect RDMA sends it to the NIC on both sides, through
  ibverbs. Exposed as a PyTorch backend plugin.

**Results (paper).** One node of 8x H200 and one of 8x MI325X, with 8x
BlueField-3 per node. LLaMA-8B, TFLOPs per GPU:

| Setup | TFLOPs/GPU |
| --- | --- |
| Gloo for all communication, mixed | 11.1 |
| Gloo for cross-vendor pipeline traffic only, mixed | 160.8 |
| same, plus one NIC per GPU, mixed | 236.5 |
| AMD only | 534.9 |
| Device-Direct, mixed | 539.6 |
| NVIDIA only | 549.7 |

**Availability.** No code, repository or license is mentioned.

### What this means for gpubridge

- **The published systems are much faster across vendors.** All three move
  cross-vendor data GPU to GPU over RDMA, or through GPU-side buffers and
  RDMA, instead of through host memory and Gloo, which is what gpubridge
  does. The PKU/BAAI HetCCL reports more than 6x Gloo's point-to-point
  bandwidth (17-19x in its abstract). In the Zettabyte numbers, the
  Gloo-based variants reach 2% to 44% of Device-Direct's throughput. The SNU
  HetCCL doesn't compare with Gloo. Expect gpubridge's bridge to be the
  bottleneck for bandwidth-bound collectives.
- **What gpubridge offers instead.**
  - It is open source (MIT) and pure Python over stock PyTorch builds (2.3 or
    later, CUDA or ROCm).
  - It needs no custom build, no `LD_PRELOAD`, no RDMA-capable NIC, no
    GPUDirect or peer-memory support, and no particular network: TCP between
    the machines is enough.
  - It installs with pip from its Git repository (it isn't on PyPI).
  - It is a working baseline you can run today, and a reference to check
    faster paths against.
- **Where the papers point.** gpubridge's `BridgeTransport` is where a
  GPU-to-GPU RDMA transport would plug in. [ROADMAP.md](ROADMAP.md), section
  2, sketches one: peer-memory or dma-buf registration, capabilities checked
  in discovery, an optional extra rather than a core dependency. *Status:
  design only, not implemented.* Its benefit depends on hardware these
  papers needed: InfiniBand or BlueField NICs, and GPUDirect RDMA or
  DirectGMA.
- **Without RDMA.** The opt-in `pipelined-reduce-bridge-broadcast` (copies
  overlapped with the bridge) and `sharded-bridge` (k bridge links instead of
  one) attack the same bottleneck on ordinary networks. Both still go through
  host memory and Gloo, so neither closes the gap to the papers, and neither
  is validated on GPUs yet (HARDWARE_VALIDATION.md items 17 to 21).
- **Not comparable yet.** gpubridge has no GPU numbers at all yet (see
  HARDWARE_VALIDATION.md), so none of the papers' figures can be put next to
  its own.

## Positioning (README paragraph)

Updated 2026-10-10 for 0.2 (all five collectives and the DDP comm hook); now
in the README.

> gpubridge lets one distributed PyTorch job span NVIDIA and AMD GPUs at the
> same time. Choosing between NCCL and RCCL on a single-vendor machine is
> already solved: PyTorch's `"nccl"` backend runs NCCL on CUDA builds and RCCL
> on ROCm builds, and Modular's MAX loads NCCL or RCCL to match the GPU it
> was built for. Neither connects the two, because an NCCL communicator and
> an RCCL communicator cannot exchange data, and Modular states that
> mixed-vendor hosts are not supported. Meta's torchcomms lists mixed-vendor
> jobs as a design goal, but each of its communicators still runs one
> vendor's library. Research systems published in 2026 (two called HetCCL,
> and Zettabyte's joint AMD and NVIDIA training) do join the vendors, moving
> data GPU to GPU over RDMA, and report much higher cross-vendor bandwidth
> than a Gloo-based bridge like gpubridge's. Their code isn't available
> (October 2026), and they rely on RDMA-capable NICs for that speed and on
> custom builds or plugins. gpubridge is the open-source option that runs on
> stock PyTorch builds over any network. It keeps each vendor on its native
> library inside an island and joins the islands with a CPU bridge. One
> `all_reduce`, `broadcast`, `all_gather_into_tensor`, `reduce_scatter_tensor`
> or `barrier` call then covers every GPU in the job, from the same code on
> every node, and DistributedDataParallel can sync gradients through it with
> a comm hook. CUDA and ROCm builds of PyTorch, from 2.9.1 to 2.14.1, have
> been shown to join one job and complete the bridge on CPU; validation on
> real GPUs is next.
