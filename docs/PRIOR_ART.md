# Prior art: Modular MAX `comm`, RAJA, Triton and Triton-distributed

Notes comparing gpubridge with four projects that also promise "write it once,
run it on NVIDIA or AMD". Each repo was read on 2026-10-06 at a fixed commit,
and every code link points at that commit:

- Modular [`85356f6`](https://github.com/modular/modular/tree/85356f6562ed57bab8762fde38448ba5b50b69c9)
- RAJA [`09626e8`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53)
  (`develop`; RAJA has no `main`)
- Triton [`4a5147d`](https://github.com/triton-lang/triton/tree/4a5147d919defd3064388d4cdd202fb5e9403b61)
- Triton-distributed [`7908e4e`](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a)
  (`main` as of 2026-09-18)

Statements marked *(reading)* are conclusions from reading the code, not from
running it.

| Project | Compute or communication | Vendors supported | Vendor chosen | Mixed vendors in one job |
| --- | --- | --- | --- | --- |
| Modular MAX `comm` | communication (vendor libraries or its own kernels) | NVIDIA, AMD | compile time, per target | no (its README says so) |
| RAJA | compute (loops and kernels) | CUDA, HIP, SYCL, OpenMP, CPU | compile time, by policy type | no |
| Triton | compute (kernel language and compiler) | NVIDIA (compute capability 8.0+), AMD (ROCm 6.2+) | run time, one driver per process; compiled per GPU architecture | no: exactly one active driver per process |
| Triton-distributed | communication inside kernels, overlapped with compute | NVIDIA (NVSHMEM), AMD (rocSHMEM or MORI, single node), MetaX | run time, per process (`nvidia-smi` / `rocm-smi` on PATH) | no: one SHMEM library over one NCCL group |
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
- Both start with a deliberately narrow v1: SUM only, a few dtypes, and
  all_reduce, broadcast and allgather-style ops.
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
   reports.*
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
   *Status: implemented as a scaffold in `policies.py`, with
   `reduce-bridge-broadcast`, `native-only`, `flat-gloo` and `auto` (the
   default, which keeps the earlier behaviour). There is no per-call override:
   a mismatch between ranks there would hang with no cheap way to detect it.*
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
   *Status: not implemented yet. The hooks it needs exist: a marked place in
   `policies.py`, plus `CollectivePolicy.setup()` / `close()` for pinned buffers
   and streams.*
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
   *Status: not implemented yet. The hook exists:
   `CollectivePolicy.select_policy(nbytes)`, which every collective goes
   through, and which a test policy already uses to route small messages
   through `flat-gloo`. Thresholds wait for real GPU benchmark data
   (`bench_all_reduce.py --policy`).*
   - For gpubridge: small tensors take one Gloo all_reduce across all ranks
     (the `FlatGloo` policy), which avoids three serialized steps. Large
     tensors take the pipelined bridge.
   - Thresholds come from each cluster's benchmark CSV.
   - Where: a `select_policy(nbytes)` in `policies.py`.
3. **Order copies on the GPU, not with host syncs.** Triton-distributed orders
   copies with stream wait/write-value on CUDA, and tiny memcpys on AMD,
   instead of blocking the host.
   *Status: not implemented yet.*
   - In gpubridge, the leader's `tensor.cpu()` blocks the host. Replace it
     with non-blocking copies into pinned memory, ordered by events, and block
     only where Gloo needs the data. This pairs with idea 1.
   - The AMD lesson, that the signalling mechanism is vendor-specific, puts
     that choice in the proposed `VendorSpec`.

## Positioning (proposed README paragraph)

> gpubridge lets one distributed PyTorch job span NVIDIA and AMD GPUs at the
> same time. Choosing between NCCL and RCCL on a single-vendor machine is
> already solved: PyTorch's `"nccl"` backend runs NCCL on CUDA builds and RCCL
> on ROCm builds, and Modular's MAX loads NCCL or RCCL to match the GPU it
> was built for.
> Neither connects the two, because an NCCL communicator and an RCCL
> communicator cannot exchange data, and Modular states that mixed-vendor
> hosts are not supported. gpubridge keeps each vendor on its native library
> inside an island and joins the islands with a CPU bridge. One `all_reduce`,
> `broadcast` or `barrier` call then covers every GPU in the job, from the
> same code on every node. CUDA and ROCm builds of PyTorch, from 2.9.1 to
> 2.14.1, have been shown to join one job and complete the bridge on CPU;
> validation on real GPUs is next.
