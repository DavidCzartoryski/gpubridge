# Prior art: Modular MAX `comm` and RAJA

Notes comparing gpubridge with two projects that also promise "write it once,
run it on NVIDIA or AMD". Read on 2026-10-06 at Modular
[`85356f6`](https://github.com/modular/modular/tree/85356f6562ed57bab8762fde38448ba5b50b69c9)
and RAJA [`09626e8`](https://github.com/llnl/RAJA/tree/09626e855db5005357eb1dac7cc4db0546cf0f53)
(`develop`; RAJA has no `main`). All links point at those commits. Statements
marked *(reading)* are conclusions from reading the code, not from running it.

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
- Their docstrings say they return `False` when the library is absent. *(reading)*
  They load it through `_find_dylib`, whose `abort_on_failure` defaults to `True`
  ([ffi](https://github.com/modular/modular/blob/85356f6562ed57bab8762fde38448ba5b50b69c9/Mojo/stdlib/std/ffi/__init__.mojo#L947-L973)),
  so a missing library probably aborts the process rather than returning `False`.
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

**Mixing NVIDIA and AMD: not supported.** You were right about this.
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
   have the right shape but probably abort when the library is missing.
   *Status: implemented as `gpubridge.probe()` and discovery-time problem
   reports.*
   - Add `detect.probe()`, which never raises. It returns this rank's build
     vendor, `dist.is_nccl_available()`, `dist.is_gloo_available()`, whether a
     GPU is visible, and the torch version.
   - Send it in `PeerInfo` during discovery (`topology.py`). Then `init()` can
     fail with one message naming every rank that lacks its island backend,
     instead of some ranks hanging in `new_group`.
   - The straggler detector also gets a per-rank capability snapshot for free.
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
   by name before calling the library. Today gpubridge leaves dtype errors to
   torch, which can fail on the bridge after the island step has already run.
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
   composed.** Today `collectives.py` hard-codes reduce-bridge-broadcast.
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
   - List the modes once in `tests/_harness.py` and parametrize the collective
     tests over them.
   - Enable the GPU modes automatically when hardware is present, so a
     self-hosted GPU runner later needs no new tests.

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
