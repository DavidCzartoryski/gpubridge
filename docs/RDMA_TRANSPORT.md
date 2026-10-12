# An RDMA bridge transport over UCX

**Status: design, with a host-memory implementation in progress. Nothing here
has run on RDMA hardware or on a GPU.** This document turns
[ROADMAP.md section 2](ROADMAP.md#2-an-rdma-bridge-transport) into a plan
that can be built and tested without RDMA hardware, then checked on it.

## Goal

A new `BridgeTransport` that carries the island leaders' bridge step over
RDMA (InfiniBand or RoCE). Where the hardware and drivers allow it, data
should go from GPU memory on one vendor's leader to GPU memory on the
other's, using GPUDirect RDMA on NVIDIA and ROCm peer memory or dma-buf on
AMD. The Gloo bridge stays the default and the fallback. Every policy keeps
working unchanged, and so does every job that doesn't ask for the new
transport.

Non-goals: replacing NCCL or RCCL inside an island, writing GPU kernels, and
making RDMA a dependency of the core package.

## 1. Library: UCX or raw ibverbs

| | UCX (UCP API) | raw ibverbs |
| --- | --- | --- |
| Transports | picks per peer: RC/DC verbs on InfiniBand, RoCE, shared memory, TCP | verbs only; TCP or shared memory would be separate code |
| Testing without RDMA | the same code runs over TCP (`UCX_TLS=tcp`) in CI and containers | needs Soft-RoCE (`rdma_rxe`) at least, which needs a kernel module |
| GPU memory | memory types `cuda` and `rocm`: registration, GPUDirect RDMA, `gdr_copy`, and staging through host memory when the NIC can't reach GPU memory, chosen by UCX | our code: `ibv_reg_mr` through peer memory, or `ibv_reg_dmabuf_mr` with an exported dma-buf, and our own staging |
| Large messages | rendezvous protocols (zero-copy get/put), pipelined through bounce buffers when needed | our code: registration cache, chunking, flow control |
| Connection setup | an opaque worker address, exchanged out of band | queue pair numbers, LIDs/GIDs, state transitions, exchanged out of band |
| Error handling | per-endpoint error callbacks (`UCP_ERR_HANDLING_MODE_PEER`) | async events and completion errors on our queue pairs |
| Dependency | `libucp` (in Debian/Ubuntu as `libucx-dev`; a module on most clusters) | `libibverbs` (always present where RDMA is) |
| Cross-vendor build | each side needs a UCX built for its GPU runtime (see section 2) | each side registers its own memory; nothing in the library knows about GPUs |
| Control | less: UCX chooses protocols and lanes; tuned with `UCX_*` variables | full |

**Decision: UCX.** It is the only option of the two that we can test end to
end in CI today, over TCP, with the same code that runs on InfiniBand, and it
already does the GPU-memory work (memory-type detection, registration,
staging when GPUDirect isn't available) that raw verbs would make us write
and debug without hardware. The costs are real and recorded here:

- UCX decides protocols itself. A bad choice on a cluster (say, staging
  through host memory where GPUDirect would work) has to be found with
  `UCX_LOG_LEVEL=info` and `UCX_PROTO_INFO=y` and fixed with `UCX_*`
  variables, not in our code. The kit records the variables and UCX's
  choices with every run.
- Both leaders need compatible UCX builds (section 2).
- Raw verbs remain possible later as a second transport behind the same
  `BridgeTransport` interface, if UCX's choices turn out to be the bottleneck.

## 2. Cross-vendor specifics

**Each side runs its own build.** An NVIDIA leader runs the CUDA build of
PyTorch and a UCX built with CUDA support; an AMD leader runs the ROCm build
of PyTorch and a UCX built with ROCm support. Neither needs the other
vendor's runtime. On Explorer this means a CUDA-enabled UCX module on the
NVIDIA side and a ROCm-enabled one (or a locally built UCX) on the AMD side.

**What has to match on the wire.**
- The UCX wire protocol. UCX keeps wire compatibility only within limits, so
  the transport exchanges each side's UCX version (`ucp_get_version`) in its
  handshake and refuses, on every rank, a pair whose major or minor versions
  differ. That is stricter than UCX promises, and can be relaxed once a mixed
  pair is shown to work.
- The network: both NICs on the same fabric (InfiniBand subnet or a routed
  RoCE network), with a transport both builds enable. `UCX_TLS` and
  `UCX_NET_DEVICES` must be set so the two sides can agree; the kit records
  both.
- The data: tensors are raw bytes on the wire. Both sides already agree on
  dtype, shape and op, because gpubridge's collectives check them before any
  communication.

**Registering GPU memory.** UCX registers GPU buffers with the NIC itself,
through one of:
- the peer-memory kernel modules (`nvidia-peermem` on NVIDIA; AMD's peer
  memory support in the `amdgpu` driver on ROCm), which let `ibv_reg_mr`
  take device pointers; or
- dma-buf (`ibv_reg_dmabuf_mr`), with the buffer exported by the GPU runtime,
  which needs a recent kernel, rdma-core, and a UCX that uses it.

**When GPUDirect isn't available**, the transport still works:
- If the local UCX was built without the GPU memory type, the transport stages
  through pinned host memory itself, as `GPUBRIDGE_PINNED_STAGING` does for the
  default bridge, and hands UCX host buffers. It finds out at start-up from
  `ucp_context_query` (`UCP_ATTR_FIELD_MEMORY_TYPES`).
- If UCX has the memory type but the NIC can't reach GPU memory (no peer
  memory module, no dma-buf), UCX stages through host memory internally. That
  is correct but no faster than staging ourselves; the kit's records show which
  path UCX took.
- Either way, the probe and the run records say which path each leader used,
  so a run never claims GPU-to-GPU transfer it didn't do.

## 3. Connection setup, lifecycle, teardown

`BridgeTransport.create(ranks, timeout=...)` is collective over the whole
world, like `new_group`: every rank calls it in the same order. The new
transport, registered as `bridge="ucx"`:

1. Numbers itself with a counter that is the same on every rank (creation
   order is the same everywhere), so its store keys never collide with another
   link's (the sharded policy creates several through `create_groups`).
2. On each member: creates a UCP context (features: tagged messages) and one
   worker, and publishes the worker address and its UCX version under
   `gpubridge/ucx/<n>/addr/<rank>` in the default c10d store, the one
   `init()` already rendezvoused through.
3. Waits for every member's address, refuses mismatched UCX versions, and
   creates one endpoint per peer with peer error handling.
4. Exchanges a small handshake message with every peer, over the endpoints,
   so a link that can't actually carry data fails here, not in the first
   collective.
5. Writes `ok` or the error text under `gpubridge/ucx/<n>/status/<rank>`.
   **Every rank, members or not, waits for all members' status keys** (within
   the timeout) and raises the same `RuntimeError`, naming each failing rank
   and its error, if any member failed. So a broken link fails `init()` on
   every rank instead of leaving the others to hang in their first island
   collective.

**Lifecycle.** One context and one worker per leader per link, owned by the
transport object; endpoints live until `close()`. Collectives on a rank run
one at a time (sync calls wait for queued async ones), so a single-threaded
worker (`UCS_THREAD_MODE_SINGLE`), driven only from the thread running the
collective, is enough.

**Teardown order** (`close()`, called by `gpubridge.destroy()`):
1. Make sure no request is outstanding (every operation completes or fails
   inside the collective that posted it, so there are none by design; a
   leftover one is cancelled and reported).
2. Close every endpoint with `ucp_ep_close_nbx` in flush mode, progressing the
   worker until each close completes or the timeout passes, then force-close
   what's left. Flush mode lets a peer still finishing its last receive get
   its data first.
3. Destroy the worker, then the context.

## 4. Stream ordering on GPUs

On the host-memory path (step 7b), nothing changes: policies stage to host
memory exactly as they do for Gloo, and the transport only sees host buffers.

On the device-memory path (step 7d), the transport receives device tensors.
A transport class says so with `device_tensors = True`; policies then skip
their host staging and pass the tensor straight through. The rules:

- **Before posting a send of device memory,** the transport records an event
  on the caller's current stream (the worker's, for async collectives) and
  waits on it on the host. That covers the island reduce (a synchronous
  `dist.reduce`, whose completion the current stream already waits for) and
  any kernel the caller queued. The host wait is for that event only, not the
  whole device.
- **Reductions run on the current stream** with ordinary torch ops, into the
  caller's tensor, after each receive completes. Before the next step sends
  data that a reduction just produced, the transport waits on an event
  recorded after it, as above.
- **After the last receive and reduction,** nothing waits on the host: the
  island broadcast that follows is queued on the same stream, behind the
  reductions. A receive that lands directly in the caller's tensor is complete
  when UCX reports it, before any later kernel on any stream is launched.
- Scratch buffers come from the caching allocator on the current stream, so
  reuse across collectives is ordered by the allocator.

## 5. Chunking, pipelining and the algorithm

**All_reduce across p leaders.**
- p = 2: each leader posts a receive of the peer's tensor into a scratch
  buffer and a send of its own, waits for both, then adds the scratch into its
  tensor. Addition is commutative in IEEE floating point, so both leaders get
  identical bits.
- p > 2: a ring, as in NCCL: p-1 reduce-scatter steps on p chunks, then p-1
  all-gather steps. Each chunk is reduced in one fixed order and then copied,
  so every leader ends with identical bits.
- MAX and MIN take the same steps with `maximum`/`minimum` instead of add.

**Broadcast** from one leader: the root sends to each other leader (p is
small, at most a handful of islands); the others receive in place.

**Chunking.** Each send and receive is split into chunks of
`GPUBRIDGE_UCX_CHUNK_BYTES` (default 4 MiB), all posted up front with tags
that carry the step and chunk number, so UCX can keep the link busy and use
its rendezvous protocol per chunk. The pipelined policy keeps working as it
is: it hands the transport one chunk at a time.

**Tags.** A 64-bit tag holds the link number, a sequence number of the
collective on this link, the step and the chunk. Receives match on the full
tag, so a late message from one collective can never land in the next.

## 6. Errors and timeouts

- Every wait has a deadline: the timeout `init()` gave its groups. A collective
  that passes it raises `RuntimeError` naming the operation, the peer and how
  many bytes were outstanding. It does not hang.
- An endpoint error (a peer died or the link dropped) fails the operation that
  saw it at once, with UCX's status text, instead of waiting for the deadline.
- A failure leaves the transport unusable: later calls raise at once, and
  `close()` force-closes. Like a NCCL error, the job is expected to stop; the
  other leaders see an endpoint error or their own deadline. Ranks that aren't
  leaders are in an island collective at that point and get NCCL's or Gloo's
  own timeout, as with the Gloo bridge today.
- Creation errors fail `init()` on every rank (section 3).
- Endpoints use UCX's peer error handling, so that a dead peer fails an
  operation at once. UCX's shared-memory transports can't report a failed
  peer, so UCX never uses them for the bridge: two leaders on one machine (a
  split test) talk over TCP or verbs loopback. Leaders normally sit on
  different hosts, where this changes nothing.

## 7. Build and packaging

**A small C++ library with a C interface, loaded with ctypes; not a torch C++
extension.** The UCX code needs no PyTorch types: it moves bytes between raw
pointers and reports completion. Building it against PyTorch's C++ headers
would tie the compiled library to one PyTorch release and build (CUDA or
ROCm, C++ ABI flags), and a mixed job has two builds by definition. A plain
shared library built against `libucp` works under either, survives PyTorch
upgrades, and builds in seconds with any C++ compiler. Device-memory
ordering (section 4) is done in Python with `torch.cuda` events, which work
on both CUDA and ROCm builds.

- Source: `src/gpubridge/ucx/bridge.cpp` (C++17), shipped in the wheel.
- Build: `python -m gpubridge.ucx build` compiles it with the system compiler
  against UCX found through `UCX_HOME` or `pkg-config ucx`, into
  `~/.cache/gpubridge/` (or `GPUBRIDGE_UCX_LIB_DIR`). `bridge="ucx"` builds it on
  first use if it isn't built yet and a compiler and UCX are there.
- Optional at install time and Linux only: nothing is compiled by
  `pip install gpubridge`. Without UCX, everything else works exactly as
  before, and `bridge="ucx"` fails `init()` on every rank with a message
  saying what's missing (no compiler, no UCX, build error) and suggesting
  `bridge="gloo"`.
- The probe records whether the library is built, the UCX version, and the
  memory types it supports.

**Decision recorded:** this departs from the "torch C++ extension" in the
original plan, for the reasons above. If a later step needs ATen types in C++
(for example, a reduction kernel fused into the receive path), a torch
extension can be added next to this library then.

## 8. Testing

- **CI (Linux, GitHub Actions):** a job installs `libucx-dev` from Ubuntu,
  builds the library, and runs the transport tests with `UCX_TLS=tcp,self`
  (and once with UCX's own choice) between processes on one runner. The other
  jobs, and macOS, skip them cleanly when the library can't be built.
- **Bit for bit:** every transport-agnostic policy test that runs on the Gloo
  bridge also runs on `bridge="ucx"`, against `flat-gloo`, with the same exact
  integer data.
- **Transport tests:** a connection failure on one rank fails `init()` on
  every rank; a peer that never answers trips the deadline with a clear
  message; teardown with an operation outstanding; large tensors, odd sizes,
  every dtype; the chunked path with a tiny chunk size; MAX and MIN; three and
  four leaders (the ring).
- **Soft-RoCE:** tried on GitHub's runners (`modprobe rdma_rxe`, then `rdma
  link add`): not possible there, see section 10. `GPUBRIDGE_TEST_UCX_TLS=rc,self`
  runs the transport tests over verbs on any machine that has them.
- **GPU and real RDMA:** tests marked `gpu` (and `rdma`) are written now and
  skipped without the hardware: device tensors on the device path, stream
  ordering behind queued kernels, the pinned-staging fallback, and a check
  that the run record says which path was used.
- **Containers:** the pinned two-core Linux containers the project already
  uses for CPU tests, with `libucx-dev` installed.

## 9. Questions for Research Computing

1. Do Explorer's GPU nodes (`gpu`, `multigpu`, and any AMD nodes in
   `sharing`) have InfiniBand or RoCE NICs, and are NVIDIA and AMD nodes on the
   same fabric?
2. Is GPUDirect RDMA enabled on the NVIDIA nodes, with `nvidia-peermem`
   loaded (or dma-buf support in the kernel and rdma-core)?
3. On any AMD GPU nodes: is ROCm peer memory (or dma-buf export) available to
   RDMA?
4. Is there a UCX module, and is it built with CUDA support? With ROCm
   support? Which version? Can users build their own UCX against the
   cluster's rdma-core?
5. Which `UCX_NET_DEVICES` / interfaces should jobs use, and are there limits
   on locked memory (`ulimit -l`) for RDMA registration in Slurm jobs?
6. Is `perftest` (`ib_write_bw` with `--use_cuda` / `--use_rocm`) installed,
   to measure the hardware ceiling before running gpubridge?

## 10. Results so far

- Host-memory path over TCP: every bridge policy on two, three and four
  leaders, bit for bit against `flat-gloo`, in Linux containers (UCX 1.18.1)
  and in CI on GitHub's Ubuntu runners (UCX 1.16).
- Soft-RoCE on GitHub's runners (tried 2026-10-12): **not possible.** They run
  an Azure kernel (6.17.0-1022-azure) without `rdma_rxe`, even after installing
  `linux-modules-extra` for it. So CI can't exercise verbs; that has to wait for
  real hardware or a self-hosted runner.
- The same runners expose an RDMA-capable device that UCX's `rc` transport
  finds but can't use (`ibv_create_srq() failed: Operation not supported`), and
  UCX's default transport choice then fails to initialize. CI pins
  `UCX_TLS=tcp,self`, and the transport's error now says to set `UCX_TLS` or
  `UCX_NET_DEVICES` when this happens. Expect the same on some cloud VMs.
- Nothing has run on RDMA hardware or a GPU.

## Plan

1. This document.
2. Library, build command, CI job; skip cleanly without UCX.
3. Host-memory path: connect, send/recv, all_reduce and broadcast across
   leaders, `bridge="ucx"`, validated across ranks.
4. Tests on the new transport (section 8).
5. Device-memory path with the pinned-staging fallback; GPU tests written,
   skipped without hardware.
6. Benchmark: `--bridge ucx` in `bench_all_reduce.py` and the job scripts,
   with a predicted-versus-measured entry.
7. Docs: README, ROADMAP status, HARDWARE_VALIDATION items (all pending).
