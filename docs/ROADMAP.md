# Roadmap: designs waiting on GPU results

These are designs, not commitments, and none of them is implemented. Each one
waits for specific results from the GPU validation kit
([GPU_VALIDATION_RUNBOOK.md](GPU_VALIDATION_RUNBOOK.md)). Without those
results there is no way to tell whether the problem it solves is the one that
matters. Any of them that gets built will follow the rules the opt-in
policies follow:

- off by default, and never picked by `auto`;
- selectable with `--policy` (or `bridge=`) in `check.py` and
  `bench_all_reduce.py`;
- any setting that changes how many messages a collective sends is checked
  across ranks during discovery;
- a numbered HARDWARE_VALIDATION.md item for every claim only GPUs can check.

Contents:

1. [Bridge-only compression](#1-bridge-only-compression)
2. [An RDMA bridge transport](#2-an-rdma-bridge-transport)
3. [A torch.distributed backend for FSDP2 and DeviceMesh](#3-a-torchdistributed-backend-for-fsdp2-and-devicemesh)

## 1. Bridge-only compression

**Problem.** Inside an island, NCCL/RCCL move data over NVLink, Infinity
Fabric or PCIe. On the bridge, the same bytes go through host memory and
Gloo over TCP. If GPU runs show the bridge step dominates large all_reduces,
sending fewer bytes over the bridge alone would help without touching the
fast island links.

**Design.**
- **Cast on the bridge only.** Island leaders cast their island sum to
  bfloat16 (or float16) before the bridge all_reduce and cast back
  afterwards. Island reductions stay in full precision. PyTorch's own
  `bf16_compress_hook` compresses the whole DDP all_reduce; this would
  compress only the slow hop.
- **Lossy, so explicit.** gpubridge policies promise not to change what a
  collective computes, and this does. So it would be an opt-in policy with
  "lossy" in its name, and it would refuse integer dtypes, MAX and MIN on
  every rank before any communication. The target dtype would be a setting
  (`GPUBRIDGE_BRIDGE_DTYPE`), checked across ranks in discovery like the
  chunk size.
- **Error feedback, later and only for DDP.** Carrying each bucket's rounding
  error into the next step needs state per bucket. That fits a DDP comm hook
  (the `state` argument of `ddp_comm_hook`), not a stateless all_reduce.
- **Not planned.** Lossless compression of float gradients saves little, and
  it costs CPU time on the leaders, which already do the host copies and run
  Gloo.

**How we'd measure it.**
- `bench_all_reduce.py --candidates` with the lossy policy next to
  reduce-bridge-broadcast, and `--phases` to see the bridge step alone.
- The cost in accuracy: loss curves from `examples/train_synthetic.py` with
  and without it, over enough steps to see drift.
- Tests on CPU: bit for bit against a local model of the cast (island sums
  cast, summed in bridge order, cast back), since FlatGloo no longer gives the
  same answer.

**Depends on GPU results.**
- HARDWARE_VALIDATION.md item 13 (the share of the bridge phase in a bridged
  all_reduce) and items 18 and 21 (whether pipelining or sharding already
  close the gap). If the bridge isn't the bottleneck, this isn't worth its
  loss of precision.

**Status: design only, not implemented.**

## 2. An RDMA bridge transport

**Problem.** The Gloo bridge copies GPU data to host memory, sends it over
TCP, and copies it back on the other side. Three 2026 papers move
cross-vendor data with RDMA instead, and report far better cross-vendor
performance than a Gloo bridge. The second HetCCL reports more than 6x Gloo's
point-to-point bandwidth (17 to 19x in its abstract). Joint Training reports
539.6 TFLOPs per GPU with its RDMA path, against 160.8 to 236.5 with Gloo
carrying the cross-vendor pipeline traffic. Details are in [PRIOR_ART.md](PRIOR_ART.md):

- HetCCL ([arXiv 2601.22585](https://arxiv.org/abs/2601.22585)) registers
  `cudaMalloc` and `hipMalloc` memory with `ibv_reg_mr` and uses InfiniBand
  verbs. It relies on NVIDIA GPUDirect RDMA and AMD DirectGMA, with no driver
  changes.
- The other HetCCL ([arXiv 2605.31000](https://arxiv.org/abs/2605.31000), a
  different group) copies data into registered RDMA buffers in GPU memory and
  sends 4 MB chunks with verbs, under a CPU proxy thread.
- "Joint Training on AMD and NVIDIA GPUs"
  ([arXiv 2602.18007](https://arxiv.org/abs/2602.18007)) adds a Device-Direct
  path with GPUDirect RDMA on both sides, for pipeline-parallel traffic.

None of them publishes code today.

**Design.**
- **A transport, not a policy.** `RdmaTransport` would subclass
  `BridgeTransport` and register as `bridge="rdma"`. Every policy would use
  it unchanged, including the sharded policy's extra links (through
  `create_groups`).
- **Device memory in, no staging.** Today a transport only sees CPU tensors.
  A class attribute such as `device_tensors = True` would let policies skip
  the host copy and pass device tensors straight to the transport. Transports
  without it keep today's behavior.
- **Two ways to register GPU memory with the NIC:**
  - `ibv_reg_mr` on device pointers, through the peer-memory kernel modules
    (`nvidia-peermem` on NVIDIA, AMD's peer memory support on ROCm). This is
    the papers' route.
  - `ibv_reg_dmabuf_mr` with a dma-buf exported by the GPU runtime
    (`cuMemGetHandleForAddressRange` on CUDA, `hsa_amd_portable_export_dmabuf`
    on ROCm), which uses the upstream kernel interface. None of the papers
    uses it. It may be the only option where the peer-memory modules aren't
    installed.
- **Capabilities in the probe.** `Probe` (shared during discovery) would
  record what each rank can do: `libibverbs` present, RDMA devices
  (`/sys/class/infiniband`), the peer-memory module loaded, dma-buf support
  in the runtime. With `bridge="rdma"`, `init()` would fail on every rank
  before any group is created, naming the rank, the missing piece, and
  `bridge="gloo"` as the fallback.
- **Not a core dependency.** The transport would load `libibverbs` at run
  time (ctypes or a small optional extension) and ship as an extra, e.g.
  `pip install gpubridge[rdma]`. The core package keeps PyTorch as its only
  dependency.
- **Reductions.** RDMA moves bytes; it doesn't add them. The second HetCCL
  reduces with each vendor's own library. gpubridge would add on the GPU
  after each receive, for example in a ring all_reduce between the
  participating ranks.

**How we'd measure it.**
- First, before writing any code: the hardware ceiling. perftest's
  `ib_write_bw` can read and write GPU memory (`--use_cuda`, `--use_rocm`, and
  dma-buf variants in recent releases). Run it between an NVIDIA node and an
  AMD node. If cross-vendor GPU-to-GPU RDMA doesn't work there, the design
  stops.
- A new kit step: `scripts/gpu/rdma_probe.py` records `ibv_devinfo`, loaded
  modules and the perftest numbers on each node, in the style of `probe.py`.
- `bench_all_reduce.py --phases` with `bridge="gloo"` and `bridge="rdma"`:
  the bridge phase alone, then the whole all_reduce, against the per-island
  baselines (item 11) and the papers' numbers.
- Correctness: `check.py` with `bridge="rdma"`, bit for bit against
  `flat-gloo`, like every other bridge.

**Depends on GPU results.**
- Hardware first: RDMA NICs between GPU nodes of both vendors, with peer
  memory or dma-buf usable. Explorer's answer is unknown until the GPU survey
  (step 0a) and step 05 run. A cloud pair needs NICs that support it.
- HARDWARE_VALIDATION.md item 8 (separate hosts on a real network) and item
  13 (how much of a bridged all_reduce the bridge takes): RDMA only pays off
  where the bridge dominates.

**Status: design only, not implemented.**

## 3. A torch.distributed backend for FSDP2 and DeviceMesh

**Problem.** `gpubridge.ddp_comm_hook` covers DistributedDataParallel.
FSDP2, DeviceMesh and DTensor have no hook point: they call a process group's
collectives directly. That takes a `torch.distributed` backend.

**Design.** A CPU prototype on torch 2.14.1 (not in this repository) settled
these points:

- **A Python `c10d.Backend` subclass.** Subclassing
  `torch._C._distributed_c10d.Backend` works; subclassing
  `dist.ProcessGroup` does not (DeviceMesh fails with "ProcessGroup name not
  set"). Register with `dist.Backend.register_backend("gpubridge", creator,
  extended_api=True, devices=["cpu", "cuda"])`. The creator receives the
  store, group rank and size, timeout and global ranks.
- **Required overrides.** `supports_splitting`, `supports_coalescing`,
  `supports_time_estimate`, `supports_shrinking` and `options` must all be
  overridden, or PyTorch recurses until it raises `RecursionError`.
- **Methods FSDP2 calls.** `all_gather_single` (forward and backward) and
  `reduce_scatter_single` (gradients), with AVG for float32 and bfloat16, SUM
  for float16, and PREMUL_SUM when a gradient divide factor is set. HSDP adds
  `allreduce` AVG on the replicate group. DTensor state dicts use
  `all_gather_single_coalesced`, and `distribute_tensor` uses `scatter`.
  The old `_allgather_base` names fail on 2.14.1. These names change between
  releases, so the backend would pin a supported range.
- **PREMUL_SUM.** Gloo lacks it, so the backend would scale, then SUM.
- **Subgroups.** `new_group(backend="gpubridge")` calls the creator on member
  ranks only. Each group needs its own islands and bridge, built from the
  store under a distinct `PrefixStore`; Gloo groups built that way worked in
  the prototype.
- **Fail together.** If the creator raises on one rank, the others hang until
  the store times out. So each rank would publish its checks to the store and
  read everyone's before building any group, the way discovery works today.
- **Work objects.** Return a `dist.Work` subclass with `wait()` and
  `get_future()` (DDP requires the future). A future that carries an
  exception crashed DDP's backward in the prototype (SIGSEGV), so failures
  would raise synchronously instead. `is_completed()` always reported False
  through the Python binding.
- **Known limits.** `TORCH_DISTRIBUTED_DEBUG=DETAIL` breaks Python backends.
  Overhead was about 120 to 129 us per tiny all_reduce, against 110 us for
  Gloo on its own, on CPU.
- **GPU risk.** FSDP2 records CUDA events right after a synchronous
  collective returns, so outputs must be ready relative to the caller's
  current stream when each method returns.
- **The alternative.** Registering as a torchcomms backend
  (PRIOR_ART.md, torchcomms idea 2) would reach the same users once its API
  and ROCm wheels settle.

**How we'd measure it.**
- Correctness on CPU first: every method through the backend, bit for bit
  against `flat-gloo` on integer-valued data, in the layouts the tests already
  use; then FSDP2 on a small transformer with the loss curve against a
  single-island run.
- On GPUs: the same FSDP2 model in a split test (step 04) and on real mixed
  ranks (step 07), with step time against FSDP2 on the Gloo backend and on
  one vendor's NCCL/RCCL.

**Depends on GPU results.**
- HARDWARE_VALIDATION.md items 15 and 16 (native collectives and async
  stream semantics on GPUs), and item 22 (DDP through the comm hook), since
  the backend would build on all three.

**Status: design only, not implemented.**
