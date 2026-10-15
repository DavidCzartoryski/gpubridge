# gpubridge as a torch.distributed backend

**Status: experimental, tested on CPU only.** The goal is for a DDP or FSDP2
training script to run on a mixed NVIDIA + AMD cluster by naming the backend:

```python
import gpubridge  # registers the name; optional on torch 2.14 and later
dist.init_process_group(backend="gpubridge")
```

This document records how PyTorch lets a third party register a backend, what
DDP and FSDP2 actually call, and what that means for gpubridge. The design
follows. Source references are to PyTorch 2.8.0 (gpubridge's floor) and 2.14.1
(the newest release CI tests), as installed by pip.

## How PyTorch registers a backend

- **`dist.Backend.register_backend(name, func, extended_api=False, devices=None)`**,
  in `torch/distributed/distributed_c10d.py` (line 304 in 2.8.0, 577 in
  2.14.1). It adds the name to `Backend.backend_list`, records the devices it
  serves in `Backend.backend_capability`, and stores `func` in
  `Backend._plugins`.
- **The creator `func`.** `init_process_group` and `new_group` reach
  `_new_process_group_helper`, which calls `func` for a plugin backend (2.8.0
  lines 2053-2065, 2.14.1 lines 2859-2874). It gets either `(store, rank,
  world_size, timeout)`, or with `extended_api=True` a
  `_DistributedBackendOptions` with `store`, `group_rank`, `group_size`,
  `timeout`, `group_id` and `global_ranks_in_group` (empty for the world
  group), plus the `pg_options` the caller passed.
- **What it may return.** Either a C++ `Backend`, which is registered per device
  on a new `ProcessGroup`, or a `ProcessGroup` subclass. The second case is
  handled right there: "If the type is a subclass of ProcessGroup then return
  this process group immediately" (`pg = backend_class; break`, 2.8.0 line 2079,
  2.14.1 line 2884).
- **Python implementations.** A Python class can subclass `dist.ProcessGroup` and
  implement collectives as methods that return a `Work`. PyTorch's own
  "threaded" test backend does exactly that:
  `torch/testing/_internal/distributed/multi_threaded_pg.py`,
  `class ProcessLocalGroup(dist.ProcessGroup)`, registered with
  `dist.Backend.register_backend("threaded", ...)`. A `Work` can be made from a
  future with `torch._C._distributed_c10d._create_work_from_future`.
- **A Python subclass of the C++ `Backend`** (which would let one process group
  keep Gloo for CPU tensors and a Python backend for CUDA tensors) is possible
  in 2.14.1 but not in 2.8.0, where constructing one fails with "No
  constructor defined" (measured). gpubridge supports 2.8, so it returns a
  `ProcessGroup` subclass.
- **Names.** `DeviceMesh` reads the group's `group_name`. torch sets the name on
  the new group, but for a Python group with no C++ backend the C++ getter
  raises "ProcessGroup name not set" (measured on both versions). So the
  subclass defines `group_name` as a property, as `ProcessLocalGroup` does, with
  the `group_id` from the extended API.
- **`device_id`.** With `init_process_group(device_id=...)`, torch asks the group
  for that device's C++ backend and whether it can split eagerly (2.8.0 line
  2123, 2.14.1 line 2944). A Python group has no per-device backend, so the
  subclass's `_get_backend()` returns a stand-in that says it can't.
- **Teardown.** `destroy_process_group()` calls `shutdown()` on every group in
  *reverse* order of group name, so the world group ("0") goes **last**, after
  every group created later (2.8.0 line 2186, 2.14.1 line 3020), then clears
  its maps.
- **Autoloading.**
  - At the end of `import torch`, `_import_device_backends()` calls every entry
    point in the `torch.backends` group, unless `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
    (`torch/__init__.py`: lines 2805 and 2866 in 2.8.0, 3469 and 3529 in 2.14.1).
    An exception there, including failing to import the entry point's module,
    makes `import torch` itself fail.
  - 2.14.0 added a lazy group, `torch.distributed.backends`, loaded only when
    an unknown backend name is first asked for: by `init_process_group`,
    `new_group` or `is_backend_available`, not by `dist.Backend(name)`
    (`Backend._ensure_backend_registered`, `distributed_c10d.py` line 543 in
    2.14.1). A failure there fails only that call. It isn't in 2.13.0's
    source, nor in 2.8.0.

## What DDP and FSDP2 call

Measured with a logging Python process group registered as a backend, on
CPU, in both versions (2 ranks, a 3-layer model, 2 training steps each; the
calls were identical on 2.8.0 and 2.14.1 apart from the renames below):

| who | when | process group method | how it gets there |
| --- | --- | --- | --- |
| DDP | constructor | `allgather` (list form) | `dist._verify_params_across_processes` (`torch/nn/parallel/distributed.py`), checking parameter shapes |
| DDP | constructor | `broadcast` per bucket | `_sync_module_states` → `dist._broadcast_coalesced` (C++) |
| DDP | every step | `allreduce`, one flat bucket tensor per call | the C++ `Reducer`'s default comm hook |
| DDP | after the first step | `broadcast` (x2) | rebuilding buckets: rank 0's bucket order |
| FSDP2 | every step | `_allgather_base` (2.8) / `all_gather_single` (2.14) | `dist.all_gather_into_tensor` / `dist.all_gather_single` in `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py` |
| FSDP2 | every step | `_reduce_scatter_base` (2.8) / `reduce_scatter_single` (2.14) | `dist.reduce_scatter_tensor` / `dist.reduce_scatter_single`, same file |
| FSDP2 (HSDP) | every step | `allreduce` on the replicate group | same file; needs a 2-D mesh ("Subgroups and HSDP", below) |
| any | `dist.barrier()` | `barrier` | |
| any | `dist._coalescing_manager` | `allreduce_coalesced`, `allgather_into_tensor_coalesced`, `reduce_scatter_tensor_coalesced`; with a device, `_start_coalescing` / `_end_coalescing` around individual calls | `distributed_c10d.py` |

Other measured facts:
- A 1-D `DeviceMesh` over every rank (what `fully_shard` builds without a mesh)
  reuses the default group: `mesh.get_group() is _get_default_group()`.
- Creating groups with `dist.new_group` inside the process group's first
  collective works in both versions: the default group exists by then.
- 2.14.1 renamed the process group's `_allgather_base` and
  `_reduce_scatter_base` to `all_gather_single` and `reduce_scatter_single`.
  A backend for both versions implements both names.

## Design

**Registration.**
- `import gpubridge` registers the backend name `"gpubridge"`, with
  `extended_api=True` and devices `cpu` and `cuda` (ROCm builds use `cuda`).
- On torch 2.14 and later, installed gpubridge also registers it through the
  lazy `torch.distributed.backends` entry point, when `backend="gpubridge"` is
  first asked for. There a script's only gpubridge line can be the backend
  name, and nothing of gpubridge runs otherwise.
- Before 2.14 (2.8 included), scripts need `import gpubridge` before
  `init_process_group`.
- **No `torch.backends` entry point.** It would register the name on older
  torch too, but torch runs that group at every `import torch`, and anything
  that goes wrong there makes `import torch` fail: an install whose metadata
  outlives its files (an editable install of a deleted worktree, say) is
  enough, whatever the hook catches. A library must never be able to break
  `import torch`, so gpubridge doesn't use it.

**Process groups: the world and subgroups.** The creator returns a
`gpubridge.torch_backend.GpubridgeProcessGroup` for the world group and for
every subgroup (`new_group`, the groups of a multi-dimensional `DeviceMesh`).
A group of every rank shares the world's topology. A subgroup gets a topology
of its own, over its members only ("Subgroups and HSDP", below). Ranks
outside a group get what torch gives any non-member.

**Setup at the first collective.** gpubridge needs its own groups: a Gloo group
of every rank (discovery, `barrier()`, `flat-gloo`), one per island on its
native backend, and the bridge. `dist.new_group` can't run inside
`init_process_group`, because the default group doesn't exist yet. So the
process group builds them at its first collective. Every rank makes the same
collectives in the same order, so every rank creates the same groups at the
same point. Discovery is gpubridge's usual one: every rank's vendor, build,
GPU and settings are exchanged, and any problem fails that first collective on
every rank with gpubridge's usual message. The policy and bridge transport,
which `gpubridge.init()` takes as arguments, come from `GPUBRIDGE_POLICY` and
`GPUBRIDGE_BRIDGE` (default `auto` and `gloo`), checked across ranks like the
other settings. A setting one rank can't read at all (an unknown policy name,
a malformed `GPUBRIDGE_*` value) also fails every rank. The Gloo groups are
created first, and every rank's error is exchanged over them before
anything else, so no rank is left waiting in `new_group`. These groups are
named like a subgroup's (below), `gpubridge:world.{ranks}.{k}`, not from
torch's count of the groups each process has created: the ranks can reach the
first collective having created different numbers of groups (torch 2.9.1's
`DeviceMesh` creates one for a (2, 1) mesh's replicate dimension where 2.14.1's
reuses the default group, and this backend's groups are created without an
exchange), and torch's names would then differ between them.

**Collectives.**
- Each torch collective maps onto gpubridge's public collective of the same
  meaning, called with `async_op=True`. The returned torch `Work` wraps the
  gpubridge `Work`'s future.
- Stream semantics are therefore gpubridge's async ones
  (`src/gpubridge/work.py`):
  - The collective waits for the caller's stream at submission.
  - `wait()` and the future make the waiting stream wait for the result.
  - Unlike NCCL, `wait()` also blocks the host until gpubridge's worker thread
    has run the collective. So FSDP2's prefetching gets less overlap of
    compute and communication than it would with NCCL.

| torch call | gpubridge |
| --- | --- |
| `allreduce` (SUM, AVG, MAX, MIN), `allreduce_coalesced` | `all_reduce`, per tensor |
| `broadcast` | `broadcast` (`rootRank` is the global rank: the world group) |
| `_allgather_base`, `all_gather_single`, `allgather_into_tensor_coalesced` | `all_gather_into_tensor` |
| `allgather` (list form) | `all_gather_into_tensor` into one buffer, then copied out |
| `_reduce_scatter_base`, `reduce_scatter_single`, `reduce_scatter_tensor_coalesced` (SUM, AVG) | `reduce_scatter_tensor` |
| `reduce_scatter` (list form) | the inputs concatenated, then `reduce_scatter_tensor` |
| `barrier` | `barrier` |
| `gather`, `scatter` | CPU tensors only, on the backend's Gloo group (below) |
| `_start_coalescing` / `_end_coalescing` | calls run as they come; `_end_coalescing` returns a `Work` for all of them |

- **CPU tensors.** gpubridge collectives take tensors on the rank's device.
  So in a GPU job, CPU tensors (some libraries all_reduce CPU flags or counts)
  go to a Gloo group of every rank that belongs to the backend. It is kept
  apart from the one gpubridge's worker thread uses, so the two threads never
  interleave collectives on one group. AVG there is a SUM and a division, as
  in gpubridge. `backend="gpubridge"` therefore behaves like
  `"cpu:gloo,cuda:gpubridge"` on every supported torch release.
- **Gather and scatter** of CPU tensors take the same group, in any job.
  That's what `gather_object` and `scatter_object_list` use, and
  `torch.distributed.checkpoint` plans its saves with them. The object
  collectives are tested; a full checkpoint save isn't.
- **Refused before any communication**, with an error naming the operation and
  the backend:
  - reductions other than SUM, AVG, MAX and MIN (PRODUCT, PREMUL_SUM, bitwise),
    on the CPU path too;
  - any-source `recv`, `alltoall` and `reduce`;
  - gather and scatter of GPU tensors;
  - list-form collectives with tensors of different sizes.

  The arguments are the same on every rank of a matching collective, so every
  rank refuses alike.

**Teardown.**
- `destroy_process_group()` calls the backend's `shutdown()` last, after it has
  shut down gpubridge's own groups. So `shutdown()` releases gpubridge's state
  without running anything still queued: those collectives fail.
- It waits at most 10 s for gpubridge's worker thread, and never raises, so
  `destroy_process_group()` always finishes and `init_process_group()` can run
  again in the same process.
- `abort()` does the same.
- In one process, use either `gpubridge.init()` or the backend, not both. With
  the backend, `gpubridge.all_reduce` and the other gpubridge functions also
  work on the groups it built.
- A custom bridge transport or policy that ran a collective on the default
  group during setup would wait on the backend's own setup. That is refused
  with an error, pointing at `topology.world_group`.

**gpubridge internals.** Two changes, invisible to the existing API:
- `Topology` gains `world_group`: a new Gloo group of every rank under the
  backend. After `gpubridge.init()` it is None, meaning the default group,
  which is looked up when used rather than held.
- Discovery, `barrier()` and `flat-gloo` use `world_group` instead of the
  default group, which under the backend is the backend itself.

## Subgroups and HSDP

**A subgroup is a smaller gpubridge job.** gpubridge's collectives compose
over islands: one native group per vendor, joined by a bridge between island
leaders. A subgroup gets the same structure over its members only:
- **Its islands** are its members grouped by vendor, from the labels the
  world's discovery already settled (a split test's labels included), so a
  subgroup runs no discovery of its own.
- **Single-vendor members** make one island and no bridge: the `auto` policy
  resolves to `native-only`, one NCCL/RCCL collective (Gloo in simulation),
  with no host staging and no bridge.
- **Mixed members** make a bridged topology like the world's: islands
  reduce onto their leaders, the leaders' bridge carries the cross-vendor
  step, and leaders broadcast back. The leaders are the subgroup's own, its
  lowest member of each vendor.
- **Policy and transport.** Single-vendor members always run `native-only`.
  Mixed members run the policy the job asked for (`GPUBRIDGE_POLICY`; `auto`
  resolves to `reduce-bridge-broadcast`), over the job's bridge transport
  (`GPUBRIDGE_BRIDGE`).
- **Group names.** torch names groups created with local synchronization from
  state that differs between ranks of asymmetric groups (2.8: how many groups
  the rank has; 2.14: a counter only creating ranks advance). So a
  subgroup's own groups are named from the subgroup's name, their ranks, and
  how many groups of the same ranks it created before: the same on every
  member, and the job's later group names stay in step.
- **That naming uses a private torch function.** While a subgroup sets up,
  gpubridge replaces `torch.distributed.distributed_c10d._process_group_name(ranks,
  use_hashed_name)` and passes `use_local_synchronization=True` to
  `dist.new_group`. A torch release can change private functions without
  notice. If the patch then missed or broke group naming, members could wait
  on different store keys until the group's timeout: a hang, not an error.
  So creating a subgroup first checks three things: the function exists with
  exactly that signature, `dist.new_group` still names groups with it
  (directly or through a `distributed_c10d` function it calls), and
  `new_group` still takes `use_local_synchronization`. If one fails,
  `new_group(..., backend="gpubridge")` raises on the subgroup's members,
  naming the torch version and what changed, and nothing is patched. Groups of
  every rank don't need the patch and keep working. Checked on torch 2.8.0
  and 2.14.1. `tests/test_torch_backend.py` simulates each change: a
  parameter added, renamed or made keyword-only, the function gone, and
  `new_group` naming groups another way or without local synchronization.
- **Ranks.** Collectives keep global ranks wherever torch wants them (the
  source of a broadcast; inside gpubridge, the destination of an island's
  reduce onto its leader). A subgroup's ranks must be ascending, as
  `new_group` sorts them by default; `sort_ranks=False` is refused. Rows of an all_gather
  and slices of a reduce_scatter follow the subgroup's own rank order, its
  members in ascending global rank, as torch orders a group.

**When its groups are created.** At the subgroup's first collective, like the
world's, and only by its members, with `new_group(...,
use_local_synchronization=True)`. torch calls the creator on members only, so
non-members can't take part in a group creation that every rank would have
to join.
- **Same order on every member:** a Gloo group of the members (barrier,
  `flat-gloo`, CPU tensors), the islands in leader order, then the bridge.
  The layout is a pure function of the members and the world's labels, so
  every member creates the same groups in the same order.
- **The world comes first.** A subgroup's first collective also sets up the
  world's topology if no collective has yet, and that setup takes every rank
  (limit 2 below).
- **One worker per rank.** Every group's async collectives run on the same
  worker thread, in the order the training thread issued them, as NCCL runs
  them on a stream. Two groups whose members issue collectives in opposite
  orders would deadlock there, as they would with NCCL.

**HSDP on a mixed cluster: shard within islands, replicate across vendors.**
FSDP2's HSDP takes a 2-D mesh `(replicate, shard)`. It shards parameters and
gradients over the shard dimension, all-gathering parameters before forward
and backward and reduce-scattering gradients after backward. It all-reduces
the reduce-scattered gradient shard over the replicate dimension once per
step. With the shard groups set to the vendor islands:

```python
# 2 islands of 4 GPUs, island ranks contiguous: rows are islands
mesh = init_device_mesh(device, (2, 4), mesh_dim_names=("replicate", "shard"))
fully_shard(model, mesh=mesh)
```

- **The shard groups are single-vendor.** Each is a `native-only` topology,
  so every all-gather and reduce-scatter is one NCCL or RCCL call inside a
  vendor's fabric (NVLink, xGMI, the vendor's network). None of it touches the
  bridge or the host.
- **The replicate groups are mixed.** Each joins rank i of every island, so
  its topology is bridged. Once per step, its all_reduce of a gradient shard
  crosses vendors: a reduce within each island (one rank per island in this
  layout, so nothing to do), then the bridge between the leaders, then a
  broadcast back.
- **What crosses the bridge:** one all_reduce per step per replicate group,
  of that rank's gradient shard: the model's gradients divided by the shard
  size. Summed over the replicate groups, the gradients cross the bridge once
  per step, in as many parallel streams as there are ranks per island. The
  frequent, latency-bound traffic, every parameter all-gather in forward and
  backward and every gradient reduce-scatter, stays inside each vendor's
  fabric.
- **Why this layout.** The bridge is the slow link: host staging, Gloo or UCX
  between leaders. Plain FSDP2 over the whole mixed job puts every
  all-gather and reduce-scatter on it, several per layer per step. Shard
  groups that span vendors would do the same. With shards inside islands, the
  bridge sees only the replicate all_reduce, once per step, overlapped with
  the next step's compute as FSDP2 schedules it. The cost is memory: each
  island holds a full copy of the model state, sharded over its own ranks
  only, so an island must be big enough to hold one.

**Two limits.**
1. **Islands must be contiguous blocks of ranks, all the same size.** That
   means the same number of GPUs from each vendor, with the ranks ordered by
   vendor: all of one vendor's ranks, then all of the next's. A Slurm
   heterogeneous job with one component per vendor gives exactly that, since
   its ranks follow the components in order. `init_device_mesh` lays out ranks
   row by row, and each row is a shard group, so only then is every shard
   group one island. With unequal islands there is no 2-D mesh to build. With
   ranks interleaved, the shard groups span vendors: HSDP still runs, but every
   all-gather and reduce-scatter crosses the bridge, which is what this layout
   is meant to avoid.
2. **The job's first collective must include every rank.** gpubridge sets
   itself up at the first collective on any of its groups, and that setup
   takes every rank. DDP, FSDP2 and HSDP meet this: their first collectives
   run on every rank. A job whose first collective is on a subgroup of some
   ranks would hang there until the group's timeout. Call `dist.barrier()`
   on the world group first.

## Point-to-point (CPU only)

`dist.send`, `recv(src=...)`, `isend`, `irecv` and `batch_isend_irecv` use
the group's `send` and `recv` methods. They accept one contiguous dense
tensor on CPU or on the rank's configured GPU. Finish setup with a world
`dist.barrier()` first; a subgroup also needs its own first collective.
P2P refuses before setup, because setup needs every member, even members
that will not participate in the subsequent exchange.

Within an island, the call goes straight to the native process group
(NCCL/RCCL on GPU, Gloo in simulation). CPU tensors in a GPU job use Gloo.
Between islands, `BridgeTransport.create_p2p` supplies an all-member link;
Gloo reuses the existing discovery group and transfers host-staged raw bytes.
This is separate from the leader-only collective link and does not create
another process group. Collective-only custom transports need to implement
the optional hook before carrying cross-island P2P.

With `GPUBRIDGE_BRIDGE=ucx`, each matched directional `(src, dst, tag)` lane
lazily connects its own two-endpoint UCX worker, including non-leaders. This
keeps the existing collective worker separate (it is not thread-safe) and
lets unrelated pairs/tags progress concurrently. Address exchange uses the
same scoped Store; it does not rendezvous with uninvolved ranks or create
another ProcessGroup. Payloads are host-staged, including pinned/chunked
options; there is no device-direct UCX P2P path. Separate workers cost a UCX
context and endpoints per used directional tag for the group's lifetime.
Use a bounded set of tags. The adapter is CPU tested over UCX TCP; RDMA and
GPU behavior are unvalidated. Closing schedules worker cleanup after any
native wait has completed or failed, without blocking the main thread.

`GPUBRIDGE_PINNED_STAGING=1` uses pinned host buffers on GPUs. Setting
`GPUBRIDGE_P2P_CHUNK_BYTES=4M` instead enables two-slot chunked staging, also
pinned on GPU: copying the next chunk can overlap the current transfer, and
copying received chunks back can overlap the following receive. This setting
must be the same at both endpoints and is checked in the message handshake.
Unset means one whole message; it does not inherit the collective policy's
chunk size. CPU tests use the same schedule without pinning. P2P is always
bit-exact and does not apply collective gradient compression. No GPU overlap
or speedup has been measured.

- **Matching.** Across islands, a message matches by group, source,
  destination and tag (integer `0` through `2**31 - 1`). Same-tag sends and
  receives match in posting order within each direction of a pair. Different
  tags and pairs run independently, including a full-duplex batched exchange.
  Sizes and dtypes must agree; shape may differ with the same element count.
  A group-scoped Store handshake validates byte count and dtype before posting
  payloads, so mismatches raise instead of triggering Gloo's fatal truncation.
  Metadata keys are removed after being read. Within an island, matching is
  the native backend's: NCCL/RCCL ignore tags; Gloo respects them.
- **Streams and lifetime.** Submission records an event on the caller's
  stream. A lane's own worker stream waits for it, and the tensor is kept
  alive and marked with `record_stream` until the operation has completed.
  `wait()` orders the caller's current stream after the final copy. The
  future is device-aware; `is_completed()` includes GPU completion. Do not
  modify or read an in-flight tensor before waiting. P2P has no ordering
  guarantee relative to collectives or other pairs.
- **Timeout and failure.** Each cross-island operation has the process group's
  timeout from submission, including time spent queued. A mismatch, deadline
  or transport error fails pending P2P handles locally and publishes a group
  failure. Ranks with an active engine observe this within the health polling
  interval (50 ms plus Store latency); all other members refuse their next
  backend operation. An idle dead peer is detected when an operation waiting
  for it reaches its deadline. There is no recovery: destroy the process group
  and restart the job. `Work.wait(timeout=...)` has a separate host timeout;
  it does not cancel the transfer. Same-island errors retain native behavior.
- **Teardown.** Pending handles fail and the engine joins lane threads for at
  most 200 ms total. In-flight native requests and their buffers stay owned by
  daemon lane threads until the transport errors, its timeout passes or torch
  destroys the group. Teardown never waits for a missing peer to post a receive.
- **Native GPU batches.** On torch 2.14, `dist.batch_isend_irecv` coalesces
  gpubridge's native NCCL/RCCL operations and combines their batch Work with
  the cross-island Works. torch 2.8 excludes Python ProcessGroup subclasses
  from its coalescing path. On that release, use
  `gpubridge.batch_isend_irecv(ops)` for a batch containing same-island GPU
  peers: it supplies the batch boundary without modifying torch's global
  functions. The explicit helper also works on 2.14 and delegates unchanged
  for other backends. Plain `dist.batch_isend_irecv` with only cross-island
  peers works on either release. Native GPU batch behavior remains designed,
  not GPU validated; the helper and batching paths are tested on CPU.

Tests cover 1+1, 2+2, uneven and three-island simulation layouts, every pair
(including non-leaders), zero/small/large payloads and dense dtypes, reverse
tag order, simultaneous pairs, full-duplex same-tag batches, subgroup ranks,
a slow receiver, missing peers and size/dtype mismatch propagation.

## What isn't supported, and why

- **Subgroups:** see "Subgroups and HSDP" above for what works. Tensor
  parallelism hasn't been tested.
- **`torch.compile`:** not tested. Functional collectives
  (`torch.ops._c10d_functional`) do reach the Python group: FSDP2's
  `full_tensor()` uses them, and it is tested.
- **Any-source receive, all-to-all, and gather and scatter of GPU tensors:**
  gpubridge has no cross-vendor implementation of them yet.
- **Overhead:** each collective goes through Python and gpubridge's worker
  thread, as `gpubridge.all_reduce(async_op=True)` does today. CPU
  measurements of both are in the README, "Per-call overhead on small
  messages".

## How it is tested

In simulation mode (CPU processes with faked vendors, two islands and a Gloo
bridge):
- **DDP** with `backend="gpubridge"` trains bit for bit like a Gloo-only run on
  integer-valued data. Also with BatchNorm buffers, unused parameters and a
  comm hook, within float64 rounding of Gloo.
- **FSDP2** (`fully_shard`) matches Gloo and a single-process reference bit for
  bit in float64. On the float32 (AVG) path it matches a single process within
  float32 rounding. Gloo isn't the reference there: torch 2.8.0's Gloo
  `reduce_scatter` returns the sum for AVG (measured), so FSDP2 on Gloo doesn't
  average.
- **Collectives:** the coalesced and bucketed paths, and the object
  collectives. The CPU-tensor path, which simulated jobs don't take on their
  own, forced: the same collectives on the backend's Gloo group.
- **Refusals:** unsupported operations fail on every rank.
- **Subgroups** (simulation, 2+2 ranks): single-vendor ones (native-only, no
  bridge), a mixed pair and a mixed group of three with an island of one
  (all_reduce, all_gather and reduce_scatter rows in group-rank order,
  broadcast from a global rank, barrier, object collectives), the world after
  them. HSDP on a `(2, 2)` mesh whose rows are the islands: bit for bit like
  one process; every all-gather and reduce-scatter native-only with no bridge
  phase; one bridged all_reduce per reduce-scatter; bridge transport calls
  only for those. A setting one rank can't read fails every rank, and so does
  a collective during setup.
- **Lifecycle:** init twice in one process, and destroy with collectives still
  queued.
- **GPU tests** (DDP on two GPUs, `device_id`) are written and skipped without
  GPUs.
- **Examples:** the FSDP2 example's `--verify` mode compares against a
  single-process replay bit for bit. The GPU kit runs it, so a stream race on
  GPUs would show even when every rank agrees.

HARDWARE_VALIDATION.md lists what a GPU run must show.
