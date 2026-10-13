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
    `new_group`, `dist.Backend(name)` or `is_backend_available`
    (`Backend._ensure_backend_registered`, `distributed_c10d.py` line 543 in
    2.14.1). A failure there fails only that call. It doesn't exist in 2.13.0
    or 2.8.0.

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
| FSDP2 (HSDP) | every step | `allreduce` | same file; needs a 2-D mesh, which gpubridge doesn't support (below) |
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

**One process group, the world.** The creator returns a
`gpubridge.torch_backend.GpubridgeProcessGroup` for the world group.
gpubridge's collectives run on the world group only, so a subgroup
(`new_group`, or a multi-dimensional `DeviceMesh`) fails where it is created:
on each of its members, saying so. Ranks outside it get what torch gives
any non-member. A group of every rank is allowed; it shares gpubridge's
topology.

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
anything else, so no rank is left waiting in `new_group`.

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
  - point-to-point `send` and `recv`, `alltoall` and `reduce`;
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

## What isn't supported, and why

- **Subgroups:** HSDP, 2-D meshes, tensor parallelism and any `new_group`. gpubridge
  composes collectives over its islands of the whole job; a subgroup would
  need its own islands and bridge. Possible later; not planned yet.
- **`torch.compile`:** not tested. Functional collectives
  (`torch.ops._c10d_functional`) do reach the Python group: FSDP2's
  `full_tensor()` uses them, and it is tested.
- **Point-to-point, all-to-all, and gather and scatter of GPU tensors:**
  gpubridge has no cross-vendor implementation of them yet.
- **Overhead:** each collective goes through Python and gpubridge's worker
  thread, as `gpubridge.all_reduce(async_op=True)` does today.

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
- **Refusals:** unsupported operations fail on every rank, and subgroups on
  their members. A setting one rank can't read fails every rank, and so does
  a collective during setup.
- **Lifecycle:** init twice in one process, and destroy with collectives still
  queued.
- **GPU tests** (DDP on two GPUs, `device_id`) are written and skipped without
  GPUs.
- **Examples:** the FSDP2 example's `--verify` mode compares against a
  single-process replay bit for bit. The GPU kit runs it, so a stream race on
  GPUs would show even when every rank agrees.

HARDWARE_VALIDATION.md lists what a GPU run must show.
