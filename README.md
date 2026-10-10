# gpubridge

gpubridge lets one distributed PyTorch job span NVIDIA and AMD GPUs at the
same time. Choosing between NCCL and RCCL on a single-vendor machine is already
solved: PyTorch's `"nccl"` backend runs NCCL on CUDA builds and RCCL on ROCm
builds, and Modular's MAX loads NCCL or RCCL to match the GPU it was built for.
Neither connects the two, because an NCCL communicator and an RCCL
communicator cannot exchange data, and Modular states that mixed-vendor hosts
are not supported. Meta's torchcomms lists mixed-vendor jobs as a design goal,
but each of its communicators still runs one vendor's library. Research
systems published in 2026 (two called HetCCL, and Zettabyte's joint AMD and
NVIDIA training) do join the vendors, moving data GPU to GPU over RDMA, and
report much higher cross-vendor bandwidth than a Gloo-based bridge like
gpubridge's. Their code isn't available (October 2026), and they rely on
RDMA-capable NICs for that speed and on custom builds or plugins. gpubridge is
the open-source option that runs on stock PyTorch builds over any network. It
keeps each vendor on its native library inside an island and joins the islands
with a CPU bridge. One `all_reduce`, `broadcast`, `all_gather_into_tensor`,
`reduce_scatter_tensor` or `barrier` call then covers every GPU in the job,
from the same code on every node, and DistributedDataParallel can sync
gradients through it with a comm hook. CUDA and ROCm builds of PyTorch, from
2.9.1 to 2.14.1, have been shown to join one job and complete the bridge on
CPU; validation on real GPUs is next.

gpubridge writes no CUDA or HIP code. It composes collectives that PyTorch
already provides. For how it relates to Modular MAX, RAJA, Triton,
Triton-distributed, Meta's torchcomms and the 2026 papers, see
[docs/PRIOR_ART.md](docs/PRIOR_ART.md).

> **Status: v0.2 alpha, correctness first.** All orchestration logic is tested
> in CPU simulation mode, and the CPU side of a mixed job has been checked with
> real CUDA and ROCm builds. It has not yet run on a real mixed cluster; see
> [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for the results and what
> still needs checking. The default path is reduce-bridge-broadcast. The
> opt-in policies (pipelined, sharded, auto-tuned) and the DDP comm hook are
> tested on CPU only and are **not validated on GPUs**.
> [docs/ROADMAP.md](docs/ROADMAP.md) holds designs that wait on GPU results.

## How it works

A PyTorch install is either a CUDA build or a ROCm build, so every process
belongs to exactly one vendor. gpubridge groups processes by vendor:

- **Island:** every rank of one vendor, in a process group on the native
  backend. That is NCCL on NVIDIA, and RCCL on AMD, which ROCm builds of
  PyTorch still call `"nccl"`.
- **Bridge:** a CPU link between one **leader** per island, where the leader
  is the island's lowest global rank. By default it is a Gloo process group;
  the transport can be swapped (see [Bridge transports](#bridge-transports)).

```
        NVIDIA island (NCCL)                   AMD island (RCCL)
   ┌──────────────────────────────┐      ┌──────────────────────────────┐
   │  rank 0*   rank 1   rank 2   │      │  rank 3*   rank 4   rank 5   │
   └────┬─────────────────────────┘      └────┬─────────────────────────┘
        │                                     │
        └──────── Gloo bridge (on CPU) ───────┘          * island leader
```

**`all_reduce`** uses reduce-bridge-broadcast:

1. `reduce` within each island onto its leader, on the native backend. Only
   the leader needs the island sum, and a reduce moves about half the bytes of
   an all_reduce.
2. Leaders copy the island sum to CPU and `all_reduce` it over the bridge.
3. Each leader copies the total back to its GPU and broadcasts it to its island.

**`broadcast(tensor, src)`** takes three hops: from `src` to the rest of its
island, from that island's leader over the bridge to the other leaders, then
from each leader to the rest of its island.

If the cluster has only one vendor there is no bridge, and every call goes
straight to the native collective.
These paths are the default [collective policies](#collective-policies).
Opt-in policies change how data travels (in chunks, or over several bridge
links at once), never what a collective computes, and `auto` never picks them.

### Initialization

`gpubridge.init()` does the following:

1. Creates the default process group on **Gloo** across all ranks. NCCL cannot
   span both vendors, so this CPU group handles discovery and `barrier`.
2. Each rank detects its vendor from `torch.version.hip` (AMD) or
   `torch.version.cuda` (NVIDIA), and runs `gpubridge.probe()`, which never
   raises: build, visible GPUs, and whether the NCCL and Gloo backends exist.
   Anything that would stop the rank (no vendor, no GPU, no NCCL/RCCL, or a
   `LOCAL_RANK` with no visible GPU behind it) is recorded as a problem instead
   of being raised on that rank alone. `torch.cuda.set_device` waits until
   discovery has passed, so a bad device index can't crash one rank and leave
   the rest waiting in rendezvous.
3. Every rank shares its vendor, probe, problems, hostname and chosen bridge
   transport with `all_gather_object`. If any rank reported a problem, or the
   ranks disagree on CPU versus GPU mode or on the bridge transport, `init()`
   raises the same error on every rank, listing every problem. Nobody is left
   waiting for a rank that already gave up.
4. Each rank computes the same islands, leaders and bridge from that shared
   map. Every rank calls `new_group` for every group in the same order, as
   `torch.distributed` requires, and the bridge transport is created the same
   way.

## Install

gpubridge isn't on PyPI. Install it with pip from the repository:

```bash
pip install git+https://github.com/DavidCzartoryski/gpubridge
```

or, for development:

```bash
uv venv
uv pip install -e ".[test]"
```

On GPU nodes, install the CUDA or ROCm build of PyTorch first. gpubridge only
needs `torch>=2.3` (and numpy), so it leaves an existing install alone.

## Usage

```python
import torch
import gpubridge

topology = gpubridge.init()  # env:// rendezvous, e.g. from torchrun

grad = torch.randn(1024, device=topology.device)
gpubridge.all_reduce(grad)              # summed across NVIDIA and AMD ranks
gpubridge.broadcast(grad, src=0)
gpubridge.barrier()

gpubridge.destroy()
```

Launch the same script on every node with `torchrun`; each node detects its
own vendor. See [`examples/all_reduce_torchrun.py`](examples/all_reduce_torchrun.py).

For a fuller demo, [`examples/train_synthetic.py`](examples/train_synthetic.py)
trains a small model with DistributedDataParallel and syncs its gradients
through gpubridge with `gpubridge.ddp_comm_hook`. It reports samples/sec and
checks that every rank ends with identical parameters:

```bash
GPUBRIDGE_VENDOR=nvidia torchrun --nproc-per-node=2 examples/train_synthetic.py  # on CPU
```

## Simulation mode (no GPUs needed)

Set `GPUBRIDGE_VENDOR=nvidia` or `GPUBRIDGE_VENDOR=amd` on a process to fake
its vendor. With the variable set:

- islands use Gloo instead of NCCL/RCCL, and
- tensors stay on CPU (`topology.device` is `cpu`).

Everything else is the code path used on real GPUs: discovery, leader
election, group creation, and reduce-bridge-broadcast. Set the variable on
every rank or on none; `init()` refuses a mix of CPU and GPU ranks.

Run the four-rank demo (2 NVIDIA + 2 AMD):

```bash
python examples/simulate_two_islands.py
```

```
island nvidia ranks=[0, 1] leader=0
island amd    ranks=[2, 3] leader=2
bridge ranks=[0, 2]

rank 0 (nvidia, leader): [10.0, 10.0, 10.0, 10.0]
rank 1 (nvidia, member): [10.0, 10.0, 10.0, 10.0]
rank 2 (amd, leader): [10.0, 10.0, 10.0, 10.0]
rank 3 (amd, member): [10.0, 10.0, 10.0, 10.0]

expected 10.0 on every rank: all ranks match
```

Or act as two nodes of different vendors with `torchrun`, using two terminals.
The commands are in the docstring of `examples/all_reduce_torchrun.py`.

### CPU-only mode (real builds, no GPUs)

`GPUBRIDGE_CPU_ONLY=1` also runs on CPU with Gloo islands, but keeps the real
vendor detection: a CUDA build of PyTorch reports NVIDIA and a ROCm build
reports AMD, even on a machine with no GPU. This is how
[`scripts/mixed_build_rendezvous.sh`](scripts/mixed_build_rendezvous.sh) checks
that CUDA-build and ROCm-build processes can share one job. If both variables
are set, `GPUBRIDGE_VENDOR` wins. As with simulation mode, `init()` refuses a
job where some ranks run on CPU and others on GPUs.

### Split-test mode (test only)

`GPUBRIDGE_SPLIT_TEST=half|alternate|node` (`1` means `half`) labels some ranks
of a single-vendor job as the other vendor. They keep their real GPUs and real
NCCL/RCCL, so one NVIDIA (or AMD) cluster can run two islands and the bridge on
device tensors. Only NCCL talking to RCCL goes untested.

The environment variable is the only switch, and split runs can't pass for
mixed-vendor ones:

- every rank logs a `SplitTestWarning`;
- `Topology.split_test` and `Topology.run_kind` (`"split-test"`) mark the run;
- the GPU validation kit flags these runs in every JSON record and summary.

`node` mode splits by machine. In split-test mode only, ranks may share a GPU.

## API

The API follows `torch.distributed`, always on the world group.

| Function | Description |
| --- | --- |
| `init(init_method=None, *, rank=None, world_size=None, timeout=None, bridge="gloo", policy="auto") -> Topology` | Join the cluster and build the islands and bridge. Same arguments as `init_process_group`, plus the bridge transport (a registered name or a `BridgeTransport` subclass) and the [collective policy](#collective-policies) (`"auto"`, a registered name, or a `CollectivePolicy` subclass). Both must be the same on every rank. |
| `all_reduce(tensor, op=ReduceOp.SUM, *, async_op=False)` | In-place reduction across all ranks: `SUM`, `AVG`, `MAX` or `MIN`. `AVG` is a SUM divided by the world size, identically on every rank, so it needs a floating-point dtype. |
| `broadcast(tensor, src, *, async_op=False)` | In-place copy from global rank `src` to all ranks. |
| `all_gather_into_tensor(output, input, *, async_op=False)` | Every rank's `input`, back to back in global rank order, in `output` (`world_size` times as many elements). |
| `reduce_scatter_tensor(output, input, op=ReduceOp.SUM, *, async_op=False)` | Rank `r` gets slice `r` of the reduced `input` (`SUM` or `AVG`). |
| `ddp_comm_hook(state, bucket)` | A DistributedDataParallel communication hook that averages each gradient bucket with an async gpubridge `all_reduce`. See [DistributedDataParallel](#distributeddataparallel). |
| `barrier(*, async_op=False)` | Wait for all ranks. On GPUs, first waits for this rank's queued GPU work. |
| `destroy()` | Tear down every group `init` created. |
| `get_topology() -> Topology` | This rank's view of the cluster (see below). |
| `is_initialized() -> bool` | True between `init` and `destroy`. |
| `probe() -> Probe` | This process's build vendor, CUDA/HIP versions, visible GPU count, and NCCL/Gloo availability. Never raises; failed checks are listed in `Probe.errors`. Works before `init`. |
| `add_observer(observer)` / `remove_observer(observer)` | Start or stop sending this rank's collective records to a `CollectiveObserver` (see [Observing collectives](#observing-collectives)). |

Every collective takes `async_op`. With `async_op=True` it returns a `Work`
(`wait()`, `is_completed()`, `exception()`, `get_future()`); see
[Async collectives](#async-collectives). `PRODUCT` is not supported yet:
NCCL/RCCL and Gloo still have to be shown to agree on it in every dtype.

Tensors must be contiguous and on `topology.device`: the rank's GPU, or CPU in
simulation mode. Their dtype must be one that both the island backend and Gloo
support (`gpubridge.collectives.SUPPORTED_DTYPES`): float16, bfloat16, float32,
float64, int8, uint8, int32 or int64. Anything else is rejected on every rank
before any communication. `bool` is excluded because a SUM of bools is a
logical OR on both backends.

`Topology` exposes `rank`, `world_size`, `vendor` (the island label),
`detected_vendor`, `run_kind` (`gpu-mixed`, `gpu-single-vendor`, `split-test` or
`cpu`), `split_test`, `device`, `simulated`, `island`, `is_leader`, `layout`
(every island, its ranks and leader), `peers`
(each rank's vendor, hostname, bridge transport and `probe` results),
`island_group`, `bridge` (the transport linking island leaders), and `policy`
/ `policy_name` (the active collective policy). `bridge_group` is the bridge's
process group when the transport has one, as the default Gloo transport does.

### Collective policies

A policy decides how a collective travels: which process groups carry the
data, and in which order. It never changes what the collective computes.
Every rank must use the same policy; `init()` fails on every rank otherwise,
or if the policy doesn't fit the cluster.

Every built-in policy carries every collective and reduction above. MAX and
MIN over the bridge also need the bridge transport to list them in
`reduce_ops`, as the default Gloo transport does. A custom policy offers
`all_gather_into_tensor` and `reduce_scatter_tensor` only if it overrides them;
otherwise those calls fail on every rank before any communication, naming the
policies that do.

| Policy | Applies to | What it does |
| --- | --- | --- |
| `reduce-bridge-broadcast` | 2+ islands | island reduce onto each leader, island leaders over the bridge, island broadcast |
| `native-only` | 1 island | one native NCCL/RCCL (or CPU Gloo) collective |
| `flat-gloo` | any | every rank copies to CPU and uses the world Gloo group: slow but obviously correct, so it's the test reference and a debugging fallback that never touches NCCL/RCCL |
| `auto` (default) | any | `native-only` for one island, `reduce-bridge-broadcast` otherwise; today's behaviour |
| `pipelined-reduce-bridge-broadcast` (opt-in, **not validated on GPUs**) | 2+ islands | reduce-bridge-broadcast with the leaders' bridge step cut into chunks: on GPUs the copy of chunk k+1 to pinned host memory, the bridge step of chunk k and the copy of chunk k-1 back overlap, ordered with CUDA events. Chunk size: `GPUBRIDGE_CHUNK_BYTES` (default `4M`) |
| `sharded-bridge` (opt-in, **not validated on GPUs**) | 2+ islands | k ranks per island share the bridge step instead of one leader, where k is the size of the smallest island: each island reduce_scatters (or reduces) the tensor onto k ranks, rank j of every island all_reduces segment j over its own bridge link, and each island all_gathers (or broadcasts) the result. With an island of one rank, k is 1 and it is reduce-bridge-broadcast. On CPU it is slower than reduce-bridge-broadcast; GPU numbers are HARDWARE_VALIDATION.md item 21 |
| `auto-tuned` (opt-in, **not validated on GPUs**) | any | picks one of the above by message size, from a thresholds file measured on the cluster (`GPUBRIDGE_THRESHOLDS`, written by `bench_all_reduce.py --write-thresholds`); without a file, it does what `auto` does |

The opt-in policies are never chosen by `auto`, and the default path is the
same as without them. Their settings (the chunk size, the thresholds) decide
how many messages a collective sends, so every rank reports them during
discovery, and `init()` fails on every rank if they differ or if one is
invalid.

```python
gpubridge.init(policy="flat-gloo")     # e.g. to rule out NCCL/RCCL while debugging
```

A custom policy subclasses `CollectivePolicy`, implements `applies_to`,
`all_reduce` and `broadcast`, and registers with `register_policy`. It may add
`all_gather_into_tensor`, `reduce_scatter_tensor`, and MAX/MIN (by listing them
in `reduce_ops` and taking `op=`). Optional hooks leave room for performance
work without changing the public API:
- `setup()` and `close()`, for staging buffers, streams or extra groups
  (`topology.timeout` is the timeout `init()` gave its own groups);
- `select_policy(nbytes)`, to choose a path by message size;
- `settings()`, a classmethod returning values read from the environment that
  every rank must share; discovery compares them.

The GPU kit's `check.py` and `bench_all_reduce.py` take `--policy`, so runs can
compare policies. `bench_all_reduce.py --candidates` times several policies in
one job, and `--write-thresholds` turns that into a file for `auto-tuned`.

### Async collectives

`async_op=True` validates the call at once, on the calling thread (so a bad
call still fails on every rank before any communication), then queues it on
this rank's gpubridge worker thread and returns a `Work`. The worker runs
queued collectives one at a time, in order. A synchronous collective first
waits for every queued one, so collectives always run in program order: the
same order on every rank.

```python
work = gpubridge.all_reduce(grad, op=gpubridge.ReduceOp.AVG, async_op=True)
...                                  # queue more GPU work; don't touch grad
work.wait()                          # grad is ready for kernels queued after this
```

Stream semantics on GPUs follow `torch.distributed` with NCCL:

- The collective starts after the work already queued on the caller's
  current stream (an event recorded at the call). Its tensors are marked with
  `record_stream`, so the caching allocator won't reuse them early.
- `wait()` blocks the host until the worker has run the collective, then makes
  the caller's current stream wait on an event recorded after it.
  `get_future()` returns a future created with the rank's device, so
  `future.wait()` synchronizes the same way.
- Between the call and `wait()`, don't read or write the tensors.

If an async collective fails, `wait()` raises with the original error as its
cause, and every later collective on that rank refuses to start, because the
process groups may be inconsistent. `destroy()` runs whatever is still queued
before tearing down.

### DistributedDataParallel

DDP needs a process group that spans every rank. In a mixed job the only one
is the Gloo world group `init()` creates, and on its own it would carry every
gradient over Gloo. Register `gpubridge.ddp_comm_hook` and DDP keeps its
bucketing and its overlap with backward, while each bucket is averaged by an
async `gpubridge.all_reduce` under the active policy:

```python
from torch.nn.parallel import DistributedDataParallel

topology = gpubridge.init()
model = DistributedDataParallel(model.to(topology.device))  # default group: gpubridge's Gloo world
model.register_comm_hook(None, gpubridge.ddp_comm_hook)
```

- DDP must span every rank, as gpubridge collectives do.
- Gloo still carries what DDP does outside the hook: the parameter shape check
  and the broadcast from rank 0 when DDP wraps the model, and buffer broadcasts
  before each forward pass with `broadcast_buffers=True` (the default). A
  model without buffers, or `broadcast_buffers=False`, keeps Gloo out of the
  training loop.
- If a bucket's all_reduce fails, `backward()` raises, and every later
  collective on that rank refuses to start.
- Tested on CPU; not validated on GPUs (HARDWARE_VALIDATION.md item 22).

This is a comm hook, not a `torch.distributed` backend: FSDP2 and DeviceMesh
can't use gpubridge yet. [docs/ROADMAP.md](docs/ROADMAP.md) sketches that
backend.

### Bridge transports

The bridge only moves CPU tensors between island leaders, so it can be swapped
without touching how collectives are composed. Gloo is the default. To use
something else, subclass `BridgeTransport`, register it, and pass it to `init()`
on every rank:

```python
from gpubridge import BridgeTransport, init, register_transport

@register_transport
class MyTransport(BridgeTransport):
    name = "my-transport"

    @classmethod
    def create(cls, ranks, *, timeout=None):
        # Called on every rank; return an instance on the leaders in `ranks`, None elsewhere.
        ...

    def all_reduce(self, tensor): ...          # in-place SUM of a CPU tensor
    def broadcast(self, tensor, src): ...      # in-place copy from global rank `src`
    # Optional: reduce_ops = frozenset({"sum", "max", "min"}) and all_reduce(tensor, op=...)
    # Optional: create_groups(groups, *, timeout=None), several links at once
    # (the default calls create() once per group)

init(bridge="my-transport")  # or init(bridge=MyTransport)
```

Every rank reports its transport's name during discovery, so a job where ranks
picked different transports fails at `init()` instead of hanging. Policies
with more than one bridge link, such as `sharded-bridge`, make the extra links
with the same transport through `create_groups`.
`tests/_transports.py` has a working example that carries the bridge over a
c10d `FileStore`.

### Observing collectives

Monitoring or profiling tools can watch every collective on a rank. After each
collective, an observer gets a `CollectiveRecord`:

- `seq`: the collective's number since `init()`, the same on every rank, so
  records from different ranks line up;
- `op`, `nbytes`, `dtype`, `src`, `policy` and `reduce_op` (`"sum"`, `"avg"`,
  `"max"` or `"min"` for reductions);
- `start` and `end` marks, and `phases`. Reduce-bridge-broadcast marks
  `island-reduce`, `bridge` (leaders only) and `island-broadcast`.
  `sharded-bridge` marks `island-reduce-scatter` and `island-all-gather`
  instead on islands of exactly k ranks, and `bridge` on every rank with a link.

```python
import gpubridge

class Timings(gpubridge.CollectiveObserver):
    def __init__(self):
        self.records = []

    def on_collective(self, record):
        self.records.append(record)  # keep it cheap; read the times later

gpubridge.init()
timings = Timings()
gpubridge.add_observer(timings)
# ... training ...
for rec in timings.records:
    if rec.ready():
        phases = {p.name: round(p.elapsed_ms(), 3) for p in rec.phases}
        print(rec.seq, rec.op, rec.policy, round(rec.elapsed_ms(), 3), phases)
```

The rules keep observers from changing what any rank communicates:

- **Observers never communicate.** `on_collective` must not call gpubridge or
  `torch.distributed`; gpubridge refuses its own collectives from inside a
  callback. A tool that needs other ranks' data exchanges it outside the
  callback.
- **Observers are local.** Sequence numbers count every collective, observed
  or not, so ranks can attach observers at different times.
- **A failing observer is detached on its own rank**, with a `RuntimeWarning`
  and a call to its `on_detach` hook. Collectives carry on on every rank.
- **Async collectives are observed when they run**, on the worker thread, in
  submission order.
- **Timing never syncs the GPU.** On GPUs a mark is a CUDA event recorded on
  the current stream, so check `record.ready()` before reading times. On CPU a
  mark is a `perf_counter_ns` reading. With no observers attached, gpubridge
  creates no events at all.

A custom policy can mark its own steps with `gpubridge.observe.phase("name")`.

## Building on gpubridge

gpubridge is meant to be a base for cluster tooling that has to understand
mixed hardware, such as monitoring or profiling tools:

- `get_topology()` tells each rank the vendor, island and host of every rank.
- [Observers](#observing-collectives) give every rank a timing record for
  each collective, with sequence numbers that match across ranks.
- Gloo works the same on both vendors and does not touch GPU streams. A tool
  can create its own Gloo group after `init()` (on every rank, in the same
  order) to carry monitoring data without disturbing training traffic.

## Limitations

- Reductions are `SUM`, `AVG`, `MAX` and `MIN` (no `PRODUCT` yet), and
  `reduce_scatter_tensor` takes `SUM` and `AVG` only. Collectives run on the
  world group (no custom groups). `async_op` runs collectives on one worker
  thread per rank, one at a time, so two async collectives never overlap each
  other.
- With NaN in the data, `MAX` and `MIN` follow whatever NCCL/RCCL and Gloo do,
  which may differ (HARDWARE_VALIDATION.md item 14).
- **Cross-vendor bandwidth is limited by design.** The bridge copies data to
  host memory and sends it with Gloo over TCP. The RDMA-based systems
  published in 2026 report much higher cross-vendor bandwidth (see
  [docs/PRIOR_ART.md](docs/PRIOR_ART.md#three-2026-papers-mixed-vendor-collectives-over-rdma)).
  gpubridge trades that for running on stock PyTorch builds over any network.
- On the default path, cross-vendor traffic goes through one leader per
  island, its CPU and Gloo, with no chunking, overlap or pinned memory. The
  opt-in `pipelined-reduce-bridge-broadcast` and `sharded-bridge` policies
  address that, but they are not validated on GPUs and may not be faster
  there (HARDWARE_VALIDATION.md items 17 to 21). Neither changes the limit
  above: both still go through host memory and Gloo.
- No custom kernels and no direct RDMA between vendors. A GPU-to-GPU transport
  would plug in through `BridgeTransport`.
- Designed but not implemented, each waiting on GPU results: bridge-only
  compression, an RDMA bridge transport, and a `torch.distributed` backend for
  FSDP2 and DeviceMesh. See [docs/ROADMAP.md](docs/ROADMAP.md).
- No GPU numbers yet: everything above is tested on CPU (see
  [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md)).

## Development

```bash
uv venv
uv pip install -e ".[test]"
pytest
```

The tests start real multi-process clusters on CPU with
`torch.multiprocessing`, using simulation mode. They cover mixed, uneven and
single-vendor clusters, leader election, several shapes and dtypes, and
`broadcast` from every rank.

To validate on real GPUs (a Slurm cluster, a rented AMD machine, and a real
mixed NVIDIA + AMD job), follow
[docs/GPU_VALIDATION_RUNBOOK.md](docs/GPU_VALIDATION_RUNBOOK.md). Every step is
a script under [`scripts/gpu/`](scripts/gpu) with a `--dry-run` flag.

## License

MIT. See [LICENSE](LICENSE).
