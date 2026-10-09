# gpubridge

gpubridge lets one distributed PyTorch job span NVIDIA and AMD GPUs at the
same time. Choosing between NCCL and RCCL on a single-vendor machine is already
solved: PyTorch's `"nccl"` backend runs NCCL on CUDA builds and RCCL on ROCm
builds, and Modular's MAX loads NCCL or RCCL to match the GPU it was built for.
Neither connects the two, because an NCCL communicator and an RCCL
communicator cannot exchange data, and Modular states that mixed-vendor hosts
are not supported. gpubridge keeps each vendor on its native library inside an
island and joins the islands with a CPU bridge. One `all_reduce`, `broadcast`
or `barrier` call then covers every GPU in the job, from the same code on every
node. CUDA and ROCm builds of PyTorch, from 2.9.1 to 2.14.1, have been shown to
join one job and complete the bridge on CPU; validation on real GPUs is next.

gpubridge writes no CUDA or HIP code. It composes collectives that PyTorch
already provides. For how it relates to Modular MAX, RAJA, Triton,
Triton-distributed and Meta's torchcomms, see
[docs/PRIOR_ART.md](docs/PRIOR_ART.md).

> **Status: v0.1, correctness first.** All orchestration logic is tested in CPU
> simulation mode, and the CPU side of a mixed job has been checked with real
> CUDA and ROCm builds. It has not yet run on a real mixed cluster; see
> [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for the results and what
> still needs checking.

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

1. `all_reduce` within each island on the native backend.
2. Leaders copy the island sum to CPU and `all_reduce` it over the bridge.
3. Each leader copies the total back to its GPU and broadcasts it to its island.

**`broadcast(tensor, src)`** takes three hops: from `src` to the rest of its
island, from that island's leader over the bridge to the other leaders, then
from each leader to the rest of its island.

If the cluster has only one vendor there is no bridge, and every call goes
straight to the native collective.
These paths are the default [collective policies](#collective-policies).

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

```bash
uv venv
uv pip install -e ".[test]"
```

On GPU nodes, install the CUDA or ROCm build of PyTorch first. gpubridge only
needs `torch>=2.3`, so it leaves an existing install alone.

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
trains a small model with data parallelism and syncs its gradients with
`gpubridge.all_reduce`. It reports samples/sec and checks that every rank ends
with identical parameters:

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
| `all_reduce(tensor, op=ReduceOp.SUM)` | In-place sum across all ranks. Only SUM is supported in v1. |
| `broadcast(tensor, src)` | In-place copy from global rank `src` to all ranks. |
| `barrier()` | Wait for all ranks. On GPUs, first waits for this rank's queued GPU work. |
| `destroy()` | Tear down every group `init` created. |
| `get_topology() -> Topology` | This rank's view of the cluster (see below). |
| `is_initialized() -> bool` | True between `init` and `destroy`. |
| `probe() -> Probe` | This process's build vendor, CUDA/HIP versions, visible GPU count, and NCCL/Gloo availability. Never raises; failed checks are listed in `Probe.errors`. Works before `init`. |
| `add_observer(observer)` / `remove_observer(observer)` | Start or stop sending this rank's collective records to a `CollectiveObserver` (see [Observing collectives](#observing-collectives)). |

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

| Policy | Applies to | What it does |
| --- | --- | --- |
| `reduce-bridge-broadcast` | 2+ islands | island collective, island leaders over the bridge, island broadcast |
| `native-only` | 1 island | one native NCCL/RCCL (or CPU Gloo) collective |
| `flat-gloo` | any | every rank copies to CPU and uses the world Gloo group: slow but obviously correct, so it's the test reference and a debugging fallback that never touches NCCL/RCCL |
| `auto` (default) | any | `native-only` for one island, `reduce-bridge-broadcast` otherwise; today's behaviour |

```python
gpubridge.init(policy="flat-gloo")     # e.g. to rule out NCCL/RCCL while debugging
```

A custom policy subclasses `CollectivePolicy`, implements `applies_to`,
`all_reduce` and `broadcast`, and registers with `register_policy`. Optional
hooks leave room for performance work without changing the public API:
- `setup()` and `close()`, for staging buffers or streams;
- `select_policy(nbytes)`, to choose a path by message size.

The GPU kit's `check.py` and `bench_all_reduce.py` take `--policy`, so runs can
compare policies.

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

init(bridge="my-transport")  # or init(bridge=MyTransport)
```

Every rank reports its transport's name during discovery, so a job where ranks
picked different transports fails at `init()` instead of hanging.
`tests/_transports.py` has a working example that carries the bridge over a
c10d `FileStore`.

### Observing collectives

Monitoring or profiling tools can watch every collective on a rank. After each
`all_reduce`, `broadcast` and `barrier`, an observer gets a `CollectiveRecord`:

- `seq`: the collective's number since `init()`, the same on every rank, so
  records from different ranks line up;
- `op`, `nbytes`, `dtype`, `src` and `policy`;
- `start` and `end` marks, and `phases`. Reduce-bridge-broadcast marks
  `island-reduce`, `bridge` (leaders only) and `island-broadcast`.

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

## Limitations in v1

- `all_reduce` supports only `SUM`. All calls are synchronous (no `async_op`)
  and run on the world group (no custom groups).
- No performance work yet. Cross-vendor traffic goes through each island
  leader's CPU and Gloo, with no chunking, overlap or pinned memory.
- No custom kernels and no direct RDMA between vendors.

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
