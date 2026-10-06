# gpubridge

Write your distributed communication code once and run it on a cluster that
mixes NVIDIA and AMD GPUs.

NVIDIA GPUs talk through NCCL and AMD GPUs through RCCL, and the two cannot
join the same communicator. Teams with both end up maintaining two training
setups, or leaving half their hardware idle. gpubridge sits on top of
`torch.distributed` and gives every process the same small API (`all_reduce`,
`broadcast`, `barrier`) whatever GPU it runs on.

gpubridge writes no CUDA or HIP code. It composes collectives that PyTorch
already provides.

> **Status: v0.1, correctness first.** All orchestration logic is tested in CPU
> simulation mode. It has not yet run on a real mixed cluster; see
> [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for what still needs checking.

## How it works

A PyTorch install is either a CUDA build or a ROCm build, so every process
belongs to exactly one vendor. gpubridge groups processes by vendor:

- **Island:** every rank of one vendor, in a process group on the native
  backend. That is NCCL on NVIDIA, and RCCL on AMD, which ROCm builds of
  PyTorch still call `"nccl"`.
- **Bridge:** a CPU process group on Gloo with one **leader** per island. The
  leader is the island's lowest global rank.

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

### Initialization

`gpubridge.init()` does the following:

1. Creates the default process group on **Gloo** across all ranks. NCCL cannot
   span both vendors, so this CPU group handles discovery and `barrier`.
2. Each rank detects its vendor from `torch.version.hip` (AMD) or
   `torch.version.cuda` (NVIDIA).
3. Every rank shares its vendor, PyTorch version and hostname with
   `all_gather_object`.
4. Each rank computes the same islands, leaders and bridge from that shared
   map. Every rank calls `new_group` for every group in the same order, as
   `torch.distributed` requires.

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

## Simulation mode (no GPUs needed)

Set `GPUBRIDGE_VENDOR=nvidia` or `GPUBRIDGE_VENDOR=amd` on a process to fake
its vendor. With the variable set:

- islands use Gloo instead of NCCL/RCCL, and
- tensors stay on CPU (`topology.device` is `cpu`).

Everything else is the code path used on real GPUs: discovery, leader
election, group creation, and reduce-bridge-broadcast. Set the variable on
every rank or on none; `init()` refuses a mix of simulated and real ranks.

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

## API

The API follows `torch.distributed`, always on the world group.

| Function | Description |
| --- | --- |
| `init(init_method=None, *, rank=None, world_size=None, timeout=None) -> Topology` | Join the cluster and build the islands and bridge. Same arguments as `init_process_group`. |
| `all_reduce(tensor, op=ReduceOp.SUM)` | In-place sum across all ranks. Only SUM is supported in v1. |
| `broadcast(tensor, src)` | In-place copy from global rank `src` to all ranks. |
| `barrier()` | Wait for all ranks. On GPUs, first waits for this rank's queued GPU work. |
| `destroy()` | Tear down every group `init` created. |
| `get_topology() -> Topology` | This rank's view of the cluster (see below). |
| `is_initialized() -> bool` | True between `init` and `destroy`. |

Tensors must be contiguous and on `topology.device`: the rank's GPU, or CPU in
simulation mode.

`Topology` exposes `rank`, `world_size`, `vendor`, `device`, `simulated`,
`island`, `is_leader`, `layout` (every island, its ranks and leader),
`peers` (each rank's vendor, PyTorch version and hostname), and the
`island_group` / `bridge_group` process groups.

## Building on gpubridge

gpubridge is meant to be a base for cluster tooling that has to understand
mixed hardware, such as per-vendor straggler detection:

- `get_topology()` tells each rank which vendor, island and host every other
  rank belongs to, so measurements can be grouped and compared per vendor.
- The default process group is a CPU Gloo group over all ranks. It works the
  same on both vendors and does not touch GPU streams, so it can carry
  monitoring data without disturbing training traffic.

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

## License

MIT. See [LICENSE](LICENSE).
