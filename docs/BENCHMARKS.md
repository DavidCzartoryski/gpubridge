# Benchmarks

What gpubridge costs, measured on GPUs. Every number here comes from the GPU kit's
records on the Explorer cluster on 2026-10-14, at commit 934f87c unless
a row says otherwise. Host names, user names and paths are left out; the raw
records keep them.

The plan was a comparison of NVIDIA with NVIDIA, AMD with AMD and NVIDIA with AMD.
So far only the NVIDIA + AMD half has run: a pair of two GPUs of one type, in one
partition, waits for Research Computing to confirm it is within the partitions'
intended use (see [Waiting on Research Computing](#waiting-on-research-computing)).

| Run | What it measures | Status |
| --- | --- | --- |
| Network check | InfiniBand/RoCE on each node type; the transport NCCL and RCCL choose | ran, one V100 node and one MI50 node |
| RDMA check | what RDMA from GPU memory would need on each node type: peer-memory modules, dma-buf, UCX | ran, one V100 node and one MI50 node: no RDMA path between them |
| A1, A2 | two V100s, native NCCL: as NCCL chooses, then TCP on the bridge's interface | waiting on Research Computing |
| B1, B2 | two MI50s, native RCCL: the same two | waiting on Research Computing |
| C, C-amd | two V100s (two MI50s) through gpubridge's bridge, one island each: gpubridge's own cost without vendor mixing | waiting on Research Computing |
| D | a V100 and an MI50 through gpubridge: every policy | ran |
| E | the same two ranks as one flat Gloo group | ran |
| Training | DDP with `gpubridge.ddp_comm_hook` on GPT-2 small and ResNet-50: each GPU alone, then both | ran: the V100 alone, the MI50 alone (torch 2.9.1+rocm6.3) and both together |

## Setup

| | NVIDIA side | AMD side |
| --- | --- | --- |
| GPU | Tesla V100-SXM2-32GB (sm_70) | AMD Instinct MI50, 32 GB (gfx906) |
| Partition | `gpu` | `sharing` |
| Driver | 545.23.08 | 5.14.0-611.5.1.el9_7 (the kernel's amdgpu, as the records report it) |
| OS kernel | 5.14.0-362.13.1.el9_3 | 5.14.0-611.5.1.el9_7 |
| Python | 3.12.15 | 3.12.15 |
| torch | 2.14.1+cu126 | 2.14.1+rocm7.2 (all_reduce matrix); 2.9.1+rocm6.3 (training) |
| CUDA / HIP | CUDA 12.6 | HIP 7.2.53211; HIP 6.3.42134 with 2.9.1+rocm6.3 |
| NCCL / RCCL | NCCL 2.29.3 | RCCL 2.27.7; RCCL 2.21.5 with 2.9.1+rocm6.3 |
| gpubridge | 0.4.0a0, commit 934f87c, clean | the same |

One GPU and one rank on each side, as one heterogeneous Slurm job across the two
partitions (`submit.sh 08`). Every MI50 job today ran on the same node, and
step 08, the V100-alone training run and the mixed training run shared one V100
node.

## Network

From `submit.sh net-nvidia` and `net-amd` (one short job on each node type:
`ibv_devinfo`, `ibstat`, `/sys/class/infiniband`, `ip -br addr`, and NCCL or
RCCL initialized alone with `NCCL_DEBUG=INFO`, as it chooses and then with
`NCCL_IB_DISABLE=1`); step 08 recorded the same on its own two nodes.

| Node | RDMA ports (link layer, state, rate) | NCCL/RCCL as it chooses | With `NCCL_IB_DISABLE=1` |
| --- | --- | --- | --- |
| V100 (`gpu`) | `mlx5_0`: InfiniBand, active, 100 Gb/s | NET/IB on `mlx5_0` (InfiniBand); bootstrap on `internal` | NET/Socket on `internal` and `internal.2137` |
| MI50 (`sharing`) | `mlx5_0`: Ethernet (RoCE), active, 10 Gb/s; `mlx5_1`: Ethernet, down, 40 Gb/s; `mlx5_2`: InfiniBand, down | NET/IB on `mlx5_0` over RoCE; bootstrap on `ibs8` | NET/Socket on `ibs8` |

gpubridge's bridge is Gloo over TCP. Its preflight chose `internal.2137` (a VLAN
on `internal`) on both nodes and measured, with one TCP stream from Python,
1173.7 MB/s from the V100 to the MI50, 1172.1 MB/s back, and a 0.048 ms round
trip. Step 07's two earlier jobs measured the same bandwidth within 2.5%.

What this means for the comparison:

- **A1 would not use the bridge's wire.** Between two V100 nodes, NCCL picks
  InfiniBand at 100 Gb/s; the bridge's one TCP stream measured about 9.4 Gb/s.
  A1 against D would mostly measure the network, which is why A2 (NCCL forced
  onto TCP on the bridge's interface) is the baseline for gpubridge's cost.
- **B1 would use RDMA over the MI50 node's 10 Gb/s RoCE port**, also not the
  bridge's wire.
- **RCCL's own TCP choice on the MI50 node is `ibs8`, which `ip` reports as
  down** (it has an address). A TCP baseline there must name the interface:
  step 08's A2 and B2 set `NCCL_SOCKET_IFNAME` to exactly the preflight's
  interface.
- In the D runs each island has one rank, so NCCL and RCCL never cross the
  network; their lines above say what they would choose, not a transport in use.
- **There is no RDMA path between the two node types**; see
  [RDMA between the node types](#rdma-between-the-node-types).

### RDMA between the node types

A second run of `net-nvidia` and `net-amd` (commit 1ac6fa9, which adds
`rdma_report.sh` to the network check; the steps before it are unchanged from
934f87c) recorded what RDMA from GPU memory would need on each node: the kernel's
dma-buf and peer-to-peer options, the RDMA and GPU kernel modules, rdma-core,
`rdma link`, the HSA runtime's dma-buf export, and UCX, with the cluster's
OpenMPI 4.1.6 module loaded for its `ucx_info` (`RDMA_MODULES`). One node of
each type: a different V100 node from the first network check's, and the same
MI50 node. The V100 node's ports matched the first one's.

| | V100 node (`gpu`) | MI50 node (`sharing`) |
| --- | --- | --- |
| RDMA devices (`rdma link`) | `mlx5_0`: InfiniBand, active | `mlx5_0`: Ethernet (RoCE) on `internal`, active, 10 Gb/s; `mlx5_1`: disabled; `mlx5_2`: InfiniBand, disabled |
| The bridge's interface (`internal`, VLAN `internal.2137`) | no RDMA device behind it | `mlx5_0`, RoCE |
| Kernel | 5.14.0-362.13.1.el9_3 | 5.14.0-611.5.1.el9_7 |
| dma-buf and peer-to-peer in the kernel | `DMABUF_MOVE_NOTIFY`, `PCI_P2PDMA` | the same, and `HSA_AMD_P2P` |
| rdma-core | 46.0; libibverbs has `ibv_reg_dmabuf_mr` | 57.0; the same |
| GPU memory for the NIC | `nvidia_peermem` (545.23.08) installed for this kernel, not loaded; the NVIDIA kernel module is the proprietary one (license `NVIDIA`) | no `amdp2p` module; torch's HSA runtime (2.14.1+rocm7.2) has `hsa_amd_portable_export_dmabuf` |
| UCX | 1.16.0, the only build found (in the OpenMPI 4.1.6 module): transports `self`, `tcp`, `sysv`, `posix`, `cma`; memory type host only | the same build, the same transports; host only |

What this means:

- **No RDMA path between a V100 node and an MI50 node.** The V100 node's only
  RDMA device is on InfiniBand; the MI50 node's only active one is on Ethernet,
  and its InfiniBand port is disabled. An InfiniBand port can't reach a RoCE
  port, and RoCE needs an RDMA device at both ends: the V100 node reaches the
  MI50 node's Ethernet network through `internal`, which has none. So the UCX
  transport's RDMA and GPUDirect paths (items 26 and 27 in
  [HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md)) can't be validated
  between these two node types; the bridge stays on TCP.
- **The cluster's UCX can't do RDMA or GPU memory even within one node type.**
  It was built without verbs and without CUDA or ROCm, so it offers TCP and
  shared memory only. RDMA through gpubridge's UCX transport on Explorer would
  need another UCX build; none was tried.
- **GPU side, for other hardware.** On the V100 nodes GPUDirect RDMA would
  need `nvidia_peermem` loaded, which it isn't; NVIDIA's dma-buf path needs the
  open kernel modules, which these nodes don't run. The MI50 node has what
  dma-buf RDMA from AMD GPU memory needs (the kernel option, the runtime's
  export call, `ibv_reg_dmabuf_mr`), but nothing was tried.
- Not recorded: `ulimit -l` in jobs and `perftest`. Neither changes the
  answer above.

## The all_reduce matrix: a V100 and an MI50

`submit.sh 08` (`scripts/gpu/explorer/08_bench_matrix.sbatch`), then
`scripts/gpu/bench_matrix.py` over its results. Two ranks, one GPU each on two
nodes; float32 all_reduce from 1 KB to 1 GB in steps of 4x; 20 timed trials per
size; the median latency of the slower rank, timed with CUDA events on each GPU.
Bus bandwidth as nccl-tests computes it, which with 2 ranks is bytes / time. All
11 runs were in one job (all 15 steps passed, in 12 minutes), so they share the
two nodes and the network.

| Run | Policy |
| --- | --- |
| D-default | `reduce-bridge-broadcast`, pageable host staging |
| D-pinned | the same with `GPUBRIDGE_PINNED_STAGING=1` |
| D-pipelined-1M to -64M | `pipelined-reduce-bridge-broadcast`, chunks of 1, 4, 16 and 64 MB |
| D-sharded | `sharded-bridge`; with one GPU per island it has one link (k = 1), so it runs the default's steps |
| D-bf16, D-fp8 | the default with `GPUBRIDGE_BRIDGE_COMPRESSION=bf16` or `fp8` |
| D-default-repeat | D-default again, last, to show the spread between two runs |
| E | one flat Gloo group of the same two ranks, no gpubridge |

Latency at five sizes, and speed at 1 GB against D-default (above 1 is faster):

| Run | 1 KB | 64 KB | 1 MB | 16 MB | 1 GB | Bus bandwidth at 1 GB (GB/s) | Speed-up over D-default at 1 GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-default | 731 us | 906 us | 2.7 ms | 32.3 ms | 1.67 s | 0.64 | 1.00x |
| D-pinned | 701 us | 1.12 ms | 3.67 ms | 27.2 ms | 1.21 s | 0.89 | 1.39x |
| D-pipelined-1M | 784 us | 1.14 ms | 3.27 ms | 26.2 ms | 1.66 s | 0.65 | 1.01x |
| D-pipelined-4M | 761 us | 895 us | 3.84 ms | 27.9 ms | 2.00 s | 0.54 | 0.84x |
| D-pipelined-16M | 836 us | 955 us | 3.22 ms | 26.6 ms | 1.07 s | 1.00 | 1.56x |
| D-pipelined-64M | 781 us | 1.22 ms | 3.19 ms | 29.3 ms | 1.22 s | 0.88 | 1.37x |
| D-sharded | 671 us | 1.22 ms | 3.93 ms | 29.8 ms | 1.58 s | 0.68 | 1.06x |
| D-bf16 | 921 us | 1.05 ms | 2.96 ms | 26 ms | 1.23 s | 0.88 | 1.37x |
| D-fp8 | 919 us | 952 us | 1.97 ms | 12 ms | 609 ms | 1.76 | 2.75x |
| D-default-repeat | 648 us | 1.01 ms | 4.35 ms | 34.6 ms | 1.60 s | 0.67 | 1.05x |
| E | 424 us | 562 us | 2.81 ms | 29.9 ms | 1.64 s | 0.66 | 1.02x |

What the records show:

- **With one GPU per vendor, the default bridge costs about what one flat Gloo
  group does from 1 MB up**: 1.67 s and 1.60 s (the repeat) at 1 GB, against
  1.64 s for E, all near 0.65 GB/s. With one rank per island there is nothing
  for NCCL or RCCL to reduce, so the bridge is Gloo over one TCP connection, as
  E is. From 1 KB to 16 KB the gpubridge runs took 650 to 930 us against 390
  to 500 us for E: the island phases add about 0.15 ms together, and the bridge
  phase alone took 0.36 to 0.64 ms.
- **Pinned staging is 1.3 to 1.4x faster than either default run from 64 MB
  up** (1.21 s at 1 GB, 0.89 GB/s; 304 ms against 417 and 389 ms at 256 MB; 77 ms
  against 104 and 100 ms at 64 MB).
  Step 07's earlier job measured 1.28 s at 1 GB with pinned staging.
- **The pipelined bridge with 16 MB chunks was the fastest uncompressed run**
  at 64 MB, 256 MB and 1 GB (1.07 s at 1 GB, 1.00 GB/s, 1.56x faster than the
  default). The other chunk sizes don't order consistently: 4 MB was the slowest
  run at 1 GB (2.00 s) but faster than the default at 256 MB. One run each, so
  the chunk size's effect is not settled beyond 16 MB being best here.
- **The sharded bridge with one link is the default** (1.58 s at 1 GB), as
  expected with k = 1.
- **fp8 compression was 2.75x faster than the default at 1 GB** (609 ms, an
  effective 1.76 GB/s), of which compressing and decompressing took 6.1 and
  4.4 ms. **bf16 was 1.37x faster** (1.23 s), not 2x: with two islands each
  leader's payload goes in turn, one direction at a time, so halving the bytes
  gives the same predicted time as the uncompressed ring (916 ms at 1 GB). bf16
  took 1.33x that prediction, the default 1.83x of the same number.
- **Two runs of one configuration differed by 23% and 61% at 4 MB and 1 MB**,
  by 3 to 15% below that, and by 4 to 7% from 16 MB up (table below). A
  difference between two runs in the middle sizes smaller than that says
  nothing.

| Size | D-default | D-default-repeat | Difference |
| --- | ---: | ---: | ---: |
| 1 KB | 731 us | 648 us | -11% |
| 4 KB | 755 us | 850 us | +13% |
| 16 KB | 812 us | 930 us | +15% |
| 64 KB | 906 us | 1.01 ms | +11% |
| 256 KB | 1.57 ms | 1.52 ms | -3% |
| 1 MB | 2.7 ms | 4.35 ms | +61% |
| 4 MB | 11.7 ms | 14.4 ms | +23% |
| 16 MB | 32.3 ms | 34.6 ms | +7% |
| 64 MB | 104 ms | 99.6 ms | -4% |
| 256 MB | 417 ms | 389 ms | -7% |
| 1 GB | 1.67 s | 1.60 s | -4% |

### Predicted against measured

The prediction is the job's preflight (one TCP stream: its slower direction and
the round trip) applied to a ring all_reduce, the whole tensor each way at once;
with compression, each leader's payload broadcast in turn. The measured bridge
phase also includes the copies between GPU and host, which the prediction
doesn't. At 1 GB the bridge took 1.17x (pipelined, 16 MB chunks) to 2.18x
(pipelined, 4 MB) the prediction, the default 1.74x and 1.83x. From 1 KB to
16 KB it took 6 to 12x: 0.36 to 0.64 ms per call against a 0.05 ms round trip.

| Run | Size | Predicted bridge (ms) | Measured bridge (ms) | Measured / predicted |
| --- | --- | ---: | ---: | ---: |
| D-default | 1 MB | 0.943 | 2.4008 | 2.55x |
| D-default | 1 GB | 916.132 | 1671.96 | 1.83x |
| D-pinned | 1 MB | 0.943 | 3.474 | 3.69x |
| D-pinned | 1 GB | 916.132 | 1204.97 | 1.32x |
| D-pipelined-1M | 1 MB | 0.943 | 2.7342 | 2.90x |
| D-pipelined-1M | 1 GB | 916.132 | 1658.01 | 1.81x |
| D-pipelined-4M | 1 MB | 0.943 | 3.6174 | 3.84x |
| D-pipelined-4M | 1 GB | 916.132 | 1997.72 | 2.18x |
| D-pipelined-16M | 1 MB | 0.943 | 2.8851 | 3.06x |
| D-pipelined-16M | 1 GB | 916.132 | 1072.77 | 1.17x |
| D-pipelined-64M | 1 MB | 0.943 | 2.9686 | 3.15x |
| D-pipelined-64M | 1 GB | 916.132 | 1220.44 | 1.33x |
| D-sharded | 1 MB | 0.943 | 3.6382 | 3.86x |
| D-sharded | 1 GB | 916.132 | 1577.45 | 1.72x |
| D-bf16 | 1 MB | 0.943 | 2.4076 | 2.55x |
| D-bf16 | 1 GB | 916.132 | 1217 | 1.33x |
| D-fp8 | 1 MB | 0.502 | 1.3871 | 2.76x |
| D-fp8 | 1 GB | 465.247 | 600.001 | 1.29x |
| D-default-repeat | 1 MB | 0.943 | 3.9748 | 4.22x |
| D-default-repeat | 1 GB | 916.132 | 1596.91 | 1.74x |

### Phases

Median ms of each phase on the slower rank, at 1 MB and 1 GB. The island phases
are NCCL or RCCL on one GPU, so they stay under 0.3 ms, except the sharded
bridge's reduce-scatter and all-gather (3 ms each at 1 GB).

| Run | Size | island-reduce | island-reduce-scatter | bridge | island-broadcast | island-all-gather | compress | decompress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-default | 1 MB | 0.075 | - | 2.4008 | 0.0842 | - | - | - |
| D-default | 1 GB | 0.1657 | - | 1671.96 | 0.1935 | - | - | - |
| D-pinned | 1 MB | 0.075 | - | 3.474 | 0.0242 | - | - | - |
| D-pinned | 1 GB | 0.1121 | - | 1204.97 | 0.0126 | - | - | - |
| D-pipelined-1M | 1 MB | 0.0751 | - | 2.7342 | 0.0495 | - | - | - |
| D-pipelined-1M | 1 GB | 0.106 | - | 1658.01 | 0.2616 | - | - | - |
| D-pipelined-4M | 1 MB | 0.0758 | - | 3.6174 | 0.063 | - | - | - |
| D-pipelined-4M | 1 GB | 0.1288 | - | 1997.72 | 0.1746 | - | - | - |
| D-pipelined-16M | 1 MB | 0.0756 | - | 2.8851 | 0.0609 | - | - | - |
| D-pipelined-16M | 1 GB | 0.1141 | - | 1072.77 | 0.0144 | - | - | - |
| D-pipelined-64M | 1 MB | 0.0701 | - | 2.9686 | 0.0532 | - | - | - |
| D-pipelined-64M | 1 GB | 0.1075 | - | 1220.44 | 0.0144 | - | - | - |
| D-sharded | 1 MB | - | 0.1096 | 3.6382 | - | 0.087 | - | - |
| D-sharded | 1 GB | - | 2.9712 | 1577.45 | - | 3.0598 | - | - |
| D-bf16 | 1 MB | 0.0714 | - | 2.4076 | 0.0829 | - | 0.1147 | 0.0924 |
| D-bf16 | 1 GB | 0.0952 | - | 1217 | 0.165 | - | 5.7661 | 2.8808 |
| D-fp8 | 1 MB | 0.079 | - | 1.3871 | 0.086 | - | 0.1184 | 0.1113 |
| D-fp8 | 1 GB | 0.1386 | - | 600.001 | 0.2202 | - | 6.0975 | 4.3595 |
| D-default-repeat | 1 MB | 0.0786 | - | 3.9748 | 0.1103 | - | - | - |
| D-default-repeat | 1 GB | 0.1429 | - | 1596.91 | 0.2032 | - | - | - |

### How to read the full matrix

`bench_matrix.py` writes this guide with the tables. Until A1, A2, B1, B2 and C
run, the x A2 and x A1 columns are empty.

- **A1 against A2 is the network**: the same NCCL on the same two GPUs in one
  job, on the transport NCCL picks and then over TCP on the bridge's interface.
- **A2 against C is gpubridge's own cost**: the same job, the same two V100s
  and the same interface; C takes gpubridge's path (each GPU an island, the sum
  staged through host memory and all_reduced by Gloo over one connection)
  instead of NCCL's socket transport, which may open several. B2 against C-amd
  is the same on two MI50s.
- **C against D is the vendor mixing, across jobs**: the same bridge with an
  MI50 and RCCL on one side, but on other nodes. A ratio marked † divides by a
  run of another job, so check the two jobs' networks before putting a
  difference down to the vendor.
- **D's variants against D-default** are what each policy buys on this pair,
  and **D-default against E** what gpubridge does better than one flat Gloo
  group.

### Full tables

#### Small: under 256 KB (latency)

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1KB | D-default | 731 us | 0.0014 | - | - | 0.049 | 0.4552 | 9.31x |
| 1KB | D-pinned | 701 us | 0.0015 | - | - | 0.049 | 0.4372 | 8.95x |
| 1KB | D-pipelined-1M | 784 us | 0.0013 | - | - | 0.049 | 0.5174 | 10.59x |
| 1KB | D-pipelined-4M | 761 us | 0.0013 | - | - | 0.049 | 0.4799 | 9.82x |
| 1KB | D-pipelined-16M | 836 us | 0.0012 | - | - | 0.049 | 0.5621 | 11.50x |
| 1KB | D-pipelined-64M | 781 us | 0.0013 | - | - | 0.049 | 0.5233 | 10.71x |
| 1KB | D-sharded | 671 us | 0.0015 | - | - | 0.049 | 0.3552 | 7.27x |
| 1KB | D-bf16 | 921 us | 0.0011 | - | - | 0.049 | 0.3763 | 7.70x |
| 1KB | D-fp8 | 919 us | 0.0011 | - | - | 0.048 | 0.364 | 7.51x |
| 1KB | D-default-repeat | 648 us | 0.0016 | - | - | 0.049 | 0.3855 | 7.89x |
| 1KB | E | 424 us | 0.0024 | - | - | - | - | - |
| 4KB | D-default | 755 us | 0.0054 | - | - | 0.051 | 0.4666 | 9.06x |
| 4KB | D-pinned | 758 us | 0.0054 | - | - | 0.051 | 0.4936 | 9.59x |
| 4KB | D-pipelined-1M | 773 us | 0.0053 | - | - | 0.051 | 0.5156 | 10.01x |
| 4KB | D-pipelined-4M | 807 us | 0.0051 | - | - | 0.051 | 0.5474 | 10.63x |
| 4KB | D-pipelined-16M | 793 us | 0.0052 | - | - | 0.051 | 0.509 | 9.88x |
| 4KB | D-pipelined-64M | 805 us | 0.0051 | - | - | 0.051 | 0.5371 | 10.43x |
| 4KB | D-sharded | 692 us | 0.0059 | - | - | 0.051 | 0.3853 | 7.48x |
| 4KB | D-bf16 | 928 us | 0.0044 | - | - | 0.051 | 0.3989 | 7.75x |
| 4KB | D-fp8 | 923 us | 0.0044 | - | - | 0.05 | 0.361 | 7.25x |
| 4KB | D-default-repeat | 850 us | 0.0048 | - | - | 0.051 | 0.5775 | 11.21x |
| 4KB | E | 391 us | 0.0105 | - | - | - | - | - |
| 16KB | D-default | 812 us | 0.0202 | - | - | 0.062 | 0.5576 | 9.00x |
| 16KB | D-pinned | 912 us | 0.018 | - | - | 0.062 | 0.639 | 10.31x |
| 16KB | D-pipelined-1M | 850 us | 0.0193 | - | - | 0.062 | 0.5558 | 8.97x |
| 16KB | D-pipelined-4M | 831 us | 0.0197 | - | - | 0.062 | 0.5663 | 9.14x |
| 16KB | D-pipelined-16M | 809 us | 0.0203 | - | - | 0.062 | 0.5463 | 8.81x |
| 16KB | D-pipelined-64M | 902 us | 0.0182 | - | - | 0.062 | 0.6295 | 10.16x |
| 16KB | D-sharded | 822 us | 0.0199 | - | - | 0.062 | 0.511 | 8.24x |
| 16KB | D-bf16 | 905 us | 0.0181 | - | - | 0.062 | 0.3881 | 6.26x |
| 16KB | D-fp8 | 923 us | 0.0178 | - | - | 0.055 | 0.3819 | 6.93x |
| 16KB | D-default-repeat | 930 us | 0.0176 | - | - | 0.062 | 0.6226 | 10.05x |
| 16KB | E | 495 us | 0.0331 | - | - | - | - | - |
| 64KB | D-default | 906 us | 0.0723 | - | - | 0.104 | 0.64 | 6.16x |
| 64KB | D-pinned | 1.12 ms | 0.0587 | - | - | 0.104 | 0.8387 | 8.07x |
| 64KB | D-pipelined-1M | 1.14 ms | 0.0574 | - | - | 0.104 | 0.8042 | 7.74x |
| 64KB | D-pipelined-4M | 895 us | 0.0732 | - | - | 0.104 | 0.6369 | 6.13x |
| 64KB | D-pipelined-16M | 955 us | 0.0686 | - | - | 0.104 | 0.6897 | 6.64x |
| 64KB | D-pipelined-64M | 1.22 ms | 0.0538 | - | - | 0.104 | 0.9281 | 8.93x |
| 64KB | D-sharded | 1.22 ms | 0.0535 | - | - | 0.104 | 0.8981 | 8.64x |
| 64KB | D-bf16 | 1.05 ms | 0.0625 | - | - | 0.104 | 0.5325 | 5.12x |
| 64KB | D-fp8 | 952 us | 0.0689 | - | - | 0.076 | 0.4065 | 5.32x |
| 64KB | D-default-repeat | 1.01 ms | 0.0652 | - | - | 0.104 | 0.7301 | 7.03x |
| 64KB | E | 562 us | 0.1166 | - | - | - | - | - |

#### Medium: 256 KB to under 64 MB

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256KB | D-default | 1.57 ms | 0.1667 | - | - | 0.272 | 1.2728 | 4.69x |
| 256KB | D-pinned | 1.68 ms | 0.1559 | - | - | 0.272 | 1.4286 | 5.26x |
| 256KB | D-pipelined-1M | 1.67 ms | 0.1574 | - | - | 0.272 | 1.4126 | 5.20x |
| 256KB | D-pipelined-4M | 1.69 ms | 0.1552 | - | - | 0.272 | 1.2455 | 4.58x |
| 256KB | D-pipelined-16M | 1.28 ms | 0.2046 | - | - | 0.272 | 0.9994 | 3.68x |
| 256KB | D-pipelined-64M | 1.5 ms | 0.1753 | - | - | 0.272 | 1.0957 | 4.03x |
| 256KB | D-sharded | 1.76 ms | 0.1491 | - | - | 0.272 | 1.5129 | 5.57x |
| 256KB | D-bf16 | 1.43 ms | 0.1832 | - | - | 0.272 | 0.8858 | 3.26x |
| 256KB | D-fp8 | 1.28 ms | 0.2053 | - | - | 0.162 | 0.7177 | 4.44x |
| 256KB | D-default-repeat | 1.52 ms | 0.1722 | - | - | 0.272 | 1.2502 | 4.60x |
| 256KB | E | 1.19 ms | 0.2207 | - | - | - | - | - |
| 1MB | D-default | 2.7 ms | 0.3886 | - | - | 0.943 | 2.4008 | 2.55x |
| 1MB | D-pinned | 3.67 ms | 0.2857 | - | - | 0.943 | 3.474 | 3.69x |
| 1MB | D-pipelined-1M | 3.27 ms | 0.3204 | - | - | 0.943 | 2.7342 | 2.90x |
| 1MB | D-pipelined-4M | 3.84 ms | 0.273 | - | - | 0.943 | 3.6174 | 3.84x |
| 1MB | D-pipelined-16M | 3.22 ms | 0.326 | - | - | 0.943 | 2.8851 | 3.06x |
| 1MB | D-pipelined-64M | 3.19 ms | 0.3284 | - | - | 0.943 | 2.9686 | 3.15x |
| 1MB | D-sharded | 3.93 ms | 0.2667 | - | - | 0.943 | 3.6382 | 3.86x |
| 1MB | D-bf16 | 2.96 ms | 0.3547 | - | - | 0.943 | 2.4076 | 2.55x |
| 1MB | D-fp8 | 1.97 ms | 0.5321 | - | - | 0.502 | 1.3871 | 2.76x |
| 1MB | D-default-repeat | 4.35 ms | 0.241 | - | - | 0.943 | 3.9748 | 4.22x |
| 1MB | E | 2.81 ms | 0.3726 | - | - | - | - | - |
| 4MB | D-default | 11.7 ms | 0.3574 | - | - | 3.626 | 10.4676 | 2.89x |
| 4MB | D-pinned | 8.19 ms | 0.512 | - | - | 3.626 | 8.0503 | 2.22x |
| 4MB | D-pipelined-1M | 11 ms | 0.3817 | - | - | 3.626 | 10.7753 | 2.97x |
| 4MB | D-pipelined-4M | 12.3 ms | 0.3415 | - | - | 3.626 | 12.1204 | 3.34x |
| 4MB | D-pipelined-16M | 8.45 ms | 0.4965 | - | - | 3.626 | 7.0456 | 1.94x |
| 4MB | D-pipelined-64M | 8.15 ms | 0.5144 | - | - | 3.626 | 8.0108 | 2.21x |
| 4MB | D-sharded | 9.22 ms | 0.4551 | - | - | 3.626 | 8.849 | 2.44x |
| 4MB | D-bf16 | 7.48 ms | 0.5605 | - | - | 3.626 | 6.7036 | 1.85x |
| 4MB | D-fp8 | 4.44 ms | 0.9454 | - | - | 1.865 | 3.8332 | 2.06x |
| 4MB | D-default-repeat | 14.4 ms | 0.2906 | - | - | 3.626 | 14.2158 | 3.92x |
| 4MB | E | 9.24 ms | 0.454 | - | - | - | - | - |
| 16MB | D-default | 32.3 ms | 0.5192 | - | - | 14.362 | 30.9117 | 2.15x |
| 16MB | D-pinned | 27.2 ms | 0.6174 | - | - | 14.362 | 27.0202 | 1.88x |
| 16MB | D-pipelined-1M | 26.2 ms | 0.6396 | - | - | 14.362 | 25.8242 | 1.80x |
| 16MB | D-pipelined-4M | 27.9 ms | 0.6016 | - | - | 14.362 | 26.7146 | 1.86x |
| 16MB | D-pipelined-16M | 26.6 ms | 0.6318 | - | - | 14.362 | 25.9052 | 1.80x |
| 16MB | D-pipelined-64M | 29.3 ms | 0.5718 | - | - | 14.362 | 28.3955 | 1.98x |
| 16MB | D-sharded | 29.8 ms | 0.5627 | - | - | 14.362 | 28.4848 | 1.98x |
| 16MB | D-bf16 | 26 ms | 0.6441 | - | - | 14.362 | 25.2471 | 1.76x |
| 16MB | D-fp8 | 12 ms | 1.3957 | - | - | 7.317 | 11.1805 | 1.53x |
| 16MB | D-default-repeat | 34.6 ms | 0.4844 | - | - | 14.362 | 33.5045 | 2.33x |
| 16MB | E | 29.9 ms | 0.5618 | - | - | - | - | - |

#### Large: 64 MB and up (bandwidth)

| Size | Run | Latency | Bus bandwidth (GB/s) | x A2 | x A1 | Bridge predicted (ms) | Bridge measured (ms) | Measured / predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64MB | D-default | 104 ms | 0.6459 | - | - | 57.303 | 103.312 | 1.80x |
| 64MB | D-pinned | 76.7 ms | 0.8745 | - | - | 57.303 | 76.5565 | 1.34x |
| 64MB | D-pipelined-1M | 99.1 ms | 0.6769 | - | - | 57.303 | 98.7684 | 1.72x |
| 64MB | D-pipelined-4M | 114 ms | 0.5894 | - | - | 57.303 | 113.6335 | 1.98x |
| 64MB | D-pipelined-16M | 73.1 ms | 0.918 | - | - | 57.303 | 72.8996 | 1.27x |
| 64MB | D-pipelined-64M | 80.8 ms | 0.8308 | - | - | 57.303 | 79.6711 | 1.39x |
| 64MB | D-sharded | 104 ms | 0.6479 | - | - | 57.303 | 101.8749 | 1.78x |
| 64MB | D-bf16 | 85.3 ms | 0.7866 | - | - | 57.303 | 83.9264 | 1.46x |
| 64MB | D-fp8 | 44 ms | 1.5249 | - | - | 29.123 | 42.8263 | 1.47x |
| 64MB | D-default-repeat | 99.6 ms | 0.6741 | - | - | 57.303 | 98.3242 | 1.72x |
| 64MB | E | 103 ms | 0.6527 | - | - | - | - | - |
| 256MB | D-default | 417 ms | 0.6432 | - | - | 229.069 | 416.2925 | 1.82x |
| 256MB | D-pinned | 304 ms | 0.8842 | - | - | 229.069 | 303.401 | 1.32x |
| 256MB | D-pipelined-1M | 418 ms | 0.6428 | - | - | 229.069 | 417.1843 | 1.82x |
| 256MB | D-pipelined-4M | 298 ms | 0.9001 | - | - | 229.069 | 297.8138 | 1.30x |
| 256MB | D-pipelined-16M | 280 ms | 0.9598 | - | - | 229.069 | 279.4966 | 1.22x |
| 256MB | D-pipelined-64M | 293 ms | 0.9149 | - | - | 229.069 | 293.2173 | 1.28x |
| 256MB | D-sharded | 401 ms | 0.6688 | - | - | 229.069 | 399.582 | 1.74x |
| 256MB | D-bf16 | 332 ms | 0.8083 | - | - | 229.069 | 320.4423 | 1.40x |
| 256MB | D-fp8 | 153 ms | 1.7522 | - | - | 116.348 | 149.5369 | 1.29x |
| 256MB | D-default-repeat | 389 ms | 0.69 | - | - | 229.069 | 387.9319 | 1.69x |
| 256MB | E | 407 ms | 0.6589 | - | - | - | - | - |
| 1GB | D-default | 1.67 s | 0.6412 | - | - | 916.132 | 1671.9611 | 1.83x |
| 1GB | D-pinned | 1.21 s | 0.8909 | - | - | 916.132 | 1204.9681 | 1.32x |
| 1GB | D-pipelined-1M | 1.66 s | 0.6475 | - | - | 916.132 | 1658.0089 | 1.81x |
| 1GB | D-pipelined-4M | 2 s | 0.5373 | - | - | 916.132 | 1997.7178 | 2.18x |
| 1GB | D-pipelined-16M | 1.07 s | 1.0003 | - | - | 916.132 | 1072.7695 | 1.17x |
| 1GB | D-pipelined-64M | 1.22 s | 0.8796 | - | - | 916.132 | 1220.4363 | 1.33x |
| 1GB | D-sharded | 1.58 s | 0.6781 | - | - | 916.132 | 1577.4468 | 1.72x |
| 1GB | D-bf16 | 1.23 s | 0.8759 | - | - | 916.132 | 1217.0022 | 1.33x |
| 1GB | D-fp8 | 609 ms | 1.7631 | - | - | 465.247 | 600.0012 | 1.29x |
| 1GB | D-default-repeat | 1.6 s | 0.6713 | - | - | 916.132 | 1596.9077 | 1.74x |
| 1GB | E | 1.64 s | 0.656 | - | - | - | - | - |

## Training throughput

`scripts/gpu/train_bench.py`: DDP with `gpubridge.ddp_comm_hook` on a real
model with synthetic data, float32, AdamW, 25 MB buckets; 3 warm-up steps, then
10 timed steps. Per rank: GPT-2 small (124 M parameters, 498 MB of gradients)
at batch 4 x 512 tokens, ResNet-50 (25.6 M parameters, 102 MB) at batch 32 x
224 x 224. The step splits into the slower rank's compute, the time the faster
rank waits for it, and the communication not hidden behind compute ("exposed").
"Comm alone" is the same buckets all_reduced with no compute alongside. Each GPU
alone (`submit.sh train-nvidia`, `train-amd`) is one rank, so its communication
is the hook with nothing to sync.

| Model | Run | Ranks | Samples/s | Step (ms) | Slower GPU's compute | Faster GPU waits | Exposed comm | Comm alone |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | V100 alone | 1 | 25.3 | 158.1 | 152.4 | - | 5.7 | 2.8 |
| GPT-2 small | MI50 alone (2.9.1+rocm6.3) | 1 | 16.4 | 243.6 | 238.9 | - | 4.7 | 2.8 |
| GPT-2 small | V100 + MI50 through gpubridge | 2 | 8.9 | 899.4 | 240.9 (MI50) | 88.6 (V100) | 658.6 | 714.6 |
| ResNet-50 | V100 alone | 1 | 315.8 | 101.3 | 98.1 | - | 3.2 | 0.9 |
| ResNet-50 | MI50 alone (2.9.1+rocm6.3) | 1 | 247.3 | 129.4 | 128.0 | - | 1.3 | 0.9 |
| ResNet-50 | V100 + MI50 through gpubridge | 2 | 327.7 | 195.3 | 128.3 (MI50) | 29.8 (V100) | 67.0 | 145.2 |

Medians of 10 timed steps. Samples/s counts every rank's batch, so the mixed
runs process twice the samples per step. The mixed runs are step 07 with
`STEPS_07=training` and the MI50 on `venv-rocm63` (torch 2.9.1+rocm6.3 there,
2.14.1+cu126 on the V100), the default bridge (`reduce-bridge-broadcast`,
pageable staging) over TCP on `internal.2137`; their preflight measured
1173.8 MB/s from the V100 to the MI50, 1169.7 MB/s back and a 0.046 ms round trip.

**Does adding the MI50 raise throughput? Not on this link.**

- **GPT-2 small: no, it cuts it to 0.35x** the V100 alone (8.9 against 25.3
  samples/s). Each step all_reduces 498 MB of gradients over the bridge, 715 ms
  on its own, and only 56 ms of it hides behind the backward pass; the MI50's
  slower compute adds 89 ms of waiting on the V100.
- **ResNet-50: about the same**: 327.7 against 315.8 samples/s, 4% more, from
  one run of 10 steps that ranged from 187 to 240 ms (the V100 alone: 100.8 to
  101.6 ms). Its 102 MB of gradients take 145 ms alone, 78 ms of it hidden
  behind the backward pass; the V100 waits 30 ms a step for the MI50.
- Within the mixed job each GPU's compute matched its single-GPU run (V100
  152.2 and 98.5 ms, MI50 240.9 and 128.3 ms, against 152.4, 98.1, 238.9 and
  128.0 ms alone), so the loss is the communication and the wait, not slower
  GPUs.
- Every rank ended each run with identical, finite parameters.

These runs used the default policy, which the matrix above measured at about
0.65 GB/s; pinned staging, pipelining and fp8 compression were 1.4x to 2.8x
faster at 1 GB there. Training with them hasn't been measured.

## Which ROCm wheels can train on an MI50 (gfx906)

The MI50's rank aborts at its first matmul with the kit's default ROCm wheel
(torch 2.14.1+rocm7.2): its rocBLAS has no kernels for gfx906. To find a wheel
whose rocBLAS does, a CPU job read the zip directory of every torch ROCm wheel
from 2.8 on for Python 3.12 (the PyTorch index and AMD's repo.radeon.com, by
HTTP range requests, not whole downloads) and counted
`torch/lib/rocblas/library/TensileLibrary*gfx906*`:

| torch | ROCm build | rocBLAS files in the wheel | gfx906 Tensile files |
| --- | --- | --- | --- |
| 2.8.0, 2.9.0, 2.9.1 | rocm6.3 | 1620 to 1675 | **155** |
| 2.8.0, 2.9.0, 2.9.1 | rocm6.4 | 1391 to 1446 | 0 |
| 2.10.0 | rocm7.0, rocm7.1 | 1794 | 0 |
| 2.11.0 to 2.13.0 | rocm7.1 | 1794 | 0 |
| 2.11.0, 2.12.x, 2.14.x | rocm7.2 | 1794 | 0 |
| 2.14.0, 2.14.1 | rocm7.14 | none: rocBLAS comes in separate `rocm-sdk-device-gfx*` wheels | the index has none for gfx906 (it has gfx908, gfx90a, gfx942, gfx950 and gfx10xx to gfx12xx) |
| AMD's builds of 2.8.0 to 2.10.0 | rocm 6.4.4 to 7.2 | 1542 to 1698, or none in the `.lw` builds | 0 |

Not read: 2.13.0+rocm7.2 (the server sent the whole file instead of the range;
the scan now refuses such replies) and two AMD development builds of 2.8.0 (no
end-of-directory record in the last 128 KB). The scan counted rocBLAS only, not
hipBLASLt or MIOpen.

**torch 2.9.1+rocm6.3 on one MI50**, in a second venv (`venv-rocm63`; the
kit's default ROCm venv is unchanged), with HIP 6.3.42134 and RCCL 2.21.5:

- `setup-rocm` with `ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.3`: the probe's
  `blas` stage passed (on the default wheel it aborts, exit -6).
- A short check on the GPU: a 1024 x 1024 float32 matmul (largest error 1.6e-6
  of the largest value, against float64 on the CPU), a float16 matmul, a conv2d,
  causal scaled-dot-product attention and a linear layer's backward all ran.
- The single-node AMD step (`submit.sh amd`, at dbed5c9) passed 7 of 9 steps:
  the probe, `device_map`, `local_rank_check`, `check_1gpu`, `rccl_version`,
  `check_split_shared_gpu` and its summary. `gpu_tests` had 20 failed, 6
  passed, 8 skipped, and `bench_compression_kernels` failed: every failure is
  this wheel's Triton (pytorch-triton-rocm 3.5.1) failing to compile
  gpubridge's compression kernels for gfx906 (`PassManager::run failed` in
  `ConvertTritonAMDGPUToLLVM`). The default venv's Triton (3.8.0) compiles
  them.
- So on an MI50, bridge compression needs the default venv and training needs
  `venv-rocm63`; the runbook's "Choosing the ROCm build" has the commands.

## Waiting on Research Computing

These need two GPUs of one type in one partition, as one heterogeneous job.
The job scripts are ready; `submit.sh` refuses them until
`SAME_PARTITION_PAIRS_OK=1` is set in `scripts/gpu/explorer/local.env` (or the
environment):

| Command | Runs |
| --- | --- |
| `scripts/gpu/explorer/submit.sh 08-nvidia-pair` | A1, A2, C and E-nvidia on two V100s |
| `scripts/gpu/explorer/submit.sh 08-amd-pair` | B1, B2, C-amd and E-amd on two MI50s |

Then `bench_matrix.py` over all three jobs' results fills the x A2 and x A1
columns; ratios between jobs are marked †.

## How to reproduce

```bash
scripts/gpu/explorer/submit.sh net-nvidia
scripts/gpu/explorer/submit.sh net-amd
RDMA_MODULES=OpenMPI/4.1.6 scripts/gpu/explorer/submit.sh net-nvidia   # the RDMA check's
RDMA_MODULES=OpenMPI/4.1.6 scripts/gpu/explorer/submit.sh net-amd      # ucx_info
scripts/gpu/explorer/submit.sh 08
scripts/gpu/explorer/submit.sh train-nvidia
VENV_ROCM=$WORK_DIR/venv-rocm63 scripts/gpu/explorer/submit.sh train-amd
VENV_ROCM=$WORK_DIR/venv-rocm63 STEPS_07=training TIME_07=00:40:00 scripts/gpu/explorer/submit.sh 07
python scripts/gpu/bench_matrix.py results/explorer/08_matrix_nvidia-amd \
    results/explorer/net_check_nvidia results/explorer/net_check_amd
```

Tonight's rocm6.3 jobs also set `RESULTS_ROOT=results/explorer/rocm63`, to keep
their results apart from the default venv's.
