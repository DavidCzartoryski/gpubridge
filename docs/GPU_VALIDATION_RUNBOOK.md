# GPU validation runbook

Step-by-step order for closing the open items in
[HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md) once GPUs are available.
Everything is scripted under [`scripts/gpu/`](../scripts/gpu), so a step is one
command, and every step writes its records to `results/<target>/`. Every shell
script accepts `--dry-run`, which prints the commands without running them.
Try that first on any new machine.

## Split-test mode, in one paragraph

`GPUBRIDGE_SPLIT_TEST=half|alternate|node` labels some ranks of an all-NVIDIA
(or all-AMD) job as the other vendor. It is **test only**, and that
environment variable is the only way to turn it on. Ranks keep their real
GPUs and real NCCL/RCCL, so a split test exercises the whole GPU code path:
two native islands, the Gloo bridge, reduce-bridge-broadcast on device
tensors, and `new_group` on ranks outside the group. Only NCCL talking to RCCL
is left untested. Split runs are flagged everywhere, so they can't pass for a
real mixed-vendor result:

- every rank logs a `SplitTestWarning`;
- every JSON record has `run.kind = "split-test"`, `run.split_test`, and a
  `run.warning`;
- every `summary.md` starts with a banner.

## Order at a glance

| # | Where | Step | What it proves | Items closed | Time (compute) | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Explorer login node | `submit.sh setup` | venv with the CUDA build of torch 2.14.1 | - | ~10 min | free |
| 1 | Explorer, 1 GPU | `submit.sh 01` | the build runs on the GPU; records the hardware | prerequisite | ~2 min | free |
| 2 | Explorer, 1 GPU | `submit.sh 02` | two single-rank NCCL islands sharing one GPU, plus the bridge | **3**, 6 (partly), 10 on CUDA | ~3 min | free |
| 3 | Explorer, 2+ GPUs | `submit.sh 03` | real NCCL island; each rank drives its own GPU | **5** on NVIDIA, baseline | ~5 min | free* |
| 4 | Explorer, 4 GPUs | `submit.sh 04` | two 2-GPU NCCL islands and the bridge, both layouts; native vs bridged benchmark | **3, 6, 7**, 9 (data), **11** | ~10 min | free* |
| 5 | Explorer, 2 nodes | `submit.sh 05` | NCCL across nodes; the bridge crossing the network between nodes | **8**, 11 across nodes | ~10 min | free* |
| 6 | Explorer, 4 GPUs | `submit.sh 06` | training throughput and scaling 1 to 4 GPUs (table for the multigpu request) | - | ~10 min | free* |
| 7 | AMD cloud machine | `amd/run_all.sh` | RCCL islands, split test on AMD, benchmark, scaling | **4, 5** on AMD, 6, 7, 10 on ROCm, 11 | 20-30 min | ~0.5 h of the hourly rate |
| 8 | NVIDIA + AMD cloud | `mixed/node.sh` on both | the real mixed-vendor job | **1, 2** on GPUs, **8** across networks | 30-45 min | ~0.75 h of each machine's rate |

\* Steps 3 to 6 need the `multigpu` partition, which needs an access request
(see [Explorer](#explorer)). Steps 1 and 2 run on the open `gpu` partition.
Times exclude queue waits. For cost, multiply by the provider's current hourly
rate; for example, at a hypothetical $3/h, step 7 costs about $1.50.

Run them in this order. Each step is cheaper than the next and catches the
problems the next one would otherwise waste time on.

## Explorer

### Settings to verify first

All site settings live at the top of
[`scripts/gpu/explorer/submit.sh`](../scripts/gpu/explorer/submit.sh). Override
any of them with an environment variable of the same name. The defaults come
from the [RC docs](https://rc-docs.northeastern.edu/en/explorer-main/gpus/accessinggpus.html)
as of 2026-10-06; **verify each one**:

| Setting | Default | Check in the RC docs |
| --- | --- | --- |
| `PARTITION_1GPU` | `gpu` | name of the open GPU partition (1 GPU per job) |
| `PARTITION_MULTI` | `multigpu` | name of the multi-GPU partition, and that you have access |
| `GPU_TYPE_1GPU`, `GPU_TYPE_MULTI` | empty (any GPU) | `--gres` type strings, e.g. `v100-pcie`, `v100-sxm2`, A100/H100/H200 names |
| `ACCOUNT` | empty | whether jobs need `--account` |
| `CPUS_PER_GPU`, `MEM_PER_GPU` | `4`, `32G` | per-GPU CPU and memory limits |
| `TIME_01` ... `TIME_06` | 10-30 min | maximum walltime per partition |
| `WORK_DIR` | `~/gpubridge-gpu` | home quota; the venv and caches need ~6 GB (scratch is an option, but check its purge policy) |
| `MODULES` | empty | none needed: the PyTorch wheel bundles CUDA and uv brings Python |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | empty (auto) | name of the fast interconnect interface for step 05, e.g. `ib0` |

Also check:

- **CUDA wheel vs GPU.** `setup_env.sh` installs the `cu126` build, because
  CUDA 13 builds dropped V100 (sm_70). Step 01 fails with a clear message if
  the build has no kernels for your GPU; set `CUDA_INDEX` (e.g. `cu128`) and
  rerun setup.
- **multigpu access.** The request form asks for run times on 1, 2, 3 and 4
  GPUs with scaling efficiency above about 0.5. Step 06 produces exactly that
  table (`results/explorer/06_scaling/summary.md`), so run it as soon as RC
  grants a test window.

### Steps

```bash
git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
scripts/gpu/explorer/submit.sh --dry-run free   # see what will be submitted
scripts/gpu/explorer/submit.sh setup            # on the login node, once (~10 min)
scripts/gpu/explorer/submit.sh free             # steps 01 and 02
squeue -u "$USER"                               # wait for them
cat results/explorer/02_split_shared_gpu/check/summary.md
# after multigpu access:
scripts/gpu/explorer/submit.sh multi            # steps 03 to 06
```

Job logs go to `$WORK_DIR/logs/`, and results to `results/explorer/<step>/`.
Commit the results directory when a step passes.

## AMD cloud machine

Pick any Linux machine or container with ROCm and AMD GPUs (e.g. MI300X). An
image that already has a ROCm build of PyTorch saves the 6+ GB download, with
`TORCH_FLAVOR=system`. Then:

```bash
git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
bash scripts/gpu/amd/run_all.sh --dry-run       # 1 second, no cost surprises
bash scripts/gpu/amd/run_all.sh                 # everything; ~20-30 min
```

The script runs:

1. setup
2. `rocm-smi` / `amd-smi` / `rocminfo` output
3. probe
4. device mapping under `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`
5. a 1-GPU check with `NCCL_DEBUG=INFO`, then greps the log for RCCL's version
   (item 4)

With 2+ GPUs it continues with:

6. an RCCL island
7. a split test
8. the native vs bridged benchmark
9. training scaling

With 1 GPU it runs a shared-GPU split test instead.

A failing step is logged and skipped, so one problem doesn't waste the
session. At the end the script prints the exact `scp` command for the results
tarball. **Download it before stopping the machine.** To rerun after fixing
something, `SKIP_SETUP=1` skips the install.

## Real mixed-vendor job (two cloud machines)

Explorer's compute nodes reach the internet only through an HTTP proxy, so
they can't join a job with a machine outside the cluster. Use two cloud
machines: one NVIDIA, one AMD.

### Network: what gpubridge needs

Gloo, which carries discovery, the world group and the bridge, connects every
rank to every other rank on **random TCP ports, in both directions**. So the
two machines need a flat network with all TCP open between them, not just the
rendezvous port. Opening every port on public IPs isn't safe; use an overlay.

**Tailscale (default).** On both machines:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # prints a login URL the first time
tailscale ip -4              # this machine's tailnet address, e.g. 100.64.0.1
ip -4 addr show tailscale0   # the interface the scripts use (IFNAME=tailscale0)
```

- Tailscale needs a TUN device. That works on VMs, and in containers started
  with `/dev/net/tun` and `NET_ADMIN`. Check with `ls -l /dev/net/tun`.
- In an unprivileged container, which is common for GPU pods, only userspace
  mode is available. That mode is a proxy, which Gloo can't use. In that case
  pick a VM, or use the provider's private network (below).
- The default tailnet policy allows all traffic between your machines. If you
  use ACLs, allow all TCP between the two.

**WireGuard (alternative).** Also needs root and a TUN-capable kernel or
container. Generate keys on each machine
(`wg genkey | tee privatekey | wg pubkey > publickey`), then write
`/etc/wireguard/wg0.conf`:

```ini
[Interface]
PrivateKey = <this machine's private key>
Address = 10.99.0.1/24            # 10.99.0.2/24 on the other machine
ListenPort = 51820

[Peer]
PublicKey = <the other machine's public key>
AllowedIPs = 10.99.0.2/32         # 10.99.0.1/32 on the other machine
Endpoint = <other public IP>:51820  # needed on at least one side
PersistentKeepalive = 25
```

Start it with `sudo wg-quick up wg0`. UDP 51820 must be reachable on at least
one side. Use `IFNAME=wg0`, and the `10.99.0.x` addresses for `MASTER_ADDR`
and `LOCAL_ADDR`.

**Same provider, private network.** If both machines are in one provider
project or region with private networking and open internal firewall rules,
use those private IPs and that interface directly.

### Launch

Use the same PyTorch release on both machines; `setup_env.sh` installs 2.14.1
by default. Start the master first, then the worker within about 90 seconds
(the preflight timeout):

```bash
# NVIDIA machine (master):
ROLE=master MASTER_ADDR=100.64.0.1 bash scripts/gpu/mixed/node.sh
# AMD machine (worker):
ROLE=worker MASTER_ADDR=100.64.0.1 bash scripts/gpu/mixed/node.sh
```

`node.sh` sets up the venv. Then it runs `preflight.py`, which checks:

1. the address resolves
2. the interface exists
3. TCP reaches the master
4. a real two-machine Gloo handshake succeeds

The preflight **stops with a specific message before torchrun starts** if any
check fails. After that, the script runs the correctness check and a bridged
benchmark, and packs the results.

Ports: torchrun uses `MASTER_PORT` (29500) and the preflight uses
`PREFLIGHT_PORT` (29501), plus Gloo's random ports. All are covered by an
overlay network.

Each machine only holds its own ranks' records. For the full verdict, copy
both `results/mixed-*/check/rank*.json` sets into one directory and run
`python scripts/gpu/summarize.py <dir>`. That run's `run.kind` must be
`gpu-mixed`, with `split_test` null.

## Reading results

- `results/<target>/<step>/rank<N>.json`: one record per rank. It holds:
  - `environment`: host, GPUs with PCI bus ids, driver-visible devices, torch,
    CUDA/HIP and NCCL/RCCL versions, relevant env vars, `gpubridge.probe()`;
  - `run`: kind, split mode, islands, device;
  - `stages`: pass/fail per stage, with full tracebacks on failure;
  - `rank<N>.stacks.txt`: a stack dump if the rank hung in a stage.
- `summary.md` / `summary.json`: the per-stage table, missing ranks, and GPUs
  shared outside a split test.
- `steps.jsonl` and `<step>.log`: exit code and full output of every scripted
  step.
- `bench/bench.csv`: latency and bandwidth per size, for native and gpubridge.

`run.kind` values:

| `run.kind` | Meaning |
| --- | --- |
| `gpu-mixed` | real GPUs, both vendors: the result that closes items 1 and 2 |
| `gpu-single-vendor` | real GPUs, one vendor, no bridge |
| `split-test` | islands faked with `GPUBRIDGE_SPLIT_TEST`: never a mixed-vendor result |
| `cpu` | simulation or CPU-only mode |

## If a step fails

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| 01: `build_supports_gpus` fails | wheel without kernels for this GPU | set `CUDA_INDEX` (cu126 for V100; cu128/cu130 for newer GPUs), rerun setup |
| 01: no GPU visible | job didn't get a GPU, or driver problem | check `nvidia-smi.txt` and `slurm-job.txt` in the results |
| `init` stuck (summary says "stuck in this stage") | new_group / NCCL init problem (item 3) | read `rank*.stacks.txt`; rerun with `NCCL_DEBUG=INFO`; this is a real finding, file it |
| `busy_gpu` wrong values | stream ordering bug in the bridge (item 6) | serious; keep the JSON and open an issue |
| `dtypes` mismatch | precision or reduction-order problem (item 7) | check `max_abs_error` in the stage |
| timeouts during large all_reduces | NCCL/RCCL watchdog vs a slow bridge (item 9) | raise `--timeout`; note the `all_reduce_seconds` of `busy_gpu` |
| 05: rendezvous never completes | wrong interface or blocked ports between nodes | set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` (e.g. `ib0`) |
| 03/AMD: `device_map` mismatch | visible-devices variable maps to a different GPU (item 5) | record it; check scheduler GPU binding |
| AMD: `rccl_version` fails | RCCL didn't log, or this isn't a ROCm build | check `check_1gpu.log` and `probe/probe.json` (`torch_hip`) |
| mixed: preflight `resolve` / `tcp` fails | wrong address, or overlay down | `tailscale status`; use `tailscale ip -4` of the master |
| mixed: preflight `gloo` fails, `tcp` passes | rendezvous port open, random ports blocked | use the overlay addresses and interface, not public IPs |
| NCCL "unhandled system error" in a container | too little shared memory | start the container with a larger `--shm-size` |
