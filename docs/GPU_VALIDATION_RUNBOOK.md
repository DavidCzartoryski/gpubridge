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
| 0a | Explorer login node | `submit.sh 00` | GPU types in `sharing`, `gpu` and `multigpu` (sinfo only): are there AMD GPUs in `sharing`? | - | seconds | free |
| 0b | Explorer, CPU job | `submit.sh setup` | venv with the CUDA build of torch 2.14.1, built on a compute node | - | ~10 min | free |
| 1 | Explorer, 1 GPU | `submit.sh 01` | the build runs on the GPU; records the hardware; a rank whose `LOCAL_RANK` has no GPU fails `init()` on every rank | prerequisite, **12** | ~2 min | free |
| 2 | Explorer, 1 GPU | `submit.sh 02` | two single-rank NCCL islands sharing one GPU, plus the bridge | **3**, 6 (partly), 10 on CUDA | ~3 min | free |
| 3 | Explorer, 2+ GPUs | `submit.sh 03` | real NCCL island; each rank drives its own GPU | **5** on NVIDIA, baseline | ~5 min | free* |
| 4 | Explorer, 4 GPUs | `submit.sh 04` | two 2-GPU NCCL islands and the bridge, both layouts; native vs bridged benchmark | **3, 6, 7**, 9 (data), **11**, 13 | ~10 min | free* |
| 5 | Explorer, 2 nodes | `submit.sh 05` | NCCL across nodes; the bridge crossing the network between nodes | **8**, 11 and 13 across nodes | ~10 min | free* |
| 6 | Explorer, 4 GPUs | `submit.sh 06` | training throughput and scaling 1 to 4 GPUs (table for the multigpu request) | - | ~10 min | free* |
| 7 | AMD cloud machine | `amd/run_all.sh` | RCCL islands, split test on AMD, benchmark, scaling | **4, 5** on AMD, 6, 7, 10 on ROCm, 11, 12, 13 | 20-30 min | ~0.5 h of the hourly rate |
| 8 | NVIDIA + AMD cloud | `mixed/node.sh` on both | the real mixed-vendor job | **1, 2** on GPUs, **8** across networks | 30-45 min | ~0.75 h of each machine's rate |
| 8 alt | Explorer `gpu` + `sharing`, one heterogeneous job | `submit.sh setup-rocm`, then `submit.sh 07` | the real mixed-vendor job inside Explorer, **if** step 0a finds AMD GPUs in `sharing`. Dry-run only until then | **1, 2** on GPUs, 3 with real RCCL ranks, 8 between nodes, 11 against per-island baselines, 13 | ~15 min + ~15 min ROCm setup | free |

\* Steps 3 to 6 need the `multigpu` partition, which needs an access request
(see [Explorer](#explorer)). Steps 1 and 2 run on the open `gpu` partition.
Step 8 alt replaces step 8, and saves the cloud machines, only if Explorer has
AMD GPUs; see [Mixed-vendor job inside Explorer](#mixed-vendor-job-inside-explorer).
Every step that runs `check.py` also covers items 14 to 16 (reductions,
all_gather, reduce_scatter, async, and where NCCL/RCCL and Gloo agree).
Steps 04, 05, 07 and the AMD script also check every opt-in policy in
`OPT_IN_POLICIES` and write `thresholds.json` (items 17 to 19). Opt-in policies
are not validated on GPUs until those steps pass.
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
| `PARTITION_SETUP` | `short` | a CPU partition for the setup job (RC's interactive example uses `short`) |
| `PARTITION_1GPU` | `gpu` | name of the open GPU partition (1 GPU per job) |
| `PARTITION_MULTI` | `multigpu` | name of the multi-GPU partition, and that you have access |
| `PARTITION_AMD` | `sharing` | the partition step 00 checks for AMD GPUs (step 07 only) |
| `GPU_TYPE_1GPU`, `GPU_TYPE_MULTI` | empty (any GPU) | `--gres` type strings, e.g. `v100-pcie`, `v100-sxm2`, A100/H100/H200 names |
| `GPU_TYPE_AMD` | empty, required for step 07 | the AMD `--gres` type from step 00's `gpu-types.txt` |
| `ACCOUNT` | empty | whether jobs need `--account` |
| `CPUS_PER_GPU`, `MEM_PER_GPU` | `4`, `32G` | per-GPU CPU and memory limits |
| `CPUS_SETUP`, `MEM_SETUP`, `TIME_SETUP` | `4`, `16G`, 1 h | limits of the setup partition |
| `TIME_01` ... `TIME_07` | 10-30 min | maximum walltime per partition |
| `PROJECT_DIR` | none: **required** | your group's project space, e.g. `/projects/<your-project>`; set it in the environment or in `scripts/gpu/explorer/local.env` (see [Steps](#steps)) |
| `WORK_DIR` | `$PROJECT_DIR/$USER/gpubridge-gpu` | venvs, caches and job logs: ~7 GB, ~25 GB with the ROCm venv. Not home, which is capped at 75 GB |
| `VENV`, `VENV_ROCM` | `$WORK_DIR/venv`, `$WORK_DIR/venv-rocm` | - |
| `ROCM_TORCH_VERSION`, `ROCM_INDEX`, `ROCM_INDEX_URL` | `2.14.1`, `rocm7.2`, the PyTorch index for `ROCM_INDEX` | the ROCm build for `setup-rocm`; see [Choosing the ROCm build](#choosing-the-rocm-build) |
| `PARTITION_MIXED_NVIDIA`, `GPUS_MIXED_NVIDIA`, `GPUS_MIXED_AMD` | `gpu`, `1`, `1` | step 07's GPUs per side |
| `SETUP_MODULES` | `explorer` | loaded before installing: it routes compute nodes through the proxy. Set it to `""` to load nothing |
| `MODULES` | empty | modules for steps 01 to 07; none needed: the PyTorch wheel bundles CUDA/ROCm and uv brings Python |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | empty (auto) | name of the fast interconnect interface for steps 05 and 07, e.g. `ib0` |

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
echo 'PROJECT_DIR=/projects/<your-project>' > scripts/gpu/explorer/local.env   # once
scripts/gpu/explorer/submit.sh 00               # GPU types per partition (sinfo, seconds)
scripts/gpu/explorer/submit.sh --dry-run free   # see what will be submitted
scripts/gpu/explorer/submit.sh setup            # a short CPU job, once (~10 min)
squeue -u "$USER"                               # wait for the setup job to finish
scripts/gpu/explorer/submit.sh free             # steps 01 and 02
squeue -u "$USER"                               # wait for them
cat results/explorer/02_split_shared_gpu/check/summary.md
# after multigpu access:
scripts/gpu/explorer/submit.sh multi            # steps 03 to 06
```

**Your settings.** Every step except 00 needs `PROJECT_DIR`, your group's
project space, and stops with an error until it's set. Put it in the
environment or in `scripts/gpu/explorer/local.env`, which git ignores. That
file is parsed, not run: one `NAME=value` per line, and any setting from
`submit.sh` works there. The environment overrides it.

**Setup runs on a compute node, never the login node.** `submit.sh setup`
submits [`setup.sbatch`](../scripts/gpu/explorer/setup.sbatch) to the
`short` partition. The job loads the `explorer` module first, because compute
nodes reach the internet only through the proxy that module sets up; then it
installs uv, Python 3.12 and torch into `$WORK_DIR/venv`. Its log is
`$WORK_DIR/logs/gpubridge-setup-cuda-<job id>.out`. `setup_env.sh` itself
refuses to run on a Slurm login node (sbatch present, no job), so a direct call
there fails instead of installing.

**Or, from an interactive job:**

```bash
srun -p short -N 1 -n 1 -c 4 --mem=16G -t 01:00:00 --pty bash   # wait for a shell on a compute node
cd gpubridge
GPUBRIDGE_REPO=$PWD bash scripts/gpu/explorer/setup.sbatch      # loads explorer, then installs
exit
```

Step 00 runs `sinfo` on the login node: it only queries Slurm, and starts
nothing. It writes to `results/explorer/00_partitions/`:

- `sinfo-<partition>.txt`: `sinfo -p <partition> -o "%20N %10c %10m %25f %10G %10t"` for `sharing`, `gpu` and `multigpu`;
- `gres-<partition>.txt`: each node's full GRES string, which that table's 10-character column truncates;
- `gpu-types.txt`: every GPU type per partition, labelled `amd`, `nvidia` or `unknown` from its name, and a closing line that says whether `sharing` has AMD GPUs.

Job logs go to `$WORK_DIR/logs/`, and results to `results/explorer/<step>/`.
Commit the results directory when a step passes.

### Mixed-vendor job inside Explorer

**Dry-run only until step 00 shows AMD GPUs in `sharing`.** If it does, one
Slurm heterogeneous job can run the real mixed-vendor test without cloud
machines:

- component 0 is on `gpu` (`PARTITION_MIXED_NVIDIA`): `GPUS_MIXED_NVIDIA`
  NVIDIA GPUs, default 1, using the CUDA venv;
- component 1 is on `sharing`: `GPUS_MIXED_AMD` AMD GPUs of type
  `GPU_TYPE_AMD`, using a ROCm venv built by a setup job on an AMD node.

Both components sit on the cluster network, so no overlay is needed.

```bash
scripts/gpu/explorer/submit.sh 00                  # then read 00_partitions/gpu-types.txt
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh --dry-run setup-rocm
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh --dry-run 07
# once the hardware is confirmed and AMD_NODES_CONFIRMED=1 is merged in submit.sh:
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh setup-rocm   # ROCm venv (~16 GB), on an AMD node
GPU_TYPE_AMD=<type> scripts/gpu/explorer/submit.sh 07
```

Until then, `setup-rocm` and `07` stop with an error unless `--dry-run` is
given. The switch, `AMD_NODES_CONFIRMED` in `submit.sh`, is deliberately not
an environment variable: turning the path on is a reviewed one-line change.

[`07_mixed_hetjob.sbatch`](../scripts/gpu/explorer/07_mixed_hetjob.sbatch)
runs each phase as one heterogeneous job step (`srun --het-group=0 ... :
--het-group=1 ...`), each side with its own venv's `python` or `torchrun`:

1. `probe` on each side (`probe_nvidia/`, `probe_amd/`), plus `rocm-smi` and `amd-smi` from the AMD node;
2. `preflight`: the gpu node as master, the AMD node as worker. The job stops here if either side lacks a usable GPU or Gloo can't connect;
3. `check`: one torchrun per node, with the c10d rendezvous on the gpu node. Both write into `check/` on the shared filesystem, so `summarize` covers every rank without copying. `run.kind` must be `gpu-mixed`, with `split_test` null;
4. `bench` (`--ops island,gpubridge`): first each island times its own native
   all_reduce, both at once (NCCL on the gpu node, RCCL on the AMD node; rows
   `island:nvidia` and `island:amd`), then the bridged all_reduce across both.
   `summary.md` adds a `gpubridge / slowest island` column. A native
   all_reduce over every rank is impossible here, because NCCL and RCCL ranks
   can't form one group, so the islands are the baselines.

**Per-island baselines need 2+ GPUs per island.** With one GPU, an island's
all_reduce has nothing to exchange, so its row only times the call itself.
`gpu` allows one GPU per job. For real baselines, set `GPUS_MIXED_AMD=2`, and,
with multigpu access, `PARTITION_MIXED_NVIDIA=multigpu GPUS_MIXED_NVIDIA=2`.

Also check with RC before the first real run:

- that heterogeneous jobs are allowed, and can span `gpu` and `sharing`;
- `sharing`'s walltime and preemption rules;
- which ROCm build supports the AMD GPUs; see below.

Interface names may differ between the two partitions' nodes, so leave
`GLOO_SOCKET_IFNAME` empty unless the preflight's interface check fails.

#### Choosing the ROCm build

Choose `setup-rocm`'s PyTorch from the AMD GPU model that step 00 finds. The
default, torch 2.14.1 for ROCm 7.2, may not support older Instinct cards. The
GRES type usually names the model:

| GPU | gfx target |
| --- | --- |
| MI50, MI60 | gfx906 |
| MI100 | gfx908 |
| MI210, MI250, MI250X | gfx90a |
| MI300A, MI300X | gfx942 |

1. Find the newest ROCm release that supports that gfx target in AMD's
   [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
2. Pick the newest PyTorch release with a wheel for that ROCm line and
   Python 3.12 on [download.pytorch.org/whl](https://download.pytorch.org/whl/).
   Index names look like `rocm6.4`.
3. Set them for `setup-rocm`, e.g.
   `ROCM_TORCH_VERSION=2.9.1 ROCM_INDEX=rocm6.4`. Set `ROCM_INDEX_URL` only
   for a mirror or another index; it defaults to
   `https://download.pytorch.org/whl/$ROCM_INDEX`.

`setup-rocm` ends with a probe on the AMD node, and fails if the build has no
kernels for the GPU. `results/explorer/setup_rocm/probe.json` lists the GPU's
target and the build's targets.

The CUDA side stays at 2.14.1. On CPU, mixed releases from 2.9.1 to 2.14.1
worked (item 2 in HARDWARE_VALIDATION.md). An older ROCm-side release is
untested, and `gpubridge.init()` warns when releases differ. For identical
releases, set `TORCH_VERSION` (and `CUDA_INDEX` if needed) for `submit.sh setup` too.

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
4. device mapping under `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`, then
   one more process than GPUs, which must fail `init()` on every rank (item 12)
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
machines: one NVIDIA, one AMD. If step 00 finds AMD GPUs on Explorer, the
[heterogeneous job](#mixed-vendor-job-inside-explorer) does the same for free.

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
| "set PROJECT_DIR" | no project space configured | put `PROJECT_DIR=/projects/<your-project>` in `scripts/gpu/explorer/local.env`, or export it |
| setup-rocm: the probe at the end fails | the ROCm build has no kernels for this AMD GPU | pick another build: [Choosing the ROCm build](#choosing-the-rocm-build) |
| setup: downloads time out or can't connect | the job has no route out | check that the log shows `module load explorer` ran; `SETUP_MODULES` must include `explorer` |
| setup: "this looks like a Slurm login node" | `setup_env.sh` was run outside a job | use `submit.sh setup`, or run it inside `srun --pty bash` |
| 00: no AMD GPU type found in `sharing` | no AMD nodes there, or an unrecognized type name | read `gres-sharing.txt`; if the AMD GPUs are there under another name, set `GPU_TYPE_AMD` to it; otherwise use step 8 |
| 07: "not a heterogeneous job" | the script was submitted with plain `sbatch` | submit with `submit.sh 07` |
| 07: stops after `preflight` | no route between the `gpu` and `sharing` nodes, or a GPU the build can't run | read `preflight-*.json` and `probe_*/probe.json`; set `GLOO_SOCKET_IFNAME` if the interface check failed |
| 01: `build_supports_gpus` fails | wheel without kernels for this GPU | set `CUDA_INDEX` (cu126 for V100; cu128/cu130 for newer GPUs), rerun setup |
| 01: no GPU visible | job didn't get a GPU, or driver problem | check `nvidia-smi.txt` and `slurm-job.txt` in the results |
| 01/AMD: `local_rank_check` fails | init() succeeded with more processes than GPUs, or failed for another reason | read `expected_init_error` in `local_rank_check/rank*.json`; if init succeeded, the visible-device count isn't what PyTorch reports (item 12) |
| `init` stuck (summary says "stuck in this stage") | new_group / NCCL init problem (item 3) | read `rank*.stacks.txt`; rerun with `NCCL_DEBUG=INFO`; this is a real finding, file it |
| `busy_gpu` wrong values | stream ordering bug in the bridge (item 6) | serious; keep the JSON and open an issue |
| `dtypes` mismatch | precision or reduction-order problem (item 7) | check `max_abs_error` in the stage |
| `reductions`, `all_gather` or `reduce_scatter` mismatch | a MAX/MIN/AVG or tensor-collective path differs on NCCL/RCCL (item 15) | the failing `cases` name the dtype and op; rerun with `--policy flat-gloo` to see whether the policy or the backend is at fault |
| `async` wrong values | the worker's stream didn't wait for the caller's, or `wait()` didn't order the caller's stream (item 16) | serious; keep the JSON and open an issue |
| `reduction_agreement` shows `product_agree` false | NCCL/RCCL and Gloo disagree on PRODUCT for that dtype (item 14) | not a failure: it's the evidence item 14 asks for; record it |
| timeouts during large all_reduces | NCCL/RCCL watchdog vs a slow bridge (item 9) | raise `--timeout`; note the `all_reduce_seconds` of `busy_gpu` |
| 05: rendezvous never completes | wrong interface or blocked ports between nodes | set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` (e.g. `ib0`) |
| 03/AMD: `device_map` mismatch | visible-devices variable maps to a different GPU (item 5) | record it; check scheduler GPU binding |
| AMD: `rccl_version` fails | RCCL didn't log, or this isn't a ROCm build | check `check_1gpu.log` and `probe/probe.json` (`torch_hip`) |
| mixed: preflight `resolve` / `tcp` fails | wrong address, or overlay down | `tailscale status`; use `tailscale ip -4` of the master |
| mixed: preflight `gloo` fails, `tcp` passes | rendezvous port open, random ports blocked | use the overlay addresses and interface, not public IPs |
| NCCL "unhandled system error" in a container | too little shared memory | start the container with a larger `--shm-size` |
