# A first session on Explorer

The exact commands for a first session on the Explorer cluster, from a fresh
login to a summary you can paste into a review. It covers rungs 1 and 2 of the
ladder ([GPU_VALIDATION_RUNBOOK.md](GPU_VALIDATION_RUNBOOK.md), "The ladder"):
one NVIDIA GPU, then the split test on one GPU. More GPUs follow if the
multigpu partition is open to you.

Nothing in this repository has run on GPUs yet: this session is the first
attempt, and any step may fail. That's what the session is for: the kit
records what happened, and `triage.py` summarizes it.

`$PROJECT_DIR` below stands for your group's project space,
`/projects/<your-project>`: the directory your group was given under
`/projects` (ask your group if you don't know it). Venvs, caches and logs go
under `$PROJECT_DIR/$USER/gpubridge-gpu`, not the home directory, which is
capped at 75 GB (only uv itself installs into `~/.local/bin`).

In a new shell, for example after a long wait in the queue, set it again and
go back to the clone first:

```bash
PROJECT_DIR=/projects/<your-project>
cd "$PROJECT_DIR/$USER/gpubridge"
```

## Before you start

- An Explorer account, and a project space you can write to.
- About an hour. By the runbook's estimates (nothing has run yet), setup
  takes about 10 minutes once it runs, steps 01 and 02 about 2 and 3
  minutes, and every job waits in the queue first.
- Nothing to install: the setup job builds the venv on a compute node.

## 1. Clone into the project space

```bash
PROJECT_DIR=/projects/<your-project>          # yours
mkdir -p "$PROJECT_DIR/$USER" && cd "$PROJECT_DIR/$USER"
git clone https://github.com/DavidCzartoryski/gpubridge && cd gpubridge
git log --oneline -1                          # note the commit: the records name it too
```

Results go to `results/explorer/` inside this clone. Cloning into the
project space keeps them off the home quota.

## 2. local.env: your settings, once

```bash
echo "PROJECT_DIR=$PROJECT_DIR" > scripts/gpu/explorer/local.env
cat scripts/gpu/explorer/local.env
```

`local.env` is gitignored, so your project path never reaches a commit.

- **Format:** `NAME=value` lines only. Lines starting with `export` are
  ignored. A `# comment` on the same line, or single quotes, become part of
  the value. A variable set in the environment wins over the file.
- **Required:** `PROJECT_DIR`, or `WORK_DIR` instead. Without it every step
  except 00 stops with "set PROJECT_DIR to your group's project space".
- **Optional, only if your site needs them:**
  - `ACCOUNT=<slurm account>`, if jobs need `--account`;
  - `GPU_TYPE_1GPU=<gres type>` and `GPU_TYPE_MULTI=<gres type>`, to ask for
    one GPU model on the gpu and multigpu partitions;
  - `PARTITION_1GPU`, `PARTITION_MULTI`, `PARTITION_SETUP`, if the defaults
    `gpu`, `multigpu` and `short` don't match;
  - `EXTRA_SBATCH`, for anything else every job needs, such as a `--qos`.

  Every setting and its default is at the top of
  `scripts/gpu/explorer/submit.sh`.

## 3. Step 00: what the partitions offer (login node, seconds)

```bash
scripts/gpu/explorer/submit.sh 00
```

It runs only `sinfo`, on the login node. It prints a table (partition, GPU
type, vendor, nodes, GPUs) and one verdict line:
- "AMD GPUs in sharing: ... Set GPU_TYPE_AMD to one of them...", or
- "No AMD GPU type found in sharing; check ... by hand".

The files land in `results/explorer/00_partitions/`: `gpu-types.txt`, and
`sinfo-*.txt` and `gres-*.txt` per partition. The AMD steps (rungs 3 and 4)
stay dry-run only until someone confirms AMD nodes: not today.

## 4. Look before you submit (seconds)

```bash
scripts/gpu/explorer/submit.sh --dry-run free
```

It prints the `sbatch` commands steps 01 and 02 would run, without submitting
anything. Check the partition, `--gres=gpu:...`, `--time` and `--account`.

## 5. The setup job (about 10 minutes, estimated, plus the queue)

```bash
scripts/gpu/explorer/submit.sh setup
```

It submits a short CPU job that builds the venv: torch 2.14.1 for CUDA 12.6
and gpubridge from this clone, under `$PROJECT_DIR/$USER/gpubridge-gpu/venv`.
It prints the job id and its log's path,
`$PROJECT_DIR/$USER/gpubridge-gpu/logs/gpubridge-setup-cuda-<job id>.out`.

```bash
squeue -u "$USER"                                  # until the job is gone
sacct -j <job id> --format=JobID,JobName%30,State,Elapsed,ExitCode
tail -3 "$PROJECT_DIR/$USER/gpubridge-gpu/logs/gpubridge-setup-cuda-<job id>.out"
```

**Pass:** `sacct` says COMPLETED 0:0, and the log ends with
`environment ready: .../venv`. Wait for this before step 6: the next
command doesn't wait for setup.

## 6. Rungs 1 and 2 on the free gpu partition

```bash
scripts/gpu/explorer/submit.sh free           # submits 01 and 02
squeue -u "$USER"
```

| Step | What it runs | Slurm limit | Expected (estimate) |
| --- | --- | --- | --- |
| 01 probe | 1 GPU: `probe.py` (the GPU, driver, torch, CUDA, NCCL), then two processes checking their local ranks | 10 min | about 2 min |
| 02 split test | 1 GPU, 2 processes as two islands: `check.py` on device tensors, a shared-GPU check that must refuse, `pytest -m gpu` | 25 min | about 3 min |

- **Logs:** `$PROJECT_DIR/$USER/gpubridge-gpu/logs/gpubridge-01_probe-<job id>.out`
  and `gpubridge-02_split_shared_gpu-<job id>.out`.
- **Results:** `results/explorer/01_probe/` and
  `results/explorer/02_split_shared_gpu/`. Each has a `<step>.log` per step, a
  line per step in `steps.jsonl`, and JSON records. A failed step's log ends
  with the tail of its NCCL logs.
- **Pass:** the job log ends with `all steps passed`. Otherwise it ends with
  `FAILED: <steps>`, and the job exits 1.

```bash
sacct -u "$USER" -S today --format=JobID,JobName%30,State,Elapsed,ExitCode
tail -2 "$PROJECT_DIR/$USER/gpubridge-gpu/logs/"gpubridge-0[12]_*-*.out
cat results/explorer/02_split_shared_gpu/check/summary.md
```

A failed step doesn't stop the job: the kit runs the next step and records
each one.

## 7. If you have the multigpu partition (rungs 1 and 2, continued)

Only once step 6's jobs passed: GPU time on a rung waits for the one below
(the runbook's ladder). Without multigpu access, `sbatch` may refuse the job
when you submit it, and `submit.sh multi` then stops at 03.

```bash
scripts/gpu/explorer/submit.sh --dry-run multi   # look first
scripts/gpu/explorer/submit.sh multi             # 03 to 06
```

| Step | GPUs | Slurm limit | Expected (estimate) |
| --- | --- | --- | --- |
| 03 single island | 2 on 1 node | 20 min | about 5 min |
| 04 split test | 4 on 1 node | 40 min | about 10 min |
| 05 two nodes | 2 nodes, 2 GPUs each | 30 min | about 10 min |
| 06 scaling | 4 on 1 node, runs on 1 to 4 of them | 30 min | about 10 min |

They are submitted back to back and queue on their own.

## 8. What to paste back for review

```bash
"$PROJECT_DIR/$USER/gpubridge-gpu/venv/bin/python" scripts/gpu/triage.py results/explorer
```

It prints a summary, under 80 lines:
- problems first: each step that failed, timed out or never finished, with the
  last error lines of its log;
- what ran where;
- the key numbers;
- one verdict line.

Host names become `host N`, IP addresses `<ip>`, and paths lose their owner.

It needs Python 3.10 or later (no torch), so it runs with the venv's Python
that setup made: a login node's `python3` may be older. If setup never
finished, any Python 3.10 or later works.

triage reads only job directories with a `steps.jsonl`. A job that died
before or during its first step has none, so triage leaves it out, and its
verdict can still say everything that ran passed. The setup job writes no
results at all. The `sacct` lines catch both.

Paste back:
1. **The triage output.**
2. **`git log --oneline -1`** of the clone.
3. **The `sacct` lines** of today's jobs (command in step 6).
4. **`cat results/explorer/00_partitions/gpu-types.txt`**: step 00's GPU
   types and counts per partition, and whether `sharing` has AMD GPUs. Not
   the `sinfo-*` or `gres-*` files: they name nodes.
5. **For every job `sacct` shows FAILED, TIMEOUT or CANCELLED that triage
   doesn't list** (or all of them, if triage says "no steps.jsonl"), and for
   a failed setup job: the last 30 lines of its job log under
   `$PROJECT_DIR/$USER/gpubridge-gpu/logs/`. Those lines are raw: replace the
   host name, the project directory and your user name in them before
   pasting (for example `job 123 on <host>: results in $PROJECT_DIR/...`).

Read the triage output before pasting all the same: its error lines come from
the step logs. The raw results stay on the cluster. Don't commit `results/explorer/` to the
public repository: step logs, stack dumps and NCCL logs are not redacted.
`validation_tables.py` makes the publishable tables later.

## If something fails

| What you see | What it means, and what to do |
| --- | --- |
| "set PROJECT_DIR to your group's project space" | `local.env` is missing or misspelled: step 2. |
| A job PENDING for a long time | `squeue -u "$USER" -o "%i %j %T %r"` shows the reason. `Resources` or `Priority` just means waiting. `AssocGrpGRES` means the request doesn't fit what you may use: check partition, account and GPU type against step 00. `ReqNodeNotAvail` usually means the nodes are down or reserved. |
| The setup job FAILED | The end of its log says why: a module that won't load, or a download that failed. `SETUP_MODULES=""` in `local.env` loads no module, but the `explorer` module is what gives compute nodes their proxy to the internet: without it, downloads fail. |
| sacct says TIMEOUT | Slurm's time limit ran out mid-step. The kit's own limit per launch (`STEP_TIMEOUT`, 20 minutes) is longer than step 01's 10-minute Slurm limit, and `probe` has none. That step then has no line in `steps.jsonl`: paste the end of the job log (step 8). |
| `FAILED: probe` or another step | triage shows the error lines. The full log is `results/explorer/<job>/<step>.log` (for example `01_probe/probe.log`), with NCCL's logs under `<job>/nccl/<step>/`. Paste the triage output. |
| `shared_gpu_check: check did not fire` | The shared-GPU refusal didn't show, or the run failed some other way (a crash, or its time limit): `shared_gpu_check.log` says which. Paste it either way. |

Don't rerun a failed step before pasting: the first failure's records are the
most useful ones.
