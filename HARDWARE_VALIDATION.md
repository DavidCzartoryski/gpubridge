# What still needs real-hardware validation

The library itself is tested in CPU simulation mode. Simulation runs the same
orchestration code, but it cannot answer the questions here, which depend on
real CUDA and ROCm builds of PyTorch running side by side. Items 1 and 2 are now
answered for the CPU side by running real builds without GPUs; see
[Mixed build results](#mixed-build-results) below.

**On GPUs:** [docs/GPU_VALIDATION_RUNBOOK.md](docs/GPU_VALIDATION_RUNBOOK.md)
maps every item below to a scripted step: Explorer (Slurm) for NVIDIA, a rented
AMD machine, and a real mixed job between two cloud machines. If Explorer's
`sharing` partition turns out to have AMD GPUs, the mixed job can instead run
inside Explorer as one heterogeneous Slurm job. For a first session there,
[docs/EXPLORER_FIRST_SESSION.md](docs/EXPLORER_FIRST_SESSION.md) has the
commands in order, from a fresh login to the ladder's first rungs.

**GPU results:** from 2026-10-14 on Explorer, in [GPU results](#gpu-results)
below, made by [`scripts/gpu/validation_tables.py`](scripts/gpu/validation_tables.py)
from the kit's results: what each job ran on (hosts, GPUs, driver, torch, CUDA
or ROCm, NCCL or RCCL, gpubridge commit, interface), a verdict per item below
from the steps and stages that test it, and the preflight's network numbers.
Runs in CPU simulation never count towards an item. The tables number hosts
instead of naming them and show no paths. See the runbook's
[The validation tables](docs/GPU_VALIDATION_RUNBOOK.md#the-validation-tables).
The *Status* lines under each item describe the CPU work; the GPU verdicts are
in the tables. What the bridge costs is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Blocking: can the job start at all?

1. **CUDA-PyTorch and ROCm-PyTorch processes rendezvous in one job.** Both
   builds must connect to one c10d store (TCPStore through `torchrun` or
   `tcp://`) and form one Gloo default group. This is expected to work because
   the store and Gloo run on CPU and are compiled into both builds, but nobody
   has checked it. Test it with a launcher that starts one `torchrun` per node,
   each with its own PyTorch build.

   **Status: works on CPU.** CUDA-build and ROCm-build processes rendezvoused
   through torchrun's c10d backend and completed discovery, all_reduce and
   broadcast in every run, with either build hosting the store. Rechecked on
   `main` after discovery started sending each rank's full probe record; see
   [Recheck on current main](#recheck-on-current-main-nested-probe-record).
   Still open: separate hosts on a real network, and the same with GPUs attached.
2. **Version matching between builds.** Use the same PyTorch release on both
   sides, e.g. `2.x.y+cu12x` and `2.x.y+rocm6.x`, and the same Python minor
   version, since discovery pickles objects with `all_gather_object`. Every
   rank needs PyTorch 2.8 or later: older releases can deadlock while tearing
   down Gloo groups
   ([pytorch/pytorch#154976](https://github.com/pytorch/pytorch/pull/154976)
   fixed it in 2.8). Pick both builds from 2.8 on.
   `init()` warns when releases differ but does not block. Things to find out:
   - Which release pairs actually work.
   - Whether a minor-version mismatch breaks the store or Gloo protocol.
   - Whether ROCm wheels lag behind CUDA wheels for the release you want.

   **Status: looser than feared on CPU.** Mixed releases from 2.9.1 to 2.14.1
   worked, in both directions, with Python 3.12.12 on every rank. CUDA and ROCm
   wheels exist for every release from 2.6 to 2.14 (Python 3.12, Linux x86_64),
   with different CUDA/ROCm versions per release. Still open: other Python
   versions, releases outside 2.9.1 to 2.14.1, and the GPU side.
3. **`new_group(backend="nccl")` on ranks outside the group.** Every rank
   creates every island group, so ROCm ranks register the NVIDIA island and the
   other way round. That should only touch the store, without initializing the
   backend. Confirm nothing tries to load NCCL on AMD ranks or RCCL on NVIDIA
   ranks.

## Correctness on GPUs

4. **`"nccl"` on ROCm really runs RCCL.** Check with `NCCL_DEBUG=INFO`; RCCL
   logs its own version.
5. **Device selection on AMD nodes.** Ranks use `cuda:$LOCAL_RANK`. Confirm this
   maps to the intended GPU under `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`
   and the cluster's scheduler.
6. **Stream ordering in the bridge step.** A leader's `tensor.cpu()` must see
   the finished island all_reduce, and the island broadcast must see the
   leader's `copy_` back to the GPU. PyTorch's synchronous collectives should
   guarantee both. Check it on both vendors with large tensors and a busy GPU.
   *Code audit (2026-10-10):* the island reduce is a synchronous `dist.reduce`,
   whose `work.wait()` makes the caller's current stream wait for it; the
   copy (`tensor.cpu()`) then runs on that same stream and blocks the host
   until it finishes. The `copy_` back and the island broadcast follow on the
   same stream. With `async_op=True` the same sequence runs on the worker's
   stream, which first waits on an event recorded on the caller's stream. The
   pipelined policy records its start event after the island reduce and makes
   the stream wait on its last copy back before the island broadcast.
   *Second audit (2026-10-11), three more cases:* (a) The only
   `non_blocking` device-to-host copies are the pipelined policy's, into
   pinned slots; the host waits on each chunk's copy event
   (`event.synchronize()`) before the bridge reads the slot. Correct.
   (b) A pinned slot reused while a copy back out of it is still queued:
   within one collective a slot's next copy in waits on the device for its
   last copy back, and a leader that only receives waits for it on the host
   before the bridge writes the slot. But a collective returns with copies
   back still queued, and the next pipelined collective started with no
   record of them, so it could overwrite a slot first (on the host when it
   only receives, from any stream; on the device from another stream).
   Fixed: the policy keeps each slot's last copy-back event from one
   collective to the next. (c) See item 16: `wait()` was correct, but
   `is_completed()` was not. All of this is unverified on GPUs.
   *Step:* `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`):
   `test_bridge_copies_wait_for_the_callers_stream_and_the_island_reduce`
   writes the input at the end of a queue of matmuls on a side stream, sync
   and async, for every bridge policy (a pipelined bridge step that read a
   slot before its copy event would fail it), and
   `test_pinned_slots_are_not_reused_before_their_copies_back` holds a
   receiving island's copies back behind matmuls and starts the next
   broadcast at once, on the same stream and on another. On CPU,
   `test_the_next_collective_waits_for_a_slots_last_copy_back`
   (`tests/test_pipelined.py`) fails without the fix.
7. **float16 and other dtypes over Gloo between real hosts**, compared bit for
   bit with a single-vendor NCCL/RCCL all_reduce on integer-valued data.

## Operational

8. **Gloo networking.** Leaders on different nodes reach each other on the right
   interface; set `GLOO_SOCKET_IFNAME` on nodes with several NICs.
   *Status:* `scripts/gpu/preflight.py` chooses it by measurement: TCP both
   ways between every routed pair of interfaces of the two leader nodes, raw
   bandwidth each way over one stream, then a Gloo group on the pair whose
   slower direction is fastest. Steps 05 and 07 run it first and give each
   node its own choice, even when the two names differ. The bandwidth is one
   TCP stream driven from Python, a lower bound that may understate fast
   links. Works on CPU on one machine (`tests/test_gpu_kit.py`,
   `tests/test_gpu_jobs.py`, where two different names are faked). Still
   open: Explorer's real interfaces between `multigpu` nodes and between
   `gpu` and `sharing`.
9. **Timeouts.** While leaders work over the bridge, the other ranks wait inside
   an island broadcast. With large tensors the bridge step can be slow enough to
   trip the NCCL/RCCL watchdog. Check whether the `timeout` passed to `init()` is
   enough or whether the watchdog needs its own setting.
10. **`barrier()` on ROCm**, which calls `torch.cuda.synchronize`.

## Not correctness, but worth measuring early

11. Bridge throughput and latency compared with a single-vendor all_reduce, to
    set a baseline before any performance work.
    *Step:* `bench_all_reduce.py`, in Explorer steps 03 to 05 and 07,
    `amd/run_all.sh` and `mixed/node.sh`. For each size (1 KB to 1 GB, x4) it
    runs warmup calls, then times trials with CUDA events on the current
    stream. By default it puts three ops side by side: native NCCL/RCCL over
    every rank (or each island's own, where one group is impossible),
    gpubridge, and flat-gloo, with busbw as nccl-tests computes it and
    gpubridge and flat-gloo as multiples of the baseline. Every row names its
    baseline; each island's own is labeled `island-native (fewer ranks, best
    case)`, since it is not a like-for-like comparison. Where a preflight ran,
    `validation_tables.py` sets the bridge phase against the time the
    preflight's bandwidth predicts (runbook, "Predicted vs measured").
    *Status: runs on CPU in simulation, host-timed; no GPU numbers yet.*

## Added with later features

Each item below came with a feature that is tested on CPU but makes a claim
only real GPUs or NICs can check. Each names the kit step that checks it.

12. **A rank whose `LOCAL_RANK` has no GPU fails `init()` on every rank.**
    `resolve_config` compares `LOCAL_RANK` with `torch.cuda.device_count()` and
    discovery reports a mismatch as that rank's problem; `set_device` now waits
    until discovery succeeds.
    *Status: works on CPU, with a faked device count
    (`test_an_out_of_range_local_rank_fails_init_on_every_rank`).* Still open:
    that the count PyTorch reports honours `CUDA_VISIBLE_DEVICES`,
    `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` and Slurm's GPU binding, so
    the check fires on real nodes. *Step:* `check.py --expect-init-error`,
    launched with one more process than the node has GPUs (Explorer step 01,
    `amd/run_all.sh`).
13. **The island step of reduce-bridge-broadcast is a reduce onto the leader**,
    not an all_reduce: only the leader needs the island sum, and a ring reduce
    moves about half the bytes.
    *Status: works on CPU: every layout still matches `flat-gloo` exactly, and
    a test counts one island reduce and one island broadcast per all_reduce.*
    Still open: the time saved on NCCL and RCCL, by tensor size. Small tensors
    may see none, because both ops are latency-bound there. *Step:*
    `bench_all_reduce.py --ops native,island,gpubridge,flat-gloo --phases` puts the
    `island-reduce` phase next to a native island all_reduce (`island:<vendor>`)
    of the same size (Explorer steps 04, 05 and 07, `amd/run_all.sh`).
14. **Where NCCL/RCCL and Gloo agree on reductions.** gpubridge mixes the island
    backend and Gloo in one reduction, so a result is only well defined where
    both agree. PRODUCT is left out until they are shown to agree in every
    supported dtype, including 8-bit overflow. MAX and MIN with NaN in the data
    follow whatever the backends do.
    *Status: not verifiable on CPU, where islands are Gloo too.* *Step:*
    `check.py` stage `reduction_agreement` (informational) compares a raw
    island-backend all_reduce with a Gloo one: PRODUCT in every dtype, MAX/MIN
    with a NaN on one rank. Read `product_agree` and `nan_max_min` in the rank
    records.
15. **MAX, MIN, AVG, `all_gather_into_tensor` and `reduce_scatter_tensor` on
    NCCL/RCCL islands.** Native-only uses NCCL's own all_gather_into_tensor
    and reduce_scatter_tensor; reduce-bridge-broadcast uses island gather and
    scatter, which torch builds from NCCL send/recv.
    *Status: works on CPU, bit for bit against `flat-gloo` in every layout and
    policy (`tests/test_ops.py`).* Still open: the same on GPUs. *Step:*
    `check.py` stages `reductions`, `all_gather` and `reduce_scatter`, and
    `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`), which compares every policy with `flat-gloo` bit for
    bit for every op, fp32/fp16/bf16 and odd sizes. Bit for bit holds although
    NCCL, RCCL and Gloo add in different orders: the data are small integers,
    so every partial sum is exact in every dtype, and the tests use 1, 2 or 4
    ranks so every average is exact too (`tests/test_gpu.py` explains).
16. **Async collectives keep their stream semantics.** The worker's stream
    waits on an event recorded on the caller's stream; `wait()` makes the
    caller's stream wait on an event recorded after the collective, on the
    worker's stream. `is_completed()` used to turn True as soon as the worker
    thread had queued the last kernels (the island broadcast), before they
    ran; with NCCL it means the GPU work is done. Fixed (2026-10-11): it also
    asks that event (`test_is_completed_waits_for_the_gpu_work_too`).
    *Status: ordering, results and failure handling work on CPU
    (`tests/test_ops.py`), where there are no streams.* Still open: the
    events and `record_stream` on real GPUs. *Step:* `check.py` stage `async`
    queues the all_reduce right behind GPU work and checks the result on the
    caller's stream after `wait()`, with no host sync in between. Also
    `pytest -m gpu tests/test_gpu.py` (Explorer steps 02 and 04, `amd/run_all.sh`), which submits from one non-default stream and waits
    on another, and reads a large result right after `is_completed()` turns
    True, without `wait()`
    (`test_is_completed_means_the_result_is_ready`).
17. **The pipelined bridge is correct on GPUs.** `pipelined-reduce-bridge-broadcast`
    (opt-in) copies chunks through pinned host memory on two side streams,
    ordered by CUDA events, waiting on the host only for the chunk the bridge
    needs next.
    *Status: works on CPU, bit for bit against `flat-gloo`, including with a
    staging layer that runs each copy only when something waits for its event,
    so a missing wait corrupts the result (`tests/test_pipelined.py` shows it
    does), and one that leaves copies back queued when a collective returns,
    as a GPU does, so the next collective must wait for them (item 6 (b)).*
    Still open: the real events, pinned buffers and `record_stream` on
    GPUs. *Step:* `check.py --policy pipelined-reduce-bridge-broadcast` (its
    `busy_gpu` stage queues work ahead of the collective), in Explorer steps
    04, 05 and 07 and `amd/run_all.sh` (`OPT_IN_POLICIES`).
18. **The overlap pays off.** For large tensors the pipelined bridge should
    beat reduce-bridge-broadcast without hurting small ones; the best chunk
    size depends on the machine.
    *Status: not measurable on CPU.* *Step:* `bench_all_reduce.py
    --candidates` (or `--write-thresholds`), rerun with a few
    `GPUBRIDGE_CHUNK_BYTES` values.
19. **Thresholds measured on GPUs drive `auto-tuned`.**
    *Status: on CPU, a file written by `--write-thresholds` loads, and
    `auto-tuned` routes each size to the rule's policy.* Still open: the
    thresholds themselves, per cluster. *Step:* the `bench_candidates` and
    `check_auto_tuned` steps write `thresholds.json` next to the results and
    rerun the check under `auto-tuned` with it.
20. **The sharded bridge is correct on GPUs.** `sharded-bridge` (opt-in) runs
    NCCL/RCCL `reduce_scatter_tensor` and `all_gather_into_tensor` on islands
    of exactly k ranks, k NCCL/RCCL reduces and broadcasts on larger ones, and
    k Gloo bridge groups at once, one per island rank below k.
    *Status: works on CPU, bit for bit against `flat-gloo` for every
    supported dtype and SUM/AVG/MAX/MIN, in 2+2, 3+1, 3+2, 2+4, 3+3,
    interleaved and three-island layouts, with sizes that need padding and
    sizes smaller than k (`tests/test_sharded.py`).* Still open: the native
    collectives on device tensors, and k bridge groups on separate hosts.
    *Step:* `check.py --policy sharded-bridge` in Explorer steps 04, 05 and 07
    and `amd/run_all.sh` (`OPT_IN_POLICIES`).
21. **More bridge links mean more cross-vendor bandwidth.** Each link carries
    1/k of the bytes, and each of the k ranks copies its own segment to the
    host, so with the bridge as the bottleneck, k links should beat one.
    *Status: not shown on CPU.* In CPU simulation (8 ranks on one 10-core
    machine), `sharded-bridge` is 1.3x to 1.8x slower than
    reduce-bridge-broadcast. Every rank shares one CPU and memory, so the
    bridge phase barely changes with 4 links (10.8 ms against 10.9 ms at
    64 MB). The island steps are Gloo there, and on CPU islands the
    reduce_scatter is an all_reduce plus a slice, which makes them slower.
    GPU islands use NCCL/RCCL for those steps.
    *Step:* `bench_candidates` in step 05, where the bridge crosses the
    network, and in step 07: compare the `policy:sharded-bridge` and
    `policy:reduce-bridge-broadcast` columns. Step 05's
    `bench_phases_sharded-bridge` times the bridge phase alone, next to the
    `bench_split_by_node` phases of reduce-bridge-broadcast.
22. **DDP with `gpubridge.ddp_comm_hook` trains correctly on GPUs.** Two
    things only GPUs show. First, DDP's own traffic runs on the Gloo world
    group with device tensors: the shape check and parameter broadcast when it
    wraps the model, on CUDA and ROCm builds. Second, the stream handoff: DDP
    calls the hook on the autograd thread, the bucket's all_reduce starts after
    the work queued on that thread's stream, and DDP reads the result through
    the device-aware future.
    *Status: works on CPU. Gradients match the exact average bit for bit (with
    integer-valued gradients) under the default, sharded, pipelined,
    native-only and flat-gloo policies, with one bucket per parameter. A
    failed bucket makes `backward()` raise on every rank, and later
    collectives refuse to start (`tests/test_ddp.py`).* *Step:*
    `examples/train_synthetic.py`, which exits non-zero if parameters diverge:
    `train_ddp` in Explorer steps 04 (split test) and 07 (mixed NVIDIA +
    AMD), `train_ddp_split` in `amd/run_all.sh`, and the single-island
    scaling runs in step 06.
23. **What `gpubridge.in_flight()` shows on GPUs.** It reflects the host: a
    thread is listed while it is inside a gpubridge call. NCCL and RCCL calls
    return once their work is queued, so on GPUs a non-leader whose island
    collective is still waiting on the device may already have left the
    call, while a leader blocks in the bridge's device-to-host copy or in
    Gloo. Record what each rank shows while one rank is late.
    *Status: works on CPU: a late rank shows nothing; the others show the
    collective's seq, op and current phase (bridge on the waiting leader,
    island-broadcast and island-reduce on the others), with and without
    observers; async collectives show on the worker thread; nested phases and
    failed calls are tracked (`tests/test_observe.py`).* *Step:* the
    `in_flight` stage of `check.py`, in every step that runs it. The late rank
    joins only once every other rank's view has stayed the same for
    `--in-flight-settle` seconds (1 by default), in an order the job's store
    enforces, so a loaded machine changes how long the stage takes, not what
    it records.
24. **No island puts two ranks on one GPU.** NCCL and RCCL refuse two ranks
    of one communicator on the same GPU, and an island is one communicator.
    Split-test mode lets ranks share GPUs, so a layout such as 4 ranks on a
    2-GPU node with `alternate` would fail inside NCCL. `init()` compares the
    ranks of each island by the GPU each one drives (its UUID and PCI
    address: device indices differ between processes under
    `CUDA_VISIBLE_DEVICES` or Slurm's GPU binding) and fails on every rank,
    before creating any group, naming the ranks. Every layout in the runbook
    passes: steps 02 (two single-rank islands share the GPU), 04 (a GPU per
    rank, both layouts) and 05, the AMD script, the GPU-only tests, and
    `half` on a 2-GPU node with up to 4 ranks. The only NCCL/RCCL groups
    gpubridge creates are the island groups; the world group, the bridge and
    the sharded bridge's links use Gloo.
    *Status: works on CPU with faked GPU identities
    (`tests/test_split_test.py`).* A GPU that reports neither a UUID nor a
    PCI address (some ROCm builds) can't be compared: `init()` then warns on
    rank 0, naming the island, and skips that island's check instead of
    failing; tested on CPU with every rank reporting nothing. Still open: that
    PyTorch reports a UUID or PCI address on both vendors. *Step:* Explorer
    step 02, `shared_gpu_check`: 4 ranks on its one GPU with `alternate` must
    fail `init()` on every rank with the duplicate-GPU error, within 60 s;
    `steps.jsonl` records `"expected failure confirmed"` or `"check did not
    fire"`.
25. **Pinned staging for the default bridge pays off, and stays correct.**
    With `GPUBRIDGE_PINNED_STAGING=1`, the leaders of
    `reduce-bridge-broadcast` copy into one pinned host buffer with
    `non_blocking=True`, wait on that copy's event (not the whole device)
    before Gloo reads it, and leave the copy back queued; the next
    collective's copy out waits for it on the device, and a larger tensor
    replaces the buffer only after it. Check that results stay bit for bit
    those of `flat-gloo`, including back-to-back collectives on different
    streams, and whether the `bridge` phase gets shorter than with pageable
    `.cpu()` copies, at which sizes.
    *Status: works on CPU, where the copies are checked with lazy pretend
    streams that run a copy only when something waits for it
    (`tests/test_pinned_staging.py`); the GPU tests are written
    (`tests/test_gpu.py`: every op against `flat-gloo`, streams, completion,
    buffer reuse).* *Step:* `check_pinned` and `bench_pinned` in Explorer
    steps 04, 05 and 07 and `amd/run_all.sh`, right after the pageable
    benchmark in the same allocation (`bench`, or `bench_split_by_node` in
    step 05); compare their gpubridge rows and `bridge` phases.
26. **The UCX bridge (`bridge="ucx"`) is correct on GPUs, between real NVIDIA
    and AMD leaders.** Each side builds the library against its own UCX
    (CUDA-built on NVIDIA, ROCm-built on AMD); the handshake refuses
    mismatched UCX versions. GPU tensors go to UCX directly where its build
    has the memory type, and through pinned host memory where it doesn't; the
    run record says which (`run.bridge_path`). Check results bit for bit
    against `flat-gloo`, and that `direct` is what was used where GPUDirect
    RDMA (or ROCm peer memory, or dma-buf) is available. On the direct path
    the transport flushes the NIC's writes into the GPU after every receive
    (`run.bridge_flush` says how; UCX doesn't, see docs/RDMA_TRANSPORT.md
    section 4): check that received data is fresh with the NIC writing GPU
    memory (`test_data_ucx_writes_into_gpu_memory_is_what_the_next_kernel_reads`
    with `GPUBRIDGE_TEST_UCX_TLS=rc,cuda_copy`, or across nodes), and how long
    a flush takes.
    *Status: works on CPU over UCX's TCP transport, every bridge policy on two,
    three and four leaders (`tests/test_ucx.py`); the flush's placement is
    tested on CPU against a fake worker whose receives land only when flushed
    (`tests/test_gdr_flush.py`); the device path's GPU tests are written
    (`tests/test_gpu.py`). Nothing has run on RDMA hardware. On Explorer, the
    V100 and MI50 nodes checked have no RDMA path between them, and no UCX
    there has a `cuda` or `rocm` transport
    ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#rdma-between-the-node-types)).*
    *Step:* `ucx_build` and `check_ucx` in Explorer steps 05 and 07, with
    `UCX_BRIDGE=1`. See [docs/RDMA_TRANSPORT.md](docs/RDMA_TRANSPORT.md) for
    the questions to settle first (InfiniBand on the GPU nodes, GPUDirect
    RDMA, a UCX module).
27. **The UCX bridge against the Gloo bridge.** The bridge phase over UCX
    (RDMA where available) next to the Gloo bridge's, at each size, and against
    the preflight's TCP prediction, which RDMA should beat.
    *Status: not run.* *Step:* `bench_ucx` in steps 05 and 07 with
    `UCX_BRIDGE=1`; compare with the Gloo benchmark of the same job.
28. **Bridge compression's Triton kernels give the reference's bits on NVIDIA
    and AMD GPUs.** `GPUBRIDGE_BRIDGE_COMPRESSION=bf16|fp8` (opt-in) promises
    that every rank still ends with the same bits, which needs the compiled
    kernels on both vendors to match the plain-PyTorch reference exactly
    (integer arithmetic on bit patterns, power-of-two scales, explicit
    flushing below 2^-100; docs/COMPRESSION.md). Check that the GPU tests pass,
    that `bench_compression.py` reports "same bits as CPU" for every mode and
    dtype, that its payload and result digests are identical on the NVIDIA
    and the AMD node, and that `bench_compressed` reports every rank the same.
    Also how close each kernel gets to the GPU's memory bandwidth.
    *Status: works on CPU, where Triton's interpreter runs the kernels and
    they match the reference bit for bit (`tests/test_compression_kernels.py`);
    in simulated jobs one leader on Triton and the other on PyTorch agree
    (`tests/test_compression.py`). The GPU tests are written
    (`test_compression_kernels.py`, marked `gpu`).* *Step:* `gpu_tests` and
    `bench_compression_kernels` in Explorer steps 04 and 07, `amd/run_all.sh`
    and `mixed/node.sh`; in step 07 compare `bench_compression_gpu` with
    `bench_compression_amd`.
29. **Where bridge compression pays off.** Compression adds two kernels and
    saves bridge bytes (2x for `bf16`, 3.9x for `fp8` on float32). From which
    size, if any, is a compressed all_reduce faster, per policy, and how much
    shorter does the `bridge` phase get? The `compress` and `decompress`
    phases show the kernels' share. And the cost in accuracy, from the error
    `bench_compressed` reports, and from loss curves with and without it.
    *Status: not run; on CPU the PyTorch reference kernels are slower than the
    bridge they save.* *Step:* `bench_compressed` (`--compression
    none,bf16,fp8 --phases`) in Explorer steps 04 and 07, `amd/run_all.sh`
    and `mixed/node.sh`; its summary.md says from which size each mode is
    faster.
30. **DDP and FSDP2 through `backend="gpubridge"` on GPUs.** The
    torch.distributed backend (docs/TORCH_BACKEND.md) carries a script's
    collectives as gpubridge async collectives. Check that
    `examples/fsdp2_torch_backend.py --verify` passes across a split test and
    across real NVIDIA + AMD ranks: on exact data, its parameters must equal a
    single-process replay bit for bit, which a collective reading or writing a
    tensor at the wrong moment would break even when every rank agrees. Also
    that its plain training run ends with every rank's parameters identical,
    and that the GPU tests pass (DDP on two GPUs split into two islands, with a
    CPU tensor carried by Gloo; `init_process_group(device_id=...)`). Step time
    against FSDP2 on one vendor's NCCL/RCCL isn't measured by the kit yet.
    *Status: works on CPU in simulation mode on torch 2.8.0 and 2.14.1: DDP and
    FSDP2 bit for bit like Gloo on integer-valued data, every collective the
    backend carries, every refusal (`tests/test_torch_backend.py`), and the
    example's `--verify`; the GPU tests are written.* *Step:*
    `verify_fsdp2_backend` and `train_fsdp2_backend` in Explorer steps 04 and
    07, `amd/run_all.sh` and `mixed/node.sh`; `gpu_tests`.
31. **HSDP through the backend: shard within islands, replicate across
    vendors.** Subgroups give each vendor island a native-only shard group
    and each replicate group a bridged topology (docs/TORCH_BACKEND.md,
    "Subgroups and HSDP"). Check that `examples/fsdp2_torch_backend.py
    --verify --hsdp 2` passes on the split test and on real NVIDIA + AMD
    ranks, bit for bit against a single-process replay, and that the plain
    `--hsdp 2` run ends with every rank's parameters identical. Its JSON has
    the median step time; set it against `train_fsdp2_backend`'s from the same
    job (plain FSDP2 over every rank, whose all-gathers and reduce-scatters
    cross the bridge). On the mixed job this needs as many NVIDIA GPUs as AMD;
    with one of each, the shard groups are single ranks and only the replicate
    all_reduce is exercised. *Status: works on CPU in simulation mode on torch
    2.14.1 and in CI on torch 2.8.0 and 2.14.1: bit for bit like one process,
    every all-gather and reduce-scatter native-only with no bridge phase
    (`tests/test_torch_backend.py`), and the example's `--hsdp 2` and
    `--verify --hsdp 2` on 4 simulated ranks.* *Step:* `train_hsdp_backend`
    and `verify_hsdp_backend` in Explorer steps 04 and 07.
33. **Data-parallel training across vendors tracks one GPU, compressed or
    not.** `train_dp.py` on `llama.py`'s model, 500 steps, the same global
    batch and seed as one V100 alone: DDP through `gpubridge.ddp_comm_hook`
    uncompressed, with bf16 and with fp8 (error feedback), FSDP2 through the
    backend, and DDP with batch shares by GPU speed. Uncompressed and
    speed-weighted runs should track the single GPU's loss curve closely (the
    same mean gradient each step, reduced in another order on other hardware);
    compressed ones within a tolerance to be justified from the measured gaps.
    Check that the ranks' parameters stay bit-identical (every 50 steps), and
    record tokens/s, the step split and the share of communication backward
    hides. *Status: works on CPU in simulation mode: DDP, uneven shares and
    FSDP2 within 1e-4 of one process over 12 steps, fp8 within 0.05
    (`tests/test_gpu_kit.py`). On a V100 and an MI50 (Explorer, 1,000 steps):
    parameters bit-identical at every check, compressed or not; over the last
    100 steps uncompressed DDP is 0.0037 to 0.0067 from four one-V100 runs,
    which are 0.0041 to 0.0056 apart, and bf16 and fp8 0.022 to 0.031, about
    what an uncompressed run leaves after a late drop off the loss plateau
    (docs/COMPRESSION.md, "Training with compression"); FSDP2 ran 500 steps,
    shares by speed and bucket sizes 100 steps each
    (docs/BENCHMARKS.md). 500 steps, as
    planned here, end mid-drop for some runs, so the curves were compared at
    1,000.* *Step:* `dp_*` in Explorer step 07 with `STEPS_07=dp`, and
    `dp_single` in `submit.sh dp-baseline`.
35. **The multitcp bridge against the Gloo bridge.** `bridge="multitcp"`
    (docs/RUST_TRANSPORT.md) carries the bridge over several TCP connections
    per pair and reduces each chunk as it arrives. Between two machines, its
    bridge phase at each size against the Gloo bridge's (default and pinned
    staging) in the same job, and against the preflight's prediction; one
    connection against several, for the connections' share. It takes host
    tensors only, so the GPU side is the policies' usual staging. Stage 1 stays
    only if, in each of two jobs, the median time of the whole float32
    all_reduce with pinned staging is at most 0.8 times the pinned Gloo
    bridge's at every size from 64 MB to 1 GB.
    *Status: works on CPU over loopback TCP, every bridge policy on two, three
    and four leaders, bit for bit against `flat-gloo` (`tests/test_multitcp.py`).
    Not run between two machines.* *Step:* `multitcp_build` and the
    `matrix_D-multitcp*` runs in Explorer step 08, with `MULTITCP_BRIDGE=1`.

32. **Point-to-point across islands through the backend on GPUs.** `send`,
    `recv`, `isend`, `irecv` and `batch_isend_irecv` on `backend="gpubridge"`
    go native within an island and through the bridge across islands. On GPUs,
    check that six dtypes and odd sizes arrive bit for bit, that a send whose
    input was copied device to device on a busy producer stream and dropped before
    it completes still sends what was written despite a same-stream allocation
    filled with other data, that the received tensor read on another
    stream right after `wait()` holds the data, and that two tags between a
    pair match out of order. *Status: works on CPU in simulation mode (1+1 and
    2+2 ranks, `tests/test_gpu.py`'s simulation test and check.py `--p2p-only`);
    not run on GPUs.* *Step:* `check_p2p` in Explorer steps 02 and 07, and
    `test_point_to_point_across_islands_keeps_stream_order_and_storage` in
    `gpu_tests`. The results table counts `check_p2p` only; historical `gpu_tests`
    records cannot show whether this test existed in the run.

## GPU results

From `scripts/gpu/validation_tables.py` over every Explorer result of
2026-10-14 (the kit's `results/explorer`, with tonight's rocm6.3 jobs in
`results/explorer/rocm63`). Hosts are numbered, not named. A folder's
`steps.jsonl` keeps every attempt run into it, so a job below can count a step
twice, and a FAIL can be an early attempt that a later one in the same folder
passed; the notes say which. GPU names are as the driver reports them: the
MI50 shows as "AMD Instinct MI60 / MI50", or "AMD Radeon Graphics" with
torch 2.9.1+rocm6.3.

### Runs

| Job | Steps passed | Failed | Hosts | GPUs | Driver | torch | CUDA / ROCm | NCCL / RCCL | gpubridge | Gloo interface |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01_probe | 5 of 6 | `probe` | host 1 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 14ed60c | auto |
| 02_a100_df80abb | 19 of 20 | `gpu_tests` | host 2 | NVIDIA A100-SXM4-80GB | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_l40s_df80abb | 24 of 25 | `gpu_tests` | host 3 | NVIDIA L40S | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_split_shared_gpu | 29 of 30 | `gpu_tests` | host 4 | Tesla T4 | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_t4_df80abb | 29 of 30 | `gpu_tests` | host 4 | Tesla T4 | 570.86.15 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb+dirty, df80abb | auto |
| 02_v100_df80abb | 14 of 15 | `gpu_tests` | host 5 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | df80abb | auto |
| 07_mixed_hetjob | 15 of 20 | `train_ddp`, `train_fsdp2_backend`, `verify_fsdp2_backend`, `train_hsdp_backend`, `verify_hsdp_backend` | host 6, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 4014234 | internal.2137, auto |
| 07_mixed_hetjob_4014234_job11084294 | 10 of 16 | `bench`, `check_pinned`, `summarize_pinned`, `bench_pinned`, `bench_compression_kernels`, `train_ddp` | host 8, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 4014234 | internal.2137, internal, auto |
| 08_matrix_nvidia-amd | 15 of 15 | - | host 9, host 7 | Tesla V100-SXM2-32GB, AMD Instinct MI60 / MI50 | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.14.1+rocm7.2 | CUDA 12.6, ROCm ? (HIP 7.2.53211) | NCCL 2.29.3, RCCL 2.27.7 | 934f87c | internal.2137, auto |
| amd_node | 9 of 9 | - | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 4014234 | auto |
| amd_node_4014234_job11083835 | 14 of 18 | `check_1gpu`, `gpu_tests`, `check_split_shared_gpu`, `summarize` | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 4014234 | auto |
| net_check_amd | 4 of 4 | - | host 7 | AMD Instinct MI60 / MI50 | 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+rocm7.2 | ROCm ? (HIP 7.2.53211) | RCCL 2.27.7 | 934f87c | auto |
| net_check_nvidia | 4 of 4 | - | host 10 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 934f87c | auto |
| rocm63/07_mixed_hetjob | 7 of 8 | `train_hsdp_backend` | host 9, host 7 | Tesla V100-SXM2-32GB, AMD Radeon Graphics | 545.23.08, 5.14.0-611.5.1.el9_7.x86_64 | 2.14.1+cu126, 2.9.1+rocm6.3 | CUDA 12.6, ROCm ? (HIP 6.3.42134-a9a80e791) | NCCL 2.29.3, RCCL 2.21.5 | 934f87c | internal.2137, auto |
| rocm63/amd_node | 7 of 9 | `gpu_tests`, `bench_compression_kernels` | host 7 | AMD Radeon Graphics | 5.14.0-611.5.1.el9_7.x86_64 | 2.9.1+rocm6.3 | ROCm ? (HIP 6.3.42134-a9a80e791) | RCCL 2.21.5 | dbed5c9 | auto |
| rocm63/train_bench_amd | 3 of 3 | - | host 7 | AMD Radeon Graphics | 5.14.0-611.5.1.el9_7.x86_64 | 2.9.1+rocm6.3 | ROCm ? (HIP 6.3.42134-a9a80e791) | RCCL 2.21.5 | 934f87c | auto |
| train_bench_nvidia | 3 of 3 | - | host 9 | Tesla V100-SXM2-32GB | 545.23.08 | 2.14.1+cu126 | CUDA 12.6 | NCCL 2.29.3 | 934f87c | auto |

### Items

`pass (n)` counts GPU runs only; one failed record makes an item `FAIL (k of
n)`. The `gpu_tests` step tests items 6, 15, 16, 17, 28 and 30 together, so its
failures count against all six whichever test failed.

| Item | Tested by | Verdict |
| --- | --- | --- |
| 1. CUDA and ROCm processes rendezvous in one job | check.py `init` (gpu-mixed) | pass (7) |
| 2. Version matching between builds | check.py `init` (gpu-mixed) | pass (7) |
| 3. `new_group(backend="nccl")` on ranks outside the group | check.py `init` (gpu-mixed, split-test) | pass (15) |
| 4. `"nccl"` on ROCm really runs RCCL | step `rccl_version` | pass (4) |
| 5. Device selection | step `device_map` | pass (4) |
| 6. Stream ordering in the bridge step | check.py `busy_gpu` (gpu-mixed, split-test); step `gpu_tests` | FAIL (7 of 43) |
| 7. float16 and other dtypes over Gloo | check.py `dtypes` (gpu-mixed, split-test) | pass (15) |
| 8. Gloo networking | step `preflight`; step `check_split_by_node` | pass (4) |
| 9. Timeouts | check.py `busy_gpu` (gpu-mixed, split-test) | pass (15) |
| 10. `barrier()` on GPUs | check.py `barrier` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 11. Bridge throughput and latency against native | step `bench`; step `bench_split_by_node` | FAIL (1 of 2) |
| 12. A rank whose `LOCAL_RANK` has no GPU fails `init()` | step `local_rank_check` | pass (6) |
| 13. The island step is a reduce onto the leader | step `bench`; step `bench_split_by_node` | FAIL (1 of 2) |
| 14. Where NCCL/RCCL and Gloo agree on reductions | check.py `reduction_agreement` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 15. MAX, MIN, AVG, all_gather and reduce_scatter on GPUs | check.py `reductions` (gpu-mixed, gpu-single-vendor, split-test); check.py `all_gather` (gpu-mixed, gpu-single-vendor, split-test); check.py `reduce_scatter` (gpu-mixed, gpu-single-vendor, split-test); step `gpu_tests` | FAIL (7 of 82) |
| 16. Async collectives keep their stream semantics | check.py `async` (gpu-mixed, gpu-single-vendor, split-test); step `gpu_tests` | FAIL (7 of 46) |
| 17. The pipelined bridge is correct on GPUs | step `check_pipelined-reduce-bridge-broadcast`; step `gpu_tests` | FAIL (7 of 30) |
| 18. The overlap pays off | step `bench_phases_pipelined-reduce-bridge-broadcast`; step `bench_candidates` | pass (2) |
| 19. Thresholds measured on GPUs drive `auto-tuned` | step `bench_candidates`; step `check_auto_tuned` | pass (2) |
| 20. The sharded bridge is correct on GPUs | step `check_sharded-bridge` | pass (2) |
| 21. More bridge links mean more cross-vendor bandwidth | step `bench_phases_sharded-bridge`; step `bench_candidates` | pass (2) |
| 22. DDP with `gpubridge.ddp_comm_hook` trains correctly | step `train_ddp(_split)?`; step `train_n\d+` | FAIL (2 of 3) |
| 23. What `gpubridge.in_flight()` shows on GPUs | check.py `in_flight` (gpu-mixed, gpu-single-vendor, split-test) | pass (18) |
| 24. No island puts two ranks on one GPU | step `shared_gpu_check` | pass (24) |
| 25. Pinned staging for the default bridge | step `check_pinned`; step `bench_pinned` | FAIL (2 of 4) |
| 26. The UCX bridge is correct on GPUs | step `ucx_build`; step `check_ucx` | not run |
| 27. The UCX bridge against the Gloo bridge | step `bench_ucx` | not run |
| 28. Bridge compression kernels give the reference's bits on GPUs | step `gpu_tests`; step `bench_compression_kernels` | FAIL (9 of 34) |
| 29. Where bridge compression pays off | step `bench_compressed` | pass (2) |
| 30. gpubridge as a torch.distributed backend: DDP and FSDP2 on GPUs | step `verify_fsdp2_backend`; step `train_fsdp2_backend`; step `gpu_tests` | FAIL (9 of 32) |
| 31. HSDP through the backend: shard within islands, replicate across vendors | step `verify_hsdp_backend`; step `train_hsdp_backend` | FAIL (3 of 3) |
| 33. Data-parallel training across vendors tracks one GPU, compressed or not | step `dp_(ddp\|fsdp2)(-.+)?`; step `dp_single` | not run |
| 35. The multitcp bridge against the Gloo bridge | step `multitcp_build`; step `matrix_D-multitcp(-pinned\|-1stream)?` | not run |

Where the FAILs come from:

- **`gpu_tests`, 7 records (items 6, 15, 16, 17, 28, 30).** Five are one
  morning attempt on a V100 at af51b20, recorded in each step 02 folder: two
  fp8 error-feedback residual tests failed, fixed in #50; every later attempt in
  those folders passed. One is the first MI50 attempt at df80abb: 2 tests
  failed in a matmul the tests used as busy work, which the wheel's rocBLAS
  can't run on gfx906, replaced in #52; the folder's later attempt passed. The
  last is the MI50 with torch 2.9.1+rocm6.3: its Triton (pytorch-triton-rocm
  3.5.1) can't compile the compression kernels for gfx906, so the 20
  compression-kernel tests failed; the other 6 passed.
- **`bench` (items 11 and 13), `check_pinned` and `bench_pinned` (item 25),
  one of the two `bench_compression_kernels` FAILs (item 28)**: step 07's first
  run, where Slurm couldn't create the job steps; all passed in the second run.
  The other `bench_compression_kernels` FAIL is the rocm6.3 Triton above.
- **`train_ddp` (item 22)**: failed in step 07's first two runs, on the default
  ROCm wheel (a Gloo timeout in the first, cause unknown; the MI50 aborting in
  rocBLAS in the second); passed with the MI50 on torch 2.9.1+rocm6.3.
- **`train_fsdp2_backend`, `verify_fsdp2_backend` (item 30)**: failed in step
  07's second run (the same rocBLAS abort); passed with torch 2.9.1+rocm6.3,
  bit for bit against a single-process replay.
- **`train_hsdp_backend`, `verify_hsdp_backend` (item 31)**: failed in step
  07's second run (the same rocBLAS abort). With the MI50 on torch
  2.9.1+rocm6.3, `train_hsdp_backend` printed nothing until the kit's step
  timeout stopped it (exit 124) or the job was cancelled, in this job and
  the later mixed jobs recorded here.
  **Where it stops**, from a CPU reproduction (gpubridge's simulation mode,
  two ranks on one node, `examples/fsdp2_torch_backend.py --hsdp 2`): with
  both ranks on torch 2.14.1, or both on 2.9.1, it completes; with rank 0 on
  2.14.1 and rank 1 on 2.9.1 it hangs the same way. Python stacks taken at
  the hang (faulthandler) show both ranks in gpubridge's one-time world setup,
  inside `dist.new_group` (`torch_backend._world_topology` -> `setup`),
  reached from FSDP's first gradient collective: an `all_reduce` on rank 0
  and a `reduce_scatter` on rank 1. The two torch releases' `DeviceMesh`
  creates a different number of groups for the (2, 1) mesh (2.14.1 reuses the
  default group for the replicate dimension; 2.9.1 calls `new_group`), and
  the world setup's groups take torch's default names, which count the groups
  each process has made, so the ranks likely wait on different store keys.
  A setup-naming fix in `src/` completed this mixed-version CPU example and
  its bit-for-bit replay check. It has not yet been validated on the GPU
  pair, whose original hang had no captured stacks. Plain `init_device_mesh`
  and a barrier with the same mixed versions, without gpubridge, don't hang.

### Network

| Job | Master (host, interface) | Worker (host, interface) | MB/s master to worker | MB/s worker to master | Round trip (ms) |
| --- | --- | --- | --- | --- | --- |
| 07_mixed_hetjob | host 6 internal.2137 | host 7 internal.2137 | 1173.4 | 1147.3 | 0.069 |
| 07_mixed_hetjob_4014234_job11084294 | host 8 internal.2137 | host 7 internal | 1171.4 | 1173.8 | 0.047 |
| 08_matrix_nvidia-amd | host 9 internal.2137 | host 7 internal.2137 | 1173.7 | 1172.1 | 0.048 |
| rocm63/07_mixed_hetjob | host 9 internal.2137 | host 7 internal.2137 | 1173.8 | 1169.7 | 0.046 |

The benchmark tables (gpubridge against each run's baseline, and the predicted
against the measured bridge time) are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) with the full matrix;
`validation_tables.py` writes them too.

What has not run on GPUs at all: the UCX bridge (items 26 and 27), more than
one GPU in an island of a mixed job (Explorer's `multigpu` partition isn't
open to this account), and more than two islands.

## Mixed build results

Run on 2026-10-06 with [`scripts/mixed_build_rendezvous.sh`](scripts/mixed_build_rendezvous.sh),
which reproduces everything below with one command. Raw results:
[`results/mixed_build/summary.md`](results/mixed_build/summary.md) (one row per run),
[`summary.json`](results/mixed_build/summary.json) (per-rank detail) and
[`envs.json`](results/mixed_build/envs.json) (builds and sizes).

### Setup

- No GPUs. A linux/amd64 Docker image (emulated on an Apple Silicon Mac) holds
  one venv per PyTorch build, installed from the official wheel indexes. Every
  venv uses Python 3.12.12.
- `GPUBRIDGE_CPU_ONLY=1` on every rank: the vendor comes from the real build
  (`torch.version.cuda` or `torch.version.hip`), islands use Gloo, tensors stay
  on CPU. `GPUBRIDGE_VENDOR` is not set.
- Each run starts two `torchrun` "nodes" on localhost, one per venv, with
  `--nnodes=2` and a c10d rendezvous. `--rdzv-conf is_host=...` chooses which
  node hosts the rendezvous store; that node also ends up as node 0, which
  hosts the store the workers use.
- Each rank runs [`scripts/rendezvous_probe.py`](scripts/rendezvous_probe.py)
  and checks, in order: rendezvous, `gpubridge.init()` (Gloo world group,
  `all_gather_object` discovery, island and bridge groups), that every rank got
  the same vendor map and that it matches each rank's own build, `all_reduce`
  SUM (exact, float32 and float16), `broadcast` from every rank, and a clean
  `destroy()` and exit.
- Negative control: with the bridge `all_reduce` removed from `collectives.py`,
  the same harness reports `FAIL (all_reduce)` with broadcast still passing,
  so the checks do catch wrong results.

| venv | torch | CUDA / HIP | Index | Download | Installed |
| --- | --- | --- | --- | --- | --- |
| cu-2.14.1 | 2.14.1+cu130 | 13.0 | `whl/cu130` | 3.0 GB | 5.6 GB |
| rocm-2.14.1 | 2.14.1+rocm7.2 | 7.2.53211 | `whl/rocm7.2` | 6.6 GB | 15.7 GB |
| cu-2.13.0 | 2.13.0+cu130 | 13.0 | `whl/cu130` | 2.7 GB | 4.8 GB |
| cu-2.9.1 | 2.9.1+cu128 | 12.8 | `whl/cu128` | 4.1 GB | 7.0 GB |
| rocm-2.9.1 | 2.9.1+rocm6.4 | 6.4.43484 | `whl/rocm6.4` | 4.8 GB | 15.4 GB |

ROCm builds are the large ones: the 2.14.1 ROCm torch wheel alone is 6.2 GB.

### Compatibility

Layouts are CUDA ranks + ROCm ranks; the build in brackets hosted the store.

| CUDA build | ROCm build | Runs | Result | Notes |
| --- | --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release |
| 2.9.1+cu128 | 2.9.1+rocm6.4 | 1+1 (CUDA), 2+2 (ROCm), 3+1 (CUDA) | **PASS** 3/3 | Matched release, older CUDA and ROCm generation |
| 2.13.0+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA), 1+1 (ROCm), 2+2 (CUDA) | **PASS** 3/3 | One minor release apart; `init()` warns |
| 2.14.1+cu130 | 2.9.1+rocm6.4 | 1+1 (CUDA 2.14.1) | **PASS** | Five minor releases apart, newer side hosts; warns |
| 2.9.1+cu128 | 2.14.1+rocm7.2 | 1+1 (CUDA 2.9.1) | **PASS** | Five minor releases apart, older side hosts; warns |
| 2.13.0+cu130 | (2.14.1+cu130, CUDA) | 1+1 | **PASS** | Control: one vendor, no bridge; warns |

Every run passed on its first attempt, with no timeouts and so no reruns.
Each took 8 to 9 seconds end to end, emulation included.

### Recheck on current main (nested probe record)

Run on 2026-10-06 against `main` at `3a815c0` (after #3). Discovery there sends
each rank's full `probe()` record nested inside its `PeerInfo`, which the
earlier runs above did not. Same setup, 2.14.1 pair only, with the wheels
installed offline from the host cache:

```
GPUBRIDGE_ENVS=2.14.1 scripts/mixed_build_rendezvous.sh --group 2.14.1 \
    --results /gpubridge/results/mixed_build_recheck
```

Raw results:
[`results/mixed_build_recheck/summary.md`](results/mixed_build_recheck/summary.md)
and [`summary.json`](results/mixed_build_recheck/summary.json).

| CUDA build | ROCm build | Layout (store host) | Result |
| --- | --- | --- | --- |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 1+1 (CUDA) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 2+2 (ROCm) | **PASS** |
| 2.14.1+cu130 | 2.14.1+rocm7.2 | 3+1 (CUDA) | **PASS** |

Every check passed (rendezvous, init, vendor map, all_reduce, broadcast,
shutdown), each run on its first attempt, in 8 seconds.

**The nested record crosses builds intact.** Every rank received every peer's
probe record, and all ranks held identical tables. A CUDA rank saw the ROCm
rank as:

```
build_vendor="amd", torch_version="2.14.1+rocm7.2", cuda_version=None,
hip_version="7.2.53211", gpu_count=0, nccl_available=True,
gloo_available=True, errors=()
```

The ROCm ranks saw the mirror image (`build_vendor="nvidia"`,
`cuda_version="13.0"`). No rank reported problems.

`nccl_available` is True on both builds even without GPUs, because it reports
whether the backend is compiled into the build. `gpu_count` is what shows the
missing hardware.

### What this means for the design

- **The bridge design stands.** The parts of a mixed job that both builds must
  share all run on CPU: the c10d store, Gloo, and pickled builtins in
  `all_gather_object`. All of them interoperate between CUDA and ROCm builds,
  and across releases from 2.9.1 to 2.14.1. No workaround was needed.
- **Version matching is a recommendation, not a requirement, on the CPU side.**
  `init()` keeps warning when releases differ, because the GPU side and other
  Python versions are untested.

### What this does not show

- Anything on GPUs: NCCL, RCCL, device selection, stream ordering (items 4 to 7, 10).
- Separate hosts on a real network. Gloo only ran over loopback (item 8).
- Item 3 (`new_group(backend="nccl")` on ranks outside the group). In CPU-only
  mode islands use Gloo, and on a machine without GPUs a member rank cannot
  create an NCCL group at all: it raises `ValueError: ProcessGroupNCCL is only
  supported with GPUs, no GPUs found!` (checked on 2.14.1+cu130).
- Python versions other than 3.12, releases outside 2.9.1 to 2.14.1, and
  non-x86 hosts.

### Setup problems worth knowing about

- Installing the 6.2 GB ROCm wheel straight from the index with uv failed once
  with "Failed to read from zip file". Downloading it with `curl` and checking
  the index's sha256 worked, and uv then installed the file without trouble.
  Whether the first download was cut off or uv mishandled the large file was
  not pinned down. ROCm wheels are now always fetched that way
  ([`scripts/fetch_wheel.sh`](scripts/fetch_wheel.sh)).
- The wheels first lived in a BuildKit cache mount, but Docker Desktop's
  BuildKit garbage collection caps cache mounts at about 2.76 GiB, so the 6+ GB
  ROCm wheels were evicted. They now live in a permanent cache on the host
  (`~/.cache/gpubridge/wheels`, filled by
  [`scripts/fetch_wheels.sh`](scripts/fetch_wheels.sh)), and the image
  installs offline from it.
- ROCm builds of torch need `libatomic.so.1`, which slim Debian images lack
  (package `libatomic1`).
- Recent macOS ships a BSD `sha256sum` that has no `--status` flag. An early
  version of `fetch_wheel.sh` used it, mistook the usage error for a checksum
  mismatch, and deleted a complete 6.2 GB wheel. The script now compares
  digests as text, preferring `shasum`.
- `download-r2.pytorch.org` answers 403 to Python's default `urllib`
  User-Agent; set one when scripting against it.
