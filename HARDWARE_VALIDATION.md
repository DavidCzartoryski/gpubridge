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
inside Explorer as one heterogeneous Slurm job.

**GPU results:** none yet. After GPU runs,
[`scripts/gpu/validation_tables.py`](scripts/gpu/validation_tables.py) turns
the kit's results into the tables for this file: what each job ran on (hosts,
GPUs, driver, torch, CUDA or ROCm, NCCL or RCCL, gpubridge commit, interface),
a verdict per item below from the steps and stages that test it, the
preflight's network numbers, benchmark ratios against their named baseline,
and the predicted against the measured bridge time. Runs in CPU simulation
never count towards an item. The tables number hosts instead of naming them
and show no paths, so they can go here as they are. See the runbook's
[The validation tables](docs/GPU_VALIDATION_RUNBOOK.md#the-validation-tables).

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
    (`tests/test_gpu.py`). Nothing has run on RDMA hardware.*
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
    torch.distributed backend (docs/TORCH_BACKEND.md) carries an unmodified
    script's collectives as gpubridge async collectives. Check that
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
