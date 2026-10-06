# What still needs real-hardware validation

Everything below was tested in CPU simulation mode only. Simulation runs the
same orchestration code, but it cannot answer the questions here, which depend
on real CUDA and ROCm builds of PyTorch running side by side.

## Blocking: can the job start at all?

1. **CUDA-PyTorch and ROCm-PyTorch processes rendezvous in one job.** Both
   builds must connect to one c10d store (TCPStore through `torchrun` or
   `tcp://`) and form one Gloo default group. This is expected to work because
   the store and Gloo run on CPU and are compiled into both builds, but nobody
   has checked it. Test it with a launcher that starts one `torchrun` per node,
   each with its own PyTorch build.
2. **Version matching between builds.** Use the same PyTorch release on both
   sides, e.g. `2.x.y+cu12x` and `2.x.y+rocm6.x`, and the same Python minor
   version, since discovery pickles objects with `all_gather_object`.
   `init()` warns when releases differ but does not block. Things to find out:
   - Which release pairs actually work.
   - Whether a minor-version mismatch breaks the store or Gloo protocol.
   - Whether ROCm wheels lag behind CUDA wheels for the release you want.
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
7. **float16 and other dtypes over Gloo between real hosts**, compared bit for
   bit with a single-vendor NCCL/RCCL all_reduce on integer-valued data.

## Operational

8. **Gloo networking.** Leaders on different nodes reach each other on the right
   interface; set `GLOO_SOCKET_IFNAME` on nodes with several NICs.
9. **Timeouts.** While leaders work over the bridge, the other ranks wait inside
   an island broadcast. With large tensors the bridge step can be slow enough to
   trip the NCCL/RCCL watchdog. Check whether the `timeout` passed to `init()` is
   enough or whether the watchdog needs its own setting.
10. **`barrier()` on ROCm**, which calls `torch.cuda.synchronize`.

## Not correctness, but worth measuring early

11. Bridge throughput and latency compared with a single-vendor all_reduce, to
    set a baseline before any performance work.
