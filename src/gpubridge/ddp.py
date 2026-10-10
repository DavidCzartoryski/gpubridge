"""A DistributedDataParallel communication hook that syncs gradients through gpubridge.

DDP needs a process group that spans every rank, and in a mixed NVIDIA + AMD
job the only one is the Gloo world group ``gpubridge.init()`` creates. On that
group alone, DDP would all_reduce every gradient bucket over Gloo, through CPU
memory, on every rank. With this hook DDP keeps its bucketing and its overlap
of communication with the rest of backward, and hands each bucket to
:func:`gpubridge.all_reduce`. Gradients then take the island collectives and
the bridge, under the active collective policy, like any other all_reduce::

    model = DistributedDataParallel(model)  # default group: gpubridge's Gloo world
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)

Gloo still carries what DDP does outside the hook. At construction, DDP checks
parameter shapes across ranks and broadcasts parameters and buffers from rank
0. With ``broadcast_buffers=True`` (the default) it also broadcasts buffers
before every forward pass. On GPUs these run on Gloo with device tensors
(HARDWARE_VALIDATION.md item 22). With a model that has no buffers, or with
``broadcast_buffers=False``, Gloo stays out of the training loop.

DDP must span every rank, because gpubridge collectives always do. Every
rank's DDP launches its buckets in the same order, so every rank queues the
same async all_reduces in the same order, as gpubridge requires.

If a bucket's all_reduce fails, its future carries the error, ``backward()``
raises, and every later gpubridge collective on that rank refuses to start
(see :mod:`gpubridge.work`).
"""

# No ``from __future__ import annotations`` here: DDP's register_comm_hook
# compares the hook's annotations with dist.GradBucket and
# torch.futures.Future[torch.Tensor], so they must be real objects, not strings.

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge.collectives import all_reduce


def ddp_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """Average one DDP gradient bucket across every rank with gpubridge.

    Register with ``model.register_comm_hook(None, gpubridge.ddp_comm_hook)``;
    ``state`` is unused. The all_reduce is async (``ReduceOp.AVG``), so
    backward carries on while the bucket crosses the islands and the bridge.
    DDP waits on the returned future before the gradients are used. DDP leaves
    the averaging to the hook when one is registered; AVG does it.
    """
    work = all_reduce(bucket.buffer(), op=ReduceOp.AVG, async_op=True)
    assert work is not None
    return work.get_future()
