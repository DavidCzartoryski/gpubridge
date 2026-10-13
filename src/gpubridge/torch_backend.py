"""gpubridge as a ``torch.distributed`` backend: ``dist.init_process_group(backend="gpubridge")``.

Unmodified DDP and FSDP2 scripts then run their collectives through gpubridge,
across vendor islands. Design and limits: docs/TORCH_BACKEND.md.

- ``import gpubridge`` registers the name (and so does ``import torch``, through
  gpubridge's ``torch.backends`` entry point; see :mod:`gpubridge_backend`).
- The process group is the world group. gpubridge's own groups (a Gloo group of
  every rank, the islands, the bridge) are created at its first collective,
  when ``dist.new_group`` can run; discovery then checks every rank as
  ``gpubridge.init()`` does. ``GPUBRIDGE_POLICY`` and ``GPUBRIDGE_BRIDGE`` pick the
  policy and bridge transport (``auto`` and ``gloo`` by default).
- Each collective runs as gpubridge's async collective, so stream semantics are
  those of ``async_op=True`` (:mod:`gpubridge.work`). On GPUs, CPU tensors go to
  the Gloo group directly.
- Unsupported calls (other reductions, point-to-point, all-to-all, gather,
  scatter, reduce, subgroups) fail before any communication, saying what isn't
  supported.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future

from gpubridge import collectives as _collectives
from gpubridge import observe as _observe
from gpubridge import topology as _topology
from gpubridge import work as _work
from gpubridge.config import BRIDGE_ENV, CPU_BACKEND, POLICY_ENV
from gpubridge.policies import DEFAULT_POLICY, resolve_policy
from gpubridge.transport import DEFAULT_TRANSPORT, resolve_transport

NAME = "gpubridge"

_setup_lock = threading.Lock()
_owner: GpubridgeProcessGroup | None = None
"""The process group that built gpubridge's topology; it releases it at shutdown."""


def _completed(result: Any) -> dist.Work:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _unsupported(what: str, why: str = "") -> NotImplementedError:
    return NotImplementedError(
        f"the {NAME} torch.distributed backend doesn't support {what}"
        + (f": {why}" if why else "") + " (docs/TORCH_BACKEND.md)")


class GpubridgeProcessGroup(dist.ProcessGroup):
    """The world group, carried by gpubridge. Created by ``init_process_group``."""

    def __init__(self, rank: int, size: int, name: str, timeout: timedelta | None) -> None:
        super().__init__(rank, size)
        self._name = name
        self._timeout = timeout
        self._coalesced: list[_work.Work] | None = None

    # ---- identity -------------------------------------------------------------------------

    @property
    def group_name(self) -> str:  # DeviceMesh reads it; a Python group can't set the C++ one
        return self._name

    @property
    def pg_name(self) -> str:
        return self._name

    def getBackendName(self) -> str:  # noqa: N802 - torch's name
        return NAME

    def __repr__(self) -> str:
        return f"GpubridgeProcessGroup(rank={self.rank()}, size={self.size()})"

    # ---- setup and teardown ---------------------------------------------------------------

    def _topology(self) -> _topology.Topology:
        """gpubridge's topology, built at the first collective (every rank's first is the same)."""
        global _owner
        with _setup_lock:
            if _owner is None:
                if _topology.is_initialized():
                    raise RuntimeError(
                        "gpubridge is already initialized in this process (gpubridge.init()); use "
                        f"either gpubridge.init() or the {NAME} torch.distributed backend")
                setup(self._timeout)
                _owner = self
            return _topology.get_topology()

    def shutdown(self) -> None:
        """Called by ``destroy_process_group()`` before it shuts down gpubridge's groups."""
        global _owner
        with _setup_lock:
            if _owner is self:
                _owner = None
                if _topology.is_initialized():
                    from gpubridge import _release

                    _release()

    # ---- running gpubridge collectives ---------------------------------------------------

    def _on_gloo(self, topology: _topology.Topology, tensors: Sequence[torch.Tensor]) -> bool:
        """Whether these are CPU tensors on a GPU job: Gloo carries them directly."""
        return topology.device.type != "cpu" and all(t.device.type == "cpu" for t in tensors)

    def _run(self, what: str, tensors: Sequence[torch.Tensor], result: Any,
             calls: Sequence[Callable[[], _work.Work | None]]) -> dist.Work:
        """Start ``calls`` (gpubridge async collectives), and return a torch Work that completes
        with ``result`` once the last one has: the worker runs them in order."""
        works: list[_work.Work] = []
        try:
            for call in calls:
                work = call()
                assert work is not None
                works.append(work)
        except (ValueError, TypeError, NotImplementedError) as exc:
            raise type(exc)(f"{NAME} backend, {what}: {exc}") from exc
        if self._coalesced is not None:
            self._coalesced.extend(works)
        if not works:
            return _completed(result)
        last = works[-1]

        def done(future: torch.futures.Future) -> Any:
            future.value()  # raises if the collective failed
            for work in works[:-1]:  # an earlier failure fails the later ones too
                if work.exception() is not None:
                    raise RuntimeError(f"{NAME} backend, {what} failed") from work.exception()
            return result

        return _create_work_from_future(last.get_future().then(done))

    # ---- collectives ----------------------------------------------------------------------

    def allreduce(self, tensors, opts=None):  # noqa: D102 - torch's API
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        topology = self._topology()
        if self._on_gloo(topology, tensors):
            return self._gloo("allreduce", tensors, lambda t: dist.all_reduce(
                t, op=op, group=topology.world_group, async_op=True))
        return self._run("allreduce", tensors, list(tensors), [
            lambda t=t: _collectives.all_reduce(t, op=op, async_op=True) for t in tensors])

    def allreduce_coalesced(self, tensors, opts=None):  # noqa: D102
        return self.allreduce(tensors, opts)

    def broadcast(self, tensors, opts=None):  # noqa: D102
        root = opts.rootRank if opts is not None else 0
        topology = self._topology()
        if self._on_gloo(topology, tensors):
            return self._gloo("broadcast", tensors, lambda t: dist.broadcast(
                t, src=root, group=topology.world_group, async_op=True))
        return self._run("broadcast", tensors, list(tensors), [
            lambda t=t: _collectives.broadcast(t, root, async_op=True) for t in tensors])

    def _allgather_base(self, output, input, opts=None):  # noqa: A002, D102 - torch 2.8's name
        topology = self._topology()
        if self._on_gloo(topology, [output, input]):
            return dist.all_gather_into_tensor(output, input, group=topology.world_group,
                                               async_op=True)
        return self._run("all_gather_into_tensor", [output], [output], [
            lambda: _collectives.all_gather_into_tensor(output, input, async_op=True)])

    all_gather_single = _allgather_base  # its name from torch 2.13 on

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):  # noqa: D102
        topology = self._topology()
        if self._on_gloo(topology, [*outputs, *inputs]):
            works = [dist.all_gather_into_tensor(o, i, group=topology.world_group, async_op=True)
                     for o, i in zip(outputs, inputs, strict=True)]
            return _all_of(works, list(outputs))
        return self._run("all_gather_into_tensor", outputs, list(outputs), [
            lambda o=o, i=i: _collectives.all_gather_into_tensor(o, i, async_op=True)
            for o, i in zip(outputs, inputs, strict=True)])

    def allgather(self, output_lists, inputs, opts=None):  # noqa: D102
        """The list form (DDP checks parameter shapes with it): gathered into one buffer per
        input, then copied out. Every rank's tensor must be the same size."""
        topology = self._topology()
        for outputs, tensor in zip(output_lists, inputs, strict=True):
            if len(outputs) != self.size() or any(o.shape != tensor.shape for o in outputs):
                raise _unsupported("allgather with tensors of different sizes",
                                   "every rank must contribute a tensor of the same shape")
        on_gloo = self._on_gloo(topology, inputs)
        for outputs, tensor in zip(output_lists, inputs, strict=True):
            buffer = torch.empty((self.size(), *tensor.shape), dtype=tensor.dtype,
                                 device=tensor.device)
            source = tensor.contiguous()
            if on_gloo:
                dist.all_gather_into_tensor(buffer, source, group=topology.world_group)
            else:
                self._run("allgather", [buffer], None, [
                    lambda b=buffer, s=source: _collectives.all_gather_into_tensor(
                        b, s, async_op=True)]).wait()
            for out, row in zip(outputs, buffer.unbind(0), strict=True):
                out.copy_(row)
        return _completed(output_lists)

    def _reduce_scatter_base(self, output, input, opts=None):  # noqa: A002, D102
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        topology = self._topology()
        if self._on_gloo(topology, [output, input]):
            return dist.reduce_scatter_tensor(output, input, op=op, group=topology.world_group,
                                              async_op=True)
        return self._run("reduce_scatter_tensor", [output], [output], [
            lambda: _collectives.reduce_scatter_tensor(output, input, op=op, async_op=True)])

    reduce_scatter_single = _reduce_scatter_base  # its name from torch 2.13 on

    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):  # noqa: D102
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        topology = self._topology()
        if self._on_gloo(topology, [*outputs, *inputs]):
            works = [dist.reduce_scatter_tensor(o, i, op=op, group=topology.world_group,
                                                async_op=True)
                     for o, i in zip(outputs, inputs, strict=True)]
            return _all_of(works, list(outputs))
        return self._run("reduce_scatter_tensor", outputs, list(outputs), [
            lambda o=o, i=i: _collectives.reduce_scatter_tensor(o, i, op=op, async_op=True)
            for o, i in zip(outputs, inputs, strict=True)])

    def reduce_scatter(self, outputs, input_lists, opts=None):  # noqa: D102
        """The list form: each rank's slices concatenated, then one reduce_scatter_tensor."""
        works = []
        for output, slices in zip(outputs, input_lists, strict=True):
            if len(slices) != self.size() or any(s.shape != output.shape for s in slices):
                raise _unsupported("reduce_scatter with slices of different sizes",
                                   "every slice must have the output's shape")
            joined = torch.cat([s.reshape(-1) for s in slices])
            works.append(self._reduce_scatter_base(output, joined, opts))
        return _all_of(works, list(outputs))

    def barrier(self, opts=None):  # noqa: D102
        self._topology()
        return self._run("barrier", [], [], [lambda: _collectives.barrier(async_op=True)])

    # ---- coalescing (dist._coalescing_manager with a device) ------------------------------

    def _start_coalescing(self, device: torch.device) -> None:
        """Calls until :meth:`_end_coalescing` run as they come, each queued in order."""
        self._coalesced = []

    def _end_coalescing(self, device: torch.device) -> dist.Work:
        works, self._coalesced = self._coalesced or [], None
        if not works:
            return _completed([])
        last = works[-1]
        return _create_work_from_future(last.get_future().then(lambda f: (f.value(), [])[1]))

    # ---- refused --------------------------------------------------------------------------

    def _gloo(self, what: str, tensors, start) -> dist.Work:
        return _all_of([start(t) for t in tensors], list(tensors))

    def send(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("send", "gpubridge has no point-to-point operations yet")

    def recv(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("recv", "gpubridge has no point-to-point operations yet")

    def recv_anysource(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("recv", "gpubridge has no point-to-point operations yet")

    def alltoall(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("all_to_all")

    def alltoall_base(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("all_to_all_single")

    def all_to_all_single(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("all_to_all_single")

    def gather(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("gather")

    def scatter(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("scatter")

    def reduce(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("reduce", "use all_reduce")


def _all_of(works: Sequence[dist.Work], result: Any) -> dist.Work:
    """A Work that completes with ``result`` once every one of ``works`` has."""
    futures = [w.get_future() for w in works]
    if not futures:
        return _completed(result)
    return _create_work_from_future(
        torch.futures.collect_all(futures).then(
            lambda f: ([g.value() for g in f.value()], result)[1]))


def setup(timeout: timedelta | None) -> _topology.Topology:
    """Build gpubridge's topology inside an existing ``torch.distributed`` job.

    Collective over every rank. Creates a Gloo group of every rank for
    discovery, then the islands and the bridge, exactly as ``gpubridge.init()``
    does after it creates its own default group.
    """
    transport = resolve_transport(os.environ.get(BRIDGE_ENV, "").strip() or DEFAULT_TRANSPORT)
    chosen = resolve_policy(os.environ.get(POLICY_ENV, "").strip() or DEFAULT_POLICY)
    local, config = _topology.describe_local(transport.name, chosen)
    world = dist.new_group(list(range(dist.get_world_size())), backend=CPU_BACKEND,
                           timeout=timeout)
    topology = _topology.build_topology(local, config, transport=transport, policy=chosen,
                                        timeout=timeout, world_group=world)
    _work.shutdown()
    _observe._reset()
    _topology._set_topology(topology)
    return topology


def create(options, backend_options=None) -> GpubridgeProcessGroup:
    """The creator ``init_process_group`` calls (``extended_api=True``)."""
    ranks = list(options.global_ranks_in_group)
    world = dist.get_world_size() if dist.is_initialized() else options.group_size
    if ranks and sorted(ranks) != list(range(world)):
        raise RuntimeError(
            f"the {NAME} torch.distributed backend runs on the world group only, but this group "
            f"has ranks {ranks}: gpubridge composes collectives over the islands of the whole "
            "job (docs/TORCH_BACKEND.md). Give subgroups another backend, e.g. "
            "new_group(ranks, backend=\"gloo\").")
    return GpubridgeProcessGroup(options.group_rank, options.group_size, options.group_id,
                                 options.timeout)


def register() -> None:
    """Make ``backend="gpubridge"`` known to ``torch.distributed``. Does nothing if it is
    already (from an earlier call, or gpubridge's ``torch.backends`` entry point)."""
    if dist.is_available() and NAME not in dist.Backend.backend_list:
        dist.Backend.register_backend(NAME, create, extended_api=True, devices=["cpu", "cuda"])
