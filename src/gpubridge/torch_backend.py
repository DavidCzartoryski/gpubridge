"""gpubridge as a ``torch.distributed`` backend: ``dist.init_process_group(backend="gpubridge")``.

DDP and FSDP2 scripts then run their collectives through gpubridge, across
vendor islands. Design and limits: docs/TORCH_BACKEND.md.

- ``import gpubridge`` registers the name. On torch 2.14 and later, so does
  asking for it (``init_process_group``, ``is_backend_available``), through
  gpubridge's lazy ``torch.distributed.backends`` entry point. ``import torch``
  never runs gpubridge code.
- The process group is the world group. gpubridge's own groups (a Gloo group of
  every rank, the islands, the bridge) are created at its first collective,
  when ``dist.new_group`` can run; discovery then checks every rank as
  ``gpubridge.init()`` does. ``GPUBRIDGE_POLICY`` and ``GPUBRIDGE_BRIDGE`` pick the
  policy and bridge transport (``auto`` and ``gloo`` by default).
- Each collective runs as gpubridge's async collective, so stream semantics are
  those of ``async_op=True`` (:mod:`gpubridge.work`). CPU tensors on a GPU job,
  and gather and scatter of CPU tensors on any job (object collectives,
  checkpointing), go to a Gloo group of the backend's own.
- Unsupported calls (other reductions, point-to-point, all-to-all, reduce,
  GPU-tensor gather and scatter, subgroups) fail before any communication,
  saying what isn't supported.
"""

from __future__ import annotations

import os
import threading
import warnings
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

_setup_lock = threading.RLock()
_owner: GpubridgeProcessGroup | None = None
"""The process group that built gpubridge's topology; it releases it at shutdown."""
_cpu_group: dist.ProcessGroup | None = None
"""A Gloo group of every rank for CPU-tensor collectives, apart from the one
gpubridge's worker thread uses, so the two threads never interleave on one group."""
_setting_up = False


def _completed(result: Any) -> dist.Work:
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _unsupported(what: str, why: str = "") -> NotImplementedError:
    return NotImplementedError(
        f"the {NAME} torch.distributed backend doesn't support {what}"
        + (f": {why}" if why else "") + " (docs/TORCH_BACKEND.md)")


class _NoBackend:
    """What ``_get_backend()`` returns: a Python group has no C++ backend per device.
    ``init_process_group(device_id=...)`` only asks whether it can split eagerly."""

    supports_splitting = False


class GpubridgeProcessGroup(dist.ProcessGroup):
    """The world group, carried by gpubridge. Created by ``init_process_group``."""

    def __init__(self, rank: int, size: int, name: str, timeout: timedelta | None) -> None:
        super().__init__(rank, size)
        self._name = name
        self._timeout = timeout
        self._coalesced: list[_work.Work] | None = None
        self._coalesced_torch: list[dist.Work] = []

    # ---- identity -------------------------------------------------------------------------

    @property
    def group_name(self) -> str:  # DeviceMesh reads it; the C++ getter raises for this group
        return self._name

    @property
    def pg_name(self) -> str:
        return self._name

    def getBackendName(self) -> str:  # noqa: N802 - torch's name
        return NAME

    def _get_backend(self, device: torch.device) -> _NoBackend:
        return _NoBackend()

    def __repr__(self) -> str:
        return f"GpubridgeProcessGroup(rank={self.rank()}, size={self.size()})"

    # ---- setup and teardown ---------------------------------------------------------------

    def _topology(self) -> _topology.Topology:
        """gpubridge's topology, built at the first collective (every rank's first is the same)."""
        global _owner, _cpu_group, _setting_up
        with _setup_lock:
            if _owner is None:
                if _setting_up:
                    raise RuntimeError(
                        f"a collective on the {NAME} backend's group ran while gpubridge was setting "
                        "itself up (from a custom bridge transport or policy?): during setup, "
                        "communicate on topology.world_group instead")
                if _topology.is_initialized():
                    raise RuntimeError(
                        "gpubridge is already initialized in this process (gpubridge.init()); use "
                        f"either gpubridge.init() or the {NAME} torch.distributed backend")
                _setting_up = True
                try:
                    _cpu_group = setup(self._timeout)
                finally:
                    _setting_up = False
                _owner = self
            return _topology.get_topology()

    def _cpu(self) -> dist.ProcessGroup:
        self._topology()
        assert _cpu_group is not None
        return _cpu_group

    def shutdown(self) -> None:
        """Called by ``destroy_process_group()``, after it has shut down gpubridge's own groups
        (it goes in reverse order of group name, the world group last): release gpubridge's
        state without running anything still queued, and never raise."""
        _teardown(self)

    def abort(self) -> None:
        """Called by ``torch.distributed`` to abort the job's groups: as :meth:`shutdown`."""
        _teardown(self)

    # ---- running gpubridge collectives ---------------------------------------------------

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

    def _on_cpu_group(self, what: str, result: Any, starts: Sequence[Callable[[], dist.Work]]
                      ) -> dist.Work:
        """Start Gloo collectives on the backend's CPU group; a Work for all of them."""
        works = [start() for start in starts]
        if self._coalesced is not None:
            self._coalesced_torch.extend(works)
        return _all_of(works, result)

    # ---- collectives ----------------------------------------------------------------------

    def allreduce(self, tensors, opts=None):  # noqa: D102 - torch's API
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        topology = self._topology()
        if _cpu_path(topology, tensors):
            name = _check_op(op, "allreduce")
            return self._on_cpu_group("allreduce", list(tensors), [
                lambda t=t: _gloo_reduce(dist.all_reduce, t, name, self.size(), self._cpu())
                for t in tensors])
        return self._run("allreduce", tensors, list(tensors), [
            lambda t=t: _collectives.all_reduce(t, op=op, async_op=True) for t in tensors])

    def allreduce_coalesced(self, tensors, opts=None):  # noqa: D102
        return self.allreduce(tensors, opts)

    def broadcast(self, tensors, opts=None):  # noqa: D102
        root = opts.rootRank if opts is not None else 0
        topology = self._topology()
        if _cpu_path(topology, tensors):
            return self._on_cpu_group("broadcast", list(tensors), [
                lambda t=t: dist.broadcast(t, src=root, group=self._cpu(), async_op=True)
                for t in tensors])
        return self._run("broadcast", tensors, list(tensors), [
            lambda t=t: _collectives.broadcast(t, root, async_op=True) for t in tensors])

    def _allgather_base(self, output, input, opts=None):  # noqa: A002, D102 - torch 2.8's name
        return self.allgather_into_tensor_coalesced([output], [input], opts)

    all_gather_single = _allgather_base  # its name from torch 2.13 on

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):  # noqa: D102
        topology = self._topology()
        if _cpu_path(topology, [*outputs, *inputs]):
            return self._on_cpu_group("all_gather_into_tensor", list(outputs), [
                lambda o=o, i=i: _gloo_all_gather(o, i, self._cpu())
                for o, i in zip(outputs, inputs, strict=True)])
        return self._run("all_gather_into_tensor", outputs, list(outputs), [
            lambda o=o, i=i: _collectives.all_gather_into_tensor(o, i, async_op=True)
            for o, i in zip(outputs, inputs, strict=True)])

    all_gather_single_coalesced = allgather_into_tensor_coalesced  # torch 2.13 on; also DTensor

    def allgather(self, output_lists, inputs, opts=None):  # noqa: D102
        """The list form (DDP checks parameter shapes with it): gathered into one buffer per
        input, then copied out. Every rank's tensor must be the same size."""
        topology = self._topology()
        for outputs, tensor in zip(output_lists, inputs, strict=True):
            if len(outputs) != self.size() or any(o.shape != tensor.shape for o in outputs):
                raise _unsupported("allgather with tensors of different sizes",
                                   "every rank must contribute a tensor of the same shape")
        on_cpu = _cpu_path(topology, inputs)
        for outputs, tensor in zip(output_lists, inputs, strict=True):
            # Flat: Gloo wants the output a concatenation along dim 0 of the inputs' shape.
            buffer = torch.empty(self.size() * tensor.numel(), dtype=tensor.dtype,
                                 device=tensor.device)
            source = tensor.contiguous().reshape(-1)
            if on_cpu:
                dist.all_gather_into_tensor(buffer, source, group=self._cpu())
            else:
                self._run("allgather", [buffer], None, [
                    lambda b=buffer, s=source: _collectives.all_gather_into_tensor(
                        b, s, async_op=True)]).wait()
            rows = buffer.view(self.size(), *tensor.shape).unbind(0)
            for out, row in zip(outputs, rows, strict=True):
                out.copy_(row)
        return _completed(output_lists)

    def _reduce_scatter_base(self, output, input, opts=None):  # noqa: A002, D102
        return self.reduce_scatter_tensor_coalesced([output], [input], opts)

    reduce_scatter_single = _reduce_scatter_base  # its name from torch 2.13 on

    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):  # noqa: D102
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        topology = self._topology()
        if _cpu_path(topology, [*outputs, *inputs]):
            name = _check_op(op, "reduce_scatter_tensor")
            if name not in ("sum", "avg"):
                raise _unsupported(f"reduce_scatter with {name.upper()}", "SUM and AVG only")
            return self._on_cpu_group("reduce_scatter_tensor", list(outputs), [
                lambda o=o, i=i: _gloo_reduce_scatter(o, i, name, self.size(), self._cpu())
                for o, i in zip(outputs, inputs, strict=True)])
        return self._run("reduce_scatter_tensor", outputs, list(outputs), [
            lambda o=o, i=i: _collectives.reduce_scatter_tensor(o, i, op=op, async_op=True)
            for o, i in zip(outputs, inputs, strict=True)])

    reduce_scatter_single_coalesced = reduce_scatter_tensor_coalesced  # torch 2.13 on

    def reduce_scatter(self, outputs, input_lists, opts=None):  # noqa: D102
        """The list form: each rank's slices concatenated, then one reduce_scatter_tensor."""
        joined = []
        for output, slices in zip(outputs, input_lists, strict=True):
            if len(slices) != self.size() or any(s.shape != output.shape for s in slices):
                raise _unsupported("reduce_scatter with slices of different sizes",
                                   "every slice must have the output's shape")
            joined.append(torch.cat([s.reshape(-1) for s in slices]))
        return self.reduce_scatter_tensor_coalesced(list(outputs), joined, opts)

    def barrier(self, opts=None):  # noqa: D102
        self._topology()
        return self._run("barrier", [], [], [lambda: _collectives.barrier(async_op=True)])

    def gather(self, outputs, inputs, opts=None):  # noqa: D102
        """CPU tensors only, on Gloo: what ``gather_object`` and checkpointing need."""
        root = opts.rootRank if opts is not None else 0
        _cpu_only("gather", [*inputs, *(t for lst in outputs for t in lst)])
        gather_list = outputs[0] if self.rank() == root else None
        return self._on_cpu_group("gather", outputs, [lambda: dist.gather(
            inputs[0], gather_list=gather_list, dst=root, group=self._cpu(), async_op=True)])

    def scatter(self, outputs, inputs, opts=None):  # noqa: D102
        """CPU tensors only, on Gloo: what ``scatter_object_list`` and checkpointing need."""
        root = opts.rootRank if opts is not None else 0
        _cpu_only("scatter", [*outputs, *(t for lst in inputs for t in lst)])
        scatter_list = inputs[0] if self.rank() == root else None
        return self._on_cpu_group("scatter", outputs, [lambda: dist.scatter(
            outputs[0], scatter_list=scatter_list, src=root, group=self._cpu(), async_op=True)])

    # ---- coalescing (dist._coalescing_manager with a device) ------------------------------

    def _start_coalescing(self, device: torch.device) -> None:
        """Calls until :meth:`_end_coalescing` run as they come, each queued in order."""
        self._coalesced, self._coalesced_torch = [], []

    def _end_coalescing(self, device: torch.device) -> dist.Work:
        works, self._coalesced = self._coalesced or [], None
        torch_works, self._coalesced_torch = self._coalesced_torch, []
        futures = [_future_of(w) for w in torch_works]
        if works:  # the worker runs gpubridge's in order: the last one ends after the others
            futures.append(works[-1].get_future())
        if not futures:
            return _completed([])
        return _create_work_from_future(torch.futures.collect_all(futures).then(
            lambda f: ([g.value() for g in f.value()], [])[1]))

    # ---- refused --------------------------------------------------------------------------

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

    def reduce(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("reduce", "use all_reduce")


def _cpu_path(topology: _topology.Topology, tensors: Sequence[torch.Tensor]) -> bool:
    """Whether these are CPU tensors on a GPU job: the backend's Gloo group carries them."""
    return topology.device.type != "cpu" and all(t.device.type == "cpu" for t in tensors)


def _cpu_only(what: str, tensors: Sequence[torch.Tensor]) -> None:
    if not all(t.device.type == "cpu" for t in tensors):
        raise _unsupported(f"{what} of GPU tensors",
                           f"{what} runs on Gloo, for CPU tensors (e.g. {what}_object)")


def _check_op(op, what: str) -> str:
    """The reduction's name, refusing the ones gpubridge refuses, on the CPU path too."""
    try:
        return _collectives._reduce_op_name(op, what)
    except ValueError as exc:
        raise ValueError(f"{NAME} backend, {what}: {exc}") from exc


def _gloo_reduce(fn, tensor: torch.Tensor, name: str, size: int, group) -> dist.Work:
    """A Gloo all_reduce of ``name``; AVG as SUM and a division, as gpubridge does."""
    op = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX,
          "min": dist.ReduceOp.MIN}[name]
    if name != "avg":
        return fn(tensor, op=op, group=group, async_op=True)
    work = fn(tensor, op=op, group=group, async_op=True)
    work.wait()
    tensor.div_(size)
    return _completed([tensor])


def _flat_output(output: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Gloo wants a tensor collective's input and output to be concatenations along dim 0;
    flat ones are. Whether the flat output is ``output`` itself (a view)."""
    if output.is_contiguous():
        return output.view(-1), True
    return output.new_empty(output.numel()), False


def _gloo_all_gather(output, input, group) -> dist.Work:  # noqa: A002
    flat, in_place = _flat_output(output)
    work = dist.all_gather_into_tensor(flat, input.reshape(-1), group=group, async_op=True)
    if in_place:
        return work
    work.wait()
    output.copy_(flat.view(output.shape))
    return _completed([output])


def _gloo_reduce_scatter(output, input, name: str, size: int, group) -> dist.Work:  # noqa: A002
    flat, in_place = _flat_output(output)
    work = dist.reduce_scatter_tensor(flat, input.reshape(-1), op=dist.ReduceOp.SUM,
                                      group=group, async_op=True)
    if name != "avg" and in_place:
        return work
    work.wait()
    if name == "avg":
        flat.div_(size)
    if not in_place:
        output.copy_(flat.view(output.shape))
    return _completed([output])


def _future_of(work: dist.Work) -> torch.futures.Future:
    """A torch Work's future. Some Gloo Works have none (reduce_scatter): wait for those."""
    try:
        return work.get_future()
    except RuntimeError as exc:
        if "not implemented" not in str(exc):
            raise
    work.wait()
    future: torch.futures.Future = torch.futures.Future()
    future.set_result(None)
    return future


def _all_of(works: Sequence[dist.Work], result: Any) -> dist.Work:
    """A Work that completes with ``result`` once every one of ``works`` has."""
    futures = [_future_of(w) for w in works]
    if not futures:
        return _completed(result)
    return _create_work_from_future(
        torch.futures.collect_all(futures).then(
            lambda f: ([g.value() for g in f.value()], result)[1]))


def _teardown(group: GpubridgeProcessGroup) -> None:
    global _owner, _cpu_group
    with _setup_lock:
        if _owner is not group:
            return
        _owner, _cpu_group = None, None
        if not _topology.is_initialized():
            return
        try:
            from gpubridge import _release

            _release(drain=False)  # gpubridge's groups are already shut down
        except Exception as exc:  # noqa: BLE001 - destroy_process_group must finish
            warnings.warn(f"{NAME} backend: releasing gpubridge's state failed: {exc}",
                          RuntimeWarning, stacklevel=3)


def setup(timeout: timedelta | None) -> dist.ProcessGroup:
    """Build gpubridge's topology inside an existing ``torch.distributed`` job; return the
    backend's Gloo group for CPU tensors.

    Collective over every rank. Creates two Gloo groups of every rank (one for
    gpubridge's discovery, ``barrier()`` and ``flat-gloo``, one for the
    backend's CPU tensors), then the islands and the bridge, as
    ``gpubridge.init()`` does after it creates its own default group. A setting
    one rank can't read fails every rank, after the groups exist, instead of
    leaving the others waiting in ``new_group``.
    """
    ranks = list(range(dist.get_world_size()))
    world = dist.new_group(ranks, backend=CPU_BACKEND, timeout=timeout)
    cpu = dist.new_group(ranks, backend=CPU_BACKEND, timeout=timeout)
    problem = None
    try:
        transport = resolve_transport(os.environ.get(BRIDGE_ENV, "").strip()
                                      or DEFAULT_TRANSPORT)
        chosen = resolve_policy(os.environ.get(POLICY_ENV, "").strip() or DEFAULT_POLICY)
        local, config = _topology.describe_local(transport.name, chosen)
    except (ValueError, TypeError) as exc:
        problem = str(exc)
    problems: list[Any] = [None] * len(ranks)
    dist.all_gather_object(problems, problem, group=world)
    failing = [(rank, p) for rank, p in enumerate(problems) if p]
    if failing:
        raise RuntimeError(
            f"gpubridge cannot start: {len(failing)} of {len(ranks)} ranks reported problems.\n"
            + "\n".join(f"  rank {rank}: {p}" for rank, p in failing))
    topology = _topology.build_topology(local, config, transport=transport, policy=chosen,
                                        timeout=timeout, world_group=world)
    _work.shutdown()
    _observe._reset()
    _topology._set_topology(topology)
    return cpu


def create(options, backend_options=None) -> GpubridgeProcessGroup:
    """The creator ``init_process_group`` and ``new_group`` call (``extended_api=True``)."""
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
    already (from an earlier call, or gpubridge's entry point)."""
    if dist.is_available() and NAME not in dist.Backend.backend_list:
        dist.Backend.register_backend(NAME, create, extended_api=True, devices=["cpu", "cuda"])
