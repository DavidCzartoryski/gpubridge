"""gpubridge as a ``torch.distributed`` backend: ``dist.init_process_group(backend="gpubridge")``.

DDP and FSDP2 scripts then run their collectives through gpubridge, across
vendor islands. Design and limits: docs/TORCH_BACKEND.md.

- ``import gpubridge`` registers the name. On torch 2.14 and later, so does
  asking for it (``init_process_group``, ``is_backend_available``), through
  gpubridge's lazy ``torch.distributed.backends`` entry point. ``import torch``
  never runs gpubridge code.
- The world group's gpubridge groups (a Gloo group of every rank, the islands,
  the bridge) are created at its first collective, when ``dist.new_group`` can
  run; discovery then checks every rank as ``gpubridge.init()`` does. A
  subgroup (``new_group``, a ``DeviceMesh`` dimension) builds a topology of its
  own over its members at its first collective: ``native-only`` if they are one
  vendor, bridged if they are mixed (docs/TORCH_BACKEND.md, "Subgroups and HSDP"). ``GPUBRIDGE_POLICY`` and ``GPUBRIDGE_BRIDGE`` pick the
  policy and bridge transport (``auto`` and ``gloo`` by default).
- Each collective runs as gpubridge's async collective, so stream semantics are
  those of ``async_op=True`` (:mod:`gpubridge.work`). CPU tensors on a GPU job,
  and gather and scatter of CPU tensors on any job (object collectives,
  checkpointing), go to a Gloo group of the backend's own.
- Unsupported calls (other reductions, any-source receive, all-to-all, reduce,
  GPU-tensor gather and scatter) fail before any communication, saying what
  isn't supported.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future

from gpubridge import collectives as _collectives
from gpubridge import observe as _observe
from gpubridge import p2p as _p2p
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
_world_setups = itertools.count()
"""Every rank enters world setup in lockstep, even when torch's group counts differ.
Never reset: internal groups and transport store keys can outlive the owning group."""
_subgroups: list[GpubridgeProcessGroup] = []
"""Subgroups whose topology is built; the world's teardown closes what they hold."""


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
    supports_coalescing = True


class GpubridgeProcessGroup(dist.ProcessGroup):
    """A group carried by gpubridge: the world (``init_process_group``), or a subgroup
    (``new_group``, a ``DeviceMesh`` dimension) with ``members``, its global ranks."""

    def __init__(self, rank: int, size: int, name: str, timeout: timedelta | None,
                 members: Sequence[int] = (), store=None) -> None:
        super().__init__(rank, size)
        self._name = name
        self._timeout = timeout
        self._members = tuple(members)
        self._sub: tuple[_topology.Topology, dist.ProcessGroup] | None = None
        self._sub_lock = threading.Lock()
        self._building = False
        self._coalesced: list[_work.Work] | None = None
        self._coalesced_torch: list[dist.Work] = []
        self._native_p2p_coalescing = None
        self._p2p: _p2p.Engine | None = None
        self._p2p_lock = threading.Lock()
        self._p2p_store = dist.PrefixStore("gpubridge-p2p", store) if store is not None else None

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
        members = f", members={list(self._members)}" if self._members else ""
        return f"GpubridgeProcessGroup(rank={self.rank()}, size={self.size()}{members})"

    def _global(self, group_rank: int) -> int:
        """The global rank of this group's rank ``group_rank`` (torch passes group ranks)."""
        return self._members[group_rank] if self._members else group_rank

    # ---- setup and teardown ---------------------------------------------------------------

    def _topology(self) -> _topology.Topology:
        """This group's topology: the world's, or for a subgroup its own, built at its first
        collective by its members (after the world's, if no collective has built that yet).

        The world's is always built and owned by the default group, with its timeout,
        whichever group's collective comes first: destroying a subgroup must not
        release gpubridge for the whole job."""
        self._check_p2p()
        world = _default_group(self)._world_topology()
        if not self._members:
            return world
        with self._sub_lock:
            if self._sub is None:
                if self._building:
                    raise RuntimeError(
                        f"a collective on a {NAME} subgroup ran while that subgroup was setting "
                        "itself up (from a custom bridge transport or policy?)")
                self._building = True
                try:
                    self._sub = _topology.build_subtopology(world, self._members, self._name,
                                                            self._timeout)
                finally:
                    self._building = False
                with _setup_lock:
                    _subgroups.append(self)
            return self._sub[0]

    def _world_topology(self) -> _topology.Topology:
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
        if self._members:
            assert self._sub is not None
            return self._sub[1]
        assert _cpu_group is not None
        return _cpu_group

    def shutdown(self) -> None:
        """Called by ``destroy_process_group()``, after it has shut down gpubridge's own groups
        (it goes in reverse order of group name, the world group last): release gpubridge's
        state without running anything still queued, and never raise. A subgroup closes
        its own topology's policy and bridge; torch destroys its groups."""
        if self._p2p is not None:
            self._p2p.close()
        if self._members:
            _close_subgroup(self)
            return
        _teardown(self)

    def abort(self) -> None:
        """Called by ``torch.distributed`` to abort the job's groups: as :meth:`shutdown`."""
        self.shutdown()

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
            lambda t=t: _collectives.all_reduce(t, op=op, async_op=True, _topology=topology)
            for t in tensors])

    def allreduce_coalesced(self, tensors, opts=None):  # noqa: D102
        return self.allreduce(tensors, opts)

    def broadcast(self, tensors, opts=None):  # noqa: D102
        root = self._global(opts.rootRank if opts is not None else 0)  # a group rank
        topology = self._topology()
        if _cpu_path(topology, tensors):
            return self._on_cpu_group("broadcast", list(tensors), [
                lambda t=t: dist.broadcast(t, src=root, group=self._cpu(), async_op=True)
                for t in tensors])
        return self._run("broadcast", tensors, list(tensors), [
            lambda t=t: _collectives.broadcast(t, root, async_op=True, _topology=topology)
            for t in tensors])

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
            lambda o=o, i=i: _collectives.all_gather_into_tensor(o, i, async_op=True,
                                                                 _topology=topology)
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
                        b, s, async_op=True, _topology=topology)]).wait()
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
            lambda o=o, i=i: _collectives.reduce_scatter_tensor(o, i, op=op, async_op=True,
                                                                _topology=topology)
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
        topology = self._topology()
        return self._run("barrier", [], [], [
            lambda: _collectives.barrier(async_op=True, _topology=topology)])

    def gather(self, outputs, inputs, opts=None):  # noqa: D102
        """CPU tensors only, on Gloo: what ``gather_object`` and checkpointing need."""
        root = opts.rootRank if opts is not None else 0  # a group rank
        _cpu_only("gather", [*inputs, *(t for lst in outputs for t in lst)])
        gather_list = outputs[0] if self.rank() == root else None
        return self._on_cpu_group("gather", outputs, [lambda: dist.gather(
            inputs[0], gather_list=gather_list, dst=self._global(root), group=self._cpu(),
            async_op=True)])

    def scatter(self, outputs, inputs, opts=None):  # noqa: D102
        """CPU tensors only, on Gloo: what ``scatter_object_list`` and checkpointing need."""
        root = opts.rootRank if opts is not None else 0  # a group rank
        _cpu_only("scatter", [*outputs, *(t for lst in inputs for t in lst)])
        scatter_list = inputs[0] if self.rank() == root else None
        return self._on_cpu_group("scatter", outputs, [lambda: dist.scatter(
            outputs[0], scatter_list=scatter_list, src=self._global(root), group=self._cpu(),
            async_op=True)])

    # ---- coalescing (dist._coalescing_manager with a device) ------------------------------

    def _start_coalescing(self, device: torch.device) -> None:
        """Calls until :meth:`_end_coalescing` run as they come, each queued in order."""
        self._coalesced, self._coalesced_torch = [], []

    def _end_coalescing(self, device: torch.device) -> dist.Work:
        if self._native_p2p_coalescing is not None:
            native, self._native_p2p_coalescing = self._native_p2p_coalescing, None
            self._coalesced_torch.append(native._end_coalescing(device))
        works, self._coalesced = self._coalesced or [], None
        torch_works, self._coalesced_torch = self._coalesced_torch, []
        futures, members = [], list(works)
        for work in torch_works:
            try:
                futures.append(work.get_future())
                members.append(work)
            except RuntimeError as exc:
                if "not implemented" not in str(exc):
                    raise
                # Gloo's P2P Works have no future: _future_of waits for it now. Not a
                # member: a second wait() on a finished Gloo send or recv waits for a
                # transfer that never comes, until the group's timeout.
                futures.append(_future_of(work))
        if works:  # the worker runs gpubridge's in order: the last one ends after the others
            futures.append(works[-1].get_future())
        if not futures:
            return _completed([])
        return _BatchWork(members, torch.futures.collect_all(futures).then(
            lambda f: ([g.value() for g in f.value()], [])[1]))

    # ---- point-to-point -------------------------------------------------------------------

    def _check_p2p(self) -> None:
        if self._p2p is not None:
            self._p2p.check()
        elif self._p2p_store is not None and self._p2p_store.check(["failure"]):
            raise RuntimeError(self._p2p_store.get("failure").decode())

    def _point_to_point(self, tensors, peer: int, tag: int, sending: bool):
        if len(tensors) != 1:
            raise ValueError("gpubridge point-to-point takes exactly one tensor")
        tensor = tensors[0]
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("gpubridge point-to-point takes contiguous dense tensors")
        if not 0 <= peer < self.size():
            raise ValueError(f"point-to-point peer {peer} is outside this group")
        if not isinstance(tag, int) or not 0 <= tag < 2**31:
            raise ValueError("point-to-point tag must be an integer in [0, 2**31)")
        # Setup is collective, whereas P2P may involve only two members. Refuse
        # locally rather than rendezvous with ranks that never entered this call.
        if _owner is None or (self._members and self._sub is None):
            raise RuntimeError("gpubridge point-to-point needs topology setup first; "
                               "call dist.barrier() on the world and this group first")
        topology = self._topology()
        if tensor.device.type != "cpu" and tensor.device != topology.device:
            raise ValueError(f"point-to-point tensor is on {tensor.device}, expected "
                             f"{topology.device} or cpu")
        global_peer = self._global(peer)
        if global_peer in topology.island.ranks:
            group = (self._cpu() if tensor.device.type == "cpu" and
                     topology.device.type != "cpu" else topology.island_group)
            fn = dist.isend if sending else dist.irecv
            native_batch = self._coalesced is not None and tensor.device.type == "cuda"
            if native_batch and self._native_p2p_coalescing is None:
                group._start_coalescing(tensor.device)
                self._native_p2p_coalescing = group
            work = fn(tensor, group=group, tag=tag,
                       **{"dst" if sending else "src": global_peer})
            if self._coalesced is not None and not native_batch:
                self._coalesced_torch.append(work)
            return work
        with self._p2p_lock:
            if self._p2p is None:
                assert topology.transport is not None and topology.world_group is not None
                assert self._p2p_store is not None
                link = topology.transport.create_p2p(topology.world_group,
                                                       store=self._p2p_store,
                                                       timeout=self._timeout)
                self._p2p = _p2p.Engine(link,
                                        topology.device, topology.rank, self._p2p_store,
                                        self._timeout, pinned=topology.pinned_staging,
                                        chunk_bytes=_p2p.chunk_bytes_setting())
        work = self._p2p.submit(tensor, global_peer, tag, sending)
        torch_work = _p2p.TorchWork(work, peer if not sending else self.rank())
        if self._coalesced is not None:
            self._coalesced_torch.append(torch_work)
        return torch_work

    def send(self, tensors, dst: int, tag: int):  # noqa: D102
        return self._point_to_point(tensors, dst, tag, True)

    def recv(self, tensors, src: int, tag: int):  # noqa: D102
        return self._point_to_point(tensors, src, tag, False)

    def recv_anysource(self, *args, **kwargs):  # noqa: D102
        raise _unsupported("any-source recv", "specify src so the island route is known")

    # ---- refused --------------------------------------------------------------------------

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


class _BatchWork(dist.Work):
    """A coalesced batch's Work (``batch_isend_irecv``, ``_coalescing_manager``).

    ``wait()`` waits on each member itself, so a batch completes on the host when its
    members do. A Work made from the members' futures would wait for those futures, which
    one completion thread per group resolves, behind user callbacks: a slow callback on
    another tag would delay the batch, and a callback waiting on the batch would wait
    forever. A finite ``wait(timeout)`` also works, as for the members. The future is
    every member's."""

    def __init__(self, works: Sequence[dist.Work], future: torch.futures.Future) -> None:
        super().__init__()
        self._works, self._future = list(works), future

    def wait(self, timeout: timedelta | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout.total_seconds()
        for work in self._works:
            if deadline is None:
                work.wait()
            else:
                work.wait(timedelta(seconds=max(0.0, deadline - time.monotonic())))
        return True

    def is_completed(self) -> bool:
        return all(work.is_completed() for work in self._works)

    def is_success(self) -> bool:
        return self.is_completed() and self.exception() is None

    def exception(self) -> BaseException | None:
        return next((e for e in (w.exception() for w in self._works) if e is not None), None)

    def get_future(self) -> torch.futures.Future:
        return self._future

    def synchronize(self) -> None:
        self.wait()


def _all_of(works: Sequence[dist.Work], result: Any) -> dist.Work:
    """A Work that completes with ``result`` once every one of ``works`` has."""
    futures = [_future_of(w) for w in works]
    if not futures:
        return _completed(result)
    return _create_work_from_future(
        torch.futures.collect_all(futures).then(
            lambda f: ([g.value() for g in f.value()], result)[1]))


def _default_group(group: GpubridgeProcessGroup) -> GpubridgeProcessGroup:
    """The job's default group if it is gpubridge's (it is under ``init_process_group(
    backend="gpubridge")``), else ``group``: the group that builds and owns the world's
    topology."""
    try:
        default = dist.distributed_c10d._get_default_group()
    except (RuntimeError, ValueError):
        return group
    return default if isinstance(default, GpubridgeProcessGroup) else group


def _close_subgroup(group: GpubridgeProcessGroup) -> None:
    """Close a subgroup's policy and bridge; never raise (destroy_process_group must finish)."""
    with _setup_lock:
        if group in _subgroups:
            _subgroups.remove(group)
        sub, group._sub = group._sub, None
    if sub is None:
        return
    topology = sub[0]
    try:
        topology.policy.close()
        if topology.bridge is not None:
            topology.bridge.close()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{NAME} backend: closing a subgroup's policy or bridge failed: {exc}",
                      RuntimeWarning, stacklevel=3)


def _teardown(group: GpubridgeProcessGroup) -> None:
    global _owner, _cpu_group
    with _setup_lock:
        if _owner is not group:
            return
        _owner, _cpu_group = None, None
        subgroups = list(_subgroups)
        if not _topology.is_initialized():
            return
        try:
            from gpubridge import _release

            _release(drain=False)  # gpubridge's groups are already shut down
        except Exception as exc:  # noqa: BLE001 - destroy_process_group must finish
            warnings.warn(f"{NAME} backend: releasing gpubridge's state failed: {exc}",
                          RuntimeWarning, stacklevel=3)
    for sub in subgroups:  # after the worker is abandoned, as for the world's own
        _close_subgroup(sub)


def _world_group_names() -> contextlib.AbstractContextManager[None]:
    """Name the world setup's groups like a subgroup's (``gpubridge:world{generation}.{ranks}.{k}``,
    :func:`gpubridge.topology._members_only`), not from torch's count of the groups each
    process has created. The ranks can reach their first collective having created
    different numbers of groups: torch 2.9.1's ``DeviceMesh`` creates one for a (2, 1)
    mesh's replicate dimension where 2.14.1's reuses the default group, and groups of this
    backend are created without any exchange, so nothing stops there. torch's names would
    then differ by one between the ranks, and they would wait on different store keys
    until the timeout. On a torch whose private naming isn't what the patch expects,
    torch's names are used as before, with a warning."""
    problem = _topology._subgroup_naming_problem()
    if problem is None:
        return _topology._members_only(f"gpubridge:world{next(_world_setups)}")
    warnings.warn(
        f"gpubridge names its setup groups the way torch {torch.__version__} does ({problem}), "
        "so every rank must have created the same number of groups before the first "
        f"collective on the {NAME} backend, or setup waits until the timeout",
        RuntimeWarning, stacklevel=3)
    return contextlib.nullcontext()


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
    with _world_group_names():
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
                f"gpubridge cannot start: {len(failing)} of {len(ranks)} ranks reported "
                "problems.\n" + "\n".join(f"  rank {rank}: {p}" for rank, p in failing))
        topology = _topology.build_topology(local, config, transport=transport, policy=chosen,
                                            timeout=timeout, world_group=world)
    _work.shutdown()
    _observe._reset()
    _topology._set_topology(topology)
    return cpu


def create(options, backend_options=None) -> GpubridgeProcessGroup:
    """The creator ``init_process_group`` and ``new_group`` call (``extended_api=True``).
    torch calls it on the group's members only."""
    ranks = list(options.global_ranks_in_group)  # empty for the world group
    if ranks != sorted(ranks):
        raise RuntimeError(
            f"the {NAME} torch.distributed backend takes a subgroup's ranks in ascending order "
            f"(its group ranks), but got {ranks}: new_group(..., sort_ranks=False) isn't "
            "supported (docs/TORCH_BACKEND.md)")
    world = dist.get_world_size() if dist.is_initialized() else options.group_size
    members = ranks if ranks and ranks != list(range(world)) else []
    if members:
        _topology.check_subgroup_naming()  # here, not at its first collective mid-training
    return GpubridgeProcessGroup(options.group_rank, options.group_size, options.group_id,
                                 options.timeout, members, options.store)


def register() -> None:
    """Make ``backend="gpubridge"`` known to ``torch.distributed``. Does nothing if it is
    already (from an earlier call, or gpubridge's entry point)."""
    if dist.is_available() and NAME not in dist.Backend.backend_list:
        dist.Backend.register_backend(NAME, create, extended_api=True, devices=["cpu", "cuda"])


def batch_isend_irecv(operations):
    """A coalesced P2P batch on every supported torch release.

    torch 2.8's batch helper excludes Python ProcessGroup subclasses from its
    coalescing path. This explicit helper supplies the batch boundary without
    changing torch's global API; other backends use torch's helper unchanged.
    torch 2.14 automatically coalesces the gpubridge backend's native GPU ops.
    """
    dist.distributed_c10d._check_p2p_op_list(operations)
    group = operations[0].group or dist.distributed_c10d._get_default_group()
    if not isinstance(group, GpubridgeProcessGroup):
        return dist.batch_isend_irecv(operations)
    device = operations[0].tensor.device
    with dist.distributed_c10d._coalescing_manager(group, device, async_ops=True) as cm:
        for op in operations:
            op.op(op.tensor, group=op.group, tag=op.tag,
                  **{"group_dst" if op.op is dist.isend else "group_src": op.group_peer})
    return cm.works
