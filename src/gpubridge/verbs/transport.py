"""``bridge="verbs"``: the island leaders' bridge over RDMA reliable connections.

Opt-in (docs/VERBS_TRANSPORT.md). The Rust library in :mod:`gpubridge.verbs`
holds an RC queue pair to every other leader on the device, port and GID it
selects, and moves tagged messages with RC SEND/RECV through its own registered
buffers, with credits and a completion thread (the protocol is the doc's "Wire
protocol"). The collectives and reductions are :class:`gpubridge.tagged.TaggedBridge`'s,
the same code as ``bridge="multitcp"``, in chunks of ``GPUBRIDGE_VERBS_CHUNK_BYTES``
(4 MiB by default). Messages of at least ``GPUBRIDGE_VERBS_RNDV_BYTES`` (256 KiB by
default) go by rendezvous: into registered memory the peer RDMA WRITEs them.

It takes GPU tensors too (``device_tensors``), per call by one of two paths, and says
which and why (``telemetry()["path"]``, ``["path_reason"]``):

- **direct**: the tensor's memory is registered with the NIC (a dma-buf, else NVIDIA
  peer memory; :mod:`gpubridge.verbs.gpu`), sent and written in place, ordered against
  the caller's stream and flushed before GPU work reads it. A receive below
  ``GPUBRIDGE_VERBS_RNDV_BYTES`` can't be written in place: it comes through a pinned
  bounce buffer, and a call with any such receive is **direct+bounce**, its reason
  saying how many bytes (``last_gpu["bounced_bytes"]``);
- **staged**: copied through a pinned host buffer registered with the NIC, on the
  caller's stream (``GPUBRIDGE_VERBS_GPU_PATH=staged`` forces it; ``direct`` refuses to
  stage and fails the call with the reason instead).

Direct is used only once every check has passed and the registration succeeded.

The leaders exchange addresses and queue pair numbers through the c10d store and
check they agree (protocol version, link layer, settings) before connecting. If
verbs can't run here (no library, no device, no usable port, a mismatch), setting
up the bridge fails on every rank and says why: payload never moves any other way,
and another bridge is only ever chosen explicitly. Every wait has the deadline
``init()`` gave its groups; a timeout, a failed completion or a lost peer raises,
and the transport refuses later calls.

``GPUBRIDGE_VERBS_OBSERVE=<file>`` emits transport observations for the straggler
detector (:mod:`gpubridge.verbs.observe`: one record per call, per peer and direction).

``telemetry()["staged_phases"]`` splits the host time of the staged calls that
succeeded: the copy out (queued, then waited for: it includes the work queued before
it on the stream), the bridge step, inside it the posts and the library's waits (the
rest is the reductions and Python), and queueing the copy back (not the copy itself,
which the host doesn't wait for). Host clocks only: no GPU time.

Not validated on RDMA hardware or GPUs. ``GPUBRIDGE_VERBS_TEST_FABRIC`` runs it on
a MOCK fabric for tests (see :mod:`gpubridge.verbs`).
"""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import ClassVar

import torch
from torch.distributed import ReduceOp

from gpubridge import verbs
from gpubridge.config import parse_size
from gpubridge.tagged import TaggedBridge
from gpubridge.transport import register_transport
from gpubridge.ucx import gdr
from gpubridge.ucx.transport import _segments, _wait_for
from gpubridge.verbs import observe
from gpubridge.verbs.gpu import (
    DeviceSpan,
    HostBuffer,
    RegisteredStaging,
    Router,
    pinned,
    stream_sync,
)

#: Chunk size of the collectives' messages, e.g. ``4M`` or ``65536``.
CHUNK_BYTES_ENV = "GPUBRIDGE_VERBS_CHUNK_BYTES"
DEFAULT_CHUNK_BYTES = 4 * 2**20
#: The RDMA device (``mlx5_0``), port and GID index; automatic where unset.
DEVICE_ENV = "GPUBRIDGE_VERBS_DEVICE"
PORT_ENV = "GPUBRIDGE_VERBS_PORT"
GID_INDEX_ENV = "GPUBRIDGE_VERBS_GID_INDEX"
#: The protocol's slot size (bytes per RC message) and credits (receives per peer).
SLOT_BYTES_ENV = "GPUBRIDGE_VERBS_SLOT_BYTES"
CREDITS_ENV = "GPUBRIDGE_VERBS_CREDITS"
SEND_SLOTS_ENV = "GPUBRIDGE_VERBS_SEND_SLOTS"
PROGRESS_ENV = "GPUBRIDGE_VERBS_PROGRESS"
PROGRESS_CPU_ENV = "GPUBRIDGE_VERBS_PROGRESS_CPU"
#: Bytes of arrived, not yet received messages a leader buffers per bridge.
UNEXPECTED_BYTES_ENV = "GPUBRIDGE_VERBS_UNEXPECTED_BYTES"
#: Messages of at least this size go by rendezvous (into registered memory: RDMA WRITE);
#: ``0`` turns it off, and with it receives into GPU memory. The same on every leader.
RNDV_BYTES_ENV = "GPUBRIDGE_VERBS_RNDV_BYTES"
DEFAULT_RNDV_BYTES = 256 * 2**10
#: ``auto`` (direct where possible, else staged), ``direct`` (never stage: fail the call
#: with the reason) or ``staged``.
GPU_PATH_ENV = "GPUBRIDGE_VERBS_GPU_PATH"
GPU_PATHS = ("auto", "direct", "staged")
#: ``0`` keeps the direct path from trying a dma-buf, or peer memory.
DMABUF_ENV = "GPUBRIDGE_VERBS_DMABUF"
PEER_MEMORY_ENV = "GPUBRIDGE_VERBS_PEER_MEMORY"
#: Bound on GPU memory registered at once (registrations are cached).
DEVICE_CACHE_BYTES_ENV = "GPUBRIDGE_VERBS_DEVICE_CACHE_BYTES"


def chunk_bytes_setting() -> int:
    raw = os.environ.get(CHUNK_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_CHUNK_BYTES
    try:
        size = parse_size(raw)
    except ValueError as exc:
        raise ValueError(f"{CHUNK_BYTES_ENV}: {exc}") from None
    if size < 8 or size % 8:
        raise ValueError(f"{CHUNK_BYTES_ENV}={raw!r} must be a positive multiple of 8 bytes")
    return size


def _number(env: str, size: bool = False) -> int | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    try:
        return parse_size(raw) if size else int(raw)
    except ValueError:
        raise ValueError(f"{env}={raw!r} must be a {'size' if size else 'whole number'}") \
            from None


def _flag(env: str) -> int | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    if raw not in ("0", "1"):
        raise ValueError(f"{env}={raw!r} must be 0 or 1")
    return int(raw)


def gpu_path_setting() -> str:
    raw = os.environ.get(GPU_PATH_ENV, "").strip().lower() or "auto"
    if raw not in GPU_PATHS:
        raise ValueError(f"{GPU_PATH_ENV}={raw!r} must be one of {', '.join(GPU_PATHS)}")
    return raw


def endpoint_settings() -> dict[str, object]:
    """The library's settings from the environment (unset ones: its defaults, except the
    rendezvous size, 256 KiB unless set)."""
    rndv = _number(RNDV_BYTES_ENV, size=True)
    return {
        "progress": os.environ.get(PROGRESS_ENV, "").strip().lower() or None,
        "progress_cpu": _number(PROGRESS_CPU_ENV),
        "rndv_bytes": DEFAULT_RNDV_BYTES if rndv is None else rndv,
        "dmabuf": _flag(DMABUF_ENV),
        "peer_memory": _flag(PEER_MEMORY_ENV),
        "device_cache_bytes": _number(DEVICE_CACHE_BYTES_ENV, size=True),
        "device": os.environ.get(DEVICE_ENV, "").strip() or None,
        "port": _number(PORT_ENV),
        "gid_index": _number(GID_INDEX_ENV),
        "slot_bytes": _number(SLOT_BYTES_ENV, size=True),
        "credits": _number(CREDITS_ENV),
        "send_slots": _number(SEND_SLOTS_ENV),
        "unexpected_bytes": _number(UNEXPECTED_BYTES_ENV, size=True),
    }


#: Memory kept for good when a close couldn't drain the library's work: it may still be
#: registered, so it must never be freed.
_LEAKED: list = []


@dataclass
class Support:
    """Whether the direct path can run on one GPU, and if not why."""

    ok: bool
    reason: str
    flush: gdr.Flush | None = None
    driver: str | None = None
    capabilities: dict | None = None


@dataclass
class Plan:
    """How one call on a GPU tensor runs."""

    path: str  # "direct" or "staged"
    reason: str
    device: int
    method: str | None = None
    flush: str | None = None
    spans: list[DeviceSpan] = field(default_factory=list)
    scratch: torch.Tensor | None = None
    sync: Callable[[], None] | None = None
    run_flush: Callable[[], None] | None = None


@register_transport
class VerbsTransport(TaggedBridge):
    """The bridge between island leaders over RDMA reliable connections (raw verbs)."""

    name = "verbs"
    device_tensors = True
    _links: ClassVar[itertools.count] = itertools.count()
    _instead = ("verbs is never replaced by another transport on its own; choose one "
                'explicitly (bridge="gloo", say) to run without it')

    def __init__(self, worker: Router, members, rank: int, timeout: timedelta,
                 chunk_bytes: int, *, emitter: observe.Emitter | None = None,
                 base_record: dict | None = None,
                 peer_nics: dict[int, dict] | None = None) -> None:
        super().__init__(worker, members, rank, timeout, chunk_bytes)
        self.gpu_path = gpu_path_setting()
        self.emitter = emitter
        self.base_record = base_record or {}
        self.peer_nics = peer_nics or {}
        #: The training step to put in observations, if the caller sets it.
        self.step: int | None = None
        #: Whether the endpoint is the MOCK fabric: fixed for its life, so read once.
        self._synthetic: bool | None = None
        self.paths = {"host": 0, "direct": 0, "direct+bounce": 0, "staged": 0}
        self.last: dict = {"path": None, "path_reason": None, "gpu": None}
        self._support: dict[int, Support] = {}
        self._drivers: dict[str, str] = {}
        self._staging: dict[int, RegisteredStaging] = {}
        self._host_scratch = HostBuffer(worker, lambda n: torch.empty(n, dtype=torch.uint8))
        self._device_scratch: torch.Tensor | None = None
        #: Memory of calls that failed: the library may hold it registered until close.
        self._keep: list = []
        #: Host seconds in each phase of the staged calls that succeeded (module docs).
        self.staged_phases = {"calls": 0, "to_host_s": 0.0, "bridge_s": 0.0, "post_s": 0.0,
                              "wait_s": 0.0, "to_device_s": 0.0}
        worker.bounce = pinned

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> VerbsTransport:
        chunk, settings = chunk_bytes_setting(), endpoint_settings()
        gpu_path_setting()  # a bad value fails setup, not the first collective
        index = members.index(rank)
        endpoint = verbs.Endpoint(len(members), index, settings)
        emitter = None
        try:
            if endpoint.telemetry().get("mock"):
                verbs.log.warning("gpubridge: bridge=\"verbs\" is running on the MOCK test "
                                  "fabric (%s=%s): not RDMA", verbs.TEST_FABRIC_ENV,
                                  verbs.test_fabric_dir())
            # The chunk size decides which messages each step sends: leaders that
            # disagreed would wait for the wrong ones, so it is checked like the protocol
            # (the library checks its own settings in the handshake line).
            agreed = f"{verbs.protocol(endpoint._lib)}|{chunk}"
            store.set(f"{key}/addr/{rank}", f"{agreed}|{endpoint.local_info()}")
            others = [m for m in members if m != rank]
            entries = _wait_for(store, [f"{key}/addr/{m}" for m in others], deadline,
                                "the other bridge members' verbs addresses",
                                failures=[f"{key}/status/{m}" for m in others])
            peer_nics = {}
            for member, entry in zip(others, entries, strict=True):
                protocol, theirs_chunk, info = entry.split("|", 2)
                theirs = f"{protocol}|{theirs_chunk}"
                if theirs != agreed:
                    raise RuntimeError(
                        f"verbs protocol|{CHUNK_BYTES_ENV} is {agreed} here but {theirs} on rank "
                        f"{member}; use the same gpubridge and settings on every leader")
                endpoint.connect(members.index(member), info)
                peer_nics[members.index(member)] = observe.peer_nic(info)
            # Nothing is sent until every member's queue pairs are ready to receive: a
            # message reaching one still in Init is dropped, and its sender's retries
            # may run out before that peer connects.
            store.set(f"{key}/ready/{rank}", "ok")
            _wait_for(store, [f"{key}/ready/{m}" for m in others], deadline,
                      "the other bridge members' queue pairs to be ready",
                      failures=[f"{key}/status/{m}" for m in others])
            emitter = observe.from_env(rank)
            base = {}
            if emitter is not None:
                base = {"schema": observe.SCHEMA, "host": observe.hostname(),
                        "job": observe.job_id(store), "rank": rank, "bridge": "verbs",
                        "nic": observe.nic_object(endpoint.telemetry())}
            transport = cls(Router(endpoint), members, rank, timeout, chunk, emitter=emitter,
                            base_record=base, peer_nics=peer_nics)
            transport._handshake(deadline)
            return transport
        except BaseException:
            if emitter is not None:
                emitter.close()
            endpoint.abort("setup failed")
            endpoint.close(timeout=1.0)
            raise

    # ---- which memory, which path ----------------------------------------------------

    def _check(self, tensor: torch.Tensor) -> None:
        if not tensor.is_contiguous() or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"the {self.name} bridge carries contiguous host or GPU tensors")

    def _is_device(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` is GPU memory (tests stand CPU tensors in)."""
        return tensor.device.type == "cuda"

    def _device_index(self, tensor: torch.Tensor) -> int:
        return tensor.device.index if tensor.device.index is not None else \
            torch.cuda.current_device()

    def _flush_for(self, index: int) -> gdr.Flush:
        return gdr.for_device(index, bool(torch.version.hip))

    def _support_for(self, index: int) -> Support:
        """Whether the direct path can run on GPU ``index`` (found once per GPU)."""
        if index in self._support:
            return self._support[index]
        endpoint = self.worker.endpoint
        vendor = "rocm" if torch.version.hip else "cuda"
        support = None
        if self.worker.rndv == 0:
            support = Support(False, f"rendezvous is off ({RNDV_BYTES_ENV}=0), so no peer may "
                                     "write into GPU memory")
        if support is None and vendor not in self._drivers:
            try:
                self._drivers[vendor] = endpoint.load_gpu(vendor)
            except verbs.TransportError as exc:
                support = Support(False, f"the {vendor} driver can't be used: {exc}")
        if support is None:
            caps = endpoint.gpu_capabilities(index)
            if caps.get("dmabuf") != "ok" and not str(caps.get("peer_memory")).startswith("ok"):
                support = Support(False, f"dma-buf: {caps.get('dmabuf')}; peer memory: "
                                         f"{caps.get('peer_memory')}", capabilities=caps)
            else:
                flush = self._flush_for(index)
                if not flush.available:
                    support = Support(False, f"no way to make the NIC's writes visible to GPU "
                                             f"{index}: {flush.method}", capabilities=caps)
                else:
                    support = Support(True, "", flush, self._drivers[vendor], caps)
        self._support[index] = support
        return support

    def _staged(self, index: int, reason: str) -> Plan:
        if self.gpu_path == "direct":
            raise RuntimeError(f"the verbs bridge's direct GPU path can't run on GPU {index} "
                               f"and staging is off ({GPU_PATH_ENV}=direct): {reason}")
        return Plan("staged", reason, index)

    def _plan(self, tensor: torch.Tensor, scratch_numel: int) -> Plan:
        """Direct, with the tensor (and scratch) registered, or staged with the reason."""
        index = self._device_index(tensor)
        if self.gpu_path == "staged":
            return Plan("staged", f"{GPU_PATH_ENV}=staged", index)
        if tensor.numel() == 0:
            return self._staged(index, "an empty tensor: nothing to register")
        support = self._support_for(index)
        if not support.ok:
            return self._staged(index, support.reason)
        endpoint = self.worker.endpoint
        flat = tensor.view(-1)
        regs = []
        reg = endpoint.register_device(index, flat.data_ptr(), flat.numel() * flat.element_size())
        if isinstance(reg, str):
            return self._staged(index, f"registering the tensor failed: {reg}")
        regs.append(DeviceSpan(reg, flat.view(torch.uint8)))
        scratch = None
        if scratch_numel:
            scratch = torch.empty(scratch_numel, dtype=flat.dtype, device=flat.device)
            reg2 = endpoint.register_device(index, scratch.data_ptr(),
                                            scratch.numel() * scratch.element_size())
            if isinstance(reg2, str):
                endpoint.release(reg)
                return self._staged(index, f"registering the scratch buffer failed: {reg2}")
            regs.append(DeviceSpan(reg2, scratch.view(torch.uint8)))
        flush = support.flush
        return Plan("direct", f"registered by {reg.method}", index, method=reg.method,
                    flush=flush.method, spans=regs, scratch=scratch,
                    sync=stream_sync(tensor.device), run_flush=flush)

    def _staging_for(self, tensor: torch.Tensor) -> RegisteredStaging:
        index = self._device_index(tensor)
        if index not in self._staging:
            self._staging[index] = RegisteredStaging(
                tensor.device, HostBuffer(self.worker, pinned))
        return self._staging[index]

    def _scratch(self, numel: int, like: torch.Tensor) -> torch.Tensor:
        if self._device_scratch is not None:
            return self._device_scratch[:numel]
        if like.device.type == "cpu":
            nbytes = numel * like.element_size()
            return self._host_scratch.get(nbytes).view(like.dtype)[:numel]
        return super()._scratch(numel, like)

    # ---- collectives -------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)  # a bad argument fails the call, not the bridge
        size = len(self.members)
        scratch = tensor.numel() if size == 2 else \
            max(length for _, length in _segments(tensor.numel(), size))
        self._call(tensor, scratch, lambda t: self._all_reduce(t, op))

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)
        self._call(tensor, 0, lambda t: self._broadcast(t, src))

    def _call(self, tensor: torch.Tensor, scratch_numel: int, run) -> None:
        if not self._is_device(tensor):
            self._run("host", None, tensor, run)
            return
        plan = self._plan(tensor, scratch_numel)
        if plan.path == "direct":
            self._run("direct", plan, tensor, run)
            return
        staging = self._staging_for(tensor)
        router = self.worker
        t0 = time.perf_counter()
        host = staging.to_host(tensor)  # once the copy out has passed
        t1 = time.perf_counter()
        post, wait = router.post_s, router.wait_s
        router.timed = True
        try:
            self._run("staged", plan, host, run)
        finally:
            router.timed = False
        t2 = time.perf_counter()
        staging.to_device(tensor, host)  # queued on the caller's stream
        t3 = time.perf_counter()
        phases = self.staged_phases
        phases["calls"] += 1
        phases["to_host_s"] += t1 - t0
        phases["bridge_s"] += t2 - t1
        phases["post_s"] += router.post_s - post
        phases["wait_s"] += router.wait_s - wait
        phases["to_device_s"] += t3 - t2

    def _run(self, path: str, plan: Plan | None, tensor: torch.Tensor, run) -> None:
        router = self.worker
        direct = plan is not None and plan.path == "direct"
        call = router.begin(plan.spans if direct else [])
        if direct:
            router.sync, router.flush = plan.sync, plan.run_flush
            self._device_scratch = plan.scratch
        status, error = "ok", None
        try:
            self._guarded(lambda: run(tensor))
        except BaseException as exc:
            text = str(exc)
            status = "timeout" if getattr(exc, "timed_out", False) else "error"
            error = f"{type(exc).__name__}: {text}"[:500]
            # The library may hold this memory registered (a peer granted a write) until
            # the endpoint closes.
            self._keep.append((tensor, plan))
            raise
        finally:
            if status != "ok" and hasattr(router.endpoint, "operation_outcome"):
                for op in call.log.ops:
                    if call.log.outcomes.get(op, {}).get("status") == "ok":
                        continue
                    outcome = router.endpoint.operation_outcome(op)
                    if outcome["status"] in ("ok", "error", "cancelled"):
                        # A timeout describes outstanding work, even though _guarded
                        # has now cancelled it to make caller memory safe.
                        if status == "timeout" and outcome["status"] == "cancelled":
                            outcome = {"status": "timeout", "error": error}
                        call.log.outcomes[op] = outcome
            router.end()
            self._device_scratch = None
            router.sync = router.flush = lambda: None
            if direct and status == "ok":
                for span in plan.spans:
                    router.endpoint.release(span.registration)
            reason = plan.reason if plan else None
            if direct and call.bounced_bytes:
                # Not purely direct: say so, and how much was staged.
                path = "direct+bounce"
                reason = (f"{reason}; {call.bounced_bytes} bytes in {call.bounced_ops} receives "
                          f"below the rendezvous size ({RNDV_BYTES_ENV}={router.rndv}) came "
                          "through a pinned host bounce buffer")
            self.paths[path] += 1
            self.last = {"path": path, "path_reason": reason,
                         "gpu": None if plan is None else {
                             "device": plan.device, "method": plan.method,
                             "flush": plan.flush, "syncs": call.syncs,
                             "flushes": call.flushes, "bounced_ops": call.bounced_ops,
                             "bounced_bytes": call.bounced_bytes}}
            self._observe(call, path, status, error)

    def _observe(self, call, path: str, status: str, error: str | None) -> None:
        if self.emitter is None:
            return
        base = {**self.base_record, "collective": self._seq}
        if self._synthetic is None:
            self._synthetic = bool(self.worker.endpoint.telemetry().get("mock"))
        if self._synthetic:
            base["synthetic"] = True
        if path != "host":
            base["path"] = path
        if self.step is not None:
            base["step"] = self.step
        for record in call.log.records(base, status, error, self.members, self.peer_nics):
            self.emitter.emit(record)

    # ---- state -------------------------------------------------------------------------

    def telemetry(self) -> dict:
        """The endpoint's state for telemetry correlation: provider (MOCK or libibverbs
        and its file), device, port, link layer, GID, MTUs, protocol version and
        operations, failure, counts (copied and in-place bytes, rendezvous messages), the
        registrations (``direct``, with the time spent making and releasing write
        windows); then the last call's ``path`` (``host``, ``direct``, ``direct+bounce``,
        ``staged``) and ``path_reason``, ``last_gpu`` (with the call's bounced receives),
        per-path counts, the staged calls' phase times (``staged_phases``, module docs),
        each GPU's direct-path support and the observation emitter's counts. Never a remote key."""
        router = self.worker
        gpus = {str(i): {"direct": s.ok, "reason": s.reason or None, "driver": s.driver,
                         "flush": s.flush.method if s.flush else None,
                         "capabilities": s.capabilities}
                for i, s in self._support.items()}
        return {**router.telemetry(), "chunk_bytes": self.chunk_bytes,
                "failed": router.failed, "path": self.last["path"] or "host",
                "path_reason": self.last["path_reason"], "last_gpu": self.last["gpu"],
                "paths": dict(self.paths), "gpu_path_setting": self.gpu_path, "gpus": gpus,
                "in_place_ops": router.in_place_ops, "plain_ops": router.plain_ops,
                "bounced_ops": router.bounced_ops,
                "staged_phases": dict(self.staged_phases),
                "observations": self.emitter.stats() if self.emitter else None}

    @property
    def state(self) -> str:
        """``ok``, ``failed`` or ``closed``."""
        if self.worker.failed is not None:
            return "failed"
        return self.worker.telemetry().get("state", "closed")

    def close(self) -> None:
        start = time.monotonic()
        if self.emitter is not None:
            self.emitter.close()
        healthy = self.worker.failed is None
        if healthy:
            # Between collectives nothing uses the staging buffers: drop their
            # registrations, then the buffers.
            try:
                for staging in self._staging.values():
                    staging.close()
                self._host_scratch.drop()
            except verbs.TransportError as exc:
                # Something still uses one: keep every buffer and close the endpoint as
                # a failed one would be (below), so it is still closed.
                verbs.log.warning("gpubridge: closing the verbs bridge: %s; keeping its "
                                  "buffers until the endpoint has closed", exc)
        drained = self.worker.close(timeout=min(10.0, self.timeout))
        if not drained:
            # Still registered: never free any of it.
            _LEAKED.append((self._keep, self._staging, self._host_scratch))
        self._keep, self._staging = [], {}
        verbs.log.debug("gpubridge: verbs bridge closed in %.3f s", time.monotonic() - start)
