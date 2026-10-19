"""GPU tensors on the verbs bridge: direct, or staged through registered host buffers.

docs/VERBS_TRANSPORT.md, "GPU memory". :class:`Router` is the worker the shared
collectives (:class:`gpubridge.tagged.TaggedBridge`) drive: it sends each message
from, and receives it into, whatever memory it lies in.

- **Registered host buffers** (the staging buffer and the scratch buffer, both the
  transport's own, registered once): the NIC reads and writes them in place.
- **Registered device memory** (the direct path): the NIC reads sends in place;
  receives of at least ``rndv_bytes`` are written in place by the peer (rendezvous);
  smaller ones arrive in a pinned bounce buffer and are copied to the GPU on the
  caller's stream, before anything after the wait runs on it.
- **Anything else** (a caller's host tensor): copied through the library's slots.

Ordering on the direct path, per NVIDIA's GPUDirect RDMA guide and the ROCm
equivalent the UCX bridge uses (:mod:`gpubridge.ucx.gdr`):

- Before the first message after a wait (and at the start) touches device memory,
  the host waits for an event recorded on the caller's stream: the NIC must not read
  data a kernel hasn't finished writing, nor write memory a queued kernel still uses.
- After a wait that completed receives into device memory, the flush for that GPU
  runs (``cuFlushGPUDirectRDMAWrites``, or nothing where the GPU orders such writes
  itself; ROCm: ``hipDeviceFlushGPUDirectRDMAWrites`` or the HDP flush register)
  before the wait returns, so the kernels launched next see the data. The library
  completes such a receive only once the sender's "written" message, sent after its
  writes on the same RC queue pair, has been polled: the writes are complete with
  respect to this thread.

Staged: :class:`RegisteredStaging` is :class:`gpubridge.staging.PinnedBridgeStaging`
(copies on the caller's stream; the host waits for the copy out before the bridge
reads; the copy back is queued, and the buffer isn't reused until it has passed)
with its buffer registered, so the bridge moves it without another host copy.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from gpubridge.staging import PinnedBridgeStaging
from gpubridge.verbs import Registration, TransportError
from gpubridge.verbs.observe import CallLog

_owners = itertools.count(1)


def new_owner() -> int:
    """A token for a host buffer's registrations, unique in the process."""
    return next(_owners)


@dataclass
class DeviceSpan:
    """Device memory registered for one collective: the registration and the tensor's
    bytes (for copies out of bounce buffers)."""

    registration: Registration
    view: torch.Tensor  # uint8, the tensor's own bytes

    def contains(self, pointer: int, nbytes: int) -> bool:
        start = self.view.data_ptr()
        return start <= pointer and pointer + nbytes <= start + self.view.numel()


@dataclass
class CallState:
    """What one collective call does on the device, and what it moved."""

    spans: list[DeviceSpan] = field(default_factory=list)
    #: The host waits for the caller's stream before the next message on device memory.
    sync_pending: bool = True
    #: Receives into device memory waiting for the flush: op ids.
    written: set[int] = field(default_factory=set)
    #: Bounce copies to do once these ops finish: op id -> (device bytes, host bounce).
    bounces: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    log: CallLog = field(default_factory=CallLog)
    syncs: int = 0
    flushes: int = 0
    #: Receives into device memory that came through a host bounce buffer (below the
    #: rendezvous size), and their bytes: a call with any isn't purely direct.
    bounced_ops: int = 0
    bounced_bytes: int = 0


class Router:
    """The worker behind :class:`gpubridge.verbs.transport.VerbsTransport`'s collectives;
    see the module docs. ``sync()`` waits for the caller's stream, ``flush()`` makes the
    NIC's writes into the current GPU visible to it; both are set per direct call."""

    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint
        self.rndv = endpoint.rndv_bytes()
        #: The transport's registered host buffers (staging, scratch).
        self.host: list[Registration] = []
        self.call: CallState | None = None
        self.sync: Callable[[], None] = lambda: None
        self.flush: Callable[[], None] = lambda: None
        self.bounce: Callable[[int], torch.Tensor] = lambda n: torch.empty(n, dtype=torch.uint8)
        #: Totals for telemetry.
        self.in_place_ops = 0
        self.plain_ops = 0
        self.bounced_ops = 0

    # ---- the worker interface TaggedBridge uses ---------------------------------------

    @property
    def failed(self) -> str | None:
        return self.endpoint.failed

    def _device(self, pointer: int, nbytes: int) -> DeviceSpan | None:
        if self.call is not None:
            for span in self.call.spans:
                if span.contains(pointer, nbytes):
                    return span
        return None

    def _host(self, pointer: int, nbytes: int) -> Registration | None:
        for reg in self.host:
            if reg.contains(pointer, nbytes):
                return reg
        return None

    def _touch_device(self) -> None:
        call = self.call
        if call is not None and call.sync_pending:
            self.sync()
            call.sync_pending = False
            call.syncs += 1

    def _log(self, op: int, member: int, direction: str, nbytes: int) -> int:
        if self.call is not None:
            self.call.log.posted(op, member, direction, nbytes)
        return op

    def send(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        try:
            return self._send(member, pointer, nbytes, tag)
        except TransportError as exc:
            if self.call is not None:
                self.call.log.failed_post(member, "send", nbytes, exc)
            raise

    def _send(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        span = self._device(pointer, nbytes) if nbytes else None
        if span is not None:
            self._touch_device()
            self.in_place_ops += 1
            op = self.endpoint.send_reg(member, span.registration, pointer, nbytes, tag)
        elif nbytes and (reg := self._host(pointer, nbytes)) is not None:
            self.in_place_ops += 1
            op = self.endpoint.send_reg(member, reg, pointer, nbytes, tag)
        else:
            self.plain_ops += 1
            op = self.endpoint.send(member, pointer, nbytes, tag)
        return self._log(op, member, "send", nbytes)

    def recv(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        try:
            return self._recv(member, pointer, nbytes, tag)
        except TransportError as exc:
            if self.call is not None:
                self.call.log.failed_post(member, "recv", nbytes, exc)
            raise

    def _recv(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        span = self._device(pointer, nbytes) if nbytes else None
        if span is not None:
            self._touch_device()
            if self.rndv and nbytes >= self.rndv:
                self.in_place_ops += 1
                op = self.endpoint.recv_reg(member, span.registration, pointer, nbytes, tag)
                self.call.written.add(op)
            else:
                # Below the rendezvous size the message arrives in host slots: take it
                # into a bounce buffer and copy it to the GPU on the caller's stream.
                self.bounced_ops += 1
                self.call.bounced_ops += 1
                self.call.bounced_bytes += nbytes
                bounce = self.bounce(nbytes)
                op = self.endpoint.recv(member, bounce.data_ptr(), nbytes, tag)
                start = pointer - span.view.data_ptr()
                self.call.bounces[op] = (span.view[start:start + nbytes], bounce)
                self._log(op, member, "recv", nbytes)
                self.call.log.bounced(op, nbytes)
                return op
        elif nbytes and (reg := self._host(pointer, nbytes)) is not None:
            self.in_place_ops += 1
            op = self.endpoint.recv_reg(member, reg, pointer, nbytes, tag)
        else:
            self.plain_ops += 1
            op = self.endpoint.recv(member, pointer, nbytes, tag)
        return self._log(op, member, "recv", nbytes)

    def wait(self, ids, deadline: float) -> None:
        self.endpoint.wait(ids, deadline)
        call = self.call
        if call is None:
            return
        call.log.completed(ids)
        done = set(ids)
        if call.written & done:
            # The NIC's writes into the GPU are complete with respect to this thread
            # (their "written" messages were polled); make them visible to its kernels.
            self.flush()
            call.flushes += 1
            call.written -= done
        for op in sorted(done & call.bounces.keys()):
            dst, bounce = call.bounces.pop(op)
            dst.copy_(bounce, non_blocking=True)  # queued on the caller's stream
        # Kernels may be queued on the stream again before the next message.
        call.sync_pending = True

    def abort(self, why: str = "aborted") -> None:
        self.endpoint.abort(why)

    def close(self, timeout: float = 10.0) -> bool:
        return self.endpoint.close(timeout=timeout)

    def telemetry(self) -> dict:
        return self.endpoint.telemetry()

    # ---- calls -------------------------------------------------------------------------

    def begin(self, spans: list[DeviceSpan] | None = None) -> CallState:
        self.call = CallState(spans=list(spans or []))
        return self.call

    def end(self) -> CallState | None:
        call, self.call = self.call, None
        return call


class HostBuffer:
    """A host buffer of the transport's, registered with the NIC once and replaced (its
    registration dropped first) when a bigger one is needed."""

    def __init__(self, router: Router, allocate: Callable[[int], torch.Tensor]) -> None:
        self.router = router
        self.allocate = allocate
        self.buffer: torch.Tensor | None = None
        self.registration: Registration | None = None
        self.owner = 0

    def get(self, nbytes: int) -> torch.Tensor:
        if self.buffer is None or self.buffer.numel() < nbytes:
            self.drop()
            self.buffer = self.allocate(max(nbytes, 64))
            self.owner = new_owner()
            self.registration = self.router.endpoint.register_host(
                self.buffer.data_ptr(), self.buffer.numel(), self.owner)
            self.router.host.append(self.registration)
        return self.buffer[:nbytes]

    def drop(self) -> None:
        """Forget the registration (between collectives: nothing uses it)."""
        if self.registration is not None:
            self.router.host.remove(self.registration)
            endpoint = self.router.endpoint
            if endpoint._handle is not None:
                endpoint.release(self.registration)
                if endpoint.invalidate_host(self.owner):
                    raise TransportError("a registered host buffer was still in use")
            self.registration = None


class RegisteredStaging(PinnedBridgeStaging):
    """:class:`PinnedBridgeStaging` (whose ordering it keeps) with its buffer registered
    with the NIC through ``host``."""

    def __init__(self, device: torch.device, host: HostBuffer) -> None:
        super().__init__(device)
        self.host = host

    def _allocate(self, nbytes: int) -> torch.Tensor:
        # PinnedBridgeStaging asks only after the old buffer's copy back has passed.
        return self.host.get(nbytes)

    def close(self) -> None:
        super().close()
        self.host.drop()


def stream_sync(device: torch.device) -> Callable[[], None]:
    """Waits (on the host) for the work queued on ``device``'s current stream so far."""

    def sync() -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        event.synchronize()

    return sync


def pinned(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

