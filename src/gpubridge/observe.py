"""Collective observers: a timing record for every collective, for tools built on gpubridge.

An observer sees each collective this rank runs (``all_reduce``, ``broadcast``,
``all_gather_into_tensor``, ``reduce_scatter_tensor``, ``barrier``), as a
:class:`CollectiveRecord`: a sequence number that matches across ranks, the
call's op, size and policy, and timing marks for the whole call and for each
phase the policy marked (reduce-bridge-broadcast marks island-reduce, bridge
and island-broadcast).

Rules that keep observers from ever changing what any rank communicates:

- **Observers never communicate.** ``on_collective`` must not call gpubridge or
  torch.distributed. gpubridge refuses its own collectives from inside a
  callback. A tool that needs other ranks' data exchanges it in its own code,
  outside the callback.
- **Observers are local.** Nothing about them is checked across ranks, so ranks
  may attach different observers at different times. Sequence numbers count
  every collective since ``init()``, observed or not, so they still match.
- **A failing observer is detached on its own rank only**, with a warning. No
  observer communicates, so that can't change which collectives any rank runs.
- **Timing never syncs the GPU.** On GPUs a mark is a CUDA event recorded on
  the current stream. Read a record once :meth:`CollectiveRecord.ready` is true;
  until then its times aren't known. On CPU (simulation and CPU-only mode) a
  mark is a ``perf_counter_ns`` reading and records are ready at once.
- **No observers, almost no cost.** Without observers, gpubridge creates no
  events and no records; each collective bumps the sequence counter and notes
  itself in the table :func:`in_flight` reads.
- **Async collectives are observed when they run.** With ``async_op=True`` a
  collective runs on gpubridge's worker thread, so its record is delivered on
  that thread, in submission order.

:func:`in_flight` is a debugging aid for hangs, and works with or without
observers: from any thread, it returns the collective each thread of this rank
is inside right now, with its sequence number, op and current phase. Run on
every rank of a stuck job, it shows which collective and step each one is
waiting in.
"""

from __future__ import annotations

import abc
import threading
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gpubridge.topology import Topology


@dataclass(frozen=True)
class Mark:
    """A point in time on this rank: when the host got there, and when the device did."""

    host_ns: int
    """``time.perf_counter_ns()`` when the host reached this point."""
    event: torch.cuda.Event | None = None
    """On GPUs, a timing event recorded on the current stream; None on CPU."""

    def ready(self) -> bool:
        """Whether the device has reached this mark. Never blocks."""
        return self.event is None or self.event.query()


def elapsed_ms(start: Mark, end: Mark) -> float:
    """Milliseconds from ``start`` to ``end``, on the device's clock when there is one.

    Both marks must be ready.
    """
    if start.event is None or end.event is None:
        return (end.host_ns - start.host_ns) / 1e6
    return start.event.elapsed_time(end.event)


@dataclass(frozen=True)
class Phase:
    """One step of a collective, as marked by its policy."""

    name: str
    start: Mark
    end: Mark

    def elapsed_ms(self) -> float:
        return elapsed_ms(self.start, self.end)


@dataclass(frozen=True)
class CollectiveRecord:
    """One gpubridge collective on this rank."""

    seq: int
    """Collectives this rank ran before this one since ``init()``. The same on every rank."""
    op: str
    """``"all_reduce"``, ``"broadcast"``, ``"all_gather_into_tensor"``,
    ``"reduce_scatter_tensor"`` or ``"barrier"``."""
    nbytes: int
    """Size of the input tensor; 0 for a barrier."""
    dtype: torch.dtype | None
    src: int | None
    """The source rank of a broadcast; None otherwise."""
    policy: str | None
    """The policy that carried the call, after ``select_policy``; None for a barrier."""
    start: Mark
    end: Mark
    phases: tuple[Phase, ...] = ()
    """The steps this rank took part in. A non-leader has no ``bridge`` phase."""
    reduce_op: str | None = None
    """``"sum"``, ``"avg"``, ``"max"`` or ``"min"`` for a reduction; None otherwise."""

    def ready(self) -> bool:
        """Whether every mark has been reached, so times can be read. Never blocks."""
        return self.end.ready() and self.start.ready() and all(
            p.start.ready() and p.end.ready() for p in self.phases
        )

    def elapsed_ms(self) -> float:
        """Duration of the whole call on this rank."""
        return elapsed_ms(self.start, self.end)

    def phase(self, name: str) -> Phase | None:
        """The first phase called ``name``, or None."""
        return next((p for p in self.phases if p.name == name), None)


class CollectiveObserver(abc.ABC):
    """Receives a record of every collective on this rank. Attach with :func:`add_observer`."""

    @abc.abstractmethod
    def on_collective(self, record: CollectiveRecord) -> None:
        """Called on the calling thread right after each collective.

        Keep it cheap: store the record and read its times later. Must not
        communicate. If it raises, the observer is detached on this rank.
        """

    def on_detach(self, error: BaseException) -> None:  # noqa: B027 - optional hook
        """Called once if the observer is detached because ``on_collective`` raised."""


@dataclass(frozen=True)
class InFlight:
    """A collective one thread of this rank is inside right now, from :func:`in_flight`."""

    seq: int
    """The collective's sequence number, the same on every rank (see :class:`CollectiveRecord`)."""
    op: str
    """``"all_reduce"``, ``"broadcast"``, ``"all_gather_into_tensor"``,
    ``"reduce_scatter_tensor"`` or ``"barrier"``."""
    policy: str | None
    """The policy carrying it; None for a barrier."""
    phase: str | None
    """The phase its policy marked that the thread is in now (``island-reduce``,
    ``bridge``...), or None: between phases, or for a policy that marks none."""
    phases_done: tuple[str, ...]
    """Phases this thread finished in this collective, in order."""
    thread: str
    """The name of the thread: the caller's for a blocking call, gpubridge's worker
    thread for ``async_op=True``."""
    seconds: float
    """How long the thread has been inside it, on this rank's monotonic clock."""


class _Flight:
    """One collective in progress on one thread. Read from other threads by in_flight()."""

    __slots__ = ("seq", "op", "policy", "phase", "done", "ident", "since")

    def __init__(self, seq: int, op: str, policy: str | None) -> None:
        self.seq, self.op, self.policy = seq, op, policy
        self.phase: str | None = None
        self.done: tuple[str, ...] = ()

    def enter(self) -> None:
        self.ident = threading.get_ident()
        self.since = time.monotonic()
        _in_flight[self.ident] = self
        _state.flight = self

    def leave(self) -> None:
        _in_flight.pop(self.ident, None)
        _state.flight = None


_observers: list[CollectiveObserver] = []
_seq = 0
_seq_lock = threading.Lock()
_in_flight: dict[int, _Flight] = {}
"""The collective each thread is inside, by thread id."""


class _ThreadState(threading.local):
    """Per thread, because async collectives run (and are observed) on a worker thread."""

    in_callback = False
    recording: _Recording | None = None
    flight: _Flight | None = None


_state = _ThreadState()


def in_flight() -> list[InFlight]:
    """The collective each thread of this rank is inside right now, oldest first.

    A debugging aid for hangs. Safe to call from any thread at any time (a
    watchdog thread, a debugger): it never blocks, communicates or touches a
    device. Empty when no thread is inside a gpubridge collective, or before
    ``init()``.

    It reflects the host: a thread is in flight from the moment it enters a
    gpubridge collective until the call returns. On GPUs a call can return
    before its device work finishes; such work isn't listed. An async
    collective is listed once gpubridge's worker thread starts it, not while it
    waits in the queue.
    """
    for _ in range(10):
        try:
            flights = list(_in_flight.values())
            break
        except RuntimeError:  # another thread entered or left a collective mid-copy
            continue
    else:
        return []
    now = time.monotonic()
    names = {t.ident: t.name for t in threading.enumerate()}
    out = [InFlight(seq=f.seq, op=f.op, policy=f.policy, phase=f.phase, phases_done=f.done,
                    thread=names.get(f.ident, str(f.ident)), seconds=now - f.since)
           for f in flights if hasattr(f, "since")]
    return sorted(out, key=lambda f: f.seq)


def add_observer(observer: CollectiveObserver) -> None:
    """Start sending this rank's collective records to ``observer``.

    Observers are per rank and last until :func:`gpubridge.destroy`.

    Raises:
        RuntimeError: if gpubridge is not initialized.
        TypeError: if ``observer`` is not a :class:`CollectiveObserver`.
    """
    from gpubridge.topology import get_topology

    get_topology()
    if not isinstance(observer, CollectiveObserver):
        raise TypeError(f"expected a CollectiveObserver, got {type(observer).__name__}")
    if observer not in _observers:
        _observers.append(observer)


def remove_observer(observer: CollectiveObserver) -> None:
    """Stop sending records to ``observer``. Does nothing if it isn't attached."""
    if observer in _observers:
        _observers.remove(observer)


def _reset() -> None:
    """Forget every observer and restart the sequence. Called by init() and destroy()."""
    global _seq
    _observers.clear()
    with _seq_lock:
        _seq = 0
    _state.in_callback = False
    _state.recording = None
    _state.flight = None
    _in_flight.clear()


def refuse_in_callback(op: str) -> None:
    """Raise if called from inside an observer callback on this thread.

    The public collectives call this after validating their arguments and
    before queueing or running anything, so a refused call never takes a
    sequence number and never waits for the async queue.

    Raises:
        RuntimeError: when an observer's ``on_collective`` is running on this thread.
    """
    if _state.in_callback:
        raise RuntimeError(
            f"gpubridge.{op}() called from inside a collective observer; observers "
            "must not communicate"
        )


def _event_device(topology: Topology) -> torch.device | None:
    """The device to record timing events on, or None to time on the host."""
    return topology.device if topology.device.type == "cuda" else None


def _new_mark(device: torch.device | None) -> Mark:
    if device is None:
        return Mark(time.perf_counter_ns())
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return Mark(time.perf_counter_ns(), event)


class _NoRecording:
    """What :func:`collective` and :func:`phase` return when nobody is watching."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NOT_RECORDING = _NoRecording()


class _Tracking:
    """What :func:`collective` returns when nobody is watching: only the in-flight entry."""

    __slots__ = ("flight",)

    def __init__(self, flight: _Flight) -> None:
        self.flight = flight

    def __enter__(self) -> None:
        self.flight.enter()

    def __exit__(self, *exc: object) -> None:
        self.flight.leave()


class _Recording:
    def __init__(self, topology: Topology, op: str, seq: int, tensor: torch.Tensor | None,
                 src: int | None, policy: str | None, reduce_op: str | None,
                 flight: _Flight) -> None:
        self.device = _event_device(topology)
        self.op, self.seq, self.tensor, self.src, self.policy = op, seq, tensor, src, policy
        self.reduce_op = reduce_op
        self.flight = flight
        self.phases: list[Phase] = []

    def __enter__(self) -> None:
        self.flight.enter()
        self.start = _new_mark(self.device)
        _state.recording = self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        _state.recording = None
        self.flight.leave()
        if exc_type is not None:
            return
        tensor = self.tensor
        record = CollectiveRecord(
            seq=self.seq,
            op=self.op,
            nbytes=0 if tensor is None else tensor.nbytes,
            dtype=None if tensor is None else tensor.dtype,
            src=self.src,
            policy=self.policy,
            start=self.start,
            end=_new_mark(self.device),
            phases=tuple(self.phases),
            reduce_op=self.reduce_op,
        )
        _deliver(record)


class _PhaseRecording:
    """One phase: always noted in the in-flight entry, timed only if anyone is watching."""

    def __init__(self, flight: _Flight, recording: _Recording | None, name: str) -> None:
        self.flight, self.recording, self.name = flight, recording, name

    def __enter__(self) -> None:
        self.outer = self.flight.phase  # policies may nest phases
        self.flight.phase = self.name
        if self.recording is not None:
            self.start = _new_mark(self.recording.device)

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        self.flight.phase = self.outer
        self.flight.done = (*self.flight.done, self.name)
        if exc_type is None and self.recording is not None:
            end = _new_mark(self.recording.device)
            self.recording.phases.append(Phase(self.name, self.start, end))


def collective(
    topology: Topology,
    op: str,
    tensor: torch.Tensor | None = None,
    *,
    src: int | None = None,
    policy: str | None = None,
    reduce_op: str | None = None,
) -> _Recording | _Tracking:
    """Wrap one validated collective: count it, note it as in flight, and record it if
    anyone is watching.

    Use as ``with collective(topology, "all_reduce", tensor, policy=name):``.
    Called on the thread that runs the collective (the worker thread for
    ``async_op=True``), in the order collectives run.

    Raises:
        RuntimeError: if called from inside an observer callback.
    """
    global _seq
    refuse_in_callback(op)
    with _seq_lock:
        seq = _seq
        _seq += 1
    flight = _Flight(seq, op, policy)
    if not _observers:
        return _Tracking(flight)
    return _Recording(topology, op, seq, tensor, src, policy, reduce_op, flight)


def phase(name: str) -> _PhaseRecording | _NoRecording:
    """Mark one step of a collective, for policies with more than one step.

    :func:`in_flight` shows the step a thread is in; it is timed only when an
    observer is watching. Use inside :meth:`CollectivePolicy.all_reduce` or
    :meth:`~CollectivePolicy.broadcast`::

        with phase("bridge"):
            topology.bridge.all_reduce(staged)
    """
    flight = _state.flight
    if flight is None:
        return _NOT_RECORDING
    return _PhaseRecording(flight, _state.recording, name)


def _deliver(record: CollectiveRecord) -> None:
    for observer in tuple(_observers):
        _state.in_callback = True
        try:
            observer.on_collective(record)
        except Exception as error:  # noqa: BLE001 - any observer failure detaches it
            remove_observer(observer)
            warnings.warn(
                f"gpubridge detached collective observer {observer!r} on this rank after it "
                f"raised {type(error).__name__}: {error}. Collectives continue without it.",
                RuntimeWarning,
                stacklevel=4,
            )
            try:
                observer.on_detach(error)
            except Exception:  # noqa: BLE001, S110 - a failing hook mustn't stop the others
                pass
        finally:
            _state.in_callback = False
