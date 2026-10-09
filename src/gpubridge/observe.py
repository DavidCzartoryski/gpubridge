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
- **No observers, no cost.** Without observers, gpubridge creates no events and
  no records; each collective only bumps the sequence counter.
- **Async collectives are observed when they run.** With ``async_op=True`` a
  collective runs on gpubridge's worker thread, so its record is delivered on
  that thread, in submission order.
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


_observers: list[CollectiveObserver] = []
_seq = 0
_seq_lock = threading.Lock()


class _ThreadState(threading.local):
    """Per thread, because async collectives run (and are observed) on a worker thread."""

    in_callback = False
    recording: _Recording | None = None


_state = _ThreadState()


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


class _Recording:
    def __init__(self, topology: Topology, op: str, seq: int, tensor: torch.Tensor | None,
                 src: int | None, policy: str | None, reduce_op: str | None) -> None:
        self.device = _event_device(topology)
        self.op, self.seq, self.tensor, self.src, self.policy = op, seq, tensor, src, policy
        self.reduce_op = reduce_op
        self.phases: list[Phase] = []

    def __enter__(self) -> None:
        self.start = _new_mark(self.device)
        _state.recording = self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        _state.recording = None
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
    def __init__(self, recording: _Recording, name: str) -> None:
        self.recording, self.name = recording, name

    def __enter__(self) -> None:
        self.start = _new_mark(self.recording.device)

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        if exc_type is None:
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
) -> _Recording | _NoRecording:
    """Wrap one validated collective: count it, and record it if anyone is watching.

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
    if not _observers:
        return _NOT_RECORDING
    return _Recording(topology, op, seq, tensor, src, policy, reduce_op)


def phase(name: str) -> _PhaseRecording | _NoRecording:
    """Mark one step of a collective, for policies with more than one step.

    Records nothing when no observer is watching. Use inside
    :meth:`CollectivePolicy.all_reduce` or :meth:`~CollectivePolicy.broadcast`::

        with phase("bridge"):
            topology.bridge.all_reduce(staged)
    """
    recording = _state.recording
    if recording is None:
        return _NOT_RECORDING
    return _PhaseRecording(recording, name)


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
