"""Collective observers: a timing record for every collective, for tools built on gpubridge.

An observer sees each ``all_reduce``, ``broadcast`` and ``barrier`` this rank
runs, as a :class:`CollectiveRecord`: a sequence number that matches across
ranks, the call's op, size and policy, and timing marks for the whole call and
for each phase the policy marked (reduce-bridge-broadcast marks island-reduce,
bridge and island-broadcast).

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
"""

from __future__ import annotations

import abc
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
    """``"all_reduce"``, ``"broadcast"`` or ``"barrier"``."""
    nbytes: int
    """Size of the tensor; 0 for a barrier."""
    dtype: torch.dtype | None
    src: int | None
    """The source rank of a broadcast; None otherwise."""
    policy: str | None
    """The policy that carried the call, after ``select_policy``; None for a barrier."""
    start: Mark
    end: Mark
    phases: tuple[Phase, ...] = ()
    """The steps this rank took part in. A non-leader has no ``bridge`` phase."""

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
_in_callback = False
_recording: _Recording | None = None


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
    global _seq, _in_callback, _recording
    _observers.clear()
    _seq = 0
    _in_callback = False
    _recording = None


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
                 src: int | None, policy: str | None) -> None:
        self.device = _event_device(topology)
        self.op, self.seq, self.tensor, self.src, self.policy = op, seq, tensor, src, policy
        self.phases: list[Phase] = []

    def __enter__(self) -> None:
        global _recording
        self.start = _new_mark(self.device)
        _recording = self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        global _recording
        _recording = None
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
) -> _Recording | _NoRecording:
    """Wrap one validated collective: count it, and record it if anyone is watching.

    Use as ``with collective(topology, "all_reduce", tensor, policy=name):``.

    Raises:
        RuntimeError: if called from inside an observer callback.
    """
    global _seq
    if _in_callback:
        raise RuntimeError(
            f"gpubridge.{op}() called from inside a collective observer; observers "
            "must not communicate"
        )
    seq = _seq
    _seq += 1
    if not _observers:
        return _NOT_RECORDING
    return _Recording(topology, op, seq, tensor, src, policy)


def phase(name: str) -> _PhaseRecording | _NoRecording:
    """Mark one step of a collective, for policies with more than one step.

    Records nothing when no observer is watching. Use inside
    :meth:`CollectivePolicy.all_reduce` or :meth:`~CollectivePolicy.broadcast`::

        with phase("bridge"):
            topology.bridge.all_reduce(staged)
    """
    recording = _recording
    if recording is None:
        return _NOT_RECORDING
    return _PhaseRecording(recording, name)


def _deliver(record: CollectiveRecord) -> None:
    global _in_callback
    for observer in tuple(_observers):
        _in_callback = True
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
            _in_callback = False
