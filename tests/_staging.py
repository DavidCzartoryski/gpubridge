"""Staging layers that check the bridge's copy ordering on CPU.

A GPU runs queued copies whenever it gets to them, so the pipeline may only
rely on the events it waits for. LazyStaging makes that literal: a copy runs
only when something waits for its event (or for a later event on the same
stream, since a stream runs in order). If the pipeline forgot a wait, a copy
would run late or never, the bridge would see the wrong bytes, and the result
would no longer match FlatGloo.
"""

from __future__ import annotations

from collections import deque

import torch

from gpubridge.policies import PipelinedReduceBridgeBroadcast
from gpubridge.staging import PinnedBridgeStaging, Staging


class LazyEvent:
    def __init__(self, stream: LazyStream, label: str) -> None:
        self.stream, self.label, self.done = stream, label, False

    def complete(self) -> None:
        if not self.done:
            self.stream.run_through(self)


class LazyStream:
    """Runs its queued copies, in order, only up to an event someone waits for."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.queue: deque = deque()

    def enqueue(self, label: str, action, after: LazyEvent | None) -> LazyEvent:
        event = LazyEvent(self, label)
        self.queue.append((event, action, after))
        return event

    def run_through(self, event: LazyEvent) -> None:
        while not event.done:
            queued, action, after = self.queue.popleft()
            if after is not None:
                after.complete()  # a device-side wait: the other stream gets there first
            action()
            queued.done = True
            self.log.append(("ran", queued.label))


class LazyStaging(Staging):
    """Separate host slots and lazy copies, with a log of every call."""

    def __init__(self, chunk_bytes: int, slots: int) -> None:
        self.buffers = [torch.empty(chunk_bytes, dtype=torch.uint8) for _ in range(slots)]
        self.log: list = []
        self.out = LazyStream(self.log)
        self.back = LazyStream(self.log)
        self.copies = 0

    def slot(self, index: int, chunk: torch.Tensor) -> torch.Tensor:
        return self.buffers[index].view(chunk.dtype)[:chunk.numel()]

    def begin(self, tensor: torch.Tensor) -> None:
        self.log.append(("begin",))

    def _label(self, kind: str) -> str:
        self.copies += 1
        return f"{kind}{self.copies}"

    def to_host(self, slot, chunk, after):
        label = self._label("out")
        self.log.append(("to_host", label, after.label if after else None))
        return self.out.enqueue(label, lambda: slot.copy_(chunk), after)

    def wait_host(self, event):
        self.log.append(("wait_host", event.label if event else None))
        if event is not None:
            event.complete()

    def to_device(self, chunk, slot):
        label = self._label("back")
        self.log.append(("to_device", label))
        return self.back.enqueue(label, lambda: chunk.copy_(slot), None)

    def end(self, event):
        self.log.append(("end", event.label if event else None))
        if event is not None:
            event.complete()


class CallerStreamStaging(LazyStaging):
    """LazyStaging whose end() leaves the copies back queued, as a GPU does.

    On a GPU, end() only makes the caller's stream wait, so a collective
    returns with its copies back still queued. :meth:`run_caller_stream` is
    the caller's stream reaching those waits later.
    """

    def __init__(self, chunk_bytes: int, slots: int) -> None:
        super().__init__(chunk_bytes, slots)
        self.caller_waits: list[LazyEvent] = []

    def end(self, event):
        self.log.append(("end", event.label if event else None))
        if event is not None:
            self.caller_waits.append(event)

    def run_caller_stream(self) -> None:
        for event in self.caller_waits:
            event.complete()
        self.caller_waits.clear()


class LazyPipelined(PipelinedReduceBridgeBroadcast):
    """The pipelined policy on LazyStaging."""

    name = "test-lazy-pipelined"

    def setup(self, topology) -> None:
        super().setup(topology)
        if self.staging is not None:
            self.staging = LazyStaging(self.chunk_bytes, self.SLOTS)


class LazyPinned(PinnedBridgeStaging):
    """PinnedBridgeStaging's protocol on lazy copies, with a log of every call.

    Two pretend streams; :attr:`stream` names the current one, as a caller's
    stream and an async worker's would be. A copy runs only when something
    waits for it, or for a later copy on its stream. :meth:`run_streams` is
    the device getting to everything still queued.
    """

    def __init__(self) -> None:
        super().__init__(torch.device("cpu"))
        self.log: list = []
        self.streams = {"caller": LazyStream(self.log), "other": LazyStream(self.log)}
        self.stream = "caller"
        self.copies = 0

    def _allocate(self, nbytes):
        self.log.append(("allocate", nbytes))
        return torch.empty(nbytes, dtype=torch.uint8)

    def _copy(self, dst, src, after):
        self.copies += 1
        label = f"copy{self.copies}"
        self.log.append(("copy", label, after.label if after else None))
        return self.streams[self.stream].enqueue(label, lambda: dst.copy_(src), after)

    def _wait(self, event):
        self.log.append(("wait", event.label))
        event.complete()

    def run_streams(self) -> None:
        for stream in self.streams.values():
            if stream.queue:
                stream.run_through(stream.queue[-1][0])


class EagerCopyBack(LazyPinned):
    """LazyPinned whose copies back run at once.

    On a GPU the island broadcast that follows is queued behind the copy back
    on the same stream; on CPU it is a host call that wouldn't wait for a lazy
    copy. Copies out stay lazy, so the bridge still needs to_host's wait.
    """

    def to_device(self, tensor, host):
        super().to_device(tensor, host)
        self._wait(self.copied_back)
