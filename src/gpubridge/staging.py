"""Host staging for the pipelined bridge: reusable host buffers, copy streams and events.

The pipelined policy splits a leader's bridge step into chunks so that, on
GPUs, the device-to-host copy of chunk k+1, the bridge step of chunk k and the
host-to-device copy of chunk k-1 overlap. This module holds everything that
touches devices, streams and events, behind one small interface, so the
pipeline itself (in :mod:`gpubridge.policies`) runs unchanged on CPU and tests
can swap in a staging layer that checks the ordering.

The protocol, as the pipeline uses it:

- :meth:`Staging.begin` once, before the first copy: the copy streams wait for
  the work already queued on the caller's stream (the island step).
- :meth:`Staging.to_host` copies a chunk into a host slot and returns an event.
  ``after`` is the event of the last host-to-device copy out of that slot; the
  copy waits for it on the device, so a slot is never overwritten while it is
  still being read.
- :meth:`Staging.wait_host` blocks the host until one event has passed. It is
  the only host wait: the bridge needs the chunk in host memory.
- :meth:`Staging.to_device` copies a slot back into its chunk and returns an event.
- :meth:`Staging.end` makes the caller's stream wait for the last copy back,
  so what comes next (the island broadcast) sees every chunk. No host wait.

Nothing here synchronizes the whole device.
"""

from __future__ import annotations

import abc

import torch

Event = object  # a torch.cuda.Event on GPUs; anything the staging layer likes elsewhere


class Staging(abc.ABC):
    """Host buffers and ordered copies for one leader's pipelined bridge."""

    @abc.abstractmethod
    def slot(self, index: int, chunk: torch.Tensor) -> torch.Tensor:
        """A host buffer for ``chunk``'s data: slot ``index``, viewed as its dtype and length."""

    def begin(self, tensor: torch.Tensor) -> None:  # noqa: B027 - optional hook
        """Order the copies after the work already queued for ``tensor``."""

    @abc.abstractmethod
    def to_host(self, slot: torch.Tensor, chunk: torch.Tensor, after: Event | None) -> Event | None:
        """Copy ``chunk`` into ``slot`` once ``after`` (if any) has passed; return its event."""

    @abc.abstractmethod
    def wait_host(self, event: Event | None) -> None:
        """Block the host until ``event`` has passed."""

    @abc.abstractmethod
    def to_device(self, chunk: torch.Tensor, slot: torch.Tensor) -> Event | None:
        """Copy ``slot`` back into ``chunk``; return its event."""

    def end(self, event: Event | None) -> None:  # noqa: B027 - optional hook
        """Order the caller's stream after ``event`` (the last copy back)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the buffers and streams."""


class DirectStaging(Staging):
    """On CPU the tensor is already in host memory, so a slot is the chunk itself."""

    def slot(self, index: int, chunk: torch.Tensor) -> torch.Tensor:
        return chunk

    def to_host(self, slot: torch.Tensor, chunk: torch.Tensor, after: Event | None) -> None:
        return None

    def wait_host(self, event: Event | None) -> None:
        return None

    def to_device(self, chunk: torch.Tensor, slot: torch.Tensor) -> None:
        return None


class CudaStaging(Staging):
    """Pinned host slots, one stream for copies out and one for copies back, ordered by events.

    Two streams, so a copy out (chunk k+1) and a copy back (chunk k-1) can run
    at the same time on the GPU's separate copy engines.
    """

    def __init__(self, device: torch.device, chunk_bytes: int, slots: int) -> None:
        self.device = device
        self.buffers = [torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                        for _ in range(slots)]
        self.out_stream = torch.cuda.Stream(device)
        self.back_stream = torch.cuda.Stream(device)

    def slot(self, index: int, chunk: torch.Tensor) -> torch.Tensor:
        return self.buffers[index].view(chunk.dtype)[:chunk.numel()]

    def begin(self, tensor: torch.Tensor) -> None:
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.out_stream.wait_event(ready)
        self.back_stream.wait_event(ready)
        # The caller may free the tensor once the collective returns: tell the
        # caching allocator it is in use on the copy streams too.
        tensor.record_stream(self.out_stream)
        tensor.record_stream(self.back_stream)

    def to_host(self, slot: torch.Tensor, chunk: torch.Tensor,
                after: torch.cuda.Event | None) -> torch.cuda.Event:
        if after is not None:
            self.out_stream.wait_event(after)
        with torch.cuda.stream(self.out_stream):
            slot.copy_(chunk, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.out_stream)
        return done

    def wait_host(self, event: torch.cuda.Event | None) -> None:
        if event is not None:
            event.synchronize()

    def to_device(self, chunk: torch.Tensor, slot: torch.Tensor) -> torch.cuda.Event:
        with torch.cuda.stream(self.back_stream):
            chunk.copy_(slot, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.back_stream)
        return done

    def end(self, event: torch.cuda.Event | None) -> None:
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)

    def close(self) -> None:
        self.buffers.clear()


def make_staging(device: torch.device, chunk_bytes: int, slots: int) -> Staging:
    """The staging layer for ``device``. Tests replace this to check the ordering on CPU."""
    if device.type == "cuda":
        return CudaStaging(device, chunk_bytes, slots)
    return DirectStaging()
