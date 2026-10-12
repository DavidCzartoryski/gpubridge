"""Host staging for the bridge: reusable pinned host buffers, copy streams and events.

Two users. The pipelined policy (below) stages chunk by chunk through several
slots. reduce-bridge-broadcast, the default, copies with pageable ``.cpu()``
unless ``GPUBRIDGE_PINNED_STAGING`` is on; then it stages the whole tensor
through one pinned buffer (:class:`PinnedBridgeStaging`).

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
  ``after`` is the event of the last host-to-device copy out of that slot,
  from this collective or an earlier one; the copy waits for it on the device,
  so a slot is never overwritten while it is still being read.
- :meth:`Staging.wait_host` blocks the host until one event has passed. It is
  the only host wait. Before the bridge reads a slot, its copy out must have
  passed; before the bridge only receives into a slot, the slot's last copy
  back must have, since the host is about to overwrite it.
- :meth:`Staging.to_device` copies a slot back into its chunk and returns an event.
- :meth:`Staging.end` makes the caller's stream wait for the last copy back,
  so what comes next (the island broadcast) sees every chunk. No host wait,
  so copies back may still be running when the collective returns: the
  pipeline keeps each slot's last copy-back event for the next collective.

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


class BridgeStaging(abc.ABC):
    """Host staging for one leader's whole-tensor bridge step (reduce-bridge-broadcast).

    The protocol: :meth:`to_host` returns a host copy of the tensor once the
    host may read it, the bridge reads and writes that copy, and
    :meth:`to_device` queues the copy back. The host buffer is reused by the
    next collective only after the copy back out of it has passed.
    """

    @abc.abstractmethod
    def to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy ``tensor`` to host memory; return the copy once the host may read it."""

    @abc.abstractmethod
    def to_device(self, tensor: torch.Tensor, host: torch.Tensor) -> None:
        """Queue the copy of ``host`` (from :meth:`to_host`) back into ``tensor``."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the buffer."""


class PinnedBridgeStaging(BridgeStaging):
    """One pinned host buffer, grown to the largest tensor staged so far.

    Copies run on the caller's current stream (the worker's, for async
    collectives), so they are ordered after the island step and before the
    island broadcast like the pageable copies they replace. The host waits on
    an event recorded after the copy out, before the bridge reads the buffer:
    only for that copy and the work queued before it, never the whole device.
    The copy back is not waited for. Its event is kept, and the next
    collective's copy out waits for it on the device (on any stream), so the
    buffer is never overwritten while a copy back still reads it; the host
    then waits on that copy out before the bridge writes the buffer. Before a
    larger buffer replaces it, the host waits for the last copy back.

    The device operations are the three methods at the end, so tests can run
    the protocol on CPU with copies that happen only when waited for.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buffer: torch.Tensor | None = None
        self.copied_back: Event | None = None

    def to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        nbytes = tensor.numel() * tensor.element_size()
        if self.buffer is None or self.buffer.numel() < nbytes:
            if self.copied_back is not None:
                self._wait(self.copied_back)  # the old buffer may still be read
            self.buffer = self._allocate(nbytes)
            self.copied_back = None
        host = self.buffer[:nbytes].view(tensor.dtype).view(tensor.shape)
        self._wait(self._copy(host, tensor, after=self.copied_back))  # before Gloo reads it
        return host

    def to_device(self, tensor: torch.Tensor, host: torch.Tensor) -> None:
        self.copied_back = self._copy(tensor, host, after=None)

    def close(self) -> None:
        if self.copied_back is not None:
            self._wait(self.copied_back)
        self.buffer = None
        self.copied_back = None

    # ---- device operations ----------------------------------------------------------

    def _allocate(self, nbytes: int) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def _copy(self, dst: torch.Tensor, src: torch.Tensor, after: Event | None) -> Event:
        """Queue ``dst.copy_(src)`` on the current stream, after ``after`` if given."""
        stream = torch.cuda.current_stream(self.device)
        if after is not None:
            stream.wait_event(after)
        dst.copy_(src, non_blocking=True)
        done = torch.cuda.Event()
        done.record(stream)
        return done

    def _wait(self, event: Event) -> None:
        """Block the host until ``event`` has passed."""
        event.synchronize()


def make_bridge_staging(device: torch.device) -> BridgeStaging | None:
    """Pinned staging for ``device``'s bridge step; None on CPU, where there is nothing to
    copy. Tests replace this to run and check the protocol on CPU."""
    if device.type == "cuda":
        return PinnedBridgeStaging(device)
    return None


def make_staging(device: torch.device, chunk_bytes: int, slots: int) -> Staging:
    """The staging layer for ``device``. Tests replace this to check the ordering on CPU."""
    if device.type == "cuda":
        return CudaStaging(device, chunk_bytes, slots)
    return DirectStaging()
