"""Independent FIFO lanes for the torch backend's cross-island P2P operations.

The collective worker cannot run these: a receive waiting for one peer must
not prevent posting a send to another peer (or in the opposite direction).
Each (peer, tag, direction) lane has its own queue and stream. It keeps the
input alive until completion, and never reorders messages in that lane.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from gpubridge.work import Work


class TorchWork(dist.Work):
    """Keep Python exceptions and stream ordering through torch's P2P entrypoints.

    ``_create_work_from_future`` erases the Python Future's exception unwrap
    callback on torch 2.14.1: wait() would report success for a failed send.
    Delegating explicitly also preserves Work.wait's optional host timeout.
    """

    def __init__(self, work: Work, source: int) -> None:
        super().__init__()
        self.work, self.source = work, source

    def wait(self, timeout=None) -> bool:
        return self.work.wait(timeout)

    def is_completed(self) -> bool:
        return self.work.is_completed()

    def is_success(self) -> bool:
        return self.is_completed() and self.work.exception() is None

    def get_future(self):
        return self.work.get_future()

    def exception(self):
        return self.work.exception()

    def synchronize(self) -> None:
        self.wait()

    def _source_rank(self) -> int:
        return self.source


class GlooLink:
    """Raw bytes over an already-created Gloo group of all members."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group

    def post(self, tensor: torch.Tensor, peer: int, tag: int, sending: bool):
        call = dist.isend if sending else dist.irecv
        arg = {"dst" if sending else "src": peer}
        return call(tensor, group=self.group, tag=tag, **arg)

    def close(self) -> None:
        pass  # torch owns the process group


class Engine:
    """One rank's P2P state. Creation is local; only submitted pairs communicate.

    A small Store handshake checks byte count and dtype before Gloo sees the
    payload: Gloo can terminate the process on a size mismatch. Metadata keys
    include the user tag and lane sequence, and are removed by their reader.
    Failures are shared with all members under this backend group's namespace.
    """

    def __init__(self, link, device: torch.device, rank: int, store,
                 timeout: timedelta | None) -> None:
        self.link, self.device, self.rank, self.store = link, device, rank, store
        self.timeout = (timeout or timedelta(minutes=30)).total_seconds()
        self._lock = threading.RLock()
        self._lanes: dict[tuple[int, int, bool], _Lane] = {}
        self._pending: set[Work] = set()
        self._closed = False
        self._failed: BaseException | None = None
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="gpubridge-p2p-health",
                                         daemon=True)
        self._watcher.start()

    def _watch(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self.check()
            except BaseException as exc:  # noqa: BLE001 - fail pending handles on store loss too
                self._fail(exc, publish=False)
                return

    def check(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("gpubridge point-to-point group was destroyed")
            if self._failed is not None:
                raise RuntimeError(f"gpubridge point-to-point group failed: {self._failed}")
        if self.store.check(["failure"]):
            error = RuntimeError(self.store.get("failure").decode())
            self._fail(error, publish=False)
            raise error

    def submit(self, tensor: torch.Tensor, peer: int, tag: int, sending: bool) -> Work:
        self.check()
        ready = None
        if tensor.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(tensor.device))
        work = Work("send" if sending else "recv", tensor.device, [tensor])
        with self._lock:
            self.check()
            key = peer, tag, sending
            if key not in self._lanes:
                self._lanes[key] = _Lane(self, peer, tag, sending)
            self._pending.add(work)
            self._lanes[key].queue.put((work, tensor, ready, time.monotonic() + self.timeout))
        return work

    def _finish(self, work: Work, error: BaseException | None = None) -> None:
        with self._lock:
            if work not in self._pending:
                return
            self._pending.remove(work)
            work._finish(error)

    def _fail(self, error: BaseException, *, publish: bool = True) -> None:
        with self._lock:
            if self._failed is not None or self._closed:
                return
            self._failed = error
            if publish:
                with contextlib.suppress(Exception):
                    self.store.compare_set("failure", "", f"rank {self.rank}: {error}")
            for work in tuple(self._pending):
                self._finish(work, error)

    def match(self, tensor: torch.Tensor, peer: int, tag: int, seq: int,
              sending: bool, deadline: float) -> None:
        self.check()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"point-to-point operation with rank {peer} timed out in queue")
        src, dst = (self.rank, peer) if sending else (peer, self.rank)
        prefix = f"{src}/{dst}/{tag}/{seq}"
        mine, theirs = (("send", "recv") if sending else ("recv", "send"))
        descriptor = json.dumps([tensor.numel() * tensor.element_size(), str(tensor.dtype)])
        self.store.set(f"{prefix}/{mine}", descriptor)
        key = f"{prefix}/{theirs}"
        while not self.store.check([key]):
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"point-to-point {mine} with rank {peer}, tag {tag} timed out")
            time.sleep(0.002)
        other = self.store.get(key).decode()
        self.store.delete_key(key)
        if descriptor != other:
            raise ValueError(f"point-to-point payload mismatch with rank {peer}, tag {tag}: "
                             f"local {descriptor}, peer {other}")

    def close(self, timeout: float = 0.2) -> None:
        """Fail queued handles immediately; never wait for a missing peer at teardown.

        In-flight Gloo requests retain their buffers in daemon lane threads
        until torch destroys the group or its timeout passes.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            for work in tuple(self._pending):
                self._finish(work, RuntimeError("point-to-point group was destroyed"))
            lanes = list(self._lanes.values())
            for lane in lanes:
                lane.queue.put(None)
        deadline = time.monotonic() + timeout
        for lane in lanes:
            lane.thread.join(max(0, deadline - time.monotonic()))
        self.link.close()


class _Lane:
    def __init__(self, engine: Engine, peer: int, tag: int, sending: bool) -> None:
        self.engine, self.peer, self.tag, self.sending = engine, peer, tag, sending
        self.queue: queue.Queue = queue.Queue()
        self.stream: torch.cuda.Stream | None = None
        self.thread = threading.Thread(target=self._run, name="gpubridge-p2p", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        seq = 0
        while True:
            item = self.queue.get()
            if item is None:
                return
            work, tensor, ready, deadline = item
            try:
                self.engine.check()
                if tensor.device.type == "cuda":
                    torch.cuda.set_device(tensor.device)
                    if self.stream is None:
                        self.stream = torch.cuda.Stream(tensor.device)
                    self.stream.wait_event(ready)
                    tensor.record_stream(self.stream)
                context = (torch.cuda.stream(self.stream) if self.stream is not None
                           else contextlib.nullcontext())
                with context:
                    self.engine.match(tensor, self.peer, self.tag, seq, self.sending, deadline)
                    self._transfer(tensor, deadline)
                    if self.stream is not None:
                        work._event = torch.cuda.Event()
                        work._event.record(self.stream)
                    self.engine._finish(work)
            except BaseException as exc:  # noqa: BLE001 - delivered through every pending Work
                self.engine._fail(exc)
            seq += 1

    def _transfer(self, tensor: torch.Tensor, deadline: float) -> None:
        # Byte views cover every dense dtype, including types Gloo cannot carry
        # natively. The GPU copy runs after the caller's readiness event.
        flat = tensor.reshape(-1).view(torch.uint8)
        host = flat.cpu() if self.sending else torch.empty(flat.numel(), dtype=torch.uint8)
        request = self.engine.link.post(host, self.peer, self.tag, self.sending)
        request.wait(timedelta(seconds=max(0.001, deadline - time.monotonic())))
        if not self.sending:
            flat.copy_(host)
