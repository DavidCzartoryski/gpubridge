"""Host-staged UCX links owned by independent directional P2P lanes.

The collective transport's worker is leader-only and not thread-safe. A
matched (src, dst, tag) lane connects its own two-member worker locally, so
an unrelated slow peer cannot block posting on another pair or tag.
"""

from __future__ import annotations

import base64
import threading
import time
from datetime import timedelta

from gpubridge import ucx
from gpubridge.ucx.transport import _wait_for


class _Request:
    def __init__(self, connection, identifier, tensor) -> None:
        self.connection, self.identifier, self.tensor = connection, identifier, tensor
        self.done = False

    def wait(self, timeout: timedelta) -> None:
        connection = self.connection
        with connection.lock:
            if self.done:
                return
            # Worker.wait retires all IDs once every native request is complete,
            # including immediately completed receives. Wait all IDs posted so
            # far together; cached Request objects then remain valid after reuse.
            pending = list(connection.pending)
            connection.worker.wait([request.identifier for request in pending],
                                     time.monotonic() + timeout.total_seconds())
            for request in pending:
                request.done = True
            connection.pending.clear()


class _Connection:
    def __init__(self, worker, sending) -> None:
        self.worker, self.sending = worker, sending
        self.lock = threading.Lock()
        self.pending: list[_Request] = []

    def post(self, tensor):
        with self.lock:
            nbytes = tensor.numel() * tensor.element_size()
            identifier = (self.worker.send(1, tensor.data_ptr(), nbytes, 0) if self.sending
                          else self.worker.recv(tensor.data_ptr(), nbytes, 0))
            request = _Request(self, identifier, tensor)
            self.pending.append(request)
            return request

    def close(self):
        with self.lock:
            self.worker.close(timeout=0.1)
            self.pending.clear()


class UcxLink:
    """Local factory; only the two endpoints of a submitted lane rendezvous.

    Payloads are host tensors. Pinning/chunking is handled by the common P2P
    engine; this does not claim a device-direct UCX P2P path.
    """

    def __init__(self, rank, store, timeout: timedelta) -> None:
        self.rank, self.store, self.timeout = rank, store, timeout.total_seconds()
        self.connections = {}
        self.lock = threading.Lock()
        self.closed = False

    def _connect(self, peer, tag, sending):
        worker = ucx.Worker(2)
        src, dst = (self.rank, peer) if sending else (peer, self.rank)
        prefix = f"ucx/{src}/{dst}/{tag}"
        mine, other = ("send", "recv") if sending else ("recv", "send")
        deadline = time.monotonic() + self.timeout
        try:
            major, minor, release = ucx.version()
            address = base64.b64encode(worker.address()).decode()
            self.store.set(f"{prefix}/{mine}", f"{major}.{minor}.{release}|{address}")
            key = f"{prefix}/{other}"
            entry, = _wait_for(self.store, [key], deadline, "P2P peer's UCX address")
            self.store.delete_key(key)
            version, encoded = entry.split("|", 1)
            if version.split(".")[:2] != [str(major), str(minor)]:
                raise RuntimeError(f"UCX P2P versions differ: {major}.{minor} vs {version}")
            worker.connect(1, base64.b64decode(encoded))
            return _Connection(worker, sending)
        except BaseException:
            worker.close(timeout=0.1)
            raise

    def post(self, tensor, peer, tag, sending):
        key = peer, tag, sending
        with self.lock:
            if self.closed:
                raise RuntimeError("UCX point-to-point link was closed")
            connection = self.connections.get(key)
        if connection is None:
            connection = self._connect(peer, tag, sending)  # no global link lock while waiting
            with self.lock:
                if self.closed:
                    connection.close()
                    raise RuntimeError("UCX point-to-point link was closed while connecting")
                self.connections[key] = connection
        return connection.post(tensor)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            connections = list(self.connections.values())
            self.connections.clear()

        def cleanup():
            for connection in connections:
                connection.close()

        # A UCX worker cannot be closed concurrently with its native wait.
        # Retain it until that wait completes/errors; never block main teardown.
        threading.Thread(target=cleanup, name="gpubridge-p2p-ucx-close", daemon=True).start()
