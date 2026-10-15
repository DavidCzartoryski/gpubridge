"""CPU scheduling model for the GPU P2P check, not GPU validation.

Model stream queues and stream-local allocator reuse, then inject the backend
faults the check claims to detect. Actual CUDA/HIP behavior still needs GPU runs.
"""

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
import gpukit  # noqa: E402


class Stream:
    def __init__(self, device=None):
        self.queued = []

    def wait_stream(self, other):
        self.queued.append(other.flush)

    def flush(self):
        while self.queued:
            self.queued.pop(0)()


class Allocator:
    """One producer stream's reusable buffers; live tensors own their blocks."""

    def __init__(self):
        self.free = []
        self.current = Stream()
        self.streams = [self.current]

    def tensor(self, shape, dtype):
        for index, block in enumerate(self.free):
            if block.shape == shape and block.dtype == dtype:
                return Tensor(self, self.free.pop(index))
        return Tensor(self, torch.empty(shape, dtype=dtype))

    def full(self, shape, value, *, device=None, dtype):
        tensor = self.tensor(shape, dtype)
        block = tensor.block
        self.current.queued.append(lambda: block.fill_(value))
        return tensor

    def full_like(self, tensor, value):
        return self.full(tensor.block.shape, value, dtype=tensor.block.dtype)

    def empty_like(self, tensor):
        return self.tensor(tensor.block.shape, tensor.block.dtype)

    def stream(self, device):
        stream = Stream()
        self.streams.append(stream)
        return stream

    @contextlib.contextmanager
    def use(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def sync(self, device):
        for stream in self.streams:
            stream.flush()


class Tensor:
    def __init__(self, allocator, block):
        self.allocator, self.block = allocator, block

    def __del__(self):
        self.allocator.free.append(self.block)

    def copy_(self, source):
        target = self.block
        if isinstance(source, Tensor):
            block = source.block
            self.allocator.current.queued.append(lambda: target.copy_(block))
        else:
            # Pageable H2D copy blocks the host, including prior GPU delay.
            self.allocator.current.flush()
            target.copy_(source)
        return self

    def clone(self):
        clone = self.allocator.empty_like(self)
        return clone.copy_(self)

    def cpu(self):
        self.allocator.current.flush()
        return self.block.clone()


@pytest.mark.parametrize("fault", [None, "ignores-producer-stream", "releases-send-storage"])
def test_p2p_check_detects_stream_and_storage_faults_without_a_gpu(monkeypatch, fault):
    allocator = Allocator()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Stream=allocator.stream, stream=allocator.use,
                             current_stream=lambda device: allocator.streams[0],
                             synchronize=allocator.sync),
        full=allocator.full, full_like=allocator.full_like, empty_like=allocator.empty_like,
        arange=torch.arange, equal=torch.equal,
        **{name: getattr(torch, name) for name in
           ("int8", "int64", "float16", "bfloat16", "float32", "float64")})
    monkeypatch.setattr(gpukit, "torch", fake_torch)
    monkeypatch.setattr(gpukit, "queue_gpu_delay",
                        lambda *args: allocator.current.queued.append(lambda: None))
    fake_dist = SimpleNamespace(irecv="recv", isend="send", barrier=lambda: None,
                               P2POp=lambda op, tensor, peer, tag: (op, tensor))
    monkeypatch.setattr(gpukit, "dist", fake_dist)

    def batch(operations):
        incoming, outgoing = operations[0][1], operations[1][1]
        producer = allocator.current
        block, receive = outgoing.block, incoming.block
        early = block.clone() if fault == "ignores-producer-stream" else None
        # Work owns outgoing unless this transport has the lifetime fault.
        keepalive = outgoing if fault != "releases-send-storage" else None

        def wait():
            producer.flush()
            source = early if early is not None else block
            # Model the ring neighbor's corresponding data (rank 1 versus 0).
            receive.copy_((source + 8 + 3).remainder(17) - 8)
        return [SimpleNamespace(wait=wait, storage_owner=keepalive)]

    monkeypatch.setattr(gpukit.gpubridge, "batch_isend_irecv", batch)
    topology = SimpleNamespace(rank=0, world_size=2, device=SimpleNamespace(type="cuda"),
                               layout=SimpleNamespace(island_of=lambda rank: 0))
    if fault is None:
        cases = gpukit.p2p_checks(topology, numels=(31,))
        assert len(cases) == 6 and all(case["ok"] for case in cases)
    else:
        with pytest.raises(AssertionError, match="P2P stream/lifetime check"):
            gpukit.p2p_checks(topology, numels=(31,))
