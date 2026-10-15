"""``bridge="multitcp"``: the island leaders' bridge over several TCP connections per pair.

Opt-in (docs/RUST_TRANSPORT.md). The Rust library in :mod:`gpubridge.multitcp`
moves tagged byte ranges between leaders over ``GPUBRIDGE_MULTITCP_STREAMS``
connections per pair (4 by default), each with its own sending and receiving
thread; this module runs the collectives and the reductions, with torch ops,
like the UCX transport. It takes host tensors only (``device_tensors`` is
False), so GPU tensors reach it through the policies' usual staging and their
stream ordering is untouched.

- all_reduce: with two leaders, an exchange in chunks of
  ``GPUBRIDGE_MULTITCP_CHUNK_BYTES`` (4 MiB by default), spread over the
  connections; each chunk is reduced as soon as it has arrived and gone out,
  while the rest are still on the wire. Both leaders get the same bits, since
  addition, max and min are commutative. With more leaders, a ring
  (reduce-scatter, then all-gather), so every element is reduced in one fixed
  order and then copied.
- broadcast: the source leader sends to each of the others, in chunks.

The reductions run on one torch thread (the thread count is set for the collective and
restored after). On a node with few CPUs, torch's intra-op threads would otherwise
compete with this transport's network threads: on Explorer's 4-CPU jobs that made each
chunk's reduction 10 to 25 times slower than on one thread. Gloo's own bridge reduces
on one thread too.

The leaders connect on the address of ``GPUBRIDGE_MULTITCP_IFNAME``, else of
``GLOO_SOCKET_IFNAME``, else the one this machine reaches ``MASTER_ADDR`` from.
Every wait has the deadline ``init()`` gave its groups; a timeout or a failed
peer raises, and the transport refuses later calls. Before anything raises out of
a collective, the library's threads are stopped, so none of them can still write
into (or read from) the collective's tensors once the caller sees the error.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import socket
import struct
import sys
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from gpubridge import multitcp
from gpubridge.config import parse_size
from gpubridge.transport import BridgeTransport, register_transport, scoped_name
from gpubridge.ucx.transport import _REDUCE, _segments, _wait_for, tag

#: Chunk size of every message, e.g. ``4M`` or ``65536``.
CHUNK_BYTES_ENV = "GPUBRIDGE_MULTITCP_CHUNK_BYTES"
DEFAULT_CHUNK_BYTES = 4 * 2**20
#: TCP connections per pair of leaders.
STREAMS_ENV = "GPUBRIDGE_MULTITCP_STREAMS"
DEFAULT_STREAMS = 4
#: The interface whose address the leaders connect on (else GLOO_SOCKET_IFNAME's).
IFNAME_ENV = "GPUBRIDGE_MULTITCP_IFNAME"
#: The deadline when ``init()`` gave none: torch's default group timeout.
DEFAULT_TIMEOUT = timedelta(minutes=30)
#: The tag step of the connection handshake.
HANDSHAKE = 0xFFF

def chunk_bytes_setting() -> int:
    raw = os.environ.get(CHUNK_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_CHUNK_BYTES
    try:
        size = parse_size(raw)
    except ValueError as exc:
        raise ValueError(f"{CHUNK_BYTES_ENV}: {exc}") from None
    if size < 8 or size % 8:
        raise ValueError(f"{CHUNK_BYTES_ENV}={raw!r} must be a positive multiple of 8 bytes")
    return size


def streams_setting() -> int:
    raw = os.environ.get(STREAMS_ENV, "").strip()
    if not raw:
        return DEFAULT_STREAMS
    if not raw.isdigit() or not 1 <= int(raw) <= 64:
        raise ValueError(f"{STREAMS_ENV}={raw!r} must be a whole number from 1 to 64")
    return int(raw)


def _pieces(numel: int, chunk: int) -> list[tuple[int, int]]:
    """(start, length) in elements of each chunk of ``numel``; one empty piece for 0."""
    return [(start, min(chunk, numel - start)) for start in range(0, numel, chunk)] or [(0, 0)]


def _interface_address(ifname: str) -> str:
    """The IPv4 address of interface ``ifname`` (SIOCGIFADDR)."""
    request = 0xC0206921 if sys.platform == "darwin" else 0x8915
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            answer = fcntl.ioctl(probe.fileno(), request, struct.pack("256s", ifname[:15].encode()))
        except OSError as exc:
            raise RuntimeError(f"no IPv4 address on interface {ifname!r} for the multitcp "
                               f"bridge ({exc}); set {IFNAME_ENV}") from None
    return socket.inet_ntoa(answer[20:24])


def local_address() -> str:
    """The address this leader listens on: see the module docstring."""
    ifname = (os.environ.get(IFNAME_ENV) or os.environ.get("GLOO_SOCKET_IFNAME") or "").strip()
    if ifname:
        return _interface_address(ifname.split(",")[0])
    master = os.environ.get("MASTER_ADDR", "").strip()
    if master:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((master, 9))  # picks the route; sends nothing
            return probe.getsockname()[0]
    return socket.gethostbyname(socket.gethostname())


@register_transport
class MultiTcpTransport(BridgeTransport):
    """The bridge between island leaders over several TCP connections per pair."""

    name = "multitcp"
    reduce_ops = frozenset({"sum", "max", "min"})

    #: Links created in this process, the same count on every rank (creation is
    #: collective and in the same order everywhere), so store keys never collide.
    #: A subgroup's links, built by its members only, take scoped_name() instead.
    _links: ClassVar[itertools.count] = itertools.count()

    def __init__(self, worker: multitcp.Worker, members: Sequence[int], rank: int,
                 timeout: timedelta, chunk_bytes: int) -> None:
        self.worker = worker
        self.members = list(members)
        self.index = self.members.index(rank)
        self.timeout = timeout.total_seconds()
        self.chunk_bytes = chunk_bytes
        self._seq = 0

    # ---- creation ---------------------------------------------------------------------

    @classmethod
    def create(cls, ranks: Sequence[int], *, timeout: timedelta | None = None
               ) -> MultiTcpTransport | None:
        timeout = timeout or DEFAULT_TIMEOUT
        link = scoped_name(ranks) or next(cls._links)
        store = dist.distributed_c10d._get_default_store()
        key = f"gpubridge/multitcp/{link}"
        members, rank = list(ranks), dist.get_rank()
        deadline = time.monotonic() + timeout.total_seconds()
        transport, error = None, None
        if rank in members:
            try:
                transport = cls._connect(store, key, members, rank, timeout, deadline)
            except Exception as exc:  # noqa: BLE001 - reported to every rank below
                error = f"{type(exc).__name__}: {exc}"
            store.set(f"{key}/status/{rank}", "ok" if error is None else error)
        # Every rank, member or not, learns whether every member connected.
        statuses = _wait_for(store, [f"{key}/status/{m}" for m in members], deadline,
                             "every bridge member to report on its multitcp connections")
        failed = [(m, s) for m, s in zip(members, statuses, strict=True) if s != "ok"]
        if failed:
            if transport is not None:
                transport.close()
            detail = "\n".join(f"  rank {m}: {s}" for m, s in failed)
            raise RuntimeError(f"The multitcp bridge between ranks {members} could not be set "
                               f'up; use bridge="gloo" to run without it.\n{detail}')
        return transport

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> MultiTcpTransport:
        chunk, streams = chunk_bytes_setting(), streams_setting()
        index = members.index(rank)
        worker = multitcp.Worker(len(members), index, streams, local_address())
        try:
            # The chunk size and stream count decide which messages each step sends and
            # where: leaders that disagreed would wait for the wrong ones, so they are
            # checked like the protocol.
            settings = f"{multitcp.protocol()}|{chunk}|{streams}"
            store.set(f"{key}/addr/{rank}", f"{settings}|{worker.address()}")
            others = [m for m in members if m != rank]
            entries = _wait_for(store, [f"{key}/addr/{m}" for m in others], deadline,
                                "the other bridge members' multitcp addresses",
                                failures=[f"{key}/status/{m}" for m in others])
            for member, entry in zip(others, entries, strict=True):
                theirs, address = entry.rsplit("|", 1)
                if theirs != settings:
                    raise RuntimeError(
                        f"multitcp protocol|{CHUNK_BYTES_ENV}|{STREAMS_ENV} is {settings} here "
                        f"but {theirs} on rank {member}; use the same gpubridge and settings "
                        "on every leader")
                worker.connect(members.index(member), address,
                               max(0.0, deadline - time.monotonic()))
            transport = cls(worker, members, rank, timeout, chunk)
            transport._handshake(deadline)
            return transport
        except BaseException:
            worker.close(timeout=1.0)
            raise

    def _handshake(self, deadline: float) -> None:
        """One small message each way with every other member, so a link that can't
        carry data fails now, not in the first collective."""
        hello = torch.tensor([self.index], dtype=torch.int64)
        replies = {}
        ids = []
        for peer in self._peers():
            replies[peer] = torch.full((1,), -1, dtype=torch.int64)
            ids.append(self.worker.recv(peer, replies[peer].data_ptr(), 8,
                                        tag(0, HANDSHAKE, peer, 0)))
            ids.append(self.worker.send(peer, hello.data_ptr(), 8,
                                        tag(0, HANDSHAKE, self.index, 0)))
        self.worker.wait(ids, deadline)
        for peer, reply in replies.items():
            if int(reply.item()) != peer:
                raise RuntimeError(f"multitcp handshake with bridge member {peer} returned "
                                   f"{int(reply.item())}")

    # ---- collectives ------------------------------------------------------------------

    def _peers(self) -> list[int]:
        return [i for i in range(len(self.members)) if i != self.index]

    @staticmethod
    def _check(tensor: torch.Tensor) -> None:
        if not tensor.is_contiguous() or tensor.device.type != "cpu":
            raise ValueError("the multitcp bridge carries contiguous host tensors")

    def _start(self, tensor: torch.Tensor) -> tuple[int, float, int]:
        self._seq += 1
        chunk = max(1, self.chunk_bytes // tensor.element_size())
        return self._seq, time.monotonic() + self.timeout, chunk

    def _send(self, peer: int, view: torch.Tensor, seq: int, step: int, chunk: int) -> list[int]:
        size = view.element_size()
        return [self.worker.send(peer, view.data_ptr() + start * size, length * size,
                                 tag(seq, step, self.index, i))
                for i, (start, length) in enumerate(_pieces(view.numel(), chunk))]

    def _recv(self, peer: int, view: torch.Tensor, seq: int, step: int, chunk: int) -> list[int]:
        size = view.element_size()
        return [self.worker.recv(peer, view.data_ptr() + start * size, length * size,
                                 tag(seq, step, peer, i))
                for i, (start, length) in enumerate(_pieces(view.numel(), chunk))]

    def _guarded(self, run) -> None:
        """``run()`` on one torch thread (see the module docstring); if anything raises,
        stop the library's threads before it propagates, so no posted operation outlives
        the tensors it points into."""
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            run()
        except BaseException:
            self.worker.abort()
            raise
        finally:
            torch.set_num_threads(threads)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)  # a bad argument fails the call, not the bridge
        self._guarded(lambda: self._all_reduce(tensor, op))

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp.RedOpType) -> None:
        seq, deadline, chunk = self._start(tensor)
        size = len(self.members)
        reduce = _REDUCE[op]
        flat = tensor.view(-1)
        if size == 2:
            (peer,) = self._peers()
            scratch = torch.empty_like(flat)
            received = self._recv(peer, scratch, seq, 0, chunk)
            sent = self._send(peer, flat, seq, 0, chunk)
            # Chunk i is reduced once it has arrived and this side's copy of it has gone
            # out (the send reads the same memory), while later chunks are on the wire.
            for (start, length), got, out in zip(_pieces(flat.numel(), chunk), received, sent,
                                                 strict=True):
                self.worker.wait([got, out], deadline)
                reduce(flat[start:start + length], scratch[start:start + length])
            return
        right, left = (self.index + 1) % size, (self.index - 1) % size
        segments = _segments(flat.numel(), size)
        scratch = torch.empty(max(length for _, length in segments), dtype=flat.dtype)

        def segment(i: int) -> torch.Tensor:
            start, length = segments[i % size]
            return flat[start:start + length]

        for step in range(size - 1):  # reduce-scatter
            incoming = segment(self.index - step - 1)
            buffer = scratch[:incoming.numel()]
            ids = self._recv(left, buffer, seq, step, chunk) + self._send(
                right, segment(self.index - step), seq, step, chunk)
            self.worker.wait(ids, deadline)
            reduce(incoming, buffer)
        for step in range(size - 1):  # all-gather: this member now owns segment index + 1
            ids = self._recv(left, segment(self.index - step), seq, size + step, chunk) + \
                self._send(right, segment(self.index + 1 - step), seq, size + step, chunk)
            self.worker.wait(ids, deadline)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if len(self.members) == 1:
            return
        self._check(tensor)
        self._guarded(lambda: self._broadcast(tensor, src))

    def _broadcast(self, tensor: torch.Tensor, src: int) -> None:
        seq, deadline, chunk = self._start(tensor)
        root = self.members.index(src)
        flat = tensor.view(-1)
        if self.index == root:
            ids = [i for peer in self._peers() for i in self._send(peer, flat, seq, 0, chunk)]
        else:
            ids = self._recv(root, flat, seq, 0, chunk)
        self.worker.wait(ids, deadline)

    def close(self) -> None:
        self.worker.close(timeout=min(10.0, self.timeout))
