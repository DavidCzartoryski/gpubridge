"""``bridge="verbs"``: the island leaders' bridge over RDMA reliable connections.

Opt-in (docs/VERBS_TRANSPORT.md). The Rust library in :mod:`gpubridge.verbs`
holds an RC queue pair to every other leader on the device, port and GID it
selects, and moves tagged messages with RC SEND/RECV through its own registered
buffers, with credits and a completion thread (the protocol is the doc's "Wire
protocol"). The collectives and reductions are :class:`gpubridge.tagged.TaggedBridge`'s,
the same code as ``bridge="multitcp"``, in chunks of ``GPUBRIDGE_VERBS_CHUNK_BYTES``
(4 MiB by default). It takes host tensors only (``device_tensors`` is False), so GPU
tensors reach it through the policies' usual staging; GPU memory paths are phase A3.

The leaders exchange addresses and queue pair numbers through the c10d store and
check they agree (protocol version, link layer, settings) before connecting. If
verbs can't run here (no library, no device, no usable port, a mismatch), setting
up the bridge fails on every rank and says why: payload never moves any other way,
and another bridge is only ever chosen explicitly. Every wait has the deadline
``init()`` gave its groups; a timeout, a failed completion or a lost peer raises,
and the transport refuses later calls.

Not validated on RDMA hardware or GPUs. ``GPUBRIDGE_VERBS_TEST_FABRIC`` runs it on
a MOCK fabric for tests (see :mod:`gpubridge.verbs`).
"""

from __future__ import annotations

import itertools
import os
import time
from datetime import timedelta
from typing import ClassVar

from gpubridge import verbs
from gpubridge.config import parse_size
from gpubridge.tagged import TaggedBridge
from gpubridge.transport import register_transport
from gpubridge.ucx.transport import _wait_for

#: Chunk size of the collectives' messages, e.g. ``4M`` or ``65536``.
CHUNK_BYTES_ENV = "GPUBRIDGE_VERBS_CHUNK_BYTES"
DEFAULT_CHUNK_BYTES = 4 * 2**20
#: The RDMA device (``mlx5_0``), port and GID index; automatic where unset.
DEVICE_ENV = "GPUBRIDGE_VERBS_DEVICE"
PORT_ENV = "GPUBRIDGE_VERBS_PORT"
GID_INDEX_ENV = "GPUBRIDGE_VERBS_GID_INDEX"
#: The protocol's slot size (bytes per RC message) and credits (receives per peer).
SLOT_BYTES_ENV = "GPUBRIDGE_VERBS_SLOT_BYTES"
CREDITS_ENV = "GPUBRIDGE_VERBS_CREDITS"
SEND_SLOTS_ENV = "GPUBRIDGE_VERBS_SEND_SLOTS"
#: Bytes of arrived, not yet received messages a leader buffers per bridge.
UNEXPECTED_BYTES_ENV = "GPUBRIDGE_VERBS_UNEXPECTED_BYTES"


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


def _number(env: str, size: bool = False) -> int | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    try:
        return parse_size(raw) if size else int(raw)
    except ValueError:
        raise ValueError(f"{env}={raw!r} must be a {'size' if size else 'whole number'}") \
            from None


def endpoint_settings() -> dict[str, object]:
    """The library's settings from the environment (unset ones: its defaults)."""
    return {
        "device": os.environ.get(DEVICE_ENV, "").strip() or None,
        "port": _number(PORT_ENV),
        "gid_index": _number(GID_INDEX_ENV),
        "slot_bytes": _number(SLOT_BYTES_ENV, size=True),
        "credits": _number(CREDITS_ENV),
        "send_slots": _number(SEND_SLOTS_ENV),
        "unexpected_bytes": _number(UNEXPECTED_BYTES_ENV, size=True),
    }


@register_transport
class VerbsTransport(TaggedBridge):
    """The bridge between island leaders over RDMA reliable connections (raw verbs)."""

    name = "verbs"
    _links: ClassVar[itertools.count] = itertools.count()
    _instead = ("verbs is never replaced by another transport on its own; choose one "
                'explicitly (bridge="gloo", say) to run without it')

    @classmethod
    def _connect(cls, store, key: str, members: list[int], rank: int, timeout: timedelta,
                 deadline: float) -> VerbsTransport:
        chunk, settings = chunk_bytes_setting(), endpoint_settings()
        index = members.index(rank)
        endpoint = verbs.Endpoint(len(members), index, settings)
        try:
            if endpoint.telemetry().get("mock"):
                verbs.log.warning("gpubridge: bridge=\"verbs\" is running on the MOCK test "
                                  "fabric (%s=%s): not RDMA", verbs.TEST_FABRIC_ENV,
                                  verbs.test_fabric_dir())
            # The chunk size decides which messages each step sends: leaders that
            # disagreed would wait for the wrong ones, so it is checked like the protocol
            # (the library checks its own settings in the handshake line).
            agreed = f"{verbs.protocol(endpoint._lib)}|{chunk}"
            store.set(f"{key}/addr/{rank}", f"{agreed}|{endpoint.local_info()}")
            others = [m for m in members if m != rank]
            entries = _wait_for(store, [f"{key}/addr/{m}" for m in others], deadline,
                                "the other bridge members' verbs addresses",
                                failures=[f"{key}/status/{m}" for m in others])
            for member, entry in zip(others, entries, strict=True):
                protocol, theirs_chunk, info = entry.split("|", 2)
                theirs = f"{protocol}|{theirs_chunk}"
                if theirs != agreed:
                    raise RuntimeError(
                        f"verbs protocol|{CHUNK_BYTES_ENV} is {agreed} here but {theirs} on rank "
                        f"{member}; use the same gpubridge and settings on every leader")
                endpoint.connect(members.index(member), info)
            # Nothing is sent until every member's queue pairs are ready to receive: a
            # message reaching one still in Init is dropped, and its sender's retries
            # may run out before that peer connects.
            store.set(f"{key}/ready/{rank}", "ok")
            _wait_for(store, [f"{key}/ready/{m}" for m in others], deadline,
                      "the other bridge members' queue pairs to be ready",
                      failures=[f"{key}/status/{m}" for m in others])
            transport = cls(endpoint, members, rank, timeout, chunk)
            transport._handshake(deadline)
            return transport
        except BaseException:
            endpoint.abort("setup failed")
            endpoint.close(timeout=1.0)
            raise

    def telemetry(self) -> dict:
        """The endpoint's state for telemetry correlation: provider (MOCK or libibverbs
        and its file), device, port, link layer, GID, MTUs, protocol version, ``path``
        (``host``: A2 carries host memory only), failure, operation and byte counts, and
        per peer its queue pair and state. No remote keys exist in this protocol."""
        return {**self.worker.telemetry(), "chunk_bytes": self.chunk_bytes,
                "failed": self.worker.failed}

    @property
    def state(self) -> str:
        """``ok``, ``failed`` or ``closed``."""
        if self.worker.failed is not None:
            return "failed"
        return self.worker.telemetry().get("state", "closed")

    def close(self) -> None:
        start = time.monotonic()
        self.worker.close(timeout=min(10.0, self.timeout))
        verbs.log.debug("gpubridge: verbs bridge closed in %.3f s", time.monotonic() - start)
