"""Transport observations for the straggler detector's NIC diagnosis.

Opt-in: ``GPUBRIDGE_VERBS_OBSERVE=<file>`` makes each verbs bridge leader append
one JSON line per record to ``<file>`` (``{rank}`` in the name is replaced by the
global rank). The records follow the detector's
``straggler.transport_observation/1`` (straggler-detector ``docs/NIC_TELEMETRY.md``),
with the top-level ``host`` (this node's hostname) and ``job`` (the run) and the
NIC's ``pci_bus_id`` every record carries.

**Rate.** One record per collective call, per peer and direction (``send``,
``recv``): the bytes moved with that peer in that call and the time from the
first post to the last completion. A failed or timed-out call gives one record
per peer and direction it involved, with its status. Nothing per chunk or
completion.

**Off the critical path, bounded.** The collective only appends a dict to a
bounded queue (``GPUBRIDGE_VERBS_OBSERVE_QUEUE`` records, 10000 by default); a
background thread serialises and writes. When the queue is full, new records are
dropped and counted (``dropped`` in the transport's telemetry). A write error
stops the writer and is reported the same way; it never fails a collective.

**Never keys.** Records carry no remote or local keys (none exist outside the
library's write grants) and no raw GIDs; :func:`check_record` refuses any record
with a key-like name at any depth, so a mistake can't leak one.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from collections import deque
from pathlib import Path

SCHEMA = "straggler.transport_observation/1"
OBSERVE_ENV = "GPUBRIDGE_VERBS_OBSERVE"
QUEUE_ENV = "GPUBRIDGE_VERBS_OBSERVE_QUEUE"
#: The run every record names; else SLURM's or torchelastic's id; else one the bridge
#: members agree on through the c10d store.
JOB_ENV = "GPUBRIDGE_JOB_ID"
JOB_FALLBACK_ENVS = ("SLURM_JOB_ID", "TORCHELASTIC_RUN_ID")
DEFAULT_QUEUE = 10_000
SYSFS = Path("/sys/class/infiniband")

#: Names refused at any depth (the detector refuses them too).
_KEYLIKE = re.compile(r"(^|_)(rkey|lkey|mkey|qkey|remote_key|key)$", re.IGNORECASE)


def check_record(record: object, path: str = "") -> None:
    """Raise ValueError if ``record`` has a key-like field name anywhere."""
    if isinstance(record, dict):
        for name, value in record.items():
            if _KEYLIKE.search(str(name)):
                raise ValueError(f"observation field {path}{name} looks like a key")
            check_record(value, f"{path}{name}.")
    elif isinstance(record, list):
        for item in record:
            check_record(item, path)


def job_id(store=None, key: str = "gpubridge/verbs/job") -> str:
    """The run's id: ``GPUBRIDGE_JOB_ID``, ``SLURM_JOB_ID``, ``TORCHELASTIC_RUN_ID``, or
    one agreed through ``store`` (the first rank to ask sets it)."""
    for env in (JOB_ENV, *JOB_FALLBACK_ENVS):
        value = os.environ.get(env, "").strip()
        if value:
            return value
    if store is not None:
        mine = uuid.uuid4().hex[:16]
        try:
            agreed = store.compare_set(key, "", mine)
            return agreed.decode() if isinstance(agreed, bytes) else str(agreed)
        except Exception:  # noqa: BLE001 - a store without compare_set
            pass
    return "unknown"


def link_layer(text: str | None) -> str | None:
    """The schema's ``InfiniBand``/``Ethernet`` from the library's link layer text."""
    if not text:
        return None
    if text.startswith("InfiniBand"):
        return "InfiniBand"
    if text.startswith("Ethernet"):
        return "Ethernet"
    return None


def nic_sysfs(device: str | None, port: int | None, root: Path = SYSFS) -> dict:
    """``pci_bus_id`` and ``rate_gbps`` of a device's port from sysfs; None where absent."""
    out: dict = {"pci_bus_id": None, "rate_gbps": None}
    if not device:
        return out
    link = root / device / "device"
    if link.exists():
        out["pci_bus_id"] = Path(os.path.realpath(link)).name or None
    try:
        rate = (root / device / "ports" / str(port) / "rate").read_text().split()
        out["rate_gbps"] = float(rate[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def nic_object(telemetry: dict, root: Path = SYSFS) -> dict:
    """The record's ``nic``: the endpoint the library chose, from its telemetry."""
    device, port = telemetry.get("device"), telemetry.get("port")
    return {"device": device, "port": port, "gid_index": telemetry.get("gid_index"),
            "link_layer": link_layer(telemetry.get("link_layer")), **nic_sysfs(device, port, root)}


def peer_nic(handshake: str) -> dict:
    """The record's ``peer_nic``: what the peer advertised in its handshake line."""
    fields = dict(p.split("=", 1) for p in handshake.split(";")[1:] if "=" in p)
    layer = {"ib": "InfiniBand", "eth": "Ethernet"}.get(fields.get("ll", ""))
    port = fields.get("port")
    return {"device": fields.get("dev"), "port": int(port) if port and port.isdigit() else None,
            "link_layer": layer}


class Emitter:
    """Appends records to a JSONL file from a background thread; see the module docs."""

    def __init__(self, path: str, limit: int = DEFAULT_QUEUE) -> None:
        self.path = path
        self.limit = max(1, limit)
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self.emitted = 0
        self.dropped = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="gpubridge-verbs-observe",
                                        daemon=True)
        self._thread.start()

    def emit(self, record: dict) -> None:
        """Queue ``record`` (never blocks; drops and counts when full or stopped)."""
        check_record(record)
        with self._lock:
            if self._stop or self.error is not None or len(self._queue) >= self.limit:
                self.dropped += 1
                return
            self._queue.append(record)
        self._wake.set()

    def _run(self) -> None:
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as out:
                while True:
                    self._wake.wait(0.5)
                    self._wake.clear()
                    with self._lock:
                        batch = list(self._queue)
                        self._queue.clear()
                        stop = self._stop
                    if batch:
                        out.write("".join(json.dumps(r, separators=(",", ":")) + "\n"
                                          for r in batch))
                        out.flush()
                        self.emitted += len(batch)
                    if stop:
                        return
        except OSError as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.dropped += len(self._queue)
                self._queue.clear()

    def close(self, timeout: float = 5.0) -> None:
        """Write what is queued and stop."""
        with self._lock:
            self._stop = True
        self._wake.set()
        self._thread.join(timeout)

    def stats(self) -> dict:
        return {"path": self.path, "emitted": self.emitted, "dropped": self.dropped,
                "queued": len(self._queue), "error": self.error,
                "rate": "one record per collective call, per peer and direction"}


def from_env(rank: int) -> Emitter | None:
    """The emitter ``GPUBRIDGE_VERBS_OBSERVE`` asks for, or None."""
    path = os.environ.get(OBSERVE_ENV, "").strip()
    if not path:
        return None
    raw = os.environ.get(QUEUE_ENV, "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_QUEUE
    except ValueError:
        raise ValueError(f"{QUEUE_ENV}={raw!r} must be a whole number") from None
    return Emitter(path.replace("{rank}", str(rank)), limit)


class CallLog:
    """What one collective call moved, per peer and direction, for its records."""

    def __init__(self) -> None:
        self.start: dict[tuple[int, str], float] = {}
        self.bytes: dict[tuple[int, str], int] = {}
        self.ops: dict[int, tuple[int, str]] = {}
        self.done: dict[tuple[int, str], float] = {}
        #: Bytes of those that went through a host bounce buffer (direct path only).
        self.bounce: dict[tuple[int, str], int] = {}
        self.outcomes: dict[int, dict] = {}
        self.post_errors: dict[tuple[int, str], str] = {}

    def failed_post(self, peer: int, direction: str, nbytes: int, exc: Exception) -> None:
        key = (peer, direction)
        self.start.setdefault(key, time.monotonic())
        self.bytes[key] = self.bytes.get(key, 0) + nbytes
        self.post_errors[key] = f"{type(exc).__name__}: {exc}"[:500]

    def posted(self, op: int, peer: int, direction: str, nbytes: int) -> None:
        key = (peer, direction)
        self.start.setdefault(key, time.monotonic())
        self.bytes[key] = self.bytes.get(key, 0) + nbytes
        self.ops[op] = key

    def bounced(self, op: int, nbytes: int) -> None:
        key = self.ops[op]
        self.bounce[key] = self.bounce.get(key, 0) + nbytes

    def completed(self, ops) -> None:
        now = time.monotonic()
        for op in ops:
            key = self.ops.get(op)
            if key is not None:
                self.done[key] = now
                self.outcomes[op] = {"status": "ok"}

    def records(self, base: dict, status: str, error: str | None, members: list[int],
                peer_nics: dict[int, dict]) -> list[dict]:
        now_wall, now_mono = time.time(), time.monotonic()
        out = []
        for (peer, direction), start in sorted(self.start.items()):
            key = (peer, direction)
            outcomes = [self.outcomes.get(op, {"status": status, "error": error})
                        for op, group in self.ops.items() if group == key]
            failed = next((o for o in outcomes if o["status"] == "error"), None)
            if key in self.post_errors:
                result = {"status": "error", "error": self.post_errors[key]}
            elif failed is not None:
                result = failed
            elif outcomes and all(o["status"] == "ok" for o in outcomes):
                result = {"status": "ok"}
            else:
                result = next((o for o in outcomes if o["status"] != "ok"),
                              {"status": status, "error": error})
            result_status = result["status"]
            end = self.done.get(key, now_mono) if result_status == "ok" else now_mono
            record = {**base, "t_wall": now_wall - (now_mono - end), "t_mono": end,
                      "peer": members[peer], "peer_nic": peer_nics.get(peer),
                      "op": direction, "bytes": self.bytes[(peer, direction)],
                      "duration_s": max(0.0, end - start), "status": result_status}
            if (peer, direction) in self.bounce:
                record["bounced_bytes"] = self.bounce[(peer, direction)]
            if result.get("error") is not None and result_status != "ok":
                record["error"] = result["error"][:500]
            if result.get("wc_status") is not None:
                record["wc_status"] = result["wc_status"]
            out.append(record)
        return out


def hostname() -> str:
    return socket.gethostname()
