"""The Rust library behind ``bridge="multitcp"`` (docs/RUST_TRANSPORT.md): build, load, call.

``crate/`` is a small Rust library (``gpubridge-transport``, the standard library
only) with a C interface: tagged messages between bridge members over several TCP
connections per pair. Nothing is compiled at install time. ``python -m
gpubridge.multitcp build`` compiles it with cargo, and :func:`load` does on first
use, into ``GPUBRIDGE_MULTITCP_LIB_DIR`` (default ``~/.cache/gpubridge``).
``GPUBRIDGE_MULTITCP_LIB`` points at a library built elsewhere instead, for nodes
without a Rust toolchain. Without either, :class:`Unavailable` says what is
missing; the rest of gpubridge never needs this module.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path

CRATE = Path(__file__).with_name("crate")
#: Where the compiled library goes (``~/.cache/gpubridge`` by default).
LIB_DIR_ENV = "GPUBRIDGE_MULTITCP_LIB_DIR"
#: A library built elsewhere (``python -m gpubridge.multitcp build`` prints its path).
LIB_ENV = "GPUBRIDGE_MULTITCP_LIB"

TIMED_OUT, PEER_FAILED = -2, -3

_lock = threading.Lock()
_library: ctypes.CDLL | None = None
log = logging.getLogger("gpubridge")


class Unavailable(RuntimeError):
    """The library can't be built or loaded here; the message says why."""


class TransportError(RuntimeError):
    """A send, receive or connection failed or timed out."""


def _sources() -> list[Path]:
    return sorted([CRATE / "Cargo.toml", CRATE / "Cargo.lock", *(CRATE / "src").rglob("*.rs")])


def _cargo() -> str | None:
    return os.environ.get("CARGO") or shutil.which("cargo")


def _suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def library_path() -> Path:
    """Where the library for this source, toolchain and machine lives once built.

    The cache is often a home directory shared by every node of a cluster, and the two
    sides of a mixed job may be different machines: anything that could change the
    result is part of the name.
    """
    cargo = _cargo()
    rustc = shutil.which("rustc")
    toolchain = subprocess.run([rustc, "--version"], capture_output=True, text=True).stdout \
        if rustc else (cargo or "")
    parts = [*(p.read_bytes() for p in _sources()), toolchain.encode(),
             platform.machine().encode(), " ".join(platform.libc_ver()).encode(),
             sys.platform.encode()]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    base = Path(os.environ.get(LIB_DIR_ENV) or Path.home() / ".cache" / "gpubridge")
    return base / f"libgpubridge_multitcp-{key[:16]}{_suffix()}"


def build(*, force: bool = False) -> Path:
    """Compile the crate with cargo (once per source and toolchain); return the library.

    Raises:
        Unavailable: without cargo, or if compiling fails.
    """
    cargo = _cargo()
    if cargo is None:
        raise Unavailable("no cargo (a Rust toolchain, https://rustup.rs) to build the "
                          f"multitcp bridge; or set {LIB_ENV} to a library built elsewhere")
    path = library_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gpubridge-multitcp-", dir=path.parent) as work:
        # A copy of the crate: cargo writes next to Cargo.toml, and an installed package
        # may be read-only.
        crate = Path(work) / "crate"
        shutil.copytree(CRATE, crate, ignore=shutil.ignore_patterns("target"))
        command = [cargo, "build", "--release", "--offline", "--quiet", "--manifest-path",
                   str(crate / "Cargo.toml"), "--target-dir", str(Path(work) / "target")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Unavailable("building the multitcp bridge failed:\n" + " ".join(command)
                              + "\n" + result.stderr[-4000:])
        built = Path(work) / "target" / "release" / f"libgpubridge_transport{_suffix()}"
        # A unique name, then an atomic rename: ranks building at once never see half a file.
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        shutil.copyfile(built, name)
        Path(name).replace(path)
    return path


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.gb_tcp_protocol.argtypes, lib.gb_tcp_protocol.restype = [], c.c_uint32
    lib.gb_tcp_create.argtypes = [c.c_int, c.c_int, c.c_int, c.c_char_p, c.c_char_p,
                                  c.c_size_t]
    lib.gb_tcp_create.restype = c.c_void_p
    lib.gb_tcp_address.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_tcp_connect.argtypes = [c.c_void_p, c.c_int, c.c_char_p, c.c_double]
    lib.gb_tcp_send.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_tcp_send.restype = c.c_int64
    lib.gb_tcp_recv.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_tcp_recv.restype = c.c_int64
    lib.gb_tcp_wait.argtypes = [c.c_void_p, c.POINTER(c.c_int64), c.c_int, c.c_double]
    lib.gb_tcp_last_error.argtypes, lib.gb_tcp_last_error.restype = [c.c_void_p], c.c_char_p
    lib.gb_tcp_close.argtypes = [c.c_void_p, c.c_double]
    for name in ("gb_tcp_address", "gb_tcp_connect", "gb_tcp_wait", "gb_tcp_close"):
        getattr(lib, name).restype = c.c_int
    return lib


def load(*, build_if_missing: bool = True) -> ctypes.CDLL:
    """The library: ``GPUBRIDGE_MULTITCP_LIB`` if set, else this source's build (made first
    if needed).

    Raises:
        Unavailable: if it isn't there and can't be built (see :func:`build`).
    """
    global _library
    with _lock:
        if _library is None:
            given = os.environ.get(LIB_ENV, "").strip()
            if given:
                path = Path(given)
            else:
                path = build() if build_if_missing else library_path()
            if not path.exists():
                raise Unavailable(f"no multitcp bridge library at {path}; run python -m "
                                  f"gpubridge.multitcp build, or set {LIB_ENV}")
            try:
                _library = _declare(ctypes.CDLL(str(path)))
            except OSError as exc:
                raise Unavailable(f"loading {path} failed: {exc}") from None
        return _library


def protocol() -> int:
    """The wire protocol version this library speaks; every leader's must match."""
    return int(load().gb_tcp_protocol())


class Worker:
    """One bridge member's end: a listening socket and ``streams`` connections to every
    other member. Not thread-safe."""

    def __init__(self, members: int, index: int, streams: int, host: str) -> None:
        self._lib = load()
        error = ctypes.create_string_buffer(512)
        handle = self._lib.gb_tcp_create(members, index, streams, host.encode(), error,
                                         len(error))
        if not handle:
            raise TransportError(error.value.decode())
        self._handle: int | None = handle
        #: The first error, after which every call raises: a timed-out collective may
        #: have left messages half sent or received.
        self.failed: str | None = None

    def _check(self, result: int) -> int:
        if result < 0:
            text = (self._lib.gb_tcp_last_error(self._handle) or b"").decode()
            kind = {TIMED_OUT: "timed out", PEER_FAILED: "peer failed"}.get(result, "failed")
            self.failed = f"multitcp bridge {kind}: {text}"
            raise TransportError(self.failed)
        return result

    def _usable(self) -> None:
        if self.failed is not None:
            raise TransportError(f"the multitcp bridge already failed ({self.failed})")

    def address(self) -> str:
        """``host:port``, for the other members' :meth:`connect`."""
        out = ctypes.create_string_buffer(256)
        self._check(self._lib.gb_tcp_address(self._handle, out, len(out)))
        return out.value.decode()

    def connect(self, member: int, address: str, timeout: float) -> None:
        self._usable()
        self._check(self._lib.gb_tcp_connect(self._handle, member, address.encode(), timeout))

    def send(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        self._usable()
        return self._check(self._lib.gb_tcp_send(self._handle, member, pointer, nbytes, tag))

    def recv(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        self._usable()
        return self._check(self._lib.gb_tcp_recv(self._handle, member, pointer, nbytes, tag))

    def wait(self, ids: Sequence[int], deadline: float) -> None:
        """Wait for operations ``ids`` until ``deadline`` (``time.monotonic()``)."""
        self._usable()
        if not ids:
            return
        array = (ctypes.c_int64 * len(ids))(*ids)
        timeout = max(0.0, deadline - time.monotonic())
        self._check(self._lib.gb_tcp_wait(self._handle, array, len(ids), timeout))

    def close(self, timeout: float = 10.0) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            if self._lib.gb_tcp_close(handle, timeout) != 0:
                log.warning("gpubridge: the multitcp bridge's threads didn't stop within %g s",
                            timeout)
