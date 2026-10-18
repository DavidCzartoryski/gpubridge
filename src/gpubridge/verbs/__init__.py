"""The Rust library behind ``bridge="verbs"`` (docs/VERBS_TRANSPORT.md): build, load, call.

``crate/`` is ``gpubridge-verbs``: RC queue pairs over libibverbs, through a C shim
compiled against rdma-core's headers, with the bridge's wire protocol and a
completion thread. Nothing is compiled at install time. ``python -m
gpubridge.verbs build`` compiles it with cargo on Linux (its ``ibverbs`` feature
needs rdma-core's headers: the development package, or
``GPUBRIDGE_VERBS_INCLUDE_DIR``), and :func:`load` does on first use, into
``GPUBRIDGE_VERBS_LIB_DIR`` (default ``~/.cache/gpubridge``).
``GPUBRIDGE_VERBS_LIB`` points at a library built elsewhere instead. Without
either, :class:`Unavailable` says what is missing; the rest of gpubridge never
needs this module.

**The test fabric.** ``GPUBRIDGE_VERBS_TEST_FABRIC=<directory>`` makes the bridge
use a MOCK fabric over Unix sockets between processes on one machine, so the
collective tests can run ``bridge="verbs"`` without RDMA hardware. It lives only in
a separate test build of the library (``build(test_fabric=True)``, with the
crate's ``test-fabric`` feature and without libibverbs), which only that setting
loads; the production library can't select it, and telemetry says MOCK. It is not
RDMA and proves nothing about a fabric.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

CRATE = Path(__file__).with_name("crate")
#: Where the compiled library goes (``~/.cache/gpubridge`` by default).
LIB_DIR_ENV = "GPUBRIDGE_VERBS_LIB_DIR"
#: A library built elsewhere (``python -m gpubridge.verbs build`` prints its path).
LIB_ENV = "GPUBRIDGE_VERBS_LIB"
#: A directory: run on the MOCK test fabric there (tests only; see the module docstring).
TEST_FABRIC_ENV = "GPUBRIDGE_VERBS_TEST_FABRIC"
#: Passed to the build (see docs/VERBS_TRANSPORT.md, "Building").
BUILD_ENVS = ("GPUBRIDGE_VERBS_INCLUDE_DIR", "GPUBRIDGE_VERBS_LIB_DIR", "CC", "AR")

TIMED_OUT, PEER_FAILED = -2, -3

_lock = threading.Lock()
_libraries: dict[bool, ctypes.CDLL] = {}
log = logging.getLogger("gpubridge")


class Unavailable(RuntimeError):
    """The library can't be built or loaded here; the message says why."""


class TransportError(RuntimeError):
    """A send, receive or connection failed or timed out."""


def _sources() -> list[Path]:
    return sorted([CRATE / "Cargo.toml", CRATE / "Cargo.lock", CRATE / "build.rs",
                   *(CRATE / "src").rglob("*.rs"), *(CRATE / "shim").glob("*.[ch]")])


def _cargo() -> str | None:
    return os.environ.get("CARGO") or shutil.which("cargo")


def _suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def features(test_fabric: bool) -> list[str]:
    """The crate features a build has: ``ibverbs`` for real use (Linux only), or only
    ``test-fabric`` for the test build."""
    return ["test-fabric"] if test_fabric else ["ibverbs"]


def library_path(*, test_fabric: bool = False) -> Path:
    """Where the library for this source, toolchain, features and machine lives once built
    (anything that could change the result is part of the name, as for multitcp)."""
    cargo = _cargo()
    rustc = shutil.which("rustc")
    toolchain = subprocess.run([rustc, "--version"], capture_output=True, text=True).stdout \
        if rustc else (cargo or "")
    parts = [*(p.read_bytes() for p in _sources()), toolchain.encode(),
             platform.machine().encode(), " ".join(platform.libc_ver()).encode(),
             sys.platform.encode(), ",".join(features(test_fabric)).encode(),
             *(os.environ.get(v, "").encode() for v in BUILD_ENVS)]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    base = Path(os.environ.get(LIB_DIR_ENV) or Path.home() / ".cache" / "gpubridge")
    kind = "-test" if test_fabric else ""
    return base / f"libgpubridge_verbs{kind}-{key[:16]}{_suffix()}"


def build(*, test_fabric: bool = False, force: bool = False) -> Path:
    """Compile the crate with cargo (once per source, toolchain and features); return the
    library.

    Raises:
        Unavailable: without cargo, off Linux (for the real library), or if compiling
            fails (for example without rdma-core's headers).
    """
    if not test_fabric and not sys.platform.startswith("linux"):
        raise Unavailable(f"the verbs bridge needs Linux with rdma-core (libibverbs); this is "
                          f"{sys.platform}")
    cargo = _cargo()
    if cargo is None:
        raise Unavailable("no cargo (a Rust toolchain, https://rustup.rs) to build the verbs "
                          f"bridge; or set {LIB_ENV} to a library built elsewhere")
    path = library_path(test_fabric=test_fabric)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "CARGO_ENCODED_RUSTFLAGS"}
    env["RUSTFLAGS"] = ""
    with tempfile.TemporaryDirectory(prefix="gpubridge-verbs-", dir=path.parent) as work:
        crate = Path(work) / "crate"
        shutil.copytree(CRATE, crate, ignore=shutil.ignore_patterns("target"))
        command = [cargo, "build", "--release", "--offline", "--quiet", "--manifest-path",
                   str(crate / "Cargo.toml"), "--target-dir", str(Path(work) / "target"),
                   "--features", ",".join(features(test_fabric))]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise Unavailable("building the verbs bridge failed:\n" + " ".join(command)
                              + "\n" + result.stderr[-4000:])
        built = Path(work) / "target" / "release" / f"libgpubridge_verbs{_suffix()}"
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        shutil.copyfile(built, name)
        Path(name).replace(path)
    return path


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.gb_verbs_protocol.argtypes, lib.gb_verbs_protocol.restype = [], c.c_uint32
    lib.gb_verbs_build_info.argtypes = [c.c_char_p, c.c_size_t]
    lib.gb_verbs_create.argtypes = [c.c_int, c.c_int, c.c_char_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_create.restype = c.c_void_p
    lib.gb_verbs_local_info.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_connect.argtypes = [c.c_void_p, c.c_int, c.c_char_p]
    lib.gb_verbs_send.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_verbs_send.restype = c.c_int64
    lib.gb_verbs_recv.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64]
    lib.gb_verbs_recv.restype = c.c_int64
    lib.gb_verbs_wait.argtypes = [c.c_void_p, c.POINTER(c.c_int64), c.c_int, c.c_double]
    lib.gb_verbs_last_error.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_telemetry.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
    lib.gb_verbs_telemetry.restype = c.c_int64
    lib.gb_verbs_abort.argtypes = [c.c_void_p, c.c_char_p]
    lib.gb_verbs_close.argtypes = [c.c_void_p, c.c_double, c.c_char_p, c.c_size_t]
    for name in ("gb_verbs_build_info", "gb_verbs_local_info", "gb_verbs_connect",
                 "gb_verbs_wait", "gb_verbs_last_error", "gb_verbs_abort", "gb_verbs_close"):
        getattr(lib, name).restype = c.c_int
    return lib


def test_fabric_dir() -> str | None:
    """The test fabric's directory if ``GPUBRIDGE_VERBS_TEST_FABRIC`` is set."""
    return os.environ.get(TEST_FABRIC_ENV, "").strip() or None


def load(*, test_fabric: bool | None = None, build_if_missing: bool = True) -> ctypes.CDLL:
    """The library: for real use ``GPUBRIDGE_VERBS_LIB`` if set, else this source's build
    (made first if needed); the test build when ``test_fabric`` (by default: when
    ``GPUBRIDGE_VERBS_TEST_FABRIC`` is set).

    Raises:
        Unavailable: if it isn't there and can't be built (see :func:`build`).
    """
    if test_fabric is None:
        test_fabric = test_fabric_dir() is not None
    with _lock:
        if test_fabric not in _libraries:
            given = "" if test_fabric else os.environ.get(LIB_ENV, "").strip()
            if given:
                path = Path(given)
            else:
                path = build(test_fabric=test_fabric) if build_if_missing else \
                    library_path(test_fabric=test_fabric)
            if not path.exists():
                raise Unavailable(f"no verbs bridge library at {path}; run python -m "
                                  f"gpubridge.verbs build, or set {LIB_ENV}")
            try:
                _libraries[test_fabric] = _declare(ctypes.CDLL(str(path)))
            except OSError as exc:
                raise Unavailable(f"loading {path} failed: {exc}") from None
        return _libraries[test_fabric]


def protocol(lib: ctypes.CDLL | None = None) -> int:
    """The wire protocol version this library speaks; every leader's must match."""
    return int((lib or load()).gb_verbs_protocol())


def build_info(lib: ctypes.CDLL | None = None) -> str:
    """``gpubridge-verbs <version> providers=...``: what the library was built with."""
    out = ctypes.create_string_buffer(256)
    (lib or load()).gb_verbs_build_info(out, len(out))
    return out.value.decode()


class Endpoint:
    """One bridge member's end: an RC queue pair to every other member, made by
    :meth:`__init__` and connected by :meth:`connect`. Thread-safe in the library; a
    failure is remembered here, after which every call raises."""

    def __init__(self, members: int, index: int, settings: Mapping[str, object]) -> None:
        fabric = test_fabric_dir()
        self._lib = load(test_fabric=fabric is not None)
        lines = {**settings}
        if fabric is not None:
            lines.update(provider="test-fabric", fabric_dir=fabric)
        text = "".join(f"{k}={v}\n" for k, v in lines.items() if v is not None)
        error = ctypes.create_string_buffer(1024)
        handle = self._lib.gb_verbs_create(members, index, text.encode(), error, len(error))
        if not handle:
            raise TransportError(error.value.decode())
        self._handle: int | None = handle
        #: The first error, after which every call raises: a timed-out collective may
        #: have left messages half sent or received.
        self.failed: str | None = None

    def _error(self) -> str:
        out = ctypes.create_string_buffer(2048)
        self._lib.gb_verbs_last_error(self._handle, out, len(out))
        return out.value.decode()

    def _check(self, result: int) -> int:
        if result < 0:
            text = self._error()
            self.failed = f"verbs bridge {text}" if result == TIMED_OUT else \
                f"verbs bridge failed: {text}"
            raise TransportError(self.failed)
        return result

    def _usable(self) -> None:
        if self.failed is not None:
            raise TransportError(f"the verbs bridge already failed ({self.failed})")
        if self._handle is None:
            raise TransportError("the verbs bridge is closed")

    def local_info(self) -> str:
        """This member's handshake line (addresses and queue pair numbers; no keys)."""
        out = ctypes.create_string_buffer(64 * 1024)
        if self._lib.gb_verbs_local_info(self._handle, out, len(out)) != 0:
            raise TransportError("the verbs handshake line didn't fit")
        return out.value.decode()

    def connect(self, member: int, info: str) -> None:
        self._usable()
        if self._lib.gb_verbs_connect(self._handle, member, info.encode()) != 0:
            raise TransportError(self._error())

    def send(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        self._usable()
        return self._check(self._lib.gb_verbs_send(self._handle, member, pointer, nbytes, tag))

    def recv(self, member: int, pointer: int, nbytes: int, tag: int) -> int:
        self._usable()
        return self._check(self._lib.gb_verbs_recv(self._handle, member, pointer, nbytes, tag))

    def wait(self, ids: Sequence[int], deadline: float) -> None:
        """Wait for operations ``ids`` until ``deadline`` (``time.monotonic()``)."""
        self._usable()
        if not ids:
            return
        array = (ctypes.c_int64 * len(ids))(*ids)
        timeout = max(0.0, deadline - time.monotonic())
        self._check(self._lib.gb_verbs_wait(self._handle, array, len(ids), timeout))

    def telemetry(self) -> dict:
        """What the endpoint reports (docs/VERBS_TRANSPORT.md, "Telemetry")."""
        if self._handle is None:
            return {"state": "closed"}
        size = 16 * 1024
        while True:
            out = ctypes.create_string_buffer(size)
            n = self._lib.gb_verbs_telemetry(self._handle, out, size)
            if n >= 0:
                return json.loads(out.value.decode())
            size = -n + 1

    def abort(self, why: str = "aborted") -> None:
        """Fail the endpoint now: once this returns, nothing reads or writes the memory of
        any operation posted (the library copies through its own registered slots and
        cancels every operation under its lock). Every later call raises."""
        if self._handle is not None:
            if self.failed is None:
                self.failed = why
            self._lib.gb_verbs_abort(self._handle, why.encode())

    def close(self, timeout: float = 10.0) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            error = ctypes.create_string_buffer(1024)
            if self._lib.gb_verbs_close(handle, timeout, error, len(error)) != 0:
                log.warning("gpubridge: closing the verbs bridge left work undrained after "
                            "%g s; its slots stay registered (leaked, never the caller's "
                            "memory): %s", timeout, error.value.decode())
