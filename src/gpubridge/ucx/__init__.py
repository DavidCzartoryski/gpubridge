"""The UCX library behind ``bridge="ucx"`` (docs/RDMA_TRANSPORT.md): build, load, call.

``bridge.cpp`` is a small C++ library over UCX's UCP API with a C interface.
Nothing is compiled at install time. ``python -m gpubridge.ucx build`` compiles
it, and :func:`load` does on first use: with the system C++ compiler, against
UCX found through ``UCX_HOME`` or ``pkg-config ucx``, into
``GPUBRIDGE_UCX_LIB_DIR`` (default ``~/.cache/gpubridge``). Linux only. Without a
compiler or UCX, :class:`Unavailable` says what is missing; the rest of
gpubridge never needs this module.
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

SOURCE = Path(__file__).with_name("bridge.cpp")
#: Where the compiled library goes (``~/.cache/gpubridge`` by default).
LIB_DIR_ENV = "GPUBRIDGE_UCX_LIB_DIR"
#: A UCX installation prefix, for clusters where pkg-config doesn't know UCX.
UCX_HOME_ENV = "UCX_HOME"

#: Memory types, as bridge.cpp numbers them.
HOST, CUDA, ROCM, DETECT = 0, 1, 2, -1
_TYPE_BITS = {"host": 1, "cuda": 2, "rocm": 4}

TIMED_OUT, PEER_FAILED = -2, -3

_lock = threading.Lock()
_library: ctypes.CDLL | None = None
log = logging.getLogger("gpubridge")


class Unavailable(RuntimeError):
    """The UCX library can't be built or loaded here; the message says why."""


class UcxError(RuntimeError):
    """A UCX operation failed or timed out."""


def _ucx_flags() -> tuple[list[str], list[str]]:
    """Compiler and linker flags for UCX."""
    home = os.environ.get(UCX_HOME_ENV, "").strip()
    if home:
        lib = next((Path(home) / d for d in ("lib", "lib64") if (Path(home) / d).is_dir()),
                   Path(home) / "lib")
        return [f"-I{home}/include"], [f"-L{lib}", f"-Wl,-rpath,{lib}", "-lucp", "-lucs"]
    if shutil.which("pkg-config") is None:
        raise Unavailable(f"UCX not found: set {UCX_HOME_ENV} or install pkg-config and UCX "
                          "(e.g. apt install libucx-dev, or module load ucx)")
    found = subprocess.run(["pkg-config", "--cflags", "--libs", "ucx"], capture_output=True,
                           text=True)
    if found.returncode != 0:
        raise Unavailable(f"UCX not found by pkg-config ({found.stderr.strip()}); set "
                          f"{UCX_HOME_ENV} or install UCX (e.g. apt install libucx-dev)")
    flags = found.stdout.split()
    return ([f for f in flags if f.startswith("-I")],
            [f for f in flags if not f.startswith("-I")])


def _compiler() -> str | None:
    return os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")


def _ucx_version_text(cflags: list[str]) -> str:
    """The UCX headers' version, so two UCX builds at the same path never share a library."""
    for flag in [*cflags, "-I/usr/include"]:
        header = Path(flag[2:]) / "ucp" / "api" / "ucp_version.h"
        if header.is_file():
            return header.read_text()
    return ""


def library_path() -> Path:
    """Where the library for this source, compiler, UCX and machine lives once built.

    The cache is often a home directory shared by every node of a cluster, and in a
    mixed job each side builds against its own UCX: anything that could change the
    result is part of the name.
    """
    cflags, ldflags = _ucx_flags()
    compiler = _compiler() or ""
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout \
        if compiler else ""
    parts = [SOURCE.read_bytes(), " ".join(cflags + ldflags).encode(), version.encode(),
             _ucx_version_text(cflags).encode(), platform.machine().encode(),
             " ".join(platform.libc_ver()).encode()]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()
    base = Path(os.environ.get(LIB_DIR_ENV) or Path.home() / ".cache" / "gpubridge")
    return base / f"libgpubridge_ucx-{key[:16]}.so"


def build(*, force: bool = False) -> Path:
    """Compile bridge.cpp against UCX (once per source and UCX); return the library's path.

    Raises:
        Unavailable: off Linux, without a C++ compiler or UCX, or if compiling fails.
    """
    if not sys.platform.startswith("linux"):
        raise Unavailable(f"the UCX bridge is Linux only (this is {sys.platform})")
    compiler = _compiler()
    if compiler is None:
        raise Unavailable("no C++ compiler (c++ or g++, or set CXX) to build the UCX bridge")
    path = library_path()
    if path.exists() and not force:
        return path
    cflags, ldflags = _ucx_flags()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique name per build: two nodes sharing the cache can have the same pid.
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    partial = Path(name)
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", *cflags, str(SOURCE),
               "-o", str(partial), *ldflags]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise Unavailable("building the UCX bridge failed:\n" + " ".join(command) + "\n"
                          + result.stderr[-4000:])
    partial.replace(path)  # atomic, so ranks building at once don't see half a file
    return path


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.gb_ucx_last_error.argtypes, lib.gb_ucx_last_error.restype = [c.c_void_p], c.c_char_p
    lib.gb_ucx_version.argtypes = [c.POINTER(c.c_uint)] * 3
    lib.gb_ucx_version.restype = None
    lib.gb_ucx_create.argtypes = [c.c_int, c.c_char_p, c.c_size_t]
    lib.gb_ucx_create.restype = c.c_void_p
    lib.gb_ucx_memory_types.argtypes = [c.c_void_p, c.POINTER(c.c_uint64)]
    lib.gb_ucx_address.argtypes = [c.c_void_p, c.POINTER(c.c_void_p), c.POINTER(c.c_size_t)]
    lib.gb_ucx_free_address.argtypes = [c.c_void_p, c.c_void_p]
    lib.gb_ucx_free_address.restype = None
    lib.gb_ucx_connect.argtypes = [c.c_void_p, c.c_int, c.c_char_p]
    lib.gb_ucx_send.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_uint64, c.c_int]
    lib.gb_ucx_send.restype = c.c_int64
    lib.gb_ucx_recv.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_uint64, c.c_int]
    lib.gb_ucx_recv.restype = c.c_int64
    lib.gb_ucx_wait.argtypes = [c.c_void_p, c.POINTER(c.c_int64), c.c_int, c.c_double]
    lib.gb_ucx_close.argtypes = [c.c_void_p, c.c_double]
    for name in ("gb_ucx_memory_types", "gb_ucx_address", "gb_ucx_connect", "gb_ucx_wait",
                 "gb_ucx_close"):
        getattr(lib, name).restype = c.c_int
    return lib


def load(*, build_if_missing: bool = True) -> ctypes.CDLL:
    """The UCX library, building it first if needed.

    Raises:
        Unavailable: if it isn't built and can't be (see :func:`build`).
    """
    global _library
    with _lock:
        if _library is None:
            path = build() if build_if_missing else library_path()
            if not path.exists():
                raise Unavailable("the UCX bridge isn't built; run python -m gpubridge.ucx build")
            try:
                _library = _declare(ctypes.CDLL(str(path)))
            except OSError as exc:
                if not build_if_missing:
                    raise Unavailable(f"loading {path} failed: {exc}") from None
                try:  # a damaged cached build: rebuild it once
                    _library = _declare(ctypes.CDLL(str(build(force=True))))
                except OSError as again:
                    raise Unavailable(f"loading {path} failed: {again}") from None
        return _library


def version() -> tuple[int, int, int]:
    """The UCX version the library runs against."""
    parts = [ctypes.c_uint() for _ in range(3)]
    load().gb_ucx_version(*(ctypes.byref(p) for p in parts))
    return parts[0].value, parts[1].value, parts[2].value


class Worker:
    """A UCP context, worker and one endpoint per bridge member. Not thread-safe."""

    def __init__(self, members: int) -> None:
        self._lib = load()
        error = ctypes.create_string_buffer(512)
        handle = self._lib.gb_ucx_create(members, error, len(error))
        if not handle:
            raise UcxError(
                f"{error.value.decode()}. UCX uses every transport it finds; if one of them "
                "can't actually be used here (common on cloud VMs), set UCX_TLS (e.g. "
                "tcp,self, or rc,self on InfiniBand) or UCX_NET_DEVICES")
        self._handle: int | None = handle

    def _check(self, result: int) -> int:
        if result < 0:
            text = self._lib.gb_ucx_last_error(self._handle).decode()
            kind = {TIMED_OUT: "timed out", PEER_FAILED: "peer failed"}.get(result, "failed")
            raise UcxError(f"UCX {kind}: {text}")
        return result

    def memory_types(self) -> set[str]:
        bits = ctypes.c_uint64()
        self._check(self._lib.gb_ucx_memory_types(self._handle, ctypes.byref(bits)))
        return {name for name, bit in _TYPE_BITS.items() if bits.value & bit}

    def address(self) -> bytes:
        pointer, length = ctypes.c_void_p(), ctypes.c_size_t()
        self._check(self._lib.gb_ucx_address(self._handle, ctypes.byref(pointer),
                                             ctypes.byref(length)))
        try:
            return ctypes.string_at(pointer, length.value)
        finally:
            self._lib.gb_ucx_free_address(self._handle, pointer)

    def connect(self, member: int, address: bytes) -> None:
        self._check(self._lib.gb_ucx_connect(self._handle, member, address))

    def send(self, member: int, pointer: int, nbytes: int, tag: int,
             memory: int = HOST) -> int:
        return self._check(self._lib.gb_ucx_send(self._handle, member, pointer, nbytes, tag,
                                                 memory))

    def recv(self, pointer: int, nbytes: int, tag: int, memory: int = HOST) -> int:
        return self._check(self._lib.gb_ucx_recv(self._handle, pointer, nbytes, tag, memory))

    def wait(self, ids: Sequence[int], deadline: float) -> None:
        """Wait for operations ``ids`` until ``deadline`` (``time.monotonic()``)."""
        if not ids:
            return
        array = (ctypes.c_int64 * len(ids))(*ids)
        timeout = max(0.0, deadline - time.monotonic())
        self._check(self._lib.gb_ucx_wait(self._handle, array, len(ids), timeout))

    def close(self, timeout: float = 10.0) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            if self._lib.gb_ucx_close(handle, timeout) != 0:
                log.warning("gpubridge: UCX endpoints didn't finish closing within %g s; the "
                            "worker was torn down anyway", timeout)
