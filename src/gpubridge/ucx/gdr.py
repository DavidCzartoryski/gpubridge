"""Making a NIC's writes into GPU memory visible to the GPU's kernels.

On the UCX bridge's direct path, the NIC writes received data straight into
GPU memory (GPUDirect RDMA, or ROCm peer memory). The CPU can see that receive
complete before those writes are visible to kernels on the GPU: NVIDIA's
GPUDirect RDMA guide says a kernel may read stale data until the writes have
completed with respect to the CPU thread that launches it, and a completion
the CPU polled doesn't establish that. NCCL and RCCL flush after such
receives; UCX (1.18 and master, read 2026-10-12) does not. So the transport
calls a :class:`Flush` after every receive on the direct path completes and
before any kernel reads the data (docs/RDMA_TRANSPORT.md, section 4):

- CUDA: ``cuFlushGPUDirectRDMAWrites`` (driver API, CUDA 11.3 and later),
  except on GPUs that report they order such writes for their own kernels
  (``CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING`` at least
  ``OWNER``, as NCCL expects of Hopper and later).
- ROCm: ``hipDeviceFlushGPUDirectRDMAWrites`` where the HIP runtime has it;
  otherwise what that function does, a write of 1 to the GPU's HDP flush
  register (``hipDeviceAttributeHdpMemFlushCntl``) and a read back.

Where a flush is needed and neither works, :attr:`Flush.available` is false
and the transport stages that GPU's tensors through pinned host memory
instead. Not validated on GPUs.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import torch

log = logging.getLogger("gpubridge")

# CUDA driver API (cuda.h): CUdevice_attribute, and the arguments of cuFlushGPUDirectRDMAWrites.
CU_FLUSH_WRITES_OPTIONS = 117  # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
CU_WRITES_ORDERING = 118       # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
OPTION_HOST = 1                # CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST
TARGET_CURRENT = 0             # ..._TARGET_CURRENT_CTX (CUDA), ...TargetCurrentDevice (HIP)
TO_OWNER = 100                 # ..._TO_OWNER, and the ordering value ..._ORDERING_OWNER
# HIP (hip_runtime_api.h): hipDeviceAttributeAmdSpecificBegin + 5, in every ROCm release.
HIP_HDP_MEM_FLUSH_CNTL = 10005


@dataclass
class Flush:
    """How the NIC's writes into one GPU's memory are made visible to its kernels."""

    method: str
    """What it does, for run records: ``cuFlushGPUDirectRDMAWrites``, ``not needed: ...``."""
    run: Callable[[], None] | None = None
    """Called after a receive into the GPU completes; None when nothing needs to run."""
    available: bool = True
    """False when a flush is needed but there is no way to do it here."""

    def __call__(self) -> None:
        if self.run is not None:
            self.run()


def for_device(index: int, hip: bool) -> Flush:
    """The flush for GPU ``index`` (torch's index) under the CUDA or the ROCm build."""

    def select() -> AbstractContextManager:
        return torch.cuda.device(index)

    flush = rocm(index, select) if hip else cuda(index, select)
    if not flush.available:
        log.warning("gpubridge: UCX bridge: GPU %d's tensors take the staged path (%s)", index,
                    flush.method)
    return flush


def cuda(index: int, select: Callable[[], AbstractContextManager], driver=None) -> Flush:
    """The flush for CUDA device ``index``. ``select()`` makes it the thread's current device."""
    try:
        cu = driver if driver is not None else ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return Flush(f"unavailable: can't load libcuda.so.1: {exc}", available=False)
    device, ordering, options = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    status = cu.cuDeviceGet(ctypes.byref(device), index)
    if status == 0:
        status = cu.cuDeviceGetAttribute(ctypes.byref(ordering), CU_WRITES_ORDERING, device)
    if status == 0:
        status = cu.cuDeviceGetAttribute(ctypes.byref(options), CU_FLUSH_WRITES_OPTIONS, device)
    if status != 0:  # e.g. a driver older than CUDA 11.3, without these attributes
        return Flush(f"unavailable: can't read CUDA device {index}'s GPUDirect RDMA attributes "
                     f"(CUresult {status})", available=False)
    if ordering.value >= TO_OWNER:
        return Flush("not needed: the GPU orders GPUDirect RDMA writes for its own kernels")
    if not options.value & OPTION_HOST:
        return Flush("unavailable: the GPU neither orders GPUDirect RDMA writes nor supports "
                     "cuFlushGPUDirectRDMAWrites", available=False)

    def run() -> None:
        with select():
            current = ctypes.c_int(-1)
            if cu.cuCtxGetDevice(ctypes.byref(current)) != 0 or current.value != device.value:
                raise RuntimeError(f"UCX bridge: can't flush GPUDirect RDMA writes into CUDA "
                                   f"device {index}: its context is not current")
            if (status := cu.cuFlushGPUDirectRDMAWrites(TARGET_CURRENT, TO_OWNER)) != 0:
                raise RuntimeError(f"UCX bridge: cuFlushGPUDirectRDMAWrites failed on CUDA "
                                   f"device {index} (CUresult {status})")

    return Flush("cuFlushGPUDirectRDMAWrites", run)


def rocm(index: int, select: Callable[[], AbstractContextManager], hip=None,
         hdp_flush: Callable[[int], None] | None = None) -> Flush:
    """The flush for ROCm device ``index``; ``hip`` is the HIP runtime PyTorch loaded and
    ``hdp_flush(address)`` writes and reads back an HDP flush register."""
    if hip is None:
        path = loaded_library("libamdhip64.so")
        if path is None:
            return Flush("unavailable: the HIP runtime isn't loaded", available=False)
        hip = ctypes.CDLL(path)  # the same copy PyTorch uses, never a second runtime
    api = getattr(hip, "hipDeviceFlushGPUDirectRDMAWrites", None)
    if api is not None:
        with select():
            status = api(TARGET_CURRENT, TO_OWNER)  # once now: says whether it's supported
        if status != 0:
            return Flush(f"unavailable: hipDeviceFlushGPUDirectRDMAWrites failed (hipError_t "
                         f"{status})", available=False)

        def run() -> None:
            with select():
                if (status := api(TARGET_CURRENT, TO_OWNER)) != 0:
                    raise RuntimeError(f"UCX bridge: hipDeviceFlushGPUDirectRDMAWrites failed "
                                       f"on ROCm device {index} (hipError_t {status})")

        return Flush("hipDeviceFlushGPUDirectRDMAWrites", run)
    register = ctypes.c_void_p()  # HIP writes the register's address through an int*
    status = hip.hipDeviceGetAttribute(ctypes.byref(register), HIP_HDP_MEM_FLUSH_CNTL, index)
    if status != 0 or not register.value:
        return Flush(f"unavailable: no HDP flush register for ROCm device {index} (hipError_t "
                     f"{status})", available=False)
    if hdp_flush is None:
        from gpubridge import ucx

        hdp_flush = ucx.load().gb_ucx_hdp_flush
    address = register.value
    return Flush("HDP flush register", lambda: hdp_flush(address))


def loaded_library(name: str, maps: Path = Path("/proc/self/maps")) -> str | None:
    """The path of a shared library this process already mapped whose file name starts
    with ``name`` (e.g. ``libamdhip64.so``), or None."""
    try:
        lines = maps.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(name):
            return fields[5]
    return None
