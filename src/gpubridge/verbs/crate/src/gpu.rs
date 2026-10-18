//! GPU memory the NIC can reach directly (docs/VERBS_TRANSPORT.md, "GPU memory").
//!
//! [`GpuMemory`] is what the direct paths need from a GPU driver. The drivers are loaded
//! at run time (`dlopen`), so the crate has no build dependency on CUDA or ROCm and a
//! host without them loses only the direct paths:
//!
//! - the allocation a device pointer lies in, and its identity ([`Allocation`]): base,
//!   size and the driver's buffer ID, which no later allocation shares even at the same
//!   address, so a cached registration of memory that was freed and handed out again is
//!   recognised as stale (the registration cache's owner token);
//! - whether the device can export dma-bufs, and the export;
//! - whether its memory can be registered through peer memory (`ibv_reg_mr` on the
//!   device pointer), apart from the kernel module, which [`PeerModules`] looks for;
//! - what to do once per new registration (CUDA: `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`);
//! - closing an exported dma-buf ([`DmabufCloser`]).
//!
//! Three implementations: [`CudaDriver`] (`libcuda.so.1`, the CUDA driver API),
//! [`RocmDriver`] (the HIP runtime and ROCr the process already loaded: a ROCm build of
//! PyTorch has both) and [`MockGpu`] (host memory behind made-up device addresses, with
//! simulated buffer IDs and dma-bufs, for tests). Names, values and signatures are from
//! CUDA 12.8's `cuda.h`, HIP 6.3's `hip_runtime_api.h` and ROCr's `hsa.h` and
//! `hsa_ext_amd.h` at rocm-6.3.0 (the doc's "GPU memory" has the list).
//!
//! What the NIC may do with the memory, and when the GPU may use what the NIC wrote, is
//! the transport's business (docs: ordering and flushes); none of it is here.

use std::any::Any;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::error::{Error, Result};
use crate::resources::DmabufCloser;

/// Whose driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Cuda,
    Rocm,
    /// [`MockGpu`]: not a GPU.
    Mock,
}

impl Vendor {
    pub fn name(self) -> &'static str {
        match self {
            Vendor::Cuda => "cuda",
            Vendor::Rocm => "rocm",
            Vendor::Mock => "mock",
        }
    }
}

/// The allocation a device pointer lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub base: u64,
    pub size: u64,
    /// Unique to this allocation in this process; a later allocation at the same address
    /// has another (`CU_POINTER_ATTRIBUTE_BUFFER_ID`, `HIP_POINTER_ATTRIBUTE_BUFFER_ID`).
    pub buffer_id: u64,
    /// The device ordinal it was allocated on.
    pub device: i32,
    /// Whether the driver says the memory can be used for GPUDirect RDMA
    /// (`CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`), where it says.
    pub rdma_capable: Option<bool>,
}

impl Allocation {
    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.size)
    }
}

/// What the direct paths need from a GPU driver; see the module docs.
pub trait GpuMemory: DmabufCloser + fmt::Debug {
    fn vendor(&self) -> Vendor;
    /// The library and its version, for telemetry.
    fn describe(&self) -> String;
    /// The allocation `ptr` lies in, which must be device memory of `device`.
    fn allocation(&self, device: i32, ptr: u64) -> Result<Allocation>;
    /// Whether `device`'s memory can be exported as a dma-buf; the reason if not.
    fn dmabuf_support(&self, device: i32) -> std::result::Result<(), String>;
    /// Exports `len` bytes at `start` of `alloc` as a dma-buf: its file descriptor (the
    /// caller owns it and closes it with [`DmabufCloser::close_dmabuf`]) and the offset of
    /// `start` in it.
    fn export_dmabuf(&self, alloc: &Allocation, start: u64, len: u64) -> Result<(i32, u64)>;
    /// Whether the driver lets `device`'s memory be registered through peer memory
    /// (besides the kernel module, which [`PeerModules`] checks); the reason if not.
    fn peer_memory_support(&self, device: i32) -> std::result::Result<(), String>;
    /// Called once for each new registration of memory at `start` of `alloc`.
    fn prepare(&self, alloc: &Allocation, start: u64) -> Result<()>;
}

/// The host's page size (dma-buf exports are page-aligned).
pub fn page_size() -> u64 {
    extern "C" {
        fn getpagesize() -> c_int;
    }
    // SAFETY: getpagesize takes nothing and only returns a number (glibc, musl, macOS).
    let n = unsafe { getpagesize() };
    if n > 0 {
        n as u64
    } else {
        4096
    }
}

// ---- loading shared libraries ----------------------------------------------------------

mod dl {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *mut c_char;
    }

    // The same value on Linux and macOS.
    const RTLD_NOW: c_int = 2;

    /// A shared library, never unloaded: the function pointers taken from it stay valid
    /// for the life of the process (GPU drivers don't support being unloaded anyway).
    pub struct Library {
        handle: *mut c_void,
        pub path: String,
    }

    // SAFETY: a dlopen handle is a process-wide token; dlsym is thread-safe.
    unsafe impl Send for Library {}
    // SAFETY: as for Send.
    unsafe impl Sync for Library {}

    fn last_error() -> String {
        // SAFETY: dlerror returns null or a NUL-terminated string valid until the next dl
        // call on this thread; it is copied at once.
        let e = unsafe { dlerror() };
        if e.is_null() {
            "unknown error".into()
        } else {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    }

    impl Library {
        pub fn open(path: &str) -> Result<Library, String> {
            let c = CString::new(path).map_err(|_| format!("bad library name {path:?}"))?;
            // SAFETY: c is NUL-terminated; loading runs the library's initialisers, which
            // is what loading a GPU driver means.
            let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return Err(format!("can't load {path}: {}", last_error()));
            }
            Ok(Library {
                handle,
                path: path.to_owned(),
            })
        }

        /// The function `name`, typed as `T`.
        ///
        /// # Safety
        ///
        /// `T` must be an `unsafe extern "C" fn` type matching the symbol's C signature.
        pub unsafe fn sym<T: Copy>(&self, name: &str) -> Option<T> {
            assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
            let c = CString::new(name).ok()?;
            // SAFETY: handle is from dlopen; c is NUL-terminated.
            let p = unsafe { dlsym(self.handle, c.as_ptr()) };
            if p.is_null() {
                return None;
            }
            // SAFETY: T is a function pointer type of the symbol's signature (the
            // caller's contract), the same size as a data pointer (asserted).
            Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
        }
    }
}

/// The path of a shared library this process already mapped whose file name starts with
/// `prefix` (e.g. `libamdhip64.so`), from `/proc/self/maps` (Linux).
pub fn loaded_library(prefix: &str) -> Option<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines().find_map(|line| {
        let path = line.split_whitespace().nth(5)?;
        let name = Path::new(path).file_name()?.to_str()?;
        name.starts_with(prefix).then(|| path.to_owned())
    })
}

/// Reads one attribute into a zeroed 8-byte cell: drivers write 1, 4 or 8 bytes
/// depending on the attribute, little-endian (both drivers run only on little-endian
/// hosts), so the low bytes hold the value either way.
fn cell() -> u64 {
    0
}

fn as_i32(v: u64) -> i32 {
    v as u32 as i32
}

// ---- NVIDIA: the CUDA driver API ----------------------------------------------------

type CuResult = c_int;
type CuDeviceptr = u64;
type CuContext = *mut c_void;

// cuda.h (12.8): CUdevice_attribute.
const CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED: c_int = 116;
const CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: c_int = 124;
// CUpointer_attribute.
const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;
const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: c_int = 6;
const CU_POINTER_ATTRIBUTE_BUFFER_ID: c_int = 7;
const CU_POINTER_ATTRIBUTE_IS_MANAGED: c_int = 8;
const CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: c_int = 9;
const CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: c_int = 15;
// CUmemorytype.
const CU_MEMORYTYPE_DEVICE: u64 = 2;
// CUmemRangeHandleType.
const CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD: c_int = 1;

#[derive(Clone, Copy)]
struct CudaFns {
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    pointer_get_attribute: unsafe extern "C" fn(*mut c_void, c_int, CuDeviceptr) -> CuResult,
    pointer_set_attribute: unsafe extern "C" fn(*const c_void, c_int, CuDeviceptr) -> CuResult,
    // cuMemGetAddressRange_v2 (what cuda.h's cuMemGetAddressRange names).
    mem_get_address_range:
        unsafe extern "C" fn(*mut CuDeviceptr, *mut usize, CuDeviceptr) -> CuResult,
    // CUDA 11.7 and later only.
    mem_get_handle_for_address_range:
        Option<unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize, c_int, u64) -> CuResult>,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, c_int) -> CuResult,
    // cuDevicePrimaryCtxRelease_v2, cuCtxPushCurrent_v2, cuCtxPopCurrent_v2.
    primary_ctx_release: unsafe extern "C" fn(c_int) -> CuResult,
    ctx_push_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_pop_current: unsafe extern "C" fn(*mut CuContext) -> CuResult,
}

/// The CUDA driver API from `libcuda.so.1`. Calls run with the device's primary context
/// pushed on the calling thread (retained once per device, released when this goes).
pub struct CudaDriver {
    lib: dl::Library,
    fns: CudaFns,
    version: i32,
    /// Retained primary contexts, by device (as `usize`: a context is a process-wide
    /// handle, usable from any thread).
    contexts: Mutex<HashMap<i32, (c_int, usize)>>,
}

impl fmt::Debug for CudaDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaDriver({}, {})", self.lib.path, self.version)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl CudaDriver {
    /// Loads `libcuda.so.1` (or `path`) and initialises the driver API.
    pub fn load(path: Option<&str>) -> Result<CudaDriver> {
        let lib = dl::Library::open(path.unwrap_or("libcuda.so.1")).map_err(Error::Unsupported)?;
        macro_rules! need {
            ($name:literal) => {
                // SAFETY: the field's type is the function's signature in cuda.h (12.8).
                unsafe { lib.sym($name) }
                    .ok_or_else(|| Error::Unsupported(format!("{} has no {}", lib.path, $name)))?
            };
        }
        type Init = unsafe extern "C" fn(c_uint) -> CuResult;
        type Version = unsafe extern "C" fn(*mut c_int) -> CuResult;
        let init: Init = need!("cuInit");
        let driver_version: Version = need!("cuDriverGetVersion");
        let fns = CudaFns {
            get_error_string: need!("cuGetErrorString"),
            device_get: need!("cuDeviceGet"),
            device_get_attribute: need!("cuDeviceGetAttribute"),
            pointer_get_attribute: need!("cuPointerGetAttribute"),
            pointer_set_attribute: need!("cuPointerSetAttribute"),
            mem_get_address_range: need!("cuMemGetAddressRange_v2"),
            // SAFETY: the signature in cuda.h (12.8).
            mem_get_handle_for_address_range: unsafe { lib.sym("cuMemGetHandleForAddressRange") },
            primary_ctx_retain: need!("cuDevicePrimaryCtxRetain"),
            primary_ctx_release: need!("cuDevicePrimaryCtxRelease_v2"),
            ctx_push_current: need!("cuCtxPushCurrent_v2"),
            ctx_pop_current: need!("cuCtxPopCurrent_v2"),
        };
        let mut version = 0;
        // SAFETY: cuInit(0) initialises the driver (idempotent); the out pointer is valid.
        let rc = unsafe { init(0) };
        let mut driver = CudaDriver {
            lib,
            fns,
            version: 0,
            contexts: Mutex::new(HashMap::new()),
        };
        driver.check("cuInit", rc)?;
        // SAFETY: valid out pointer.
        driver.check("cuDriverGetVersion", unsafe {
            driver_version(&mut version)
        })?;
        driver.version = version;
        Ok(driver)
    }

    fn error_text(&self, rc: CuResult) -> String {
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: valid out pointer; the string is static in the driver.
        if unsafe { (self.fns.get_error_string)(rc, &mut s) } == 0 && !s.is_null() {
            // SAFETY: NUL-terminated, static.
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        } else {
            format!("CUresult {rc}")
        }
    }

    fn check(&self, call: &str, rc: CuResult) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(Error::Gpu(format!(
                "{call} failed: {} ({rc})",
                self.error_text(rc)
            )))
        }
    }

    /// Runs `f` with `device`'s primary context current on this thread.
    fn with_ctx<T>(&self, device: i32, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let ctx = {
            let mut contexts = lock(&self.contexts);
            match contexts.get(&device) {
                Some(&(_, ctx)) => ctx,
                None => {
                    let mut dev: c_int = 0;
                    // SAFETY: valid out pointers; device is checked by the driver.
                    self.check("cuDeviceGet", unsafe {
                        (self.fns.device_get)(&mut dev, device)
                    })?;
                    let mut ctx: CuContext = std::ptr::null_mut();
                    // SAFETY: as above.
                    self.check("cuDevicePrimaryCtxRetain", unsafe {
                        (self.fns.primary_ctx_retain)(&mut ctx, dev)
                    })?;
                    contexts.insert(device, (dev, ctx as usize));
                    ctx as usize
                }
            }
        };
        // SAFETY: ctx is a retained primary context (released only when self drops).
        self.check("cuCtxPushCurrent", unsafe {
            (self.fns.ctx_push_current)(ctx as CuContext)
        })?;
        let out = f();
        let mut popped: CuContext = std::ptr::null_mut();
        // SAFETY: pops what was pushed above, on the same thread.
        unsafe { (self.fns.ctx_pop_current)(&mut popped) };
        out
    }

    fn device_attribute(&self, device: i32, attr: c_int) -> Result<i32> {
        let mut dev: c_int = 0;
        let mut v: c_int = 0;
        // SAFETY: valid out pointers.
        self.check("cuDeviceGet", unsafe {
            (self.fns.device_get)(&mut dev, device)
        })?;
        // SAFETY: valid out pointer; attr is a CUdevice_attribute value.
        self.check("cuDeviceGetAttribute", unsafe {
            (self.fns.device_get_attribute)(&mut v, attr, dev)
        })?;
        Ok(v)
    }

    fn pointer_attribute(&self, ptr: u64, attr: c_int) -> Result<u64> {
        let mut v = cell();
        // SAFETY: an 8-byte out cell holds every attribute asked for here.
        self.check("cuPointerGetAttribute", unsafe {
            (self.fns.pointer_get_attribute)((&mut v as *mut u64).cast(), attr, ptr)
        })?;
        Ok(v)
    }

    /// The driver version (`cuDriverGetVersion`: 12080 for 12.8).
    pub fn version(&self) -> i32 {
        self.version
    }
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        for (_, (dev, _)) in lock(&self.contexts).drain() {
            // SAFETY: each was retained once in with_ctx.
            unsafe { (self.fns.primary_ctx_release)(dev) };
        }
    }
}

impl DmabufCloser for CudaDriver {
    fn close_dmabuf(&self, fd: i32) -> Result<()> {
        close_fd(fd)
    }
}

fn close_fd(fd: i32) -> Result<()> {
    extern "C" {
        fn close(fd: c_int) -> c_int;
    }
    // SAFETY: fd is a descriptor this process owns (DmabufFd's contract), closed once.
    if unsafe { close(fd) } == 0 {
        Ok(())
    } else {
        Err(Error::Gpu(format!(
            "closing dma-buf descriptor {fd}: {}",
            std::io::Error::last_os_error()
        )))
    }
}

impl GpuMemory for CudaDriver {
    fn vendor(&self) -> Vendor {
        Vendor::Cuda
    }

    fn describe(&self) -> String {
        format!(
            "CUDA driver {}.{} ({})",
            self.version / 1000,
            (self.version % 1000) / 10,
            self.lib.path
        )
    }

    fn allocation(&self, device: i32, ptr: u64) -> Result<Allocation> {
        self.with_ctx(device, || {
            let kind = self.pointer_attribute(ptr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE)?;
            if kind != CU_MEMORYTYPE_DEVICE {
                return Err(Error::Unsupported(format!(
                    "{ptr:#x} is not device memory (CUmemorytype {kind})"
                )));
            }
            if self.pointer_attribute(ptr, CU_POINTER_ATTRIBUTE_IS_MANAGED)? & 0xff != 0 {
                return Err(Error::Unsupported(format!(
                    "{ptr:#x} is managed memory, which neither path registers"
                )));
            }
            let owner = as_i32(self.pointer_attribute(ptr, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)?);
            if owner != device {
                return Err(Error::invalid(format!(
                    "{ptr:#x} belongs to CUDA device {owner}, not {device}"
                )));
            }
            // Querying the buffer ID also sets SYNC_MEMOPS on the allocation (cuda.h);
            // prepare() sets it explicitly as well.
            let buffer_id = self.pointer_attribute(ptr, CU_POINTER_ATTRIBUTE_BUFFER_ID)?;
            let (mut base, mut size) = (0u64, 0usize);
            // SAFETY: valid out pointers.
            self.check("cuMemGetAddressRange", unsafe {
                (self.fns.mem_get_address_range)(&mut base, &mut size, ptr)
            })?;
            let rdma_capable = self
                .pointer_attribute(ptr, CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE)
                .ok()
                .map(|v| v & 0xff != 0);
            Ok(Allocation {
                base,
                size: size as u64,
                buffer_id,
                device,
                rdma_capable,
            })
        })
    }

    fn dmabuf_support(&self, device: i32) -> std::result::Result<(), String> {
        if self.fns.mem_get_handle_for_address_range.is_none() {
            return Err(format!(
                "{} has no cuMemGetHandleForAddressRange (CUDA 11.7 or later)",
                self.lib.path
            ));
        }
        match self.device_attribute(device, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED) {
            Ok(1) => Ok(()),
            Ok(v) => Err(format!(
                "the driver reports no dma-buf export on CUDA device {device} \
                 (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = {v})"
            )),
            Err(e) => Err(format!(
                "can't read CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED of CUDA device {device}: {e}"
            )),
        }
    }

    fn export_dmabuf(&self, alloc: &Allocation, start: u64, len: u64) -> Result<(i32, u64)> {
        let export = self.fns.mem_get_handle_for_address_range.ok_or_else(|| {
            Error::Unsupported("no cuMemGetHandleForAddressRange in this driver".into())
        })?;
        let page = page_size();
        if start % page != 0 || len % page != 0 {
            return Err(Error::invalid(format!(
                "a dma-buf export must be page-aligned: {len} bytes at {start:#x}"
            )));
        }
        self.with_ctx(alloc.device, || {
            let mut fd: c_int = -1;
            // SAFETY: for CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD the handle is an int the driver
            // fills in (cuda.h); start/len are page-aligned and inside one allocation.
            self.check("cuMemGetHandleForAddressRange", unsafe {
                export(
                    (&mut fd as *mut c_int).cast(),
                    start,
                    len as usize,
                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                    0,
                )
            })?;
            if fd < 0 {
                return Err(Error::Gpu(
                    "cuMemGetHandleForAddressRange returned no descriptor".into(),
                ));
            }
            // The handle covers exactly [start, start + len).
            Ok((fd, 0))
        })
    }

    fn peer_memory_support(&self, device: i32) -> std::result::Result<(), String> {
        match self.device_attribute(device, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED) {
            Ok(0) => Err(format!(
                "the driver reports no GPUDirect RDMA on CUDA device {device} \
                 (CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 0)"
            )),
            // A driver without the attribute (before CUDA 11.3) may still have it: the
            // registration decides.
            _ => Ok(()),
        }
    }

    fn prepare(&self, alloc: &Allocation, start: u64) -> Result<()> {
        // NVIDIA's GPUDirect RDMA guide: set SYNC_MEMOPS on memory the NIC accesses, so
        // synchronous CUDA memory operations on it complete before they return.
        let one: c_uint = 1;
        self.with_ctx(alloc.device, || {
            // SAFETY: the value is an unsigned int (cuda.h); start lies in alloc.
            self.check("cuPointerSetAttribute(SYNC_MEMOPS)", unsafe {
                (self.fns.pointer_set_attribute)(
                    (&one as *const c_uint).cast(),
                    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                    start,
                )
            })
        })
    }
}

// ---- AMD: the HIP runtime and ROCr -------------------------------------------------------

// hip_runtime_api.h / driver_types.h (6.3): hipPointer_attribute, hipMemoryType.
const HIP_POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;
const HIP_POINTER_ATTRIBUTE_BUFFER_ID: c_int = 7;
const HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: c_int = 9;
const HIP_MEMORY_TYPE_DEVICE: u64 = 2;
// hsa.h (rocm-6.3.0): hsa_system_info_t.
const HSA_SYSTEM_INFO_VERSION_MAJOR: c_int = 0;
const HSA_SYSTEM_INFO_VERSION_MINOR: c_int = 1;
const HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED: c_int = 0x204;

#[derive(Clone, Copy)]
struct HipFns {
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    pointer_get_attribute: unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> c_int,
    mem_get_address_range: unsafe extern "C" fn(*mut *mut c_void, *mut usize, *mut c_void) -> c_int,
}

#[derive(Clone, Copy)]
struct HsaFns {
    status_string: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int,
    system_get_info: unsafe extern "C" fn(c_int, *mut c_void) -> c_int,
    export_dmabuf:
        Option<unsafe extern "C" fn(*const c_void, usize, *mut c_int, *mut u64) -> c_int>,
    close_dmabuf: Option<unsafe extern "C" fn(c_int) -> c_int>,
}

/// ROCm through the HIP runtime (allocation identity: `hipPointerGetAttribute`,
/// `hipMemGetAddressRange`) and ROCr (dma-buf export: `hsa_amd_portable_export_dmabuf`,
/// `hsa_amd_portable_close_dmabuf`), both the copies the process already loaded, never
/// a second runtime. HIP 6.3 has no dma-buf export of its own.
pub struct RocmDriver {
    hip: dl::Library,
    hsa: dl::Library,
    hipf: HipFns,
    hsaf: HsaFns,
    hip_version: i32,
    hsa_version: (u16, u16),
}

impl fmt::Debug for RocmDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RocmDriver({}, {})", self.hip.path, self.hsa.path)
    }
}

impl RocmDriver {
    /// The HIP runtime and ROCr this process loaded (`libamdhip64.so*`,
    /// `libhsa-runtime64.so*` in `/proc/self/maps`), or the paths given.
    pub fn load(hip_path: Option<&str>, hsa_path: Option<&str>) -> Result<RocmDriver> {
        let hip_path = hip_path
            .map(str::to_owned)
            .or_else(|| loaded_library("libamdhip64.so"));
        let hsa_path = hsa_path
            .map(str::to_owned)
            .or_else(|| loaded_library("libhsa-runtime64.so"));
        let (Some(hip_path), Some(hsa_path)) = (hip_path, hsa_path) else {
            return Err(Error::Unsupported(
                "the HIP runtime and ROCr aren't loaded in this process (a ROCm build of \
                 PyTorch loads both)"
                    .into(),
            ));
        };
        let hip = dl::Library::open(&hip_path).map_err(Error::Unsupported)?;
        let hsa = dl::Library::open(&hsa_path).map_err(Error::Unsupported)?;
        macro_rules! need {
            ($lib:expr, $name:literal) => {
                // SAFETY: the field's type is the function's signature in the 6.3 headers.
                unsafe { $lib.sym($name) }
                    .ok_or_else(|| Error::Unsupported(format!("{} has no {}", $lib.path, $name)))?
            };
        }
        type Version = unsafe extern "C" fn(*mut c_int) -> c_int;
        type Init = unsafe extern "C" fn() -> c_int;
        let runtime_version: Version = need!(hip, "hipRuntimeGetVersion");
        let hsa_init: Init = need!(hsa, "hsa_init");
        let hipf = HipFns {
            get_error_string: need!(hip, "hipGetErrorString"),
            pointer_get_attribute: need!(hip, "hipPointerGetAttribute"),
            mem_get_address_range: need!(hip, "hipMemGetAddressRange"),
        };
        let hsaf = HsaFns {
            status_string: need!(hsa, "hsa_status_string"),
            system_get_info: need!(hsa, "hsa_system_get_info"),
            // SAFETY: the signatures in hsa_ext_amd.h (rocm-6.3.0).
            export_dmabuf: unsafe { hsa.sym("hsa_amd_portable_export_dmabuf") },
            // SAFETY: as above.
            close_dmabuf: unsafe { hsa.sym("hsa_amd_portable_close_dmabuf") },
        };
        let mut driver = RocmDriver {
            hip,
            hsa,
            hipf,
            hsaf,
            hip_version: 0,
            hsa_version: (0, 0),
        };
        let mut v: c_int = 0;
        // SAFETY: valid out pointer.
        driver.hip_check("hipRuntimeGetVersion", unsafe { runtime_version(&mut v) })?;
        driver.hip_version = v;
        // HIP initialised ROCr already; this takes one more reference to the runtime,
        // never given back (it lives as long as the process, like HIP's).
        // SAFETY: hsa_init takes nothing.
        driver.hsa_check("hsa_init", unsafe { hsa_init() })?;
        let (mut major, mut minor) = (0u16, 0u16);
        // SAFETY: both attributes are uint16_t (hsa.h).
        driver.hsa_check("hsa_system_get_info", unsafe {
            (driver.hsaf.system_get_info)(
                HSA_SYSTEM_INFO_VERSION_MAJOR,
                (&mut major as *mut u16).cast(),
            )
        })?;
        // SAFETY: as above.
        driver.hsa_check("hsa_system_get_info", unsafe {
            (driver.hsaf.system_get_info)(
                HSA_SYSTEM_INFO_VERSION_MINOR,
                (&mut minor as *mut u16).cast(),
            )
        })?;
        driver.hsa_version = (major, minor);
        Ok(driver)
    }

    fn hip_check(&self, call: &str, rc: c_int) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: returns a static NUL-terminated string (or null).
        let s = unsafe { (self.hipf.get_error_string)(rc) };
        let text = if s.is_null() {
            format!("hipError_t {rc}")
        } else {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        };
        Err(Error::Gpu(format!("{call} failed: {text} ({rc})")))
    }

    fn hsa_check(&self, call: &str, rc: c_int) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: valid out pointer; the string is static in ROCr.
        let text = if unsafe { (self.hsaf.status_string)(rc, &mut s) } == 0 && !s.is_null() {
            // SAFETY: NUL-terminated, static.
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        } else {
            format!("hsa_status_t {rc:#x}")
        };
        Err(Error::Gpu(format!("{call} failed: {text}")))
    }

    fn pointer_attribute(&self, ptr: u64, attr: c_int) -> Result<u64> {
        let mut v = cell();
        // SAFETY: an 8-byte out cell holds every attribute asked for here (HIP 6.3 writes
        // uint32_t or int for these).
        self.hip_check("hipPointerGetAttribute", unsafe {
            (self.hipf.pointer_get_attribute)((&mut v as *mut u64).cast(), attr, ptr as *mut c_void)
        })?;
        Ok(v)
    }
}

impl DmabufCloser for RocmDriver {
    fn close_dmabuf(&self, fd: i32) -> Result<()> {
        let close = self
            .hsaf
            .close_dmabuf
            .ok_or_else(|| Error::Unsupported("no hsa_amd_portable_close_dmabuf".into()))?;
        // SAFETY: fd came from hsa_amd_portable_export_dmabuf, closed once.
        self.hsa_check("hsa_amd_portable_close_dmabuf", unsafe { close(fd) })
    }
}

impl GpuMemory for RocmDriver {
    fn vendor(&self) -> Vendor {
        Vendor::Rocm
    }

    fn describe(&self) -> String {
        format!(
            "HIP runtime {} ({}), ROCr {}.{} ({})",
            self.hip_version, self.hip.path, self.hsa_version.0, self.hsa_version.1, self.hsa.path
        )
    }

    fn allocation(&self, device: i32, ptr: u64) -> Result<Allocation> {
        // HIP renumbered hipMemoryType in 6.0; the values here are 6.x's.
        if self.hip_version < 60_000_000 {
            return Err(Error::Unsupported(format!(
                "HIP runtime {} is older than 6.0, whose pointer attributes this doesn't read",
                self.hip_version
            )));
        }
        let kind = self.pointer_attribute(ptr, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE)? & 0xffff_ffff;
        if kind != HIP_MEMORY_TYPE_DEVICE {
            return Err(Error::Unsupported(format!(
                "{ptr:#x} is not device memory (hipMemoryType {kind})"
            )));
        }
        let owner = as_i32(self.pointer_attribute(ptr, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL)?);
        if owner != device {
            return Err(Error::invalid(format!(
                "{ptr:#x} belongs to ROCm device {owner}, not {device}"
            )));
        }
        // A uint32_t in HIP 6.3 (a per-process counter, hipamd's getUniqueId()).
        let buffer_id = self.pointer_attribute(ptr, HIP_POINTER_ATTRIBUTE_BUFFER_ID)? & 0xffff_ffff;
        // Not HIP_POINTER_ATTRIBUTE_RANGE_SIZE: HIP 6.3 writes it as a uint32_t.
        let (mut base, mut size) = (std::ptr::null_mut::<c_void>(), 0usize);
        // SAFETY: valid out pointers.
        self.hip_check("hipMemGetAddressRange", unsafe {
            (self.hipf.mem_get_address_range)(&mut base, &mut size, ptr as *mut c_void)
        })?;
        Ok(Allocation {
            base: base as u64,
            size: size as u64,
            buffer_id,
            device,
            rdma_capable: None,
        })
    }

    fn dmabuf_support(&self, _device: i32) -> std::result::Result<(), String> {
        if self.hsaf.export_dmabuf.is_none() || self.hsaf.close_dmabuf.is_none() {
            return Err(format!(
                "{} has no hsa_amd_portable_export_dmabuf/close_dmabuf",
                self.hsa.path
            ));
        }
        let mut supported = cell();
        // SAFETY: the attribute is a bool (hsa.h); the cell is larger.
        let rc = unsafe {
            (self.hsaf.system_get_info)(
                HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED,
                (&mut supported as *mut u64).cast(),
            )
        };
        if let Err(e) = self.hsa_check("hsa_system_get_info(DMABUF_SUPPORTED)", rc) {
            return Err(e.to_string());
        }
        if supported & 0xff == 0 {
            return Err("ROCr reports no dma-buf support from the kernel driver \
                        (HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = false)"
                .into());
        }
        Ok(())
    }

    fn export_dmabuf(&self, alloc: &Allocation, start: u64, len: u64) -> Result<(i32, u64)> {
        let export = self
            .hsaf
            .export_dmabuf
            .ok_or_else(|| Error::Unsupported("no hsa_amd_portable_export_dmabuf".into()))?;
        if start < alloc.base || start.saturating_add(len) > alloc.end() {
            return Err(Error::invalid("a dma-buf export outside its allocation"));
        }
        let (mut fd, mut offset): (c_int, u64) = (-1, 0);
        // SAFETY: the range lies in one allocation (checked); valid out pointers.
        self.hsa_check("hsa_amd_portable_export_dmabuf", unsafe {
            export(start as *const c_void, len as usize, &mut fd, &mut offset)
        })?;
        Ok((fd, offset))
    }

    fn peer_memory_support(&self, _device: i32) -> std::result::Result<(), String> {
        Err("not offered on ROCm: its direct path is a dma-buf from ROCr".into())
    }

    fn prepare(&self, _alloc: &Allocation, _start: u64) -> Result<()> {
        // CU_POINTER_ATTRIBUTE_SYNC_MEMOPS is NVIDIA's rule; nothing in ROCm's dma-buf
        // export asks for an equivalent.
        Ok(())
    }
}

// ---- peer-memory kernel modules ---------------------------------------------------------

/// Looks for the kernel module that lets `ibv_reg_mr` take NVIDIA device pointers
/// (`nvidia_peermem`, or the older `nv_peer_mem`), under `root` (`/` on a real node, a
/// fixture directory in tests).
#[derive(Clone, Debug)]
pub struct PeerModules {
    root: PathBuf,
}

impl PeerModules {
    pub fn system() -> PeerModules {
        PeerModules::at("/")
    }

    pub fn at(root: impl Into<PathBuf>) -> PeerModules {
        PeerModules { root: root.into() }
    }

    /// The loaded module's name, or why there is none.
    pub fn nvidia(&self) -> std::result::Result<String, String> {
        const NAMES: [&str; 2] = ["nvidia_peermem", "nv_peer_mem"];
        for name in NAMES {
            if self.root.join("sys/module").join(name).is_dir() {
                return Ok(name.to_owned());
            }
        }
        if let Ok(modules) = std::fs::read_to_string(self.root.join("proc/modules")) {
            for line in modules.lines() {
                if let Some(name) = line.split_whitespace().next() {
                    if NAMES.contains(&name) {
                        return Ok(name.to_owned());
                    }
                }
            }
        }
        Err("neither nvidia_peermem nor nv_peer_mem is loaded (/sys/module, /proc/modules)".into())
    }
}

// ---- the mock ---------------------------------------------------------------------------

/// Host memory behind a mock allocation, kept alive by the allocation and by every
/// dma-buf or registration of it (as a real exporter keeps exported memory).
struct Backing {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

// SAFETY: plain heap memory; accesses go through the mock provider's or MockGpu's
// copies, ordered by their locks.
unsafe impl Send for Backing {}
// SAFETY: as for Send.
unsafe impl Sync for Backing {}

impl Drop for Backing {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in MockGpu::alloc, freed once.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

struct MockAlloc {
    va: u64,
    size: u64,
    buffer_id: u64,
    device: i32,
    backing: Arc<Backing>,
}

/// A dma-buf a [`MockGpu`] exported: `len` bytes from `offset` of a backing.
struct MockFd {
    backing: Arc<Backing>,
    offset: u64,
    len: u64,
    /// Registrations importing it (the mock provider counts them).
    imports: u32,
    gpu: std::sync::Weak<MockGpuShared>,
}

/// The mock fabric's view of mock device memory: what `reg_dmabuf_mr` and `reg_mr` on a
/// mock device address resolve to.
pub(crate) struct MockImport {
    /// The host address of the first byte.
    pub host: u64,
    /// Keeps the memory alive while registered.
    pub keep: Arc<dyn Any + Send + Sync>,
}

#[derive(Default)]
struct Registry {
    fds: HashMap<i32, MockFd>,
    /// Live mock allocations: (va, size, backing).
    vas: Vec<(u64, u64, Arc<Backing>)>,
}

fn registry() -> MutexGuard<'static, Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    lock(REGISTRY.get_or_init(|| Mutex::new(Registry::default())))
}

// Far from anything real: fds no real call ever sees, addresses no real allocation has.
static NEXT_FD: AtomicI32 = AtomicI32::new(1 << 20);
static NEXT_VA: AtomicU64 = AtomicU64::new(0x7e00_0000_0000);
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// The memory a mock dma-buf fd names, `len` bytes from `offset`; counts an import.
pub(crate) fn mock_import_dmabuf(fd: i32, offset: u64, len: u64) -> Option<MockImport> {
    let mut reg = registry();
    let f = reg.fds.get_mut(&fd)?;
    if offset.checked_add(len)? > f.len {
        return None;
    }
    f.imports += 1;
    Some(MockImport {
        host: f.backing.ptr as u64 + f.offset + offset,
        keep: f.backing.clone(),
    })
}

/// A registration of mock dma-buf `fd` went away.
pub(crate) fn mock_release_dmabuf(fd: i32) {
    if let Some(f) = registry().fds.get_mut(&fd) {
        f.imports = f.imports.saturating_sub(1);
    }
}

/// What `reg_mr` on `len` bytes at mock device address `va` reaches (the mock's peer
/// memory), or None if `va` isn't mock device memory.
pub(crate) fn mock_import_va(va: u64, len: u64) -> Option<MockImport> {
    let reg = registry();
    let (base, size, backing) = reg.vas.iter().find(|(b, s, _)| va >= *b && va < b + s)?;
    if va.checked_add(len)? > base + size {
        return None;
    }
    Some(MockImport {
        host: backing.ptr as u64 + (va - base),
        keep: backing.clone(),
    })
}

#[derive(Default)]
struct MockGpuState {
    allocs: Vec<MockAlloc>,
    dmabuf: Option<String>,
    peer: Option<String>,
    exported: u64,
    closed: HashMap<i32, u32>,
    sync_memops: Vec<u64>,
    violations: Vec<String>,
    fail_export: Option<String>,
}

struct MockGpuShared {
    st: Mutex<MockGpuState>,
}

/// A mock GPU for tests: allocations are host memory behind made-up device addresses
/// (the CPU can't use them as pointers, as with real device memory), with buffer IDs
/// that are never reused, dma-bufs the mock provider imports, and peer memory the mock
/// provider resolves. Records what a test needs to check: dma-buf descriptors closed
/// (and whether any was closed while still imported), SYNC_MEMOPS calls.
#[derive(Clone)]
pub struct MockGpu {
    shared: Arc<MockGpuShared>,
}

impl fmt::Debug for MockGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MockGpu")
    }
}

impl Default for MockGpu {
    fn default() -> Self {
        MockGpu::new()
    }
}

impl MockGpu {
    /// A mock GPU with both paths available.
    pub fn new() -> MockGpu {
        MockGpu {
            shared: Arc::new(MockGpuShared {
                st: Mutex::new(MockGpuState::default()),
            }),
        }
    }

    fn st(&self) -> MutexGuard<'_, MockGpuState> {
        lock(&self.shared.st)
    }

    /// Makes dma-buf export unavailable (`Some(reason)`) or available (`None`).
    pub fn set_dmabuf_unsupported(&self, why: Option<&str>) {
        self.st().dmabuf = why.map(str::to_owned);
    }

    /// Makes the driver refuse peer memory (`Some(reason)`) or allow it (`None`).
    pub fn set_peer_unsupported(&self, why: Option<&str>) {
        self.st().peer = why.map(str::to_owned);
    }

    /// Makes the next exports fail with `why` (or succeed again with `None`).
    pub fn fail_exports(&self, why: Option<&str>) {
        self.st().fail_export = why.map(str::to_owned);
    }

    /// A zeroed allocation of `size` bytes on `device`; its device address.
    pub fn alloc(&self, device: i32, size: u64) -> u64 {
        let span = size.div_ceil(2 << 20).max(1) * (2 << 20);
        let va = NEXT_VA.fetch_add(span + (2 << 20), Ordering::Relaxed);
        self.alloc_at(device, va, size)
    }

    fn alloc_at(&self, device: i32, va: u64, size: u64) -> u64 {
        let layout = std::alloc::Layout::from_size_align(size.max(1) as usize, 4096)
            .expect("a mock allocation's layout");
        // SAFETY: nonzero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "mock allocation of {size} bytes failed");
        let backing = Arc::new(Backing { ptr, layout });
        let buffer_id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        registry().vas.push((va, size, backing.clone()));
        self.st().allocs.push(MockAlloc {
            va,
            size,
            buffer_id,
            device,
            backing,
        });
        va
    }

    /// Frees the allocation at `va` (its memory lives on while a dma-buf or registration
    /// holds it, as a real exporter's does).
    pub fn free(&self, va: u64) {
        let mut st = self.st();
        st.allocs.retain(|a| a.va != va);
        registry().vas.retain(|(b, _, _)| *b != va);
    }

    /// Frees the allocation at `va` and allocates `size` bytes at the same address: a
    /// new allocation with a new buffer ID, as an allocator handing an address out again.
    pub fn realloc_same_address(&self, va: u64, size: u64) -> u64 {
        let device = self
            .st()
            .allocs
            .iter()
            .find(|a| a.va == va)
            .map_or(0, |a| a.device);
        self.free(va);
        self.alloc_at(device, va, size)
    }

    fn host(&self, va: u64, len: u64) -> *mut u8 {
        let st = self.st();
        let a = st
            .allocs
            .iter()
            .find(|a| va >= a.va && va + len <= a.va + a.size)
            .expect("inside a mock allocation");
        // SAFETY: inside the backing (checked).
        unsafe { a.backing.ptr.add((va - a.va) as usize) }
    }

    /// Copies `data` to device address `va`.
    pub fn write(&self, va: u64, data: &[u8]) {
        let p = self.host(va, data.len() as u64);
        // SAFETY: inside a live backing (host() checks); data is separate memory.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
    }

    /// Copies `len` bytes from device address `va`.
    pub fn read(&self, va: u64, len: usize) -> Vec<u8> {
        let p = self.host(va, len as u64);
        let mut out = vec![0u8; len];
        // SAFETY: as in write().
        unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), len) };
        out
    }

    /// dma-bufs exported so far.
    pub fn exported(&self) -> u64 {
        self.st().exported
    }

    /// Times descriptor `fd` was closed.
    pub fn closes(&self, fd: i32) -> u32 {
        self.st().closed.get(&fd).copied().unwrap_or(0)
    }

    /// Exported descriptors not closed yet.
    pub fn open_fds(&self) -> usize {
        let me = Arc::downgrade(&self.shared);
        registry()
            .fds
            .values()
            .filter(|f| f.gpu.ptr_eq(&me))
            .count()
    }

    /// Addresses SYNC_MEMOPS was set on, in order.
    pub fn sync_memops(&self) -> Vec<u64> {
        self.st().sync_memops.clone()
    }

    /// Descriptors closed while still imported, or twice.
    pub fn violations(&self) -> Vec<String> {
        self.st().violations.clone()
    }
}

impl DmabufCloser for MockGpu {
    fn close_dmabuf(&self, fd: i32) -> Result<()> {
        let removed = registry().fds.remove(&fd);
        let mut st = self.st();
        *st.closed.entry(fd).or_default() += 1;
        match removed {
            None => {
                st.violations.push(format!("dma-buf {fd} closed again"));
                Err(Error::Gpu(format!("dma-buf {fd} is not open")))
            }
            Some(f) if f.imports > 0 => {
                st.violations.push(format!(
                    "dma-buf {fd} closed while {} registration(s) still import it",
                    f.imports
                ));
                Ok(())
            }
            Some(_) => Ok(()),
        }
    }
}

impl GpuMemory for MockGpu {
    fn vendor(&self) -> Vendor {
        Vendor::Mock
    }

    fn describe(&self) -> String {
        "MOCK GPU (host memory)".into()
    }

    fn allocation(&self, device: i32, ptr: u64) -> Result<Allocation> {
        let st = self.st();
        let a = st
            .allocs
            .iter()
            .find(|a| ptr >= a.va && ptr < a.va + a.size)
            .ok_or_else(|| Error::Unsupported(format!("{ptr:#x} is not mock device memory")))?;
        if a.device != device {
            return Err(Error::invalid(format!(
                "{ptr:#x} belongs to mock device {}, not {device}",
                a.device
            )));
        }
        Ok(Allocation {
            base: a.va,
            size: a.size,
            buffer_id: a.buffer_id,
            device: a.device,
            rdma_capable: Some(true),
        })
    }

    fn dmabuf_support(&self, _device: i32) -> std::result::Result<(), String> {
        match &self.st().dmabuf {
            Some(why) => Err(why.clone()),
            None => Ok(()),
        }
    }

    fn export_dmabuf(&self, alloc: &Allocation, start: u64, len: u64) -> Result<(i32, u64)> {
        let mut st = self.st();
        if let Some(why) = &st.fail_export {
            return Err(Error::Gpu(why.clone()));
        }
        let a = st
            .allocs
            .iter()
            .find(|a| a.buffer_id == alloc.buffer_id)
            .ok_or_else(|| Error::Gpu("the allocation was freed".into()))?;
        if start < a.va || start + len > a.va + a.size {
            return Err(Error::invalid("a dma-buf export outside its allocation"));
        }
        let fd = NEXT_FD.fetch_add(1, Ordering::Relaxed);
        registry().fds.insert(
            fd,
            MockFd {
                backing: a.backing.clone(),
                offset: start - a.va,
                len,
                imports: 0,
                gpu: Arc::downgrade(&self.shared),
            },
        );
        st.exported += 1;
        Ok((fd, 0))
    }

    fn peer_memory_support(&self, _device: i32) -> std::result::Result<(), String> {
        match &self.st().peer {
            Some(why) => Err(why.clone()),
            None => Ok(()),
        }
    }

    fn prepare(&self, _alloc: &Allocation, start: u64) -> Result<()> {
        self.st().sync_memops.push(start);
        Ok(())
    }
}

/// Loads the GPU driver named `vendor` (`cuda` or `rocm`).
pub fn load(vendor: &str) -> Result<Arc<dyn GpuMemory>> {
    match vendor {
        "cuda" => Ok(Arc::new(CudaDriver::load(None)?)),
        "rocm" => Ok(Arc::new(RocmDriver::load(None, None)?)),
        other => Err(Error::invalid(format!(
            "unknown GPU vendor {other:?} (cuda or rocm)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_modules_are_found_in_sysfs_or_proc() {
        let dir = std::env::temp_dir().join(format!("gbv-peer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("proc")).unwrap();
        std::fs::write(dir.join("proc/modules"), "mlx5_ib 1 0 - Live 0x0\n").unwrap();
        let m = PeerModules::at(&dir);
        assert!(m.nvidia().unwrap_err().contains("nvidia_peermem"));
        std::fs::write(
            dir.join("proc/modules"),
            "mlx5_ib 1 0 - Live 0x0\nnv_peer_mem 16384 0 - Live 0x0\n",
        )
        .unwrap();
        assert_eq!(m.nvidia().unwrap(), "nv_peer_mem");
        std::fs::create_dir_all(dir.join("sys/module/nvidia_peermem")).unwrap();
        assert_eq!(m.nvidia().unwrap(), "nvidia_peermem");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_driver_is_an_unsupported_error() {
        let e = CudaDriver::load(Some("libgpubridge-no-such-driver.so")).unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)), "{e}");
        assert!(page_size() >= 4096);
    }

    #[test]
    fn mock_reallocation_at_the_same_address_has_a_new_buffer_id() {
        let gpu = MockGpu::new();
        let va = gpu.alloc(0, 1 << 20);
        let first = gpu.allocation(0, va + 100).unwrap();
        assert_eq!(first.base, va);
        let again = gpu.realloc_same_address(va, 1 << 20);
        assert_eq!(again, va);
        let second = gpu.allocation(0, va).unwrap();
        assert_ne!(first.buffer_id, second.buffer_id);
        assert!(gpu.allocation(1, va).is_err(), "another device's memory");
        gpu.write(va + 8, b"abc");
        assert_eq!(gpu.read(va + 8, 3), b"abc");
    }
}
