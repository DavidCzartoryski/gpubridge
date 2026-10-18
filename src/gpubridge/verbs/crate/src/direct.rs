//! Registering memory the NIC reads and writes in place (docs/VERBS_TRANSPORT.md, "GPU
//! memory" and "Registered host memory"): device memory through a dma-buf or peer
//! memory, and host buffers the caller owns, each through a bounded [`MrCache`].
//!
//! **Selection.** For device memory, in order, each only where its checks pass:
//!
//! 1. **dma-buf**: the driver says the device exports dma-bufs
//!    ([`GpuMemory::dmabuf_support`]), the range can be exported (page-aligned, inside one
//!    allocation), the export succeeds and `ibv_reg_dmabuf_mr` takes it.
//! 2. **peer memory** (NVIDIA only): `nvidia_peermem` or `nv_peer_mem` is loaded
//!    ([`PeerModules`]), the driver allows it ([`GpuMemory::peer_memory_support`]) and
//!    `ibv_reg_mr` on the device pointer succeeds.
//!
//! A registration is reported direct only once one of them has succeeded. Otherwise the
//! caller gets every reason (`"dma-buf: ...; peer memory: ..."`) and stages the tensor
//! through host memory instead, saying so; nothing pretends.
//!
//! **Identity and lifetime.** A device registration is cached under the allocation's
//! buffer ID (CUDA's and HIP's `BUFFER_ID`, never reused within a process), so the same
//! address in a later allocation is a different owner: the cache drops the old entry
//! (or refuses, if posted work still holds it). The cached range is the asked-for range
//! rounded out to 2 MiB inside the allocation, so neighbouring tensors of one allocator
//! segment share it. A dma-buf region owns its exported descriptor and closes it once,
//! after deregistering (`resources.rs`). What keeps the memory itself safe: a dma-buf
//! import pins the exported memory for as long as it is registered, and the peer-memory
//! module pins the GPU pages and revokes the registration if the allocation is freed;
//! the device only touches a registration while work posted with it is outstanding or a
//! peer holds a write grant for an outstanding receive (`endpoint.rs`).
//!
//! Host buffers are registered under an owner token the caller picks (its buffer's
//! identity), and the caller invalidates it before freeing the buffer.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::cache::{CacheConfig, CacheKey, CacheStats, Invalidated, MrCache};
use crate::error::{Error, Result};
use crate::gpu::{page_size, Allocation, GpuMemory, PeerModules};
use crate::resources::{DmabufCloser, DmabufFd, MemoryRegion, ProtectionDomain};
use crate::types::Access;

/// Device registrations are rounded out to this, inside their allocation.
pub const DEVICE_ROUND: u64 = 2 << 20;

/// How memory was registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// A host buffer the caller registered: the CPU can read and write it too.
    Host,
    DmaBuf,
    PeerMemory,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Host => "host",
            Method::DmaBuf => "dmabuf",
            Method::PeerMemory => "peer_memory",
        }
    }
}

/// A registration the endpoint can move data through in place. Only this module makes
/// them, so `cpu` (whether the CPU may read and write the memory through its address)
/// can be trusted by the endpoint's copies.
#[derive(Clone, Debug)]
pub struct Registered {
    region: MemoryRegion,
    method: Method,
    device: Option<i32>,
}

impl Registered {
    pub fn region(&self) -> &MemoryRegion {
        &self.region
    }

    pub fn method(&self) -> Method {
        self.method
    }

    /// The device, for device memory.
    pub fn device(&self) -> Option<i32> {
        self.device
    }

    /// Whether the CPU can use the memory (host memory), so a message that has to come
    /// through the endpoint's slots can be copied in or out of it.
    pub fn cpu(&self) -> bool {
        self.method == Method::Host
    }

    pub fn addr(&self) -> usize {
        self.region.addr()
    }

    pub fn len(&self) -> usize {
        self.region.len()
    }

    pub fn is_empty(&self) -> bool {
        self.region.is_empty()
    }
}

/// Which device paths may be tried, and the caches' bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectConfig {
    pub dmabuf: bool,
    pub peer_memory: bool,
    pub device_cache: CacheConfig,
    pub host_cache: CacheConfig,
}

impl Default for DirectConfig {
    fn default() -> Self {
        DirectConfig {
            dmabuf: true,
            peer_memory: true,
            device_cache: CacheConfig {
                max_entries: 256,
                max_bytes: 8 << 30,
            },
            host_cache: CacheConfig {
                max_entries: 64,
                max_bytes: 8 << 30,
            },
        }
    }
}

/// What the device paths can do on one device, found once.
#[derive(Clone, Debug)]
struct DeviceCaps {
    dmabuf: std::result::Result<(), String>,
    peer_memory: std::result::Result<String, String>,
}

#[derive(Default)]
struct Counts {
    dmabuf: u64,
    peer_memory: u64,
    host: u64,
    failures: u64,
    last_failure: Option<String>,
}

/// The GPU driver this registrar uses, once loaded.
struct Gpu {
    driver: Arc<dyn GpuMemory>,
    modules: PeerModules,
    caps: HashMap<i32, DeviceCaps>,
}

/// Registers device and host memory for one endpoint; see the module docs.
pub struct Registrar {
    config: DirectConfig,
    access: Access,
    device_cache: MrCache,
    host_cache: MrCache,
    gpu: Mutex<Option<Gpu>>,
    load_error: Mutex<Option<String>>,
    counts: Mutex<Counts>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Adapts a driver to the closer a dma-buf region keeps.
struct Closer(Arc<dyn GpuMemory>);

impl DmabufCloser for Closer {
    fn close_dmabuf(&self, fd: i32) -> Result<()> {
        self.0.close_dmabuf(fd)
    }
}

impl Registrar {
    /// Registrations on `pd` with `access` (local write, plus remote write when peers may
    /// write into them).
    pub fn new(pd: &ProtectionDomain, access: Access, config: DirectConfig) -> Result<Registrar> {
        Ok(Registrar {
            config,
            access,
            device_cache: MrCache::new(pd.clone(), config.device_cache)?,
            host_cache: MrCache::new(pd.clone(), config.host_cache)?,
            gpu: Mutex::new(None),
            load_error: Mutex::new(None),
            counts: Mutex::new(Counts::default()),
        })
    }

    /// Uses `driver` for device memory (and `modules` for the peer-memory module).
    pub fn set_gpu(&self, driver: Arc<dyn GpuMemory>, modules: PeerModules) {
        *lock(&self.gpu) = Some(Gpu {
            driver,
            modules,
            caps: HashMap::new(),
        });
        *lock(&self.load_error) = None;
    }

    /// Remembers why no driver could be loaded, for the reasons and telemetry.
    pub fn set_gpu_error(&self, why: String) {
        *lock(&self.load_error) = Some(why);
    }

    /// The driver's description, if one is loaded.
    pub fn gpu_description(&self) -> Option<String> {
        lock(&self.gpu).as_ref().map(|g| g.driver.describe())
    }

    fn caps(gpu: &mut Gpu, device: i32) -> DeviceCaps {
        let driver = gpu.driver.clone();
        let modules = gpu.modules.clone();
        gpu.caps
            .entry(device)
            .or_insert_with(|| DeviceCaps {
                dmabuf: driver.dmabuf_support(device),
                peer_memory: driver
                    .peer_memory_support(device)
                    .and_then(|()| modules.nvidia()),
            })
            .clone()
    }

    /// What each device path can do on `device`, as JSON: `{"dmabuf": "ok" or the reason,
    /// "peer_memory": "ok (<module>)" or the reason}`.
    pub fn capabilities(&self, device: i32) -> String {
        let mut g = lock(&self.gpu);
        let Some(gpu) = g.as_mut() else {
            let why = lock(&self.load_error)
                .clone()
                .unwrap_or_else(|| "no GPU driver loaded".into());
            let q = json_str(&why);
            return format!("{{\"driver\":null,\"dmabuf\":{q},\"peer_memory\":{q}}}");
        };
        let caps = Self::caps(gpu, device);
        let text = |r: std::result::Result<String, String>, on: bool| match (on, r) {
            (false, _) => "disabled by the settings".to_owned(),
            (true, Ok(s)) => s,
            (true, Err(e)) => e,
        };
        format!(
            "{{\"driver\":{},\"dmabuf\":{},\"peer_memory\":{}}}",
            json_str(&gpu.driver.describe()),
            json_str(&text(caps.dmabuf.map(|()| "ok".into()), self.config.dmabuf)),
            json_str(&text(
                caps.peer_memory.map(|m| format!("ok ({m})")),
                self.config.peer_memory
            )),
        )
    }

    fn failed(&self, why: String) -> String {
        let mut c = lock(&self.counts);
        c.failures += 1;
        c.last_failure = Some(why.clone());
        why
    }

    /// A direct registration covering `len` bytes at device address `ptr` on `device`,
    /// or every reason none was possible.
    pub fn register_device(
        &self,
        device: i32,
        ptr: u64,
        len: u64,
    ) -> std::result::Result<Registered, String> {
        if len == 0 {
            return Err("an empty range needs no registration".into());
        }
        let mut g = lock(&self.gpu);
        let Some(gpu) = g.as_mut() else {
            let why = lock(&self.load_error)
                .clone()
                .unwrap_or_else(|| "no GPU driver loaded".into());
            drop(g);
            return Err(self.failed(why));
        };
        let driver = gpu.driver.clone();
        let caps = Self::caps(gpu, device);
        drop(g);
        let alloc = match driver.allocation(device, ptr) {
            Ok(a) => a,
            Err(e) => return Err(self.failed(format!("{ptr:#x}: {e}"))),
        };
        if ptr < alloc.base || ptr.saturating_add(len) > alloc.end() {
            return Err(self.failed(format!(
                "{len} bytes at {ptr:#x} run past their allocation ({} bytes at {:#x})",
                alloc.size, alloc.base
            )));
        }
        let start = (ptr - ptr % DEVICE_ROUND).max(alloc.base);
        let end = ptr
            .saturating_add(len)
            .div_ceil(DEVICE_ROUND)
            .saturating_mul(DEVICE_ROUND)
            .min(alloc.end());
        let want = CacheKey {
            addr: ptr as usize,
            len: len as usize,
            access: self.access,
            owner: alloc.buffer_id,
        };
        let cover = CacheKey {
            addr: start as usize,
            len: (end - start) as usize,
            ..want
        };
        let mut reasons: Vec<String> = Vec::new();
        let mut used = None;
        let got = self.device_cache.get_with(want, cover, || {
            match self.try_dmabuf(&driver, &caps, &alloc, start, end - start) {
                Ok(region) => {
                    used = Some(Method::DmaBuf);
                    return Ok(region);
                }
                Err(why) => reasons.push(format!("dma-buf: {why}")),
            }
            match self.try_peer(&driver, &caps, &alloc, start, end - start) {
                Ok(region) => {
                    used = Some(Method::PeerMemory);
                    Ok(region)
                }
                Err(why) => {
                    reasons.push(format!("peer memory: {why}"));
                    Err(Error::Unsupported(reasons.join("; ")))
                }
            }
        });
        match got {
            Ok(region) => {
                let method = if region.is_dmabuf() {
                    Method::DmaBuf
                } else {
                    Method::PeerMemory
                };
                if let Some(m) = used {
                    // A new registration: the driver's per-registration step (SYNC_MEMOPS).
                    if let Err(e) = driver.prepare(&alloc, start) {
                        self.device_cache.invalidate_owner(alloc.buffer_id);
                        return Err(self.failed(format!(
                            "{}: registered, but preparing the memory failed: {e}",
                            m.name()
                        )));
                    }
                    let mut c = lock(&self.counts);
                    match m {
                        Method::DmaBuf => c.dmabuf += 1,
                        _ => c.peer_memory += 1,
                    }
                }
                Ok(Registered {
                    region,
                    method,
                    device: Some(device),
                })
            }
            Err(Error::Unsupported(why)) if !reasons.is_empty() => Err(self.failed(why)),
            Err(e) => Err(self.failed(format!("registration cache: {e}"))),
        }
    }

    fn try_dmabuf(
        &self,
        driver: &Arc<dyn GpuMemory>,
        caps: &DeviceCaps,
        alloc: &Allocation,
        start: u64,
        len: u64,
    ) -> std::result::Result<MemoryRegion, String> {
        if !self.config.dmabuf {
            return Err("disabled by the settings".into());
        }
        caps.dmabuf.clone()?;
        let page = page_size();
        if start % page != 0 || len % page != 0 {
            return Err(format!(
                "the allocation's range ({len} bytes at {start:#x}) isn't page-aligned"
            ));
        }
        let (fd, offset) = driver
            .export_dmabuf(alloc, start, len)
            .map_err(|e| e.to_string())?;
        let fd = DmabufFd::new(fd, Arc::new(Closer(driver.clone())));
        self.device_cache
            .pd()
            .register_dmabuf(fd, offset, len as usize, start, self.access)
            .map_err(|e| format!("ibv_reg_dmabuf_mr: {e}"))
    }

    fn try_peer(
        &self,
        driver: &Arc<dyn GpuMemory>,
        caps: &DeviceCaps,
        alloc: &Allocation,
        start: u64,
        len: u64,
    ) -> std::result::Result<MemoryRegion, String> {
        if !self.config.peer_memory {
            return Err("disabled by the settings".into());
        }
        caps.peer_memory.clone()?;
        if alloc.rdma_capable == Some(false) {
            return Err(
                "the driver says this allocation can't be used for GPUDirect RDMA \
                        (CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 0)"
                    .into(),
            );
        }
        let _ = driver;
        // SAFETY: the address is device memory, never dereferenced by the CPU: the peer
        // memory module resolves it to GPU pages and pins them for the registration's
        // lifetime, revoking the registration if the allocation is freed first, so the
        // device can't reach memory handed to someone else.
        unsafe {
            self.device_cache
                .pd()
                .register(start as usize as *mut u8, len as usize, self.access)
        }
        .map_err(|e| format!("ibv_reg_mr on device memory: {e}"))
    }

    /// Registers `len` bytes of host memory at `ptr`, which the caller identifies as
    /// `owner` (a token that changes whenever that memory is a different buffer).
    ///
    /// # Safety
    ///
    /// The memory must stay valid until the caller has invalidated `owner`
    /// ([`invalidate_host`](Registrar::invalidate_host)) and every handle to the
    /// registration (the returned one, and work posted with it) is gone.
    pub unsafe fn register_host(&self, ptr: usize, len: usize, owner: u64) -> Result<Registered> {
        let key = CacheKey {
            addr: ptr,
            len,
            access: self.access,
            owner,
        };
        // SAFETY: the caller's contract, passed on.
        let region = unsafe { self.host_cache.get(key) }.map_err(|e| {
            self.failed(format!("registering host memory: {e}"));
            e
        })?;
        lock(&self.counts).host += 1;
        Ok(Registered {
            region,
            method: Method::Host,
            device: None,
        })
    }

    /// Drops host registrations of `owner` (its buffer is being freed).
    pub fn invalidate_host(&self, owner: u64) -> Invalidated {
        self.host_cache.invalidate_owner(owner)
    }

    /// Drops every cached registration (they stay registered while still in use).
    pub fn clear(&self) -> (Invalidated, Invalidated) {
        (self.device_cache.clear(), self.host_cache.clear())
    }

    pub fn device_stats(&self) -> CacheStats {
        self.device_cache.stats()
    }

    pub fn host_stats(&self) -> CacheStats {
        self.host_cache.stats()
    }

    /// Telemetry as a JSON object.
    pub fn telemetry(&self) -> String {
        let c = lock(&self.counts);
        let mut o = String::from("{");
        let driver = self.gpu_description();
        let _ = write!(
            o,
            "\"gpu_driver\":{},\"gpu_driver_error\":{},\"dmabuf_enabled\":{},\
             \"peer_memory_enabled\":{},\"registered\":{{\"dmabuf\":{},\"peer_memory\":{},\
             \"host\":{}}},\"registration_failures\":{},\"last_registration_failure\":{},",
            driver.as_deref().map_or("null".into(), json_str),
            lock(&self.load_error)
                .as_deref()
                .map_or("null".into(), json_str),
            self.config.dmabuf,
            self.config.peer_memory,
            c.dmabuf,
            c.peer_memory,
            c.host,
            c.failures,
            c.last_failure.as_deref().map_or("null".into(), json_str),
        );
        for (name, cache) in [
            ("device_cache", &self.device_cache),
            ("host_cache", &self.host_cache),
        ] {
            let s = cache.stats();
            let _ = write!(
                o,
                "\"{name}\":{{\"entries\":{},\"bytes\":{},\"hits\":{},\"misses\":{},\
                 \"evictions\":{},\"stale\":{},\"invalidations\":{},\"refused\":{}}},",
                cache.len(),
                cache.bytes(),
                s.hits,
                s.misses,
                s.evictions,
                s.stale,
                s.invalidations,
                s.refused
            );
        }
        o.pop();
        o.push('}');
        o
    }
}

/// A JSON string literal.
pub(crate) fn json_str(v: &str) -> String {
    let mut out = String::from("\"");
    for ch in v.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
