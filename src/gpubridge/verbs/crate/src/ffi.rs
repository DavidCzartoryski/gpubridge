//! The C interface Python's `gpubridge.verbs` loads (docs/VERBS_TRANSPORT.md): an
//! [`Endpoint`] behind an opaque handle, with the same shape as the multitcp library's
//! (`gb_tcp_*`), so the Python collectives are shared.
//!
//! Every function catches panics. Errors are a negative return value and a message
//! [`gb_verbs_last_error`] copies out. A null handle is refused, never dereferenced:
//! each function returns its failure value (0 from `gb_verbs_rndv_bytes`).
//!
//! Registrations (`gb_verbs_reg_device`, `gb_verbs_reg_host`) come back as handles the
//! caller passes to `gb_verbs_send_reg`/`gb_verbs_recv_reg` and gives back with
//! `gb_verbs_release`; a handle keeps its registration from being evicted, and closing
//! the endpoint releases what is left.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::direct::{Method, Registered};
use crate::endpoint::{Config, Endpoint, WaitError};
use crate::error::{Error, Result};
use crate::protocol::PROTOCOL_VERSION;
use crate::provider::Provider;

const FAILED: i64 = -1;
const TIMED_OUT: c_int = -2;
const PEER_FAILED: c_int = -3;

struct Handle {
    endpoint: Endpoint,
    last_error: Mutex<String>,
    regions: Mutex<HashMap<u64, Registered>>,
    next_region: AtomicU64,
}

impl Handle {
    fn set_error(&self, e: impl ToString) {
        *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = e.to_string();
    }

    /// Files `reg` under a new handle; writes the handle, the registration's address and
    /// its length to `out[0..3]`.
    ///
    /// # Safety
    /// `out` points to 3 writable u64s.
    unsafe fn file(&self, reg: Registered, out: *mut u64) -> c_int {
        if out.is_null() {
            return FAILED as c_int;
        }
        let id = self.next_region.fetch_add(1, Ordering::Relaxed) + 1;
        let (addr, len) = (reg.addr() as u64, reg.len() as u64);
        self.regions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, reg);
        // SAFETY: 3 writable u64s (the caller's contract).
        unsafe {
            *out = id;
            *out.add(1) = addr;
            *out.add(2) = len;
        }
        0
    }

    fn region(&self, id: u64) -> Option<Registered> {
        self.regions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&id)
            .cloned()
    }
}

fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

fn write_c(text: &str, out: *mut c_char, size: usize) -> bool {
    if out.is_null() || size == 0 {
        return false;
    }
    let bytes = text.as_bytes();
    let n = bytes.len().min(size - 1);
    // SAFETY: the caller passes a buffer of `size` bytes; n + 1 <= size are written.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
    }
    n == bytes.len()
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(if value.is_finite() && value > 0.0 {
        value.min(1e9)
    } else {
        0.0
    })
}

/// The handle, or None for a null one, which every function refuses (it returns its
/// failure value) rather than dereference.
///
/// # Safety
/// `h` must be null, or come from gb_verbs_create and not be closed yet.
unsafe fn handle<'a>(h: *mut c_void) -> Option<&'a Handle> {
    // SAFETY: the caller's contract.
    unsafe { (h as *const Handle).as_ref() }
}

const NULL_HANDLE: &str = "a null handle";

/// The provider the settings name: `provider=libibverbs` (the default) or
/// `provider=test-fabric` with `fabric_dir=` (a test build only).
fn provider(settings: &str) -> Result<Arc<dyn Provider>> {
    let get = |key: &str| {
        settings
            .lines()
            .filter_map(|l| l.trim().split_once('='))
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.to_owned())
    };
    match get("provider").as_deref().unwrap_or("libibverbs") {
        "libibverbs" => {
            #[cfg(feature = "ibverbs")]
            {
                Ok(Arc::new(crate::ibv::IbvProvider::new()?))
            }
            #[cfg(not(feature = "ibverbs"))]
            {
                Err(Error::Unsupported(
                    "this library was built without libibverbs (the ibverbs feature: Linux \
                     with rdma-core's headers)"
                        .into(),
                ))
            }
        }
        "test-fabric" => {
            #[cfg(feature = "test-fabric")]
            {
                let dir = get("fabric_dir")
                    .ok_or_else(|| Error::invalid("the test fabric needs fabric_dir"))?;
                Ok(Arc::new(crate::uds::UdsProvider::open(dir.as_ref())?))
            }
            #[cfg(not(feature = "test-fabric"))]
            {
                Err(Error::Unsupported(
                    "this library has no test fabric (a test-only build feature)".into(),
                ))
            }
        }
        other => Err(Error::invalid(format!("unknown provider {other:?}"))),
    }
}

#[no_mangle]
pub extern "C" fn gb_verbs_protocol() -> u32 {
    PROTOCOL_VERSION
}

/// An endpoint for member `index` of `members`, from `settings` (`key=value` lines,
/// [`Config::parse`] plus `provider` and `fabric_dir`), or null with the reason in
/// `error`.
///
/// # Safety
/// `settings` NUL-terminated; `error` holds `error_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_create(
    members: c_int,
    index: c_int,
    settings: *const c_char,
    error: *mut c_char,
    error_size: usize,
) -> *mut c_void {
    guard(std::ptr::null_mut(), || {
        // SAFETY: NUL-terminated (the caller's contract).
        let settings = unsafe { CStr::from_ptr(settings) }
            .to_string_lossy()
            .into_owned();
        let made = (|| {
            let members = usize::try_from(members).map_err(|_| Error::invalid("members"))?;
            let index = usize::try_from(index).map_err(|_| Error::invalid("index"))?;
            let config = Config::parse(&settings)?;
            Endpoint::new(provider(&settings)?, members, index, config)
        })();
        match made {
            Ok(endpoint) => Box::into_raw(Box::new(Handle {
                endpoint,
                last_error: Mutex::new(String::new()),
                regions: Mutex::new(HashMap::new()),
                next_region: AtomicU64::new(0),
            })) as *mut c_void,
            Err(e) => {
                write_c(&e.to_string(), error, error_size);
                std::ptr::null_mut()
            }
        }
    })
}

/// This member's handshake line into `out`: 0, or -1 if `out` is too small.
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_local_info(
    h: *mut c_void,
    out: *mut c_char,
    size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        if write_c(&h.endpoint.local_info(), out, size) {
            0
        } else {
            FAILED as c_int
        }
    })
}

/// Connects to `member` from its handshake line: 0, or -1 (see gb_verbs_last_error).
///
/// # Safety
/// `h` from gb_verbs_create; `info` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_connect(
    h: *mut c_void,
    member: c_int,
    info: *const c_char,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        // SAFETY: NUL-terminated (the caller's contract).
        let info = unsafe { CStr::from_ptr(info) }.to_string_lossy();
        let member = usize::try_from(member).unwrap_or(usize::MAX);
        match h.endpoint.connect(member, &info) {
            Ok(()) => 0,
            Err(e) => {
                h.set_error(e);
                FAILED as c_int
            }
        }
    })
}

/// An operation id (positive), or -1 (see gb_verbs_last_error).
///
/// # Safety
/// `h` from gb_verbs_create; `size` bytes at `data` valid (and unchanged) until the
/// operation finishes, fails, or gb_verbs_abort / gb_verbs_close returns.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_send(
    h: *mut c_void,
    member: c_int,
    data: *const c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let member = usize::try_from(member).unwrap_or(usize::MAX);
        // SAFETY: the caller keeps the memory valid as Endpoint::send requires.
        match unsafe { h.endpoint.send(member, data as usize, size as u64, tag) } {
            Ok(id) => id,
            Err(e) => {
                h.set_error(e);
                FAILED
            }
        }
    })
}

/// As gb_verbs_send, for a receive into `size` bytes at `data`.
///
/// # Safety
/// As gb_verbs_send; nothing else may use the memory meanwhile.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_recv(
    h: *mut c_void,
    member: c_int,
    data: *mut c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let member = usize::try_from(member).unwrap_or(usize::MAX);
        // SAFETY: the caller keeps the memory valid as Endpoint::recv requires.
        match unsafe { h.endpoint.recv(member, data as usize, size as u64, tag) } {
            Ok(id) => id,
            Err(e) => {
                h.set_error(e);
                FAILED
            }
        }
    })
}

/// The rendezvous threshold the endpoint was made with (0: off).
///
/// # Safety
/// `h` from gb_verbs_create.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_rndv_bytes(h: *mut c_void) -> u64 {
    guard(0, || {
        // SAFETY: the caller's contract.
        unsafe { handle(h) }.map_or(0, |h| h.endpoint.rndv_bytes())
    })
}

/// Loads the GPU driver for `vendor` (`cuda` or `rocm`) at run time: 0 with its
/// description in `out`, or -1 (see gb_verbs_last_error).
///
/// # Safety
/// `h` from gb_verbs_create; `vendor` NUL-terminated; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_gpu_load(
    h: *mut c_void,
    vendor: *const c_char,
    out: *mut c_char,
    size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        // SAFETY: NUL-terminated (the caller's contract).
        let vendor = unsafe { CStr::from_ptr(vendor) }.to_string_lossy();
        match h.endpoint.load_gpu(&vendor) {
            Ok(text) => {
                write_c(&text, out, size);
                0
            }
            Err(e) => {
                h.set_error(e);
                FAILED as c_int
            }
        }
    })
}

/// What the direct paths can do on `device`, JSON into `out`: its length, or -(needed + 1).
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_gpu_caps(
    h: *mut c_void,
    device: c_int,
    out: *mut c_char,
    size: usize,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let text = h.endpoint.gpu_capabilities(device);
        if write_c(&text, out, size) {
            text.len() as i64
        } else {
            -(text.len() as i64 + 1)
        }
    })
}

/// Registers `len` bytes of device memory at `ptr` on `device` directly (dma-buf, then
/// peer memory): 0 with `out[0..3]` = handle, registration address, registration length
/// and `method` = 1 (dma-buf) or 2 (peer memory); or -1 with every reason in
/// gb_verbs_last_error (the caller stages instead).
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds 3 u64s; `method` null or writable.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_reg_device(
    h: *mut c_void,
    device: c_int,
    ptr: u64,
    len: u64,
    out: *mut u64,
    method: *mut c_int,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        match h.endpoint.register_device(device, ptr, len) {
            Ok(reg) => {
                if !method.is_null() {
                    let m = if reg.method() == Method::DmaBuf { 1 } else { 2 };
                    // SAFETY: writable (the caller's contract).
                    unsafe { *method = m };
                }
                // SAFETY: out holds 3 u64s (the caller's contract).
                unsafe { h.file(reg, out) }
            }
            Err(why) => {
                h.set_error(why);
                FAILED as c_int
            }
        }
    })
}

/// Registers `len` bytes of host memory at `ptr`, identified by the caller as `owner`:
/// 0 with `out[0..3]` as for gb_verbs_reg_device, or -1.
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds 3 u64s. The memory stays valid until the caller
/// has called gb_verbs_invalidate_host for `owner`, released every handle to it and every
/// operation on it is done, failed, or the endpoint closed.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_reg_host(
    h: *mut c_void,
    ptr: *mut c_void,
    len: usize,
    owner: u64,
    out: *mut u64,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        // SAFETY: the caller keeps the memory valid as Endpoint::register_host requires.
        match unsafe { h.endpoint.register_host(ptr as usize, len, owner) } {
            // SAFETY: out holds 3 u64s.
            Ok(reg) => unsafe { h.file(reg, out) },
            Err(e) => {
                h.set_error(e);
                FAILED as c_int
            }
        }
    })
}

/// Gives back a registration handle: 0, or -1 for an unknown one.
///
/// # Safety
/// `h` from gb_verbs_create.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_release(h: *mut c_void, region: u64) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        let gone = h
            .regions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&region);
        if gone.is_some() {
            0
        } else {
            FAILED as c_int
        }
    })
}

/// Drops the host registrations of `owner`: how many of them were still in use (they
/// stay registered until that use ends).
///
/// # Safety
/// `h` from gb_verbs_create.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_invalidate_host(h: *mut c_void, owner: u64) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        h.endpoint.invalidate_host(owner).in_use as c_int
    })
}

/// As gb_verbs_send, from `size` bytes at `offset` of registration `region`.
///
/// # Safety
/// `h` from gb_verbs_create; the memory unchanged until the operation finishes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_send_reg(
    h: *mut c_void,
    member: c_int,
    region: u64,
    offset: u64,
    size: u64,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let member = usize::try_from(member).unwrap_or(usize::MAX);
        let Some(reg) = h.region(region) else {
            h.set_error(format!("no registration {region}"));
            return FAILED;
        };
        match h
            .endpoint
            .send_registered(member, &reg, offset as usize, size, tag)
        {
            Ok(id) => id,
            Err(e) => {
                h.set_error(e);
                FAILED
            }
        }
    })
}

/// As gb_verbs_recv, into `size` bytes at `offset` of registration `region`. A peer may
/// be granted an RDMA WRITE into exactly those bytes (rendezvous); a failure anywhere on
/// the endpoint, as a failed gb_verbs_wait reports it, moves every queue pair to the error
/// state, so from then on no peer can write them (as after gb_verbs_abort).
///
/// # Safety
/// `h` from gb_verbs_create; nothing else uses the memory until the operation finishes,
/// fails, or the endpoint is aborted or closed; the registration's memory stays allocated
/// until gb_verbs_close (a write the device had already begun may still land).
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_recv_reg(
    h: *mut c_void,
    member: c_int,
    region: u64,
    offset: u64,
    size: u64,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let member = usize::try_from(member).unwrap_or(usize::MAX);
        let Some(reg) = h.region(region) else {
            h.set_error(format!("no registration {region}"));
            return FAILED;
        };
        match h
            .endpoint
            .recv_registered(member, &reg, offset as usize, size, tag)
        {
            Ok(id) => id,
            Err(e) => {
                h.set_error(e);
                FAILED
            }
        }
    })
}

/// 0 once every operation in `ids` finished; -2 timed out, -3 failed (see
/// gb_verbs_last_error). A failure fails the whole endpoint, as gb_verbs_abort does: every
/// operation is cancelled and every queue pair is in the error state, so no peer can
/// write a receive's granted window any more; later calls fail. A timeout fails nothing.
///
/// # Safety
/// `h` from gb_verbs_create; `ids` holds `count` ids.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_wait(
    h: *mut c_void,
    ids: *const i64,
    count: c_int,
    timeout: f64,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        let ids = match usize::try_from(count) {
            // SAFETY: ids holds count entries (the caller's contract).
            Ok(n) if n > 0 => unsafe { std::slice::from_raw_parts(ids, n) },
            _ => &[],
        };
        match h.endpoint.wait(ids, seconds(timeout)) {
            Ok(()) => 0,
            Err(WaitError::TimedOut(e)) => {
                h.set_error(e);
                TIMED_OUT
            }
            Err(WaitError::Failed(e)) => {
                h.set_error(e);
                PEER_FAILED
            }
        }
    })
}

/// One operation's outcome as JSON: length, or -(needed + 1) if `out` is too small.
/// Valid after a failed wait or abort, until close; a successful wait forgets its IDs.
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_op_outcome(
    h: *mut c_void,
    id: i64,
    out: *mut c_char,
    size: usize,
) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let text = h.endpoint.operation_outcome(id);
        if write_c(&text, out, size) {
            text.len() as i64
        } else {
            -(text.len() as i64 + 1)
        }
    })
}

/// The last error's text into `out` (truncated to fit).
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_last_error(
    h: *mut c_void,
    out: *mut c_char,
    size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            write_c(NULL_HANDLE, out, size);
            return FAILED as c_int;
        };
        let text = h
            .last_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        write_c(&text, out, size);
        0
    })
}

/// Telemetry JSON into `out`: its length, or -(needed + 1) if `out` is too small.
///
/// # Safety
/// `h` from gb_verbs_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_telemetry(h: *mut c_void, out: *mut c_char, size: usize) -> i64 {
    guard(FAILED, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED;
        };
        let text = h.endpoint.telemetry();
        if write_c(&text, out, size) {
            text.len() as i64
        } else {
            -(text.len() as i64 + 1)
        }
    })
}

/// Fails the endpoint now (see Endpoint::abort): once this returns nothing reads or
/// writes memory any operation named. Later calls fail. Returns 0.
///
/// # Safety
/// `h` from gb_verbs_create; `why` null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_abort(h: *mut c_void, why: *const c_char) -> c_int {
    guard(FAILED as c_int, || {
        // SAFETY: the caller's contract.
        let Some(h) = (unsafe { handle(h) }) else {
            return FAILED as c_int;
        };
        let why = if why.is_null() {
            "aborted".into()
        } else {
            // SAFETY: NUL-terminated (the caller's contract).
            unsafe { CStr::from_ptr(why) }
                .to_string_lossy()
                .into_owned()
        };
        h.endpoint.abort(&why);
        0
    })
}

/// Closes and frees the endpoint: 0, or 1 if work didn't drain within `timeout`, with the
/// reason in `error`. Then that work was leaked rather than released: its slots, and any
/// registration of the caller's memory it held (a registered send or receive, a write
/// window), stay registered for good, so the caller must never free that memory. Plain
/// caller memory is never held.
///
/// It frees the handle once the endpoint has closed, so no other call may be in the
/// library with `h`, or enter it, from when this starts: abort first so a waiter returns,
/// and if one doesn't, leak the handle rather than close it (Python's `Endpoint.close`
/// does both).
///
/// # Safety
/// `h` from gb_verbs_create, not used again afterwards (nor by any other thread meanwhile);
/// `error` holds `error_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_close(
    h: *mut c_void,
    timeout: f64,
    error: *mut c_char,
    error_size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        if h.is_null() {
            write_c(NULL_HANDLE, error, error_size);
            return FAILED as c_int;
        }
        // SAFETY: from gb_verbs_create and closed once (the caller's contract).
        let h = unsafe { Box::from_raw(h as *mut Handle) };
        match h.endpoint.close(seconds(timeout)) {
            Ok(()) => 0,
            Err(e) => {
                write_c(&e.to_string(), error, error_size);
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null handle (say, a caller that raced a close) is refused by every function,
    /// never dereferenced.
    #[test]
    fn every_function_refuses_a_null_handle() {
        let null = std::ptr::null_mut();
        let mut buf = [0 as c_char; 64];
        let (out, size) = (buf.as_mut_ptr(), buf.len());
        let mut regs = [0u64; 3];
        let ids = [1i64];
        let text = c"x".as_ptr();
        // SAFETY: every pointer but the handle is valid; the null handle is what each
        // function must refuse.
        unsafe {
            assert_eq!(gb_verbs_local_info(null, out, size), -1);
            assert_eq!(gb_verbs_connect(null, 1, text), -1);
            assert_eq!(gb_verbs_send(null, 1, out.cast(), 4, 1), -1);
            assert_eq!(gb_verbs_recv(null, 1, out.cast(), 4, 1), -1);
            assert_eq!(gb_verbs_rndv_bytes(null), 0);
            assert_eq!(gb_verbs_gpu_load(null, text, out, size), -1);
            assert_eq!(gb_verbs_gpu_caps(null, 0, out, size), -1);
            let method = std::ptr::null_mut();
            assert_eq!(
                gb_verbs_reg_device(null, 0, 0, 4, regs.as_mut_ptr(), method),
                -1
            );
            assert_eq!(
                gb_verbs_reg_host(null, out.cast(), 4, 1, regs.as_mut_ptr()),
                -1
            );
            assert_eq!(gb_verbs_release(null, 1), -1);
            assert_eq!(gb_verbs_invalidate_host(null, 1), -1);
            assert_eq!(gb_verbs_send_reg(null, 1, 1, 0, 4, 1), -1);
            assert_eq!(gb_verbs_recv_reg(null, 1, 1, 0, 4, 1), -1);
            assert_eq!(gb_verbs_wait(null, ids.as_ptr(), 1, 0.0), -1);
            assert_eq!(gb_verbs_op_outcome(null, 1, out, size), -1);
            assert_eq!(gb_verbs_telemetry(null, out, size), -1);
            assert_eq!(gb_verbs_abort(null, text), -1);
            assert_eq!(gb_verbs_last_error(null, out, size), -1);
            assert_eq!(CStr::from_ptr(out).to_str(), Ok("a null handle"));
            buf[0] = 0;
            assert_eq!(gb_verbs_close(null, 0.0, buf.as_mut_ptr(), size), -1);
            assert_eq!(CStr::from_ptr(buf.as_ptr()).to_str(), Ok("a null handle"));
        }
    }
}
