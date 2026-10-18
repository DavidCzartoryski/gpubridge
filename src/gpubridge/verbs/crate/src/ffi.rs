//! The C interface Python's `gpubridge.verbs` loads (docs/VERBS_TRANSPORT.md): an
//! [`Endpoint`] behind an opaque handle, with the same shape as the multitcp library's
//! (`gb_tcp_*`), so the Python collectives are shared.
//!
//! Every function catches panics. Errors are a negative return value and a message
//! [`gb_verbs_last_error`] copies out.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
}

impl Handle {
    fn set_error(&self, e: impl ToString) {
        *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = e.to_string();
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

/// # Safety
/// `handle` must come from gb_verbs_create and not be closed yet.
unsafe fn handle<'a>(h: *mut c_void) -> &'a Handle {
    // SAFETY: the caller's contract.
    unsafe { &*(h as *const Handle) }
}

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
        let h = unsafe { handle(h) };
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
        let (h, info) = unsafe { (handle(h), CStr::from_ptr(info).to_string_lossy()) };
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
        let h = unsafe { handle(h) };
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
        let h = unsafe { handle(h) };
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

/// 0 once every operation in `ids` finished; -2 timed out, -3 failed (see
/// gb_verbs_last_error).
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
        let h = unsafe { handle(h) };
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
        let h = unsafe { handle(h) };
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
        let h = unsafe { handle(h) };
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
        let h = unsafe { handle(h) };
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

/// Closes and frees the endpoint: 0, or 1 if work didn't drain within `timeout` and its
/// slots were leaked (never the caller's memory), with the reason in `error`.
///
/// # Safety
/// `h` from gb_verbs_create, not used again afterwards; `error` holds `error_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_close(
    h: *mut c_void,
    timeout: f64,
    error: *mut c_char,
    error_size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
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
