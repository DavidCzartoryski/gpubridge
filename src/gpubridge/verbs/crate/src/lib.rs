//! The foundation of gpubridge's raw-verbs bridge (docs/VERBS_TRANSPORT.md): owned
//! libibverbs resources over a [`Provider`], the queue pair state machine, work tracking
//! that keeps memory registered while the device can use it, a bounded registration cache,
//! and a deterministic mock provider.
//!
//! - [`Context`], [`ProtectionDomain`], [`CompletionQueue`], [`QueuePair`] and
//!   [`MemoryRegion`] own their verbs resources and release them in a correct order
//!   whatever order they are dropped in (`resources.rs` says how).
//! - [`MrCache`] caches registrations by allocation identity, within fixed bounds.
//! - [`MockFabric`] / [`MockProvider`]: an in-process fabric with injectable faults.
//! - [`IbvProvider`] (feature `ibverbs`, Linux): libibverbs through a C shim compiled
//!   against rdma-core's headers.
//!
//! There is no wire protocol here and nothing Python calls yet (phase A2), and no GPU
//! memory path (phase A3). Nothing here has run on RDMA hardware.

mod cache;
mod endpoint;
mod error;
mod ffi;
#[cfg(feature = "ibverbs")]
mod ibv;
mod mock;
pub mod protocol;
mod provider;
mod resources;
mod types;
#[cfg(feature = "test-fabric")]
mod uds;

pub use cache::{CacheConfig, CacheKey, CacheStats, Invalidated, MrCache};
pub use endpoint::{Config, Counters, Endpoint, WaitError, TICK};
pub use error::{Error, Result};
#[cfg(feature = "ibverbs")]
pub use ibv::{check_layouts, library_path, wc_status_text, IbvProvider};
pub use mock::{
    Call, Fault, LiveCounts, MockDevice, MockFabric, MockLimits, MockPort, MockProvider, NodeId,
};
pub use provider::{BatchResult, Handle, MrKeys, Provider, QpCreated};
pub use resources::{
    CompletionChannel, CompletionQueue, Context, MemoryRegion, ProtectionDomain, QueuePair,
    SendRequest, Sge, StatsSnapshot, DEFAULT_DRAIN_TIMEOUT,
};
pub use types::{
    Access, Address, AsyncEvent, AsyncEventKind, Completion, DeviceAttr, Events, Gid, GidEntry,
    GidType, GlobalRoute, InitAttr, LinkLayer, Mtu, PortAttr, PortState, QpCaps, QpState,
    RawCompletion, RawRecv, RawSend, RawSge, RemoteTarget, RtrAttr, RtsAttr, SendOp, Transition,
    WcStatus, WorkId, WorkKind, MAX_SGE,
};
#[cfg(feature = "test-fabric")]
pub use uds::UdsProvider;

use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// What this build is, for the Python side to check before it uses the library:
/// `"gpubridge-verbs <version> providers=mock[,libibverbs][,test-fabric]"`, NUL-terminated
/// into `out`.
/// Returns 0, or -1 if `out` is too small or null. Opens nothing.
///
/// # Safety
///
/// `out` must be null or point to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_verbs_build_info(out: *mut c_char, len: usize) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // The layout check fails only if the shim was built from another source.
        #[cfg(feature = "ibverbs")]
        let providers = if check_layouts().is_ok() {
            "mock,libibverbs"
        } else {
            "mock"
        };
        #[cfg(not(feature = "ibverbs"))]
        let providers = "mock";
        #[cfg(feature = "test-fabric")]
        let providers = format!("{providers},test-fabric");
        let text = format!(
            "gpubridge-verbs {} providers={providers}",
            env!("CARGO_PKG_VERSION")
        );
        if out.is_null() || text.len() + 1 > len {
            return -1;
        }
        // SAFETY: out has len writable bytes (the caller's contract) and text.len() + 1
        // of them are written.
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), out.cast::<u8>(), text.len());
            *out.add(text.len()) = 0;
        }
        0
    }))
    .unwrap_or(-1)
}
