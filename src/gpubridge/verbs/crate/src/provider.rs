//! The verbs calls the owned resources make, on opaque handles.
//!
//! Two implementations: [`IbvProvider`](crate::IbvProvider) calls libibverbs through the C
//! shim (the `ibverbs` feature), [`MockProvider`](crate::MockProvider) is an in-process
//! fabric for tests. The owned layer (`resources.rs`) is the same code over both, so what
//! the mock tests show about release order, state checks and work tracking is about the
//! code that drives the real provider.
//!
//! A provider checks nothing the owned layer promises (release order, outstanding work);
//! like libibverbs, it does what it's told. The mock records violations of those promises
//! so the tests can show the owned layer keeps them.

use std::fmt;

use crate::error::Result;
use crate::types::{
    Access, DeviceAttr, GidEntry, PortAttr, QpCaps, RawCompletion, RawSge, SendOp, Transition,
};

/// A provider's resource: a pointer value for libibverbs, a number for the mock.
pub type Handle = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MrKeys {
    pub handle: Handle,
    pub lkey: u32,
    pub rkey: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QpCreated {
    pub handle: Handle,
    pub qp_num: u32,
    /// What the provider actually gave (at least what was asked for).
    pub caps: QpCaps,
}

pub trait Provider: Send + Sync + fmt::Debug {
    /// "libibverbs" or "mock".
    fn name(&self) -> &str;
    fn list_devices(&self) -> Result<Vec<String>>;
    fn open_device(&self, name: &str) -> Result<Handle>;
    fn close_device(&self, ctx: Handle) -> Result<()>;
    fn query_device(&self, ctx: Handle) -> Result<DeviceAttr>;
    fn query_port(&self, ctx: Handle, port: u8) -> Result<PortAttr>;
    fn query_gid(&self, ctx: Handle, port: u8, index: u32) -> Result<GidEntry>;

    fn alloc_pd(&self, ctx: Handle) -> Result<Handle>;
    fn dealloc_pd(&self, pd: Handle) -> Result<()>;

    /// Returns the queue and its actual depth (at least `depth`).
    fn create_cq(&self, ctx: Handle, depth: u32) -> Result<(Handle, u32)>;
    fn destroy_cq(&self, cq: Handle) -> Result<()>;

    /// A reliable-connected queue pair on which every send is signaled.
    fn create_rc_qp(
        &self,
        pd: Handle,
        send_cq: Handle,
        recv_cq: Handle,
        caps: &QpCaps,
    ) -> Result<QpCreated>;
    fn modify_qp(&self, qp: Handle, transition: &Transition) -> Result<()>;
    fn destroy_qp(&self, qp: Handle) -> Result<()>;

    /// Registers `len` bytes at `addr`.
    ///
    /// # Safety
    ///
    /// The memory must stay valid (allocated, not unmapped) until `dereg_mr` of the
    /// returned handle has returned: the device may read or write it until then.
    unsafe fn reg_mr(
        &self,
        pd: Handle,
        addr: *mut u8,
        len: usize,
        access: Access,
    ) -> Result<MrKeys>;
    /// Registers `len` bytes at `offset` of the dma-buf `fd`, addressed as `iova`. The
    /// kernel holds its own reference to the dma-buf; `fd` may be closed afterwards.
    fn reg_dmabuf_mr(
        &self,
        pd: Handle,
        offset: u64,
        len: usize,
        iova: u64,
        fd: i32,
        access: Access,
    ) -> Result<MrKeys>;
    fn dereg_mr(&self, mr: Handle) -> Result<()>;

    fn post_send(&self, qp: Handle, wr_id: u64, op: &SendOp, sges: &[RawSge]) -> Result<()>;
    fn post_recv(&self, qp: Handle, wr_id: u64, sges: &[RawSge]) -> Result<()>;
    /// Appends up to `max` completions to `out`; returns how many.
    fn poll_cq(&self, cq: Handle, out: &mut Vec<RawCompletion>, max: usize) -> Result<usize>;
}
