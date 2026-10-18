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
use std::time::Duration;

use crate::error::{Error, Result};
use crate::types::{
    Access, DeviceAttr, Events, ForkStatus, GidEntry, PortAttr, QpCaps, RawCompletion, RawRecv,
    RawSend, RawSge, SendOp, Transition,
};

/// A provider's resource: a pointer value for libibverbs, a number for the mock.
pub type Handle = u64;

/// A batch post's result: on an error, how many were posted before it.
pub type BatchResult = std::result::Result<(), (usize, Error)>;

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
    /// A completion queue that notifies `channel` (after [`req_notify_cq`]) instead of
    /// having to be polled to find out whether anything completed.
    ///
    /// [`req_notify_cq`]: Provider::req_notify_cq
    fn create_cq_on_channel(
        &self,
        ctx: Handle,
        depth: u32,
        channel: Handle,
    ) -> Result<(Handle, u32)>;
    fn destroy_cq(&self, cq: Handle) -> Result<()>;

    /// A completion channel (`ibv_create_comp_channel`): a file descriptor that becomes
    /// readable when a completion queue on it has a notification.
    fn create_comp_channel(&self, ctx: Handle) -> Result<Handle>;
    /// Only once every completion queue on it is destroyed.
    fn destroy_comp_channel(&self, channel: Handle) -> Result<()>;
    /// Asks for one notification on the queue's channel when its next completion arrives.
    fn req_notify_cq(&self, cq: Handle) -> Result<()>;
    /// Blocks up to `timeout` until `channel` has a completion notification or `ctx` an
    /// asynchronous event, then takes and acknowledges what is there. Returns at once if
    /// something is already waiting; returns empty [`Events`] at the timeout.
    fn wait_events(&self, ctx: Handle, channel: Handle, timeout: Duration) -> Result<Events>;

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

    /// Posts several sends at once (one doorbell where the provider can). On an error,
    /// the first `n` were posted and the rest weren't: `Err((n, error))`.
    fn post_send_batch(&self, qp: Handle, wrs: &[RawSend<'_>]) -> BatchResult {
        for (n, wr) in wrs.iter().enumerate() {
            self.post_send(qp, wr.wr_id, &wr.op, wr.sges)
                .map_err(|e| (n, e))?;
        }
        Ok(())
    }

    /// Posts several receives at once; errors as for [`post_send_batch`].
    ///
    /// [`post_send_batch`]: Provider::post_send_batch
    fn post_recv_batch(&self, qp: Handle, wrs: &[RawRecv<'_>]) -> BatchResult {
        for (n, wr) in wrs.iter().enumerate() {
            self.post_recv(qp, wr.wr_id, wr.sges).map_err(|e| (n, e))?;
        }
        Ok(())
    }

    /// What this provider is, for telemetry: the library it calls (for libibverbs, the
    /// file it was loaded from, whose name carries rdma-core's version) or that it is a
    /// mock.
    fn describe(&self) -> String {
        self.name().to_owned()
    }

    /// True for providers that are not RDMA (the mocks): telemetry labels them MOCK.
    fn is_mock(&self) -> bool {
        true
    }

    /// Whether registered memory survives a `fork()` (`ibv_is_fork_initialized`). The
    /// mocks' default: no device reads their memory, so there is nothing to protect.
    fn fork_status(&self) -> Result<ForkStatus> {
        Ok(ForkStatus::Unneeded)
    }

    /// `ibv_fork_init`: keeps registered pages out of forked children. Only before any
    /// memory is registered in the process.
    fn fork_init(&self) -> Result<()> {
        Ok(())
    }
    /// Appends up to `max` completions to `out`; returns how many.
    fn poll_cq(&self, cq: Handle, out: &mut Vec<RawCompletion>, max: usize) -> Result<usize>;
}
