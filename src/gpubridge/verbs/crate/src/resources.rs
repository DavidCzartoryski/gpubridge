//! Owned verbs resources: context, protection domain, completion queue, queue pair and
//! memory region, over any [`Provider`].
//!
//! **Release order is structural.** Each resource holds an `Arc` of what it was made from:
//! a queue pair its completion queues and protection domain, a memory region its
//! protection domain, a protection domain or completion queue its context. A resource is
//! released when the last reference to it goes, so whatever order the caller drops
//! handles in, a queue pair goes before its completion queues, memory regions and queue
//! pairs before their protection domain, and the context last. A setup that fails halfway
//! drops what it made, which releases it in reverse.
//!
//! **Work holds memory.** Every posted work request is filed on its completion queue with
//! a clone of each memory region it reads or writes, and stays filed until its completion
//! is polled. So a region is deregistered (and a buffer it owns freed) only once no posted
//! work references it, whatever the caller does with its own handle. A queue pair closed
//! with work outstanding first moves to the error state, which makes the device flush
//! every request posted on it, and polls until each has completed; only then is it
//! destroyed. If that drain runs out of time, the requests' regions are leaked on purpose
//! (never deregistered, their buffers never freed) and `close` returns
//! [`Error::DrainTimeout`].

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::provider::{Handle, MrKeys, Provider};
use crate::types::{
    Access, Address, Completion, DeviceAttr, GidEntry, GlobalRoute, InitAttr, LinkLayer, PortAttr,
    PortState, QpCaps, QpState, RawCompletion, RawSge, RtrAttr, RtsAttr, SendOp, Transition,
    WcStatus, WorkId, WorkKind, MAX_SGE,
};

/// How long dropping a queue pair waits for its outstanding work to drain.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
/// Completions taken from the provider per poll.
const POLL_BATCH: usize = 64;
/// Owned buffers are page-aligned.
const BUFFER_ALIGN: usize = 4096;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic elsewhere must not stop resources from being released.
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What couldn't be released cleanly, per context.
#[derive(Debug, Default)]
struct Stats {
    release_errors: AtomicU64,
    leaked_regions: AtomicU64,
    leaked_work: AtomicU64,
    stray_completions: AtomicU64,
    discarded_completions: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatsSnapshot {
    /// Destroy, deallocate or deregister calls that failed (the resource is leaked).
    pub release_errors: u64,
    /// Memory regions kept registered forever because work on them never drained, or
    /// because deregistering failed.
    pub leaked_regions: u64,
    /// Work requests abandoned at a drain deadline.
    pub leaked_work: u64,
    /// Completions for work this layer didn't post (never expected).
    pub stray_completions: u64,
    /// Completions of a queue pair closed before anyone polled them.
    pub discarded_completions: u64,
}

struct CtxInner {
    provider: Arc<dyn Provider>,
    handle: Handle,
    name: String,
    attr: DeviceAttr,
    stats: Stats,
}

impl CtxInner {
    fn release_failed(&self) {
        self.stats.release_errors.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for CtxInner {
    fn drop(&mut self) {
        if self.provider.close_device(self.handle).is_err() {
            self.release_failed();
        }
    }
}

/// An open device.
#[derive(Clone)]
pub struct Context {
    inner: Arc<CtxInner>,
}

impl Context {
    pub fn open(provider: Arc<dyn Provider>, device: &str) -> Result<Context> {
        let handle = provider.open_device(device)?;
        let attr = match provider.query_device(handle) {
            Ok(attr) => attr,
            Err(e) => {
                // Nothing else holds the context yet: close it here.
                let _ = provider.close_device(handle);
                return Err(e);
            }
        };
        Ok(Context {
            inner: Arc::new(CtxInner {
                provider,
                handle,
                name: device.to_owned(),
                attr,
                stats: Stats::default(),
            }),
        })
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.inner.provider
    }

    pub fn attr(&self) -> &DeviceAttr {
        &self.inner.attr
    }

    fn check_port(&self, port: u8) -> Result<()> {
        if port == 0 || port > self.inner.attr.phys_port_cnt {
            return Err(Error::invalid(format!(
                "{} has ports 1 to {}, not {port}",
                self.inner.name, self.inner.attr.phys_port_cnt
            )));
        }
        Ok(())
    }

    pub fn query_port(&self, port: u8) -> Result<PortAttr> {
        self.check_port(port)?;
        self.inner.provider.query_port(self.inner.handle, port)
    }

    pub fn query_gid(&self, port: u8, index: u32) -> Result<GidEntry> {
        let attr = self.query_port(port)?;
        if index >= attr.gid_tbl_len {
            return Err(Error::invalid(format!(
                "GID index {index} outside port {port}'s table of {}",
                attr.gid_tbl_len
            )));
        }
        self.inner
            .provider
            .query_gid(self.inner.handle, port, index)
    }

    pub fn alloc_pd(&self) -> Result<ProtectionDomain> {
        let handle = self.inner.provider.alloc_pd(self.inner.handle)?;
        Ok(ProtectionDomain {
            inner: Arc::new(PdInner {
                ctx: self.inner.clone(),
                handle,
            }),
        })
    }

    /// A completion queue with room for `depth` completions. Queue pairs are created only
    /// on queues with room for all the work they can have outstanding.
    pub fn create_cq(&self, depth: u32) -> Result<CompletionQueue> {
        if depth == 0 || depth > self.inner.attr.max_cqe {
            return Err(Error::invalid(format!(
                "completion queue depth {depth} outside 1 to {}",
                self.inner.attr.max_cqe
            )));
        }
        let (handle, _actual) = self.inner.provider.create_cq(self.inner.handle, depth)?;
        Ok(CompletionQueue {
            inner: Arc::new(CqInner {
                ctx: self.inner.clone(),
                handle,
                depth,
                state: Mutex::new(CqState::default()),
            }),
        })
    }

    pub fn stats(&self) -> StatsSnapshot {
        let s = &self.inner.stats;
        StatsSnapshot {
            release_errors: s.release_errors.load(Ordering::Relaxed),
            leaked_regions: s.leaked_regions.load(Ordering::Relaxed),
            leaked_work: s.leaked_work.load(Ordering::Relaxed),
            stray_completions: s.stray_completions.load(Ordering::Relaxed),
            discarded_completions: s.discarded_completions.load(Ordering::Relaxed),
        }
    }
}

struct PdInner {
    ctx: Arc<CtxInner>,
    handle: Handle,
}

impl Drop for PdInner {
    fn drop(&mut self) {
        if self.ctx.provider.dealloc_pd(self.handle).is_err() {
            self.ctx.release_failed();
        }
    }
}

/// A protection domain: memory regions and queue pairs of one PD work together.
#[derive(Clone)]
pub struct ProtectionDomain {
    inner: Arc<PdInner>,
}

impl ProtectionDomain {
    pub fn context(&self) -> Context {
        Context {
            inner: self.inner.ctx.clone(),
        }
    }

    fn check_region(&self, len: usize, access: Access) -> Result<()> {
        access.validate()?;
        if len == 0 {
            return Err(Error::invalid("a memory region can't be empty"));
        }
        if len as u64 > self.inner.ctx.attr.max_mr_size {
            return Err(Error::Exhausted(format!(
                "{len} bytes exceeds the device's largest region ({})",
                self.inner.ctx.attr.max_mr_size
            )));
        }
        Ok(())
    }

    /// Allocates `len` zeroed, page-aligned bytes and registers them. The region owns the
    /// buffer, which is freed only after it is deregistered.
    pub fn register_buffer(&self, len: usize, access: Access) -> Result<MemoryRegion> {
        self.check_region(len, access)?;
        let buffer = OwnedBuffer::new(len)?;
        // SAFETY: the buffer moves into the region below and is freed only after
        // `dereg_mr` returns (MrInner's drop deregisters before its fields drop), or never
        // if deregistering fails. If registering fails, nothing is registered.
        let keys = unsafe {
            self.inner
                .ctx
                .provider
                .reg_mr(self.inner.handle, buffer.ptr.as_ptr(), len, access)
        }?;
        let addr = buffer.ptr.as_ptr() as usize;
        Ok(MemoryRegion::new(
            &self.inner,
            keys,
            addr,
            len,
            access,
            Some(buffer),
        ))
    }

    /// Registers memory the region doesn't own.
    ///
    /// # Safety
    ///
    /// `len` bytes at `addr` must stay valid (allocated, not unmapped) until the region is
    /// released, which is when the returned handle, all its clones and every work request
    /// posted with it are gone. Freeing the memory earlier lets the device write into
    /// freed memory.
    pub unsafe fn register(
        &self,
        addr: *mut u8,
        len: usize,
        access: Access,
    ) -> Result<MemoryRegion> {
        self.check_region(len, access)?;
        if addr.is_null() {
            return Err(Error::invalid("null address"));
        }
        // SAFETY: the caller keeps the memory valid until the region is released (this
        // function's contract); the region deregisters it before it is released.
        let keys = unsafe {
            self.inner
                .ctx
                .provider
                .reg_mr(self.inner.handle, addr, len, access)
        }?;
        Ok(MemoryRegion::new(
            &self.inner,
            keys,
            addr as usize,
            len,
            access,
            None,
        ))
    }

    /// A reliable-connected queue pair in the `Reset` state. Both completion queues must
    /// have room for all of its work (`max_send_wr` on `send_cq`, `max_recv_wr` on
    /// `recv_cq`) on top of other queue pairs': a completion queue that overflows breaks
    /// every queue pair on it.
    pub fn create_qp(
        &self,
        send_cq: &CompletionQueue,
        recv_cq: &CompletionQueue,
        caps: QpCaps,
    ) -> Result<QueuePair> {
        let ctx = &self.inner.ctx;
        for cq in [send_cq, recv_cq] {
            if !Arc::ptr_eq(&cq.inner.ctx, ctx) {
                return Err(Error::invalid("completion queue from another context"));
            }
        }
        let attr = &ctx.attr;
        let sge_limit = MAX_SGE.min(attr.max_sge);
        if caps.max_send_wr == 0 || caps.max_send_wr > attr.max_qp_wr {
            return Err(Error::invalid(format!(
                "max_send_wr {} outside 1 to {}",
                caps.max_send_wr, attr.max_qp_wr
            )));
        }
        if caps.max_recv_wr > attr.max_qp_wr {
            return Err(Error::invalid(format!(
                "max_recv_wr {} above {}",
                caps.max_recv_wr, attr.max_qp_wr
            )));
        }
        if caps.max_send_sge == 0 || caps.max_send_sge > sge_limit {
            return Err(Error::invalid(format!(
                "max_send_sge {} outside 1 to {sge_limit}",
                caps.max_send_sge
            )));
        }
        if caps.max_recv_sge > sge_limit || (caps.max_recv_wr > 0 && caps.max_recv_sge == 0) {
            return Err(Error::invalid(format!(
                "max_recv_sge {} outside 1 to {sge_limit}",
                caps.max_recv_sge
            )));
        }
        send_cq.inner.reserve(caps.max_send_wr)?;
        if let Err(e) = recv_cq.inner.reserve(caps.max_recv_wr) {
            send_cq.inner.unreserve(caps.max_send_wr);
            return Err(e);
        }
        let created = match ctx.provider.create_rc_qp(
            self.inner.handle,
            send_cq.inner.handle,
            recv_cq.inner.handle,
            &caps,
        ) {
            Ok(created) => created,
            Err(e) => {
                send_cq.inner.unreserve(caps.max_send_wr);
                recv_cq.inner.unreserve(caps.max_recv_wr);
                return Err(e);
            }
        };
        Ok(QueuePair {
            send_cq: send_cq.inner.clone(),
            recv_cq: recv_cq.inner.clone(),
            pd: self.inner.clone(),
            handle: created.handle,
            shared: Arc::new(QpShared {
                qp_num: created.qp_num,
                sends: AtomicU32::new(0),
                recvs: AtomicU32::new(0),
                failed: AtomicBool::new(false),
            }),
            caps,
            granted: created.caps,
            st: Mutex::new(QpRuntime {
                state: QpState::Reset,
                port: None,
            }),
            next_wr: AtomicU64::new(1),
            closed: false,
        })
    }
}

/// Page-aligned heap memory a region owns.
struct OwnedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: plain heap memory with no thread affinity. Copies in and out happen under the
// region's lock (MemoryRegion::read/write); the device's accesses are ordered by the work
// requests and completions that the same lock counts.
unsafe impl Send for OwnedBuffer {}
// SAFETY: as for Send: shared access goes through the region's lock.
unsafe impl Sync for OwnedBuffer {}

impl OwnedBuffer {
    fn new(len: usize) -> Result<OwnedBuffer> {
        let layout = Layout::from_size_align(len, BUFFER_ALIGN)
            .map_err(|_| Error::invalid(format!("{len} bytes is too large to allocate")))?;
        // SAFETY: the layout has a nonzero size (check_region refuses empty regions).
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::Exhausted(format!("allocating {len} bytes failed")))?;
        Ok(OwnedBuffer { ptr, layout })
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in new() with this layout, freed once (here).
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

struct MrInner {
    pd: Arc<PdInner>,
    handle: Handle,
    addr: usize,
    len: usize,
    lkey: u32,
    rkey: u32,
    access: Access,
    /// Work requests posted with this region and not completed yet. Posting takes this
    /// lock to count one, and so do copies in and out of an owned buffer.
    inflight: Mutex<u64>,
    owned: Option<OwnedBuffer>,
}

impl Drop for MrInner {
    fn drop(&mut self) {
        // Every posted request holds a clone of this region until its completion is
        // polled, so none is outstanding here.
        debug_assert_eq!(*lock(&self.inflight), 0);
        let ctx = &self.pd.ctx;
        if ctx.provider.dereg_mr(self.handle).is_err() {
            ctx.release_failed();
            ctx.stats.leaked_regions.fetch_add(1, Ordering::Relaxed);
            // Possibly still registered: the device may still reach the buffer.
            if let Some(buffer) = self.owned.take() {
                mem::forget(buffer);
            }
        }
        // `owned` (if any) is freed after this, once deregistered.
    }
}

/// A registered memory region. Clones share one registration, released when the last
/// clone, and the last work request posted with it, are gone.
#[derive(Clone)]
pub struct MemoryRegion {
    inner: Arc<MrInner>,
}

impl MemoryRegion {
    fn new(
        pd: &Arc<PdInner>,
        keys: MrKeys,
        addr: usize,
        len: usize,
        access: Access,
        owned: Option<OwnedBuffer>,
    ) -> MemoryRegion {
        MemoryRegion {
            inner: Arc::new(MrInner {
                pd: pd.clone(),
                handle: keys.handle,
                addr,
                len,
                lkey: keys.lkey,
                rkey: keys.rkey,
                access,
                inflight: Mutex::new(0),
                owned,
            }),
        }
    }

    pub fn addr(&self) -> usize {
        self.inner.addr
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    pub fn lkey(&self) -> u32 {
        self.inner.lkey
    }

    pub fn rkey(&self) -> u32 {
        self.inner.rkey
    }

    pub fn access(&self) -> Access {
        self.inner.access
    }

    /// Whether the region owns its buffer (from `register_buffer`).
    pub fn owns_buffer(&self) -> bool {
        self.inner.owned.is_some()
    }

    /// Posted work requests that reference this region and haven't completed.
    pub fn inflight(&self) -> u64 {
        *lock(&self.inner.inflight)
    }

    /// Whether `self` and `other` are the same registration.
    pub fn same(&self, other: &MemoryRegion) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Handles to this registration, including the ones posted work holds.
    pub(crate) fn holders(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    fn owned_range(&self, offset: usize, len: usize) -> Result<&OwnedBuffer> {
        let buffer = self.inner.owned.as_ref().ok_or_else(|| {
            Error::Unsupported("copying in or out of memory the region doesn't own".into())
        })?;
        if offset
            .checked_add(len)
            .map_or(true, |end| end > self.inner.len)
        {
            return Err(Error::invalid(format!(
                "{len} bytes at {offset} outside a region of {}",
                self.inner.len
            )));
        }
        Ok(buffer)
    }

    /// Copies `data` into an owned buffer at `offset`. Refused while work is posted on it.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<()> {
        let buffer = self.owned_range(offset, data.len())?;
        let inflight = lock(&self.inner.inflight);
        if *inflight > 0 {
            return Err(Error::Busy(format!(
                "{} work requests use this region",
                *inflight
            )));
        }
        // SAFETY: in bounds (owned_range); no posted work uses the buffer and none can be
        // posted while this lock is held. Writes by a peer into a remotely writable
        // region are ordered by the protocol, as for any RDMA buffer.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr().add(offset), data.len())
        };
        Ok(())
    }

    /// Copies from an owned buffer at `offset` into `out`. Refused while work is posted.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        let buffer = self.owned_range(offset, out.len())?;
        let inflight = lock(&self.inner.inflight);
        if *inflight > 0 {
            return Err(Error::Busy(format!(
                "{} work requests use this region",
                *inflight
            )));
        }
        // SAFETY: as in write().
        unsafe {
            ptr::copy_nonoverlapping(buffer.ptr.as_ptr().add(offset), out.as_mut_ptr(), out.len())
        };
        Ok(())
    }

    fn pd(&self) -> &Arc<PdInner> {
        &self.inner.pd
    }
}

impl std::fmt::Debug for MemoryRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryRegion")
            .field("addr", &format_args!("{:#x}", self.inner.addr))
            .field("len", &self.inner.len)
            .field("lkey", &format_args!("{:#x}", self.inner.lkey))
            .field("access", &self.inner.access)
            .finish()
    }
}

/// A region referenced by posted work: counted while the request is outstanding.
struct InFlight(MemoryRegion);

impl InFlight {
    fn new(region: &MemoryRegion) -> InFlight {
        *lock(&region.inner.inflight) += 1;
        InFlight(region.clone())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        *lock(&self.0.inner.inflight) -= 1;
    }
}

/// A part of a region for a work request.
#[derive(Clone, Copy)]
pub struct Sge<'a> {
    pub region: &'a MemoryRegion,
    pub offset: usize,
    pub len: u32,
}

impl<'a> Sge<'a> {
    pub fn new(region: &'a MemoryRegion, offset: usize, len: u32) -> Sge<'a> {
        Sge {
            region,
            offset,
            len,
        }
    }
}

/// A send-side work request.
#[derive(Clone, Copy)]
pub struct SendRequest<'a> {
    /// Returned in the completion.
    pub tag: u64,
    pub op: SendOp,
    pub sges: &'a [Sge<'a>],
}

/// Shared between a queue pair and its outstanding work.
struct QpShared {
    qp_num: u32,
    sends: AtomicU32,
    recvs: AtomicU32,
    /// A completion failed: the device has moved the queue pair to the error state.
    failed: AtomicBool,
}

/// A posted work request, filed on its completion queue until its completion is polled.
struct Posted {
    tag: u64,
    kind: WorkKind,
    shared: Arc<QpShared>,
    regions: Vec<InFlight>,
}

impl Drop for Posted {
    fn drop(&mut self) {
        let counter = if self.kind == WorkKind::Recv {
            &self.shared.recvs
        } else {
            &self.shared.sends
        };
        counter.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Default)]
struct CqState {
    /// Work the queue pairs on this queue could have outstanding at once.
    committed: u64,
    outstanding: HashMap<(u32, u64), Posted>,
    per_qp: HashMap<u32, usize>,
    ready: VecDeque<Completion>,
    raw: Vec<RawCompletion>,
}

impl CqState {
    fn remove(&mut self, qp_num: u32, wr_id: u64) -> Option<Posted> {
        let posted = self.outstanding.remove(&(qp_num, wr_id))?;
        if let Entry::Occupied(mut count) = self.per_qp.entry(qp_num) {
            *count.get_mut() -= 1;
            if *count.get() == 0 {
                count.remove();
            }
        }
        Some(posted)
    }
}

struct CqInner {
    ctx: Arc<CtxInner>,
    handle: Handle,
    depth: u32,
    state: Mutex<CqState>,
}

impl CqInner {
    fn reserve(&self, n: u32) -> Result<()> {
        let mut st = lock(&self.state);
        if st.committed + u64::from(n) > u64::from(self.depth) {
            return Err(Error::Exhausted(format!(
                "a completion queue of depth {} already has {} committed; {n} more won't fit",
                self.depth, st.committed
            )));
        }
        st.committed += u64::from(n);
        Ok(())
    }

    fn unreserve(&self, n: u32) {
        let mut st = lock(&self.state);
        st.committed -= u64::from(n);
    }

    fn track(&self, qp_num: u32, wr_id: u64, posted: Posted) {
        let mut st = lock(&self.state);
        st.outstanding.insert((qp_num, wr_id), posted);
        *st.per_qp.entry(qp_num).or_insert(0) += 1;
    }

    fn untrack(&self, qp_num: u32, wr_id: u64) -> Option<Posted> {
        lock(&self.state).remove(qp_num, wr_id)
    }

    fn outstanding_for(&self, qp_num: u32) -> usize {
        lock(&self.state).per_qp.get(&qp_num).copied().unwrap_or(0)
    }

    /// Polls the provider once and files what it returns. The finished requests come
    /// back to be dropped once the lock is released (dropping one may deregister a
    /// region). Returns how many completions the provider gave.
    fn pump(&self, st: &mut CqState, finished: &mut Vec<Posted>) -> Result<usize> {
        let mut raw = mem::take(&mut st.raw);
        raw.clear();
        let result = self.ctx.provider.poll_cq(self.handle, &mut raw, POLL_BATCH);
        let n = raw.len();
        for wc in raw.drain(..) {
            match st.remove(wc.qp_num, wc.wr_id) {
                Some(posted) => {
                    if wc.status != WcStatus::Success {
                        posted.shared.failed.store(true, Ordering::Release);
                    }
                    st.ready.push_back(Completion {
                        qp_num: wc.qp_num,
                        work: WorkId(wc.wr_id),
                        tag: posted.tag,
                        kind: posted.kind,
                        status: wc.status,
                        byte_len: wc.byte_len,
                        imm: wc.imm,
                        vendor_err: wc.vendor_err,
                    });
                    finished.push(posted);
                }
                None => {
                    self.ctx
                        .stats
                        .stray_completions
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        st.raw = raw;
        result.map(|_| n)
    }

    fn pump_once(&self) -> Result<usize> {
        let mut finished = Vec::new();
        let result = {
            let mut st = lock(&self.state);
            self.pump(&mut st, &mut finished)
        };
        drop(finished);
        result
    }

    /// Drops completions of a closed queue pair that nobody polled.
    fn discard(&self, qp_num: u32) {
        let mut st = lock(&self.state);
        let before = st.ready.len();
        st.ready.retain(|c| c.qp_num != qp_num);
        let gone = (before - st.ready.len()) as u64;
        self.ctx
            .stats
            .discarded_completions
            .fetch_add(gone, Ordering::Relaxed);
    }

    /// Takes every request still filed for `qp_num` (to leak them).
    fn take(&self, qp_num: u32) -> Vec<Posted> {
        let mut st = lock(&self.state);
        let keys: Vec<(u32, u64)> = st
            .outstanding
            .keys()
            .filter(|(q, _)| *q == qp_num)
            .copied()
            .collect();
        keys.into_iter()
            .filter_map(|(q, w)| st.remove(q, w))
            .collect()
    }
}

impl Drop for CqInner {
    fn drop(&mut self) {
        if self.ctx.provider.destroy_cq(self.handle).is_err() {
            self.ctx.release_failed();
        }
    }
}

/// A completion queue. Clones share it; it is destroyed once the last clone and the last
/// queue pair on it are gone.
#[derive(Clone)]
pub struct CompletionQueue {
    inner: Arc<CqInner>,
}

impl CompletionQueue {
    pub fn depth(&self) -> u32 {
        self.inner.depth
    }

    /// Work requests filed on this queue and not completed yet.
    pub fn outstanding(&self) -> usize {
        lock(&self.inner.state).outstanding.len()
    }

    /// Up to `max` completions, oldest first: ones already taken from the provider (by a
    /// queue pair's drain, say), then a poll of the provider if those aren't enough.
    /// A finished request releases its hold on the memory regions it used.
    pub fn poll(&self, max: usize) -> Result<Vec<Completion>> {
        let mut finished = Vec::new();
        let result = {
            let mut st = lock(&self.inner.state);
            let pumped = if st.ready.len() < max {
                self.inner.pump(&mut st, &mut finished).map(|_| ())
            } else {
                Ok(())
            };
            let n = max.min(st.ready.len());
            let out: Vec<Completion> = st.ready.drain(..n).collect();
            match pumped {
                Err(e) if out.is_empty() => Err(e),
                _ => Ok(out),
            }
        };
        drop(finished);
        result
    }
}

struct QpRuntime {
    state: QpState,
    /// The port and its attributes from the `Init` transition.
    port: Option<(u8, PortAttr)>,
}

/// A reliable-connected queue pair.
///
/// Dropping it drains and destroys it as [`close`](QueuePair::close) does, with
/// [`DEFAULT_DRAIN_TIMEOUT`].
pub struct QueuePair {
    // Fields drop in this order once the queue pair is destroyed: its completion queues
    // before its protection domain, the conventional teardown order.
    send_cq: Arc<CqInner>,
    recv_cq: Arc<CqInner>,
    pd: Arc<PdInner>,
    handle: Handle,
    shared: Arc<QpShared>,
    caps: QpCaps,
    granted: QpCaps,
    st: Mutex<QpRuntime>,
    next_wr: AtomicU64,
    closed: bool,
}

fn check_bits(what: &str, value: u32, bits: u32) -> Result<()> {
    if value >= 1 << bits {
        return Err(Error::invalid(format!(
            "{what} {value:#x} doesn't fit in {bits} bits"
        )));
    }
    Ok(())
}

fn check_grh(grh: &GlobalRoute, port: u8, attr: &PortAttr) -> Result<()> {
    if grh.dgid.is_zero() {
        return Err(Error::invalid("a global route needs the peer's GID"));
    }
    if u32::from(grh.sgid_index) >= attr.gid_tbl_len {
        return Err(Error::invalid(format!(
            "GID index {} outside port {port}'s table of {}",
            grh.sgid_index, attr.gid_tbl_len
        )));
    }
    if grh.hop_limit == 0 {
        return Err(Error::invalid("a global route's hop limit must be nonzero"));
    }
    check_bits("flow label", grh.flow_label, 20)
}

impl QueuePair {
    fn provider(&self) -> &Arc<dyn Provider> {
        &self.pd.ctx.provider
    }

    pub fn qp_num(&self) -> u32 {
        self.shared.qp_num
    }

    /// What it was created for (the bounds this layer enforces).
    pub fn caps(&self) -> QpCaps {
        self.caps
    }

    /// What the provider actually granted (at least `caps`).
    pub fn granted_caps(&self) -> QpCaps {
        self.granted
    }

    fn current(&self, rt: &mut QpRuntime) -> QpState {
        if self.shared.failed.load(Ordering::Acquire) && rt.state != QpState::Reset {
            rt.state = QpState::Err;
        }
        rt.state
    }

    /// The state, including the device's own move to `Err` after a failed completion
    /// (seen once that completion is polled).
    pub fn state(&self) -> QpState {
        let mut rt = lock(&self.st);
        self.current(&mut rt)
    }

    /// Outstanding (sends, receives).
    pub fn outstanding(&self) -> (u32, u32) {
        (
            self.shared.sends.load(Ordering::Acquire),
            self.shared.recvs.load(Ordering::Acquire),
        )
    }

    fn require(&self, rt: &mut QpRuntime, state: QpState, wanted: &'static str) -> Result<()> {
        let now = self.current(rt);
        if now != state {
            return Err(Error::InvalidState { state: now, wanted });
        }
        Ok(())
    }

    /// `Reset -> Init` on `attr.port`.
    pub fn to_init(&self, attr: &InitAttr) -> Result<()> {
        let mut rt = lock(&self.st);
        self.require(&mut rt, QpState::Reset, "move to Init")?;
        let ctx = Context {
            inner: self.pd.ctx.clone(),
        };
        let port = ctx.query_port(attr.port)?;
        self.provider()
            .modify_qp(self.handle, &Transition::Init(*attr))?;
        rt.state = QpState::Init;
        rt.port = Some((attr.port, port));
        Ok(())
    }

    /// `Init -> Rtr`, connected to the peer queue pair `attr` describes.
    pub fn to_rtr(&self, attr: &RtrAttr) -> Result<()> {
        let mut rt = lock(&self.st);
        self.require(&mut rt, QpState::Init, "move to Rtr")?;
        let (port, _) = rt.port.expect("a queue pair in Init has a port");
        // The port as it is now: it may have gone down since Init.
        let ctx = Context {
            inner: self.pd.ctx.clone(),
        };
        let pattr = ctx.query_port(port)?;
        if pattr.state != PortState::Active {
            return Err(Error::invalid(format!(
                "port {port} is {:?}, not Active",
                pattr.state
            )));
        }
        check_bits("destination queue pair number", attr.dest_qp_num, 24)?;
        if attr.dest_qp_num < 2 {
            return Err(Error::invalid(
                "queue pairs 0 and 1 are special, not RC peers",
            ));
        }
        check_bits("receive PSN", attr.rq_psn, 24)?;
        check_bits("minimum RNR timer", u32::from(attr.min_rnr_timer), 5)?;
        check_bits("service level", u32::from(attr.sl), 4)?;
        if attr.path_mtu > pattr.active_mtu {
            return Err(Error::invalid(format!(
                "path MTU {} above port {port}'s active MTU {}",
                attr.path_mtu.bytes(),
                pattr.active_mtu.bytes()
            )));
        }
        if u32::from(attr.max_dest_rd_atomic) > self.pd.ctx.attr.max_qp_rd_atom {
            return Err(Error::invalid(format!(
                "max_dest_rd_atomic {} above the device's {}",
                attr.max_dest_rd_atomic, self.pd.ctx.attr.max_qp_rd_atom
            )));
        }
        match (&attr.address, pattr.link_layer) {
            (Address::Ib { lid, grh }, LinkLayer::InfiniBand) => {
                if *lid == 0 || *lid >= 0xC000 {
                    return Err(Error::invalid(format!(
                        "LID {lid:#x} isn't a unicast LID (1 to 0xbfff)"
                    )));
                }
                if let Some(grh) = grh {
                    check_grh(grh, port, &pattr)?;
                }
            }
            (Address::Roce(grh), LinkLayer::Ethernet) => check_grh(grh, port, &pattr)?,
            (Address::Ib { .. }, LinkLayer::Ethernet) => {
                return Err(Error::invalid(format!(
                    "port {port} is Ethernet (RoCE): address the peer by GID"
                )))
            }
            (Address::Roce(_), LinkLayer::InfiniBand) => {
                return Err(Error::invalid(format!(
                    "port {port} is InfiniBand: address the peer by LID"
                )))
            }
        }
        self.provider()
            .modify_qp(self.handle, &Transition::Rtr { port, attr: *attr })?;
        rt.state = QpState::Rtr;
        rt.port = Some((port, pattr));
        Ok(())
    }

    /// `Rtr -> Rts`.
    pub fn to_rts(&self, attr: &RtsAttr) -> Result<()> {
        let mut rt = lock(&self.st);
        self.require(&mut rt, QpState::Rtr, "move to Rts")?;
        check_bits("send PSN", attr.sq_psn, 24)?;
        check_bits("timeout", u32::from(attr.timeout), 5)?;
        check_bits("retry count", u32::from(attr.retry_cnt), 3)?;
        check_bits("RNR retry count", u32::from(attr.rnr_retry), 3)?;
        if u32::from(attr.max_rd_atomic) > self.pd.ctx.attr.max_qp_init_rd_atom {
            return Err(Error::invalid(format!(
                "max_rd_atomic {} above the device's {}",
                attr.max_rd_atomic, self.pd.ctx.attr.max_qp_init_rd_atom
            )));
        }
        self.provider()
            .modify_qp(self.handle, &Transition::Rts(*attr))?;
        rt.state = QpState::Rts;
        Ok(())
    }

    /// Any state -> `Err`. The device then completes every outstanding request with
    /// [`WcStatus::WrFlushErr`].
    pub fn to_error(&self) -> Result<()> {
        let mut rt = lock(&self.st);
        if rt.state == QpState::Err {
            return Ok(());
        }
        self.provider().modify_qp(self.handle, &Transition::Error)?;
        rt.state = QpState::Err;
        Ok(())
    }

    /// Any state -> `Reset`, only with no work outstanding: verbs would discard it without
    /// completions, and with them the record of when its memory is free.
    pub fn reset(&self) -> Result<()> {
        let mut rt = lock(&self.st);
        let (sends, recvs) = self.outstanding();
        if sends + recvs > 0 {
            return Err(Error::Busy(format!(
                "{sends} sends and {recvs} receives outstanding; drain them first"
            )));
        }
        self.provider().modify_qp(self.handle, &Transition::Reset)?;
        rt.state = QpState::Reset;
        rt.port = None;
        self.shared.failed.store(false, Ordering::Release);
        Ok(())
    }

    fn raw_sges(&self, sges: &[Sge<'_>], limit: u32, receive: bool) -> Result<Vec<RawSge>> {
        if sges.len() > limit as usize {
            return Err(Error::invalid(format!(
                "{} scatter-gather entries, at most {limit}",
                sges.len()
            )));
        }
        sges.iter()
            .map(|s| {
                if !Arc::ptr_eq(s.region.pd(), &self.pd) {
                    return Err(Error::invalid(
                        "memory region from another protection domain",
                    ));
                }
                if s.offset
                    .checked_add(s.len as usize)
                    .map_or(true, |end| end > s.region.len())
                {
                    return Err(Error::invalid(format!(
                        "{} bytes at {} outside a region of {}",
                        s.len,
                        s.offset,
                        s.region.len()
                    )));
                }
                if receive && !s.region.access().contains(Access::LOCAL_WRITE) {
                    return Err(Error::invalid("receiving needs a locally writable region"));
                }
                Ok(RawSge {
                    addr: (s.region.addr() + s.offset) as u64,
                    len: s.len,
                    lkey: s.region.lkey(),
                })
            })
            .collect()
    }

    fn reserve(counter: &AtomicU32, limit: u32, queue: &'static str) -> Result<()> {
        let mut n = counter.load(Ordering::Acquire);
        loop {
            if n >= limit {
                return Err(Error::QueueFull { queue, limit });
            }
            match counter.compare_exchange_weak(n, n + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(()),
                Err(now) => n = now,
            }
        }
    }

    /// Posts a send, an RDMA write, or either with immediate data. Only in `Rts`. Fails
    /// with [`Error::QueueFull`] when `max_send_wr` sends are outstanding.
    pub fn post_send(&self, request: &SendRequest<'_>) -> Result<WorkId> {
        let mut rt = lock(&self.st);
        self.require(&mut rt, QpState::Rts, "post a send")?;
        let raw = self.raw_sges(request.sges, self.caps.max_send_sge, false)?;
        Self::reserve(&self.shared.sends, self.caps.max_send_wr, "send")?;
        let wr_id = self.next_wr.fetch_add(1, Ordering::Relaxed);
        let posted = Posted {
            tag: request.tag,
            kind: request.op.kind(),
            shared: self.shared.clone(),
            regions: request
                .sges
                .iter()
                .map(|s| InFlight::new(s.region))
                .collect(),
        };
        // Filed before it is posted, so its completion always finds it.
        self.send_cq.track(self.shared.qp_num, wr_id, posted);
        if let Err(e) = self
            .provider()
            .post_send(self.handle, wr_id, &request.op, &raw)
        {
            drop(self.send_cq.untrack(self.shared.qp_num, wr_id));
            return Err(e);
        }
        Ok(WorkId(wr_id))
    }

    /// Posts a receive into `sges` (none, for RDMA writes with immediate data). From
    /// `Init` on. Fails with [`Error::QueueFull`] when `max_recv_wr` are outstanding.
    pub fn post_recv(&self, tag: u64, sges: &[Sge<'_>]) -> Result<WorkId> {
        let mut rt = lock(&self.st);
        let state = self.current(&mut rt);
        if !matches!(state, QpState::Init | QpState::Rtr | QpState::Rts) {
            return Err(Error::InvalidState {
                state,
                wanted: "post a receive",
            });
        }
        let raw = self.raw_sges(sges, self.caps.max_recv_sge, true)?;
        Self::reserve(&self.shared.recvs, self.caps.max_recv_wr, "receive")?;
        let wr_id = self.next_wr.fetch_add(1, Ordering::Relaxed);
        let posted = Posted {
            tag,
            kind: WorkKind::Recv,
            shared: self.shared.clone(),
            regions: sges.iter().map(|s| InFlight::new(s.region)).collect(),
        };
        self.recv_cq.track(self.shared.qp_num, wr_id, posted);
        if let Err(e) = self.provider().post_recv(self.handle, wr_id, &raw) {
            drop(self.recv_cq.untrack(self.shared.qp_num, wr_id));
            return Err(e);
        }
        Ok(WorkId(wr_id))
    }

    fn filed(&self) -> usize {
        let qp = self.shared.qp_num;
        let mut n = self.send_cq.outstanding_for(qp);
        if !Arc::ptr_eq(&self.send_cq, &self.recv_cq) {
            n += self.recv_cq.outstanding_for(qp);
        }
        n
    }

    /// Polls both completion queues until none of this queue pair's work is outstanding,
    /// or `deadline`. Returns how much is left.
    fn drain(&self, deadline: Instant) -> usize {
        loop {
            let left = self.filed();
            if left == 0 || Instant::now() >= deadline {
                return left;
            }
            let mut got = match self.send_cq.pump_once() {
                Ok(n) => n,
                Err(_) => return self.filed(),
            };
            if !Arc::ptr_eq(&self.send_cq, &self.recv_cq) {
                got += match self.recv_cq.pump_once() {
                    Ok(n) => n,
                    Err(_) => return self.filed(),
                };
            }
            if got == 0 {
                thread::sleep(Duration::from_micros(100));
            }
        }
    }

    fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let qp_num = self.shared.qp_num;
        let mut result = Ok(());
        if self.filed() > 0 {
            // The error state flushes everything posted: each request then completes.
            let in_error = lock(&self.st).state == QpState::Err;
            if !in_error {
                if let Err(e) = self.provider().modify_qp(self.handle, &Transition::Error) {
                    result = Err(e);
                }
            }
            lock(&self.st).state = QpState::Err;
            let left = self.drain(Instant::now() + timeout);
            if left > 0 {
                // The device may still use this memory: keep every region these requests
                // hold registered (and every owned buffer allocated) forever.
                let mut leaked = self.send_cq.take(qp_num);
                if !Arc::ptr_eq(&self.send_cq, &self.recv_cq) {
                    leaked.extend(self.recv_cq.take(qp_num));
                }
                let stats = &self.pd.ctx.stats;
                stats
                    .leaked_work
                    .fetch_add(leaked.len() as u64, Ordering::Relaxed);
                let regions: usize = leaked.iter().map(|p| p.regions.len()).sum();
                stats
                    .leaked_regions
                    .fetch_add(regions as u64, Ordering::Relaxed);
                for posted in leaked {
                    mem::forget(posted);
                }
                result = Err(Error::DrainTimeout {
                    qp_num,
                    outstanding: left,
                });
            }
        }
        self.send_cq.discard(qp_num);
        if !Arc::ptr_eq(&self.send_cq, &self.recv_cq) {
            self.recv_cq.discard(qp_num);
        }
        match self.provider().destroy_qp(self.handle) {
            Ok(()) => {
                self.send_cq.unreserve(self.caps.max_send_wr);
                self.recv_cq.unreserve(self.caps.max_recv_wr);
            }
            Err(e) => {
                // Still there: its completion queue capacity stays committed.
                self.pd.ctx.release_failed();
                if result.is_ok() {
                    result = Err(e);
                }
            }
        }
        result
    }

    /// Drains and destroys the queue pair. With work outstanding, it first moves to `Err`
    /// and waits up to `timeout` for every request to complete (flushed); requests still
    /// outstanding then are leaked with their memory (see the module docs).
    pub fn close(mut self, timeout: Duration) -> Result<()> {
        self.shutdown(timeout)
    }
}

impl Drop for QueuePair {
    fn drop(&mut self) {
        let _ = self.shutdown(DEFAULT_DRAIN_TIMEOUT);
    }
}

impl std::fmt::Debug for QueuePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuePair")
            .field("qp_num", &format_args!("{:#x}", self.shared.qp_num))
            .field("state", &lock(&self.st).state)
            .finish()
    }
}
