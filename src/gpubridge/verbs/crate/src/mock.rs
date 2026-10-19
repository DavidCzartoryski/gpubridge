//! A deterministic in-process fabric behind the [`Provider`] trait, for tests.
//!
//! A [`MockFabric`] has nodes, each with devices and ports (InfiniBand with LIDs, or
//! Ethernet/RoCE with GIDs); [`MockFabric::provider`] gives one node's provider. Queue
//! pairs connect across nodes as RC queue pairs do: a send moves only between two queue
//! pairs in `Rtr`/`Rts` that name each other, whose addresses match the peer's port and
//! whose PSNs agree; otherwise it fails with `RetryExcErr` after the retry budget, as a
//! wrong address or a dead peer does on a real fabric. Sends, RDMA writes and immediate
//! data copy real bytes between registered memory, checked against the regions' keys,
//! bounds, protection domains and access rights.
//!
//! Nothing moves on its own: every `poll_cq` advances the fabric one tick and moves what
//! is ready, so a test's sequence of calls fixes the outcome. Faults ([`Fault`]) make a
//! call fail, delay sends or completions, fail a send or raise an asynchronous event;
//! [`MockFabric::kill`] loses a node; device limits ([`MockLimits`]) make resources run
//! out.
//!
//! Completion channels work as in verbs: a queue asked for a notification
//! (`req_notify_cq`) marks its channel when its next completion arrives, and
//! `wait_events` returns it. A thread blocked in `wait_events` is the fabric's clock while
//! it waits: it advances a tick about every millisecond and whenever another call changes
//! something, so threads driving several nodes (the protocol tests) make progress without
//! anyone polling. Those runs are not tick-deterministic; single-threaded ones still are.
//!
//! Like libibverbs, the mock does what it is told; where verbs would let a caller break
//! memory safety (deregister memory that posted work still uses, destroy a queue pair or
//! reset one with work outstanding, overrun a completion queue) it also records a
//! violation ([`MockFabric::violations`]), which the tests require to stay empty.

use std::collections::{BTreeMap, VecDeque};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::provider::{Handle, MrKeys, Provider, QpCreated};
use crate::types::{
    Access, Address, AsyncEvent, AsyncEventKind, DeviceAttr, Events, ForkStatus, Gid, GidEntry,
    GidType, LinkLayer, Mtu, PortAttr, PortState, QpCaps, QpState, RawCompletion, RawSge, RtrAttr,
    RtsAttr, SendOp, Transition, WcStatus,
};

pub type NodeId = usize;

const EINVAL: i32 = 22;
const ENOMEM: i32 = 12;
const EBUSY: i32 = 16;
const EIO: i32 = 5;
const ENODEV: i32 = 19;

#[derive(Clone, Debug)]
pub struct MockPort {
    pub link_layer: LinkLayer,
    pub state: PortState,
    pub lid: u16,
    pub gids: Vec<GidEntry>,
    pub active_mtu: Mtu,
    pub max_mtu: Mtu,
}

impl MockPort {
    /// An active InfiniBand port with this LID and a link-local GID derived from it.
    pub fn infiniband(lid: u16) -> MockPort {
        let mut gid = [0u8; 16];
        gid[0] = 0xfe;
        gid[1] = 0x80;
        gid[14..].copy_from_slice(&lid.to_be_bytes());
        MockPort {
            link_layer: LinkLayer::InfiniBand,
            state: PortState::Active,
            lid,
            gids: vec![GidEntry {
                gid: Gid(gid),
                gid_type: GidType::Ib,
                ndev_ifindex: 0,
            }],
            active_mtu: Mtu::Mtu4096,
            max_mtu: Mtu::Mtu4096,
        }
    }

    /// An active RoCE port: its GID table holds a RoCE v1 link-local GID at index 0 and
    /// `gid` as RoCE v2 at index 1.
    pub fn roce(gid: Gid) -> MockPort {
        let mut link_local = [0u8; 16];
        link_local[0] = 0xfe;
        link_local[1] = 0x80;
        link_local[8..].copy_from_slice(&gid.0[8..]);
        MockPort {
            link_layer: LinkLayer::Ethernet,
            state: PortState::Active,
            lid: 0,
            gids: vec![
                GidEntry {
                    gid: Gid(link_local),
                    gid_type: GidType::RoceV1,
                    ndev_ifindex: 2,
                },
                GidEntry {
                    gid,
                    gid_type: GidType::RoceV2,
                    ndev_ifindex: 2,
                },
            ],
            active_mtu: Mtu::Mtu1024,
            max_mtu: Mtu::Mtu4096,
        }
    }
}

/// A device's limits: creating more than these fails with `ENOMEM`, asking for more per
/// object with `EINVAL`.
#[derive(Clone, Copy, Debug)]
pub struct MockLimits {
    pub max_pd: u32,
    pub max_cq: u32,
    pub max_cqe: u32,
    pub max_qp: u32,
    pub max_qp_wr: u32,
    pub max_sge: u32,
    pub max_mr: u32,
    pub max_mr_size: u64,
    pub max_qp_rd_atom: u32,
}

impl Default for MockLimits {
    fn default() -> Self {
        MockLimits {
            max_pd: 16,
            max_cq: 16,
            max_cqe: 4096,
            max_qp: 64,
            max_qp_wr: 1024,
            max_sge: 16,
            max_mr: 256,
            max_mr_size: 1 << 32,
            max_qp_rd_atom: 16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MockDevice {
    pub name: String,
    pub ports: Vec<MockPort>,
    pub limits: MockLimits,
}

impl MockDevice {
    pub fn new(name: &str, ports: Vec<MockPort>) -> MockDevice {
        MockDevice {
            name: name.to_owned(),
            ports,
            limits: MockLimits::default(),
        }
    }
}

/// The calls a [`Fault::FailCall`] can fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Call {
    OpenDevice,
    QueryDevice,
    QueryPort,
    AllocPd,
    CreateCq,
    CreateQp,
    ModifyQp,
    RegMr,
    PostSend,
    PostRecv,
    PollCq,
    DestroyQp,
    DestroyCq,
    DeallocPd,
    DeregMr,
    CloseDevice,
    CreateCompChannel,
    DestroyCompChannel,
    ReqNotifyCq,
    /// `wait_events` (`ibv_get_cq_event` / `ibv_get_async_event`).
    GetEvent,
    /// Every event returned by GetEvent is acknowledged before it returns.
    AckCqEvents,
}

impl Call {
    fn verb(self) -> &'static str {
        match self {
            Call::OpenDevice => "ibv_open_device",
            Call::QueryDevice => "ibv_query_device",
            Call::QueryPort => "ibv_query_port",
            Call::AllocPd => "ibv_alloc_pd",
            Call::CreateCq => "ibv_create_cq",
            Call::CreateQp => "ibv_create_qp",
            Call::ModifyQp => "ibv_modify_qp",
            Call::RegMr => "ibv_reg_mr",
            Call::PostSend => "ibv_post_send",
            Call::PostRecv => "ibv_post_recv",
            Call::PollCq => "ibv_poll_cq",
            Call::DestroyQp => "ibv_destroy_qp",
            Call::DestroyCq => "ibv_destroy_cq",
            Call::DeallocPd => "ibv_dealloc_pd",
            Call::DeregMr => "ibv_dereg_mr",
            Call::CloseDevice => "ibv_close_device",
            Call::CreateCompChannel => "ibv_create_comp_channel",
            Call::DestroyCompChannel => "ibv_destroy_comp_channel",
            Call::ReqNotifyCq => "ibv_req_notify_cq",
            Call::GetEvent => "ibv_get_cq_event",
            Call::AckCqEvents => "ibv_ack_cq_events",
        }
    }

    fn releases(self) -> bool {
        matches!(
            self,
            Call::DestroyQp
                | Call::DestroyCq
                | Call::DeallocPd
                | Call::DeregMr
                | Call::CloseDevice
                | Call::DestroyCompChannel
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Of the calls of `call` on `node` from now on, the `nth` (0 = the next) fails with
    /// `errno`.
    FailCall {
        node: NodeId,
        call: Call,
        nth: u32,
        errno: i32,
    },
    /// Sends posted on `node` from now on move no sooner than `ticks` ticks after posting.
    SendDelay { node: NodeId, ticks: u64 },
    /// Completions on `node` become visible `ticks` ticks after the work finished.
    CompletionDelay { node: NodeId, ticks: u64 },
    /// The next send `node` moves completes with `status`; its queue pair goes to `Err`.
    FailNextSend { node: NodeId, status: WcStatus },
    /// Every open context on `node` gets this asynchronous event (`qp_num` and `port` as
    /// given).
    AsyncEvent { node: NodeId, event: AsyncEvent },
    /// Queue a notification without a corresponding WC (legal in libibverbs).
    SpuriousNotification { node: NodeId },
    /// Once, hide a newly visible completion from an empty poll and expose it just
    /// after that poll, without notifying. Models the poll/arm boundary exactly.
    CompletionAtPollBoundary { node: NodeId },
}

/// Live resources on a node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveCounts {
    pub contexts: usize,
    pub pds: usize,
    pub cqs: usize,
    pub qps: usize,
    pub mrs: usize,
    pub channels: usize,
}

impl LiveCounts {
    pub fn total(&self) -> usize {
        self.contexts + self.pds + self.cqs + self.qps + self.mrs + self.channels
    }
}

struct Node {
    devices: Vec<MockDevice>,
    alive: bool,
    send_delay: u64,
    completion_delay: u64,
    fail_next_send: Option<WcStatus>,
    log: Vec<String>,
}

struct CtxRec {
    node: NodeId,
    dev: usize,
    events: VecDeque<AsyncEvent>,
}

struct ChanRec {
    ctx: Handle,
    /// A notification is waiting to be taken.
    pending: VecDeque<Handle>,
}

struct PdRec {
    ctx: Handle,
}

struct CqRec {
    ctx: Handle,
    depth: u32,
    // Visibility tick, WC, and whether it has passed its notification edge.
    entries: VecDeque<(u64, RawCompletion, bool)>,
    overrun: bool,
    channel: Option<Handle>,
    /// Notify the channel when the next completion becomes visible.
    armed: bool,
    hide_once: bool,
    boundary_unpolled: bool,
}

struct SendJob {
    wr_id: u64,
    op: SendOp,
    sges: Vec<RawSge>,
    ready_at: u64,
    unreachable_since: Option<u64>,
    rnr_since: Option<u64>,
}

struct RecvJob {
    wr_id: u64,
    sges: Vec<RawSge>,
}

struct QpRec {
    node: NodeId,
    ctx: Handle,
    pd: Handle,
    send_cq: Handle,
    recv_cq: Handle,
    qp_num: u32,
    caps: QpCaps,
    state: QpState,
    port: u8,
    rtr: Option<RtrAttr>,
    rts: Option<RtsAttr>,
    sq: VecDeque<SendJob>,
    rq: VecDeque<RecvJob>,
    /// The remote access the queue pair grants (its `Init` attributes).
    access: Access,
}

struct MrRec {
    node: NodeId,
    pd: Handle,
    /// The address work requests use for its first byte (the virtual address, or a
    /// dma-buf region's iova, or a mock device address).
    addr: u64,
    /// Where that byte is in this process (the same as `addr` for host memory).
    host: u64,
    len: u64,
    lkey: u32,
    rkey: u32,
    access: Access,
    /// The mock dma-buf a region imports.
    dmabuf: Option<i32>,
    /// Keeps mock device memory alive while registered (as pinning does).
    _keep: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl MrRec {
    fn covers(&self, addr: u64, len: u64) -> bool {
        addr >= self.addr
            && addr
                .checked_add(len)
                .is_some_and(|e| e <= self.addr + self.len)
    }

    /// The host address of `addr` (inside the region).
    fn host_of(&self, addr: u64) -> *mut u8 {
        (self.host + (addr - self.addr)) as usize as *mut u8
    }
}

struct FaultRule {
    fault: Fault,
    seen: u32,
}

struct Fabric {
    tick: u64,
    retry_ticks: u64,
    rnr_ticks: u64,
    nodes: Vec<Node>,
    next_handle: Handle,
    next_qpn: u32,
    next_key: u32,
    ctxs: BTreeMap<Handle, CtxRec>,
    pds: BTreeMap<Handle, PdRec>,
    cqs: BTreeMap<Handle, CqRec>,
    qps: BTreeMap<Handle, QpRec>,
    mrs: BTreeMap<Handle, MrRec>,
    channels: BTreeMap<Handle, ChanRec>,
    faults: Vec<FaultRule>,
    violations: Vec<String>,
    /// What `fork_status` reports, and how `fork_init` answers (an errno to fail with);
    /// how often `fork_init` was called.
    fork: (ForkStatus, Option<i32>),
    fork_inits: u32,
}

/// The fabric: create it, add nodes, take each node's provider.
pub struct MockFabric {
    inner: Mutex<Fabric>,
    /// Signalled whenever a call may have changed what a waiting thread could see.
    wake: Condvar,
}

impl Default for MockFabric {
    fn default() -> Self {
        MockFabric {
            inner: Mutex::new(Fabric {
                tick: 0,
                retry_ticks: 16,
                rnr_ticks: 4,
                nodes: Vec::new(),
                next_handle: 1,
                next_qpn: 0x100,
                next_key: 1,
                ctxs: BTreeMap::new(),
                pds: BTreeMap::new(),
                cqs: BTreeMap::new(),
                qps: BTreeMap::new(),
                mrs: BTreeMap::new(),
                channels: BTreeMap::new(),
                faults: Vec::new(),
                violations: Vec::new(),
                fork: (ForkStatus::Unneeded, None),
                fork_inits: 0,
            }),
            wake: Condvar::new(),
        }
    }
}

fn verbs(call: Call, errno: i32) -> Error {
    Error::Verbs {
        call: call.verb(),
        errno,
    }
}

impl MockFabric {
    pub fn new() -> Arc<MockFabric> {
        Arc::new(MockFabric::default())
    }

    fn lock(&self) -> MutexGuard<'_, Fabric> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn add_node(&self, devices: Vec<MockDevice>) -> NodeId {
        let mut f = self.lock();
        f.nodes.push(Node {
            devices,
            alive: true,
            send_delay: 0,
            completion_delay: 0,
            fail_next_send: None,
            log: Vec::new(),
        });
        f.nodes.len() - 1
    }

    pub fn provider(self: &Arc<Self>, node: NodeId) -> Arc<dyn Provider> {
        assert!(node < self.lock().nodes.len(), "no node {node}");
        Arc::new(MockProvider {
            fabric: self.clone(),
            node,
        })
    }

    pub fn inject(&self, fault: Fault) {
        let mut f = self.lock();
        match fault {
            Fault::FailCall { .. } => f.faults.push(FaultRule { fault, seen: 0 }),
            Fault::SendDelay { node, ticks } => f.nodes[node].send_delay = ticks,
            Fault::CompletionDelay { node, ticks } => f.nodes[node].completion_delay = ticks,
            Fault::FailNextSend { node, status } => f.nodes[node].fail_next_send = Some(status),
            Fault::AsyncEvent { node, event } => {
                for c in f.ctxs.values_mut().filter(|c| c.node == node) {
                    c.events.push_back(event);
                }
            }
            Fault::SpuriousNotification { node } => {
                let cqs: Vec<(Handle, Handle)> = f
                    .cqs
                    .iter()
                    .filter_map(|(cq, c)| {
                        (f.ctxs[&c.ctx].node == node)
                            .then_some(c.channel.map(|ch| (*cq, ch)))
                            .flatten()
                    })
                    .collect();
                for (cq, ch) in cqs {
                    f.channels.get_mut(&ch).unwrap().pending.push_back(cq);
                }
            }
            Fault::CompletionAtPollBoundary { node } => {
                let ctxs: Vec<Handle> = f
                    .ctxs
                    .iter()
                    .filter(|(_, c)| c.node == node)
                    .map(|(h, _)| *h)
                    .collect();
                for c in f.cqs.values_mut().filter(|c| ctxs.contains(&c.ctx)) {
                    c.hide_once = true;
                    c.armed = false;
                }
            }
        }
        drop(f);
        self.wake.notify_all();
    }

    /// What every provider of this fabric says about fork safety: `status` from
    /// `fork_status`, and `fork_init` failing with `init_errno` if given (else enabling
    /// it).
    pub fn set_fork(&self, status: ForkStatus, init_errno: Option<i32>) {
        self.lock().fork = (status, init_errno);
    }

    /// How often `fork_init` was called.
    pub fn fork_inits(&self) -> u32 {
        self.lock().fork_inits
    }

    /// The node is gone: nothing reaches it, it moves nothing, and its calls fail (except
    /// releases, so its owner's cleanup still runs).
    pub fn kill(&self, node: NodeId) {
        self.lock().nodes[node].alive = false;
        self.wake.notify_all();
    }

    /// Changes a port's state; contexts on its device get a `PortErr` event when it leaves
    /// `Active` and a `PortActive` one when it returns.
    pub fn set_port_state(&self, node: NodeId, device: usize, port: u8, state: PortState) {
        let mut f = self.lock();
        let p = &mut f.nodes[node].devices[device].ports[usize::from(port) - 1];
        let was = p.state;
        p.state = state;
        let kind = match (was == PortState::Active, state == PortState::Active) {
            (true, false) => Some(AsyncEventKind::PortErr),
            (false, true) => Some(AsyncEventKind::PortActive),
            _ => None,
        };
        if let Some(kind) = kind {
            let event = AsyncEvent {
                kind,
                qp_num: None,
                port: Some(port),
            };
            for c in f
                .ctxs
                .values_mut()
                .filter(|c| c.node == node && c.dev == device)
            {
                c.events.push_back(event);
            }
        }
        drop(f);
        self.wake.notify_all();
    }

    /// How many ticks a send keeps trying an unreachable peer, and per RNR retry.
    pub fn set_retry(&self, retry_ticks: u64, rnr_ticks: u64) {
        let mut f = self.lock();
        f.retry_ticks = retry_ticks;
        f.rnr_ticks = rnr_ticks;
    }

    /// Advances the fabric `n` ticks (each `poll_cq` advances it one).
    pub fn step(&self, n: u64) {
        let mut f = self.lock();
        for _ in 0..n {
            f.progress();
        }
        drop(f);
        self.wake.notify_all();
    }

    pub fn tick(&self) -> u64 {
        self.lock().tick
    }

    /// Promises the provider's caller broke (see the module docs). Empty in a correct run.
    pub fn violations(&self) -> Vec<String> {
        self.lock().violations.clone()
    }

    /// The node's resource calls, in order: `"<verb> #<handle>"`.
    pub fn log(&self, node: NodeId) -> Vec<String> {
        self.lock().nodes[node].log.clone()
    }

    pub fn live(&self, node: NodeId) -> LiveCounts {
        let f = self.lock();
        let on = |ctx: Handle| f.ctxs.get(&ctx).is_some_and(|c| c.node == node);
        LiveCounts {
            contexts: f.ctxs.values().filter(|c| c.node == node).count(),
            pds: f.pds.values().filter(|p| on(p.ctx)).count(),
            cqs: f.cqs.values().filter(|c| on(c.ctx)).count(),
            qps: f.qps.values().filter(|q| q.node == node).count(),
            mrs: f.mrs.values().filter(|m| m.node == node).count(),
            channels: f.channels.values().filter(|c| on(c.ctx)).count(),
        }
    }

    /// Work requests the fabric still holds for queue pair `qp_num` (sends, receives).
    pub fn queued(&self, qp_num: u32) -> (usize, usize) {
        let f = self.lock();
        f.qps
            .values()
            .find(|q| q.qp_num == qp_num)
            .map_or((0, 0), |q| (q.sq.len(), q.rq.len()))
    }
}

impl Fabric {
    fn handle(&mut self) -> Handle {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }

    fn log(&mut self, node: NodeId, call: Call, handle: Handle) {
        self.nodes[node]
            .log
            .push(format!("{} #{handle}", call.verb()));
    }

    /// Every call goes through here: a dead node refuses all but releases, and an injected
    /// failure fires.
    fn enter(&mut self, node: NodeId, call: Call) -> Result<()> {
        if !self.nodes[node].alive && !call.releases() {
            return Err(verbs(call, EIO));
        }
        let mut fired = None;
        for (i, rule) in self.faults.iter_mut().enumerate() {
            if let Fault::FailCall {
                node: n,
                call: c,
                nth,
                errno,
            } = rule.fault
            {
                if n == node && c == call {
                    if rule.seen == nth {
                        fired = Some((i, errno));
                        break;
                    }
                    rule.seen += 1;
                }
            }
        }
        if let Some((i, errno)) = fired {
            self.faults.remove(i);
            self.nodes[node].log.push(format!("{} failed", call.verb()));
            return Err(verbs(call, errno));
        }
        Ok(())
    }

    fn ctx(&self, node: NodeId, ctx: Handle, call: Call) -> Result<&CtxRec> {
        self.ctxs
            .get(&ctx)
            .filter(|c| c.node == node)
            .ok_or_else(|| verbs(call, EINVAL))
    }

    fn device(&self, ctx: Handle) -> &MockDevice {
        let c = &self.ctxs[&ctx];
        &self.nodes[c.node].devices[c.dev]
    }

    fn port(&self, ctx: Handle, port: u8) -> Option<&MockPort> {
        self.device(ctx)
            .ports
            .get(usize::from(port).checked_sub(1)?)
    }

    /// How many of the contexts `ctxs` yields are on the same device as `ctx`.
    fn on_device(&self, ctx: Handle, ctxs: impl Iterator<Item = Handle>) -> usize {
        let c = &self.ctxs[&ctx];
        let (node, dev) = (c.node, c.dev);
        ctxs.filter(|other| {
            self.ctxs
                .get(other)
                .is_some_and(|o| o.node == node && o.dev == dev)
        })
        .count()
    }

    fn push_completion(&mut self, cq: Handle, node: NodeId, wc: RawCompletion) {
        let visible = self.tick + self.nodes[node].completion_delay;
        let rec = self
            .cqs
            .get_mut(&cq)
            .expect("a queue pair's completion queue exists");
        if rec.entries.len() >= rec.depth as usize {
            rec.overrun = true;
            self.violations.push(format!(
                "completion queue #{cq} overran its depth {}",
                rec.depth
            ));
            return;
        }
        rec.entries.push_back((visible, wc, false));
        self.notify_visible();
    }

    /// Marks the channel of every armed queue whose oldest completion is visible now.
    fn notify_visible(&mut self) {
        let tick = self.tick;
        let mut ring = Vec::new();
        for (&cq, rec) in self.cqs.iter_mut() {
            if rec.hide_once {
                continue;
            }
            let mut new = false;
            for (visible, _, seen) in &mut rec.entries {
                if *visible <= tick && !*seen {
                    *seen = true;
                    new = true;
                }
            }
            if rec.armed && new {
                rec.armed = false;
                ring.extend(rec.channel.map(|ch| (ch, cq)));
            }
        }
        for (ch, cq) in ring {
            if let Some(c) = self.channels.get_mut(&ch) {
                c.pending.push_back(cq);
            }
        }
    }

    fn complete(
        &mut self,
        qp: Handle,
        recv: bool,
        wr_id: u64,
        status: WcStatus,
        len: u32,
        imm: Option<u32>,
    ) {
        let q = &self.qps[&qp];
        let (cq, node, qp_num) = (if recv { q.recv_cq } else { q.send_cq }, q.node, q.qp_num);
        self.push_completion(
            cq,
            node,
            RawCompletion {
                wr_id,
                qp_num,
                status,
                byte_len: len,
                imm,
                vendor_err: 0,
            },
        );
    }

    /// The queue pair goes to `Err`: everything still queued completes flushed.
    fn fail_qp(&mut self, qp: Handle) {
        let rec = self.qps.get_mut(&qp).expect("queue pair exists");
        rec.state = QpState::Err;
        let sends: Vec<u64> = rec.sq.drain(..).map(|j| j.wr_id).collect();
        let recvs: Vec<u64> = rec.rq.drain(..).map(|j| j.wr_id).collect();
        for wr_id in sends {
            self.complete(qp, false, wr_id, WcStatus::WrFlushErr, 0, None);
        }
        for wr_id in recvs {
            self.complete(qp, true, wr_id, WcStatus::WrFlushErr, 0, None);
        }
    }

    /// Finishes the send at the head of `qp`'s queue.
    fn finish_send(&mut self, qp: Handle, status: WcStatus, len: u32) {
        let job = self
            .qps
            .get_mut(&qp)
            .and_then(|q| q.sq.pop_front())
            .expect("a send to finish");
        self.complete(qp, false, job.wr_id, status, len, None);
        if status != WcStatus::Success {
            self.fail_qp(qp);
        }
    }

    fn find_mr(&self, node: NodeId, pd: Handle, key: u32, remote: bool) -> Option<&MrRec> {
        self.mrs.values().find(|m| {
            m.node == node && m.pd == pd && if remote { m.rkey == key } else { m.lkey == key }
        })
    }

    fn address_matches(&self, address: &Address, ctx: Handle, port: u8) -> bool {
        let Some(p) = self.port(ctx, port) else {
            return false;
        };
        match address {
            Address::Ib { lid, .. } => p.link_layer == LinkLayer::InfiniBand && p.lid == *lid,
            Address::Roce(grh) => {
                p.link_layer == LinkLayer::Ethernet && p.gids.iter().any(|g| g.gid == grh.dgid)
            }
        }
    }

    /// The queue pair `src` is connected to, if a packet could get there and back now.
    fn route(&self, src: Handle) -> Option<Handle> {
        let s = &self.qps[&src];
        let (rtr, rts) = (s.rtr?, s.rts?);
        let t_handle = *self
            .qps
            .iter()
            .find(|(_, q)| q.qp_num == rtr.dest_qp_num)?
            .0;
        let t = &self.qps[&t_handle];
        let t_rtr = t.rtr?;
        let ports_up = [(s.ctx, s.port), (t.ctx, t.port)].iter().all(|&(c, p)| {
            self.port(c, p)
                .is_some_and(|p| p.state == PortState::Active)
        });
        let ok = self.nodes[t.node].alive
            && ports_up
            && matches!(t.state, QpState::Rtr | QpState::Rts)
            && self.address_matches(&rtr.address, t.ctx, t.port)
            && self.address_matches(&t_rtr.address, s.ctx, s.port)
            && t_rtr.dest_qp_num == s.qp_num
            && t_rtr.rq_psn == rts.sq_psn
            && self
                .port(t.ctx, t.port)
                .is_some_and(|p| rtr.path_mtu <= p.active_mtu);
        ok.then_some(t_handle)
    }

    /// The bytes `sges` name in `node`'s memory, or None for a bad key or range.
    fn gather(&self, node: NodeId, pd: Handle, sges: &[RawSge]) -> Option<Vec<u8>> {
        let mut data = Vec::new();
        for sge in sges {
            let mr = self.find_mr(node, pd, sge.lkey, false)?;
            if !mr.covers(sge.addr, u64::from(sge.len)) {
                return None;
            }
            let start = data.len();
            data.resize(start + sge.len as usize, 0);
            // SAFETY: the range lies inside a registered region (checked above), and a
            // provider's caller keeps registered memory valid until it is deregistered
            // (Provider::reg_mr's contract), or the region keeps mock device memory alive;
            // it is still registered here.
            unsafe {
                ptr::copy_nonoverlapping(
                    mr.host_of(sge.addr),
                    data.as_mut_ptr().add(start),
                    sge.len as usize,
                )
            };
        }
        Some(data)
    }

    /// Writes `data` across `sges` in `node`'s memory. Err(true) if it doesn't fit,
    /// Err(false) for a bad key, range or access right.
    fn scatter(
        &self,
        node: NodeId,
        pd: Handle,
        sges: &[RawSge],
        data: &[u8],
    ) -> std::result::Result<(), bool> {
        let room: u64 = sges.iter().map(|s| u64::from(s.len)).sum();
        if (data.len() as u64) > room {
            return Err(true);
        }
        let mut targets = Vec::with_capacity(sges.len());
        for sge in sges {
            let mr = self.find_mr(node, pd, sge.lkey, false).ok_or(false)?;
            if !mr.covers(sge.addr, u64::from(sge.len)) || !mr.access.contains(Access::LOCAL_WRITE)
            {
                return Err(false);
            }
            targets.push(mr.host_of(sge.addr));
        }
        let mut done = 0;
        for (sge, target) in sges.iter().zip(targets) {
            let n = (data.len() - done).min(sge.len as usize);
            // SAFETY: inside a registered, locally writable region (checked above), kept
            // valid by Provider::reg_mr's contract or by the region; `data` is a separate
            // buffer.
            unsafe { ptr::copy_nonoverlapping(data.as_ptr().add(done), target, n) };
            done += n;
        }
        Ok(())
    }

    /// Tries to move the send at the head of `src`'s queue. True if it finished.
    fn try_send(&mut self, src: Handle) -> bool {
        let tick = self.tick;
        let (node, pd, rts, op, sges, unreachable_since, rnr_since) = {
            let q = &self.qps[&src];
            let Some(job) = q.sq.front() else {
                return false;
            };
            if q.state != QpState::Rts || job.ready_at > tick {
                return false;
            }
            (
                q.node,
                q.pd,
                q.rts.expect("Rts has its attributes"),
                job.op,
                job.sges.clone(),
                job.unreachable_since,
                job.rnr_since,
            )
        };
        if !self.nodes[node].alive {
            return false;
        }
        if let Some(status) = self.nodes[node].fail_next_send.take() {
            self.finish_send(src, status, 0);
            return true;
        }
        let Some(target) = self.route(src) else {
            let since = unreachable_since.unwrap_or(tick);
            if tick - since >= self.retry_ticks {
                self.finish_send(src, WcStatus::RetryExcErr, 0);
                return true;
            }
            self.qps.get_mut(&src).unwrap().sq[0].unreachable_since = Some(since);
            return false;
        };
        let Some(data) = self.gather(node, pd, &sges) else {
            self.finish_send(src, WcStatus::LocProtErr, 0);
            return true;
        };
        let (t_node, t_pd) = {
            let t = &self.qps[&target];
            (t.node, t.pd)
        };
        let len = data.len() as u32;
        let (imm, consumes_recv) = match op {
            SendOp::Send => (None, true),
            SendOp::SendWithImm(imm) => (Some(imm), true),
            SendOp::Write(_) => (None, false),
            SendOp::WriteWithImm(_, imm) => (Some(imm), true),
        };
        if consumes_recv && self.qps[&target].rq.is_empty() {
            // Receiver not ready: retry until the RNR budget runs out (7 = forever).
            let since = rnr_since.unwrap_or(tick);
            let limit = if rts.rnr_retry == 7 {
                u64::MAX
            } else {
                (u64::from(rts.rnr_retry) + 1) * self.rnr_ticks
            };
            if tick - since >= limit {
                self.finish_send(src, WcStatus::RnrRetryExcErr, 0);
                return true;
            }
            self.qps.get_mut(&src).unwrap().sq[0].rnr_since = Some(since);
            return false;
        }
        if let SendOp::Write(remote) | SendOp::WriteWithImm(remote, _) = op {
            // The responder's queue pair must grant remote writes, and the region too.
            let qp_ok = self.qps[&target].access.contains(Access::REMOTE_WRITE);
            let target_host = self
                .find_mr(t_node, t_pd, remote.rkey, true)
                .filter(|mr| {
                    qp_ok
                        && mr.access.contains(Access::REMOTE_WRITE)
                        && mr.covers(remote.addr, u64::from(len))
                })
                .map(|mr| mr.host_of(remote.addr));
            let Some(target_host) = target_host else {
                self.finish_send(src, WcStatus::RemAccessErr, 0);
                return true;
            };
            // SAFETY: inside the peer's registered, remotely writable region (checked
            // above), kept valid by Provider::reg_mr's contract or by the region; `data` is
            // a separate buffer.
            unsafe { ptr::copy_nonoverlapping(data.as_ptr(), target_host, data.len()) };
        }
        if consumes_recv {
            let recv = self
                .qps
                .get_mut(&target)
                .and_then(|t| t.rq.pop_front())
                .expect("a receive is posted");
            let written = if matches!(op, SendOp::Send | SendOp::SendWithImm(_)) {
                self.scatter(t_node, t_pd, &recv.sges, &data)
            } else {
                Ok(())
            };
            match written {
                Ok(()) => {
                    self.complete(target, true, recv.wr_id, WcStatus::Success, len, imm);
                }
                Err(too_long) => {
                    let status = if too_long {
                        WcStatus::LocLenErr
                    } else {
                        WcStatus::LocProtErr
                    };
                    self.complete(target, true, recv.wr_id, status, 0, None);
                    self.fail_qp(target);
                    let status = if too_long {
                        WcStatus::RemInvReqErr
                    } else {
                        WcStatus::RemOpErr
                    };
                    self.finish_send(src, status, 0);
                    return true;
                }
            }
        }
        self.finish_send(src, WcStatus::Success, len);
        true
    }

    /// One tick: every queue pair in `Rts` moves what it can, in order.
    fn progress(&mut self) {
        self.tick += 1;
        let qps: Vec<Handle> = self.qps.keys().copied().collect();
        for qp in qps {
            while self.qps.contains_key(&qp) && self.try_send(qp) {}
        }
        // Delayed completions that became visible with this tick.
        self.notify_visible();
    }
}

/// One node's provider.
pub struct MockProvider {
    fabric: Arc<MockFabric>,
    node: NodeId,
}

impl std::fmt::Debug for MockProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockProvider(node {})", self.node)
    }
}

impl MockProvider {
    fn enter(&self, call: Call) -> Result<MutexGuard<'_, Fabric>> {
        let mut f = self.fabric.lock();
        f.enter(self.node, call)?;
        Ok(f)
    }

    /// Wakes threads blocked in `wait_events` after a call that changed something.
    fn changed(&self) {
        self.fabric.wake.notify_all();
    }

    fn new_cq(&self, ctx: Handle, depth: u32, channel: Option<Handle>) -> Result<(Handle, u32)> {
        let mut f = self.enter(Call::CreateCq)?;
        f.ctx(self.node, ctx, Call::CreateCq)?;
        if let Some(ch) = channel {
            if f.channels.get(&ch).map(|c| c.ctx) != Some(ctx) {
                return Err(verbs(Call::CreateCq, EINVAL));
            }
        }
        let limits = f.device(ctx).limits;
        if depth == 0 || depth > limits.max_cqe {
            return Err(verbs(Call::CreateCq, EINVAL));
        }
        if f.on_device(ctx, f.cqs.values().map(|c| c.ctx)) >= limits.max_cq as usize {
            return Err(verbs(Call::CreateCq, ENOMEM));
        }
        let h = f.handle();
        f.cqs.insert(
            h,
            CqRec {
                ctx,
                depth,
                entries: VecDeque::new(),
                overrun: false,
                channel,
                armed: false,
                hide_once: false,
                boundary_unpolled: false,
            },
        );
        f.log(self.node, Call::CreateCq, h);
        Ok((h, depth))
    }
}

impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn describe(&self) -> String {
        "MOCK: in-process fabric (not RDMA)".into()
    }

    fn fork_status(&self) -> Result<ForkStatus> {
        Ok(self.fabric.lock().fork.0)
    }

    fn fork_init(&self) -> Result<()> {
        let mut f = self.fabric.lock();
        f.fork_inits += 1;
        f.nodes[self.node].log.push("ibv_fork_init".into());
        match f.fork.1 {
            Some(errno) => Err(Error::Verbs {
                call: "ibv_fork_init",
                errno,
            }),
            None => {
                f.fork.0 = ForkStatus::Enabled;
                Ok(())
            }
        }
    }

    fn list_devices(&self) -> Result<Vec<String>> {
        let f = self.fabric.lock();
        Ok(f.nodes[self.node]
            .devices
            .iter()
            .map(|d| d.name.clone())
            .collect())
    }

    fn open_device(&self, name: &str) -> Result<Handle> {
        let mut f = self.enter(Call::OpenDevice)?;
        let dev = f.nodes[self.node]
            .devices
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| verbs(Call::OpenDevice, ENODEV))?;
        let h = f.handle();
        f.ctxs.insert(
            h,
            CtxRec {
                node: self.node,
                dev,
                events: VecDeque::new(),
            },
        );
        f.log(self.node, Call::OpenDevice, h);
        Ok(h)
    }

    fn close_device(&self, ctx: Handle) -> Result<()> {
        let mut f = self.enter(Call::CloseDevice)?;
        f.ctx(self.node, ctx, Call::CloseDevice)?;
        let children = f.pds.values().filter(|p| p.ctx == ctx).count()
            + f.cqs.values().filter(|c| c.ctx == ctx).count()
            + f.qps.values().filter(|q| q.ctx == ctx).count()
            + f.channels.values().filter(|c| c.ctx == ctx).count();
        if children > 0 {
            f.violations.push(format!(
                "context #{ctx} closed with {children} live children"
            ));
            return Err(verbs(Call::CloseDevice, EBUSY));
        }
        f.ctxs.remove(&ctx);
        f.log(self.node, Call::CloseDevice, ctx);
        Ok(())
    }

    fn query_device(&self, ctx: Handle) -> Result<DeviceAttr> {
        let f = self.enter(Call::QueryDevice)?;
        let c = f.ctx(self.node, ctx, Call::QueryDevice)?;
        let dev = &f.nodes[c.node].devices[c.dev];
        let l = dev.limits;
        Ok(DeviceAttr {
            max_mr_size: l.max_mr_size,
            node_guid: ((self.node as u64) << 16) | c.dev as u64,
            max_qp: l.max_qp,
            max_qp_wr: l.max_qp_wr,
            max_sge: l.max_sge,
            max_cq: l.max_cq,
            max_cqe: l.max_cqe,
            max_mr: l.max_mr,
            max_pd: l.max_pd,
            max_qp_rd_atom: l.max_qp_rd_atom,
            max_qp_init_rd_atom: l.max_qp_rd_atom,
            vendor_id: 0,
            vendor_part_id: 0,
            phys_port_cnt: dev.ports.len() as u8,
            fw_ver: "mock".into(),
        })
    }

    fn query_port(&self, ctx: Handle, port: u8) -> Result<PortAttr> {
        let f = self.enter(Call::QueryPort)?;
        f.ctx(self.node, ctx, Call::QueryPort)?;
        let p = f
            .port(ctx, port)
            .ok_or_else(|| verbs(Call::QueryPort, EINVAL))?;
        Ok(PortAttr {
            state: p.state,
            max_mtu: p.max_mtu,
            active_mtu: p.active_mtu,
            gid_tbl_len: p.gids.len() as u32,
            lid: p.lid,
            sm_lid: if p.link_layer == LinkLayer::InfiniBand {
                1
            } else {
                0
            },
            link_layer: p.link_layer,
        })
    }

    fn query_gid(&self, ctx: Handle, port: u8, index: u32) -> Result<GidEntry> {
        let f = self.enter(Call::QueryPort)?;
        f.ctx(self.node, ctx, Call::QueryPort)?;
        let p = f
            .port(ctx, port)
            .ok_or_else(|| verbs(Call::QueryPort, EINVAL))?;
        p.gids
            .get(index as usize)
            .copied()
            .ok_or_else(|| verbs(Call::QueryPort, EINVAL))
    }

    fn alloc_pd(&self, ctx: Handle) -> Result<Handle> {
        let mut f = self.enter(Call::AllocPd)?;
        f.ctx(self.node, ctx, Call::AllocPd)?;
        let count = f.on_device(ctx, f.pds.values().map(|p| p.ctx));
        if count >= f.device(ctx).limits.max_pd as usize {
            return Err(verbs(Call::AllocPd, ENOMEM));
        }
        let h = f.handle();
        f.pds.insert(h, PdRec { ctx });
        f.log(self.node, Call::AllocPd, h);
        Ok(h)
    }

    fn dealloc_pd(&self, pd: Handle) -> Result<()> {
        let mut f = self.enter(Call::DeallocPd)?;
        if !f.pds.contains_key(&pd) {
            return Err(verbs(Call::DeallocPd, EINVAL));
        }
        if f.qps.values().any(|q| q.pd == pd) || f.mrs.values().any(|m| m.pd == pd) {
            f.violations
                .push(format!("protection domain #{pd} deallocated while in use"));
            return Err(verbs(Call::DeallocPd, EBUSY));
        }
        f.pds.remove(&pd);
        f.log(self.node, Call::DeallocPd, pd);
        Ok(())
    }

    fn create_cq(&self, ctx: Handle, depth: u32) -> Result<(Handle, u32)> {
        self.new_cq(ctx, depth, None)
    }

    fn create_cq_on_channel(
        &self,
        ctx: Handle,
        depth: u32,
        channel: Handle,
    ) -> Result<(Handle, u32)> {
        self.new_cq(ctx, depth, Some(channel))
    }

    fn create_comp_channel(&self, ctx: Handle) -> Result<Handle> {
        let mut f = self.enter(Call::CreateCompChannel)?;
        f.ctx(self.node, ctx, Call::CreateCompChannel)?;
        let h = f.handle();
        f.channels.insert(
            h,
            ChanRec {
                ctx,
                pending: VecDeque::new(),
            },
        );
        f.log(self.node, Call::CreateCompChannel, h);
        Ok(h)
    }

    fn destroy_comp_channel(&self, channel: Handle) -> Result<()> {
        let mut f = self.enter(Call::DestroyCompChannel)?;
        if !f.channels.contains_key(&channel) {
            return Err(verbs(Call::DestroyCompChannel, EINVAL));
        }
        if f.cqs.values().any(|c| c.channel == Some(channel)) {
            f.violations.push(format!(
                "completion channel #{channel} destroyed while a completion queue uses it"
            ));
            return Err(verbs(Call::DestroyCompChannel, EBUSY));
        }
        f.channels.remove(&channel);
        f.log(self.node, Call::DestroyCompChannel, channel);
        Ok(())
    }

    fn req_notify_cq(&self, cq: Handle) -> Result<()> {
        let mut f = self.enter(Call::ReqNotifyCq)?;
        let rec = f
            .cqs
            .get_mut(&cq)
            .ok_or_else(|| verbs(Call::ReqNotifyCq, EINVAL))?;
        if rec.channel.is_none() {
            return Err(verbs(Call::ReqNotifyCq, EINVAL));
        }
        // The next visible completion marks the channel. Callers poll after arming, as
        // with verbs, so one already queued is never missed either way.
        rec.armed = true;
        Ok(())
    }

    fn wait_events(&self, ctx: Handle, channel: Handle, timeout: Duration) -> Result<Events> {
        let deadline = Instant::now() + timeout;
        let mut f = self.enter(Call::GetEvent)?;
        loop {
            if f.channels.get(&channel).map(|c| c.ctx) != Some(ctx) {
                return Err(verbs(Call::GetEvent, EINVAL));
            }
            f.ctx(self.node, ctx, Call::GetEvent)?;
            if !f.nodes[self.node].alive {
                return Err(verbs(Call::GetEvent, EIO));
            }
            f.progress();
            if !timeout.is_zero()
                && f.cqs
                    .values()
                    .any(|c| c.channel == Some(channel) && c.boundary_unpolled)
            {
                f.violations
                    .push("slept with a poll/arm boundary completion left unpolled".into());
                return Err(verbs(Call::GetEvent, EIO));
            }
            let ch = f.channels.get_mut(&channel).expect("checked above");
            let pending = std::mem::take(&mut ch.pending);
            let completion = !pending.is_empty();
            for cq in pending {
                f.log(self.node, Call::AckCqEvents, cq);
            }
            let async_events: Vec<AsyncEvent> = f
                .ctxs
                .get_mut(&ctx)
                .expect("checked above")
                .events
                .drain(..)
                .collect();
            if completion || !async_events.is_empty() {
                return Ok(Events {
                    completion,
                    async_events,
                });
            }
            // Wake an event waiter to exercise the injected empty-poll boundary. This
            // is a spurious wake, not a CQ notification; no event is fabricated/acked.
            if f.cqs.values().any(|c| {
                c.channel == Some(channel)
                    && c.hide_once
                    && c.entries.front().is_some_and(|(v, _, _)| *v <= f.tick)
            }) {
                return Ok(Events::default());
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(Events::default());
            }
            let nap = (deadline - now).min(Duration::from_millis(1));
            f = self
                .fabric
                .wake
                .wait_timeout(f, nap)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    fn destroy_cq(&self, cq: Handle) -> Result<()> {
        let mut f = self.enter(Call::DestroyCq)?;
        if !f.cqs.contains_key(&cq) {
            return Err(verbs(Call::DestroyCq, EINVAL));
        }
        if f.qps.values().any(|q| q.send_cq == cq || q.recv_cq == cq) {
            f.violations.push(format!(
                "completion queue #{cq} destroyed while a queue pair uses it"
            ));
            return Err(verbs(Call::DestroyCq, EBUSY));
        }
        if f.channels.values().any(|ch| ch.pending.contains(&cq)) {
            f.violations.push(format!(
                "completion queue #{cq} destroyed with unacknowledged events"
            ));
            return Err(verbs(Call::DestroyCq, EBUSY));
        }
        f.cqs.remove(&cq);
        f.log(self.node, Call::DestroyCq, cq);
        Ok(())
    }

    fn create_rc_qp(
        &self,
        pd: Handle,
        send_cq: Handle,
        recv_cq: Handle,
        caps: &QpCaps,
    ) -> Result<QpCreated> {
        let mut f = self.enter(Call::CreateQp)?;
        let ctx = f
            .pds
            .get(&pd)
            .ok_or_else(|| verbs(Call::CreateQp, EINVAL))?
            .ctx;
        for cq in [send_cq, recv_cq] {
            if f.cqs.get(&cq).map(|c| c.ctx) != Some(ctx) {
                return Err(verbs(Call::CreateQp, EINVAL));
            }
        }
        let limits = f.device(ctx).limits;
        if caps.max_send_wr > limits.max_qp_wr
            || caps.max_recv_wr > limits.max_qp_wr
            || caps.max_send_sge > limits.max_sge
            || caps.max_recv_sge > limits.max_sge
        {
            return Err(verbs(Call::CreateQp, EINVAL));
        }
        if f.qps.values().filter(|q| q.ctx == ctx).count() >= limits.max_qp as usize {
            return Err(verbs(Call::CreateQp, ENOMEM));
        }
        let h = f.handle();
        let qp_num = f.next_qpn;
        f.next_qpn += 1;
        f.qps.insert(
            h,
            QpRec {
                node: self.node,
                ctx,
                pd,
                send_cq,
                recv_cq,
                qp_num,
                caps: *caps,
                state: QpState::Reset,
                port: 0,
                rtr: None,
                rts: None,
                sq: VecDeque::new(),
                rq: VecDeque::new(),
                access: Access::NONE,
            },
        );
        f.log(self.node, Call::CreateQp, h);
        Ok(QpCreated {
            handle: h,
            qp_num,
            caps: *caps,
        })
    }

    fn modify_qp(&self, qp: Handle, transition: &Transition) -> Result<()> {
        let mut f = self.enter(Call::ModifyQp)?;
        let bad = || verbs(Call::ModifyQp, EINVAL);
        let (state, ctx) = {
            let q = f.qps.get(&qp).ok_or_else(bad)?;
            (q.state, q.ctx)
        };
        match transition {
            Transition::Init(attr) => {
                if state != QpState::Reset || f.port(ctx, attr.port).is_none() {
                    return Err(bad());
                }
                let q = f.qps.get_mut(&qp).unwrap();
                q.state = QpState::Init;
                q.port = attr.port;
                q.access = attr.access;
            }
            Transition::Rtr { port, attr } => {
                let p = f.port(ctx, *port).ok_or_else(bad)?;
                let addressing_ok = match attr.address {
                    Address::Ib { lid, .. } => p.link_layer == LinkLayer::InfiniBand && lid != 0,
                    Address::Roce(_) => p.link_layer == LinkLayer::Ethernet,
                };
                if state != QpState::Init || !addressing_ok || attr.path_mtu > p.active_mtu {
                    return Err(bad());
                }
                let q = f.qps.get_mut(&qp).unwrap();
                q.state = QpState::Rtr;
                q.rtr = Some(*attr);
            }
            Transition::Rts(attr) => {
                if state != QpState::Rtr {
                    return Err(bad());
                }
                let q = f.qps.get_mut(&qp).unwrap();
                q.state = QpState::Rts;
                q.rts = Some(*attr);
            }
            Transition::Error => f.fail_qp(qp),
            Transition::Reset => {
                let q = f.qps.get_mut(&qp).unwrap();
                let dropped = q.sq.len() + q.rq.len();
                q.sq.clear();
                q.rq.clear();
                q.state = QpState::Reset;
                q.rtr = None;
                q.rts = None;
                if dropped > 0 {
                    f.violations.push(format!(
                        "queue pair #{qp} reset with {dropped} work requests outstanding"
                    ));
                }
            }
        }
        drop(f);
        self.changed();
        Ok(())
    }

    fn destroy_qp(&self, qp: Handle) -> Result<()> {
        let mut f = self.enter(Call::DestroyQp)?;
        let q = f
            .qps
            .remove(&qp)
            .ok_or_else(|| verbs(Call::DestroyQp, EINVAL))?;
        let pending = q.sq.len() + q.rq.len();
        if pending > 0 {
            f.violations.push(format!(
                "queue pair {:#x} destroyed with {pending} work requests not completed",
                q.qp_num
            ));
        }
        f.log(self.node, Call::DestroyQp, qp);
        Ok(())
    }

    unsafe fn reg_mr(
        &self,
        pd: Handle,
        addr: *mut u8,
        len: usize,
        access: Access,
    ) -> Result<MrKeys> {
        let mut f = self.enter(Call::RegMr)?;
        let ctx = f
            .pds
            .get(&pd)
            .ok_or_else(|| verbs(Call::RegMr, EINVAL))?
            .ctx;
        let limits = f.device(ctx).limits;
        if addr.is_null() || len == 0 || len as u64 > limits.max_mr_size {
            return Err(verbs(Call::RegMr, EINVAL));
        }
        if access.contains(Access::REMOTE_WRITE) && !access.contains(Access::LOCAL_WRITE) {
            return Err(verbs(Call::RegMr, EINVAL));
        }
        let count = f.on_device(
            ctx,
            f.mrs
                .values()
                .filter_map(|m| f.pds.get(&m.pd).map(|p| p.ctx)),
        );
        if count >= limits.max_mr as usize {
            return Err(verbs(Call::RegMr, ENOMEM));
        }
        // A mock GPU's device address registers as peer memory would: the pages behind
        // it, kept while registered.
        let (host, keep) = match crate::gpu::mock_import_va(addr as usize as u64, len as u64) {
            Some(import) => (import.host, Some(import.keep)),
            None => (addr as usize as u64, None),
        };
        let h = f.handle();
        let key = f.next_key;
        f.next_key += 1;
        let (lkey, rkey) = (0x1000 + key, 0x8000_0000 | key);
        f.mrs.insert(
            h,
            MrRec {
                node: self.node,
                pd,
                addr: addr as usize as u64,
                host,
                len: len as u64,
                lkey,
                rkey,
                access,
                dmabuf: None,
                _keep: keep,
            },
        );
        f.log(self.node, Call::RegMr, h);
        Ok(MrKeys {
            handle: h,
            lkey,
            rkey,
        })
    }

    /// Imports a [`MockGpu`](crate::MockGpu)'s dma-buf: any other descriptor fails, as a
    /// descriptor that isn't a dma-buf does in the kernel.
    fn reg_dmabuf_mr(
        &self,
        pd: Handle,
        offset: u64,
        len: usize,
        iova: u64,
        fd: i32,
        access: Access,
    ) -> Result<MrKeys> {
        let mut f = self.enter(Call::RegMr)?;
        let ctx = f
            .pds
            .get(&pd)
            .ok_or_else(|| verbs(Call::RegMr, EINVAL))?
            .ctx;
        let limits = f.device(ctx).limits;
        if len == 0 || len as u64 > limits.max_mr_size || iova == 0 {
            return Err(verbs(Call::RegMr, EINVAL));
        }
        if access.contains(Access::REMOTE_WRITE) && !access.contains(Access::LOCAL_WRITE) {
            return Err(verbs(Call::RegMr, EINVAL));
        }
        let import = crate::gpu::mock_import_dmabuf(fd, offset, len as u64)
            .ok_or_else(|| verbs(Call::RegMr, EINVAL))?;
        let h = f.handle();
        let key = f.next_key;
        f.next_key += 1;
        let (lkey, rkey) = (0x1000 + key, 0x8000_0000 | key);
        f.mrs.insert(
            h,
            MrRec {
                node: self.node,
                pd,
                addr: iova,
                host: import.host,
                len: len as u64,
                lkey,
                rkey,
                access,
                dmabuf: Some(fd),
                _keep: Some(import.keep),
            },
        );
        f.log(self.node, Call::RegMr, h);
        Ok(MrKeys {
            handle: h,
            lkey,
            rkey,
        })
    }

    fn dereg_mr(&self, mr: Handle) -> Result<()> {
        let mut f = self.enter(Call::DeregMr)?;
        let rec = f
            .mrs
            .remove(&mr)
            .ok_or_else(|| verbs(Call::DeregMr, EINVAL))?;
        let uses_local = |sges: &[RawSge]| sges.iter().any(|s| s.lkey == rec.lkey);
        let mut users = 0;
        for q in f.qps.values().filter(|q| q.node == rec.node) {
            users += q.sq.iter().filter(|j| uses_local(&j.sges)).count();
            users += q.rq.iter().filter(|j| uses_local(&j.sges)).count();
        }
        // Writes from anywhere still headed for it.
        for q in f.qps.values() {
            users +=
                q.sq.iter()
                    .filter(|j| match j.op {
                        SendOp::Write(r) | SendOp::WriteWithImm(r, _) => r.rkey == rec.rkey,
                        _ => false,
                    })
                    .count();
        }
        if users > 0 {
            f.violations.push(format!(
                "memory region #{mr} deregistered while {users} work requests use it"
            ));
        }
        if let Some(fd) = rec.dmabuf {
            crate::gpu::mock_release_dmabuf(fd);
        }
        f.log(self.node, Call::DeregMr, mr);
        Ok(())
    }

    fn post_send(&self, qp: Handle, wr_id: u64, op: &SendOp, sges: &[RawSge]) -> Result<()> {
        let mut f = self.enter(Call::PostSend)?;
        let delay = f.nodes[self.node].send_delay;
        let ready_at = f.tick + delay;
        let q = f
            .qps
            .get_mut(&qp)
            .ok_or_else(|| verbs(Call::PostSend, EINVAL))?;
        if q.state != QpState::Rts || sges.len() > q.caps.max_send_sge as usize {
            return Err(verbs(Call::PostSend, EINVAL));
        }
        if q.sq.len() >= q.caps.max_send_wr as usize {
            return Err(verbs(Call::PostSend, ENOMEM));
        }
        q.sq.push_back(SendJob {
            wr_id,
            op: *op,
            sges: sges.to_vec(),
            ready_at,
            unreachable_since: None,
            rnr_since: None,
        });
        drop(f);
        self.changed();
        Ok(())
    }

    fn post_recv(&self, qp: Handle, wr_id: u64, sges: &[RawSge]) -> Result<()> {
        let mut f = self.enter(Call::PostRecv)?;
        let q = f
            .qps
            .get_mut(&qp)
            .ok_or_else(|| verbs(Call::PostRecv, EINVAL))?;
        if !matches!(q.state, QpState::Init | QpState::Rtr | QpState::Rts)
            || sges.len() > q.caps.max_recv_sge as usize
        {
            return Err(verbs(Call::PostRecv, EINVAL));
        }
        if q.rq.len() >= q.caps.max_recv_wr as usize {
            return Err(verbs(Call::PostRecv, ENOMEM));
        }
        q.rq.push_back(RecvJob {
            wr_id,
            sges: sges.to_vec(),
        });
        drop(f);
        self.changed();
        Ok(())
    }

    fn poll_cq(&self, cq: Handle, out: &mut Vec<RawCompletion>, max: usize) -> Result<usize> {
        let mut f = self.enter(Call::PollCq)?;
        if !f.cqs.contains_key(&cq) {
            return Err(verbs(Call::PollCq, EINVAL));
        }
        f.progress();
        let tick = f.tick;
        let rec = f.cqs.get_mut(&cq).unwrap();
        if rec.overrun {
            return Err(verbs(Call::PollCq, EIO));
        }
        if rec.hide_once && rec.entries.front().is_some_and(|(v, _, _)| *v <= tick) {
            rec.hide_once = false;
            rec.boundary_unpolled = true;
            for (visible, _, seen) in &mut rec.entries {
                if *visible <= tick {
                    *seen = true;
                }
            }
            f.nodes[self.node]
                .log
                .push("MOCK poll/arm boundary completion exposed".into());
            return Ok(0);
        }
        let mut n = 0;
        while n < max {
            match rec.entries.front() {
                Some((visible, _, _)) if *visible <= tick => {
                    out.push(rec.entries.pop_front().unwrap().1);
                    n += 1;
                }
                _ => break,
            }
        }
        if n > 0 {
            rec.boundary_unpolled = false;
        }
        Ok(n)
    }
}
