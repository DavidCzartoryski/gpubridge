//! One bridge member's end of the verbs bridge (docs/VERBS_TRANSPORT.md, "Wire
//! protocol"): an RC queue pair to every other member, tagged messages over them with
//! credits, and a completion thread.
//!
//! **Memory.** Two kinds of caller memory. Plain caller memory ([`Endpoint::send`],
//! [`Endpoint::recv`]) is never registered: a message is copied from it into a send slot
//! and sent; it arrives in a receive slot and is copied into it (or, if no receive is
//! posted for it yet, into a bounded heap buffer). Every such copy happens with the
//! endpoint's lock held, and [`Endpoint::abort`] cancels every operation under that lock:
//! once it returns, nothing reads or writes that memory again. Registered memory
//! ([`Registered`], from [`Endpoint::register_device`] or [`Endpoint::register_host`];
//! [`Endpoint::send_registered`], [`Endpoint::recv_registered`]) is read and written by
//! the device in place: a send gathers its payload from it, and with rendezvous on a
//! receive grants the peer an RDMA WRITE into it, through a window registered for exactly
//! that receive (docs: "Wire protocol"). That memory is the registration's to keep valid;
//! a queue pair that never drains leaks the registrations its work holds, never releases
//! them early.
//!
//! **Failure is cancellation.** An operation fails only by being cancelled under the
//! endpoint's lock ([`State::fail_op`]), and a failed endpoint cancels them all and posts
//! nothing more, before any wait can report it: nothing copies to or from plain caller
//! memory after a failure has been reported.
//!
//! **Rendezvous** (`rndv_bytes` > 0, the same on every member). A message of at least
//! `rndv_bytes` bytes waits until its receiver posts the receive and clears it: with a
//! write grant when the receive is into registered memory (the sender WRITEs the payload
//! there, then SENDs a "written" message, which completes the receive), or "send it in
//! chunks" otherwise. Waiting messages don't hold up others to the same peer, except
//! later ones with the same tag.
//!
//! **Threads.** The caller's thread posts (and copies what fits into free slots). The
//! completion thread blocks on the completion channel (`poll(2)` on its descriptor and
//! the device's asynchronous event descriptor, waking at least every [`TICK`] to notice
//! a stop), polls the completion queue when notified, and does the rest: copies arrived
//! chunks out, reposts receives, returns credits, sends what credits now allow, and
//! wakes waiters. No caller code runs per completion.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use crate::cache::Invalidated;
use crate::direct::{json_str, DirectConfig, Registered, Registrar, Window};
use crate::error::{Error, Result};
use crate::gpu::{GpuMemory, PeerModules};
use crate::protocol::{self, Header, Info, Kind, Selected, HEADER_BYTES, PROTOCOL_VERSION};
use crate::provider::Provider;
use crate::resources::{
    CompletionChannel, CompletionQueue, MemoryRegion, QueuePair, SendRequest, Sge,
};
use crate::types::{
    Access, Address, Completion, ForkStatus, GlobalRoute, InitAttr, LinkLayer, Mtu, QpCaps,
    RemoteTarget, RtrAttr, RtsAttr, SendOp, WcStatus,
};

/// How long the completion thread blocks at most before it checks whether to stop.
pub const TICK: Duration = Duration::from_millis(100);
/// Completions taken per poll.
const POLL_BATCH: usize = 64;
/// The most one RDMA WRITE from registered memory carries (below the 2 GiB message limit
/// of common devices).
pub const WRITE_CHUNK: u64 = 1 << 30;

/// What [`Endpoint::wait`] found when it didn't succeed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitError {
    TimedOut(String),
    Failed(String),
}

/// The endpoint's settings. Every member must use the same `slot_bytes`, `credits` and
/// `ctrl_credits` (the handshake checks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub device: Option<String>,
    pub port: Option<u8>,
    pub gid_index: Option<u32>,
    /// Bytes per message, header included: the size of every slot.
    pub slot_bytes: u32,
    /// Data receives posted for each peer: how many data messages a peer may have in
    /// flight to this member.
    pub credits: u32,
    /// Receives posted for each peer's control messages (credit returns, goodbye).
    pub ctrl_credits: u32,
    /// Send slots per peer: messages this member may have in flight to one peer.
    pub send_slots: u32,
    /// Operations posted and not yet waited for, at most.
    pub max_pending: usize,
    /// Bytes of arrived messages nobody has posted a receive for yet that this member
    /// buffers; beyond that it stops returning the sender's credits until receives are
    /// posted (backpressure).
    pub unexpected_bytes: u64,
    /// The local ACK timeout exponent (4.096 µs * 2^n) and retry count: how long until a
    /// lost peer fails a send.
    pub timeout_exp: u8,
    pub retry_cnt: u8,
    pub service_level: u8,
    pub traffic_class: u8,
    /// Messages of at least this many bytes go by rendezvous (docs: "Wire protocol");
    /// 0: none do, and peers get no write access at all.
    pub rndv_bytes: u64,
    /// Which device paths registrations may use, and the registration caches' bounds.
    pub direct: DirectConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            device: None,
            port: None,
            gid_index: None,
            slot_bytes: 128 * 1024,
            credits: 32,
            ctrl_credits: 4,
            send_slots: 32,
            max_pending: 1 << 16,
            unexpected_bytes: 256 << 20,
            timeout_exp: 14,
            retry_cnt: 7,
            service_level: 0,
            traffic_class: 0,
            rndv_bytes: 0,
            direct: DirectConfig::default(),
        }
    }
}

impl Config {
    /// From `key=value` lines (the C interface's format). Unknown keys are refused.
    pub fn parse(text: &str) -> Result<Config> {
        let mut c = Config::default();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| Error::invalid(format!("setting without '=': {line:?}")))?;
            let bad = || Error::invalid(format!("bad value for {key}: {value:?}"));
            let n = || value.parse::<u64>().map_err(|_| bad());
            let flag = || match value {
                "0" => Ok(false),
                "1" => Ok(true),
                _ => Err(bad()),
            };
            match key {
                "device" => c.device = (!value.is_empty()).then(|| value.to_owned()),
                "port" => c.port = Some(u8::try_from(n()?).map_err(|_| bad())?),
                "gid_index" => c.gid_index = Some(u32::try_from(n()?).map_err(|_| bad())?),
                "slot_bytes" => c.slot_bytes = u32::try_from(n()?).map_err(|_| bad())?,
                "credits" => c.credits = u32::try_from(n()?).map_err(|_| bad())?,
                "ctrl_credits" => c.ctrl_credits = u32::try_from(n()?).map_err(|_| bad())?,
                "send_slots" => c.send_slots = u32::try_from(n()?).map_err(|_| bad())?,
                "max_pending" => c.max_pending = n()? as usize,
                "unexpected_bytes" => c.unexpected_bytes = n()?,
                "timeout_exp" => c.timeout_exp = u8::try_from(n()?).map_err(|_| bad())?,
                "retry_cnt" => c.retry_cnt = u8::try_from(n()?).map_err(|_| bad())?,
                "service_level" => c.service_level = u8::try_from(n()?).map_err(|_| bad())?,
                "traffic_class" => c.traffic_class = u8::try_from(n()?).map_err(|_| bad())?,
                "rndv_bytes" => c.rndv_bytes = n()?,
                "dmabuf" => c.direct.dmabuf = flag()?,
                "peer_memory" => c.direct.peer_memory = flag()?,
                "device_cache_entries" => c.direct.device_cache.max_entries = n()? as usize,
                "device_cache_bytes" => c.direct.device_cache.max_bytes = n()? as usize,
                "host_cache_entries" => c.direct.host_cache.max_entries = n()? as usize,
                "host_cache_bytes" => c.direct.host_cache.max_bytes = n()? as usize,
                // The C interface's own keys (which provider), handled there.
                "provider" | "fabric_dir" | "gpu" => {}
                _ => return Err(Error::invalid(format!("unknown setting {key:?}"))),
            }
        }
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<()> {
        let min = HEADER_BYTES as u32 + 64;
        if !(min..=16 << 20).contains(&self.slot_bytes) || self.slot_bytes % 64 != 0 {
            return Err(Error::invalid(format!(
                "slot_bytes {} must be a multiple of 64 from {min} to 16 MiB",
                self.slot_bytes
            )));
        }
        for (name, v, max) in [
            ("credits", self.credits, 4096),
            ("ctrl_credits", self.ctrl_credits, 64),
            ("send_slots", self.send_slots, 4096),
        ] {
            if !(1..=max).contains(&v) {
                return Err(Error::invalid(format!("{name} {v} outside 1 to {max}")));
            }
        }
        if self.max_pending == 0 {
            return Err(Error::invalid("max_pending must be positive"));
        }
        for (name, cache) in [
            ("device cache", self.direct.device_cache),
            ("host cache", self.direct.host_cache),
        ] {
            if cache.max_entries == 0 || cache.max_bytes == 0 {
                return Err(Error::invalid(format!(
                    "the {name} needs room for at least one entry and byte"
                )));
            }
        }
        if self.timeout_exp > 31 || self.retry_cnt > 7 || self.service_level > 15 {
            return Err(Error::invalid(
                "timeout_exp is at most 31, retry_cnt 7, service_level 15",
            ));
        }
        Ok(())
    }

    fn payload(&self) -> u64 {
        u64::from(self.slot_bytes) - HEADER_BYTES as u64
    }
}

/// Counters for telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub sends: u64,
    pub recvs: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub chunks_sent: u64,
    pub chunks_received: u64,
    pub control_sent: u64,
    pub control_received: u64,
    /// Times a send waited because the peer had no free receive (no credit).
    pub credit_stalls: u64,
    pub unexpected_bytes: u64,
    pub unexpected_peak: u64,
    /// Arrived chunks held in their slots because the unexpected buffer was full.
    pub held_chunks: u64,
    pub completions: u64,
    pub completion_errors: u64,
    pub notifications: u64,
    pub async_events: u64,
    pub batches_posted: u64,
    /// Clear-to-send messages sent: write grants and "send in chunks".
    pub grants_sent: u64,
    pub go_sent: u64,
    /// Clear-to-send messages received.
    pub grants_received: u64,
    pub go_received: u64,
    /// RDMA WRITEs posted and their bytes; "written" messages sent and received.
    pub writes_posted: u64,
    pub bytes_written: u64,
    pub fins_sent: u64,
    pub fins_received: u64,
    /// Bytes the device read from or wrote to registered memory in place (sends gathered
    /// from it, writes from it, receives written into it), by kind.
    pub host_registered_bytes: u64,
    pub device_bytes: u64,
    /// Bytes copied by the CPU between caller memory and the slots.
    pub copied_bytes: u64,
    /// Write windows (one per granted receive) deregistered once written, and kept until
    /// their queue pair closes because their receive failed.
    pub exposures_closed: u64,
    pub exposures_retired: u64,
}

/// Where an arriving message goes.
enum Dest {
    /// A posted receive (an operation id).
    Op(i64),
    /// A heap buffer, until a receive is posted.
    Buffer(Vec<u8>),
    /// An operation that was cancelled: arriving bytes are dropped.
    Discard,
}

struct Incoming {
    msg_len: u64,
    received: u64,
    dest: Dest,
}

#[derive(Clone, Copy)]
enum SlotUse {
    Free,
    /// A chunk, an RDMA WRITE or a "written" message of a send operation.
    Data {
        op: Option<i64>,
    },
    /// A credit or goodbye message.
    Control,
    /// A clear-to-send message.
    Grant,
}

/// What a receiver cleared one message to be sent as.
#[derive(Clone, Copy)]
struct Grant {
    len: u64,
    /// RDMA WRITE to this address with this key; None: in chunks.
    write: Option<(u64, u32)>,
}

struct Peer {
    member: usize,
    qp: QueuePair,
    connected: bool,
    path_mtu: Option<Mtu>,
    /// Set once the queue pair failed or the peer left: why. Nothing more is posted.
    lost: Option<String>,
    /// The peer said goodbye: it sends nothing more.
    said_bye: bool,
    send_slots: Vec<MemoryRegion>,
    slot_use: Vec<SlotUse>,
    free_send: Vec<u32>,
    recv_slots: Vec<MemoryRegion>,
    // Credits (docs: "Credits").
    data_sent: u32,
    data_returned: u32,
    ctrl_sent: u32,
    ctrl_returned: u32,
    data_freed: u32,
    ctrl_freed: u32,
    data_reported: u32,
    send_seq: u32,
    recv_seq: u32,
    bye_pending: bool,
    bye_sent: bool,
    stalled: bool,
    /// Send operations not started, in order.
    sends: VecDeque<i64>,
    /// The send operation being posted (chunks, or writes and then its "written").
    current: Option<i64>,
    /// Clear-to-send messages received and not yet taken by a send, by tag, in order.
    grants: HashMap<u64, VecDeque<Grant>>,
    /// Receive operations whose clear-to-send message is still to go out, in order.
    grants_out: VecDeque<i64>,
    /// Receives granted a write, waiting for the "written" message, by tag.
    writing: HashMap<u64, i64>,
    /// Receives posted and not yet matched to an arriving message, by tag.
    posted: HashMap<u64, i64>,
    incoming: HashMap<u64, Incoming>,
    /// Whole messages nobody posted a receive for yet, by tag, oldest first.
    complete: HashMap<u64, VecDeque<Vec<u8>>>,
    /// Arrived data chunks still in their receive slots (the unexpected buffer was full),
    /// in arrival order; while any is held, later ones queue behind it.
    held: VecDeque<(u32, Header)>,
    reposts: Vec<u32>,
    /// Write windows of receives that failed (each holds the registration it lies in):
    /// the peer may have been granted one and still be writing until this queue pair is
    /// closed, so they are released only after that (or never, if it didn't drain).
    retired: Vec<Window>,
}

/// How a send goes out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Waiting for the receiver's clear-to-send (rendezvous).
    Waiting,
    /// In chunks through the peer's receive slots.
    Chunks,
    /// RDMA WRITEs to the granted address, then a "written" message.
    Write { addr: u64, key: u32 },
}

enum OpKind {
    Send {
        mode: Mode,
        /// Payload bytes posted so far.
        next: u64,
        /// Work requests posted, and completed.
        posted: u64,
        acked: u64,
        /// Work requests in all (chunks; or writes plus the "written" message).
        chunks: u64,
    },
    Recv,
}

/// An operation's memory.
enum Mem {
    /// Caller memory, copied through the slots by the CPU.
    Plain(usize),
    /// A registration the device reads or writes in place, from `offset`.
    Reg { reg: Registered, offset: usize },
}

impl Mem {
    /// Where the CPU may copy to or from, if it may.
    fn cpu_addr(&self) -> Option<usize> {
        match self {
            Mem::Plain(p) => Some(*p),
            Mem::Reg { reg, offset } if reg.cpu() => Some(reg.addr() + offset),
            Mem::Reg { .. } => None,
        }
    }
}

struct Op {
    peer: usize,
    tag: u64,
    mem: Mem,
    len: u64,
    kind: OpKind,
    done: bool,
    /// Set once the operation failed, which cancels it ([`State::fail_op`]).
    error: Option<String>,
    /// A receive the peer will RDMA WRITE into: the registration of exactly its bytes
    /// with remote write, made for this one receive ([`Registrar::expose`]). Its key is
    /// the one the write grant carries; it is deregistered once the "written" message
    /// arrives, or kept until the queue pair is closed if the receive fails.
    exposure: Option<Window>,
}

struct State {
    /// Set once the endpoint failed: every operation is cancelled with it
    /// ([`Shared::fail_endpoint`]).
    failed: Option<String>,
    closing: bool,
    peers: Vec<Option<Peer>>,
    ops: HashMap<i64, Op>,
    next_id: i64,
    unexpected: u64,
    counters: Counters,
    /// Write windows to release once the lock is let go (deregistering can be slow).
    release: Vec<Window>,
}

/// What the handshake and telemetry report about this end.
struct Local {
    info: Info,
    device: String,
    port: u8,
    link_layer: LinkLayer,
    gid_index: Option<u32>,
    gid_type: Option<crate::types::GidType>,
    fw_ver: String,
    vendor: (u32, u32),
    provider: String,
    mock: bool,
    /// Whether registered memory survives a fork here, and what was done about it.
    fork: String,
}

/// Makes sure memory this process registers stays what the device reads and writes
/// across a `fork()` (DataLoader workers, subprocesses): on a kernel that doesn't keep
/// registered pages with the parent, `ibv_fork_init` before anything is registered, or
/// fail with the reason. Returns what telemetry says about it.
fn fork_safety(provider: &dyn Provider) -> Result<String> {
    let status = provider.fork_status()?;
    let why = match status {
        ForkStatus::Unneeded => {
            return Ok("unneeded: the kernel keeps registered pages with the parent on fork".into())
        }
        ForkStatus::Enabled => {
            return Ok("enabled: ibv_fork_init already ran (or RDMAV_FORK_SAFE is set)".into())
        }
        ForkStatus::Disabled => "the kernel doesn't keep registered pages with the parent on fork",
        ForkStatus::Unknown => "this libibverbs can't say whether the kernel is fork-safe",
    };
    match provider.fork_init() {
        Ok(()) => Ok(format!(
            "enabled by gpubridge (ibv_fork_init): {why}, so registered pages are kept out of \
             forked children"
        )),
        Err(e) => Err(Error::Unsupported(format!(
            "{why}, and ibv_fork_init failed ({e}): a fork() (a DataLoader worker, a \
             subprocess) would leave the device reading and writing pages the parent no \
             longer uses. ibv_fork_init must run before any memory is registered in the \
             process: set RDMAV_FORK_SAFE=1 before starting it, or open the verbs bridge \
             before anything else registers memory"
        ))),
    }
}

struct Shared {
    config: Config,
    index: usize,
    members: usize,
    local: Local,
    /// Registrations of caller memory (device and host), on the endpoint's protection
    /// domain. Dropped after the queue pairs are closed.
    registrar: Registrar,
    /// Whether work requests may carry two scatter-gather entries (a header from a slot
    /// and a payload from registered memory).
    gather: bool,
    cq: CompletionQueue,
    channel: CompletionChannel,
    state: Mutex<State>,
    changed: Condvar,
    stop: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A completion's tag: which peer, which slot, send or receive.
fn slot_tag(member: usize, recv: bool, slot: u32) -> u64 {
    ((member as u64) << 33) | (u64::from(recv) << 32) | u64::from(slot)
}

fn untag(tag: u64) -> (usize, bool, u32) {
    ((tag >> 33) as usize, (tag >> 32) & 1 == 1, tag as u32)
}

fn random_psn(salt: u64) -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(salt);
    h.write_u128(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    (h.finish() as u32) & 0xFF_FFFF
}

fn status_text(status: WcStatus) -> &'static str {
    match status {
        WcStatus::RetryExcErr => {
            ": the peer didn't acknowledge within the retry budget (it exited, or the path to \
             it is down)"
        }
        WcStatus::RnrRetryExcErr => ": the peer had no receive posted (a protocol error)",
        WcStatus::RemAccessErr | WcStatus::RemInvReqErr | WcStatus::RemOpErr => {
            ": the peer refused the request"
        }
        WcStatus::WrFlushErr => ": flushed (the queue pair had already failed)",
        _ => "",
    }
}

/// A bridge member's end. See the module docs.
pub struct Endpoint {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Endpoint {
    /// Opens the device `config` selects and makes everything for `members` members, of
    /// which this is `index`: a queue pair per other member, in `Init` with its receives
    /// posted, ready for [`connect`](Endpoint::connect). Fails with the reason if verbs
    /// can't be used here; nothing falls back to another transport.
    pub fn new(
        provider: Arc<dyn Provider>,
        members: usize,
        index: usize,
        config: Config,
    ) -> Result<Endpoint> {
        config.check()?;
        if members < 2 || index >= members || members > 4096 {
            return Err(Error::invalid(format!(
                "member {index} of {members}: need 2 to 4096 members"
            )));
        }
        // Before anything is registered (the slots, below).
        let fork = fork_safety(provider.as_ref())?;
        let Selected {
            ctx,
            port,
            attr,
            gid_index,
            gid,
            gid_type,
        } = protocol::select(
            &provider,
            config.device.as_deref(),
            config.port,
            config.gid_index,
        )?;
        let dev = ctx.attr().clone();
        let peers = (members - 1) as u32;
        let per_peer = config.send_slots + config.credits + config.ctrl_credits;
        let depth = peers
            .checked_mul(per_peer)
            .filter(|d| *d <= dev.max_cqe)
            .ok_or_else(|| {
                Error::Exhausted(format!(
                    "{peers} peers with {per_peer} work requests each need a completion queue \
                     of {} entries; {} allows {}: lower the credits or send slots",
                    u64::from(peers) * u64::from(per_peer),
                    ctx.name(),
                    dev.max_cqe
                ))
            })?;
        let pd = ctx.alloc_pd()?;
        let channel = ctx.create_channel()?;
        let cq = ctx.create_cq_on(depth, &channel)?;
        let gather = dev.max_sge >= 2;
        let caps = QpCaps {
            max_send_wr: config.send_slots,
            max_recv_wr: config.credits + config.ctrl_credits,
            max_send_sge: if gather { 2 } else { 1 },
            max_recv_sge: 1,
            max_inline_data: 0,
        };
        // Peers may write only with rendezvous on, and then only through the window of
        // one granted receive (the queue pair's access is necessary, not sufficient: the
        // region's key and bounds decide). No other registration is remotely writable:
        // not the slots, not the cached registrations of caller memory.
        let qp_access = if config.rndv_bytes > 0 {
            Access::LOCAL_WRITE | Access::REMOTE_WRITE
        } else {
            Access::NONE
        };
        let mut slots = Vec::new();
        let mut qpns = vec![0; members];
        let mut psns = vec![0; members];
        for member in (0..members).filter(|m| *m != index) {
            let qp = pd.create_qp(&cq, &cq, caps)?;
            qp.to_init(&InitAttr {
                port,
                pkey_index: 0,
                access: qp_access,
            })?;
            let size = config.slot_bytes as usize;
            let send_slots = (0..config.send_slots)
                .map(|_| pd.register_buffer(size, Access::NONE))
                .collect::<Result<Vec<_>>>()?;
            let recv_slots = (0..config.credits + config.ctrl_credits)
                .map(|_| pd.register_buffer(size, Access::LOCAL_WRITE))
                .collect::<Result<Vec<_>>>()?;
            let sges: Vec<[Sge<'_>; 1]> = recv_slots
                .iter()
                .map(|r| [Sge::new(r, 0, config.slot_bytes)])
                .collect();
            let reqs: Vec<(u64, &[Sge<'_>])> = sges
                .iter()
                .enumerate()
                .map(|(i, s)| (slot_tag(member, true, i as u32), &s[..]))
                .collect();
            qp.post_recvs(&reqs)?;
            qpns[member] = qp.qp_num();
            psns[member] = random_psn(member as u64 ^ (u64::from(qp.qp_num()) << 32));
            slots.push(Peer {
                member,
                qp,
                connected: false,
                path_mtu: None,
                lost: None,
                said_bye: false,
                slot_use: vec![SlotUse::Free; send_slots.len()],
                free_send: (0..config.send_slots).rev().collect(),
                send_slots,
                recv_slots,
                data_sent: 0,
                data_returned: 0,
                ctrl_sent: 0,
                ctrl_returned: 0,
                data_freed: 0,
                ctrl_freed: 0,
                data_reported: 0,
                send_seq: 0,
                recv_seq: 0,
                bye_pending: false,
                bye_sent: false,
                stalled: false,
                sends: VecDeque::new(),
                current: None,
                grants: HashMap::new(),
                grants_out: VecDeque::new(),
                writing: HashMap::new(),
                posted: HashMap::new(),
                incoming: HashMap::new(),
                complete: HashMap::new(),
                held: VecDeque::new(),
                reposts: Vec::new(),
                retired: Vec::new(),
            });
        }
        let grh = attr.link_layer == LinkLayer::Ethernet || gid_index.is_some();
        let info = Info {
            version: PROTOCOL_VERSION,
            link_layer: attr.link_layer,
            lid: attr.lid,
            gid,
            grh,
            active_mtu: attr.active_mtu,
            slot_bytes: config.slot_bytes,
            credits: config.credits,
            ctrl_credits: config.ctrl_credits,
            rndv: config.rndv_bytes,
            qpns,
            psns,
            device: ctx.name().to_owned(),
            port,
        };
        let mut peers_by_member: Vec<Option<Peer>> = (0..members).map(|_| None).collect();
        for p in slots {
            let m = p.member;
            peers_by_member[m] = Some(p);
        }
        let registrar = Registrar::new(&pd, config.direct)?;
        drop(pd);
        let shared = Arc::new(Shared {
            local: Local {
                info,
                device: ctx.name().to_owned(),
                port,
                link_layer: attr.link_layer,
                gid_index,
                gid_type,
                fw_ver: dev.fw_ver.clone(),
                vendor: (dev.vendor_id, dev.vendor_part_id),
                provider: provider.describe(),
                mock: provider.is_mock(),
                fork,
            },
            config,
            index,
            members,
            registrar,
            gather,
            cq,
            channel,
            state: Mutex::new(State {
                failed: None,
                closing: false,
                peers: peers_by_member,
                ops: HashMap::new(),
                next_id: 1,
                unexpected: 0,
                counters: Counters::default(),
                release: Vec::new(),
            }),
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let worker = shared.clone();
        let thread = thread::Builder::new()
            .name("gpubridge-verbs".into())
            .spawn(move || worker.run())
            .map_err(|e| Error::Exhausted(format!("starting the completion thread: {e}")))?;
        Ok(Endpoint {
            shared,
            thread: Some(thread),
        })
    }

    /// This member's handshake line, for every other member.
    pub fn local_info(&self) -> String {
        self.shared.local.info.encode()
    }

    /// Connects to `member` from the handshake line it published: checks the two ends
    /// agree (protocol version, link layer, settings) and moves the queue pair to `Rtr`
    /// and `Rts`.
    pub fn connect(&self, member: usize, info: &str) -> Result<()> {
        let theirs = Info::decode(info)
            .map_err(|e| Error::invalid(format!("member {member}'s handshake: {e}")))?;
        let s = &self.shared;
        let mine = &s.local.info;
        let path_mtu = mine.compatible(&theirs, member, s.index)?;
        let mut st = lock(&s.state);
        let peer = st
            .peers
            .get_mut(member)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::invalid(format!("no member {member} to connect to")))?;
        if peer.connected {
            return Err(Error::invalid(format!(
                "member {member} is already connected"
            )));
        }
        let route = GlobalRoute {
            dgid: theirs.gid,
            sgid_index: s.local.gid_index.unwrap_or(0) as u8,
            hop_limit: 64,
            traffic_class: s.config.traffic_class,
            flow_label: 0,
        };
        let address = match s.local.link_layer {
            LinkLayer::Ethernet => Address::Roce(route),
            LinkLayer::InfiniBand => Address::Ib {
                lid: theirs.lid,
                grh: s.local.gid_index.map(|_| route),
            },
        };
        if s.local.gid_index.is_some_and(|i| i > u32::from(u8::MAX)) {
            return Err(Error::invalid("GID index above 255"));
        }
        peer.qp.to_rtr(&RtrAttr {
            dest_qp_num: theirs.qpns[s.index],
            rq_psn: theirs.psns[s.index],
            path_mtu,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 12,
            sl: s.config.service_level,
            address,
        })?;
        peer.qp.to_rts(&RtsAttr {
            sq_psn: mine.psns[member],
            timeout: s.config.timeout_exp,
            retry_cnt: s.config.retry_cnt,
            // Credits mean a receive is always posted for what arrives: running out of
            // RNR retries is a protocol error, reported rather than waited on forever.
            rnr_retry: 6,
            max_rd_atomic: 0,
        })?;
        peer.connected = true;
        peer.path_mtu = Some(path_mtu);
        Ok(())
    }

    fn post(&self, member: usize, mem: Mem, len: u64, tag: u64, send: bool) -> Result<i64> {
        let s = &self.shared;
        let rndv = s.config.rndv_bytes;
        if let Mem::Reg { reg, offset } = &mem {
            if offset
                .checked_add(len as usize)
                .map_or(true, |e| e > reg.len())
            {
                return Err(Error::invalid(format!(
                    "{len} bytes at {offset} outside a registration of {}",
                    reg.len()
                )));
            }
            if !reg.cpu() && !send && (rndv == 0 || len < rndv) {
                return Err(Error::invalid(format!(
                    "a receive of {len} bytes into device memory: only messages of at least \
                     rndv_bytes ({rndv}; 0 is off) are written into registered memory, smaller \
                     ones arrive in host slots"
                )));
            }
            if !s.gather && len > 0 && !(rndv > 0 && len >= rndv) {
                return Err(Error::Unsupported(format!(
                    "{} allows one scatter-gather entry per request, so registered memory can't \
                     be sent in chunks",
                    s.local.device
                )));
            }
        }
        // A receive the rendezvous will have the peer write: exactly its bytes, registered
        // with remote write for this receive alone (before the lock: registering is slow).
        let exposure = match &mem {
            Mem::Reg { reg, offset } if !send && rndv > 0 && len >= rndv => {
                Some(s.registrar.expose(reg, *offset, len as usize)?)
            }
            _ => None,
        };
        let mut st = lock(&s.state);
        if let Some(why) = &st.failed {
            return Err(Error::Busy(format!(
                "the verbs bridge already failed ({why})"
            )));
        }
        if st.closing {
            return Err(Error::Busy("the verbs bridge is closing".into()));
        }
        if st.ops.len() >= s.config.max_pending {
            return Err(Error::QueueFull {
                queue: "operation",
                limit: s.config.max_pending as u32,
            });
        }
        if len > 0 && matches!(mem, Mem::Plain(0)) {
            return Err(Error::invalid("a null buffer"));
        }
        let peer = st
            .peers
            .get(member)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::invalid(format!("no member {member}")))?;
        if !peer.connected {
            return Err(Error::invalid(format!("member {member} isn't connected")));
        }
        // A message that arrived before the peer left can still be received.
        let buffered = !send && peer.complete.contains_key(&tag);
        if let (Some(why), false) = (&peer.lost, buffered) {
            return Err(Error::Busy(format!("member {member}: {why}")));
        }
        if !send && (peer.posted.contains_key(&tag) || peer.writing.contains_key(&tag)) {
            return Err(Error::invalid(format!(
                "a receive from member {member} with tag {tag:#x} is already posted"
            )));
        }
        if send && peer.said_bye {
            return Err(Error::Busy(format!(
                "member {member} closed its end of the bridge"
            )));
        }
        let id = st.next_id;
        st.next_id += 1;
        let payload = s.config.payload();
        let kind = if send {
            let waiting = rndv > 0 && len >= rndv;
            OpKind::Send {
                mode: if waiting { Mode::Waiting } else { Mode::Chunks },
                next: 0,
                posted: 0,
                acked: 0,
                chunks: len.div_ceil(payload).max(1),
            }
        } else {
            OpKind::Recv
        };
        st.ops.insert(
            id,
            Op {
                peer: member,
                tag,
                mem,
                len,
                kind,
                done: false,
                error: None,
                exposure,
            },
        );
        if send {
            st.counters.sends += 1;
            st.peer(member).sends.push_back(id);
        } else {
            st.counters.recvs += 1;
            s.match_recv(&mut st, member, id);
        }
        s.pump(&mut st, member);
        s.changed.notify_all();
        Ok(id)
    }

    /// Sends `len` bytes at `ptr` to `member` with `tag`; returns an operation id for
    /// [`wait`](Endpoint::wait). Messages to one member with one tag arrive in the order
    /// sent.
    ///
    /// # Safety
    ///
    /// `len` bytes at `ptr` must stay valid and unchanged until `wait` reports the
    /// operation done, it fails, or [`abort`](Endpoint::abort) or `close` returns. A
    /// failure cancels the operation (and every other one, the endpoint having failed)
    /// under the lock every copy takes, so once `wait` has reported it nothing reads the
    /// memory again.
    pub unsafe fn send(&self, member: usize, ptr: usize, len: u64, tag: u64) -> Result<i64> {
        self.post(member, Mem::Plain(ptr), len, tag, true)
    }

    /// Receives the message `member` sends with `tag` into `len` bytes at `ptr` (it must
    /// be exactly `len` bytes long).
    ///
    /// # Safety
    ///
    /// `len` bytes at `ptr` must stay valid, and not be used otherwise, until `wait`
    /// reports the operation done, it fails, or `abort` or `close` returns. As for
    /// [`send`](Endpoint::send), a failure is a cancellation: nothing writes the memory
    /// after `wait` has reported it.
    pub unsafe fn recv(&self, member: usize, ptr: usize, len: u64, tag: u64) -> Result<i64> {
        self.post(member, Mem::Plain(ptr), len, tag, false)
    }

    /// Sends `len` bytes at `offset` of the registration `reg` to `member` with `tag`:
    /// the device reads them in place (gathered after a header from a slot, or by RDMA
    /// WRITE). The memory must not change until the operation is done.
    pub fn send_registered(
        &self,
        member: usize,
        reg: &Registered,
        offset: usize,
        len: u64,
        tag: u64,
    ) -> Result<i64> {
        let mem = Mem::Reg {
            reg: reg.clone(),
            offset,
        };
        self.post(member, mem, len, tag, true)
    }

    /// Receives the message `member` sends with `tag` into `len` bytes at `offset` of
    /// `reg`. With rendezvous on and a message of at least `rndv_bytes`, the peer writes
    /// it there by RDMA WRITE, through a window registered now for exactly those bytes
    /// (deregistered once written); a smaller one (host memory only) is copied out of a
    /// slot.
    /// For device memory, completion means the write has completed with respect to this
    /// endpoint (its "written" message, sent after the writes on the same RC queue pair,
    /// was polled); making it visible to GPU work is the caller's flush (docs: "GPU
    /// memory").
    pub fn recv_registered(
        &self,
        member: usize,
        reg: &Registered,
        offset: usize,
        len: u64,
        tag: u64,
    ) -> Result<i64> {
        let mem = Mem::Reg {
            reg: reg.clone(),
            offset,
        };
        self.post(member, mem, len, tag, false)
    }

    /// Uses `driver` for device memory, and `modules` to look for the peer-memory module.
    pub fn set_gpu(&self, driver: Arc<dyn GpuMemory>, modules: PeerModules) {
        self.shared.registrar.set_gpu(driver, modules);
    }

    /// Loads the GPU driver for `vendor` (`cuda` or `rocm`) at run time; its description,
    /// or why it can't be used (remembered for the reasons registrations give).
    pub fn load_gpu(&self, vendor: &str) -> Result<String> {
        match crate::gpu::load(vendor) {
            Ok(driver) => {
                let text = driver.describe();
                self.set_gpu(driver, PeerModules::system());
                Ok(text)
            }
            Err(e) => {
                self.shared.registrar.set_gpu_error(e.to_string());
                Err(e)
            }
        }
    }

    /// What each device path can do on `device`, as JSON (see
    /// [`Registrar::capabilities`]).
    pub fn gpu_capabilities(&self, device: i32) -> String {
        self.shared.registrar.capabilities(device)
    }

    /// A direct registration of `len` bytes of device memory at `ptr` on `device` (a
    /// dma-buf or peer memory, whichever passes its checks first; docs: "GPU memory"), or
    /// every reason why not, for the caller to stage instead and say so.
    pub fn register_device(
        &self,
        device: i32,
        ptr: u64,
        len: u64,
    ) -> std::result::Result<Registered, String> {
        self.shared.registrar.register_device(device, ptr, len)
    }

    /// Registers `len` bytes of host memory at `ptr` that the caller identifies as `owner`.
    ///
    /// # Safety
    ///
    /// As [`Registrar::register_host`]: valid until `owner` is invalidated and every
    /// handle to the registration (and work posted with it) is gone.
    pub unsafe fn register_host(&self, ptr: usize, len: usize, owner: u64) -> Result<Registered> {
        // SAFETY: the caller's contract, passed on.
        unsafe { self.shared.registrar.register_host(ptr, len, owner) }
    }

    /// Drops the host registrations of `owner` (it is about to be freed).
    pub fn invalidate_host(&self, owner: u64) -> Invalidated {
        self.shared.registrar.invalidate_host(owner)
    }

    /// The setting messages go by rendezvous from (0: never).
    pub fn rndv_bytes(&self) -> u64 {
        self.shared.config.rndv_bytes
    }

    /// Waits until every operation in `ids` is done (and forgets them), one fails, the
    /// endpoint fails, or `timeout` passes.
    pub fn wait(&self, ids: &[i64], timeout: Duration) -> std::result::Result<(), WaitError> {
        let s = &self.shared;
        let start = Instant::now();
        let deadline = start + timeout;
        let mut st = lock(&s.state);
        loop {
            let mut unfinished = 0;
            for id in ids {
                match st.ops.get(id) {
                    None => {
                        return Err(WaitError::Failed(format!(
                            "unknown operation {id} (already waited for?)"
                        )))
                    }
                    Some(op) => {
                        if let Some(e) = op.error.clone() {
                            // One failed operation fails the endpoint, which cancels the
                            // rest.
                            s.fail_endpoint(&mut st, &e);
                            return Err(WaitError::Failed(e));
                        }
                        if !op.done {
                            unfinished += 1;
                        }
                    }
                }
            }
            if unfinished == 0 {
                for id in ids {
                    st.ops.remove(id);
                }
                return Ok(());
            }
            if let Some(why) = &st.failed {
                return Err(WaitError::Failed(why.clone()));
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(WaitError::TimedOut(format!(
                    "timed out: {unfinished} of {} operations unfinished after {:.1} s",
                    ids.len(),
                    timeout.as_secs_f64()
                )));
            }
            st = s
                .changed
                .wait_timeout(st, deadline - now)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    /// Fails the endpoint now: every operation is cancelled (so nothing reads or writes
    /// caller memory after this returns) and every queue pair moves to the error state,
    /// which flushes its work and makes the peers' sends to it fail.
    pub fn abort(&self, why: &str) {
        let s = &self.shared;
        let mut st = lock(&s.state);
        s.fail_endpoint(&mut st, why);
        for peer in st.peers.iter_mut().flatten() {
            let _ = peer.qp.to_error();
            peer.lost.get_or_insert_with(|| why.to_owned());
        }
        s.changed.notify_all();
    }

    /// Says goodbye to every peer (so they don't take this member's leaving as a
    /// failure), cancels what is left, flushes every queue pair and waits up to `timeout`
    /// for the flushes, stops the completion thread and releases everything. Work that
    /// didn't drain in time keeps its slots registered and allocated (leaked, never
    /// freed): that is the error returned.
    pub fn close(mut self, timeout: Duration) -> Result<()> {
        self.shutdown(timeout)
    }

    fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        let s = &self.shared;
        let deadline = Instant::now() + timeout;
        {
            let mut st = lock(&s.state);
            st.closing = true;
            if st.failed.is_none() {
                let members: Vec<usize> = st.peers.iter().flatten().map(|p| p.member).collect();
                for m in members {
                    let p = st.peer(m);
                    if p.connected && p.lost.is_none() {
                        p.bye_pending = true;
                    }
                    s.pump(&mut st, m);
                }
                // The goodbyes are sent once their send completes (the peer has it).
                let bye_deadline = deadline.min(Instant::now() + Duration::from_secs(1));
                while st
                    .peers
                    .iter()
                    .flatten()
                    .any(|p| p.bye_pending && p.lost.is_none())
                    && Instant::now() < bye_deadline
                {
                    st = s
                        .changed
                        .wait_timeout(st, bye_deadline - Instant::now())
                        .unwrap_or_else(|p| p.into_inner())
                        .0;
                }
            }
            s.fail_endpoint(&mut st, "closed");
            for peer in st.peers.iter_mut().flatten() {
                let _ = peer.qp.to_error();
                peer.lost.get_or_insert_with(|| "closed".into());
            }
            // The completion thread takes the flushed completions.
            while s.cq.outstanding() > 0 && Instant::now() < deadline {
                st = s
                    .changed
                    .wait_timeout(st, (deadline - Instant::now()).min(TICK))
                    .unwrap_or_else(|p| p.into_inner())
                    .0;
            }
        }
        s.stop.store(true, Ordering::Release);
        let _ = thread.join();
        let peers: Vec<Peer> = {
            let mut st = lock(&s.state);
            st.peers.iter_mut().filter_map(Option::take).collect()
        };
        let mut result = Ok(());
        for peer in peers {
            // Whatever didn't drain is leaked with its slots (QueuePair::close).
            let left = deadline.saturating_duration_since(Instant::now());
            match peer.qp.close(left) {
                // Closed: the peer can't write through a window it was granted any more.
                Ok(()) => drop(peer.retired),
                Err(e) => {
                    // It may still be able to: keep those windows (and the memory they
                    // lie in) registered for good.
                    std::mem::forget(peer.retired);
                    if result.is_ok() {
                        result = Err(e);
                    }
                }
            }
        }
        result
    }

    /// Telemetry as one JSON object (docs/VERBS_TRANSPORT.md, "Telemetry"). Remote keys
    /// only ever cross the fabric in write grants; none is kept where this reads, so none
    /// can appear here.
    pub fn telemetry(&self) -> String {
        let s = &self.shared;
        let l = &s.local;
        let direct = s.registrar.telemetry();
        let st = lock(&s.state);
        let c = &st.counters;
        let mut o = String::from("{");
        let q = |v: &str| json_str(v);
        let _ = write!(
            o,
            "\"protocol\":{PROTOCOL_VERSION},\"provider\":{},\"mock\":{},\"library\":{},\
             \"device\":{},\"port\":{},\"link_layer\":{},\"gid_index\":{},\"gid_type\":{},\
             \"active_mtu\":{},\"firmware\":{},\"vendor_id\":{},\"vendor_part_id\":{},\
             \"operations\":{},\"rndv_bytes\":{},\"gather\":{},\"slot_bytes\":{},\
             \"credits\":{},\"ctrl_credits\":{},\"send_slots\":{},\"members\":{},\"index\":{},\
             \"fork_safety\":{},\"direct\":{direct},",
            q(&l.provider),
            l.mock,
            q(&format!("gpubridge-verbs {}", env!("CARGO_PKG_VERSION"))),
            q(&l.device),
            l.port,
            q(protocol::link_text(l.link_layer)),
            l.gid_index.map_or("null".into(), |i| i.to_string()),
            l.gid_type.map_or("null".into(), |t| q(&format!("{t:?}"))),
            l.info.active_mtu.bytes(),
            q(&l.fw_ver),
            l.vendor.0,
            l.vendor.1,
            q(if s.config.rndv_bytes > 0 {
                "RC SEND/RECV, RDMA WRITE (rendezvous)"
            } else {
                "RC SEND/RECV"
            }),
            s.config.rndv_bytes,
            s.gather,
            s.config.slot_bytes,
            s.config.credits,
            s.config.ctrl_credits,
            s.config.send_slots,
            s.members,
            s.index,
            q(&l.fork),
        );
        let state = match (&st.failed, st.closing) {
            (_, true) => "closed",
            (Some(_), false) => "failed",
            (None, false) => "ok",
        };
        let _ = write!(
            o,
            "\"state\":{},\"failure\":{},",
            q(state),
            st.failed.as_deref().map_or("null".into(), q)
        );
        let _ = write!(
            o,
            "\"sends\":{},\"recvs\":{},\"bytes_sent\":{},\"bytes_received\":{},\
             \"chunks_sent\":{},\"chunks_received\":{},\"control_sent\":{},\
             \"control_received\":{},\"credit_stalls\":{},\"unexpected_bytes\":{},\
             \"unexpected_peak\":{},\"held_chunks\":{},\"completions\":{},\
             \"completion_errors\":{},\"notifications\":{},\"async_events\":{},\
             \"batches_posted\":{},\"grants_sent\":{},\"go_sent\":{},\"grants_received\":{},\
             \"go_received\":{},\"writes_posted\":{},\"bytes_written\":{},\"fins_sent\":{},\
             \"fins_received\":{},\"host_registered_bytes\":{},\"device_bytes\":{},\
             \"copied_bytes\":{},\"write_windows_closed\":{},\"write_windows_retired\":{},\
             \"pending_operations\":{},\"peers\":[",
            c.sends,
            c.recvs,
            c.bytes_sent,
            c.bytes_received,
            c.chunks_sent,
            c.chunks_received,
            c.control_sent,
            c.control_received,
            c.credit_stalls,
            st.unexpected,
            c.unexpected_peak,
            c.held_chunks,
            c.completions,
            c.completion_errors,
            c.notifications,
            c.async_events,
            c.batches_posted,
            c.grants_sent,
            c.go_sent,
            c.grants_received,
            c.go_received,
            c.writes_posted,
            c.bytes_written,
            c.fins_sent,
            c.fins_received,
            c.host_registered_bytes,
            c.device_bytes,
            c.copied_bytes,
            c.exposures_closed,
            c.exposures_retired,
            st.ops.len()
        );
        let mut first = true;
        for p in st.peers.iter().flatten() {
            if !first {
                o.push(',');
            }
            first = false;
            let _ = write!(
                o,
                "{{\"member\":{},\"qpn\":{},\"connected\":{},\"path_mtu\":{},\"closed_by_peer\":{},\
                 \"lost\":{}}}",
                p.member,
                p.qp.qp_num(),
                p.connected,
                p.path_mtu.map_or("null".into(), |m| m.bytes().to_string()),
                p.said_bye,
                p.lost.as_deref().map_or("null".into(), q)
            );
        }
        o.push_str("]}");
        o
    }

    /// The counters, for tests.
    pub fn counters(&self) -> Counters {
        let st = lock(&self.shared.state);
        Counters {
            unexpected_bytes: st.unexpected,
            ..st.counters
        }
    }

    /// Why the endpoint failed, if it did.
    pub fn failure(&self) -> Option<String> {
        lock(&self.shared.state).failed.clone()
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        let _ = self.shutdown(Duration::from_secs(10));
    }
}

impl Peer {
    /// Drops every operation queued, posted, granted or in progress with this peer (they
    /// stay in the operation table, failed or cancelled, holding their registrations
    /// until waited for or the endpoint goes: a peer granted a write may still be writing
    /// until its queue pair is in the error state).
    fn forget_work(&mut self) {
        self.sends.clear();
        self.current = None;
        self.posted.clear();
        self.writing.clear();
        self.grants.clear();
        self.grants_out.clear();
    }
}

impl State {
    fn peer(&mut self, member: usize) -> &mut Peer {
        self.peers[member].as_mut().expect("a peer")
    }

    /// Fails an operation, which cancels it: from here on nothing copies to or from its
    /// memory (it leaves every queue, an arriving message for it is dropped, its posted
    /// chunks no longer count for it), all under the endpoint lock that every copy takes.
    fn fail_op(&mut self, id: i64, why: &str) {
        let Some(op) = self.ops.get_mut(&id) else {
            return;
        };
        if op.done || op.error.is_some() {
            return;
        }
        op.error = Some(why.to_owned());
        let member = op.peer;
        let Some(peer) = self.peers.get_mut(member).and_then(Option::as_mut) else {
            return;
        };
        let exposure = op.exposure.take();
        peer.sends.retain(|s| *s != id);
        if peer.current == Some(id) {
            peer.current = None;
        }
        peer.posted.retain(|_, r| *r != id);
        peer.writing.retain(|_, r| *r != id);
        peer.grants_out.retain(|r| *r != id);
        for inc in peer.incoming.values_mut() {
            if matches!(inc.dest, Dest::Op(o) if o == id) {
                inc.dest = Dest::Discard;
            }
        }
        for u in peer.slot_use.iter_mut() {
            if matches!(u, SlotUse::Data { op: Some(o) } if *o == id) {
                *u = SlotUse::Data { op: None };
            }
        }
        // A granted write window stays registered until the peer can't write any more
        // (its queue pair is closed): see `Peer::retired`.
        if let Some(window) = exposure {
            peer.retired.push(window);
            self.counters.exposures_retired += 1;
        }
    }
}

impl Shared {
    // ---- the completion thread ------------------------------------------------------

    fn run(&self) {
        while !self.stop.load(Ordering::Acquire) {
            if self.poll_and_handle() > 0 {
                continue;
            }
            // Arm, then poll once more: a completion between the empty poll and the arm
            // would otherwise wait for the next one.
            if let Err(e) = self.cq.req_notify() {
                self.fail_all(&format!("asking for completion notifications failed: {e}"));
                thread::sleep(TICK);
                continue;
            }
            if self.poll_and_handle() > 0 {
                continue;
            }
            match self.channel.wait(TICK) {
                Ok(events) => {
                    if events.completion {
                        lock(&self.state).counters.notifications += 1;
                    }
                    for e in events.async_events {
                        self.async_event(e);
                    }
                }
                Err(e) => {
                    self.fail_all(&format!("waiting for completions failed: {e}"));
                    thread::sleep(TICK);
                }
            }
        }
    }

    fn async_event(&self, e: crate::types::AsyncEvent) {
        let mut st = lock(&self.state);
        st.counters.async_events += 1;
        if !e.kind.fatal() {
            return;
        }
        let what = match (e.qp_num, e.port) {
            (Some(q), _) => format!(" on queue pair {q:#x}"),
            (_, Some(p)) => format!(" on port {p}"),
            _ => String::new(),
        };
        let why = format!("{} reported {:?}{what}", self.local.device, e.kind);
        match e.qp_num {
            Some(q) => {
                let member = st
                    .peers
                    .iter()
                    .flatten()
                    .find(|p| p.qp.qp_num() == q)
                    .map(|p| p.member);
                if let Some(m) = member {
                    self.lose(&mut st, m, &why);
                }
            }
            None => self.fail_endpoint(&mut st, &why),
        }
        self.changed.notify_all();
    }

    fn fail_all(&self, why: &str) {
        let mut st = lock(&self.state);
        self.fail_endpoint(&mut st, why);
        self.changed.notify_all();
    }

    /// The endpoint failed: every operation is cancelled, and nothing is posted or
    /// copied from here on ([`State::fail_op`], [`pump`](Shared::pump)).
    fn fail_endpoint(&self, st: &mut State, why: &str) {
        st.failed.get_or_insert_with(|| why.to_owned());
        self.cancel_all(st);
    }

    /// Polls once and handles what came; returns how many.
    fn poll_and_handle(&self) -> usize {
        let completions = match self.cq.poll(POLL_BATCH) {
            Ok(c) => c,
            Err(e) => {
                self.fail_all(&format!("polling the completion queue failed: {e}"));
                return 0;
            }
        };
        if completions.is_empty() {
            return 0;
        }
        let n = completions.len();
        let mut st = lock(&self.state);
        st.counters.completions += n as u64;
        let mut touched = Vec::new();
        for c in &completions {
            let (member, _, _) = untag(c.tag);
            self.completion(&mut st, c);
            if !touched.contains(&member) {
                touched.push(member);
            }
        }
        for m in touched {
            self.repost(&mut st, m);
            self.pump(&mut st, m);
        }
        let release = std::mem::take(&mut st.release);
        drop(st);
        self.changed.notify_all();
        drop(release);
        n
    }

    fn completion(&self, st: &mut State, c: &Completion) {
        let (member, recv, slot) = untag(c.tag);
        if st.peers.get(member).and_then(Option::as_ref).is_none() {
            return;
        }
        if !c.ok() {
            st.counters.completion_errors += 1;
            let peer = st.peer(member);
            if recv {
                // A flushed receive: its slot isn't reposted (the queue pair is done).
            } else if let SlotUse::Data { op: Some(id) } = peer.slot_use[slot as usize] {
                peer.slot_use[slot as usize] = SlotUse::Free;
                peer.free_send.push(slot);
                let why = format!(
                    "member {member}: a send completed with {:?} (vendor error {:#x}){}",
                    c.status,
                    c.vendor_err,
                    status_text(c.status)
                );
                st.fail_op(id, &why);
            } else {
                peer.slot_use[slot as usize] = SlotUse::Free;
                peer.free_send.push(slot);
            }
            let peer = st.peer(member);
            if peer.lost.is_none() {
                let why = if peer.said_bye {
                    format!("member {member} closed its end of the bridge")
                } else {
                    format!(
                        "member {member}: {} completed with {:?} (vendor error {:#x}){}",
                        if recv { "a receive" } else { "a send" },
                        c.status,
                        c.vendor_err,
                        status_text(c.status)
                    )
                };
                self.lose(st, member, &why);
            }
            return;
        }
        if recv {
            self.arrived(st, member, slot, c.byte_len);
        } else {
            let peer = st.peer(member);
            let used = std::mem::replace(&mut peer.slot_use[slot as usize], SlotUse::Free);
            peer.free_send.push(slot);
            match used {
                SlotUse::Data { op: Some(id) } => {
                    if let Some(op) = st.ops.get_mut(&id) {
                        if let OpKind::Send { acked, chunks, .. } = &mut op.kind {
                            *acked += 1;
                            if *acked == *chunks && op.error.is_none() {
                                op.done = true;
                            }
                        }
                    }
                }
                SlotUse::Control => {
                    let peer = st.peer(member);
                    if peer.bye_sent {
                        peer.bye_pending = false;
                    }
                }
                _ => {}
            }
        }
    }

    /// A message arrived in receive slot `slot` from `member`.
    fn arrived(&self, st: &mut State, member: usize, slot: u32, byte_len: u32) {
        let mut raw = [0u8; HEADER_BYTES];
        let header = {
            let peer = st.peer(member);
            if (byte_len as usize) < HEADER_BYTES {
                Err(format!(
                    "a message of {byte_len} bytes, shorter than a header"
                ))
            } else {
                peer.recv_slots[slot as usize]
                    .read(0, &mut raw)
                    .map_err(|e| e.to_string())
                    .and_then(|()| Header::decode(&raw, byte_len))
            }
        };
        let header = match header.and_then(|h| self.check_header(st, member, h)) {
            Ok(h) => h,
            Err(e) => {
                let why = format!("member {member} broke the protocol: {e}");
                self.protocol_error(st, member, &why);
                return;
            }
        };
        match header.kind {
            Kind::Data => {
                st.counters.chunks_received += 1;
                let peer = st.peer(member);
                if !peer.held.is_empty() {
                    peer.held.push_back((slot, header));
                    st.counters.held_chunks += 1;
                    return;
                }
                if !self.place(st, member, slot, &header) {
                    st.peer(member).held.push_back((slot, header));
                    st.counters.held_chunks += 1;
                    return;
                }
                let peer = st.peer(member);
                peer.data_freed = peer.data_freed.wrapping_add(1);
                peer.reposts.push(slot);
            }
            Kind::Cts | Kind::Go => {
                if header.kind == Kind::Cts {
                    st.counters.grants_received += 1;
                } else {
                    st.counters.go_received += 1;
                }
                let peer = st.peer(member);
                peer.data_freed = peer.data_freed.wrapping_add(1);
                peer.reposts.push(slot);
                let write = (header.kind == Kind::Cts).then_some((header.offset, header.key));
                peer.grants.entry(header.tag).or_default().push_back(Grant {
                    len: header.msg_len,
                    write,
                });
            }
            Kind::Fin => {
                st.counters.fins_received += 1;
                let peer = st.peer(member);
                peer.data_freed = peer.data_freed.wrapping_add(1);
                peer.reposts.push(slot);
                let Some(id) = peer.writing.remove(&header.tag) else {
                    let why = format!(
                        "member {member} broke the protocol: a \"written\" message for tag {:#x}, \
                         which this member granted no write for",
                        header.tag
                    );
                    self.protocol_error(st, member, &why);
                    return;
                };
                let Some(op) = st.ops.get_mut(&id) else {
                    return;
                };
                if op.len != header.msg_len {
                    let why = format!(
                        "member {member} wrote {} bytes for tag {:#x}, granted {}",
                        header.msg_len, header.tag, op.len
                    );
                    self.protocol_error(st, member, &why);
                    return;
                }
                op.done = op.error.is_none();
                // The writes came before this message on the same RC queue pair, so they
                // are complete: the peer can't write through the window any more, and it
                // goes (deregistered once the lock is let go).
                if let Some(window) = op.exposure.take() {
                    st.release.push(window);
                    st.counters.exposures_closed += 1;
                }
                let Some(op) = st.ops.get(&id) else {
                    return;
                };
                let device = matches!(&op.mem, Mem::Reg { reg, .. } if !reg.cpu());
                let len = op.len;
                st.counters.bytes_received += len;
                if device {
                    st.counters.device_bytes += len;
                } else {
                    st.counters.host_registered_bytes += len;
                }
            }
            Kind::Credit | Kind::Bye => {
                st.counters.control_received += 1;
                let peer = st.peer(member);
                peer.ctrl_freed = peer.ctrl_freed.wrapping_add(1);
                peer.reposts.push(slot);
                if header.kind == Kind::Bye {
                    peer.said_bye = true;
                    // Nothing more comes from it: receives it can't still meet fail.
                    let waiting: Vec<i64> = peer
                        .posted
                        .drain()
                        .chain(peer.writing.drain())
                        .map(|(_, id)| id)
                        .collect();
                    // And it takes nothing more: sends not yet (wholly) posted to it fail
                    // now rather than at their deadline. Ones already posted complete as
                    // the peer acknowledges them.
                    let unsent: Vec<i64> =
                        peer.current.iter().chain(&peer.sends).copied().collect();
                    let why =
                        format!("member {member} closed its end of the bridge before sending this");
                    for id in waiting {
                        st.fail_op(id, &why);
                    }
                    let why = format!("member {member} closed its end of the bridge");
                    for id in unsent {
                        st.fail_op(id, &why);
                    }
                }
            }
        }
    }

    /// Sequence and credit checks; applies the credit counters.
    fn check_header(
        &self,
        st: &mut State,
        member: usize,
        h: Header,
    ) -> std::result::Result<Header, String> {
        let cfg = &self.config;
        let peer = st.peer(member);
        if h.seq != peer.recv_seq {
            return Err(format!(
                "message {} arrived where {} was next",
                h.seq, peer.recv_seq
            ));
        }
        if h.kind == Kind::Data && u64::from(h.chunk_len) > cfg.payload() {
            return Err(format!("a chunk of {} bytes", h.chunk_len));
        }
        // Cumulative counts only grow, and never past what was sent.
        let data_back = h.data_freed.wrapping_sub(peer.data_returned);
        let ctrl_back = h.ctrl_freed.wrapping_sub(peer.ctrl_returned);
        if data_back > peer.data_sent.wrapping_sub(peer.data_returned)
            || ctrl_back > peer.ctrl_sent.wrapping_sub(peer.ctrl_returned)
        {
            return Err("returned more credits than it had".into());
        }
        peer.recv_seq = peer.recv_seq.wrapping_add(1);
        peer.data_returned = h.data_freed;
        peer.ctrl_returned = h.ctrl_freed;
        Ok(h)
    }

    fn protocol_error(&self, st: &mut State, member: usize, why: &str) {
        st.failed.get_or_insert_with(|| why.to_owned());
        self.lose(st, member, why);
        self.fail_endpoint(st, why);
    }

    /// The link to `member` is gone: fail what involves it and stop using it.
    fn lose(&self, st: &mut State, member: usize, why: &str) {
        let ids: Vec<i64> = st
            .ops
            .iter()
            .filter(|(_, op)| op.peer == member && !op.done)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            st.fail_op(id, why);
        }
        let peer = st.peer(member);
        peer.lost.get_or_insert_with(|| why.to_owned());
        peer.forget_work();
        let _ = peer.qp.to_error();
    }

    /// Cancels every operation: none of them reads or writes caller memory after this.
    fn cancel_all(&self, st: &mut State) {
        let why = st.failed.clone().unwrap_or_else(|| "cancelled".into());
        let ids: Vec<i64> = st.ops.keys().copied().collect();
        for id in ids {
            st.fail_op(id, &why);
        }
        for peer in st.peers.iter_mut().flatten() {
            peer.forget_work();
            for inc in peer.incoming.values_mut() {
                if matches!(inc.dest, Dest::Op(_)) {
                    inc.dest = Dest::Discard;
                }
            }
            for u in peer.slot_use.iter_mut() {
                if let SlotUse::Data { op } = u {
                    *op = None;
                }
            }
        }
    }

    /// Puts a data chunk where it goes; false if it has to stay in its slot.
    fn place(&self, st: &mut State, member: usize, slot: u32, h: &Header) -> bool {
        let limit = self.config.unexpected_bytes;
        let peer = st.peer(member);
        if !peer.incoming.contains_key(&h.tag) {
            if h.offset != 0 {
                let why = format!(
                    "member {member} broke the protocol: a chunk at {} of a message not started",
                    h.offset
                );
                self.protocol_error(st, member, &why);
                return true;
            }
            let dest = match peer.posted.remove(&h.tag) {
                Some(id) => {
                    let len = st.ops[&id].len;
                    if len != h.msg_len {
                        let why = format!(
                            "member {member} sent {} bytes for tag {:#x}, posted for {len} bytes",
                            h.msg_len, h.tag
                        );
                        self.protocol_error(st, member, &why);
                        return true;
                    }
                    Dest::Op(id)
                }
                None => {
                    if h.msg_len > 0 && st.unexpected + h.msg_len > limit {
                        return false;
                    }
                    st.unexpected += h.msg_len;
                    st.counters.unexpected_peak = st.counters.unexpected_peak.max(st.unexpected);
                    Dest::Buffer(vec![0u8; h.msg_len as usize])
                }
            };
            st.peer(member).incoming.insert(
                h.tag,
                Incoming {
                    msg_len: h.msg_len,
                    received: 0,
                    dest,
                },
            );
        }
        let peer = st.peers[member].as_mut().expect("a peer");
        let inc = peer.incoming.get_mut(&h.tag).expect("just made");
        if h.msg_len != inc.msg_len || h.offset != inc.received {
            let why = format!(
                "member {member} broke the protocol: chunk at {} of {} bytes where {} of {} \
                 was next",
                h.offset, h.msg_len, inc.received, inc.msg_len
            );
            self.protocol_error(st, member, &why);
            return true;
        }
        // A receive that failed (or an endpoint that did) was cancelled: State::fail_op
        // and cancel_all turn its destination into Discard; this only makes sure.
        if let Dest::Op(id) = inc.dest {
            if st.failed.is_some() || st.ops.get(&id).map_or(true, |op| op.error.is_some()) {
                inc.dest = Dest::Discard;
            }
        }
        let n = h.chunk_len as usize;
        let src = &peer.recv_slots[slot as usize];
        let mut registered = false;
        let mut kept = true;
        let copied = match &mut inc.dest {
            Dest::Op(id) => match st.ops.get(id) {
                Some(op) if n > 0 => match op.mem.cpu_addr() {
                    Some(addr) => {
                        registered = matches!(op.mem, Mem::Reg { .. });
                        // SAFETY: the op's memory holds op.len bytes (recv's contract, or
                        // the registration's, which the op holds) and offset + n <=
                        // msg_len == op.len (checked by Header::decode and above). The
                        // op has no error and the endpoint hasn't failed (checked just
                        // above, with the endpoint lock held, as it still is): a failure
                        // cancels the op under this lock before anyone is told of it, so
                        // the caller still keeps the memory valid.
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((addr + h.offset as usize) as *mut u8, n)
                        };
                        src.read(HEADER_BYTES, out)
                    }
                    // Device memory is only ever written by RDMA WRITE (post() refuses
                    // receives into it that could arrive in chunks).
                    None => Err(Error::invalid("a chunk for device memory")),
                },
                _ => Ok(()),
            },
            Dest::Buffer(buf) => src.read(
                HEADER_BYTES,
                &mut buf[h.offset as usize..h.offset as usize + n],
            ),
            Dest::Discard => {
                kept = false;
                Ok(())
            }
        };
        if let Err(e) = copied {
            let why = format!("copying a chunk from member {member} failed: {e}");
            self.protocol_error(st, member, &why);
            return true;
        }
        inc.received += u64::from(h.chunk_len);
        st.counters.bytes_received += u64::from(h.chunk_len);
        if registered {
            st.counters.host_registered_bytes += u64::from(h.chunk_len);
        } else if kept {
            st.counters.copied_bytes += u64::from(h.chunk_len);
        }
        let peer = st.peers[member].as_mut().expect("a peer");
        let inc = &peer.incoming[&h.tag];
        if inc.received == inc.msg_len {
            let inc = peer.incoming.remove(&h.tag).expect("present");
            match inc.dest {
                Dest::Op(id) => {
                    if let Some(op) = st.ops.get_mut(&id) {
                        op.done = op.error.is_none();
                    }
                }
                Dest::Buffer(buf) => peer.complete.entry(h.tag).or_default().push_back(buf),
                Dest::Discard => {}
            }
        }
        true
    }

    /// A receive operation was posted: meet it from what already arrived if possible.
    fn match_recv(&self, st: &mut State, member: usize, id: i64) {
        let (tag, len, cpu) = {
            let op = &st.ops[&id];
            (op.tag, op.len, op.mem.cpu_addr())
        };
        let peer = st.peer(member);
        let whole = peer.complete.get_mut(&tag).and_then(VecDeque::pop_front);
        if peer.complete.get(&tag).is_some_and(VecDeque::is_empty) {
            peer.complete.remove(&tag);
        }
        if let Some(buf) = whole {
            st.unexpected -= buf.len() as u64;
            if buf.len() as u64 != len {
                let why = format!(
                    "member {member} sent {} bytes for tag {tag:#x}, posted for {len} bytes",
                    buf.len()
                );
                self.protocol_error(st, member, &why);
                return;
            }
            if len > 0 {
                let Some(ptr) = cpu else {
                    let why = format!(
                        "member {member} sent tag {tag:#x} in chunks where device memory waits"
                    );
                    self.protocol_error(st, member, &why);
                    return;
                };
                // SAFETY: the op's memory holds len bytes (recv's contract, or the
                // registration's), still valid: it was just posted and the endpoint lock
                // is held.
                unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr as *mut u8, buf.len()) };
                st.counters.copied_bytes += len;
            }
            st.ops.get_mut(&id).expect("posted").done = true;
        } else if let Some(inc) = peer.incoming.get_mut(&tag) {
            if let Dest::Buffer(buf) = &inc.dest {
                if inc.msg_len != len {
                    let why = format!(
                        "member {member} sent {} bytes for tag {tag:#x}, posted for {len} bytes",
                        inc.msg_len
                    );
                    self.protocol_error(st, member, &why);
                    return;
                }
                let Some(ptr) = cpu else {
                    let why = format!(
                        "member {member} sent tag {tag:#x} in chunks where device memory waits"
                    );
                    self.protocol_error(st, member, &why);
                    return;
                };
                let have = inc.received as usize;
                if have > 0 {
                    // SAFETY: as above; have <= msg_len == len.
                    unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr as *mut u8, have) };
                }
                let freed = buf.len() as u64;
                inc.dest = Dest::Op(id);
                st.unexpected -= freed;
                st.counters.copied_bytes += have as u64;
            } else {
                // Another receive is already taking this message: this one waits for the
                // next with the same tag.
                self.defer_recv(st, member, id);
            }
        } else if peer.said_bye {
            let why = format!("member {member} closed its end of the bridge before sending this");
            st.fail_op(id, &why);
        } else {
            self.defer_recv(st, member, id);
        }
        self.release_held(st, member);
        self.repost(st, member);
    }

    /// A receive nothing has arrived for yet: posted for chunks, or, for a message the
    /// rendezvous covers, cleared to be sent (a write grant into registered memory, or
    /// "in chunks" into plain memory).
    fn defer_recv(&self, st: &mut State, member: usize, id: i64) {
        let rndv = self.config.rndv_bytes;
        let (tag, len, registered) = {
            let op = &st.ops[&id];
            (op.tag, op.len, matches!(op.mem, Mem::Reg { .. }))
        };
        let peer = st.peer(member);
        if rndv > 0 && len >= rndv {
            if registered {
                peer.writing.insert(tag, id);
            } else {
                peer.posted.insert(tag, id);
            }
            peer.grants_out.push_back(id);
        } else {
            peer.posted.insert(tag, id);
        }
    }

    /// Places held chunks, oldest first, while they fit.
    fn release_held(&self, st: &mut State, member: usize) {
        while let Some(&(slot, h)) = st.peer(member).held.front() {
            if !self.place(st, member, slot, &h) {
                break;
            }
            let peer = st.peer(member);
            peer.held.pop_front();
            peer.data_freed = peer.data_freed.wrapping_add(1);
            peer.reposts.push(slot);
        }
    }

    fn repost(&self, st: &mut State, member: usize) {
        let peer = st.peer(member);
        if peer.reposts.is_empty() || peer.lost.is_some() {
            peer.reposts.clear();
            return;
        }
        let slots = std::mem::take(&mut peer.reposts);
        let len = self.config.slot_bytes;
        let sges: Vec<[Sge<'_>; 1]> = slots
            .iter()
            .map(|&i| [Sge::new(&peer.recv_slots[i as usize], 0, len)])
            .collect();
        let reqs: Vec<(u64, &[Sge<'_>])> = slots
            .iter()
            .zip(&sges)
            .map(|(&i, s)| (slot_tag(member, true, i), &s[..]))
            .collect();
        let result = peer.qp.post_recvs(&reqs);
        drop(sges);
        st.counters.batches_posted += 1;
        if let Err(e) = result {
            let why = format!("member {member}: reposting receives failed: {e}");
            self.lose(st, member, &why);
        }
    }

    /// Takes the next send to `member` that may start: the first queued one that isn't
    /// waiting for a clear-to-send, and that no earlier queued send has the same tag as
    /// (messages with one tag go in order). Binds a waiting one to its clear-to-send.
    fn next_send(&self, st: &mut State, member: usize) -> std::result::Result<Option<i64>, String> {
        let State { peers, ops, .. } = st;
        let peer = peers[member].as_mut().expect("a peer");
        let mut seen = std::collections::HashSet::new();
        let mut pick = None;
        peer.sends
            .retain(|id| ops.get(id).is_some_and(|op| op.error.is_none()));
        for (i, id) in peer.sends.iter().enumerate() {
            let op = &ops[id];
            if !seen.insert(op.tag) {
                continue;
            }
            let waiting = matches!(
                op.kind,
                OpKind::Send {
                    mode: Mode::Waiting,
                    ..
                }
            );
            if !waiting || peer.grants.get(&op.tag).is_some_and(|g| !g.is_empty()) {
                pick = Some(i);
                break;
            }
        }
        let Some(i) = pick else {
            return Ok(None);
        };
        let id = peer.sends.remove(i).expect("picked");
        let op = ops.get_mut(&id).expect("queued");
        let OpKind::Send { mode, chunks, .. } = &mut op.kind else {
            unreachable!("only sends are queued")
        };
        if *mode == Mode::Waiting {
            let queue = peer.grants.get_mut(&op.tag).expect("granted");
            let grant = queue.pop_front().expect("granted");
            if queue.is_empty() {
                peer.grants.remove(&op.tag);
            }
            if grant.len != op.len {
                return Err(format!(
                    "member {member} posted {} bytes for tag {:#x}; this member sends {}",
                    grant.len, op.tag, op.len
                ));
            }
            match grant.write {
                Some((addr, key)) => {
                    let piece = match op.mem {
                        Mem::Plain(_) => u64::from(self.config.slot_bytes),
                        Mem::Reg { .. } => WRITE_CHUNK,
                    };
                    *mode = Mode::Write { addr, key };
                    *chunks = op.len.div_ceil(piece) + 1;
                }
                None => *mode = Mode::Chunks,
            }
        }
        Ok(Some(id))
    }

    /// Sends what slots and credits allow to `member`: control messages first, then
    /// clear-to-send messages, then the current send's next chunks (or writes, then its
    /// "written" message). One batch post.
    fn pump(&self, st: &mut State, member: usize) {
        let cfg = &self.config;
        let payload = cfg.payload();
        let threshold = (cfg.credits / 2).max(1);
        let mut batch: Vec<Wr> = Vec::new();
        let mut bytes = 0;
        loop {
            // A failed endpoint posts nothing more: its operations are cancelled.
            if st.failed.is_some() {
                break;
            }
            let peer = st.peers[member].as_mut().expect("a peer");
            // A peer that said goodbye wants nothing more, credits included.
            if !peer.connected || peer.lost.is_some() || peer.bye_sent || peer.said_bye {
                break;
            }
            if peer.current.is_none() {
                match self.next_send(st, member) {
                    Ok(next) => st.peer(member).current = next,
                    Err(e) => {
                        let why = format!("member {member} broke the protocol: {e}");
                        self.protocol_error(st, member, &why);
                        return;
                    }
                }
            }
            // Failing an operation takes it out of `current` (State::fail_op); this only
            // makes sure.
            if let Some(id) = st.peer(member).current {
                if st.ops.get(&id).map_or(true, |op| op.error.is_some()) {
                    st.peer(member).current = None;
                    continue;
                }
            }
            let peer = st.peers[member].as_mut().expect("a peer");
            let data_avail = cfg
                .credits
                .wrapping_sub(peer.data_sent.wrapping_sub(peer.data_returned));
            let ctrl_avail = cfg
                .ctrl_credits
                .wrapping_sub(peer.ctrl_sent.wrapping_sub(peer.ctrl_returned));
            let owed = peer.data_freed.wrapping_sub(peer.data_reported);
            // The next data-class work request, and whether it takes a data credit (an
            // RDMA WRITE takes none: it consumes no receive).
            let next_needs_credit = if !peer.grants_out.is_empty() {
                Some(true)
            } else {
                peer.current.map(|id| match &st.ops[&id].kind {
                    OpKind::Send {
                        mode: Mode::Write { .. },
                        next,
                        ..
                    } => *next >= st.ops[&id].len,
                    _ => true,
                })
            };
            let has_data = next_needs_credit.is_some();
            let need_credit = owed >= threshold && (!has_data || data_avail == 0);
            let need_bye = peer.bye_pending && !has_data && peer.sends.is_empty();
            if (need_credit || need_bye) && ctrl_avail > 0 {
                let Some(slot) = peer.free_send.pop() else {
                    break;
                };
                let kind = if need_bye { Kind::Bye } else { Kind::Credit };
                let h = Header {
                    kind,
                    seq: peer.send_seq,
                    chunk_len: 0,
                    data_freed: peer.data_freed,
                    ctrl_freed: peer.ctrl_freed,
                    key: 0,
                    tag: 0,
                    msg_len: 0,
                    offset: 0,
                };
                if peer.send_slots[slot as usize]
                    .write(0, &h.encode())
                    .is_err()
                {
                    peer.free_send.push(slot);
                    break;
                }
                peer.send_seq = peer.send_seq.wrapping_add(1);
                peer.data_reported = peer.data_freed;
                peer.ctrl_sent = peer.ctrl_sent.wrapping_add(1);
                peer.slot_use[slot as usize] = SlotUse::Control;
                if kind == Kind::Bye {
                    peer.bye_sent = true;
                }
                batch.push(Wr::Send {
                    slot,
                    len: HEADER_BYTES as u32,
                });
                st.counters.control_sent += 1;
                if kind == Kind::Bye {
                    break;
                }
                continue;
            }
            let Some(needs_credit) = next_needs_credit else {
                break;
            };
            if needs_credit && data_avail == 0 {
                if !peer.stalled {
                    peer.stalled = true;
                    st.counters.credit_stalls += 1;
                }
                break;
            }
            let Some(slot) = peer.free_send.pop() else {
                break;
            };
            if needs_credit {
                peer.stalled = false;
            }
            // A clear-to-send for the oldest receive that needs one.
            if let Some(rid) = peer.grants_out.pop_front() {
                let grant = match st.ops.get(&rid) {
                    Some(op) if !op.done && op.error.is_none() => match (&op.mem, &op.exposure) {
                        // The key of the receive's own window: exactly its bytes, nothing
                        // around them, and only until its "written" message.
                        (Mem::Reg { .. }, Some(window)) => (
                            Kind::Cts,
                            window.region().rkey(),
                            window.region().addr() as u64,
                            op.tag,
                            op.len,
                        ),
                        (Mem::Reg { .. }, None) => {
                            unreachable!("post() exposes every registered rendezvous receive")
                        }
                        (Mem::Plain(_), _) => (Kind::Go, 0, 0, op.tag, op.len),
                    },
                    // Cancelled meanwhile.
                    _ => {
                        st.peer(member).free_send.push(slot);
                        continue;
                    }
                };
                let (kind, key, addr, tag, len) = grant;
                let peer = st.peers[member].as_mut().expect("a peer");
                let h = Header {
                    kind,
                    seq: peer.send_seq,
                    chunk_len: 0,
                    data_freed: peer.data_freed,
                    ctrl_freed: peer.ctrl_freed,
                    key,
                    tag,
                    msg_len: len,
                    offset: addr,
                };
                if let Err(e) = peer.send_slots[slot as usize].write(0, &h.encode()) {
                    peer.free_send.push(slot);
                    let why = format!("member {member}: filling a send slot failed: {e}");
                    self.lose(st, member, &why);
                    return;
                }
                peer.send_seq = peer.send_seq.wrapping_add(1);
                peer.data_reported = peer.data_freed;
                peer.data_sent = peer.data_sent.wrapping_add(1);
                peer.slot_use[slot as usize] = SlotUse::Grant;
                batch.push(Wr::Send {
                    slot,
                    len: HEADER_BYTES as u32,
                });
                if kind == Kind::Cts {
                    st.counters.grants_sent += 1;
                } else {
                    st.counters.go_sent += 1;
                }
                continue;
            }
            let id = peer.current.expect("data to send");
            let (seq, data_freed, ctrl_freed) = (peer.send_seq, peer.data_freed, peer.ctrl_freed);
            let slot_region = peer.send_slots[slot as usize].clone();
            let op = st.ops.get_mut(&id).expect("the current send");
            let OpKind::Send {
                mode,
                next,
                posted,
                chunks,
                ..
            } = &mut op.kind
            else {
                unreachable!("only sends are current")
            };
            let header = |kind: Kind, chunk_len: u32, offset: u64| Header {
                kind,
                seq,
                chunk_len,
                data_freed,
                ctrl_freed,
                key: 0,
                tag: op.tag,
                msg_len: op.len,
                offset,
            };
            // What this request is: (the request, payload bytes, takes a receive, kind
            // of memory the device reads in place: None for a copy).
            let made: Result<(Wr, u64, bool, Option<bool>)> = match *mode {
                Mode::Waiting => unreachable!("a send is current only once cleared"),
                Mode::Chunks => {
                    let n = (op.len - *next).min(payload);
                    let h = header(Kind::Data, n as u32, *next);
                    slot_region
                        .write(0, &h.encode())
                        .and_then(|()| match &op.mem {
                            Mem::Plain(ptr) => {
                                if n > 0 {
                                    // SAFETY: the op's buffer holds op.len bytes and next + n
                                    // <= op.len. The op is current and has no error (checked
                                    // above, under the endpoint lock, which is held): it
                                    // isn't done, and it hasn't failed or been cancelled,
                                    // which State::fail_op does under this lock before any
                                    // failure is reported. So the caller still keeps the
                                    // memory valid (send's contract).
                                    let data = unsafe {
                                        std::slice::from_raw_parts(
                                            (ptr + *next as usize) as *const u8,
                                            n as usize,
                                        )
                                    };
                                    slot_region.write(HEADER_BYTES, data)?;
                                }
                                let len = HEADER_BYTES as u32 + n as u32;
                                Ok((Wr::Send { slot, len }, n, true, None))
                            }
                            Mem::Reg { .. } if n == 0 => Ok((
                                Wr::Send {
                                    slot,
                                    len: HEADER_BYTES as u32,
                                },
                                0,
                                true,
                                None,
                            )),
                            Mem::Reg { reg, offset } => Ok((
                                Wr::Gather {
                                    slot,
                                    region: reg.region().clone(),
                                    off: offset + *next as usize,
                                    n: n as u32,
                                },
                                n,
                                true,
                                Some(reg.cpu()),
                            )),
                        })
                }
                Mode::Write { addr, key } if *next < op.len => {
                    let remote = RemoteTarget {
                        addr: addr + *next,
                        rkey: key,
                    };
                    match &op.mem {
                        Mem::Plain(ptr) => {
                            let n = (op.len - *next).min(u64::from(self.config.slot_bytes));
                            // SAFETY: as for a chunk above (current, no error, lock
                            // held).
                            let data = unsafe {
                                std::slice::from_raw_parts(
                                    (ptr + *next as usize) as *const u8,
                                    n as usize,
                                )
                            };
                            slot_region.write(0, data).map(|()| {
                                let wr = Wr::Write {
                                    slot,
                                    local: None,
                                    off: 0,
                                    n: n as u32,
                                    remote,
                                };
                                (wr, n, false, None)
                            })
                        }
                        Mem::Reg { reg, offset } => {
                            let n = (op.len - *next).min(WRITE_CHUNK);
                            let wr = Wr::Write {
                                slot,
                                local: Some(reg.region().clone()),
                                off: offset + *next as usize,
                                n: n as u32,
                                remote,
                            };
                            Ok((wr, n, false, Some(reg.cpu())))
                        }
                    }
                }
                Mode::Write { .. } => {
                    let h = header(Kind::Fin, 0, 0);
                    slot_region.write(0, &h.encode()).map(|()| {
                        let wr = Wr::Send {
                            slot,
                            len: HEADER_BYTES as u32,
                        };
                        (wr, 0, true, None)
                    })
                }
            };
            let (wr, n, takes_receive, in_place) = match made {
                Ok(m) => m,
                Err(e) => {
                    st.peer(member).free_send.push(slot);
                    let why = format!("member {member}: filling a send slot failed: {e}");
                    self.lose(st, member, &why);
                    return;
                }
            };
            let is_fin = matches!(*mode, Mode::Write { .. }) && takes_receive;
            let is_write = matches!(wr, Wr::Write { .. });
            *next += n;
            *posted += 1;
            let finished = *posted == *chunks;
            bytes += n;
            let c = &mut st.counters;
            match in_place {
                Some(true) => c.host_registered_bytes += n,
                Some(false) => c.device_bytes += n,
                None if n > 0 => c.copied_bytes += n,
                None => {}
            }
            if is_write {
                c.writes_posted += 1;
                c.bytes_written += n;
            } else if is_fin {
                c.fins_sent += 1;
            } else {
                c.chunks_sent += 1;
            }
            let peer = st.peers[member].as_mut().expect("a peer");
            if takes_receive {
                peer.send_seq = peer.send_seq.wrapping_add(1);
                peer.data_reported = peer.data_freed;
                peer.data_sent = peer.data_sent.wrapping_add(1);
            }
            peer.slot_use[slot as usize] = SlotUse::Data { op: Some(id) };
            if finished {
                peer.current = None;
            }
            batch.push(wr);
        }
        if batch.is_empty() {
            return;
        }
        st.counters.bytes_sent += bytes;
        st.counters.batches_posted += 1;
        let peer = st.peers[member].as_mut().expect("a peer");
        let sges: Vec<Vec<Sge<'_>>> = batch
            .iter()
            .map(|w| match w {
                Wr::Send { slot, len } => vec![Sge::new(&peer.send_slots[*slot as usize], 0, *len)],
                Wr::Gather {
                    slot,
                    region,
                    off,
                    n,
                } => vec![
                    Sge::new(&peer.send_slots[*slot as usize], 0, HEADER_BYTES as u32),
                    Sge::new(region, *off, *n),
                ],
                Wr::Write {
                    slot,
                    local,
                    off,
                    n,
                    ..
                } => vec![Sge::new(
                    local.as_ref().unwrap_or(&peer.send_slots[*slot as usize]),
                    *off,
                    *n,
                )],
            })
            .collect();
        let reqs: Vec<SendRequest<'_>> = batch
            .iter()
            .zip(&sges)
            .map(|(w, sge)| SendRequest {
                tag: slot_tag(member, false, w.slot()),
                op: match w {
                    Wr::Write { remote, .. } => SendOp::Write(*remote),
                    _ => SendOp::Send,
                },
                sges: &sge[..],
            })
            .collect();
        let result = peer.qp.post_sends(&reqs);
        drop(reqs);
        drop(sges);
        if let Err(e) = result {
            // What was posted completes (flushed) and frees its slot; what wasn't keeps
            // its slot marked. The link is never used again either way.
            let why = format!("member {member}: posting sends failed: {e}");
            self.lose(st, member, &why);
        }
    }
}

/// One work request a pump posts.
enum Wr {
    /// A SEND of `len` bytes from a slot.
    Send { slot: u32, len: u32 },
    /// A SEND of the slot's header followed by `n` bytes at `off` of `region`.
    Gather {
        slot: u32,
        region: MemoryRegion,
        off: usize,
        n: u32,
    },
    /// An RDMA WRITE of `n` bytes at `off` of `local` (None: the slot) to `remote`.
    Write {
        slot: u32,
        local: Option<MemoryRegion>,
        off: usize,
        n: u32,
        remote: RemoteTarget,
    },
}

impl Wr {
    fn slot(&self) -> u32 {
        match self {
            Wr::Send { slot, .. } | Wr::Gather { slot, .. } | Wr::Write { slot, .. } => *slot,
        }
    }
}
