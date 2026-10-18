//! One bridge member's end of the verbs bridge (docs/VERBS_TRANSPORT.md, "Wire
//! protocol"): an RC queue pair to every other member, tagged messages over them with
//! credits, and a completion thread.
//!
//! **Memory.** The device only ever reads and writes this endpoint's own registered
//! slots: a message is copied from the caller's memory into a send slot and sent; it
//! arrives in a receive slot and is copied into the caller's memory (or, if no receive is
//! posted for it yet, into a bounded heap buffer). The caller's memory is never
//! registered, so nothing the device does can touch it, and a queue pair that never
//! drains can only leak this endpoint's slots. Every copy into or out of the caller's
//! memory happens with the endpoint's lock held, and [`Endpoint::abort`] cancels every
//! operation under that lock: once it returns, nothing reads or writes caller memory
//! again.
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

use crate::error::{Error, Result};
use crate::protocol::{self, Header, Info, Kind, Selected, HEADER_BYTES, PROTOCOL_VERSION};
use crate::provider::Provider;
use crate::resources::{
    CompletionChannel, CompletionQueue, MemoryRegion, ProtectionDomain, QueuePair, SendRequest, Sge,
};
use crate::types::{
    Access, Address, Completion, GlobalRoute, InitAttr, LinkLayer, Mtu, QpCaps, RtrAttr, RtsAttr,
    SendOp, WcStatus,
};

/// How long the completion thread blocks at most before it checks whether to stop.
pub const TICK: Duration = Duration::from_millis(100);
/// Completions taken per poll.
const POLL_BATCH: usize = 64;

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
                // The C interface's own keys (which provider), handled there.
                "provider" | "fabric_dir" => {}
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
    Data { op: Option<i64> },
    Control,
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
    /// Send operations, in order; the front one is being chunked.
    sends: VecDeque<i64>,
    /// Receives posted and not yet matched to an arriving message, by tag.
    posted: HashMap<u64, i64>,
    incoming: HashMap<u64, Incoming>,
    /// Whole messages nobody posted a receive for yet, by tag, oldest first.
    complete: HashMap<u64, VecDeque<Vec<u8>>>,
    /// Arrived data chunks still in their receive slots (the unexpected buffer was full),
    /// in arrival order; while any is held, later ones queue behind it.
    held: VecDeque<(u32, Header)>,
    reposts: Vec<u32>,
}

enum OpKind {
    Send {
        next: u64,
        posted: u64,
        acked: u64,
        chunks: u64,
    },
    Recv,
}

struct Op {
    peer: usize,
    tag: u64,
    ptr: usize,
    len: u64,
    kind: OpKind,
    done: bool,
    error: Option<String>,
}

struct State {
    failed: Option<String>,
    closing: bool,
    peers: Vec<Option<Peer>>,
    ops: HashMap<i64, Op>,
    next_id: i64,
    unexpected: u64,
    counters: Counters,
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
}

struct Shared {
    config: Config,
    index: usize,
    members: usize,
    local: Local,
    /// Every queue pair and slot holds it too; kept here as the seam where phase A3
    /// registers device memory for direct transfers.
    _pd: ProtectionDomain,
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
        let caps = QpCaps {
            max_send_wr: config.send_slots,
            max_recv_wr: config.credits + config.ctrl_credits,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        };
        let mut slots = Vec::new();
        let mut qpns = vec![0; members];
        let mut psns = vec![0; members];
        for member in (0..members).filter(|m| *m != index) {
            let qp = pd.create_qp(&cq, &cq, caps)?;
            // No remote access: this protocol only sends.
            qp.to_init(&InitAttr {
                port,
                pkey_index: 0,
                access: Access::NONE,
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
                posted: HashMap::new(),
                incoming: HashMap::new(),
                complete: HashMap::new(),
                held: VecDeque::new(),
                reposts: Vec::new(),
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
            },
            config,
            index,
            members,
            _pd: pd,
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

    fn post(&self, member: usize, ptr: usize, len: u64, tag: u64, send: bool) -> Result<i64> {
        let s = &self.shared;
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
        if len > 0 && ptr == 0 {
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
        if !send && peer.posted.contains_key(&tag) {
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
            OpKind::Send {
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
                ptr,
                len,
                kind,
                done: false,
                error: None,
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
    /// operation done, it fails, or [`abort`](Endpoint::abort) or `close` returns.
    pub unsafe fn send(&self, member: usize, ptr: usize, len: u64, tag: u64) -> Result<i64> {
        self.post(member, ptr, len, tag, true)
    }

    /// Receives the message `member` sends with `tag` into `len` bytes at `ptr` (it must
    /// be exactly `len` bytes long).
    ///
    /// # Safety
    ///
    /// `len` bytes at `ptr` must stay valid, and not be used otherwise, until `wait`
    /// reports the operation done, it fails, or `abort` or `close` returns.
    pub unsafe fn recv(&self, member: usize, ptr: usize, len: u64, tag: u64) -> Result<i64> {
        self.post(member, ptr, len, tag, false)
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
                        if let Some(e) = &op.error {
                            let e = e.clone();
                            st.failed.get_or_insert_with(|| e.clone());
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
        st.failed.get_or_insert_with(|| why.to_owned());
        s.cancel_all(&mut st);
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
            st.failed.get_or_insert_with(|| "closed".into());
            s.cancel_all(&mut st);
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
            if let Err(e) = peer.qp.close(left) {
                if result.is_ok() {
                    result = Err(e);
                }
            }
        }
        result
    }

    /// Telemetry as one JSON object (docs/VERBS_TRANSPORT.md, "Telemetry"). No remote
    /// keys exist in this protocol, so none can appear here.
    pub fn telemetry(&self) -> String {
        let s = &self.shared;
        let l = &s.local;
        let st = lock(&s.state);
        let c = &st.counters;
        let mut o = String::from("{");
        let q = |v: &str| {
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
        };
        let _ = write!(
            o,
            "\"protocol\":{PROTOCOL_VERSION},\"provider\":{},\"mock\":{},\"library\":{},\
             \"device\":{},\"port\":{},\"link_layer\":{},\"gid_index\":{},\"gid_type\":{},\
             \"active_mtu\":{},\"firmware\":{},\"vendor_id\":{},\"vendor_part_id\":{},\
             \"path\":\"host\",\"operations\":\"RC SEND/RECV\",\"slot_bytes\":{},\
             \"credits\":{},\"ctrl_credits\":{},\"send_slots\":{},\"members\":{},\"index\":{},",
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
            s.config.slot_bytes,
            s.config.credits,
            s.config.ctrl_credits,
            s.config.send_slots,
            s.members,
            s.index,
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
             \"batches_posted\":{},\"pending_operations\":{},\"peers\":[",
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

impl State {
    fn peer(&mut self, member: usize) -> &mut Peer {
        self.peers[member].as_mut().expect("a peer")
    }

    fn fail_op(&mut self, id: i64, why: &str) {
        if let Some(op) = self.ops.get_mut(&id) {
            if !op.done && op.error.is_none() {
                op.error = Some(why.to_owned());
            }
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
            None => {
                st.failed.get_or_insert(why);
                self.cancel_all(&mut st);
            }
        }
        self.changed.notify_all();
    }

    fn fail_all(&self, why: &str) {
        let mut st = lock(&self.state);
        st.failed.get_or_insert_with(|| why.to_owned());
        let ids: Vec<i64> = st.ops.keys().copied().collect();
        for id in ids {
            st.fail_op(id, why);
        }
        self.changed.notify_all();
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
        self.changed.notify_all();
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
            Kind::Credit | Kind::Bye => {
                st.counters.control_received += 1;
                let peer = st.peer(member);
                peer.ctrl_freed = peer.ctrl_freed.wrapping_add(1);
                peer.reposts.push(slot);
                if header.kind == Kind::Bye {
                    peer.said_bye = true;
                    // Nothing more comes from it: receives it can't still meet fail.
                    let waiting: Vec<i64> = peer.posted.drain().map(|(_, id)| id).collect();
                    let why =
                        format!("member {member} closed its end of the bridge before sending this");
                    for id in waiting {
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
        let ids: Vec<i64> = st.ops.keys().copied().collect();
        for id in ids {
            st.fail_op(id, why);
        }
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
        peer.sends.clear();
        peer.posted.clear();
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
            peer.sends.clear();
            peer.posted.clear();
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
        let n = h.chunk_len as usize;
        let src = &peer.recv_slots[slot as usize];
        let copied = match &mut inc.dest {
            Dest::Op(id) => match st.ops.get(id) {
                Some(op) if n > 0 => {
                    // SAFETY: the op's buffer holds op.len bytes (send/recv's contract)
                    // and offset + n <= msg_len == op.len (checked by Header::decode and
                    // above); the op isn't done or cancelled, so the caller still keeps it
                    // valid, and the endpoint lock is held.
                    let out = unsafe {
                        std::slice::from_raw_parts_mut((op.ptr + h.offset as usize) as *mut u8, n)
                    };
                    src.read(HEADER_BYTES, out)
                }
                _ => Ok(()),
            },
            Dest::Buffer(buf) => src.read(
                HEADER_BYTES,
                &mut buf[h.offset as usize..h.offset as usize + n],
            ),
            Dest::Discard => Ok(()),
        };
        if let Err(e) = copied {
            let why = format!("copying a chunk from member {member} failed: {e}");
            self.protocol_error(st, member, &why);
            return true;
        }
        inc.received += u64::from(h.chunk_len);
        st.counters.bytes_received += u64::from(h.chunk_len);
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
        let (tag, len, ptr) = {
            let op = &st.ops[&id];
            (op.tag, op.len, op.ptr)
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
                // SAFETY: the op's buffer holds len bytes (recv's contract), still valid:
                // it was just posted and the endpoint lock is held.
                unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr as *mut u8, buf.len()) };
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
                let have = inc.received as usize;
                if have > 0 {
                    // SAFETY: as above; have <= msg_len == len.
                    unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr as *mut u8, have) };
                }
                let freed = buf.len() as u64;
                inc.dest = Dest::Op(id);
                st.unexpected -= freed;
            } else {
                // Another receive is already taking this message: this one waits for the
                // next with the same tag.
                peer.posted.insert(tag, id);
            }
        } else if peer.said_bye {
            let why = format!("member {member} closed its end of the bridge before sending this");
            st.fail_op(id, &why);
        } else {
            peer.posted.insert(tag, id);
        }
        self.release_held(st, member);
        self.repost(st, member);
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

    /// Sends what slots and credits allow to `member`: control messages first, then the
    /// front send operation's next chunks. One batch post.
    fn pump(&self, st: &mut State, member: usize) {
        let cfg = &self.config;
        let payload = cfg.payload();
        let threshold = (cfg.credits / 2).max(1);
        // (slot, bytes to send from it)
        let mut batch: Vec<(u32, u32)> = Vec::new();
        let mut bytes = 0;
        loop {
            let peer = st.peers[member].as_mut().expect("a peer");
            // A peer that said goodbye wants nothing more, credits included.
            if !peer.connected || peer.lost.is_some() || peer.bye_sent || peer.said_bye {
                break;
            }
            let data_avail = cfg
                .credits
                .wrapping_sub(peer.data_sent.wrapping_sub(peer.data_returned));
            let ctrl_avail = cfg
                .ctrl_credits
                .wrapping_sub(peer.ctrl_sent.wrapping_sub(peer.ctrl_returned));
            let owed = peer.data_freed.wrapping_sub(peer.data_reported);
            let front = peer.sends.front().copied();
            let need_credit = owed >= threshold && (front.is_none() || data_avail == 0);
            let need_bye = peer.bye_pending && front.is_none();
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
                batch.push((slot, HEADER_BYTES as u32));
                st.counters.control_sent += 1;
                if kind == Kind::Bye {
                    break;
                }
                continue;
            }
            let Some(id) = front else {
                break;
            };
            if data_avail == 0 {
                if !peer.stalled {
                    peer.stalled = true;
                    st.counters.credit_stalls += 1;
                }
                break;
            }
            let Some(slot) = peer.free_send.pop() else {
                break;
            };
            peer.stalled = false;
            let op = st.ops.get_mut(&id).expect("a queued send");
            let OpKind::Send {
                next,
                posted,
                chunks,
                ..
            } = &mut op.kind
            else {
                unreachable!("only sends are queued")
            };
            let n = (op.len - *next).min(payload);
            let peer = st.peers[member].as_mut().expect("a peer");
            let h = Header {
                kind: Kind::Data,
                seq: peer.send_seq,
                chunk_len: n as u32,
                data_freed: peer.data_freed,
                ctrl_freed: peer.ctrl_freed,
                tag: op.tag,
                msg_len: op.len,
                offset: *next,
            };
            let region = &peer.send_slots[slot as usize];
            let mut written = region.write(0, &h.encode());
            if written.is_ok() && n > 0 {
                // SAFETY: the op's buffer holds op.len bytes and next + n <= op.len; the
                // op is queued, so not done or cancelled, and the caller keeps its memory
                // valid until then (send's contract); the endpoint lock is held.
                let data = unsafe {
                    std::slice::from_raw_parts((op.ptr + *next as usize) as *const u8, n as usize)
                };
                written = region.write(HEADER_BYTES, data);
            }
            if let Err(e) = written {
                peer.free_send.push(slot);
                let why = format!("member {member}: filling a send slot failed: {e}");
                self.lose(st, member, &why);
                return;
            }
            *next += n;
            *posted += 1;
            let finished = *posted == *chunks;
            bytes += n;
            peer.send_seq = peer.send_seq.wrapping_add(1);
            peer.data_reported = peer.data_freed;
            peer.data_sent = peer.data_sent.wrapping_add(1);
            peer.slot_use[slot as usize] = SlotUse::Data { op: Some(id) };
            if finished {
                peer.sends.pop_front();
            }
            batch.push((slot, HEADER_BYTES as u32 + n as u32));
            st.counters.chunks_sent += 1;
        }
        if batch.is_empty() {
            return;
        }
        st.counters.bytes_sent += bytes;
        st.counters.batches_posted += 1;
        let peer = st.peers[member].as_mut().expect("a peer");
        let sges: Vec<[Sge<'_>; 1]> = batch
            .iter()
            .map(|&(s, l)| [Sge::new(&peer.send_slots[s as usize], 0, l)])
            .collect();
        let reqs: Vec<SendRequest<'_>> = batch
            .iter()
            .zip(&sges)
            .map(|(&(s, _), sge)| SendRequest {
                tag: slot_tag(member, false, s),
                op: SendOp::Send,
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
