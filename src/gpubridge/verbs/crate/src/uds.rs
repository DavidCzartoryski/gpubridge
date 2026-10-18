//! A cross-process MOCK fabric for tests (the `test-fabric` feature): RC queue pairs
//! between processes on one machine, carried over Unix sockets in a directory the test
//! names. It is not RDMA and moves no data over a network: it lets the Python tests run
//! `bridge="verbs"` between real processes, through the same endpoint, protocol and
//! completion thread as libibverbs. Only a library built with this feature has it, only
//! an explicit setting selects it, and telemetry says MOCK.
//!
//! Each provider is a node with one InfiniBand-like device (`mockfab0`, one port) whose
//! LID it takes by creating `lid-<n>` in the directory, and a socket `node-<n>.sock`
//! there. A send copies its bytes out of registered memory when posted (as a NIC may) and
//! a node thread writes them to the peer node's socket; the peer node's reader thread puts
//! them into the receive posted on the destination queue pair (or keeps them until one is
//! posted: an infinite RNR retry) and acknowledges; the acknowledgement completes the
//! send. A send whose peer node is gone, or whose destination queue pair doesn't exist
//! or isn't connected back with matching PSNs, completes with `RetryExcErr`. Completion
//! channels and notifications work as in verbs. There are no asynchronous events, RDMA
//! writes or reads.
//!
//! No thread blocks on a socket while holding the node's lock: every frame leaves through
//! one sender thread's queue, so frames of one queue pair keep their order.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::provider::{Handle, MrKeys, Provider, QpCreated};
use crate::types::{
    Access, Address, DeviceAttr, Events, Gid, GidEntry, GidType, LinkLayer, Mtu, PortAttr,
    PortState, QpCaps, QpState, RawCompletion, RawSge, SendOp, Transition, WcStatus,
};

const DEVICE: &str = "mockfab0";
const EINVAL: i32 = 22;
const EIO: i32 = 5;
const ENODEV: i32 = 19;
/// How long an unacknowledged send waits before the node checks its peer is still there.
const ACK_PATIENCE: Duration = Duration::from_secs(2);
/// How often blocked threads look up to see whether the node is stopping.
const NAP: Duration = Duration::from_millis(50);

const FRAME_HELLO: u8 = 1;
const FRAME_DATA: u8 = 2;
const FRAME_ACK: u8 = 3;

const ACK_OK: u8 = 0;
/// The receive was too short (the sender sees `RemInvReqErr`).
const ACK_TOO_LONG: u8 = 1;
/// No such queue pair, or not connected to the sender (the sender sees `RetryExcErr`).
const ACK_NO_QP: u8 = 2;

fn verbs(call: &'static str, errno: i32) -> Error {
    Error::Verbs { call, errno }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

struct Mr {
    pd: Handle,
    addr: u64,
    len: u64,
    lkey: u32,
    access: Access,
}

struct Cq {
    depth: u32,
    entries: VecDeque<RawCompletion>,
    channel: Option<Handle>,
    armed: bool,
}

struct Unacked {
    seq: u64,
    wr_id: u64,
    since: Instant,
}

struct Arrived {
    seq: u64,
    src_lid: u16,
    imm: Option<u32>,
    payload: Vec<u8>,
}

struct Qp {
    qpn: u32,
    pd: Handle,
    send_cq: Handle,
    recv_cq: Handle,
    caps: QpCaps,
    state: QpState,
    /// (peer LID, peer queue pair number, PSN expected from it) from Rtr.
    peer: Option<(u16, u32, u32)>,
    sq_psn: u32,
    next_seq: u64,
    unacked: VecDeque<Unacked>,
    rq: VecDeque<(u64, Vec<RawSge>)>,
    arrived: VecDeque<Arrived>,
}

#[derive(Default)]
struct State {
    next_handle: Handle,
    next_qpn: u32,
    next_key: u32,
    contexts: usize,
    pds: HashSet<Handle>,
    channels: HashMap<Handle, bool>,
    cqs: HashMap<Handle, Cq>,
    qps: HashMap<Handle, Qp>,
    by_qpn: HashMap<u32, Handle>,
    mrs: HashMap<Handle, Mr>,
    dead: HashSet<u16>,
    probed: HashMap<u16, Instant>,
}

/// A frame for the node at a LID.
type Outgoing = (u16, Vec<u8>);

struct Shared {
    dir: PathBuf,
    lid: u16,
    state: Mutex<State>,
    wake: Condvar,
    out: Mutex<Option<Sender<Outgoing>>>,
    stop: AtomicBool,
}

/// One process's node on the test fabric. See the module docs.
pub struct UdsProvider {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for UdsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdsProvider(lid {})", self.shared.lid)
    }
}

fn socket_path(dir: &Path, lid: u16) -> PathBuf {
    dir.join(format!("node-{lid}.sock"))
}

impl UdsProvider {
    /// Joins the fabric in `dir` (an existing directory the test owns).
    pub fn open(dir: &Path) -> Result<UdsProvider> {
        if !dir.is_dir() {
            return Err(Error::invalid(format!(
                "test fabric directory {} doesn't exist",
                dir.display()
            )));
        }
        let io = |e: std::io::Error| Error::Unsupported(format!("test fabric: {e}"));
        // A LID nobody else on this fabric holds.
        let mut lid = 1u16;
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.join(format!("lid-{lid}")))
            {
                Ok(_) => break,
                Err(e) if e.kind() == ErrorKind::AlreadyExists && lid < 0xBFFF => lid += 1,
                Err(e) => return Err(io(e)),
            }
        }
        let path = socket_path(dir, lid);
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).map_err(io)?;
        listener.set_nonblocking(true).map_err(io)?;
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            dir: dir.to_owned(),
            lid,
            state: Mutex::new(State {
                next_handle: 1,
                next_qpn: 0x100,
                next_key: 1,
                ..State::default()
            }),
            wake: Condvar::new(),
            out: Mutex::new(Some(tx)),
            stop: AtomicBool::new(false),
        });
        let s = shared.clone();
        let accept = thread::Builder::new()
            .name("verbs-test-fabric-accept".into())
            .spawn(move || s.accept(listener))
            .map_err(io)?;
        let s = shared.clone();
        let send = thread::Builder::new()
            .name("verbs-test-fabric-send".into())
            .spawn(move || s.sender(rx))
            .map_err(io)?;
        Ok(UdsProvider {
            shared,
            threads: vec![accept, send],
        })
    }
}

impl Drop for UdsProvider {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        lock(&self.shared.out).take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let _ = fs::remove_file(socket_path(&self.shared.dir, self.shared.lid));
        let _ = fs::remove_file(self.shared.dir.join(format!("lid-{}", self.shared.lid)));
    }
}

fn frame(kind: u8, body: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(5 + body.len());
    f.extend_from_slice(&(body.len() as u32 + 1).to_le_bytes());
    f.push(kind);
    f.extend_from_slice(body);
    f
}

fn read_frame(stream: &mut UnixStream, stop: &AtomicBool) -> Option<(u8, Vec<u8>)> {
    let mut len = [0u8; 4];
    read_all(stream, &mut len, stop)?;
    let n = u32::from_le_bytes(len) as usize;
    if n == 0 {
        return None;
    }
    let mut body = vec![0u8; n];
    read_all(stream, &mut body, stop)?;
    let kind = body.remove(0);
    Some((kind, body))
}

/// read_exact that gives up when the node stops (the socket has a read timeout).
fn read_all(stream: &mut UnixStream, mut buf: &mut [u8], stop: &AtomicBool) -> Option<()> {
    while !buf.is_empty() {
        match stream.read(buf) {
            Ok(0) => return None,
            Ok(n) => buf = &mut buf[n..],
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    Some(())
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(b[i..i + 2].try_into().unwrap())
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

impl Shared {
    fn queue(&self, lid: u16, frame: Vec<u8>) {
        if let Some(tx) = lock(&self.out).as_ref() {
            let _ = tx.send((lid, frame));
        }
    }

    fn accept(self: Arc<Self>, listener: UnixListener) {
        let mut readers = Vec::new();
        while !self.stop.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let s = self.clone();
                    readers.push(thread::spawn(move || s.read(stream)));
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(_) => thread::sleep(NAP),
            }
        }
        for r in readers {
            let _ = r.join();
        }
    }

    /// One accepted connection: a peer node's frames to this one.
    fn read(self: Arc<Self>, mut stream: UnixStream) {
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(NAP));
        let Some((FRAME_HELLO, body)) = read_frame(&mut stream, &self.stop) else {
            return; // a liveness probe, or garbage
        };
        if body.len() < 2 {
            return;
        }
        let peer = u16_at(&body, 0);
        while let Some((kind, body)) = read_frame(&mut stream, &self.stop) {
            match kind {
                FRAME_DATA if body.len() >= 25 => self.data(peer, &body),
                FRAME_ACK if body.len() >= 13 => self.ack(&body),
                _ => break,
            }
        }
        if !self.stop.load(Ordering::Acquire) {
            self.peer_gone(peer);
        }
    }

    fn sender(self: Arc<Self>, rx: Receiver<Outgoing>) {
        let mut conns: HashMap<u16, UnixStream> = HashMap::new();
        loop {
            let (lid, frame) = match rx.recv_timeout(NAP) {
                Ok(m) => m,
                Err(RecvTimeoutError::Timeout) => {
                    if self.stop.load(Ordering::Acquire) {
                        return;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            };
            if lock(&self.state).dead.contains(&lid) {
                continue;
            }
            let ok = match conns.entry(lid) {
                std::collections::hash_map::Entry::Occupied(mut c) => {
                    c.get_mut().write_all(&frame).is_ok()
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    match UnixStream::connect(socket_path(&self.dir, lid)) {
                        Ok(mut c) => {
                            // A peer that stops reading fails rather than hangs the node.
                            let _ = c.set_write_timeout(Some(Duration::from_secs(10)));
                            let hello = frame_hello(self.lid);
                            let ok = c.write_all(&hello).is_ok() && c.write_all(&frame).is_ok();
                            v.insert(c);
                            ok
                        }
                        Err(_) => false,
                    }
                }
            };
            if !ok {
                conns.remove(&lid);
                self.peer_gone(lid);
            }
        }
    }

    /// The node at `lid` is gone: its sends fail and nothing more goes there.
    fn peer_gone(&self, lid: u16) {
        let mut st = lock(&self.state);
        st.dead.insert(lid);
        let failed: Vec<Handle> = st
            .qps
            .iter()
            .filter(|(_, q)| q.peer.is_some_and(|p| p.0 == lid) && !q.unacked.is_empty())
            .map(|(h, _)| *h)
            .collect();
        for qp in failed {
            fail_head(&mut st, qp, WcStatus::RetryExcErr);
        }
        drop(st);
        self.wake.notify_all();
    }

    fn data(&self, from: u16, b: &[u8]) {
        let (src_qpn, dst_qpn, psn, seq) =
            (u32_at(b, 0), u32_at(b, 4), u32_at(b, 8), u64_at(b, 12));
        let imm = (b[20] != 0).then(|| u32_at(b, 21));
        let payload = b[25.min(b.len())..].to_vec();
        let mut st = lock(&self.state);
        let target = st.by_qpn.get(&dst_qpn).copied().filter(|h| {
            let q = &st.qps[h];
            matches!(q.state, QpState::Rtr | QpState::Rts) && q.peer == Some((from, src_qpn, psn))
        });
        let Some(qp) = target else {
            drop(st);
            self.queue(from, ack_frame(src_qpn, seq, ACK_NO_QP));
            return;
        };
        st.qps.get_mut(&qp).unwrap().arrived.push_back(Arrived {
            seq,
            src_lid: from,
            imm,
            payload,
        });
        let acks = deliver(&mut st, qp);
        drop(st);
        for (lid, f) in acks {
            self.queue(lid, f);
        }
        self.wake.notify_all();
    }

    fn ack(&self, b: &[u8]) {
        let (qpn, seq, status) = (u32_at(b, 0), u64_at(b, 4), b[12]);
        let mut st = lock(&self.state);
        let Some(&qp) = st.by_qpn.get(&qpn) else {
            return;
        };
        let head = st.qps[&qp].unacked.front().map(|u| (u.seq, u.wr_id));
        if let Some((s, wr_id)) = head {
            if s == seq {
                match status {
                    ACK_OK => {
                        st.qps.get_mut(&qp).unwrap().unacked.pop_front();
                        let cq = st.qps[&qp].send_cq;
                        push(&mut st, cq, wr_id, qpn, WcStatus::Success, 0, None);
                    }
                    ACK_TOO_LONG => fail_head(&mut st, qp, WcStatus::RemInvReqErr),
                    _ => fail_head(&mut st, qp, WcStatus::RetryExcErr),
                }
            }
        }
        drop(st);
        self.wake.notify_all();
    }

    /// Unacknowledged sends to a node that no longer answers fail.
    fn check_peers(&self, st: &mut State) {
        let now = Instant::now();
        let overdue: Vec<(Handle, u16)> = st
            .qps
            .iter()
            .filter_map(|(h, q)| {
                let u = q.unacked.front()?;
                let lid = q.peer?.0;
                (now.duration_since(u.since) > ACK_PATIENCE).then_some((*h, lid))
            })
            .collect();
        for (qp, lid) in overdue {
            let recent = st
                .probed
                .get(&lid)
                .is_some_and(|t| now.duration_since(*t) < ACK_PATIENCE);
            if !st.dead.contains(&lid) && !recent {
                st.probed.insert(lid, now);
                // A connect that fails means the node's process is gone.
                if UnixStream::connect(socket_path(&self.dir, lid)).is_err() {
                    st.dead.insert(lid);
                }
            }
            if st.dead.contains(&lid) {
                fail_head(st, qp, WcStatus::RetryExcErr);
            }
        }
    }
}

fn frame_hello(lid: u16) -> Vec<u8> {
    frame(FRAME_HELLO, &lid.to_le_bytes())
}

fn ack_frame(qpn: u32, seq: u64, status: u8) -> Vec<u8> {
    let mut b = Vec::with_capacity(13);
    b.extend_from_slice(&qpn.to_le_bytes());
    b.extend_from_slice(&seq.to_le_bytes());
    b.push(status);
    frame(FRAME_ACK, &b)
}

/// Queues a completion; notifies the channel if the queue is armed.
fn push(
    st: &mut State,
    cq: Handle,
    wr_id: u64,
    qp_num: u32,
    status: WcStatus,
    len: u32,
    imm: Option<u32>,
) {
    let Some(rec) = st.cqs.get_mut(&cq) else {
        return;
    };
    if rec.entries.len() >= rec.depth as usize {
        // An overrun: the owned layer never lets this happen.
        return;
    }
    rec.entries.push_back(RawCompletion {
        wr_id,
        qp_num,
        status,
        byte_len: len,
        imm,
        vendor_err: 0,
    });
    if rec.armed {
        rec.armed = false;
        if let Some(ch) = rec.channel {
            st.channels.insert(ch, true);
        }
    }
}

/// The head send failed with `status`: the queue pair goes to `Err` and flushes the rest.
fn fail_head(st: &mut State, qp: Handle, status: WcStatus) {
    let Some(q) = st.qps.get_mut(&qp) else {
        return;
    };
    let qpn = q.qpn;
    let (send_cq, recv_cq) = (q.send_cq, q.recv_cq);
    let head = q.unacked.pop_front();
    if let Some(u) = head {
        push(st, send_cq, u.wr_id, qpn, status, 0, None);
    }
    flush(st, qp, send_cq, recv_cq);
}

fn flush(st: &mut State, qp: Handle, send_cq: Handle, recv_cq: Handle) {
    let q = st.qps.get_mut(&qp).unwrap();
    q.state = QpState::Err;
    let qpn = q.qpn;
    let sends: Vec<u64> = q.unacked.drain(..).map(|u| u.wr_id).collect();
    let recvs: Vec<u64> = q.rq.drain(..).map(|r| r.0).collect();
    q.arrived.clear();
    for wr_id in sends {
        push(st, send_cq, wr_id, qpn, WcStatus::WrFlushErr, 0, None);
    }
    for wr_id in recvs {
        push(st, recv_cq, wr_id, qpn, WcStatus::WrFlushErr, 0, None);
    }
}

/// Moves arrived messages into posted receives; returns the acknowledgements to send.
fn deliver(st: &mut State, qp: Handle) -> Vec<Outgoing> {
    let mut acks = Vec::new();
    loop {
        let q = st.qps.get_mut(&qp).unwrap();
        if q.state == QpState::Err || q.arrived.is_empty() || q.rq.is_empty() {
            return acks;
        }
        let msg = q.arrived.pop_front().unwrap();
        let (wr_id, sges) = q.rq.pop_front().unwrap();
        let (qpn, pd, recv_cq, send_cq) = (q.qpn, q.pd, q.recv_cq, q.send_cq);
        let peer_qpn = q.peer.map_or(0, |p| p.1);
        match scatter(st, pd, &sges, &msg.payload) {
            Ok(()) => {
                let len = msg.payload.len() as u32;
                push(st, recv_cq, wr_id, qpn, WcStatus::Success, len, msg.imm);
                acks.push((msg.src_lid, ack_frame(peer_qpn, msg.seq, ACK_OK)));
            }
            Err(()) => {
                push(st, recv_cq, wr_id, qpn, WcStatus::LocLenErr, 0, None);
                flush(st, qp, send_cq, recv_cq);
                acks.push((msg.src_lid, ack_frame(peer_qpn, msg.seq, ACK_TOO_LONG)));
                return acks;
            }
        }
    }
}

fn find_mr(st: &State, pd: Handle, lkey: u32, addr: u64, len: u64) -> Option<&Mr> {
    st.mrs.values().find(|m| {
        m.pd == pd
            && m.lkey == lkey
            && addr >= m.addr
            && addr.checked_add(len).is_some_and(|e| e <= m.addr + m.len)
    })
}

fn gather(st: &State, pd: Handle, sges: &[RawSge]) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    for s in sges {
        find_mr(st, pd, s.lkey, s.addr, u64::from(s.len))?;
        let start = data.len();
        data.resize(start + s.len as usize, 0);
        // SAFETY: inside a registered region (checked above), which its owner keeps valid
        // until it is deregistered (Provider::reg_mr's contract).
        unsafe {
            ptr::copy_nonoverlapping(
                s.addr as usize as *const u8,
                data.as_mut_ptr().add(start),
                s.len as usize,
            )
        };
    }
    Some(data)
}

fn scatter(st: &State, pd: Handle, sges: &[RawSge], data: &[u8]) -> std::result::Result<(), ()> {
    let room: u64 = sges.iter().map(|s| u64::from(s.len)).sum();
    if data.len() as u64 > room {
        return Err(());
    }
    for s in sges {
        let ok = find_mr(st, pd, s.lkey, s.addr, u64::from(s.len))
            .is_some_and(|m| m.access.contains(Access::LOCAL_WRITE));
        if !ok {
            return Err(());
        }
    }
    let mut done = 0;
    for s in sges {
        let n = (data.len() - done).min(s.len as usize);
        // SAFETY: inside a registered, locally writable region (checked above), valid by
        // Provider::reg_mr's contract; data is a separate buffer.
        unsafe { ptr::copy_nonoverlapping(data.as_ptr().add(done), s.addr as usize as *mut u8, n) };
        done += n;
    }
    Ok(())
}

impl UdsProvider {
    fn st(&self) -> MutexGuard<'_, State> {
        lock(&self.shared.state)
    }

    fn handle(st: &mut State) -> Handle {
        let h = st.next_handle;
        st.next_handle += 1;
        h
    }

    fn port(&self) -> PortAttr {
        PortAttr {
            state: PortState::Active,
            max_mtu: Mtu::Mtu4096,
            active_mtu: Mtu::Mtu4096,
            gid_tbl_len: 1,
            lid: self.shared.lid,
            sm_lid: 1,
            link_layer: LinkLayer::InfiniBand,
        }
    }
}

impl Provider for UdsProvider {
    fn name(&self) -> &str {
        "test-fabric"
    }

    fn describe(&self) -> String {
        "MOCK: cross-process test fabric over Unix sockets (not RDMA)".into()
    }

    fn list_devices(&self) -> Result<Vec<String>> {
        Ok(vec![DEVICE.into()])
    }

    fn open_device(&self, name: &str) -> Result<Handle> {
        if name != DEVICE {
            return Err(verbs("ibv_open_device", ENODEV));
        }
        let mut st = self.st();
        st.contexts += 1;
        Ok(Self::handle(&mut st))
    }

    fn close_device(&self, _ctx: Handle) -> Result<()> {
        self.st().contexts -= 1;
        Ok(())
    }

    fn query_device(&self, _ctx: Handle) -> Result<DeviceAttr> {
        Ok(DeviceAttr {
            max_mr_size: 1 << 40,
            node_guid: u64::from(self.shared.lid),
            max_qp: 1 << 16,
            max_qp_wr: 1 << 14,
            max_sge: 16,
            max_cq: 1 << 10,
            max_cqe: 1 << 20,
            max_mr: 1 << 20,
            max_pd: 1 << 10,
            max_qp_rd_atom: 16,
            max_qp_init_rd_atom: 16,
            vendor_id: 0,
            vendor_part_id: 0,
            phys_port_cnt: 1,
            fw_ver: "mock".into(),
        })
    }

    fn query_port(&self, _ctx: Handle, port: u8) -> Result<PortAttr> {
        if port != 1 {
            return Err(verbs("ibv_query_port", EINVAL));
        }
        Ok(self.port())
    }

    fn query_gid(&self, _ctx: Handle, port: u8, index: u32) -> Result<GidEntry> {
        if port != 1 || index != 0 {
            return Err(verbs("ibv_query_gid_ex", EINVAL));
        }
        let mut g = [0u8; 16];
        g[0] = 0xfe;
        g[1] = 0x80;
        g[14..].copy_from_slice(&self.shared.lid.to_be_bytes());
        Ok(GidEntry {
            gid: Gid(g),
            gid_type: GidType::Ib,
            ndev_ifindex: 0,
        })
    }

    fn alloc_pd(&self, _ctx: Handle) -> Result<Handle> {
        let mut st = self.st();
        let h = Self::handle(&mut st);
        st.pds.insert(h);
        Ok(h)
    }

    fn dealloc_pd(&self, pd: Handle) -> Result<()> {
        self.st().pds.remove(&pd);
        Ok(())
    }

    fn create_cq(&self, _ctx: Handle, depth: u32) -> Result<(Handle, u32)> {
        let mut st = self.st();
        let h = Self::handle(&mut st);
        st.cqs.insert(
            h,
            Cq {
                depth,
                entries: VecDeque::new(),
                channel: None,
                armed: false,
            },
        );
        Ok((h, depth))
    }

    fn create_cq_on_channel(
        &self,
        _ctx: Handle,
        depth: u32,
        channel: Handle,
    ) -> Result<(Handle, u32)> {
        let mut st = self.st();
        if !st.channels.contains_key(&channel) {
            return Err(verbs("ibv_create_cq", EINVAL));
        }
        let h = Self::handle(&mut st);
        st.cqs.insert(
            h,
            Cq {
                depth,
                entries: VecDeque::new(),
                channel: Some(channel),
                armed: false,
            },
        );
        Ok((h, depth))
    }

    fn destroy_cq(&self, cq: Handle) -> Result<()> {
        self.st().cqs.remove(&cq);
        Ok(())
    }

    fn create_comp_channel(&self, _ctx: Handle) -> Result<Handle> {
        let mut st = self.st();
        let h = Self::handle(&mut st);
        st.channels.insert(h, false);
        Ok(h)
    }

    fn destroy_comp_channel(&self, channel: Handle) -> Result<()> {
        self.st().channels.remove(&channel);
        Ok(())
    }

    fn req_notify_cq(&self, cq: Handle) -> Result<()> {
        let mut st = self.st();
        let rec = st
            .cqs
            .get_mut(&cq)
            .ok_or_else(|| verbs("ibv_req_notify_cq", EINVAL))?;
        rec.armed = true;
        if !rec.entries.is_empty() {
            rec.armed = false;
            if let Some(ch) = rec.channel {
                st.channels.insert(ch, true);
            }
        }
        Ok(())
    }

    fn wait_events(&self, _ctx: Handle, channel: Handle, timeout: Duration) -> Result<Events> {
        let deadline = Instant::now() + timeout;
        let mut st = self.st();
        loop {
            self.shared.check_peers(&mut st);
            match st.channels.get_mut(&channel) {
                None => return Err(verbs("ibv_get_cq_event", EINVAL)),
                Some(pending) if *pending => {
                    *pending = false;
                    return Ok(Events {
                        completion: true,
                        async_events: Vec::new(),
                    });
                }
                Some(_) => {}
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(Events::default());
            }
            st = self
                .shared
                .wake
                .wait_timeout(st, (deadline - now).min(NAP))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    fn create_rc_qp(
        &self,
        pd: Handle,
        send_cq: Handle,
        recv_cq: Handle,
        caps: &QpCaps,
    ) -> Result<QpCreated> {
        let mut st = self.st();
        if !st.pds.contains(&pd) || !st.cqs.contains_key(&send_cq) || !st.cqs.contains_key(&recv_cq)
        {
            return Err(verbs("ibv_create_qp", EINVAL));
        }
        let h = Self::handle(&mut st);
        let qpn = st.next_qpn;
        st.next_qpn += 1;
        st.qps.insert(
            h,
            Qp {
                qpn,
                pd,
                send_cq,
                recv_cq,
                caps: *caps,
                state: QpState::Reset,
                peer: None,
                sq_psn: 0,
                next_seq: 0,
                unacked: VecDeque::new(),
                rq: VecDeque::new(),
                arrived: VecDeque::new(),
            },
        );
        st.by_qpn.insert(qpn, h);
        Ok(QpCreated {
            handle: h,
            qp_num: qpn,
            caps: *caps,
        })
    }

    fn modify_qp(&self, qp: Handle, transition: &Transition) -> Result<()> {
        let mut st = self.st();
        let bad = || verbs("ibv_modify_qp", EINVAL);
        let q = st.qps.get_mut(&qp).ok_or_else(bad)?;
        match transition {
            Transition::Init(a) if q.state == QpState::Reset && a.port == 1 => {
                q.state = QpState::Init;
            }
            Transition::Rtr { attr, .. } if q.state == QpState::Init => {
                let Address::Ib { lid, .. } = attr.address else {
                    return Err(bad());
                };
                q.peer = Some((lid, attr.dest_qp_num, attr.rq_psn));
                q.state = QpState::Rtr;
            }
            Transition::Rts(a) if q.state == QpState::Rtr => {
                q.sq_psn = a.sq_psn;
                q.state = QpState::Rts;
            }
            Transition::Error => {
                let (s, r) = (q.send_cq, q.recv_cq);
                flush(&mut st, qp, s, r);
            }
            Transition::Reset => {
                q.state = QpState::Reset;
                q.peer = None;
                q.unacked.clear();
                q.rq.clear();
                q.arrived.clear();
            }
            _ => return Err(bad()),
        }
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    fn destroy_qp(&self, qp: Handle) -> Result<()> {
        let mut st = self.st();
        if let Some(q) = st.qps.remove(&qp) {
            st.by_qpn.remove(&q.qpn);
        }
        Ok(())
    }

    unsafe fn reg_mr(
        &self,
        pd: Handle,
        addr: *mut u8,
        len: usize,
        access: Access,
    ) -> Result<MrKeys> {
        let mut st = self.st();
        if !st.pds.contains(&pd) || addr.is_null() || len == 0 {
            return Err(verbs("ibv_reg_mr", EINVAL));
        }
        let h = Self::handle(&mut st);
        let key = st.next_key;
        st.next_key += 1;
        st.mrs.insert(
            h,
            Mr {
                pd,
                addr: addr as usize as u64,
                len: len as u64,
                lkey: key,
                access,
            },
        );
        Ok(MrKeys {
            handle: h,
            lkey: key,
            rkey: key,
        })
    }

    fn reg_dmabuf_mr(
        &self,
        _: Handle,
        _: u64,
        _: usize,
        _: u64,
        _: i32,
        _: Access,
    ) -> Result<MrKeys> {
        Err(Error::Unsupported(
            "the test fabric has no dma-buf memory".into(),
        ))
    }

    fn dereg_mr(&self, mr: Handle) -> Result<()> {
        self.st().mrs.remove(&mr);
        Ok(())
    }

    fn post_send(&self, qp: Handle, wr_id: u64, op: &SendOp, sges: &[RawSge]) -> Result<()> {
        let imm = match op {
            SendOp::Send => None,
            SendOp::SendWithImm(imm) => Some(*imm),
            _ => {
                return Err(Error::Unsupported(
                    "the test fabric has no RDMA writes".into(),
                ))
            }
        };
        let mut st = self.st();
        let q = st
            .qps
            .get(&qp)
            .ok_or_else(|| verbs("ibv_post_send", EINVAL))?;
        if q.state != QpState::Rts || q.unacked.len() >= q.caps.max_send_wr as usize {
            return Err(verbs("ibv_post_send", EINVAL));
        }
        let (pd, (lid, dst, _), psn, src) =
            (q.pd, q.peer.expect("Rts has a peer"), q.sq_psn, q.qpn);
        let Some(payload) = gather(&st, pd, sges) else {
            return Err(verbs("ibv_post_send", EINVAL));
        };
        let q = st.qps.get_mut(&qp).unwrap();
        let seq = q.next_seq;
        q.next_seq += 1;
        q.unacked.push_back(Unacked {
            seq,
            wr_id,
            since: Instant::now(),
        });
        if st.dead.contains(&lid) {
            fail_head(&mut st, qp, WcStatus::RetryExcErr);
            drop(st);
            self.shared.wake.notify_all();
            return Ok(());
        }
        let mut b = Vec::with_capacity(25 + payload.len());
        b.extend_from_slice(&src.to_le_bytes());
        b.extend_from_slice(&dst.to_le_bytes());
        b.extend_from_slice(&psn.to_le_bytes());
        b.extend_from_slice(&seq.to_le_bytes());
        b.push(u8::from(imm.is_some()));
        b.extend_from_slice(&imm.unwrap_or(0).to_le_bytes());
        b.extend_from_slice(&payload);
        // Queued while the lock is held, so this queue pair's frames keep their order.
        self.shared.queue(lid, frame(FRAME_DATA, &b));
        Ok(())
    }

    fn post_recv(&self, qp: Handle, wr_id: u64, sges: &[RawSge]) -> Result<()> {
        let mut st = self.st();
        let q = st
            .qps
            .get_mut(&qp)
            .ok_or_else(|| verbs("ibv_post_recv", EINVAL))?;
        if !matches!(q.state, QpState::Init | QpState::Rtr | QpState::Rts)
            || q.rq.len() >= q.caps.max_recv_wr as usize
        {
            return Err(verbs("ibv_post_recv", EINVAL));
        }
        q.rq.push_back((wr_id, sges.to_vec()));
        let acks = deliver(&mut st, qp);
        drop(st);
        for (lid, f) in acks {
            self.shared.queue(lid, f);
        }
        self.shared.wake.notify_all();
        Ok(())
    }

    fn poll_cq(&self, cq: Handle, out: &mut Vec<RawCompletion>, max: usize) -> Result<usize> {
        let mut st = self.st();
        let rec = st
            .cqs
            .get_mut(&cq)
            .ok_or_else(|| verbs("ibv_poll_cq", EIO))?;
        let n = max.min(rec.entries.len());
        out.extend(rec.entries.drain(..n));
        Ok(n)
    }
}
