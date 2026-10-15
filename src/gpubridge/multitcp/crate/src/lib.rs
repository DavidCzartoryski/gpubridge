//! Multi-stream TCP messaging for gpubridge's `bridge="multitcp"` (docs/RUST_TRANSPORT.md).
//!
//! A [`Worker`] is one bridge member's end: a listening socket, and `streams` TCP
//! connections to every other member (the member with the lower index dials). A message
//! is a tagged byte range in the caller's memory. It travels as one frame (a 16-byte
//! header, the tag and the length, then the bytes) on one of that peer's connections,
//! chosen round-robin, so consecutive messages to one peer go out in parallel. Every
//! connection has its own sending and receiving thread; bytes go straight from the
//! sender's memory to the socket and from the socket into the receiver's memory.
//!
//! A receive is matched by (peer, tag), and must be posted before the frame can be read:
//! a frame whose receive isn't posted yet waits for it, holding back its connection, not
//! the others. Python numbers its messages so tags never repeat within a collective, and
//! posts a collective's receives before waiting on any of them.
//!
//! The C functions below are what gpubridge.multitcp loads with ctypes. Every one catches
//! panics, so a bug here can't unwind into Python. Not thread-safe across callers: one
//! Python thread drives a worker, as the bridge does.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Wire protocol version: both ends must agree (Python also checks it at rendezvous).
pub const PROTOCOL: u32 = 1;
const MAGIC: u32 = 0x6762_7470; // "gbtp"
pub const FAILED: i64 = -1;
pub const TIMED_OUT: i64 = -2;
pub const PEER_FAILED: i64 = -3;

struct Job {
    id: i64,
    tag: u64,
    ptr: usize,
    len: usize,
}

struct Recv {
    id: i64,
    ptr: usize,
    len: usize,
}

#[derive(Default)]
struct State {
    next_id: i64,
    finished: HashSet<i64>,
    /// Posted receives, by (peer, tag).
    posted: HashMap<(usize, u64), Recv>,
    /// One queue per connection: index peer * streams + stream.
    queues: Vec<VecDeque<Job>>,
    next_stream: Vec<usize>,
    connected: Vec<bool>,
    failed: Option<String>,
    closing: bool,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A thread that panicked while holding the lock leaves the state usable for a
        // failure report, which is all that is left to do.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fail(&self, message: String) {
        let mut state = self.lock();
        if state.failed.is_none() && !state.closing {
            state.failed = Some(message);
        }
        self.changed.notify_all();
    }
}

/// Connections accepted from lower-index members, by (member, stream), until connect()
/// takes them.
type Accepted = Arc<(Mutex<HashMap<(usize, usize), TcpStream>>, Condvar)>;

pub struct Worker {
    members: usize,
    index: usize,
    streams: usize,
    address: String,
    shared: Arc<Shared>,
    accepted: Accepted,
    sockets: Vec<TcpStream>,
    threads: Vec<JoinHandle<()>>,
    last_error: CString,
}

fn write_header(socket: &mut TcpStream, words: &[u64]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    socket.write_all(&bytes)
}

fn read_words<const N: usize>(socket: &mut TcpStream) -> std::io::Result<[u64; N]> {
    let mut bytes = vec![0u8; N * 8];
    socket.read_exact(&mut bytes)?;
    let mut words = [0u64; N];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    }
    Ok(words)
}

impl Worker {
    pub fn new(members: usize, index: usize, streams: usize, host: &str) -> Result<Worker, String> {
        if members == 0 || index >= members || streams == 0 {
            return Err(format!(
                "bad worker: member {index} of {members}, {streams} stream(s)"
            ));
        }
        let listener =
            TcpListener::bind((host, 0)).map_err(|e| format!("listen on {host}: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let address = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        let state = State {
            queues: (0..members * streams).map(|_| VecDeque::new()).collect(),
            next_stream: vec![0; members],
            connected: vec![false; members],
            ..State::default()
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            changed: Condvar::new(),
        });
        let accepted: Accepted = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
        let mut worker = Worker {
            members,
            index,
            streams,
            address,
            shared,
            accepted,
            sockets: Vec::new(),
            threads: Vec::new(),
            last_error: CString::default(),
        };
        let expected = index * streams; // every lower-index member dials `streams` times
        let (shared, accepted) = (worker.shared.clone(), worker.accepted.clone());
        worker.threads.push(thread::spawn(move || {
            accept_loop(listener, expected, members, streams, shared, accepted)
        }));
        Ok(worker)
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// Open (lower index) or take the accepted (higher index) connections to `member`,
    /// then start their threads. `timeout` bounds the whole call.
    pub fn connect(
        &mut self,
        member: usize,
        address: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        if member >= self.members || member == self.index {
            return Err(format!(
                "no member {member} to connect to (this is {})",
                self.index
            ));
        }
        if self.shared.lock().connected[member] {
            return Err(format!("member {member} is already connected"));
        }
        let deadline = Instant::now() + timeout;
        let mut sockets = Vec::with_capacity(self.streams);
        if self.index < member {
            let target: SocketAddr = address
                .to_socket_addrs()
                .map_err(|e| format!("member {member}'s address {address}: {e}"))?
                .next()
                .ok_or_else(|| {
                    format!("member {member}'s address {address} resolves to nothing")
                })?;
            for stream in 0..self.streams {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!(
                        "timed out connecting to member {member} at {address}"
                    ));
                }
                let mut socket = TcpStream::connect_timeout(&target, left)
                    .map_err(|e| format!("connect to member {member} at {address}: {e}"))?;
                socket.set_nodelay(true).map_err(|e| e.to_string())?;
                write_header(
                    &mut socket,
                    &[
                        MAGIC as u64 | (PROTOCOL as u64) << 32,
                        self.index as u64,
                        stream as u64,
                    ],
                )
                .map_err(|e| format!("hello to member {member}: {e}"))?;
                sockets.push(socket);
            }
        } else {
            let (lock, arrived) = &*self.accepted;
            let mut map = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(error) = self.shared.lock().failed.clone() {
                    return Err(error);
                }
                if (0..self.streams).all(|s| map.contains_key(&(member, s))) {
                    break;
                }
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!(
                        "timed out waiting for member {member}'s {} connection(s)",
                        self.streams
                    ));
                }
                map = arrived
                    .wait_timeout(map, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            for stream in 0..self.streams {
                sockets.push(map.remove(&(member, stream)).unwrap());
            }
        }
        for (stream, socket) in sockets.into_iter().enumerate() {
            let queue = member * self.streams + stream;
            let reader = socket.try_clone().map_err(|e| e.to_string())?;
            let writer = socket.try_clone().map_err(|e| e.to_string())?;
            let shared = self.shared.clone();
            self.threads.push(thread::spawn(move || {
                send_loop(writer, queue, member, shared)
            }));
            let shared = self.shared.clone();
            self.threads
                .push(thread::spawn(move || recv_loop(reader, member, shared)));
            self.sockets.push(socket);
        }
        self.shared.lock().connected[member] = true;
        Ok(())
    }

    fn check_peer(&self, state: &State, member: usize) -> Result<(), String> {
        if member >= self.members || member == self.index || !state.connected[member] {
            return Err(format!("member {member} is not connected"));
        }
        Ok(())
    }

    /// Queue `len` bytes at `ptr` for `member`. They must stay valid until wait() says the
    /// send finished: written to the socket, so the memory can be reused.
    pub fn send(&self, member: usize, ptr: usize, len: usize, tag: u64) -> Result<i64, String> {
        let mut state = self.shared.lock();
        self.check_peer(&state, member)?;
        if let Some(error) = &state.failed {
            return Err(error.clone());
        }
        state.next_id += 1;
        let id = state.next_id;
        let stream = state.next_stream[member];
        state.next_stream[member] = (stream + 1) % self.streams;
        state.queues[member * self.streams + stream].push_back(Job { id, tag, ptr, len });
        self.shared.changed.notify_all();
        Ok(id)
    }

    /// Post a receive of exactly `len` bytes into `ptr` for the message `tag` from `member`.
    pub fn recv(&self, member: usize, ptr: usize, len: usize, tag: u64) -> Result<i64, String> {
        let mut state = self.shared.lock();
        self.check_peer(&state, member)?;
        if let Some(error) = &state.failed {
            return Err(error.clone());
        }
        if state.posted.contains_key(&(member, tag)) {
            return Err(format!(
                "a receive for tag {tag:#x} from member {member} is already posted"
            ));
        }
        state.next_id += 1;
        let id = state.next_id;
        state.posted.insert((member, tag), Recv { id, ptr, len });
        self.shared.changed.notify_all();
        Ok(id)
    }

    /// Wait until every one of `ids` finished: 0, or TIMED_OUT, or PEER_FAILED.
    pub fn wait(&mut self, ids: &[i64], timeout: Duration) -> i64 {
        let deadline = Instant::now() + timeout;
        let shared = self.shared.clone();
        let mut state = shared.lock();
        loop {
            if ids.iter().all(|id| state.finished.contains(id)) {
                for id in ids {
                    state.finished.remove(id);
                }
                return 0;
            }
            if let Some(error) = state.failed.clone() {
                drop(state);
                self.set_error(error);
                return PEER_FAILED;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let missing = ids.iter().filter(|id| !state.finished.contains(id)).count();
                drop(state);
                self.set_error(format!(
                    "{missing} of {} operation(s) unfinished",
                    ids.len()
                ));
                return TIMED_OUT;
            }
            state = shared
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn set_error(&mut self, error: String) {
        self.last_error = CString::new(error.replace('\0', " ")).unwrap_or_default();
    }

    /// Stop every thread and close every connection. Returns false if a thread outlived
    /// `timeout` (it is left to end with the process).
    pub fn close(mut self, timeout: Duration) -> bool {
        {
            let mut state = self.shared.lock();
            state.closing = true;
            self.shared.changed.notify_all();
        }
        self.accepted.1.notify_all();
        for socket in &self.sockets {
            let _ = socket.shutdown(Shutdown::Both);
        }
        let deadline = Instant::now() + timeout;
        let mut clean = true;
        for handle in self.threads.drain(..) {
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            if handle.is_finished() {
                let _ = handle.join();
            } else {
                clean = false;
            }
        }
        clean
    }
}

fn accept_loop(
    listener: TcpListener,
    expected: usize,
    members: usize,
    streams: usize,
    shared: Arc<Shared>,
    accepted: Accepted,
) {
    let mut count = 0;
    while count < expected {
        if shared.lock().closing {
            return;
        }
        match listener.accept() {
            Ok((mut socket, _)) => {
                let hello = socket
                    .set_nonblocking(false)
                    .and_then(|_| socket.set_read_timeout(Some(Duration::from_secs(30))))
                    .and_then(|_| read_words::<3>(&mut socket))
                    .and_then(|words| socket.set_read_timeout(None).map(|_| words));
                let [first, member, stream] = match hello {
                    Ok(words) => words,
                    Err(_) => continue, // not one of ours (a port scan, a dead dialer)
                };
                let (member, stream) = (member as usize, stream as usize);
                if first as u32 != MAGIC || member >= members || stream >= streams {
                    continue;
                }
                if (first >> 32) as u32 != PROTOCOL {
                    shared.fail(format!(
                        "member {member} speaks protocol {} and this one {PROTOCOL}: rebuild \
                         gpubridge.multitcp from the same gpubridge on every leader",
                        first >> 32
                    ));
                    return;
                }
                let _ = socket.set_nodelay(true);
                let (lock, arrived) = &*accepted;
                let mut map = lock.lock().unwrap_or_else(|e| e.into_inner());
                if map.insert((member, stream), socket).is_none() {
                    count += 1;
                }
                arrived.notify_all();
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                shared.fail(format!("accepting bridge connections: {e}"));
                return;
            }
        }
    }
}

fn send_loop(mut socket: TcpStream, queue: usize, member: usize, shared: Arc<Shared>) {
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if state.closing || state.failed.is_some() {
                    return;
                }
                if let Some(job) = state.queues[queue].pop_front() {
                    break job;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        // SAFETY: the caller keeps the bytes valid until wait() reports this send finished.
        let bytes = unsafe { std::slice::from_raw_parts(job.ptr as *const u8, job.len) };
        let sent = write_header(&mut socket, &[job.tag, job.len as u64])
            .and_then(|_| socket.write_all(bytes));
        if let Err(e) = sent {
            shared.fail(format!("sending to member {member}: {e}"));
            return;
        }
        let mut state = shared.lock();
        state.finished.insert(job.id);
        shared.changed.notify_all();
    }
}

fn recv_loop(mut socket: TcpStream, member: usize, shared: Arc<Shared>) {
    loop {
        let [tag, len] = match read_words::<2>(&mut socket) {
            Ok(words) => words,
            Err(e) => {
                let closed = e.kind() == ErrorKind::UnexpectedEof;
                shared.fail(if closed {
                    format!("member {member} closed its bridge connection")
                } else {
                    format!("receiving from member {member}: {e}")
                });
                return;
            }
        };
        let recv = {
            let mut state = shared.lock();
            loop {
                if state.closing || state.failed.is_some() {
                    return;
                }
                if let Some(recv) = state.posted.remove(&(member, tag)) {
                    break recv;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        if recv.len as u64 != len {
            shared.fail(format!(
                "member {member} sent {len} bytes for tag {tag:#x}, posted for {} bytes",
                recv.len
            ));
            return;
        }
        // SAFETY: the caller keeps the buffer valid until wait() reports this receive done.
        let bytes = unsafe { std::slice::from_raw_parts_mut(recv.ptr as *mut u8, recv.len) };
        if let Err(e) = socket.read_exact(bytes) {
            shared.fail(format!("receiving from member {member}: {e}"));
            return;
        }
        let mut state = shared.lock();
        state.finished.insert(recv.id);
        shared.changed.notify_all();
    }
}

// ---- C interface ---------------------------------------------------------------------

fn write_c(text: &str, out: *mut c_char, size: usize) {
    if out.is_null() || size == 0 {
        return;
    }
    let bytes = text.as_bytes();
    let n = bytes.len().min(size - 1);
    // SAFETY: the caller passes a buffer of `size` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, n);
        *out.add(n) = 0;
    }
}

fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// # Safety
/// `handle` must come from gb_tcp_create and not be closed yet.
unsafe fn worker<'a>(handle: *mut c_void) -> &'a mut Worker {
    &mut *(handle as *mut Worker)
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    })
}

#[no_mangle]
pub extern "C" fn gb_tcp_protocol() -> u32 {
    PROTOCOL
}

/// A worker listening on `host` (an address of this machine), or null with the reason
/// in `error`.
///
/// # Safety
/// `host` NUL-terminated; `error` holds `error_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_create(
    members: c_int,
    index: c_int,
    streams: c_int,
    host: *const c_char,
    error: *mut c_char,
    error_size: usize,
) -> *mut c_void {
    guard(std::ptr::null_mut(), || {
        let host = CStr::from_ptr(host).to_string_lossy().into_owned();
        let counts = (
            usize::try_from(members),
            usize::try_from(index),
            usize::try_from(streams),
        );
        let result = match counts {
            (Ok(m), Ok(i), Ok(s)) => Worker::new(m, i, s, &host),
            _ => Err("negative member, index or stream count".to_string()),
        };
        match result {
            Ok(w) => Box::into_raw(Box::new(w)) as *mut c_void,
            Err(e) => {
                write_c(&e, error, error_size);
                std::ptr::null_mut()
            }
        }
    })
}

/// This worker's address ("host:port") into `out`, NUL-terminated.
///
/// # Safety
/// `handle` from gb_tcp_create; `out` holds `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_address(
    handle: *mut c_void,
    out: *mut c_char,
    size: usize,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        if w.address().len() + 1 > size {
            return FAILED as c_int;
        }
        write_c(w.address(), out, size);
        0
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `address` NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_connect(
    handle: *mut c_void,
    member: c_int,
    address: *const c_char,
    timeout: f64,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        let address = CStr::from_ptr(address).to_string_lossy().into_owned();
        match w.connect(member.max(0) as usize, &address, seconds(timeout)) {
            Ok(()) => 0,
            Err(e) => {
                w.set_error(e);
                FAILED as c_int
            }
        }
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `data` valid for `size` bytes until the send finishes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_send(
    handle: *mut c_void,
    member: c_int,
    data: *const c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        let w = worker(handle);
        w.send(member.max(0) as usize, data as usize, size, tag)
            .unwrap_or_else(|e| {
                w.set_error(e);
                FAILED
            })
    })
}

/// # Safety
/// `handle` from gb_tcp_create; `data` valid for `size` bytes until the receive finishes.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_recv(
    handle: *mut c_void,
    member: c_int,
    data: *mut c_void,
    size: usize,
    tag: u64,
) -> i64 {
    guard(FAILED, || {
        let w = worker(handle);
        w.recv(member.max(0) as usize, data as usize, size, tag)
            .unwrap_or_else(|e| {
                w.set_error(e);
                FAILED
            })
    })
}

/// 0 once every operation in `ids` finished; TIMED_OUT or PEER_FAILED (see last_error).
///
/// # Safety
/// `handle` from gb_tcp_create; `ids` holds `count` ids.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_wait(
    handle: *mut c_void,
    ids: *const i64,
    count: c_int,
    timeout: f64,
) -> c_int {
    guard(FAILED as c_int, || {
        let w = worker(handle);
        let ids = if count > 0 {
            std::slice::from_raw_parts(ids, count as usize)
        } else {
            &[]
        };
        w.wait(ids, seconds(timeout)) as c_int
    })
}

/// The last error's text, valid until the next call on this worker.
///
/// # Safety
/// `handle` from gb_tcp_create.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_last_error(handle: *mut c_void) -> *const c_char {
    guard(std::ptr::null(), || worker(handle).last_error.as_ptr())
}

/// Close the worker and free it: 0, or 1 if a thread didn't stop within `timeout`.
///
/// # Safety
/// `handle` from gb_tcp_create; not used again afterwards.
#[no_mangle]
pub unsafe extern "C" fn gb_tcp_close(handle: *mut c_void, timeout: f64) -> c_int {
    guard(FAILED as c_int, || {
        let w = Box::from_raw(handle as *mut Worker);
        if w.close(seconds(timeout)) {
            0
        } else {
            1
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(streams: usize) -> (Worker, Worker) {
        let mut a = Worker::new(2, 0, streams, "127.0.0.1").unwrap();
        let mut b = Worker::new(2, 1, streams, "127.0.0.1").unwrap();
        let (address_a, address_b) = (a.address().to_string(), b.address().to_string());
        let timeout = Duration::from_secs(10);
        let dialer = thread::spawn(move || {
            a.connect(1, &address_b, timeout).unwrap();
            a
        });
        b.connect(0, &address_a, timeout).unwrap();
        (dialer.join().unwrap(), b)
    }

    #[test]
    fn messages_arrive_intact_across_streams_in_any_order() {
        let (mut a, mut b) = pair(3);
        let sizes = [0usize, 1, 7, 4096, 1 << 20, 3_000_001];
        let data: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| (0..n).map(|j| (j * 31 + i * 7) as u8).collect())
            .collect();
        let mut out: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
        // Receives posted in reverse order, sends forward: matching is by tag.
        let mut ids_b: Vec<i64> = (0..sizes.len())
            .rev()
            .map(|i| {
                b.recv(0, out[i].as_mut_ptr() as usize, sizes[i], i as u64)
                    .unwrap()
            })
            .collect();
        let ids_a: Vec<i64> = (0..sizes.len())
            .map(|i| {
                a.send(1, data[i].as_ptr() as usize, sizes[i], i as u64)
                    .unwrap()
            })
            .collect();
        ids_b.reverse();
        assert_eq!(a.wait(&ids_a, Duration::from_secs(10)), 0);
        assert_eq!(b.wait(&ids_b, Duration::from_secs(10)), 0);
        assert_eq!(out, data);
        assert!(a.close(Duration::from_secs(5)) & b.close(Duration::from_secs(5)));
    }

    #[test]
    fn a_frame_waits_for_its_receive_to_be_posted() {
        let (mut a, mut b) = pair(1);
        let data = vec![5u8; 1000];
        let mut out = vec![0u8; 1000];
        let sent = a.send(1, data.as_ptr() as usize, data.len(), 9).unwrap();
        thread::sleep(Duration::from_millis(50));
        let got = b.recv(0, out.as_mut_ptr() as usize, out.len(), 9).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), 0);
        assert_eq!(a.wait(&[sent], Duration::from_secs(5)), 0);
        assert_eq!(out, data);
    }

    #[test]
    fn a_missing_message_times_out() {
        let (_a, mut b) = pair(2);
        let mut out = [0u8; 8];
        let id = b.recv(0, out.as_mut_ptr() as usize, 8, 1).unwrap();
        assert_eq!(b.wait(&[id], Duration::from_millis(100)), TIMED_OUT);
        assert!(b.last_error.to_str().unwrap().contains("unfinished"));
    }

    #[test]
    fn a_size_mismatch_or_a_closed_peer_fails_the_wait() {
        let (a, mut b) = pair(1);
        let data = [1u8; 16];
        let mut out = [0u8; 8];
        let got = b.recv(0, out.as_mut_ptr() as usize, 8, 3).unwrap();
        a.send(1, data.as_ptr() as usize, 16, 3).unwrap();
        assert_eq!(b.wait(&[got], Duration::from_secs(5)), PEER_FAILED);
        assert!(b.last_error.to_str().unwrap().contains("sent 16 bytes"));

        let (a, mut b) = pair(2);
        let id = b.recv(0, out.as_mut_ptr() as usize, 8, 4).unwrap();
        a.close(Duration::from_secs(5));
        assert_eq!(b.wait(&[id], Duration::from_secs(5)), PEER_FAILED);
        assert!(b
            .last_error
            .to_str()
            .unwrap()
            .contains("closed its bridge connection"));
    }

    #[test]
    fn bad_arguments_are_refused() {
        assert!(Worker::new(2, 2, 1, "127.0.0.1").is_err());
        assert!(Worker::new(2, 0, 0, "127.0.0.1").is_err());
        let (a, _b) = pair(1);
        assert!(a.send(0, 0, 0, 1).is_err()); // itself
        assert!(a.send(5, 0, 0, 1).is_err());
    }
}
