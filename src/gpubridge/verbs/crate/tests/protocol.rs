//! The wire protocol between endpoints on the mock fabric: what arrives, in what order,
//! under pressure, and how failures, deadlines and teardown behave. Each endpoint runs
//! its own completion thread, as it does over libibverbs.

mod common;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use common::{eventually, garbage, ib_fabric, roce_gid, RawMember};
use gpubridge_verbs::{
    AsyncEvent, AsyncEventKind, Config, Endpoint, Fault, ForkStatus, MockDevice, MockFabric,
    MockPort, Mtu, NodeId, PortState, WaitError, WcStatus,
};

const LONG: Duration = Duration::from_secs(20);

/// Small slots (208 bytes of payload) so messages are many chunks.
fn small() -> Config {
    Config {
        slot_bytes: 256,
        credits: 4,
        ctrl_credits: 2,
        send_slots: 4,
        ..Config::default()
    }
}

fn endpoints(fabric: &Arc<MockFabric>, nodes: &[NodeId], config: &Config) -> Vec<Endpoint> {
    let n = nodes.len();
    let eps: Vec<Endpoint> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| Endpoint::new(fabric.provider(node), n, i, config.clone()).unwrap())
        .collect();
    let infos: Vec<String> = eps.iter().map(Endpoint::local_info).collect();
    for (i, ep) in eps.iter().enumerate() {
        for (j, info) in infos.iter().enumerate() {
            if i != j {
                ep.connect(j, info).unwrap();
            }
        }
    }
    eps
}

fn pair(config: &Config) -> (Arc<MockFabric>, Endpoint, Endpoint) {
    let (fabric, a, b) = ib_fabric();
    let mut eps = endpoints(&fabric, &[a, b], config);
    let b = eps.pop().unwrap();
    let a = eps.pop().unwrap();
    (fabric, a, b)
}

fn send(ep: &Endpoint, to: usize, data: &[u8], tag: u64) -> i64 {
    // SAFETY: every test waits for (or aborts) its operations before the data goes.
    unsafe { ep.send(to, data.as_ptr() as usize, data.len() as u64, tag) }.unwrap()
}

fn recv(ep: &Endpoint, from: usize, out: &mut [u8], tag: u64) -> i64 {
    // SAFETY: as for send.
    unsafe { ep.recv(from, out.as_mut_ptr() as usize, out.len() as u64, tag) }.unwrap()
}

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn messages_of_every_size_arrive_intact_in_chunks() {
    let (fabric, a, b) = pair(&small());
    let payload = 256 - 48;
    let sizes = [
        0,
        1,
        payload - 1,
        payload,
        payload + 1,
        3 * payload + 7,
        1 << 16,
    ];
    for (i, &n) in sizes.iter().enumerate() {
        let data = pattern(n, i as u8);
        let mut out = vec![0u8; n];
        // Receive posted first for even sizes, after the send for odd ones.
        let (r, s) = if i % 2 == 0 {
            let r = recv(&b, 0, &mut out, i as u64);
            (r, send(&a, 1, &data, i as u64))
        } else {
            let s = send(&a, 1, &data, i as u64);
            thread::sleep(Duration::from_millis(5));
            (recv(&b, 0, &mut out, i as u64), s)
        };
        a.wait(&[s], LONG).unwrap();
        b.wait(&[r], LONG).unwrap();
        assert!(out == data, "size {n}");
    }
    let (ca, cb) = (a.counters(), b.counters());
    let chunks: u64 = sizes
        .iter()
        .map(|&n| (n as u64).div_ceil(payload as u64).max(1))
        .sum();
    assert_eq!(ca.chunks_sent, chunks);
    assert_eq!(cb.chunks_received, chunks);
    assert_eq!(cb.bytes_received, sizes.iter().sum::<usize>() as u64);
    assert!(
        cb.notifications > 0,
        "the completion thread waited on its channel"
    );
    assert_eq!(cb.unexpected_bytes, 0);
    a.close(LONG).unwrap();
    b.close(LONG).unwrap();
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn both_directions_and_many_tags_at_once() {
    let (fabric, a, b) = pair(&small());
    let msgs: Vec<Vec<u8>> = (0..12).map(|i| pattern(100 * i + 3, i as u8)).collect();
    let mut at_a: Vec<Vec<u8>> = msgs.iter().map(|m| vec![0; m.len()]).collect();
    let mut at_b: Vec<Vec<u8>> = msgs.iter().map(|m| vec![0; m.len()]).collect();
    let mut ida = Vec::new();
    let mut idb = Vec::new();
    // Receives in reverse tag order, sends forward: matching is by tag.
    for (i, (oa, ob)) in at_a.iter_mut().zip(at_b.iter_mut()).enumerate().rev() {
        ida.push(recv(&a, 1, oa, i as u64));
        idb.push(recv(&b, 0, ob, i as u64));
    }
    for (i, m) in msgs.iter().enumerate() {
        ida.push(send(&a, 1, m, i as u64));
        idb.push(send(&b, 0, m, i as u64));
    }
    a.wait(&ida, LONG).unwrap();
    b.wait(&idb, LONG).unwrap();
    assert_eq!(at_a, msgs);
    assert_eq!(at_b, msgs);
    drop((a, b));
    assert!(fabric.violations().is_empty());
}

#[test]
fn messages_with_one_tag_arrive_in_the_order_sent() {
    let (_fabric, a, b) = pair(&small());
    let msgs: Vec<Vec<u8>> = (0..6).map(|i| vec![i as u8; 500]).collect();
    let ids: Vec<i64> = msgs.iter().map(|m| send(&a, 1, m, 7)).collect();
    for m in &msgs {
        let mut out = vec![0u8; 500];
        let r = recv(&b, 0, &mut out, 7);
        b.wait(&[r], LONG).unwrap();
        assert_eq!(&out, m);
    }
    a.wait(&ids, LONG).unwrap();
}

#[test]
fn credits_bound_what_is_in_flight_and_stalled_senders_resume() {
    let config = Config {
        credits: 2,
        send_slots: 2,
        ctrl_credits: 1,
        ..small()
    };
    let (fabric, a, b) = pair(&config);
    let data = pattern(50_000, 1); // 241 chunks through 2 credits
    let mut out = vec![0u8; data.len()];
    let s = send(&a, 1, &data, 1);
    thread::sleep(Duration::from_millis(20));
    let r = recv(&b, 0, &mut out, 1);
    a.wait(&[s], LONG).unwrap();
    b.wait(&[r], LONG).unwrap();
    assert!(out == data);
    let (ca, cb) = (a.counters(), b.counters());
    assert!(ca.credit_stalls > 0, "{ca:?}");
    assert!(
        cb.control_sent > 0,
        "credits came back on control messages: {cb:?}"
    );
    assert!(fabric.violations().is_empty());
}

#[test]
fn unclaimed_messages_are_buffered_up_to_the_limit_then_held_in_their_slots() {
    let config = Config {
        unexpected_bytes: 1000,
        ..small()
    };
    let (fabric, a, b) = pair(&config);
    let msgs: Vec<Vec<u8>> = (0..10).map(|i| pattern(400, i as u8)).collect();
    let ids: Vec<i64> = msgs
        .iter()
        .enumerate()
        .map(|(i, m)| send(&a, 1, m, i as u64))
        .collect();
    // Two fit the buffer; the rest wait in receive slots, then at the sender (credits).
    let start = Instant::now();
    while b.counters().held_chunks == 0 && start.elapsed() < LONG {
        thread::sleep(Duration::from_millis(2));
    }
    let held = b.counters();
    assert!(
        held.unexpected_bytes <= 1000 && held.held_chunks > 0,
        "{held:?}"
    );
    let mut outs: Vec<Vec<u8>> = msgs.iter().map(|m| vec![0; m.len()]).collect();
    let rids: Vec<i64> = outs
        .iter_mut()
        .enumerate()
        .rev()
        .map(|(i, o)| recv(&b, 0, o, i as u64))
        .collect();
    b.wait(&rids, LONG).unwrap();
    a.wait(&ids, LONG).unwrap();
    assert_eq!(outs, msgs);
    let c = b.counters();
    assert_eq!(c.unexpected_bytes, 0);
    assert!(c.unexpected_peak <= 1000);
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_receive_of_the_wrong_size_fails_with_both_sizes() {
    let (_fabric, a, b) = pair(&small());
    let data = [1u8; 16];
    let mut out = [0u8; 8];
    let r = recv(&b, 0, &mut out, 77);
    send(&a, 1, &data, 77);
    match b.wait(&[r], LONG) {
        Err(WaitError::Failed(e)) => {
            assert!(
                e.contains("sent 16 bytes for tag 0x4d, posted for 8 bytes"),
                "{e}"
            )
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(out, [0u8; 8]);
}

#[test]
fn too_many_pending_operations_are_refused() {
    let config = Config {
        max_pending: 3,
        ..small()
    };
    let (_fabric, a, _b) = pair(&config);
    let mut bufs = [[0u8; 4]; 4];
    for (i, buf) in bufs.iter_mut().take(3).enumerate() {
        recv(&a, 1, buf, i as u64);
    }
    // SAFETY: refused, so never used.
    let fourth = unsafe { a.recv(1, bufs[3].as_mut_ptr() as usize, 4, 3) };
    assert!(fourth
        .unwrap_err()
        .to_string()
        .contains("operation queue full"));
}

#[test]
fn a_wait_past_its_deadline_times_out_and_a_late_message_never_lands() {
    let (fabric, a, b) = pair(&small());
    let mut out = vec![0u8; 300];
    let r = recv(&b, 0, &mut out, 5);
    let start = Instant::now();
    match b.wait(&[r], Duration::from_millis(200)) {
        Err(WaitError::TimedOut(e)) => {
            assert!(e.contains("timed out: 1 of 1 operations unfinished"), "{e}")
        }
        other => panic!("{other:?}"),
    }
    assert!(start.elapsed() < Duration::from_secs(5));
    b.abort("timed out");
    // The sender is late: its message must not reach the cancelled receive's memory.
    let data = vec![9u8; 300];
    let s = send(&a, 1, &data, 5);
    match a.wait(&[s], LONG) {
        Err(WaitError::Failed(e)) => assert!(e.contains("retry budget"), "{e}"),
        other => panic!("{other:?}"),
    }
    assert!(out.iter().all(|&x| x == 0));
    // A failed endpoint refuses new work.
    // SAFETY: refused.
    let again = unsafe { b.recv(0, out.as_mut_ptr() as usize, 300, 6) };
    assert!(again.unwrap_err().to_string().contains("already failed"));
    a.close(LONG).ok();
    b.close(LONG).ok();
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_late_completion_after_an_abort_touches_nothing() {
    let (fabric, a, b) = pair(&small());
    // Completions on b become visible long after the data landed in b's slot.
    fabric.inject(Fault::CompletionDelay {
        node: 1,
        ticks: 200,
    });
    let mut out = vec![0u8; 100];
    let r = recv(&b, 0, &mut out, 1);
    let data = vec![7u8; 100];
    let s = send(&a, 1, &data, 1);
    a.wait(&[s], LONG).unwrap();
    assert!(b.wait(&[r], Duration::from_millis(10)).is_err());
    b.abort("gave up");
    thread::sleep(Duration::from_millis(400)); // the completion surfaces now
    assert!(out.iter().all(|&x| x == 0));
    b.close(LONG).ok();
    drop(a);
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_lost_peer_fails_the_operations_on_it() {
    let (fabric, a, b) = pair(&small());
    fabric.kill(1);
    let data = vec![1u8; 10];
    let s = send(&a, 1, &data, 1);
    match a.wait(&[s], LONG) {
        Err(WaitError::Failed(e)) => {
            assert!(e.contains("member 1") && e.contains("RetryExcErr"), "{e}")
        }
        other => panic!("{other:?}"),
    }
    let t = a.telemetry();
    assert!(t.contains("\"lost\":\"member 1"), "{t}");
    a.close(LONG).ok();
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
    // The dead node can't drain: its receives are leaked with their slots, not freed.
    let e = b.close(Duration::from_millis(200)).unwrap_err().to_string();
    assert!(e.contains("leaked, not released"), "{e}");
}

#[test]
fn a_failed_completion_and_a_fatal_async_event_propagate() {
    let (fabric, a, b) = pair(&small());
    fabric.inject(Fault::FailNextSend {
        node: 0,
        status: WcStatus::RemAccessErr,
    });
    let data = [1u8; 4];
    let s = send(&a, 1, &data, 1);
    match a.wait(&[s], LONG) {
        Err(WaitError::Failed(e)) => assert!(e.contains("RemAccessErr"), "{e}"),
        other => panic!("{other:?}"),
    }
    fabric.inject(Fault::AsyncEvent {
        node: 1,
        event: AsyncEvent {
            kind: AsyncEventKind::DeviceFatal,
            qp_num: None,
            port: None,
        },
    });
    let start = Instant::now();
    while b.failure().is_none() && start.elapsed() < LONG {
        thread::sleep(Duration::from_millis(2));
    }
    assert!(b.failure().unwrap().contains("DeviceFatal"));
    assert!(b.telemetry().contains("\"state\":\"failed\""));
}

#[test]
fn a_port_going_down_fails_the_endpoint() {
    let (fabric, a, _b) = pair(&small());
    fabric.set_port_state(0, 0, 1, PortState::Down);
    let start = Instant::now();
    while a.failure().is_none() && start.elapsed() < LONG {
        thread::sleep(Duration::from_millis(2));
    }
    assert!(a.failure().unwrap().contains("PortErr on port 1"));
}

#[test]
fn closing_with_work_outstanding_drains_and_releases_everything() {
    let (fabric, na, nb) = ib_fabric();
    let eps = endpoints(&fabric, &[na, nb], &small());
    // Sends that won't move for a while, and receives nobody will meet.
    fabric.inject(Fault::SendDelay {
        node: na,
        ticks: 1_000_000,
    });
    let data = vec![3u8; 2000];
    let mut out = vec![0u8; 64];
    send(&eps[0], 1, &data, 1);
    recv(&eps[0], 1, &mut out, 2);
    recv(&eps[1], 0, &mut out, 3);
    let start = Instant::now();
    for ep in eps {
        ep.close(Duration::from_secs(5)).unwrap();
    }
    assert!(start.elapsed() < Duration::from_secs(10));
    assert_eq!(fabric.live(na).total(), 0, "{:?}", fabric.live(na));
    assert_eq!(fabric.live(nb).total(), 0);
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

#[test]
fn a_peer_that_said_goodbye_left_its_messages_and_no_failure() {
    let (fabric, a, b) = pair(&small());
    let data = pattern(1000, 3);
    let s = send(&a, 1, &data, 4);
    a.wait(&[s], LONG).unwrap();
    a.close(LONG).unwrap();
    thread::sleep(Duration::from_millis(50));
    let mut out = vec![0u8; 1000];
    let r = recv(&b, 0, &mut out, 4);
    b.wait(&[r], LONG).unwrap();
    assert_eq!(out, data);
    assert!(b.failure().is_none());
    assert!(b.telemetry().contains("\"closed_by_peer\":true"));
    let mut more = [0u8; 4];
    // A receive it can never meet now fails at once rather than at the deadline.
    let r = recv(&b, 0, &mut more, 5);
    assert!(matches!(b.wait(&[r], LONG), Err(WaitError::Failed(e)) if e.contains("closed")));
    b.close(LONG).ok();
    assert!(fabric.violations().is_empty());
}

/// Member 1 breaks the protocol while member 0 is part-way through a plain send to
/// member 2 and a plain receive from it. Once member 0's wait has reported the failure,
/// nothing reads the send's buffer or writes the receive's again, though member 2's
/// credits come back and its chunks keep arriving (docs: "Failure is cancellation").
#[test]
fn a_failure_cancels_plain_operations_so_nothing_touches_their_memory_after_it() {
    let fabric = MockFabric::new();
    let nodes: Vec<NodeId> = (1..=3)
        .map(|lid| {
            fabric.add_node(vec![MockDevice::new(
                "mlx0",
                vec![MockPort::infiniband(lid)],
            )])
        })
        .collect();
    // Nothing unexpected is buffered: a message nobody receives yet stays in its slots,
    // so its sender runs out of credits part-way.
    let config = Config {
        unexpected_bytes: 0,
        ..small()
    };
    let a = Endpoint::new(fabric.provider(nodes[0]), 3, 0, config.clone()).unwrap();
    let c = Endpoint::new(fabric.provider(nodes[2]), 3, 2, config.clone()).unwrap();
    let b = RawMember::new(&fabric, nodes[1], 3, 0, &config);
    b.connect(&a.local_info(), 1);
    a.connect(1, &b.info).unwrap();
    a.connect(2, &c.local_info()).unwrap();
    c.connect(0, &a.local_info()).unwrap();

    // 0 -> 2: stalls after its credits' worth of chunks (2 holds them).
    let payload = 256 - 48;
    let outgoing = pattern(40 * payload, 1);
    let s = send(&a, 2, &outgoing, 1);
    eventually(LONG, "the send stalls on credits", || {
        a.counters().credit_stalls > 0
    });
    // 2 -> 0: into a posted receive, each batch of chunks 200 ticks after the last.
    fabric.inject(Fault::SendDelay {
        node: nodes[2],
        ticks: 200,
    });
    let mut incoming = vec![0u8; 40 * payload];
    let r = recv(&a, 2, &mut incoming, 2);
    let theirs = pattern(40 * payload, 2);
    let t = send(&c, 0, &theirs, 2);
    eventually(LONG, "the first chunks land", || {
        a.counters().chunks_received > 0
    });

    // 1 sends a message without the protocol's magic number.
    assert_eq!(b.send(&garbage()), WcStatus::Success);
    for id in [s, r] {
        match a.wait(&[id], LONG) {
            Err(WaitError::Failed(e)) => assert!(e.contains("member 1 broke the protocol"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    let at_failure = a.counters();
    let received_then = incoming.clone();
    assert!(
        at_failure.chunks_received < 40,
        "the receive was part-way when it failed"
    );

    // 2 receives what 0 sent so far: its credits go back to 0, which (before this fix)
    // copied the rest of the send's buffer into slots and sent it.
    let mut drained = vec![0u8; outgoing.len()];
    let u = recv(&c, 0, &mut drained, 1);
    // And 2's chunks keep landing at 0 (each batch 200 ticks after the last).
    eventually(LONG, "more of member 2's chunks arrive", || {
        a.counters().chunks_received > at_failure.chunks_received
    });
    thread::sleep(Duration::from_millis(300));
    let after = a.counters();
    assert_eq!(after.chunks_sent, at_failure.chunks_sent, "no slot copies");
    assert_eq!(
        after.copied_bytes, at_failure.copied_bytes,
        "no copies at all"
    );
    assert_eq!(after.credit_stalls, at_failure.credit_stalls);
    assert!(a.telemetry().contains("\"state\":\"failed\""));
    a.abort("test over");
    c.abort("test over");
    // Nothing more was written into the receive's buffer after the failure.
    assert!(c.wait(&[t, u], Duration::from_millis(10)).is_err());
    a.close(LONG).ok();
    c.close(LONG).ok();
    assert!(incoming == received_then);
    drop(b);
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

/// Sends queued to a member, or part-way, fail as soon as it says goodbye, with that
/// reason, rather than at their deadline.
#[test]
fn queued_sends_to_a_member_that_said_goodbye_fail_at_once_with_its_reason() {
    let config = Config {
        unexpected_bytes: 0,
        ..small()
    };
    let (fabric, a, b) = pair(&config);
    let first = pattern(20 * 208, 1);
    let second = pattern(100, 2);
    let s1 = send(&a, 1, &first, 1);
    eventually(LONG, "the first send stalls on credits", || {
        a.counters().credit_stalls > 0
    });
    let s2 = send(&a, 1, &second, 2);
    let start = Instant::now();
    b.close(LONG).ok();
    for id in [s1, s2] {
        match a.wait(&[id], Duration::from_secs(60)) {
            Err(WaitError::Failed(e)) => {
                assert!(e.contains("member 1 closed its end of the bridge"), "{e}")
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    a.close(LONG).ok();
    assert!(fabric.violations().is_empty(), "{:?}", fabric.violations());
}

/// Fork safety is checked when an endpoint opens, before it registers anything:
/// `ibv_fork_init` where the kernel (or an old libibverbs) can't promise it, a failure
/// with the reason if that fails, and the outcome in telemetry.
#[test]
fn fork_safety_is_checked_at_open_before_anything_is_registered() {
    for (status, init, inits, expect) in [
        (
            ForkStatus::Unneeded,
            None,
            0,
            Ok("unneeded: the kernel keeps"),
        ),
        (
            ForkStatus::Enabled,
            None,
            0,
            Ok("enabled: ibv_fork_init already ran"),
        ),
        (
            ForkStatus::Disabled,
            None,
            1,
            Ok("enabled by gpubridge (ibv_fork_init)"),
        ),
        (
            ForkStatus::Unknown,
            None,
            1,
            Ok("enabled by gpubridge (ibv_fork_init)"),
        ),
        (
            ForkStatus::Disabled,
            Some(22),
            1,
            Err("ibv_fork_init failed"),
        ),
    ] {
        let (fabric, na, _nb) = ib_fabric();
        fabric.set_fork(status, init);
        let opened = Endpoint::new(fabric.provider(na), 2, 0, small());
        assert_eq!(fabric.fork_inits(), inits, "{status:?}");
        let log = fabric.log(na);
        if inits > 0 {
            let first_reg = log.iter().position(|l| l.starts_with("ibv_reg_mr"));
            assert_eq!(
                log.first().map(String::as_str),
                Some("ibv_fork_init"),
                "{log:?}"
            );
            assert!(first_reg.map_or(true, |i| i > 0));
        }
        match (opened, expect) {
            (Ok(ep), Ok(text)) => {
                let t = ep.telemetry();
                assert!(t.contains(&format!("\"fork_safety\":\"{text}")), "{t}");
            }
            (Err(e), Err(text)) => {
                let e = e.to_string();
                assert!(e.contains(text) && e.contains("RDMAV_FORK_SAFE"), "{e}");
                assert!(!log.iter().any(|l| l.starts_with("ibv_reg_mr")), "{log:?}");
                assert_eq!(fabric.live(na).total(), 0);
            }
            (got, want) => panic!("{status:?}: {:?} where {want:?}", got.map(|_| ())),
        }
    }
}

#[test]
fn three_members_pass_a_ring() {
    let fabric = MockFabric::new();
    let nodes: Vec<NodeId> = (1..=3)
        .map(|lid| {
            fabric.add_node(vec![MockDevice::new(
                "mlx0",
                vec![MockPort::infiniband(lid)],
            )])
        })
        .collect();
    let eps = endpoints(&fabric, &nodes, &small());
    let data: Vec<Vec<u8>> = (0..3).map(|i| pattern(1234, i as u8)).collect();
    let mut outs = vec![vec![0u8; 1234]; 3];
    let mut ids = vec![Vec::new(); 3];
    for (i, out) in outs.iter_mut().enumerate() {
        ids[i].push(recv(&eps[i], (i + 2) % 3, out, 9));
    }
    for i in 0..3 {
        ids[i].push(send(&eps[i], (i + 1) % 3, &data[i], 9));
    }
    for (ep, ids) in eps.iter().zip(&ids) {
        ep.wait(ids, LONG).unwrap();
    }
    for i in 0..3 {
        assert_eq!(outs[i], data[(i + 2) % 3]);
    }
}

#[test]
fn mismatched_ends_are_refused_at_connect() {
    // A different protocol version.
    let (fabric, a, b) = ib_fabric();
    let ea = Endpoint::new(fabric.provider(a), 2, 0, small()).unwrap();
    let eb = Endpoint::new(fabric.provider(b), 2, 1, small()).unwrap();
    let older = eb.local_info().replacen("gbverbs 2;", "gbverbs 1;", 1);
    let e = ea.connect(1, &older).unwrap_err().to_string();
    assert!(e.contains("protocol 1, this side 2"), "{e}");
    // Different settings.
    let ec = Endpoint::new(
        fabric.provider(b),
        2,
        1,
        Config {
            credits: 8,
            ..small()
        },
    )
    .unwrap();
    let e = ea.connect(1, &ec.local_info()).unwrap_err().to_string();
    assert!(e.contains("same settings"), "{e}");
    // InfiniBand and RoCE.
    let fabric = MockFabric::new();
    let ib = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(1)])]);
    let eth = fabric.add_node(vec![MockDevice::new(
        "roce0",
        vec![MockPort::roce(roce_gid(2))],
    )]);
    let e_ib = Endpoint::new(fabric.provider(ib), 2, 0, small()).unwrap();
    let e_eth = Endpoint::new(fabric.provider(eth), 2, 1, small()).unwrap();
    let e = e_ib
        .connect(1, &e_eth.local_info())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("incompatible link layers: InfiniBand here"),
        "{e}"
    );
    let e = e_eth
        .connect(0, &e_ib.local_info())
        .unwrap_err()
        .to_string();
    assert!(e.contains("Ethernet (RoCE) here"), "{e}");
}

#[test]
fn the_path_mtu_is_the_smaller_active_mtu() {
    let fabric = MockFabric::new();
    let mut slow = MockPort::infiniband(2);
    slow.active_mtu = Mtu::Mtu1024;
    let a = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(1)])]);
    let b = fabric.add_node(vec![MockDevice::new("mlx0", vec![slow])]);
    let eps = endpoints(&fabric, &[a, b], &small());
    assert!(eps[0].telemetry().contains("\"path_mtu\":1024"));
    assert!(eps[0].telemetry().contains("\"active_mtu\":4096"));
    let data = pattern(5000, 1);
    let mut out = vec![0u8; 5000];
    let r = recv(&eps[1], 0, &mut out, 1);
    let s = send(&eps[0], 1, &data, 1);
    eps[0].wait(&[s], LONG).unwrap();
    eps[1].wait(&[r], LONG).unwrap();
    assert_eq!(out, data);
}

#[test]
fn devices_ports_and_gids_are_chosen_by_the_documented_rules() {
    let fabric = MockFabric::new();
    let mut down = MockPort::infiniband(5);
    down.state = PortState::Down;
    let node = fabric.add_node(vec![
        MockDevice::new("mlx_b", vec![MockPort::roce(roce_gid(9))]),
        MockDevice::new("mlx_a", vec![down]),
    ]);
    let p = fabric.provider(node);
    // Automatic: mlx_a first by name, but its port is down; then mlx_b's RoCE v2 GID.
    let ep = Endpoint::new(p.clone(), 2, 0, small()).unwrap();
    let t = ep.telemetry();
    assert!(
        t.contains("\"device\":\"mlx_b\"") && t.contains("\"gid_index\":1"),
        "{t}"
    );
    assert!(
        t.contains("\"gid_type\":\"RoceV2\"") && t.contains("\"mock\":true"),
        "{t}"
    );
    assert!(t.contains("MOCK"), "{t}");
    drop(ep);
    // Explicit: the named device is used or the endpoint fails; never another one.
    let named = |device: &str| Config {
        device: Some(device.into()),
        ..small()
    };
    let e = Endpoint::new(p.clone(), 2, 0, named("mlx_a"))
        .err()
        .unwrap();
    assert!(
        e.to_string().contains("mlx_a port 1: Down, not Active"),
        "{e}"
    );
    let e = Endpoint::new(p.clone(), 2, 0, named("mlx_c"))
        .err()
        .unwrap();
    assert!(e.to_string().contains("there are mlx_a, mlx_b"), "{e}");
    let v1 = Config {
        gid_index: Some(0),
        ..named("mlx_b")
    };
    let ep = Endpoint::new(p, 2, 0, v1).unwrap();
    assert!(ep.telemetry().contains("\"gid_type\":\"RoceV1\""));
}

#[test]
fn settings_parse_and_bad_ones_are_refused() {
    let c =
        Config::parse("device=mlx5_0\nport=1\nslot_bytes=4096\ncredits=8\nprovider=x\n").unwrap();
    assert_eq!(c.device.as_deref(), Some("mlx5_0"));
    assert_eq!((c.port, c.slot_bytes, c.credits), (Some(1), 4096, 8));
    for bad in [
        "slot_bytes=100",
        "credits=0",
        "nonsense=1",
        "port",
        "retry_cnt=9",
    ] {
        assert!(Config::parse(bad).is_err(), "{bad}");
    }
}
