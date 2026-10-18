//! Work on connected queue pairs over the mock fabric: what moves, what completes, and
//! how failures surface.

mod common;

use common::*;
use gpubridge_verbs::{
    Access, Call, Error, Fault, MockDevice, MockFabric, MockPort, QpState, RemoteTarget, SendOp,
    SendRequest, Sge, WcStatus, WorkKind,
};

fn send<'a>(tag: u64, sges: &'a [Sge<'a>]) -> SendRequest<'a> {
    SendRequest {
        tag,
        op: SendOp::Send,
        sges,
    }
}

#[test]
fn a_send_lands_in_the_posted_receive() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(8, 8, 4));
    let src = a.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    src.write(0, b"hello, verbs").unwrap();
    b.qp.post_recv(7, &[Sge::new(&dst, 0, 64)]).unwrap();
    a.qp.post_send(&send(9, &[Sge::new(&src, 0, 12)])).unwrap();
    let sent = wait(&a.cq, 1, 4);
    let got = wait(&b.cq, 1, 4);
    assert_eq!(
        (sent[0].tag, sent[0].kind, sent[0].status),
        (9, WorkKind::Send, WcStatus::Success)
    );
    assert_eq!(
        (got[0].tag, got[0].kind, got[0].byte_len),
        (7, WorkKind::Recv, 12)
    );
    let mut out = [0u8; 12];
    dst.read(0, &mut out).unwrap();
    assert_eq!(&out, b"hello, verbs");
    assert!(fabric.violations().is_empty());
}

#[test]
fn empty_and_scattered_messages_arrive_in_order() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(16, 16, 4));
    let size = 1 << 20;
    let src = a.pd.register_buffer(size, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(size, Access::LOCAL_WRITE).unwrap();
    let pattern: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    src.write(0, &pattern).unwrap();
    let quarter = (size / 4) as u32;
    // An empty receive and send first, then 1 MiB gathered from four pieces into two.
    b.qp.post_recv(1, &[]).unwrap();
    b.qp.post_recv(
        2,
        &[
            Sge::new(&dst, 0, 2 * quarter),
            Sge::new(&dst, size / 2, 2 * quarter),
        ],
    )
    .unwrap();
    a.qp.post_send(&send(1, &[])).unwrap();
    let pieces: Vec<Sge> = (0..4)
        .map(|i| Sge::new(&src, i * size / 4, quarter))
        .collect();
    a.qp.post_send(&send(2, &pieces)).unwrap();
    let got = wait(&b.cq, 2, 4);
    assert_eq!(
        got.iter().map(|c| (c.tag, c.byte_len)).collect::<Vec<_>>(),
        [(1, 0), (2, size as u32)]
    );
    wait(&a.cq, 2, 4);
    let mut out = vec![0u8; size];
    dst.read(0, &mut out).unwrap();
    assert!(out == pattern);
    assert!(fabric.violations().is_empty());
}

#[test]
fn rdma_writes_and_immediate_data() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(8, 8, 1));
    let src = a.pd.register_buffer(32, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(64, remote_access()).unwrap();
    src.write(0, &[5u8; 32]).unwrap();
    let target = RemoteTarget {
        addr: dst.addr() as u64 + 16,
        rkey: dst.rkey(),
    };
    // A plain write needs no receive; one with immediate data consumes one.
    a.qp.post_send(&SendRequest {
        tag: 1,
        op: SendOp::Write(target),
        sges: &[Sge::new(&src, 0, 16)],
    })
    .unwrap();
    b.qp.post_recv(3, &[]).unwrap();
    a.qp.post_send(&SendRequest {
        tag: 2,
        op: SendOp::WriteWithImm(
            RemoteTarget {
                addr: target.addr + 16,
                ..target
            },
            0xabcd,
        ),
        sges: &[Sge::new(&src, 16, 16)],
    })
    .unwrap();
    let sent = wait(&a.cq, 2, 4);
    assert!(sent.iter().all(|c| c.ok()));
    let got = wait(&b.cq, 1, 4);
    assert_eq!(
        (got[0].tag, got[0].imm, got[0].byte_len),
        (3, Some(0xabcd), 16)
    );
    let mut out = [0u8; 64];
    dst.read(0, &mut out).unwrap();
    assert_eq!(&out[..16], &[0u8; 16]);
    assert_eq!(&out[16..48], &[5u8; 32]);
    // Send with immediate data.
    let rbuf = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(4, &[Sge::new(&rbuf, 0, 8)]).unwrap();
    a.qp.post_send(&SendRequest {
        tag: 5,
        op: SendOp::SendWithImm(42),
        sges: &[Sge::new(&src, 0, 8)],
    })
    .unwrap();
    assert_eq!(wait(&b.cq, 1, 4)[0].imm, Some(42));
    assert!(fabric.violations().is_empty());
}

#[test]
fn roce_queue_pairs_connect_by_gid() {
    let (fabric, na, nb) = roce_fabric();
    let (a, b) = pair(&fabric, na, nb, "roce0", caps(4, 4, 1));
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 8)]).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    assert!(wait(&b.cq, 1, 4)[0].ok());
}

fn first_send_fails_with(a: &Side, b: &Side, polls: usize) -> WcStatus {
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 8)]).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    a.qp.post_send(&send(2, &[Sge::new(&src, 0, 8)])).unwrap();
    let done = wait(&a.cq, 2, polls);
    // The second send never ran: flushed once the first failed.
    assert_eq!(done[1].status, WcStatus::WrFlushErr);
    assert_eq!(a.qp.state(), QpState::Err);
    done[0].status
}

#[test]
fn mismatched_connections_run_out_of_retries() {
    // A PSN that doesn't match the peer's, a wrong destination queue pair, a wrong LID.
    for case in 0..3 {
        let (fabric, na, nb) = ib_fabric();
        let a = open(&fabric, na, "mlx0", caps(4, 4, 1));
        let b = open(&fabric, nb, "mlx0", caps(4, 4, 1));
        let mut to_b = rtr(&b, PSN_B);
        let mut a_psn = PSN_A;
        match case {
            0 => a_psn += 1,
            1 => to_b.dest_qp_num = 0x7777,
            _ => {
                to_b.address = gpubridge_verbs::Address::Ib { lid: 9, grh: None };
            }
        }
        a.qp.to_rtr(&to_b).unwrap();
        b.qp.to_rtr(&rtr(&a, PSN_A)).unwrap();
        a.qp.to_rts(&rts(a_psn, 7)).unwrap();
        b.qp.to_rts(&rts(PSN_B, 7)).unwrap();
        fabric.set_retry(5, 1);
        assert_eq!(
            first_send_fails_with(&a, &b, 20),
            WcStatus::RetryExcErr,
            "case {case}"
        );
        assert!(fabric.violations().is_empty());
    }
}

#[test]
fn a_lost_peer_fails_the_send_and_cleanup_still_runs() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    fabric.set_retry(3, 1);
    fabric.kill(nb);
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    assert_eq!(wait(&a.cq, 1, 10)[0].status, WcStatus::RetryExcErr);
    assert_eq!(a.qp.state(), QpState::Err);
    drop((a, b, src));
    assert_eq!(fabric.live(na).total(), 0);
    assert_eq!(fabric.live(nb).total(), 0);
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_receiver_without_receives_runs_out_of_rnr_retries() {
    let (fabric, na, nb) = ib_fabric();
    let a = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let b = open(&fabric, nb, "mlx0", caps(4, 4, 1));
    a.qp.to_rtr(&rtr(&b, PSN_B)).unwrap();
    b.qp.to_rtr(&rtr(&a, PSN_A)).unwrap();
    a.qp.to_rts(&rts(PSN_A, 1)).unwrap();
    b.qp.to_rts(&rts(PSN_B, 7)).unwrap();
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    assert_eq!(wait(&a.cq, 1, 20)[0].status, WcStatus::RnrRetryExcErr);
}

#[test]
fn infinite_rnr_retry_waits_for_a_late_receive() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    fabric.step(100);
    assert!(a.cq.poll(1).unwrap().is_empty());
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 8)]).unwrap();
    assert!(wait(&a.cq, 1, 4)[0].ok());
}

#[test]
fn a_message_longer_than_the_receive_fails_both_sides() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let src = a.pd.register_buffer(16, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(16, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 8)]).unwrap();
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 16)])).unwrap();
    assert_eq!(wait(&b.cq, 1, 4)[0].status, WcStatus::LocLenErr);
    assert_eq!(wait(&a.cq, 1, 4)[0].status, WcStatus::RemInvReqErr);
    assert_eq!((a.qp.state(), b.qp.state()), (QpState::Err, QpState::Err));
}

#[test]
fn a_write_without_remote_access_is_refused_by_the_peer() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&SendRequest {
        tag: 1,
        op: SendOp::Write(RemoteTarget {
            addr: dst.addr() as u64,
            rkey: dst.rkey(),
        }),
        sges: &[Sge::new(&src, 0, 8)],
    })
    .unwrap();
    assert_eq!(wait(&a.cq, 1, 4)[0].status, WcStatus::RemAccessErr);
}

#[test]
fn an_injected_send_failure_moves_the_queue_pair_to_error() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    fabric.inject(Fault::FailNextSend {
        node: na,
        status: WcStatus::GeneralErr,
    });
    assert_eq!(first_send_fails_with(&a, &b, 4), WcStatus::GeneralErr);
    // The device moved it to Err: sends are refused until it is reset.
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    assert!(matches!(
        a.qp.post_send(&send(3, &[Sge::new(&src, 0, 8)])),
        Err(Error::InvalidState {
            state: QpState::Err,
            ..
        })
    ));
}

#[test]
fn sends_beyond_the_queue_pairs_capacity_are_refused_until_some_complete() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(2, 4, 1));
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(32, Access::LOCAL_WRITE).unwrap();
    for i in 0..3 {
        b.qp.post_recv(i, &[Sge::new(&dst, 8 * i as usize, 8)])
            .unwrap();
    }
    a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)])).unwrap();
    a.qp.post_send(&send(2, &[Sge::new(&src, 0, 8)])).unwrap();
    assert_eq!(
        a.qp.post_send(&send(3, &[Sge::new(&src, 0, 8)]))
            .unwrap_err(),
        Error::QueueFull {
            queue: "send",
            limit: 2
        }
    );
    assert_eq!(a.qp.outstanding(), (2, 0));
    wait(&a.cq, 2, 4);
    assert_eq!(a.qp.outstanding(), (0, 0));
    a.qp.post_send(&send(3, &[Sge::new(&src, 0, 8)])).unwrap();
    wait(&a.cq, 1, 4);
}

#[test]
fn a_post_the_provider_refuses_holds_nothing() {
    let (fabric, na, nb) = ib_fabric();
    let (a, _b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    fabric.inject(Fault::FailCall {
        node: na,
        call: Call::PostSend,
        nth: 0,
        errno: 12,
    });
    let err =
        a.qp.post_send(&send(1, &[Sge::new(&src, 0, 8)]))
            .unwrap_err();
    assert_eq!(err.errno(), Some(12));
    assert_eq!(a.qp.outstanding(), (0, 0));
    assert_eq!(src.inflight(), 0);
    assert_eq!(a.cq.outstanding(), 0);
}

#[test]
fn bad_scatter_gather_entries_are_refused_before_posting() {
    let (fabric, na, nb) = ib_fabric();
    let (a, _b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 2));
    let src = a.pd.register_buffer(8, Access::NONE).unwrap();
    let other_pd = a.ctx.alloc_pd().unwrap();
    let foreign = other_pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let cases: [Vec<Sge>; 3] = [
        vec![Sge::new(&src, 4, 8)],
        vec![Sge::new(&foreign, 0, 8)],
        vec![
            Sge::new(&src, 0, 1),
            Sge::new(&src, 1, 1),
            Sge::new(&src, 2, 1),
        ],
    ];
    for sges in &cases {
        assert!(matches!(
            a.qp.post_send(&send(1, sges)),
            Err(Error::InvalidArgument(_))
        ));
    }
    // Receiving needs local write access.
    assert!(matches!(
        a.qp.post_recv(1, &[Sge::new(&src, 0, 8)]),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(a.qp.outstanding(), (0, 0));
}

#[test]
fn a_shared_completion_queue_keeps_other_queue_pairs_completions_across_a_drain() {
    let fabric = MockFabric::new();
    let na = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(1)])]);
    let nb = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(2)])]);
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    // A second queue pair on A's completion queue.
    let qp2 = a.pd.create_qp(&a.cq, &a.cq, caps(4, 4, 1)).unwrap();
    qp2.to_init(&gpubridge_verbs::InitAttr {
        port: 1,
        pkey_index: 0,
        access: Access::NONE,
    })
    .unwrap();
    let buf = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    let dst = b.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 8)]).unwrap();
    a.qp.post_send(&send(77, &[Sge::new(&buf, 0, 8)])).unwrap();
    fabric.step(1);
    // Dropping qp2 with a receive posted drains it through the shared queue, which also
    // takes the send's completion: it must still reach the caller.
    qp2.post_recv(5, &[Sge::new(&buf, 0, 8)]).unwrap();
    drop(qp2);
    let got = wait(&a.cq, 1, 4);
    assert_eq!((got[0].tag, got[0].qp_num), (77, a.qp.qp_num()));
    assert!(a.cq.poll(8).unwrap().is_empty());
    assert_eq!(a.ctx.stats().discarded_completions, 1);
    assert!(fabric.violations().is_empty());
}
