//! Deterministic software checks for every native progress policy. No real NIC/GPU.
mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{eventually, ib_fabric};
use gpubridge_verbs::{
    Access, AsyncEvent, AsyncEventKind, Call, Config, Context, Endpoint, Fault, InitAttr,
    MockFabric, Progress, SendOp, SendRequest, Sge, WaitError, WcStatus,
};

const POLICIES: [Progress; 3] = [Progress::Latency, Progress::Balanced, Progress::Economy];
const LONG: Duration = Duration::from_secs(5);

fn pair(policy: Progress) -> (Arc<MockFabric>, Endpoint, Endpoint) {
    let (f, na, nb) = ib_fabric();
    let c = Config {
        progress: policy,
        slot_bytes: 256,
        credits: 4,
        ctrl_credits: 2,
        send_slots: 4,
        unexpected_bytes: 0,
        ..Config::default()
    };
    let a = Endpoint::new(f.provider(na), 2, 0, c.clone()).unwrap();
    let b = Endpoint::new(f.provider(nb), 2, 1, c).unwrap();
    a.connect(1, &b.local_info()).unwrap();
    b.connect(0, &a.local_info()).unwrap();
    (f, a, b)
}

fn send(e: &Endpoint, peer: usize, data: &[u8], tag: u64) -> i64 {
    // SAFETY: tests keep each buffer until wait succeeds or abort/close cancels it.
    unsafe { e.send(peer, data.as_ptr() as usize, data.len() as u64, tag) }.unwrap()
}
fn recv(e: &Endpoint, peer: usize, data: &mut [u8], tag: u64) -> i64 {
    // SAFETY: as for send; outputs are inspected only after completion.
    unsafe { e.recv(peer, data.as_mut_ptr() as usize, data.len() as u64, tag) }.unwrap()
}

#[test]
fn every_policy_handles_empty_queues_spurious_events_boundary_completions_and_bursts() {
    for policy in POLICIES {
        let (f, a, b) = pair(policy);
        assert!(a
            .telemetry()
            .contains(&format!("\"progress\":{:?}", policy.name())));
        f.inject(Fault::SpuriousNotification { node: 1 });
        eventually(LONG, "spurious event consumed", || {
            b.counters().notifications > 0
        });
        assert_eq!(b.counters().completions, 0);
        f.inject(Fault::CompletionAtPollBoundary { node: 1 });
        let messages: Vec<Vec<u8>> = (0..24).map(|n| vec![n as u8; n * 97]).collect();
        let mut outputs: Vec<Vec<u8>> = messages.iter().map(|m| vec![0; m.len()]).collect();
        let mut sends = Vec::new();
        let mut recvs = Vec::new();
        for (tag, (message, output)) in messages.iter().zip(&mut outputs).enumerate() {
            recvs.push(recv(&b, 0, output, tag as u64));
            sends.push(send(&a, 1, message, tag as u64));
        }
        a.wait(&sends, LONG).unwrap();
        b.wait(&recvs, LONG).unwrap();
        assert_eq!(outputs, messages, "{policy:?}");
        assert!(
            f.log(1)
                .iter()
                .any(|l| l.contains("boundary completion exposed")),
            "{policy:?}"
        );
        assert_eq!(a.counters().bytes_sent, b.counters().bytes_received);
        // A queued spurious event at shutdown must be acked even if the CQ is empty.
        f.inject(Fault::SpuriousNotification { node: 0 });
        a.close(LONG).unwrap();
        b.close(LONG).unwrap();
        assert!(
            f.violations().is_empty(),
            "{policy:?}: {:?}",
            f.violations()
        );
        for node in [0, 1] {
            assert_eq!(f.live(node).total(), 0, "{policy:?}");
            assert!(f
                .log(node)
                .iter()
                .any(|l| l.starts_with("ibv_ack_cq_events")));
        }
    }
}

#[test]
fn every_policy_preserves_deadlines_local_cancellation_and_shutdown_with_pending_work() {
    for policy in POLICIES {
        let (f, a, b) = pair(policy);
        let mut out = [0u8; 4];
        let r = recv(&a, 1, &mut out, 77);
        assert!(matches!(
            a.wait(&[r], Duration::from_millis(5)),
            Err(WaitError::TimedOut(_))
        ));
        a.abort("local cancellation");
        assert!(a.operation_outcome(r).contains("\"status\":\"cancelled\""));
        a.close(LONG).unwrap();
        b.close(LONG).unwrap();
        assert_eq!(out, [0; 4]);
        assert!(
            f.violations().is_empty(),
            "{policy:?}: {:?}",
            f.violations()
        );
    }
}

#[test]
fn every_policy_reports_root_wc_failure_and_services_fatal_async_events() {
    for policy in POLICIES {
        let (f, a, b) = pair(policy);
        let mut out = [0u8; 4];
        let r = recv(&a, 1, &mut out, 88);
        f.inject(Fault::FailNextSend {
            node: 0,
            status: WcStatus::RetryExcErr,
        });
        let data = [1; 4];
        let s = send(&a, 1, &data, 99);
        assert!(matches!(a.wait(&[s, r], LONG), Err(WaitError::Failed(_))));
        assert!(a.operation_outcome(s).contains("IBV_WC_RETRY_EXC_ERR"));
        assert!(a.operation_outcome(r).contains("\"status\":\"cancelled\""));
        f.inject(Fault::AsyncEvent {
            node: 1,
            event: AsyncEvent {
                kind: AsyncEventKind::DeviceFatal,
                qp_num: None,
                port: None,
            },
        });
        eventually(LONG, "fatal async event", || b.failure().is_some());
        assert!(b.failure().unwrap().contains("DeviceFatal"));
        a.close(LONG).unwrap();
        b.close(LONG).unwrap();
        assert!(
            f.violations().is_empty(),
            "{policy:?}: {:?}",
            f.violations()
        );
    }
}

#[test]
fn every_policy_refuses_queued_sends_to_a_disconnected_peer_without_a_deadline_wait() {
    for policy in POLICIES {
        let (f, a, b) = pair(policy);
        let first = vec![1u8; 8192];
        let second = [2; 4];
        let s1 = send(&a, 1, &first, 1);
        eventually(LONG, "send stalled on credits", || {
            a.counters().credit_stalls > 0
        });
        let s2 = send(&a, 1, &second, 2);
        b.close(LONG).unwrap();
        for s in [s1, s2] {
            match a.wait(&[s], LONG) {
                Err(WaitError::Failed(why)) => assert!(why.contains("closed its end"), "{why}"),
                other => panic!("{policy:?}: {other:?}"),
            }
        }
        let data = [1; 4];
        // SAFETY: the buffer stays live; posting must be rejected before any use.
        let error = unsafe { a.send(1, data.as_ptr() as usize, 4, 1) }.unwrap_err();
        assert!(error.to_string().contains("closed its end"));
        a.close(LONG).unwrap();
        assert!(
            f.violations().is_empty(),
            "{policy:?}: {:?}",
            f.violations()
        );
    }
}

#[test]
fn every_policy_attributes_a_native_post_failure_to_its_send() {
    for policy in POLICIES {
        for rejected in [0, 1] {
            let (f, a, b) = pair(policy);
            let mut out = [0u8; 4];
            let r = recv(&a, 1, &mut out, 88);
            f.inject(Fault::FailCall {
                node: 0,
                call: Call::PostSend,
                nth: rejected,
                errno: 5,
            });
            let data = vec![1; 2048];
            let s = send(&a, 1, &data, 99);
            assert!(matches!(a.wait(&[s, r], LONG), Err(WaitError::Failed(_))));
            let cause = a.operation_outcome(s);
            assert!(cause.contains("\"status\":\"error\""), "{cause}");
            assert!(cause.contains("posting sends failed"), "{cause}");
            assert!(
                cause.contains("\"wc_status\":null"),
                "no WC exists for a rejected post"
            );
            assert!(a.operation_outcome(r).contains("\"status\":\"cancelled\""));
            a.close(LONG).unwrap();
            b.close(LONG).unwrap();
            assert!(
                f.violations().is_empty(),
                "{policy:?}: {:?}",
                f.violations()
            );
        }
    }
}

#[test]
fn policies_are_local_settings_with_economy_default_and_affinity_only_for_latency() {
    assert_eq!(Config::default().progress, Progress::Economy);
    for p in POLICIES {
        assert_eq!(
            Config::parse(&format!("progress={}", p.name()))
                .unwrap()
                .progress,
            p
        );
    }
    for bad in [
        "progress=unknown",
        "progress_cpu=1",
        "progress=balanced\nprogress_cpu=0",
        "progress_cpu=-1",
    ] {
        assert!(Config::parse(bad).is_err(), "{bad}");
    }
    let (f, node, _) = ib_fabric();
    let config = Config {
        progress: Progress::Latency,
        progress_cpu: Some(u32::MAX),
        ..Config::default()
    };
    assert!(Endpoint::new(f.provider(node), 2, 0, config).is_err());
    assert_eq!(f.live(node).total(), 0);
}

#[test]
fn mock_notification_is_one_shot_not_retroactive_and_requires_rearm() {
    let (f, node, _) = ib_fabric();
    let ctx = Context::open(f.provider(node), "mlx0").unwrap();
    let channel = ctx.create_channel().unwrap();
    let cq = ctx.create_cq_on(16, &channel).unwrap();
    let pd = ctx.alloc_pd().unwrap();
    let qp = pd.create_qp(&cq, &cq, common::caps(4, 4, 1)).unwrap();
    qp.to_init(&InitAttr {
        port: 1,
        pkey_index: 0,
        access: Access::NONE,
    })
    .unwrap();
    let dst = pd.register_buffer(4, Access::LOCAL_WRITE).unwrap();
    qp.post_recv(1, &[Sge::new(&dst, 0, 4)]).unwrap();
    // Transition to Err generates a flush CQE before notification is armed.
    qp.to_error().unwrap();
    cq.req_notify().unwrap();
    assert!(!channel.wait(Duration::ZERO).unwrap().completion);
    assert_eq!(cq.poll(4).unwrap().len(), 1);
    qp.close(LONG).unwrap();
    drop(cq);
    drop(channel);
    assert!(!f
        .log(node)
        .iter()
        .any(|l| l.starts_with("ibv_ack_cq_events")));
}

#[test]
fn mock_notifications_require_rearm_after_each_event() {
    let (f, na, nb) = ib_fabric();
    let open = |node| {
        let ctx = Context::open(f.provider(node), "mlx0").unwrap();
        let channel = ctx.create_channel().unwrap();
        let cq = ctx.create_cq_on(16, &channel).unwrap();
        let pd = ctx.alloc_pd().unwrap();
        let qp = pd.create_qp(&cq, &cq, common::caps(4, 4, 1)).unwrap();
        qp.to_init(&InitAttr {
            port: 1,
            pkey_index: 0,
            access: Access::NONE,
        })
        .unwrap();
        (
            common::Side {
                node,
                ctx,
                pd,
                cq,
                qp,
            },
            channel,
        )
    };
    let (a, _channel_a) = open(na);
    let (b, channel_b) = open(nb);
    common::connect(&a, &b);
    let src = a.pd.register_buffer(4, Access::NONE).unwrap();
    let dst = b.pd.register_buffer(4, Access::LOCAL_WRITE).unwrap();
    for n in 0..3 {
        if n != 1 {
            b.cq.req_notify().unwrap();
        }
        b.qp.post_recv(n, &[Sge::new(&dst, 0, 4)]).unwrap();
        a.qp.post_send(&SendRequest {
            tag: n,
            op: SendOp::Send,
            sges: &[Sge::new(&src, 0, 4)],
        })
        .unwrap();
        common::wait(&a.cq, 1, 10);
        assert_eq!(channel_b.wait(Duration::ZERO).unwrap().completion, n != 1);
        common::wait(&b.cq, 1, 10);
    }
    drop(a);
    drop(b);
    assert!(f.violations().is_empty(), "{:?}", f.violations());
    assert_eq!(
        f.log(nb)
            .iter()
            .filter(|l| l.starts_with("ibv_ack_cq_events"))
            .count(),
        2
    );
}
