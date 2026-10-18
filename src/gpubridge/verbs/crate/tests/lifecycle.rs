//! Resource lifetimes on the mock provider: the queue pair state machine, release order,
//! cleanup after a failed setup, and memory held by outstanding work.

mod common;

use std::time::Duration;

use common::*;
use gpubridge_verbs::{
    Access, Address, Call, Context, Error, Fault, GlobalRoute, InitAttr, MockDevice, MockFabric,
    MockLimits, MockPort, Mtu, PortState, QpState, SendOp, SendRequest, Sge, WcStatus,
};

fn kinds(log: &[String]) -> Vec<&str> {
    log.iter()
        .map(|l| l.split(' ').next().unwrap())
        .filter(|v| *v != "ibv_reg_mr" && *v != "ibv_dereg_mr")
        .collect()
}

#[test]
fn the_state_machine_goes_reset_init_rtr_rts_and_refuses_skips() {
    let (fabric, na, nb) = ib_fabric();
    let ctx = Context::open(fabric.provider(na), "mlx0").unwrap();
    let pd = ctx.alloc_pd().unwrap();
    let cq = ctx.create_cq(64).unwrap();
    let qp = pd.create_qp(&cq, &cq, caps(4, 4, 1)).unwrap();
    let b = open(&fabric, nb, "mlx0", caps(4, 4, 1));
    assert_eq!(qp.state(), QpState::Reset);
    let to_b = rtr(&b, PSN_B);
    let wanted = |r: Result<(), Error>, state| {
        assert!(
            matches!(r, Err(Error::InvalidState { state: s, .. }) if s == state),
            "{r:?}"
        );
    };
    wanted(qp.to_rtr(&to_b), QpState::Reset);
    wanted(qp.to_rts(&rts(PSN_A, 7)), QpState::Reset);
    let buf = pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    wanted(
        qp.post_recv(1, &[Sge::new(&buf, 0, 8)]).map(|_| ()),
        QpState::Reset,
    );
    let init = InitAttr {
        port: 1,
        pkey_index: 0,
        access: Access::REMOTE_WRITE,
    };
    // A port the device doesn't have.
    for port in [0, 2] {
        assert!(matches!(
            qp.to_init(&InitAttr { port, ..init }),
            Err(Error::InvalidArgument(_))
        ));
    }
    qp.to_init(&init).unwrap();
    wanted(qp.to_init(&init), QpState::Init);
    wanted(qp.to_rts(&rts(PSN_A, 7)), QpState::Init);
    // Receives can be posted from Init, sends only in Rts.
    qp.post_recv(1, &[Sge::new(&buf, 0, 8)]).unwrap();
    let send = SendRequest {
        tag: 1,
        op: SendOp::Send,
        sges: &[],
    };
    wanted(qp.post_send(&send).map(|_| ()), QpState::Init);
    qp.to_rtr(&to_b).unwrap();
    wanted(qp.post_send(&send).map(|_| ()), QpState::Rtr);
    qp.to_rts(&rts(PSN_A, 7)).unwrap();
    assert_eq!(qp.state(), QpState::Rts);
    // Reset with work outstanding would lose its completions.
    assert!(matches!(qp.reset(), Err(Error::Busy(_))));
    qp.to_error().unwrap();
    assert_eq!(wait(&cq, 1, 2)[0].status, WcStatus::WrFlushErr);
    qp.reset().unwrap();
    qp.to_init(&init).unwrap();
    assert!(fabric.violations().is_empty());
}

#[test]
fn transition_attributes_are_checked_before_verbs_sees_them() {
    let (fabric, na, nb) = ib_fabric();
    let a = open(&fabric, na, "mlx0", caps(4, 4, 1));
    let b = open(&fabric, nb, "mlx0", caps(4, 4, 1));
    let good = rtr(&b, PSN_B);
    let grh = GlobalRoute {
        dgid: b.ctx.query_gid(1, 0).unwrap().gid,
        sgid_index: 0,
        hop_limit: 64,
        traffic_class: 0,
        flow_label: 0,
    };
    let bad = [
        gpubridge_verbs::RtrAttr {
            dest_qp_num: 1 << 24,
            ..good
        },
        gpubridge_verbs::RtrAttr {
            dest_qp_num: 1,
            ..good
        },
        gpubridge_verbs::RtrAttr {
            rq_psn: 1 << 24,
            ..good
        },
        gpubridge_verbs::RtrAttr {
            min_rnr_timer: 32,
            ..good
        },
        gpubridge_verbs::RtrAttr { sl: 16, ..good },
        gpubridge_verbs::RtrAttr {
            max_dest_rd_atomic: 200,
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Ib { lid: 0, grh: None },
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Ib {
                lid: 0xc000,
                grh: None,
            },
            ..good
        },
        // A RoCE address on an InfiniBand port.
        gpubridge_verbs::RtrAttr {
            address: Address::Roce(grh),
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Ib {
                lid: 2,
                grh: Some(GlobalRoute {
                    sgid_index: 5,
                    ..grh
                }),
            },
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Ib {
                lid: 2,
                grh: Some(GlobalRoute {
                    dgid: Default::default(),
                    ..grh
                }),
            },
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Ib {
                lid: 2,
                grh: Some(GlobalRoute {
                    flow_label: 1 << 20,
                    ..grh
                }),
            },
            ..good
        },
    ];
    for (i, attr) in bad.iter().enumerate() {
        assert!(
            matches!(a.qp.to_rtr(attr), Err(Error::InvalidArgument(_))),
            "case {i}"
        );
        assert_eq!(a.qp.state(), QpState::Init, "case {i}");
    }
    // An InfiniBand route with a valid GRH is fine.
    a.qp.to_rtr(&gpubridge_verbs::RtrAttr {
        address: Address::Ib {
            lid: 2,
            grh: Some(grh),
        },
        ..good
    })
    .unwrap();
    let ok = rts(PSN_A, 7);
    let bad = [
        gpubridge_verbs::RtsAttr {
            sq_psn: 1 << 24,
            ..ok
        },
        gpubridge_verbs::RtsAttr { timeout: 32, ..ok },
        gpubridge_verbs::RtsAttr { retry_cnt: 8, ..ok },
        gpubridge_verbs::RtsAttr { rnr_retry: 8, ..ok },
        gpubridge_verbs::RtsAttr {
            max_rd_atomic: 200,
            ..ok
        },
    ];
    for (i, attr) in bad.iter().enumerate() {
        assert!(
            matches!(a.qp.to_rts(attr), Err(Error::InvalidArgument(_))),
            "case {i}"
        );
        assert_eq!(a.qp.state(), QpState::Rtr);
    }
    a.qp.to_rts(&ok).unwrap();
}

#[test]
fn roce_ports_need_a_global_route_and_a_fitting_mtu() {
    let (fabric, na, nb) = roce_fabric();
    let a = open(&fabric, na, "roce0", caps(4, 4, 1));
    let b = open(&fabric, nb, "roce0", caps(4, 4, 1));
    let good = rtr(&b, PSN_B);
    let Address::Roce(grh) = good.address else {
        panic!("RoCE addresses by GID")
    };
    let bad = [
        gpubridge_verbs::RtrAttr {
            address: Address::Ib { lid: 2, grh: None },
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Roce(GlobalRoute {
                sgid_index: 2,
                ..grh
            }),
            ..good
        },
        gpubridge_verbs::RtrAttr {
            address: Address::Roce(GlobalRoute {
                hop_limit: 0,
                ..grh
            }),
            ..good
        },
        // The mock RoCE port's active MTU is 1024.
        gpubridge_verbs::RtrAttr {
            path_mtu: Mtu::Mtu4096,
            ..good
        },
    ];
    for (i, attr) in bad.iter().enumerate() {
        assert!(
            matches!(a.qp.to_rtr(attr), Err(Error::InvalidArgument(_))),
            "case {i}"
        );
    }
    // A port that went down since Init.
    fabric.set_port_state(na, 0, 1, PortState::Down);
    assert!(matches!(a.qp.to_rtr(&good), Err(Error::InvalidArgument(_))));
    fabric.set_port_state(na, 0, 1, PortState::Active);
    a.qp.to_rtr(&good).unwrap();
}

#[test]
fn handles_dropped_in_any_order_release_in_the_right_one() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let mr = a.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    let Side {
        ctx, pd, cq, qp, ..
    } = a;
    // Parents first: nothing is released until its children are.
    drop(ctx);
    drop(pd);
    drop(cq);
    assert_eq!(fabric.live(na).total(), 5);
    drop(mr);
    drop(qp);
    let log = fabric.log(na);
    assert_eq!(
        kinds(&log),
        [
            "ibv_open_device",
            "ibv_alloc_pd",
            "ibv_create_cq",
            "ibv_create_qp",
            "ibv_destroy_qp",
            "ibv_destroy_cq",
            "ibv_dealloc_pd",
            "ibv_close_device"
        ]
    );
    // The region went before its protection domain.
    let at = |verb: &str| log.iter().position(|l| l.starts_with(verb)).unwrap();
    assert!(at("ibv_dereg_mr") < at("ibv_dealloc_pd"));
    assert_eq!(fabric.live(na).total(), 0);
    drop(b);
    assert_eq!(fabric.live(nb).total(), 0);
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_setup_that_fails_halfway_releases_what_it_made() {
    for (call, nth) in [
        (Call::QueryDevice, 0),
        (Call::AllocPd, 0),
        (Call::CreateCq, 0),
        (Call::CreateQp, 0),
        (Call::ModifyQp, 0),
        (Call::RegMr, 0),
    ] {
        let (fabric, na, _) = ib_fabric();
        fabric.inject(Fault::FailCall {
            node: na,
            call,
            nth,
            errno: 12,
        });
        let setup = || -> Result<(), Error> {
            let ctx = Context::open(fabric.provider(na), "mlx0")?;
            let pd = ctx.alloc_pd()?;
            let cq = ctx.create_cq(16)?;
            let qp = pd.create_qp(&cq, &cq, caps(4, 4, 1))?;
            qp.to_init(&InitAttr {
                port: 1,
                pkey_index: 0,
                access: Access::NONE,
            })?;
            let _mr = pd.register_buffer(64, Access::LOCAL_WRITE)?;
            Ok(())
        };
        let err = setup().unwrap_err();
        assert_eq!(err.errno(), Some(12), "{call:?}");
        assert_eq!(
            fabric.live(na).total(),
            0,
            "{call:?} left {:?}",
            fabric.live(na)
        );
        assert!(fabric.violations().is_empty(), "{call:?}");
    }
}

#[test]
fn device_limits_and_completion_queue_capacity_are_enforced() {
    let fabric = MockFabric::new();
    let mut dev = MockDevice::new("mlx0", vec![MockPort::infiniband(1)]);
    dev.limits = MockLimits {
        max_qp: 1,
        max_mr: 2,
        ..MockLimits::default()
    };
    let na = fabric.add_node(vec![dev]);
    let ctx = Context::open(fabric.provider(na), "mlx0").unwrap();
    let pd = ctx.alloc_pd().unwrap();
    let cq = ctx.create_cq(16).unwrap();
    // 8 sends and 8 receives fill a queue of 16; one more of either doesn't fit.
    assert!(matches!(
        pd.create_qp(&cq, &cq, caps(8, 9, 1)),
        Err(Error::Exhausted(_))
    ));
    let qp = pd.create_qp(&cq, &cq, caps(8, 8, 1)).unwrap();
    // The device allows one queue pair: ENOMEM, and the capacity it reserved comes back.
    let cq2 = ctx.create_cq(16).unwrap();
    assert_eq!(
        pd.create_qp(&cq2, &cq2, caps(1, 1, 1)).unwrap_err().errno(),
        Some(12)
    );
    drop(qp);
    let qp = pd.create_qp(&cq, &cq, caps(8, 8, 1)).unwrap();
    let _m1 = pd.register_buffer(8, Access::NONE).unwrap();
    let _m2 = pd.register_buffer(8, Access::NONE).unwrap();
    assert_eq!(
        pd.register_buffer(8, Access::NONE).unwrap_err().errno(),
        Some(12)
    );
    // Bad requests are refused before verbs: empty regions, remote write without local.
    assert!(matches!(
        pd.register_buffer(0, Access::NONE),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        pd.register_buffer(8, Access::REMOTE_WRITE),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(ctx.create_cq(0), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        pd.create_qp(&cq, &cq, caps(0, 1, 1)),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        pd.create_qp(&cq, &cq, caps(1, 1, 17)),
        Err(Error::InvalidArgument(_))
    ));
    drop(qp);
    assert!(fabric.violations().is_empty());
}

#[test]
fn a_region_stays_registered_while_posted_work_uses_it() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let dst = b.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    let keep = dst.clone();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 64)]).unwrap();
    assert_eq!(dst.inflight(), 1);
    // The device may write into it: copies in and out are refused.
    assert!(matches!(keep.write(0, b"x"), Err(Error::Busy(_))));
    assert!(matches!(keep.read(0, &mut [0u8; 1]), Err(Error::Busy(_))));
    drop(dst);
    drop(keep);
    // Every handle the caller had is gone; the receive still holds the registration.
    assert_eq!(fabric.live(nb).mrs, 1);
    let src = a.pd.register_buffer(8, Access::LOCAL_WRITE).unwrap();
    a.qp.post_send(&SendRequest {
        tag: 1,
        op: SendOp::Send,
        sges: &[Sge::new(&src, 0, 8)],
    })
    .unwrap();
    wait(&b.cq, 1, 4);
    assert_eq!(fabric.live(nb).mrs, 0);
    assert!(fabric.violations().is_empty());
}

#[test]
fn dropping_a_queue_pair_with_work_outstanding_drains_it_first() {
    let (fabric, na, nb) = ib_fabric();
    let (a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    // Completions arrive late: the drain has to wait for them.
    fabric.inject(Fault::CompletionDelay {
        node: nb,
        ticks: 50,
    });
    let dst = b.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    for i in 0..3 {
        b.qp.post_recv(i, &[Sge::new(&dst, 0, 64)]).unwrap();
    }
    drop(dst);
    let Side { qp, cq, .. } = b;
    let before = fabric.tick();
    qp.close(Duration::from_secs(10)).unwrap();
    assert!(
        fabric.tick() - before >= 50,
        "the drain waited for the late completions"
    );
    let log = fabric.log(nb);
    let at = |verb: &str| log.iter().position(|l| l.starts_with(verb)).unwrap();
    // Flushed completions released the region; the queue pair went after its work.
    assert!(at("ibv_dereg_mr") < at("ibv_destroy_qp"));
    assert_eq!(cq.outstanding(), 0);
    assert!(
        cq.poll(8).unwrap().is_empty(),
        "a closed queue pair's completions are dropped"
    );
    assert!(fabric.violations().is_empty());
    drop(a);
}

#[test]
fn a_drain_that_runs_out_of_time_leaks_the_memory_instead_of_freeing_it() {
    let (fabric, na, nb) = ib_fabric();
    let (_a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    // The flush completions never become visible.
    fabric.inject(Fault::CompletionDelay {
        node: nb,
        ticks: u64::MAX / 2,
    });
    let dst = b.pd.register_buffer(64, Access::LOCAL_WRITE).unwrap();
    b.qp.post_recv(1, &[Sge::new(&dst, 0, 64)]).unwrap();
    let Side {
        ctx, pd, cq, qp, ..
    } = b;
    let err = qp.close(Duration::from_millis(20)).unwrap_err();
    assert!(
        matches!(err, Error::DrainTimeout { outstanding: 1, .. }),
        "{err:?}"
    );
    let stats = ctx.stats();
    assert_eq!((stats.leaked_work, stats.leaked_regions), (1, 1));
    drop(dst);
    drop((ctx, pd, cq));
    // The region (and so its protection domain and context) stay: never deregistered.
    let live = fabric.live(nb);
    assert_eq!((live.mrs, live.pds, live.contexts, live.qps), (1, 1, 1, 0));
    assert!(fabric.violations().is_empty());
}

#[test]
fn the_mock_catches_a_caller_that_breaks_the_rules() {
    // The tests above rely on the mock's violation checks; make sure they fire.
    let (fabric, na, nb) = ib_fabric();
    let (_a, b) = pair(&fabric, na, nb, "mlx0", caps(4, 4, 1));
    let p = fabric.provider(nb);
    let mut buf = vec![0u8; 64];
    // Through the raw provider: register, post a receive, deregister while it's posted.
    let ctx = p.open_device("mlx0").unwrap();
    let pd = p.alloc_pd(ctx).unwrap();
    let (cq, _) = p.create_cq(ctx, 8).unwrap();
    let qp = p.create_rc_qp(pd, cq, cq, &caps(1, 1, 1)).unwrap();
    p.modify_qp(
        qp.handle,
        &gpubridge_verbs::Transition::Init(InitAttr {
            port: 1,
            pkey_index: 0,
            access: Access::NONE,
        }),
    )
    .unwrap();
    // SAFETY: `buf` outlives the registration (deregistered below, before buf drops).
    let mr = unsafe { p.reg_mr(pd, buf.as_mut_ptr(), buf.len(), Access::LOCAL_WRITE) }.unwrap();
    p.post_recv(
        qp.handle,
        1,
        &[gpubridge_verbs::RawSge {
            addr: buf.as_ptr() as u64,
            len: 64,
            lkey: mr.lkey,
        }],
    )
    .unwrap();
    p.dereg_mr(mr.handle).unwrap();
    assert!(p.destroy_cq(cq).is_err());
    p.destroy_qp(qp.handle).unwrap();
    let v = fabric.violations();
    assert!(
        v.iter()
            .any(|v| v.contains("deregistered while 1 work requests")),
        "{v:?}"
    );
    assert!(
        v.iter()
            .any(|v| v.contains("destroyed while a queue pair uses it")),
        "{v:?}"
    );
    assert!(
        v.iter()
            .any(|v| v.contains("destroyed with 1 work requests")),
        "{v:?}"
    );
    drop(b);
}

#[test]
fn the_library_reports_what_it_was_built_with() {
    let mut buf = [0 as std::ffi::c_char; 128];
    // SAFETY: buf has 128 writable bytes.
    assert_eq!(
        unsafe { gpubridge_verbs::gb_verbs_build_info(buf.as_mut_ptr(), 128) },
        0
    );
    // SAFETY: on success it wrote a NUL-terminated string.
    let text = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        text.starts_with("gpubridge-verbs 0.1.0 providers=mock"),
        "{text}"
    );
    assert_eq!(cfg!(feature = "ibverbs"), text.ends_with(",libibverbs"));
    // SAFETY: as above, with a buffer too small.
    assert_eq!(
        unsafe { gpubridge_verbs::gb_verbs_build_info(buf.as_mut_ptr(), 8) },
        -1
    );
}
