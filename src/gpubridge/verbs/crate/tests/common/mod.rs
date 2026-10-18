//! Two mock nodes with connected queue pairs.

#![allow(dead_code)]

use std::sync::Arc;

use gpubridge_verbs::{
    Access, Address, Completion, CompletionQueue, Context, Gid, GlobalRoute, InitAttr, MockDevice,
    MockFabric, MockPort, Mtu, NodeId, ProtectionDomain, QpCaps, QueuePair, RtrAttr, RtsAttr,
};

pub const PSN_A: u32 = 0x10;
pub const PSN_B: u32 = 0x20;

pub struct Side {
    pub node: NodeId,
    pub ctx: Context,
    pub pd: ProtectionDomain,
    pub cq: CompletionQueue,
    pub qp: QueuePair,
}

/// Two nodes, one InfiniBand device each ("mlx0", LIDs 1 and 2).
pub fn ib_fabric() -> (Arc<MockFabric>, NodeId, NodeId) {
    let fabric = MockFabric::new();
    let a = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(1)])]);
    let b = fabric.add_node(vec![MockDevice::new("mlx0", vec![MockPort::infiniband(2)])]);
    (fabric, a, b)
}

pub fn roce_gid(last: u8) -> Gid {
    let mut g = [0u8; 16];
    g[10] = 0xff;
    g[11] = 0xff;
    g[12..].copy_from_slice(&[10, 0, 0, last]);
    Gid(g)
}

/// Two nodes, one RoCE device each ("roce0"), RoCE v2 GIDs at index 1.
pub fn roce_fabric() -> (Arc<MockFabric>, NodeId, NodeId) {
    let fabric = MockFabric::new();
    let a = fabric.add_node(vec![MockDevice::new(
        "roce0",
        vec![MockPort::roce(roce_gid(1))],
    )]);
    let b = fabric.add_node(vec![MockDevice::new(
        "roce0",
        vec![MockPort::roce(roce_gid(2))],
    )]);
    (fabric, a, b)
}

pub fn caps(send: u32, recv: u32, sge: u32) -> QpCaps {
    QpCaps {
        max_send_wr: send,
        max_recv_wr: recv,
        max_send_sge: sge,
        max_recv_sge: sge,
        max_inline_data: 0,
    }
}

pub fn remote_access() -> Access {
    Access::LOCAL_WRITE | Access::REMOTE_WRITE | Access::REMOTE_READ
}

/// A context, PD, one CQ and a QP in Init on port 1 of `device` on `node`.
pub fn open(fabric: &Arc<MockFabric>, node: NodeId, device: &str, qp_caps: QpCaps) -> Side {
    let ctx = Context::open(fabric.provider(node), device).unwrap();
    let pd = ctx.alloc_pd().unwrap();
    let cq = ctx.create_cq(256).unwrap();
    let qp = pd.create_qp(&cq, &cq, qp_caps).unwrap();
    qp.to_init(&InitAttr {
        port: 1,
        pkey_index: 0,
        access: remote_access(),
    })
    .unwrap();
    Side {
        node,
        ctx,
        pd,
        cq,
        qp,
    }
}

/// How `to` is addressed from `from`'s port: by LID on InfiniBand, by GID index 1 on RoCE.
pub fn address_of(to: &Side) -> Address {
    let port = to.ctx.query_port(1).unwrap();
    if port.lid != 0 {
        Address::Ib {
            lid: port.lid,
            grh: None,
        }
    } else {
        Address::Roce(GlobalRoute {
            dgid: to.ctx.query_gid(1, 1).unwrap().gid,
            sgid_index: 1,
            hop_limit: 64,
            traffic_class: 0,
            flow_label: 0,
        })
    }
}

pub fn rtr(peer: &Side, rq_psn: u32) -> RtrAttr {
    RtrAttr {
        dest_qp_num: peer.qp.qp_num(),
        rq_psn,
        path_mtu: Mtu::Mtu1024,
        max_dest_rd_atomic: 1,
        min_rnr_timer: 12,
        sl: 0,
        address: address_of(peer),
    }
}

pub fn rts(sq_psn: u32, rnr_retry: u8) -> RtsAttr {
    RtsAttr {
        sq_psn,
        timeout: 14,
        retry_cnt: 7,
        rnr_retry,
        max_rd_atomic: 1,
    }
}

/// Both sides to Rts, connected to each other (A sends from PSN_A, B from PSN_B).
pub fn connect(a: &Side, b: &Side) {
    a.qp.to_rtr(&rtr(b, PSN_B)).unwrap();
    b.qp.to_rtr(&rtr(a, PSN_A)).unwrap();
    a.qp.to_rts(&rts(PSN_A, 7)).unwrap();
    b.qp.to_rts(&rts(PSN_B, 7)).unwrap();
}

pub fn pair(
    fabric: &Arc<MockFabric>,
    a: NodeId,
    b: NodeId,
    device: &str,
    qp_caps: QpCaps,
) -> (Side, Side) {
    let sa = open(fabric, a, device, qp_caps);
    let sb = open(fabric, b, device, qp_caps);
    connect(&sa, &sb);
    (sa, sb)
}

/// Polls until `n` completions came, within `polls` polls (each a fabric tick).
pub fn wait(cq: &CompletionQueue, n: usize, polls: usize) -> Vec<Completion> {
    let mut out = Vec::new();
    for _ in 0..polls {
        out.extend(cq.poll(n - out.len()).unwrap());
        if out.len() == n {
            return out;
        }
    }
    panic!(
        "{} of {n} completions after {polls} polls: {out:?}",
        out.len()
    );
}
