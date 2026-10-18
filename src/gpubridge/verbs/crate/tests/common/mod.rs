//! Two mock nodes with connected queue pairs, and a bare queue pair that poses as a bridge
//! member.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpubridge_verbs::protocol::{Header, Info, Kind, HEADER_BYTES, PROTOCOL_VERSION};
use gpubridge_verbs::{
    Access, Address, Completion, CompletionQueue, Config, Context, Gid, GlobalRoute, InitAttr,
    LinkLayer, MemoryRegion, MockDevice, MockFabric, MockPort, Mtu, NodeId, ProtectionDomain,
    QpCaps, QueuePair, RemoteTarget, RtrAttr, RtsAttr, SendOp, SendRequest, Sge, WcStatus,
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

/// A bare queue pair posing as member `index` of a bridge to one endpoint (InfiniBand
/// device "mlx0" on its node): it sends whatever a test writes, including what no
/// endpoint would, and reads what the endpoint sends it.
pub struct RawMember {
    pub side: Side,
    recv_slots: Vec<MemoryRegion>,
    slot_bytes: u32,
    pub info: String,
    psn: u32,
    seq: u32,
}

impl RawMember {
    /// A member of `members`, talking to member `peer` only, with `config`'s slot,
    /// credit and rendezvous settings (the handshake compares them).
    pub fn new(
        fabric: &Arc<MockFabric>,
        node: NodeId,
        members: usize,
        peer: usize,
        config: &Config,
    ) -> RawMember {
        let recvs = config.credits + config.ctrl_credits;
        let side = open(fabric, node, "mlx0", caps(64, recvs, 1));
        let recv_slots: Vec<MemoryRegion> = (0..recvs)
            .map(|_| {
                side.pd
                    .register_buffer(config.slot_bytes as usize, Access::LOCAL_WRITE)
                    .unwrap()
            })
            .collect();
        for (i, r) in recv_slots.iter().enumerate() {
            side.qp
                .post_recv(i as u64, &[Sge::new(r, 0, config.slot_bytes)])
                .unwrap();
        }
        let port = side.ctx.query_port(1).unwrap();
        let psn = 0x77;
        let mut qpns = vec![0; members];
        let mut psns = vec![0; members];
        qpns[peer] = side.qp.qp_num();
        psns[peer] = psn;
        let info = Info {
            version: PROTOCOL_VERSION,
            link_layer: LinkLayer::InfiniBand,
            lid: port.lid,
            gid: side.ctx.query_gid(1, 0).unwrap().gid,
            grh: false,
            active_mtu: port.active_mtu,
            slot_bytes: config.slot_bytes,
            credits: config.credits,
            ctrl_credits: config.ctrl_credits,
            rndv: config.rndv_bytes,
            qpns,
            psns,
            device: "mlx0".into(),
            port: 1,
        }
        .encode();
        RawMember {
            side,
            recv_slots,
            slot_bytes: config.slot_bytes,
            info,
            psn,
            seq: 0,
        }
    }

    /// Connects to the endpoint (member `peer`'s handshake line `theirs`), as this
    /// member `index`.
    pub fn connect(&self, theirs: &str, index: usize) {
        let info = Info::decode(theirs).unwrap();
        self.side
            .qp
            .to_rtr(&RtrAttr {
                dest_qp_num: info.qpns[index],
                rq_psn: info.psns[index],
                path_mtu: info.active_mtu,
                max_dest_rd_atomic: 0,
                min_rnr_timer: 12,
                sl: 0,
                address: Address::Ib {
                    lid: info.lid,
                    grh: None,
                },
            })
            .unwrap();
        self.side.qp.to_rts(&rts(self.psn, 7)).unwrap();
    }

    /// The next header this member would send, of `kind` (a sequence number of its own;
    /// it never returns credits).
    pub fn header(&mut self, kind: Kind, tag: u64, msg_len: u64) -> Header {
        let h = Header {
            kind,
            seq: self.seq,
            chunk_len: 0,
            data_freed: 0,
            ctrl_freed: 0,
            key: 0,
            tag,
            msg_len,
            offset: 0,
        };
        self.seq += 1;
        h
    }

    /// SENDs `bytes` as one message and waits for its completion.
    pub fn send(&self, bytes: &[u8]) -> WcStatus {
        let buf = self
            .side
            .pd
            .register_buffer(bytes.len().max(1), Access::NONE)
            .unwrap();
        buf.write(0, bytes).unwrap();
        let sge = [Sge::new(&buf, 0, bytes.len() as u32)];
        self.post(SendOp::Send, &sge)
    }

    /// RDMA WRITEs `data` to `addr` with `rkey` and waits for its completion.
    pub fn write(&self, data: &[u8], addr: u64, rkey: u32) -> WcStatus {
        let buf = self
            .side
            .pd
            .register_buffer(data.len().max(1), Access::NONE)
            .unwrap();
        buf.write(0, data).unwrap();
        let sge = [Sge::new(&buf, 0, data.len() as u32)];
        self.post(SendOp::Write(RemoteTarget { addr, rkey }), &sge)
    }

    fn post(&self, op: SendOp, sges: &[Sge<'_>]) -> WcStatus {
        self.side
            .qp
            .post_send(&SendRequest {
                tag: 1 << 40,
                op,
                sges,
            })
            .unwrap();
        loop {
            let got = self.poll(Duration::from_secs(10));
            if got.tag == 1 << 40 {
                return got.status;
            }
        }
    }

    fn poll(&self, timeout: Duration) -> Completion {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(c) = self.side.cq.poll(1).unwrap().pop() {
                return c;
            }
            assert!(
                Instant::now() < deadline,
                "nothing completed in {timeout:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The next message the endpoint sent this member: its header (the slot isn't
    /// reposted: tests read only a few).
    pub fn receive(&self, timeout: Duration) -> Header {
        let c = self.poll(timeout);
        assert!(c.ok(), "{c:?}");
        let mut raw = [0u8; HEADER_BYTES];
        self.recv_slots[c.tag as usize].read(0, &mut raw).unwrap();
        Header::decode(&raw, c.byte_len).unwrap()
    }
}

/// A message every endpoint refuses: no magic number.
pub fn garbage() -> Vec<u8> {
    vec![0u8; HEADER_BYTES]
}

/// Waits up to `timeout` for `cond`.
pub fn eventually(timeout: Duration, what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while !cond() {
        assert!(Instant::now() < deadline, "not within {timeout:?}: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}
