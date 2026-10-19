//! Plain values passed between the owned resources and a [`Provider`](crate::Provider):
//! attributes, addresses, work requests and completions. None of them mirrors a libibverbs
//! struct; the C shim translates (docs/VERBS_TRANSPORT.md, "Bindings").

use std::fmt;
use std::ops::BitOr;

use crate::error::{Error, Result};

/// Memory region access rights.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Access(u32);

impl Access {
    pub const NONE: Access = Access(0);
    pub const LOCAL_WRITE: Access = Access(1);
    pub const REMOTE_WRITE: Access = Access(2);
    pub const REMOTE_READ: Access = Access(4);
    /// PCIe relaxed ordering where the device supports it (ignored where it doesn't).
    pub const RELAXED_ORDERING: Access = Access(8);
    const ALL: u32 = 15;

    /// From raw bits (the C shim's `GBV_ACCESS_*`), refusing unknown ones.
    pub fn from_bits(bits: u32) -> Result<Access> {
        if bits & !Self::ALL != 0 {
            return Err(Error::invalid(format!("unknown access bits {bits:#x}")));
        }
        Ok(Access(bits))
    }

    pub fn bits(self) -> u32 {
        self.0
    }

    pub fn contains(self, other: Access) -> bool {
        self.0 & other.0 == other.0
    }

    /// `self | other`, in constants.
    pub const fn union(self, other: Access) -> Access {
        Access(self.0 | other.0)
    }

    /// Remote write or read access needs local write too (verbs refuses it otherwise).
    pub(crate) fn validate(self) -> Result<()> {
        if (self.contains(Access::REMOTE_WRITE)) && !self.contains(Access::LOCAL_WRITE) {
            return Err(Error::invalid(
                "remote write access needs local write access",
            ));
        }
        Ok(())
    }
}

impl BitOr for Access {
    type Output = Access;
    fn bitor(self, rhs: Access) -> Access {
        Access(self.0 | rhs.0)
    }
}

impl fmt::Debug for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = [
            (Access::LOCAL_WRITE, "LOCAL_WRITE"),
            (Access::REMOTE_WRITE, "REMOTE_WRITE"),
            (Access::REMOTE_READ, "REMOTE_READ"),
            (Access::RELAXED_ORDERING, "RELAXED_ORDERING"),
        ];
        let set: Vec<&str> = names
            .iter()
            .filter(|(a, _)| self.contains(*a))
            .map(|(_, n)| *n)
            .collect();
        if set.is_empty() {
            write!(f, "Access(NONE)")
        } else {
            write!(f, "Access({})", set.join("|"))
        }
    }
}

/// A path MTU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mtu {
    Mtu256,
    Mtu512,
    Mtu1024,
    Mtu2048,
    Mtu4096,
}

impl Mtu {
    pub fn bytes(self) -> u32 {
        match self {
            Mtu::Mtu256 => 256,
            Mtu::Mtu512 => 512,
            Mtu::Mtu1024 => 1024,
            Mtu::Mtu2048 => 2048,
            Mtu::Mtu4096 => 4096,
        }
    }

    pub fn from_bytes(bytes: u32) -> Option<Mtu> {
        Some(match bytes {
            256 => Mtu::Mtu256,
            512 => Mtu::Mtu512,
            1024 => Mtu::Mtu1024,
            2048 => Mtu::Mtu2048,
            4096 => Mtu::Mtu4096,
            _ => return None,
        })
    }
}

/// Whether memory registered in this process stays what the device reads and writes
/// after a `fork()` (`ibv_is_fork_initialized`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkStatus {
    /// Not protected: a fork, then a write by the parent to a registered page, gives the
    /// parent a copy the device never sees (`IBV_FORK_DISABLED`).
    Disabled,
    /// `ibv_fork_init` ran (or `RDMAV_FORK_SAFE` was set): registered pages are kept out
    /// of children (`IBV_FORK_ENABLED`).
    Enabled,
    /// The kernel keeps registered pages with the parent on fork (Linux 5.13 and later,
    /// FOLL_PIN copy-on-fork): nothing to do (`IBV_FORK_UNNEEDED`).
    Unneeded,
    /// This libibverbs predates `ibv_is_fork_initialized` (rdma-core 35): can't tell.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortState {
    Down,
    Init,
    Armed,
    Active,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkLayer {
    InfiniBand,
    /// RoCE: addressed by GID, always with a global route.
    Ethernet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GidType {
    Ib,
    RoceV1,
    RoceV2,
    Other,
}

/// A 128-bit GID, in network byte order as on the wire.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Gid(pub [u8; 16]);

impl Gid {
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
    }
}

impl fmt::Debug for Gid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let words: Vec<String> = self
            .0
            .chunks(2)
            .map(|w| format!("{:02x}{:02x}", w[0], w[1]))
            .collect();
        write!(f, "Gid({})", words.join(":"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GidEntry {
    pub gid: Gid,
    pub gid_type: GidType,
    /// The network interface of a RoCE GID (0 for InfiniBand).
    pub ndev_ifindex: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeviceAttr {
    pub max_mr_size: u64,
    pub node_guid: u64,
    pub max_qp: u32,
    pub max_qp_wr: u32,
    pub max_sge: u32,
    pub max_cq: u32,
    pub max_cqe: u32,
    pub max_mr: u32,
    pub max_pd: u32,
    pub max_qp_rd_atom: u32,
    pub max_qp_init_rd_atom: u32,
    pub vendor_id: u32,
    pub vendor_part_id: u32,
    pub phys_port_cnt: u8,
    pub fw_ver: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortAttr {
    pub state: PortState,
    pub max_mtu: Mtu,
    pub active_mtu: Mtu,
    pub gid_tbl_len: u32,
    pub lid: u16,
    pub sm_lid: u16,
    pub link_layer: LinkLayer,
}

/// What a queue pair may hold at once. The owned layer refuses work beyond these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QpCaps {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

impl Default for QpCaps {
    fn default() -> Self {
        QpCaps {
            max_send_wr: 64,
            max_recv_wr: 64,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        }
    }
}

/// The most scatter-gather entries one work request may have (the C shim's `GBV_MAX_SGE`).
pub const MAX_SGE: u32 = 16;

/// Queue pair states, as in verbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QpState {
    Reset,
    Init,
    /// Ready to receive.
    Rtr,
    /// Ready to send.
    Rts,
    Err,
}

/// `Reset -> Init`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitAttr {
    pub port: u8,
    pub pkey_index: u16,
    /// What the peer may do to this side's memory through this queue pair.
    pub access: Access,
}

/// A global route (GRH): mandatory on RoCE, optional on InfiniBand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalRoute {
    pub dgid: Gid,
    /// This side's GID table index to send from.
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
    pub flow_label: u32,
}

/// Where the peer queue pair is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    /// InfiniBand: the peer port's LID, with a global route only across subnets.
    Ib { lid: u16, grh: Option<GlobalRoute> },
    /// RoCE (Ethernet link layer): GID addressing.
    Roce(GlobalRoute),
}

/// `Init -> Rtr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtrAttr {
    pub dest_qp_num: u32,
    pub rq_psn: u32,
    pub path_mtu: Mtu,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub sl: u8,
    pub address: Address,
}

/// `Rtr -> Rts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtsAttr {
    pub sq_psn: u32,
    pub timeout: u8,
    pub retry_cnt: u8,
    /// 7 means retry forever.
    pub rnr_retry: u8,
    pub max_rd_atomic: u8,
}

/// One state change, as the provider performs it (already validated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transition {
    Init(InitAttr),
    Rtr { port: u8, attr: RtrAttr },
    Rts(RtsAttr),
    Error,
    Reset,
}

/// A scatter-gather entry as the provider sees it: an address inside a registered region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawSge {
    pub addr: u64,
    pub len: u32,
    pub lkey: u32,
}

/// Where an RDMA write lands on the peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteTarget {
    pub addr: u64,
    pub rkey: u32,
}

/// The send-side operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendOp {
    Send,
    SendWithImm(u32),
    Write(RemoteTarget),
    /// Consumes a receive on the peer, which completes with the immediate value.
    WriteWithImm(RemoteTarget, u32),
}

impl SendOp {
    pub fn kind(&self) -> WorkKind {
        match self {
            SendOp::Send => WorkKind::Send,
            SendOp::SendWithImm(_) => WorkKind::SendWithImm,
            SendOp::Write(_) => WorkKind::Write,
            SendOp::WriteWithImm(..) => WorkKind::WriteWithImm,
        }
    }
}

/// What a tracked work request was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkKind {
    Send,
    SendWithImm,
    Write,
    WriteWithImm,
    Recv,
}

/// Completion status. Anything but `Success` leaves an RC queue pair in the error state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WcStatus {
    Success,
    LocLenErr,
    LocQpOpErr,
    LocProtErr,
    /// The queue pair went to the error state before this request ran.
    WrFlushErr,
    BadRespErr,
    LocAccessErr,
    RemInvReqErr,
    RemAccessErr,
    RemOpErr,
    /// The peer didn't answer within the retry budget (a lost peer, a wrong address).
    RetryExcErr,
    /// The peer had no receive posted within the RNR retry budget.
    RnrRetryExcErr,
    GeneralErr,
    /// Another status: the provider's raw value.
    Other(u32),
}

impl WcStatus {
    /// Stable libibverbs spelling for observation consumers (never parsed from prose).
    pub fn name(self) -> &'static str {
        match self {
            Self::Success => "IBV_WC_SUCCESS",
            Self::LocLenErr => "IBV_WC_LOC_LEN_ERR",
            Self::LocQpOpErr => "IBV_WC_LOC_QP_OP_ERR",
            Self::LocProtErr => "IBV_WC_LOC_PROT_ERR",
            Self::WrFlushErr => "IBV_WC_WR_FLUSH_ERR",
            Self::BadRespErr => "IBV_WC_BAD_RESP_ERR",
            Self::LocAccessErr => "IBV_WC_LOC_ACCESS_ERR",
            Self::RemInvReqErr => "IBV_WC_REM_INV_REQ_ERR",
            Self::RemAccessErr => "IBV_WC_REM_ACCESS_ERR",
            Self::RemOpErr => "IBV_WC_REM_OP_ERR",
            Self::RetryExcErr => "IBV_WC_RETRY_EXC_ERR",
            Self::RnrRetryExcErr => "IBV_WC_RNR_RETRY_EXC_ERR",
            Self::GeneralErr => "IBV_WC_GENERAL_ERR",
            Self::Other(_) => "IBV_WC_UNKNOWN",
        }
    }
}

/// What the provider reported for one finished work request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawCompletion {
    pub wr_id: u64,
    pub qp_num: u32,
    pub status: WcStatus,
    pub byte_len: u32,
    pub imm: Option<u32>,
    pub vendor_err: u32,
}

/// The work request numbers the owned layer gives out (unique per queue pair).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkId(pub u64);

/// A finished work request, as [`CompletionQueue::poll`](crate::CompletionQueue::poll)
/// returns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub qp_num: u32,
    pub work: WorkId,
    /// The caller's tag from posting.
    pub tag: u64,
    pub kind: WorkKind,
    pub status: WcStatus,
    /// Bytes received (receives) or sent; meaningful on success only.
    pub byte_len: u32,
    pub imm: Option<u32>,
    pub vendor_err: u32,
}

impl Completion {
    pub fn ok(&self) -> bool {
        self.status == WcStatus::Success
    }
}

/// One send of a batch, as the provider posts it (one `ibv_post_send` for the batch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawSend<'a> {
    pub wr_id: u64,
    pub op: SendOp,
    pub sges: &'a [RawSge],
}

/// One receive of a batch (one `ibv_post_recv` for the batch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawRecv<'a> {
    pub wr_id: u64,
    pub sges: &'a [RawSge],
}

/// An asynchronous event's type (`enum ibv_event_type`, translated by the shim).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AsyncEventKind {
    CqErr,
    QpFatal,
    QpReqErr,
    QpAccessErr,
    CommEst,
    SqDrained,
    PathMig,
    PathMigErr,
    DeviceFatal,
    PortActive,
    PortErr,
    LidChange,
    PkeyChange,
    SmChange,
    SrqErr,
    SrqLimitReached,
    QpLastWqeReached,
    ClientReregister,
    GidChange,
    /// Another type: the provider's raw value.
    Other(u32),
}

impl AsyncEventKind {
    /// Whether the event breaks what it names: a queue pair or completion queue error, a
    /// fatal device error, a port that went down. The others are informational.
    pub fn fatal(self) -> bool {
        matches!(
            self,
            AsyncEventKind::CqErr
                | AsyncEventKind::QpFatal
                | AsyncEventKind::QpReqErr
                | AsyncEventKind::QpAccessErr
                | AsyncEventKind::PathMigErr
                | AsyncEventKind::DeviceFatal
                | AsyncEventKind::PortErr
        )
    }
}

/// An asynchronous event from a device (`ibv_get_async_event`), already acknowledged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AsyncEvent {
    pub kind: AsyncEventKind,
    /// The queue pair it is about, for queue pair events.
    pub qp_num: Option<u32>,
    /// The port it is about, for port events.
    pub port: Option<u8>,
}

/// What [`Provider::wait_events`](crate::Provider::wait_events) found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Events {
    /// A completion queue notification arrived (and was acknowledged): poll the queues on
    /// the channel and ask for the next notification.
    pub completion: bool,
    pub async_events: Vec<AsyncEvent>,
}
