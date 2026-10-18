//! The libibverbs provider: the C shim (`shim/gbverbs.c`), compiled against rdma-core's
//! headers, linked to `libibverbs.so.1`.
//!
//! Rust holds only opaque pointers (as [`Handle`]s) and the `gbv_*` structs the shim
//! defines. Their layouts are asserted here at compile time with the same numbers the
//! shim asserts with `_Static_assert`, and [`check_layouts`] compares them with the
//! compiled shim's at run time. No libibverbs struct, enum or constant appears in Rust.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem::{offset_of, size_of};

use crate::error::{Error, Result};
use crate::provider::{Handle, MrKeys, Provider, QpCreated};
use crate::types::{
    Access, Address, DeviceAttr, Gid, GidEntry, GidType, LinkLayer, Mtu, PortAttr, PortState,
    QpCaps, RawCompletion, RawSge, SendOp, Transition, WcStatus, MAX_SGE,
};

#[repr(C)]
struct GbvDeviceAttr {
    max_mr_size: u64,
    node_guid: u64,
    max_qp: u32,
    max_qp_wr: u32,
    max_sge: u32,
    max_cq: u32,
    max_cqe: u32,
    max_mr: u32,
    max_pd: u32,
    max_qp_rd_atom: u32,
    max_qp_init_rd_atom: u32,
    vendor_id: u32,
    vendor_part_id: u32,
    phys_port_cnt: u8,
    pad: [u8; 3],
    fw_ver: [u8; 64],
}

impl Default for GbvDeviceAttr {
    fn default() -> Self {
        GbvDeviceAttr {
            max_mr_size: 0,
            node_guid: 0,
            max_qp: 0,
            max_qp_wr: 0,
            max_sge: 0,
            max_cq: 0,
            max_cqe: 0,
            max_mr: 0,
            max_pd: 0,
            max_qp_rd_atom: 0,
            max_qp_init_rd_atom: 0,
            vendor_id: 0,
            vendor_part_id: 0,
            phys_port_cnt: 0,
            pad: [0; 3],
            fw_ver: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct GbvPortAttr {
    state: u32,
    max_mtu: u32,
    active_mtu: u32,
    gid_tbl_len: u32,
    lid: u16,
    sm_lid: u16,
    link_layer: u8,
    active_width: u8,
    active_speed: u8,
    pad: u8,
}

#[repr(C)]
#[derive(Default)]
struct GbvGid {
    raw: [u8; 16],
    gid_type: u32,
    ndev_ifindex: u32,
}

#[repr(C)]
#[derive(Default)]
struct GbvQpCaps {
    max_send_wr: u32,
    max_recv_wr: u32,
    max_send_sge: u32,
    max_recv_sge: u32,
    max_inline_data: u32,
}

#[repr(C)]
struct GbvInitAttr {
    access: u32,
    pkey_index: u16,
    port_num: u8,
    pad: u8,
}

#[repr(C)]
#[derive(Default)]
struct GbvRtrAttr {
    dgid: [u8; 16],
    dest_qp_num: u32,
    rq_psn: u32,
    path_mtu: u32,
    flow_label: u32,
    dlid: u16,
    max_dest_rd_atomic: u8,
    min_rnr_timer: u8,
    port_num: u8,
    sl: u8,
    is_global: u8,
    sgid_index: u8,
    hop_limit: u8,
    traffic_class: u8,
    pad: [u8; 2],
}

#[repr(C)]
struct GbvRtsAttr {
    sq_psn: u32,
    timeout: u8,
    retry_cnt: u8,
    rnr_retry: u8,
    max_rd_atomic: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GbvSge {
    addr: u64,
    length: u32,
    lkey: u32,
}

#[repr(C)]
struct GbvSendWr {
    wr_id: u64,
    remote_addr: u64,
    opcode: u32,
    rkey: u32,
    imm_data: u32,
    num_sge: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GbvWc {
    wr_id: u64,
    status: u32,
    opcode: u32,
    byte_len: u32,
    qp_num: u32,
    imm_data: u32,
    vendor_err: u32,
    raw_status: u32,
    has_imm: u32,
}

/// (layout id in gbverbs.h, what Rust has): struct sizes and field offsets.
const LAYOUTS: [(u32, usize); 23] = [
    (1, size_of::<GbvDeviceAttr>()),
    (2, offset_of!(GbvDeviceAttr, fw_ver)),
    (3, size_of::<GbvPortAttr>()),
    (4, offset_of!(GbvPortAttr, lid)),
    (5, offset_of!(GbvPortAttr, link_layer)),
    (6, size_of::<GbvGid>()),
    (7, offset_of!(GbvGid, gid_type)),
    (8, size_of::<GbvQpCaps>()),
    (9, offset_of!(GbvQpCaps, max_inline_data)),
    (10, size_of::<GbvInitAttr>()),
    (11, offset_of!(GbvInitAttr, port_num)),
    (12, size_of::<GbvRtrAttr>()),
    (13, offset_of!(GbvRtrAttr, dest_qp_num)),
    (14, offset_of!(GbvRtrAttr, dlid)),
    (15, offset_of!(GbvRtrAttr, traffic_class)),
    (16, size_of::<GbvRtsAttr>()),
    (17, offset_of!(GbvRtsAttr, max_rd_atomic)),
    (18, size_of::<GbvSge>()),
    (19, offset_of!(GbvSge, lkey)),
    (20, size_of::<GbvSendWr>()),
    (21, offset_of!(GbvSendWr, num_sge)),
    (22, size_of::<GbvWc>()),
    (23, offset_of!(GbvWc, has_imm)),
];

/// The numbers gbverbs.c asserts with _Static_assert, in LAYOUTS' order.
const EXPECTED: [usize; 23] = [
    128, 64, 24, 16, 20, 24, 16, 20, 16, 8, 6, 44, 16, 32, 41, 8, 7, 16, 12, 32, 28, 40, 36,
];

const _: () = {
    let mut i = 0;
    while i < LAYOUTS.len() {
        assert!(
            LAYOUTS[i].1 == EXPECTED[i],
            "a gbv_* layout differs from gbverbs.c's"
        );
        i += 1;
    }
};

const ABI_VERSION: u32 = 1;

extern "C" {
    fn gbv_abi_version() -> u32;
    fn gbv_layout(id: u32) -> u64;
    fn gbv_wc_status_str(raw_status: u32) -> *const c_char;
    fn gbv_list_devices(names: *mut c_char, len: usize, count: *mut u32) -> c_int;
    fn gbv_open_device(name: *const c_char, out: *mut *mut c_void) -> c_int;
    fn gbv_close_device(ctx: *mut c_void) -> c_int;
    fn gbv_query_device(ctx: *mut c_void, out: *mut GbvDeviceAttr) -> c_int;
    fn gbv_query_port(ctx: *mut c_void, port: u8, out: *mut GbvPortAttr) -> c_int;
    fn gbv_query_gid(ctx: *mut c_void, port: u8, index: u32, out: *mut GbvGid) -> c_int;
    fn gbv_alloc_pd(ctx: *mut c_void, out: *mut *mut c_void) -> c_int;
    fn gbv_dealloc_pd(pd: *mut c_void) -> c_int;
    fn gbv_create_cq(
        ctx: *mut c_void,
        depth: u32,
        out: *mut *mut c_void,
        actual: *mut u32,
    ) -> c_int;
    fn gbv_destroy_cq(cq: *mut c_void) -> c_int;
    fn gbv_create_rc_qp(
        pd: *mut c_void,
        send_cq: *mut c_void,
        recv_cq: *mut c_void,
        caps: *mut GbvQpCaps,
        out: *mut *mut c_void,
        qp_num: *mut u32,
    ) -> c_int;
    fn gbv_modify_qp_init(qp: *mut c_void, attr: *const GbvInitAttr) -> c_int;
    fn gbv_modify_qp_rtr(qp: *mut c_void, attr: *const GbvRtrAttr) -> c_int;
    fn gbv_modify_qp_rts(qp: *mut c_void, attr: *const GbvRtsAttr) -> c_int;
    fn gbv_modify_qp_error(qp: *mut c_void) -> c_int;
    fn gbv_modify_qp_reset(qp: *mut c_void) -> c_int;
    fn gbv_destroy_qp(qp: *mut c_void) -> c_int;
    fn gbv_reg_mr(
        pd: *mut c_void,
        addr: *mut c_void,
        len: usize,
        access: u32,
        out: *mut *mut c_void,
        lkey: *mut u32,
        rkey: *mut u32,
    ) -> c_int;
    fn gbv_reg_dmabuf_mr(
        pd: *mut c_void,
        offset: u64,
        len: usize,
        iova: u64,
        fd: c_int,
        access: u32,
        out: *mut *mut c_void,
        lkey: *mut u32,
        rkey: *mut u32,
    ) -> c_int;
    fn gbv_dereg_mr(mr: *mut c_void) -> c_int;
    fn gbv_post_send(qp: *mut c_void, wr: *const GbvSendWr, sges: *const GbvSge) -> c_int;
    fn gbv_post_recv(qp: *mut c_void, wr_id: u64, sges: *const GbvSge, num_sge: u32) -> c_int;
    fn gbv_poll_cq(cq: *mut c_void, out: *mut GbvWc, max: u32, n: *mut u32) -> c_int;
}

/// Compares the compiled shim's struct layouts and ABI version with this file's. Calls no
/// libibverbs function and opens nothing.
pub fn check_layouts() -> Result<()> {
    // SAFETY: gbv_abi_version and gbv_layout take plain integers and return constants.
    let abi = unsafe { gbv_abi_version() };
    if abi != ABI_VERSION {
        return Err(Error::Unsupported(format!(
            "the shim's ABI is {abi}, this crate's {ABI_VERSION}"
        )));
    }
    for (id, ours) in LAYOUTS {
        // SAFETY: as above.
        let theirs = unsafe { gbv_layout(id) };
        if theirs != ours as u64 {
            return Err(Error::Unsupported(format!(
                "layout {id}: the shim has {theirs}, Rust {ours}"
            )));
        }
    }
    Ok(())
}

/// libibverbs' text for a raw completion status (`ibv_wc_status_str`). Opens nothing.
pub fn wc_status_text(raw_status: u32) -> String {
    // SAFETY: ibv_wc_status_str returns a pointer to a static NUL-terminated string (an
    // "unknown" one for values it doesn't know), never null.
    unsafe { CStr::from_ptr(gbv_wc_status_str(raw_status)) }
        .to_string_lossy()
        .into_owned()
}

fn check(call: &'static str, rc: c_int) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::Verbs { call, errno: rc })
    }
}

fn ptr(handle: Handle) -> *mut c_void {
    handle as usize as *mut c_void
}

fn handle(ptr: *mut c_void) -> Handle {
    ptr as usize as Handle
}

fn mtu(bytes: u32) -> Mtu {
    // The shim reports 0 for an MTU enum value it doesn't know; the smallest is safe.
    Mtu::from_bytes(bytes).unwrap_or(Mtu::Mtu256)
}

fn wc_status(code: u32, raw: u32) -> WcStatus {
    match code {
        0 => WcStatus::Success,
        1 => WcStatus::LocLenErr,
        2 => WcStatus::LocQpOpErr,
        3 => WcStatus::LocProtErr,
        4 => WcStatus::WrFlushErr,
        5 => WcStatus::BadRespErr,
        6 => WcStatus::LocAccessErr,
        7 => WcStatus::RemInvReqErr,
        8 => WcStatus::RemAccessErr,
        9 => WcStatus::RemOpErr,
        10 => WcStatus::RetryExcErr,
        11 => WcStatus::RnrRetryExcErr,
        12 => WcStatus::GeneralErr,
        _ => WcStatus::Other(raw),
    }
}

fn sges(sges: &[RawSge]) -> Result<Vec<GbvSge>> {
    if sges.len() > MAX_SGE as usize {
        return Err(Error::invalid(format!(
            "{} scatter-gather entries, at most {MAX_SGE}",
            sges.len()
        )));
    }
    Ok(sges
        .iter()
        .map(|s| GbvSge {
            addr: s.addr,
            length: s.len,
            lkey: s.lkey,
        })
        .collect())
}

/// libibverbs, through the shim. Handles are the shim's pointers.
#[derive(Debug, Default)]
pub struct IbvProvider {
    _private: (),
}

impl IbvProvider {
    /// Checks the shim's layouts first (opens nothing).
    pub fn new() -> Result<IbvProvider> {
        check_layouts()?;
        Ok(IbvProvider { _private: () })
    }
}

impl Provider for IbvProvider {
    fn name(&self) -> &str {
        "libibverbs"
    }

    fn list_devices(&self) -> Result<Vec<String>> {
        let mut names = vec![0 as c_char; 4096];
        let mut count = 0u32;
        // SAFETY: names is a writable buffer of the length passed; count a valid u32.
        check("ibv_get_device_list", unsafe {
            gbv_list_devices(names.as_mut_ptr(), names.len(), &mut count)
        })?;
        // SAFETY: on success the shim wrote a NUL-terminated string into names.
        let text = unsafe { CStr::from_ptr(names.as_ptr()) }.to_string_lossy();
        Ok(text.lines().map(str::to_owned).collect())
    }

    fn open_device(&self, name: &str) -> Result<Handle> {
        let name = CString::new(name).map_err(|_| Error::invalid("device name with a NUL"))?;
        let mut out = std::ptr::null_mut();
        // SAFETY: name is NUL-terminated; out a valid pointer slot.
        check("ibv_open_device", unsafe {
            gbv_open_device(name.as_ptr(), &mut out)
        })?;
        Ok(handle(out))
    }

    fn close_device(&self, ctx: Handle) -> Result<()> {
        // SAFETY: ctx came from open_device and is closed once: the owned layer releases
        // each handle once, after everything made from it.
        check("ibv_close_device", unsafe { gbv_close_device(ptr(ctx)) })
    }

    fn query_device(&self, ctx: Handle) -> Result<DeviceAttr> {
        let mut a = GbvDeviceAttr::default();
        // SAFETY: ctx is an open context; a is a gbv_device_attr the shim fills.
        check("ibv_query_device", unsafe {
            gbv_query_device(ptr(ctx), &mut a)
        })?;
        let end = a
            .fw_ver
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(a.fw_ver.len());
        Ok(DeviceAttr {
            max_mr_size: a.max_mr_size,
            node_guid: a.node_guid,
            max_qp: a.max_qp,
            max_qp_wr: a.max_qp_wr,
            max_sge: a.max_sge,
            max_cq: a.max_cq,
            max_cqe: a.max_cqe,
            max_mr: a.max_mr,
            max_pd: a.max_pd,
            max_qp_rd_atom: a.max_qp_rd_atom,
            max_qp_init_rd_atom: a.max_qp_init_rd_atom,
            vendor_id: a.vendor_id,
            vendor_part_id: a.vendor_part_id,
            phys_port_cnt: a.phys_port_cnt,
            fw_ver: String::from_utf8_lossy(&a.fw_ver[..end]).into_owned(),
        })
    }

    fn query_port(&self, ctx: Handle, port: u8) -> Result<PortAttr> {
        let mut a = GbvPortAttr::default();
        // SAFETY: ctx is an open context; a is a gbv_port_attr the shim fills.
        check("ibv_query_port", unsafe {
            gbv_query_port(ptr(ctx), port, &mut a)
        })?;
        Ok(PortAttr {
            state: match a.state {
                1 => PortState::Down,
                2 => PortState::Init,
                3 => PortState::Armed,
                4 => PortState::Active,
                _ => PortState::Other,
            },
            max_mtu: mtu(a.max_mtu),
            active_mtu: mtu(a.active_mtu),
            gid_tbl_len: a.gid_tbl_len,
            lid: a.lid,
            sm_lid: a.sm_lid,
            link_layer: if a.link_layer == 2 {
                LinkLayer::Ethernet
            } else {
                LinkLayer::InfiniBand
            },
        })
    }

    fn query_gid(&self, ctx: Handle, port: u8, index: u32) -> Result<GidEntry> {
        let mut g = GbvGid::default();
        // SAFETY: ctx is an open context; g is a gbv_gid the shim fills.
        check("ibv_query_gid_ex", unsafe {
            gbv_query_gid(ptr(ctx), port, index, &mut g)
        })?;
        Ok(GidEntry {
            gid: Gid(g.raw),
            gid_type: match g.gid_type {
                1 => GidType::Ib,
                2 => GidType::RoceV1,
                3 => GidType::RoceV2,
                _ => GidType::Other,
            },
            ndev_ifindex: g.ndev_ifindex,
        })
    }

    fn alloc_pd(&self, ctx: Handle) -> Result<Handle> {
        let mut out = std::ptr::null_mut();
        // SAFETY: ctx is an open context; out a valid pointer slot.
        check("ibv_alloc_pd", unsafe { gbv_alloc_pd(ptr(ctx), &mut out) })?;
        Ok(handle(out))
    }

    fn dealloc_pd(&self, pd: Handle) -> Result<()> {
        // SAFETY: pd came from alloc_pd; released once, after its regions and queue pairs.
        check("ibv_dealloc_pd", unsafe { gbv_dealloc_pd(ptr(pd)) })
    }

    fn create_cq(&self, ctx: Handle, depth: u32) -> Result<(Handle, u32)> {
        let mut out = std::ptr::null_mut();
        let mut actual = 0u32;
        // SAFETY: ctx is an open context; out and actual are valid slots.
        check("ibv_create_cq", unsafe {
            gbv_create_cq(ptr(ctx), depth, &mut out, &mut actual)
        })?;
        Ok((handle(out), actual))
    }

    fn destroy_cq(&self, cq: Handle) -> Result<()> {
        // SAFETY: cq came from create_cq; released once, after every queue pair on it.
        check("ibv_destroy_cq", unsafe { gbv_destroy_cq(ptr(cq)) })
    }

    fn create_rc_qp(
        &self,
        pd: Handle,
        send_cq: Handle,
        recv_cq: Handle,
        caps: &QpCaps,
    ) -> Result<QpCreated> {
        let mut c = GbvQpCaps {
            max_send_wr: caps.max_send_wr,
            max_recv_wr: caps.max_recv_wr,
            max_send_sge: caps.max_send_sge,
            max_recv_sge: caps.max_recv_sge,
            max_inline_data: caps.max_inline_data,
        };
        let mut out = std::ptr::null_mut();
        let mut qp_num = 0u32;
        // SAFETY: pd and both queues are live handles of one context; c, out and qp_num
        // are valid for the shim to read and write.
        check("ibv_create_qp", unsafe {
            gbv_create_rc_qp(
                ptr(pd),
                ptr(send_cq),
                ptr(recv_cq),
                &mut c,
                &mut out,
                &mut qp_num,
            )
        })?;
        Ok(QpCreated {
            handle: handle(out),
            qp_num,
            caps: QpCaps {
                max_send_wr: c.max_send_wr,
                max_recv_wr: c.max_recv_wr,
                max_send_sge: c.max_send_sge,
                max_recv_sge: c.max_recv_sge,
                max_inline_data: c.max_inline_data,
            },
        })
    }

    fn modify_qp(&self, qp: Handle, transition: &Transition) -> Result<()> {
        let qp = ptr(qp);
        let rc = match transition {
            Transition::Init(a) => {
                let attr = GbvInitAttr {
                    access: a.access.bits(),
                    pkey_index: a.pkey_index,
                    port_num: a.port,
                    pad: 0,
                };
                // SAFETY: qp is a live queue pair; attr outlives the call.
                unsafe { gbv_modify_qp_init(qp, &attr) }
            }
            Transition::Rtr { port, attr: a } => {
                let mut attr = GbvRtrAttr {
                    dest_qp_num: a.dest_qp_num,
                    rq_psn: a.rq_psn,
                    path_mtu: a.path_mtu.bytes(),
                    max_dest_rd_atomic: a.max_dest_rd_atomic,
                    min_rnr_timer: a.min_rnr_timer,
                    port_num: *port,
                    sl: a.sl,
                    ..GbvRtrAttr::default()
                };
                let grh = match a.address {
                    Address::Ib { lid, grh } => {
                        attr.dlid = lid;
                        grh
                    }
                    Address::Roce(grh) => Some(grh),
                };
                if let Some(g) = grh {
                    attr.is_global = 1;
                    attr.dgid = g.dgid.0;
                    attr.sgid_index = g.sgid_index;
                    attr.hop_limit = g.hop_limit;
                    attr.traffic_class = g.traffic_class;
                    attr.flow_label = g.flow_label;
                }
                // SAFETY: qp is a live queue pair; attr outlives the call.
                unsafe { gbv_modify_qp_rtr(qp, &attr) }
            }
            Transition::Rts(a) => {
                let attr = GbvRtsAttr {
                    sq_psn: a.sq_psn,
                    timeout: a.timeout,
                    retry_cnt: a.retry_cnt,
                    rnr_retry: a.rnr_retry,
                    max_rd_atomic: a.max_rd_atomic,
                };
                // SAFETY: qp is a live queue pair; attr outlives the call.
                unsafe { gbv_modify_qp_rts(qp, &attr) }
            }
            // SAFETY: qp is a live queue pair.
            Transition::Error => unsafe { gbv_modify_qp_error(qp) },
            // SAFETY: qp is a live queue pair.
            Transition::Reset => unsafe { gbv_modify_qp_reset(qp) },
        };
        check("ibv_modify_qp", rc)
    }

    fn destroy_qp(&self, qp: Handle) -> Result<()> {
        // SAFETY: qp came from create_rc_qp; destroyed once, after its work drained.
        check("ibv_destroy_qp", unsafe { gbv_destroy_qp(ptr(qp)) })
    }

    unsafe fn reg_mr(
        &self,
        pd: Handle,
        addr: *mut u8,
        len: usize,
        access: Access,
    ) -> Result<MrKeys> {
        let mut out = std::ptr::null_mut();
        let (mut lkey, mut rkey) = (0u32, 0u32);
        // SAFETY: pd is a live protection domain; the caller keeps the memory valid until
        // dereg_mr (this function's contract); the out slots are valid.
        check("ibv_reg_mr", unsafe {
            gbv_reg_mr(
                ptr(pd),
                addr.cast(),
                len,
                access.bits(),
                &mut out,
                &mut lkey,
                &mut rkey,
            )
        })?;
        Ok(MrKeys {
            handle: handle(out),
            lkey,
            rkey,
        })
    }

    fn reg_dmabuf_mr(
        &self,
        pd: Handle,
        offset: u64,
        len: usize,
        iova: u64,
        fd: i32,
        access: Access,
    ) -> Result<MrKeys> {
        let mut out = std::ptr::null_mut();
        let (mut lkey, mut rkey) = (0u32, 0u32);
        // SAFETY: pd is a live protection domain; the kernel validates fd as a dma-buf and
        // takes its own reference to it, so no process memory is involved; the out slots
        // are valid.
        check("ibv_reg_dmabuf_mr", unsafe {
            gbv_reg_dmabuf_mr(
                ptr(pd),
                offset,
                len,
                iova,
                fd,
                access.bits(),
                &mut out,
                &mut lkey,
                &mut rkey,
            )
        })?;
        Ok(MrKeys {
            handle: handle(out),
            lkey,
            rkey,
        })
    }

    fn dereg_mr(&self, mr: Handle) -> Result<()> {
        // SAFETY: mr came from reg_mr; deregistered once, after the last work using it.
        check("ibv_dereg_mr", unsafe { gbv_dereg_mr(ptr(mr)) })
    }

    fn post_send(&self, qp: Handle, wr_id: u64, op: &SendOp, list: &[RawSge]) -> Result<()> {
        let list = sges(list)?;
        let (opcode, remote, imm) = match *op {
            SendOp::Send => (1, None, 0),
            SendOp::SendWithImm(imm) => (2, None, imm),
            SendOp::Write(r) => (3, Some(r), 0),
            SendOp::WriteWithImm(r, imm) => (4, Some(r), imm),
        };
        let wr = GbvSendWr {
            wr_id,
            remote_addr: remote.map_or(0, |r| r.addr),
            opcode,
            rkey: remote.map_or(0, |r| r.rkey),
            imm_data: imm,
            num_sge: list.len() as u32,
        };
        // SAFETY: qp is a live queue pair; wr and list (num_sge entries) outlive the call;
        // each entry lies in a registered region kept registered until this request
        // completes (the owned layer files it with clones of those regions).
        check("ibv_post_send", unsafe {
            gbv_post_send(ptr(qp), &wr, list.as_ptr())
        })
    }

    fn post_recv(&self, qp: Handle, wr_id: u64, list: &[RawSge]) -> Result<()> {
        let list = sges(list)?;
        // SAFETY: as for post_send.
        check("ibv_post_recv", unsafe {
            gbv_post_recv(ptr(qp), wr_id, list.as_ptr(), list.len() as u32)
        })
    }

    fn poll_cq(&self, cq: Handle, out: &mut Vec<RawCompletion>, max: usize) -> Result<usize> {
        let max = max.min(256);
        let mut wcs = vec![GbvWc::default(); max];
        let mut n = 0u32;
        // SAFETY: cq is a live completion queue; wcs has room for max entries.
        check("ibv_poll_cq", unsafe {
            gbv_poll_cq(ptr(cq), wcs.as_mut_ptr(), max as u32, &mut n)
        })?;
        out.extend(wcs[..n as usize].iter().map(|w| RawCompletion {
            wr_id: w.wr_id,
            qp_num: w.qp_num,
            status: wc_status(w.status, w.raw_status),
            byte_len: w.byte_len,
            imm: (w.has_imm != 0).then_some(w.imm_data),
            vendor_err: w.vendor_err,
        }));
        Ok(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These two call into the linked shim and libibverbs but open no device.

    #[test]
    fn the_shim_was_compiled_with_these_layouts() {
        check_layouts().unwrap();
    }

    #[test]
    fn libibverbs_names_completion_statuses() {
        // IBV_WC_SUCCESS is 0 in every rdma-core; the text comes from libibverbs.so.1.
        assert_eq!(wc_status_text(0), "success");
    }

    /// Opens a real device: only where `GPUBRIDGE_VERBS_DEVICE` names one (an allocated
    /// node), never on a login node.
    #[test]
    #[ignore = "opens an RDMA device: set GPUBRIDGE_VERBS_DEVICE on an allocated node"]
    fn opens_a_device_and_creates_resources() {
        use crate::resources::Context;
        use std::sync::Arc;
        let device = std::env::var("GPUBRIDGE_VERBS_DEVICE").expect("GPUBRIDGE_VERBS_DEVICE");
        let provider: Arc<dyn Provider> = Arc::new(IbvProvider::new().unwrap());
        let ctx = Context::open(provider, &device).unwrap();
        let pd = ctx.alloc_pd().unwrap();
        let cq = ctx.create_cq(16).unwrap();
        let caps = QpCaps {
            max_send_wr: 4,
            max_recv_wr: 4,
            ..QpCaps::default()
        };
        let qp = pd.create_qp(&cq, &cq, caps).unwrap();
        let mr = pd.register_buffer(4096, Access::LOCAL_WRITE).unwrap();
        assert!(mr.lkey() != 0 || mr.rkey() != 0);
        drop(qp);
    }
}
