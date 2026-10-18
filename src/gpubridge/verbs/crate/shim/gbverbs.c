/*
 * libibverbs behind gbverbs.h. Compiled against rdma-core's <infiniband/verbs.h>, so the
 * header's static inline functions (ibv_post_send, ibv_post_recv, ibv_poll_cq, ibv_reg_mr,
 * ibv_query_gid_ex, ibv_query_port) are the header's own code, and linked to the system's
 * libibverbs.so.1. Rust sees none of the ibv_* types: see gbverbs.h.
 */
/* dladdr (gbv_library_path). */
#define _GNU_SOURCE
#include "gbverbs.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <infiniband/verbs.h>

/* The layouts Rust relies on (src/ibv.rs asserts the same numbers). */
_Static_assert(sizeof(struct gbv_device_attr) == 128, "gbv_device_attr size");
_Static_assert(offsetof(struct gbv_device_attr, fw_ver) == 64, "gbv_device_attr.fw_ver");
_Static_assert(sizeof(struct gbv_port_attr) == 24, "gbv_port_attr size");
_Static_assert(offsetof(struct gbv_port_attr, lid) == 16, "gbv_port_attr.lid");
_Static_assert(offsetof(struct gbv_port_attr, link_layer) == 20, "gbv_port_attr.link_layer");
_Static_assert(sizeof(struct gbv_gid) == 24, "gbv_gid size");
_Static_assert(offsetof(struct gbv_gid, gid_type) == 16, "gbv_gid.gid_type");
_Static_assert(sizeof(struct gbv_qp_caps) == 20, "gbv_qp_caps size");
_Static_assert(offsetof(struct gbv_qp_caps, max_inline_data) == 16, "gbv_qp_caps.max_inline_data");
_Static_assert(sizeof(struct gbv_init_attr) == 8, "gbv_init_attr size");
_Static_assert(offsetof(struct gbv_init_attr, port_num) == 6, "gbv_init_attr.port_num");
_Static_assert(sizeof(struct gbv_rtr_attr) == 44, "gbv_rtr_attr size");
_Static_assert(offsetof(struct gbv_rtr_attr, dest_qp_num) == 16, "gbv_rtr_attr.dest_qp_num");
_Static_assert(offsetof(struct gbv_rtr_attr, dlid) == 32, "gbv_rtr_attr.dlid");
_Static_assert(offsetof(struct gbv_rtr_attr, traffic_class) == 41, "gbv_rtr_attr.traffic_class");
_Static_assert(sizeof(struct gbv_rts_attr) == 8, "gbv_rts_attr size");
_Static_assert(offsetof(struct gbv_rts_attr, max_rd_atomic) == 7, "gbv_rts_attr.max_rd_atomic");
_Static_assert(sizeof(struct gbv_sge) == 16, "gbv_sge size");
_Static_assert(offsetof(struct gbv_sge, lkey) == 12, "gbv_sge.lkey");
_Static_assert(sizeof(struct gbv_send_wr) == 32, "gbv_send_wr size");
_Static_assert(offsetof(struct gbv_send_wr, num_sge) == 28, "gbv_send_wr.num_sge");
_Static_assert(sizeof(struct gbv_wc) == 40, "gbv_wc size");
_Static_assert(offsetof(struct gbv_wc, has_imm) == 36, "gbv_wc.has_imm");
_Static_assert(sizeof(struct gbv_async_event) == 16, "gbv_async_event size");
_Static_assert(offsetof(struct gbv_async_event, has_port) == 14, "gbv_async_event.has_port");
/* Our scatter-gather entries go straight into ibv_sge's fields, one by one. */
_Static_assert(sizeof(((struct ibv_sge *)0)->length) == sizeof(uint32_t), "ibv_sge.length");

#define CTX(p) ((struct ibv_context *)(p))
#define PD(p) ((struct ibv_pd *)(p))
#define CQ(p) ((struct ibv_cq *)(p))
#define QP(p) ((struct ibv_qp *)(p))
#define MR(p) ((struct ibv_mr *)(p))
#define CH(p) ((struct ibv_comp_channel *)(p))

/* errno after a call that failed without one: never report success for a failure. */
static int fail(void) { return errno > 0 ? errno : EIO; }

/* A return value that is itself an errno (ibv_modify_qp, ibv_post_*, ibv_destroy_*). */
static int status(int rc) { return rc == 0 ? 0 : (rc > 0 ? rc : (errno > 0 ? errno : EIO)); }

static int nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return fail();
    return 0;
}

uint32_t gbv_abi_version(void) { return GBV_ABI_VERSION; }

uint64_t gbv_layout(uint32_t id) {
    switch (id) {
    case GBV_L_DEVICE_ATTR: return sizeof(struct gbv_device_attr);
    case GBV_L_DEVICE_ATTR_FW_VER: return offsetof(struct gbv_device_attr, fw_ver);
    case GBV_L_PORT_ATTR: return sizeof(struct gbv_port_attr);
    case GBV_L_PORT_ATTR_LID: return offsetof(struct gbv_port_attr, lid);
    case GBV_L_PORT_ATTR_LINK_LAYER: return offsetof(struct gbv_port_attr, link_layer);
    case GBV_L_GID: return sizeof(struct gbv_gid);
    case GBV_L_GID_TYPE: return offsetof(struct gbv_gid, gid_type);
    case GBV_L_QP_CAPS: return sizeof(struct gbv_qp_caps);
    case GBV_L_QP_CAPS_INLINE: return offsetof(struct gbv_qp_caps, max_inline_data);
    case GBV_L_INIT_ATTR: return sizeof(struct gbv_init_attr);
    case GBV_L_INIT_ATTR_PORT: return offsetof(struct gbv_init_attr, port_num);
    case GBV_L_RTR_ATTR: return sizeof(struct gbv_rtr_attr);
    case GBV_L_RTR_ATTR_DEST_QP_NUM: return offsetof(struct gbv_rtr_attr, dest_qp_num);
    case GBV_L_RTR_ATTR_DLID: return offsetof(struct gbv_rtr_attr, dlid);
    case GBV_L_RTR_ATTR_TRAFFIC_CLASS: return offsetof(struct gbv_rtr_attr, traffic_class);
    case GBV_L_RTS_ATTR: return sizeof(struct gbv_rts_attr);
    case GBV_L_RTS_ATTR_MAX_RD_ATOMIC: return offsetof(struct gbv_rts_attr, max_rd_atomic);
    case GBV_L_SGE: return sizeof(struct gbv_sge);
    case GBV_L_SGE_LKEY: return offsetof(struct gbv_sge, lkey);
    case GBV_L_SEND_WR: return sizeof(struct gbv_send_wr);
    case GBV_L_SEND_WR_NUM_SGE: return offsetof(struct gbv_send_wr, num_sge);
    case GBV_L_WC: return sizeof(struct gbv_wc);
    case GBV_L_WC_HAS_IMM: return offsetof(struct gbv_wc, has_imm);
    case GBV_L_ASYNC_EVENT: return sizeof(struct gbv_async_event);
    case GBV_L_ASYNC_EVENT_HAS_PORT: return offsetof(struct gbv_async_event, has_port);
    default: return UINT64_MAX;
    }
}

const char *gbv_wc_status_str(uint32_t raw_status) {
    return ibv_wc_status_str((enum ibv_wc_status)raw_status);
}

static int access_flags(uint32_t access, int *out) {
    if (access & ~(GBV_ACCESS_LOCAL_WRITE | GBV_ACCESS_REMOTE_WRITE | GBV_ACCESS_REMOTE_READ |
                   GBV_ACCESS_RELAXED_ORDERING))
        return EINVAL;
    int flags = 0;
    if (access & GBV_ACCESS_LOCAL_WRITE) flags |= IBV_ACCESS_LOCAL_WRITE;
    if (access & GBV_ACCESS_REMOTE_WRITE) flags |= IBV_ACCESS_REMOTE_WRITE;
    if (access & GBV_ACCESS_REMOTE_READ) flags |= IBV_ACCESS_REMOTE_READ;
    if (access & GBV_ACCESS_RELAXED_ORDERING) flags |= IBV_ACCESS_RELAXED_ORDERING;
    *out = flags;
    return 0;
}

static uint32_t mtu_bytes(enum ibv_mtu mtu) {
    switch (mtu) {
    case IBV_MTU_256: return 256;
    case IBV_MTU_512: return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
    }
    return 0;
}

static int mtu_enum(uint32_t bytes, enum ibv_mtu *out) {
    switch (bytes) {
    case 256: *out = IBV_MTU_256; return 0;
    case 512: *out = IBV_MTU_512; return 0;
    case 1024: *out = IBV_MTU_1024; return 0;
    case 2048: *out = IBV_MTU_2048; return 0;
    case 4096: *out = IBV_MTU_4096; return 0;
    }
    return EINVAL;
}

int gbv_list_devices(char *names, size_t len, uint32_t *count) {
    if (!names || len == 0 || !count) return EINVAL;
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) return fail();
    size_t used = 0;
    int rc = 0;
    names[0] = '\0';
    for (int i = 0; i < n; i++) {
        const char *name = ibv_get_device_name(list[i]);
        size_t l = name ? strlen(name) : 0;
        if (used + l + 2 > len) {
            rc = ENOSPC;
            break;
        }
        memcpy(names + used, name, l);
        names[used + l] = '\n';
        used += l + 1;
        names[used] = '\0';
    }
    ibv_free_device_list(list);
    *count = rc == 0 ? (uint32_t)n : 0;
    return rc;
}

int gbv_open_device(const char *name, gbv_context **out) {
    if (!name || !out) return EINVAL;
    *out = NULL;
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) return fail();
    int rc = ENODEV;
    for (int i = 0; i < n; i++) {
        const char *dev = ibv_get_device_name(list[i]);
        if (dev && strcmp(dev, name) == 0) {
            errno = 0;
            struct ibv_context *ctx = ibv_open_device(list[i]);
            rc = ctx ? 0 : fail();
            if (ctx) {
                /* gbv_wait_events polls this descriptor and must never block reading it. */
                rc = nonblocking(ctx->async_fd);
                if (rc) {
                    ibv_close_device(ctx);
                    ctx = NULL;
                }
            }
            *out = (gbv_context *)ctx;
            break;
        }
    }
    /* An open context keeps its device: the list can go. */
    ibv_free_device_list(list);
    return rc;
}

int gbv_close_device(gbv_context *ctx) {
    if (!ctx) return EINVAL;
    return status(ibv_close_device(CTX(ctx)));
}

int gbv_query_device(gbv_context *ctx, struct gbv_device_attr *out) {
    if (!ctx || !out) return EINVAL;
    struct ibv_device_attr attr;
    memset(&attr, 0, sizeof(attr));
    int rc = status(ibv_query_device(CTX(ctx), &attr));
    if (rc) return rc;
    memset(out, 0, sizeof(*out));
    out->max_mr_size = attr.max_mr_size;
    out->node_guid = be64toh(attr.node_guid);
    out->max_qp = (uint32_t)attr.max_qp;
    out->max_qp_wr = (uint32_t)attr.max_qp_wr;
    out->max_sge = (uint32_t)attr.max_sge;
    out->max_cq = (uint32_t)attr.max_cq;
    out->max_cqe = (uint32_t)attr.max_cqe;
    out->max_mr = (uint32_t)attr.max_mr;
    out->max_pd = (uint32_t)attr.max_pd;
    out->max_qp_rd_atom = (uint32_t)attr.max_qp_rd_atom;
    out->max_qp_init_rd_atom = (uint32_t)attr.max_qp_init_rd_atom;
    out->vendor_id = attr.vendor_id;
    out->vendor_part_id = attr.vendor_part_id;
    out->phys_port_cnt = attr.phys_port_cnt;
    _Static_assert(sizeof(out->fw_ver) <= sizeof(attr.fw_ver), "fw_ver fits");
    memcpy(out->fw_ver, attr.fw_ver, sizeof(out->fw_ver) - 1);
    out->fw_ver[sizeof(out->fw_ver) - 1] = '\0';
    return 0;
}

int gbv_query_port(gbv_context *ctx, uint8_t port, struct gbv_port_attr *out) {
    if (!ctx || !out) return EINVAL;
    struct ibv_port_attr attr;
    memset(&attr, 0, sizeof(attr));
    int rc = status(ibv_query_port(CTX(ctx), port, &attr));
    if (rc) return rc;
    memset(out, 0, sizeof(*out));
    switch (attr.state) {
    case IBV_PORT_DOWN: out->state = GBV_PORT_DOWN; break;
    case IBV_PORT_INIT: out->state = GBV_PORT_INIT; break;
    case IBV_PORT_ARMED: out->state = GBV_PORT_ARMED; break;
    case IBV_PORT_ACTIVE: out->state = GBV_PORT_ACTIVE; break;
    default: out->state = GBV_PORT_OTHER; break;
    }
    out->max_mtu = mtu_bytes(attr.max_mtu);
    out->active_mtu = mtu_bytes(attr.active_mtu);
    out->gid_tbl_len = attr.gid_tbl_len > 0 ? (uint32_t)attr.gid_tbl_len : 0;
    out->lid = attr.lid;
    out->sm_lid = attr.sm_lid;
    /* Unspecified means InfiniBand (verbs.h, enum ibv_link_layer). */
    out->link_layer =
        attr.link_layer == IBV_LINK_LAYER_ETHERNET ? GBV_LINK_ETHERNET : GBV_LINK_INFINIBAND;
    out->active_width = attr.active_width;
    out->active_speed = attr.active_speed;
    return 0;
}

int gbv_query_gid(gbv_context *ctx, uint8_t port, uint32_t index, struct gbv_gid *out) {
    if (!ctx || !out) return EINVAL;
    struct ibv_gid_entry entry;
    memset(&entry, 0, sizeof(entry));
    int rc = status(ibv_query_gid_ex(CTX(ctx), port, index, &entry, 0));
    if (rc) return rc;
    memset(out, 0, sizeof(*out));
    _Static_assert(sizeof(out->raw) == sizeof(entry.gid.raw), "gid size");
    memcpy(out->raw, entry.gid.raw, sizeof(out->raw));
    switch (entry.gid_type) {
    case IBV_GID_TYPE_IB: out->gid_type = GBV_GID_IB; break;
    case IBV_GID_TYPE_ROCE_V1: out->gid_type = GBV_GID_ROCE_V1; break;
    case IBV_GID_TYPE_ROCE_V2: out->gid_type = GBV_GID_ROCE_V2; break;
    default: out->gid_type = GBV_GID_OTHER; break;
    }
    out->ndev_ifindex = entry.ndev_ifindex;
    return 0;
}

int gbv_alloc_pd(gbv_context *ctx, gbv_pd **out) {
    if (!ctx || !out) return EINVAL;
    errno = 0;
    struct ibv_pd *pd = ibv_alloc_pd(CTX(ctx));
    *out = (gbv_pd *)pd;
    return pd ? 0 : fail();
}

int gbv_dealloc_pd(gbv_pd *pd) {
    if (!pd) return EINVAL;
    return status(ibv_dealloc_pd(PD(pd)));
}

int gbv_create_cq(gbv_context *ctx, uint32_t depth, gbv_cq **out, uint32_t *actual) {
    if (!ctx || !out || !actual || depth == 0 || depth > INT_MAX) return EINVAL;
    errno = 0;
    struct ibv_cq *cq = ibv_create_cq(CTX(ctx), (int)depth, NULL, NULL, 0);
    *out = (gbv_cq *)cq;
    if (!cq) return fail();
    *actual = cq->cqe > 0 ? (uint32_t)cq->cqe : 0;
    return 0;
}

int gbv_destroy_cq(gbv_cq *cq) {
    if (!cq) return EINVAL;
    return status(ibv_destroy_cq(CQ(cq)));
}

int gbv_create_rc_qp(gbv_pd *pd, gbv_cq *send_cq, gbv_cq *recv_cq, struct gbv_qp_caps *caps,
                     gbv_qp **out, uint32_t *qp_num) {
    if (!pd || !send_cq || !recv_cq || !caps || !out || !qp_num) return EINVAL;
    if (caps->max_send_sge > GBV_MAX_SGE || caps->max_recv_sge > GBV_MAX_SGE) return EINVAL;
    struct ibv_qp_init_attr init;
    memset(&init, 0, sizeof(init));
    init.send_cq = CQ(send_cq);
    init.recv_cq = CQ(recv_cq);
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 1; /* every send completes visibly: the Rust side tracks each one */
    init.cap.max_send_wr = caps->max_send_wr;
    init.cap.max_recv_wr = caps->max_recv_wr;
    init.cap.max_send_sge = caps->max_send_sge;
    init.cap.max_recv_sge = caps->max_recv_sge;
    init.cap.max_inline_data = caps->max_inline_data;
    errno = 0;
    struct ibv_qp *qp = ibv_create_qp(PD(pd), &init);
    *out = (gbv_qp *)qp;
    if (!qp) return fail();
    caps->max_send_wr = init.cap.max_send_wr;
    caps->max_recv_wr = init.cap.max_recv_wr;
    caps->max_send_sge = init.cap.max_send_sge;
    caps->max_recv_sge = init.cap.max_recv_sge;
    caps->max_inline_data = init.cap.max_inline_data;
    *qp_num = qp->qp_num;
    return 0;
}

int gbv_modify_qp_init(gbv_qp *qp, const struct gbv_init_attr *a) {
    if (!qp || !a) return EINVAL;
    int access = 0;
    int rc = access_flags(a->access, &access);
    if (rc) return rc;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = a->pkey_index;
    attr.port_num = a->port_num;
    attr.qp_access_flags = (unsigned int)access;
    return status(ibv_modify_qp(QP(qp), &attr,
                                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                    IBV_QP_ACCESS_FLAGS));
}

int gbv_modify_qp_rtr(gbv_qp *qp, const struct gbv_rtr_attr *a) {
    if (!qp || !a) return EINVAL;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    int rc = mtu_enum(a->path_mtu, &attr.path_mtu);
    if (rc) return rc;
    attr.dest_qp_num = a->dest_qp_num;
    attr.rq_psn = a->rq_psn;
    attr.max_dest_rd_atomic = a->max_dest_rd_atomic;
    attr.min_rnr_timer = a->min_rnr_timer;
    attr.ah_attr.dlid = a->dlid;
    attr.ah_attr.sl = a->sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = a->port_num;
    attr.ah_attr.is_global = a->is_global ? 1 : 0;
    if (a->is_global) {
        _Static_assert(sizeof(attr.ah_attr.grh.dgid.raw) == sizeof(a->dgid), "dgid size");
        memcpy(attr.ah_attr.grh.dgid.raw, a->dgid, sizeof(a->dgid));
        attr.ah_attr.grh.flow_label = a->flow_label;
        attr.ah_attr.grh.sgid_index = a->sgid_index;
        attr.ah_attr.grh.hop_limit = a->hop_limit;
        attr.ah_attr.grh.traffic_class = a->traffic_class;
    }
    return status(ibv_modify_qp(QP(qp), &attr,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                    IBV_QP_MIN_RNR_TIMER));
}

int gbv_modify_qp_rts(gbv_qp *qp, const struct gbv_rts_attr *a) {
    if (!qp || !a) return EINVAL;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = a->sq_psn;
    attr.timeout = a->timeout;
    attr.retry_cnt = a->retry_cnt;
    attr.rnr_retry = a->rnr_retry;
    attr.max_rd_atomic = a->max_rd_atomic;
    return status(ibv_modify_qp(QP(qp), &attr,
                                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
}

static int modify_state(gbv_qp *qp, enum ibv_qp_state state) {
    if (!qp) return EINVAL;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = state;
    return status(ibv_modify_qp(QP(qp), &attr, IBV_QP_STATE));
}

int gbv_modify_qp_error(gbv_qp *qp) { return modify_state(qp, IBV_QPS_ERR); }

int gbv_modify_qp_reset(gbv_qp *qp) { return modify_state(qp, IBV_QPS_RESET); }

int gbv_destroy_qp(gbv_qp *qp) {
    if (!qp) return EINVAL;
    return status(ibv_destroy_qp(QP(qp)));
}

int gbv_reg_mr(gbv_pd *pd, void *addr, size_t len, uint32_t access, gbv_mr **out, uint32_t *lkey,
               uint32_t *rkey) {
    if (!pd || !out || !lkey || !rkey) return EINVAL;
    int flags = 0;
    int rc = access_flags(access, &flags);
    if (rc) return rc;
    errno = 0;
    struct ibv_mr *mr = ibv_reg_mr(PD(pd), addr, len, flags);
    *out = (gbv_mr *)mr;
    if (!mr) return fail();
    *lkey = mr->lkey;
    *rkey = mr->rkey;
    return 0;
}

int gbv_reg_dmabuf_mr(gbv_pd *pd, uint64_t offset, size_t len, uint64_t iova, int fd,
                      uint32_t access, gbv_mr **out, uint32_t *lkey, uint32_t *rkey) {
    if (!pd || !out || !lkey || !rkey || fd < 0) return EINVAL;
    int flags = 0;
    int rc = access_flags(access, &flags);
    if (rc) return rc;
    errno = 0;
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(PD(pd), offset, len, iova, fd, flags);
    *out = (gbv_mr *)mr;
    if (!mr) return fail();
    *lkey = mr->lkey;
    *rkey = mr->rkey;
    return 0;
}

int gbv_dereg_mr(gbv_mr *mr) {
    if (!mr) return EINVAL;
    return status(ibv_dereg_mr(MR(mr)));
}

static int fill_sges(struct ibv_sge *out, const struct gbv_sge *in, uint32_t n) {
    if (n > GBV_MAX_SGE || (n > 0 && !in)) return EINVAL;
    for (uint32_t i = 0; i < n; i++) {
        out[i].addr = in[i].addr;
        out[i].length = in[i].length;
        out[i].lkey = in[i].lkey;
    }
    return 0;
}

/* One send work request from ours; sg must hold w->num_sge entries (already filled). */
static int fill_send(struct ibv_send_wr *wr, const struct gbv_send_wr *w, struct ibv_sge *sg) {
    memset(wr, 0, sizeof(*wr));
    wr->wr_id = w->wr_id;
    wr->sg_list = w->num_sge ? sg : NULL;
    wr->num_sge = (int)w->num_sge;
    wr->send_flags = IBV_SEND_SIGNALED;
    switch (w->opcode) {
    case GBV_WR_SEND: wr->opcode = IBV_WR_SEND; break;
    case GBV_WR_SEND_WITH_IMM:
        wr->opcode = IBV_WR_SEND_WITH_IMM;
        wr->imm_data = htonl(w->imm_data);
        break;
    case GBV_WR_RDMA_WRITE:
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->wr.rdma.remote_addr = w->remote_addr;
        wr->wr.rdma.rkey = w->rkey;
        break;
    case GBV_WR_RDMA_WRITE_WITH_IMM:
        wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr->imm_data = htonl(w->imm_data);
        wr->wr.rdma.remote_addr = w->remote_addr;
        wr->wr.rdma.rkey = w->rkey;
        break;
    default: return EINVAL;
    }
    return 0;
}

int gbv_post_send(gbv_qp *qp, const struct gbv_send_wr *w, const struct gbv_sge *sges) {
    if (!qp || !w) return EINVAL;
    struct ibv_sge sg[GBV_MAX_SGE];
    int rc = fill_sges(sg, sges, w->num_sge);
    if (rc) return rc;
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad = NULL;
    rc = fill_send(&wr, w, sg);
    if (rc) return rc;
    return status(ibv_post_send(QP(qp), &wr, &bad));
}

int gbv_post_send_batch(gbv_qp *qp, const struct gbv_send_wr *wrs, uint32_t n,
                        const struct gbv_sge *sges, uint32_t *posted) {
    if (!qp || !posted || n == 0 || n > GBV_MAX_BATCH || !wrs) return EINVAL;
    *posted = 0;
    struct ibv_send_wr wr[GBV_MAX_BATCH];
    struct ibv_sge sg[GBV_MAX_BATCH_SGE];
    uint32_t used = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = wrs[i].num_sge;
        if (k > GBV_MAX_SGE || used + k > GBV_MAX_BATCH_SGE) return EINVAL;
        int rc = fill_sges(sg + used, sges + used, k);
        if (!rc) rc = fill_send(&wr[i], &wrs[i], sg + used);
        if (rc) return rc;
        wr[i].next = i + 1 < n ? &wr[i + 1] : NULL;
        used += k;
    }
    struct ibv_send_wr *bad = NULL;
    int rc = status(ibv_post_send(QP(qp), wr, &bad));
    /* Requests before bad_wr were posted; bad_wr and after were not. */
    *posted = rc == 0 ? n : (bad ? (uint32_t)(bad - wr) : 0);
    return rc;
}

int gbv_post_recv_batch(gbv_qp *qp, const uint64_t *wr_ids, const uint32_t *num_sges, uint32_t n,
                        const struct gbv_sge *sges, uint32_t *posted) {
    if (!qp || !posted || n == 0 || n > GBV_MAX_BATCH || !wr_ids || !num_sges) return EINVAL;
    *posted = 0;
    struct ibv_recv_wr wr[GBV_MAX_BATCH];
    struct ibv_sge sg[GBV_MAX_BATCH_SGE];
    uint32_t used = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = num_sges[i];
        if (k > GBV_MAX_SGE || used + k > GBV_MAX_BATCH_SGE) return EINVAL;
        int rc = fill_sges(sg + used, sges + used, k);
        if (rc) return rc;
        memset(&wr[i], 0, sizeof(wr[i]));
        wr[i].wr_id = wr_ids[i];
        wr[i].sg_list = k ? sg + used : NULL;
        wr[i].num_sge = (int)k;
        wr[i].next = i + 1 < n ? &wr[i + 1] : NULL;
        used += k;
    }
    struct ibv_recv_wr *bad = NULL;
    int rc = status(ibv_post_recv(QP(qp), wr, &bad));
    *posted = rc == 0 ? n : (bad ? (uint32_t)(bad - wr) : 0);
    return rc;
}

int gbv_post_recv(gbv_qp *qp, uint64_t wr_id, const struct gbv_sge *sges, uint32_t num_sge) {
    if (!qp) return EINVAL;
    struct ibv_sge sg[GBV_MAX_SGE];
    int rc = fill_sges(sg, sges, num_sge);
    if (rc) return rc;
    struct ibv_recv_wr wr;
    struct ibv_recv_wr *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = num_sge ? sg : NULL;
    wr.num_sge = (int)num_sge;
    return status(ibv_post_recv(QP(qp), &wr, &bad));
}

static uint32_t wc_status(enum ibv_wc_status s) {
    switch (s) {
    case IBV_WC_SUCCESS: return GBV_WC_SUCCESS;
    case IBV_WC_LOC_LEN_ERR: return GBV_WC_LOC_LEN_ERR;
    case IBV_WC_LOC_QP_OP_ERR: return GBV_WC_LOC_QP_OP_ERR;
    case IBV_WC_LOC_PROT_ERR: return GBV_WC_LOC_PROT_ERR;
    case IBV_WC_WR_FLUSH_ERR: return GBV_WC_WR_FLUSH_ERR;
    case IBV_WC_BAD_RESP_ERR: return GBV_WC_BAD_RESP_ERR;
    case IBV_WC_LOC_ACCESS_ERR: return GBV_WC_LOC_ACCESS_ERR;
    case IBV_WC_REM_INV_REQ_ERR: return GBV_WC_REM_INV_REQ_ERR;
    case IBV_WC_REM_ACCESS_ERR: return GBV_WC_REM_ACCESS_ERR;
    case IBV_WC_REM_OP_ERR: return GBV_WC_REM_OP_ERR;
    case IBV_WC_RETRY_EXC_ERR: return GBV_WC_RETRY_EXC_ERR;
    case IBV_WC_RNR_RETRY_EXC_ERR: return GBV_WC_RNR_RETRY_EXC_ERR;
    case IBV_WC_GENERAL_ERR: return GBV_WC_GENERAL_ERR;
    default: return GBV_WC_OTHER;
    }
}

static uint32_t wc_opcode(enum ibv_wc_opcode op) {
    switch (op) {
    case IBV_WC_SEND: return GBV_WC_OP_SEND;
    case IBV_WC_RDMA_WRITE: return GBV_WC_OP_RDMA_WRITE;
    case IBV_WC_RECV: return GBV_WC_OP_RECV;
    case IBV_WC_RECV_RDMA_WITH_IMM: return GBV_WC_OP_RECV_RDMA_WITH_IMM;
    default: return GBV_WC_OP_OTHER;
    }
}

int gbv_poll_cq(gbv_cq *cq, struct gbv_wc *out, uint32_t max, uint32_t *n) {
    if (!cq || !n || (max > 0 && !out)) return EINVAL;
    *n = 0;
    struct ibv_wc wc[16];
    while (*n < max) {
        uint32_t want = max - *n;
        if (want > 16) want = 16;
        int got = ibv_poll_cq(CQ(cq), (int)want, wc);
        /* ibv_poll_cq reports no errno. Completions already taken are returned first; the
         * next call reports the error. */
        if (got < 0) return *n > 0 ? 0 : EIO;
        for (int i = 0; i < got; i++) {
            struct gbv_wc *o = &out[*n + (uint32_t)i];
            memset(o, 0, sizeof(*o));
            o->wr_id = wc[i].wr_id;
            o->raw_status = (uint32_t)wc[i].status;
            o->status = wc_status(wc[i].status);
            /* opcode and byte_len are defined only for successful completions. */
            o->opcode = wc[i].status == IBV_WC_SUCCESS ? wc_opcode(wc[i].opcode) : GBV_WC_OP_OTHER;
            o->byte_len = wc[i].byte_len;
            o->qp_num = wc[i].qp_num;
            o->vendor_err = wc[i].vendor_err;
            if (wc[i].status == IBV_WC_SUCCESS && (wc[i].wc_flags & IBV_WC_WITH_IMM)) {
                o->has_imm = 1;
                o->imm_data = ntohl(wc[i].imm_data);
            }
        }
        *n += (uint32_t)got;
        if ((uint32_t)got < want) break;
    }
    return 0;
}

int gbv_create_comp_channel(gbv_context *ctx, gbv_channel **out) {
    if (!ctx || !out) return EINVAL;
    errno = 0;
    struct ibv_comp_channel *channel = ibv_create_comp_channel(CTX(ctx));
    *out = NULL;
    if (!channel) return fail();
    int rc = nonblocking(channel->fd);
    if (rc) {
        ibv_destroy_comp_channel(channel);
        return rc;
    }
    *out = (gbv_channel *)channel;
    return 0;
}

int gbv_destroy_comp_channel(gbv_channel *channel) {
    if (!channel) return EINVAL;
    return status(ibv_destroy_comp_channel(CH(channel)));
}

int gbv_create_cq_on_channel(gbv_context *ctx, uint32_t depth, gbv_channel *channel,
                             gbv_cq **out, uint32_t *actual) {
    if (!ctx || !channel || !out || !actual || depth == 0 || depth > INT_MAX) return EINVAL;
    errno = 0;
    struct ibv_cq *cq = ibv_create_cq(CTX(ctx), (int)depth, NULL, CH(channel), 0);
    *out = (gbv_cq *)cq;
    if (!cq) return fail();
    *actual = cq->cqe > 0 ? (uint32_t)cq->cqe : 0;
    return 0;
}

int gbv_req_notify_cq(gbv_cq *cq) {
    if (!cq) return EINVAL;
    return status(ibv_req_notify_cq(CQ(cq), 0));
}

static uint32_t event_type(enum ibv_event_type t, uint8_t *has_qp, uint8_t *has_port) {
    *has_qp = 0;
    *has_port = 0;
    switch (t) {
    case IBV_EVENT_CQ_ERR: return GBV_EV_CQ_ERR;
    case IBV_EVENT_QP_FATAL: *has_qp = 1; return GBV_EV_QP_FATAL;
    case IBV_EVENT_QP_REQ_ERR: *has_qp = 1; return GBV_EV_QP_REQ_ERR;
    case IBV_EVENT_QP_ACCESS_ERR: *has_qp = 1; return GBV_EV_QP_ACCESS_ERR;
    case IBV_EVENT_COMM_EST: *has_qp = 1; return GBV_EV_COMM_EST;
    case IBV_EVENT_SQ_DRAINED: *has_qp = 1; return GBV_EV_SQ_DRAINED;
    case IBV_EVENT_PATH_MIG: *has_qp = 1; return GBV_EV_PATH_MIG;
    case IBV_EVENT_PATH_MIG_ERR: *has_qp = 1; return GBV_EV_PATH_MIG_ERR;
    case IBV_EVENT_QP_LAST_WQE_REACHED: *has_qp = 1; return GBV_EV_QP_LAST_WQE_REACHED;
    case IBV_EVENT_DEVICE_FATAL: return GBV_EV_DEVICE_FATAL;
    case IBV_EVENT_PORT_ACTIVE: *has_port = 1; return GBV_EV_PORT_ACTIVE;
    case IBV_EVENT_PORT_ERR: *has_port = 1; return GBV_EV_PORT_ERR;
    case IBV_EVENT_LID_CHANGE: *has_port = 1; return GBV_EV_LID_CHANGE;
    case IBV_EVENT_PKEY_CHANGE: *has_port = 1; return GBV_EV_PKEY_CHANGE;
    case IBV_EVENT_SM_CHANGE: *has_port = 1; return GBV_EV_SM_CHANGE;
    case IBV_EVENT_CLIENT_REREGISTER: *has_port = 1; return GBV_EV_CLIENT_REREGISTER;
    case IBV_EVENT_GID_CHANGE: *has_port = 1; return GBV_EV_GID_CHANGE;
    case IBV_EVENT_SRQ_ERR: return GBV_EV_SRQ_ERR;
    case IBV_EVENT_SRQ_LIMIT_REACHED: return GBV_EV_SRQ_LIMIT_REACHED;
    default: return GBV_EV_OTHER;
    }
}

int gbv_wait_events(gbv_context *ctx, gbv_channel *channel, int timeout_ms, uint32_t *completion,
                    struct gbv_async_event *events, uint32_t max_events, uint32_t *n_events) {
    if (!ctx || !channel || !completion || !n_events || (max_events > 0 && !events))
        return EINVAL;
    *completion = 0;
    *n_events = 0;
    struct pollfd fds[2];
    fds[0].fd = CH(channel)->fd;
    fds[0].events = POLLIN;
    fds[0].revents = 0;
    fds[1].fd = CTX(ctx)->async_fd;
    fds[1].events = POLLIN;
    fds[1].revents = 0;
    int ready = poll(fds, 2, timeout_ms < 0 ? 0 : timeout_ms);
    if (ready < 0) return errno == EINTR ? 0 : fail();
    if (ready == 0) return 0;
    if ((fds[0].revents | fds[1].revents) & (POLLERR | POLLNVAL)) return EIO;
    if (fds[0].revents & POLLIN) {
        struct ibv_cq *cq = NULL;
        void *cq_context = NULL;
        errno = 0;
        if (ibv_get_cq_event(CH(channel), &cq, &cq_context) == 0) {
            /* Acknowledged at once: ibv_destroy_cq waits for every event to be. */
            ibv_ack_cq_events(cq, 1);
            *completion = 1;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            return fail();
        }
    }
    if (fds[1].revents & POLLIN) {
        while (*n_events < max_events) {
            struct ibv_async_event ev;
            memset(&ev, 0, sizeof(ev));
            if (ibv_get_async_event(CTX(ctx), &ev) != 0) break; /* EAGAIN: no more */
            struct gbv_async_event *o = &events[*n_events];
            memset(o, 0, sizeof(*o));
            o->raw_type = (uint32_t)ev.event_type;
            o->type = event_type(ev.event_type, &o->has_qp, &o->has_port);
            if (o->has_qp && ev.element.qp) o->qp_num = ev.element.qp->qp_num;
            else o->has_qp = 0;
            if (o->has_port) o->port_num = (uint8_t)ev.element.port_num;
            ibv_ack_async_event(&ev);
            *n_events += 1;
        }
    }
    return 0;
}

int gbv_library_path(char *out, size_t len) {
    if (!out || len == 0) return EINVAL;
    Dl_info info;
    memset(&info, 0, sizeof(info));
    /* Any function libibverbs.so.1 itself defines (not a header inline). */
    if (!dladdr((void *)&ibv_get_device_list, &info) || !info.dli_fname) return ENOENT;
    char resolved[PATH_MAX];
    const char *path = realpath(info.dli_fname, resolved) ? resolved : info.dli_fname;
    size_t l = strlen(path);
    if (l + 1 > len) return ENOSPC;
    memcpy(out, path, l + 1);
    return 0;
}
