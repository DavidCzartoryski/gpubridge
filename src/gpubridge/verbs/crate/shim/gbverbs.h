/*
 * The C shim between gpubridge-verbs (Rust) and libibverbs (docs/VERBS_TRANSPORT.md).
 *
 * Rust never sees a libibverbs struct, enum or constant. It holds the opaque pointers
 * below and fills or reads the gbv_* structs defined here; gbverbs.c, compiled against
 * <infiniband/verbs.h>, translates them. Every struct here has a fixed layout of fixed-width
 * fields, checked with _Static_assert in gbverbs.c and with the same numbers in Rust
 * (src/ibv.rs), and gbv_layout() reports them at run time for a test to compare.
 *
 * Every function returns 0 on success or a positive errno value.
 */
#ifndef GBVERBS_H
#define GBVERBS_H

#include <stddef.h>
#include <stdint.h>

#define GBV_ABI_VERSION 2u
#define GBV_MAX_SGE 16u
/* At most this many work requests per batch post, and this many scatter-gather entries in
 * all of them together. */
#define GBV_MAX_BATCH 64u
#define GBV_MAX_BATCH_SGE 256u

/* Opaque handles: struct ibv_context, ibv_pd, ibv_cq, ibv_qp, ibv_mr. Never defined. */
typedef struct gbv_context gbv_context;
typedef struct gbv_pd gbv_pd;
typedef struct gbv_cq gbv_cq;
typedef struct gbv_qp gbv_qp;
typedef struct gbv_mr gbv_mr;
/* struct ibv_comp_channel. */
typedef struct gbv_channel gbv_channel;

/* Access flags (our bits; translated to IBV_ACCESS_*). */
#define GBV_ACCESS_LOCAL_WRITE 1u
#define GBV_ACCESS_REMOTE_WRITE 2u
#define GBV_ACCESS_REMOTE_READ 4u
#define GBV_ACCESS_RELAXED_ORDERING 8u

/* Port states and link layers. */
enum { GBV_PORT_DOWN = 1, GBV_PORT_INIT = 2, GBV_PORT_ARMED = 3, GBV_PORT_ACTIVE = 4,
       GBV_PORT_OTHER = 5 };
enum { GBV_LINK_INFINIBAND = 1, GBV_LINK_ETHERNET = 2 };
enum { GBV_GID_IB = 1, GBV_GID_ROCE_V1 = 2, GBV_GID_ROCE_V2 = 3, GBV_GID_OTHER = 4 };

/* Send opcodes. */
enum { GBV_WR_SEND = 1, GBV_WR_SEND_WITH_IMM = 2, GBV_WR_RDMA_WRITE = 3,
       GBV_WR_RDMA_WRITE_WITH_IMM = 4 };

/* Completion statuses: our numbering, translated from enum ibv_wc_status. */
enum {
    GBV_WC_SUCCESS = 0,
    GBV_WC_LOC_LEN_ERR = 1,
    GBV_WC_LOC_QP_OP_ERR = 2,
    GBV_WC_LOC_PROT_ERR = 3,
    GBV_WC_WR_FLUSH_ERR = 4,
    GBV_WC_BAD_RESP_ERR = 5,
    GBV_WC_LOC_ACCESS_ERR = 6,
    GBV_WC_REM_INV_REQ_ERR = 7,
    GBV_WC_REM_ACCESS_ERR = 8,
    GBV_WC_REM_OP_ERR = 9,
    GBV_WC_RETRY_EXC_ERR = 10,
    GBV_WC_RNR_RETRY_EXC_ERR = 11,
    GBV_WC_GENERAL_ERR = 12,
    GBV_WC_OTHER = 13 /* anything else: raw_status has libibverbs' value */
};

/* Completion opcodes (valid only when status is GBV_WC_SUCCESS). */
enum { GBV_WC_OP_SEND = 1, GBV_WC_OP_RDMA_WRITE = 2, GBV_WC_OP_RECV = 3,
       GBV_WC_OP_RECV_RDMA_WITH_IMM = 4, GBV_WC_OP_OTHER = 5 };

struct gbv_device_attr {
    uint64_t max_mr_size;
    uint64_t node_guid; /* host byte order */
    uint32_t max_qp;
    uint32_t max_qp_wr;
    uint32_t max_sge;
    uint32_t max_cq;
    uint32_t max_cqe;
    uint32_t max_mr;
    uint32_t max_pd;
    uint32_t max_qp_rd_atom;
    uint32_t max_qp_init_rd_atom;
    uint32_t vendor_id;
    uint32_t vendor_part_id;
    uint8_t phys_port_cnt;
    uint8_t pad[3];
    char fw_ver[64];
};

struct gbv_port_attr {
    uint32_t state;      /* GBV_PORT_* */
    uint32_t max_mtu;    /* bytes */
    uint32_t active_mtu; /* bytes */
    uint32_t gid_tbl_len;
    uint16_t lid;
    uint16_t sm_lid;
    uint8_t link_layer; /* GBV_LINK_* */
    uint8_t active_width;
    uint8_t active_speed;
    uint8_t pad;
};

struct gbv_gid {
    uint8_t raw[16];
    uint32_t gid_type; /* GBV_GID_* */
    uint32_t ndev_ifindex;
};

struct gbv_qp_caps {
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

struct gbv_init_attr {
    uint32_t access; /* GBV_ACCESS_* */
    uint16_t pkey_index;
    uint8_t port_num;
    uint8_t pad;
};

struct gbv_rtr_attr {
    uint8_t dgid[16];
    uint32_t dest_qp_num;
    uint32_t rq_psn;
    uint32_t path_mtu; /* bytes */
    uint32_t flow_label;
    uint16_t dlid;
    uint8_t max_dest_rd_atomic;
    uint8_t min_rnr_timer;
    uint8_t port_num;
    uint8_t sl;
    uint8_t is_global;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
    uint8_t pad[2];
};

struct gbv_rts_attr {
    uint32_t sq_psn;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t max_rd_atomic;
};

struct gbv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

struct gbv_send_wr {
    uint64_t wr_id;
    uint64_t remote_addr;
    uint32_t opcode; /* GBV_WR_* */
    uint32_t rkey;
    uint32_t imm_data; /* host byte order; sent in network order */
    uint32_t num_sge;
};

struct gbv_wc {
    uint64_t wr_id;
    uint32_t status; /* GBV_WC_* */
    uint32_t opcode; /* GBV_WC_OP_* */
    uint32_t byte_len;
    uint32_t qp_num;
    uint32_t imm_data; /* host byte order, when has_imm */
    uint32_t vendor_err;
    uint32_t raw_status; /* libibverbs' enum ibv_wc_status value */
    uint32_t has_imm;
};

/* Asynchronous event types: our numbering, translated from enum ibv_event_type. */
enum {
    GBV_EV_CQ_ERR = 1,
    GBV_EV_QP_FATAL = 2,
    GBV_EV_QP_REQ_ERR = 3,
    GBV_EV_QP_ACCESS_ERR = 4,
    GBV_EV_COMM_EST = 5,
    GBV_EV_SQ_DRAINED = 6,
    GBV_EV_PATH_MIG = 7,
    GBV_EV_PATH_MIG_ERR = 8,
    GBV_EV_DEVICE_FATAL = 9,
    GBV_EV_PORT_ACTIVE = 10,
    GBV_EV_PORT_ERR = 11,
    GBV_EV_LID_CHANGE = 12,
    GBV_EV_PKEY_CHANGE = 13,
    GBV_EV_SM_CHANGE = 14,
    GBV_EV_SRQ_ERR = 15,
    GBV_EV_SRQ_LIMIT_REACHED = 16,
    GBV_EV_QP_LAST_WQE_REACHED = 17,
    GBV_EV_CLIENT_REREGISTER = 18,
    GBV_EV_GID_CHANGE = 19,
    GBV_EV_OTHER = 20 /* raw_type has libibverbs' value */
};

struct gbv_async_event {
    uint32_t type;     /* GBV_EV_* */
    uint32_t raw_type; /* libibverbs' enum ibv_event_type value */
    uint32_t qp_num;   /* when has_qp */
    uint8_t port_num;  /* when has_port */
    uint8_t has_qp;
    uint8_t has_port;
    uint8_t pad;
};

/* Layout ids for gbv_layout(): a struct's size, or a field's offset. */
enum {
    GBV_L_DEVICE_ATTR = 1, GBV_L_DEVICE_ATTR_FW_VER,
    GBV_L_PORT_ATTR, GBV_L_PORT_ATTR_LID, GBV_L_PORT_ATTR_LINK_LAYER,
    GBV_L_GID, GBV_L_GID_TYPE,
    GBV_L_QP_CAPS, GBV_L_QP_CAPS_INLINE,
    GBV_L_INIT_ATTR, GBV_L_INIT_ATTR_PORT,
    GBV_L_RTR_ATTR, GBV_L_RTR_ATTR_DEST_QP_NUM, GBV_L_RTR_ATTR_DLID, GBV_L_RTR_ATTR_TRAFFIC_CLASS,
    GBV_L_RTS_ATTR, GBV_L_RTS_ATTR_MAX_RD_ATOMIC,
    GBV_L_SGE, GBV_L_SGE_LKEY,
    GBV_L_SEND_WR, GBV_L_SEND_WR_NUM_SGE,
    GBV_L_WC, GBV_L_WC_HAS_IMM,
    GBV_L_ASYNC_EVENT, GBV_L_ASYNC_EVENT_HAS_PORT,
    GBV_L_END
};

uint32_t gbv_abi_version(void);
/* A struct's size or a field's offset (GBV_L_*), or UINT64_MAX for an unknown id. */
uint64_t gbv_layout(uint32_t id);
/* libibverbs' own text for a raw completion status (ibv_wc_status_str). Opens nothing. */
const char *gbv_wc_status_str(uint32_t raw_status);

/* Device names, each followed by '\n', into names (NUL-terminated); ENOSPC if too small. */
int gbv_list_devices(char *names, size_t len, uint32_t *count);
int gbv_open_device(const char *name, gbv_context **out);
int gbv_close_device(gbv_context *ctx);
int gbv_query_device(gbv_context *ctx, struct gbv_device_attr *out);
int gbv_query_port(gbv_context *ctx, uint8_t port, struct gbv_port_attr *out);
int gbv_query_gid(gbv_context *ctx, uint8_t port, uint32_t index, struct gbv_gid *out);

int gbv_alloc_pd(gbv_context *ctx, gbv_pd **out);
int gbv_dealloc_pd(gbv_pd *pd);

int gbv_create_cq(gbv_context *ctx, uint32_t depth, gbv_cq **out, uint32_t *actual);
int gbv_destroy_cq(gbv_cq *cq);
/* Only the calling native progress thread; validate against its inherited allowed set. */
int gbv_progress_cpu(uint32_t cpu, int apply);

/* An RC queue pair, every send signaled. caps: what to ask for in, what was given out. */
int gbv_create_rc_qp(gbv_pd *pd, gbv_cq *send_cq, gbv_cq *recv_cq, struct gbv_qp_caps *caps,
                     gbv_qp **out, uint32_t *qp_num);
int gbv_modify_qp_init(gbv_qp *qp, const struct gbv_init_attr *attr);
int gbv_modify_qp_rtr(gbv_qp *qp, const struct gbv_rtr_attr *attr);
int gbv_modify_qp_rts(gbv_qp *qp, const struct gbv_rts_attr *attr);
int gbv_modify_qp_error(gbv_qp *qp);
int gbv_modify_qp_reset(gbv_qp *qp);
int gbv_destroy_qp(gbv_qp *qp);

int gbv_reg_mr(gbv_pd *pd, void *addr, size_t len, uint32_t access, gbv_mr **out, uint32_t *lkey,
               uint32_t *rkey);
int gbv_reg_dmabuf_mr(gbv_pd *pd, uint64_t offset, size_t len, uint64_t iova, int fd,
                      uint32_t access, gbv_mr **out, uint32_t *lkey, uint32_t *rkey);
int gbv_dereg_mr(gbv_mr *mr);

/* One work request each; sges has wr->num_sge (at most GBV_MAX_SGE) entries. */
int gbv_post_send(gbv_qp *qp, const struct gbv_send_wr *wr, const struct gbv_sge *sges);
int gbv_post_recv(gbv_qp *qp, uint64_t wr_id, const struct gbv_sge *sges, uint32_t num_sge);
/* Up to max completions into out; *n says how many. */
int gbv_poll_cq(gbv_cq *cq, struct gbv_wc *out, uint32_t max, uint32_t *n);

/* n (at most GBV_MAX_BATCH) work requests in one ibv_post_send / ibv_post_recv call,
 * chained in order. sges holds every request's entries one after the other (at most
 * GBV_MAX_BATCH_SGE). On an error, *posted says how many were posted before the one that
 * failed (libibverbs' bad_wr); on success it is n. */
int gbv_post_send_batch(gbv_qp *qp, const struct gbv_send_wr *wrs, uint32_t n,
                        const struct gbv_sge *sges, uint32_t *posted);
int gbv_post_recv_batch(gbv_qp *qp, const uint64_t *wr_ids, const uint32_t *num_sges, uint32_t n,
                        const struct gbv_sge *sges, uint32_t *posted);

/* A completion channel, its descriptor non-blocking (gbv_open_device makes the context's
 * asynchronous event descriptor non-blocking too). */
int gbv_create_comp_channel(gbv_context *ctx, gbv_channel **out);
/* Only after every completion queue on it is destroyed. */
int gbv_destroy_comp_channel(gbv_channel *channel);
int gbv_create_cq_on_channel(gbv_context *ctx, uint32_t depth, gbv_channel *channel,
                             gbv_cq **out, uint32_t *actual);
/* One notification for the next completion (ibv_req_notify_cq, any completion). */
int gbv_req_notify_cq(gbv_cq *cq);
/* Waits up to timeout_ms (poll(2) on the channel's and the context's descriptors) for a
 * completion notification or asynchronous events, then takes and acknowledges them:
 * *completion is 1 if a notification came; up to max_events events go into events, *n_events
 * says how many. An interrupted wait returns 0 with nothing. */
int gbv_wait_events(gbv_context *ctx, gbv_channel *channel, int timeout_ms, uint32_t *completion,
                    struct gbv_async_event *events, uint32_t max_events, uint32_t *n_events);
/* The file libibverbs was loaded from, resolved (its name carries rdma-core's version, e.g.
 * libibverbs.so.1.14.57.0), NUL-terminated; ENOENT if unknown, ENOSPC if len is too small. */
int gbv_library_path(char *out, size_t len);

/* Fork safety of registered memory: ibv_is_fork_initialized(), looked up at run time (it is in
 * rdma-core 35 and later), as GBV_FORK_*; GBV_FORK_UNKNOWN where this libibverbs lacks it. */
enum { GBV_FORK_DISABLED = 1, GBV_FORK_ENABLED = 2, GBV_FORK_UNNEEDED = 3, GBV_FORK_UNKNOWN = 4 };
int gbv_fork_status(uint32_t *out);
/* ibv_fork_init(): only before any memory is registered in the process (EINVAL after). */
int gbv_fork_init(void);

#endif
