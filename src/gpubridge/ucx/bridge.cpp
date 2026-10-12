// UCX endpoints for gpubridge's "ucx" bridge transport (docs/RDMA_TRANSPORT.md).
//
// A thin layer over UCP tagged messages with a C interface, loaded with ctypes
// by gpubridge/ucx/__init__.py. It moves bytes between raw pointers and
// reports completion; it knows nothing about PyTorch, so one build works under
// both the CUDA and the ROCm builds of PyTorch. Algorithms (exchange, ring),
// reductions and stream ordering live in Python.
//
// Every call returns 0 (or a non-negative id) on success and a negative value
// on failure, with the reason in gb_ucx_last_error(). Every wait has a deadline.
// Calls must not overlap on one handle (UCS_THREAD_MODE_SERIALIZED): gpubridge
// runs a rank's collectives one at a time, possibly from different threads.

#include <ucp/api/ucp.h>

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum MemoryType : int { kHost = 0, kCuda = 1, kRocm = 2, kDetect = -1 };

// Error codes returned to Python.
enum Error : int {
    kFailed = -1,       // UCX or argument error; see gb_ucx_last_error()
    kTimedOut = -2,     // the deadline passed with operations still outstanding
    kPeerFailed = -3,   // an endpoint reported an error (peer gone, link down)
};

struct Op {
    void* request = nullptr;  // null once complete
    ucs_status_t status = UCS_OK;
    int peer = -1;            // -1 for a receive (any peer: the tag says which)
    size_t bytes = 0;
    size_t received = 0;      // a receive's actual length, once complete
    bool send = false;
};

struct Peer {
    ucp_ep_h ep = nullptr;
    ucs_status_t failed = UCS_OK;  // set by the error callback
};

struct Handle {
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
    std::vector<Peer> peers;
    std::vector<Op> ops;
    std::string error;
    bool broken = false;  // after a timeout or peer failure, every later call fails
};

double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int fail(Handle* h, int code, const char* format, ...) {
    char text[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof text, format, args);
    va_end(args);
    if (h != nullptr) {
        h->error = text;
        if (code == kTimedOut || code == kPeerFailed) h->broken = true;
    }
    return code;
}

void on_endpoint_error(void* arg, ucp_ep_h, ucs_status_t status) {
    auto* peer = static_cast<Peer*>(arg);
    peer->failed = status;
}

ucs_memory_type_t to_ucs(int memory_type) {
    switch (memory_type) {
        case kCuda: return UCS_MEMORY_TYPE_CUDA;
        case kRocm: return UCS_MEMORY_TYPE_ROCM;
        default: return UCS_MEMORY_TYPE_HOST;
    }
}

void cancel_all(Handle* h, double timeout_s);

int64_t track(Handle* h, ucs_status_ptr_t result, int peer, size_t bytes, bool send) {
    if (UCS_PTR_IS_ERR(result)) {
        // Whatever this step already posted can't complete as intended: cancel it now,
        // so no request outlives the buffers it points to, and refuse later calls.
        cancel_all(h, 1.0);
        int code = fail(h, kFailed, "UCX %s of %zu bytes%s failed to start: %s",
                        send ? "send" : "receive", bytes, send ? " to a peer" : "",
                        ucs_status_string(UCS_PTR_STATUS(result)));
        h->broken = true;
        h->ops.clear();
        return code;
    }
    Op op;
    op.request = result;  // null: already complete
    op.peer = peer;
    op.bytes = bytes;
    op.send = send;
    h->ops.push_back(op);
    return static_cast<int64_t>(h->ops.size() - 1);
}

// Progress until no request is outstanding or the deadline passes.
int progress_until(Handle* h, const std::vector<int64_t>& ids, double deadline) {
    for (;;) {
        bool pending = false;
        for (int64_t id : ids) {
            Op& op = h->ops[static_cast<size_t>(id)];
            if (op.request == nullptr) continue;
            ucs_status_t status;
            if (op.send) {
                status = ucp_request_check_status(op.request);
            } else {
                ucp_tag_recv_info_t info{};
                status = ucp_tag_recv_request_test(op.request, &info);
                if (status == UCS_OK) op.received = info.length;
            }
            if (status == UCS_INPROGRESS) {
                pending = true;
                continue;
            }
            op.status = status;
            ucp_request_free(op.request);
            op.request = nullptr;
        }
        for (int64_t id : ids) {
            const Op& op = h->ops[static_cast<size_t>(id)];
            if (!op.send && op.request == nullptr && op.status == UCS_OK &&
                op.received != op.bytes) {
                return fail(h, kFailed,
                            "a UCX message of %zu bytes arrived where %zu were expected: the "
                            "leaders don't agree on how to split messages", op.received,
                            op.bytes);
            }
        }
        for (int64_t id : ids) {
            const Op& op = h->ops[static_cast<size_t>(id)];
            if (op.request == nullptr && op.status != UCS_OK) {
                int code = op.status == UCS_ERR_CANCELED ? kFailed : kPeerFailed;
                return fail(h, code, "UCX %s of %zu bytes failed: %s",
                            op.send ? "send" : "receive", op.bytes,
                            ucs_status_string(op.status));
            }
        }
        if (!pending) return 0;
        for (size_t p = 0; p < h->peers.size(); ++p) {
            if (h->peers[p].failed != UCS_OK) {
                return fail(h, kPeerFailed, "the UCX endpoint to bridge member %zu failed: %s",
                            p, ucs_status_string(h->peers[p].failed));
            }
        }
        if (now() > deadline) {
            size_t bytes = 0;
            int outstanding = 0;
            for (int64_t id : ids) {
                const Op& op = h->ops[static_cast<size_t>(id)];
                if (op.request != nullptr) {
                    bytes += op.bytes;
                    ++outstanding;
                }
            }
            return fail(h, kTimedOut,
                        "timed out with %d UCX operation(s) outstanding (%zu bytes)",
                        outstanding, bytes);
        }
        if (ucp_worker_progress(h->worker) == 0) sched_yield();
    }
}

void cancel_all(Handle* h, double timeout_s) {
    double deadline = now() + timeout_s;
    for (Op& op : h->ops) {
        if (op.request != nullptr) ucp_request_cancel(h->worker, op.request);
    }
    for (Op& op : h->ops) {
        while (op.request != nullptr &&
               ucp_request_check_status(op.request) == UCS_INPROGRESS && now() < deadline) {
            ucp_worker_progress(h->worker);
        }
        if (op.request != nullptr) {
            ucp_request_free(op.request);
            op.request = nullptr;
        }
    }
}

}  // namespace

extern "C" {

const char* gb_ucx_last_error(void* handle) {
    return handle == nullptr ? "no handle" : static_cast<Handle*>(handle)->error.c_str();
}

void gb_ucx_version(unsigned* major, unsigned* minor, unsigned* release) {
    ucp_get_version(major, minor, release);
}

// Context and worker. `error` receives the reason if this returns null.
void* gb_ucx_create(int peers, char* error, size_t error_len) {
    auto* h = new Handle();
    ucp_config_t* config = nullptr;
    ucs_status_t status = ucp_config_read(nullptr, nullptr, &config);
    if (status == UCS_OK) {
        ucp_params_t params{};
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features = UCP_FEATURE_TAG;
        status = ucp_init(&params, config, &h->context);
        ucp_config_release(config);
    }
    if (status == UCS_OK) {
        ucp_worker_params_t params{};
        params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
        status = ucp_worker_create(h->context, &params, &h->worker);
    }
    if (status != UCS_OK) {
        std::snprintf(error, error_len, "UCX initialization failed: %s",
                      ucs_status_string(status));
        if (h->context != nullptr) ucp_cleanup(h->context);
        delete h;
        return nullptr;
    }
    h->peers.resize(static_cast<size_t>(peers));
    return h;
}

// Bit 0: host memory, bit 1: CUDA, bit 2: ROCm.
int gb_ucx_memory_types(void* handle, uint64_t* types) {
    auto* h = static_cast<Handle*>(handle);
    ucp_context_attr_t attr{};
    attr.field_mask = UCP_ATTR_FIELD_MEMORY_TYPES;
    ucs_status_t status = ucp_context_query(h->context, &attr);
    if (status != UCS_OK) {
        return fail(h, kFailed, "ucp_context_query failed: %s", ucs_status_string(status));
    }
    *types = 0;
    if (attr.memory_types & UCS_BIT(UCS_MEMORY_TYPE_HOST)) *types |= 1;
    if (attr.memory_types & UCS_BIT(UCS_MEMORY_TYPE_CUDA)) *types |= 2;
    if (attr.memory_types & UCS_BIT(UCS_MEMORY_TYPE_ROCM)) *types |= 4;
    return 0;
}

// The worker's address; the caller copies it, then frees it with gb_ucx_free_address.
int gb_ucx_address(void* handle, void** address, size_t* length) {
    auto* h = static_cast<Handle*>(handle);
    ucp_address_t* raw = nullptr;
    ucs_status_t status = ucp_worker_get_address(h->worker, &raw, length);
    if (status != UCS_OK) {
        return fail(h, kFailed, "ucp_worker_get_address failed: %s", ucs_status_string(status));
    }
    *address = raw;
    return 0;
}

void gb_ucx_free_address(void* handle, void* address) {
    ucp_worker_release_address(static_cast<Handle*>(handle)->worker,
                               static_cast<ucp_address_t*>(address));
}

// An endpoint to bridge member `peer`, from its worker address.
int gb_ucx_connect(void* handle, int peer, const void* address) {
    auto* h = static_cast<Handle*>(handle);
    if (peer < 0 || static_cast<size_t>(peer) >= h->peers.size()) {
        return fail(h, kFailed, "no bridge member %d", peer);
    }
    Peer& p = h->peers[static_cast<size_t>(peer)];
    ucp_ep_params_t params{};
    params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                        UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params.address = static_cast<const ucp_address_t*>(address);
    params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    params.err_handler.cb = on_endpoint_error;
    params.err_handler.arg = &p;
    ucs_status_t status = ucp_ep_create(h->worker, &params, &p.ep);
    if (status != UCS_OK) {
        return fail(h, kFailed, "ucp_ep_create to bridge member %d failed: %s", peer,
                    ucs_status_string(status));
    }
    return 0;
}

// Post a send to `peer`; returns an operation id for gb_ucx_wait.
int64_t gb_ucx_send(void* handle, int peer, const void* buffer, size_t bytes, uint64_t tag,
                    int memory_type) {
    auto* h = static_cast<Handle*>(handle);
    if (h->broken) return fail(h, kFailed, "the UCX bridge already failed: %s", h->error.c_str());
    if (peer < 0 || static_cast<size_t>(peer) >= h->peers.size() ||
        h->peers[static_cast<size_t>(peer)].ep == nullptr) {
        return fail(h, kFailed, "no endpoint to bridge member %d", peer);
    }
    ucp_request_param_t param{};
    if (memory_type != kDetect) {
        param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE;
        param.memory_type = to_ucs(memory_type);
    }
    ucs_status_ptr_t result = ucp_tag_send_nbx(h->peers[static_cast<size_t>(peer)].ep, buffer,
                                               bytes, tag, &param);
    return track(h, result, peer, bytes, true);
}

// Post a receive of exactly `tag`; returns an operation id for gb_ucx_wait.
int64_t gb_ucx_recv(void* handle, void* buffer, size_t bytes, uint64_t tag, int memory_type) {
    auto* h = static_cast<Handle*>(handle);
    if (h->broken) return fail(h, kFailed, "the UCX bridge already failed: %s", h->error.c_str());
    ucp_request_param_t param{};
    ucp_tag_recv_info_t info{};  // filled only if it completes right away
    param.op_attr_mask = UCP_OP_ATTR_FIELD_RECV_INFO;
    param.recv_info.tag_info = &info;
    if (memory_type != kDetect) {
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
        param.memory_type = to_ucs(memory_type);
    }
    ucs_status_ptr_t result = ucp_tag_recv_nbx(h->worker, buffer, bytes, tag,
                                               ~static_cast<ucp_tag_t>(0), &param);
    int64_t id = track(h, result, -1, bytes, false);
    if (id >= 0 && result == nullptr) h->ops[static_cast<size_t>(id)].received = info.length;
    return id;
}

// Wait for operations `ids` (from gb_ucx_send / gb_ucx_recv), at most `timeout_s`.
// On a timeout or failure every outstanding operation is cancelled and the
// handle refuses further sends and receives.
int gb_ucx_wait(void* handle, const int64_t* ids, int count, double timeout_s) {
    auto* h = static_cast<Handle*>(handle);
    std::vector<int64_t> wanted(ids, ids + count);
    for (int64_t id : wanted) {
        if (id < 0 || static_cast<size_t>(id) >= h->ops.size()) {
            return fail(h, kFailed, "no UCX operation %lld", static_cast<long long>(id));
        }
    }
    int result = progress_until(h, wanted, now() + timeout_s);
    if (result != 0) {
        std::string reason = h->error;
        cancel_all(h, 1.0);
        h->error = reason;
        h->broken = true;
    }
    bool idle = true;
    for (const Op& op : h->ops) idle = idle && op.request == nullptr;
    if (idle) h->ops.clear();  // ids are only valid until every posted operation is waited for
    return result;
}

// Close every endpoint (flushing, then forcing after `timeout_s`), then the worker
// and context. Frees the handle.
int gb_ucx_close(void* handle, double timeout_s) {
    auto* h = static_cast<Handle*>(handle);
    cancel_all(h, timeout_s);
    double deadline = now() + timeout_s;
    std::vector<void*> closing;
    for (Peer& p : h->peers) {
        if (p.ep == nullptr) continue;
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = (h->broken || p.failed != UCS_OK) ? UCP_EP_CLOSE_FLAG_FORCE : 0;
        ucs_status_ptr_t request = ucp_ep_close_nbx(p.ep, &param);
        if (UCS_PTR_IS_PTR(request)) closing.push_back(request);
        p.ep = nullptr;
    }
    int result = 0;
    for (void* request : closing) {
        while (ucp_request_check_status(request) == UCS_INPROGRESS && now() < deadline) {
            if (ucp_worker_progress(h->worker) == 0) sched_yield();
        }
        // Past the deadline the flush is abandoned: freeing the request lets UCX drop
        // it, and destroying the worker below tears the endpoint down.
        if (ucp_request_check_status(request) == UCS_INPROGRESS) result = kTimedOut;
        ucp_request_free(request);
    }
    ucp_worker_destroy(h->worker);
    ucp_cleanup(h->context);
    delete h;
    return result;
}

// Makes a NIC's writes into an AMD GPU's memory visible to its kernels, given the
// address of the GPU's HDP flush register (hipDeviceAttributeHdpMemFlushCntl):
// a write of 1 and a read back, as hipDeviceFlushGPUDirectRDMAWrites does, for
// HIP releases without that function. Volatile, so neither access is dropped.
void gb_ucx_hdp_flush(void* reg) {
    std::atomic_thread_fence(std::memory_order_seq_cst);  // after the completion was seen
    volatile uint32_t* flush = static_cast<volatile uint32_t*>(reg);
    *flush = 1u;
    static_cast<void>(*flush);
}

}  // extern "C"
