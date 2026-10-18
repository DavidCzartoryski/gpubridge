# A raw-verbs bridge transport in Rust

**Status: phases A1 and A2. `bridge="verbs"` exists, opt-in, for host tensors.
Not validated on RDMA hardware or GPUs.** Everything above the provider (the
owned resources, the wire protocol, the completion thread, the Python bridge) is
tested against mock providers: an in-process fabric, and a test-only fabric
between processes. The libibverbs provider compiles and links against
rdma-core's headers on Linux. None of that is a transfer over a real or
software RDMA device.

The plan has three phases, each its own change:

- **A1:** the binding, a C shim compiled against rdma-core's headers, owned
  verbs resources (device context, protection domain, completion queue, queue
  pair, memory region) with their release order, the queue pair state machine,
  a bounded registration cache, and a deterministic mock provider with
  injectable faults.
- **A2, this one:** a versioned reliable-connected (RC) wire protocol on top of
  it (["Wire protocol"](#wire-protocol-version-1)), the endpoint selection rules,
  completion channels and asynchronous events in the shim, a completion thread,
  the Python `BridgeTransport` as `bridge="verbs"` with telemetry, and the
  collective tests bit for bit against `flat-gloo` on a test-only fabric.
- **A3:** the host-staged GPU path, the NVIDIA peer-memory and AMD dma-buf
  registration paths behind capability checks, the detector's
  transport-observation emission, and the hardware test commands.

Nothing here inherits a validation claim from the UCX transport
([RDMA_TRANSPORT.md](RDMA_TRANSPORT.md), CPU and TCP only) or from the
multitcp transport ([RUST_TRANSPORT.md](RUST_TRANSPORT.md), TCP, which missed
its hardware gate).

## Status

IMPLEMENTED: the code exists. SOFTWARE-VERIFIED: tests pass against a mock
provider. EMULATED: tested over a software RDMA provider (`rdma_rxe`, `siw`).
HARDWARE-VERIFIED: tested on RDMA hardware.

| Feature | Status |
| --- | --- |
| C shim over libibverbs (open, query, PD, CQ, QP, MR, dma-buf MR, post, poll; completion channels, `ibv_req_notify_cq`, asynchronous events, batched posts) | IMPLEMENTED; compiles and links against rdma-core 57.0 headers and the system `libibverbs.so.1` on Linux (build evidence, not a transfer) |
| Owned resources, release order, cleanup after a failed setup | IMPLEMENTED, SOFTWARE-VERIFIED |
| Queue pair state machine, IB (LID) and RoCE (GID) addressing checks | IMPLEMENTED, SOFTWARE-VERIFIED |
| Work tracking: no deregistration while work is outstanding, drain before release | IMPLEMENTED, SOFTWARE-VERIFIED |
| Bounded registration cache | IMPLEMENTED, SOFTWARE-VERIFIED (unused by A2, which registers no caller memory) |
| Wire protocol v1: handshake, chunking, credits, ordering, goodbye | IMPLEMENTED, SOFTWARE-VERIFIED |
| Completion thread on a completion channel, asynchronous events | IMPLEMENTED, SOFTWARE-VERIFIED (the mocks' channels; the shim's `poll(2)` path is compiled only) |
| Deadlines, completion errors, lost peers, teardown with work outstanding | IMPLEMENTED, SOFTWARE-VERIFIED |
| Device, port and GID selection | IMPLEMENTED, SOFTWARE-VERIFIED (mock devices) |
| `bridge="verbs"`: every policy, compression, subgroups, async, bit for bit against `flat-gloo` | IMPLEMENTED, SOFTWARE-VERIFIED (the MOCK test fabric between processes) |
| Any of the above on a software RDMA provider | not run: none available, and loading one needs privileges (EMULATED would need it) |
| Any of the above on RDMA hardware | BLOCKED: hardware/authorization. Needs two allocated nodes with an RDMA device each on one fabric, permission to open the devices from a job, the library built there, and kit steps (phase A3). HARDWARE_VALIDATION.md item 38 |
| GPU memory (staged, NVIDIA peer memory, AMD dma-buf) | not started (A3) |

## Bindings

What the transport needs from libibverbs: device listing and opening, port
and GID queries (with the GID type, for RoCE v2), protection domains,
completion queues, RC queue pairs and `ibv_modify_qp`, `ibv_reg_mr`, and for
A3 `ibv_reg_dmabuf_mr`, plus the hot-path calls `ibv_post_send`,
`ibv_post_recv` and `ibv_poll_cq`. Those last three, and `ibv_reg_mr` and
`ibv_query_gid_ex` in current rdma-core, are `static inline` functions in
`<infiniband/verbs.h>`: they reach the provider through function pointers in
`struct ibv_context`, so a binding either compiles C against the header or
re-implements the inline code against a copy of the struct layout.

Checked on 2026-10-18 (crates.io and GitHub):

| | `ibverbs` / `ibverbs-sys` (jonhoo/rust-ibverbs) | `sideway` / `rdma-mummy-sys` (RDMA-Rust) | `rdma-sys`, `async-rdma` (datenlord) | C shim (chosen) |
| --- | --- | --- | --- | --- |
| Release | 0.9.2 / 0.3.2+55.0, 2025-03-08; master moved on (rdma-core 65.0, edition 2024) unreleased | 0.4.3, 2026-06-04 / 0.2.6, 2026-10-10 | 0.3.0 / 0.5.0, 2023-02-01; no pushes since 2023 | ours |
| License | MIT or Apache-2.0 | MPL-2.0 / MIT | MIT / GPL-3.0 | MIT, the repo's |
| MSRV | 1.82 (0.3.2); 1.85 on master | not declared | not declared | 1.77 (`offset_of!`, for the layout checks) |
| Build | vendors rdma-core 55.0 as a submodule, runs cmake on it, runs bindgen (libclang) | builds a static "mummy" libibverbs + mlx5 with cmake that loads the real providers at run time; bindgen | bindgen + pkg-config against installed headers | a C compiler and the headers; no build dependencies |
| Static inlines | re-implemented in Rust through bindgen's layout of the vendored header (`ops.post_send`, `ops.poll_cq`) | written by hand case by case ("inline function and nested structure are not handled properly", its README) | bindgen + hand-written inlines | compiled from the real header |
| `ibv_reg_dmabuf_mr` | not in 0.9.2; master has an mlx5 "data direct" method only | yes | no | yes |
| Dependencies | bindgen, cmake (+ serde) | bindgen, cmake, libc, memoffset, paste, serde, tabled, thiserror, ... | bindgen, libc, memoffset, paste | none (std) |

**Decision: a minimal C shim** (`src/gpubridge/verbs/crate/shim/gbverbs.c`),
compiled by the crate's `build.rs` with the system C compiler against
rdma-core's real headers and linked dynamically to `libibverbs.so.1`. Why:

- **Header-compiled, no copied layouts.** Every libibverbs call, inline or
  not, is compiled by a C compiler against `<infiniband/verbs.h>`, so the
  inline code that runs is the header's own. Rust sees only opaque pointers
  and the few structs the shim defines itself (attributes in, completions
  out), each with `_Static_assert`s on its size and field offsets in C, the
  same constants asserted in Rust at compile time, and a test that compares
  the two at run time. No `ibv_*` struct is mirrored in Rust and there are no
  hand-written offsets. The shim also translates every enum and flag (access
  flags, MTU, QP state, opcodes, completion status), so no libibverbs
  constant is copied either.
- **The cluster ABI.** It links the system's `libibverbs.so.1` by its
  SONAME (`-l:libibverbs.so.1`), so it builds on nodes with the runtime but
  no `-devel` package (Explorer's login node: rdma-core 57.0, no headers) and
  runs against whatever providers the node has. libibverbs keeps its ABI
  within SONAME 1 with versioned symbols. The built library needs
  `IBVERBS_1.1`, `1.8` (`ibv_reg_mr_iova2`, which the header's `ibv_reg_mr`
  calls), `1.11` (`_ibv_query_gid_ex`) and `1.12` (`ibv_reg_dmabuf_mr`)
  (`readelf -V`). The headers should be the installed release's (here
  rdma-core v57.0); a runtime too old for them fails at load time with a
  missing symbol version, not later.
- **Builds offline with the repo's toolchain.** Like the multitcp crate, it
  uses the standard library only (`build.rs` runs the C compiler and `ar`
  itself rather than through the `cc` crate), so `cargo build --offline`
  works on a node with a Rust toolchain and a C compiler. `ibverbs-sys` needs
  cmake (not on Explorer's login node) and libclang, and its newest release
  fixes the bindings to rdma-core 55.0's header.
- **Coverage.** The shim wraps exactly what the transport uses, including
  `ibv_reg_dmabuf_mr` and `ibv_query_gid_ex`, which the released `ibverbs`
  crate lacks. `sideway` has them, but brings MPL-2.0, a statically linked
  copy of rdma-core's providers and nine other crates.
- **Cost.** About 400 lines of C to keep correct, each wrapper a direct call.
  Adding a verb is a C function and its Rust declaration. If a maintained
  crate later covers dma-buf, builds without cmake and compiles the inlines
  from the installed header, it can replace the shim behind the same
  `Provider` trait.

## The crate

`gpubridge-verbs`, in [`src/gpubridge/verbs/crate`](../src/gpubridge/verbs/crate),
laid out like the multitcp crate. It builds and tests anywhere with only the
mock provider; the libibverbs provider is behind the `ibverbs` feature.

**Build switch.**

```bash
cd src/gpubridge/verbs/crate
cargo test                          # mock provider only: macOS, or Linux without rdma-core headers
cargo test --features ibverbs       # also the libibverbs provider; needs rdma-core's headers
cargo test --features test-fabric   # also the cross-process test fabric (any Unix)
```

| Variable | Meaning |
| --- | --- |
| `GPUBRIDGE_VERBS_INCLUDE_DIR` | a directory holding `infiniband/verbs.h` (and `rdma/`), when the headers aren't in the compiler's default path; rdma-core's source tree has them in `libibverbs/` and `kernel-headers/rdma/` |
| `GPUBRIDGE_VERBS_LIB_DIR` | where `libibverbs.so.1` is, when not in the linker's default path |
| `CC`, `AR` | the C compiler and archiver (`cc` and `ar` by default) |

Without the headers, `--features ibverbs` fails at build time and says
which variable to set.

**Layers.**

- `Provider` (`provider.rs`): the verbs calls, on opaque handles. Two
  implementations: `IbvProvider` (`ibverbs` feature: the shim) and
  `MockProvider` (always).
- Owned resources (`resources.rs`): `Context`, `ProtectionDomain`,
  `CompletionQueue`, `QueuePair`, `MemoryRegion`. Each holds a reference to
  what it was made from, so the release order is structural: a queue pair
  holds its completion queues and protection domain, a memory region its
  protection domain, and both of those the context. Dropping things in any
  order releases them in the right one, and a setup that fails halfway drops
  what it had made, in reverse.
- Work tracking: every posted work request is filed on its completion queue
  with references to the memory regions it reads or writes, and is removed
  only when its completion is polled. A memory region (and, for one that owns
  its buffer, the buffer) is released only after the last such reference
  goes. A queue pair that is closed or dropped with work outstanding first
  moves to the error state, which flushes everything posted on it, and polls
  until each request has completed; only then is it destroyed. If that drain
  runs out of time, the regions it still references are leaked on purpose
  (never deregistered, never freed) and the close reports it: a leak is
  recoverable, the NIC writing into freed memory is not.
- Bounded queues: posting more sends or receives than the queue pair was
  created for fails with `QueueFull` rather than overrunning it, and a queue
  pair can only be created on completion queues with room for all its work.
- `MrCache` (`cache.rs`): the registration cache.
- `Endpoint` (`endpoint.rs`): one bridge member's end of the wire protocol,
  over any provider; `protocol.rs` has the header, the handshake line and the
  endpoint selection; `ffi.rs` the C interface Python loads (`gb_verbs_*`).
- `UdsProvider` (`uds.rs`, the `test-fabric` feature only): the test fabric
  between processes (["Testing"](#testing)).

**Queue pair states.** `Reset -> Init -> Rtr -> Rts`, and `Err` from any
of them. Each transition checks its attributes first: queue pair numbers and
PSNs fit in 24 bits, retry counts in 3 bits, timeouts in 5; the path MTU is
at most the port's active MTU; an InfiniBand port takes a nonzero LID (and an
optional global route), an Ethernet (RoCE) port must have a global route with
a GID index inside the port's table and a nonzero destination GID. Receives
can be posted from `Init`, sends only in `Rts`. Going back to `Reset` is
refused while work is outstanding (verbs would discard it without
completions).

**The registration cache.** Keyed by the allocation's identity: address,
length, access flags and an owner token the caller supplies (an allocation
generation, so the same address handed out again is a different key).
Bounded by entries and by bytes, evicting the least recently used idle
entry. A lookup inside a cached registration of the same allocation reuses
it. Registering an allocation drops cached entries of other owners that
overlap it (that memory was freed and reused). `invalidate` drops entries
when an allocation is freed. Nothing is evicted or deregistered while it is
in use: an entry handed out or referenced by posted work stays registered
until both are gone, and a cache with no idle entry to evict refuses the new
registration (`Exhausted`) instead of growing. Holding the address is not
holding the registration; holding the `MemoryRegion` is.

**Unsafe code.** Every `unsafe` block has a `// SAFETY:` comment. They are
in the libibverbs provider (calls into the shim), in the mock providers (their
"DMA" copies between registered memory), in `MemoryRegion`'s owned
buffers, in the endpoint's copies between its slots and the caller's memory
(`Endpoint::send` and `recv` are `unsafe fn`: the caller keeps the memory valid
until the operation finishes, fails or is aborted), and in the C interface.
Registering memory the region doesn't own is an `unsafe fn`: the caller
promises the memory stays valid until the region is released.

## Wire protocol (version 1)

**Verbs operations used:** `IBV_WR_SEND` with `IBV_SEND_SIGNALED` (every send
completes visibly) and receives (`ibv_post_recv`), on RC queue pairs, one
scatter-gather entry each; nothing else crosses the fabric. No RDMA WRITE, READ
or atomic is posted and the queue pairs grant no remote access
(`qp_access_flags` 0), so no memory is remotely accessible and no remote key
exists to leak. Several sends or receives are posted as one chained request
list (one doorbell).

**Guarantees this rests on (RC):** messages on one queue pair arrive once, in
order, intact; a send's successful completion means the peer's adapter has the
message in a posted receive buffer (not that the peer's program has read it); a
failed request moves the queue pair to the error state, which completes
everything after it with `WR_FLUSH_ERR`.

**Endpoints.** Each member has one completion channel and one completion queue,
and one RC queue pair per other member. Per peer it registers `send_slots`
send buffers and `credits + ctrl_credits` receive buffers of `slot_bytes` each
(defaults 32, 32 + 4 and 128 KiB), owned by the endpoint and page-aligned.
Every receive buffer is posted while the queue pair is in `Init`, before the
member publishes its handshake. **The caller's memory is never registered**:
a message is copied from the caller's tensor into a send slot, and out of a
receive slot into the caller's tensor (or a heap buffer, see below). So the
device can only ever read or write the endpoint's own slots, and A1's
drain-or-leak on close can only ever leak those slots, never Python's memory.
This is the "staged/host" path; zero-copy registration of caller memory (with
the cache keyed by storage identity and a generation) is left to A3 along with
GPU memory.

**Handshake (through the c10d store).** Each member publishes one line under
`gpubridge/verbs/<link>/addr/<rank>`: `<protocol>|<chunk bytes>|` then

```
gbverbs 1;ll=ib|eth;lid=<LID>;gid=<32 hex digits>;grh=0|1;mtu=<active MTU>;
slot=<slot_bytes>;credits=<n>;ctrl=<n>;qpn=<per member>;psn=<per member>;dev=<name>;port=<n>
```

(`qpn` and `psn`: the queue pair number and first send PSN this member uses for
each member index, 0 for itself.) Readers ignore keys they don't know. Each
member refuses to connect, and setup fails on every rank with the reason, when
the protocol version, the link layer (InfiniBand against Ethernet), or
`slot`/`credits`/`ctrl` differ, when an InfiniBand peer has no unicast LID, or
a GID-addressed peer sent no GID. The path MTU is the smaller of the two active
MTUs. Then `Init -> Rtr -> Rts` (RTR: `min_rnr_timer` 12, no RDMA reads; RTS:
ACK timeout exponent 14 and 7 retries by default, `rnr_retry` 6). Each member
then publishes `ready` and **sends nothing until every member is ready**: a
message reaching a queue pair still in `Init` is dropped, and its sender's
retries can run out before the peer connects. Finally one 8-byte message each
way checks the data path.

**Header.** Every message starts with 48 bytes, little-endian:

| Bytes | Field |
| --- | --- |
| 0-1 | magic `0x4256` |
| 2 | protocol version (1) |
| 3 | kind: 1 data, 2 credit, 3 goodbye |
| 4-7 | sequence number of this message on this queue pair, from 0, wrapping |
| 8-11 | payload bytes after the header (0 for credit and goodbye) |
| 12-15 | data receives the sender has reposted for its peer, cumulative, wrapping |
| 16-19 | control receives the sender has reposted for its peer, cumulative, wrapping |
| 20-23 | zero |
| 24-31 | tag |
| 32-39 | the whole message's length |
| 40-47 | this chunk's offset in the message |

A receiver checks the magic, version, sequence number, that the completion's
byte count is the header plus the payload, that a chunk lies inside its message
and follows the last one, and that credits never come back beyond what was
sent; any violation fails the endpoint ("member N broke the protocol: ...").

**Messages and chunking.** A message is `(tag, bytes)` to one peer, split into
chunks of up to `slot_bytes - 48` bytes, sent in order (an empty message is one
empty chunk). Messages to one peer go out in the order they were posted, each
whole before the next; messages with the same tag from one peer are received
in the order sent. A posted receive takes the next message with its tag and
must be exactly its length ("member N sent X bytes for tag T, posted for Y
bytes" otherwise). The Python collectives use 64-bit tags of collective number,
step, sender and chunk (the multitcp bridge's).

**Credits and receive provisioning.** A sender may have at most `credits` data
messages and `ctrl_credits` control messages to a peer that the peer hasn't
reported reposting. A receiver reposts a receive buffer as soon as it has
copied the message out, and reports the cumulative counts in the header of
every message it sends that peer. When it owes at least `credits / 2` data
credits and has no data to send that peer (or no credits to send it with), it
sends a credit message, which uses a control credit. So a send always finds a
posted receive (running out of RNR retries is a protocol error, reported), no
credit message is needed to return control credits (every credit message
returns at least one data credit, so the peer's next data carries the count),
and credit messages can't deadlock. Data credits returned beyond those sent
fail the endpoint.

**Unexpected messages and backpressure.** A message that arrives before its
receive is posted is copied into a heap buffer, up to `unexpected_bytes` (256
MiB by default) per endpoint. Past that, it and every later data message from
that peer stay in their receive buffers, unreposted, so their credits don't go
back and the peer stops sending; posting receives drains them in arrival
order. Posted operations are bounded too (`max_pending`, 65536); beyond that a
post fails. Sends wait for credits and free send slots inside the library, not
in the caller's thread.

**Completion thread.** One thread per endpoint: poll the completion queue
until empty, ask for a notification, poll once more (a completion between the
two would otherwise wait for the next), then block in `poll(2)` on the
completion channel's descriptor and the device's asynchronous event descriptor
(both non-blocking), at most 100 ms so it notices a close. Notifications and
events are acknowledged as they are taken. It does everything a completion
causes (copy out, repost, return credits, send what credits now allow, wake
waiters) in batches; no Python runs per completion and nothing spins.

**Deadlines.** Every wait has one (the collective's, from `init()`'s timeout);
it fails with "timed out: k of n operations unfinished after s". The Python
bridge then aborts the endpoint: a timed-out collective may have left messages
half sent or received, so the bridge refuses later calls.

**Errors and peer loss.** A completion with an error status fails every
operation with that peer and stops using its queue pair, with the status and
vendor error in the message (`RETRY_EXC_ERR`: "the peer didn't acknowledge
within the retry budget (it exited, or the path to it is down)"). An
asynchronous event that breaks something (`QP_FATAL`, `QP_REQ_ERR`,
`QP_ACCESS_ERR`, `PATH_MIG_ERR`, `CQ_ERR`, `PORT_ERR`, `DEVICE_FATAL`) fails
that peer, or the whole endpoint when it names no queue pair. A lost peer is
noticed when something is sent to it; a receive from a peer that died silently
waits for its deadline.

**Teardown.** Closing first sends each connected peer a goodbye (best effort,
up to a second for it to complete), so the peer takes what follows as a close,
not a failure: messages that arrived before it stay receivable, receives that
can no longer be met fail at once, sends to it are refused. Then every
operation is cancelled under the endpoint's lock (from then on nothing reads
or writes caller memory), every queue pair moves to the error state, the
completion thread takes the flushed completions until none is outstanding or
the close's timeout passes, the thread stops, and the queue pairs close (A1's
drain-or-leak: work still outstanding keeps its slots registered and allocated,
and the close says so). `abort` is the same without the goodbye or the release.

## Choosing the endpoint

`GPUBRIDGE_VERBS_DEVICE`, `GPUBRIDGE_VERBS_PORT` and `GPUBRIDGE_VERBS_GID_INDEX`
first: a named device or port is used or setup fails with the reason; it is
never swapped for another. Otherwise: devices in name order, their ports in
order, the first `Active` port with a usable address. An InfiniBand port needs
a unicast LID (a subnet manager) and is addressed by LID without a global
route, unless a GID index is set. A RoCE port needs a GID: the set index, or
the best in its table: RoCE v2 with an IPv4-mapped address, then other RoCE v2,
then link-local RoCE v2, then RoCE v1, the lowest index among equals. Members
on different link layers refuse to connect. Telemetry reports what was chosen.

## Telemetry

`VerbsTransport.telemetry()` (and `state`: `ok`, `failed`, `closed`) returns
the endpoint's JSON: `protocol`, `provider` (`libibverbs (<the loaded file>)`,
whose name carries rdma-core's version, or `MOCK: ...`), `mock`, `library`
(the crate's version), `device`, `port`, `link_layer`, `gid_index`, `gid_type`,
`active_mtu`, `firmware`, `vendor_id`, `vendor_part_id`, `path` (`host`: A2
moves host memory only), `operations` (`RC SEND/RECV`), the protocol settings,
`state`, `failure`, counts (operations, bytes, chunks, control messages, credit
stalls, unexpected and held bytes, completions, errors, notifications,
asynchronous events, batches), and per peer its queue pair number, path MTU,
whether it said goodbye and why it was lost. There are no remote keys to report.

## The Python bridge

`bridge="verbs"` (`src/gpubridge/verbs/transport.py`), opt-in; the default
stays Gloo, and UCX and multitcp are unchanged. `device_tensors` is False: GPU
tensors reach it through the policies' usual staging. Its collectives are
`gpubridge.tagged.TaggedBridge`'s, shared with `bridge="multitcp"` (moved there
from the multitcp transport unchanged): with two leaders a chunked exchange,
with more a ring, so every element is reduced in one fixed order. Chunks of
`GPUBRIDGE_VERBS_CHUNK_BYTES` (4 MiB); the library's settings from
`GPUBRIDGE_VERBS_SLOT_BYTES`, `_CREDITS`, `_SEND_SLOTS`, `_UNEXPECTED_BYTES`.

The library is built like multitcp's: `python -m gpubridge.verbs build`
(Linux, the `ibverbs` feature, rdma-core's headers; `GPUBRIDGE_VERBS_LIB` for a
library built elsewhere), and `python -m gpubridge.verbs info` loads it without
opening a device. If it can't be built or loaded, or there is no usable device
or port, setup fails on every rank with the reason. Payload never moves any
other way: another bridge is only ever chosen explicitly.

## Testing

Rust, on the in-process mock (`tests/protocol.rs`): every message size around
the chunk boundary and 64 KiB, both directions and many tags at once, ordering
within a tag, a 50 KB message through 2 credits (stalls and credit messages),
unexpected messages buffered to the limit and then held, a receive of the wrong
size, the bound on posted operations, deadlines, a message arriving after the
receive's deadline and abort (it never lands), a completion surfacing after an
abort, a lost peer, a failed completion, a fatal device event, a port going
down, closing with sends and receives outstanding (everything released, no
violation), goodbye semantics, three members, mismatched versions, settings and
link layers, the path MTU, and the selection rules.

**The test fabric** (`uds.rs`, feature `test-fabric`): a MOCK provider whose
nodes are processes on one machine, carrying RC queue pairs over Unix sockets
in a directory. It emulates RC's ordering, acknowledgements and failures
(a send to a missing, unconnected or mismatched queue pair, or to a node that
went away, completes with `RETRY_EXC_ERR`). It exists only in a separate test
build of the library (`gpubridge.verbs.build(test_fabric=True)`, without
libibverbs) that only `GPUBRIDGE_VERBS_TEST_FABRIC=<directory>` loads; the
production library can't select it, and telemetry and a warning say MOCK.
`tests/test_verbs.py` runs every bridge policy on two, three and four leaders
with every dtype and reduction, bridge compression, large tensors in many
chunks, subgroups through `backend="gpubridge"`, async collectives, empty and
bad tensors, a receiver after the sender closed, a bad setting and mismatched
settings failing init everywhere, a deadline, and teardown with a receive
outstanding, on it, bit for bit against `flat-gloo`; and that without it
`bridge="verbs"` fails init with the reason. These are software tests of the
code above the provider, not of any fabric.

## Building on a node without rdma-core's headers

Explorer's login node has rdma-core 57.0's runtime (`/usr/lib64/libibverbs.so.1`
and its providers) but not its `-devel` headers. The headers of the same
release, from rdma-core's own tag, give the same ABI:

```bash
git clone --depth 1 --branch v57.0 https://github.com/linux-rdma/rdma-core
mkdir -p inc/infiniband inc/rdma
cp rdma-core/libibverbs/verbs.h rdma-core/libibverbs/verbs_api.h \
   rdma-core/libibverbs/tm_types.h \
   rdma-core/kernel-headers/rdma/ib_user_ioctl_verbs.h inc/infiniband/
cp rdma-core/kernel-headers/rdma/*.h inc/rdma/
GPUBRIDGE_VERBS_INCLUDE_DIR=$PWD/inc cargo test --features ibverbs \
    --manifest-path src/gpubridge/verbs/crate/Cargo.toml
```

The tests that run there use the mock provider. The libibverbs provider's
own tests that need no device (layouts, status strings) run too; the one that
opens a device is ignored, and belongs on an allocated node:
`GPUBRIDGE_VERBS_DEVICE=mlx5_0 cargo test --features ibverbs -- --ignored`.

## Results so far

- **A2, macOS (mock providers):** `cargo fmt --check` and `cargo clippy
  --all-targets -D warnings` with and without `test-fabric` are clean; `cargo
  test --features test-fabric` passes 57 tests (55 without the feature).
  `tests/test_verbs.py`: 34 passed (every policy on two to four leaders,
  compression, large tensors, subgroups, async, failures and teardown on the
  test fabric; init failing with the reason without it). The multitcp bridge's
  tests still pass after its collectives moved to `TaggedBridge` (41).
- **A2, Linux build (2026-10-18), explorer-01:** at commit bf80da8, against
  rdma-core v57.0 headers and the node's rdma-core 57.0 runtime, debug builds at
  `cargo -j 1`: `cargo test --features ibverbs` 59 passed (the device test
  ignored), `--features test-fabric` 57, mock only 55 (its first compile was
  SIGKILLed by the login node's limits; the one retry passed); clippy clean with
  either feature; the shim has no warnings. The library needs `libibverbs.so.1`
  (`IBVERBS_1.0` to `1.12`) and resolves it to the system library; the
  completion channel and event calls are linked, the post and poll calls still
  inline. Build evidence, not a transfer: nothing opened the node's device.
- **A1:** the same checks at A1's commit (33 mock tests; release build on
  explorer-01).
- **CI:** the `verbs crate` job also lints and tests the `test-fabric` feature;
  the pytest jobs run `tests/test_verbs.py` where cargo is installed (not run
  yet with this change).
- Nothing has run over a software RDMA provider or RDMA hardware: not validated
  on RDMA hardware or GPUs. That needs two allocated nodes with usable devices
  on one fabric (HARDWARE_VALIDATION.md item 38).
