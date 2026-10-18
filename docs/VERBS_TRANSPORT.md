# A raw-verbs bridge transport in Rust

**Status: phase A1 (the foundation) only. There is no `bridge="verbs"` yet:
nothing here moves a tensor, and nothing has run on RDMA hardware or GPUs.**
The crate's resource handling is tested against a mock provider (an
in-process fabric), and its libibverbs provider compiles and links against
rdma-core's headers on Linux. Neither is a transfer over a real or software
RDMA device. Nothing here is validated on RDMA hardware or GPUs.

The plan has three phases, each its own change:

- **A1, this one:** the binding, a C shim compiled against rdma-core's
  headers, owned verbs resources (device context, protection domain,
  completion queue, queue pair, memory region) with their release order, the
  queue pair state machine, a bounded registration cache, and a deterministic
  mock provider with injectable faults.
- **A2:** a versioned reliable-connected (RC) wire protocol on top of it
  (handshake through the c10d store, receive provisioning and credits,
  chunking, a completion thread, deadlines, peer failure, teardown), the
  Python `BridgeTransport` as `bridge="verbs"`, and the collective tests bit
  for bit against `flat-gloo`.
- **A3:** the host-staged GPU path, the NVIDIA peer-memory and AMD dma-buf
  registration paths behind capability checks, telemetry, and the hardware
  test commands.

Nothing here inherits a validation claim from the UCX transport
([RDMA_TRANSPORT.md](RDMA_TRANSPORT.md), CPU and TCP only) or from the
multitcp transport ([RUST_TRANSPORT.md](RUST_TRANSPORT.md), TCP, which missed
its hardware gate).

## Status

| Feature | Status |
| --- | --- |
| C shim over libibverbs (open, query, PD, CQ, QP, MR, dma-buf MR, post, poll) | IMPLEMENTED; compiles and links against rdma-core 57.0 headers and the system `libibverbs.so.1` on Linux (build evidence, not a transfer) |
| Owned resources, release order, cleanup after a failed setup | IMPLEMENTED, SOFTWARE-VERIFIED (mock provider) |
| Queue pair state machine, IB (LID) and RoCE (GID) addressing checks | IMPLEMENTED, SOFTWARE-VERIFIED (mock provider) |
| Work tracking: no deregistration while work is outstanding, drain before release | IMPLEMENTED, SOFTWARE-VERIFIED (mock provider) |
| Bounded registration cache | IMPLEMENTED, SOFTWARE-VERIFIED (mock provider) |
| Mock provider: faults, delays, late completions, peer loss, exhaustion | IMPLEMENTED, SOFTWARE-VERIFIED |
| Any of the above on a software RDMA provider (`rdma_rxe`) | not run: none available (EMULATED would need one) |
| Any of the above on RDMA hardware | BLOCKED: needs an allocated node with an RDMA device, permission to open it, and a peer on the same fabric |
| `bridge="verbs"`, the wire protocol, GPU memory | not started (A2, A3) |

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
| MSRV | 1.82 (0.3.2); 1.85 on master | not declared | not declared | the crate's `rust-version` |
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
  within SONAME 1 with versioned symbols; the newest the shim needs are
  `ibv_reg_dmabuf_mr` (`IBVERBS_1.12`, rdma-core 34) and `_ibv_query_gid_ex`
  (`IBVERBS_1.11`, rdma-core 32). The headers should be the installed
  release's (here rdma-core v57.0); an older runtime fails at load time with a
  missing symbol version, not at run time.
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
cargo test                      # mock provider only: macOS, or Linux without rdma-core headers
cargo test --features ibverbs   # also the libibverbs provider; needs rdma-core's headers
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
in the libibverbs provider (calls into the shim), in the mock provider (its
"DMA" copies between registered memory) and in `MemoryRegion`'s owned
buffers. Registering memory the region doesn't own is an `unsafe fn`: the
caller promises the memory stays valid until the region is released.

## Building on a node without rdma-core's headers

Explorer's login node has rdma-core 57.0's runtime (`/usr/lib64/libibverbs.so.1`
and its providers) but not its `-devel` headers. The headers of the same
release, from rdma-core's own tag, give the same ABI:

```bash
git clone --depth 1 --branch v57.0 https://github.com/linux-rdma/rdma-core
mkdir -p inc/infiniband inc/rdma
cp rdma-core/libibverbs/verbs.h rdma-core/libibverbs/verbs_api.h \
   rdma-core/libibverbs/tm_types.h inc/infiniband/
cp rdma-core/kernel-headers/rdma/*.h inc/rdma/
GPUBRIDGE_VERBS_INCLUDE_DIR=$PWD/inc cargo test --features ibverbs \
    --manifest-path src/gpubridge/verbs/crate/Cargo.toml
```

The tests that run there use the mock provider. The libibverbs provider's
own tests that need no device (layouts, status strings) run too; those that
open a device are ignored unless `GPUBRIDGE_VERBS_DEVICE` names one, and
belong on an allocated node.
