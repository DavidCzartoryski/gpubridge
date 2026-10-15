# A multi-stream TCP bridge transport in Rust

**Status: stage 1 implemented as an opt-in transport, `bridge="multitcp"`,
tested on CPU only (loopback TCP between processes). Nothing here has run on
GPUs or between two machines yet; the comparison that decides whether it is
worth keeping is step 08 with `MULTITCP_BRIDGE=1` (below).** The Gloo bridge
stays the default and the reference.

## Why

On Explorer, between a V100 node and an MI50 node, the bridge crosses one TCP
path of about 1.17 GB/s (one stream, measured by the preflight). A 1 GB float32
all_reduce took 1.67 s through the default Gloo bridge, 1.21 s with pinned
staging and 1.07 s pipelined in 16 MB chunks
([BENCHMARKS.md](BENCHMARKS.md#the-all_reduce-matrix-a-v100-and-an-mi50)).
Moving 1 GB each way at the measured rate takes about 0.9 s, so roughly
0.2 to 0.8 s of each of those is something other than the wire: Gloo's one
connection per pair, and its transfer and reduction running one after the
other. This transport attacks both, on an ordinary network first. RDMA would
be stage 2, but Explorer has no RDMA path between those two node types
([BENCHMARKS.md](BENCHMARKS.md#rdma-between-the-node-types)), so stage 2 isn't
started.

## Design

**Shape.** A Rust crate, `gpubridge-transport`
([`src/gpubridge/multitcp/crate`](../src/gpubridge/multitcp/crate)), with a C
interface (`gb_tcp_*`), loaded from Python with ctypes like the UCX transport.
It uses the standard library only, so it builds offline on any node with a
Rust toolchain (Explorer's login node has cargo 1.88 from the OS packages;
the crate builds and passes its tests with 1.70 and 1.98). `python -m gpubridge.multitcp build` compiles it
into `~/.cache/gpubridge` (`GPUBRIDGE_MULTITCP_LIB_DIR`), keyed by the source,
the toolchain, the machine and libc, always without `RUSTFLAGS` (a cache shared
by nodes of different CPUs must not hold a `target-cpu=native` build); `load()` builds on first use;
`GPUBRIDGE_MULTITCP_LIB` takes a library built elsewhere, for nodes without
cargo.

**Split of work.** The Rust side moves tagged byte ranges between bridge
members and nothing else. The collectives (an exchange for two leaders, a ring
for more, a fan-out for broadcast) and every reduction stay in Python with
torch ops, exactly as in the UCX transport, so the results are the same bits
as Gloo's with the same code paths that already pass every policy against
`flat-gloo`.

**Connections.** Each member listens on one port of its chosen address
(`GPUBRIDGE_MULTITCP_IFNAME`, else `GLOO_SOCKET_IFNAME`, else the address this
machine reaches `MASTER_ADDR` from) and publishes `protocol|chunk|streams|host:port`
in the job's c10d store. For each pair, the member with the lower index dials
`GPUBRIDGE_MULTITCP_STREAMS` connections (4 by default), each opened with a
hello naming the sender and the connection's number; an accepting thread on
the other side files them. Leaders whose protocol, chunk size or stream count
differ fail `init()` on every rank, and a one-message handshake each way checks
that every link carries data, as the UCX transport does.

**Threading.** Every connection has a sending thread and a receiving thread,
started at connect and stopped at `close()`. Python calls in on one thread
(ctypes releases the GIL for the call), posts sends and receives, and waits on
the ids they return. The native threads never call back into Python, so no
work depends on the GIL.

**Messages.** A message is one frame, a 16-byte header (tag, length) then the
bytes, on one connection: sends to a peer go round-robin over its connections,
so the chunks of one tensor travel in parallel. Python splits each tensor into
chunks of `GPUBRIDGE_MULTITCP_CHUNK_BYTES` (4 MiB) and tags each one (the
collective's number, its step, the sender and the chunk). The receiving thread
matches a frame to the receive posted for that (peer, tag) and reads its bytes
straight into the destination; a frame whose receive isn't posted yet holds
back that one connection until it is. Each collective posts all its receives
before it waits on any, and tags never repeat within a collective, so a
connection held back this way is always freed by the next post.

**Buffers.** No copies in the transport: a send writes from the caller's
tensor and a receive reads into it. The caller (the transport's Python side)
keeps both alive until `wait()` says the operation finished. A send is
finished once its bytes are in the kernel's socket buffer, so the memory can
be reused. With two leaders, the reduction runs chunk by chunk: chunk *i* is
reduced as soon as it has arrived and this side's copy of it has gone out
(the send reads the same memory), while later chunks are still on the wire.

**GPU memory and stream ordering.** The transport takes host tensors only
(`device_tensors` is False). The policies stage GPU tensors to host memory
before the bridge step and back after it (pageable, or pinned with
`GPUBRIDGE_PINNED_STAGING=1`), exactly as for the Gloo bridge, so every
ordering guarantee the GPU tests enforce for the default bridge holds
unchanged: the transport never sees a device pointer or a stream.

**Failures and time limits.** Every wait has the deadline `init()` gave its
groups (30 minutes without one). A connection that ends at a frame boundary has
delivered everything it carried, so a member closing fails only what can no
longer arrive: a receive from it once all its connections have ended, a send on
a connection whose writer broke (and those queued behind it), a receive cut off
mid-frame. A leader that has finished its part of a collective and closed fails
nothing still arriving on its other connections or from other leaders. A frame of the wrong size fails the worker; a wait that runs out of time,
or any wait after a failure, raises, and the transport refuses every later call.
**Before a failed wait returns, the worker stops**: every connection is shut
down and every sending and receiving thread joined, without a time limit (shutting
a socket down wakes a thread blocked on it), so no thread can still read or write
a collective's tensors once Python sees the error and frees them. Only the
accepting thread, which never touches tensor memory, may outlive the stop's time
limit. If Python's own
code raises in the middle of a collective, the transport stops the worker the
same way (`gb_tcp_abort`) before the error propagates. An empty message never
touches its pointer (an empty tensor's is null). The accepting thread files
only the connections lower-index members dial, each number once. `close()`
stops the worker and frees it. Every C function catches Rust panics, so none
unwinds into Python.

## Tests (CPU)

- The crate's own (`cargo test`): messages of 0 bytes to 3 MB over three
  connections, matched by tag in any order; a frame that arrives before its
  receive is posted; a missing message timing out, and the frame that comes
  after it never reaching the buffer; empty messages with null pointers; a
  frame of the wrong size and a closed peer failing the wait; a member that
  closes failing only the waits on it, with three members, and frames it sent
  before closing still arriving on its other connections; stray and
  repeated dials not taking a real connection's place; bad arguments refused.
- `tests/test_multitcp.py`, between processes over loopback TCP: every bridge
  policy at two, three and four leaders, bit for bit against `flat-gloo`, for
  every dtype and reduction, broadcast, reduce-scatter and an async collective
  (16-byte chunks over three connections, so most tensors are several
  messages); bf16 and fp8 bridge compression; tensors of a million elements in
  64 KB chunks; subgroups (the torch.distributed backend); a bad setting or
  mismatched settings on one leader failing `init()` on every rank; a leader
  that never joins tripping the deadline, then every later call refused;
  teardown with a receive still posted, and a new bridge after it; a receiver
  that posts only after the broadcast's root has closed; a message of
  the wrong size; empty tensors, and a bad argument leaving the bridge working;
  the address and setting rules; and the library's build and load errors.
- CI: `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`, then
  the Python tests over the built library (job `multitcp bridge`).

## Stage 1 on hardware, and the gate

Step 08 with `MULTITCP_BRIDGE=1` builds the library on both nodes and adds three
runs to the comparison matrix, in the same job as the Gloo bridge's:

| Run | What |
| --- | --- |
| `D-multitcp` | the default policy over `bridge="multitcp"`, pageable staging |
| `D-multitcp-pinned` | the same with pinned staging: against `D-pinned` |
| `D-multitcp-1stream` | one connection: what the extra connections add, apart from the chunked overlap |

```bash
MULTITCP_BRIDGE=1 scripts/gpu/explorer/submit.sh 08
```

**The gate, fixed before any run:** the metric is the whole float32
all_reduce's median time, `median_us` in each run's `bench.json` (gpubridge's op,
default policy), at each of step 08's sizes from 64 MB to 1 GB. Stage 1 stays only
if, in each of two step-08 jobs on its own, `D-multitcp-pinned` takes at most 0.8
times `D-pinned`'s time at every one of those sizes. Otherwise the measurements
get written up and the work stops there; it doesn't go on to stage 2 on that
basis alone. Either way the numbers go in
[BENCHMARKS.md](BENCHMARKS.md) and the CHANGELOG, measured, not predicted.

## Not done

- Stage 2 (ibverbs with GPUDirect RDMA on NVIDIA and dma-buf on AMD, with
  registration caching): no RDMA path between Explorer's V100 and MI50 nodes.
- Device tensors. Taking GPU memory directly would need the stream-ordering
  work the UCX transport does (an event per step, a flush after receives), for
  a TCP path that copies through host memory anyway.
- Several leaders per pair of islands (the sharded policy's links each get
  their own transport, as with Gloo).
