# Integrated verbs capability audit

Audit base: `75922a05f49dab0b1d30c0197b55d5118f971ece`, 2026-10-18.
Paths below are relative to `src/gpubridge/verbs/`; Rust paths start at
`crate/src/`. IMPLEMENTED means reachable code, SOFTWARE-VERIFIED means mock,
CPU or test-fabric checks; EMULATED requires a software RDMA device, and
HARDWARE-VERIFIED requires real devices. No row here is EMULATED or
HARDWARE-VERIFIED. **Every hardware gate remains BLOCKED.** The predecessor's
confirming safety review of `3df48a1..75922a0` was WAIVED BY USER / NOT RUN;
its regression tests are evidence, not an independent review.

| Capability | Implementation and selection/gates | Software evidence | Remaining hardware check |
| --- | --- | --- | --- |
| Native host verbs | `ibv.rs:IbvProvider`, `shim/gbverbs.c:gbv_post_send_batch`, `endpoint.rs:Endpoint::post`, `transport.py:VerbsTransport`; opt-in `bridge="verbs"`, Linux `ibverbs` build, usable compatible port/GID, ready barrier | Predecessor Linux rdma-core v57.0 check/clippy; `tests/protocol.rs`, `tests/transfers.rs`, `tests/test_verbs.py` (repository tests) | Allocated two-node SEND/RECV and rendezvous integrity, failure and deadline tests |
| Staged GPU | `gpu.py:RegisteredStaging`, `HostBuffer`, `Router`, `transport.py:VerbsTransport._call`; forced `GPUBRIDGE_VERBS_GPU_PATH=staged`, otherwise explicit fallback with reason | `tests/test_verbs_gpu.py`: lazy staged copies on CPU tensors, expected sums over MOCK fabric | Real pinned copies and stream ordering on CUDA and ROCm |
| NVIDIA dma-buf | `gpu.rs:CudaDriver::{allocation,export_dmabuf,prepare}`, `direct.rs:Registrar::{register_device,try_dmabuf,expose_dmabuf}`; dma-buf first, export attribute/symbol, aligned range, registration success, incoming-write flush required | Header ABI check in predecessor; `tests/direct.rs`: selection, identity reuse, descriptor lifetime, exact receive windows | NVIDIA driver/kernel/NIC export-import compatibility, CUDA visibility and data integrity |
| NVIDIA peer memory | `gpu.rs:PeerModules::nvidia`, `CudaDriver::peer_memory_support`, `direct.rs:Registrar::try_peer`; fallback after dma-buf, loaded module and successful registration | Module fixtures and mock selection in `tests/direct.rs` | Real `nvidia_peermem` registration, allocator reuse and invalidation |
| AMD dma-buf | `gpu.rs:RocmDriver::{load,allocation,export_dmabuf}`, `direct.rs:Registrar`; existing HIP/ROCr runtimes only, HIP >=6, export/close symbols and ROCr support, flush available; no AMD peer-memory fallback | HIP/ROCr ABI checks in predecessor; shared mock registration/window tests | ROCm export/import, allocation identity, cross-thread calls, HDP/driver flush sufficiency |
| NVIDIA ↔ AMD | `protocol.rs:Info`, shared `Endpoint` engine and per-process `Registrar`; each process loads one vendor, native NCCL/RCCL island operations stay in the existing backend | `tests/direct.rs:every_pairing_of_plain_host_registered_and_device_memory_moves_the_same_bytes` proves engine memory combinations, **not vendor interoperability** | Compatible mixed allocation on one RDMA fabric; both directions, staged/direct and training |
| Progress (base) | `endpoint.rs:Shared::run`; dedicated native thread, drain/arm/re-poll/event wait, 100 ms stop/async tick; no async-runtime worker polling | Mock completion channels; protocol and shutdown tests | Real completion-event race, async errors and idle CPU |
| Selectable latency/balanced/economy | `progress.rs:Progress`, `endpoint.rs:Shared::{run,events}`, `shim/gbverbs.c:gbv_progress_cpu`; `Config.progress` / `GPUBRIDGE_VERBS_PROGRESS`; default economy preserves the original event loop, optional latency-only CPU affinity restricted by live `sched_getaffinity` | `tests/progress.rs`: every policy handles poll/arm boundary, spurious/empty, bursts, WC/async failures, deadlines, cancellation, queued disconnect and shutdown; direct memory pairings and granted-write abort run under every policy; strict mock detects missing re-poll/re-arm and unacked destruction | Real CPU/latency tradeoffs, NUMA placement and allowed-set affinity execution |
| Observations | `observe.py:{Emitter,CallLog}`, `gpu.py:Router`, `transport.py:VerbsTransport._observe`; `GPUBRIDGE_VERBS_OBSERVE`, bounded queue/background writer, key-field rejection | `tests/test_verbs_gpu.py`; detector F1–F8 integration report is SYNTHETIC | Real NIC identity/rate and completed transfer byte correlation |

## Data flow and lifetime findings

| Area | Audit conclusion and regression evidence |
| --- | --- |
| Allocation identity/reuse | No new defect found: each registration lookup queries BUFFER_ID and checks device/range; `tests/direct.rs:a_reallocated_address_gets_a_new_registration_and_the_stale_one_goes`. Allocator subranges share an allocation identity, but cached regions have local access only. |
| Cache ownership/eviction | No new defect found: `cache.rs:MrCache::get_with` serializes lookup/registration, refuses busy stale entries and evicts idle entries only; cache tests exercise entry/byte bounds and outstanding WR ownership. |
| Preparation before publication | Fixed a race in `direct.rs:Registrar::register_device`: SYNC_MEMOPS ran after insertion into the cache, letting a concurrent lookup return an unprepared MR. Preparation now completes under the cache creation lock; failure drops the unpublished MR/export. `a_registration_is_not_published_before_its_gpu_preparation_finishes` and `failed_gpu_preparation_is_not_cached_and_closes_its_export_once` test both cases. |
| dma-buf close-once | No new defect found: `resources.rs:DmabufFd` owns the descriptor; region drop deregisters before close and leaks on deregistration failure; `tests/direct.rs` tests successful and failed registration lifetimes. |
| Write windows (waived fix) | `direct.rs:Registrar::expose` creates an uncached exact receive range; `endpoint.rs:State::fail_op` retires failed windows until QP close. `a_write_grant_reaches_exactly_the_receive_and_only_until_it_is_written` checks host/dma-buf/peer memory. A granted peer can still write within that window until written or QP error: retained risk, not arbitrary cached-range access. |
| Queue/backpressure | No new defect found: `Config::max_pending`, QP send/recv limits, CQ reservation and credits bound posted work; `unexpected_bytes` stops returning credits when the buffer budget is reached. Protocol/resource tests cover exhaustion and delayed receives. |
| Incoming-write visibility | `gpu.py:Router.wait` flushes completed direct receives before dependent Torch work; bounce copies use caller stream; recording endpoint test checks flush order. Actual AMD visibility is hardware-blocked. |
| Stream/context ownership | Caller thread synchronizes its stream before device access; native progress thread may create a receive's export. CUDA `CudaDriver::with_ctx` retains/pushes the target primary context and pops it on the same thread. HIP allocation queries use explicit pointers/device validation, ROCr exports use explicit addresses; real cross-thread driver behavior remains a hardware gate. |
| Async/subgroups | Shared `TaggedBridge` and backend scheduling preserve existing async and subgroup semantics; repository verbs tests compare policies, compression, subgroups and async bit-for-bit with flat Gloo. No native Torch kernels changed. |
| Failure/teardown (waived fixes) | `fail_endpoint` cancels under the state lock before exposing failure, and `pump`/`place` refuse failed work. Goodbye fails queued sends. Rust protocol tests check no subsequent plain-memory access, prompt goodbye failure and drain/release; Python close test covers a busy staging buffer. Drain timeout deliberately leaks DMA-reachable memory. |
| Fork safety (waived fix) | `endpoint.rs:fork_safety` gates resource creation through the real shim's fork-status/init calls; mock protocol test covers supported, initialized and failure states. |

## Detector findings at audit base

F1 (missing synthetic marker), F2 (local abort labelled error), F4 (one failure
copied to all records), F5 (free-text-only WC status) and F8 (failed post absent)
are fixed by the integration observation changes. Native `operation_outcome`
preserves completion failures and distinguishes cancellations; `CallLog` preserves
completed work and failed posts. Rust protocol tests cover root WC error versus
collateral cancellation and local abort; Python observation tests cover aggregation,
synthetic records and failed send/receive posts. F3 is resolved by explicitly
documenting pre-transfer refusals as exceptions/telemetry only, without records;
the forced-direct refusal test verifies that boundary. F6 is a
non-issue under the agreed contract: non-ok `bytes` means requested bytes, not
traffic moved. F7 (peer rate/PCI/host advertisement) is an optional enhancement;
the base wire handshake carries only device, port and link layer. Never infer
peer identity from a matching NIC name on separate hosts.

Evidence for this integration run belongs under the run's `evidence/` directory;
the rows above distinguish inherited evidence from checks rerun on this branch.
