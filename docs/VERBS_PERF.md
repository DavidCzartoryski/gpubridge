# Verbs bridge implementation cost: budgets, profile and results

**Every number in this document is CPU/MOCK, not a GPU or RDMA speedup.** The
measurements run `bridge="verbs"` on the MOCK test fabric (the test-only build of
the library, whose provider carries RC queue pairs over Unix sockets between
processes on one machine). They measure what the bridge's own code costs per call:
Python, the Python/Rust crossings, the protocol's control messages, registration
and copies. They predict nothing about RDMA bandwidth, latency on a real NIC, GPU
paths or training speed. Hardware tuning (chunk, slot, credit and rendezvous sizes,
progress affinity) stays a parameter for the reservation; a mock fabric can't
choose it.

## 1. Pre-registered budgets and rules

This section was written and committed before any measurement in this document was
taken. It is not edited after measuring; any deviation from it is recorded as a
labelled deviation in the results, never by changing these rules.

### Workloads (matched)

Every comparison runs the same workload in both arms: the same harness
(`scripts/verbs_perf.py`, from this tree, whatever source tree it imports), sizes,
dtype, ranks, repeats, warmup, settings and process setup. Only the code under
comparison differs.

| Series | What runs | Ranks (leaders) | Sizes, fp32 | Warmup / timed calls |
| --- | --- | --- | --- | --- |
| `host` | `topology.bridge.all_reduce(t)` (the verbs bridge itself, not the island policy around it) on host tensors of integer values | 2 | 4 KiB, 64 KiB, 1 MiB, 16 MiB | 20/500, 20/300, 10/100, 3/20 |
| `ring` | the same, ring all_reduce | 3 | 64 KiB, 1 MiB | 20/300, 10/100 |
| `observe` | `host` with `GPUBRIDGE_VERBS_OBSERVE` writing observations | 2 | 4 KiB, 1 MiB | as `host` |
| `pingpong` | `verbs.Endpoint` send/recv/wait, 8 bytes each way, no collectives | 2 | 8 B | 100/2000 round trips |
| `idle` | a connected bridge doing nothing | 2 | none | 2 s window |

Each series runs under each progress policy (`latency`, `balanced`, `economy`;
`GPUBRIDGE_VERBS_PROGRESS`) with all other settings at their defaults (4 MiB chunks,
256 KiB rendezvous). Ranks run with one torch thread each, as the test harness does.
Every result is checked: each all_reduce's output must equal the exact integer sum.

The matrix is bounded to these series. No thread, chunk or queue sweeps: their best
values on a mock fabric say nothing about a NIC.

GPU paths (staged, direct) are not timed here: the staged path only runs with CPU
stand-ins and the direct path only against a fake endpoint, so their timings would
be timings of the stand-ins. Their per-call crossings and registrations are counted
(deterministic), not timed. Fault conditions (timeouts, aborts) are correctness
tests, not performance series.

### Metrics

Per series and arm, from each rank (rank 0's reported; every rank's kept raw):

- **latency**: wall time per call on the rank's own monotonic clock, after a
  barrier-free warmup; p50 and p99 over the timed calls;
- **throughput**: payload bytes / p50 latency (algorithm bytes, not wire bytes);
- **CPU per op**: process user+system CPU time (all threads, so the native progress
  thread too) over the timed loop / calls;
- **idle CPU**: process CPU seconds per wall second in the `idle` window;
- **RSS**: max RSS of the process at the end, and its growth over the timed loop;
- **copies**: the library's `copied_bytes` per call (bytes moved through its own
  slots rather than read or written in place);
- **registrations**: host, dma-buf and peer-memory registrations per call, from the
  library's telemetry, after warmup;
- **crossings**: Python-to-native calls per collective call (counted in a separate,
  untimed pass that wraps every exported function);
- **control messages**: the library's `control_sent` per call.

### Budgets

Absolute targets, CPU/MOCK, judged on the after state. Failing one is reported; it
doesn't by itself block a change that improves things.

| ID | Budget |
| --- | --- |
| B1 | Steady state: 0 registrations per host-path call after warmup, under every policy. |
| B2 | Python-to-native crossings per 2-leader host all_reduce: at most 2 per chunk plus 2 per call (one post of the chunk's send and receive, one wait), excluding telemetry. |
| B3 | Idle CPU: `economy` at most 0.02 core-s/s, `balanced` at most 0.05; `latency` is a disclosed cost (up to one core per endpoint), recorded, not budgeted. |
| B4 | Observation export on: p50 at 4 KiB at most 10% above export off; no dropped observation. |
| B5 | RSS growth over a timed series at most 2 MiB (no per-call leak). |
| B6 | Copies per host-path call no more than today's (protocol-inherent copies through slots are disclosed, not budgeted to zero). |

### Rules for a change (applied as written)

An optimization is measured against its parent commit, interleaved: pairs of runs
alternating A B B A A B ... (5 pairs per series and policy), every run a fresh set
of processes with the same settings, `uptime` load and the machine model recorded
with each run. For each metric, the **spread** is the larger of the two arms'
(max - min) / median over their 5 runs.

1. **Correct.** Every result in both arms equals the exact sum. Any mismatch fails
   the change.
2. **Wins** only where the after median is lower than the before median by more than
   the spread, and lower in at least 4 of 5 pairs. A difference inside the spread is
   **no change**.
3. **No regression.** No budgeted metric (p50, p99, throughput, CPU per op, idle CPU,
   RSS, under any policy and series) gets worse by more than its spread; counted
   metrics (copies, registrations, crossings, control messages) don't increase at all.
4. A change that passes 1 to 3 and wins somewhere may become the default. A change
   that fails 3 anywhere, or wins nowhere, stays **opt-in** (off by default) or is
   dropped. A lower median bought with more CPU or a worse p99 is a tradeoff, so it
   fails 3.
5. A machine-state explanation for a failure (load from another worker, power
   state) is recorded as a hypothesis; it does not turn a failure into a pass.

Results are reported for every metric of every series, passing or failing, with the
raw runs in `results/verbs-perf/`.
