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

## 2. Profile (base `ef73a07`, CPU/MOCK, not a GPU or RDMA speedup)

Mac mini (Mac16,10, Apple M4, 10 cores), Python 3.12, torch 2.14.1, the test library
in cargo's release mode. Another worker ran tests on this machine at the same time
(load 1.9 to 2.4 during these runs). One run per series and policy; raw records and
cProfile text in `results/verbs-perf/profile-ef73a07/`. These are single runs, for
finding hot spots; the comparisons in section 3 use interleaved pairs.

Where a call's time goes (rank 0, economy; "native" is time inside the library's
exported functions, "Python" the rest of the call):

| Series, size | p50 | native wait | native send/recv posts | Python and torch |
| --- | --- | --- | --- | --- |
| host 4 KiB | 70 us | ~24 us | ~4 us | ~22 us |
| host 1 MiB | 987 us | ~940 us | ~8 us | ~47 us |
| host 16 MiB | 14.8 ms | ~14.0 ms | ~45 us | ~0.8 ms (mostly torch's `add_`) |
| observe 4 KiB | 117 us | ~21 us | ~18 us | ~55 us |
| pingpong 8 B (one round trip) | 38 us | all | | |

`python-cost` (rank 0's Python alone, against an endpoint that completes at once):
7.7 us per 4 KiB call and 31 us per 1 MiB call without observations; **38.4 us and
61.3 us with observations on**.

Findings, each with its evidence:

1. **H1, observation export (demonstrated).** With `GPUBRIDGE_VERBS_OBSERVE` set, a
   4 KiB call's p50 rose from 70 to 117 us (economy; 74 to 122 balanced) and its
   Python from ~22 to ~55 us; `python-cost` attributes +30.7 us per call to it with
   no fabric at all. cProfile (`cprofile-observe-economy-4k.txt`): `check_record`,
   the recursive key-name check, runs 26 times per record (104,000 calls for 4,000
   records) and is the largest single cost, then the per-record wake of the writer
   thread and its `json.dumps` under the GIL. Budget B4 (at most +10%) fails by a
   wide margin.
2. **H2, a full telemetry read per observed call (demonstrated).** `_observe` calls
   `endpoint.telemetry()` on every call to read the constant `mock` flag: one more
   native crossing per call (`gb_verbs_telemetry`, ~4.4 us inside the library) plus
   allocating a 16 KiB buffer and parsing the whole telemetry JSON.
3. **Copies through slots (measured, protocol-inherent here).** `copied_bytes` equals
   the payload on every host call: the caller's tensor is not registered, so the
   library copies sends through its slots (the receive lands in place in the
   transport's registered scratch). Eager messages that arrive before their receive
   is posted are copied once more (4 KiB: 4,194 to 5,702 bytes per 4,096-byte call).
   Removing this means registering caller memory per call, which trades a copy for a
   registration whose cost only a real NIC can tell: a reservation tuning question,
   not changed here.
4. **Crossings (counted).** 3 per chunk (post receive, post send, wait), 12 for a
   16 MiB call: budget B2 (2 per chunk + 2) fails for multi-chunk calls. A crossing
   costs ~1 to 3 us here, under 5% of a 4 KiB call and below its run-to-run spread,
   so batching posts is not a demonstrated hot spot and is not changed.
5. **Python bookkeeping without observations: not a hot spot.** 7.7 us per call spread
   over ~85 small calls; the largest, `_all_reduce` itself, is ~2.5 us. Thread-count
   switching (`torch.set_num_threads`) costs 0.03 us.
6. **Reduction kernels: kept.** At 16 MiB most of the Python share is torch's native
   `add_` on 4M floats per call; it stays as is.
7. **Registrations: none in steady state** (B1 passes on the host path, all
   policies; the staged path's buffer is registered once). The direct path calls
   `register_device` and `release` per call and allocates its scratch per call; the
   library's device cache should make the second registration a hit, but that is a
   GPU-only check (no GPU here).
8. **Progress policies on a shared CPU.** `latency` costs a full core idle (1.00
   core-s/s) and, with two ranks plus the mock's socket threads on a loaded machine,
   was *slower* than economy above 4 KiB (1 MiB: 1,463 vs 987 us; CPU per op 2.6x).
   `economy` and `balanced` idle at 0.003 to 0.005 core-s/s (B3 passes). This is a
   property of an oversubscribed mock, disclosed, not tuned.

## 3. Changes and matched comparisons (CPU/MOCK, not a GPU or RDMA speedup)

Each change ran against its parent: `scripts/verbs_perf.py ab`, series `observe` and
`host`, every policy, 5 interleaved pairs (A B, B A, ...), 60 runs per comparison, load
average 1.7 to 3.6 (another worker's tests). Cells are before → after medians (change,
spread as defined in section 1, pairs better, verdict). Every metric of every size,
RSS and counts included: `results/verbs-perf/ab-*/summary.md`. All 240 runs: 0 wrong
results.

### Deviations from section 1 (labelled after the perf review)

The comparisons did not apply rule 3 as section 1 wrote it in two places (perf review
r1, finding 4). The verdicts below and in `results/verbs-perf/ab-*/summary.*` are the
ones the harness judged, kept as recorded. Rechecked from the raw records with the
harness's own `compare` (which reproduces all 204 recorded verdicts of each
comparison), once as judged and once under rule 3 as written:

- **D1: copies judged as a timed metric.** Rule 3 lists copies among the counted
  metrics, which must not increase at all. `scripts/verbs_perf.py` leaves
  `copied_bytes` out of `COUNTED` and judges it against its spread, because an eager
  message that arrives before its receive is posted is copied once more, so copied
  bytes per call depend on timing. Under the rule as written, change 1 regresses on
  copies at observe/latency/4 KiB (5242.88 → 5357.57 bytes per call, +2.19%, spread
  19%): rule 4 would keep it opt-in or drop it, not make it the default. Change 2
  regresses on copies at host/balanced/4 KiB (+0.77%) and observe/latency/4 KiB
  (+2.13%) and stays dropped either way. No other verdict changes. So change 1's
  default status rests on this deviation, or on its deterministic crossing reduction
  (4 → 3 per observed call in every pair), which rule 2 doesn't list as a win; its
  one rule-2 win is a single cell, about the size of the drift between sessions
  (the same code's p50 there was 1011 us as comparison 1's after arm and 1028 us as
  comparison 2's before arm). That is David's to
  accept or reject, with the change-2 proposal below.
- **D2: a 2 MiB floor for RSS.** The harness calls a max-RSS or RSS-growth difference
  under 2 MiB "no change", a floor rule 3 doesn't have. Judged against the spread
  instead, no RSS row of either comparison is a regression or a win: D2 changed no
  verdict.

### Change 1: read the endpoint's mock flag once (`b89b4d9`): DEFAULT

`_observe` parsed the whole native telemetry on every observed call to read a flag
that is fixed for the endpoint's life (H2). Now read on the first observed call.
Pure-Python probe: 38.6 → 32.5 us per 4 KiB observed call. Counted: crossings per
observed call 4 → 3 in every pair (the per-call
`gb_verbs_telemetry` is gone). Rule 2: wins on observe/economy 1 MiB (p50 -2.2%,
spread 1.1%, 5/5; throughput +2.2%). Rule 3: no metric regressed beyond its spread
under any policy or series, with copies judged against their spread (deviation D1;
as written, copies regress at observe/latency/4 KiB). Rule 4: passes under D1; it is
the default, pending David's decision on D1.

| series | policy | size | p50 us | p99 us | CPU per op us |
| --- | --- | --- | --- | --- | --- |
| host | balanced | idle | idle CPU core-s/s: 0.004 → 0.004 (+8%, spread 33%, 2/5; inside spread) | | |
| host | balanced | 4 KiB | 74 → 63 (-15%, spread 49%, 3/5; inside spread) | 162 → 127 (-21%, spread 55%, 4/5; inside spread) | 107 → 87 (-18%, spread 37%, 4/5; inside spread) |
| host | balanced | 64 KiB | 96 → 96 (-0%, spread 21%, 4/5; inside spread) | 125 → 124 (-2%, spread 58%, 5/5; inside spread) | 117 → 116 (-1%, spread 8%, 5/5; inside spread) |
| host | balanced | 1 MiB | 1000 → 1002 (+0%, spread 25%, 4/5; inside spread) | 1158 → 1064 (-8%, spread 60%, 4/5; inside spread) | 1186 → 1171 (-1%, spread 20%, 4/5; inside spread) |
| host | balanced | 16 MiB | 14866 → 14888 (+0%, spread 1%, 1/5; inside spread) | 15490 → 15488 (-0%, spread 13%, 2/5; inside spread) | 18607 → 18429 (-1%, spread 2%, 3/5; inside spread) |
| host | economy | idle | idle CPU core-s/s: 0.003 → 0.004 (+4%, spread 28%, 2/5; inside spread) | | |
| host | economy | 4 KiB | 71 → 71 (-0%, spread 34%, 2/5; inside spread) | 158 → 161 (+2%, spread 27%, 1/5; inside spread) | 89 → 91 (+2%, spread 25%, 1/5; inside spread) |
| host | economy | 64 KiB | 95 → 94 (-1%, spread 2%, 5/5; inside spread) | 125 → 123 (-2%, spread 4%, 4/5; inside spread) | 109 → 108 (-1%, spread 3%, 3/5; inside spread) |
| host | economy | 1 MiB | 988 → 992 (+0%, spread 18%, 3/5; inside spread) | 1060 → 1274 (+20%, spread 84%, 2/5; inside spread) | 1140 → 1144 (+0%, spread 12%, 1/5; inside spread) |
| host | economy | 16 MiB | 14803 → 14867 (+0%, spread 14%, 1/5; inside spread) | 15917 → 15148 (-5%, spread 68%, 2/5; inside spread) | 18376 → 18332 (-0%, spread 15%, 3/5; inside spread) |
| host | latency | idle | idle CPU core-s/s: 1.001 → 1.000 (-0%, spread 0%, 3/5; inside spread) | | |
| host | latency | 4 KiB | 52 → 52 (+0%, spread 14%, 3/5; inside spread) | 82 → 81 (-1%, spread 37%, 2/5; inside spread) | 93 → 93 (-1%, spread 8%, 3/5; inside spread) |
| host | latency | 64 KiB | 132 → 133 (+0%, spread 4%, 3/5; inside spread) | 167 → 168 (+1%, spread 39%, 3/5; inside spread) | 253 → 255 (+1%, spread 5%, 3/5; inside spread) |
| host | latency | 1 MiB | 1463 → 1466 (+0%, spread 18%, 2/5; inside spread) | 1582 → 1880 (+19%, spread 42%, 1/5; inside spread) | 2917 → 2947 (+1%, spread 8%, 1/5; inside spread) |
| host | latency | 16 MiB | 22020 → 22162 (+1%, spread 16%, 1/5; inside spread) | 23293 → 28285 (+21%, spread 43%, 0/5; inside spread) | 45939 → 47792 (+4%, spread 14%, 1/5; inside spread) |
| observe | balanced | idle | idle CPU core-s/s: 0.004 → 0.004 (+2%, spread 13%, 1/5; inside spread) | | |
| observe | balanced | 4 KiB | 120 → 108 (-10%, spread 20%, 4/5; inside spread) | 328 → 282 (-14%, spread 52%, 4/5; inside spread) | 201 → 185 (-8%, spread 31%, 4/5; inside spread) |
| observe | balanced | 1 MiB | 1036 → 1014 (-2%, spread 31%, 5/5; inside spread) | 1100 → 1141 (+4%, spread 104%, 2/5; inside spread) | 1212 → 1196 (-1%, spread 26%, 4/5; inside spread) |
| observe | economy | idle | idle CPU core-s/s: 0.004 → 0.004 (+0%, spread 24%, 2/5; inside spread) | | |
| observe | economy | 4 KiB | 119 → 104 (-13%, spread 32%, 5/5; inside spread) | 300 → 259 (-14%, spread 47%, 3/5; inside spread) | 181 → 167 (-8%, spread 39%, 3/5; inside spread) |
| observe | economy | 1 MiB | 1033 → 1011 (-2%, spread 1%, 5/5; win) | 1138 → 1073 (-6%, spread 43%, 4/5; inside spread) | 1189 → 1158 (-3%, spread 3%, 5/5; inside spread) |
| observe | latency | idle | idle CPU core-s/s: 0.999 → 1.000 (+0%, spread 0%, 0/5; inside spread) | | |
| observe | latency | 4 KiB | 117 → 102 (-13%, spread 26%, 5/5; inside spread) | 166 → 159 (-4%, spread 57%, 3/5; inside spread) | 240 → 215 (-10%, spread 18%, 5/5; inside spread) |
| observe | latency | 1 MiB | 1536 → 1496 (-3%, spread 30%, 4/5; inside spread) | 1951 → 1587 (-19%, spread 39%, 5/5; inside spread) | 3078 → 2984 (-3%, spread 24%, 4/5; inside spread) |

### Change 2: remember field names already checked (`04bfc24`): DROPPED (`613a26c`)

`check_record` ran a regular expression and a recursive call for every field of every
record (H1). Remembering names that passed (bounded) and not recursing into scalars
cut the pure-Python probe from 32.5 to 24.9 us per 4 KiB observed call (runs within
±3%). On the pre-registered series, 4 KiB observe p50 fell 6 to 11% under every
policy in 3 or 4 of 5 pairs, but inside spreads of 17 to 34%, and nothing regressed.
Rule 2: wins nowhere. Rule 4: not the default; the code is reverted to the original
check (its refusal test stays). **Deviation proposal for David, not applied:** accept
the low-noise `python-cost` probe as the deciding measurement for pure-Python changes
with no regression, which would make change 2 the default. Until approved it stays
out.

| series | policy | size | p50 us | p99 us | CPU per op us |
| --- | --- | --- | --- | --- | --- |
| host | balanced | idle | idle CPU core-s/s: 0.004 → 0.004 (-2%, spread 19%, 4/5; inside spread) | | |
| host | balanced | 4 KiB | 75 → 67 (-11%, spread 37%, 4/5; inside spread) | 166 → 168 (+1%, spread 37%, 2/5; inside spread) | 109 → 97 (-11%, spread 32%, 3/5; inside spread) |
| host | balanced | 64 KiB | 97 → 96 (-0%, spread 7%, 3/5; inside spread) | 126 → 125 (-1%, spread 17%, 2/5; inside spread) | 116 → 117 (+0%, spread 6%, 3/5; inside spread) |
| host | balanced | 1 MiB | 994 → 992 (-0%, spread 1%, 2/5; inside spread) | 1132 → 1065 (-6%, spread 21%, 2/5; inside spread) | 1169 → 1174 (+0%, spread 3%, 2/5; inside spread) |
| host | balanced | 16 MiB | 14742 → 14829 (+1%, spread 3%, 2/5; inside spread) | 15343 → 15253 (-1%, spread 32%, 2/5; inside spread) | 18496 → 18595 (+1%, spread 3%, 1/5; inside spread) |
| host | economy | idle | idle CPU core-s/s: 0.004 → 0.004 (+1%, spread 51%, 1/5; inside spread) | | |
| host | economy | 4 KiB | 70 → 70 (+0%, spread 9%, 4/5; inside spread) | 166 → 155 (-7%, spread 17%, 4/5; inside spread) | 90 → 89 (-1%, spread 10%, 5/5; inside spread) |
| host | economy | 64 KiB | 94 → 94 (-0%, spread 3%, 2/5; inside spread) | 125 → 122 (-2%, spread 8%, 3/5; inside spread) | 109 → 108 (-0%, spread 2%, 3/5; inside spread) |
| host | economy | 1 MiB | 989 → 987 (-0%, spread 3%, 3/5; inside spread) | 1035 → 1046 (+1%, spread 27%, 1/5; inside spread) | 1144 → 1142 (-0%, spread 3%, 3/5; inside spread) |
| host | economy | 16 MiB | 14751 → 14779 (+0%, spread 4%, 0/5; inside spread) | 16806 → 15356 (-9%, spread 72%, 4/5; inside spread) | 18345 → 18383 (+0%, spread 1%, 2/5; inside spread) |
| host | latency | idle | idle CPU core-s/s: 1.001 → 1.002 (+0%, spread 0%, 1/5; inside spread) | | |
| host | latency | 4 KiB | 52 → 51 (-1%, spread 3%, 3/5; inside spread) | 80 → 78 (-2%, spread 11%, 3/5; inside spread) | 92 → 92 (+0%, spread 3%, 3/5; inside spread) |
| host | latency | 64 KiB | 130 → 132 (+2%, spread 2%, 0/5; inside spread) | 167 → 164 (-1%, spread 7%, 3/5; inside spread) | 247 → 252 (+2%, spread 3%, 1/5; inside spread) |
| host | latency | 1 MiB | 1474 → 1465 (-1%, spread 2%, 3/5; inside spread) | 1571 → 1602 (+2%, spread 22%, 2/5; inside spread) | 2968 → 2921 (-2%, spread 5%, 4/5; inside spread) |
| host | latency | 16 MiB | 22201 → 22055 (-1%, spread 2%, 2/5; inside spread) | 23003 → 22653 (-2%, spread 23%, 3/5; inside spread) | 45916 → 45684 (-1%, spread 3%, 4/5; inside spread) |
| observe | balanced | idle | idle CPU core-s/s: 0.004 → 0.004 (-1%, spread 24%, 4/5; inside spread) | | |
| observe | balanced | 4 KiB | 107 → 97 (-9%, spread 17%, 3/5; inside spread) | 270 → 246 (-9%, spread 49%, 3/5; inside spread) | 183 → 172 (-6%, spread 25%, 3/5; inside spread) |
| observe | balanced | 1 MiB | 1010 → 1006 (-0%, spread 2%, 3/5; inside spread) | 1081 → 1074 (-1%, spread 20%, 3/5; inside spread) | 1195 → 1188 (-1%, spread 2%, 3/5; inside spread) |
| observe | economy | idle | idle CPU core-s/s: 0.004 → 0.003 (-5%, spread 18%, 4/5; inside spread) | | |
| observe | economy | 4 KiB | 96 → 91 (-6%, spread 23%, 4/5; inside spread) | 213 → 243 (+14%, spread 64%, 4/5; inside spread) | 146 → 146 (+1%, spread 34%, 4/5; inside spread) |
| observe | economy | 1 MiB | 1028 → 1007 (-2%, spread 30%, 5/5; inside spread) | 1201 → 1151 (-4%, spread 111%, 3/5; inside spread) | 1190 → 1160 (-3%, spread 32%, 4/5; inside spread) |
| observe | latency | idle | idle CPU core-s/s: 1.002 → 1.002 (-0%, spread 0%, 3/5; inside spread) | | |
| observe | latency | 4 KiB | 104 → 92 (-11%, spread 34%, 4/5; inside spread) | 201 → 140 (-30%, spread 75%, 3/5; inside spread) | 219 → 189 (-14%, spread 29%, 4/5; inside spread) |
| observe | latency | 1 MiB | 1494 → 1488 (-0%, spread 3%, 3/5; inside spread) | 1638 → 1742 (+6%, spread 249%, 1/5; inside spread) | 2994 → 2968 (-1%, spread 12%, 3/5; inside spread) |

### Tried and discarded without a comparison

- Waking the observation writer once per 64 records instead of every record: the
  `python-cost` probe, interleaved 3 times per arm, showed no difference (24.5 to
  25.1 us both ways), so it was not a demonstrated improvement and was not kept.
- Batching each chunk's receive and send post into one crossing (B2): a crossing
  costs 1 to 3 us here, under the spread of every series; not attempted.

## 4. Budgets after the changes (default code at `613a26c`)

From the `before` arm of comparison 2 (that code is today's default).

| Budget | Result | Value |
| --- | --- | --- |
| B1 registrations per host call after warmup, all policies | PASS | 0 |
| B2 crossings per 2-leader call <= 2 per chunk + 2 | PASS for 1 chunk, FAIL for several | 3 per chunk: 3 (4 KiB to 1 MiB), 12 at 16 MiB (budget 10) |
| B3 idle CPU, economy <= 0.02 and balanced <= 0.05 core-s/s | PASS | 0.0035 and 0.0037 (corrected from 0.0037 and 0.0038, which didn't match the raw records); latency 1.00 (disclosed: a full core per endpoint, idle included) |
| B4 observation export: 4 KiB p50 <= +10%, no drops | FAIL (drops: PASS) | +37% economy, +42% balanced, +100% latency at 4 KiB; +1 to +4% at 1 MiB; 0 dropped of 2,920 per run |
| B5 RSS growth over a timed series <= 2 MiB | FAIL as written at 16 MiB | over both comparisons' 240 runs, 46 of 720 rank-series over, all `host` 16 MiB (up to 4.38 MiB, 4,587,520 bytes: corrected from "4.1 MiB", which was another run's 4,112,384 bytes read as decimal MB); every other size within it |
| B6 copies per host call no more than before | PASS | unchanged: the payload once through slots, plus timing-dependent copies of eager messages that arrived before their receive |

Budgets aren't rule 3, so neither deviation (section 3) affects this table. B5's
failures are max-RSS high-water marks at the largest size after only 3 warmup
calls; whether they are a leak or allocator growth is not shown (hypothesis: the
latter, since smaller sizes never exceed the budget). B4 remains the open hot spot:
the `python-cost` probe puts observation export at 32.5 us of Python per 4 KiB call
after change 1, against 7.8 us without it (38.6 before change 1; 24.9 with change 2;
`results/verbs-perf/python-cost-<commit>.json`, the same probe run at each commit).

## 5. What this does not show

No RDMA hardware, GPU or real NIC was used. The mock's own socket I/O dominates its
wait times, so absolute latencies and throughputs mean nothing for a fabric; only
the bridge's own costs (Python, crossings, copies, registrations, observation export)
carry over. The `latency` policy was slower than `economy` here because its spinning
thread competes with two ranks and the mock's threads on a shared machine; on a node
with a dedicated core the tradeoff differs, and its affinity and value are
reservation parameters. The direct GPU path's per-call `register_device`/`release`
and per-call scratch allocation are counted only on fakes; whether the device cache
makes them cheap needs a GPU (a hardware check for the reservation).
