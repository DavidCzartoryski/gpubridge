# Staged verbs path: what it costs, and a one-node A/B plan (2026-10-19)

**Status.** Measurement only. This change adds timers and counters to the verbs
bridge and a harness to run on one GPU node; it changes no data path. Every number
below that this change produced is a **local CPU/MOCK measurement on an Apple M1 Max**
(a MacBook Pro, a shared machine with a load average of 35 to 54 from other work
throughout), never GPU or NIC evidence. No verbs run on GPU or RDMA hardware is
recorded in this repository (HARDWARE_VALIDATION.md, items 38 to 42: not run).
Section 7 is the conditional A/B of one opt-in change built on these measurements
(`GPUBRIDGE_VERBS_STAGED_OVERLAP`); it claims no speedup.

## 1. Why: what is not measured today

A staged call (a GPU tensor copied through the registered pinned buffer,
`GPUBRIDGE_VERBS_GPU_PATH=staged`) does, on each island leader
(`src/gpubridge/verbs/transport.py`, `_call`):

1. `RegisteredStaging.to_host`: queue the copy out on the caller's stream, then block
   the host until it has passed. That wait also covers whatever was queued before it on
   the stream: under reduce-bridge-broadcast, the island's NCCL reduce.
2. The bridge step (`TaggedBridge._all_reduce`, two leaders): post every chunk's
   receive (a receive of at least `GPUBRIDGE_VERBS_RNDV_BYTES`, 256 KiB, makes a write
   window, one `ibv_reg_mr` of exactly its bytes, on the posting thread), post every
   chunk's send, then per chunk wait for its send and receive and reduce it on the CPU
   (one torch thread).
3. `to_device`: queue the copy back. The host doesn't wait for it.

The library deregisters each write window on its completion thread once the peer's
"written" message has arrived (`endpoint.rs`, `poll_and_handle`).

What was missing: the time of each of those host phases, and the cost of the write
windows. The library counted windows (`direct.write_windows`) but not their time, and
`scripts/verbs_perf.py`'s "registrations" metric counts only cached registrations,
not windows. A 25 MiB call at the default 4 MiB chunk makes 7 windows per leader; a
150 MiB call 38; a GPT-2 small step (497,759,232 bytes of fp32 gradients) at least
119.

## 2. What this change adds

| Where | What | Cost when on (local CPU, M1 Max) |
| --- | --- | --- |
| `crate/src/direct.rs`, `Registrar::expose` | `write_window_register_ns`: nanoseconds spent making write windows, failed attempts included | one `Instant::now()` and one `elapsed()` per window: 44 ns (macOS) and 47 ns (Linux aarch64 container) per timed region, against a registration of at least microseconds on a NIC |
| `crate/src/direct.rs`, `Registrar::release`, called from `endpoint.rs` `poll_and_handle` | `write_windows_released`, `write_window_release_ns`: windows deregistered once written, and the time that took on the completion thread | the same per batch of windows released, plus one uncontended lock; nothing when a poll released nothing |
| `verbs/gpu.py`, `Router` | `post_s`, `wait_s`: host seconds in posts and in the library's waits, failed ones too | read the clock only while `timed`, which only staged calls turn on |
| `verbs/transport.py`, `_call` | `telemetry()["staged_phases"]`: for staged calls that succeeded, `calls`, `to_host_s`, `bridge_s`, `post_s`, `wait_s`, `to_device_s` | four clock reads per staged call |
| `scripts/gpu/verbs_staged_ab.py` | one arm of the on-node A/B (section 4), and `compare` | none on the library |

Derived per staged call: **reductions plus Python = `bridge_s` - `post_s` - `wait_s`**.
Window share of a call: `(write_window_register_ns + write_window_release_ns) / 1e9`
over `bridge_s` (release runs on another thread: it delays the completion thread's
next poll, it is not inside `bridge_s`).

**Still unavailable, by design or not measurable from the host:**

- **H2D, the copy back: UNAVAILABLE.** The host never waits for it, so nothing times
  it, and `to_device_s` is only its queueing. The A/B harness's CUDA-event latency of
  the whole call includes it, but also the island broadcast and framework work, and
  the host's enqueueing can overlap GPU work: that latency minus the host phases is
  no estimate of the copy back, and nothing in section 5 uses it;
- inside `to_host_s`, the island reduce's GPU time against the copy out's: the 2-rank
  (1+1) arm has no island step, so there it is the copy out alone;
- windows of failed receives, released when their queue pair closes (counted in
  `write_windows_retired`, not timed);
- staged calls that raised (not in `staged_phases`);
- what a window costs on mlx5: unknown until the hardware run.

Observation records (`GPUBRIDGE_VERBS_OBSERVE`) are unchanged.

### Measured cost of the instrumentation (local CPU/MOCK, M1 Max, loaded)

- **Rust clock reads**: 44 ns per timed region on macOS, 47 ns in a Linux aarch64
  container on the same machine (`Instant::now()` then `elapsed().as_nanos()`, best of
  7 runs of 5 million).
- **Router, per send + receive + wait** against an endpoint that does nothing (best of
  9 runs of 100,000, 5 runs each, interleaved): base `cd9fbac` median 2,203 ns; this
  change with timing off 2,316 ns (+0.11 us, runs ranged 2,127 to 2,815 so inside the
  noise); timing on 2,548 ns (+0.35 us). A staged 25 MiB call has 7 such triples:
  about 2.4 us of clock reads, against a call of milliseconds.
- **Python per host all_reduce** (`verbs_perf.py python-cost`, 5 interleaved runs per
  arm): 4 KiB 14.54 us before, 15.60 us after; 1 MiB 52.42 and 53.87 us. Both inside
  the run-to-run range (13.6 to 17.4 us at 4 KiB).
- **MOCK A/B** (`verbs_perf.py ab`, host and observe series, economy, 5 pairs): see
  section 3.

## 3. The MOCK A/B of the instrumentation (CPU/MOCK, not a GPU or RDMA speedup)

`results/verbs-perf/ab-3-phase-timers/`: `scripts/verbs_perf.py ab` of the base
(`cd9fbac`, `src` tree `ea6c77b9`) against this change (`src` tree `042b6444`),
host and observe series, `economy`, 5 interleaved pairs, the section 1 rules of
docs/VERBS_PERF.md as written. Every timed cell is inside its spread (16% to 46% on
this loaded machine) and every counted metric is the same (registrations,
crossings, control messages, mismatches). Host 4 KiB p50 was +5.1%, after slower in
5 of 5 pairs, inside a 27.9% spread; the direct probes above put the change's cost
there at about 0.1 us per call.

An earlier version of this change read the clock on every path. Its MOCK A/B (same
series, 5 pairs, kept as recorded in the same directory's `superseded/`) gave one
cell outside the spread: observe, 1 MiB, p50 +21.8% against a spread of 21.3%, after
slower in 5 of 5 pairs. The direct probes above put that version's cost at about
1 us per host call, too much to leave on by default; the router now reads the clock
only during staged calls.

## 4. On-node A/B (owner runs; one node, four V100s, NIC loopback between two leaders)

**Not cross-vendor and not inter-node evidence.** `GPUBRIDGE_SPLIT_TEST=half` makes two
2-GPU islands on one node; the bridge between their leaders runs over the node's own
RDMA NIC. Nothing here is GPUDirect: every gpubridge arm is `staged` and says so.

### The "about 40% slower than native" figure

It cannot be reproduced from this repository: no verbs training or collective record
is committed. The definitions of the run it most likely comes from (a two-island staged
verbs arm against a flat four-rank NCCL arm, one run each, observation export on in
the verbs arm only) mix the hierarchy's cost with the transport's. Quote it only from
its own records, with those definitions. The matrix below gives a topology-matched
control (`gloo` over the same split) and matched observation settings.

### Setup (once, on the node, before any arm)

```bash
# Two checkouts: BASE at cd9fbac, PR0 at this branch's head. No site paths here: set
# them in your shell. The harness always comes from PR0; each arm imports its tree.
export BASE=... PR0=... OUT=...      # OUT: an empty results directory
mkdir -p "$OUT/m1" "$OUT/m2"          # every command below writes into these
for T in "$BASE" "$PR0"; do
  PYTHONPATH="$T/src" GPUBRIDGE_VERBS_LIB_DIR="$T/.verbs-lib" \
    python -m gpubridge.verbs build                      # prints the library path
done
export LIB_BASE=$(ls "$BASE"/.verbs-lib/libgpubridge_verbs-*.so)
export LIB_PR0=$(ls "$PR0"/.verbs-lib/libgpubridge_verbs-*.so)
sha256sum "$LIB_BASE" "$LIB_PR0" > "$OUT/libraries.sha256"
(cd "$BASE" && git rev-parse HEAD; cd "$PR0" && git rev-parse HEAD) > "$OUT/revs.txt"
# Matched settings in every arm: no observation export, no compression, defaults.
unset GPUBRIDGE_VERBS_OBSERVE GPUBRIDGE_VERBS_CHUNK_BYTES GPUBRIDGE_VERBS_RNDV_BYTES \
      GPUBRIDGE_BRIDGE_COMPRESSION GPUBRIDGE_POLICY GPUBRIDGE_PINNED_STAGING
export GPUBRIDGE_SPLIT_TEST=half GPUBRIDGE_VERBS_GPU_PATH=staged GPUBRIDGE_VERBS_PROGRESS=economy
env | grep -E '^(GPUBRIDGE|NCCL|CUDA)_' | sort > "$OUT/env.txt"
```

Before and after each run: `uptime >> "$OUT/load.txt"` and
`nvidia-smi -q -d CLOCK >> "$OUT/clocks.txt"`. Change no clocks, drivers, modules or
permissions.

### M1: collectives

One run is (`ARM`, `TREE`, `LIB`, `R` and extra settings per row below):

```bash
uptime >> "$OUT/load.txt"; nvidia-smi -q -d CLOCK >> "$OUT/clocks.txt"
PYTHONPATH="$TREE/src" GPUBRIDGE_VERBS_LIB="$LIB" /usr/bin/time -v -o "$OUT/m1/$ARM-r$R.time" \
  torchrun --nproc-per-node=4 "$PR0/scripts/gpu/verbs_staged_ab.py" run \
  --out "$OUT/m1" --arm "$ARM" --repeat "$R" --op gpubridge --bridge verbs
```

| Arm | Tree, library | Differs from P0 by | Repeats |
| --- | --- | --- | --- |
| `N` native NCCL | PR0 | `--op native` (one NCCL group of all 4 ranks; same job layout) | 2 |
| `B0` staged, base | BASE, `LIB_BASE` | the code before this change | 3 |
| `P0` staged, this change | PR0, `LIB_PR0` | - | 3 |
| `D1` diagnostic | PR0 | `GPUBRIDGE_VERBS_CHUNK_BYTES=16M` (fewer, bigger windows) | 2 |
| `D2` diagnostic | PR0 | `GPUBRIDGE_VERBS_RNDV_BYTES=0` (no windows: slot copies) | 2 |
| `D3` diagnostic | PR0 | `--policy pipelined-reduce-bridge-broadcast` (copies overlapped, unregistered slots) | 2 |
| `D4` transport control | PR0 | `--bridge gloo` (same split, TCP bridge) | 2 |
| `P0x2` pure bridge step | PR0 | `CUDA_VISIBLE_DEVICES=0,1`, `--nproc-per-node=2` (1+1: no island step) | 3 |

Order (fixed now, rotated so no arm always runs first or last): repeat 0: N B0 P0 D1
D2 D3 D4; repeat 1: P0 B0 D4 D3 D2 D1 N; repeat 2: B0 P0; then P0x2 r0, r1, r2. Sizes,
warmup and timed calls are the harness's fixed table: 64 KiB 20/300, 1 MiB 10/100,
4 MiB 10/50, 25 MiB 5/30, 150 MiB 3/10, fp32 integer values (exact sums).

Each run's record keeps every call's CUDA-event and host latency on every rank, the
per-call and whole-region process CPU (all threads, the progress thread included),
max RSS and its growth, warmup and timed wall times, the run's full wall time, the
actual payload bytes, mismatches, and on both leaders the verbs telemetry deltas
(sends, receives, bytes, writes, write windows and their nanoseconds, copies, control
messages, completions, notifications, staged phase times, paths) plus the git commit,
the imported sources' hash and the verbs library's hash.

### M2: one training workload (the detector stage's definition)

```bash
# N: flat 4-rank NCCL DDP, no gpubridge in the gradient path
PYTHONPATH="$PR0/src" /usr/bin/time -v -o "$OUT/m2/N-r$R.time" torchrun --nproc-per-node=4 \
  "$PR0/scripts/gpu/train_bench.py" --model gpt2-small --steps 50 --warmup-steps 5 \
  --global-batch 32 --bucket-mb 25 --grad-sync native --json "$OUT/m2/N-r$R.json"
# S: staged verbs bridge, this change (same command, these differences)
#   GPUBRIDGE_VERBS_LIB="$LIB_PR0" ... --grad-sync gpubridge --bridge verbs
# G: topology-matched transport control
#   ... --grad-sync gpubridge --bridge gloo
```

Order: r0 N S G, r1 S G N, r2 G N S. Kept: the train_bench JSON (samples_per_s,
compute, exposed and hidden communication, communication alone, buckets, memory,
losses, params_identical), the `/usr/bin/time -v` output (the whole process tree's
CPU and max RSS: full-process figures, not attribution) and wall time. train_bench
does not record the bridge's telemetry: M1 is where the phases come from.

**Budget**: M1 about 45 minutes, M2 about 30. Hard stop at 100 minutes, cutting M2
from its end. No arm is rerun after its result is seen; a run that fails is recorded
with its error and counted as a failed run.

### Compression

Not in this matrix: this change doesn't touch it. Compare it on its own
(`scripts/gpu/bench_compression.py`, `bench_all_reduce.py --compression`) with its
error tolerances, never against these rows.

## 5. Predeclared acceptance and gates

```bash
H="$PR0/scripts/gpu/verbs_staged_ab.py"
python "$H" compare "$OUT/m1" --before B0 --after P0 --before-repeats 3 --after-repeats 3 \
  --differ code --wire                                                   # G4
python "$H" compare "$OUT/m1" --before P0 --after D1 --before-repeats 3 --after-repeats 2 \
  --differ env.GPUBRIDGE_VERBS_CHUNK_BYTES --differ verbs.chunk_bytes
python "$H" compare "$OUT/m1" --before P0 --after D2 --before-repeats 3 --after-repeats 2 \
  --differ env.GPUBRIDGE_VERBS_RNDV_BYTES --differ verbs.rndv_bytes
python "$H" compare "$OUT/m1" --before P0 --after D3 --before-repeats 3 --after-repeats 2 \
  --differ policy
python "$H" compare "$OUT/m1" --before P0 --after D4 --before-repeats 3 --after-repeats 2 \
  --differ bridge --differ verbs --differ code.verbs
python "$H" compare "$OUT/m1" --before N --after P0 --before-repeats 2 --after-repeats 3 \
  --differ op                                                            # G5 ratios
```

Each writes `compare-<before>-<after>.json` (its status and reasons, whatever the
outcome) and, when it judges, a table. It judges only the comparison declared on its
command line:

- **Complete.** Each arm has exactly its predeclared repeats, 0 to N-1, each a whole
  record (every size of the table, every rank, every call's samples, integer
  mismatch counts). A repeat missing, extra or truncated, a run that failed (no
  record) included, or an arm's file (`<arm>-r<k>.json`) that is not a readable JSON
  object, is `INCOMPLETE` (exit 4): no verdict on the repeats that are there. With `--wire`,
  every island leader's telemetry must hold every counted message (below), else
  `INCOMPLETE`; an arm with no verbs telemetry at all is `UNSUPPORTED` (exit 3).
- **Correct.** Every run of both arms is checked for wrong results, the unpaired
  repeats of the longer arm and incomplete runs included: any is
  `FAIL (wrong result)` (exit 1), whatever else holds.
- **Matched.** The arms' operation, bridge, policy (requested and resolved), table,
  sizes and calls, topology (world size, islands, split), verbs settings, the
  recorded `GPUBRIDGE_`/`NCCL_`/`GLOO_`/`UCX_`/`TORCH_` and device settings, code
  (commit, source and library hashes) and host must be equal, every repeat to its
  arm's repeat 0, and arm to arm except the keys the command declares with
  `--differ` (a key or its group: `code` is every `code.` key). Anything else is
  `REFUSED` (exit 5), with the keys and values.

Timing pairs repeat k of one arm with repeat k of the other, for the repeats both
have (P0's third has no D1 partner). Its rules:
for each size, the spread is the larger arm's (max - min) / median of the per-repeat p50
(slowest rank per call); `win` needs the p50 better by more than the spread in all but at
most one pair; `REGRESSION` is p50, p99 or CPU per call worse by more than the spread, or
(with `--wire`) any differing count of sends, receives, their bytes, RDMA writes, their
bytes, write windows or their bytes (exit 1). `copied_bytes` and `control_sent` depend
on timing (late receives, credit returns) and are reported, not gated. Exit 0: judged,
no failure.

| Gate | PASS | FAIL | UNSUPPORTED |
| --- | --- | --- | --- |
| G1 correctness | 0 mismatches in every call of every arm; train_bench `params_identical` true | any mismatch | - |
| G2 path | every timed call of B0, P0, D1, D2 and P0x2 is `staged` on both leaders (`path_staged` delta = calls), and for P0, D1, D2 and P0x2 `staged_calls` delta = calls; D3 hands the bridge host slots, so its calls are `host` | another path, or phases missing | verbs setup fails (no usable device): the record says why |
| G3 windows | P0 per leader per call: 0 windows at 64 KiB, 1 at 1 and 4 MiB, 7 at 25 MiB, 38 at 150 MiB; D1: 2 at 25 MiB, 10 at 150 MiB; D2: 0 | any other count | - |
| G4 instrumentation cost | `compare B0 P0 ... --wire` judges (exit 0 or 1): no REGRESSION at any size | any REGRESSION (then the timers are not negligible on this node: report it with the raw data) | the compare is `INCOMPLETE`, `UNSUPPORTED` or `REFUSED`: G4 is not judged, its reasons are in the compare JSON, nothing is rerun |
| G5 native baseline | N runs; ratios are reported | - | `native` refuses (exit 3) |

The record is what the next change is chosen from, with these thresholds fixed now
(P0 and P0x2, at 25 and 150 MiB, medians over repeats, per leader, except where a row
names one arm). Every quantity is measured; none is a residual. The rows are read top
to bottom and the first row met picks the next change; the rows below it are reported,
not acted on. The first row is decided by a command (below):

| Observation | Next change |
| --- | --- |
| In P0x2, at 25 or 150 MiB, on either leader: the median over its 3 repeats of `(staged_to_host_s / staged_calls) / p50` is at least 0.30, where `staged_to_host_s / staged_calls` is the D2H copy out, queued and waited for (1+1 has no island reduce before it), and p50 is that repeat's call latency (slowest rank, CUDA events); or `compare P0 D3` says `win` at 25 or 150 MiB | overlap the copies with the transfer, chunk by chunk (candidate A, opt-in) |
| `(write_window_register_ns + write_window_release_ns) / 1e9` (nanoseconds to seconds) at least 20% of `bridge_s`, or D1 or D2 beats P0 by more than the spread | amortize windows (first the chunk size setting) |
| `bridge_s - post_s - wait_s` (reductions and Python) at least 30% of `bridge_s` | reduce on the GPU for two staged leaders |
| none, `wait_s` dominant | the NIC loopback or the island NCCL steps bound it: no bridge change |

P0's `to_host_s` also covers the island reduce it waits for, and the copy back (H2D)
is not measured (section 2): both are reported and trigger nothing.

```bash
python "$H" trigger "$OUT/m1" --arm P0x2 --repeats 3 --compare compare-P0-D3.json \
  > "$OUT/m1/trigger.txt" 2>&1                                           # first row
```

It prints to stdout only (so the file's order is the order printed): for each P0x2
leader at 25 and 150 MiB, each repeat's `(staged_to_host_s / staged_calls) / p50` and
their median, then the D3 compare's status and the sizes it says `win` at, then the
reasons (the leader and size, or the D3 win, that met it; or why nothing is decided),
and last a line that is exactly `TRIGGER MET`, `TRIGGER NOT MET` or `INCOMPLETE`.
`INCOMPLETE` (exit 4) decides nothing, and candidate A is not run: a P0x2 repeat
missing, extra, unreadable or not whole; a file named for P0x2 that holds no P0x2
record; repeats that differ in workload or are not 1+1 on verbs; staged phases not
from exactly the recorded leaders, or with `staged_calls` other than the timed calls;
other than 3 shares for each leader and size; or the D3 compare JSON missing, of other
arms than its name, not judged (`judged` or `REGRESSION`; `INCOMPLETE`, `UNSUPPORTED`,
`REFUSED` or a wrong result are not) or without a row at either size.

Noise and no benefit: a difference inside the spread is "no change", reported with the
raw samples; load or clock explanations are recorded as hypotheses, not used to drop or
rerun runs. Ratios against N and D4 are reported with these definitions and claim no
parity.

### Expected artifacts

`$OUT/libraries.sha256`, `revs.txt`, `env.txt`, `load.txt`, `clocks.txt`,
`m1/<arm>-r<k>.json` (19 runs) with `.time` files, `m1/compare-*.json` (6),
`m1/trigger.txt`,
`m2/<arm>-r<k>.json` (9 runs) with `.time` files. Every JSON string the harness
writes is redacted of home and project paths (`gpukit.write_json`).

## 6. Build impact

The crate stays std-only: no new dependency, no feature change, `Cargo.lock`
unchanged. Its source changes, so the library's cache name (a hash of the sources)
changes and each node rebuilds once (`python -m gpubridge.verbs build`, offline).
Checked offline in a Debian bookworm aarch64 container (rust 1.99.1, rdma-core 44.0,
no RDMA device, no rxe): `cargo fmt --check`, `clippy --all-targets -D warnings` and
`cargo test` with no feature, `test-fabric` and `ibverbs`, and a release build with
`ibverbs` that links `libibverbs.so.1`. The cluster nodes are x86_64: their library
is built there.

## 7. Candidate A, conditional: chunk-by-chunk staged copies (`GPUBRIDGE_VERBS_STAGED_OVERLAP=1`)

**Status.** Opt-in, off by default. Software-verified only (CPU tensors standing in,
lazy copies, the MOCK fabric). **No speedup is claimed.** Run this A/B only if the
section 4 records meet section 5's first threshold, which uses measured phases only:
in P0x2, at 25 or 150 MiB, on either leader, the median over its 3 repeats of
`(staged_to_host_s / staged_calls) / p50` (the copy out's share of the call) is at
least 0.30, or `compare P0 D3` says `win` at 25 or 150 MiB. The copy back (H2D) is not
measured and is not part of it; neither is the call latency minus the host phases.
If the threshold is not met, or G1, G2 or G4 failed or was not judged, or the section 4
run didn't happen, record "not run: threshold not met" (or why) with those numbers
and stop.
No benefit, or a regression, is a result to record, not a reason to rerun.

### What it changes, and what it doesn't

The serialization it removes is certain from the code. How much it costs is not
known; finding that out is the point of this A/B. Without the option, a staged
two-leader all_reduce (a) blocks the host until the whole tensor's copy out has
passed before posting anything, (b) posts every chunk's receive, each making its
write window, before the first send, and (c) queues the whole copy back only after the
last chunk is reduced. With it, every chunk's copy out is queued at once; per chunk
the receive is posted, the host waits for that chunk's copy out, its send is posted;
per chunk, once both have completed, it is reduced and its copy back queued. All of
it stays on the caller's stream (no new streams, no device-wide waits). The trade
it makes: receives are now posted one chunk at a time, so the peer's write of chunk
k + 1 can wait for this leader's copy out of chunk k, and on one stream the copies
out and back still run one after another on the GPU. Either can cancel the gain,
which is why only the A/B decides.

Unchanged: the messages (the same buffers, registrations, tags and sizes, each
direction in the same order, so G3's window counts hold), the reductions and their
order (the same bits), the progress policy, and every call the option doesn't cover:
broadcasts, more than two leaders, and tensors of one chunk. **At the default 4 MiB
chunk, the 64 KiB, 1 MiB and 4 MiB rows run the unchanged code with the option on**:
those rows are an A/A control inside the A arm. Behaviour that differs: an
observation's `duration_s` starts before the first chunk's copy out has passed, and a
call that fails part way may already have copied some reduced chunks back into the
caller's tensor (it still raises, and the bridge refuses later calls).
`staged_phases` keeps its meaning as a split of the call's host time: for overlapped
calls `to_host_s` is queueing the copies out plus the host's waits for them (the
waits the overlap didn't hide) and `to_device_s` is queueing each copy back.

Software evidence (local CPU/MOCK on an Apple M1 Max, never GPU or NIC evidence):
`tests/test_verbs_staged_overlap.py` (docs/VERBS_TRANSPORT.md, "Tests"); with the
option unset, a recording of 12 staged scenarios (692 ordered copies, waits and posts,
with the bytes each send read when it was posted) is byte-identical to the base
`0a3ffbe`; and `results/verbs-perf/ab-4-staged-overlap/` holds the host-path Python
cost before and after, inside the run-to-run range.

### Setup (after section 4's)

```bash
export PRA=...    # a checkout of this change; same shell as section 4 (BASE, PR0, OUT)
PYTHONPATH="$PRA/src" GPUBRIDGE_VERBS_LIB_DIR="$PRA/.verbs-lib" python -m gpubridge.verbs build
export LIB_PRA=$(ls "$PRA"/.verbs-lib/libgpubridge_verbs-*.so)
sha256sum "$LIB_PRA" >> "$OUT/libraries.sha256"  # crate unchanged: same name and hash as LIB_PR0
(cd "$PRA" && git rev-parse HEAD) >> "$OUT/revs.txt"
unset GPUBRIDGE_VERBS_STAGED_OVERLAP            # set per run below, never exported
env | grep -E '^(GPUBRIDGE|NCCL|CUDA)_' | sort > "$OUT/env-a.txt"
mkdir -p "$OUT/m1a" "$OUT/m2a"
export H="$PRA/scripts/gpu/verbs_staged_ab.py"  # one harness for every M1-A run
```

### M1-A: collectives

One run (`ARM`, `TREE`, `LIB`, `OVERLAP`, `NPROC`, `GPUS`, `OP`, `R` per row):

```bash
uptime >> "$OUT/load.txt"; nvidia-smi -q -d CLOCK >> "$OUT/clocks.txt"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$TREE/src" GPUBRIDGE_VERBS_LIB="$LIB" \
  GPUBRIDGE_VERBS_STAGED_OVERLAP="$OVERLAP" /usr/bin/time -v -o "$OUT/m1a/$ARM-r$R.time" \
  torchrun --nproc-per-node="$NPROC" "$H" run \
  --out "$OUT/m1a" --arm "$ARM" --repeat "$R" --op "$OP" --bridge verbs
```

| Arm | `TREE`, `LIB` | `OVERLAP` | `NPROC`, `GPUS` | `OP` | Repeats |
| --- | --- | --- | --- | --- | --- |
| `Na` native NCCL | `PRA`, `LIB_PRA` | `0` | 4, `0,1,2,3` | `native` | 2 |
| `P0a` staged, before | `PR0`, `LIB_PR0` | `0` (PR0 has no such option) | 4, `0,1,2,3` | `gpubridge` | 5 |
| `A1` staged, overlapped | `PRA`, `LIB_PRA` | `1` | 4, `0,1,2,3` | `gpubridge` | 5 |
| `P0ax2` pure bridge step, before | `PR0`, `LIB_PR0` | `0` | 2, `0,1` | `gpubridge` | 3 |
| `A1x2` pure bridge step, overlapped | `PRA`, `LIB_PRA` | `1` | 2, `0,1` | `gpubridge` | 3 |

Order (fixed now): `Na` r0; r0 `P0a A1`, r1 `A1 P0a`, r2 `P0a A1`, r3 `A1 P0a`, r4
`P0a A1`; r0 `P0ax2 A1x2`, r1 `A1x2 P0ax2`, r2 `P0ax2 A1x2`; `Na` r1. 18 runs, sizes,
warmup and timed calls as section 4's table. Section 4's records are never edited or
reused as these arms; `P0a` is rerun here so both arms of each pair run minutes apart
with the same harness.

```bash
# The arms differ by design in the code, the option and its recorded setting only.
D="--differ code --differ env.GPUBRIDGE_VERBS_STAGED_OVERLAP --differ verbs.staged_overlap_setting"
python "$H" compare "$OUT/m1a" --before P0a --after A1 --before-repeats 5 --after-repeats 5 $D --wire
python "$H" compare "$OUT/m1a" --before P0ax2 --after A1x2 --before-repeats 3 --after-repeats 3 $D --wire
# GA2: which code each timed call ran, on both leaders
python - "$OUT/m1a" > "$OUT/m1a/ga2.txt" <<'PY'
import json, pathlib, sys
bad = 0
for p in sorted(pathlib.Path(sys.argv[1]).glob("*-r*.json")):
    record = json.loads(p.read_text())
    if record["op"] != "gpubridge":
        continue
    on = record["arm"].startswith("A1")
    settings = [v["verbs"].get("staged_overlap_setting")
                for v in record["provenance"] if "verbs" in v]
    if settings != [True, True] if on else any(settings):
        bad += 1
        print(p.name, "staged_overlap_setting", settings, "FAIL")
    for ranks in record["sizes"]:
        for x in (x for x in ranks if x["verbs_delta"] is not None):
            d, many = x["verbs_delta"], x["bytes"] > 4 * 2**20
            ok = d["path_staged"] == x["calls"] == d["staged_calls"] and \
                d.get("staged_overlapped_calls", 0) == (x["calls"] if on and many else 0) and \
                x["last_staging"] == (("overlapped" if many else "whole (one chunk)") if on
                                      else None)
            bad += not ok
            print(p.name, x["bytes"], x["rank"], x["last_staging"], "ok" if ok else "FAIL")
print("GA2", "PASS" if bad == 0 else f"FAIL ({bad})")
PY
```

### M2-A: the training workload

Section 4's M2 command, `--grad-sync gpubridge --bridge verbs`, as two arms:

- `S`: `PYTHONPATH="$PR0/src" GPUBRIDGE_VERBS_LIB="$LIB_PR0"`, scripts from `PR0`;
- `SA`: `PYTHONPATH="$PRA/src" GPUBRIDGE_VERBS_LIB="$LIB_PRA" GPUBRIDGE_VERBS_STAGED_OVERLAP=1`,
  scripts from `PRA`;

each `--json "$OUT/m2a/<arm>-r$R.json"` under `/usr/bin/time -v -o "$OUT/m2a/<arm>-r$R.time"`.
Order: r0 `S SA`, r1 `SA S`, r2 `S SA`. GPT-2 small's 25 MiB buckets are many chunks,
so `SA` overlaps nearly every gradient byte. Section 4's `N` runs are its native
reference, reported as a ratio from another block of the same session.

**Budget**: M1-A about 45 minutes, M2-A about 18. Hard stop at 80 minutes: cut M2-A
from its end, then the 2-rank pairs from theirs. A failed run is recorded with its
error, counted as failed, and not rerun.

### Predeclared gates and outcomes (`P0a` against `A1`; section 5's rules)

| Gate | PASS | FAIL |
| --- | --- | --- |
| GA1 correctness | 0 mismatches in every call of `A1`, `A1x2`, `P0a`, `P0ax2`; `SA` and `S` `params_identical` true | any mismatch |
| GA2 code path | `ga2.txt` ends `GA2 PASS`: every timed call staged on both leaders; overlapped exactly at 25 and 150 MiB in `A1`/`A1x2`, `whole (one chunk)` below; never in `P0a`/`P0ax2`; provenance `staged_overlap_setting` true on both `A1` leaders | anything else: the arm did not test what it says |
| GA3 wire | both compares: no `wire counts differ` at any size; G3's window counts hold in `A1` | any differing count |
| GA4 no regression | both compares judge (exit 0 or 1): no `REGRESSION` (p50, p99, CPU per call) at any size, including the 64 KiB to 4 MiB A/A rows | any `REGRESSION` |
| GA5 benefit | the 4-rank compare says `win` at 25 MiB or 150 MiB | - |

UNSUPPORTED: verbs setup fails on the node (the record says why), or a compare is
`INCOMPLETE`, `UNSUPPORTED` or `REFUSED` (its reasons are in its JSON); then nothing
here is judged and nothing is rerun.

- **GA1 to GA4 pass and GA5 wins**: "helped on this node at these sizes". It stays
  opt-in; the records go to the maintainer as evidence for any default change. M2-A
  is reported beside it: `samples_per_s` counts as a win only if `SA` beats `S` by more
  than the larger arm's (max - min) / median in 3 of 3 rotations; else it is reported
  as no change.
- **GA1 to GA4 pass, no win**: "no benefit", recorded with every raw sample. The
  option stays off by default (the maintainer may drop it). Not rerun, not retuned.
- **Any FAIL**: recorded as a regression (or a wrong result); the change is dropped.

Load and clock differences are recorded as hypotheses only. CPU per call is gated
(GA4), so a lower median bought with more CPU fails. All arms use `economy` progress
(no busy polling). Ratios against `Na`, section 4's `N` and D4 are reported with
section 5's definitions; no parity with NCCL is claimed, and every arm is `staged`
(host-staged through pinned memory), never GPUDirect.

### Expected artifacts

`$OUT/m1a/<arm>-r<k>.json` (18) with `.time` files, `compare-P0a-A1.json`,
`compare-P0ax2-A1x2.json`, `ga2.txt`; `$OUT/m2a/<arm>-r<k>.json` (6) with `.time`
files; `env-a.txt` and the appended `libraries.sha256`, `revs.txt`, `load.txt` and
`clocks.txt`.

### Build impact

Python only: no crate, dependency or build change. `LIB_PRA` is built from the same
sources as `LIB_PR0` and has the same cache name.
