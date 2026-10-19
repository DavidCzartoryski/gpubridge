# Verbs bridge implementation cost: raw data

**CPU/MOCK, not a GPU or RDMA speedup.** Every record here comes from
`scripts/verbs_perf.py` running `bridge="verbs"` on the MOCK test fabric (the
test-only library; RC queue pairs carried over Unix sockets between processes on one
Mac mini). They measure the bridge's own per-call cost: Python, Python/Rust
crossings, control messages, copies, registrations. They say nothing about a NIC's
bandwidth or latency, GPU paths or training speed. Budgets, rules and the reading of
these numbers: `docs/VERBS_PERF.md`.

Machine: Mac16,10 (Apple M4, 10 cores), macOS, Python 3.12, torch 2.14.1, test
library built by `cargo build --release` with Rust 1.99.0. Another worker ran tests on
the same machine throughout; every record has `uptime` before and after its run.

## Layout

- `profile-ef73a07/`: one run per series and policy at the profiled base, the
  `python-cost` probe, and cProfile text for 4 KiB host and observe calls. For finding
  hot spots only.
- `python-cost-<commit>.json`: the pure-Python probe at `49a6ad7`, `b89b4d9` and
  `04bfc24`, run one after another on the same machine.
- `ab-N-<change>/`: one matched comparison. Files are
  `<series>-<policy>-pair<k>-<arm>.json`, one per run (one fresh set of rank processes),
  run in the order A B, B A, A B, ... per series and policy. `summary.json` and
  `summary.md` are `verbs_perf.py compare` of the directory: medians, spread, pairs
  and the rule verdict of docs/VERBS_PERF.md, section 1, for every metric.
  - `ab-1-mock-flag/`: before `49a6ad7`, after `b89b4d9`.
  - `ab-2-key-check/`: before `b89b4d9`, after `04bfc24` (dropped by the rule; reverted in `613a26c`).

**Labelled deviations (after the perf review).** The `summary.*` verdicts are the
harness's, kept as recorded, and depart from docs/VERBS_PERF.md rule 3 in two ways:
copies (`copied_bytes`) are judged against their spread rather than as a count that
must not increase (D1), and RSS differences under 2 MiB are "no change" (D2). Under the
rule as written, D1 turns copies at observe/latency/4 KiB into a regression for
`ab-1` (change 1, so its default status rests on D1) and two copies cells for `ab-2`
(already dropped); D2 changes no verdict. Details: docs/VERBS_PERF.md, section 3,
"Deviations from section 1".

Each arm's source is an exported tree of that commit (`git archive <commit> src`), so
its `src_rev` field is null and the arm name plus this list identify it. The harness
is the same file for both arms.

## Record fields (rank 0's in `ranks[0]`, every rank's kept)

`latency_s` (every timed call), `p50_us`, `p99_us`, `throughput_mb_s` (payload / p50),
`cpu_per_op_us` (process CPU, all threads, summed over the timed calls / calls),
`loop_cpu_per_wall`, `maxrss_bytes`, `rss_growth_bytes`, `per_call` (library telemetry
deltas per call: `copied_bytes`, `control_sent`, `completions`, `registrations`, ...),
`crossings_per_call` (exported-function calls per collective, from an untimed wrapped
pass), `breakdown_us` (that pass's mean call time and time inside each exported
function), `mismatches` (calls whose result wasn't the exact sum: always 0 here),
`idle_cpu_per_s` (2 s with the bridge connected and idle).

## Reproduce

```
export GPUBRIDGE_VERBS_PROGRESS=  # unset: the harness sets it per run
git archive <before> src | tar -x -C /tmp/before
git archive <after> src | tar -x -C /tmp/after
python scripts/verbs_perf.py ab --arm before=/tmp/before --arm after=/tmp/after \
    --series observe,host --pairs 5 --out results/verbs-perf/<dir>
python scripts/verbs_perf.py compare results/verbs-perf/<dir>
```

It needs cargo (the test library is built on first use) and runs on macOS or Linux.
