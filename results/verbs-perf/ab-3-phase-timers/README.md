# ab-3: the cost of the staged-path timers (CPU/MOCK, not a GPU or RDMA speedup)

Every record here is a local CPU/MOCK measurement on an Apple M1 Max (MacBookPro18,2,
10 cores, macOS 15.7, Python 3.12, torch 2.14.1 CPU, test library built by
`python -m gpubridge.verbs` with Rust 1.98.0). It is not GPU, NIC or training
evidence. Other work shared the machine throughout (load average 6 to 54; each record
has `uptime` before and after its run).

## The comparison

`scripts/verbs_perf.py ab`, host and observe series, `economy`, 5 interleaved pairs
(A B, B A, ...), with docs/VERBS_PERF.md section 1's rules as written:

- before: `cd9fbac` (`src` tree `ea6c77b9`);
- after: this change, `3216833` (`src` tree `042b6444`).

Each arm is an exported tree (`git archive <commit> src`). `summary.md` and
`summary.json` are `verbs_perf.py compare` of this directory. **Every timed cell is
"no change (inside spread)" and every counted metric is the same**: registrations 0,
crossings 3 per call (12 at 16 MiB), control messages 0, mismatches 0. The spreads
were 16% to 46% on this loaded machine. The one direction worth stating is host,
4 KiB, p50: +5.1%, after slower in 5 of 5 pairs, against a 27.9% spread. The direct
probes below put the change's cost on that path at about 0.1 us per call.

## superseded/

The same comparison of an earlier version of this change (`src` tree `768c16e9`, never
on the branch), whose router read the clock around every post and wait on every
path. One cell broke the rule: observe, 1 MiB, p50 +21.8% against a spread of 21.3%,
after slower in 5 of 5 pairs. Kept as recorded. That version's direct cost (about
1 us per host call, below) is why the router now reads the clock only during staged
calls.

## probes/

- `python-cost-{before,after}-<k>.json`: `verbs_perf.py python-cost`, run 5 times per
  arm, interleaved. Python per 2-leader host all_reduce against an endpoint that does
  nothing. Median of the run medians: 4 KiB 14.54 us before and 15.60 us after
  (range 13.6 to 17.4 us); 1 MiB 52.42 and 53.87 us. The superseded version measured
  16.41 and 17.53 us in 3 runs.
- `router-ops.jsonl` (`router_ops.py`): one `Router` send + receive + wait against an
  endpoint that does nothing, best of 9 runs of 100,000 calls, 5 runs per mode. The
  modes are base, timing off and timing on (as during a staged call). Medians:
  2,203 ns, 2,316 ns and 2,548 ns. Off is inside the run-to-run range (2,009 to
  2,862 ns at base).
- `instant-macos.json`, `instant-linux-aarch64.json` (`instant.rs`): what the library
  adds per write window, `Instant::now()` then `elapsed().as_nanos()`, best of 7 runs of
  5 million: 44 ns on macOS and 47 ns in a Debian aarch64 container on the same machine.
