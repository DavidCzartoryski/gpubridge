# ab-4: `GPUBRIDGE_VERBS_STAGED_OVERLAP` off is the code it was (CPU/MOCK, not a GPU or RDMA speedup)

Every record here is a local CPU measurement on an Apple M1 Max (MacBookPro18,2,
10 cores, macOS 15.7, Python 3.12, torch 2.14.1 CPU). None of it is GPU, NIC or
training evidence, and none of it says whether the option is faster: that is the
conditional on-node A/B in docs/verbs-perf-20261019.md, section 7. Other work shared
the machine throughout (load average 20 to 25; each python-cost record has `uptime`).

Arms are exported trees: before `0a3ffbe` (`src` tree `042b6444`), after `8376253`
(`src` tree `d8093332`), the change that adds the option.

## probes/

- `default_log.py`, `default-path.json`: the staged path with the option unset, over a
  recording fake wire that plays the peer, with CPU tensors standing in and lazy copies
  (`tests/test_verbs_gpu.py`'s `LazyRegistered`). Twelve scenarios (1, 15, 16, 17, 56
  and 200 float32 elements at a 64-byte chunk, sum and max, three all_reduce calls
  across two streams and a broadcast). Each logs every copy, host wait, device wait,
  post (its registration, offset, size, tag and the bytes the send read when posted)
  and library wait in one order, plus the results and the last copy back: 692 entries.
  `python -I default_log.py <tree>` printed the same bytes for both trees
  (sha256 `81d64fc2bd31540e046468ec5f4d00f00bda763c38f24deda43cd5f5153c058d`);
  `default-path.json` is that output.
- `python-cost-{before,after}-<k>.json`: `scripts/verbs_perf.py python-cost`, 5 runs per
  arm, interleaved (before after, after before, ...). Python per 2-leader host
  all_reduce against an endpoint that does nothing; the host path gained one keyword
  argument. Median of the run medians (range over runs):

  | bytes | observe | before (us) | after (us) | change |
  | --- | --- | --- | --- | --- |
  | 4096 | no | 16.79 (15.70 to 19.00) | 15.68 (15.62 to 18.44) | -6.6% |
  | 1048576 | no | 59.41 (52.63 to 62.68) | 59.66 (52.64 to 63.37) | +0.4% |
  | 4096 | yes | 61.15 (52.38 to 62.11) | 60.66 (55.55 to 63.29) | -0.8% |
  | 1048576 | yes | 106.47 (87.81 to 115.62) | 104.70 (90.89 to 113.76) | -1.7% |

  Every change is smaller than the before runs' own spread, (max - min) / median of 15%
  to 26%: no change measurable here.

No `verbs_perf.py ab` was run for this change: off, the staged call is the same
sequence of operations (above) and the host path differs by one argument.
