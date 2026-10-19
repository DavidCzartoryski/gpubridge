# CPU/MOCK, not a GPU or RDMA speedup
| series | policy | bytes | metric | before | after | change | spread | pairs better | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host | balanced | 4096 | p50_us | 75.06 | 66.92 | -10.9% | +37.1% | 4/5 | no change (inside spread) |
| host | balanced | 4096 | p99_us | 165.8 | 168 | +1.3% | +36.8% | 2/5 | no change (inside spread) |
| host | balanced | 4096 | throughput_mb_s | 54.57 | 61.21 | +12.2% | +42.9% | 4/5 | no change (inside spread) |
| host | balanced | 4096 | cpu_per_op_us | 109.4 | 96.94 | -11.4% | +32.0% | 3/5 | no change (inside spread) |
| host | balanced | 4096 | copied_bytes | 4243 | 4276 | +0.8% | +5.4% | 2/5 | no change (inside spread) |
| host | balanced | 4096 | rss_growth_bytes | 8.192e+04 | 8.192e+04 | +0.0% | +80.0% | 2/5 | no change (< 2 MiB) |
| host | balanced | 4096 | maxrss_bytes | 2.266e+08 | 2.264e+08 | -0.1% | +0.3% | 4/5 | no change (< 2 MiB) |
| host | balanced | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 65536 | p50_us | 96.79 | 96.44 | -0.4% | +7.1% | 3/5 | no change (inside spread) |
| host | balanced | 65536 | p99_us | 125.6 | 124.8 | -0.6% | +17.4% | 2/5 | no change (inside spread) |
| host | balanced | 65536 | throughput_mb_s | 677.1 | 679.6 | +0.4% | +7.1% | 3/5 | no change (inside spread) |
| host | balanced | 65536 | cpu_per_op_us | 116.2 | 116.6 | +0.3% | +6.1% | 3/5 | no change (inside spread) |
| host | balanced | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 65536 | rss_growth_bytes | 0 | 0 | +0.0% | inf | 1/5 | no change (< 2 MiB) |
| host | balanced | 65536 | maxrss_bytes | 2.27e+08 | 2.268e+08 | -0.1% | +0.3% | 4/5 | no change (< 2 MiB) |
| host | balanced | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 1048576 | p50_us | 994.2 | 992.3 | -0.2% | +1.0% | 2/5 | no change (inside spread) |
| host | balanced | 1048576 | p99_us | 1132 | 1065 | -5.9% | +21.2% | 2/5 | no change (inside spread) |
| host | balanced | 1048576 | throughput_mb_s | 1055 | 1057 | +0.2% | +1.0% | 2/5 | no change (inside spread) |
| host | balanced | 1048576 | cpu_per_op_us | 1169 | 1174 | +0.4% | +2.5% | 2/5 | no change (inside spread) |
| host | balanced | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 1048576 | rss_growth_bytes | 1.966e+05 | 2.949e+05 | +50.0% | +150.0% | 2/5 | no change (< 2 MiB) |
| host | balanced | 1048576 | maxrss_bytes | 2.33e+08 | 2.329e+08 | -0.1% | +0.3% | 4/5 | no change (< 2 MiB) |
| host | balanced | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 16777216 | p50_us | 1.474e+04 | 1.483e+04 | +0.6% | +2.5% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | p99_us | 1.534e+04 | 1.525e+04 | -0.6% | +31.8% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | throughput_mb_s | 1138 | 1131 | -0.6% | +2.5% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | cpu_per_op_us | 1.85e+04 | 1.859e+04 | +0.5% | +2.9% | 1/5 | no change (inside spread) |
| host | balanced | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 16777216 | rss_growth_bytes | 1.311e+05 | 3.408e+06 | +2500.0% | +2612.5% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | maxrss_bytes | 3.004e+08 | 3.036e+08 | +1.0% | +1.3% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced |  | idle_cpu_per_s | 0.003689 | 0.00363 | -1.6% | +19.1% | 4/5 | no change (inside spread) |
| host | economy | 4096 | p50_us | 70.25 | 70.37 | +0.2% | +9.1% | 4/5 | no change (inside spread) |
| host | economy | 4096 | p99_us | 166.4 | 155.2 | -6.7% | +16.5% | 4/5 | no change (inside spread) |
| host | economy | 4096 | throughput_mb_s | 58.31 | 58.2 | -0.2% | +8.9% | 4/5 | no change (inside spread) |
| host | economy | 4096 | cpu_per_op_us | 90.33 | 89.46 | -1.0% | +9.8% | 5/5 | no change (inside spread) |
| host | economy | 4096 | copied_bytes | 4162 | 4162 | +0.0% | +3.9% | 2/5 | no change (inside spread) |
| host | economy | 4096 | rss_growth_bytes | 8.192e+04 | 6.554e+04 | -20.0% | +50.0% | 2/5 | no change (< 2 MiB) |
| host | economy | 4096 | maxrss_bytes | 2.261e+08 | 2.266e+08 | +0.2% | +0.3% | 1/5 | no change (< 2 MiB) |
| host | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 65536 | p50_us | 94.23 | 94.19 | -0.0% | +2.8% | 2/5 | no change (inside spread) |
| host | economy | 65536 | p99_us | 125.4 | 122.5 | -2.3% | +7.9% | 3/5 | no change (inside spread) |
| host | economy | 65536 | throughput_mb_s | 695.5 | 695.8 | +0.0% | +2.8% | 2/5 | no change (inside spread) |
| host | economy | 65536 | cpu_per_op_us | 108.9 | 108.4 | -0.5% | +2.4% | 3/5 | no change (inside spread) |
| host | economy | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 65536 | rss_growth_bytes | 0 | 0 | +0.0% | inf | 1/5 | no change (< 2 MiB) |
| host | economy | 65536 | maxrss_bytes | 2.266e+08 | 2.27e+08 | +0.2% | +0.3% | 1/5 | no change (< 2 MiB) |
| host | economy | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 1048576 | p50_us | 989 | 987.3 | -0.2% | +2.8% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | p99_us | 1035 | 1046 | +1.0% | +26.7% | 1/5 | no change (inside spread) |
| host | economy | 1048576 | throughput_mb_s | 1060 | 1062 | +0.2% | +2.9% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | cpu_per_op_us | 1144 | 1142 | -0.2% | +3.2% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 1048576 | rss_growth_bytes | 1.966e+05 | 1.311e+05 | -33.3% | +150.0% | 4/5 | no change (< 2 MiB) |
| host | economy | 1048576 | maxrss_bytes | 2.325e+08 | 2.329e+08 | +0.1% | +0.2% | 1/5 | no change (< 2 MiB) |
| host | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 16777216 | p50_us | 1.475e+04 | 1.478e+04 | +0.2% | +4.5% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | p99_us | 1.681e+04 | 1.536e+04 | -8.6% | +72.1% | 4/5 | no change (inside spread) |
| host | economy | 16777216 | throughput_mb_s | 1137 | 1135 | -0.2% | +4.7% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | cpu_per_op_us | 1.834e+04 | 1.838e+04 | +0.2% | +1.3% | 2/5 | no change (inside spread) |
| host | economy | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | rss_growth_bytes | 3.391e+06 | 2.458e+05 | -92.8% | +1453.3% | 3/5 | no change (inside spread) |
| host | economy | 16777216 | maxrss_bytes | 3.032e+08 | 3.038e+08 | +0.2% | +3.8% | 2/5 | no change (< 2 MiB) |
| host | economy | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy |  | idle_cpu_per_s | 0.003546 | 0.003577 | +0.9% | +51.0% | 1/5 | no change (inside spread) |
| host | latency | 4096 | p50_us | 51.71 | 51.25 | -0.9% | +2.9% | 3/5 | no change (inside spread) |
| host | latency | 4096 | p99_us | 79.75 | 77.79 | -2.5% | +11.0% | 3/5 | no change (inside spread) |
| host | latency | 4096 | throughput_mb_s | 79.21 | 79.92 | +0.9% | +2.9% | 3/5 | no change (inside spread) |
| host | latency | 4096 | cpu_per_op_us | 91.83 | 91.9 | +0.1% | +2.7% | 3/5 | no change (inside spread) |
| host | latency | 4096 | copied_bytes | 4473 | 4407 | -1.5% | +6.6% | 4/5 | no change (inside spread) |
| host | latency | 4096 | rss_growth_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +125.0% | 3/5 | no change (< 2 MiB) |
| host | latency | 4096 | maxrss_bytes | 2.265e+08 | 2.264e+08 | -0.0% | +0.3% | 3/5 | no change (< 2 MiB) |
| host | latency | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 65536 | p50_us | 129.8 | 131.9 | +1.6% | +2.4% | 0/5 | no change (inside spread) |
| host | latency | 65536 | p99_us | 166.5 | 164.2 | -1.4% | +7.2% | 3/5 | no change (inside spread) |
| host | latency | 65536 | throughput_mb_s | 504.8 | 496.7 | -1.6% | +2.4% | 0/5 | no change (inside spread) |
| host | latency | 65536 | cpu_per_op_us | 247.3 | 252 | +1.9% | +3.4% | 1/5 | no change (inside spread) |
| host | latency | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | latency | 65536 | rss_growth_bytes | 0 | 0 | +0.0% | inf | 1/5 | no change (< 2 MiB) |
| host | latency | 65536 | maxrss_bytes | 2.27e+08 | 2.269e+08 | -0.0% | +0.3% | 3/5 | no change (< 2 MiB) |
| host | latency | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 1048576 | p50_us | 1474 | 1465 | -0.6% | +2.1% | 3/5 | no change (inside spread) |
| host | latency | 1048576 | p99_us | 1571 | 1602 | +2.0% | +22.4% | 2/5 | no change (inside spread) |
| host | latency | 1048576 | throughput_mb_s | 711.6 | 715.6 | +0.6% | +2.1% | 3/5 | no change (inside spread) |
| host | latency | 1048576 | cpu_per_op_us | 2968 | 2921 | -1.6% | +4.7% | 4/5 | no change (inside spread) |
| host | latency | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | latency | 1048576 | rss_growth_bytes | 2.785e+05 | 1.311e+05 | -52.9% | +200.0% | 3/5 | no change (< 2 MiB) |
| host | latency | 1048576 | maxrss_bytes | 2.33e+08 | 2.329e+08 | -0.0% | +0.3% | 3/5 | no change (< 2 MiB) |
| host | latency | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 16777216 | p50_us | 2.22e+04 | 2.205e+04 | -0.7% | +1.9% | 2/5 | no change (inside spread) |
| host | latency | 16777216 | p99_us | 2.3e+04 | 2.265e+04 | -1.5% | +23.4% | 3/5 | no change (inside spread) |
| host | latency | 16777216 | throughput_mb_s | 755.7 | 760.7 | +0.7% | +1.9% | 2/5 | no change (inside spread) |
| host | latency | 16777216 | cpu_per_op_us | 4.592e+04 | 4.568e+04 | -0.5% | +2.8% | 4/5 | no change (inside spread) |
| host | latency | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | latency | 16777216 | rss_growth_bytes | 3.408e+06 | 3.375e+06 | -1.0% | +116.8% | 3/5 | no change (< 2 MiB) |
| host | latency | 16777216 | maxrss_bytes | 3.038e+08 | 3.031e+08 | -0.2% | +1.5% | 3/5 | no change (< 2 MiB) |
| host | latency | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency |  | idle_cpu_per_s | 1.001 | 1.002 | +0.1% | +0.4% | 1/5 | no change (inside spread) |
| observe | balanced | 4096 | p50_us | 106.7 | 96.96 | -9.1% | +16.9% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | p99_us | 270.3 | 246 | -9.0% | +49.1% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | throughput_mb_s | 38.39 | 42.25 | +10.0% | +19.9% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | cpu_per_op_us | 183 | 171.7 | -6.2% | +25.4% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | copied_bytes | 5226 | 5145 | -1.6% | +10.7% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | rss_growth_bytes | 8.192e+04 | 9.83e+04 | +20.0% | +160.0% | 1/5 | no change (< 2 MiB) |
| observe | balanced | 4096 | maxrss_bytes | 2.263e+08 | 2.263e+08 | -0.0% | +0.3% | 3/5 | no change (< 2 MiB) |
| observe | balanced | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | balanced | 1048576 | p50_us | 1010 | 1006 | -0.4% | +2.0% | 3/5 | no change (inside spread) |
| observe | balanced | 1048576 | p99_us | 1081 | 1074 | -0.6% | +20.5% | 3/5 | no change (inside spread) |
| observe | balanced | 1048576 | throughput_mb_s | 1038 | 1042 | +0.4% | +2.0% | 3/5 | no change (inside spread) |
| observe | balanced | 1048576 | cpu_per_op_us | 1195 | 1188 | -0.6% | +2.2% | 3/5 | no change (inside spread) |
| observe | balanced | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | balanced | 1048576 | rss_growth_bytes | 2.785e+05 | 1.638e+05 | -41.2% | +80.0% | 2/5 | no change (< 2 MiB) |
| observe | balanced | 1048576 | maxrss_bytes | 2.325e+08 | 2.326e+08 | +0.1% | +0.3% | 2/5 | no change (< 2 MiB) |
| observe | balanced | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | balanced |  | idle_cpu_per_s | 0.003905 | 0.003881 | -0.6% | +24.1% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | p50_us | 96.21 | 90.85 | -5.6% | +23.0% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | p99_us | 212.9 | 242.7 | +14.0% | +63.5% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | throughput_mb_s | 42.57 | 45.08 | +5.9% | +27.5% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | cpu_per_op_us | 145.5 | 146.4 | +0.6% | +33.8% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | copied_bytes | 5243 | 5128 | -2.2% | +8.4% | 2/5 | no change (inside spread) |
| observe | economy | 4096 | rss_growth_bytes | 6.554e+04 | 1.311e+05 | +100.0% | +150.0% | 1/5 | no change (< 2 MiB) |
| observe | economy | 4096 | maxrss_bytes | 2.265e+08 | 2.261e+08 | -0.2% | +0.3% | 3/5 | no change (< 2 MiB) |
| observe | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy | 1048576 | p50_us | 1028 | 1007 | -2.0% | +30.0% | 5/5 | no change (inside spread) |
| observe | economy | 1048576 | p99_us | 1201 | 1151 | -4.1% | +111.1% | 3/5 | no change (inside spread) |
| observe | economy | 1048576 | throughput_mb_s | 1020 | 1041 | +2.1% | +23.7% | 5/5 | no change (inside spread) |
| observe | economy | 1048576 | cpu_per_op_us | 1190 | 1160 | -2.5% | +31.8% | 4/5 | no change (inside spread) |
| observe | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | economy | 1048576 | rss_growth_bytes | 1.311e+05 | 1.311e+05 | +0.0% | +1312.5% | 2/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | maxrss_bytes | 2.33e+08 | 2.324e+08 | -0.3% | +1.1% | 5/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy |  | idle_cpu_per_s | 0.003689 | 0.003487 | -5.5% | +18.1% | 4/5 | no change (inside spread) |
| observe | latency | 4096 | p50_us | 103.5 | 92.27 | -10.9% | +34.2% | 4/5 | no change (inside spread) |
| observe | latency | 4096 | p99_us | 200.8 | 140 | -30.3% | +74.8% | 3/5 | no change (inside spread) |
| observe | latency | 4096 | throughput_mb_s | 39.56 | 44.39 | +12.2% | +26.8% | 4/5 | no change (inside spread) |
| observe | latency | 4096 | cpu_per_op_us | 219 | 188.7 | -13.8% | +29.5% | 4/5 | no change (inside spread) |
| observe | latency | 4096 | copied_bytes | 5374 | 5489 | +2.1% | +24.8% | 1/5 | no change (inside spread) |
| observe | latency | 4096 | rss_growth_bytes | 8.192e+04 | 6.554e+04 | -20.0% | +100.0% | 3/5 | no change (< 2 MiB) |
| observe | latency | 4096 | maxrss_bytes | 2.265e+08 | 2.267e+08 | +0.1% | +0.7% | 2/5 | no change (< 2 MiB) |
| observe | latency | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | latency | 1048576 | p50_us | 1494 | 1488 | -0.4% | +2.9% | 3/5 | no change (inside spread) |
| observe | latency | 1048576 | p99_us | 1638 | 1742 | +6.4% | +249.1% | 1/5 | no change (inside spread) |
| observe | latency | 1048576 | throughput_mb_s | 701.9 | 704.5 | +0.4% | +3.0% | 3/5 | no change (inside spread) |
| observe | latency | 1048576 | cpu_per_op_us | 2994 | 2968 | -0.9% | +12.1% | 3/5 | no change (inside spread) |
| observe | latency | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | latency | 1048576 | rss_growth_bytes | 1.475e+05 | 1.475e+05 | +0.0% | +1166.7% | 3/5 | no change (< 2 MiB) |
| observe | latency | 1048576 | maxrss_bytes | 2.33e+08 | 2.333e+08 | +0.1% | +1.6% | 1/5 | no change (< 2 MiB) |
| observe | latency | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | latency |  | idle_cpu_per_s | 1.002 | 1.002 | -0.0% | +0.5% | 3/5 | no change (inside spread) |
