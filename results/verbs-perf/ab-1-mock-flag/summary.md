# CPU/MOCK, not a GPU or RDMA speedup
| series | policy | bytes | metric | before | after | change | spread | pairs better | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host | balanced | 4096 | p50_us | 74.27 | 63.42 | -14.6% | +48.9% | 3/5 | no change (inside spread) |
| host | balanced | 4096 | p99_us | 161.5 | 127 | -21.4% | +55.3% | 4/5 | no change (inside spread) |
| host | balanced | 4096 | throughput_mb_s | 55.15 | 64.59 | +17.1% | +56.6% | 3/5 | no change (inside spread) |
| host | balanced | 4096 | cpu_per_op_us | 106.6 | 87.31 | -18.1% | +37.0% | 4/5 | no change (inside spread) |
| host | balanced | 4096 | copied_bytes | 4293 | 4260 | -0.8% | +6.2% | 3/5 | no change (inside spread) |
| host | balanced | 4096 | rss_growth_bytes | 8.192e+04 | 8.192e+04 | +0.0% | +160.0% | 2/5 | no change (< 2 MiB) |
| host | balanced | 4096 | maxrss_bytes | 2.265e+08 | 2.266e+08 | +0.0% | +0.2% | 1/5 | no change (< 2 MiB) |
| host | balanced | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 65536 | p50_us | 95.98 | 95.54 | -0.5% | +21.1% | 4/5 | no change (inside spread) |
| host | balanced | 65536 | p99_us | 125.5 | 123.5 | -1.6% | +58.3% | 5/5 | no change (inside spread) |
| host | balanced | 65536 | throughput_mb_s | 682.8 | 685.9 | +0.5% | +18.5% | 4/5 | no change (inside spread) |
| host | balanced | 65536 | cpu_per_op_us | 117.5 | 116.3 | -1.0% | +8.1% | 5/5 | no change (inside spread) |
| host | balanced | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 65536 | rss_growth_bytes | 0 | 0 | +0.0% | inf | 1/5 | no change (< 2 MiB) |
| host | balanced | 65536 | maxrss_bytes | 2.269e+08 | 2.27e+08 | +0.0% | +0.4% | 3/5 | no change (< 2 MiB) |
| host | balanced | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 1048576 | p50_us | 999.6 | 1002 | +0.2% | +24.7% | 4/5 | no change (inside spread) |
| host | balanced | 1048576 | p99_us | 1158 | 1064 | -8.1% | +59.7% | 4/5 | no change (inside spread) |
| host | balanced | 1048576 | throughput_mb_s | 1049 | 1047 | -0.2% | +19.9% | 4/5 | no change (inside spread) |
| host | balanced | 1048576 | cpu_per_op_us | 1186 | 1171 | -1.3% | +20.0% | 4/5 | no change (inside spread) |
| host | balanced | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 1048576 | rss_growth_bytes | 1.475e+05 | 2.949e+05 | +100.0% | +200.0% | 2/5 | no change (< 2 MiB) |
| host | balanced | 1048576 | maxrss_bytes | 2.33e+08 | 2.33e+08 | +0.0% | +0.2% | 3/5 | no change (< 2 MiB) |
| host | balanced | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced | 16777216 | p50_us | 1.487e+04 | 1.489e+04 | +0.2% | +1.1% | 1/5 | no change (inside spread) |
| host | balanced | 16777216 | p99_us | 1.549e+04 | 1.549e+04 | -0.0% | +13.2% | 2/5 | no change (inside spread) |
| host | balanced | 16777216 | throughput_mb_s | 1129 | 1127 | -0.2% | +1.1% | 1/5 | no change (inside spread) |
| host | balanced | 16777216 | cpu_per_op_us | 1.861e+04 | 1.843e+04 | -1.0% | +2.3% | 3/5 | no change (inside spread) |
| host | balanced | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | balanced | 16777216 | rss_growth_bytes | 1.638e+04 | 1.638e+04 | +0.0% | +20600.0% | 1/5 | no change (< 2 MiB) |
| host | balanced | 16777216 | maxrss_bytes | 3.003e+08 | 3.003e+08 | +0.0% | +1.5% | 3/5 | no change (< 2 MiB) |
| host | balanced | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | balanced | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | balanced |  | idle_cpu_per_s | 0.003513 | 0.00378 | +7.6% | +33.5% | 2/5 | no change (inside spread) |
| host | economy | 4096 | p50_us | 70.71 | 70.67 | -0.1% | +34.3% | 2/5 | no change (inside spread) |
| host | economy | 4096 | p99_us | 157.6 | 161.4 | +2.4% | +27.3% | 1/5 | no change (inside spread) |
| host | economy | 4096 | throughput_mb_s | 57.93 | 57.96 | +0.1% | +50.7% | 2/5 | no change (inside spread) |
| host | economy | 4096 | cpu_per_op_us | 88.98 | 90.7 | +1.9% | +25.0% | 1/5 | no change (inside spread) |
| host | economy | 4096 | copied_bytes | 4162 | 4162 | +0.0% | +2.4% | 2/5 | no change (inside spread) |
| host | economy | 4096 | rss_growth_bytes | 4.915e+04 | 8.192e+04 | +66.7% | +100.0% | 2/5 | no change (< 2 MiB) |
| host | economy | 4096 | maxrss_bytes | 2.266e+08 | 2.263e+08 | -0.1% | +0.4% | 3/5 | no change (< 2 MiB) |
| host | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 65536 | p50_us | 95.33 | 94.5 | -0.9% | +2.0% | 5/5 | no change (inside spread) |
| host | economy | 65536 | p99_us | 125.1 | 123 | -1.7% | +4.3% | 4/5 | no change (inside spread) |
| host | economy | 65536 | throughput_mb_s | 687.4 | 693.5 | +0.9% | +2.1% | 5/5 | no change (inside spread) |
| host | economy | 65536 | cpu_per_op_us | 108.8 | 107.8 | -1.0% | +2.8% | 3/5 | no change (inside spread) |
| host | economy | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 65536 | rss_growth_bytes | 1.638e+04 | 0 | -100.0% | inf | 2/5 | no change (< 2 MiB) |
| host | economy | 65536 | maxrss_bytes | 2.269e+08 | 2.266e+08 | -0.1% | +0.3% | 3/5 | no change (< 2 MiB) |
| host | economy | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 1048576 | p50_us | 988.4 | 992.1 | +0.4% | +18.0% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | p99_us | 1060 | 1274 | +20.1% | +84.4% | 2/5 | no change (inside spread) |
| host | economy | 1048576 | throughput_mb_s | 1061 | 1057 | -0.4% | +15.4% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | cpu_per_op_us | 1140 | 1144 | +0.3% | +11.9% | 1/5 | no change (inside spread) |
| host | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 1048576 | rss_growth_bytes | 6.554e+04 | 1.475e+05 | +125.0% | +250.0% | 2/5 | no change (< 2 MiB) |
| host | economy | 1048576 | maxrss_bytes | 2.328e+08 | 2.326e+08 | -0.1% | +0.3% | 3/5 | no change (< 2 MiB) |
| host | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 16777216 | p50_us | 1.48e+04 | 1.487e+04 | +0.4% | +14.0% | 1/5 | no change (inside spread) |
| host | economy | 16777216 | p99_us | 1.592e+04 | 1.515e+04 | -4.8% | +68.1% | 2/5 | no change (inside spread) |
| host | economy | 16777216 | throughput_mb_s | 1133 | 1128 | -0.4% | +12.5% | 1/5 | no change (inside spread) |
| host | economy | 16777216 | cpu_per_op_us | 1.838e+04 | 1.833e+04 | -0.2% | +14.6% | 3/5 | no change (inside spread) |
| host | economy | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | rss_growth_bytes | 1.966e+05 | 3.277e+04 | -83.3% | +10450.0% | 3/5 | no change (< 2 MiB) |
| host | economy | 16777216 | maxrss_bytes | 3.037e+08 | 3.034e+08 | -0.1% | +1.2% | 4/5 | no change (< 2 MiB) |
| host | economy | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy |  | idle_cpu_per_s | 0.003411 | 0.003553 | +4.2% | +27.6% | 2/5 | no change (inside spread) |
| host | latency | 4096 | p50_us | 52.06 | 52.21 | +0.3% | +14.1% | 3/5 | no change (inside spread) |
| host | latency | 4096 | p99_us | 81.88 | 80.92 | -1.2% | +37.5% | 2/5 | no change (inside spread) |
| host | latency | 4096 | throughput_mb_s | 78.67 | 78.45 | -0.3% | +15.7% | 3/5 | no change (inside spread) |
| host | latency | 4096 | cpu_per_op_us | 93.11 | 92.52 | -0.6% | +8.2% | 3/5 | no change (inside spread) |
| host | latency | 4096 | copied_bytes | 4456 | 4391 | -1.5% | +7.1% | 3/5 | no change (inside spread) |
| host | latency | 4096 | rss_growth_bytes | 8.192e+04 | 6.554e+04 | -20.0% | +120.0% | 4/5 | no change (< 2 MiB) |
| host | latency | 4096 | maxrss_bytes | 2.265e+08 | 2.266e+08 | +0.0% | +0.1% | 2/5 | no change (< 2 MiB) |
| host | latency | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 65536 | p50_us | 132.4 | 132.6 | +0.2% | +4.5% | 3/5 | no change (inside spread) |
| host | latency | 65536 | p99_us | 166.9 | 168.1 | +0.7% | +38.7% | 3/5 | no change (inside spread) |
| host | latency | 65536 | throughput_mb_s | 495.1 | 494.1 | -0.2% | +4.6% | 3/5 | no change (inside spread) |
| host | latency | 65536 | cpu_per_op_us | 253.1 | 254.5 | +0.5% | +5.3% | 3/5 | no change (inside spread) |
| host | latency | 65536 | copied_bytes | 6.554e+04 | 6.554e+04 | +0.0% | +1.3% | 1/5 | no change (inside spread) |
| host | latency | 65536 | rss_growth_bytes | 0 | 0 | +0.0% | inf | 1/5 | no change (< 2 MiB) |
| host | latency | 65536 | maxrss_bytes | 2.27e+08 | 2.27e+08 | +0.0% | +0.2% | 2/5 | no change (< 2 MiB) |
| host | latency | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 1048576 | p50_us | 1463 | 1466 | +0.1% | +17.6% | 2/5 | no change (inside spread) |
| host | latency | 1048576 | p99_us | 1582 | 1880 | +18.8% | +41.7% | 1/5 | no change (inside spread) |
| host | latency | 1048576 | throughput_mb_s | 716.5 | 715.5 | -0.1% | +15.3% | 2/5 | no change (inside spread) |
| host | latency | 1048576 | cpu_per_op_us | 2917 | 2947 | +1.0% | +8.2% | 1/5 | no change (inside spread) |
| host | latency | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | latency | 1048576 | rss_growth_bytes | 1.311e+05 | 2.13e+05 | +62.5% | +125.0% | 0/5 | no change (< 2 MiB) |
| host | latency | 1048576 | maxrss_bytes | 2.329e+08 | 2.33e+08 | +0.0% | +0.9% | 2/5 | no change (< 2 MiB) |
| host | latency | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency | 16777216 | p50_us | 2.202e+04 | 2.216e+04 | +0.6% | +16.2% | 1/5 | no change (inside spread) |
| host | latency | 16777216 | p99_us | 2.329e+04 | 2.828e+04 | +21.4% | +42.9% | 0/5 | no change (inside spread) |
| host | latency | 16777216 | throughput_mb_s | 761.9 | 757 | -0.6% | +14.6% | 1/5 | no change (inside spread) |
| host | latency | 16777216 | cpu_per_op_us | 4.594e+04 | 4.779e+04 | +4.0% | +14.1% | 1/5 | no change (inside spread) |
| host | latency | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | latency | 16777216 | rss_growth_bytes | 6.554e+04 | 1.311e+05 | +100.0% | +5175.0% | 3/5 | no change (< 2 MiB) |
| host | latency | 16777216 | maxrss_bytes | 3.007e+08 | 3.009e+08 | +0.1% | +1.6% | 2/5 | no change (< 2 MiB) |
| host | latency | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | latency | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | latency |  | idle_cpu_per_s | 1.001 | 0.9998 | -0.1% | +0.4% | 3/5 | no change (inside spread) |
| observe | balanced | 4096 | p50_us | 120.4 | 107.9 | -10.4% | +20.2% | 4/5 | no change (inside spread) |
| observe | balanced | 4096 | p99_us | 327.9 | 282.5 | -13.9% | +51.7% | 4/5 | no change (inside spread) |
| observe | balanced | 4096 | throughput_mb_s | 34.02 | 37.98 | +11.6% | +24.4% | 4/5 | no change (inside spread) |
| observe | balanced | 4096 | cpu_per_op_us | 200.5 | 185.2 | -7.7% | +30.7% | 4/5 | no change (inside spread) |
| observe | balanced | 4096 | copied_bytes | 5177 | 5112 | -1.3% | +6.1% | 2/5 | no change (inside spread) |
| observe | balanced | 4096 | rss_growth_bytes | 6.554e+04 | 8.192e+04 | +25.0% | +100.0% | 1/5 | no change (< 2 MiB) |
| observe | balanced | 4096 | maxrss_bytes | 2.265e+08 | 2.266e+08 | +0.1% | +0.2% | 2/5 | no change (< 2 MiB) |
| observe | balanced | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 4096 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | balanced | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | balanced | 1048576 | p50_us | 1036 | 1014 | -2.1% | +30.9% | 5/5 | no change (inside spread) |
| observe | balanced | 1048576 | p99_us | 1100 | 1141 | +3.7% | +103.6% | 2/5 | no change (inside spread) |
| observe | balanced | 1048576 | throughput_mb_s | 1012 | 1034 | +2.2% | +23.8% | 5/5 | no change (inside spread) |
| observe | balanced | 1048576 | cpu_per_op_us | 1212 | 1196 | -1.3% | +26.2% | 4/5 | no change (inside spread) |
| observe | balanced | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | balanced | 1048576 | rss_growth_bytes | 2.949e+05 | 2.949e+05 | +0.0% | +388.9% | 3/5 | no change (< 2 MiB) |
| observe | balanced | 1048576 | maxrss_bytes | 2.33e+08 | 2.33e+08 | -0.0% | +0.7% | 3/5 | no change (< 2 MiB) |
| observe | balanced | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | balanced | 1048576 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | balanced | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | balanced |  | idle_cpu_per_s | 0.003742 | 0.003811 | +1.8% | +13.1% | 1/5 | no change (inside spread) |
| observe | economy | 4096 | p50_us | 119.2 | 104 | -12.8% | +32.0% | 5/5 | no change (inside spread) |
| observe | economy | 4096 | p99_us | 300.3 | 258.8 | -13.8% | +47.2% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | throughput_mb_s | 34.35 | 39.38 | +14.6% | +44.9% | 5/5 | no change (inside spread) |
| observe | economy | 4096 | cpu_per_op_us | 181.3 | 167.2 | -7.7% | +38.9% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | copied_bytes | 5095 | 5079 | -0.3% | +9.3% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | rss_growth_bytes | 9.83e+04 | 6.554e+04 | -33.3% | +275.0% | 4/5 | no change (< 2 MiB) |
| observe | economy | 4096 | maxrss_bytes | 2.266e+08 | 2.267e+08 | +0.0% | +0.3% | 3/5 | no change (< 2 MiB) |
| observe | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy | 1048576 | p50_us | 1033 | 1011 | -2.2% | +1.1% | 5/5 | win |
| observe | economy | 1048576 | p99_us | 1138 | 1073 | -5.8% | +42.5% | 4/5 | no change (inside spread) |
| observe | economy | 1048576 | throughput_mb_s | 1015 | 1037 | +2.2% | +1.1% | 5/5 | win |
| observe | economy | 1048576 | cpu_per_op_us | 1189 | 1158 | -2.6% | +3.1% | 5/5 | no change (inside spread) |
| observe | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | economy | 1048576 | rss_growth_bytes | 0 | 1.311e+05 | inf | inf | 1/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | maxrss_bytes | 2.328e+08 | 2.329e+08 | +0.0% | +0.4% | 3/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy |  | idle_cpu_per_s | 0.003705 | 0.003722 | +0.5% | +23.8% | 2/5 | no change (inside spread) |
| observe | latency | 4096 | p50_us | 117.1 | 101.8 | -13.1% | +25.9% | 5/5 | no change (inside spread) |
| observe | latency | 4096 | p99_us | 166.1 | 159.1 | -4.2% | +56.7% | 3/5 | no change (inside spread) |
| observe | latency | 4096 | throughput_mb_s | 34.98 | 40.25 | +15.0% | +21.7% | 5/5 | no change (inside spread) |
| observe | latency | 4096 | cpu_per_op_us | 240.2 | 215.1 | -10.4% | +17.6% | 5/5 | no change (inside spread) |
| observe | latency | 4096 | copied_bytes | 5243 | 5358 | +2.2% | +19.4% | 2/5 | no change (inside spread) |
| observe | latency | 4096 | rss_growth_bytes | 6.554e+04 | 8.192e+04 | +25.0% | +125.0% | 1/5 | no change (< 2 MiB) |
| observe | latency | 4096 | maxrss_bytes | 2.267e+08 | 2.267e+08 | -0.0% | +0.9% | 3/5 | no change (< 2 MiB) |
| observe | latency | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 4096 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | latency | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | latency | 1048576 | p50_us | 1536 | 1496 | -2.6% | +30.3% | 4/5 | no change (inside spread) |
| observe | latency | 1048576 | p99_us | 1951 | 1587 | -18.7% | +38.9% | 5/5 | no change (inside spread) |
| observe | latency | 1048576 | throughput_mb_s | 682.7 | 701 | +2.7% | +25.1% | 4/5 | no change (inside spread) |
| observe | latency | 1048576 | cpu_per_op_us | 3078 | 2984 | -3.1% | +23.9% | 4/5 | no change (inside spread) |
| observe | latency | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | latency | 1048576 | rss_growth_bytes | 1.311e+05 | 1.311e+05 | +0.0% | +350.0% | 2/5 | no change (< 2 MiB) |
| observe | latency | 1048576 | maxrss_bytes | 2.331e+08 | 2.33e+08 | -0.0% | +1.0% | 3/5 | no change (< 2 MiB) |
| observe | latency | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | latency | 1048576 | crossings_total_per_call | 4 | 3 | -25.0% | +0.0% | 5/5 | better |
| observe | latency | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | latency |  | idle_cpu_per_s | 0.9987 | 1 | +0.2% | +0.4% | 0/5 | no change (inside spread) |
