# CPU/MOCK, not a GPU or RDMA speedup
| series | policy | bytes | metric | before | after | change | spread | pairs better | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host | economy | 4096 | p50_us | 212.4 | 99.29 | -53.3% | +199.4% | 5/5 | no change (inside spread) |
| host | economy | 4096 | p99_us | 1624 | 172.8 | -89.4% | +1874.9% | 4/5 | no change (inside spread) |
| host | economy | 4096 | throughput_mb_s | 19.28 | 41.25 | +113.9% | +182.3% | 5/5 | no change (inside spread) |
| host | economy | 4096 | cpu_per_op_us | 143.5 | 109.9 | -23.4% | +69.8% | 4/5 | no change (inside spread) |
| host | economy | 4096 | copied_bytes | 5030 | 4506 | -10.4% | +22.8% | 5/5 | no change (inside spread) |
| host | economy | 4096 | rss_growth_bytes | 1.638e+04 | 6.554e+04 | +300.0% | +800.0% | 1/5 | no change (< 2 MiB) |
| host | economy | 4096 | maxrss_bytes | 2.189e+08 | 2.188e+08 | -0.0% | +4.9% | 2/5 | no change (< 2 MiB) |
| host | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 65536 | p50_us | 940.6 | 235.4 | -75.0% | +311.3% | 5/5 | no change (inside spread) |
| host | economy | 65536 | p99_us | 1.641e+04 | 368.4 | -97.8% | +12607.0% | 4/5 | no change (inside spread) |
| host | economy | 65536 | throughput_mb_s | 69.67 | 278.4 | +299.6% | +338.1% | 5/5 | no change (inside spread) |
| host | economy | 65536 | cpu_per_op_us | 354.4 | 230.1 | -35.1% | +71.4% | 5/5 | no change (inside spread) |
| host | economy | 65536 | copied_bytes | 7.384e+04 | 6.554e+04 | -11.2% | +21.3% | 4/5 | no change (inside spread) |
| host | economy | 65536 | rss_growth_bytes | 8.192e+05 | 1.802e+05 | -78.0% | +563.6% | 4/5 | no change (< 2 MiB) |
| host | economy | 65536 | maxrss_bytes | 2.198e+08 | 2.194e+08 | -0.2% | +4.1% | 4/5 | no change (< 2 MiB) |
| host | economy | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 1048576 | p50_us | 6719 | 2827 | -57.9% | +277.2% | 5/5 | no change (inside spread) |
| host | economy | 1048576 | p99_us | 9.67e+04 | 2.514e+04 | -74.0% | +619.2% | 4/5 | no change (inside spread) |
| host | economy | 1048576 | throughput_mb_s | 156.1 | 370.9 | +137.6% | +232.2% | 5/5 | no change (inside spread) |
| host | economy | 1048576 | cpu_per_op_us | 2620 | 2619 | -0.1% | +40.2% | 3/5 | no change (inside spread) |
| host | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 1048576 | rss_growth_bytes | 5.669e+06 | 5.308e+06 | -6.4% | +82.9% | 3/5 | no change (< 2 MiB) |
| host | economy | 1048576 | maxrss_bytes | 2.384e+08 | 2.363e+08 | -0.9% | +3.2% | 4/5 | no change (< 2 MiB) |
| host | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 16777216 | p50_us | 4.285e+04 | 4.266e+04 | -0.4% | +558.9% | 4/5 | no change (inside spread) |
| host | economy | 16777216 | p99_us | 9.677e+04 | 1.219e+05 | +25.9% | +510.4% | 3/5 | no change (inside spread) |
| host | economy | 16777216 | throughput_mb_s | 391.5 | 393.2 | +0.4% | +107.9% | 4/5 | no change (inside spread) |
| host | economy | 16777216 | cpu_per_op_us | 3.999e+04 | 4.379e+04 | +9.5% | +29.5% | 2/5 | no change (inside spread) |
| host | economy | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | rss_growth_bytes | 1.193e+07 | 1.139e+07 | -4.5% | +101.4% | 3/5 | no change (< 2 MiB) |
| host | economy | 16777216 | maxrss_bytes | 3.319e+08 | 3.309e+08 | -0.3% | +3.0% | 4/5 | no change (< 2 MiB) |
| host | economy | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy |  | idle_cpu_per_s | 0.000932 | 0.0009991 | +7.2% | +41.0% | 1/5 | no change (inside spread) |
| observe | economy | 4096 | p50_us | 150.6 | 152.3 | +1.1% | +10.4% | 0/5 | no change (inside spread) |
| observe | economy | 4096 | p99_us | 299.6 | 238.5 | -20.4% | +44.4% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | throughput_mb_s | 27.19 | 26.9 | -1.1% | +10.2% | 0/5 | no change (inside spread) |
| observe | economy | 4096 | cpu_per_op_us | 209 | 205.2 | -1.8% | +7.2% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | copied_bytes | 4948 | 5046 | +2.0% | +19.2% | 2/5 | no change (inside spread) |
| observe | economy | 4096 | rss_growth_bytes | 3.277e+04 | 8.192e+04 | +150.0% | +300.0% | 1/5 | no change (< 2 MiB) |
| observe | economy | 4096 | maxrss_bytes | 2.183e+08 | 2.183e+08 | -0.0% | +1.4% | 3/5 | no change (< 2 MiB) |
| observe | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy | 1048576 | p50_us | 2144 | 2611 | +21.8% | +21.3% | 0/5 | REGRESSION |
| observe | economy | 1048576 | p99_us | 2950 | 3778 | +28.1% | +351.6% | 1/5 | no change (inside spread) |
| observe | economy | 1048576 | throughput_mb_s | 489 | 401.7 | -17.9% | +23.3% | 0/5 | no change (inside spread) |
| observe | economy | 1048576 | cpu_per_op_us | 2220 | 2582 | +16.3% | +18.6% | 1/5 | no change (inside spread) |
| observe | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | economy | 1048576 | rss_growth_bytes | 5.095e+06 | 4.604e+06 | -9.6% | +78.6% | 3/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | maxrss_bytes | 2.33e+08 | 2.344e+08 | +0.6% | +1.9% | 1/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy |  | idle_cpu_per_s | 0.001422 | 0.001208 | -15.1% | +45.9% | 4/5 | no change (inside spread) |
