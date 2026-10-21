# CPU/MOCK, not a GPU or RDMA speedup
| series | policy | bytes | metric | before | after | change | spread | pairs better | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host | economy | 4096 | p50_us | 104.4 | 109.8 | +5.1% | +27.9% | 0/5 | no change (inside spread) |
| host | economy | 4096 | p99_us | 158 | 224.8 | +42.3% | +82.9% | 0/5 | no change (inside spread) |
| host | economy | 4096 | throughput_mb_s | 39.23 | 37.32 | -4.9% | +37.2% | 0/5 | no change (inside spread) |
| host | economy | 4096 | cpu_per_op_us | 111.9 | 115.9 | +3.6% | +27.5% | 0/5 | no change (inside spread) |
| host | economy | 4096 | copied_bytes | 4424 | 4555 | +3.0% | +16.9% | 1/5 | no change (inside spread) |
| host | economy | 4096 | rss_growth_bytes | 1.147e+05 | 9.83e+04 | -14.3% | +257.1% | 2/5 | no change (< 2 MiB) |
| host | economy | 4096 | maxrss_bytes | 2.173e+08 | 2.183e+08 | +0.4% | +1.9% | 2/5 | no change (< 2 MiB) |
| host | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 65536 | p50_us | 261 | 260 | -0.4% | +46.4% | 4/5 | no change (inside spread) |
| host | economy | 65536 | p99_us | 353.2 | 373.7 | +5.8% | +209.2% | 2/5 | no change (inside spread) |
| host | economy | 65536 | throughput_mb_s | 251.1 | 252 | +0.4% | +81.4% | 4/5 | no change (inside spread) |
| host | economy | 65536 | cpu_per_op_us | 263.3 | 251.6 | -4.4% | +45.3% | 3/5 | no change (inside spread) |
| host | economy | 65536 | copied_bytes | 6.597e+04 | 6.641e+04 | +0.7% | +7.9% | 1/5 | no change (inside spread) |
| host | economy | 65536 | rss_growth_bytes | 3.113e+05 | 1.638e+05 | -47.4% | +278.9% | 3/5 | no change (< 2 MiB) |
| host | economy | 65536 | maxrss_bytes | 2.189e+08 | 2.189e+08 | -0.0% | +1.3% | 3/5 | no change (< 2 MiB) |
| host | economy | 65536 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 65536 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 1048576 | p50_us | 2805 | 2783 | -0.8% | +40.4% | 4/5 | no change (inside spread) |
| host | economy | 1048576 | p99_us | 3447 | 3455 | +0.2% | +136.6% | 2/5 | no change (inside spread) |
| host | economy | 1048576 | throughput_mb_s | 373.8 | 376.7 | +0.8% | +63.2% | 4/5 | no change (inside spread) |
| host | economy | 1048576 | cpu_per_op_us | 2941 | 2829 | -3.8% | +41.2% | 4/5 | no change (inside spread) |
| host | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 1048576 | rss_growth_bytes | 4.522e+06 | 4.342e+06 | -4.0% | +71.4% | 3/5 | no change (< 2 MiB) |
| host | economy | 1048576 | maxrss_bytes | 2.332e+08 | 2.34e+08 | +0.3% | +2.8% | 2/5 | no change (< 2 MiB) |
| host | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy | 16777216 | p50_us | 4.149e+04 | 4.029e+04 | -2.9% | +28.1% | 2/5 | no change (inside spread) |
| host | economy | 16777216 | p99_us | 4.349e+04 | 4.63e+04 | +6.5% | +35.4% | 3/5 | no change (inside spread) |
| host | economy | 16777216 | throughput_mb_s | 404.4 | 416.5 | +3.0% | +34.5% | 2/5 | no change (inside spread) |
| host | economy | 16777216 | cpu_per_op_us | 4.557e+04 | 4.471e+04 | -1.9% | +26.0% | 3/5 | no change (inside spread) |
| host | economy | 16777216 | copied_bytes | 1.678e+07 | 1.678e+07 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| host | economy | 16777216 | rss_growth_bytes | 1.081e+07 | 1.057e+07 | -2.3% | +109.1% | 3/5 | no change (< 2 MiB) |
| host | economy | 16777216 | maxrss_bytes | 3.289e+08 | 3.289e+08 | -0.0% | +0.7% | 4/5 | no change (< 2 MiB) |
| host | economy | 16777216 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | crossings_total_per_call | 12 | 12 | +0.0% | +0.0% | 0/5 | same |
| host | economy | 16777216 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| host | economy |  | idle_cpu_per_s | 0.001519 | 0.0013 | -14.4% | +40.8% | 4/5 | no change (inside spread) |
| observe | economy | 4096 | p50_us | 162.4 | 158.8 | -2.2% | +16.2% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | p99_us | 317.3 | 243.2 | -23.3% | +41.1% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | throughput_mb_s | 25.23 | 25.8 | +2.3% | +16.7% | 3/5 | no change (inside spread) |
| observe | economy | 4096 | cpu_per_op_us | 212.7 | 212.6 | -0.1% | +15.4% | 2/5 | no change (inside spread) |
| observe | economy | 4096 | copied_bytes | 4964 | 5177 | +4.3% | +9.2% | 2/5 | no change (inside spread) |
| observe | economy | 4096 | rss_growth_bytes | 3.277e+04 | 1.638e+05 | +400.0% | +1200.0% | 1/5 | no change (< 2 MiB) |
| observe | economy | 4096 | maxrss_bytes | 2.179e+08 | 2.174e+08 | -0.2% | +1.2% | 3/5 | no change (< 2 MiB) |
| observe | economy | 4096 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 4096 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy | 1048576 | p50_us | 2754 | 2668 | -3.1% | +42.1% | 3/5 | no change (inside spread) |
| observe | economy | 1048576 | p99_us | 3465 | 3443 | -0.6% | +58.0% | 3/5 | no change (inside spread) |
| observe | economy | 1048576 | throughput_mb_s | 380.7 | 393 | +3.2% | +53.6% | 3/5 | no change (inside spread) |
| observe | economy | 1048576 | cpu_per_op_us | 2765 | 2757 | -0.3% | +40.6% | 3/5 | no change (inside spread) |
| observe | economy | 1048576 | copied_bytes | 1.049e+06 | 1.049e+06 | +0.0% | +0.0% | 0/5 | no change (inside spread) |
| observe | economy | 1048576 | rss_growth_bytes | 4.407e+06 | 4.342e+06 | -1.5% | +91.1% | 2/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | maxrss_bytes | 2.341e+08 | 2.34e+08 | -0.1% | +0.8% | 4/5 | no change (< 2 MiB) |
| observe | economy | 1048576 | control_sent | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | registrations | 0 | 0 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | crossings_total_per_call | 3 | 3 | +0.0% | +0.0% | 0/5 | same |
| observe | economy | 1048576 | mismatches | 0 | 0 | +0.0% | +0.0% | 0/5 | ok |
| observe | economy |  | idle_cpu_per_s | 0.001039 | 0.0009608 | -7.5% | +38.3% | 4/5 | no change (inside spread) |
