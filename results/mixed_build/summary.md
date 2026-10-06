| Experiment | Host node (store) | Guest node | Result | Rdzv | Init | Vendor map | all_reduce | broadcast | Shutdown | Wall time | Attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.14.1-1+1 | 1x cu-2.14.1 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 9.0s | 1 |
| 2.14.1-2+2 | 2x rocm-2.14.1 | 2x cu-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| 2.14.1-3+1 | 3x cu-2.14.1 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 9.0s | 1 |
| 2.9.1-1+1 | 1x cu-2.9.1 | 1x rocm-2.9.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| 2.9.1-2+2 | 2x rocm-2.9.1 | 2x cu-2.9.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| 2.9.1-3+1 | 3x cu-2.9.1 | 1x rocm-2.9.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| minor-1+1-cu-hosts | 1x cu-2.13.0 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 9.0s | 1 |
| minor-1+1-rocm-hosts | 1x rocm-2.14.1 | 1x cu-2.13.0 | PASS | ok | ok | ok | ok | ok | ok | 9.0s | 1 |
| minor-2+2 | 2x cu-2.13.0 | 2x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 9.0s | 1 |
| wide-cu2.14-rocm2.9 | 1x cu-2.14.1 | 1x rocm-2.9.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| wide-cu2.9-rocm2.14 | 1x cu-2.9.1 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| control-cu2.13-cu2.14 | 1x cu-2.13.0 | 1x cu-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
