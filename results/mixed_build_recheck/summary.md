| Experiment | Host node (store) | Guest node | Result | Rdzv | Init | Vendor map | all_reduce | broadcast | Shutdown | Wall time | Attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.14.1-1+1 | 1x cu-2.14.1 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| 2.14.1-2+2 | 2x rocm-2.14.1 | 2x cu-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
| 2.14.1-3+1 | 3x cu-2.14.1 | 1x rocm-2.14.1 | PASS | ok | ok | ok | ok | ok | ok | 8.0s | 1 |
