# Results: omx_stack_v3

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Stack/omx_stack_v3_s0 | 599 | 115.91 | 100% | 100 | 117.05 ± 2.89 | 100% |
| OpenManipulatorX/Stack/omx_stack_v3_s1 | 599 | 114.92 | 98% | 100 | 62.83 ± 7.39 | 0% |
| OpenManipulatorX/Stack/omx_stack_v3_s2 | 599 | 113.01 | 94% | 100 | 90.96 ± 28.00 | 61% |
| **mean over 3 seeds** | | | | | 90.28 ± 22.14 | 54% ± 41% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
