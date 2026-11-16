# Results: omx_stack

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Stack/omx_stack_s0 | 599 | 67.31 | 0% | 100 | 22.04 ± 28.10 | 0% |
| OpenManipulatorX/Stack/omx_stack_s1 | 599 | 101.58 | 98% | 100 | 102.74 ± 31.86 | 100% |
| OpenManipulatorX/Stack/omx_stack_s2 | 599 | 23.48 | 0% | 100 | 71.65 ± 46.37 | 50% |
| **mean over 3 seeds** | | | | | 65.48 ± 33.23 | 50% ± 41% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
