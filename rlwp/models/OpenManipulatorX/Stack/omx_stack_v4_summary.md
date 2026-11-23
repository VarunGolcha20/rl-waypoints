# Results: omx_stack_v4

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Stack/omx_stack_v4_s0 | 799 | 100.33 | 80% | 100 | 76.20 ± 43.58 | 52% |
| OpenManipulatorX/Stack/omx_stack_v4_s1 | 799 | 100.20 | 78% | 100 | 108.90 ± 16.89 | 94% |
| OpenManipulatorX/Stack/omx_stack_v4_s2 | 799 | 101.55 | 84% | 100 | 93.30 ± 32.07 | 71% |
| **mean over 3 seeds** | | | | | 92.80 ± 13.36 | 72% ± 17% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
