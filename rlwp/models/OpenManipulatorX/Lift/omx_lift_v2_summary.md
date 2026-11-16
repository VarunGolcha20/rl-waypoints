# Results: omx_lift_v2

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Lift/omx_lift_v2_s0 | 399 | 66.19 | 94% | 100 | 69.14 ± 5.57 | 99% |
| OpenManipulatorX/Lift/omx_lift_v2_s1 | 399 | 64.32 | 90% | 100 | 69.74 ± 0.43 | 100% |
| OpenManipulatorX/Lift/omx_lift_v2_s2 | 399 | 64.53 | 90% | 100 | 62.67 ± 15.98 | 83% |
| **mean over 3 seeds** | | | | | 67.18 ± 3.20 | 94% ± 8% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
