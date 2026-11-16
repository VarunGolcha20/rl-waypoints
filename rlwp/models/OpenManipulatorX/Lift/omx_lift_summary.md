# Results: omx_lift

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Lift/omx_lift_s0 | 399 | 23.60 | 4% | 100 | 20.06 ± 4.60 | 9% |
| OpenManipulatorX/Lift/omx_lift_s1 | 399 | 39.83 | 76% | 100 | 44.27 ± 20.67 | 81% |
| OpenManipulatorX/Lift/omx_lift_s2 | 399 | 35.87 | 60% | 100 | 35.18 ± 20.44 | 60% |
| **mean over 3 seeds** | | | | | 33.17 ± 9.98 | 50% ± 30% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
