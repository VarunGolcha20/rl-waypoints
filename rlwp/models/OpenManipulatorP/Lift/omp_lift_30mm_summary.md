# Results: omp_lift_30mm

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorP/Lift/omp_lift_30mm_s0 | 399 | 66.26 | 98% | 100 | 68.67 ± 0.18 | 100% |
| OpenManipulatorP/Lift/omp_lift_30mm_s1 | 399 | 53.82 | 78% | 100 | 57.26 ± 20.66 | 77% |
| OpenManipulatorP/Lift/omp_lift_30mm_s2 | 399 | 66.56 | 100% | 100 | 58.80 ± 16.73 | 85% |
| **mean over 3 seeds** | | | | | 61.58 ± 5.05 | 87% ± 10% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
