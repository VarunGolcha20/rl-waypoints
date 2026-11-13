# Results: omp_lift_55mm

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorP/Lift/omp_lift_55mm_s0 | 399 | 57.98 | 70% | 100 | 44.29 ± 21.48 | 50% |
| OpenManipulatorP/Lift/omp_lift_55mm_s1 | 399 | 67.13 | 94% | 100 | 69.56 ± 3.27 | 99% |
| OpenManipulatorP/Lift/omp_lift_55mm_s2_best (BEST) | 399 | 68.21 | 100% | 100 | 69.76 ± 4.79 | 99% |
| **mean over 3 seeds** | | | | | 61.21 ± 11.96 | 83% ± 23% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
