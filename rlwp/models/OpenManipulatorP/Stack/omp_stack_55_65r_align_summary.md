# Results: omp_stack_55_65r_align

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorP/Stack/omp_stack_55_65r_align_s0 | 299 | 122.35 | 96% | 100 | 119.91 ± 17.53 | 90% |
| OpenManipulatorP/Stack/omp_stack_55_65r_align_s1_best (BEST) | 299 | 126.01 | 100% | 100 | 124.28 ± 9.82 | 99% |
| OpenManipulatorP/Stack/omp_stack_55_65r_align_s2 | 299 | 31.33 | 0% | 100 | 25.39 ± 0.17 | 0% |
| **mean over 3 seeds** | | | | | 89.86 ± 45.62 | 63% ± 45% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)
