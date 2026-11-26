# Results: omx_stack_rel

| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |
|---|---|---|---|---|---|---|
| OpenManipulatorX/Stack/omx_stack_rel_s0 | 799 | 115.42 | 100% | 100 | 115.61 ± 3.22 | 100% |
| OpenManipulatorX/Stack/omx_stack_rel_s1 | 799 | 116.90 | 100% | 100 | 116.94 ± 9.59 | 99% |
| OpenManipulatorX/Stack/omx_stack_rel_s2 | 799 | 117.00 | 100% | 100 | 117.99 ± 3.36 | 100% |
| **mean over 3 seeds** | | | | | 116.85 ± 0.97 | 100% ± 0% |

(late-train = last 50 training episodes, i.e. the final waypoint's bandit)

Evaluation: 100 episodes per seed, 20 optimiser restarts. Trained with commit a9a9610, task Stack_omx_rel.

Success map, cubeA grid, landing block at (5.0 17.5) cm, relative model (best relative seed: omx_stack_rel_s0):

```

61/71 succeed in simulation (248 s, 20 optimiser restarts)   # = success, . = fail, o = not tested (cubes overlap), [ ] = trained area x 16.3..18.3, y -3.0..0.0
  y(cm)     10    12    14    16    18    20   <- x (cm)
    +12     o     .     #     #     #     . 
    +10     #     #     #     #     #     . 
     +8     #     #     #     #     #     . 
     +6     #     #     #     #     #     . 
     +4     #     #     #     #     #     . 
     +2     #     #     #     #     #     . 
     +0     #     #     #     #    [#]    # 
     -2     #     #     #     #    [#]    # 
     -4     #     #     #     #     #     . 
     -6     #     #     #     #     #     # 
     -8     #     #     #     #     #     . 
    -10     #     #     #     #     #     . 
```

Success map, cubeA grid, landing block at (5.0 17.5) cm, v4 model (best relative seed: omx_stack_rel_s0):

```

9/71 succeed in simulation (220 s, 20 optimiser restarts)   # = success, . = fail, o = not tested (cubes overlap), [ ] = trained area x 16.3..18.3, y -3.0..0.0
  y(cm)     10    12    14    16    18    20   <- x (cm)
    +12     o     .     .     .     .     . 
    +10     .     .     .     .     .     . 
     +8     .     .     .     .     .     . 
     +6     .     .     .     .     .     . 
     +4     .     .     .     .     .     . 
     +2     .     .     .     #     #     . 
     +0     .     .     .     #    [#]    . 
     -2     .     .     .     #    [#]    . 
     -4     .     .     .     .     #     . 
     -6     .     .     .     .     #     . 
     -8     .     .     .     #     .     . 
    -10     .     .     .     .     .     . 
```
