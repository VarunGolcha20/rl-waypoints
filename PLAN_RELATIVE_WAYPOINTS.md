# Plan: object-relative waypoints for Stack on the OpenMANIPULATOR-X

Status: **plan only, not implemented** (2026-11-23). Branch `omx-port`. Related open problems in
[OMX_PORT.md](OMX_PORT.md) §5.3: P1 (off-centre grasp), P6 (landing area), P8 (several optima over a large area).

## 1. Problem

The trained Stack models only work for cube positions they were trained on. Hardware dry run, 2026-11-23, model
`omx_stack_v4_s1`, cubeA at (14, +9) cm and the landing block at (5.0, 17.5) cm (robot base frame: origin on joint 1's
axis, x straight out, y left):

* waypoint 1 (the grasp) went to (16.5, 12.9), 4.7 cm away from cubeA, and the gripper never closed;
* both waypoints sat exactly on the reach ring's outer edge (20.9 / 21.0 cm from joint 1). In the trained layout the
  release (waypoint 2) also sits on that edge (21.0 cm), whatever the block's distance.

Cause. rlwp does not learn a policy. It learns a **reward model** (an ensemble of MLPs) that scores
`[waypoints, absolute object positions]`, then picks the waypoints with an optimiser (SLSQP, random restarts). The
waypoints are absolute points (`target = start_pose + w`). To grasp a moved cube, the model would have to learn "the
reward is high where the waypoint equals the cube's position", a relationship between two inputs. The training data
never showed it: cubeA only spawned in a 2 × 3 cm patch, and the open gripper (82 mm) is wide enough for one fixed grasp
point to catch a 55 mm cube anywhere in that patch. So the model learned "grasp around here", not "grasp at the cube".
For an unseen cube position its input is out of distribution, and the optimiser finds the maximum of an extrapolated
function, typically at the boundary of the search region.

## 2. Idea

Make each waypoint an **offset from the object it acts on**, and let the reward model see **only the offsets**.

```
today:     target = start_pose + w            w: where on the table
planned:   target = object_pos + R(θ) · w     w: where relative to the object, in the arm's frame
```

* Waypoint 1 (grasp) is relative to cubeA, waypoint 2 (release) relative to the landing block (cubeB).
* The offset is expressed in the **arm's frame** around joint 1: radial (away from joint 1), tangential (sideways), up.
  θ is the object's bearing seen from joint 1, R(θ) the rotation by θ about the vertical axis.
* The reward models get the offsets (and gripper commands) only, no absolute positions.

Why this generalizes, rather than extrapolating better:

1. **The inputs stop being new.** With offsets only, the reward model can't tell where the cube is: a cube at (14, +9)
   gives it exactly the inputs a trained cube gives. Nothing is extrapolated, because nothing in the input is unseen.
2. **The assumption behind it holds on this arm.** Offsets-only assumes the right grasp relative to the cube doesn't
   depend on where the cube is. The OM-X is rotationally symmetric about joint 1 (joint 1 turns the whole arm, joints
   2–4 bend in one vertical plane), so a cube at bearing 33° is the trained situation rotated about joint 1, provided
   (a) offsets are in the arm's frame and (b) the cube is turned to face joint 1 (the gripper always faces joint 1, so
   the fingers then meet the cube face-on at every bearing). What remains is the **distance from joint 1**: cubeA at
   (14, +9) is 16.6 cm away, inside the trained band of 16.3–18.5 cm.
3. **All training episodes teach the same thing** (one good offset), so the data needed is far smaller than for
   learning the position dependence.

What it does not fix: distances outside the trained band (other arm configurations, tracking error near the reach
limit); anything that breaks the symmetry (a cube not turned toward joint 1, obstacles); the reach limit itself (cube
position + offset must stay reachable); P1 (the learned offset may still be ~1 cm off-centre, a reward change).

## 3. Design

All new behaviour sits behind new config keys. **Without them every existing config runs exactly as before** (this
fork's rule; checked by seeded bit-identical smoke runs, §5).

### 3.1 Config (new task `rlwp/cfg/task/Stack_omx_rel.yaml`)

A copy of `Stack_omx.yaml` (same layout, rewards, schedule and workspace) plus:

```yaml
env:
  env_kwargs:
    face_center: [-0.158, 0.0]          # new Stack option: cubeA and cubeB are turned (yaw) to face this point (joint 1)
  relative_waypoints:
    center: [-0.158, 0.0]               # joint 1's axis (world xy): the arm frame's origin
    refs: [cubeA, cubeB]                # per waypoint: the object it is relative to (obs key <ref>_pos)
    bounds:                             # per waypoint offset bounds (m): [radial, tangential, up], up from the object's
      - low:  [-0.03, -0.03, -0.015]    #   observed centre. Grasp: cubeA's centre is observed at table + 3.75 cm (the
        high: [ 0.03,  0.03,  0.060]    #   1 cm spawn drop), so up -1.5..+6 cm = 2.25..9.75 cm above the table
      - low:  [-0.04, -0.04,  0.050]    # release: block centre at table + 2.4 cm, so up +5..+8.1 cm = 7.4..10.5 cm
        high: [ 0.04,  0.04,  0.081]    #   (cubeA's bottom clears the 4.8 cm block top from 7.55 cm)
    object_inputs: none                 # none: reward models see the offsets only (the point of this plan);
                                        #   radius: + each ref object's distance from center; absolute: + positions
```

The existing absolute `workspace:` box and reach ring stay and become a **safety clip on the final absolute targets**
(box + ring projection), so a target can never leave the arm's verified reach or go below the fingertip floor.

Training layout: **cubeA keeps its 2 × 3 cm patch**, the block keeps its ring (17–20 cm, ±40–90°). Training on the same
small patch as v4 is deliberate: it tests the claim that generalization comes from the representation, not from more
spawn area. Schedule as v4 (400 episodes per waypoint, `save_final_models: True`).

### 3.2 Code changes

| File | Change |
|---|---|
| `rlwp/common.py` | `relative_cfg(env_cfg)`; `policy_objs(env_cfg, objs, obs)`: the reward model's object input (`none` → empty, `radius` → distances, default/absent → `objs`, the original); `waypoint_targets(env_cfg, wp_xyz, state, obs)`: absolute targets, original `wp + state` when not relative, else `obj + R(θ)·w` then the safety clip; `get_workspace()` returns a **per-waypoint list of offset bounds** when relative (unchanged otherwise) |
| `rlwp/method.py`, `rlwp/eval.py` (`Method`) | accept a per-waypoint bounds list: before optimising (or sampling, at init and in exploration) waypoint *i*, switch `lb` / `ub` / exploration bounds / constraints to waypoint *i*'s bounds; no reach-ring constraint in offset space. With a tuple workspace (all existing configs) the code path and the random-number order stay as today |
| `rlwp/train.py`, `rlwp/eval.py` (loops) | network input `policy_objs(...)` instead of `objs` (Method construction, `traj_opt`, `memory.push`); `traj_mat = waypoint_targets(...)` instead of `wp + state`. The per-step controller rule, gripper rule and `lift_first` are unchanged |
| `robosuite/.../manipulation/stack.py` | `face_center=None` option: when set, cubeA's free-joint quaternion and the fixed cubeB's `body_quat` get yaw = bearing from that point. Default `None` = today |
| `omx_hw/plan_episode.py` | `place_cubes` applies the same yaw when the config has `face_center`; targets via `waypoint_targets`, network input via `policy_objs`; `TASKS` gets the new model; `--config Stack_omx_rel`. Retreat check, finger-clearance raising and IK adjustments work on absolute joint angles and need no change |
| `omx_hw/sim_sweep.py` | same two substitutions; default grid for the relative model covers the left side too (e.g. cubeA x 10–20, y −10…+12, inside the reach ring and ≥ 9 cm from the block) |
| `omx_hw/run_trials.py` | when the plan's config has `face_center`: tell the user to **turn each cube so one face points at joint 1** (with the angle in degrees); random cubeA positions from a wider area with `--allow-outside` |
| `omx_hw/plan_episode.py` viewer | draws each cube target with its facing direction |
| `omp_tools/view_policy.py`, `omp_tools/check_real_success.py` | same two substitutions (they compute `traj[:, :3] + start` today) |
| Docs | `OMX_PORT.md` (new section + P1 / P6 / P8 notes), `omx_hw/README.md` (cube orientation, new model, generalization map) |

### 3.3 Implementation details to get right

* **Bearing per episode**: θ from the object's *reset* observation (the same observation the reward model's inputs
  come from), fixed for the episode.
* **Offsets in the reward model**: the 4 values per waypoint are (radial, tangential, up, gripper). Input size for
  waypoint *k* is `4k` with `object_inputs: none` (today `4k + 6`); models are therefore not interchangeable with v4.
* **Noise and exploration** (`NOISE ADDED TO POSE`, restarts with `N(0, 0.1)`, uniform exploration) act on offsets and
  are clipped to the offset bounds; no change in the schedule.
* **Safety clip** must not silently change what the reward model is credited for: log how often targets are clipped
  in training (expected ~0 with cubeA in its patch and the block 17–20 cm out).

## 4. Training run (detached, survives logout)

1. Smoke run first: `omp_tools/train_omp.sh -t Stack_omx_rel --smoke -s "0" -n omx_stack_rel_smoke` (~1 min).
2. Full run: 3 seeds **in parallel** (one launcher per seed, same `-n`), training only, then evaluation with 20
   optimiser restarts (v4 showed 5 restarts underrate the models, P8), then a generalization map. One wrapper script,
   started with `setsid nohup` so it is detached from the terminal and VS Code (`KillUserProcesses=no` on this machine,
   so it survives logout; `tmux` / `screen` are not installed):

   ```bash
   cd ~/Documents/RL/rl-waypoints
   setsid nohup omp_tools/train_rel_detached.sh > rlwp/logs/omx_stack_rel.out 2>&1 < /dev/null &
   # progress:  tail -f rlwp/logs/omx_stack_rel.out     per seed: tail -f rlwp/logs/omx_stack_rel/seed0_train.log
   ```

   `train_rel_detached.sh` (to be written): `train_omp.sh -t Stack_omx_rel -s <k> -n omx_stack_rel --train-only` for
   k = 0, 1, 2 in parallel, `wait`, then `train_omp.sh ... --eval-only -- n_inits=20` per seed, `summarize_runs.py`,
   and `omx_hw/sim_sweep.py` for the best seed vs `omx_stack_v4_s1` on the same grid. Final results go to
   `rlwp/logs/omx_stack_rel/summary.md` and the end of `omx_stack_rel.out`.
3. Expected time: ~2–3 h for training (3 × 800 episodes in parallel on 16 cores, as v4), ~10 min evaluation, ~10 min
   for the two maps.

## 5. Verification

| Check | Pass criterion |
|---|---|
| Unchanged defaults: seeded smoke runs of `Stack_omx`, `Lift_omx`, Panda `Lift`, OM-P `Stack` before vs after | training and evaluation data bit-identical |
| `waypoint_targets` unit check: offset (0, 0, up) at several bearings | target directly above the object; radial offset points away from joint 1 |
| Scripted grasp, no learning (`check_omp_tasks`-style): offset (0, 0, grasp height) then lift, cubeA facing joint 1, at (17.3, −1.5), (14, +9), (12, −10) | lifted at all three: confirms the offset really is position-independent on this arm |
| Smoke run of `Stack_omx_rel` | trains, saves, evaluates end to end |
| Evaluation in the trained layout (100 episodes, 20 restarts) | best seed ≥ v4_s1's 94 % |
| Generalization map (`sim_sweep.py`, cubeA grid incl. the left side) | at distances 16.3–18.5 cm from joint 1: success at most points regardless of bearing, where v4 fails; (14, +9) succeeds |
| Hardware | `run_trials.py --task Stack --config Stack_omx_rel --allow-outside` with cubes turned to face joint 1: in-the-air dry run at (14, +9), then real trials |

## 6. Risks and open questions

* **Distance dependence**: offsets-only gives one average offset for all distances. If the map shows failures at the
  edges of the distance band, switch to `object_inputs: radius` (rotation-invariant, 1-D extrapolation only) or
  widen cubeA's spawn range in distance.
* **Release tracking**: the release's tracking error grows with the block's distance (11–18 mm at 17–20 cm); with
  offsets only the model learns an average correction. Within the 25 mm centre tolerance this should hold; watch the
  off-centre distances in the evaluation.
* **Cube orientation on hardware**: the cubes must be turned to face joint 1 (the planner prints the angle). A cube
  square to the table at 33° bearing would be grasped on its edges.
* **P1 (off-centre grasp)** is not addressed by this change; a centred-grasp reward could go into the same retrain.
* **P7 (v4 grazes cubeA before the grasp)**: check whether it persists in the map / viewer.
* If the scripted grasp check (§5) fails at other bearings, stop: the symmetry assumption is wrong for this setup and
  the plan needs revising before any training.
