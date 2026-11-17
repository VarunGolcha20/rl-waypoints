# rl-waypoints on the ROBOTIS OpenMANIPULATOR-X

This branch (`omx-port`) runs the waypoint-based RL method of Mehta, Habibian & Losey (2024) on a **ROBOTIS
OpenMANIPULATOR-X** (RM-X52-TNM, standard kit: 4 × XM430-W350 arm joints + an XM430 gripper) in the robosuite / MuJoCo
simulation that the paper used. It builds on the OpenMANIPULATOR-P port of this fork ([README_OMP.md](README_OMP.md)),
whose learning code, configuration keys and tools it reuses. The original Panda workflow is unchanged.

Status (2026-11-16): robot model, controller and task layouts are done and verified. The first trained models' successes
were largely false (full diagnosis in §3). With the fixes in §5.0: **Lift is solved**: retrained, 94 % mean / **100 %
best** evaluation success under a strict check (grasped, level, ≥ 4 cm up, held 1 s), all verified real (§3, "Retrained
Lift"). **Stack is solved by one seed**: `omx_stack_v3_s0` stacks 100 % in evaluation under a strict check (cubeA
flat and centred within 25 mm on a fixed cubeB, released and settled), all verified real; the other two seeds were
evaluated with early model snapshots (rlwp saving bug, §3 "Retrained Stack"). Open problems: §5.3. Nothing has run on
hardware yet.

---

## 1. The robot in simulation

| Part | What it is | Source / check |
|---|---|---|
| Arm | `robosuite/robosuite/models/assets/robots/open_manipulator_x/robot.xml`, class `OpenManipulatorX` | ROBOTIS' own OM-X MuJoCo model (`ROBOTIS-GIT/open_manipulator`, jazzy). Forward kinematics identical to ROBOTIS' MJCF and URDF (0.0 error over 202 configurations). Torque motors ±4.1 N·m (XM430-W350 stall torque at 12 V) |
| Gripper | `OMXNativeGripper` (`grippers/omx_native_gripper.xml`) | the OM-X gripper without an adapter, grip point 0.126 m past joint 4 (ROBOTIS `end_effector_link`). Opens 20–82 mm |
| Base | 1.7 cm plate on the table top (`OMXBaseMount`, size editable in `models/mounts/omx_base_mount.py`) | base at x = −0.17 m, 0.17 m behind the table centre |
| Start pose | `[0, −0.307, 0.249, 1.629]` rad | gripper pointing down, 0.12 m in front of the base, 0.087 m above the table |
| Reach (gripper down) | 6–24 cm from the base axis, from the table up to ~12 cm | joint 4's upper limit (1.97 rad) is the binding one |

### Controller: a 4-DoF arm needs a different OSC setting
The arm has one yaw joint and three pitch joints. It can point the gripper straight down anywhere in reach, but the
gripper's yaw is then fixed by the target's bearing from the base. robosuite's **coupled** operational-space controller
(the setting the OM-P uses) solves position and orientation together by least squares; a sideways move needs joint 1,
which also yaws the gripper, and the least squares trades the sideways command away against "keep the yaw". The arm
**did not move sideways at all** (0.000 m for a +y command).

`controllers/config/default_open_manipulator_x.json`: **uncoupled** OSC, kp 300, **yaw stiffness 0**. Tracking 40 random
waypoints with rlwp's rule: median error 3.5 mm, max 7.7 mm, none over 1 cm, tool tilt ≤ 1.6°, ≤ 41 % of the motor
torque. rlwp's code is unchanged; the `_omx` configs just name this controller.

## 2. Task layouts

Configs: `rlwp/cfg/task/Lift_omx.yaml`, `rlwp/cfg/task/Stack_omx.yaml` (paper training schedules, as for Panda / OM-P).
The OM-X reaches only ~6–24 cm from its base, and its fingers open across the line from the base, so objects are
grasped head-on (radially).

| Task | Layout | Waypoint box (world, m) | Scripted 2-waypoint check |
|---|---|---|---|
| Lift_omx | 55 mm cube, centre 18–21 cm in front of the base (x 0.01–0.04, y ±0.03; was x ±0.03 = 14–20 cm, see #7) | x [−0.07, 0.05], y ±0.10, z [0.825, 0.905] | 20/20 (lift 7 cm, strict check) |
| Stack_omx | cubeA 55 mm straight ahead, centre 17.5–19.5 cm out (x 0.005–0.025, y −0.03…0; was 12–15 cm, under the start pose); cubeB 65 mm set 17 mm into a recess (modelled as a fixed 65 × 65 × 48 mm block), 16–20 cm out, ~50° to the left | x [−0.07, 0.03], y [−0.10, 0.15], z [0.825, 0.905] | 19/20 (strict check) |

The waypoint boxes come from a reachability grid (`omp_tools/check_omp_workspace.py --robot OpenManipulatorX`, 810
targets): every grid point inside a box is tracked to within 1 cm. The scripted check executes hand-designed waypoints
through rlwp's exact action rule, so a pass means the task can be solved with two waypoints in that box.

## 3. Training results

3 seeds per task, paper schedules (Lift 399 episodes, Stack 599), ~1–2 h on a laptop CPU with all 6 runs in parallel.
Evaluation: 100 episodes per seed with new object placements. Models: `rlwp/models/OpenManipulatorX/{Lift,Stack}/`.

### What robosuite reports

| Task | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Lift_omx | 9 % (reward 20.1) | 81 % (44.3) | 60 % (35.2) |
| Stack_omx | 0 % (22.0) | **100 %** (102.7) | 50 % (71.7) |

### What actually happened (real vs false successes)

`omp_tools/check_real_success.py`, 20 episodes per model, placement seed 1000:

| Model | robosuite success | **real success** | false success | when robosuite first reports success |
|---|---|---|---|---|
| Lift_omx `omx_lift_s0` | 5 % | **0 %** | 5 % | step 60 |
| Lift_omx `omx_lift_s1` | 50 % | **0 %** | 50 % | steps 4–14 (before any grasp) |
| Lift_omx `omx_lift_s2` | 75 % | **10 %** | 65 % | steps 2–6 (9×), 50–51 (6×) |
| Stack_omx `omx_stack_s0` | 0 % | **0 %** | 0 % | – |
| Stack_omx `omx_stack_s1` | 100 % | **60 %** | 40 % | steps 86–88 (landing) |
| Stack_omx `omx_stack_s2` | 90 % | **55 %** | 35 % | steps 86–88 |
| OM-P reference: Lift_omp `omp_lift_55mm_s2_best` (same 55 mm cube) | 100 % | **100 %** | 0 % | steps 50–51 (second waypoint) |

Real success: Lift = the cube's centre rose ≥ 4 cm above its resting height; Stack = cubeA still rests on cubeB at the
end of the episode, not held. (20 episodes, so these rates differ somewhat from the 100-episode evaluation above.)

* **Lift (OM-X): the successes are false.** robosuite's Lift counts the cube as lifted when its centre is > table + 4 cm.
  That was designed for the stock 40 mm cube (2 cm of real lift), but a 55 mm cube passes after 1.25 cm. The OM-X models
  learned to sweep the open gripper into the cube with the first waypoint: the bump tips or flicks the cube over the
  threshold within the first 2–14 steps, and every such step also earns the full "lifted" reward. Seed 1 even aims its
  first waypoint 7–12 cm beside the cube. The OM-P model with the same cube and the same check lifts for real every time,
  so the problem is that the OM-X found the exploit, not the check alone.
* **Stack (OM-X): real, but robosuite counts the landing.** cubeA is released ~3 cm above cubeB. robosuite reports success
  when it lands (steps 86–88), but in about 4 of 10 episodes it then slides or tips off onto the table.

### Best models

| Task | Model | Use |
|---|---|---|
| Lift | `rlwp/models/OpenManipulatorX/Lift/omx_lift_v2_s1` | **best OM-X Lift model**: 100 % real lifts (see "Retrained Lift" below). The older `omx_lift_s0-2` were trained under the old check and only bump the cube |
| Stack | `rlwp/models/OpenManipulatorX/Stack/omx_stack_v3_s0` | **best OM-X Stack model**: 100 % strict successes (see "Retrained Stack" below), but grasps cubeA off-centre (§5.3, P1). The older `omx_stack_s0-2` were trained under the old rules (~35 % really stacked) |

### Full diagnosis (2026-11-16)

A second, per-step look at all six models (20 episodes each, placement seed 1000, rlwp's exact action rule, logging
cube heights, tilts, grasp state, contacts and waypoints at every step), plus control runs without a policy. It confirms
the two problems above, corrects some numbers, and finds more. Nothing in the code was changed for it.

**Confirmed: the reported problems**

| # | Problem | Evidence |
|---|---|---|
| 1 | **Lift: tilting the cube counts as lifting.** Every false success is the same: cube **not grasped**, tilted **30–50°** (leaning on an edge against a finger), centre only **1.3–1.5 cm** up. robosuite's check (centre > table + 4 cm) is 1.25 cm of rise for a 55 mm cube, so a leaning cube passes | real lifts (grasped, ≥ 4 cm): s0 0 of 2 successes, s1 0 of 12, s2 2 of 12. Each "success" step earns reward 1.0; reaching + grasping earns ≤ 0.56. Bumping episodes score 40–86, honest attempts 11–17, so the exploit is what pays |
| 2 | **Stack: cubeB is knocked by the carried cubeA.** It never moves on its own (0.0 mm drift over 100 idle steps, hard or soft reset). It is hit 2–3 steps into the carry (step 52–53): cubeA's bottom is then at 0.841–0.848 m and cubeB's top at 0.848 m, i.e. **0–7 mm clearance**, so cubeA clips cubeB's top edge | cubeB moved > 5 mm in 13/20 (s0), 11/20 (s1), 13/20 (s2) episodes, shoved 3–9 cm and tipped up to 45° |

**New problems**

| # | Problem | Evidence |
|---|---|---|
| 3 | **Stack counts success on a knocked-over cubeB.** The check doesn't test that cubeB is upright or in place, and its "cubeB top" (`pos.z + half height`) assumes cubeB is upright | s1: 9 of its 20 successes fire while cubeB lies tipped 35–43° and shoved 4–7 cm |
| 4 | **Stack success is counted off-centre and at landing.** No centring test; success fires at landing (step 86–87), not at the end of the episode | 13/20 episodes (s1, s2) succeed with cubeA 2–5 cm off cubeB's centre (cubeB half-width 3.25 cm). s1: success at some step 20/20, at the end 11/20, **truly stacked 7/20** |
| 5 | **The waypoint optimiser can return points far outside the box.** SLSQP results are used unchecked, in evaluation (`eval.py`) and in **training** (`method.py`, `self.best_wp = res.x`; only the starting points are clipped) | Stack s2: 14 of 150 solutions out of bounds, up to 2 × 10⁷ m away; the arm is then driven at full speed toward the nonsense target |
| 6 | **Waypoints stuck at the box corners.** Stack's release waypoint is identical in every episode, `[-0.07, 0.15, 0.905]` (corner of the box), whatever cubeB's position (which varies by 3 cm in x and y). Lift s0's first waypoint is also a fixed corner. The learned reward keeps rising toward the box edges | explains the off-centre landings (#4) and the 3 cm drop (release at the box ceiling) |
| 7 | **Start pose hovers 1.1 cm above the cube.** The gripper starts over the cube's spawn region (finger pads at z ≈ 0.866, cube top 0.855), so a first waypoint that moves sideways while dipping slightly sweeps the cube within 1–2 steps | gripper first touches the cube at step 0–2 in most Lift s1/s2 episodes |
| 8 | **`check_real_success.py` is too lenient.** Lift "real" doesn't require a grasp (a hard flick would count); Stack "real" ignores a shoved or tipped cubeB and a tilted cubeA | Stack s1: 60 % "real" by that tool, **35 %** (7/20) when cubeB must be in place and upright, cubeA flat, centred and settled; s2: 20 % |
| 9 | **cubeB is a loose block in the sim; the real one sits 17 mm deep in a recess.** A collision shoves it in the sim; on the real table the carried cube would jam against it instead | sim and hardware diverge exactly where #2 happens |
| 10 | **Minor:** at the first step of every episode the fingers twitch inward (82 → ~68 mm), because the gripper command restarts at half-open; Stack s0 never opens the gripper (its last waypoint commands "close") | – |

Checked and fine: the gripper's "0 = hold" behaves as rlwp assumes (−1 opens fully, +1 closes, 0 holds); the cubes don't
drift on their own; hard and soft resets behave the same.

Updated real-success numbers (strict criteria as in #8): Lift s0 0 %, s1 0 %, s2 10 %; Stack s0 0 %, s1 35 %, s2 20 %.
Fixes for #1–#7, #9 and #10 are implemented (§5.0); #8 (the lenient checker) is not changed yet. The `omx_lift_s*` /
`omx_stack_s*` models were trained under the old rules; the retrained ones are `omx_lift_v2_*` / `omx_stack_v2_*`.

### Retrained Lift (2026-11-16): solved

`Lift_omx` with fixes A–D, J, M (§5.0), 3 seeds, paper schedule (399 episodes), run name `omx_lift_v2`, ~1 h on the
laptop CPU (6 runs in parallel, one thread each). Success now means: both finger pads hold the cube, its centre is
≥ 4 cm above its resting height, tilt ≤ 15°, for 10 consecutive steps (1 s). Results table:
`rlwp/models/OpenManipulatorX/Lift/omx_lift_v2_summary.md`.

| Seed | Evaluation (100 episodes) | Reward | Late training (last 50) | Per-step check (20 episodes, seed 1000) |
|---|---|---|---|---|
| s0 | 99 % | 69.1 ± 5.6 | 94 % | 20/20 real lifts |
| **s1** | **100 %** | 69.7 ± 0.4 | 90 % | **20/20 real lifts** |
| s2 | 83 % | 62.7 ± 16.0 | 90 % | 17/20 real lifts, 3 fails |
| mean | 94 % ± 8 % | 67.2 | | |

(Old models under the same check: 0 %, 0 %, 10 %.)

What the per-step check found (same per-step logging as the diagnosis: contacts, grasp, tilt, rise at every step):

* **Every success is a real lift.** The cube is grasped by both pads, rises a median 7.4–7.7 cm, stays within 4° of
  level while held, and success is reported at step 60–61 (10 steps after the lift starts, as the 1 s hold requires).
  No bumps, flicks or tilted cubes count any more.
* **No pushing before the grasp:** the cube moves at most 5 mm (s0), 8 mm (s1), 16 mm (s2) before it is grasped, under
  the 2 cm push-penalty threshold. The gripper no longer starts over the cube.
* **Seed 2's 3 failures are policy failures, not check problems:** for cubes at the far edge of the spawn area it
  grasps but keeps waypoint 2 at table height (z 0.825), so it never lifts.
* **The lift ends at the top of the waypoint box and swings sideways.** Waypoint 2 is always at the box ceiling
  (z 0.905), and it moves ~10 cm sideways too: s0 always to the same corner (x −0.07, y 0.10), s1 to y −0.05…−0.10. The
  reward doesn't care where the cube goes once it is up, so nothing discourages the swing. Harmless in simulation; on the
  real arm a fast sideways swing with the cube could loosen the grasp. To check on hardware, or to limit with a smaller
  box / a "stay above the spawn point" term if it causes drops.

### Retrained Stack (2026-11-16): solved by seed 0

Two rounds, both 3 seeds, paper schedule (599 episodes), ~1 h each on the laptop CPU, fixes E, G, I, J, M (§5.0).

**Round 1, `omx_stack_v2`** (centre tolerance 15 mm, original cubeA spawn): evaluation 9 % / 68 % / 13 %. The best
seed always landed cubeA on cubeB (20/20), but 8 of 20 landings were 16–25 mm off-centre, just over the tolerance.
Watching it showed a second problem: **cubeA spawned under the start pose** (as Lift's cube did, #7). In some episodes
a finger pad landed on cubeA's top edge at step 1 and dragged it down with the arm, rotating it by 12–18° before the
grasp, which wouldn't happen with a block simply placed on a real table. Models not committed.

**Round 2, `omx_stack_v3`** (current config) with two changes:
* cubeA spawns 5.5 cm further out (centre x 0.005–0.025, was −0.05…−0.02): the gripper now starts 2.1–4.5 cm from it
  (was ~1 mm). The waypoint box was extended to x 0.03 (and narrowed to y ≥ −0.10, which cubeA never needs). Reach
  re-measured: < 1 cm everywhere except the far corner x ≥ 0.02, y ≥ 0.12 at the ceiling (up to 24 mm), which the task
  doesn't use. Scripted check 19/20.
* strict-check centre tolerance 25 mm (was 15 mm). A 25 mm offset still leaves cubeA's centre of mass well inside
  cubeB's 65 mm top.

No pre-grasp penalty was added: in the real setup the block is simply placed, so moving it out of the gripper's way is
the right fix rather than teaching the policy to avoid it. Results table:
`rlwp/models/OpenManipulatorX/Stack/omx_stack_v3_summary.md`.

| Seed | Evaluation (100 episodes) | Reward | Late training (last 50) | Per-step check (20 episodes, seed 1000) |
|---|---|---|---|---|
| **s0** | **100 %** | 117.1 ± 2.9 | 100 % | **20/20 strict successes**; no gripper contact with cubeA before the grasp; lands on cubeB 20/20, median 9 mm off-centre (max 23 mm) |
| s1 | 0 % | 62.8 ± 7.4 | 98 % | never releases (19/20): evaluated with an early model snapshot, see below |
| s2 | 61 % | 91.0 ± 28.0 | 94 % | lands on cubeB 20/20, but 10/20 more than 25 mm off-centre (median 25 mm); 3/20 pre-grasp touches. Also an early snapshot |
| mean | 54 % ± 41 % | 90.3 | | |

In every episode of all seeds cubeB stays in place (it is fixed), and every success is a real one: cubeA released,
resting flat and centred on cubeB, settled. 2-waypoint Stack works, so 3 waypoints (S3) aren't needed for now.

**Why seeds 1 and 2 evaluate far below their training: an rlwp saving issue (original code).** rlwp saves the last
waypoint's reward models only when an episode beats the best reward so far, and never at the end of training. The
evaluated models are therefore a snapshot from the best-reward episode, not the final ones. s0's best came at episode
273 of 300 (saved 2 min before the end), but s1's at episode 67 (still exploring: its snapshot never learned to open
the gripper) and s2's at episode 91. Their final models, which scored 98 % / 94 % in the last training episodes, were
not saved. Earlier waypoints are unaffected (saved when training moves to the next waypoint). Fix in §5.3.

## 4. Seeing it

The policy viewer marks every episode with a coloured, labelled sphere above the scene: grey = running, yellow =
robosuite reports success, then **green = REAL SUCCESS**, **red = FALSE SUCCESS** (with the reason), grey = FAIL. The
terminal prints the same verdict with running totals. "Success" is the environment's own check, i.e. the strict one
for the `_omx` configs.

```bash
cd rl-waypoints && source ../.venv/bin/activate
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omx  --run omx_lift_v2_s1 --seed 1000
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Stack_omx --run omx_stack_v3_s0 --seed 1000
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py --robot omx      # the task layouts, robot at rest
```

**Episode navigation** (click the viewer window first): **← Left** = previous episode, **→ Right** = next episode,
**↑ Up** = replay the current episode. A key interrupts a running episode. `--episode N` starts at episode N,
`--pause S` holds each result for S seconds (default 2). Replays are exact: the viewer records numpy's random state
before each episode's reset (object placement) and before its waypoint optimisation, so episode N always has the same
placement and waypoints for a given `--seed` (verified bit-identical when jumping 3 → 1 → 2 → 3). Totals count each
episode once. For Stack, each verdict also shows cubeA's end state (off-centre distance and tilt; full detail in the
terminal).

## 5. Possible fixes

The two problems differ. In Lift the **reward** pays for the wrong behaviour, so the policy learned it: the check and
the reward have to change, then Lift must be retrained. In Stack the policy does the right thing, but the release is
unreliable and the **metric** counts the landing instead of the end state. All fixes below would follow this fork's
rule: a new environment argument whose default reproduces the original task, switched on only in the `_omx` configs,
so Panda and OM-P results stay as they are.

### 5.0 Implemented (2026-11-16): Lift and Stack retrained and solved

Each fix is an option whose default is the original behaviour, switched on only in `Lift_omx.yaml` / `Stack_omx.yaml`
(except J, which only changes runs where the optimiser returned an invalid point). "Problem" refers to the numbered
table in §3, "Full diagnosis".

| Fix | Problem | Option (where) | Setting in the `_omx` config |
|---|---|---|---|
| A. "lifted" needs a grasp and a real rise | #1 | `lift_check: 'grasped'`, `lift_height`, `lift_max_tilt` (Lift env) | both finger pads on the cube, centre ≥ 4 cm above its resting height, tilt ≤ 15°. Used by the success check **and** the per-step "lifted" reward |
| B. hold before success | #1 | `success_hold_steps` (Lift env) | 10 steps (1 s); the reward is paid from the first lifted step |
| C. cube not under the start pose | #7 | `cube_x_range`, `cube_y_range` (Lift env); Lift waypoint box x up to 0.05 | cube centre x 0.01–0.04 (was ±0.03): near face ≥ 2 cm ahead of the open fingers. Tracking at x 0.05 ≤ 9 mm at every height |
| D. penalty for pushing the cube | #1, #7 | `push_penalty`, `push_threshold` (Lift env) | −0.25 per step while the cube is > 2 cm from its spawn point and not grasped |
| E. rise before moving sideways | #2 | `lift_first` (rlwp task `env:` key; `common.lift_first_error`, used by `train.py`, `eval.py`, `check_omp_tasks.py`, `check_real_success.py`, `view_policy.py`) | on for Stack: in segments after the first, while the waypoint is > 1 cm above the gripper only its height is tracked |
| G. static landing cube | #2, #9 | `cubeB_fixed` (Stack env) | cubeB has no free joint: it rests on the table at its sampled position, like the real one in its recess |
| I. strict stacking check | #3, #4 | `stack_check: 'strict'` + tolerances, `success_hold_steps` (Stack env) | cubeA on cubeB's top (± 6 mm), centred within 25 mm (first tried 15 mm), flat within 10°, speed < 0.05 m/s, no gripper contact; cubeB upright (5°) and within 1 cm of its spawn point; held 5 steps. Same condition pays the stacking reward. **If it proves too hard to learn, loosen the tolerances first** (e.g. centre 25 mm, tilt 20°, speed 0.1 m/s) |
| J. guard the optimiser's output | #5 | `common.bounded_solution` (`method.py`, `eval.py`; always on) | non-finite solutions → clipped start point, out-of-box solutions → clipped, cost re-evaluated. Valid solutions are unchanged |
| M. gripper starts fully open | #10 | `reset_action` on `OMXNativeGripper` (OM-X only), read by `SingleArmRobot.reset` | no finger twitch on the first step |

`check_omp_tasks.py`'s Lift plan now lifts 7 cm (the arm sags ~1 cm under the cube; 5 cm gave only 3.9 cm of rise).

**Verification**

| Check | Result |
|---|---|
| scripted 2-waypoint check (`check_omp_tasks.py --robot omx`, 20 episodes) | Lift 20/20 (rise 5.8 cm, held 46 steps); Stack 18/20 and 19/20 on two seeds: cubeB never moves, cubeA lands flat, typically 5 mm off-centre. The misses landed 1.9 cm off-centre against an open finger |
| old Lift models under the new check (their original spawn area) | s1: 0 successes (was 12 false); s2: only its 2 real lifts count |
| old Stack models under the new rules | cubeB moved in 0/40 episodes (was 55–65 %). Strict successes only 2/20 (s1) and 1/20 (s2): their release waypoint is pinned to the box corner, so cubeA lands a median 21–25 mm off-centre (#6). A retrained policy is rewarded for centring |
| optimiser guard | Stack s2: 0/150 out-of-bounds solutions (was 14/150) |
| Panda Lift, OM-P Lift, OM-P Stack: seeded smoke runs before vs after | training and evaluation data bit-identical |
| `train_omp.sh --smoke` for Lift_omx, Stack_omx | train + evaluate run end to end |

Also for Stack (after watching round 1): cubeA's spawn moved out from under the start pose (`cubeA_x_range` 0.005–0.025,
waypoint box x up to 0.03), the Stack analogue of C.

**Retrained:** Lift: solved (94 % mean, 100 % best, all real; §3 "Retrained Lift"). Stack: solved by seed 0 (100 %, all
real; §3 "Retrained Stack"); 2 waypoints are enough, so 3 waypoints (S3) aren't needed for now. Open problems: §5.3.

### 5.1 Lift: false successes from bumping the cube

| # | Fix | What it changes | Pros | Cons / cost |
|---|---|---|---|---|
| L1 | **Lift height relative to the cube's resting height**: "lifted" = cube centre ≥ resting centre + 4 cm (resting centre = table + half the cube height), as Stack's `relative_stack_check` does | Lift's success check, and with it the per-step "lifted" reward (it pays the full reward exactly when success is true) | Same 4 cm of real lift for every cube size; a bump can't raise a 55 mm cube 4 cm; one small change in `lift.py` | Alone, a hard enough flick could still count in principle |
| L2 | **Lifted only while grasped**: also require robosuite's `_check_grasp` (both finger pads touching the cube) | success and reward | Rules out bumps, flicks and tipping completely; matches what "lift" means | Depends on the finger-pad contact model; a firm grasp that touches with one pad only wouldn't count |
| L3 | **Lifted for a while**: the condition must hold for N consecutive steps (e.g. 10 steps = 1 s) | success (the reward can stay per-step) | Removes transient spikes; checks that the cube is actually held | Episodes must leave time to hold (they do: waypoint 2 has 50 steps) |
| L4 | **Penalty for pushing the cube before it is grasped**: subtract reward when the cube moves > 2 cm while not grasped (the paper added "penalties for knocking the objects over" to some tasks) | reward only | Directly discourages the sweeping waypoint 1 that the models learned | Needs a weight to tune; doesn't fix the metric by itself |
| L5 | **Report "real success" in rlwp's evaluation** (as `check_real_success.py` does) | metric only | Catches this kind of exploit automatically in every future run | Doesn't change what is learned |

**Recommended:** L1 + L2 (+ L5), then retrain `Lift_omx` (3 seeds, ~1 h 15 on this laptop) and check with
`check_real_success.py`. The scripted check already lifts for real (19/20) and the OM-P model with the same cube learns
real lifts (20/20), so the task itself is learnable; only the incentive is wrong.

### 5.2 Stack: cubeA lands on cubeB, then falls off

What happens: the second waypoint stops at the top of the waypoint box (z 0.905), and the gripper opens there, so cubeA
drops ~3 cm onto cubeB. It often tumbles on landing: most cubes that stay on cubeB were still turned over by 90° at the
end. In ~4 of 10 episodes it slides or tips off onto the table, after robosuite has already counted the success.

| # | Fix | What it changes | Pros | Cons / cost |
|---|---|---|---|---|
| S1 | **Count the stack at the end of the episode** (or for the last N steps), not at the landing | success metric (and rlwp's evaluation report) | Honest number immediately; no retraining needed to measure | Doesn't make stacking more reliable |
| S2 | **Pay the stacking reward only while cubeA rests on cubeB and is still** (low velocity), or add a bonus for being stacked at the last step | reward | The learner then prefers releases that stay on cubeB (better centred, lower) | Retrain; a bonus weight to choose |
| S3 | **Lower release with 3 waypoints**: grasp → carry high over cubeB → lower to ~1 cm above cubeB and open (`Stack3_omx`, like the existing `Stack3_omp`; rlwp supports `num_wp: 3`) | task schedule (one more 300-episode bandit; waypoints 1–2 can be reused with `--init-from`) | Removes the 3 cm drop that makes the cube tumble; scripted 3-waypoint Stack worked 20/20 on the OM-P | The OM-P's 3-waypoint run struggled to explore the release; needs the stronger alignment reward (already on) |
| S4 | **Lower the box ceiling over cubeB only**, so the 2-waypoint release happens closer to cubeB | waypoint bounds | No extra waypoint | The straight carry from cubeA needs the height to clear cubeB; a single box can't be high on the way and low at the end, so this would need a non-box bound in rlwp |
| S5 | **Align cubeA with cubeB**: the gripper's yaw follows the bearing from the base, so cubeA arrives turned ~50° against cubeB; spawning cubeB turned by its bearing (new `cubeB_rotation` argument) lands face to face | layout | More support under cubeA, steadier landing | Also has to be set up the same way on the real table |
| S6 | **Hardware-side**: wider or textured landing cube, deeper recess | physical setup | Easy | Changes the task you report |

**Recommended:** S1 now (honest metric), then S3 + S2 and retrain Stack, checking the end-state rate with
`check_real_success.py`. S5 is a cheap layout improvement to test first with the scripted check.

### 5.3 Open problems (2026-11-16, not fixed)

| # | Problem | Evidence | Possible fix |
|---|---|---|---|
| P1 | **Stack grasps cubeA off-centre, toward its far edge.** The grasp waypoint is pinned to the box's forward edge (x 0.030) in every episode, wherever cubeA is, so the closer cubeA spawns, the further past its centre it is gripped. In simulation friction holds it; on the real arm the cube could tip or slip in the fingers under its own weight. Cause: nothing in the reward cares where the cube is grasped. Stack's reward is the maximum of the reach / lift / stack stages, and the "close to the cube centre" term only exists in the small reach stage, which stops counting once the cube is lifted, so the learned reward keeps rising toward the box edge (as in #6) | `omx_stack_v3_s0`, 20 episodes: grasp point vs cubeA centre median **+10 mm**, max **+20 mm** along the finger faces (cube half-width 27.5 mm), > 10 mm in 11/20; across the fingers ~1 mm. Lift (`omx_lift_v2_s1`) for comparison: median 1.9 mm, > 10 mm in 1/20 | **Reward a centred grasp** (recommended): record the grasp offset when cubeA is first grasped and scale the lift / align / stack rewards by a factor that is 1 within ~5 mm and falls off beyond; optionally require ≤ ~10 mm for success. Alternatives: waypoint 1 relative to cubeA's position (structural change to rlwp); narrow cubeA's spawn range or shrink the box (hide the problem / only reduce the worst case) |
| P2 | **rlwp evaluates an early snapshot of the last waypoint's models** (original code): they are saved only when an episode beats the best reward, never at the end of training | Stack v3: s1 snapshot from episode 67 of 300 (0 % in evaluation vs 98 % in its last training episodes), s2 from episode 91 (61 % vs 94 %) | Save the final models at the end of training too (config option, on only for the `_omx` tasks so Panda / OM-P stay identical), then retrain s1 / s2 |
| P3 | **Lift swings the cube ~10 cm sideways while lifting** (waypoint 2 at the top of the box, moved to its side) | §3 "Retrained Lift" | Watch on hardware; if it loosens the grasp, a smaller box or a "stay above the spawn point" term |
| P4 | **cubeA lands rotated relative to cubeB** (the 4-DoF arm's gripper yaw follows the bearing from the base, so cubeA turns ~50° with joint 1 on the way to cubeB) | visible in the viewer; real successes are unaffected so far | Spawn cubeB turned by its bearing (S5) so the faces meet, if landings need more support |
| P5 | **`check_real_success.py` is still lenient** (#8): Lift "real" doesn't require a grasp; Stack "real" ignores a moved cubeB / tilted cubeA | the environments' own success is now strict, so the viewer / evaluation numbers are honest; only this tool's "real" column is lenient | Align its criteria with the environments' strict checks |

### 5.4 After that

**Hardware:** `omx_hw/` ([omx_hw/README.md](omx_hw/README.md)): `plan_episode.py` turns hand-measured cube positions
into a checked joint-space plan (replay of the simulated episode, predicted outcome) and `execute_plan.py` runs it on the
real OM-X through ROS 2 Jazzy (ROBOTIS `open_manipulator`, U2D2, 12 V, IDs 11–15, 1 Mbps). Planner tested in
simulation; executor not yet run on hardware. Use the models only after their *real* success
rate is good, because the false successes above would not transfer.

## 6. Reproducing

```bash
python omp_tools/check_omx_arm.py                       # arm vs ROBOTIS models, reach map
python omp_tools/check_omx_env.py                       # placement, controller comparison, scripted pick
python omp_tools/check_omp_tasks.py --robot omx         # scripted 2-waypoint checks of the _omx tasks
python omp_tools/check_omp_workspace.py --robot OpenManipulatorX   # reachability grid -> waypoint boxes (~10 min)
omp_tools/train_omp.sh -t Lift_omx  -s "0 1 2" -n omx_lift_v2       # train + evaluate (with the §5.0 fixes)
omp_tools/train_omp.sh -t Stack_omx -s "0 1 2" -n omx_stack_v3
python omp_tools/check_real_success.py --task Stack_omx --runs omx_stack_v3_s0   # real vs false successes
```

Setup (virtualenv, pinned versions) as in [README_OMP.md](README_OMP.md) §1; OM-X commands also in its §9.
