# rl-waypoints on the ROBOTIS OpenMANIPULATOR-X

This branch (`omx-port`) runs the waypoint-based RL method of Mehta, Habibian & Losey (2024) on a **ROBOTIS
OpenMANIPULATOR-X** (RM-X52-TNM, standard kit: 4 × XM430-W350 arm joints + an XM430 gripper) in the robosuite / MuJoCo
simulation that the paper used. It builds on the OpenMANIPULATOR-P port of this fork ([README_OMP.md](README_OMP.md)),
whose learning code, configuration keys and tools it reuses. The original Panda workflow is unchanged.

Status (2026-11-16): robot model, controller and task layouts are done and verified. **Stack is learned (best model
stacks 100 % by robosuite's measure, ~60 % stay stacked). Lift is not: the trained models exploit robosuite's lift check
instead of lifting (details in §3, possible fixes in §5). It needs a check fix and retraining.** Nothing has run on
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
| Lift_omx | 55 mm cube, 14–20 cm in front of the base (robosuite's ±3 cm spawn) | x [−0.07, 0.03], y ±0.10, z [0.825, 0.905] | 19/20 |
| Stack_omx | cubeA 55 mm straight ahead, 12–15 cm out; cubeB 65 mm set 17 mm into a recess (48 mm above the table), 16–20 cm out, ~50° to the left | x [−0.07, −0.02], y ±0.15, z [0.825, 0.905] | 20/20 |

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
| Stack | `rlwp/models/OpenManipulatorX/Stack/omx_stack_s1` | best OM-X model: grasps, carries over cubeB, releases; ~60 % stay stacked |
| Lift | none yet | retrain after the lift-check fix (§5) |

## 4. Seeing it

The policy viewer marks every episode with a coloured, labelled sphere above the scene: grey = running, yellow =
robosuite reports success, then **green = REAL SUCCESS**, **red = FALSE SUCCESS** (with the reason), grey = FAIL. The
terminal prints the same verdict with running totals.

```bash
cd rl-waypoints && source ../.venv/bin/activate
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omx  --run omx_lift_s2  --seed 1000
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Stack_omx --run omx_stack_s1 --seed 1000
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py --robot omx      # the task layouts, robot at rest
```

## 5. Possible fixes

The two problems differ. In Lift the **reward** pays for the wrong behaviour, so the policy learned it: the check and
the reward have to change, then Lift must be retrained. In Stack the policy does the right thing, but the release is
unreliable and the **metric** counts the landing instead of the end state. All fixes below would follow this fork's
rule: a new environment argument whose default reproduces the original task, switched on only in the `_omx` configs,
so Panda and OM-P results stay as they are.

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

### 5.3 After that

**Hardware:** a runner that executes the learned waypoints on the real OM-X through the U2D2, using the simulator's
kinematics for IK and the ROBOTIS Jazzy setup (12 V, IDs 11–15, 1 Mbps). Use the models only after their *real* success
rate is good, because the false successes above would not transfer.

## 6. Reproducing

```bash
python omp_tools/check_omx_arm.py                       # arm vs ROBOTIS models, reach map
python omp_tools/check_omx_env.py                       # placement, controller comparison, scripted pick
python omp_tools/check_omp_tasks.py --robot omx         # scripted 2-waypoint checks of the _omx tasks
python omp_tools/check_omp_workspace.py --robot OpenManipulatorX   # reachability grid -> waypoint boxes (~10 min)
omp_tools/train_omp.sh -t Lift_omx  -s "0 1 2" -n omx_lift          # train + evaluate
omp_tools/train_omp.sh -t Stack_omx -s "0 1 2" -n omx_stack
python omp_tools/check_real_success.py --task Stack_omx --runs omx_stack_s1   # real vs false successes
```

Setup (virtualenv, pinned versions) as in [README_OMP.md](README_OMP.md) §1; OM-X commands also in its §9.
