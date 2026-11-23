# Running the trained models on the real OpenMANIPULATOR-X

Two scripts take a trained Lift / Stack model to the real OM-X (U2D2, ROS 2 Jazzy, ROBOTIS' `open_manipulator` stack):

| Script | Runs where | Needs | Does |
|---|---|---|---|
| `plan_episode.py` | this repo's virtualenv | nothing on the robot side | from hand-measured cube positions: asks the trained model for its waypoints, **simulates the episode with exactly those positions** (predicted outcome), turns it into a joint-space plan for the robot, checks it, writes a plan file (JSON); `--view` shows it |
| `scan_bus.py` | anywhere with `dynamixel_sdk`, bringup NOT running | the U2D2 | read-only check of the 5 motors: IDs, voltage, temperature, errors, joint angles; nothing moves |
| `run_trials.py` | this repo's virtualenv, ROS 2 sourced | both of the above | repeated trials at random positions: plan + MuJoCo viewer (with the measuring axes), you place the cubes, the robot runs, you log the outcome |
| `sim_sweep.py` | this repo's virtualenv | nothing | success map of a trained model on a grid of positions (simulation), to see where it works beyond its trained area |
| `execute_plan.py` | where ROS 2 Jazzy is sourced (natively or in ROBOTIS' Docker container) | `rclpy`, `control_msgs`, `trajectory_msgs`, `sensor_msgs` | plays the plan on the robot, step by step, asking before each step; `--gripper-test` checks the gripper |

**Status (2026-11-17): Lift runs on the real arm** (bus check, start pose, gripper test, and the in-the-air Lift plan,
which already lifted the cube by its top). **Stack has not run on hardware yet.** Everything, including Stack and
`run_trials.py`, is also tested against ROBOTIS' bringup with mock hardware (`use_mock_hardware:=true`: moves, gripper,
Ctrl+C cancel). The planner needs
the project virtualenv `../.venv` (CPU torch, MuJoCo 3.1.6, gym, hydra-core, robosuite editable). Its ROS interfaces follow
ROBOTIS' Jazzy configuration (`open_manipulator_bringup/config/open_manipulator_x/`): `arm_controller`
(JointTrajectoryController, `/arm_controller/follow_joint_trajectory`, joints `joint1`..`joint4`) and
`gripper_controller` (GripperActionController on `gripper_left_joint`, −0.011 … 0.020 m). Two things those files
don't state, so the first session must check them: the gripper action's name (the executor looks for the
`GripperCommand` action on the system, falling back to `/gripper_controller/gripper_cmd`) and which gripper direction
is open (`--gripper-test`).

## How the plan is made

The policy outputs two waypoints (grip-point position + open/close) from the cube positions. In simulation the arm is
then driven to them by rlwp's controller. On the robot, the planner **replays the simulated episode's joint angles**,
4× slower: the simulator's arm model is ROBOTIS' own MJCF (same joint names, zeros and directions as their ROS 2 URDF;
forward kinematics verified identical), so the angles carry over directly. This is the motion that was just verified to
succeed, it is reachable by construction and it includes rlwp's way of moving (e.g. Stack's rise-first carry). Two
adjustments by inverse kinematics (gripper pointing down):

* points where the fingers would come within 3 mm of the table are raised (in simulation the table stops the arm; a
  position-controlled real arm would press into it). Typically ≤ 3.5 mm, during the Lift grasp;
* optional `--z-offset-cm` (dry runs above the cubes) and `--plate-cm` (a different base height).

After the policy's steps, an optional **retreat** retraces the motion backwards with the gripper open (Lift first puts
the cube back down where it was), ending in the start pose.

## Setup

1. **Hardware** (ROBOTIS e-Manual): OM-X, U2D2 + U2D2 power hub board, 12 V supply, Dynamixel IDs 11–15, 1 Mbps.
2. **Base height**: the robot base stands on a **1.7 cm plate** on the table top (as in the simulation). If yours
   differs, pass `--plate-cm <height>` to the planner.
3. **Objects**: Lift: a 55 mm cube. Stack: a 55 mm cube to pick, a 65 mm landing cube set 17 mm deep into a recess
   (48 mm above the table, held fixed).
4. **ROS 2 Jazzy + ROBOTIS `open_manipulator`** (branch `jazzy`; ROBOTIS recommends their Docker setup). Check the arm
   with ROBOTIS' own tools first.
5. Copy `omx_hw/execute_plan.py` and the plan files to the ROS machine / container (only these two are needed there).

## Measuring the cube positions

All positions are **centimetres in the robot base frame**:

* **origin**: joint 1's rotation axis (the centre of the base motor's shaft), on the table;
* **x**: straight out from the arm when joint 1 = 0 (the direction the arm points in ROBOTIS' home pose);
* **y**: to the robot's left.

Measure the **centre** of each cube, and place the cubes **square to the x axis** (faces parallel / perpendicular to
it): in training the cubes were never rotated. Stay inside the areas the models were trained on (the planner refuses
other positions unless `--allow-outside`):

| Task | Cube | x (cm) | y (cm) |
|---|---|---|---|
| Lift | cube | 16.8 … 19.8 | −3.0 … 3.0 |
| Stack | cubeA (picked) | 16.3 … 18.3 | −3.0 … 0.0 |
| Stack | cubeB (landing) | **0 … 15** (x ≥ 0) | **−20 … +20**, and 17–20 cm from joint 1's axis, not within 40° of straight ahead: see below |

**Stack landing area** (`Stack_omx.yaml`, since 2026-11-22): 17–20 cm from joint 1's axis, in front of the axis
(x ≥ 0), y between −20 and +20 cm, but not within 40° of straight ahead (where cubeA is), so 40°–90° to the left or to
the right (|y| ≥ ~11 cm), and at least 9 cm (centre to centre) from cubeA. Examples: x 12.0, y 13.0 (47°, left);
x 2.0, y −18.0 (−84°, right); x 6.0, y 17.0 (71°, left). Trained model: `omx_stack_v4_s1` (the planner's default;
94 % in simulation). The planner optimises Stack's waypoints with 20 random restarts (`--n-inits`): with fewer, the
same placement can come out with a closed instead of an open release. The planner prints
each cube's distance and angle and refuses positions outside the area. The previous model `omx_stack_v3_s0` (old
landing area x 8.8 … 11.8, y 12.0 … 15.0) is still usable with `--config Stack_omx_v3 --run omx_stack_v3_s0`.

## Running

```bash
# 1) plan (in rl-waypoints, virtualenv). Check the printout: inside the trained area, predicted outcome, warnings.
../.venv/bin/python omx_hw/plan_episode.py --task Lift --cube 18.5 -1.0 --out plan_lift.json --view
../.venv/bin/python omx_hw/plan_episode.py --task Stack --cubeA 17.5 -1.5 --cubeB 12.0 13.0 --out plan_stack.json --view

# 0) bringup NOT running: read-only bus check (IDs 11-15, voltage, errors; torque stays off, nothing moves)
python3 omx_hw/scan_bus.py

# 2) ROS 2 terminal 1: ROBOTIS bringup with the U2D2. init_position:=false: torque on, the arm holds where it is
#    (without it, ROBOTIS' node moves the arm to their home [0, -1, 1, 0] within 3 s, unprompted)
ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0 init_position:=false
#    check: ros2 action list   -> /arm_controller/follow_joint_trajectory and /gripper_controller/gripper_cmd

# 3) ROS 2 terminal 2, first session only: move to the simulation start pose (asks first), then does "open" open
#    and "close" close?  (--home alone: only the move to the simulation start pose)
python3 execute_plan.py --gripper-test --time-scale 2

# 4) dry run in the air, no cubes on the table: everything 3 cm higher
../.venv/bin/python omx_hw/plan_episode.py --task Lift --cube 18.5 -1.0 --z-offset-cm 3 --out plan_lift_dry.json
python3 execute_plan.py plan_lift_dry.json

# 5) the real run, cubes at the measured positions
python3 execute_plan.py plan_lift.json --dry-run    # print the steps
python3 execute_plan.py plan_lift.json              # Enter = run step, s = skip, q = quit; Ctrl+C cancels a motion
```

### Repeated trials at random positions (`run_trials.py`)

Runs the whole loop for N random cube positions inside the trained Lift area (0.5 cm grid, 3 mm inside the edges):
plan with the learned waypoints → MuJoCo viewer for that position (planned motion + simulated episode, on repeat) →
you place the cube → the robot picks, lifts, puts it back → you say whether it lifted. Bringup running in terminal 1;
in terminal 2 (ROS 2 sourced):

```bash
../.venv/bin/python omx_hw/run_trials.py              # 10 trials; -n 5, --seed 3, --confirm-steps, --no-viewer, --time-scale
```

Stack: `--task Stack --cubeB X Y` (landing block fixed at its measured position, cubeA random each trial). Dry run in
the air first, with only the block on the table: `--task Stack --cubeB X Y --z-offset-cm 3 -n 1`.

Outside the trained areas: `--allow-outside` (each plan is still simulated first; a failing or outside position asks
before running), `--expand-cm N` to grow the random areas, `--cube` / `--cubeA` / `--cubeB X Y` for fixed positions.
Simulated success beyond the trained areas (2026-11-17, 1 cm grid, one simulation per point, **Stack rows: the previous
model `omx_stack_v3_s0`**; the waypoint boxes are hard limits: Lift x 8.8–20.8 / y ±10; Stack then x 8.8–18.8 /
y −10…15, now x −1.2–18.8 / y ±21 cut to 10–21 cm from joint 1):

| Object | Trained area (x, y) | Succeeded at every grid point in | Never works |
|---|---|---|---|
| Lift cube | 16.8–19.8, −3…3 | x 17–20, y −4…+6 | x ≥ 21 with y ≤ −2 / y −8 mostly |
| Stack cubeA (block at 10.3, 13.5) | 16.3–18.3, −3…0 | x 16–18, y −6…+3 | x ≥ 20 (beyond the grasp box) |
| Stack landing block (cubeA at 17.3, −1.5) | 8.8–11.8, 12…15 | x 7–13, y 12…15 | y ≤ 11, y ≥ 17, x ≥ 15 |

Reproduce / re-check with `../.venv/bin/python omx_hw/sim_sweep.py --task Stack --object cubeB` (see its `--help`).
The landing block's narrow y band was a trained-in limit of `omx_stack_v3_s0` (the release sat at the box edge
y = 15 cm, `OMX_PORT.md` P6). Since 2026-11-22 Stack is retrained for the larger landing area (17–20 cm from joint 1,
x ≥ 0, y −20…+20 cm; `omx_stack_v4_s1`, `OMX_PORT.md` §3 "Larger landing area"); a sweep map of v4 is still to do
(`sim_sweep.py`, default grid now covers the new area).

Ctrl+C stops the robot's motion (the arm holds) and the script asks how to go on. Results per session in
`omx_hw/trials/<date-time>/`: `results.csv` (position, simulated prediction, robot run, real outcome, note) plus every
plan and log.

Executor options: `--yes` (no confirmations), `--no-retreat`, `--time-scale 2` (twice as slow again),
`--max-effort`, `--arm-action`, `--gripper-action`.

## Safety

* Keep a hand on the 12 V power switch during the first runs. **Ctrl+C in the executor's terminal** cancels the running
  motion and the arm holds its pose (torque stays on; the executor waits for the controller's confirmation).
  **Stopping the bringup (Ctrl+C) or switching off the power turns the torque off: the arm drops.** Support the arm
  before you do either.
* Do the gripper test and the in-the-air dry run (step 4) before putting cubes down.
* The first move goes from wherever the arm is to the start pose, slowly in joint space; later moves ease in if the arm
  isn't where the plan expects. Nothing checks for obstacles other than the table, and the cubes only via the simulated
  episode: keep the workspace clear.
* Re-measure and re-plan for every new cube placement: the plan is only valid for the positions it was made with.

## Expected differences from the simulation

* **Stack retreat**: after the release the open fingers straddle the stacked cube, so retracing the carry (the first
  version) would have swept them 22 mm into it. The planner now pulls the gripper straight back toward the base
  (6 cm, rising 1.5 cm; up to 12 cm / 5 cm if needed), then returns to the start pose, and checks the whole retreat
  against the simulated end state (gripper > 5 mm from both cubes). If cubeA lands elsewhere on the real block, watch
  the retreat (or use `--no-retreat` / `--confirm-steps`).
* **Stack grasps cubeA ~1 cm off-centre**, toward its far edge (open problem P1 in `OMX_PORT.md` §5.3): the real cube
  may tip in the fingers. Consider fixing P1 before relying on Stack.
* **Lift swings the cube ~10 cm sideways while lifting** (P3): watch the grasp; slow down with `--time-scale` if needed.
* **Grip force** comes from the gripper motor's current limit in ROBOTIS' config (current-based position mode, Goal
  Current 200); the simulated gripper may grip differently.
* The simulated arm settles slightly under the cube's weight; the plan replays those simulated angles, so on a stiffer
  real arm positions can differ by a few millimetres.
* Measurement errors move the grasp directly: aim for ±3 mm on the cube positions.
