# Running the trained models on the real OpenMANIPULATOR-X

Two scripts take a trained Lift / Stack model to the real OM-X (U2D2, ROS 2 Jazzy, ROBOTIS' `open_manipulator` stack):

| Script | Runs where | Needs | Does |
|---|---|---|---|
| `plan_episode.py` | this repo's virtualenv | nothing on the robot side | from hand-measured cube positions: asks the trained model for its waypoints, **simulates the episode with exactly those positions** (predicted outcome), turns it into a joint-space plan for the robot, checks it, writes a plan file (JSON); `--view` shows it |
| `execute_plan.py` | where ROS 2 Jazzy is sourced (natively or in ROBOTIS' Docker container) | `rclpy`, `control_msgs`, `trajectory_msgs`, `sensor_msgs` | plays the plan on the robot, step by step, asking before each step; `--gripper-test` checks the gripper |

**Status: the planner is tested in simulation; the executor has not run on hardware yet.** Its ROS interfaces follow
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
| Stack | cubeB (landing) | 8.8 … 11.8 | 12.0 … 15.0 |

## Running

```bash
# 1) plan (in rl-waypoints, virtualenv). Check the printout: inside the trained area, predicted outcome, warnings.
.venv/bin/python omx_hw/plan_episode.py --task Lift --cube 18.5 -1.0 --out plan_lift.json --view
.venv/bin/python omx_hw/plan_episode.py --task Stack --cubeA 17.5 -1.5 --cubeB 10.5 13.5 --out plan_stack.json --view

# 2) ROS 2 terminal 1: ROBOTIS bringup with the U2D2 (the arm moves to its home pose)
ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0
#    check: ros2 action list   -> /arm_controller/follow_joint_trajectory and a gripper action

# 3) ROS 2 terminal 2, first session only: does "open" open and "close" close?
python3 execute_plan.py --gripper-test

# 4) dry run in the air, no cubes on the table: everything 3 cm higher
.venv/bin/python omx_hw/plan_episode.py --task Lift --cube 18.5 -1.0 --z-offset-cm 3 --out plan_lift_dry.json
python3 execute_plan.py plan_lift_dry.json

# 5) the real run, cubes at the measured positions
python3 execute_plan.py plan_lift.json --dry-run    # print the steps
python3 execute_plan.py plan_lift.json              # Enter = run step, s = skip, q = quit; Ctrl+C cancels a motion
```

Executor options: `--yes` (no confirmations), `--no-retreat`, `--time-scale 2` (twice as slow again),
`--max-effort`, `--arm-action`, `--gripper-action`.

## Safety

* Keep a hand on the 12 V power switch during the first runs; Ctrl+C cancels the running motion.
* Do the gripper test and the in-the-air dry run (step 4) before putting cubes down.
* The first move goes from wherever the arm is to the start pose, slowly in joint space; later moves ease in if the arm
  isn't where the plan expects. Nothing checks for obstacles other than the table, and the cubes only via the simulated
  episode: keep the workspace clear.
* Re-measure and re-plan for every new cube placement: the plan is only valid for the positions it was made with.

## Expected differences from the simulation

* **Stack grasps cubeA ~1 cm off-centre**, toward its far edge (open problem P1 in `OMX_PORT.md` §5.3): the real cube
  may tip in the fingers. Consider fixing P1 before relying on Stack.
* **Lift swings the cube ~10 cm sideways while lifting** (P3): watch the grasp; slow down with `--time-scale` if needed.
* **Grip force** comes from the gripper motor's current limit in ROBOTIS' config (current-based position mode, Goal
  Current 200); the simulated gripper may grip differently.
* The simulated arm settles slightly under the cube's weight; the plan replays those simulated angles, so on a stiffer
  real arm positions can differ by a few millimetres.
* Measurement errors move the grasp directly: aim for ±3 mm on the cube positions.
