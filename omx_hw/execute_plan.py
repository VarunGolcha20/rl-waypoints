#!/usr/bin/env python3
"""
Execute a plan from omx_hw/plan_episode.py on the real OpenMANIPULATOR-X through ROS 2 Jazzy (ROBOTIS'
open_manipulator stack: arm_controller = JointTrajectoryController on joint1..joint4, gripper_controller =
GripperActionController on gripper_left_joint). Needs only standard ROS 2 packages (rclpy, control_msgs,
trajectory_msgs, sensor_msgs); run it where ROS 2 is sourced (natively or inside ROBOTIS' Docker container).

    # terminal 1: ROBOTIS bringup (U2D2), moves the arm to its home pose
    ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0
    # terminal 2:
    python3 omx_hw/execute_plan.py --gripper-test            # first time: check the gripper opens / closes as expected
    python3 omx_hw/execute_plan.py plan_lift.json --dry-run  # print the steps only (no ROS needed)
    python3 omx_hw/execute_plan.py plan_lift.json            # run, confirming every step (Enter / s = skip / q = quit)

Safety: every step waits for Enter unless --yes; Ctrl+C cancels the running motion. The first move goes from wherever
the arm is to the plan's start pose, slowly; later moves are stretched if the arm isn't where the plan expects.
Keep a hand on the power switch for the first runs.
"""
import argparse
import json
import sys
import time

ARM_ACTION = "/arm_controller/follow_joint_trajectory"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
# start pose of every simulated episode (OpenManipulatorX.init_qpos): gripper pointing straight down, grip point 12 cm
# in front of joint 1 and 8.7 cm above the table (1.7 cm base plate). Not ROBOTIS' bringup home [0, -1, 1, 0].
SIM_HOME = [0.0, -0.307, 0.249, 1.629]
GRIPPER_ACTION_DEFAULT = "/gripper_controller/gripper_cmd"
GRIPPER_JOINT = "gripper_left_joint"
# exit status of a plan run (omx_hw/run_trials.py reads it)
EXIT_OK, EXIT_ERROR, EXIT_QUIT, EXIT_INTERRUPTED = 0, 1, 3, 130


def describe(plan, steps):
    print("plan: %s with model %s, measured %s (cm, robot base frame), plate %.1f cm, z offset %.1f cm" % (
        plan["task"], plan["run"], plan["measured_cm"], plan["plate_cm"], plan["z_offset_cm"]))
    print("simulated outcome for these positions: %s" % plan["predicted"])
    for w in plan.get("warnings", []):
        print("WARNING from the planner: %s" % w)
    for i, s in enumerate(steps):
        what = ("%d points, %.1f s" % (len(s["points"]), s["points"][-1]["t"]) if s["type"] == "arm"
                else "gripper_left_joint -> %.3f m" % s["position"])
        print("  %2d. [%s]%s %s: %s" % (i + 1, s["type"], " (optional)" if s.get("optional") else "", s["label"], what))


class Robot:
    def __init__(self, args):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from sensor_msgs.msg import JointState

        from rclpy.signals import SignalHandlerOptions

        self.rclpy, self.FollowJointTrajectory, self.GripperCommand = rclpy, FollowJointTrajectory, GripperCommand
        # no rclpy signal handler: by default Ctrl+C shuts the ROS context down before our KeyboardInterrupt handler
        # runs, so the cancel request could not be sent and the controller would finish the motion
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.pending = None
        self.node = Node("rlwp_execute_plan")
        self.joints = {}
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.arm = ActionClient(self.node, FollowJointTrajectory, args.arm_action)
        gripper_action = args.gripper_action or self._find_gripper_action()
        self.gripper = ActionClient(self.node, GripperCommand, gripper_action)
        for client, name in ((self.arm, args.arm_action), (self.gripper, gripper_action)):
            if not client.wait_for_server(timeout_sec=10.0):
                names = [n for n, _ in rclpy.action.get_action_names_and_types(self.node)]
                sys.exit("action server %s not available (is the ROBOTIS bringup running?). Available actions: %s" % (name, names))
        print("connected: arm %s, gripper %s" % (args.arm_action, gripper_action))
        t0 = time.time()
        while not all(j in self.joints for j in ("joint1", "joint2", "joint3", "joint4")):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - t0 > 10:
                sys.exit("no /joint_states for joint1..joint4 received")
        self.goal_handle = None

    def _on_joints(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p

    def _find_gripper_action(self):
        """the GripperCommand action served by the gripper controller (name not fixed in ROBOTIS' config files)"""
        t0 = time.time()
        while time.time() - t0 < 5.0:
            found = [n for n, types in self.rclpy.action.get_action_names_and_types(self.node)
                     if "control_msgs/action/GripperCommand" in types]
            if found:
                return sorted(found, key=lambda n: "gripper" not in n)[0]
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        print("no GripperCommand action found yet, trying %s" % GRIPPER_ACTION_DEFAULT)
        return GRIPPER_ACTION_DEFAULT

    def arm_now(self, names):
        self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return [self.joints[n] for n in names]

    def _wait(self, future, timeout):
        t0 = time.time()
        while not future.done():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.time() - t0 > timeout:
                return False
        return True

    def run_trajectory(self, names, points):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(names)
        for p in points:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in p["q"]]
            sec = int(p["t"])
            pt.time_from_start = Duration(sec=sec, nanosec=int(round((p["t"] - sec) * 1e9)))
            goal.trajectory.points.append(pt)
        fut = self.pending = self.arm.send_goal_async(goal)
        if not self._wait(fut, 5.0) or not fut.result().accepted:
            raise RuntimeError("arm trajectory rejected by the controller")
        self.goal_handle, self.pending = fut.result(), None
        res = self.goal_handle.get_result_async()
        if not self._wait(res, points[-1]["t"] + 10.0):
            raise RuntimeError("arm trajectory did not finish in time")
        self.goal_handle = None
        code = res.result().result.error_code
        if code != 0:
            raise RuntimeError("arm trajectory failed, error code %d (%s)" % (code, res.result().result.error_string))

    def run_gripper(self, position, max_effort, timeout=5.0):
        goal = self.GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        fut = self.pending = self.gripper.send_goal_async(goal)
        if not self._wait(fut, 5.0) or not fut.result().accepted:
            raise RuntimeError("gripper command rejected")
        self.goal_handle, self.pending = fut.result(), None
        res = self.goal_handle.get_result_async()
        finished = self._wait(res, timeout)
        self.goal_handle = None
        # closing on a cube stalls before the target: the controller may report it as aborted / stalled; that is the
        # expected outcome of a grasp, so only report it
        r = res.result().result if finished else None
        pos = self.joints.get(GRIPPER_JOINT, float("nan"))
        return "gripper_left_joint now %.4f m%s" % (pos, "" if r is None else " (reached_goal %s, stalled %s)" % (
            r.reached_goal, r.stalled))

    def cancel(self):
        """cancel the running goal and wait for the controller to confirm (the arm controller then holds position)"""
        if self.goal_handle is None and self.pending is not None:  # interrupted while the goal was being sent
            if self._wait(self.pending, 2.0) and self.pending.result().accepted:
                self.goal_handle = self.pending.result()
        self.pending = None
        if self.goal_handle is None:
            return
        fut = self.goal_handle.cancel_goal_async()
        self.goal_handle = None
        if self._wait(fut, 2.0):
            print("   cancel confirmed by the controller: motion stopped, the arm holds its position")
        else:
            print("   WARNING: no cancel confirmation within 2 s. If the arm is still moving, stop the bringup "
                  "(torque off: support the arm) or switch off the 12 V supply.")

    def shutdown(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def ask(prompt, auto):
    if auto:
        print(prompt + " [auto]")
        return "run"
    a = input(prompt + "  [Enter = run, s = skip, q = quit] ").strip().lower()
    return {"": "run", "s": "skip", "q": "quit"}.get(a, "skip")


def stretched(points, q_now, joint_speed, time_scale, first_move):
    """time-scale the plan's points; make the start of the move safe from the arm's actual position"""
    pts = [{"t": p["t"] * time_scale, "q": p["q"]} for p in points]
    if first_move:  # go from wherever the arm is straight to the plan's start pose, slowly (joint space)
        q_end = pts[-1]["q"]
        d = max(abs(a - b) for a, b in zip(q_end, q_now))
        n = max(2, int(d / 0.02) + 1)
        dur = max(3.0, d / joint_speed) * time_scale
        return [{"t": dur * k / n, "q": [a + (b - a) * k / n for a, b in zip(q_now, q_end)]} for k in range(1, n + 1)]
    d = max(abs(a - b) for a, b in zip(pts[0]["q"], q_now))
    lead = d / joint_speed - pts[0]["t"]
    if d > 0.05:
        print("   note: the arm is %.2f rad away from where this move starts; easing in over %.1f s" % (d, max(lead, 0) + pts[0]["t"]))
    if lead > 0:
        pts = [{"t": p["t"] + lead, "q": p["q"]} for p in pts]
    return pts


def go_home(robot, args):
    """move the arm slowly (joint space) from wherever it is to the simulation's start pose; False if the user quit"""
    q_now = robot.arm_now(ARM_JOINTS)
    d = max(abs(a - b) for a, b in zip(q_now, SIM_HOME))
    print("arm now [%s], simulation start pose [%s] (largest joint distance %.2f rad)" % (
        ", ".join("%+.3f" % v for v in q_now), ", ".join("%+.3f" % v for v in SIM_HOME), d))
    choice = ask("move the arm to the simulation start pose (gripper pointing down)", args.yes)
    if choice != "run":
        return choice != "quit"
    pts = stretched([{"t": 1.0, "q": SIM_HOME}], q_now, args.joint_speed, args.time_scale, first_move=True)
    print("   moving (%.1f s)..." % pts[-1]["t"])
    robot.run_trajectory(ARM_JOINTS, pts)
    err = max(abs(a - b) for a, b in zip(robot.arm_now(ARM_JOINTS), SIM_HOME))
    print("   done; largest joint error %.3f rad. The gripper should point straight down, ~12 cm in front of the base "
          "axis, its grip point ~8.7 cm above the table." % err)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="plan JSON from plan_episode.py")
    ap.add_argument("--dry-run", action="store_true", help="print the steps only (no ROS needed)")
    ap.add_argument("--gripper-test", action="store_true",
                    help="move to the simulation start pose (asks first), then open, close, open the gripper")
    ap.add_argument("--home", action="store_true", help="only move the arm to the simulation start pose")
    ap.add_argument("--yes", action="store_true", help="don't ask before each step")
    ap.add_argument("--no-retreat", action="store_true", help="stop after the policy's steps (skip the optional retreat)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="extra slow-down factor on top of the plan's timing")
    ap.add_argument("--joint-speed", type=float, default=0.4, help="speed for easing into a move (rad/s)")
    ap.add_argument("--max-effort", type=float, default=1.0, help="GripperCommand max_effort")
    ap.add_argument("--arm-action", default=ARM_ACTION)
    ap.add_argument("--gripper-action", default=None, help="default: the GripperCommand action found on the system")
    args = ap.parse_args()

    if args.home and not args.gripper_test:
        robot = Robot(args)
        try:
            go_home(robot, args)
        except KeyboardInterrupt:
            print("\ninterrupted: cancelling the motion (the arm holds where it is)")
            robot.cancel()
        except RuntimeError as e:
            print("\nERROR: %s" % e)
            robot.cancel()
        finally:
            robot.shutdown()
        return

    if args.gripper_test:
        robot = Robot(args)
        try:
            if not go_home(robot, args):
                return
            for label, pos in (("open", 0.019), ("close (nothing between the fingers)", -0.010), ("open", 0.019)):
                if ask("gripper: %s -> gripper_left_joint %.3f m" % (label, pos), args.yes) == "quit":
                    break
                print("   " + robot.run_gripper(pos, args.max_effort))
            print("Check: 'open' must spread the fingers (~82 mm apart), 'close' must bring them together. If it is the "
                  "other way round, don't run plans: tell us, the plan's gripper positions must be swapped.")
        except KeyboardInterrupt:
            print("\ninterrupted: cancelling the motion (the arm holds where it is)")
            robot.cancel()
        except RuntimeError as e:
            print("\nERROR: %s" % e)
            robot.cancel()
        finally:
            robot.shutdown()
        return

    if not args.plan:
        ap.error("a plan file is required (or --gripper-test)")
    plan = json.load(open(args.plan))
    steps = [s for s in plan["steps"] if not (args.no_retreat and s.get("optional"))]
    describe(plan, steps)
    if args.dry_run:
        return

    robot = Robot(args)
    names = plan["arm_joints"]
    first_arm = True
    code = EXIT_OK
    try:
        for i, s in enumerate(steps):
            title = "\nstep %d/%d: %s" % (i + 1, len(steps), s["label"])
            if s["type"] == "gripper":
                choice = ask(title + " -> %s" % s["command"], args.yes)
            else:
                choice = ask(title, args.yes)
            if choice == "quit":
                print("stopped")
                code = EXIT_QUIT
                break
            if choice == "skip":
                continue
            if s["type"] == "arm":
                pts = stretched(s["points"], robot.arm_now(names), args.joint_speed, args.time_scale, first_arm)
                first_arm = False
                print("   moving (%.1f s)..." % pts[-1]["t"])
                robot.run_trajectory(names, pts)
                now = robot.arm_now(names)
                err = max(abs(a - b) for a, b in zip(now, pts[-1]["q"]))
                print("   done; largest joint error at the end %.3f rad" % err)
            else:
                print("   " + robot.run_gripper(s["position"], args.max_effort))
        if code == EXIT_OK:
            print("\nfinished")
    except KeyboardInterrupt:
        print("\ninterrupted: cancelling the motion")
        robot.cancel()
        code = EXIT_INTERRUPTED
    except RuntimeError as e:
        print("\nERROR: %s" % e)
        robot.cancel()
        code = EXIT_ERROR
    finally:
        robot.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
