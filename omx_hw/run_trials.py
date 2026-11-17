#!/usr/bin/env python3
"""
Real-robot Lift / Stack trials on the OpenMANIPULATOR-X with the learned waypoints, one random position after another.

For every trial:
  1. picks random cube positions inside the areas the model was trained on (robot base frame, cm);
  2. plans the episode with the trained model (omx_hw/plan_episode.py: waypoints, simulated outcome, robot plan) and
     opens the MuJoCo viewer for those positions (with the measuring frame): planned motion, then the simulated
     episode, on repeat;
  3. tells you where to put the cube(s) and waits for you;
  4. runs the plan on the robot (omx_hw/execute_plan.py through ROS 2);
       Lift:  grasp, lift, put the cube back, return to the start pose;
       Stack: grasp cubeA, carry it over the landing block, release, pull back, return to the start pose;
  5. asks for the real outcome and logs it next to the simulated prediction.

Run it in the project virtualenv, in a terminal where ROS 2 is sourced, with the ROBOTIS bringup running in another:

    # terminal 1
    source ~/ros2_ws/install/setup.bash
    ros2 launch open_manipulator_bringup open_manipulator_x.launch.py port_name:=/dev/ttyUSB0 init_position:=false
    # terminal 2
    source ~/ros2_ws/install/setup.bash
    cd ~/Documents/RL/rl-waypoints
    ../.venv/bin/python omx_hw/run_trials.py                                    # Lift, 10 trials
    ../.venv/bin/python omx_hw/run_trials.py --task Stack --cubeB 10.5 13.5     # Stack, landing block fixed there
    ../.venv/bin/python omx_hw/run_trials.py --task Stack --cubeB 10.5 13.5 --z-offset-cm 3 -n 1   # dry run in the air

Ctrl+C while the robot moves: stops the motion (the arm holds its pose, torque stays on), then asks how to go on.
Results: omx_hw/trials/<date-time>/results.csv (+ the plan, planner log and executor log of every trial).
"""
import argparse
import csv
import datetime
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLANNER = os.path.join(HERE, "plan_episode.py")
EXECUTOR = os.path.join(HERE, "execute_plan.py")
ROS_PYTHON = "/usr/bin/python3"  # the system Python ROS 2 Jazzy is built for (rclpy); the planner runs in this venv

# trained spawn areas, robot base frame (cm), ((x0, x1), (y0, y1)): plan_episode.py prints the same from the task
# configs and refuses positions outside them
AREAS = {
    "Lift": {"cube": ((16.8, 19.8), (-3.0, 3.0))},
    "Stack": {"cubeA": ((16.3, 18.3), (-3.0, 0.0)), "cubeB": ((8.8, 11.8), (12.0, 15.0))},
}
# waypoint boxes of the task configs (robot base frame, cm): the model can only send the gripper inside these, so
# positions beyond them can't work whatever the planner is told (Lift_omx.yaml / Stack_omx.yaml `workspace:`)
BOXES = {"Lift": ((8.8, 20.8), (-10.0, 10.0)), "Stack": ((8.8, 18.8), (-10.0, 15.0))}
WHAT = {"cube": "the 55 mm cube", "cubeA": "cubeA (55 mm, the one to pick)",
        "cubeB": "the landing block (65 mm, top 48 mm above the table, fixed)"}
QUESTION = {"Lift": "Did the real robot lift the cube?",
            "Stack": "Is cubeA resting on the landing block at the end (stacked, not fallen off)?"}
EXIT_NAMES = {0: "finished", 1: "error", 3: "quit by you", 130: "interrupted (Ctrl+C)"}
NOISE = ("robosuite WARNING", "Gym has been unmaintained", "Please upgrade to Gymnasium", "Users of this version of Gym",
         "See the migration guide")


def grid(lo, hi, margin, step):
    """positions from lo + margin to hi - margin, on a grid of `step` cm (easy to measure); step 0 = continuous"""
    a, b = lo + margin, hi - margin
    if step <= 0:
        return None, (a, b)
    first, last = math.ceil(round(a / step, 6)) * step, math.floor(round(b / step, 6)) * step
    n = int(round((last - first) / step)) + 1
    return [round(first + k * step, 2) for k in range(n)], (a, b)


class Sampler:
    """random position per cube, distinct grid cells while they last (or continuous with --step-cm 0)"""

    def __init__(self, area, margin, step, rng):
        self.rng = rng
        self.xs, self.xr = grid(*area[0], margin, step)
        self.ys, self.yr = grid(*area[1], margin, step)
        self.cells = None if self.xs is None else [(x, y) for x in self.xs for y in self.ys]
        if self.cells:
            rng.shuffle(self.cells)

    def next(self):
        if self.cells is None:
            return round(self.rng.uniform(*self.xr), 1), round(self.rng.uniform(*self.yr), 1)
        if not self.cells:  # all used: start over
            self.cells = [(x, y) for x in self.xs for y in self.ys]
            self.rng.shuffle(self.cells)
        return self.cells.pop()


def describe(x, y):
    side = "on the centre line" if y == 0 else "%.1f cm to the robot's %s" % (abs(y), "LEFT" if y > 0 else "RIGHT")
    return "x = %.1f cm forward from joint 1's axis, y = %+.1f cm (%s)" % (x, y, side)


def run_child(cmd, log_path):
    """run a child in the foreground of this terminal, showing and logging its output. Ctrl+C reaches the child too
    (same process group): this process just keeps waiting, the child handles it (the executor cancels the motion)."""
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO, text=True, bufsize=1)
        while True:
            try:
                for line in p.stdout:
                    log.write(line)
                    if not line.startswith("[WARN]") and "ROS_LOCALHOST_ONLY" not in line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                return p.wait()
            except KeyboardInterrupt:
                continue


def start_planner(task, pos, plan_path, log_path, args):
    """plan_episode.py with the viewer on repeat, in its own session (Ctrl+C in this terminal doesn't close it);
    returns the process once the plan file exists, or None if planning failed"""
    if os.path.exists(plan_path):
        os.remove(plan_path)
    cmd = [sys.executable, "-u", PLANNER, "--task", task, "--out", plan_path, "--plate-cm", str(args.plate_cm),
           "--seed", str(args.plan_seed), "--z-offset-cm", str(args.z_offset_cm)]
    if args.allow_outside:
        cmd.append("--allow-outside")
    for k, (x, y) in pos.items():
        cmd += ["--" + k, str(x), str(y)]
    if not args.no_viewer:
        cmd.append("--view-loop")
    env = dict(os.environ)
    env.pop("MUJOCO_GL", None)  # the viewer needs GLFW, not EGL
    env.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
    with open(log_path, "w") as log:  # the child keeps its own copy of the file descriptor
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env, start_new_session=True)
    t0 = time.time()
    try:
        while not os.path.exists(plan_path):
            if p.poll() is not None or time.time() - t0 > 300:
                stop(p)
                print(open(log_path).read())
                return None
            time.sleep(0.2)
    except KeyboardInterrupt:  # its own session: Ctrl+C doesn't reach it
        stop(p)
        raise
    return p


def stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def ask(prompt, options):
    while True:
        a = input(prompt).strip().lower()
        if a in options:
            return a
        print("   please answer one of: %s" % ", ".join(repr(o) if o else "Enter" for o in options))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(AREAS), default="Lift")
    ap.add_argument("-n", "--trials", type=int, default=10, help="number of trials (default 10)")
    ap.add_argument("--cube", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Lift: the cube always here")
    ap.add_argument("--cubeA", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Stack: cubeA always here")
    ap.add_argument("--cubeB", nargs=2, type=float, metavar=("X_CM", "Y_CM"),
                    help="Stack: the landing block stays at this measured position (recommended: fix it once); "
                         "default: a new random position every trial")
    ap.add_argument("--allow-outside", action="store_true",
                    help="plan positions outside the trained areas too (the simulated outcome still shows whether the "
                         "model copes; a failing simulation asks before running)")
    ap.add_argument("--expand-cm", type=float, default=0.0,
                    help="with --allow-outside: draw random positions from the trained areas grown by this much on every "
                         "side (clipped to the waypoint box)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the random positions (default: random)")
    ap.add_argument("--step-cm", type=float, default=0.5,
                    help="positions on a grid of this step, easy to measure (default 0.5 cm; 0 = any position)")
    ap.add_argument("--margin-cm", type=float, default=0.3,
                    help="stay this far inside the trained area, for placement errors (default 0.3 cm)")
    ap.add_argument("--plate-cm", type=float, default=1.7, help="robot base height above the table (planner option)")
    ap.add_argument("--z-offset-cm", type=float, default=0.0,
                    help="dry runs: everything this much higher on the robot (planner option; leave cubeA away)")
    ap.add_argument("--plan-seed", type=int, default=0, help="seed of the waypoint optimiser (planner option)")
    ap.add_argument("--time-scale", type=float, default=2.0,
                    help="executor slow-down on top of the plan's timing (default 2, as in the first tests)")
    ap.add_argument("--confirm-steps", action="store_true", help="ask before every step on the robot (Enter / s / q)")
    ap.add_argument("--no-retreat", action="store_true", help="stop after the policy's steps (skip the retreat)")
    ap.add_argument("--no-viewer", action="store_true", help="don't open the MuJoCo viewer")
    args = ap.parse_args()
    task = args.task
    for k in ("cube", "cubeA", "cubeB"):
        if getattr(args, k) and k not in AREAS[task]:
            ap.error("--%s is not used by %s (its cubes: %s)" % (k, task, ", ".join(AREAS[task])))
    if args.expand_cm and not args.allow_outside:
        ap.error("--expand-cm needs --allow-outside")

    if subprocess.run([ROS_PYTHON, "-c", "import rclpy, control_msgs"], capture_output=True).returncode != 0:
        sys.exit("ROS 2 is not available to %s: run 'source ~/ros2_ws/install/setup.bash' in this terminal first."
                 % ROS_PYTHON)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(10000)
    rng = random.Random(seed)
    fixed = {k: tuple(getattr(args, k)) for k in AREAS[task] if getattr(args, k)}
    (bx0, bx1), (by0, by1) = BOXES[task]
    e = args.expand_cm
    areas = {k: ((max(x0 - e, bx0), min(x1 + e, bx1)), (max(y0 - e, by0), min(y1 + e, by1)))
             for k, ((x0, x1), (y0, y1)) in AREAS[task].items()}
    samplers = {k: Sampler(a, args.margin_cm, args.step_cm, rng) for k, a in areas.items() if k not in fixed}
    for k, (x, y) in fixed.items():
        (x0, x1), (y0, y1) = AREAS[task][k]
        if not (x0 - 0.5 <= x <= x1 + 0.5 and y0 - 0.5 <= y <= y1 + 0.5) and not args.allow_outside:
            sys.exit("--%s %.1f %.1f is outside the trained area (x %.1f..%.1f, y %.1f..%.1f): add --allow-outside to "
                     "try it anyway" % (k, x, y, x0, x1, y0, y1))
    keys = list(AREAS[task])

    out_dir = os.path.join(HERE, "trials", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + task.lower())
    os.makedirs(out_dir)
    csv_path = os.path.join(out_dir, "results.csv")
    fields = ["trial", "attempt"] + ["%s_%s_cm" % (k, c) for k in keys for c in "xy"] + [
        "z_offset_cm", "sim_prediction", "sim_success", "robot_run", "real_success", "note", "plan"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    json.dump(vars(args) | {"seed": seed}, open(os.path.join(out_dir, "settings.json"), "w"), indent=1)

    print("%s trials on the real OM-X: %d trials, positions seed %d%s" % (
        task, args.trials, seed, ", %.1f cm grid" % args.step_cm if args.step_cm > 0 else ""))
    for k, smp in samplers.items():
        print("  %-6s random in x %.1f..%.1f cm, y %+.1f..%+.1f cm" % (k, *smp.xr, *smp.yr))
    for k, (x, y) in fixed.items():
        print("  %-6s fixed at (%.1f, %+.1f) cm" % (k, x, y))
    if args.allow_outside:
        print("  OUTSIDE the trained areas allowed: trust the simulated outcome of each position, not the model")
    if args.z_offset_cm:
        print("  DRY RUN: everything %.1f cm higher than planned; leave the cube to pick off the table" % args.z_offset_cm)
    print("results -> %s" % csv_path)
    print("Positions: robot base frame, origin on joint 1's axis, x straight out along the arm at joint1 = 0, y to the "
          "robot's left (the viewer draws these axes). Place each cube's centre there, faces square to the x axis.\n")

    done, real_ok, sim_ok = 0, 0, 0
    viewer = None
    try:
        trial, attempt = 0, 0
        while trial < args.trials:
            trial += 1
            attempt += 1
            pos = {k: (fixed[k] if k in fixed else samplers[k].next()) for k in keys}
            tag = "attempt%02d" % attempt
            plan_path = os.path.join(out_dir, tag + "_plan.json")
            print("=" * 100)
            print("TRIAL %d/%d: %s. Planning with the learned waypoints%s..." % (
                trial, args.trials, ", ".join("%s at (%.1f, %+.1f) cm" % (k, *v) for k, v in pos.items()),
                " and opening the MuJoCo viewer" if not args.no_viewer else ""))
            stop(viewer)
            viewer = start_planner(task, pos, plan_path, os.path.join(out_dir, tag + "_planner.log"), args)
            if viewer is None:
                print("planning failed (log above); next position")
                trial -= 1
                continue
            for line in open(os.path.join(out_dir, tag + "_planner.log")):
                if line.strip() and not any(n in line for n in NOISE) and "plan written" not in line:
                    print("   " + line.rstrip())
            plan = json.load(open(plan_path))
            sim_success = "success True" in plan["predicted"]
            row = {"trial": trial, "attempt": attempt, "z_offset_cm": args.z_offset_cm, "sim_prediction": plan["predicted"],
                   "sim_success": sim_success, "plan": os.path.basename(plan_path)}
            for k, (x, y) in pos.items():
                row["%s_x_cm" % k], row["%s_y_cm" % k] = x, y

            print()
            for k, (x, y) in pos.items():
                if args.z_offset_cm and k in ("cube", "cubeA"):
                    print(">>> DRY RUN: keep %s OFF the table (it would be at %s)." % (WHAT[k], describe(x, y)))
                else:
                    print(">>> Put %s at %s%s." % (WHAT[k], describe(x, y),
                                                   " (same as before)" if k in fixed and attempt > 1 else ""))
            if plan["warnings"]:
                print(">>> The planner warns: %s" % "; ".join(plan["warnings"]))
                a = ask(">>> Run it anyway? [r = run, Enter = skip to a new position, q = quit] ", ("r", "", "q"))
                a = {"r": "", "": "s"}.get(a, a)
            else:
                a = ask(">>> In place and workspace clear? [Enter = run on the robot, s = skip to a new position, "
                        "q = quit] ", ("", "s", "q"))
            if a == "q":
                break
            if a == "s":
                row.update(trial="", robot_run="skipped", real_success="", note="")
                trial -= 1  # a skipped position doesn't count as a trial
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fields).writerow(row)
                continue

            cmd = [ROS_PYTHON, "-u", EXECUTOR, plan_path, "--time-scale", str(args.time_scale)]
            if not args.confirm_steps:
                cmd.append("--yes")
            if args.no_retreat:
                cmd.append("--no-retreat")
            print("\nrunning on the robot (Ctrl+C stops the motion; the arm then holds its pose)\n")
            code = run_child(cmd, os.path.join(out_dir, tag + "_robot.log"))
            row["robot_run"] = EXIT_NAMES.get(code, "exit %d" % code)
            print("\nrobot run: %s" % row["robot_run"])

            if args.z_offset_cm:
                ok = ask(">>> Dry run: did the motion look right (no contact with the block)? [y / n / Enter = don't "
                         "count] ", ("y", "n", ""))
            else:
                ok = ask(">>> %s [y / n / Enter = don't count] " % QUESTION[task], ("y", "n", ""))
            row["real_success"] = {"y": True, "n": False, "": ""}[ok]
            row["note"] = input(">>> Note (optional, Enter = none): ").strip()
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
            if ok:
                done += 1
                real_ok += ok == "y"
                sim_ok += sim_success
                print("so far: real robot %d/%d, simulation predicted %d/%d" % (real_ok, done, sim_ok, done))

            if code != 0:
                print(">>> The run did not finish: the arm may have stopped mid-motion, maybe holding the cube. The next "
                      "run first moves the arm straight from where it is to the start pose, and only then opens the "
                      "gripper: make sure that path is clear.")
                if ask(">>> Remove the cube / clear the arm, then: [c = continue, q = quit] ", ("c", "q")) == "q":
                    break
            elif task == "Lift" and not args.no_retreat:
                print(">>> The cube is back where you put it; the arm is at the start pose.")
            elif task == "Stack" and not args.z_offset_cm:
                print(">>> Take cubeA off the block before the next trial%s." % (
                    "" if "cubeB" in fixed else " (the block moves too)"))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop(viewer)

    print("\n%d counted trials: real robot %d, simulation predicted %d. Results: %s" % (done, real_ok, sim_ok, csv_path))


if __name__ == "__main__":
    main()
