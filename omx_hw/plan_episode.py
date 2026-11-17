"""
Plan one Lift / Stack episode for the real OpenMANIPULATOR-X from hand-measured cube positions (no ROS, no hardware).

    cd rl-waypoints && ../.venv/bin/python omx_hw/plan_episode.py --task Lift  --cube 19.5 -1.0 --out plan_lift.json
    cd rl-waypoints && ../.venv/bin/python omx_hw/plan_episode.py --task Stack --cubeA 18.5 -1.5 --cubeB 11.5 13.5 --out plan_stack.json
    ... add --view to watch the planned motion and the simulated episode in MuJoCo's viewer

Positions are in centimetres in the ROBOT BASE frame: origin on joint 1's rotation axis, x forward (straight out from
the arm at joint1 = 0), y to the robot's left; cube centres, cubes square to the x axis (as in training).

What it does:
  1. builds the trained environment (task YAML) and puts the cubes at the measured positions, exactly as the simulator
     would have spawned them, and asks the trained reward models for the two waypoints (as rlwp's evaluation does);
  2. simulates that episode with rlwp's own action rule -> predicted outcome (success / not) for these positions;
  3. turns that simulated episode into the robot plan: the arm's joint angles at every simulation step, replayed slower
     (the simulator's arm model is ROBOTIS' MJCF: same joint names, zeros and directions as their ROS 2 URDF), with the
     gripper commands where the simulation gave them. This is the motion that was just verified, it is reachable by
     construction, and it includes rlwp's way of moving (rise-first for Stack, the arm settling under the cube). Only the
     optional retreat and the --z-offset-cm / --plate-cm adjustments use inverse kinematics (gripper pointing down);
  4. checks joint limits, IK accuracy and finger clearance above the table along every move;
  5. writes the plan (JSON) for omx_hw/execute_plan.py, which runs it on the robot through ROS 2.
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RLWP = os.path.join(HERE, "..", "rlwp")
sys.path.insert(0, RLWP)
os.chdir(RLWP)  # rlwp loads models from ./models
from common import get_controller_config, get_env_kwargs, get_save_name, get_workspace, lift_first_error  # noqa: E402
from eval import Method  # noqa: E402

import robosuite as suite  # noqa: E402

TASKS = {"Lift": ("Lift_omx", "omx_lift_v2_s1"), "Stack": ("Stack_omx", "omx_stack_v3_s0")}
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ROS_HOME = [0.0, -1.0, 1.0, 0.0]          # pose ROBOTIS' bringup moves to (open_manipulator_bringup initial_positions.yaml)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.019, -0.010  # gripper_left_joint (m); limits -0.011 .. 0.020, 0.020 = fully open (82 mm)
SPAWN_DROP = 0.01                           # robosuite spawns free cubes 1 cm above the table (z_offset); the policy
                                            # was trained on that reset observation


# ---------------------------------------------------------------------------------------------------------------- env
def make_env(task_name):
    task = yaml.safe_load(open(os.path.join("cfg", "task", task_name + ".yaml")))
    ec = task["env"]
    env = suite.make(task["name"], robots=ec["robot"], controller_configs=get_controller_config(ec), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True, control_freq=10,
                     initialization_noise=None, hard_reset=False, **get_env_kwargs(ec, None, ec["use_latch"]))
    return env, task


def place_cubes(env, name, cubes_world):
    """put the cubes where the simulator's sampler would have put them (no rotation, free cubes 1 cm up) and return the
    reset observation"""
    table_z = env.table_offset[2]
    if name == "Lift":
        (x, y), = cubes_world
        env.sim.data.set_joint_qpos(env.cube.joints[0], [x, y, table_z + env.cube.size[2] + SPAWN_DROP, 1, 0, 0, 0])
        env._cube_start_xy = np.array([x, y])
    else:
        (ax, ay), (bx, by) = cubes_world
        env.sim.data.set_joint_qpos(env.cubeA.joints[0], [ax, ay, table_z + env.cubeA.size[2] + SPAWN_DROP, 1, 0, 0, 0])
        b = np.array([bx, by, table_z + env.cubeB.size[2]])
        if env.cubeB_fixed:
            env.sim.model.body_pos[env.cubeB_body_id] = b
            env.sim.model.body_quat[env.cubeB_body_id] = [1, 0, 0, 0]
        else:
            env.sim.data.set_joint_qpos(env.cubeB.joints[0], np.r_[b + [0, 0, SPAWN_DROP], 1, 0, 0, 0])
        env._cubeB_start = b
    env.sim.forward()
    return env._get_observations(force_update=True)


def policy_objs(name, obs):
    return obs["cube_pos"] if name == "Lift" else np.concatenate([obs["cubeA_pos"], obs["cubeB_pos"]])


def rollout(env, task, traj, targets, obs, viewer=None):
    """the episode with rlwp's action rule (as eval.py / view_policy.py); returns success, the end state and a record
    of the arm's joint angles per step with the phase each step belongs to ("move i" / "gripper i")"""
    ec, name = task["env"], task["name"]
    wp_steps, gripper_steps = ec["wp_steps"], ec["gripper_steps"]
    time_s, success, record = 0, False, []
    for t in range(len(targets) * wp_steps):
        i = t // wp_steps
        err = lift_first_error(targets[i] - obs["robot0_eef_pos"], i, ec.get("lift_first", False))
        if t < 10:
            a = np.r_[10 * err, 0, 0, 0, -1.0]
        elif time_s >= wp_steps - gripper_steps:
            a = np.r_[np.zeros(6), traj[i, 3:]]
        else:
            a = np.r_[10 * err, 0, 0, 0, 0.0]
        phase = ("gripper %d" if t >= 10 and time_s >= wp_steps - gripper_steps else "move %d") % i
        time_s = time_s + 1 if time_s + 1 < wp_steps else 1
        obs = env.step(a)[0]
        success |= env._check_success()
        record.append((phase, env.robots[0]._joint_positions.copy()))
        if viewer is not None:
            viewer.sync()
            time.sleep(env.control_timestep)
    return success, obs, record


# ----------------------------------------------------------------------------------------------- kinematics on MuJoCo
class Kinematics:
    """FK / IK / clearance on the simulator's arm model (a private MjData, the env is not disturbed)"""

    def __init__(self, env, start_q):
        self.m = env.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = env.sim.data.qpos
        robot = env.robots[0]
        self.qidx = np.array(robot._ref_joint_pos_indexes)
        self.lo, self.hi = self.m.jnt_range[robot._ref_joint_indexes].T
        self.site = robot.eef_site_id
        self.table_z = env.table_offset[2]
        self.grip_geoms = [i for i in range(self.m.ngeom) if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                           .startswith("gripper0_") and self.m.geom_contype[i]]
        # which axis of the grip-point frame points down in the start pose (the approach axis)
        self.fk(start_q)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        self.k = int(np.argmax(np.abs(R[2, :])))
        self.sign = -np.sign(R[2, self.k])

    def fk(self, q):
        self.d.qpos[self.qidx] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()

    def approach_axis(self):
        return self.sign * self.d.site_xmat[self.site].reshape(3, 3)[:, self.k]

    def tilt_deg(self, q):
        self.fk(q)
        return float(np.degrees(np.arccos(np.clip(-self.approach_axis()[2], -1, 1))))

    def finger_clearance(self, q):
        """lowest point of the gripper's collision geoms above the table (m)"""
        self.fk(q)
        low = np.inf
        for g in self.grip_geoms:
            c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
            R = self.d.geom_xmat[g].reshape(3, 3)
            low = min(low, (self.d.geom_xpos[g] + R @ c)[2] - (np.abs(R) @ h)[2])
        return float(low - self.table_z)

    def ik(self, target, q0, iters=300, w_ori=0.05):
        """damped least squares: grip point at target, approach axis straight down (yaw is free on this arm)"""
        down = np.array([0.0, 0.0, -1.0])
        q = np.array(q0, dtype=float)

        def residual(qq):
            p = self.fk(qq)
            return np.r_[target - p, w_ori * np.cross(self.approach_axis(), down)]

        for _ in range(iters):
            e = residual(q)
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < w_ori * 2e-3:
                break
            J = np.zeros((6, 4))
            for j in range(4):
                dq = np.zeros(4)
                dq[j] = 1e-6
                J[:, j] = (residual(q + dq) - e) / 1e-6
            J = -J  # residual = target - f(q)  ->  d(residual)/dq = -df/dq
            step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + step, self.lo, self.hi)
        p = self.fk(q)
        return q, float(np.linalg.norm(target - p)), self.tilt_deg(q)


def retreat_contacts(env, end_qpos, steps, margin=0.005):
    """replay the plan's retreat steps (gripper open) in the simulated end state and report the closest approach of the
    robot / gripper to a cube (None if nothing comes within `margin`). The planner's playback is kinematic, so without
    this nothing would notice the arm sweeping through the stacked cube"""
    m = env.sim.model._model
    d = mujoco.MjData(m)
    d.qpos[:] = end_qpos
    robot = env.robots[0]
    qi, gq = robot._ref_joint_pos_indexes, robot._ref_gripper_joint_pos_indexes
    cubes = {i for i in range(m.ngeom) if m.geom_bodyid[i] in (env.cubeA_body_id, env.cubeB_body_id)}
    arm = {i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("robot0_", "gripper0_"))}
    saved = {g: m.geom_margin[g] for g in cubes}
    worst = None
    try:
        for g in cubes:
            m.geom_margin[g] = margin
        for s in steps:
            if s["type"] != "arm" or not s["label"].startswith("retreat"):
                continue
            for pt in s["points"]:
                d.qpos[qi] = pt["q"]
                d.qpos[gq] = [0.02, -0.02]
                mujoco.mj_forward(m, d)
                for c in d.contact[:d.ncon]:
                    pair = {c.geom1, c.geom2}
                    if pair & cubes and pair & arm and (worst is None or c.dist < worst[0]):
                        worst = (c.dist, s["label"].split(":")[1].strip()[:40], pt["t"])
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v
    return None if worst is None else "%.1f mm at t %.1f s of '%s'" % (1000 * worst[0], worst[2], worst[1])


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--run", default=None, help="trained model run name (default: best model of the task)")
    ap.add_argument("--cube", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Lift: cube centre, robot base frame")
    ap.add_argument("--cubeA", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Stack: cube to pick (55 mm)")
    ap.add_argument("--cubeB", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Stack: landing cube (65 mm, in recess)")
    ap.add_argument("--plate-cm", type=float, default=1.7,
                    help="height of the robot base above the table top (the simulated base plate is 1.7 cm)")
    ap.add_argument("--z-offset-cm", type=float, default=0.0,
                    help="raise every waypoint by this much on the robot (dry runs above the cubes); the prediction "
                         "still uses the real waypoints")
    ap.add_argument("--time-scale", type=float, default=4.0,
                    help="replay the simulated motion this many times slower than the simulation (default 4)")
    ap.add_argument("--joint-speed", type=float, default=0.5,
                    help="joint speed limit (rad/s): slower moves are stretched to respect it")
    ap.add_argument("--min-clearance-mm", type=float, default=3.0,
                    help="keep the fingers at least this far above the table: in the simulation the table stops the "
                         "arm, a position-controlled real arm would press into it instead")
    ap.add_argument("--retreat-pull-cm", type=float, default=6.0,
                    help="Stack retreat: pull the open gripper this far straight back toward the base after the release")
    ap.add_argument("--retreat-rise-cm", type=float, default=1.5,
                    help="Stack retreat: rise this much while pulling back")
    ap.add_argument("--allow-outside", action="store_true", help="plan even if a cube is outside the trained spawn area")
    ap.add_argument("--view", action="store_true", help="show the planned robot motion, then the simulated episode")
    ap.add_argument("--view-loop", action="store_true", help="as --view, but replay both until the window is closed")
    ap.add_argument("--out", default=None, help="plan file (JSON) for execute_plan.py")
    ap.add_argument("--seed", type=int, default=0, help="seed of the waypoint optimiser's random restarts")
    args = ap.parse_args()

    name = args.task
    task_name, default_run = TASKS[name]
    run = args.run or default_run
    need = ["cube"] if name == "Lift" else ["cubeA", "cubeB"]
    for k in need:
        if getattr(args, k) is None:
            ap.error("--%s X_CM Y_CM is required for %s" % (k, name))
    np.random.seed(args.seed)

    env, task = make_env(task_name)
    ec = task["env"]
    obs = env.reset()
    start = obs["robot0_eef_pos"].copy()
    start_q = env.robots[0]._joint_positions.copy()
    axis = env.sim.data.xanchor[env.sim.model.joint_name2id("robot0_joint1")][:2].copy()  # joint 1's axis (x, y)

    # measured cube positions (robot base frame, cm) -> simulator world frame (m)
    measured = {k: np.array(getattr(args, k)) for k in need}
    cubes_world = [axis + measured[k] / 100.0 for k in need]

    # trained spawn areas, in the robot base frame
    kw = ec["env_kwargs"]
    ranges = ({"cube": (kw["cube_x_range"], kw["cube_y_range"])} if name == "Lift" else
              {"cubeA": (kw["cubeA_x_range"], kw["cubeA_y_range"]), "cubeB": (kw["cubeB_x_range"], kw["cubeB_y_range"])})
    warnings, areas = [], {}
    print("\n%s with model %s; positions in the robot base frame (cm), trained spawn area in brackets:" % (task_name, run))
    for k in need:
        xr, yr = [(np.array(r) + env.table_offset[i] - axis[i]) * 100 for i, r in enumerate(ranges[k])]
        areas[k] = (tuple(xr), tuple(yr))
        x, y = measured[k]
        inside = xr[0] - 0.5 <= x <= xr[1] + 0.5 and yr[0] - 0.5 <= y <= yr[1] + 0.5
        print("  %-6s x %5.1f [%5.1f, %5.1f]  y %5.1f [%5.1f, %5.1f]  %s" % (k, x, xr[0], xr[1], y, yr[0], yr[1],
                                                                            "ok" if inside else "OUTSIDE the trained area"))
        if not inside:
            warnings.append("%s is outside the trained spawn area: the model never saw such a position" % k)
    if warnings and not args.allow_outside:
        sys.exit("\n".join(warnings) + "\nMove the cube (or pass --allow-outside to plan anyway).")

    # waypoints from the trained reward models (as rlwp's evaluation: frozen ensemble, no exploration)
    obs = place_cubes(env, name, cubes_world)
    agent = Method(state_dim=ec["state_dim"], objs=policy_objs(name, obs), wp_id=2,
                   save_name=get_save_name(name, ec["robot"], run, None, ec["use_latch"]),
                   config={"task": task, "n_inits": 5}, workspace=get_workspace(ec, start))
    traj = agent.traj_opt(1, policy_objs(name, obs)).reshape(2, ec["state_dim"])
    targets = traj[:, :3] + start
    grip = ["close" if g > 0 else "open" for g in traj[:, 3]]

    # predicted outcome: the same episode in the simulator
    success, end, record = rollout(env, task, traj, targets, obs)
    end_qpos = env.sim.data.qpos.copy()  # simulated end state (cubes where the episode left them), for the retreat check
    if name == "Lift":
        rise = end["cube_pos"][2] - (env.table_offset[2] + env.cube.size[2])
        outcome = "lifted %.1f cm, success %s" % (100 * rise, success)
    else:
        a, b = end["cubeA_pos"], end["cubeB_pos"]
        outcome = "cubeA %.0f mm off cubeB's centre, success %s" % (1000 * np.linalg.norm(a[:2] - b[:2]), success)
    print("\nsimulated episode with these positions: %s" % outcome)
    if not success:
        warnings.append("the simulated episode with these positions does NOT succeed")

    # ---- robot plan: the simulated episode's joint angles, replayed slower
    kin = Kinematics(env, start_q)
    dz_plate = 0.017 - args.plate_cm / 100.0  # base higher than in the sim -> targets lower in the robot's frame
    kin.table_z = env.table_offset[2] - dz_plate  # table as seen from the robot
    z_off = args.z_offset_cm / 100.0
    shift = z_off - dz_plate  # robot-frame height adjustment of every simulated point (0 normally)
    steps, worst = [], {"ik_err": 0.0, "tilt": 0.0, "clear": np.inf, "raised": 0, "raised_max": 0.0}
    min_clear = args.min_clearance_mm / 1000.0
    dt = env.control_timestep * args.time_scale

    def adjust(q_seq):
        """height adjustments by IK: --z-offset-cm / --plate-cm, and raising any point whose fingers would come closer
        than --min-clearance-mm to the table"""
        out, qq = [], np.array(q_seq[0])
        for q in q_seq:
            q = np.array(q)
            dz = shift
            c = kin.finger_clearance(q) + shift  # clearance after the global shift (fingers move rigidly with the grip point)
            if c < min_clear:
                dz += min_clear - c + 0.0005
                worst["raised"] += 1
                worst["raised_max"] = max(worst["raised_max"], min_clear - c + 0.0005)
            if abs(dz) > 1e-9:
                q_new, e, tilt = kin.ik(kin.fk(q) + [0, 0, dz], qq)
                worst["ik_err"], worst["tilt"] = max(worst["ik_err"], e), max(worst["tilt"], tilt)
                q = q_new
            out.append(q)
            qq = q
        return out

    def arm_step(label, q_seq, q_prev, base_dt):
        """a trajectory through q_seq (starting from q_prev), base_dt per point, stretched to the joint speed limit"""
        pts, t, qp = [], 0.0, np.array(q_prev)
        for q in q_seq:
            worst["clear"] = min(worst["clear"], kin.finger_clearance(q))
            t += max(base_dt, np.max(np.abs(q - qp)) / args.joint_speed)
            pts.append({"t": round(t, 3), "q": [round(float(v), 5) for v in q]})
            qp = q
        steps.append({"type": "arm", "label": label, "points": pts})
        return qp

    def gripper(label, cmd):
        steps.append({"type": "gripper", "label": label, "command": cmd,
                      "position": GRIPPER_CLOSED if cmd == "close" else GRIPPER_OPEN})

    def interp(q0, q1, step=0.02):
        n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / step)))
        return [q0 + (q1 - q0) * k / n for k in range(1, n + 1)]

    moves = {i: adjust([q for ph, q in record if ph == "move %d" % i]) for i in range(2)}
    q_start = adjust([start_q])[0]
    q = arm_step("go to the start pose of every simulated episode (gripper down, %.1f cm in front of joint 1, %.1f cm "
                 "above the table)" % (100 * (start[0] - axis[0]), 100 * (start[2] - env.table_offset[2])),
                 interp(np.array(ROS_HOME), q_start),
                 ROS_HOME, 0.05)
    gripper("open the gripper (as every simulated episode starts)", "open")
    for i in range(2):
        tgt = targets[i]
        q = arm_step("waypoint %d -> x %.1f y %.1f z %.1f cm (base frame; z above the table), simulated motion" % (
            i + 1, *((tgt[:2] - axis) * 100), 100 * (tgt[2] - env.table_offset[2] + z_off)), moves[i], q, dt)
        gripper("waypoint %d gripper: %s" % (i + 1, grip[i]), grip[i])
    # retreat (optional on the robot), by retracing the simulated motion backwards with the gripper open: Lift first
    # puts the cube back down where it was picked (retrace waypoint 2 while still holding it)
    if name == "Lift" and grip[1] == "close":
        q = arm_step("retreat: put the cube back down (waypoint 2 motion backwards)", moves[1][::-1][1:] + [moves[0][-1]], q, dt)
        gripper("retreat: release the cube", "open")
        q = arm_step("retreat: back to the start pose (waypoint 1 motion backwards)", moves[0][::-1][1:] + [q_start], q, dt)
    else:
        # after a release (Stack) the open fingers straddle the cube just stacked: retracing the carry would sweep them
        # through it (checked: 22 mm deep), and straight up is out of reach at this distance. Pull straight back toward
        # the base instead (the fingers slide off along the cube's faces), rising a little, then go to the start pose.
        # The cube lands where the off-centre grasp (P1) puts it, so a fixed pull isn't always enough: the joint-space
        # move to the start pose then swings a finger through it. Try larger pulls / rises until the retreat is
        # reachable and clear of both cubes in the simulated end state.
        p0 = kin.fk(q)
        radial = np.r_[p0[:2] - axis, 0.0]
        radial /= np.linalg.norm(radial)
        q_rel, n_before = np.array(q), len(steps)
        options = [(args.retreat_pull_cm, args.retreat_rise_cm)] + [o for o in ((8, 3), (10, 4), (12, 5))
                                                                     if o[0] > args.retreat_pull_cm]
        for i_opt, (pull, rise) in enumerate(options):
            del steps[n_before:]
            n = max(2, int(np.ceil(pull / 0.5)))
            seq, qq, e_max, t_max = [], q_rel, 0.0, 0.0
            for k in range(1, n + 1):
                f = k / n
                qq, e, tilt = kin.ik(p0 - f * pull / 100.0 * radial + [0, 0, f * rise / 100.0], qq)
                e_max, t_max = max(e_max, e), max(t_max, tilt)
                seq.append(qq)
            q = arm_step("retreat: pull the open gripper %.0f cm straight back toward the base (rising %.1f cm), out "
                         "from around the stacked cube" % (pull, rise), seq, q_rel, dt)
            q = arm_step("retreat: back to the start pose", interp(np.array(q), q_start), q, 0.05)
            if e_max <= 0.003 and not retreat_contacts(env, end_qpos, steps[n_before:]):
                break
        worst["ik_err"], worst["tilt"] = max(worst["ik_err"], e_max), max(worst["tilt"], t_max)
    for st in steps:
        st["optional"] = st["label"].startswith("retreat")

    if name == "Stack":
        hit = retreat_contacts(env, end_qpos, steps)
        if hit:
            warnings.append("the retreat brings the gripper within 5 mm of a cube in the simulated end state (%s)" % hit)
        else:
            print("retreat checked against the simulated end state: the gripper stays > 5 mm from both cubes")

    if worst["ik_err"] > 0.003:
        warnings.append("IK error up to %.1f mm" % (1000 * worst["ik_err"]))
    if worst["clear"] < min_clear - 0.0005:
        warnings.append("fingers come within %.1f mm of the table" % (1000 * worst["clear"]))

    print("\nwaypoints (robot base frame, cm; z above the table) and gripper commands:")
    for i in range(2):
        print("  %d: x %5.1f  y %5.1f  z %4.1f   gripper %s" % (i + 1, *((targets[i][:2] - axis) * 100),
                                                            100 * (targets[i][2] - env.table_offset[2] + z_off), grip[i]))
    print("plan: %d steps (%d optional), IK error <= %.1f mm, tool tilt <= %.1f deg, lowest finger point %.1f mm above "
          "the table, total %.0f s" % (len(steps), sum(s["optional"] for s in steps), 1000 * worst["ik_err"], worst["tilt"],
                                       1000 * worst["clear"],
                                       sum(s["points"][-1]["t"] for s in steps if s["type"] == "arm")))
    if worst["raised"]:
        print("raised %d simulated points by up to %.1f mm to keep the fingers %.0f mm above the table (in the simulation "
              "they touched it)" % (worst["raised"], 1000 * worst["raised_max"], args.min_clearance_mm))
    for w in warnings:
        print("WARNING: " + w)

    if args.out:
        plan = {"task": task_name, "run": run, "measured_cm": {k: measured[k].tolist() for k in need},
                "plate_cm": args.plate_cm, "z_offset_cm": args.z_offset_cm, "waypoints_base_cm": [
                    ((targets[i][:2] - axis) * 100).round(2).tolist() + [round(100 * (targets[i][2] - env.table_offset[2]), 2)]
                    for i in range(2)], "gripper": grip, "predicted": outcome, "warnings": warnings,
                "arm_joints": ARM_JOINTS, "expected_start": ROS_HOME, "steps": steps}
        out = os.path.abspath(os.path.join(HERE, "..", args.out) if not os.path.isabs(args.out) else args.out)
        json.dump(plan, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)  # atomic: run_trials.py waits for this file while the viewer opens
        print("\nplan written to %s" % out, flush=True)

    if args.view or args.view_loop:  # after writing the plan, so the plan can be used while the viewer is open
        frame = (axis, env.table_offset[2], [(k, tuple(measured[k]), areas[k]) for k in need])
        view(env, kin, steps, task, traj, targets, cubes_world, name, start_q, loop=args.view_loop, frame=frame)


# ------------------------------------------------------------------------------------------- measuring aids in viewer
def add_base_frame(scn, axis, table_z, cubes, x_len_cm=26, y_half_cm=15):
    """draw the robot base frame used for measuring into a MuJoCo scene (viewer.user_scn or a Renderer's scene):
    origin on joint 1's axis at table height, x (red) straight out along the arm at joint1 = 0, y (green) to the
    robot's left; ticks every cm, labelled every 5 cm; per cube (label, (x, y) cm, ((x0, x1), (y0, y1))
    trained area cm or None): the trained area (yellow), the target cross, guide lines to the axes and its coordinates"""
    z0 = table_z + 0.0008  # just above the table surface

    def w(x_cm, y_cm, dz=0.0):  # base frame (cm) -> world (m)
        return np.array([axis[0] + x_cm / 100.0, axis[1] + y_cm / 100.0, z0 + dz])

    def geom(gtype, rgba, label=""):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        g.label = label
        scn.ngeom += 1
        return g

    def line(a, b, rgba, width=0.0012, gtype=mujoco.mjtGeom.mjGEOM_CAPSULE, label=""):
        g = geom(gtype, rgba, label)
        if g is not None:
            mujoco.mjv_connector(g, gtype, width, a, b)

    def text(pos, label, rgba=(1, 1, 1, 1)):
        g = geom(mujoco.mjtGeom.mjGEOM_SPHERE, rgba, label)
        if g is not None:
            g.size[:] = [0.0015, 0, 0]
            g.pos[:] = pos

    RED, GREEN, YELLOW, CUBE = (0.9, 0.1, 0.1, 1), (0.1, 0.75, 0.1, 1), (1, 0.85, 0, 1), (1, 0.2, 0.9, 1)
    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    line(w(0, 0), w(x_len_cm, 0), RED, 0.002, arrow)
    line(w(0, -y_half_cm), w(0, y_half_cm), GREEN, 0.0015)
    line(w(0, 0), w(0, y_half_cm + 1), GREEN, 0.002, arrow)
    text(w(x_len_cm + 2, 0), "x (cm)", RED)
    text(w(0, y_half_cm + 6), "y (cm, robot's left)", GREEN)
    text(w(-1, -1), "0 (joint 1 axis)")
    for c in range(1, x_len_cm):
        big = c % 5 == 0
        h = 0.6 if big else 0.3
        line(w(c, -h), w(c, h), RED, 0.0009 if big else 0.0006)
        if big:
            text(w(c, -1.6), "%d" % c, RED)
    for c in range(-y_half_cm, y_half_cm + 1):
        if c == 0:
            continue
        big = c % 5 == 0
        h = 0.6 if big else 0.3
        line(w(-h, c), w(h, c), GREEN, 0.0009 if big else 0.0006)
        if big:
            text(w(-1.8, c), "%+d" % c, GREEN)
    for label, (cx, cy), area in cubes:
        if area is not None:
            (x0, x1), (y0, y1) = area
            for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                line(w(*a), w(*b), YELLOW, 0.0008)
            text(w(x0, y0 - 9), "trained area" if len(cubes) == 1 else "trained area " + label, YELLOW)
        line(w(cx, 0), w(cx, cy), CUBE, 0.0006)  # guide lines: x along the x axis, y across
        line(w(0, cy), w(cx, cy), CUBE, 0.0006)
        line(w(cx - 1.5, cy), w(cx + 1.5, cy), CUBE, 0.0009)
        line(w(cx, cy - 1.5), w(cx, cy + 1.5), CUBE, 0.0009)
        text(w(cx, cy, 0.075), "%s (x %.1f, y %+.1f) cm" % (label, cx, cy), CUBE)


def view(env, kin, steps, task, traj, targets, cubes_world, name, start_q, loop=False, frame=None):
    """1) the robot plan (kinematic playback, cubes in place), 2) the simulated episode; with loop, again and again.
    Close the window to go on."""
    import mujoco.viewer
    print("\nviewer: first the planned robot motion, then the simulated episode%s (close the window to finish)" % (
        ", repeated" if loop else ""), flush=True)
    with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (-0.06, 0.04, 0.86), 0.65, 145, -25
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            if frame is not None:  # measuring aids: robot base frame, cm ticks, trained areas, cube targets
                add_base_frame(viewer.user_scn, *frame)
        _view_once(env, viewer, steps, task, traj, targets, cubes_world, name)
        while loop and viewer.is_running():
            time.sleep(1.5)
            _view_once(env, viewer, steps, task, traj, targets, cubes_world, name)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def _view_once(env, viewer, steps, task, traj, targets, cubes_world, name):
    """one playback of the robot plan, then the simulated episode"""
    env.reset()
    place_cubes(env, name, cubes_world)
    robot = env.robots[0]
    gq = robot._ref_gripper_joint_pos_indexes
    for s in steps:
        if not viewer.is_running():
            return
        if s["type"] == "gripper":  # kinematic playback: show the command, don't simulate contacts
            env.sim.data.qpos[gq] = [0.02, -0.02] if s["command"] == "open" else [0.005, -0.005]
            env.sim.forward()
            viewer.sync()
            time.sleep(0.5)
            continue
        t0 = 0.0
        for pt in s["points"]:
            env.sim.data.qpos[robot._ref_joint_pos_indexes] = pt["q"]
            env.sim.forward()
            viewer.sync()
            time.sleep(min(0.05, max(0.0, pt["t"] - t0) / 4))  # 4x faster than the robot
            t0 = pt["t"]
    time.sleep(1.0)
    if not viewer.is_running():
        return
    obs = env.reset()
    obs = place_cubes(env, name, cubes_world)
    rollout(env, task, traj, targets, obs, viewer)


if __name__ == "__main__":
    main()
