"""
Checks that the OpenManipulatorX robot works inside a robosuite environment built the way rlwp/train.py builds it, and
compares OSC controller settings for this 4-DoF arm.

    MUJOCO_GL=egl python omp_tools/check_omx_env.py [--targets 40] [--out omp_tools/out]

Checks: placement (base / eef / cube), contacts at reset, holding still, OSC step response, gripper open/close,
waypoint tracking with rlwp's rule (action = 10 * (target - eef), 50 steps) under several controller settings, and a
scripted pick of a 55 mm cube with the robot's own controller config.

Why the controller comparison: the arm has 4 joints (joint1 = yaw, joints 2-4 = pitch). It can point the gripper
down anywhere in reach, but the gripper's yaw is fixed by where the target is (atan2(y, x) from the base). An OSC
that also tries to hold a yaw fights joint1 whenever the target is off to the side.
"""
import argparse
import copy
import os

import mujoco
import numpy as np
import PIL.Image
from scipy.spatial.transform import Rotation as R

import robosuite as suite
from robosuite.controllers import load_controller_config

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(suite.__file__), "controllers", "config")
CUBE = 0.055  # the user's Lift cube (m)
TABLE = 0.8


def controller_cfg(kp=None, uncouple=None):
    cfg = load_controller_config(custom_fpath=os.path.join(CFG_DIR, "default_open_manipulator_x.json"))
    cfg = copy.deepcopy(cfg)
    if kp is not None:
        cfg["kp"] = kp
    if uncouple is not None:
        cfg["uncouple_pos_ori"] = uncouple
    return cfg


def make_env(cfg=None):
    # Same arguments as rlwp/train.py (+ the 55 mm cube). cfg None = the robot's own default_open_manipulator_x
    return suite.make(
        env_name="Lift",
        robots="OpenManipulatorX",
        controller_configs=cfg,
        has_renderer=False,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        hard_reset=False,
        cube_full_size=[CUBE] * 3,
    )


def contacts(env):
    m, d = env.sim.model, env.sim.data
    return sorted({(m.geom_id2name(c.geom1), m.geom_id2name(c.geom2)) for c in d.contact[: d.ncon]})


def render(env, path, lookat, azimuth, elevation, distance=0.6):
    with mujoco.Renderer(env.sim.model._model, 480, 640) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # hide collision geoms, show the visual meshes
        r.update_scene(env.sim.data._data, camera=cam, scene_option=opt)
        PIL.Image.fromarray(r.render()).save(path)


def tool_tilt_deg(obs):
    """angle between the approach axis (grip site +z) and straight down"""
    z = R.from_quat(obs["robot0_eef_quat"]).as_matrix()[:, 2]
    return np.degrees(np.arccos(np.clip(-z[2], -1, 1)))


def rot_action(obs, mode, start_rot):
    """rotation part of the OSC delta action (in [-1, 1], 0.5 rad per unit)"""
    if mode == "zero":
        return np.zeros(3)
    cur = R.from_quat(obs["robot0_eef_quat"])
    if mode == "hold_full":  # rlwp's orientation_hold_action: back to the full start orientation (incl. yaw)
        rv = (start_rot * cur.inv()).as_rotvec()
    else:  # "hold_tilt": only turn the approach axis back to straight down, yaw left free
        a = cur.as_matrix()[:, 2]
        axis = np.cross(a, [0, 0, -1.0])
        s, c = np.linalg.norm(axis), -a[2]
        rv = np.zeros(3) if s < 1e-9 else axis / s * np.arctan2(s, c)
    return np.clip(rv / 0.5, -1, 1)


def park_cube(env):
    """move the cube out of the way (targets may pass through its spawn area)"""
    env.sim.data.set_joint_qpos(env.cube.joints[0], np.r_[0.25, 0.3, TABLE + CUBE / 2, 1, 0, 0, 0])
    env.sim.forward()


def track(env, target, mode, steps=50):
    """rlwp's rule for one waypoint: P-control on the grip point, gripper open for the first 10 steps"""
    obs = env.reset()
    park_cube(env)
    start_rot = R.from_quat(obs["robot0_eef_quat"])
    rob = env.robots[0]
    lim = rob.sim.model.jnt_range[rob._ref_joint_indexes]
    peak_tau = 0.0
    for t in range(steps):
        a = np.r_[10.0 * (target - obs["robot0_eef_pos"]), rot_action(obs, mode, start_rot), -1.0 if t < 10 else 0.0]
        obs, *_ = env.step(a)
        peak_tau = max(peak_tau, np.max(np.abs(rob.torques) / 4.1))
    q = np.array(rob._joint_positions)
    at_limit = np.any(np.minimum(q - lim[:, 0], lim[:, 1] - q) < 0.02)
    return np.linalg.norm(obs["robot0_eef_pos"] - target), tool_tilt_deg(obs), at_limit, peak_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=40, help="random targets per controller setting")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env()
    obs = env.reset()
    m, d = env.sim.model, env.sim.data
    base = d.body_xpos[m.body_name2id("robot0_base")].copy()
    eef, cube = obs["robot0_eef_pos"].copy(), obs["cube_pos"].copy()
    rob = env.robots[0]
    print("[make] Lift with OpenManipulatorX: ok (action dim %d, gripper %s, controller kp %s, uncoupled %s)" % (
        env.action_dim, type(rob.gripper).__name__, rob.controller.kp, rob.controller.uncoupling))
    print("[place] base %s | eef %s | cube %s" % (np.round(base, 3), np.round(eef, 3), np.round(cube, 3)))
    print("[place] cube %.3f m from the base axis; eef %.3f m in front of the base, %.3f m above the table; tool tilt %.2f deg"
          % (np.linalg.norm(cube[:2] - base[:2]), eef[0] - base[0], eef[2] - TABLE, tool_tilt_deg(obs)))
    print("[contacts] at reset: %s" % (contacts(env) or "none"))
    render(env, os.path.join(args.out, "omx_Lift_reset_persp.png"), (base[0] + 0.12, 0, 0.88), 150, -25, 0.6)
    render(env, os.path.join(args.out, "omx_Lift_reset_side.png"), (base[0] + 0.12, 0, 0.88), 90, -5, 0.55)

    # 1. Hold still
    for _ in range(30):
        obs, *_ = env.step(np.zeros(env.action_dim))
    print("[hold] eef drift after 30 zero-action steps: %.4f m" % np.linalg.norm(obs["robot0_eef_pos"] - eef))

    # 2. OSC step response: 5 steps of 0.5 per axis (commanded 0.025 m/step; Panda moves ~0.05 m, OM-P 0.050 / 0.053)
    for axis, name in [(0, "+x"), (1, "+y"), (2, "+z")]:
        obs = env.reset()
        park_cube(env)
        p0 = obs["robot0_eef_pos"].copy()
        a = np.zeros(env.action_dim)
        a[axis] = 0.5
        for _ in range(5):
            obs, *_ = env.step(a)
        print("[osc] %s x5: moved %s m, tool tilt %.1f deg" % (name, np.round(obs["robot0_eef_pos"] - p0, 3),
                                                               tool_tilt_deg(obs)))

    # 3. Gripper
    for g, name in [(1, "close"), (-1, "open")]:
        for _ in range(5):
            obs, *_ = env.step(np.r_[np.zeros(6), g])
        print("[gripper] %s: finger qpos %s" % (name, np.round(obs["robot0_gripper_qpos"], 4)))
    env.close()

    # 4. Controller comparison: random targets in the tool-down reach band, around and beside the object area
    rng = np.random.default_rng(0)
    targets = []
    while len(targets) < args.targets:
        r, th, h = rng.uniform(0.10, 0.20), np.radians(rng.uniform(-40, 40)), rng.uniform(0.025, 0.09)
        targets.append(base + [r * np.cos(th), r * np.sin(th), 0] + [0, 0, TABLE + h - base[2]])
    variants = [
        ("coupled, kp 150, rot 0 (OM-P setting)", controller_cfg(kp=150, uncouple=False), "zero"),
        ("coupled, kp 150, hold full ori", controller_cfg(kp=150, uncouple=False), "hold_full"),
        ("coupled, kp 150 / yaw 0, rot 0", controller_cfg(kp=[150] * 5 + [0], uncouple=False), "zero"),
        ("uncoupled, kp 150, rot 0", controller_cfg(kp=150), "zero"),
        ("uncoupled, kp 150 / yaw 0, rot 0", controller_cfg(kp=[150] * 5 + [0]), "zero"),
        ("uncoupled, kp 300 / yaw 0, rot 0 (default)", controller_cfg(), "zero"),
        ("uncoupled, kp 300 / yaw 0, hold full ori", controller_cfg(), "hold_full"),
        ("uncoupled, kp 300 / yaw 0, hold tilt", controller_cfg(), "hold_tilt"),
    ]
    print("[ctrl] %d random targets (0.10-0.20 m from the base axis, +-40 deg, 2.5-9 cm above the table), 50 steps "
          "of rlwp's rule each:" % len(targets))
    print("       %-42s %9s %9s %8s %9s %9s %9s" % ("setting", "med err", "max err", ">1 cm", "max tilt", "at limit", "peak tau"))
    for name, cfg, mode in variants:
        e = make_env(cfg)
        res = np.array([track(e, t, mode) for t in targets], dtype=float)
        e.close()
        print("       %-42s %7.1f mm %6.1f mm %7.0f %% %7.1f deg %7.0f %% %8.0f %%" % (
            name, 1000 * np.median(res[:, 0]), 1000 * res[:, 0].max(), 100 * np.mean(res[:, 0] > 0.01),
            res[:, 1].max(), 100 * res[:, 2].mean(), 100 * res[:, 3].max()))

    # 5. Scripted pick of the 55 mm cube with the robot's own controller config (rlwp-style P-control, no rotation)
    env = make_env()
    rob = env.robots[0]
    results = []
    for ep in range(10):
        obs = env.reset()
        for _ in range(3):  # let the cube settle (robosuite drops objects from 1 cm)
            obs, *_ = env.step(np.r_[np.zeros(6), -1])
        cube = obs["cube_pos"].copy()
        grasp = cube + [0, 0, 0.0]
        peak = 0.0

        def go(target, g, n):
            nonlocal obs, peak
            for _ in range(n):
                obs, *_ = env.step(np.r_[10.0 * (target - obs["robot0_eef_pos"]), 0, 0, 0, g])
                peak = max(peak, np.max(np.abs(rob.torques) / 4.1))

        go(grasp + [0, 0, 0.06], -1, 15)
        go(grasp, -1, 20)
        err = np.linalg.norm(obs["robot0_eef_pos"] - grasp)
        go(grasp, 1, 10)
        grasped = env._check_grasp(gripper=rob.gripper, object_geoms=env.cube)
        if ep == 0:
            render(env, os.path.join(args.out, "omx_Lift_grasp.png"), (cube[0] - 0.05, 0, 0.86), 150, -20, 0.5)
        go(grasp + [0, 0, 0.06], 1, 20)
        lifted = obs["cube_pos"][2] - cube[2]
        if ep == 0:
            render(env, os.path.join(args.out, "omx_Lift_lifted.png"), (cube[0] - 0.05, 0, 0.88), 150, -20, 0.5)
        results.append((env._check_success(), grasped, lifted, err, tool_tilt_deg(obs), peak))
        print("[pick] ep %d: cube %s, reach error %.4f m, grasped %s, lifted %.3f m, success %s, tilt %.1f deg, peak "
              "|tau|/limit %.2f" % (ep, np.round(cube, 3), err, grasped, lifted, env._check_success(),
                                    tool_tilt_deg(obs), peak))
    print("[pick] success %d/%d" % (sum(r[0] for r in results), len(results)))
    print("[render] PNGs written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
