"""
Checks that the OpenManipulatorP robot works inside robosuite environments, built the same way rlwp/train.py does.

    MUJOCO_GL=egl python omp_tools/check_omp_env.py [--env Lift] [--out omp_tools/out]

Checks: placement (base / eef / object), contacts at reset, holding still under zero action, OSC response
direction, gripper open/close, and a scripted pick with rlwp-style P-control (action = 10 * (target - eef)).
"""
import argparse
import os

import mujoco
import numpy as np
import PIL.Image

import robosuite as suite
from robosuite.controllers import load_controller_config

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(env_name, controller="rlwp"):
    # Same arguments as rlwp/train.py
    cfg = load_controller_config(default_controller="OSC_POSE") if controller == "rlwp" else None
    return suite.make(
        env_name=env_name,
        robots="OpenManipulatorP",
        controller_configs=cfg,
        has_renderer=False,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        single_object_mode=2,
        object_type=None,
        use_latch=False,
    )


def contacts(env):
    m, d = env.sim.model, env.sim.data
    return sorted({(m.geom_id2name(c.geom1), m.geom_id2name(c.geom2)) for c in d.contact[: d.ncon]})


def render(env, path, lookat, azimuth, elevation, distance=1.0):
    with mujoco.Renderer(env.sim.model._model, 480, 640) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # hide collision geoms (robosuite recolours them), show the visual meshes
        r.update_scene(env.sim.data._data, camera=cam, scene_option=opt)
        PIL.Image.fromarray(r.render()).save(path)


def p_control(env, obs, target, grip, steps):
    """rlwp's tracking rule: OSC delta = 10 * (target - eef), orientation held, gripper +-1 / 0."""
    for _ in range(steps):
        a = np.r_[10.0 * (target - obs["robot0_eef_pos"]), 0, 0, 0, grip]
        obs, reward, _, _ = env.step(a)
    return obs, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Lift")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 0. The robot's own default controller config loads too
    make_env(args.env, controller="default").close()
    print("[make] default_open_manipulator_p controller config: ok")

    env = make_env(args.env)
    obs = env.reset()
    m, d = env.sim.model, env.sim.data
    base = d.body_xpos[m.body_name2id("robot0_base")].copy()
    eef = obs["robot0_eef_pos"].copy()
    obj_key = {"Lift": "cube_pos", "Stack": "cubeA_pos", "Door": "handle_pos"}.get(args.env)
    obj = obs[obj_key].copy()
    print("[make] %s with OpenManipulatorP: ok (action dim %d)" % (args.env, env.action_dim))
    print("[place] base %s | eef %s | %s %s" % (np.round(base, 3), np.round(eef, 3), obj_key, np.round(obj, 3)))
    print("[place] object: %.3f m horizontal from base, %.3f m below base; eef %.3f m above table" % (
        np.linalg.norm(obj[:2] - base[:2]), base[2] - obj[2], eef[2] - 0.8))
    print("[place] joints %s (init_qpos %s)" % (
        np.round(obs["robot0_joint_pos_cos"], 3) if False else np.round(env.robots[0]._joint_positions, 3),
        np.round(env.robots[0].init_qpos, 3)))
    print("[contacts] at reset: %s" % (contacts(env) or "none"))
    riser = m.geom_name2id("mount0_riser_collision")
    print("[place] riser box spans z %.3f .. %.3f (table top 0.8, base %.3f)" % (
        d.geom_xpos[riser][2] - m.geom_size[riser][2], d.geom_xpos[riser][2] + m.geom_size[riser][2], base[2]))

    render(env, os.path.join(args.out, "env_%s_reset_persp.png" % args.env), (base[0] + 0.25, 0, 0.9), 150, -25, 1.1)
    render(env, os.path.join(args.out, "env_%s_reset_side.png" % args.env), (base[0] + 0.25, 0, 0.9), 90, -5, 0.9)

    # 1. Hold still under zero action (gravity is compensated by OSC)
    for _ in range(30):
        obs, *_ = env.step(np.zeros(env.action_dim))
    drift = np.linalg.norm(obs["robot0_eef_pos"] - eef)
    print("[hold] eef drift after 30 zero-action steps (3 s): %.4f m -> %s" % (drift, "ok" if drift < 0.01 else "CHECK"))

    # 2. OSC direction response: 5 steps of +x, then +z (expected ~0.025 m/step per axis at action 0.5)
    for axis, name in [(0, "x"), (2, "z")]:
        p0 = obs["robot0_eef_pos"].copy()
        a = np.zeros(env.action_dim)
        a[axis] = 0.5
        for _ in range(5):
            obs, *_ = env.step(a)
        dp = obs["robot0_eef_pos"] - p0
        print("[osc] +%s x5: moved %s m -> %s" % (name, np.round(dp, 3), "ok" if dp[axis] > 0.05 and
                                                 np.abs(np.delete(dp, axis)).max() < 0.5 * dp[axis] else "CHECK"))

    # 3. Gripper close / open through the env action
    for g, name in [(1, "close"), (-1, "open")]:
        for _ in range(5):
            obs, *_ = env.step(np.r_[np.zeros(6), g])
        print("[gripper] %s: finger qpos %s" % (name, np.round(obs["robot0_gripper_qpos"], 4)))

    # 4. Scripted pick (Lift only): above cube -> descend -> close -> lift
    if args.env == "Lift":
        obs = env.reset()
        cube = obs["cube_pos"].copy()
        grasp = cube + [0, 0, 0.005]  # grip point slightly above the cube centre (fingertips 21 mm past it)
        obs, _ = p_control(env, obs, grasp + [0, 0, 0.08], -1, 15)
        obs, _ = p_control(env, obs, grasp, -1, 15)
        err = np.linalg.norm(obs["robot0_eef_pos"] - grasp)
        obs, _ = p_control(env, obs, grasp, 1, 10)
        grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube)
        render(env, os.path.join(args.out, "env_Lift_grasp.png"), (cube[0] - 0.1, 0, 0.88), 150, -20, 0.7)
        obs, reward = p_control(env, obs, grasp + [0, 0, 0.12], 1, 20)
        lifted = obs["cube_pos"][2] - cube[2]
        success = env._check_success()
        render(env, os.path.join(args.out, "env_Lift_lifted.png"), (cube[0] - 0.1, 0, 0.9), 150, -20, 0.8)
        print("[pick] reach error %.4f m, grasped=%s, cube lifted %.3f m, success=%s, last reward %.2f -> %s" % (
            err, grasped, lifted, success, reward, "ok" if success else "CHECK"))
    print("[render] PNGs written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
