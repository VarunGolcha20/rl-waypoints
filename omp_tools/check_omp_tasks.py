"""
Is each OM-P task solvable with rlwp's 2-waypoint scheme? For every task this builds the env exactly as rlwp does
(from rlwp/cfg/task/<Task>_omp.yaml: controller, env_kwargs, wp_steps, gripper_steps, orientation hold) and executes a
hand-designed pair of waypoints with rlwp's own action rule (50-step segments, gripper open for the first 10 steps,
gripper command only in the last gripper_steps of each segment). Reports success rate, whether the scripted waypoints
lie inside the task's workspace box (they are clipped into it, as rlwp's optimiser is; "clipped" counts episodes where
that moved a waypoint by more than 5 mm), and the worst tool tilt.

    MUJOCO_GL=egl python omp_tools/check_omp_tasks.py [--tasks Lift Stack ...] [--episodes 15]

A pass means the task layout and the waypoint parametrisation work on this robot. It says nothing about how fast
rlwp will learn those waypoints.
"""
import argparse
import os
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
RLWP = os.path.join(HERE, "..", "rlwp")
sys.path.insert(0, RLWP)
from common import get_controller_config, get_env_kwargs, orientation_hold_action  # noqa: E402

import robosuite as suite  # noqa: E402

OBJECT = {"PickPlace": "can"}  # rlwp's top-level `object` for PickPlace


# --- hand-designed waypoints: (target xyz, gripper command) x 2, from settled object positions -------------------
def plan_lift(env, obs):
    c = obs["cube_pos"]
    return [(c + [0, 0, 0.005], 1.0), (c + [0, 0, 0.05], 1.0)]


def plan_stack(env, obs):
    a, b = obs["cubeA_pos"], obs["cubeB_pos"]
    over_b = np.r_[b[:2], 0.905]  # as high as the workspace allows: the straight carry path must clear cubeB
    return [(a + [0, 0, 0.005], 1.0), (over_b, -1.0)]


def plan_nut(env, obs):
    nut = obs["RoundNut_pos"]
    table = env.table_offset[2]
    peg = env.sim.data.body_xpos[env.peg2_body_id]
    # grip the handle bar 7 cm out from the nut centre (the bar spans 5.4-8.5 cm; at the 6 cm handle site the open
    # fingers clip the ring's octagon corners), fingertips just above the table
    to_handle = 0.07 * np.array([np.cos(env.nut_rotation), np.sin(env.nut_rotation)])
    handle = np.r_[nut[:2] + to_handle, table + 0.025]
    over_peg = np.r_[peg[:2] + to_handle, table + 0.085]  # nut bottom ~5 cm over the table
    return [(handle, 1.0), (over_peg, -1.0)]


def plan_pickplace(env, obs):
    can = obs["can_pos"]
    target = env.target_bin_placements[env.object_to_id["can"]]
    # grasp low on the can (its bottom then rises sooner on the straight carry), release high over the compartment
    return [(can + [0, 0, -0.015], 1.0), (np.r_[target[:2], 0.905], -1.0)]


def plan_door(env, obs):
    # grasp the handle bar (fingers straddle it), then pull it 0.35 rad along the door's opening arc
    h = obs["handle_pos"]
    hinge = env.sim.data.xanchor[env.sim.model.joint_name2id(env.door.joints[0])][:2]
    a = 0.35  # success: hinge > 0.3 rad; the door opens counter-clockwise about +z
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return [(h, 1.0), (np.r_[hinge + rot @ (h[:2] - hinge), h[2]], 1.0)]


PLANS = {"Lift": plan_lift, "Stack": plan_stack, "NutAssembly": plan_nut, "PickPlace": plan_pickplace,
         "Door": plan_door}


def load_task(name):
    cfg = yaml.safe_load(open(os.path.join(RLWP, "cfg", "task", name + "_omp.yaml")))
    return cfg["name"], cfg["env"]


def run(task, episodes, seed):
    env_name, env_cfg = load_task(task)
    env = suite.make(env_name, robots=env_cfg["robot"], controller_configs=get_controller_config(env_cfg),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True,
                     control_freq=10, initialization_noise=None, hard_reset=False,
                     **get_env_kwargs(env_cfg, OBJECT.get(env_name), env_cfg["use_latch"]))
    wp_steps, gripper_steps = env_cfg["wp_steps"], env_cfg["gripper_steps"]
    ws = env_cfg.get("workspace")
    np.random.seed(seed)
    ok, outside, tilt = 0, 0, 0.0
    for _ in range(episodes):
        env.reset()
        for _ in range(3):  # let objects settle before reading their positions
            obs = env.step(np.zeros(env.action_dim))[0]
        wps = PLANS[task](env, obs)
        if ws is not None:
            clipped = [(np.clip(w, ws["low"], ws["high"]), g) for w, g in wps]
            outside += any(np.linalg.norm(w - c) > 0.006 for (w, _), (c, _) in zip(wps, clipped))
            wps = clipped
        ori_goal = R.from_quat(obs["robot0_eef_quat"])
        success, time_s = False, 0
        for t in range(2 * wp_steps):  # rlwp's get_action
            i = t // wp_steps
            err = wps[i][0] - obs["robot0_eef_pos"]
            if t < 10:
                a = np.r_[10 * err, 0, 0, 0, -1.0]
            elif time_s >= wp_steps - gripper_steps:
                a = np.r_[np.zeros(6), wps[i][1]]
            else:
                a = np.r_[10 * err, 0, 0, 0, 0.0]
            if env_cfg.get("orientation_hold"):
                a[3:6] = orientation_hold_action(obs, ori_goal)
            time_s = time_s + 1 if time_s + 1 < wp_steps else 1
            obs = env.step(a)[0]
            success = success or env._check_success()
            Rm = R.from_quat(obs["robot0_eef_quat"]).as_matrix()
            tilt = max(tilt, np.degrees(np.arccos(np.clip(-Rm[2, 2], -1, 1))))
        ok += success
    env.close()
    return ok, outside, tilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(PLANS))
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for task in args.tasks:
        ok, outside, tilt = run(task, args.episodes, args.seed)
        print("%-12s scripted 2-waypoint success %2d/%d | episodes with a waypoint clipped > 5 mm by the workspace box: %d | "
              "max tool tilt %.1f deg" % (task, ok, args.episodes, outside, tilt), flush=True)


if __name__ == "__main__":
    main()
