"""
Watch a trained (or still training) rlwp model in MuJoCo's viewer: each episode picks the waypoints with the saved
reward models (as rlwp's evaluation does, no exploration noise) and executes them with rlwp's action rule.
Read-only: loads rlwp/models/..., never writes.

    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Stack_omp --run omp_stack_30mm_s2
    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omp --run omp_lift_30mm_s0

Prints one line per episode (grasp / lift / success). Close the window to quit.
--force-open replaces the last waypoint's gripper command with "open", to see where a release would land.
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
RLWP = os.path.join(HERE, "..", "rlwp")
sys.path.insert(0, RLWP)
os.chdir(RLWP)  # rlwp loads models from ./models
from common import get_controller_config, get_env_kwargs, get_save_name, get_workspace, orientation_hold_action  # noqa: E402
from eval import Method  # noqa: E402

import robosuite as suite  # noqa: E402

OBJECTS = {"Lift": ["cube_pos"], "Stack": ["cubeA_pos", "cubeB_pos"], "NutAssembly": ["RoundNut_pos"],
           "PickPlace": ["can_pos"], "Door": ["door_pos", "handle_pos"]}
GRASP_TARGET = {"Lift": "cube", "Stack": "cubeA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Stack_omp", help="rlwp task config (cfg/task/<task>.yaml)")
    ap.add_argument("--run", required=True, help="run name, e.g. omp_stack_30mm_s2")
    ap.add_argument("--object", default="can", help="PickPlace object")
    ap.add_argument("--force-open", action="store_true",
                    help="override the last waypoint's gripper command with 'open' (what if it released there?)")
    args = ap.parse_args()

    task = yaml.safe_load(open(os.path.join("cfg", "task", args.task + ".yaml")))
    name, ec = task["name"], task["env"]
    obj = args.object if name == "PickPlace" else None
    env = suite.make(name, robots=ec["robot"], controller_configs=get_controller_config(ec), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True, control_freq=10,
                     initialization_noise=None, hard_reset=False, **get_env_kwargs(ec, obj, ec["use_latch"]))
    obs = env.reset()
    get_objs = lambda o: np.concatenate([o[k] for k in OBJECTS[name]])
    state = lambda o: o["robot0_eef_pos"]
    save_name = get_save_name(name, ec["robot"], args.run, obj, ec["use_latch"])
    agent = Method(state_dim=ec["state_dim"], objs=get_objs(obs), wp_id=2, save_name=save_name,
                   config={"task": task, "n_inits": 5}, workspace=get_workspace(ec, state(obs)))
    wp_steps, gripper_steps = ec["wp_steps"], ec["gripper_steps"]
    print("loaded models/%s" % save_name, flush=True)

    with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (-0.12, 0.05, 0.85), 1.0, 145, -25
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
        episode = 0
        while viewer.is_running():
            episode += 1
            traj = agent.traj_opt(episode, get_objs(obs)).reshape(2, ec["state_dim"])
            if args.force_open:
                traj[-1, 3] = -1.0
            targets = traj[:, :3] + state(obs)
            goal = R.from_quat(obs["robot0_eef_quat"])
            time_s, grasped, success, reward = 0, False, False, 0.0
            for t in range(2 * wp_steps):
                if not viewer.is_running():
                    return
                t0 = time.time()
                i = t // wp_steps
                err = targets[i] - state(obs)
                if t < 10:
                    a = np.r_[10 * err, 0, 0, 0, -1.0]
                elif time_s >= wp_steps - gripper_steps:
                    a = np.r_[np.zeros(6), traj[i, 3:]]
                else:
                    a = np.r_[10 * err, 0, 0, 0, 0.0]
                if ec.get("orientation_hold"):
                    a[3:6] = orientation_hold_action(obs, goal)
                time_s = time_s + 1 if time_s + 1 < wp_steps else 1
                obs, r, _, _ = env.step(a)
                reward += r
                success |= env._check_success()
                if t == wp_steps - 1 and name in GRASP_TARGET:
                    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=getattr(env, GRASP_TARGET[name]))
                viewer.sync()
                time.sleep(max(0.0, env.control_timestep - (time.time() - t0)))
            print("episode %d: waypoints %s, gripper %s | grasped after wp1: %s | success: %s | reward %.1f" % (
                episode, np.round(targets, 3).tolist(), np.round(traj[:, 3], 2).tolist(), grasped, success, reward), flush=True)
            obs = env.reset()
            viewer.sync()


if __name__ == "__main__":
    main()
