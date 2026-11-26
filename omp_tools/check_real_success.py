"""
Robosuite's success flag vs what physically happened, for trained rlwp models (Lift / Stack). Each episode picks the
waypoints like rlwp's evaluation (frozen reward models, no exploration) and executes them with rlwp's action rule.

    MUJOCO_GL=egl python omp_tools/check_real_success.py --task Lift_omx --runs omx_lift_s0 omx_lift_s1 omx_lift_s2
    MUJOCO_GL=egl python omp_tools/check_real_success.py --task Stack_omx --runs omx_stack_s1 --episodes 50

real success:
  Lift  - the cube's centre rose >= 4 cm above its resting height at some step. robosuite only asks for the centre to
          be > table + 4 cm, which a 55 mm cube passes after 1.25 cm (a bump can do it).
  Stack - at the end of the episode cubeA rests on cubeB (bottom within 6 mm of cubeB's top, centre over cubeB) and is
          not held. robosuite counts the moment cubeA lands, even if it then slides off.
The verdicts are the same as the coloured marker in omp_tools/view_policy.py.
"""
import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from view_policy import OBJECTS, lifted_for_real, stacked_for_real  # noqa: E402  (also puts rlwp/ on sys.path, cwd = rlwp/)
from common import get_controller_config, get_env_kwargs, get_save_name, get_workspace, lift_first_error  # noqa: E402
from common import policy_objs, run_turn, waypoint_targets  # noqa: E402
from eval import Method  # noqa: E402

import robosuite as suite  # noqa: E402


def check(task_name, run, episodes, seed):
    task = yaml.safe_load(open(os.path.join("cfg", "task", task_name + ".yaml")))
    name, ec = task["name"], task["env"]
    assert name in ("Lift", "Stack"), "only Lift / Stack have a real-success criterion"
    env = suite.make(name, robots=ec["robot"], controller_configs=get_controller_config(ec), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True, control_freq=10,
                     initialization_noise=None, hard_reset=False, **get_env_kwargs(ec, None, ec["use_latch"]))
    get_objs = lambda o: policy_objs(ec, np.concatenate([o[k] for k in OBJECTS[name]]), o)  # reward-model input
    np.random.seed(seed)
    obs = env.reset()
    start = obs["robot0_eef_pos"].copy()
    num_wp = task["task"].get("num_wp", 2)
    agent = Method(state_dim=ec["state_dim"], objs=get_objs(obs), wp_id=num_wp,
                   save_name=get_save_name(name, ec["robot"], run, None, ec["use_latch"]),
                   config={"task": task, "n_inits": 5}, workspace=get_workspace(ec, start))
    wp_steps, gripper_steps = ec["wp_steps"], ec["gripper_steps"]
    counts = {"real": 0, "false": 0, "fail": 0}
    first_steps = []
    for ep in range(episodes):
        obs = env.reset()
        obs = run_turn(env, ec, obs)  # object-relative configs: turn toward the first object first
        traj = agent.traj_opt(ep, get_objs(obs)).reshape(num_wp, ec["state_dim"])
        targets = waypoint_targets(ec, traj[:, :3], start, obs)
        time_s, success, real, first = 0, False, False, None
        for t in range(num_wp * wp_steps):  # rlwp's get_action (position-only configs)
            i = t // wp_steps
            err = lift_first_error(targets[i] - obs["robot0_eef_pos"], i, ec.get("lift_first", False))
            if t < 10:
                a = np.r_[10 * err, 0, 0, 0, -1.0]
            elif time_s >= wp_steps - gripper_steps:
                a = np.r_[np.zeros(6), traj[i, 3:]]
            else:
                a = np.r_[10 * err, 0, 0, 0, 0.0]
            time_s = time_s + 1 if time_s + 1 < wp_steps else 1
            obs = env.step(a)[0]
            if env._check_success() and not success:
                success, first = True, t
            if name == "Lift" and lifted_for_real(env, obs):
                real = True
        if name == "Stack":
            real = stacked_for_real(env, obs)
        counts["real" if success and real else "false" if success else "fail"] += 1
        if success:
            first_steps.append(first)
    env.close()
    n = episodes
    print("%-10s %-32s robosuite success %3d%% | real %3d%% | false %3d%% | robosuite's first-success steps %s" % (
        task_name, run, round(100 * (counts["real"] + counts["false"]) / n), round(100 * counts["real"] / n),
        round(100 * counts["false"] / n), sorted(first_steps)), flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="rlwp task config, e.g. Lift_omx")
    ap.add_argument("--runs", nargs="+", required=True, help="run names, e.g. omx_lift_s0 omx_lift_s1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="placement seed (1000 = rlwp eval seed of seed-0 runs)")
    args = ap.parse_args()
    for run in args.runs:
        check(args.task, run, args.episodes, args.seed)


if __name__ == "__main__":
    main()
