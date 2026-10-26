"""
Interactive MuJoCo viewer for the OpenManipulatorP in a robosuite task, running a scripted pick in a loop.

    python omp_tools/demo_omp_viewer.py              # Lift, scripted pick, repeats with a new random cube
    python omp_tools/demo_omp_viewer.py --static     # just show the robot at its initial pose

Viewer: left-drag rotate, right-drag pan, scroll zoom, double-click a body to select, Ctrl+right-drag to push it.
Press Tab / Shift+Tab to hide the side panels. Close the window to quit.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Lift")
    ap.add_argument("--static", action="store_true")
    args = ap.parse_args()

    env = suite.make(
        env_name=args.env,
        robots="OpenManipulatorP",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=10,
        initialization_noise=None,
        hard_reset=False,  # keep the same MjModel/MjData across resets so the viewer stays attached
        single_object_mode=2,
        object_type=None,
        use_latch=False,
    )
    obs = env.reset()
    model, data = env.sim.model._model, env.sim.data._data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [-0.15, 0, 0.88]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 145, -22
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0  # hide collision geoms; press "0" in the viewer to toggle them back

        def step(action):
            t0 = time.time()
            o = env.step(action)[0]
            viewer.sync()
            time.sleep(max(0.0, env.control_timestep - (time.time() - t0)))  # real time
            return o

        if args.static:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return

        episode = 0
        while viewer.is_running():
            episode += 1
            cube = obs["cube_pos"].copy()
            grasp = cube + [0, 0, 0.005]
            # (target, gripper, steps): above -> descend -> close -> lift -> hold -> release
            plan = [(grasp + [0, 0, 0.08], -1, 15), (grasp, -1, 15), (grasp, 1, 10),
                    (grasp + [0, 0, 0.12], 1, 20), (grasp + [0, 0, 0.12], 1, 10), (grasp + [0, 0, 0.12], -1, 10)]
            success = False
            for target, grip, n in plan:
                for _ in range(n):
                    if not viewer.is_running():
                        return
                    obs = step(np.r_[10.0 * (target - obs["robot0_eef_pos"]), 0, 0, 0, grip])
                    success |= env._check_success()
            print("episode %d: cube at %s, lift success=%s" % (episode, np.round(cube[:2], 3), success), flush=True)
            obs = env.reset()
            viewer.sync()


if __name__ == "__main__":
    main()
