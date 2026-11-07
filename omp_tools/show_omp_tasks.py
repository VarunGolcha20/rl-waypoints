"""
Shows the rlwp task environments with the OpenManipulatorP: the robot holds its start pose (zero OSC action) while the
physics runs, so objects settle and you can push them around (Ctrl + right-drag). No learned or scripted motion.

    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py               # all tasks, one window after another
    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py --env Door    # one task
    MUJOCO_GL=egl python omp_tools/show_omp_tasks.py --snapshot                 # PNGs to omp_tools/out/, no window

In the viewer: close the window to go to the next task, Backspace resets the scene (new random object placement),
`0` toggles collision geoms, left-drag rotate, right-drag pan, scroll zoom.
"""
import argparse
import os
import time

import mujoco
import numpy as np
import PIL.Image

import robosuite as suite

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["Lift", "Stack", "Door", "NutAssembly", "PickPlace"]
# extra suite.make arguments per task, as rlwp passes them: PickPlace object (rlwp's `object`) + the env_kwargs of the
# OM-P task configs (rlwp/cfg/task/<Task>_omp.yaml), i.e. the adapted layouts
TASK_KWARGS = {"PickPlace": dict(object_type="can")}
RLWP_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rlwp", "cfg", "task")


def omp_env_kwargs(name):
    import yaml

    path = os.path.join(RLWP_TASKS, name + "_omp.yaml")
    return (yaml.safe_load(open(path))["env"].get("env_kwargs") or {}) if os.path.exists(path) else {}


# camera (lookat, distance, azimuth, elevation) per task
CAMERAS = {"Door": ((-0.12, -0.02, 0.87), 1.1, 225, -25)}  # from the robot side of the door
DEFAULT_CAMERA = ((-0.12, 0.0, 0.85), 1.1, 145, -22)


def make_env(name):
    kwargs = dict(single_object_mode=2, object_type=None, use_latch=False)
    kwargs.update(TASK_KWARGS.get(name, {}))
    kwargs.update(omp_env_kwargs(name))
    return suite.make(name, robots="OpenManipulatorP", has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False, initialization_noise=None, control_freq=10, hard_reset=False, **kwargs)


def set_camera(cam, name):
    lookat, dist, az, el = CAMERAS.get(name, DEFAULT_CAMERA)
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el


def snapshot(env, name, out):
    for _ in range(10):  # let objects settle
        env.step(np.zeros(env.action_dim))
    with mujoco.Renderer(env.sim.model._model, 480, 640) as r:
        cam = mujoco.MjvCamera()
        set_camera(cam, name)
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # visual meshes only
        r.update_scene(env.sim.data._data, camera=cam, scene_option=opt)
        path = os.path.join(out, "task_%s.png" % name)
        PIL.Image.fromarray(r.render()).save(path)
    return path


def view(env, name):
    import mujoco.viewer

    model, data = env.sim.model._model, env.sim.data._data
    print("%s: close the window for the next task" % name, flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer.cam, name)
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            t0 = time.time()
            env.step(np.zeros(env.action_dim))  # OSC holds the start pose; physics keeps running
            viewer.sync()
            time.sleep(max(0.0, env.control_timestep - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=TASKS, help="one task (default: all)")
    ap.add_argument("--snapshot", action="store_true", help="save PNGs instead of opening the viewer")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    for name in [args.env] if args.env else TASKS:
        env = make_env(name)
        env.reset()
        if args.snapshot:
            os.makedirs(args.out, exist_ok=True)
            print(name, "->", snapshot(env, name, args.out))
        else:
            view(env, name)
        env.close()


if __name__ == "__main__":
    main()
