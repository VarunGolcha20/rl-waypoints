"""
Shows the rlwp task environments with the OpenManipulatorP (default) or, with --robot omx, the OpenManipulatorX: the
robot holds its start pose (zero OSC action, gripper open) while the physics runs, so objects settle and you can push
them around (Ctrl + right-drag). No learned or scripted motion.

    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py --robot omx   # the OM-X tasks (<Task>_omx.yaml)

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


ROBOTS = {"omp": "OpenManipulatorP", "omx": "OpenManipulatorX"}


def task_yaml(name, robot):
    return os.path.join(RLWP_TASKS, "%s_%s.yaml" % (name, robot))


def omp_env_kwargs(name, robot="omp"):
    import yaml

    path = task_yaml(name, robot)
    return (yaml.safe_load(open(path))["env"].get("env_kwargs") or {}) if os.path.exists(path) else {}


# camera (lookat, distance, azimuth, elevation) per task
CAMERAS = {"Door": ((-0.12, -0.02, 0.87), 1.1, 225, -25)}  # from the robot side of the door
DEFAULT_CAMERA = ((-0.12, 0.0, 0.85), 1.1, 145, -22)
OMX_CAMERA = ((-0.06, 0.04, 0.85), 0.7, 145, -25)  # OM-X: smaller robot, base at x = -0.17


def make_env(name, robot="omp"):
    kwargs = dict(single_object_mode=2, object_type=None, use_latch=False)
    kwargs.update(TASK_KWARGS.get(name, {}))
    kwargs.update(omp_env_kwargs(name, robot))
    return suite.make(name, robots=ROBOTS[robot], has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False, initialization_noise=None, control_freq=10, hard_reset=False, **kwargs)


def set_camera(cam, name, robot="omp"):
    lookat, dist, az, el = OMX_CAMERA if robot == "omx" else CAMERAS.get(name, DEFAULT_CAMERA)
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el


def hold(env):
    """zero OSC action (hold the start pose), gripper open as rlwp keeps it at the start of an episode"""
    return np.r_[np.zeros(env.action_dim - 1), -1.0]


def snapshot(env, name, out, robot="omp"):
    for _ in range(10):  # let objects settle
        env.step(hold(env))
    with mujoco.Renderer(env.sim.model._model, 480, 640) as r:
        cam = mujoco.MjvCamera()
        set_camera(cam, name, robot)
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # visual meshes only
        r.update_scene(env.sim.data._data, camera=cam, scene_option=opt)
        path = os.path.join(out, "task_%s.png" % name if robot == "omp" else "omx_task_%s.png" % name)
        PIL.Image.fromarray(r.render()).save(path)
    return path


def view(env, name, robot="omp"):
    import mujoco.viewer

    model, data = env.sim.model._model, env.sim.data._data
    print("%s: close the window for the next task" % name, flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer.cam, name, robot)
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            t0 = time.time()
            env.step(hold(env))  # OSC holds the start pose; physics keeps running
            viewer.sync()
            time.sleep(max(0.0, env.control_timestep - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="omp", choices=list(ROBOTS), help="task configs <Task>_<robot>.yaml")
    ap.add_argument("--env", choices=TASKS, help="one task (default: all)")
    ap.add_argument("--snapshot", action="store_true", help="save PNGs instead of opening the viewer")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    tasks = [args.env] if args.env else TASKS
    if args.robot != "omp":  # only the tasks adapted to this robot
        tasks = [t for t in tasks if os.path.exists(task_yaml(t, args.robot))]
    for name in tasks:
        env = make_env(name, args.robot)
        env.reset()
        if args.snapshot:
            os.makedirs(args.out, exist_ok=True)
            print(name, "->", snapshot(env, name, args.out, args.robot))
        else:
            view(env, name, args.robot)
        env.close()


if __name__ == "__main__":
    main()
