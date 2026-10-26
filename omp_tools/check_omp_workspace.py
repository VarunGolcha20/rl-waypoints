"""
Waypoint reachability map for the OpenManipulatorP, as seen by rlwp: each grid point is a waypoint (grip-point
target, world frame), tracked for one segment (50 steps) with rlwp's rule action = 10 * (target - eef) from the start
pose (the Lift cube is moved out of the way first, so it can't block targets). Reports tracking error / joint limits /
tool tilt, and for each task the largest y-symmetric grid box that contains the task's object spawn region and whose grid
points are all reached, as a `workspace:` block for the rlwp task YAML.

    MUJOCO_GL=egl python omp_tools/check_omp_workspace.py [--controller default_open_manipulator_p] [--tol 0.01]
"""
import argparse
import itertools
import json
import os

import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLLER_DIR = os.path.join(HERE, "..", "robosuite", "robosuite", "controllers", "config")

XS = np.round(np.arange(-0.35, 0.151, 0.05), 3)
YS = np.round(np.arange(0.0, 0.251, 0.05), 3)  # mirrored to -y
ZS = np.round(np.arange(0.83, 1.081, 0.05), 3)  # 0.83: fingertips ~9 mm above the 0.80 table

# Object spawn regions (world frame x/y, from the bundled envs) that each task's waypoints must cover
TASK_REGIONS = {
    "Lift": ([-0.03, -0.03], [0.03, 0.03]),  # cube
    "Stack": ([-0.08, -0.08], [0.08, 0.08]),  # cubeA x [-0.08, -0.04], cubeB x [0.02, 0.08], both y +-0.08
}
PARKED_CUBE = [0.35, 0.35, 0.82, 1, 0, 0, 0]  # table corner, outside the grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="default_open_manipulator_p")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = load_controller_config(custom_fpath=os.path.join(CONTROLLER_DIR, args.controller + ".json"))
    env = suite.make("Lift", robots="OpenManipulatorP", controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, initialization_noise=None,
                     control_freq=10, hard_reset=False)
    robot = env.robots[0]
    lo, hi = env.sim.model.jnt_range[robot._ref_joint_indexes].T
    eef_body = env.sim.model.body_name2id(robot.robot_model.eef_name)

    res = {}
    for x, y, z in itertools.product(XS, YS, ZS):
        env.reset()
        env.sim.data.set_joint_qpos(env.cube.joints[0], PARKED_CUBE)
        env.sim.forward()
        obs = env._get_observations(force_update=True)
        target = np.array([x, y, z])
        for _ in range(args.steps):
            obs = env.step(np.r_[10.0 * (target - obs["robot0_eef_pos"]), 0, 0, 0, 0])[0]
        q = robot._joint_positions
        R = env.sim.data.body_xmat[eef_body].reshape(3, 3)
        res[(x, y, z)] = dict(
            err=float(np.linalg.norm(target - obs["robot0_eef_pos"])),
            at_limit=[int(j + 1) for j in np.where(np.minimum(q - lo, hi - q) < 0.05)[0]],
            tilt=float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))),
        )
    start = env.reset()["robot0_eef_pos"]
    ok = {k: v["err"] < args.tol for k, v in res.items()}

    print("controller %s, tolerance %.0f mm, %d steps per waypoint; start eef %s; base x = %.2f" % (
        args.controller, 1000 * args.tol, args.steps, np.round(start, 3),
        env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")][0]))
    limits = {}
    for v in res.values():
        for j in v["at_limit"]:
            limits["J%d" % j] = limits.get("J%d" % j, 0) + 1
    print("grid points ending at a joint limit: %s; max tool tilt %.1f deg" % (
        limits or "none", max(v["tilt"] for v in res.values())))
    for y in YS:
        print("\n|y| = %.2f   tracking error (cm), '.' = within tolerance" % y)
        print("   z \\ x " + "".join("%6.2f" % x for x in XS))
        for z in ZS[::-1]:
            print("  %.2f  " % z + "".join("     ." if ok[(x, y, z)] else "%6.0f" % (100 * res[(x, y, z)]["err"]) for x in XS))

    # Per task: largest box [x0, x1] x [-yh, yh] x [z0, z1] (grid-aligned) containing the task's spawn region,
    # with every grid point inside it within tolerance
    for task, (rlo, rhi) in TASK_REGIONS.items():
        best, best_vol = None, -1
        for i0, i1 in itertools.combinations_with_replacement(range(len(XS)), 2):
            if not (XS[i0] <= rlo[0] and rhi[0] <= XS[i1]):
                continue
            for k0, k1 in itertools.combinations_with_replacement(range(len(ZS)), 2):
                for jh in range(len(YS)):
                    if YS[jh] < max(-rlo[1], rhi[1]):
                        continue
                    pts = itertools.product(XS[i0:i1 + 1], YS[:jh + 1], ZS[k0:k1 + 1])
                    if all(ok[p] for p in pts):
                        vol = (XS[i1] - XS[i0]) * YS[jh] * (ZS[k1] - ZS[k0])
                        if vol > best_vol:
                            best, best_vol = (XS[i0], XS[i1], YS[jh], ZS[k0], ZS[k1]), vol
        print("\n%s (objects x [%.2f, %.2f], y [%.2f, %.2f]):" % (task, rlo[0], rhi[0], rlo[1], rhi[1]))
        if best is None:
            print("  no fully reachable box contains the spawn region -> move the base or shrink the spawn region")
            continue
        x0, x1, yh, z0, z1 = best
        print("  box x [%.2f, %.2f], y [%.2f, %.2f], z [%.2f, %.2f]  (relative to start eef: dx [%+.2f, %+.2f], "
              "dz [%+.2f, %+.2f])" % (x0, x1, -yh, yh, z0, z1, x0 - start[0], x1 - start[0], z0 - start[2], z1 - start[2]))
        print("  rlwp task YAML (env:):\n    workspace:\n      low:  [%.2f, %.2f, %.2f]\n      high: [%.2f, %.2f, %.2f]" % (
            x0, -yh, z0, x1, yh, z1))

    path = os.path.join(args.out, "workspace_%s.json" % args.controller)
    json.dump({"%.2f,%.2f,%.2f" % k: v for k, v in res.items()}, open(path, "w"), indent=1)
    print("\nraw results:", os.path.abspath(path))


if __name__ == "__main__":
    main()
