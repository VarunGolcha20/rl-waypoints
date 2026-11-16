"""
Waypoint reachability map for the OpenManipulatorP (default) or OpenManipulatorX, as seen by rlwp: each grid point is a
waypoint (grip-point target, world frame), tracked for one segment (50 steps) with rlwp's rule
action = 10 * (target - eef) from the start pose (the Lift cube is moved out of the way first, so it can't block
targets). Reports tracking error / joint limits / tool tilt, and for each task the largest grid box that contains the
task's object spawn region and whose grid points are all reached, as a `workspace:` block for the rlwp task YAML.
OM-P boxes are y-symmetric; OM-X boxes may be asymmetric in y (its tasks put the second object to one side).

    MUJOCO_GL=egl python omp_tools/check_omp_workspace.py [--robot OpenManipulatorX] [--controller ...] [--tol 0.01]
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
# 0.825: fingertips 4 mm above the 0.80 table (grasp height for flat objects); then 0.83 and 2.5 cm steps
ZS = np.round(np.r_[0.825, np.arange(0.83, 1.031, 0.025)], 3)

# Per task (adapted OM-P layouts, rlwp/cfg/task/<Task>_omp.yaml): the x/y region the waypoints must reach (objects,
# grasp points and drop / pull targets), the lowest acceptable box ceiling, and the task's z shift relative to this grid
# (the grid is measured with the base on a 0.80 m table; NutAssembly's table is at 0.82)
TASK_REGIONS = {
    "Lift": ([-0.03, -0.03], [0.03, 0.03], 0.86, 0.0),  # cube; success = cube centre > 0.84 (+ margin)
    "Stack": ([-0.04, -0.05], [0.01, 0.16], 0.905, 0.0),  # cubeA in front, cubeB 12-16 cm to the side; carry high
    "NutAssembly": ([-0.06, -0.18], [-0.04, 0.03], 0.885, 0.02),  # handle grasp points, over the round peg
    "PickPlace": ([-0.08, -0.065], [0.025, 0.125], 0.905, 0.0),  # can in the object bin, target compartment
    "Door": ([-0.07, -0.01], [0.01, 0.09], 0.86, 0.0),  # handle bar, pulled 0.35 rad along the arc
}
PARKED_CUBE = [0.35, 0.35, 0.82, 1, 0, 0, 0]  # table corner, outside the grid

# OpenManipulatorX (base at x = -0.17 on the 1.7 cm plate; gripper pointing down it reaches ~0.06-0.24 m from the base
# axis, up to ~12 cm above the table): finer, smaller grid
OMX_XS = np.round(np.arange(-0.12, 0.081, 0.025), 3)
OMX_YS = np.round(np.arange(0.0, 0.226, 0.025), 3)  # mirrored to -y
OMX_ZS = np.round(np.r_[0.825, np.arange(0.83, 0.936, 0.015)], 3)
# adapted OM-X layouts, rlwp/cfg/task/<Task>_omx.yaml: (x/y region lo, hi, minimum ceiling, z shift, maximum floor).
# The floor limit keeps the box search (which maximises volume) from giving up the grasp height: the 55 mm cubes are
# gripped at z ~0.83
OMX_TASK_REGIONS = {
    "Lift": ([-0.03, -0.03], [0.03, 0.03], 0.86, 0.0, 0.83),  # cube; lifted = cube centre > table + 4 cm
    # cubeA straight ahead 12-15 cm out (grasp), cubeB 16-20 cm out, ~45-55 deg to the left (release over it, carry high)
    "Stack": ([-0.07, -0.03], [-0.02, 0.15], 0.905, 0.0, 0.83),
}

ROBOTS = {
    "OpenManipulatorP": dict(controller="default_open_manipulator_p", xs=XS, ys=YS, zs=ZS, regions=TASK_REGIONS,
                             symmetric=True),
    "OpenManipulatorX": dict(controller="default_open_manipulator_x", xs=OMX_XS, ys=OMX_YS, zs=OMX_ZS,
                             regions=OMX_TASK_REGIONS, symmetric=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="OpenManipulatorP", choices=list(ROBOTS))
    ap.add_argument("--controller", default=None, help="controller config name (default: the robot's own)")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--reuse", action="store_true", help="recompute the boxes from the saved grid results instead of simulating")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rc = ROBOTS[args.robot]
    args.controller = args.controller or rc["controller"]
    XS, YS, ZS, TASK_REGIONS = rc["xs"], rc["ys"], rc["zs"], rc["regions"]

    cfg = load_controller_config(custom_fpath=os.path.join(CONTROLLER_DIR, args.controller + ".json"))
    env = suite.make("Lift", robots=args.robot, controller_configs=cfg, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, initialization_noise=None,
                     control_freq=10, hard_reset=False)
    robot = env.robots[0]
    lo, hi = env.sim.model.jnt_range[robot._ref_joint_indexes].T
    eef_body = env.sim.model.body_name2id(robot.robot_model.eef_name)

    path = os.path.join(args.out, "workspace_%s.json" % args.controller)
    res = {}
    if args.reuse:
        snap = lambda v, grid: grid[np.abs(grid - v).argmin()]  # older files stored 2 decimals
        for k, v in json.load(open(path)).items():
            x, y, z = (float(c) for c in k.split(","))
            res[(snap(x, XS), snap(y, YS), snap(z, ZS))] = v
    for x, y, z in ([] if args.reuse else itertools.product(XS, YS, ZS)):
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
            print(" %.3f  " % z + "".join("     ." if ok[(x, y, z)] else "%6.0f" % (100 * res[(x, y, z)]["err"]) for x in XS))

    # Per task: largest box [x0, x1] x [y0, y1] x [z0, z1] (grid-aligned) containing the task's spawn region, with every
    # grid point inside it within tolerance. y0 = -y1 for symmetric robots (OM-P); the grid is mirrored to -y.
    YF = np.unique(np.r_[-YS, YS])
    okf = lambda x, y, z: ok[(x, YS[np.abs(YS - abs(y)).argmin()], z)]
    for task, region in TASK_REGIONS.items():
        rlo, rhi, ztop, zshift = region[:4]
        zfloor = region[4] if len(region) > 4 else np.inf  # highest acceptable box floor (OM-P regions: no limit)
        best, best_vol = None, -1
        if rc["symmetric"]:
            ypairs = [(-yh, yh) for yh in YS]
        else:
            ypairs = [(YF[j0], YF[j1]) for j0, j1 in itertools.combinations_with_replacement(range(len(YF)), 2)]
        ypairs = [(y0, y1) for y0, y1 in ypairs if y0 <= rlo[1] + 1e-9 and rhi[1] <= y1 + 1e-9]
        for i0, i1 in itertools.combinations_with_replacement(range(len(XS)), 2):
            if not (XS[i0] <= rlo[0] + 1e-9 and rhi[0] <= XS[i1] + 1e-9):
                continue
            for k0, k1 in itertools.combinations_with_replacement(range(len(ZS)), 2):
                if ZS[k1] < ztop or ZS[k0] > zfloor + 1e-9:
                    continue
                for y0, y1 in ypairs:
                    ys = YF[(YF >= y0 - 1e-9) & (YF <= y1 + 1e-9)]
                    if all(okf(x, y, z) for x, y, z in itertools.product(XS[i0:i1 + 1], ys, ZS[k0:k1 + 1])):
                        vol = (XS[i1] - XS[i0]) * (y1 - y0) * (ZS[k1] - ZS[k0])
                        if vol > best_vol:
                            best, best_vol = (XS[i0], XS[i1], y0, y1, ZS[k0], ZS[k1]), vol
        print("\n%s (objects x [%.2f, %.2f], y [%.2f, %.2f], ceiling >= %.3f%s):" % (
            task, rlo[0], rhi[0], rlo[1], rhi[1], ztop, ", floor <= %.3f" % zfloor if np.isfinite(zfloor) else ""))
        if best is None:
            print("  no fully reachable box contains the spawn region up to that ceiling -> move the base or shrink the spawn region")
            continue
        x0, x1, y0, y1, z0, z1 = best
        print("  box x [%.3f, %.3f], y [%.3f, %.3f], z [%.3f, %.3f]  (relative to start eef: dx [%+.3f, %+.3f], "
              "dz [%+.3f, %+.3f])" % (x0, x1, y0, y1, z0, z1, x0 - start[0], x1 - start[0], z0 - start[2], z1 - start[2]))
        print("  rlwp task YAML (env:)%s:\n    workspace:\n      low:  [%.3f, %.3f, %.3f]\n      high: [%.3f, %.3f, %.3f]" % (
            " (z shifted by %+.2f for this task's table)" % zshift if zshift else "", x0, y0, z0 + zshift, x1, y1, z1 + zshift))

    if not args.reuse:
        json.dump({"%.3f,%.3f,%.3f" % k: v for k, v in res.items()}, open(path, "w"), indent=1)
    print("\nraw results:", os.path.abspath(path))


if __name__ == "__main__":
    main()
