#!/usr/bin/env python3
"""
Where does a trained model work? Simulates the full episode (learned waypoints + rlwp's controller, the same as
plan_episode.py's prediction) for one object on a grid of positions, the other object fixed, and prints a success map
with the trained area marked. Positions in the robot base frame (cm), as everywhere in omx_hw/.

    ../.venv/bin/python omx_hw/sim_sweep.py --task Lift                                   # cube, default grid
    ../.venv/bin/python omx_hw/sim_sweep.py --task Stack --object cubeA --other 10.3 13.5
    ../.venv/bin/python omx_hw/sim_sweep.py --task Stack --object cubeB --other 17.3 -1.5 --x 6 15 --y 8 17
    ... --run omx_stack_v4_s0   (another model)   --json out.json   (save the map)

One simulation per grid point (~1.5 s each): outside the trained area expect scattered failures, so treat it as a map,
not a guarantee. Lift: success = lifted (strict check); Stack: strict stacking check.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_episode as pe  # noqa: E402  (also changes into rlwp/, where the models are)

DEFAULT_GRID = {("Lift", "cube"): ((14, 22), (-8, 8)), ("Stack", "cubeA"): ((13, 20), (-6, 3)),
                ("Stack", "cubeB"): ((-1, 17), (-21, 21))}  # cubeB: covers the 17-20 cm landing ring (2026-11-22)
DEFAULT_OTHER = {("Stack", "cubeA"): (12.0, 13.0), ("Stack", "cubeB"): (17.3, -1.5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(pe.TASKS), required=True)
    ap.add_argument("--object", default=None, help="cube (Lift), cubeA or cubeB (Stack); default cube / cubeA")
    ap.add_argument("--other", nargs=2, type=float, metavar=("X_CM", "Y_CM"), help="Stack: the other cube's position")
    ap.add_argument("--x", nargs=2, type=float, metavar=("FROM", "TO"), help="grid x range (cm)")
    ap.add_argument("--y", nargs=2, type=float, metavar=("FROM", "TO"), help="grid y range (cm)")
    ap.add_argument("--step", type=float, default=1.0, help="grid step (cm), default 1")
    ap.add_argument("--run", default=None, help="model run name (default: the task's best model)")
    ap.add_argument("--config", default=None, help="task config instead of the task's default (e.g. Stack_omx_rel for an "
                                                     "object-relative model, Stack_omx_v3 for omx_stack_v3_s0)")
    ap.add_argument("--json", default=None, help="also save the map as JSON")
    args = ap.parse_args()

    name = args.task
    obj = args.object or ("cube" if name == "Lift" else "cubeA")
    if (name, obj) not in DEFAULT_GRID:
        ap.error("--object %s is not a %s object" % (obj, name))
    (gx, gy) = DEFAULT_GRID[(name, obj)]
    (xa, xb), (ya, yb) = args.x or gx, args.y or gy
    xs, ys = np.arange(xa, xb + 1e-9, args.step), np.arange(ya, yb + 1e-9, args.step)
    other = tuple(args.other) if args.other else DEFAULT_OTHER.get((name, obj))
    task_name, default_run = pe.TASKS[name]
    task_name = args.config or task_name
    run = args.run or default_run

    env, task = pe.make_env(task_name)
    ec = task["env"]
    obs = env.reset()
    start = obs["robot0_eef_pos"].copy()
    axis = env.sim.data.xanchor[env.sim.model.joint_name2id("robot0_joint1")][:2].copy()
    kw = ec["env_kwargs"]
    ring = kw.get("cubeB_ring") if obj == "cubeB" else None
    if ring is not None:  # ring-sector landing area (Stack_omx since 2026-11-22): distance / angle around its centre
        rc = (np.array(ring["center"]) - axis) * 100
        r0, r1 = [100 * v for v in ring["r_range"]]
        area = {"center_cm": rc.tolist(), "r_range_cm": [r0, r1], "angle_ranges": ring["angle_ranges"]}
        area_txt = "%.0f-%.0f cm from joint 1, angle %s" % (r0, r1, " or ".join("%+.0f..%+.0f" % tuple(a) for a in ring["angle_ranges"]))

        def inside(x, y):
            r, th = np.hypot(x - rc[0], y - rc[1]), np.degrees(np.arctan2(y - rc[1], x - rc[0]))
            return r0 - 0.01 <= r <= r1 + 0.01 and any(lo - 0.01 <= th <= hi + 0.01 for lo, hi in ring["angle_ranges"])
    else:
        area = [(np.array(kw["%s_%s_range" % (obj, c)]) + env.table_offset[i] - axis[i]) * 100 for i, c in enumerate("xy")]
        (x0, x1), (y0, y1) = area
        area_txt = "x %.1f..%.1f, y %.1f..%.1f" % (x0, x1, y0, y1)

        def inside(x, y):
            return x0 - 0.01 <= x <= x1 + 0.01 and y0 - 0.01 <= y <= y1 + 0.01

    print("%s %s with model %s, %d positions%s" % (task_name, obj, run, len(xs) * len(ys),
                                                   "" if other is None else ", other cube at (%.1f, %+.1f)" % other))
    res, t0 = {}, time.time()
    for x in xs:
        for y in ys:
            cubes = [(x, y)] if name == "Lift" else ([(x, y), other] if obj == "cubeA" else [other, (x, y)])
            np.random.seed(0)
            env.reset()
            if len(cubes) == 2 and np.linalg.norm(np.subtract(cubes[0], cubes[1])) < 8.0:
                res[(round(x, 2), round(y, 2))] = None  # the cubes would overlap: not tested
                continue
            obs = pe.place_cubes(env, name, [axis + np.array(c, float) / 100 for c in cubes])
            obs = pe.run_turn(env, ec, obs)  # object-relative configs: turn toward cubeA first (no-op otherwise)
            pobjs = pe.model_objs(ec, pe.policy_objs(name, obs), obs)
            agent = pe.Method(state_dim=ec["state_dim"], objs=pobjs, wp_id=2,
                              save_name=pe.get_save_name(name, ec["robot"], run, None, ec["use_latch"]),
                              config={"task": task, "n_inits": pe.N_INITS[name]}, workspace=pe.get_workspace(ec, start))
            traj = agent.traj_opt(1, pobjs).reshape(2, ec["state_dim"])
            ok, _, _ = pe.rollout(env, task, traj, pe.waypoint_targets(ec, traj[:, :3], start, obs), obs)
            res[(round(x, 2), round(y, 2))] = bool(ok)

    print("\n%d/%d succeed in simulation (%.0f s, %d optimiser restarts)   # = success, . = fail, o = not tested (cubes overlap), [ ] = trained area %s"
          % (sum(v is True for v in res.values()), sum(v is not None for v in res.values()), time.time() - t0, pe.N_INITS[name], area_txt))
    print("  y(cm) " + "".join("%6g" % x for x in xs) + "   <- x (cm)")
    for y in ys[::-1]:
        row = ""
        for x in xs:
            v = res[(round(x, 2), round(y, 2))]
            row += ("   [%s]" if inside(x, y) else "    %s ") % ("o" if v is None else "#" if v else ".")
        print("  %+5g %s" % (y, row))
    if args.json:
        json.dump({"task": task_name, "object": obj, "run": run, "other": other, "trained_area": area if isinstance(area, dict) else [list(area[0]), list(area[1])],
                   "results": [[x, y, ok] for (x, y), ok in res.items()]}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
