"""
Standalone checks for the OMXGripper (robosuite gripper model), using robosuite's GripperTester world
(gripper on a vertical slider above a table with a cube).

    MUJOCO_GL=egl python omp_tools/check_omx_gripper.py [--out omp_tools/out]

Checks: open/closed gap, grasp + lift of cubes of several sizes, both fingerpads in contact (what
robosuite's _check_grasp() tests), release; renders PNGs.
"""
import argparse
import os

import mujoco
import numpy as np
import PIL.Image

from robosuite.models.grippers import GripperTester, OMXGripper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_Z = 0.35  # gripper root height (pointing down)
GRIP_SITE_OFFSET = 0.156  # flange -> grip_site along the approach axis (see omx_gripper.xml)
PAD_HALF_THICKNESS = 0.002
FINGERTIP_PAST_GRIP_SITE = 0.021  # fingertip at OM-X x = 0.147, grip_site at 0.126
MIN_LIFT = 0.1  # slider travel low -> high is ~0.13 m


def make_tester(half_size, density):
    t = GripperTester(
        gripper=OMXGripper(),
        pos="0 0 %.3f" % ROOT_Z,
        quat="0 0 1 0",  # approach axis (+z) pointing down
        gripper_low_pos=-0.2,  # replaced after the cube has settled
        gripper_high_pos=0.05,
        box_size=[half_size] * 3,
        box_density=density,
        render=False,
    )
    t.start_simulation()
    return t


def pad_ids(t):
    g = t.gripper.important_geoms
    return [t.sim.model.geom_name2id(g["left_fingerpad"][0]), t.sim.model.geom_name2id(g["right_fingerpad"][0])]


def finger_gap(t):
    a, b = (t.sim.data.geom_xpos[i] for i in pad_ids(t))
    return np.linalg.norm(a - b) - 2 * PAD_HALF_THICKNESS


def pads_touch_cube(t):
    pads = set(pad_ids(t))
    touching = set()
    for i in range(t.sim.data.ncon):
        c = t.sim.data.contact[i]
        for g_pad, g_other in [(c.geom1, c.geom2), (c.geom2, c.geom1)]:
            if g_pad in pads and t.sim.model.geom_bodyid[g_other] == t.object_id:
                touching.add(g_pad)
    return len(touching) == 2


def run_phase(t, low, closed, n=400):
    t.gripper_z_is_low, t.gripper_is_closed = low, closed
    for _ in range(n):
        t.step()


def snapshot(t, path):
    m, d = t.sim.model._model, t.sim.data._data
    with mujoco.Renderer(m, 480, 640) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0, 0, 0.2), 0.45, 0, -10
        r.update_scene(d, camera=cam)
        PIL.Image.fromarray(r.render()).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Opening range, gripper high, no contact
    t = make_tester(0.02, 1000)
    run_phase(t, low=False, closed=False)
    gap_open = finger_gap(t)
    run_phase(t, low=False, closed=True)
    gap_closed = finger_gap(t)
    print("[gap] open %.1f mm, closed %.1f mm (expected ~82 / ~20)" % (1000 * gap_open, 1000 * gap_closed))
    t.close()

    # 2. Grasp + lift: 24 mm, 40 mm (Lift cube ~40-44 mm), 50 mm (Stack cubeB); ~64 g for 40 mm at density 1000
    all_ok = True
    for half, density in [(0.012, 1000), (0.02, 1000), (0.025, 1000), (0.02, 3000)]:
        t = make_tester(half, density)
        run_phase(t, low=False, closed=False, n=200)  # let the cube settle
        cube_z0 = t.sim.data.body_xpos[t.object_id][2]
        # put grip_site at the cube centre, but keep the fingertips (21 mm past grip_site) 4 mm above the table:
        # root_z - offset + low = target_z  (slider moves down for negative q)
        target_z = max(cube_z0, cube_z0 - half + FINGERTIP_PAST_GRIP_SITE + 0.004)
        t.gripper_low_pos = target_z + GRIP_SITE_OFFSET - ROOT_Z
        run_phase(t, low=True, closed=False)
        run_phase(t, low=True, closed=True)
        grasp = pads_touch_cube(t)
        gap = finger_gap(t)
        run_phase(t, low=False, closed=True)
        lift = t.sim.data.body_xpos[t.object_id][2] - cube_z0
        held = pads_touch_cube(t)
        if half == 0.02 and density == 1000:
            snapshot(t, os.path.join(args.out, "gripper_lift_40mm.png"))
        run_phase(t, low=False, closed=False)
        dropped = t.sim.data.body_xpos[t.object_id][2] - cube_z0 < 0.01
        ok = grasp and held and lift > MIN_LIFT and dropped
        all_ok &= ok
        mass = density * (2 * half) ** 3
        print("[grasp] cube %2d mm, %3.0f g: pads-touch=%s gap=%.1f mm lifted=%.3f m held=%s released=%s -> %s" % (
            2000 * half, 1000 * mass, grasp, 1000 * gap, lift, held, dropped, "ok" if ok else "FAIL"))
        t.close()
    print("[result]", "all grasp tests passed" if all_ok else "SOME GRASP TESTS FAILED")
    print("[render] PNGs written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
