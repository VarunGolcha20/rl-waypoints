"""
Standalone sanity checks for the OpenMANIPULATOR-P arm MJCF (robosuite asset), without robosuite.

    MUJOCO_GL=egl python omp_tools/check_omp_arm.py [--out omp_tools/out]

Checks: compile, FK vs DH, self-collision, gravity torque vs motor limits, PD hold stability; renders PNGs.
"""
import argparse
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "..", "robosuite", "robosuite", "models", "assets", "robots", "open_manipulator_p", "robot.xml")

# Candidate "gripper pointing down" pose: q2 + q3 + q5 = pi/2, q4 = q6 = 0, J5 away from 0 (wrist singularity)
DOWN_POSE = np.array([0.0, 0.3, 0.6, 0.0, 0.67, 0.0])
# Payload used for the torque check: OM-X gripper + adapter (~0.35 kg) at ~8 cm past the flange
TOOL_MASS, TOOL_OFFSET = 0.35, 0.08


def contacts(m, d):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        out.append((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, round(c.dist, 4)))
    return out


def render(m, d, path, azimuth, elevation, distance=1.1, lookat=(0.2, 0, 0.3)):
    with mujoco.Renderer(m, 480, 640) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        r.update_scene(d, camera=cam)
        import PIL.Image

        PIL.Image.fromarray(r.render()).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(ARM_XML)
    d = mujoco.MjData(m)
    print("[compile] ok: nq=%d, nu=%d, total mass=%.3f kg" % (m.nq, m.nu, m.body_subtreemass[0]))
    for j in range(m.njnt):
        print("  %-7s range=[%+.3f, %+.3f]  armature=%.2f  torque=+-%.1f" % (
            m.joint(j).name, *m.jnt_range[j], m.dof_armature[j], m.actuator_ctrlrange[j, 1]))

    # FK at q = 0 vs DH (upper arm 264 mm + 30 mm offset, forearm 195 + 63 mm, wrist 123 + 11.5 mm flange)
    mujoco.mj_kinematics(m, d)
    hand = d.body("right_hand")
    expected = np.array([0.03 + 0.195 + 0.063 + 0.123 + 0.0115, 0.0, 0.126 + 0.033 + 0.264 + 0.03])
    err = np.linalg.norm(hand.xpos - expected)
    print("[fk] right_hand @ q=0: %s (expected %s, err %.1e) -> %s" % (
        np.round(hand.xpos, 4), expected, err, "ok" if err < 1e-6 else "FAIL"))

    # Self-collision at the reference poses and over a random sweep
    for name, q in [("zero", np.zeros(6)), ("down", DOWN_POSE)]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        print("[collision] %-4s pose: %d contacts %s" % (name, d.ncon, contacts(m, d)))
        print("            right_hand pos=%s, tool axis=%s" % (
            np.round(d.body("right_hand").xpos, 3), np.round(d.body("right_hand").xmat.reshape(3, 3)[:, 2], 3)))
    rng = np.random.default_rng(0)
    n_hit, pairs = 0, {}
    for _ in range(2000):
        d.qpos[:] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        mujoco.mj_forward(m, d)
        if d.ncon:
            n_hit += 1
            for a, b, _ in contacts(m, d):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
    print("[collision] random sweep: %.1f%% of 2000 configs self-collide; pairs: %s" % (
        100 * n_hit / 2000, dict(sorted(pairs.items(), key=lambda x: -x[1])[:6])))

    # Gravity torque (arm + estimated tool) vs motor limits
    mt = mujoco.MjModel.from_xml_path(ARM_XML)
    b = mt.body("end_link").id
    old_m, old_c = mt.body_mass[b], mt.body_ipos[b].copy()
    new_c = (old_m * old_c + TOOL_MASS * np.array([0.0115 + TOOL_OFFSET, 0, 0])) / (old_m + TOOL_MASS)
    mt.body_mass[b], mt.body_ipos[b] = old_m + TOOL_MASS, new_c
    dt = mujoco.MjData(mt)
    lim = mt.actuator_ctrlrange[:, 1]
    worst = np.zeros(6)
    for _ in range(2000):
        dt.qpos[:] = rng.uniform(mt.jnt_range[:, 0], mt.jnt_range[:, 1])
        mujoco.mj_forward(mt, dt)
        worst = np.maximum(worst, np.abs(dt.qfrc_bias))
    print("[torque] worst static gravity torque (with %.2f kg tool) / limit: %s" % (
        TOOL_MASS, np.round(worst / lim, 2)))

    # PD + gravity compensation hold of the down pose (3 s): checks numerical stability and torque headroom
    d.qpos[:] = DOWN_POSE + 0.05
    d.qvel[:] = 0
    kp, kd = np.array([60, 60, 40, 20, 8, 8.0]), np.array([8, 8, 6, 3, 1, 1.0])
    peak = np.zeros(6)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_forward(m, d)
        tau = kp * (DOWN_POSE - d.qpos) - kd * d.qvel + d.qfrc_bias
        d.ctrl[:] = np.clip(tau, -lim, lim)
        peak = np.maximum(peak, np.abs(tau))
        mujoco.mj_step(m, d)
    hold_err = np.abs(d.qpos - DOWN_POSE).max()
    print("[hold] final max joint error %.4f rad, peak |tau|/limit %s, warnings: %s -> %s" % (
        hold_err, np.round(peak / lim, 2), [w.number for w in d.warning if w.number],
        # PD without integral term: frictionloss 0.1 / kp 8 (J5) leaves ~0.0125 rad steady-state error by design
        "ok" if hold_err < 0.02 and np.isfinite(d.qpos).all() else "CHECK"))

    # Renders
    for name, q in [("zero", np.zeros(6)), ("down", DOWN_POSE)]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for view, (az, el) in {"side": (90, -10), "persp": (135, -25)}.items():
            render(m, d, os.path.join(args.out, "arm_%s_%s.png" % (name, view)), az, el)
    print("[render] PNGs written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
