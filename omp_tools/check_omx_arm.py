"""
Standalone checks for the OpenMANIPULATOR-X arm MJCF (robosuite asset), without robosuite.

    MUJOCO_GL=egl python omp_tools/check_omx_arm.py [--robotis ~/Desktop/IRob/ros2_ws/src/open_manipulator]

Checks: compile; FK of every link and of the grip point vs ROBOTIS' own MuJoCo model and URDF (from a local clone of
ROBOTIS-GIT/open_manipulator, jazzy branch); the start pose (init_qpos) points the gripper down; the tool-down reach
map (where the grip point can be with the gripper pointing straight down, within the joint limits).
"""
import argparse
import os
import re
import tempfile

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "..", "robosuite", "robosuite", "models", "assets", "robots", "open_manipulator_x", "robot.xml")
GRIP = 0.126  # grip point (ROBOTIS end_effector_link) along OM-X link5 +x, from the joint4 axis
INIT_QPOS = np.array([0.0, -0.307, 0.249, 1.629])  # OpenManipulatorX.init_qpos
PLATE = 0.017  # base plate height (OMX_BASE_PLATE_SIZE[2]): table top is at base z = -PLATE
OURS = ["base", "link2", "link3", "link4", "link5"]
THEIRS = ["link1", "link2", "link3", "link4", "link5"]


def poses(m, d, q, names):
    d.qpos[: len(q)] = q
    mujoco.mj_kinematics(m, d)
    out = []
    for n in names:
        b = m.body(n).id
        out.append((d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()))
    return out


def grip_point(pose5):
    pos, rot = pose5
    return pos + rot @ np.array([GRIP, 0, 0]), rot[:, 0]  # grip point, approach axis


def load_urdf_without_meshes(path):
    """ROBOTIS' URDF with visual / collision elements removed (package:// meshes) and fixed links kept."""
    txt = open(path).read()
    txt = re.sub(r"<visual>.*?</visual>", "", txt, flags=re.S)
    txt = re.sub(r"<collision>.*?</collision>", "", txt, flags=re.S)
    txt = txt.replace("<robot name=\"open_manipulator_x\">",
                      "<robot name=\"open_manipulator_x\"><mujoco><compiler fusestatic=\"false\" discardvisual=\"false\"/></mujoco>", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(txt)
    try:
        return mujoco.MjModel.from_xml_path(f.name)
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotis", default=os.path.expanduser("~/Desktop/IRob/ros2_ws/src/open_manipulator"),
                    help="local clone of ROBOTIS-GIT/open_manipulator (jazzy)")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(ARM_XML)
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    print("[compile] ok: nq=%d, nu=%d, arm mass (without gripper) %.3f kg, ranges %s" % (
        m.nq, m.nu, m.body_mass.sum(), np.round(m.jnt_range, 2).tolist()))

    # 1. FK vs ROBOTIS' MuJoCo model and URDF at random configurations (+ zero / start pose)
    desc = os.path.join(args.robotis, "open_manipulator_description")
    refs = []
    mj_path = os.path.join(desc, "mujoco", "open_manipulator_x", "open_manipulator_x.xml")
    urdf_path = os.path.join(desc, "urdf", "open_manipulator_x", "open_manipulator_x.urdf")
    if os.path.exists(mj_path):
        refs.append(("ROBOTIS MJCF", mujoco.MjModel.from_xml_path(mj_path)))
    if os.path.exists(urdf_path):
        refs.append(("ROBOTIS URDF", load_urdf_without_meshes(urdf_path)))
    if not refs:
        print("[fk] SKIPPED: no ROBOTIS model under", desc)
    rng = np.random.default_rng(0)
    qs = [np.zeros(4), INIT_QPOS] + [lo + rng.random(4) * (hi - lo) for _ in range(200)]
    for name, rm in refs:
        rd = mujoco.MjData(rm)
        jidx = [rm.joint(j).qposadr[0] for j in ["joint1", "joint2", "joint3", "joint4"]]
        err_pos = err_rot = err_grip = 0.0
        for q in qs:
            rq = np.zeros(rm.nq)
            rq[jidx] = q
            a = poses(m, d, q, OURS)
            b = poses(rm, rd, rq, THEIRS)
            err_pos = max(err_pos, max(np.linalg.norm(pa - pb) for (pa, _), (pb, _) in zip(a, b)))
            err_rot = max(err_rot, max(np.abs(ra - rb).max() for (_, ra), (_, rb) in zip(a, b)))
            ee = rd.xpos[rm.body("end_effector_link").id]
            err_grip = max(err_grip, np.linalg.norm(grip_point(a[4])[0] - ee))
        ok = max(err_pos, err_rot, err_grip) < 1e-9
        print("[fk] vs %s, %d configs: max link position error %.1e m, rotation %.1e, grip point vs end_effector_link "
              "%.1e m -> %s" % (name, len(qs), err_pos, err_rot, err_grip, "ok" if ok else "CHECK"))

    # 2. Start pose
    p, ax = grip_point(poses(m, d, INIT_QPOS, OURS)[4])
    print("[start] init_qpos %s: grip point %s m from the base (%.3f m above the table), approach axis %s, "
          "J4 margin to limit %.2f rad" % (INIT_QPOS, np.round(p, 4), p[2] + PLATE, np.round(ax, 4), hi[3] - INIT_QPOS[3]))

    # 3. Tool-down reach map (planar IK, joint1 = 0; by symmetry the same at every joint1 angle)
    j2 = np.array([0.012, 0.0595])
    la, phi, lb = np.hypot(0.024, 0.128), np.arctan2(0.024, 0.128), 0.124

    def ik_down(r, z):
        w = np.array([r, z + GRIP]) - j2
        c = (w @ w - la**2 - lb**2) / (2 * la * lb)
        sols = []
        if abs(c) <= 1:
            for s in (1, -1):
                g = s * np.arccos(c)
                alpha = np.arctan2(w[1], w[0]) + np.arctan2(lb * np.sin(g), la + lb * np.cos(g))
                q2 = np.pi / 2 - phi - alpha
                q3 = -(alpha - g) - q2
                q = np.array([0.0, q2, q3, np.pi / 2 - q2 - q3])
                if np.all(q >= lo) and np.all(q <= hi):
                    gp, gax = grip_point(poses(m, d, q, OURS)[4])
                    if np.linalg.norm(gp - [r, 0, z]) < 1e-6 and gax[2] < -0.999:
                        sols.append(q)
        return sols

    rs = np.arange(0.06, 0.301, 0.02)
    print("[reach] gripper pointing straight down: rows = grip point height above the table (cm), columns = distance "
          "from the base axis (cm), value = joint-limit margin (rad) of the best IK solution, '.' = unreachable")
    print("        " + " ".join("%5d" % round(r * 100) for r in rs))
    for h in np.arange(0.14, -0.001, -0.02):
        row = []
        for r in rs:
            sols = ik_down(r, h - PLATE)
            row.append("%5.2f" % max(min((q - lo).min(), (hi - q).min()) for q in sols) if sols else "    .")
        print("  %4d  " % round(h * 100) + " ".join(row))


if __name__ == "__main__":
    main()
