#!/usr/bin/env python3
"""
Read-only check of the OpenMANIPULATOR-X Dynamixel bus (U2D2): pings IDs 11-15 and prints model, mode, torque,
position, voltage, temperature and hardware errors. Writes nothing to the motors, so torque stays as it is (off after
power-up) and the arm does not move. Run it while the ROBOTIS bringup is NOT running (it needs the serial port).

    python3 omx_hw/scan_bus.py                    # /dev/ttyUSB0, 1 Mbps, IDs 11-15
    python3 omx_hw/scan_bus.py --all              # nothing found? try every ID and the common baud rates

Needs dynamixel_sdk (ROS 2 workspace sourced, or pip install dynamixel-sdk).
"""
import argparse
import math
import sys

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# XM430-W350 control table (protocol 2.0): name -> (address, size)
REG = {
    "drive_mode": (10, 1), "operating_mode": (11, 1), "homing_offset": (20, 4), "torque": (64, 1),
    "hw_error": (70, 1), "current": (126, 2), "position": (132, 4), "voltage": (144, 2), "temperature": (146, 1),
}
MODELS = {1020: "XM430-W350", 1030: "XM430-W210", 1060: "XM540-W270", 1120: "XM540-W150"}
MODES = {0: "current", 1: "velocity", 3: "position", 4: "ext. position", 5: "current-based position", 16: "PWM"}
JOINTS = {11: "joint1", 12: "joint2", 13: "joint3", 14: "joint4", 15: "gripper"}
# simulation start pose (OpenManipulatorX.init_qpos) and ROBOTIS' bringup home, rad
SIM_HOME = [0.0, -0.307, 0.249, 1.629]
ROS_HOME = [0.0, -1.0, 1.0, 0.0]
HW_ERRORS = {0: "input voltage", 2: "overheating", 3: "motor encoder", 4: "electrical shock", 5: "overload"}


def signed(v, size):
    bits = 8 * size
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def read(pk, port, dxl_id, name):
    addr, size = REG[name]
    fn = {1: pk.read1ByteTxRx, 2: pk.read2ByteTxRx, 4: pk.read4ByteTxRx}[size]
    val, res, err = fn(port, dxl_id, addr)
    if res != COMM_SUCCESS:
        return None
    return signed(val, size) if name in ("homing_offset", "current", "position") else val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--all", action="store_true", help="scan IDs 0-252 at 57600, 115200, 1M, 2M, 3M, 4M")
    args = ap.parse_args()

    port = PortHandler(args.port)
    if not port.openPort():
        sys.exit("cannot open %s (is the ROBOTIS bringup or another program using it?)" % args.port)
    pk = PacketHandler(2.0)
    try:
        if args.all:
            for baud in (57600, 115200, 1000000, 2000000, 3000000, 4000000):
                port.setBaudRate(baud)
                found = []
                for i in range(253):
                    m, r, _ = pk.ping(port, i)
                    if r == COMM_SUCCESS:
                        found.append((i, m))
                print("%7d bps: %s" % (baud, ", ".join("ID %d (%s)" % (i, MODELS.get(m, m)) for i, m in found) or "nothing"))
            return

        port.setBaudRate(args.baud)
        print("%s @ %d bps, protocol 2.0\n" % (args.port, args.baud))
        print("%-3s %-8s %-11s %-23s %-6s %-9s %-8s %-7s %-6s %-8s %s" % (
            "ID", "joint", "model", "operating mode", "torque", "pos (raw)", "pos rad", "volt", "temp", "current", "hw error"))
        ok, q = 0, {}
        for dxl_id, joint in JOINTS.items():
            model, res, err = pk.ping(port, dxl_id)
            if res != COMM_SUCCESS:
                print("%-3d %-8s NO RESPONSE (%s)" % (dxl_id, joint, pk.getTxRxResult(res)))
                continue
            ok += 1
            r = {n: read(pk, port, dxl_id, n) for n in REG}
            rad = None if r["position"] is None else (r["position"] - 2048) * 2 * math.pi / 4096
            if joint != "gripper" and rad is not None:
                q[joint] = rad
            errs = [txt for bit, txt in HW_ERRORS.items() if r["hw_error"] and r["hw_error"] >> bit & 1]
            print("%-3d %-8s %-11s %-23s %-6s %-9s %-8s %-7s %-6s %-8s %s" % (
                dxl_id, joint, MODELS.get(model, str(model)), MODES.get(r["operating_mode"], r["operating_mode"]),
                {0: "off", 1: "ON"}.get(r["torque"], "?"), r["position"],
                "-" if rad is None else "%+.3f" % rad,
                "-" if r["voltage"] is None else "%.1f V" % (r["voltage"] / 10),
                "-" if r["temperature"] is None else "%d C" % r["temperature"],
                r["current"], ", ".join(errs) if errs else ("none" if r["hw_error"] == 0 else r["hw_error"])))
        print("\n%d/5 motors responded" % ok)
        if len(q) == 4:
            arm = [q["joint%d" % i] for i in range(1, 5)]
            print("arm now (rad, ROBOTIS convention: 0 at raw 2048): [%s]" % ", ".join("%+.3f" % v for v in arm))
            for name, ref in (("sim start pose", SIM_HOME), ("ROBOTIS home", ROS_HOME)):
                print("  largest joint distance to %-15s %s: %.2f rad" % (
                    name, ref, max(abs(a - b) for a, b in zip(arm, ref))))
        if ok < 5:
            print("missing motors: check the 12 V supply, the U2D2 power hub, the daisy-chain cables, or run --all")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
