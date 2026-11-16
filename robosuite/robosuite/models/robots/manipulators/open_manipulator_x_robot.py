import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OpenManipulatorX(ManipulatorModel):
    """
    OpenMANIPULATOR-X (RM-X52-TNM) is a 4-DoF arm from ROBOTIS (4x XM430-W350) with its own gripper (OMXNativeGripper,
    a 5th XM430), sitting on a box-shaped base plate on the table top (OMXBaseMount; size set in omx_base_mount.py).

    4 DoF: joint1 turns the arm about the vertical axis, joints 2-4 are pitch joints. So the gripper can point straight
    down anywhere in reach (q2 + q3 + q4 = pi/2), but its yaw is always atan2(y, x) of the target seen from the base:
    the fingers open perpendicular to the line from the base, not along a fixed world axis.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    # base_xpos_offset["table"] puts the base plate on a 0.8 m table top; envs with another table height shift it
    table_mounted = True

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/open_manipulator_x/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "OMXBaseMount"

    @property
    def default_gripper(self):
        return "OMXNativeGripper"

    @property
    def default_controller_config(self):
        return "default_open_manipulator_x"

    @property
    def init_qpos(self):
        # Gripper pointing straight down (q2 + q3 + q4 = pi/2), grip point 0.12 m in front of the base and 0.07 m above
        # it (0.087 m above the table with the 1.7 cm base plate, as for the OM-P). Planar IK; J4 is the limiting joint
        # (0.34 rad below its 1.97 limit), J2 / J3 have > 1 rad margin. See IMPLEMENTATION_LOG X1.
        return np.array([0.0, -0.307, 0.249, 1.629])

    @property
    def base_xpos_offset(self):
        # The robot stands on the table top (base plate bottom at z = 0.8) with the base 0.17 m behind the table centre,
        # where the tasks place their objects: with the gripper pointing down the OM-X reaches ~0.06-0.24 m from its
        # base axis, best around 0.12-0.20 m. PROVISIONAL, set from the workspace grid in task X2.
        return {
            "bins": (-0.17, 0.0, 0.8),
            "empty": (-0.17, 0, 0),
            "table": lambda table_length: (-table_length / 2 + 0.23, 0, 0.8),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.45))

    @property
    def _horizontal_radius(self):
        return 0.2

    @property
    def arm_type(self):
        return "single"
