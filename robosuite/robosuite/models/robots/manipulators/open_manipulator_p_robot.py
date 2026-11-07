import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OpenManipulatorP(ManipulatorModel):
    """
    OpenMANIPULATOR-P (RM-P60-RNH) is a 6-DoF arm from ROBOTIS, here with the OpenMANIPULATOR-X gripper,
    sitting on a ~9 mm base plate (TurtleBot3 Waffle Pi waffle plate) on the table top.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    # base_xpos_offset["table"] puts the base plate on a 0.8 m table top; envs with another table height shift it
    table_mounted = True

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/open_manipulator_p/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "OMPRiserMount"

    @property
    def default_gripper(self):
        return "OMXGripper"

    @property
    def default_controller_config(self):
        return "default_open_manipulator_p"

    @property
    def init_qpos(self):
        # Gripper pointing straight down, grip point 0.22 m in front of the base and 0.07 m above it
        # (0.079 m above the table with the 9 mm base plate). J5 = 1.213 keeps 0.36 rad to its limit and stays
        # clear of the J5 = 0 singularity.
        return np.array([0.0, -0.235, 0.592, 0.0, 1.213, 0.0])

    @property
    def base_xpos_offset(self):
        # The robot stands on the table top (base plate bottom at z = 0.8, the table height used by Lift / Stack / Door)
        # instead of on a floor pedestal: OM-P's 0.645 m reach needs the objects ~0.3 m from the base.
        return {
            # PickPlace: same place as for the table tasks, on a stand at bin height (added by the env). The original
            # bins are out of reach; use the smaller, lower bins from rlwp/cfg/task/PickPlace_omp.yaml
            "bins": (-0.32, 0.0, 0.8),
            "empty": (-0.3, 0, 0),
            "table": lambda table_length: (-table_length / 2 + 0.08, 0, 0.8),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.75))

    @property
    def _horizontal_radius(self):
        return 0.3

    @property
    def arm_type(self):
        return "single"
