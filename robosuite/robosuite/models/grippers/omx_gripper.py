"""
Gripper from ROBOTIS OpenMANIPULATOR-X (two parallel fingers on a rack-and-pinion).

OMXGripper: mounted on another arm (the OpenMANIPULATOR-P) via an adapter plate.
OMXNativeGripper: on its own OpenMANIPULATOR-X arm (no adapter, gripper root on the joint4 axis).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OMXGripperBase(GripperModel):
    """
    OpenMANIPULATOR-X gripper with one actuator per finger.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    # gripper MJCF (subclasses for other mountings override this)
    XML = "grippers/omx_gripper.xml"

    def __init__(self, idn=0):
        super().__init__(xml_path_completion(self.XML), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        # fully open (82 mm gap)
        return np.array([0.02, -0.02])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class OMXGripper(OMXGripperBase):
    """
    Modifies OMXGripperBase to only take one action (same convention as PandaGripper).
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


class OMXNativeGripper(OMXGripper):
    """
    OMXGripper on its own OpenMANIPULATOR-X arm: no adapter plate, gripper root = joint4 axis, grip point 0.126 m
    along the approach axis (ROBOTIS end_effector_link). Same fingers, actuators and action convention.
    """

    XML = "grippers/omx_native_gripper.xml"

    # Start every episode with the internal command at "fully open" (the state init_qpos puts the fingers in). With the
    # default zero command the fingers first moved toward half-open (82 -> ~68 mm) on the first step and then reopened.
    # Only the OM-X's own gripper: the OM-P (OMXGripper) keeps the original behaviour of its trained models.
    reset_action = (1.0, -1.0)
