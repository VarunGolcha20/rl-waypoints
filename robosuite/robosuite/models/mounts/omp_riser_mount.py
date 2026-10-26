"""
Table-top riser block for the OpenMANIPULATOR-P.
"""
import numpy as np

from robosuite.models.mounts.mount_model import MountModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OMPRiserMount(MountModel):
    """
    10 cm riser block that the OpenMANIPULATOR-P base sits on (placed on the table top).

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("mounts/omp_riser_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.1
