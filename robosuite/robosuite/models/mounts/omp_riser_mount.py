"""
Base plate under the OpenMANIPULATOR-P: a box on the table top whose size you can change below.
"""
import numpy as np

from robosuite.models.mounts.mount_model import MountModel
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion

# ---------------------------------------------------------------------------------------------------------------------
# Base plate box: EDIT HERE. Sizes in metres: length along x (towards the work area), width along y, height along z.
# The height lifts the robot and shifts its whole vertical working range: after changing it, rerun
# omp_tools/check_omp_workspace.py (and omp_tools/check_omp_tasks.py) and update the workspace blocks in
# rlwp/cfg/task/*_omp.yaml.
BASE_PLATE_SIZE = (0.14, 0.14, 0.017)  # TurtleBot3 Waffle Pi plate, ~1.7 cm thick (measured by the user)
BASE_PLATE_OFFSET = (0.0, 0.0)  # x, y of the plate centre relative to the robot base (0, 0 = centred under it)
BASE_PLATE_DENSITY = 1200.0  # kg / m^3 (plastic); the plate is static, so this only affects the reported mass
# ---------------------------------------------------------------------------------------------------------------------


class OMPRiserMount(MountModel):
    """
    Box-shaped base plate that the OpenMANIPULATOR-P stands on, on the table top. Its size comes from
    BASE_PLATE_SIZE / BASE_PLATE_OFFSET above (or the @size / @offset arguments).

    robosuite moves the mount root onto the robot base (top_offset = 0), so the box hangs below the root, and the
    mount's bottom offset (-height) is what lifts the robot by the plate height above the surface it is placed on.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
        size (None or 3-tuple): (length x, width y, height z) in metres. None = BASE_PLATE_SIZE
        offset (None or 2-tuple): plate centre (x, y) relative to the robot base. None = BASE_PLATE_OFFSET
    """

    def __init__(self, idn=0, size=None, offset=None):
        super().__init__(xml_path_completion("mounts/omp_riser_mount.xml"), idn=idn)
        length, width, height = BASE_PLATE_SIZE if size is None else size
        ox, oy = BASE_PLATE_OFFSET if offset is None else offset
        self.plate_size = np.array([length, width, height], dtype=float)
        self.plate_offset = np.array([ox, oy], dtype=float)

        half = array_to_string([length / 2.0, width / 2.0, height / 2.0])
        centre = array_to_string([ox, oy, -height / 2.0])
        for geom in self.worldbody.iter("geom"):
            geom.set("size", half)
            geom.set("pos", centre)
        inertial = self._elements["root_body"].find("inertial")
        mass = max(BASE_PLATE_DENSITY * length * width * height, 0.01)
        inertial.set("pos", centre)
        inertial.set("mass", str(mass))
        inertial.set("diaginertia", array_to_string(
            mass / 12.0 * np.array([width**2 + height**2, length**2 + height**2, length**2 + width**2])))
        # bottom of the plate, read by the robot model to stand the robot on top of it
        self._elements["root_body"].set("pos", array_to_string([0, 0, -height]))
        self._base_offset = np.array([0.0, 0.0, -height])

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return float(np.hypot(*(self.plate_size[:2] / 2.0 + np.abs(self.plate_offset))))
