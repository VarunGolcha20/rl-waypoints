"""
Base plate under the OpenMANIPULATOR-X: a box on the table top whose size you can change below.
Same mechanism as the OpenMANIPULATOR-P's OMPRiserMount (see omp_riser_mount.py), with its own numbers, so the two
robots can stand on different plates.
"""
from robosuite.models.mounts.omp_riser_mount import OMPRiserMount

# ---------------------------------------------------------------------------------------------------------------------
# Base plate box: EDIT HERE. Sizes in metres: length along x (towards the work area), width along y, height along z.
# The height lifts the robot and shifts its whole vertical working range: after changing it, rerun
# omp_tools/check_omp_workspace.py (and omp_tools/check_omp_tasks.py) and update the workspace blocks in
# rlwp/cfg/task/*_omx.yaml.
OMX_BASE_PLATE_SIZE = (0.14, 0.14, 0.017)  # the user's measured plate (~1.7 cm, TurtleBot3 Waffle Pi); to be confirmed
OMX_BASE_PLATE_OFFSET = (0.0, 0.0)  # x, y of the plate centre relative to the robot base (0, 0 = centred under it)
# ROBOTIS' own OM-X kit plate (meshes/open_manipulator_x/plate.stl) would be (0.35, 0.30, 0.008) with offset (0.1272, 0)
# ---------------------------------------------------------------------------------------------------------------------


class OMXBaseMount(OMPRiserMount):
    """
    Box-shaped base plate that the OpenMANIPULATOR-X stands on, on the table top (OMX_BASE_PLATE_SIZE / _OFFSET above,
    or the @size / @offset arguments).

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
        size (None or 3-tuple): (length x, width y, height z) in metres. None = OMX_BASE_PLATE_SIZE
        offset (None or 2-tuple): plate centre (x, y) relative to the robot base. None = OMX_BASE_PLATE_OFFSET
    """

    def __init__(self, idn=0, size=None, offset=None):
        super().__init__(
            idn=idn,
            size=OMX_BASE_PLATE_SIZE if size is None else size,
            offset=OMX_BASE_PLATE_OFFSET if offset is None else offset,
        )
