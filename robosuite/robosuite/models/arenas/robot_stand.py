"""
Static stand for robots that are placed on a table top (e.g. OpenManipulatorP, `table_mounted = True`) in arenas that
have no table where the robot stands (bins, door).
"""
import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import array_to_string


def add_robot_stand(arena, xy, top_z, half_size=0.08, name="robot_stand"):
    """
    Adds a box from the floor up to @top_z, centred under the robot base at @xy.

    Args:
        arena (Arena): arena to add the stand to (before the model is compiled)
        xy (2-array): robot base position in the world frame
        top_z (float): height of the stand's top (where the robot's base plate sits)
        half_size (float): half of the stand's footprint side
        name (str): body / geom name prefix
    """
    body = ET.Element("body", name=name, pos=array_to_string([xy[0], xy[1], top_z / 2.0]))
    size = array_to_string([half_size, half_size, top_z / 2.0])
    body.append(ET.Element("geom", name=name + "_collision", type="box", size=size, group="0", rgba="0.4 0.4 0.4 1"))
    body.append(ET.Element("geom", name=name + "_visual", type="box", size=size, group="1", contype="0",
                           conaffinity="0", rgba="0.35 0.35 0.38 1"))
    arena.worldbody.append(body)
