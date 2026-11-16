from .mount_model import MountModel
from .mount_factory import mount_factory

from .rethink_mount import RethinkMount
from .rethink_minimal_mount import RethinkMinimalMount
from .null_mount import NullMount
from .omp_riser_mount import OMPRiserMount
from .omx_base_mount import OMXBaseMount


MOUNT_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "OMPRiserMount": OMPRiserMount,
    "OMXBaseMount": OMXBaseMount,
    None: NullMount,
}

ALL_MOUNTS = MOUNT_MAPPING.keys()
