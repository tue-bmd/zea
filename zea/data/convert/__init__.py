"""All functions in this module are used to convert different datasets to the zea format."""

from .camus import convert_camus
from .images import convert_image_dataset
from .matlab import zea_from_matlab_raw
from .picmus import convert_picmus
