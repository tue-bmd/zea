"""This submodule contains the definition of the zea data format as well as
all code to convert other datasets into the zea data format.

"""

from .convert.camus import sitk_load
from .data_format import *
from .file import File, load_zea_file
