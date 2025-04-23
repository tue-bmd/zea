"""This submodule contains the definition of the USBMD data format as well as
all code to convert other datasets into the USBMD data format.

.. include:: ./README.md
"""

from .convert.camus import sitk_load
from .data_format import *
from .dataloader import H5Dataloader
from .file import File, load_usbmd_file
