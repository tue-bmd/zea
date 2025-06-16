"""This submodule contains the definition of the zea data format as well as
all code to convert other datasets into the zea data format.

"""

from .convert.camus import sitk_load
from .data_format import (
    generate_example_dataset,
    generate_zea_dataset,
    validate_input_data,
    DatasetElement,
)
from .dataloader import H5Generator
from .datasets import Dataset, Folder
from .file import File, load_file
