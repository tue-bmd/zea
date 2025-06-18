"""Data subpackage for working with the ``zea`` data format.

This subpackage provides core classes and utilities for working with the zea data format,
including file and dataset access, validation, and data loading. For more information on the
``zea`` data format, see :doc:`../data-acquisition`.

Main classes
------------

- :class:`zea.data.File` -- Open and access a single zea HDF5 data file.
- :class:`zea.data.Dataset` -- Manage and iterate over a collection of zea data files.

See the data notebook for a more detailed example: :doc:`../notebooks/data/zea_data_example`

Examples usage
^^^^^^^^^^^^^^

.. code-block:: python

    from zea import File, Dataset

    # Open a single zea data file
    with File("path/to/file.hdf5", mode="r") as file:
        file.summary()
        data = file.load_data("raw_data", indices=[0])
        scan = file.scan()
        probe = file.probe()

    # Work with a dataset (folder or list of files)
    dataset = Dataset("path/to/folder", key="raw_data")
    for file in dataset:
        print(file)
    dataset.close()

Subpackage layout
-----------------

- ``file.py``: Implements :class:`zea.File` and related file utilities.
- ``datasets.py``: Implements :class:`zea.Dataset` and folder management.
- ``dataloader.py``: Data loading utilities for batching and shuffling.
- ``data_format.py``: Data validation and example dataset generation.
- ``convert/``: Data conversion tools (e.g., from external formats).

"""  # noqa: E501

from .convert.camus import sitk_load
from .data_format import (
    DatasetElement,
    generate_example_dataset,
    generate_zea_dataset,
    validate_input_data,
)
from .dataloader import H5Generator
from .datasets import Dataset, Folder
from .file import File, load_file
