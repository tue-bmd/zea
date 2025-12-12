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

.. doctest::

    >>> from zea import File, Dataset

    >>> # Work with a single file
    >>> path_to_file = (
    ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
    ... )

    >>> with File(path_to_file, mode="r") as file:
    ...     # file.summary()
    ...     data = file.load_data("raw_data", indices=[0])
    ...     scan = file.scan()
    ...     probe = file.probe()

    >>> # Work with a dataset (folder or list of files)
    >>> dataset = Dataset("hf://zeahub/picmus", key="raw_data")
    >>> files = []
    >>> for file in dataset:
    ...     files.append(file)  # process each file as needed
    >>> dataset.close()

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
