"""Data subpackage for working with the ``zea`` data format.

This subpackage provides core classes and utilities for working with the zea data format,
including file and dataset access, validation, and data loading. For more information on the
``zea`` data format, see :doc:`../data-acquisition`.

Main classes
------------

- :class:`zea.File` — open, create, and validate a single zea HDF5 file.
- :class:`zea.Dataset` — manage and iterate over a collection of zea data files.
- :class:`zea.Dataloader` — Data loader for training pipelines.

See the data notebook for a more detailed example: :doc:`../notebooks/data/zea_data_example`

Example usage
^^^^^^^^^^^^^

.. doctest::

    >>> from zea import File, Dataset, Dataloader

    >>> # Work with a single file
    >>> path_to_file = (
    ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
    ... )

    >>> with File(path_to_file, mode="r") as file:
    ...     data = file.data.raw_data[0]  # first frame
    ...     params = file.load_parameters()

    >>> # Work with a dataset (folder or list of files)
    >>> dataset = Dataset("hf://zeahub/picmus")
    >>> files = []
    >>> for file in dataset:
    ...     files.append(file)  # process each file as needed
    >>> dataset.close()

    >>> # Use a dataloader for training
    >>> dataloader = Dataloader(
    ...     "hf://zeahub/camus-sample/",
    ...     key="data/image/values",
    ...     batch_size=4,
    ...     image_size=(256, 256),
    ...     shuffle=True,
    ... )
    >>> for batch in dataloader:
    ...     # process batch for training
    ...     pass

"""  # noqa: E501

from .convert.camus import sitk_load
from .dataloader import Dataloader
from .datasets import Dataset, Folder
from .file import CustomElement, File, load_file
