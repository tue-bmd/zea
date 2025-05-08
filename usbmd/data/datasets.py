"""The datasets module contains classes for loading different types of
ultrasound datasets.

- **Author(s)**     : Tristan Stevens, Wessel van Nierop
- **Date**          : November 18th, 2021
"""

from collections import OrderedDict
from pathlib import Path
from typing import List

import tqdm

from usbmd.data.file import File, get_shape_hdf5_file, validate_dataset
from usbmd.datapaths import format_data_path
from usbmd.utils import (
    calculate_file_hash,
    date_string_to_readable,
    get_date_string,
    log,
)
from usbmd.utils.io_lib import search_file_tree

_CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE = 10000
_VALIDATED_FLAG_FILE = "validated.flag"
FILE_HANDLE_CACHE_CAPACITY = 128
FILE_TYPES = [".hdf5", ".h5"]


class H5FileHandleCache:
    """Cache for HDF5 file handles.

    This class manages a cache of HDF5 file handles to avoid reopening files
    multiple times. It uses an OrderedDict to maintain the order of file
    access and closes the least recently used file when the cache reaches
    its capacity."""

    def __init__(
        self,
        file_handle_cache_capacity: int = FILE_HANDLE_CACHE_CAPACITY,
    ):
        self._file_handle_cache = OrderedDict()
        self.file_handle_cache_capacity = file_handle_cache_capacity

    @staticmethod
    def _check_if_open(file):
        """Check if a file is open."""
        return bool(file.id.valid)

    def get_file(self, file_path) -> File:
        """Open an HDF5 file and cache it."""
        # If file is already in cache, return it and move it to the end
        if file_path in self._file_handle_cache:
            self._file_handle_cache.move_to_end(file_path)
            file = self._file_handle_cache[file_path]
            # if file was closed, reopen:
            if not self._check_if_open(file):
                file = File(file_path, "r")
                self._file_handle_cache[file_path] = file
        # If file is not in cache, open it and add it to the cache
        else:
            # If cache is full, close the least recently used file
            if len(self._file_handle_cache) >= self.file_handle_cache_capacity:
                _, close_file = self._file_handle_cache.popitem(last=False)
                close_file.close()
            file = File(file_path, "r")
            self._file_handle_cache[file_path] = file

        return self._file_handle_cache[file_path]

    def __del__(self):
        """Ensure cached files are closed."""
        for _, file in self._file_handle_cache.items():
            file.close()
        self._file_handle_cache = OrderedDict()


def find_h5_files_from_directory(
    directory: str | list,
    key: str,
    search_file_tree_kwargs: dict | None = None,
    additional_axes_iter: tuple | None = None,
):
    """
    Find HDF5 files from a directory or list of directories and retrieve their shapes.

    Args:
        directory (str or list): A single directory path, a list of directory paths,
            or a single HDF5 file path.
        key (str): The key to access the HDF5 dataset.
        search_file_tree_kwargs (dict, optional): Additional keyword arguments for the
            search_file_tree function. Defaults to None.
        additional_axes_iter (tuple, optional): Additional axes to iterate over if dataset_info
            contains file_shapes. Defaults to None.

    Returns:
        - file_paths (list): List of file paths to the HDF5 files.
        - file_shapes (list): List of shapes of the HDF5 datasets.
    """

    file_paths = []
    file_shapes = []

    if search_file_tree_kwargs is None:
        search_file_tree_kwargs = {}

    # 'directory' is actually just a single hdf5 file
    if not isinstance(directory, list) and Path(directory).is_file():
        filename = directory
        file_shapes = [get_shape_hdf5_file(filename, key)]
        file_paths = [str(filename)]
        return file_paths, file_shapes

    dataset_info = search_file_tree(
        directory,
        filetypes=FILE_TYPES,
        hdf5_key_for_length=key,
        **search_file_tree_kwargs,
    )
    file_paths = dataset_info["file_paths"]
    file_paths = [str(Path(directory) / file_path) for file_path in file_paths]
    if "file_shapes" not in dataset_info:
        assert additional_axes_iter is None, (
            "additional_axes_iter is only supported if the dataset_info "
            "contains file_shapes. please remove dataset_info.yaml files and rerun."
        )
        # since in this case we only need to iterate over the first axis it is
        # okay we only have the lengths of the files (and not the full shape)
        file_shapes.extend([[length] for length in dataset_info["file_lengths"]])
    else:
        file_shapes.extend(dataset_info["file_shapes"])

    return file_paths, file_shapes


class Dataset(H5FileHandleCache):
    """Read multiple usbmd.File objects from a folder."""

    def __init__(
        self,
        key: str,
        file_paths: List[str],
        file_shapes: List[tuple],
        additional_axes_iter: tuple = None,
        **kwargs,
    ):
        """Initializes the Dataset.

        Args:
            config (utils.config.Config): The config.data configuration object.
        """
        super().__init__(**kwargs)
        self.key = key
        self.file_paths = file_paths
        self.file_shapes = file_shapes
        if additional_axes_iter is None:
            additional_axes_iter = []
        self.additional_axes_iter = additional_axes_iter

        assert self.n_files > 0, f"No files in file_paths: {file_paths}"

    @classmethod
    def from_path(
        cls,
        path: str | list,
        key: str,
        search_file_tree_kwargs: dict | None = None,
        additional_axes_iter: tuple = None,
        validate=True,
        **kwargs,
    ):
        """Creates a Dataset from a path.

        Saves a dataset_info.yaml file in the directory with information about the dataset.
        This file is used to load the dataset later on, which speeds up the initial loading
        of the dataset for very large datasets.
        """
        path = Path(path)
        file_paths, file_shapes = find_h5_files_from_directory(
            path,
            key,
            search_file_tree_kwargs,
            additional_axes_iter,
        )
        assert len(file_paths) > 0, f"No files in directories:\n{path}"
        instance = cls(
            key=key,
            file_paths=file_paths,
            file_shapes=file_shapes,
            **kwargs,
        )
        if validate:
            instance.validate_dataset(path)
        return instance

    @classmethod
    def from_config(cls, dataset_folder, dtype, user=None, **kwargs):
        """Creates a Dataset from a config file."""
        path = format_data_path(dataset_folder, user)

        if "file_path" in kwargs:
            log.warning(
                "Found 'file_path' in config, this will be ignored since a Dataset is "
                + "always multiple files."
            )

        return cls.from_path(path, key=dtype)

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.n_files

    @property
    def n_files(self):
        """Return number of files in dataset."""
        return len(self.file_paths)

    def __getitem__(self, index) -> File:
        """Retrieves an item from the dataset."""
        file = self.get_file(self.file_paths[index])
        return file

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        for idx in range(self.n_files):
            yield self[idx]

    def __call__(self):
        return iter(self)

    @property
    def total_frames(self):
        """Return total number of frames in dataset."""
        return sum(self.get_file(file_path).num_frames for file_path in self.file_paths)

    def validate_dataset(self, path):
        """Validate dataset contents.
        Furthermore, it checks if all files in the dataset have the same scan parameters.

        TODO: I don't think it actually checks for the same scan parameters.

        If a validation file exists, it checks if the dataset was validated on the same date.
        If the validation file was corrupted, it raises an error.
        If the validation file was not corrupted and validated, it prints a message and returns.
        """

        validation_file_path = path / _VALIDATED_FLAG_FILE
        # for error logging
        validation_error_file_path = Path(
            path, get_date_string() + "_validation_errors.log"
        )
        validation_error_log = []

        if validation_file_path.is_file():
            self._assert_validation_file(validation_file_path)
            return

        if self.n_files > _CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE:
            log.warning(
                "Checking scan parameters in more than "
                f"{_CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE} files takes too long. "
                f"Found {self.n_files} files in dataset. "
                "Not checking scan parameters."
            )
            return

        num_frames_per_file = []
        validated_succesfully = True
        for file_path in tqdm.tqdm(
            self.file_paths,
            total=self.n_files,
            desc="Checking dataset files on validity (USBMD format)",
        ):
            try:
                validate_dataset(file_path)
            except Exception as e:
                validation_error_log.append(
                    f"File {file_path} is not a valid USBMD dataset.\n{e}\n"
                )
                # convert into warning
                log.warning(f"Error in file {file_path}.\n{e}")
                validated_succesfully = False

        if not validated_succesfully:
            log.warning(
                "Not all files in dataset have the same scan parameters. "
                "Check warnings above for details. No validation file was created. "
                f"See {validation_error_file_path} for details."
            )
            try:
                with open(validation_error_file_path, "w", encoding="utf-8") as f:
                    for error in validation_error_log:
                        f.write(error)
            except Exception as e:
                log.error(
                    f"Could not write validation errors to {validation_error_file_path}."
                    f"\n{e}"
                )
            return

        # Create the validated flag file
        self._write_validation_file(path, num_frames_per_file)
        log.info(
            f"{log.green('Dataset validated.')} Check {validation_file_path} for details."
        )

    @staticmethod
    def _assert_validation_file(validation_file_path):
        """Check if validation file exists and is valid."""
        with open(validation_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            try:
                validation_date = lines[1].split(": ")[1].strip()
                read_validation_file_hash = lines[-1].split(": ")[1].strip()
            except Exception as exc:
                raise ValueError(
                    log.error(
                        f"Validation file {log.yellow(validation_file_path)} is corrupted. "
                        "Remove it if you want to redo validation."
                    )
                ) from exc

            log.info(
                "Dataset was validated on "
                f"{log.green(date_string_to_readable(validation_date))}"
            )
            log.info(
                f"Remove {log.yellow(validation_file_path)} if you want to redo validation."
            )
        # check if validation file was corrupted
        validation_file_hash = calculate_file_hash(
            validation_file_path, omit_line_str="hash"
        )
        assert validation_file_hash == read_validation_file_hash, log.error(
            f"Validation file {log.yellow(validation_file_path)} was corrupted.\n"
            f"Remove it if you want to redo validation.\n"
        )

    def get_data_types(self, file_path):
        """Get data types from file."""
        file = self.get_file(file_path)
        if "data" in file:
            data_types = list(file["data"].keys())
        else:
            data_types = list(file["event_0"]["data"].keys())
        return data_types

    def _write_validation_file(self, path, num_frames_per_file):
        """Write validation file."""
        validation_file_path = Path(path, _VALIDATED_FLAG_FILE)

        # Read data types from the first file
        data_types = self.get_data_types(self.file_paths[0])

        number_of_frames = sum(num_frames_per_file)
        with open(validation_file_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {path}\n")
            f.write(f"Validated on: {get_date_string()}\n")
            f.write(f"Number of files: {self.n_files}\n")
            f.write(f"Number of frames: {number_of_frames}\n")
            f.write(f"Data types: {', '.join(data_types)}\n")
            f.write(f"{'-' * 80}\n")
            # write all file names (not entire path) with number of frames on a new line
            for file_path, num_frames in zip(self.file_paths, num_frames_per_file):
                f.write(f"{file_path.name}: {num_frames}\n")
            f.write(f"{'-' * 80}\n")

        # Write the hash of the validation file
        validation_file_hash = calculate_file_hash(validation_file_path)
        with open(validation_file_path, "a", encoding="utf-8") as f:
            # *** validation file hash *** (80 total line length)
            f.write("*** validation file hash ***\n")
            f.write(f"hash: {validation_file_hash}")
