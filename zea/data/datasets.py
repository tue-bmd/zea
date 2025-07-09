"""
zea.data.datasets
===================

This module provides classes and utilities for loading, validating, and managing
ultrasound datasets stored in HDF5 format. It supports both local and Hugging Face
Hub datasets, and offers efficient file handle caching for large collections of files.

Main Classes
------------

- H5FileHandleCache: Caches open HDF5 file handles to optimize repeated access.
- Folder: Represents a group of HDF5 files in a directory, with optional validation.
- Dataset: Provides an iterable interface over multiple HDF5 files or folders, with
    support for directory-based splitting and validation.

Functions
---------

- find_h5_files: Recursively finds HDF5 files and retrieves their dataset shapes.
- split_files_by_directory: Splits files among directories according to specified ratios.
- count_samples_per_directory: Counts the number of files per directory.

Features
--------

- Validation of dataset integrity with flag files and error logging.
- Support for Hugging Face Hub datasets with local caching.
- Utilities for dataset splitting and sample counting.
- Example usage provided in the module's main block.

"""

from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tqdm

from zea import log
from zea.data.file import File, validate_file
from zea.data.preset_utils import (
    HF_DATASETS_DIR,
    HF_PREFIX,
    _hf_list_files,
    _hf_parse_path,
    _hf_resolve_path,
)
from zea.datapaths import format_data_path
from zea.io_lib import search_file_tree
from zea.tools.hf import HFPath
from zea.utils import (
    calculate_file_hash,
    date_string_to_readable,
    get_date_string,
    reduce_to_signature,
)

_CHECK_MAX_DATASET_SIZE = 10000
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
        if hasattr(self, "_file_handle_cache"):
            for _, file in self._file_handle_cache.items():
                if file is not None and self._check_if_open(file):
                    file.close()
            self._file_handle_cache = OrderedDict()


def find_h5_files(paths: str | list, key: str = None, search_file_tree_kwargs: dict | None = None):
    """
    Find HDF5 files from a directory or list of directories and optionally retrieve their shapes.

    Args:
        paths (str or list): A single directory path, a list of directory paths,
            or a single HDF5 file path.
        key (str, optional): The key to get the file shapes for.
        search_file_tree_kwargs (dict, optional): Additional keyword arguments for the
            search_file_tree function. Defaults to None.

    Returns:
        - file_paths (list): List of file paths to the HDF5 files.
        - file_shapes (list): List of shapes of the HDF5 datasets.
    """

    if search_file_tree_kwargs is None:
        search_file_tree_kwargs = {}

    # Make sure paths is a list
    if not isinstance(paths, (tuple, list)):
        paths = [paths]

    file_shapes = []
    file_paths = []
    for path in paths:
        if isinstance(path, (str, Path, HFPath)) and str(path).startswith(HF_PREFIX):
            # Let File handle HF path resolution
            resolved = _hf_resolve_path(str(path))
            path = resolved

        if Path(path).is_file():
            path = Path(path)
            # If the path is a file, get its shape directly
            if key is not None:
                file_shapes.append(File.get_shape(path, key))
            file_paths.append(str(path))
            continue

        dataset_info = search_file_tree(
            path,
            filetypes=FILE_TYPES,
            hdf5_key_for_length=key,
            **search_file_tree_kwargs,
        )
        file_shapes += dataset_info["file_shapes"]
        file_paths += [str(Path(path) / file_path) for file_path in dataset_info["file_paths"]]

    return file_paths, file_shapes


class Folder:
    """Group of HDF5 files in a folder that can be validated.
    Mostly used internally, you might want to use the Dataset class instead.
    """

    def __init__(
        self,
        folder_path: list[str] | list[Path],
        key: str = None,
        search_file_tree_kwargs: dict | None = None,
        validate: bool = True,
        hf_cache_dir: str = HF_DATASETS_DIR,
        **kwargs,
    ):
        # Hugging Face support
        if isinstance(folder_path, (str, Path, HFPath)):
            folder_path_str = str(folder_path)
            if folder_path_str.startswith(HF_PREFIX):
                # Only resolve to local dir if it's a directory, else let File handle it
                repo_id, subpath = _hf_parse_path(folder_path_str)
                files = _hf_list_files(repo_id)
                if subpath and any(f == subpath for f in files):
                    # It's a file, let File handle it
                    pass
                else:
                    folder_path = _hf_resolve_path(folder_path_str, cache_dir=hf_cache_dir)

        super().__init__(**kwargs)

        self.folder_path = Path(folder_path)
        self.key = key
        self.search_file_tree_kwargs = search_file_tree_kwargs
        self.validate = validate
        self.file_paths, self.file_shapes = find_h5_files(
            folder_path, self.key, self.search_file_tree_kwargs
        )
        assert self.n_files > 0, f"No files in folder: {folder_path}"
        if self.validate:
            self.validate_folder()

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.n_files

    @property
    def n_files(self):
        """Return number of files in dataset."""
        return len(self.file_paths)

    def validate_folder(self):
        """Validate dataset contents.

        If a validation file exists, it checks if the dataset was validated on the same date.
        If the validation file was corrupted, it raises an error.
        If the validation file was not corrupted and validated, it prints a message and returns.
        """

        validation_file_path = self.folder_path / _VALIDATED_FLAG_FILE
        # for error logging
        validation_error_file_path = Path(
            self.folder_path, get_date_string() + "_validation_errors.log"
        )
        validation_error_log = []

        if validation_file_path.is_file():
            self._assert_validation_file(validation_file_path)
            return

        if self.n_files > _CHECK_MAX_DATASET_SIZE:
            log.warning(
                "Checking dataset in more than "
                f"{_CHECK_MAX_DATASET_SIZE} files takes too long. "
                f"Found {self.n_files} files in dataset. "
            )
            return

        num_frames_per_file = []
        validated_succesfully = True
        for file_path in tqdm.tqdm(
            self.file_paths,
            total=self.n_files,
            desc="Checking dataset files on validity (zea format)",
        ):
            try:
                validate_file(file_path)
            except Exception as e:
                validation_error_log.append(f"File {file_path} is not a valid zea dataset.\n{e}\n")
                # convert into warning
                log.warning(f"Error in file {file_path}.\n{e}")
                validated_succesfully = False

        if not validated_succesfully:
            log.warning(
                "Check warnings above for details. No validation file was created. "
                f"See {validation_error_file_path} for details."
            )
            try:
                with open(validation_error_file_path, "w", encoding="utf-8") as f:
                    for error in validation_error_log:
                        f.write(error)
            except Exception as e:
                log.error(
                    f"Could not write validation errors to {validation_error_file_path}.\n{e}"
                )
            return

        # Create the validated flag file
        self._write_validation_file(self.folder_path, num_frames_per_file)
        log.info(f"{log.green('Dataset validated.')} Check {validation_file_path} for details.")

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
                f"Dataset was validated on {log.green(date_string_to_readable(validation_date))}"
            )
            log.info(f"Remove {log.yellow(validation_file_path)} if you want to redo validation.")
        # check if validation file was corrupted
        validation_file_hash = calculate_file_hash(validation_file_path, omit_line_str="hash")
        assert validation_file_hash == read_validation_file_hash, log.error(
            f"Validation file {log.yellow(validation_file_path)} was corrupted.\n"
            f"Remove it if you want to redo validation.\n"
        )

    @staticmethod
    def get_data_types(file_path):
        """Get data types from file."""
        with File(file_path) as file:
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

    def __repr__(self):
        return (
            f"<zea.data.datasets.Folder at 0x{id(self):x}: "
            f"{self.n_files} files, key='{self.key}', folder='{self.folder_path}'>"
        )

    def __str__(self):
        return f"Folder with {self.n_files} files in '{self.folder_path}' (key='{self.key}')"

    def copy(self, to_path: str | Path, key: str = None, mode: str | None = None):
        """Copy the data for all or a specific key to a new location.

        Has the option to copy all keys or only a specific key. By default, it only copies if the
        destination file does not already contain the key. You can change the mode to 'w' to
        overwrite the destination file. Will always copy metadata such as dataset attributes and
        scan object.

        Args:
            to_path (str or Path): The destination path where files will be copied.
            key (str, optional): The key to copy from the source files.
                If 'all' or '*', all keys will be copied. Defaults to None, which
                uses the key set in the Folder instance.
            mode (str): The mode in which to open the destination files.
                Defaults to 'a' (append mode), and 'w' (write mode) if key is 'all' or '*'.
                See: https://docs.h5py.org/en/stable/high/file.html#opening-creating-files
        """
        if key is None and self.key is None:
            raise ValueError(
                "No key specified. Please provide a key to copy the data for, or set the "
                "key attribute of the Folder instance."
            )
        elif key is None:
            key = self.key

        all_keys = key == "all" or key == "*"

        if mode is None:
            mode = "a" if not all_keys else "w"

        if all_keys:
            key_msg = "Including all keys."
            assert mode in ["w", "x"], (
                "If you want to copy all keys, the destination file must be opened "
                "in 'w' or 'x' mode, which means it will be overwritten or created."
            )
        else:
            key_msg = f"Only copying key '{key}'."
            assert mode in ["a", "w", "r+", "x"], (
                f"Invalid mode '{mode}'. Must be one of 'a', 'w', 'r+', or 'x'."
            )

        to_path = Path(to_path)
        to_path.mkdir(parents=True, exist_ok=True)

        for file_path in tqdm.tqdm(
            self.file_paths,
            total=self.n_files,
            desc=f"Copying dataset from {self.folder_path} to {to_path}. {key_msg}",
        ):
            dst_path = Path(file_path).relative_to(self.folder_path)
            with File(file_path) as src, File(to_path / dst_path, mode) as dst:
                if all_keys:
                    for obj in src.keys():
                        src.copy(obj, dst)
                else:
                    src.copy_key(key, dst)


class Dataset(H5FileHandleCache):
    """Iterate over File(s) and Folder(s)."""

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str,
        search_file_tree_kwargs: dict | None = None,
        validate: bool = True,
        directory_splits: list | None = None,
        **kwargs,
    ):
        """Initializes the Dataset.

        Args:
            file_paths (str or list): (list of) path(s) to the folder(s) containing the HDF5 file(s)
                or list of HDF5 file paths. Can be a mixed list of folders and files.
            key (str): The key to access the HDF5 dataset.
            search_file_tree_kwargs (dict, optional): Additional keyword arguments for the
                search_file_tree function. These are only used when `file_paths` are directories.
                Defaults to None.
            validate (bool, optional): Whether to validate the dataset. Defaults to True.
            directory_splits (list, optional): List of directory split by. Is a list of floats
                between 0 and 1, with the same length as the number of file_paths given.
                If none, all files in file_paths are used.

        """
        super().__init__(**kwargs)
        self.key = key
        self.search_file_tree_kwargs = search_file_tree_kwargs
        self.validate = validate

        self.file_paths, self.file_shapes = self.find_files_and_shapes(file_paths)

        if directory_splits is not None:
            # Split the files according to their parent directories
            self.file_paths, self.file_shapes = split_files_by_directory(
                self.file_paths,
                self.file_shapes,
                directory_list=file_paths,
                directory_splits=directory_splits,
            )

        assert self.n_files > 0, f"No files in file_paths: {file_paths}"

    def find_files_and_shapes(self, paths):
        """Find files and shapes in the dataset."""
        # Initialize file paths and shapes
        file_paths = []
        file_shapes = []

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        for file_path in paths:
            if isinstance(file_path, (list, tuple)):
                # If the path is a list, recursively call find_files_and_shapes
                _file_paths, _file_shapes = self.find_files_and_shapes(file_path)
                file_paths += _file_paths
                file_shapes += _file_shapes
                continue

            file_path = str(file_path)
            if file_path.startswith(HF_PREFIX):
                file_path = HFPath(file_path)
            else:
                file_path = Path(file_path)

            if file_path.is_dir():
                folder = Folder(file_path, self.key, self.search_file_tree_kwargs, self.validate)
                file_paths += folder.file_paths
                file_shapes += folder.file_shapes
                del folder
            elif file_path.is_file():
                file_paths.append(file_path)
                with File(file_path) as file:
                    file_shapes.append(file.shape(self.key))
                    if self.validate:
                        file.validate()

        return file_paths, file_shapes

    @classmethod
    def from_config(cls, dataset_folder, dtype, user=None, **kwargs):
        """Creates a Dataset from a config file."""
        path = format_data_path(dataset_folder, user)

        if "file_path" in kwargs:
            log.warning(
                "Found 'file_path' in config, this will be ignored since a Dataset is "
                + "always multiple files."
            )

        reduced_params = reduce_to_signature(cls.__init__, kwargs)
        return cls(path, key=dtype, **reduced_params)

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
        return sum(self.get_file(file_path).n_frames for file_path in self.file_paths)

    def __repr__(self):
        return (
            f"<zea.data.datasets.Dataset at 0x{id(self):x}: {self.n_files} files, key='{self.key}'>"
        )

    def __str__(self):
        return f"Dataset with {self.n_files} files (key='{self.key}')"

    def close(self):
        """Close all cached file handles."""
        for file in self._file_handle_cache.values():
            if file is not None and file.id.valid:
                file.close()
        self._file_handle_cache.clear()
        log.info("Closed all cached file handles.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def split_files_by_directory(file_names, file_shapes, directory_list, directory_splits):
    """Split files according to their parent directories and given split ratios.

    Args:
        file_names (list): List of file paths.
        file_shapes (list): List of shapes for each file.
        directory_list (list): List of directory paths to split by.
        directory_splits (list): List of split ratios (0-1) for each directory.

    Returns:
        tuple: (split_file_names, split_file_shapes)
    """
    if isinstance(directory_list, str):
        directory_list = [directory_list]
    if isinstance(directory_splits, (float, int)):
        directory_splits = [directory_splits]

    assert len(directory_splits) == len(directory_list), (
        "Number of directory splits must be equal to the number of directories."
    )
    assert all(0 <= split <= 1 for split in directory_splits), (
        "Directory splits must be between 0 and 1."
    )

    # Get directory sizes using the new count function implementation
    directory_counts = count_samples_per_directory(file_names, directory_list)
    directory_sizes = [directory_counts[str(dir_path)] for dir_path in directory_list]

    # take percentage of the files from each directory
    split_indices = [int(split * size) for split, size in zip(directory_splits, directory_sizes)]

    # offset split indices by each total number of files
    start_datasets = [0] + list(np.cumsum(directory_sizes))
    split_indices = [
        (start_datasets[i], start_datasets[i] + split) for i, split in enumerate(split_indices)
    ]

    # split the files
    split_file_names = []
    split_file_shapes = []

    for start, end in split_indices:
        split_file_names.extend(file_names[start:end])
        split_file_shapes.extend(file_shapes[start:end])

    # verify the split size
    expected_size = sum((d * s) for d, s in zip(directory_sizes, directory_splits))
    expected_size = int(expected_size)
    assert len(split_file_names) == expected_size, (
        "Number of files in split directories does not match the expected number. "
        "Please check the directory splits."
    )

    return split_file_names, split_file_shapes


def count_samples_per_directory(file_names, directories):
    """Count number of samples per directory.

    Args:
        file_names (list): List of file paths
        directories (str or list): Directory or list of directories

    Returns:
        dict: Dictionary with directory paths as keys and sample counts as values
    """
    if not isinstance(directories, list):
        directories = [directories]

    # Convert all paths to strings with normalized separators
    dir_paths = [str(Path(d)) for d in directories]

    file_paths = [str(Path(f)) for f in file_names]

    # Count files per directory using string matching
    counts = {
        dir_path: sum(1 for f in file_paths if f.startswith(dir_path)) for dir_path in dir_paths
    }

    # Assert that the total counts match the number of files
    total_count = sum(counts.values())
    assert total_count == len(file_paths), (
        f"Total count of files ({total_count}) does not match the number of files provided "
        f"({len(file_paths)}). Some files may not belong to any of the specified directories."
    )

    return counts


if __name__ == "__main__":
    src_folder = Folder(
        "/mnt/z/Ultrasound-BMd/data/USBMD_datasets/CAMUS/val/patient0450", "image", validate=False
    )
    src_folder.copy("./CAMIUS", key="all")
