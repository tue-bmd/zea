"""The datasets module contains classes for loading different types of
ultrasound datasets.
"""

from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tqdm

from usbmd import log
from usbmd.data.file import File, validate_file
from usbmd.datapaths import format_data_path
from usbmd.io_lib import search_file_tree
from usbmd.utils import (
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


def find_h5_files(
    paths: str | list, key: str, search_file_tree_kwargs: dict | None = None
):
    """
    Find HDF5 files from a directory or list of directories and retrieve their shapes.

    Args:
        paths (str or list): A single directory path, a list of directory paths,
            or a single HDF5 file path.
        key (str): The key to access the HDF5 dataset.
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

    paths = [Path(path) for path in paths]

    file_shapes = []
    file_paths = []
    for path in paths:
        if path.is_file():
            # If the path is a file, get its shape directly
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
        file_paths += [
            str(Path(path) / file_path) for file_path in dataset_info["file_paths"]
        ]

    return file_paths, file_shapes


class Folder:
    """Group of HDF5 files in a folder that can be validated.
    Mostly used internally, you might want to use the Dataset class instead.
    """

    def __init__(
        self,
        folder_path: list[str] | list[Path],
        key: str,
        search_file_tree_kwargs: dict | None = None,
        validate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.folder_path = folder_path
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
            desc="Checking dataset files on validity (USBMD format)",
        ):
            try:
                validate_file(file_path)
            except Exception as e:
                validation_error_log.append(
                    f"File {file_path} is not a valid USBMD dataset.\n{e}\n"
                )
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
                    f"Could not write validation errors to {validation_error_file_path}."
                    f"\n{e}"
                )
            return

        # Create the validated flag file
        self._write_validation_file(self.folder_path, num_frames_per_file)
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
            f"<usbmd.data.datasets.Folder at 0x{id(self):x}: "
            f"{self.n_files} files, key='{self.key}', folder='{self.folder_path}'>"
        )

    def __str__(self):
        return f"Folder with {self.n_files} files in '{self.folder_path}' (key='{self.key}')"


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
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                if file_path.is_dir():
                    folder = Folder(
                        file_path, self.key, self.search_file_tree_kwargs, self.validate
                    )
                    file_paths += folder.file_paths
                    file_shapes += folder.file_shapes
                    del folder
                elif file_path.is_file():
                    file_paths.append(file_path)
                    with File(file_path) as file:
                        file_shapes.append(file.shape(self.key))
                        if self.validate:
                            file.validate()
                else:
                    raise ValueError(f"File {file_path} is not a file or directory.")
            elif isinstance(file_path, (list, tuple)):
                # If the path is a list, recursively call find_files_and_shapes
                _file_paths, _file_shapes = self.find_files_and_shapes(file_path)
                file_paths += _file_paths
                file_shapes += _file_shapes
            else:
                raise ValueError(f"File {file_path} is not a string or Path object.")

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
            f"<usbmd.data.datasets.Dataset at 0x{id(self):x}: "
            f"{self.n_files} files, key='{self.key}'>"
        )

    def __str__(self):
        return f"Dataset with {self.n_files} files (key='{self.key}')"


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

    assert len(directory_splits) == len(
        directory_list
    ), "Number of directory splits must be equal to the number of directories."
    assert all(
        0 <= split <= 1 for split in directory_splits
    ), "Directory splits must be between 0 and 1."

    # Get directory sizes using the new count function implementation
    directory_counts = count_samples_per_directory(file_names, directory_list)
    directory_sizes = [directory_counts[str(dir_path)] for dir_path in directory_list]

    # take percentage of the files from each directory
    split_indices = [
        int(split * size) for split, size in zip(directory_splits, directory_sizes)
    ]

    # offset split indices by each total number of files
    start_datasets = [0] + list(np.cumsum(directory_sizes))
    split_indices = [
        (start_datasets[i], start_datasets[i] + split)
        for i, split in enumerate(split_indices)
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
        dir_path: sum(1 for f in file_paths if f.startswith(dir_path))
        for dir_path in dir_paths
    }

    # Assert that the total counts match the number of files
    total_count = sum(counts.values())
    assert total_count == len(file_paths), (
        f"Total count of files ({total_count}) does not match the number of files provided "
        f"({len(file_paths)}). Some files may not belong to any of the specified directories."
    )

    return counts
