"""
H5 dataloader for loading images from zea datasets.
"""

import re
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from zea import log
from zea.data.datasets import Dataset, H5FileHandleCache, count_samples_per_directory
from zea.data.file import File
from zea.data.utils import json_dumps
from zea.io_lib import retry_on_io_error
from zea.utils import map_negative_indices

DEFAULT_NORMALIZATION_RANGE = (0, 1)
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 0.1


def generate_h5_indices(
    file_paths: List[str],
    file_shapes: list,
    n_frames: int,
    frame_index_stride: int,
    key: str = "data/image",
    initial_frame_axis: int = 0,
    additional_axes_iter: List[int] | None = None,
    sort_files: bool = True,
    overlapping_blocks: bool = False,
    limit_n_frames: int | None = None,
):
    """Generate indices for h5 files.

    Generates a list of indices to extract images from hdf5 files. Length of this list
    is the length of the extracted dataset.

    Args:
        file_paths (list): List of file paths.
        file_shapes (list): List of file shapes.
        n_frames (int): Number of frames to load from each hdf5 file.
        frame_index_stride (int): Interval between frames to load.
        key (str, optional): Key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): Axis to iterate over. Defaults to 0.
        additional_axes_iter (list, optional): Additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): Sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): Will take n_frames from sequence, then move by 1.
            Defaults to False.
        limit_n_frames (int, optional): Limit the number of frames to load from each file. This
            means n_frames per data file will be used. These will be the first frames in the file.
            Defaults to None.

    Returns:
        list: List of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.

    Example:
        .. code-block:: python

            [
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (range(0, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (range(1, 2), slice(None, 256, None), slice(None, 256, None)),
                ),
                ...,
            ]
    """
    if not limit_n_frames:
        limit_n_frames = np.inf

    assert len(file_paths) == len(file_shapes), "file_paths and file_shapes must have same length"

    if additional_axes_iter:
        # cannot contain initial_frame_axis
        assert initial_frame_axis not in additional_axes_iter, (
            "initial_frame_axis cannot be in additional_axes_iter. "
            "We are already iterating over that axis."
        )
    else:
        additional_axes_iter = []

    if sort_files:
        try:
            # this is like an np.argsort, returns the indices that would sort the array
            indices_sorting_file_paths = sorted(
                range(len(file_paths)),
                key=lambda i: int(re.findall(r"\d+", file_paths[i])[-2]),
            )
            file_paths = [file_paths[i] for i in indices_sorting_file_paths]
            file_shapes = [file_shapes[i] for i in indices_sorting_file_paths]
        except Exception:
            log.warning("H5Generator: Could not sort file_paths by number.")

    # block size with stride included
    block_size = n_frames * frame_index_stride

    if not overlapping_blocks:
        block_step_size = block_size
    else:
        # now blocks overlap by n_frames - 1
        block_step_size = 1

    def axis_indices_files():
        # For every file
        for shape in file_shapes:
            n_frames_in_file = shape[initial_frame_axis]
            # Optionally limit frames to load from each file
            n_frames_in_file = min(n_frames_in_file, limit_n_frames)
            indices = [
                list(range(i, i + block_size, frame_index_stride))
                for i in range(0, n_frames_in_file - block_size + 1, block_step_size)
            ]
            yield [indices]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(file_paths, file_shapes, list(axis_indices_files())):
        # remove all the files that have empty list at initial_frame_axis
        # this can happen if the file is too small to fit a block
        if not axis_indices[0]:  # initial_frame_axis is the first entry in axis_indices
            skipped_files += 1
            continue

        if additional_axes_iter:
            axis_indices += [list(range(shape[axis])) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
            indices.append((file, key, tuple(full_indices)))

    if skipped_files > 0:
        log.warning(
            f"H5Generator: Skipping {skipped_files} files with not enough frames "
            f"which is about {skipped_files / len(file_paths) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _h5_reopen_on_io_error(
    dataloader_obj: H5FileHandleCache,
    file,
    key,
    indices,
    retry_count,
    **kwargs,
):
    """Reopen the file if an I/O error occurs.
    Also removes the file from the cache and try to close file.
    """
    file_name = indices[0]
    try:
        file_handle = dataloader_obj._file_handle_cache.pop(file_name, None)
        if file_handle is not None:
            file_handle.close()
    except Exception:
        pass

    log.warning(
        f"H5Generator: I/O error occurred while reading file {file_name}. "
        f"Retry opening file. Retry count: {retry_count}."
    )


class H5Generator(Dataset):
    """H5Generator class for iterating over hdf5 files in an advanced way.
    Mostly used internally, you might want to use the Dataloader class instead.
    Loads one item at a time. Always outputs numpy arrays.
    """

    def __init__(
        self,
        file_paths: List[str],
        key: str = "data/image",
        n_frames: int = 1,
        shuffle: bool = True,
        return_filename: bool = False,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        seed: int | None = None,
        cache: bool = False,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        initial_frame_axis: int = 0,
        insert_frame_axis: bool = True,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        validate: bool = True,
        **kwargs,
    ):
        super().__init__(file_paths, key, validate=validate, **kwargs)

        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = int(initial_frame_axis)
        self.return_filename = return_filename
        self.shuffle = shuffle
        self.sort_files = sort_files
        self.overlapping_blocks = overlapping_blocks
        self.limit_n_samples = limit_n_samples
        self.limit_n_frames = limit_n_frames
        self.seed = seed
        self.additional_axes_iter = additional_axes_iter or []

        assert self.frame_index_stride > 0, (
            f"`frame_index_stride` must be greater than 0, got {self.frame_index_stride}"
        )
        assert self.n_frames > 0, f"`n_frames` must be greater than 0, got {self.n_frames}"

        # Extract some general information about the dataset
        image_shapes = np.array(self.file_shapes)
        image_shapes = np.delete(
            image_shapes, (self.initial_frame_axis, *self.additional_axes_iter), axis=1
        )
        n_dims = len(image_shapes[0])

        self.equal_file_shapes = np.all(image_shapes == image_shapes[0])
        if not self.equal_file_shapes:
            log.warning(
                "H5Generator: Not all files have the same shape. "
                "This can lead to issues when resizing images later...."
            )
            self.shape = np.array([None] * n_dims)
        else:
            self.shape = np.array(image_shapes[0])

        if insert_frame_axis:
            _frame_axis = map_negative_indices([frame_axis], len(self.shape) + 1)
            self.shape = np.insert(self.shape, _frame_axis, 1)
        if self.shape[frame_axis]:
            self.shape[frame_axis] = self.shape[frame_axis] * n_frames

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)

        self.indices = generate_h5_indices(
            file_paths=self.file_paths,
            file_shapes=self.file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=self.sort_files,
            overlapping_blocks=self.overlapping_blocks,
            limit_n_frames=self.limit_n_frames,
        )

        if not self.shuffle:
            log.warning("H5Generator: Not shuffling data.")

        if limit_n_samples:
            log.warning(
                f"H5Generator: Limiting number of samples to {limit_n_samples} "
                f"out of {len(self.indices)}"
            )
            self.indices = self.indices[:limit_n_samples]

        self.shuffled_items = list(range(len(self.indices)))

        # Retry count for I/O errors
        self.retry_count = 0

        # Create a cache for the data
        self.cache = cache
        self._data_cache = {}

    def _get_single_item(self, idx):
        # Check if the item is already in the cache
        if self.cache and idx in self._data_cache:
            return self._data_cache[idx]

        # Get the data
        file_name, key, indices = self.indices[idx]
        file = self.get_file(file_name)
        image = self.load(file, key, indices)
        file_data = json_dumps(
            {
                "fullpath": file.filename,
                "filename": file.stem,
                "indices": indices,
            }
        )

        if self.cache:
            # Store the image and file data in the cache
            self._data_cache[idx] = [image, file_data]

        return image, file_data

    def __getitem__(self, index):
        image, file_data = self._get_single_item(self.shuffled_items[index])

        if self.return_filename:
            return image, file_data
        else:
            return image

    @retry_on_io_error(
        max_retries=MAX_RETRY_ATTEMPTS,
        initial_delay=INITIAL_RETRY_DELAY,
        retry_action=_h5_reopen_on_io_error,
    )
    def load(
        self,
        file: File,
        key: str,
        indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    ):
        """Extract data from hdf5 file.
        Args:
            file_name (str): name of the file to extract image from.
            key (str): key of the hdf5 dataset to grab data from.
            indices (tuple): indices to extract image from (tuple of slices)
        Returns:
            np.ndarray: image extracted from hdf5 file and indexed by indices.
        """
        try:
            images = file.load_data(key, indices)
        except (OSError, IOError):
            # Let the decorator handle I/O errors
            raise
        except Exception as exc:
            # For non-I/O errors, provide detailed context
            raise ValueError(
                f"Could not load image at index {indices} "
                f"and file {file.name} of shape {file[key].shape}"
            ) from exc

        # stack frames along frame_axis
        if self.insert_frame_axis:
            # move frames axis to self.frame_axis
            initial_frame_axis = self.initial_frame_axis
            if self.additional_axes_iter:
                # offset initial_frame_axis if we have additional axes that are before
                # the initial_frame_axis
                additional_axes_before = sum(
                    axis < self.initial_frame_axis for axis in self.additional_axes_iter
                )
                initial_frame_axis = initial_frame_axis - additional_axes_before

            images = np.moveaxis(images, initial_frame_axis, self.frame_axis)
        else:
            # append frames to existing axis
            images = np.concatenate(images, axis=self.frame_axis)

        return images

    def _shuffle(self):
        self.rng.shuffle(self.shuffled_items)
        log.info("H5Generator: Shuffled data.")

    def __len__(self):
        return len(self.indices)

    def iterator(self):
        """Generator that yields images from the hdf5 files."""
        if self.shuffle:
            self._shuffle()
        for idx in range(len(self)):
            yield self[idx]

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        return self.iterator()

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} at 0x{id(self):x}: "
            f"{len(self)} batches, n_frames={self.n_frames}, key='{self.key}', "
            f"shuffle={self.shuffle}, file_paths={len(self.file_paths)}>"
        )

    def __str__(self):
        return (
            f"H5Generator with {len(self)} batches from {len(self.file_paths)} files "
            f"(key='{self.key}')"
        )

    def summary(self):
        """Return a string with dataset statistics and per-directory breakdown."""
        total_samples = len(self.indices)
        file_names = [idx[0] for idx in self.indices]
        # Try to infer directories from file_names
        directories = sorted({str(Path(f).parent) for f in file_names})
        samples_per_dir = count_samples_per_directory(file_names, directories)

        parts = [f"H5Generator with {total_samples} total samples:"]
        for dir_path, count in samples_per_dir.items():
            percentage = (count / total_samples) * 100 if total_samples else 0
            parts.append(f"  {dir_path}: {count} samples ({percentage:.1f}%)")
        print("\n".join(parts))
