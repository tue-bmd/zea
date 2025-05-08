"""
H5 dataloader for loading images from USBMD datasets.

This module can be used with any backend.
"""

import copy
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import List

import keras
import numpy as np
from keras import ops
from keras.src.utils import backend_utils

from usbmd.data.datasets import Dataset
from usbmd.data.file import File
from usbmd.data.layers import Resizer
from usbmd.utils import log, map_negative_indices, translate
from usbmd.utils.io_lib import retry_on_io_error

if keras.backend.backend() == "jax":
    from usbmd.backend.jax import on_device_jax as on_device
elif keras.backend.backend() == "tensorflow":
    from usbmd.backend.tensorflow import on_device_tf as on_device
elif keras.backend.backend() == "torch":
    from usbmd.backend.torch import on_device_torch as on_device
else:

    def on_device(func, *args, **kwargs):
        """Dummy function for non-backend specific code."""
        return func(*args, **kwargs)


DEFAULT_IMAGE_RANGE = (0, 255)
DEFAULT_NORMALIZATION_RANGE = (0, 1)
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 0.1


def json_dumps(obj):
    """Used to serialize objects that contain range and slice objects.
    Args:
        obj: object to serialize (most likely a dictionary).
    Returns:
        str: serialized object (json string).
    """
    return json.dumps(obj, cls=USBMDJSONEncoder)


def json_loads(obj):
    """Used to deserialize objects that contain range and slice objects.
    Args:
        obj: object to deserialize (most likely a json string).
    Returns:
        object: deserialized object (dictionary).
    """
    return json.loads(obj, object_hook=_usbmd_datasets_json_decoder)


def decode_file_info(file_info):
    """Decode file info from a json string.
    A batch of H5Generator can return a list of file_info that are json strings.
    This function decodes the json strings and returns a list of dictionaries
    with the information, namely:
    - full_path: full path to the file
    - file_name: file name
    - indices: indices used to extract the image from the file
    """

    if file_info.ndim == 0:
        file_info = [file_info]

    decoded_info = []
    for info in file_info:
        info = ops.convert_to_numpy(info)[()].decode("utf-8")
        decoded_info.append(json_loads(info))
    return decoded_info


def _usbmd_datasets_json_decoder(dct):
    """Wrapper for json.loads to decode range and slice objects."""
    if "__type__" in dct:
        if dct["__type__"] == "range":
            return range(dct["start"], dct["stop"], dct["step"])
        if dct["__type__"] == "slice":
            return slice(dct["start"], dct["stop"], dct["step"])
    return dct


def generate_h5_indices(
    file_names: List[str],
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
        file_names (list): list of file names.
        file_shapes (list): list of file shapes.
        n_frames (int): number of frames to load from each hdf5 file.
        frame_index_stride (int): interval between frames to load.
        key (str, optional): key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): axis to iterate over. Defaults to 0.
        additional_axes_iter (list, optional): additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): will take n_frames from sequence, then move by 1.
            Defaults to False.
        limit_n_frames (int, optional): limit the number of frames to load from each file. This
            means n_frames per data file will be used. These will be the first frames in the file.
            Defaults to None

    Returns:
        list: list of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.
            example: [
                ('/folder/path_to_file.hdf5', 'data/image', [range(0, 1), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ('/folder/path_to_file.hdf5', 'data/image', [range(1, 2), slice(None, 256, None), slice(None, 256, None)]), # pylint: disable=line-too-long
                ...
            ]

    """
    if not limit_n_frames:
        limit_n_frames = np.inf

    assert len(file_names) == len(
        file_shapes
    ), "file_names and file_shapes must have same length"

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
            indices_sorting_file_names = sorted(
                range(len(file_names)),
                key=lambda i: int(re.findall(r"\d+", file_names[i])[-2]),
            )
            file_names = [file_names[i] for i in indices_sorting_file_names]
            file_shapes = [file_shapes[i] for i in indices_sorting_file_names]
        except:
            log.warning("H5Generator: Could not sort file_names by number.")

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
                range(i, i + block_size, frame_index_stride)
                for i in range(0, n_frames_in_file - block_size + 1, block_step_size)
            ]
            yield [indices]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(
        file_names, file_shapes, list(axis_indices_files())
    ):
        # remove all the files that have empty list at initial_frame_axis
        # this can happen if the file is too small to fit a block
        if not axis_indices[0]:  # initial_frame_axis is the first entry in axis_indices
            skipped_files += 1
            continue

        if additional_axes_iter:
            axis_indices += [range(shape[axis]) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
            indices.append((file, key, full_indices))

    if skipped_files > 0:
        log.warning(
            f"H5Generator: Skipping {skipped_files} files with not enough frames "
            f"which is about {skipped_files / len(file_names) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _h5_reopen_on_io_error(
    dataloader_obj,
    indices,
    exception,  # pylint: disable=unused-argument
    retry_count,
):
    """Reopen the file if an I/O error occurs.
    Also removes the file from the cache and try to close file.
    """
    file_name = indices[0]
    try:
        file = dataloader_obj._file_handle_cache.pop(file_name, None)
        if file is not None:
            file.close()
    except Exception:
        pass

    log.warning(
        f"H5Generator: I/O error occurred while reading file {file_name}. "
        f"Retry opening file. Retry count: {retry_count}."
    )


class H5Generator(Dataset, keras.utils.PyDataset):
    """Generator from h5 file using provided indices."""

    def __init__(
        self,
        directory: str | List[str] = None,
        file_names: List[str] = None,
        file_shapes: List[tuple] = None,
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = True,
        initial_frame_axis: int = 0,
        return_filename: bool = False,
        key: str = "data/image",
        shuffle: bool = True,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        seed: int | None = None,
        batch_size: int = 1,
        as_tensor: bool = True,
        **kwargs,
    ):
        assert (directory is not None) ^ (
            file_names is not None and file_shapes is not None
        ), "Either `directory` or `file_names` and `file_shapes` must be provided."

        super().__init__(path=directory, key=key, **kwargs)

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
        self.batch_size = batch_size
        self.as_tensor = as_tensor

        self.maybe_tensor = ops.convert_to_tensor if self.as_tensor else lambda x: x

        assert (
            self.frame_index_stride > 0
        ), f"`frame_index_stride` must be greater than 0, got {self.frame_index_stride}"
        assert (
            self.n_frames > 0
        ), f"`n_frames` must be greater than 0, got {self.n_frames}"

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
            file_names=self.file_names,
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

        # Retry count for I/O errors
        self.retry_count = 0

    def __repr__(self):
        return f"{self.__class__.__name__} containing {len(self)} batches."

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            self._shuffle()

        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.indices))
        indices_list = self.indices[low:high]

        images = []
        filenames = []
        for file_name, key, indices in indices_list:
            file = self.get_file(file_name)
            images.append(self.load(file, key, indices))
            filenames.append(
                json_dumps(
                    {
                        "fullpath": file.filename,
                        "filename": file.stem,
                        "indices": indices,
                    }
                )
            )

        if self.batch_size == 1:
            images = images[0]
        else:
            images = np.stack(images)

        if self.batch_size == 1:
            filenames = filenames[0]

        if self.return_filename:
            return self.maybe_tensor(images), filenames
        else:
            return self.maybe_tensor(images)

    @retry_on_io_error(
        max_retries=MAX_RETRY_ATTEMPTS,
        initial_delay=INITIAL_RETRY_DELAY,
        retry_action=_h5_reopen_on_io_error,
    )
    # TODO: move retry to File?
    def load(self, file: File, key: str, indices: tuple | str):
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
        self.rng.shuffle(self.indices)
        log.info("H5Generator: Shuffled data.")

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


class H5Dataloader(H5Generator):
    """Dataloader for h5 files. Can resize images and normalize them."""

    def __init__(
        self,
        resize_type: str = "center_crop",
        image_size: tuple | None = None,
        image_range: tuple = DEFAULT_IMAGE_RANGE,
        normalization_range: tuple = DEFAULT_NORMALIZATION_RANGE,
        resize_kwargs: dict = None,
        map_fns: list = None,
        augmentation: callable = None,
        assert_image_range: bool = True,
        clip_image_range: bool = False,
        backend=None,
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resize_type = resize_type
        self.image_size = image_size
        self.image_range = image_range
        self.normalization_range = normalization_range
        self.resize_kwargs = resize_kwargs or {}
        self.resizer = Resizer(
            resize_type=resize_type,
            image_size=image_size,
            seed=self.seed,
            **self.resize_kwargs,
        )
        self.map_fns = map_fns or []
        if augmentation is not None:
            self.map_fns.append(augmentation)
        self.assert_image_range = assert_image_range
        self.clip_image_range = clip_image_range
        if backend is None:
            backend = keras.backend.backend()
        self.backend = backend_utils.DynamicBackend(backend)
        self.device = device

    def map(self, fn):
        """Add a mapping function to the dataloader.

        Args:
            fn (callable): Function to map over the images.

        Example usage:
            dataloader = dataloader.map(lambda x: x / 255)

        """
        dl = copy.copy(self)
        dl.map_fns.append(fn)
        return dl

    def preprocess(self, out):
        """Preprocess the images such as resizing and normalizing them."""
        if self.return_filename:
            images, filenames = out
        else:
            images = out

        # add channel dim
        if len(self.shape) != 3:
            images = self.backend.numpy.expand_dims(images, axis=-1)

        # Check if there are outliers in the image range
        if self.assert_image_range:
            minval = self.backend.numpy.min(images)
            maxval = self.backend.numpy.max(images)
            assert self.image_range[0] <= minval and maxval <= self.image_range[1], (
                f"Image range {self.image_range} is not in the range of the data "
                f"{minval} - {maxval}"
            )

        # Clip to image range
        if self.clip_image_range:
            images = self.backend.numpy.clip(images, *self.image_range[0])

        # Translate to normalization range
        images = translate(images, self.image_range, self.normalization_range)

        # resize
        if self.image_size is not None:
            images = self.resizer(images)

        for map_fn in self.map_fns:
            images = map_fn(images)

        if self.return_filename:
            return images, filenames
        else:
            return images

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if self.device is None:
            return self.preprocess(out)
        else:
            return on_device(self.preprocess, out, self.device)


class USBMDJSONEncoder(json.JSONEncoder):
    """Wrapper for json.dumps to encode range and slice objects.

    Example:
        >>> json.dumps(range(10), cls=USBMDJSONEncoder)
        '{"__type__": "range", "start": 0, "stop": 10, "step": 1}'

    Note:
        Probably you would use the `usbmd.data.dataloader.json_dumps()`
        function instead of using this class directly.
    """

    def default(self, o):
        if isinstance(o, range):
            return {
                "__type__": "range",
                "start": o.start,
                "stop": o.stop,
                "step": o.step,
            }
        if isinstance(o, slice):
            return {
                "__type__": "slice",
                "start": o.start,
                "stop": o.stop,
                "step": o.step,
            }
        if isinstance(o, Path):
            return str(o)
        return super().default(o)
