"""
Script to convert the EchoNet database to zea format.

.. note::
    Will segment the images and convert them to polar coordinates.

For more information about the dataset, resort to the following links:

- The original dataset can be found at `this link <https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a>`_.
- The project page is available `here <https://echonet.github.io/dynamic/>`_.

"""

import collections
import functools
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm

from zea import log
from zea.data.convert.utils import load_avi, unzip_beside
from zea.data.file import File
from zea.func.tensor import translate
from zea.internal.preset_utils import HF_PREFIX, _hf_resolve_path

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

# Process-shared counter, bound per worker process by ``count_init`` (see below)
# and read by ``H5Processor``. Annotation-only on purpose: the name genuinely
# does not exist until an initializer runs, but declaring it here gives the
# ``global COUNTER`` statement a module-scope declaration to rebind.
COUNTER: "Synchronized[int]"


def segment(tensor, number_erasing=0, min_clip=0):
    """Segments the background of the echonet images by setting it to 0 and creating a hard edge.

    Args:
        tensor (ndarray): Input image (sc) with 3 dimensions. (N, 112, 112)
        number_erasing (float, optional): number to fill the background with.
        min_clip (float, optional): If > 0, values on the computed cone edge will be clipped
            to be at least this value. Defaults to 0.
    Returns:
        tensor (ndarray): Segmented matrix of same dimensions as input

    """
    # Start with the upper part

    # Height of the diagonal lines for the columns [0, 112]
    rows_left = np.linspace(67, 7, 61)
    rows_right = np.linspace(7, 57, 51)
    rows = np.concatenate([rows_left, rows_right], axis=0)
    for idx, row in enumerate(rows.astype(np.int32)):
        # Set everything above the edge to the number_erasing value.
        # Rows count up from 0 to 112 so row-1 is above.
        tensor[:, 0 : row - 1, idx] = number_erasing

        # Set minimum values for the edge
        if min_clip > 0:
            tensor[:, row, idx] = np.clip(tensor[:, row, idx], min_clip, 1)

    # Bottom left curve (manual fit)
    cols_left = np.linspace(0, 20, 21).astype(np.int32)
    rows_left = np.array(
        [
            102,
            103,
            103,
            104,
            104,
            105,
            105,
            106,
            106,
            107,
            107,
            107,
            108,
            108,
            109,
            109,
            109,
            110,
            110,
            111,
            111,
        ]
    )

    # Bottom right curve (manual fit)
    cols_right = np.linspace(89, 111, 23).astype(np.int32)
    rows_right = np.array(
        [
            111,
            111,
            111,
            110,
            110,
            110,
            109,
            109,
            109,
            108,
            108,
            107,
            107,
            107,
            106,
            106,
            105,
            105,
            104,
            104,
            103,
            103,
            102,
        ]
    )

    rows = np.concatenate([rows_left, rows_right], axis=0)
    cols = np.concatenate([cols_left, cols_right], axis=0)

    for row, col in zip(rows, cols):
        # Set everything under the edge to the number_erasing value.
        # Rows count up from 0 to 112 so row-1 is above.
        tensor[:, row:, col] = number_erasing
        # Set minimum values for the edge
        if min_clip > 0:
            tensor[:, row - 1, col] = np.clip(tensor[:, row - 1, col], min_clip, 1)

    return tensor


def accept_shape(tensor):
    """Acceptance algorithm that determines whether to reject an image
    based on left and right corner data.

    Args:
        tensor (ndarray): Input image (sc) with 2 dimensions. (112, 112)

    Returns:
        decision (bool): Whether or not the tensor should be rejected.

    """

    decision = True

    # Test one, check if left bottom corner is populated with values
    rows_lower = np.linspace(78, 47, 21).astype(np.int32)
    rows_upper = np.linspace(67, 47, 21).astype(np.int32)
    counter = 0
    for idx, row in enumerate(rows_lower):
        counter += np.sum(tensor[rows_upper[idx] : row, idx])

    # If it is not populated, reject the image
    if counter < 0.1:
        decision = False

    # Test two, check if the bottom right cornered with values (that are not artifacts)
    cols = np.linspace(70, 111, 42).astype(np.int32)
    rows_bot = np.linspace(17, 57, 42).astype(np.int32)
    rows_top = np.linspace(17, 80, 42).astype(np.int32)

    # List all the values
    counter = []
    for i, col in enumerate(cols):
        counter += [tensor[rows_bot[i] : rows_top[i], col]]

    flattened_counter = [float(item) for sublist in counter for item in sublist]
    # Sort and exclude the first 50 (likely artifacts)
    flattened_counter.sort(reverse=True)
    value = sum(flattened_counter[100:])

    # Reject if the baseline is too low
    if value < 5:
        decision = False

    return decision


def rotate_coordinates(data_points, degrees):
    """Function that rotates the datapoints by a certain degree.

    Args:
        data_points (ndarray): tensor containing [N,2] (x and y) datapoints.
        degrees (int): angle to rotate the datapoints with

    Returns:
       rotated_points (ndarray): the rotated data_points.

    """

    angle_radians = np.radians(degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_points = rotation_matrix @ data_points.T

    return rotated_points.T


@functools.lru_cache
def _polar_grid(rows, cols, tip, r_max, angle):
    # The triangulation only depends on the grid, so it is shared by all frames and files.
    center_x, center_y = tip

    # Create cartesian coordinates of the image data
    x = np.linspace(-center_x, cols - center_x - 1, cols)
    y = np.linspace(-center_y, rows - center_y - 1, rows)
    x, y = np.meshgrid(x, y)
    data_points = np.column_stack((x.ravel(), y.ravel()))
    data_points = rotate_coordinates(data_points, -90)

    # Define new points to sample from in the region of the data.
    # R_max and Theta are found manually. R_max differs from the number of rows in EchoNet!
    r = np.linspace(0, r_max, rows)
    theta = np.linspace(-angle, angle, cols)
    r, theta = np.meshgrid(r, theta)

    x_polar = r * np.cos(theta)
    y_polar = r * np.sin(theta)
    new_points = np.column_stack((x_polar.ravel(), y_polar.ravel()))
    return Delaunay(data_points), new_points


@functools.lru_cache
def _nearest_indices(rows, cols, tip, r_max, angle):
    # What NearestNDInterpolator looks up, without rebuilding its KD-tree on every call.
    tri, new_points = _polar_grid(rows, cols, tip, r_max, angle)
    return cKDTree(tri.points).query(new_points)[1]


def cartesian_to_polar_matrix(
    cartesian_matrix, tip=(61, 7), r_max=107, angle=0.79, interpolation="nearest"
):
    """
    Function that converts a timeseries of a cartesian cone to a polar representation
    that is more compatible with CNN's/action selection.

    Args:
        - cartesian_matrix (ndarray): (rows, cols) scan-converted (Cartesian) image, or a
            (n_frames, rows, cols) time sequence of them.
        - tip (tuple, optional): coordinates (in indices) of the tip of the cone.
            Defaults to (61, 7).
        - r_max (int, optional): expected radius of the cone. Defaults to 107.
        - angle (float, optional): expected angle of the cone, will be used as (-angle, angle).
            Defaults to 0.79.
        - interpolation (str, optional): can be [nearest, linear, cubic]. Defaults to 'nearest'.

    Returns:
        polar_matrix (ndarray): polar conversion of the input, same shape as the input.
    """
    *batch, rows, cols = cartesian_matrix.shape
    tri, new_points = _polar_grid(rows, cols, tuple(tip), r_max, angle)

    # (rows * cols, n_frames) so that all frames are interpolated at once
    data_values = cartesian_matrix.reshape(-1, rows * cols).T

    if interpolation == "nearest":
        polar_values = data_values[_nearest_indices(rows, cols, tuple(tip), r_max, angle)]
    elif interpolation == "linear":
        polar_values = LinearNDInterpolator(tri, data_values, fill_value=0)(new_points)
    elif interpolation == "cubic":
        polar_values = CloughTocher2DInterpolator(tri, data_values, fill_value=0)(new_points)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")

    # Reshape to (n_frames, rows, cols)
    polar_matrix = np.rot90(polar_values.T.reshape(-1, cols, rows), k=-1, axes=(1, 2))
    return polar_matrix.reshape(*batch, rows, cols)


def find_split_for_file(file_dict, target_file):
    """
    Locate which split contains a given filename.

    Parameters:
        file_dict (dict): Mapping from split name (e.g., "train", "val", "test", "rejected")
            to an iterable of filenames.
        target_file (str): Filename to search for within the split lists.

    Returns:
        str: The split name that contains `target_file`, or `"rejected"` if the file is not found.
            Warns about a file that is not found only when `file_dict` lists the rejected
            files explicitly; otherwise unlisted files are rejected by definition.
    """
    for split, files in file_dict.items():
        if target_file in files:
            return split
    if "rejected" in file_dict:
        log.warning(f"File {target_file} not found in any split, defaulting to rejected.")
    return "rejected"


# The published EchoNet-Dynamic split (train.yaml, val.yaml and test.yaml)
ECHONET_SPLIT_PATH = "hf://zeahub/echonet-dynamic"


def _hf_resolve_split(hf_path):
    # Download only the split files
    if hf_path.endswith(".yaml"):
        return _hf_resolve_path(hf_path)
    hf_path = hf_path.rstrip("/")
    try:
        return _hf_resolve_path(f"{hf_path}/split.yaml")
    except FileNotFoundError:
        files = [_hf_resolve_path(f"{hf_path}/{split}.yaml") for split in ["train", "val", "test"]]
        return files[0].parent


def load_split(split_path):
    """Load an EchoNet dataset split.

    Two formats are supported:

    * ``split.yaml``, as written by :func:`convert_echonet`: maps ``train``, ``val``,
      ``test`` and ``rejected`` to lists of file names.
    * ``train.yaml``, ``val.yaml`` and ``test.yaml``, as published on
      `zeahub/echonet-dynamic <https://huggingface.co/datasets/zeahub/echonet-dynamic>`_:
      each has ``file_paths`` and ``file_lengths`` (frames per file). Files that are in
      none of them are rejected.

    Args:
        split_path (str, Path): A folder containing either format, a ``split.yaml``
            file, or an ``hf://`` path to a Hugging Face dataset repository.

    Returns:
        tuple: The split, mapping split names to sets of ``.hdf5`` file names, and the
        expected number of frames per file name (empty for ``split.yaml``).
    """
    text = str(split_path)
    if text.startswith(HF_PREFIX):
        split_path = _hf_resolve_split(text)
    split_path = Path(split_path)

    if split_path.is_file():
        split_path, split_file = split_path.parent, split_path
        if split_file.name != "split.yaml":
            # One of the published per-split files; read all three from its folder.
            split_file = split_path / "split.yaml"
    else:
        split_file = split_path / "split.yaml"

    if split_file.exists():
        with open(split_file, "r") as f:
            splits = yaml.safe_load(f)
        return {split: set(names or []) for split, names in splits.items()}, {}

    splits, frames = {}, {}
    for split in ["train", "val", "test"]:
        yaml_file = split_path / f"{split}.yaml"
        if not yaml_file.exists():
            raise FileNotFoundError(f"Neither split.yaml nor {split}.yaml found in {split_path}.")
        with open(yaml_file, "r") as f:
            content = yaml.safe_load(f)
        names = [Path(p).name for p in content["file_paths"]]
        splits[split] = set(names)
        frames.update(zip(names, content.get("file_lengths", [])))
    return splits, frames


def count_init(shared_counter):
    """
    Initialize the module-level shared counter used by worker processes.

    Parameters:
        shared_counter (multiprocessing.Value): A process-shared integer Value that
            will be assigned to the module-global COUNTER for coordinated counting
            across processes.
    """
    global COUNTER
    COUNTER = shared_counter


class H5Processor:
    """
    Stores a few variables and paths to allow for hyperthreading.
    """

    def __init__(
        self,
        path_out_h5,
        num_val=500,
        num_test=500,
        range_from=(0, 255),
        range_to=(-60, 0),
        splits=None,
        expected_frames=None,
    ):
        self.path_out_h5 = Path(path_out_h5)
        self.num_val = num_val
        self.num_test = num_test
        self.range_from = range_from
        self.range_to = range_to
        self.splits = splits
        # Frames per file name, as listed by the published split
        self.expected_frames = expected_frames or {}
        self._process_range = (0, 1)

        # Ensure train, val, test, rejected paths exist
        for folder in ["train", "val", "test", "rejected"]:
            (self.path_out_h5 / folder).mkdir(parents=True, exist_ok=True)

    def _translate(self, data):
        """Translate the data from the processing range to final range."""
        return translate(data, self._process_range, self.range_to)

    def get_split(self, hdf5_file: str, sequence):
        """
        Determine the dataset split label for a given file and its image sequence.

        This method checks acceptance based on the first frame of `sequence`.
        If explicit splits were provided to the processor, it returns the split
        found for `hdf5_file`, even when the acceptance check disagrees: the check
        thresholds decoded pixels, so a different decoder can flip borderline videos,
        and a published split should not change because of that. Disagreements are
        logged.
        If no explicit splits are provided, rejected sequences are labeled `"rejected"`.
        Accepted sequences increment a shared counter and are assigned
        `"val"`, `"test"`, or `"train"` according to the processor's
        `num_val` and `num_test` quotas.

        Args:
            hdf5_file (str): Filename or identifier used to look up an existing split
                when splits are provided.
            sequence (array-like): Time-ordered sequence of images; the first frame is
                used for acceptance checking.

        Returns:
            str: One of `"train"`, `"val"`, `"test"`, or `"rejected"` indicating the assigned split.
        """
        # Always check acceptance
        accepted = accept_shape(sequence[0])

        # Previous split
        if self.splits is not None:
            split = find_split_for_file(self.splits, hdf5_file)
            if accepted != (split != "rejected"):
                log.warning(
                    "%s: the acceptance check %s it, but the given split puts it in %s. "
                    "Keeping the given split.",
                    hdf5_file,
                    "accepts" if accepted else "rejects",
                    split,
                )
            return split

        # New split
        if not accepted:
            return "rejected"

        # Increment the hyperthreading counter
        # Note that some threads will start on subsequent splits
        # while others are still processing
        with COUNTER.get_lock():
            COUNTER.value += 1
            n = COUNTER.value

        # Determine the split
        if n <= self.num_val:
            return "val"
        elif n <= self.num_val + self.num_test:
            return "test"
        else:
            return "train"

    def validate_split_copy(self, split_file):
        """
        Validate that a generated split YAML matches the original splits provided to the processor.

        Reads the YAML at `split_file` and compares its `train`, `val`, `test`, and `rejected` lists
        (or other split keys present in `self.splits`) against `self.splits`; logs confirmation
        when a split matches and logs which entries are missing or extra when they differ. If the
        processor was not initialized with `splits`, validation is skipped and a message is logged.

        Args:
            split_file (str or os.PathLike): Path to the YAML file containing the
                generated dataset splits.
        """
        if self.splits is not None:
            # Read the split_file and ensure contents of the train, val and split match
            with open(split_file, "r") as f:
                new_splits = yaml.safe_load(f)
            for split in self.splits.keys():
                if set(new_splits[split]) == set(self.splits[split]):
                    log.info(f"Split {split} copied correctly.")
                else:
                    # Log which entry is missing or extra in the split_file
                    missing = set(self.splits[split]) - set(new_splits[split])
                    extra = set(new_splits[split]) - set(self.splits[split])
                    if missing:
                        log.warning(f"New dataset split {split} is missing entries: {missing}")
                    if extra:
                        log.warning(f"New dataset split {split} has extra entries: {extra}")
        else:
            log.info(
                "Processor not initialized with a split, not validating if the split was copied."
            )

    def __call__(self, avi_file):
        """
        Convert a single AVI file into a zea dataset entry.
        Loads the AVI, validates and rescales pixel ranges, applies segmentation,
        assigns a data split (train/val/test/rejected), converts accepted frames
        to polar coordinates and saves as a zea HDF5 file via File.create.

        Args:
            avi_file (pathlib.Path): Path to the source .avi file to process.

        Raises:
            ValueError: If the converted image has non-finite values.
        """
        hdf5_file = avi_file.stem + ".hdf5"
        sequence = load_avi(avi_file)

        expected = self.expected_frames.get(hdf5_file)
        if expected is not None and len(sequence) != expected:
            log.warning(
                "%s has %d frames, but the published split lists %d.",
                hdf5_file,
                len(sequence),
                expected,
            )

        assert sequence.min() >= self.range_from[0], f"{sequence.min()} < {self.range_from[0]}"
        assert sequence.max() <= self.range_from[1], f"{sequence.max()} > {self.range_from[1]}"

        # Translate to [0, 1]
        sequence = translate(sequence, self.range_from, self._process_range)

        sequence = segment(sequence, number_erasing=0, min_clip=0)

        split = self.get_split(hdf5_file, sequence)
        accepted = split != "rejected"

        out_h5 = self.path_out_h5 / split / hdf5_file

        if accepted:
            polar_im_set = cartesian_to_polar_matrix(sequence, interpolation="cubic")
            polar_im_set = np.clip(polar_im_set, *self._process_range)

        # Check the ranges
        assert sequence.min() >= self._process_range[0], sequence.min()
        assert sequence.max() <= self._process_range[1], sequence.max()

        if accepted:
            # Store the polar (pre-scan-conversion) representation as the image.
            polar_db = self._translate(polar_im_set)
            polar_float32 = polar_db.astype(np.float32)
            polar_float32 = np.expand_dims(polar_float32, axis=-1)  # add y dim
            data = {"image": {"values": polar_float32}}
        else:
            # Rejected sequences have no polar representation; store the original
            # scan-converted (Cartesian) frames as the image instead.
            cartesian_db = self._translate(sequence).astype(np.float32)
            data = {"image": {"values": cartesian_db}}

        # Values are clipped to range_to, but NaNs survive clipping
        if not np.isfinite(data["image"]["values"]).all():
            raise ValueError(f"{hdf5_file} has non-finite values.")

        File.create(
            path=out_h5,
            data=data,
            probe={"name": "generic"},
            description="EchoNet dataset converted to zea format",
        )


def _find_videos(folder: Path) -> Path:
    # EchoNet-Dynamic.zip nests everything under a top-level EchoNet-Dynamic/ folder, so
    # unzip_beside yields EchoNet-Dynamic/EchoNet-Dynamic/Videos, while a manual extraction
    # into the folder holding the zip yields EchoNet-Dynamic/Videos.
    for candidate in (folder / "Videos", folder / "EchoNet-Dynamic" / "Videos"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Videos folder found in {folder}.")


def _resolve_path(src: str | Path) -> Path:
    src = Path(src)

    if src.is_file() and src.suffix == ".zip":
        return _find_videos(unzip_beside(src))

    if (src / "Videos").exists():
        return src / "Videos"

    zip_name = "EchoNet-Dynamic.zip"
    folder_name = "EchoNet-Dynamic"

    if (src / folder_name).exists():
        return _find_videos(src / folder_name)

    return _find_videos(unzip_beside(src / zip_name))


def convert_echonet(args):
    """
    Convert an EchoNet dataset into zea files, organizing results
    into train/val/test/rejected splits.

    Args:
        args (argparse.Namespace): An object with the following attributes.

            - src (str|Path): Path to the source archive or directory containing .avi files.
                Will be unzipped if needed.
            - dst (str|Path): Destination directory for generated zea files
                per-split subdirectories (train, val, test, rejected) and a split.yaml
                are created or updated.
            - split_path (str|Path|None, optional): The split to reproduce, in any format
                :func:`load_split` reads. Defaults to the published split,
                :data:`ECHONET_SPLIT_PATH`.
            - new_split (bool, optional): Create a new random split instead of reproducing
                one. Defaults to False.
            - no_hyperthreading (bool): When false, processing uses a ProcessPoolExecutor
                with a shared counter; when true, processing runs sequentially.

    Note:
        - May unzip the source into a working directory.
        - Writes zea files into dst.
        - Writes a split.yaml into dst summarizing produced files per split.
        - Logs progress and validation results, including files whose frame count or
          acceptance check disagrees with the given split.

    Raises:
        RuntimeError: If any file failed to convert. The split.yaml is still written for
            the files that succeeded.
    """
    # Check if unzip is needed
    src = _resolve_path(args.src)

    split_path = getattr(args, "split_path", ECHONET_SPLIT_PATH)
    if getattr(args, "new_split", False) or split_path is None:
        splits, expected_frames = None, {}
        log.info("Creating a new train-val-test split.")
    else:
        # Reproduce a previous split...
        splits, expected_frames = load_split(split_path)
        log.info(f"Processor initialized with train-val-test split from {split_path}.")

    # List the files that have an entry in path_out_h5 already
    files_done = []
    for _, _, filenames in os.walk(args.dst):
        for filename in filenames:
            files_done.append(filename.replace(".hdf5", ""))

    # List all files of echonet and exclude those already processed
    path_in = Path(src)
    h5_files = list(path_in.glob("*.avi"))
    if not h5_files:
        raise FileNotFoundError(f"No .avi files found in {path_in}.")
    h5_files = [file for file in h5_files if file.stem not in files_done]
    log.info(f"Files left to process: {len(h5_files)}")

    # Run the processor
    processor = H5Processor(path_out_h5=args.dst, splits=splits, expected_frames=expected_frames)

    log.info("Starting the conversion process.")

    failures = {}

    def collect(file, run):
        try:
            run()
        except Exception as exc:
            failures[file.name] = exc
            log.error(
                "Failed to convert %s:\n%s",
                log.yellow(file),
                "".join(traceback.format_exception(exc)),
            )

    if not args.no_hyperthreading:
        shared_counter = Value("i", 0)
        with ProcessPoolExecutor(initializer=count_init, initargs=(shared_counter,)) as executor:
            futures = {executor.submit(processor, file): file for file in h5_files}
            for future in tqdm(as_completed(futures), total=len(futures)):
                collect(futures[future], future.result)
    else:
        # Initialize global variable for counting
        count_init(Value("i", 0))
        for file in tqdm(h5_files):
            collect(file, lambda file=file: processor(file))

    log.info("All tasks are completed.")

    # Write to yaml split files
    full_list = {}
    for split in ["train", "val", "test", "rejected"]:
        split_dir = Path(args.dst) / split
        full_list[split] = sorted(f.name for f in split_dir.glob("*.hdf5"))

    # Resumes skip files that exist in any split, so a duplicate was put there by hand.
    counts = collections.Counter(name for names in full_list.values() for name in names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(
            f"{len(duplicates)} files are in more than one split of {args.dst}, e.g. "
            f"{', '.join(duplicates[:5])}. Remove the stale copies and rerun."
        )

    with open(Path(args.dst) / "split.yaml", "w") as f:
        yaml.dump(full_list, f)

    # Validate that the split was copied correctly
    processor.validate_split_copy(Path(args.dst) / "split.yaml")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(h5_files)} files failed to convert; see the errors above."
        )
