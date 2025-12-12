"""
Script to convert the EchoNet database to zea format.

.. note::
    Will segment the images and convert them to polar coordinates.

For more information about the dataset, resort to the following links:

- The original dataset can be found at `this link <https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a>`_.
- The project page is available `here <https://echonet.github.io/>`_.

"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import griddata
from tqdm import tqdm

from zea import log
from zea.data import generate_zea_dataset
from zea.data.convert.utils import load_avi, unzip
from zea.tensor_ops import translate


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


def cartesian_to_polar_matrix(
    cartesian_matrix, tip=(61, 7), r_max=107, angle=0.79, interpolation="nearest"
):
    """
    Function that converts a timeseries of a cartesian cone to a polar representation
    that is more compatible with CNN's/action selection.

    Args:
        - cartesian_matrix (2d array): (rows, cols) matrix containing time sequence
            of image_sc data.
        - tip (tuple, optional): coordinates (in indices) of the tip of the cone.
            Defaults to (61, 7).
        - r_max (int, optional): expected radius of the cone. Defaults to 107.
        - angle (float, optional): expected angle of the cone, will be used as (-angle, angle).
            Defaults to 0.79.
        - interpolation (str, optional): can be [nearest, linear, cubic]. Defaults to 'nearest'.

    Returns:
        polar_matrix (2d array): polar conversion of the input.
    """
    rows, cols = cartesian_matrix.shape
    center_x, center_y = tip

    # Create cartesian coordinates of the image data
    x = np.linspace(-center_x, cols - center_x - 1, cols)
    y = np.linspace(-center_y, rows - center_y - 1, rows)
    x, y = np.meshgrid(x, y)

    # Flatten the grid and values
    data_points = np.column_stack((x.ravel(), y.ravel()))
    data_points = rotate_coordinates(data_points, -90)
    data_values = cartesian_matrix.ravel()

    # Define new points to sample from in the region of the data.
    # R_max and Theta are found manually. R_max differs from the number of rows in EchoNet!
    r = np.linspace(0, r_max, rows)
    theta = np.linspace(-angle, angle, cols)
    r, theta = np.meshgrid(r, theta)

    x_polar = r * np.cos(theta)
    y_polar = r * np.sin(theta)
    new_points = np.column_stack((x_polar.ravel(), y_polar.ravel()))

    # Interpolate and reshape to 2D matrix
    polar_values = griddata(
        data_points, data_values, new_points, method=interpolation, fill_value=0
    )
    polar_matrix = np.rot90(polar_values.reshape(cols, rows), k=-1)
    return polar_matrix


def find_split_for_file(file_dict, target_file):
    """
    Locate which split contains a given filename.

    Parameters:
        file_dict (dict): Mapping from split name (e.g., "train", "val", "test", "rejected")
            to an iterable of filenames.
        target_file (str): Filename to search for within the split lists.

    Returns:
        str: The split name that contains `target_file`, or `"rejected"` if the file is not found.
    """
    for split, files in file_dict.items():
        if target_file in files:
            return split
    log.warning(f"File {target_file} not found in any split, defaulting to rejected.")
    return "rejected"


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
    ):
        self.path_out_h5 = Path(path_out_h5)
        self.num_val = num_val
        self.num_test = num_test
        self.range_from = range_from
        self.range_to = range_to
        self.splits = splits
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
        found for `hdf5_file` (and asserts that the acceptance result matches the split).
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
            assert accepted == (split != "rejected"), "Rejection mismatch"
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
        to polar coordinates.
        Constructs and returns the zea dataset descriptor used by
        generate_zea_dataset; the descriptor always includes `path`, `image_sc`,
        `probe_name`, and `description`, and includes `image` when the file is accepted.

        Args:
            avi_file (pathlib.Path): Path to the source .avi file to process.

        Returns:
            dict: The value returned by generate_zea_dataset containing the dataset
                entry for the processed file.
        """
        hdf5_file = avi_file.stem + ".hdf5"
        sequence = load_avi(avi_file)

        assert sequence.min() >= self.range_from[0], f"{sequence.min()} < {self.range_from[0]}"
        assert sequence.max() <= self.range_from[1], f"{sequence.max()} > {self.range_from[1]}"

        # Translate to [0, 1]
        sequence = translate(sequence, self.range_from, self._process_range)

        sequence = segment(sequence, number_erasing=0, min_clip=0)

        split = self.get_split(hdf5_file, sequence)
        accepted = split != "rejected"

        out_h5 = self.path_out_h5 / split / hdf5_file

        polar_im_set = []
        for _, im in enumerate(sequence):
            if not accepted:
                continue

            polar_im = cartesian_to_polar_matrix(im, interpolation="cubic")
            polar_im = np.clip(polar_im, *self._process_range)
            polar_im_set.append(polar_im)

        if accepted:
            polar_im_set = np.stack(polar_im_set, axis=0)

        # Check the ranges
        assert sequence.min() >= self._process_range[0], sequence.min()
        assert sequence.max() <= self._process_range[1], sequence.max()

        zea_dataset = {
            "path": out_h5,
            "image_sc": self._translate(sequence),
            "probe_name": "generic",
            "description": "EchoNet dataset converted to zea format",
        }
        if accepted:
            zea_dataset["image"] = self._translate(polar_im_set)
        return generate_zea_dataset(**zea_dataset)


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
            - split_path (str|Path|None): If provided, must contain a split.yaml to reproduce
                an existing split; function asserts the file exists.
            - no_hyperthreading (bool): When false, processing uses a ProcessPoolExecutor
                with a shared counter; when true, processing runs sequentially.

    Note:
        - May unzip the source into a working directory.
        - Writes zea files into dst.
        - Writes a split.yaml into dst summarizing produced files per split.
        - Logs progress and validation results.
        - Asserts that split.yaml exists at split_path when split reproduction is requested.
    """
    # Check if unzip is needed
    src = unzip(args.src, "echonet")

    if args.split_path is not None:
        # Reproduce a previous split...
        yaml_file = Path(args.split_path) / "split.yaml"
        assert yaml_file.exists(), f"File {yaml_file} does not exist."
        splits = {"train": None, "val": None, "test": None, "rejected": None}
        with open(yaml_file, "r") as f:
            splits = yaml.safe_load(f)
        log.info(f"Processor initialized with train-val-test split from {yaml_file}.")
    else:
        splits = None

    # List the files that have an entry in path_out_h5 already
    files_done = []
    for _, _, filenames in os.walk(args.dst):
        for filename in filenames:
            files_done.append(filename.replace(".hdf5", ""))

    # List all files of echonet and exclude those already processed
    path_in = Path(src)
    h5_files = path_in.glob("*.avi")
    h5_files = [file for file in h5_files if file.stem not in files_done]
    log.info(f"Files left to process: {len(h5_files)}")

    # Run the processor
    processor = H5Processor(path_out_h5=args.dst, splits=splits)

    log.info("Starting the conversion process.")

    if not args.no_hyperthreading:
        shared_counter = Value("i", 0)
        with ProcessPoolExecutor(initializer=count_init, initargs=(shared_counter,)) as executor:
            futures = [executor.submit(processor, file) for file in h5_files]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception:
                    log.warning("Task raised an exception")
    else:
        # Initialize global variable for counting
        count_init(Value("i", 0))
        for file in tqdm(h5_files):
            processor(file)

    log.info("All tasks are completed.")

    # Write to yaml split files
    full_list = {}
    for split in ["train", "val", "test", "rejected"]:
        split_dir = Path(args.dst) / split

        # Get only files (skip directories)
        file_list = [f.name for f in split_dir.iterdir() if f.is_file()]
        full_list[split] = file_list

    with open(Path(args.dst) / "split.yaml", "w") as f:
        yaml.dump(full_list, f)

    # Validate that the split was copied correctly
    processor.validate_split_copy(Path(args.dst) / "split.yaml")
