"""
Script to convert the EchoNet database to .npy and USBMD formats.
Will segment the images and convert them to polar coordinates.
"""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from usbmd.config import Config
from usbmd.data import generate_usbmd_dataset
from usbmd.utils.io_lib import load_video
from usbmd.utils.utils import translate


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert EchoNet to USBMD format")
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/EchoNet-Dynamic/Videos",
    )
    parser.add_argument(
        "--output", type=str, default="/mnt/z/Ultrasound-BMd/data/Wessel/echonet_v2025"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/EchoNet-Dynamic/splits",
    )
    parser.add_argument("--output_numpy", type=str, default=None)
    parser.add_argument("--no_hyperthreading", action="store_true")
    args = parser.parse_args()
    return args


def segment(tensor, number_erasing=0, min_clip=0):
    """Segments the background of the echonet images by setting it to 0 and creating a hard edge.

    Args:
        tensor (ndarray): Input image (sc) with 3 dimensions. (N, 112, 112)
        number_erasing (float, optional): number to fill the background with.
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


def find_key_for_file(file_dict, target_file):
    for key, files in file_dict.items():
        if target_file in files:
            return key
    return "rejected"


class H5Processor:
    """
    Stores a few variables and paths to allow for hyperthreading.
    """

    def __init__(
        self,
        path_out_h5,
        path_out=None,
        num_val=500,
        num_test=500,
        range_from=(0, 255),
        range_to=(-60, 0),
        splits=None,
    ):
        self.path_out_h5 = Path(path_out_h5)
        self.path_out = Path(path_out) if path_out else None
        self.num_val = num_val
        self.num_test = num_test
        self.range_from = range_from
        self.range_to = range_to
        self.splits = splits
        self._process_range = (0, 1)

        # Ensure train, val, test, rejected paths exist
        for folder in ["train", "val", "test", "rejected"]:
            if self._to_numpy:
                (self.path_out / folder).mkdir(parents=True, exist_ok=True)
            (self.path_out_h5 / folder).mkdir(parents=True, exist_ok=True)

    @property
    def _to_numpy(self):
        return self.path_out is not None

    def translate(self, tensor):
        return translate(tensor, self._process_range, self.range_to)

    def get_split(self, hdf5_file: str, tensor):
        # Always check acceptance
        accepted = accept_shape(tensor[0])

        # Previous split
        if self.splits is not None:
            split = find_key_for_file(self.splits, hdf5_file)
            assert accepted == (split != "rejected"), "Rejection mismatch"
            return split

        # New split
        if not accepted:
            return "rejected"

        # This inefficient counter works with hyperthreading
        # TODO: but it is not reproducible!
        val_counter = len(list((self.path_out_h5 / "val").iterdir()))
        test_counter = len(list((self.path_out_h5 / "test").iterdir()))

        # Determine the split
        if val_counter < self.num_val:
            return "val"
        elif test_counter < self.num_test:
            return "test"
        else:
            return "train"

    def __call__(self, avi_file):
        """
        Processes a single h5 file using the class variables and the filename given.
        """
        hdf5_file = avi_file.stem + ".hdf5"
        tensor = load_video(avi_file)

        assert (
            tensor.min() >= self.range_from[0]
        ), f"{tensor.min()} < {self.range_from[0]}"
        assert (
            tensor.max() <= self.range_from[1]
        ), f"{tensor.max()} > {self.range_from[1]}"

        # Translate to [0, 1]
        tensor = translate(tensor, self.range_from, self._process_range)

        tensor = segment(tensor, number_erasing=0, min_clip=0)

        split = self.get_split(hdf5_file, tensor)
        accepted = split != "rejected"

        out_h5 = self.path_out_h5 / split / hdf5_file
        if self._to_numpy:
            out_dir = self.path_out / split / avi_file.stem
            out_dir.mkdir(parents=True, exist_ok=True)

        polar_im_set = []
        for i, im in enumerate(tensor):
            if self._to_numpy:
                np.save(out_dir / f"sc{str(i).zfill(3)}.npy", im)

            if not accepted:
                continue

            polar_im = cartesian_to_polar_matrix(im, interpolation="cubic")
            if self._to_numpy:
                np.save(
                    out_dir / f"polar{str(i).zfill(3)}.npy",
                    polar_im,
                )
            polar_im_set.append(polar_im)
        polar_im_set = np.stack(polar_im_set, axis=0)

        usbmd_dataset = {
            "path": out_h5,
            "image": self.translate(tensor),
            "probe_name": "generic",
            "description": "EchoNet dataset converted to USBMD format",
        }
        if accepted:
            usbmd_dataset["image_sc"] = self.translate(polar_im_set)
        return generate_usbmd_dataset(**usbmd_dataset)


if __name__ == "__main__":
    args = get_args()

    if args.splits is not None:
        # Reproduce a previous split...
        split_yaml_dir = Path(args.splits)
        splits = {"train": None, "val": None, "test": None}
        for split in splits:
            yaml_file = split_yaml_dir / (split + ".yaml")
            assert yaml_file.exists(), f"File {yaml_file} does not exist."
            splits[split] = Config.load_from_yaml(yaml_file)["file_paths"]
    else:
        splits = None

    # List the files that have an entry in path_out_h5 already
    files_done = []
    for _, _, filenames in os.walk(args.output):
        for filename in filenames:
            files_done.append(filename.replace(".hdf5", ""))

    # List all files of echonet and exclude those already processed
    path_in = Path(args.source)
    h5_files = path_in.glob("*.avi")
    h5_files = [file for file in h5_files if file.stem not in files_done]
    print(f"Files left to process: {len(h5_files)}")

    # Run the processor
    processor = H5Processor(
        path_out_h5=args.output, path_out=args.output_numpy, splits=splits
    )

    if not args.no_hyperthreading:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(processor, h5_files), total=len(h5_files)))
    else:
        for file in tqdm(h5_files):
            processor(file)

    print("All tasks are completed.")
