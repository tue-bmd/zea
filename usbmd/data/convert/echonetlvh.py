"""
Script to convert the EchoNet-LVH database to USBMD format.
Will segment the images and convert them to polar coordinates.
"""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

import numpy as np
from tqdm import tqdm

from usbmd.data.convert.echonet import H5Processor, segment, cartesian_to_polar_matrix
from usbmd.utils.io_lib import load_video
from usbmd.utils.utils import translate
from usbmd.data import generate_usbmd_dataset


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert EchoNet-LVH to USBMD format")
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/echonetlvh",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh_v2025",
    )
    parser.add_argument("--output_numpy", type=str, default=None)
    parser.add_argument(
        "--use_hyperthreading", action="store_true", help="Enable hyperthreading"
    )
    args = parser.parse_args()
    return args


def load_splits(source_dir):
    """Load splits from MeasurementsList.csv"""
    csv_path = Path(source_dir) / "MeasurementsList.csv"
    df = pd.read_csv(csv_path)

    # Create dictionary of filename to split mapping
    splits = {"train": [], "val": [], "test": []}

    # Group by HashedFileName to get unique files and their splits
    for filename, group in df.groupby("HashedFileName"):
        # Get the split for this file (should all be the same)
        split = group["split"].iloc[0]
        splits[split].append(filename + ".hdf5")

    return splits


def find_avi_file(source_dir, hashed_filename):
    """Find AVI file in any of the batch directories"""
    for batch_dir in Path(source_dir).glob("Batch*"):
        avi_path = batch_dir / f"{hashed_filename}.avi"
        if avi_path.exists():
            return avi_path
    return None


def manual_crop(sequence):
    """Manually crop sequence to remove zero padding for two known shapes."""
    F, H, W = sequence.shape
    # Case 1:
    if (H, W) == (768, 1024):
        return sequence[:, 86:-50, 78:-78]
    # Case 2:
    elif (H, W) == (600, 800):
        return sequence[:, 70:-40, 60:-60]
    # Add more cases as needed
    else:
        raise ValueError(f"Unexpected image shape: {sequence.shape}")


def adaptive_segment(tensor, number_erasing=0, min_clip=0):
    """Segments the background of echonet images adaptively based on image size.

    Args:
        tensor (ndarray): Input image with shape (N, H, W)
        number_erasing (float): Value to fill background with
        min_clip (float): Minimum value for edge pixels
    """
    H, W = tensor.shape[1:]
    center_x = W // 2
    center_y = H // 2

    # Find the cone boundaries by detecting intensity changes
    # Sample columns vertically to find top edge
    top_edge = []
    for x in range(W):
        col = tensor[0, :, x]  # Take first frame
        # Find first significant intensity change from top
        edges = np.where(np.diff(col > 0.1) > 0)[0]
        if len(edges) > 0:
            top_edge.append((x, edges[0]))

    # Sample rows horizontally to find side edges
    side_edges = []
    for y in range(center_y, H):
        row = tensor[0, y, :]
        # Find leftmost and rightmost significant intensity
        edges = np.where(row > 0.1)[0]
        if len(edges) > 1:
            side_edges.append((y, edges[0], edges[-1]))

    # Fit curves to detected edges
    if len(top_edge) > 0 and len(side_edges) > 0:
        # Create mask
        mask = np.ones_like(tensor)

        # Apply top edge mask
        top_x, top_y = zip(*top_edge)
        for x, y in zip(top_x, top_y):
            mask[:, :y, x] = number_erasing
            if min_clip > 0:
                mask[:, y, x] = np.clip(tensor[:, y, x], min_clip, 1)

        # Apply side edge mask
        for y, left, right in side_edges:
            mask[:, y, :left] = number_erasing
            mask[:, y, right:] = number_erasing
            if min_clip > 0:
                mask[:, y, left] = np.clip(tensor[:, y, left], min_clip, 1)
                mask[:, y, right] = np.clip(tensor[:, y, right], min_clip, 1)

        return tensor * mask
    return tensor


def adaptive_accept_shape(tensor):
    """Determines whether to accept an image based on content analysis.

    Args:
        tensor (ndarray): Input image with shape (H, W)
    """
    H, W = tensor.shape
    center_x = W // 2
    center_y = H // 2

    # Check bottom corners for valid data
    left_roi = tensor[center_y:, : W // 4]
    right_roi = tensor[center_y:, 3 * W // 4 :]

    # Compute content metrics
    left_content = np.mean(left_roi > 0.1)
    right_content = np.mean(right_roi > 0.1)

    # Accept if both corners have sufficient content
    return left_content > 0.1 and right_content > 0.1


def adaptive_cartesian_to_polar(cartesian_matrix, angle):
    """Converts cartesian image to polar coordinates adaptively.

    Args:
        cartesian_matrix (ndarray): Input image with shape (H, W)
    """
    H, W = cartesian_matrix.shape
    center_x = W // 2

    # Find tip of cone (assume it's at the top center)
    tip_y = 0
    for y in range(H // 4):
        if np.any(cartesian_matrix[y, center_x - 10 : center_x + 10] > 0.1):
            tip_y = y
            break

    # Use existing polar conversion with adapted parameters
    return cartesian_to_polar_matrix(
        cartesian_matrix,
        tip=(center_x, tip_y),
        r_max=H,
        angle=angle,
        interpolation="cubic",
    )


class LVHProcessor(H5Processor):
    """Modified H5Processor for EchoNet-LVH dataset."""

    def get_split(self, hdf5_file: str, sequence):
        """Override split determination to use predefined splits."""
        # First check acceptance
        # accepted = self.accept_shape(sequence[0])
        # if not accepted:
        #     return "rejected"

        # Use predefined split from csv
        for split, files in self.splits.items():
            if hdf5_file in files:
                return split

        raise UserWarning("Unknown split for file: " + hdf5_file)

    def __call__(self, avi_file):
        """Process a single AVI file."""
        hdf5_file = avi_file.stem + ".hdf5"
        # sequence = load_video(avi_file)
        sequence = load_video(
            "/mnt/z/Ultrasound-BMd/data/USBMD_datasets/_RAW/echonetlvh/Batch4/0XB4FC1CB94F182CE1.avi"
        )

        # Normalize to [0,1]
        sequence = translate(sequence, self.range_from, self._process_range)

        # Segment adaptively
        # sequence = adaptive_segment(sequence)
        sequence = manual_crop(sequence)

        split = self.get_split(hdf5_file, sequence)
        accepted = split != "rejected"

        out_h5 = self.path_out_h5 / split / hdf5_file
        if self._to_numpy:
            out_dir = self.path_out / split / avi_file.stem
            out_dir.mkdir(parents=True, exist_ok=True)

        polar_im_set = []
        for i, im in enumerate(sequence):
            if self._to_numpy:
                np.save(out_dir / f"sc{str(i).zfill(3)}.npy", im)

            if not accepted:
                continue

            polar_im = adaptive_cartesian_to_polar(im, angle=np.deg2rad(42))
            polar_im = np.clip(polar_im, *self._process_range)
            if self._to_numpy:
                np.save(out_dir / f"polar{str(i).zfill(3)}.npy", polar_im)
            polar_im_set.append(polar_im)

        if accepted:
            polar_im_set = np.stack(polar_im_set, axis=0)

        # Create USBMD dataset
        usbmd_dataset = {
            "path": out_h5,
            "image_sc": self.translate(sequence),
            "probe_name": "generic",
            "description": "EchoNet-LVH dataset converted to USBMD format",
        }
        if accepted:
            usbmd_dataset["image"] = self.translate(polar_im_set)
        return generate_usbmd_dataset(**usbmd_dataset)


if __name__ == "__main__":
    args = get_args()
    source_path = Path(args.source)

    # Load splits from CSV
    splits = load_splits(source_path)

    # Get list of all unique files from CSV
    files_to_process = []
    for split_files in splits.values():
        for hdf5_file in split_files:
            avi_file = find_avi_file(args.source, hdf5_file[:-5])  # Remove .hdf5
            if avi_file:
                files_to_process.append(avi_file)
            else:
                print(f"Warning: Could not find AVI file for {hdf5_file}")

    # List files that have already been processed
    files_done = []
    for _, _, filenames in os.walk(args.output):
        for filename in filenames:
            files_done.append(filename.replace(".hdf5", ""))

    # Filter out already processed files
    files_to_process = [f for f in files_to_process if f.stem not in files_done]
    print(f"Files left to process: {len(files_to_process)}")

    # Initialize processor with splits
    processor = LVHProcessor(
        path_out_h5=args.output,
        path_out=args.output_numpy,
        splits=splits,
    )

    print("Starting the conversion process.")

    if args.use_hyperthreading:
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(processor, file): file for file in files_to_process
            }
            for future in tqdm(as_completed(futures), total=len(files_to_process)):
                future.result()
    else:
        for file in tqdm(files_to_process):
            processor(file)

    print("All tasks are completed.")
