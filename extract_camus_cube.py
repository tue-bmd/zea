"""Extract an image sequence from the CAMUS dataset and store it as a numpy data cube."""

import argparse
from pathlib import Path

import numpy as np
from imagelib import Image, LimitsND

import zea

DEFAULT_DATASET = "hf://zeahub/camus-sample/val"
IMAGE_KEY = "data/image/values"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="zea dataset path or hf:// URI")
    parser.add_argument("--index", type=int, default=0, help="Index of the file in the dataset")
    parser.add_argument(
        "--output", type=Path, default=Path("camus_cube.hdf5"), help="Output .npy path"
    )
    return parser.parse_args()


def load_image_sequence(dataset_path, index):
    """Load a sequence of images as a (frames, height, width) array."""
    dataset = zea.Dataset(dataset_path)
    with dataset[index] as file:
        return np.asarray(file.data.image.values)


def _get_extent(cube):
    width = (cube.shape[2] - 1) * 0.31e-3
    height = (cube.shape[1] - 1) * 0.31e-3
    return LimitsND((0, cube.shape[0] - 1, -width / 2, width / 2, 0, height))


def save_cube(cube, output_path):
    """Save the image cube to a numpy file."""
    print(cube.shape)
    return Image(cube, limits=_get_extent(cube)).save(output_path)


def main():
    """Extract a CAMUS image sequence and store it as a numpy cube."""
    arguments = parse_arguments()
    cube = load_image_sequence(arguments.dataset, arguments.index)
    image = save_cube(cube, arguments.output)


if __name__ == "__main__":
    main()
