"""
Script to convert the PICMUS database to the zea format.

Example usage:
```bash
python zea/data/convert/picmus.py \
--src_dir /mnt/z/Ultrasound-BMd/data/PICMUS \
--output_dir converted_PICMUS_dir
```
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from zea import log
from zea.beamform.delays import compute_t0_delays_planewave
from zea.data.data_format import generate_zea_dataset


def convert_picmus(source_path, output_path, overwrite=False):
    """Converts the PICMUS database to the zea format.

    Args:
        source_path (str, pathlike): The path to the original PICMUS file.
        output_path (str, pathlike): The path to the output file.
        overwrite (bool, optional): Set to True to overwrite existing file.
            Defaults to False.
    """

    # Check if output file already exists and remove
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            logging.warning("Output file already exists. Skipping conversion.")
            return

    # Open the file
    file = h5py.File(source_path, "r")

    # Get the group containing the dataset
    file = file["US"]["US_DATASET0000"]

    if "data" not in file:
        raise ValueError("The file does not contain the data group.")

    # Extract I- and Q-data (shape (tx, el, ax))
    i_data = file["data"]["real"][:]
    q_data = file["data"]["imag"][:]

    if np.abs(np.sum(q_data)) < 0.1:
        # Use only the I-data, add dummy dimension (shape (tx, el, ax, ch=1))
        raw_data = i_data[..., None]
    else:
        # Stack I- and Q-data (shape (tx, el, ax, 2))
        raw_data = np.stack([i_data, q_data], axis=-1)

    # Add dummy frame dimension (shape (frame=1, tx, el, ax, ch=1))
    raw_data = raw_data[None]

    raw_data = np.transpose(raw_data, (0, 1, 3, 2, 4))

    _, n_tx, _, n_el, _ = raw_data.shape

    center_frequency = int(file["modulation_frequency"][:][0])
    # Fix a mistake in one of the PICMUS files
    if center_frequency == 0:
        center_frequency = 5.208e6
    sampling_frequency = int(file["sampling_frequency"][:][0])
    probe_geometry = np.transpose(file["probe_geometry"][:], (1, 0))

    sound_speed = float(file["sound_speed"][:][0])
    focus_distances = np.zeros((n_tx,), dtype=np.float32)
    polar_angles = file["angles"][:]
    azimuth_angles = np.zeros((n_tx,), dtype=np.float32)
    t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)

    initial_times = np.zeros((n_tx,))
    for n in range(n_tx):
        v = np.array([np.sin(polar_angles[n]), 0, np.cos(0)])
        initial_times[n] = -np.min(np.sum(probe_geometry * v[None], axis=1)) / sound_speed

        t0_delays[n] = compute_t0_delays_planewave(
            probe_geometry=probe_geometry,
            polar_angles=polar_angles[n],
            sound_speed=sound_speed,
        )
        # This line changes the data format to work with the old beamformer,
        # which is not in accordance with the new zea format

    generate_zea_dataset(
        path=output_path,
        raw_data=raw_data,
        center_frequency=center_frequency,
        sampling_frequency=sampling_frequency,
        probe_geometry=probe_geometry,
        initial_times=initial_times,
        sound_speed=sound_speed,
        t0_delays=t0_delays,
        focus_distances=focus_distances,
        polar_angles=polar_angles,
        azimuth_angles=azimuth_angles,
        tx_apodizations=tx_apodizations,
        probe_name="verasonics_l11_4v",
        description="PICMUS dataset converted to zea format",
    )


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Converts the PICMUS database to the zea format. The "
            "src_dir is scanned for hdf5 files ending in iq or rf. These files are"
            "converted and stored in output_dir under the same relative path as "
            "they came from in src_dir."
        )
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Source directory where the original PICMUS data is stored.",
    )

    parser.add_argument("--output_dir", type=str, help="Output directory of the converted database")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()

    # Get the source and output directories
    base_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    # Check if the source directory exists and create the output directory
    assert base_dir.exists(), f"Source directory {base_dir} does not exist."
    output_dir.mkdir(parents=True, exist_ok=False)

    # Traverse the source directory and convert all files
    for file in base_dir.rglob("*.hdf5"):
        str_file = str(file)

        # Select only the data files that actually contain rf or iq data
        # (There are also files containing the geometry of the phantoms or
        # images)
        if (
            not str_file.endswith("iq.hdf5") or not str_file.endswith("rf.hdf5")
        ) and "img" in str_file:
            log.info("Skipping %s", file.name)
            continue

        log.info("Converting %s", file.name)

        # Find the folder relative to the base directory to retain the
        # folder structure in the output directory
        output_file = output_dir / file.relative_to(base_dir)

        # Define the output path
        # NOTE: I added output_file.stem to put each file in its own
        # folder. This makes it possible to use it as a dataset because
        # it ensures there are never different types of data file in
        # the same folder.
        output_file = output_file.parent / output_file.stem / f"{output_file.stem}.hdf5"

        # Convert the file
        try:
            # Create the output directory if it does not exist already
            output_file.parent.mkdir(parents=True, exist_ok=True)

            convert_picmus(file, output_file, overwrite=True)
        except Exception:
            output_file.parent.rmdir()
            log.error("Failed to convert %s", str_file)
            continue
