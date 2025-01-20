"""
Script to convert the PICMUS database to the USBMD format using `usbmd.data.convert.picmus`.

Example usage:
```bash
python usbmd/scripts/convert_picmus.py \
--src_dir /mnt/z/Ultrasound-BMd/data/PICMUS \
--output_dir converted_PICMUS_dir
```
"""

import argparse
from pathlib import Path

from usbmd.data.convert import convert_picmus
from usbmd.utils import log


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Converts the PICMUS database to the USBMD format. The "
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

    parser.add_argument(
        "--output_dir", type=str, help="Output directory of the converted database"
    )
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
        except:
            output_file.parent.rmdir()
            log.error("Failed to convert %s", str_file)
            continue
