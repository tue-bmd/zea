"""
Script to convert the PICMUS database to the USBMD format using `usbmd.
data_format.convert_picmus.convert_picmus`.
"""

import os
import logging
from pathlib import Path
import argparse

from usbmd.data_format.convert_picmus import convert_picmus

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Converts the PICMUS database'
                                     ' to the USBMD format. The src_dir is '
                                     'scanned for hdf5 files ending in iq or '
                                     'rf. These files are converted and '
                                     'stored in output_dir under the same '
                                     'relative path as they came from in '
                                     'src_dir.')

    parser.add_argument('--src_dir', type=str,
                        help='Source directory where the original PICMUS data '
                             'is stored.')

    parser.add_argument('--output_dir', type=str,
                        help='Output directory of the converted database')

    args = parser.parse_args()

    # Get the source and output directories
    base_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    # Traverse the source directory and convert all files
    for root, dirs, files in os.walk(base_dir):

        for file in files:
            logging.info('Processing %s', file)
            # Select only the data files that actually contain rf or iq data
            # (There are also files containing the geometry of the phantoms or
            # images)
            if (file.endswith('iq.hdf5') or file.endswith('rf.hdf5')) \
                and not 'img' in file:

                # Get the full path to the file
                path = os.path.join(root, file)

                # Find the folder relative to the base directory to retain the
                # folder structure in the output directory
                relative_folder_path = Path(root).relative_to(base_dir)

                # Define the output path
                # NOTE: I added Path(file).stem to put each file in its own
                # folder. This makes it possible to use it as a dataset because
                # it ensures there are never different types of data file in
                # the same folder.
                output_path = Path(output_dir,
                                   relative_folder_path,
                                   Path(file).stem,
                                   file)

                # Create the output directory if it does not exist already
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert the file
                try:
                    convert_picmus(path, output_path, overwrite=True)
                except:
                    logging.error('Failed to convert %s', Path(path))
                    continue

                print(f'Converted file saved to {Path(output_path)}')
