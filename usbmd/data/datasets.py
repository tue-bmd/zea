"""The datasets module contains classes for loading different types of
ultrasound datasets.

- **Author(s)**     : Tristan Stevens, Wessel van Nierop
- **Date**          : November 18th, 2021
"""

from pathlib import Path

import tqdm

from usbmd.data.legacy_datasets import DataSet
from usbmd.scan import Scan
from usbmd.utils import (
    calculate_file_hash,
    date_string_to_readable,
    get_date_string,
    log,
)
from usbmd.utils.checks import get_check, validate_dataset

_CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE = 10000
_VALIDATED_FLAG_FILE = "validated.flag"


class USBMDDataSet(DataSet):
    """Class to read dataset in USBMD format."""

    def __init__(self, config, validate=True, verbose=True):
        """Initializes the USBMDDataSet class.

        Args:
            config (utils.config.Config): The config.data configuration object.
        """
        self.config = config
        self.dtype = config.dtype
        self.verbose = verbose

        if "user" in config and config.user is not None:
            self.data_root = Path(config.user["data_root"]) / config["dataset_folder"]
        else:
            self.data_root = config["dataset_folder"]
        super().__init__(
            config, datafolder=self.data_root, filetype="hdf5", reader="hdf5"
        )

        if validate:
            self.validate_dataset()

    def __getitem__(self, index):
        """Retrieves an item from the dataset."""
        if isinstance(index, int):
            frame_no = None
        elif len(index) == 2:
            index, frame_no = index
        else:
            raise ValueError(
                "Index should either be an integer (indicating file index), "
                "or tuple containing file index and frame number!"
            )
        self.file = super().__getitem__(index)

        self.frame_no = self.get_frame_no(frame_no)

        if "data" in self.file:
            data = self.file[f"data/{self.dtype}"][self.frame_no]
        else:
            assert not isinstance(self.frame_no, list), (
                "Reading multiple frames from event structure is not supported. "
                "Please specify a single frame number."
            )
            data = self.file[f"event_{self.frame_no}/data/{self.dtype}"][0]

        if isinstance(self.frame_no, list):
            get_check(self.dtype)(data, with_batch_dim=True)
        else:
            get_check(self.dtype)(data)

        return data

    def validate_dataset(self):
        """Validate dataset contents.
        Furthermore, it checks if all files in the dataset have the same scan parameters.

        If a validation file exists, it checks if the dataset was validated on the same date.
        If the validation file was corrupted, it raises an error.
        If the validation file was not corrupted and validated, it prints a message and returns.
        """

        validation_file_path = Path(self.data_root, _VALIDATED_FLAG_FILE)
        # for error logging
        validation_error_file_path = Path(
            self.data_root, get_date_string() + "_validation_errors.log"
        )
        validation_error_log = []

        if validation_file_path.is_file():
            self._assert_validation_file(validation_file_path)
            return

        if len(self.file_paths) > _CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE:
            log.warning(
                "Checking scan parameters in more than "
                f"{_CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE} files takes too long. "
                f"Found {len(self.file_paths)} files in dataset. "
                "Not checking scan parameters."
            )
            return

        num_frames_per_file = []
        validated_succesfully = True
        for file_path in tqdm.tqdm(
            self.file_paths,
            total=len(self),
            desc="Checking dataset files on validity (USBMD format)",
            disable=not self.verbose,
        ):
            try:
                validate_dataset(file_path)
            except Exception as e:
                validation_error_log.append(
                    f"File {file_path} is not a valid USBMD dataset.\n{e}\n"
                )
                # convert into warning
                log.warning(f"Error in file {file_path}.\n{e}")
                validated_succesfully = False

        if not validated_succesfully:
            log.warning(
                "Not all files in dataset have the same scan parameters. "
                "Check warnings above for details. No validation file was created. "
                f"See {validation_error_file_path} for details."
            )
            try:
                with open(validation_error_file_path, "w", encoding="utf-8") as f:
                    for error in validation_error_log:
                        f.write(error)
            except Exception as e:
                log.error(
                    f"Could not write validation errors to {validation_error_file_path}."
                    f"\n{e}"
                )
            return

        # Create the validated flag file
        self._write_validation_file(num_frames_per_file)
        log.info(
            f"{log.green('Dataset validated.')} Check {validation_file_path} for details."
        )

    @staticmethod
    def _assert_validation_file(validation_file_path):
        """Check if validation file exists and is valid."""
        with open(validation_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            try:
                validation_date = lines[1].split(": ")[1].strip()
                read_validation_file_hash = lines[-1].split(": ")[1].strip()
            except Exception as exc:
                raise ValueError(
                    log.error(
                        f"Validation file {log.yellow(validation_file_path)} is corrupted. "
                        "Remove it if you want to redo validation."
                    )
                ) from exc

            log.info(
                "Dataset was validated on "
                f"{log.green(date_string_to_readable(validation_date))}"
            )
            log.info(
                f"Remove {log.yellow(validation_file_path)} if you want to redo validation."
            )
        # check if validation file was corrupted
        validation_file_hash = calculate_file_hash(
            validation_file_path, omit_line_str="hash"
        )
        assert validation_file_hash == read_validation_file_hash, log.error(
            f"Validation file {log.yellow(validation_file_path)} was corrupted.\n"
            f"Remove it if you want to redo validation.\n"
        )

    def _write_validation_file(self, num_frames_per_file):
        """Write validation file."""
        validation_file_path = Path(self.data_root, _VALIDATED_FLAG_FILE)
        self.file = self.get_file(0)  # read a file to initiate self.file
        if "data" in self.file:
            data_types = list(self.file["data"].keys())
        else:
            data_types = list(self.file["event_0"]["data"].keys())

        number_of_files = len(self.file_paths)
        number_of_frames = sum(num_frames_per_file)
        with open(validation_file_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {self.data_root}\n")
            f.write(f"Validated on: {get_date_string()}\n")
            f.write(f"Number of files: {number_of_files}\n")
            f.write(f"Number of frames: {number_of_frames}\n")
            f.write(f"Data types: {', '.join(data_types)}\n")
            f.write(f"{'-' * 80}\n")
            # write all file names (not entire path) with number of frames on a new line
            for file_path, num_frames in zip(self.file_paths, num_frames_per_file):
                f.write(f"{file_path.name}: {num_frames}\n")
            f.write(f"{'-' * 80}\n")

        # Write the hash of the validation file
        validation_file_hash = calculate_file_hash(validation_file_path)
        with open(validation_file_path, "a", encoding="utf-8") as f:
            # *** validation file hash *** (80 total line length)
            f.write("*** validation file hash ***\n")
            f.write(f"hash: {validation_file_hash}")
