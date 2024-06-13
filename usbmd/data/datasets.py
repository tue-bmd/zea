"""The datasets module contains classes for loading different types of
ultrasound datasets.

TODO:
    - Context manager ReadH5
    - Check if legacy datasets are still working
    - Add warnings to legacy datasets and point to USBMD dataset
    which is current standard.
    - Move all processing to processing class (also in legacy datasets)
    - Move legacy datasets to different file

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import tqdm
from deepdiff import DeepDiff

from usbmd.probes import get_probe
from usbmd.registry import dataset_registry
from usbmd.scan import PlaneWaveScan, Scan, cast_scan_parameters
from usbmd.utils import log
from usbmd.utils.checks import get_check, validate_dataset
from usbmd.utils.config import Config
from usbmd.data.read_h5 import ReadH5, recursively_load_dict_contents_from_group
from usbmd.utils.utils import (
    calculate_file_hash,
    date_string_to_readable,
    get_date_string,
    update_dictionary,
)

_CHECK_SCAN_PARAMETERS_MAX_DATASET_SIZE = 10000
_VALIDATED_FLAG_FILE = "validated.flag"


def get_dataset(config):
    """Get dataset instance given configuration file."""
    name = config.dataset_name

    dataset_class = dataset_registry[name]

    return dataset_class(config)


class DataSet:
    """Base class for ultrasound datasets.

    A dataset is initialized with a datafolder. The DataSet base class will
    then search for files in the datafolder with the correct filetype and store
    the paths to these files.

    Do not use on its own but rather inherit the class.

    """

    def __init__(self, config, datafolder, filetype, reader):
        """
        Args:
            config (object): config.data object / dict.
            datafolder (str): folder in which the dataset is located.
            filetype (str): Filetype / extension of the files in the dataset.
                Either mat or hdf5.
            reader (str): Which kind of reader to use when loading the data.
                Either mat or hdf5. Can be different from filetype.
                Depending on the version of the .mat file, etc...
        """
        #: Stores the config.data key internally
        self.config = config
        #: The folder to read data files from
        self.datafolder = datafolder
        #: The type of data to read (either 'hdf5' or 'mat')
        self.filetype = filetype
        #: The kind of reader to use when loading the data
        #: (either 'hdf5' or 'mat'). Can be different from filetype.
        self.reader = reader
        #: The folders to search for data files
        self.datafolders = None
        #: The paths to the data files
        self.file_paths = None
        #: The name of the current file
        self.file_name = None
        #: The current file container
        self.file = None
        #: The current frame number
        self.frame_no = None
        # h5 reader object for reading hdf5 files.
        # This class is defined in `usbmd.utils.read_h5`.
        self.h5_reader = None
        #: Total number of frames in current file
        self.num_frames = None
        #: The scan parameters of the current file
        self.scan_parameters = None

        assert filetype in ["hdf5", "mat"], ValueError(
            f"Unsupported filteype: {filetype}."
        )

        if self.reader is None:
            self.reader = self.filetype

        if not isinstance(datafolder, list):
            datafolder = [datafolder]

        self.datafolders = [Path(folder) for folder in datafolder]

        for datafolder_ in self.datafolders:
            if not datafolder_.is_dir():
                log.error(
                    f"{log.yellow(datafolder_)} is not a directory. "
                    "Check for errors in the path!"
                )
                sys.exit(1)

        list_of_file_paths = [
            list(folder.rglob("*." + filetype)) for folder in self.datafolders
        ]

        self.file_paths = [file for sublist in list_of_file_paths for file in sublist]

        if len(self.file_paths) == 0:
            raise ValueError(
                "Could not find any files with data type "
                f"{self.config.dtype} in:\n{self.datafolders}"
            )

    def __str__(self):
        """Print info of current dataset."""
        return f"{self.__class__.__name__} dataset containing {len(self)} files"

    def __len__(self):
        """Return number of files in dataset."""
        return len(self.file_paths)

    def __getitem__(self, index):
        """Read file at index place in file_paths.

        Args:
            index (int): Index of which file in dataset to read.

        Raises:
            ValueError: Raises when filetype is not hdf5 or mat.

        Returns:
            file (h5py or mat): File container.

        """
        return self.get_file(index)

    def get_file(self, index: int):
        """Read file at index place in file_paths.

        Args:
            index (int): Index of which file in dataset to read.

        Raises:
            ValueError: Raises when filetype is not hdf5 or mat.

        Returns:
            file (h5py or mat): File container.

        """
        assert self.reader in ["hdf5", "mat"], f"Unsupported filteype {self.reader}."
        self.file_name = self.file_paths[index]
        # TODO: probably should be done in context manager.
        if self.reader == "hdf5":
            self.h5_reader = ReadH5(self.file_name)
            file = self.h5_reader.open()
            self.num_frames = len(self.h5_reader)

        elif self.reader == "mat":
            file = sio.loadmat(self.file_name)
            self.num_frames = len(file)

        else:
            raise ValueError("Unsupported filteype.")

        return file

    def get_frame_no(self, frame_no=None):
        """set frame number"""
        if self.num_frames == 1:
            frame_no = 0
        else:
            if frame_no is None:
                if self.config.get("frame_no") is None:
                    frame_no = int(input(f"Frame number (0 / {self.num_frames - 1}): "))
                else:
                    frame_no = self.config.frame_no
        return frame_no

    @classmethod
    def get_probe_name(cls):
        """Returns the name of the probe type corresponding to this dataset."""
        return dataset_registry.get_parameter(cls_or_name=cls, parameter="probe_name")

    @classmethod
    def get_scan_class(cls):
        """Returns the Scan class corresponding to the dataset."""
        return dataset_registry.get_parameter(cls_or_name=cls, parameter="scan_class")

    def get_default_scan_parameters(self):
        """Returns a dictionary of default parameters to initialize a scan
        object that works with the dataset.

        Returns:
            dict: The default parameters (the keys are identical to the
                __init__ parameters of the Scan class).
        """
        default_scan_parameters = get_probe(
            self.get_probe_name()
        ).get_default_scan_parameters()
        file_scan_parameters = self.get_scan_parameters_from_file()
        return update_dictionary(default_scan_parameters, file_scan_parameters)

    def get_scan_parameters_from_file(self, file=None):
        """Returns a dictionary of parameters to initialize a scan
        object that comes with the dataset (stored inside datafile).

        If there are no scan parameters in the hdf5 file, returns
        an empty dictionary.

        Args:
            file (h5py or mat): File container.

        Returns:
            dict: The scan parameters.
        """
        if file is None:
            file = self.get_file(0)
        scan_parameters = {}
        if "scan" in file:
            scan_parameters = recursively_load_dict_contents_from_group(file, "scan")
        return scan_parameters

    def close(self):
        """Close the file."""
        if self.reader == "hdf5":
            self.h5_reader.close()


@dataset_registry(
    name="usbmd",
    probe_name="generic",
    scan_class=Scan,
)
class USBMDDataSet(DataSet):
    """Class to read dataset in USBMD format."""

    def __init__(self, config):
        """Initializes the USBMDDataSet class.

        Args:
            config (utils.config.Config): The config.data configuration object.
        """
        self.config = config
        self.dtype = config.dtype

        if "user" in config and config.user is not None:
            self.data_root = Path(config.user["data_root"]) / config["dataset_folder"]
        else:
            self.data_root = config["dataset_folder"]

        super().__init__(
            config, datafolder=self.data_root, filetype="hdf5", reader="hdf5"
        )

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

        data = self.file[f"data/{self.dtype}"]

        if self.frame_no == "all":
            data = data[:]
            get_check(self.dtype)(data, with_batch_dim=True)
        else:
            data = data[self.frame_no]
            get_check(self.dtype)(data)

        data = self._correct_deprecated_dim_order(data)
        return data

    def _correct_deprecated_dim_order(self, data):
        """Correct data dimension order if it is in the old usbmd format."""
        if self.dtype not in ["raw_data", "aligned_data"]:
            return data

        *_, n_ax, n_el, _ = data.shape

        if n_ax < n_el:
            log.warning(
                "\nThe data has the wrong dimension order: (n_tx, n_el, n_ax, n_ch).\n"
                "Transposing data to correct dimension order: (n_tx, n_ax, n_el, n_ch).\n"
                "This will be removed in a future version of USBMD. "
                "Please update your dataset to the new format."
            )
            if len(data.shape) == 4:
                data = np.transpose(data, (0, 2, 1, 3))
            else:
                data = np.transpose(data, (0, 1, 3, 2, 4))
        return data

    # pylint: disable=arguments-differ
    def get_probe_name(self):
        """Reads the probe name from the data file and returns it."""
        # Get the first file from the dataset
        # We assume all files in a dataset have the same scan parameters
        # so the first one will do.
        if self.file is None:
            self.file = super().__getitem__(0)

        # Read the probe name from the file
        assert "probe" in self.file.attrs, (
            "Probe name not found in file attributes. "
            "Make sure you are using a USBMD dataset. "
            f"Found attributes: {list(self.file.attrs)}"
        )
        probe_name = self.file.attrs["probe"]

        return probe_name

    def get_default_scan_parameters(self, file_idx=0):
        # Get the first file from the dataset
        # We assume all files in a dataset have the same scan parameters
        # so the first one will do.
        self.file = super().__getitem__(file_idx)

        probe_name = self.get_probe_name()

        default_scan_parameters = get_probe(probe_name).get_default_scan_parameters()

        file_scan_parameters = self.get_scan_parameters_from_file(self.file)
        file_scan_parameters = cast_scan_parameters(file_scan_parameters)

        # lookup all arguments from Scan class and populate default_scan_parameters if
        # they can be found in file_scan_parameters
        for arg in Scan.__init__.__code__.co_varnames:
            if arg in file_scan_parameters:
                default_scan_parameters[arg] = file_scan_parameters[arg]

        return default_scan_parameters

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
        for i, file_path in tqdm.tqdm(
            enumerate(self.file_paths),
            total=len(self),
            desc="Checking dataset files on validity (USBMD format)",
        ):
            try:
                validate_dataset(file_path)
            except Exception as e:
                validation_error_log.append(
                    f"File {file_path} is not a valid USBMD dataset.\n{e}\n"
                )
                # convert into warning
                log.warning(f"⚠️ Error in file {file_path}.\n{e}")
                validated_succesfully = False

            try:
                scan_parameters = self.get_default_scan_parameters(i)
                if i == 0:
                    self.scan_parameters = scan_parameters
                else:
                    diff = DeepDiff(self.scan_parameters, scan_parameters)
                    assert not diff, (
                        "Scan parameters are not the same in all files in dataset."
                        f"\n{diff}"
                    )
                num_frames_per_file.append(self.num_frames)
            except Exception as e:
                # convert into warning
                log.warning(f"Error in scan parameters from file {file_path}.\n{e}")
                validation_error_log.append(
                    f"File {file_path} has different scan parameters than "
                    f"file {self.file_paths[0]}."
                    f"\n{diff}\n"
                )
                validated_succesfully = False

        if not validated_succesfully:
            log.warning(
                "Not all files in dataset have the same scan parameters. "
                "Check warnings above for details. No validation file was created. "
                f"See {validation_error_file_path} for details."
            )
            with open(validation_error_file_path, "w", encoding="utf-8") as f:
                for error in validation_error_log:
                    f.write(error)
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
        data_types = list(self.file["data"].keys())
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
            f.write("Scan parameters:\n")
            scan_parameters = dict(self.scan_parameters)
            if scan_parameters:
                for key, value in scan_parameters.items():
                    if isinstance(value, np.ndarray):
                        scan_parameters[key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        scan_parameters[key] = np.array(value).tolist()
                    elif isinstance(value, str):
                        scan_parameters[key] = str(value)
                    elif int(value) == value:
                        scan_parameters[key] = int(value)
                    else:
                        scan_parameters[key] = float(value)
                f.write(json.dumps(scan_parameters, indent=4))
            else:
                f.write("\tEmpty\n")
            f.write(f"{'-' * 80}\n")

        # Write the hash of the validation file
        validation_file_hash = calculate_file_hash(validation_file_path)
        with open(validation_file_path, "a", encoding="utf-8") as f:
            # *** validation file hash *** (80 total line length)
            f.write("*** validation file hash ***\n")
            f.write(f"hash: {validation_file_hash}")


@dataset_registry(
    name="dummy",
    probe_name="verasonics_l11_4v",
    scan_class=PlaneWaveScan,
)
class DummyDataset(DataSet):
    """Dummy dataset."""

    def __init__(self, config=None, modtype="rf"):
        """
        Initializes dummy dataset which returns normally distributed random
        values.

        If no modtype nor config is provided the dataset will assume raw_data in rf
        modulated form of shape (75, 128, 2048, 1).

        If a config and modtype is supplied the ouptput depends on the dtype
        elements:
        - (dtype:raw_data, modtype:"rf") - (75, 128, 2048, 1)
        - (dtype:raw_data, modtype:"iq") - (75, 128, 2048, 2)
        - (dtype:beamformed_data) - (128, 356, 1)

        Args:
            config (Config, optional): config.data section from a config file.
            Defaults to None.
        """
        try:
            # Run baseclass init to define all the internal variables
            super().__init__(None, None, None, None)
        # The base init will raise an error because the __init__ method was
        # called with just None inputs, which is expected.
        except AssertionError:
            pass

        self.data_root = "dummy"
        self.file_paths = [Path(f"dummy/dummy_file_{i}.hdf5") for i in range(len(self))]
        # use raw rf data if no config is supplied
        if config is None:
            self.dtype = "raw_data"
        else:
            self.dtype = config.dtype

        if modtype == "rf":
            self.n_ch = 1
        elif modtype == "iq":
            self.n_ch = 2
        else:
            raise ValueError(
                f"Modulation type {modtype} not available for this dataset"
            )

        self.n_ax = 2048
        self.n_tx = 75
        self.Nz = 128

    def __len__(self):
        """
        Return number of files in dataset. The number is arbitrary for the dummy
        dataset because it can generate as much data as needed.
        """
        return 5

    def __getitem__(self, index):
        """Read file at index place in file_paths.

        Args:
            index (int): Index of which file in dataset to read.

        Raises:
            ValueError: Raises when filetype is not hdf5 or mat.

        Returns:
            file (h5py or mat): File container.

        """
        self.file = self.get_file(index)
        if self.dtype == "raw_data":
            rf_data = np.random.randn(self.n_tx, self.n_ax, self.Nz, self.n_ch)
            return rf_data

        elif self.dtype == "beamformed_data":
            return np.random.randn(self.Nz, 256, 1)

        else:
            raise ValueError(f"Data type {self.dtype} not available for this dataset")

    def get_file(self, index: int):
        """Returns fake file object because dummy dataset has no files."""
        file = {}
        file["attrs"] = {}
        file["attrs"]["probe"] = "generic"
        file["attrs"]["description"] = "Dummy dataset"
        file = Config(file)
        return file

    def get_default_scan_parameters(self):
        probe_parameters = get_probe(
            self.get_probe_name()
        ).get_default_scan_parameters()
        probe_parameters["n_ax"] = self.n_ax
        probe_parameters["n_tx"] = self.n_tx
        probe_parameters["Nz"] = self.Nz

        scan_parameters = {
            "angles": np.linspace(-0.27925268, 0.27925268, 75),
            **probe_parameters,
        }
        return scan_parameters

    def get_scan_parameters_from_file(self, file=None):
        return self.get_default_scan_parameters()
