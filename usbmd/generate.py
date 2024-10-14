"""Generate ultrasound dataset from any data type to another and save to disk.

Supports both saving to png and hdf5. saving to png is only supported for image data.

Example:
    Run from command line to generate PICMUS dataset:
    >>> usbmd -c configs/config_picmus_rf.yaml -t generate

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""

from pathlib import Path
from typing import Union
import inspect

import numpy as np
import tqdm

from usbmd import Config
from usbmd.data import get_dataset
from usbmd.data.data_format import generate_usbmd_dataset
from usbmd.display import to_8bit
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.utils import get_function_args, log, update_dictionary
from usbmd.utils.checks import _DATA_TYPES


class GenerateDataSet:
    """Class for generating and saving ultrasound dataset to disk."""

    def __init__(
        self,
        config,
        to_dtype: str = "image",
        destination_folder: Union[None, str] = None,
        retain_folder_structure: bool = True,
        filetype: str = "hdf5",
        overwrite: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            config (object): Config object.
            to_dtype (str): output dtype, default is `image`.
            destination_folder (bool, optional): Folder to which dataset should
                be saved. Defaults to None. If None, new folder with dtype suffix
                is created in the parent folder of the original dataset folder.
                If relative path, folder will be created in parent folder
                of source dataset.
            retain_folder_structure (bool, optional): Whether to exactly copy
                the folder structure of the original dataset or put all output
                files in one folder. Defaults to True.
            filetype (str, optional): Filetype to save to. Defaults to "hdf5".
            overwrite (bool, optional): Whether to overwrite existing files.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        """
        self.config = Config(config)
        self.to_dtype = to_dtype
        assert self.to_dtype in _DATA_TYPES, ValueError(
            f"Unsupported dtype: {self.to_dtype}."
        )
        self.retain_folder_structure = retain_folder_structure
        self.filetype = filetype
        assert self.filetype in ["hdf5", "png"], ValueError(
            f"Unsupported filteype: {self.filetype}."
        )

        if self.to_dtype not in ["image", "image_sc"] and self.filetype == "png":
            raise ValueError(
                "Cannot save to png if to_dtype is not image. "
                "Please set filetype to hdf5."
            )
        self.overwrite = overwrite
        self.verbose = verbose

        # intialize dataset
        self.dataset = get_dataset(self.config.data)

        # Initialize scan based on dataset (if it can find proper scan parameters)
        scan_class = self.dataset.get_scan_class()
        file_scan_params = self.dataset.get_scan_parameters_from_file()
        file_probe_params = self.dataset.get_probe_parameters_from_file()

        if len(file_scan_params) == 0:
            log.info(
                f"Could not find proper scan parameters in {self.dataset} at "
                f"{log.yellow(str(self.dataset.datafolder))}."
            )
            log.info("Proceeding without scan class.")

            self.scan = None
        else:
            config_scan_params = self.config.scan
            # dict merging of manual config and dataset default scan parameters
            scan_params = update_dictionary(file_scan_params, config_scan_params)
            # Retrieve the argument names of the Scan class
            sig = inspect.signature(scan_class.__init__)

            # Filter out the arguments that are not part of the Scan class
            reduced_scan_params = {
                key: scan_params[key] for key in sig.parameters if key in scan_params
            }

            self.scan = scan_class(**reduced_scan_params)

        # initialize probe
        probe_name = self.dataset.get_probe_name()

        if probe_name == "generic":
            self.probe = get_probe(probe_name, **file_probe_params)
        else:
            self.probe = get_probe(probe_name)

        # intialize process class
        self.process = Process(self.config, self.scan, self.probe)
        if self.config.preprocess.operation_chain is None:
            self.process.set_pipeline(
                dtype=self.config.data.dtype,
                to_dtype=self.to_dtype,
                verbose=self.verbose,
            )
        else:
            self.process.set_pipeline(
                operation_chain=self.config.preprocess.operation_chain,
                verbose=self.verbose,
            )

        if self.dataset.datafolder is None:
            self.dataset.datafolder = Path(".")

        if destination_folder is None:
            self.destination_folder = (
                self.dataset.datafolder.parent
                / f"{self.dataset.config.dataset_name}_{to_dtype}"
            )
        else:
            self.destination_folder = Path(destination_folder)
            if not self.destination_folder.is_absolute():
                self.destination_folder = (
                    self.dataset.datafolder.parent / self.destination_folder
                )

        if self.destination_folder.exists():
            if self.overwrite:
                log.warning(
                    "Possibly overwriting files existing folder "
                    f"{self.destination_folder}. "
                    "Press enter to continue..."
                )
                input()

    def generate(self):
        """Generate the dataset.

        Generates a dataset based on `filetype` that is being set during initalization.
        Either a `png` or `hdf5` dataset.

        Returns:
            bool: if succesfull returns `True`.

        """
        total_num_frames = self.dataset.total_num_frames
        pbar = tqdm.tqdm(
            total=total_num_frames,
            desc=f"Generating dataset ({self.to_dtype}, {self.filetype})",
        )
        for idx in range(len(self.dataset)):
            try:
                frame_no = "all"
                data = self.dataset[(idx, frame_no)]

                base_name = self.dataset.file_paths[idx]

                if self.filetype == "png":
                    for i, image in enumerate(data):
                        name = base_name.parent / base_name.stem / str(i)

                        path = self.get_path_from_name(name, ".png")
                        if self.skip_path(path):
                            pbar.update(1)
                            continue

                        image = self.process.run(image)
                        self.save_image(np.squeeze(image), path)
                        pbar.update(1)

                elif self.filetype == "hdf5":
                    path = self.get_path_from_name(base_name, ".hdf5")
                    if self.skip_path(path):
                        pbar.update(data.shape[0])
                        continue

                    data_list = []
                    for d in data:
                        d = self.process.run(d)
                        data_list.append(d)
                        pbar.update(1)
                    data = np.stack(data_list, axis=0)
                    self.save_data(data, path)
            except Exception as e:
                log.error(f"Error processing {base_name}: {e}")
                raise

        log.success(f"Created dataset in {self.destination_folder}")
        return True

    def skip_path(self, path):
        """Check if path exists and if we should skip it.
        Skips when file exists and overwrite is False.
        Overwrites when file exists and overwrite is True.
        Nothing happens when file does not exist, we will proceed.
        """
        if path.is_file() and not self.overwrite:
            log.warning(f"Skipping {path}, already exists.")
            return True
        elif path.is_file() and self.overwrite:
            log.warning(f"Overwriting {path}.")
            path.unlink()
        return False

    def get_path_from_name(self, name, suffix):
        """Simple helper function that return proper path"""
        name = Path(name)
        if self.retain_folder_structure:
            path = name.relative_to(self.dataset.datafolder)
        else:
            path = name.name
        path = self.destination_folder / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix(suffix)
        return path

    @staticmethod
    def save_image(image, path):
        """Save images to disk

        Args:
            image (ndarray): input image
            path (str): file path
        """
        image = to_8bit(image)
        image.save(path)

    def save_data(self, data, path):
        """Save data to disk in hdf5 format

        Args:
            image (ndarray): input data
            path (str): file path
        """
        file_scan_parameters = self.scan.get_scan_parameters()

        gen_kwargs = {
            str(self.to_dtype): data,
            **file_scan_parameters,
            "probe_name": self.dataset.file.attrs["probe"],
            "description": self.dataset.file.attrs["description"],
        }

        # automatically get correct gen_kwargs for generate_usbmd_dataset function
        # some scan parameters are not needed for the function and derived from
        # other parameters. we are only passing the necessary parameters
        func_args = get_function_args(generate_usbmd_dataset)
        gen_kwargs = {
            key: value for key, value in gen_kwargs.items() if key in func_args
        }
        generate_usbmd_dataset(
            path=path,
            **gen_kwargs,
        )
