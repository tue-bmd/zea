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

import numpy as np
import tqdm

from usbmd.config import Config
from usbmd.data.data_format import generate_usbmd_dataset
from usbmd.data.datasets import Dataset
from usbmd.data.file import File
from usbmd.datapaths import format_data_path
from usbmd.display import to_8bit
from usbmd.ops import Pipeline
from usbmd.scan import Scan
from usbmd.utils import get_function_args, log
from usbmd.utils.checks import _DATA_TYPES


class GenerateDataSet:
    """Class for generating and saving ultrasound dataset to disk."""

    def __init__(
        self,
        config,
        to_dtype: str,
        destination_folder: Union[None, str],
        retain_folder_structure: bool = True,
        filetype: str = "hdf5",
        overwrite: bool = False,
        verbose: bool = True,
        jit_options: Union[None, dict] = "ops",
        **kwargs,
    ):
        """
        Args:
            config (object): Config object.
            to_dtype (str): output dtype, default is `image`.
            destination_folder (bool, optional): Folder to which dataset should
                be saved.
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
        self.dataset = Dataset.from_config(**self.config.data, **kwargs)
        self.path = format_data_path(
            self.config.data.dataset_folder, self.config.data.user
        )

        # initialize Pipeline
        assert (
            "pipeline" in self.config
        ), "Pipeline not found in config, please specify pipeline in config."

        self.process = Pipeline.from_config(
            self.config.pipeline, with_batch_dim=False, jit_options=jit_options
        )

        self.destination_folder = Path(destination_folder)

        if self.destination_folder.exists():
            if self.overwrite:
                log.warning(
                    "Possibly overwriting files existing folder "
                    f"{self.destination_folder}. "
                    "Press enter to continue..."
                )
                input()

    def prepare_parameters(self, file: File):
        """Prepare parameters for processing based on the file and config."""
        scan = file.scan(**self.config.scan)
        probe = file.probe()
        parameters = self.process.prepare_parameters(probe, scan, self.config)
        return parameters

    def _process_file(self, file: File, pbar: tqdm.tqdm):
        parameters = self.prepare_parameters(file)

        if self.dtype.value in ["raw_data", "aligned_data"]:
            data = file.load_transmits(self.dtype, parameters["selected_transmits"])
        else:
            data = file.load_data(self.dtype)

        if self.filetype == "png":
            for i, image in enumerate(data):
                name = file.path.parent / file.stem / str(i)

                path = self.get_path_from_name(name, ".png")
                if self.skip_path(path):
                    pbar.update(1)
                    return

                image = self.process_data(image, parameters)

                self.save_image(np.squeeze(image), path)
                pbar.update(1)

        elif self.filetype == "hdf5":
            path = self.get_path_from_name(file.path, ".hdf5")
            if self.skip_path(path):
                pbar.update(data.shape[0])
                return

            data_list = []
            for d in data:
                d = self.process_data(d, parameters)
                data_list.append(d)
                pbar.update(1)
            data = np.stack(data_list, axis=0)

            self.save_data(
                data,
                path,
                file.get_scan_parameters(),
                file.probe_name,
                file.description,
            )

    def generate(self):
        """Generate the dataset.

        Generates a dataset based on `filetype` that is being set during initalization.
        Either a `png` or `hdf5` dataset.
        """
        pbar = tqdm.tqdm(
            total=self.dataset.total_frames,
            desc=f"Generating dataset ({self.to_dtype}, {self.filetype})",
            disable=not self.verbose,
        )
        for file in iter(self.dataset):
            try:
                self._process_file(file, pbar)

            except Exception as e:
                log.error(f"Error processing {file.path}: {e}")
                raise

        log.success(f"Created dataset in {self.destination_folder}")

    @property
    def dtype(self):
        """Get the input data type of the pipeline"""
        return self.process.operations[0].input_data_type

    def process_data(self, data, parameters):
        """Small wrapper for processing data with the pipeline"""
        inputs = {self.process.key: data}

        outputs = self.process(**inputs, **parameters)

        image = outputs[self.process.output_key]

        return image

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
            path = name.relative_to(self.path)
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

    def save_data(self, data, path, scan_parameters, probe_name, description):
        """Save data to disk in hdf5 format

        Args:
            image (ndarray): input data
            path (str): file path
        """

        gen_kwargs = {
            str(self.to_dtype): data,
            **scan_parameters,
            "probe_name": probe_name,
            "description": description,
        }

        # automatically get correct gen_kwargs for generate_usbmd_dataset function
        # some scan parameters are not needed for the function and derived from
        # other parameters. we are only passing the necessary parameters
        func_args = get_function_args(generate_usbmd_dataset)
        gen_kwargs = {
            key: value for key, value in gen_kwargs.items() if key in func_args
        }
        generate_usbmd_dataset(path=path, **gen_kwargs)
