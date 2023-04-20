"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : generate.py

    Author(s)     : Tristan Stevens
    Date          : Thu Nov 18 2021

==============================================================================
"""
from pathlib import Path

import h5py
import numpy as np
import tqdm
from PIL import Image

from usbmd.datasets import get_dataset
from usbmd.probes import get_probe
from usbmd.processing import Process, to_8bit, _DATA_TYPES
from usbmd.utils.utils import update_dictionary


class GenerateDataSet:
    """Class for generating and saving ultrasound dataset to disk."""
    def __init__(
        self,
        config,
        to_dtype: str='image',
        destination_folder: str=None,
        retain_folder_structure: bool=True,
        filetype: str='hdf5',
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

        """
        self.config = config
        self.to_dtype = to_dtype
        assert self.to_dtype in _DATA_TYPES, \
            ValueError(f'Unsupported dtype: {self.to_dtype}.')
        self.retain_folder_structure = retain_folder_structure
        self.filetype = filetype
        assert self.filetype in ['hdf5', 'png'], \
            ValueError(f'Unsupported filteype: {self.filetype}.')

        if self.to_dtype != 'image' and self.filetype == 'png':
            raise ValueError(
                'Cannot save to png if to_dtype is not image. '
                'Please set filetype to hdf5.'
            )

        # intialize dataset
        self.dataset = get_dataset(self.config.data)

        # Initialize scan based on dataset
        scan_class = self.dataset.get_scan_class()
        default_scan_params = self.dataset.get_default_scan_parameters()
        config_scan_params = self.config.scan

        # dict merging of manual config and dataset default scan parameters
        scan_params = update_dictionary(default_scan_params, config_scan_params)
        self.scan = scan_class(**scan_params, modtype=self.config.data.modtype)

        # initialize probe
        self.probe = get_probe(self.dataset.get_probe_name())

        # intialize process class
        self.process = Process(config, self.scan, self.probe)

        if destination_folder is None:
            self.destination_folder = self.dataset.datafolder.parent / \
                f'{self.dataset.config.dataset_name}_image'
        else:
            self.destination_folder = Path(destination_folder)
            if not self.destination_folder.is_absolute():
                self.destination_folder = self.dataset.datafolder.parent / self.destination_folder

        if self.destination_folder.exists():
            raise ValueError(
                f'Cannot create dataset in {self.destination_folder}, folder already exists!'
            )

    def generate(self):
        """Generate the dataset."""
        for idx in tqdm.tqdm(
            range(len(self.dataset)),
            desc=f'Generating dataset ({self.to_dtype}, {self.filetype})',
        ):
            data = self.dataset[idx]
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
                single_frame = True
            else:
                single_frame = False

            base_name = self.dataset.file_paths[idx]

            if self.filetype == 'png':
                for i, image in enumerate(data):
                    if single_frame:
                        name = base_name
                    else:
                        name = base_name.parent / str(i)

                    path = self.get_path_from_name(name, '.png')

                    image = self.process.run(image, self.config.data.dtype, self.to_dtype)
                    self.save_image(image, path)

            elif self.filetype == 'hdf5':
                data_list = []
                for d in data:
                    d = self.process.run(d, self.config.data.dtype, self.to_dtype)
                    data_list.append(d)
                data = np.stack(data_list, axis=0)
                path = self.get_path_from_name(base_name, '.hdf5')
                self.save_data(data, path)

        print(f'Succesfully created dataset in {self.destination_folder}')
        return True

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
        image = Image.fromarray(image)
        image.save(path)

    def save_data(self, data, path):
        """Save data to disk in hdf5 format

        Args:
            image (ndarray): input data
            path (str): file path
        """
        with h5py.File(path, 'w') as h5file:
            h5file[f'data/{self.to_dtype}'] = data
