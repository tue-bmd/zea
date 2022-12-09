"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : ui.py

    Author(s)     : Tristan Stevens
    Date          : Thu Nov 18 2021

==============================================================================
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

import usbmd.utils.git_info as git
from usbmd.datasets import _DATASETS, get_dataset
from usbmd.probes import get_probe
from usbmd.processing import (_BEAMFORMER_TYPES, _DATA_TYPES, _MOD_TYPES,
                              Process)
from usbmd.tensorflow_ultrasound.dataloader import GenerateDataSet
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.utils import (filename_from_window_dialog,
                               plt_window_has_been_closed, strtobool, to_image)


def check_config_file(config):
    """Check config file for inconsistencies in set parameters.

    Add necessary assertion checks for each parameter to be able to check
    its validity.

    Args:
        config (dict): config file.

    Returns:
        config (dict): config file.

    """
    config.data.dataset_name = config.data.dataset_name.lower()
    assert config.data.dataset_name in _DATASETS, \
        f'Dataset {config.data.dataset_name} does not exist,'\
        f'should be in:\n{_DATASETS}'
    assert config.data.dtype in _DATA_TYPES, \
        f'Dtype {config.data.dtype} does not exist,' \
        f'should be in:\n{_DATA_TYPES}'
    assert config.data.get('modtype') in _MOD_TYPES, \
        'Modulation type does not exist,' \
        f'should be in:\n{_MOD_TYPES}'
    assert config.model.beamformer.type in _BEAMFORMER_TYPES, \
        f'Beamformer {config.model.beamformer.type} does not exist,' \
        f'should be in:\n{_BEAMFORMER_TYPES}'

    return config

class DataLoaderUI:
    """UI for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    """

    def __init__(self, config=None, verbose=True):
        self.config = config
        self.verbose = verbose

        # intialize dataset
        self.dataset = get_dataset(self.config.data.dataset_name)(config=config.data)

        self.probe = get_probe(config, self.dataset)
        self.dataset.probe = self.probe

        # intialize process class
        self.process = Process(config, self.probe)

        self.data = None
        self.image = None
        self.file_path = None
        self.mpl_img = None
        self.fig = None
        self.ax = None

    def run(self, plot=True):
        """Run ui. Will retrieve, process and plot data if set to True."""

        if self.config.data.get('frame_no') == 'all':
            ## run movie
            self.run_movie()
        else:
            ## plot single frame
            self.data = self.get_data()

            self.image = self.process.run(self.data, dtype=self.config.data.dtype)

            if plot:
                save = self.config.get('plot', {}).get('save')
                axis = self.config.get('plot', {}).get('axis', True)
                self.plot(self.image, block=True, save=save, axis=axis)

        return self.image

    def get_data(self):
        """Get data. Chosen datafile should be listed in the dataset.

        Using either file specified in config or if None, the ui window.

        """
        if self.config.data.file_path:
            path = Path(self.config.data.file_path)
            if path.is_absolute():
                self.file_path = path
            else:
                self.file_path = self.dataset.data_root / path
        else:
            filtetype = self.dataset.filetype
            initialdir = self.dataset.data_root
            self.file_path = filename_from_window_dialog(
                f'Choose .{filtetype} file',
                filetypes=((filtetype, '*.' + filtetype),),
                initialdir=initialdir,
            )
            self.config.data.file_path = self.file_path

        if self.verbose:
            print(f'Selected {self.file_path}')

        # find file in dataset
        if self.file_path in self.dataset.file_paths:
            file_idx = self.dataset.file_paths.index(self.file_path)
        else:
            raise ValueError(f'Chosen datafile {self.file_path} does not exist in dataset!')

        if self.config.data.get('frame_no') == 'all':
            print('Will run all frames as `all` was chosen in config...')

        data = self.dataset[file_idx]

        return data

    def plot(
        self,
        image,
        image_range: tuple=None,
        save: bool=False,
        movie: bool=False,
        block: bool=True,
        axis: bool=True,
    ):
        """Plot image.

        Args:
            image (ndarray): Log compressed enveloped detected image.
            image_range (tuple, optional): dynamic range of plot. Defaults to None,
                in that case the dynamic range in config is used.
            save (bool): wheter to save the image to disk.
            movie (bool, optional): if True it will assume a figure object
                already exists and will overwrite the frame (to create a movie).
                If False will just create a new figure with each call. Defaults to False.
            block (bool, optional): halt program after plotting. Defaults to True.
            axis (bool, optional): type of plotting, with or without axis.
                axis set to `True` will result in matplotlib plot with axis.
                axis set to `False` will result in png image without axis and will
                use opencv for video rendering. Defaults to True.
        Returns:
            fig (fig): figure object.

        """
        if self.probe.probe_type == 'phased':
            image = self.process.run(image, dtype='image', to_dtype='image_sc')

        if not movie and axis:
            self.fig, self.ax = plt.subplots()

            extent = [
                self.probe.xlims[0] * 1e3,
                self.probe.xlims[1] * 1e3,
                self.probe.zlims[1] * 1e3,
                self.probe.zlims[0] * 1e3,
            ]

            if image_range is None:
                vmin, vmax = self.config.data.dynamic_range
            else:
                vmin, vmax = image_range

        if movie:
            if axis:
                if self.mpl_img is None:
                    raise ValueError('First run plot function without movie.')
                self.mpl_img.set_data(image)
                self.fig.canvas.draw_idle()
                return self.fig
            else:
                image = to_image(image, self.config.data.dynamic_range, pillow=False)
                cv2.imshow('frame', image)
                return
        else:
            if axis:
                self.mpl_img = self.ax.imshow(
                    image, cmap='gray', vmin=vmin, vmax=vmax,
                    origin='upper', extent=extent, interpolation='none',
                )

                self.ax.set_xlabel('Lateral Width (mm)')
                self.ax.set_ylabel('Axial length (mm)')
                divider = make_axes_locatable(self.ax)

                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(self.mpl_img, cax=cax)

                self.fig.tight_layout()

                if save:
                    self.save_image(self.fig)

                plt.show(block=block)
                return self.fig

            else:
                image = to_image(image, self.config.data.dynamic_range)
                image.show()
                self.save_image(image)
                return image

    def run_movie(self):
        """Run all frames in file in sequence"""

        print('Playing video, press "q" to exit...')
        axis = self.config.get('plot', {}).get('axis', True)
        self.config.data.frame_no = 0
        self.data = self.get_data()
        n_frames = len(self.dataset.h5object)

        # plot initial frame
        self.image = self.process.run(self.data, dtype=self.config.data.dtype)
        if axis:
            self.plot(self.image, axis=axis, block=False)
        else:
            self.plot(self.image, movie=True, axis=axis, block=False)

        # plot remaining frames in a loop
        self.verbose = False
        while True:
            for i in range(1, n_frames):
                self.config.data.frame_no = i
                self.data = self.get_data()
                image = self.process.run(self.data, dtype=self.config.data.dtype)
                self.plot(image, movie=True, axis=axis)
                print(f'frame {i}', end='\r')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
                if axis:
                    if plt_window_has_been_closed(self.fig):
                        return
            # clear line, frame number
            print('\x1b[2K', end='\r')

    def save_image(self, fig, path=None):
        """Save image to disk.

        Args:
            fig (fig object): figure.
            path (str, optional): path to save image to. Defaults to None.

        """
        if path is None:
            if self.dataset.frame_no is not None:
                filename = self.file_path.stem + '-' + str(self.dataset.frame_no) + '.png'
            else:
                filename = self.file_path.stem + '.png'

            path = Path('./figures', filename)
            Path('./figures').mkdir(parents=True, exist_ok=True)

        if isinstance(fig, plt.Figure):
            fig.savefig(path, transparent=True)
        elif isinstance(fig, Image.Image):
            fig.save(path)
        else:
            raise ValueError('Figure is not PIL image or matplotlib figure object.')

        if self.verbose:
            print(f'Image saved to {path}')


def setup(file=None):
    """Setup function. Retrieves config file and checks for validity.

    Args:
        file (str, optional): file path to config yaml. Defaults to None.
            if None, argparser is checked. If that is None as well, the window
            ui will pop up for choosing the config file manually.

    Returns:
        config (dict): config object / dict.

    """
    if file is None:
        # if no argument is provided resort to UI window
        if args.config is None:
            filetype = 'yaml'
            config_file = filename_from_window_dialog(
                f'Choose .{filetype} file',
                filetypes=((filetype, '*.' + filetype),),
                initialdir='./configs',
            )
        else:
            config_file = args.config
    else:
        config_file = file

    config = load_config_from_yaml(Path(config_file))
    config = check_config_file(config)

    print(f'Using config file: {config_file}')

    ## git
    cwd = Path.cwd().stem
    if cwd in ('Ultrasound-BMd', 'usbmd'):
        try:
            print('Git branch and commit: ')
            config['git'] = git.get_git_branch() + '=' + git.get_git_commit_hash()
            print(config['git'])
        except Exception:
            print('Cannot find Git')

    return config

def get_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(description='Process ultrasound data.')
    parser.add_argument('-c', '--config', type=str, default=None, help='path to config file.')
    parser.add_argument('-t', '--task',
        default='run', choices=['run', 'generate'],  type=str,
        help='which task to run')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = setup()

    if args.task == 'run':
        ui = DataLoaderUI(config)
        image = ui.run()
    elif args.task == 'generate':
        destination_folder = input('Give destination folder path: ')
        to_dtype = input(f'Specify data type \n{_DATA_TYPES}: ')
        retain_folder_structure = input('Retain folder structure? (Y/N): ')
        retain_folder_structure = strtobool(retain_folder_structure)
        generator = GenerateDataSet(
            config,
            to_dtype=to_dtype,
            destination_folder=destination_folder,
            retain_folder_structure=retain_folder_structure,
        )
        generator.generate()
