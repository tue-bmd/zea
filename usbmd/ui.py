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
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from usbmd.common import set_data_paths
from usbmd.datasets import get_dataset
from usbmd.generate import GenerateDataSet
from usbmd.probes import get_probe
from usbmd.processing import _DATA_TYPES, Process, get_contrast_boost_func
from usbmd.utils.config import Config, load_config_from_yaml
from usbmd.utils.config_validation import check_config
from usbmd.utils.git_info import get_git_summary
from usbmd.utils.utils import (filename_from_window_dialog,
                               plt_window_has_been_closed, save_to_gif,
                               strtobool, to_image)


class DataLoaderUI:
    """UI for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    """

    def __init__(self, config=None, verbose=True):
        self.config = config
        self.verbose = verbose

        # intialize dataset
        self.dataset = get_dataset(self.config.data)

        # Initialize scan based on dataset
        scan_class = self.dataset.get_scan_class()
        default_scan_params = self.dataset.get_default_scan_parameters()
        config_scan_params = self.config.scan

        # dict merging python > 3.9: default_scan_params | config_scan_params
        scan_params = {**default_scan_params, **config_scan_params}
        self.scan = scan_class(**scan_params)

        # initialize probe
        self.probe = get_probe(self.dataset.get_probe_name())

        # intialize process class
        self.process = Process(config, self.scan, self.probe)

        # initialize attributes for UI class
        self.data = None
        self.image = None
        self.file_path = None
        self.mpl_img = None
        self.fig = None
        self.ax = None
        self.headless = False

        # initialize post processing tools
        if 'postprocess' in self.config:
            if 'contrast_boost' in self.config.postprocess:
                self.contrast_boost = get_contrast_boost_func()
            if 'lista' in self.config.postprocess:
                # initialize neural network
                pass
            # etc...

        self.check_for_display()

    def check_for_display(self):
        """check if in headless mode (no monitor available)"""
        # first read from config, headless could be an option
        if self.config.plot.headless is not None:
            self.headless = self.config.plot.headless
        else:
            self.headles = False
        # check if non headless mode is possible
        if self.headless is False:
            if plt.rcParams['backend'].lower() == 'agg':
                self.headless = True
                warnings.warn('Could not connect to display, running headless.')
        else:
            print('Running in headless mode as set by config.')

    def run(self, plot=True, to_dtype=None):
        """Run ui. Will retrieve, process and plot data if set to True."""

        to_dtype = 'image' if to_dtype is None else to_dtype
        save = self.config.plot.save
        axis = self.config.plot.axis

        if self.config.data.get('frame_no') == 'all':
            if to_dtype != 'image':
                warnings.warn(
                    f'Image to_dtype: {to_dtype} not yet supported for movies.\
                        falling back to  to_dtype: `image`')
            ## run movie
            self.run_movie(save=save)
        else:
            ## plot single frame
            self.data = self.get_data()

            self.image = self.process.run(
                self.data,
                dtype=self.config.data.dtype,
                to_dtype=to_dtype)

            if plot:
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

    def postprocess(self, image):
        """Post processing in image domain."""
        if not 'postprocess' in self.config:
            return image

        if 'contrast_boost' in self.config.postprocess:
            if self.config.data.dtype not in ['raw_data', 'aligned_data']:
                warnings.warn(f'contrast boost not possible with {self.config.data.dtype}')
                return image
            apodization = self.config.data.apodization
            self.config.data.apodization = 'checkerboard'
            noise = self.process.run(self.data, dtype=self.config.data.dtype)
            self.config.data.apodization = apodization
            image = self.contrast_boost(image, noise, **self.config.postprocess.contrast_boost)

        return image

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
                self.scan.xlims[0] * 1e3,
                self.scan.xlims[1] * 1e3,
                self.scan.zlims[1] * 1e3,
                self.scan.zlims[0] * 1e3,
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
                self.fig.canvas.flush_events()
                return self.fig
            else:
                image = to_image(image, self.config.data.dynamic_range, pillow=False)
                if not self.headless:
                    cv2.imshow('frame', image)
                return image
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
                if not self.headless:
                    plt.show(block=block)
                return self.fig

            else:
                image = to_image(image, self.config.data.dynamic_range)
                image.show()
                self.save_image(image)
                return image

    def run_movie(self, save: bool=False):
        """Run all frames in file in sequence"""

        print('Playing video, press "q" to exit...')
        axis = self.config.plot.axis
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
        images = []
        self.verbose = False
        while True:
            for i in range(1, n_frames):
                self.config.data.frame_no = i
                self.data = self.get_data()

                image = self.process.run(self.data, dtype=self.config.data.dtype)

                if 'postprocess' in self.config:
                    image = self.postprocess(image)

                image = self.plot(image, movie=True, axis=axis)
                print(f'frame {i}', end='\r')

                if save:
                    if len(images) < n_frames:
                        images.append(image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.save_video(images)
                    return
                if axis:
                    if plt_window_has_been_closed(self.fig):
                        self.save_video(images)
                        return

                if self.headless:
                    if len(images) == n_frames:
                        self.save_video(images)
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
            if self.config.plot.tag:
                tag = '_' + self.config.plot.tag
            else:
                tag = ''

            if self.dataset.frame_no is not None:
                filename = self.file_path.stem + '-' + str(self.dataset.frame_no) + tag + '.png'
            else:
                filename = self.file_path.stem + tag + '.png'

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

    def save_video(self, images, path=None):
        """Save video to disk.

        Args:
            images (list): list of images.
            path (str, optional): path to save image to. Defaults to None.

        TODO: can only save gif and not mp4
        TODO: plt figures (axis=true in config) are not supported with save_video

        """
        if path is None:
            if self.config.plot.tag:
                tag = '_' + self.config.plot.tag
            else:
                tag = ''
            filename = self.file_path.stem + tag + '.gif'

            path = Path('./figures', filename)
            Path('./figures').mkdir(parents=True, exist_ok=True)

        if isinstance(images[0], plt.Figure):
            raise NotImplementedError('Saving videos using matplotlib '\
                                      '(`axis = True` in config) not yet supported')
        if isinstance(images[0], np.ndarray):
            fps = self.config.plot.fps
            save_to_gif(images, path, fps=fps)
        else:
            raise ValueError('Figure is not a numpy array or matplotlib figure object.')

        if self.verbose:
            print(f'Video saved to {path}')

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
        filetype = 'yaml'
        try:
            file = filename_from_window_dialog(
                f'Choose .{filetype} file',
                filetypes=((filetype, '*.' + filetype),),
                initialdir='./configs',
            )
        except Exception as e:
            raise ValueError (
                'Please specify the path to a config file through --config flag ' \
                'if GUI is not working (usually on headless servers).') from e

    config = load_config_from_yaml(Path(file))
    print(f'Using config file: {file}')
    config = check_config(config.serialize())
    config = Config(config)

    ## git
    cwd = Path.cwd().stem
    if cwd in ('Ultrasound-BMd', 'usbmd'):
        config['git'] = get_git_summary()

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

def main():
    """main entrypoint for UI script USBMD"""
    args = get_args()
    set_data_paths()
    config = setup(file=args.config)

    if args.task == 'run':
        ui = DataLoaderUI(config)
        image = ui.run()
        return image
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

if __name__ == '__main__':
    main()
