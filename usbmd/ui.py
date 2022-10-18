"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : ui.py

    Author(s)     : Tristan Stevens
    Date          : Thu Nov 18 2021
    
==============================================================================
"""
import sys
from pathlib import Path
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

import argparse

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import usbmd.utils.git_info as git
from usbmd.datasets import _DATASETS, get_dataset
from usbmd.probes import get_probe
from usbmd.processing import (_BEAMFORMER_TYPES, _DATA_TYPES, _MOD_TYPES,
                              Process)
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.utils import (filename_from_window_dialog,
                               plt_window_has_been_closed)


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
        
    def run(self, plot=True):
        """Run ui. Will retrieve, process and plot data if set to True."""
        if self.config.data.get('frame_no') == 'all':
            self.run_movie()
        else:
            self.data = self.get_data()

        self.image = self.process.run(self.data, dtype=self.config.data.dtype)

        if plot:
            save = self.config.get('plot', {}).get('save')
            self.plot(self.image, save=save, block=True)

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
    
    def plot(self, image, image_range=None, save=False, movie=False, block=True):
        """Plot image.
        
        Args:
            image (ndarray): Log compressed enveloped detected image.

        Returns:
            fig (fig): figure object.
            save (bool): wheter to save the image to disk.

        """
        if not movie:
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

        if self.probe.probe_type == 'phased':
            image = self.process.run(image, dtype='image', to_dtype='image_sc')
            
        if movie:
            self.im.set_data(image)
            self.fig.canvas.draw_idle()
        else:
            self.im = self.ax.imshow(
                image, cmap='gray', vmin=vmin, vmax=vmax, 
                origin='upper', extent=extent, interpolation='none',
            )
        
            self.ax.set_xlabel('Lateral Width (mm)')
            self.ax.set_ylabel('Axial length (mm)')
            divider = make_axes_locatable(self.ax)
            
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(self.im, cax=cax)
            
            self.fig.tight_layout()
        
            if save:
                self.save_image(self.fig)
            
            plt.show(block=block)
        
        return self.fig

    def run_movie(self):
        """Run all frames in file in sequence"""

        print('Playing video, press "q" to exit...')
        self.config.data.frame_no = 0
        self.data = self.get_data()
        n_frames = len(self.dataset.h5object)
        self.image = self.process.run(self.data, dtype=self.config.data.dtype)
        self.plot(self.image, save=self.config.plot.save, block=False)
        
        self.verbose = False
        while True:
            for i in range(1, n_frames):
                self.config.data.frame_no = i
                self.data = self.get_data()
                image = self.process.run(self.data, dtype=self.config.data.dtype)
                self.plot(image, movie=True)
                print(f'frame {i}', end='\r')
                cv2.waitKey(1)
                if plt_window_has_been_closed(self.ax):
                    return self.image
            # clear line, frame number
            print('\x1b[2K', end='\r')
            
    def save_image(self, fig, path=None):
        """Save image to disk.

        Args:
            fig (fig object): figure.
            path (str, optional): path to save image to. Defaults to None.

        """
        if path is None:
            filename = ui.file_path.stem + '.png'
            path = Path('./figures', filename)
            Path('./figures').mkdir(parents=True, exist_ok=True)
        fig.savefig(path, transparent=True)
        if self.verbose:
            print('Image saved to {}'.format(path))


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
        # argparse option
        parser = argparse.ArgumentParser(description='Process ultrasound data.')
        parser.add_argument('-c', '--config', type=str, default=None, help='path to config file.')
        args = parser.parse_args()
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
    if cwd == 'Ultrasound-BMd' or cwd == 'usbmd':
        try:
            print('Git branch and commit: ')
            config['git'] = git.get_git_branch() + '=' + git.get_git_commit_hash()
            print(config['git'])
        except:
            print('Cannot find Git')
        
    return config
    
if __name__ == '__main__':
    config = setup()
    ui = DataLoaderUI(config)
    image = ui.run()