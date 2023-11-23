"""The UI module runs a complete ultrasound beamforming pipeline and displays
the results in a GUI.

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from usbmd.datasets import get_dataset
from usbmd.display import to_8bit
from usbmd.generate import GenerateDataSet
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.setup_usbmd import setup
from usbmd.usbmd_gui import USBMDApp
from usbmd.utils.checks import _DATA_TYPES, _NON_IMAGE_DATA_TYPES
from usbmd.utils.config import Config
from usbmd.utils.io_lib import (
    filename_from_window_dialog,
    matplotlib_figure_to_numpy,
    running_in_notebook,
)
from usbmd.utils.selection_tool import interactive_selector_with_plot_and_metric
from usbmd.utils.utils import (
    plt_window_has_been_closed,
    save_to_gif,
    strtobool,
    update_dictionary,
)


class DataLoaderUI:
    """UI for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    """

    def __init__(self, config=None, verbose=True):
        self.config = Config(config)
        self.verbose = verbose

        # intialize dataset
        self.dataset = get_dataset(self.config.data)

        # Initialize scan based on dataset (if not image data)
        if self.config.data.dtype in _NON_IMAGE_DATA_TYPES:
            scan_class = self.dataset.get_scan_class()
            default_scan_params = self.dataset.get_default_scan_parameters()
            config_scan_params = self.config.scan

            # dict merging of manual config and dataset default scan parameters
            scan_params = update_dictionary(default_scan_params, config_scan_params)
            self.scan = scan_class(**scan_params)
        else:
            self.scan = None

        # initialize probe
        self.probe = get_probe(self.dataset.get_probe_name())

        # intialize process class
        self.process = Process(self.config, self.scan, self.probe)

        # initialize attributes for UI class
        self.data = None
        self.image = None
        self.file_path = None
        self.mpl_img = None
        self.fig = None
        self.ax = None
        self.gui = None

        if self.config.plot.headless is None:
            self.headless = False
        else:
            self.headless = self.config.plot.headless

        self.check_for_display()
        self.set_backend_for_notebooks()

    def check_for_display(self):
        """check if in headless mode (no monitor available)"""
        if self.headless is False:
            if matplotlib.get_backend().lower() == "agg":
                self.headless = True
                warnings.warn("Could not connect to display, running headless.")
        else:
            print("Running in headless mode as set by config.")

    def set_backend_for_notebooks(self):
        """Set backend to QtAgg if running in notebook"""
        if running_in_notebook() and not self.headless:
            matplotlib.use("QtAgg")

    def run(self, plot=True, to_dtype=None):
        """Run ui. Will retrieve, process and plot data if set to True."""

        to_dtype = self.config.data.to_dtype if to_dtype is None else to_dtype
        save = self.config.plot.save
        plot_lib = self.config.plot.plot_lib

        if self.config.data.get("frame_no") == "all":
            # run movie
            self.run_movie(save=save, to_dtype=to_dtype)
        else:
            # plot single frame
            self.data = self.get_data()

            self.image = self.process.run(
                self.data,
                dtype=self.config.data.dtype,
                to_dtype=to_dtype,
            )
            if self.process.postprocess:
                self.image = self.process.postprocess.run(self.image[None, ..., None])
                self.image = np.squeeze(self.image)

            if plot:
                self.plot(
                    self.image, block=(not self.gui), save=save, plot_lib=plot_lib
                )

        return self.image

    def get_data(self):
        """Get data. Chosen datafile should be listed in the dataset.

        Using either file specified in config or if None, the ui window.

        Returns:
            data (np.ndarray): data array of shape (n_tx, n_el, n_ax, N_ch)
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
                f"Choose .{filtetype} file",
                filetypes=((filtetype, "*." + filtetype),),
                initialdir=initialdir,
            )
            self.config.data.file_path = self.file_path

        if self.verbose:
            print(f"Selected {self.file_path}")

        # find file in dataset
        if self.file_path in self.dataset.file_paths:
            file_idx = self.dataset.file_paths.index(self.file_path)
        else:
            raise ValueError(
                f"Chosen datafile {self.file_path} does not exist in dataset!"
            )

        if self.config.data.get("frame_no") == "all":
            print("Will run all frames as `all` was chosen in config...")

        data = self.dataset[file_idx]

        return data

    def plot(
        self,
        image,
        image_range: tuple = None,
        save: bool = False,
        movie: bool = False,
        block: bool = True,
        plot_lib: str = "matplotlib",
    ):
        """Plot image.

        Args:
            image (ndarray): Log compressed enveloped detected image.
            image_range (tuple, optional): dynamic range of plot. Defaults to None,
                in that case the dynamic range in config is used.
            save (bool): whether to save the image to disk.
            movie (bool, optional): if True it will assume a figure object
                already exists and will overwrite the frame (to create a movie).
                If False will just create a new figure with each call. Defaults to False.
            block (bool, optional): halt program after plotting. Defaults to True.
            plot_lib (str, optional): type of plotting, with matplotlib or opencv.
        Returns:
            fig (fig): figure object.

        """
        if self.probe.probe_type == "phased":
            image = self.process.run(image, dtype="image", to_dtype="image_sc")

        if image_range is None:
            image_range = self.config.data.dynamic_range

        # match orientation
        image = np.fliplr(image)

        if movie:
            if plot_lib == "matplotlib":
                if self.mpl_img is None:
                    movie = False
                else:
                    self.mpl_img.set_data(image)
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                    image = matplotlib_figure_to_numpy(self.fig)
                    return image
            elif plot_lib == "opencv":
                image = to_8bit(image, image_range, pillow=False)
                if not self.headless:
                    cv2.imshow("frame", image)
                return image

        if not movie:
            if plot_lib == "matplotlib":
                self.fig, self.ax = plt.subplots()

                if self.scan:
                    extent = [
                        self.scan.xlims[0] * 1e3,
                        self.scan.xlims[1] * 1e3,
                        self.scan.zlims[1] * 1e3,
                        self.scan.zlims[0] * 1e3,
                    ]
                else:
                    extent = None

                self.mpl_img = self.ax.imshow(
                    image,
                    cmap="gray",
                    vmin=image_range[0],
                    vmax=image_range[1],
                    origin="upper",
                    extent=extent,
                    interpolation="none",
                )

                self.ax.set_xlabel("Lateral Width (mm)")
                self.ax.set_ylabel("Axial length (mm)")
                divider = make_axes_locatable(self.ax)

                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(self.mpl_img, cax=cax)

                self.fig.tight_layout()

                if self.config.plot.selector:
                    interactive_selector_with_plot_and_metric(
                        image,
                        self.ax,
                        extent=extent,
                        selector=self.config.plot.selector,
                        metric=self.config.plot.selector_metric,
                    )

                if save:
                    self.save_image(self.fig)
                if not self.headless:
                    plt.show(block=block)
                image = matplotlib_figure_to_numpy(self.fig)
                return image

            elif plot_lib == "opencv":
                image = to_8bit(image, self.config.data.dynamic_range)
                image.show()
                self.save_image(image)
                return image

    def run_movie(self, save: bool = False, to_dtype: str = "image"):
        """Run all frames in file in sequence"""

        if to_dtype not in ["image", "image_sc"]:
            warnings.warn(
                f"Image to_dtype: {to_dtype} not supported for movies."
                "falling back to  to_dtype: `image`"
            )
            to_dtype = "image"

        print('Playing video, press/hold "q" while the window is active to exit...')
        images = self.run_movie_loop(save, to_dtype)

        if save:
            self.save_video(images)

    def run_movie_loop(self, save, to_dtype):
        """
        Process data and plot it in real time.
        NOTE: when plot loop is terminated by user, it will only save the shown frames

        Args:
            plot_lib (str): The plotting library to use (either "matplotlib" or "opencv").
            n_frames (int): The total number of frames to plot.
            save (bool): Whether to save the plotted images.
            to_dtype (str): The data type to convert the plotted images to.

        Returns:
            list: A list of the plotted images.
        """
        # Initialize list of images
        images = []

        # Load correct number of frames (needs to get_data first)
        self.config.data.frame_no = 0
        self.get_data()
        n_frames = len(self.dataset.h5_reader)

        self.verbose = False
        plot_lib = self.config.plot.plot_lib
        while True:
            # first frame is already plotted during initialization of plotting
            for i in range(n_frames):
                if self.gui:
                    self.gui.check_freeze()

                self.config.data.frame_no = i
                self.data = self.get_data()

                image = self.process.run(
                    self.data, dtype=self.config.data.dtype, to_dtype=to_dtype
                )
                if self.process.postprocess:
                    image = self.process.postprocess.run(image[None, ..., None])
                    image = np.squeeze(image)

                image = self.plot(image, movie=True, plot_lib=plot_lib, block=False)

                print(f"frame {i}", end="\r")

                if save:
                    if len(images) < n_frames:
                        images.append(image)

                # For opencv, show frame for 25 ms and check if "q" is pressed
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return images
                # For matplotlib, check if window has been closed
                if plot_lib == "matplotlib":
                    if plt_window_has_been_closed(self.fig):
                        return images
                # For headless mode, check if all frames have been plotted
                if self.headless:
                    if len(images) == n_frames:
                        return images

            # clear line, frame number
            print("\x1b[2K", end="\r")

    def save_image(self, fig, path=None):
        """Save image to disk.

        Args:
            fig (fig object): figure.
            path (str, optional): path to save image to. Defaults to None.

        """
        if path is None:
            if self.config.plot.tag:
                tag = "_" + self.config.plot.tag
            else:
                tag = ""

            if self.dataset.frame_no is not None:
                filename = (
                    self.file_path.stem
                    + "-"
                    + str(self.dataset.frame_no)
                    + tag
                    + ".png"
                )
            else:
                filename = self.file_path.stem + tag + ".png"

            path = Path("./figures", filename)
            Path("./figures").mkdir(parents=True, exist_ok=True)

        if isinstance(fig, plt.Figure):
            fig.savefig(path, transparent=True)
        elif isinstance(fig, Image.Image):
            fig.save(path)
        else:
            raise ValueError("Figure is not PIL image or matplotlib figure object.")

        if self.verbose:
            print(f"Image saved to {path}")

    def save_video(self, images, path=None):
        """Save video to disk.

        Args:
            images (list): list of images.
            path (str, optional): path to save image to. Defaults to None.

        TODO: can only save gif and not mp4
        """
        if path is None:
            if self.config.plot.tag:
                tag = "_" + self.config.plot.tag
            else:
                tag = ""
            filename = self.file_path.stem + tag + ".gif"

            path = Path("./figures", filename)
            Path("./figures").mkdir(parents=True, exist_ok=True)

        if not isinstance(images[0], np.ndarray):
            raise ValueError("Images are not numpy arrays.")

        fps = self.config.plot.fps
        save_to_gif(images, path, fps=fps)

        if self.verbose:
            print(f"Video saved to {path}")


def get_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(description="Process ultrasound data.")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="path to config file."
    )
    parser.add_argument(
        "-t",
        "--task",
        default="run",
        choices=["run", "generate"],
        type=str,
        help="which task to run",
    )
    # pylint: disable=no-member
    parser.add_argument("--gui", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def main():
    """main entrypoint for UI script USBMD"""
    args = get_args()
    if args.gui:
        warnings.warn("GUI is very much in beta, please report any bugs to the Github.")
        gui = USBMDApp(title="USBMD GUI", resolution=(600, 300), verbose=True)

    config = setup(args.config)

    if args.task == "run":
        ui = DataLoaderUI(config)

        if args.gui:
            gui.ui = ui
            ui.gui = gui  # haha
            gui.build(config)
            gui.mainloop()
        else:
            ui.run()

    elif args.task == "generate":
        destination_folder = input(">> Give destination folder path: ")
        to_dtype = input(f">> Specify data type \n{_DATA_TYPES}: ")
        retain_folder_structure = input(">> Retain folder structure? (Y/N): ")
        retain_folder_structure = strtobool(retain_folder_structure)
        filetype = input(">> Filetype (hdf5, png): ")
        generator = GenerateDataSet(
            config,
            to_dtype=to_dtype,
            destination_folder=destination_folder,
            retain_folder_structure=retain_folder_structure,
            filetype=filetype,
        )
        generator.generate()


if __name__ == "__main__":
    main()
