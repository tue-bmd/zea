"""The UI module runs a complete ultrasound beamforming pipeline and displays
the results in a GUI.

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
from usbmd.utils import log
from usbmd.utils.checks import _DATA_TYPES
from usbmd.utils.config import Config
from usbmd.utils.io_lib import (
    ImageViewerMatplotlib,
    ImageViewerOpenCV,
    filename_from_window_dialog,
    matplotlib_figure_to_numpy,
    running_in_notebook,
    start_async_app,
)
from usbmd.utils.utils import save_to_gif, save_to_mp4, strtobool, update_dictionary


class DataLoaderUI:
    """UI for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    """

    def __init__(self, config=None, verbose=True):
        self.config = Config(config)
        self.verbose = verbose

        # intialize dataset
        self.dataset = get_dataset(self.config.data)

        # Initialize scan based on dataset (if it can find proper scan parameters)
        scan_class = self.dataset.get_scan_class()
        default_scan_params = self.dataset.get_default_scan_parameters()

        if len(default_scan_params) == 0:
            log.info(
                f"Could not find proper scan parameters in {self.dataset} at "
                f"{log.yellow(str(self.dataset.datafolder))}."
            )
            log.info("Proceeding without scan class.")

            self.scan = None
        else:
            config_scan_params = self.config.scan
            # dict merging of manual config and dataset default scan parameters
            scan_params = update_dictionary(default_scan_params, config_scan_params)

            if "n_ax" not in scan_params:
                self.scan = scan_class(**scan_params, n_ax=2000, n_el=192, n_ch=1)
            else:
                self.scan = scan_class(**scan_params)

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
        self.image_viewer = None

        self.plot_lib = self.config.plot.plot_lib

        if self.config.plot.headless is None:
            self.headless = False
        else:
            self.headless = self.config.plot.headless

        self.check_for_display()
        self.set_backend_for_notebooks()

        if hasattr(self.dataset.file_name, "name"):
            window_name = str(self.dataset.file_name.name)
        else:
            window_name = "usbmd"

        if self.plot_lib == "opencv":
            self.image_viewer = ImageViewerOpenCV(
                self.data_to_display,
                window_name=window_name,
                num_threads=1,
            )
        elif self.plot_lib == "matplotlib":
            self.image_viewer = ImageViewerMatplotlib(
                self.data_to_display,
                window_name=window_name,
                num_threads=1,
            )

    @property
    def dtype(self):
        """Data type of data when loaded from file."""
        return self.config.data.dtype

    @dtype.setter
    def dtype(self, value):
        self.config.data.dtype = value

    @property
    def to_dtype(self):
        """Data type to convert to for display."""
        return self.config.data.to_dtype

    @to_dtype.setter
    def to_dtype(self, value):
        self.config.data.to_dtype = value

    def check_for_display(self):
        """check if in headless mode (no monitor available)"""
        if self.headless is False:
            if matplotlib.get_backend().lower() == "agg":
                self.headless = True
                log.warning("Could not connect to display, running headless.")
        else:
            matplotlib.use("agg")
            log.info("Running in headless mode as set by config.")

    def set_backend_for_notebooks(self):
        """Set backend to QtAgg if running in notebook"""
        if running_in_notebook() and not self.headless:
            matplotlib.use("QtAgg")

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
            if self.headless:
                raise ValueError(
                    "No file path specified for data file, which is required "
                    "in headless mode as window dialog cannot be opened."
                )
            filtetype = self.dataset.filetype
            initialdir = self.dataset.data_root
            self.file_path = filename_from_window_dialog(
                f"Choose .{filtetype} file",
                filetypes=((filtetype, "*." + filtetype),),
                initialdir=initialdir,
            )
            self.config.data.file_path = self.file_path

        if self.verbose:
            log.info(f"Selected {log.yellow(self.file_path)}")

        # find file in dataset
        if self.file_path in self.dataset.file_paths:
            file_idx = self.dataset.file_paths.index(self.file_path)
        else:
            raise ValueError(
                f"Chosen datafile {self.file_path} does not exist in dataset!"
            )

        if self.config.data.get("frame_no") == "all":
            log.info("Will run all frames as `all` was chosen in config...")

        data = self.dataset[file_idx]

        return data

    def data_to_display(self, data=None):
        """Get data and convert to display to_dtype."""
        if data is None:
            self.data = self.get_data()
        else:
            self.data = data

        if self.to_dtype not in ["image", "image_sc"]:
            log.warning(
                f"Image to_dtype: {self.to_dtype} not supported for displaying data."
                "falling back to  to_dtype: `image_sc`"
            )
            self.to_dtype = "image_sc"

        # we only need processing for dtypes other than image_sc
        if self.dtype != "image_sc":
            # if to_dtype == image_sc, first to go to dtype = image
            # then do processing and then convert to image_sc
            if self.to_dtype == "image_sc":
                to_dtype = "image"
            else:
                to_dtype = self.to_dtype

            self.image = self.process.run(
                self.data,
                dtype=self.dtype,
                to_dtype=to_dtype,
            )

            if self.process.postprocess:
                self.image = self.process.postprocess.run(self.image[None, ..., None])
                self.image = np.squeeze(self.image)

        else:
            # data is already in image_sc format
            self.image = self.data

        if self.to_dtype == "image_sc":
            self.image = self.process.run(
                self.image,
                dtype="image",
                to_dtype="image_sc",
            )

        # match orientation if necessary
        if self.config.plot.fliplr:
            self.image = np.fliplr(self.image)
        # opencv requires 8 bit images
        if self.plot_lib == "opencv":
            self.image = to_8bit(self.image, self.config.data.dynamic_range)
        return self.image

    def run(self, plot=False, block=True):
        """Run ui. Will retrieve, process and plot data if set to True."""
        save = self.config.plot.save

        if self.config.data.get("frame_no") == "all":
            if not asyncio.get_event_loop().is_running():
                asyncio.run(self.run_movie(save))
            else:
                asyncio.create_task(self.run_movie(save))

        else:
            if plot:
                self.image = self.plot(
                    save=save,
                    block=block,
                )
            else:
                self.image = self.data_to_display()

        return self.image

    def plot(
        self,
        data: np.ndarray = None,
        save: bool = False,
        block: bool = True,
    ):
        """Plot image using matplotlib or opencv.

        Args:
            save (bool): whether to save the image to disk.
            block (bool): whether to block the UI while plotting.
        Returns:
            image (np.ndarray): plotted image (grabbed from figure).
        """
        assert self.image_viewer is not None, "Image viewer not initialized."

        self.image_viewer.threading = False

        if self.plot_lib == "matplotlib":
            if self.image_viewer.fig is None:
                self._init_plt_figure()
            self.image_viewer.show(data)
            if save:
                self.save_image(self.fig)
            if not self.headless and block:
                plt.show(block=True)
            self.image = matplotlib_figure_to_numpy(self.fig)
            return self.image

        elif self.plot_lib == "opencv":
            self.image_viewer.show(data)
            if not self.headless and block:
                cv2.waitKey(0)
            self.save_image(self.image)
            return self.image

    def _init_plt_figure(self):
        figsize = (10, 10)
        if self.scan:
            extent = [
                self.scan.xlims[0] * 1e3,
                self.scan.xlims[1] * 1e3,
                self.scan.zlims[1] * 1e3,
                self.scan.zlims[0] * 1e3,
            ]
            # set figure aspect ratio to match scan
            aspect_ratio = abs(extent[1] - extent[0]) / abs(extent[3] - extent[2])
            figsize = tuple(np.array(figsize) * aspect_ratio)
        else:
            extent = None

        self.fig, self.ax = plt.subplots(figsize=figsize)
        # darkmode
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        text_color = "gray"
        for spine in self.ax.spines.values():
            spine.set_color(text_color)

        image_range = self.config.data.dynamic_range
        imshow_kwargs = {
            "cmap": "gray",
            "vmin": image_range[0],
            "vmax": image_range[1],
            "origin": "upper",
            "extent": extent,
            "interpolation": "none",
        }
        cax_kwargs = {
            "pad": 0.05,
            "position": "right",
            "color": text_color,
            "size": "5%",
        }

        self.ax.set_xlabel("Lateral Width (mm)", color=text_color)
        self.ax.set_ylabel("Axial length (mm)", color=text_color)
        self.ax.tick_params(axis="x", colors=text_color)
        self.ax.tick_params(axis="y", colors=text_color)

        # assign properties of fig, ax to image viewer
        self.image_viewer.imshow_kwargs = imshow_kwargs
        self.image_viewer.cax_kwargs = cax_kwargs
        self.image_viewer.fig = self.fig
        self.image_viewer.ax = self.ax

    async def run_movie(self, save: bool = False):
        """Run all frames in file in sequence"""

        log.info('Playing video, press/hold "q" while the window is active to exit...')
        self.image_viewer.threading = True
        images = await self._movie_loop(save)

        if save:
            self.save_video(images)

    async def _movie_loop(self, save: bool = False) -> List[np.ndarray]:
        """Process data and plot it in real time.

        NOTE: when plot loop is terminated by user, it will only save the shown frames.
        This is to prevent long waiting times when saving a movie (for large datasets).

        Args:
            save (bool): Whether to save the plotted images.

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
        # pylint: disable=too-many-nested-blocks
        try:
            while True:
                # first frame is already plotted during initialization of plotting
                start_time = time.time()
                frame_counter = 0
                self.image_viewer.frame_no = 0
                while frame_counter < n_frames:
                    if self.gui:
                        await self.gui.check_freeze()

                    await asyncio.sleep(0.01)

                    self.config.data.frame_no = frame_counter

                    if frame_counter == 0:
                        if self.plot_lib == "matplotlib":
                            if self.image_viewer.fig is None:
                                self._init_plt_figure()

                    self.image_viewer.show()

                    # set counter to frame number of image viewer (possibly not updated)
                    frame_counter = self.image_viewer.frame_no

                    # check if frame counter updated
                    if frame_counter != self.config.data.frame_no:
                        fps = frame_counter / (time.time() - start_time)
                        print(
                            f"frame {frame_counter} / {n_frames} ({fps:.2f} fps)",
                            end="\r",
                        )
                        if save and (len(images) < n_frames):
                            if self.plot_lib == "matplotlib":
                                # grab image from plt figure
                                image = matplotlib_figure_to_numpy(self.fig)
                            else:
                                image = np.array(self.image)
                            images.append(image)

                    # For opencv, show frame for 25 ms and check if "q" is pressed
                    if self.plot_lib == "opencv":
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            self.image_viewer.close()
                            return images
                        if self.image_viewer.has_been_closed():
                            return images
                    # For matplotlib, check if window has been closed
                    elif self.plot_lib == "matplotlib":
                        if cv2.waitKey(25) and self.image_viewer.has_been_closed():
                            return images
                    # For headless mode, check if all frames have been plotted
                    if self.headless:
                        if len(images) == n_frames:
                            return images

                # clear line, frame number
                print("\x1b[2K", end="\r")

                # only loop once if in headless mode
                if self.headless:
                    return images

        except KeyboardInterrupt:
            if save:
                if len(images) > 0:
                    self.save_video(images)
            raise

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
                    + "."
                    + self.config.plot.image_extension
                )
            else:
                filename = (
                    self.file_path.stem + tag + "." + self.config.plot.image_extension
                )

            path = Path("./figures", filename)
            Path("./figures").mkdir(parents=True, exist_ok=True)

        if isinstance(fig, plt.Figure):
            fig.savefig(path, transparent=True)
        elif isinstance(fig, Image.Image):
            fig.save(path)
        else:
            raise ValueError(
                f"Figure is not PIL image or matplotlib figure object, got {type(fig)}"
            )

        if self.verbose:
            log.info(f"Image saved to {log.yellow(path)}")

    def save_video(self, images, path=None):
        """Save video to disk.

        Args:
            images (list): list of images.
            path (str, optional): path to save image to. Defaults to None.

        """
        if path is None:
            if self.config.plot.tag:
                tag = "_" + self.config.plot.tag
            else:
                tag = ""
            filename = (
                self.file_path.stem + tag + "." + self.config.plot.video_extension
            )

            path = Path("./figures", filename)
            Path("./figures").mkdir(parents=True, exist_ok=True)

        if not isinstance(images[0], np.ndarray):
            raise ValueError("Images are not numpy arrays.")

        fps = self.config.plot.fps

        if self.config.plot.video_extension == "gif":
            save_to_gif(images, path, fps=fps)
        elif self.config.plot.video_extension == "mp4":
            save_to_mp4(images, path, fps=fps)

        if self.verbose:
            log.info(f"Video saved to {log.yellow(path)}")


def _try(fn, args=None, required_set=None):
    """Keep trying to run a function until it succeeds.
    Args:
        fn (function): function to run
        args (dict, optional): arguments to pass to function
        required_set (set, optional): set of required outputs
            if output is not in required_set, function will be rerun
    """
    while True:
        try:
            out = fn(**args) if args is not None else fn()
            if required_set is not None:
                assert out is not None
                assert out in required_set, f"Output {out} not in {required_set}"
            return out
        except Exception as e:
            print(e)


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
    config = setup(args.config)

    if args.task == "run":
        ui = DataLoaderUI(config)

        if args.gui:
            log.warning(
                "GUI is very much in beta, please report any bugs to "
                "https://github.com/tue-bmd/ultrasound-toolbox."
            )
            try:
                asyncio.run(
                    start_async_app(
                        USBMDApp,
                        title="USBMD GUI",
                        ui=ui,
                        resolution=(600, 300),
                        verbose=True,
                        config=config,
                    )
                )
            except RuntimeError as e:
                # probably a better way to handle this...
                if str(e) == "Event loop stopped before Future completed.":
                    log.info("GUI closed.")
                else:
                    raise e
        else:
            ui.run(plot=True)

    elif args.task == "generate":
        destination_folder = _try(
            lambda: input(
                ">> Give destination folder path"
                + " (if relative path, will be relative to the original dataset): "
            )
        )
        to_dtype = _try(
            lambda: input(f">> Specify data type \n{_DATA_TYPES}: "),
            required_set=_DATA_TYPES,
        )
        retain_folder_structure = _try(
            lambda: strtobool(input(">> Retain folder structure? (Y/N): "))
        )
        filetype = _try(
            lambda: input(">> Filetype (hdf5, png): "), required_set=["hdf5", "png"]
        )

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
