"""The UI module runs a complete ultrasound beamforming pipeline and displays
the results in a GUI.

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

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
from usbmd.utils.checks import _DATA_TYPES, _NON_IMAGE_DATA_TYPES
from usbmd.utils.config import Config
from usbmd.utils.io_lib import (
    ImageViewerMatplotlib,
    ImageViewerOpenCV,
    filename_from_window_dialog,
    matplotlib_figure_to_numpy,
    running_in_notebook,
)
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
            self.scan = scan_class(**scan_params, modtype=self.config.data.modtype)
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

        if not self.headless:
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

    def data_to_display(self, to_dtype: str = "image_sc"):
        """Get data and convert to display dtype."""
        self.data = self.get_data()

        # if image_sc, first to image for postprocessing and afterwards to image_sc
        _to_dtype = to_dtype if to_dtype != "image_sc" else "image"
        self.image = self.process.run(
            self.data,
            dtype=self.config.data.dtype,
            to_dtype=_to_dtype,
        )

        if self.process.postprocess:
            self.image = self.process.postprocess.run(self.image[None, ..., None])
            self.image = np.squeeze(self.image)

        if to_dtype == "image_sc":
            self.image = self.process.run(
                self.image, dtype="image", to_dtype="image_sc"
            )

        # match orientation if necessary
        if self.config.plot.fliplr:
            self.image = np.fliplr(self.image)
        # opencv requires 8 bit images
        if self.plot_lib == "opencv":
            self.image = to_8bit(self.image, self.config.data.dynamic_range)
        return self.image

    def run(self, plot=False, to_dtype=None):
        """Run ui. Will retrieve, process and plot data if set to True."""

        to_dtype = self.config.data.to_dtype if to_dtype is None else to_dtype

        if self.config.data.get("frame_no") == "all":
            self.run_movie(save=self.config.plot.save, to_dtype=to_dtype)
        else:
            if plot:
                self.image = self.plot(
                    save=self.config.plot.save,
                )
            else:
                self.image = self.data_to_display(to_dtype=to_dtype)

        return self.image

    def plot(
        self,
        save: bool = False,
    ):
        """Plot image using matplotlib or opencv.

        Args:
            save (bool): whether to save the image to disk.
        Returns:
            image (np.ndarray): plotted image (grabbed from figure).
        """
        if self.headless:
            return self.data_to_display()

        assert self.image_viewer is not None, "Image viewer not initialized."

        self.image_viewer.threading = False

        if self.plot_lib == "matplotlib":
            if self.image_viewer.fig is None:
                self._init_plt_figure()
            self.image_viewer.show()
            if save:
                self.save_image(self.fig)
            if not self.headless:
                plt.show(block=True)
            self.image = matplotlib_figure_to_numpy(self.fig)
            return self.image

        elif self.plot_lib == "opencv":
            self.image_viewer.show()
            self.save_image(self.image)
            cv2.waitKey(0)
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

    def run_movie(self, save: bool = False, to_dtype: str = "image"):
        """Run all frames in file in sequence"""

        if to_dtype not in ["image", "image_sc"]:
            warnings.warn(
                f"Image to_dtype: {to_dtype} not supported for movies."
                "falling back to  to_dtype: `image`"
            )
            to_dtype = "image"

        print('Playing video, press/hold "q" while the window is active to exit...')
        self.image_viewer.threading = True
        images = self.run_movie_loop(save)

        if save:
            self.save_video(images)

    def run_movie_loop(self, save):
        """
        Process data and plot it in real time.
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
        # self.get_data()
        n_frames = len(self.dataset.h5_reader)

        self.verbose = False
        while True:
            # first frame is already plotted during initialization of plotting
            start_time = time.time()
            frame_counter = 0
            self.image_viewer.frame_no = 0

            while frame_counter < n_frames:
                if self.gui:
                    self.gui.check_freeze()

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
                # For matplotlib, check if window has been closed
                if self.plot_lib == "matplotlib":
                    if cv2.waitKey(25) & plt_window_has_been_closed(self.fig):
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
            ui.run(plot=True)

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
