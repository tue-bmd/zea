"""The interface module runs a complete ultrasound beamforming pipeline and displays
the results in a GUI.

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""

import asyncio
import time
from pathlib import Path
from typing import List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from usbmd.config import Config
from usbmd.data import get_dataset
from usbmd.display import to_8bit
from usbmd.probes import get_probe
from usbmd.processing import Process
from usbmd.utils import (
    log,
    safe_initialize_class,
    save_to_gif,
    save_to_mp4,
    update_dictionary,
)
from usbmd.utils.io_lib import (
    ImageViewerMatplotlib,
    ImageViewerOpenCV,
    filename_from_window_dialog,
    matplotlib_figure_to_numpy,
    running_in_notebook,
)
from usbmd.utils.utils import keep_trying


class Interface:
    """Interface for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    """

    def __init__(self, config=None, verbose=True, dataset_kwargs=None):
        self.config = Config(config)
        self.verbose = verbose

        # intialize dataset
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.dataset = get_dataset(self.config.data, **dataset_kwargs)

        # Initialize scan based on dataset (if it can find proper scan parameters)
        scan_class = self.dataset.get_scan_class()
        file_scan_params = self.dataset.get_scan_parameters_from_file(event=0)
        file_probe_params = self.dataset.get_probe_parameters_from_file(event=0)

        self.scan = None
        if len(file_scan_params) == 0:
            log.info(
                f"Could not find proper scan parameters in {self.dataset} at "
                f"{log.yellow(str(self.dataset.datafolder))}."
            )
            log.info("Proceeding without scan class.")
        else:
            self.config_scan_params = self.config.scan
            # dict merging of manual config and dataset default scan parameters
            self.scan_params = update_dictionary(
                file_scan_params, self.config_scan_params
            )
            try:
                self.scan = safe_initialize_class(scan_class, **self.scan_params)
            except Exception as e:
                log.error(
                    "Could not initialize scan class with parameters: "
                    f"{self.scan_params}\n{e}"
                )

        # initialize probe
        probe_name = self.dataset.get_probe_name()

        if probe_name == "generic":
            self.probe = get_probe(probe_name, **file_probe_params)
        else:
            self.probe = get_probe(probe_name)

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

        if hasattr(self.dataset.file_name, "name"):
            window_name = str(self.dataset.file_name.name)
        else:
            window_name = "usbmd"

        if self.plot_lib == "opencv":
            self.image_viewer = ImageViewerOpenCV(
                self.data_to_display,
                window_name=window_name,
                num_threads=1,
                headless=self.headless,
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
            # self.plot_lib = "matplotlib"  # force matplotlib in headless mode
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
            log.info("Please select file from window dialog...")
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

        # grab frame number from config or user input if not set in config
        frame_no = self.config.data.get("frame_no")

        if frame_no == "all":
            log.info("Will run all frames as `all` was chosen in config...")
        elif frame_no is None:
            if self.dataset.num_frames == 1:
                frame_no = 0
            else:
                frame_no = keep_trying(
                    lambda: int(
                        input(f">> Frame number (0 / {self.dataset.num_frames - 1}): ")
                    )
                )

        # get data from dataset
        data = self.dataset[(file_idx, frame_no)]

        ## Update scan class (probably a cleaner way to do this)
        # check if event data by checking self.dataset.file keys start with event
        if self.dataset.event_structure:
            # this is still under development
            scan_params = self.dataset.get_scan_parameters_from_file(
                file_idx=self.dataset.file, event=self.dataset.frame_no
            )
            scan_class = self.dataset.get_scan_class()

            scan_params = update_dictionary(scan_params, self.config_scan_params)
            self.scan_params = update_dictionary(self.scan_params, scan_params)
            self.scan = scan_class(**self.scan_params)

            # TODO: use adaptive beamformer processing instead of reinit
            self.process = Process(self.config, self.scan, self.probe)
            # print(f"frame: {self.dataset.frame_no}, angles: {scan_params['polar_angles']}")

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

        if self.process.pipeline is None:
            if self.config.preprocess.operation_chain is None:
                self.process.set_pipeline(
                    dtype=self.dtype,
                    to_dtype=self.to_dtype,
                    verbose=self.verbose,
                )
            else:
                self.process.set_pipeline(
                    operation_chain=self.config.preprocess.operation_chain,
                    verbose=self.verbose,
                )

        self.image = self.process.run(self.data)

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
            "size": "5%",
        }

        self.ax.set_xlabel("Lateral Width (mm)", size=15)
        self.ax.set_ylabel("Axial length (mm)", size=15)
        self.ax.tick_params(axis="x")
        self.ax.tick_params(axis="y")

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
        n_frames = self.dataset.num_frames

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
                    if not self.headless:
                        if self.plot_lib == "opencv":
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                self.image_viewer.close()
                                return images
                            if self.image_viewer.has_been_closed():
                                return images
                        # For matplotlib, check if window has been closed
                        elif self.plot_lib == "matplotlib":
                            if (
                                time.sleep(0.025)
                                and self.image_viewer.has_been_closed()
                            ):
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
