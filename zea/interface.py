"""Convenience interface for loading and displaying ultrasound data.

Example usage
^^^^^^^^^^^^^^

.. code-block:: python

    import zea
    from zea.internal.setup_zea import setup_config

    config = setup_config("hf://zeahub/configs/config_camus.yaml")

    interface = zea.Interface(config)
    interface.run(plot=True)

"""

import asyncio
import time
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zea import log
from zea.config import Config
from zea.data.file import File
from zea.datapaths import format_data_path
from zea.display import to_8bit
from zea.internal.core import DataTypes
from zea.internal.viewer import (
    ImageViewerMatplotlib,
    ImageViewerOpenCV,
    filename_from_window_dialog,
    running_in_notebook,
)
from zea.io_lib import matplotlib_figure_to_numpy
from zea.ops import Pipeline
from zea.utils import keep_trying, save_to_gif, save_to_mp4


class Interface:
    """Interface for selecting / loading / processing single ultrasound images.

    Useful for inspecting datasets and single ultrasound images.

    # TODO: maybe we can refactor such that it is clear what needs to be in config.
    """

    def __init__(self, config: Config = None, verbose: bool = True, validate_file: bool = True):
        """Initialize Interface.

        Args:
            config (Config): Configuration object.
            verbose (bool): Whether to print verbose output.
            validate_file (bool): Whether to validate the file.
        """
        self.config = Config(config)
        self.verbose = verbose

        self.file = File(self.file_path)

        if validate_file:
            self.file.validate()

        # get probe and scan from file
        self.probe = self.file.probe()
        self.scan = self.file.scan(**self.config.scan)

        # initialize Pipeline
        assert "pipeline" in self.config, (
            "Pipeline not found in config, please specify pipeline in config."
        )

        self.process = Pipeline.from_config(
            self.config.pipeline,
            with_batch_dim=False,
            jit_options=None,
        )
        self.parameters = self.process.prepare_parameters(self.probe, self.scan, self.config)

        # initialize attributes for UI class
        self.data = None
        self.image = None
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

        if self.plot_lib == "opencv":
            self.image_viewer = ImageViewerOpenCV(
                self.data_to_display,
                window_name=self.file.name,
                num_threads=1,
                headless=self.headless,
            )
        elif self.plot_lib == "matplotlib":
            self.image_viewer = ImageViewerMatplotlib(
                self.data_to_display,
                window_name=self.file.name,
                num_threads=1,
            )

    @property
    def dtype(self):
        """Data type of data when loaded from file."""
        return self.config.data.dtype

    @property
    def dataset_folder(self):
        """Path to dataset folder."""
        return format_data_path(self.config.data.dataset_folder, self.config.data.user)

    @property
    def file_path(self):
        """Path to data file."""
        if self.config.data.file_path:
            return self.dataset_folder / self.config.data.file_path
        else:
            return self.choose_file_path()

    @file_path.setter
    def file_path(self, value):
        """Set file path to data file."""
        self.config.data.file_path = value

    def choose_file_path(self):
        """Choose file path from window dialog."""
        if self.headless:
            raise ValueError(
                "No file path specified for data file, which is required "
                "in headless mode as window dialog cannot be opened."
            )
        filetype = "hdf5"
        log.info("Please select file from window dialog...")
        self.file_path = filename_from_window_dialog(
            f"Choose .{filetype} file",
            filetypes=((filetype, "*." + filetype),),
            initialdir=self.dataset_folder,
        )
        return self.file_path

    @property
    def data_root(self):
        """Root path to data file."""
        return Path(self.config.user.data_root)

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

    @property
    def frame_no(self):
        """Frame number to display."""
        return self.config.data.get("frame_no")

    @frame_no.setter
    def frame_no(self, value):
        self.config.data.frame_no = value

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
        if self.verbose:
            log.info(f"Selected {log.yellow(self.file_path)}")

        # grab frame number from config or user input if not set in config
        if self.frame_no == "all":
            log.info("Will run all frames as `all` was chosen in config...")
        elif self.frame_no is None:
            if self.file.n_frames == 1:
                self.frame_no = 0
            else:
                self.frame_no = keep_trying(
                    lambda: int(input(f">> Frame number (0 / {self.file.n_frames - 1}): "))
                )

        # get data from dataset
        data = self.file.load_data(self.dtype, self.frame_no)

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

        # select transmits if raw or aligned data
        data_type = self.process.operations[0].input_data_type
        if data_type in [DataTypes.RAW_DATA, DataTypes.ALIGNED_DATA]:
            n_tx = self.data.shape[0]
            assert len(self.scan.selected_transmits) <= n_tx, (
                f"Number of selected transmits {len(self.scan.selected_transmits)} "
                f"exceeds number of transmits in raw data {n_tx}"
            )
            self.data = np.take(self.data, self.scan.selected_transmits, axis=0)

        inputs = {self.process.key: self.data}

        outputs = self.process(**inputs, **self.parameters)

        self.image = outputs[self.process.output_key]

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

        if self.frame_no == "all":
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
                self.image_viewer._cv2.waitKey(0)
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
        self.frame_no = 0
        self.get_data()
        n_frames = self.file.n_frames

        self.verbose = False
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

                    self.frame_no = frame_counter

                    if frame_counter == 0:
                        if self.plot_lib == "matplotlib":
                            if self.image_viewer.fig is None:
                                self._init_plt_figure()

                    self.image_viewer.show()

                    # set counter to frame number of image viewer (possibly not updated)
                    frame_counter = self.image_viewer.frame_no

                    # check if frame counter updated
                    if frame_counter != self.frame_no:
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
                            if self.image_viewer._cv2.waitKey(25) & 0xFF == ord("q"):
                                self.image_viewer.close()
                                return images
                            if self.image_viewer.has_been_closed():
                                return images
                        # For matplotlib, check if window has been closed
                        elif self.plot_lib == "matplotlib":
                            if time.sleep(0.025) and self.image_viewer.has_been_closed():
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

            if self.frame_no is not None:
                filename = self.file_path.stem + "-" + str(self.frame_no) + tag
            else:
                filename = self.file_path.stem + tag

            ext = f".{self.config.plot.image_extension.lstrip('.')}"

            path = Path("./figures", filename).with_suffix(ext)
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
            filename = self.file_path.stem + tag + "." + self.config.plot.video_extension

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

    def __del__(self):
        try:
            if self.image_viewer is not None:
                self.image_viewer.close()
        except Exception:
            pass
        try:
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
