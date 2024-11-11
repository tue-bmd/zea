"""Input/output functions

Use to quickly read and write files or interact with file system.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 12th, 2023
"""

import abc
import asyncio
import functools
import multiprocessing
import os
import sys
import warnings
from collections import deque
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import Callable, Optional

import cv2
import h5py
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tqdm
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from pydicom.pixel_data_handlers import convert_color_space
from PyQt5.QtCore import QRect

from usbmd.utils import log

_SUPPORTED_VID_TYPES = [".avi", ".mp4", ".gif", ""]
_SUPPORTED_IMG_TYPES = [".jpg", ".png", ".JPEG", ".PNG", ".jpeg"]
_SUPPORTED_USBMD_TYPES = [".hdf5", ".h5"]


def running_in_notebook():
    """Check whether code is running in a Jupyter Notebook or not."""
    return "ipykernel_launcher" in sys.argv[0]


def plt_window_has_been_closed(fig):
    """Checks whether matplotlib plot window is closed"""
    return not plt.fignum_exists(fig.number)


def filename_from_window_dialog(window_name=None, filetypes=None, initialdir=None):
    """Get filename through dialog window
    Args:
        window_name: string with name of window
        filetypes: tuple of tuples containing (name, filetypes)
            example:
                (('mat or hdf5 or whatever you want', '*.mat *.hdf5 *'), (ckpt, *.ckpt))
        initialdir: path to directory where window will start
    Returns:
        filename: string containing path to selected file
    """
    if filetypes is None:
        filetypes = (("all files", "*.*"),)

    try:
        root = Tk()
    except Exception as error:
        raise ValueError(
            "Cannot run USBMD GUI on a server, unless a X11 server is properly setup"
        ) from error

    # open in foreground
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-topmost", False)

    # we don't want a full GUI, so keep the root window from appearing
    if not running_in_notebook():
        root.withdraw()

    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(
        parent=root,
        title=window_name,
        filetypes=filetypes,
        initialdir=initialdir,
    )
    root.destroy()

    # check whether a file was selected
    if filename:
        return Path(filename)
    else:
        raise ValueError("No file selected.")


def load_video(filename):
    """Load a video file and return a numpy array of frames.

    Supported file types: avi, mp4, gif, dcm.

    Args:
        filename (str): The path to the video file.

    Returns:
        numpy.ndarray: A numpy array of frames.
    Raises:
        ValueError: If the file extension is not supported.
    """
    filename = Path(filename)
    assert Path(filename).exists(), f"File {filename} does not exist"
    extension = filename.suffix
    assert (
        extension in _SUPPORTED_VID_TYPES
    ), f"File extension {extension} not supported"

    if extension in [".avi", ".mp4"]:
        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif extension == ".gif":
        frames = imageio.mimread(filename)
    elif extension == "":
        ds = pydicom.dcmread(filename)
        frames = convert_color_space(ds.pixel_array, "YBR_FULL", "RGB")
    else:
        raise ValueError("Unsupported file extension")
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    return np.array(frames)


def load_image(filename, grayscale=True, color_order="RGB"):
    """Load an image file and return a numpy array.

    Supported file types: jpg, png.

    Args:
        filename (str): The path to the image file.
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to True.
        color_order (str, optional): The desired color channel ordering. Defaults to 'RGB'.

    Returns:
        numpy.ndarray: A numpy array of the image.

    Raises:
        ValueError: If the file extension is not supported.
    """
    filename = Path(filename)
    assert Path(filename).exists(), f"File {filename} does not exist"
    extension = filename.suffix
    assert (
        extension in _SUPPORTED_IMG_TYPES
    ), f"File extension {extension} not supported"

    image = cv2.imread(str(filename))

    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if color_order == "BGR":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_order == "RGB":
        pass
    else:
        raise ValueError(f"Unsupported color order: {color_order}")

    return image


def _get_length_hdf5_file(filepath, key):
    """Retrieve the length of a dataset in an hdf5 file."""
    with h5py.File(filepath, "r") as f:
        return len(f[key])


def search_file_tree(
    directory,
    filetypes=None,
    write=True,
    dataset_info_filename="dataset_info.yaml",
    hdf5_key_for_length=None,
    redo=False,
    parallel=True,
):
    """Lists all files in directory and sub-directories.

    If dataset_info.yaml is detected in the directory, that file is read and used
    to deduce the file paths. If not, the file paths are searched for in the
    directory and written to a dataset_info.yaml file.

    Args:
        directory (str): path to base directory to start file search
        filetypes (str, list, optional): filetypes to look for in directory
            Defaults to image types (.png etc.). make sure to include the dot.
        write (bool, optional): Whether to write to dataset_info.yaml file.
            Defaults to True. If False, the file paths are not written to file
            and simply returned.
        dataset_info_filename (str, optional): name of dataset info file.
            Defaults to "dataset_info.yaml", but can be changed to any name.
        hdf5_key_for_length (str, optional): key to use for getting length of hdf5 files.
            Defaults to None. If set, the number of frames in each hdf5 file is
            calculated and stored in the dataset_info.yaml file. This is extra
            functionality of `search_file_tree` and only works with hdf5 files.
        redo (bool, optional): Whether to redo the search and overwrite the dataset_info.yaml file.

    Returns:
        dict: dictionary containing file paths and total number of files.
            has the following structure:
                {
                    "file_paths": list of file paths,
                    "total_num_files": total number of files
                    "file_lengths": list of number of frames in each hdf5 file
                    "total_num_frames": total number of frames in all hdf5 files
                }

    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(
            log.error(
                f"Directory {directory} does not exist. Please provide a valid directory."
            )
        )
    assert Path(dataset_info_filename).suffix == ".yaml", (
        "Currently only YAML files are supported for dataset info file when "
        f"using `search_file_tree`, got {dataset_info_filename}"
    )

    if (directory / dataset_info_filename).is_file() and not redo:
        log.info(
            "Using pregenerated dataset info file: "
            f"{log.yellow(directory / dataset_info_filename)} ..."
        )
        log.info(f"...for reading file paths in {log.yellow(directory)}")
        with open(directory / dataset_info_filename, "r", encoding="utf-8") as file:
            dataset_info = yaml.load(file, Loader=yaml.FullLoader)
        return dataset_info

    if redo:
        log.info(
            f"Overwriting dataset info file: {log.yellow(directory / dataset_info_filename)}"
        )

    # set default file type
    if filetypes is None:
        filetypes = _SUPPORTED_IMG_TYPES + _SUPPORTED_VID_TYPES + _SUPPORTED_USBMD_TYPES

    file_paths = []
    file_lengths = []

    if isinstance(filetypes, str):
        filetypes = [filetypes]

    if hdf5_key_for_length is not None:
        assert isinstance(
            hdf5_key_for_length, str
        ), "hdf5_key_for_length must be a string"
        assert set(filetypes) == {".hdf5", ".h5"}, (
            "hdf5_key_for_length only works with when filetypes is set to "
            f"`.hdf5` or `.h5`, got {filetypes}"
        )

    # Traverse file tree to index all files from filetypes
    log.info(f"Searching {log.yellow(directory)} for {filetypes} files...")
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if Path(file).suffix in filetypes:
                file_path = Path(dirpath) / file
                file_path = file_path.relative_to(directory)
                file_paths.append(str(file))

    if hdf5_key_for_length is not None:
        # using multiprocessing to speed up reading hdf5 files
        # and getting the number of frames in each file
        log.info("Getting number of frames in each hdf5 file...")

        # make sure to call search_file_tree from within a function
        # or use if __name__ == "__main__":
        # to avoid freezing the main process
        absolute_file_paths = [directory / file for file in file_paths]
        if parallel:
            with multiprocessing.Pool() as pool:
                file_lengths = list(
                    tqdm.tqdm(
                        pool.imap(
                            functools.partial(
                                _get_length_hdf5_file, key=hdf5_key_for_length
                            ),
                            absolute_file_paths,
                        ),
                        total=len(file_paths),
                        desc="Getting number of frames in each hdf5 file",
                    )
                )
        else:
            for file_path in tqdm.tqdm(
                absolute_file_paths, desc="Getting number of frames in each hdf5 file"
            ):
                file_lengths.append(
                    _get_length_hdf5_file(file_path, hdf5_key_for_length)
                )

    assert len(file_paths) > 0, f"No image files were found in: {directory}"
    log.info(f"Found {len(file_paths)} image files in {log.yellow(directory)}")
    log.info(f"Writing dataset info to {log.yellow(directory / dataset_info_filename)}")

    dataset_info = {"file_paths": file_paths, "total_num_files": len(file_paths)}
    if len(file_lengths) > 0:
        dataset_info["file_lengths"] = file_lengths
        dataset_info["total_num_frames"] = sum(file_lengths)

    if write:
        with open(directory / dataset_info_filename, "w", encoding="utf-8") as file:
            yaml.dump(dataset_info, file)

    return dataset_info


def move_matplotlib_figure(figure, position, size=None):
    """Move matplotlib figure to a specific position on the screen.
    Args:
        figure (plt.figure): matplotlib figure
        position (tuple): x and y position of figure in pixels
        size (tuple, optional): width and height of figure in pixels

    """
    x, y = position

    if size is not None:
        width, height = size
        figure.set_size_inches(width / figure.dpi, height / figure.dpi)

    backend = matplotlib.get_backend()

    if backend == "TkAgg":
        figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == "WXAgg":
        figure.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        figure.canvas.manager.window.move(x, y)


def get_matplotlib_figure_props(figure):
    """Return a dictionary of matplotlib figure properties.
    Args:
        figure (plt.figure): matplotlib figure
    Returns:
        tuple: position and size of figure in pixels
            position (tuple): x and y position of figure in pixels
            size (tuple): width and height of figure in pixels
    """
    # Get size and position in the format of "widthxheight+X+Y"
    geometry = figure.canvas.manager.window.geometry()

    # Get the geometry object based on the backend
    backend_name = matplotlib.get_backend()

    try:
        if backend_name == "TkAgg":
            assert isinstance(
                geometry, str
            ), f"Unsupported geometry type: {type(geometry)} for backend: {backend_name}"
            # format: "widthxheight+X+Y"
            # Split the geometry string by '+' to extract size and position
            size_str, *pos_str = geometry.split("+")
            # Extract width and height from the size string
            size = map(int, size_str.split("x"))
            # Extract X and Y position values as integers
            position = map(int, pos_str)
        elif backend_name == "QtAgg":
            assert isinstance(
                geometry, QRect
            ), f"Unsupported geometry type: {type(geometry)} for backend: {backend_name}"
            # format: QRect object
            position = geometry.x(), geometry.y()
            size = geometry.size().width(), geometry.size().height()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    except Exception as error:
        log.warning(f"Could not get figure properties: {error}")
        position, size = None, None

    return position, size


def raise_matplotlib_window(figname=None):
    """Raise the matplotlib window for Figure figname to the foreground.

    If no argument is given, raise the current figure.

    This function will only work with a Qt or Tk graphics backend.
    """
    # check for backend
    backend = matplotlib.get_backend()
    if backend in ["Qt5Agg", "TkAgg"]:
        log.warning("This function only works with a Qt or Tk graphics backend")

    if figname:
        plt.figure(figname)

    cfm = plt.get_current_fig_manager()

    if backend in ["QtAgg", "Qt4Agg", "Qt5Agg"]:
        cfm.window.activateWindow()
        cfm.window.raise_()
    elif backend == "TkAgg":
        cfm.window.attributes("-topmost", True)
        cfm.window.attributes("-topmost", False)


def matplotlib_figure_to_numpy(fig):
    """Convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.figure.Figure): figure to convert.

    Returns:
        np.ndarray: numpy array of figure.

    """
    try:
        if matplotlib.get_backend() == "Qt5Agg":
            canvas = FigureCanvasQTAgg(fig)
        elif matplotlib.get_backend() == "TkAgg":
            canvas = FigureCanvasTkAgg(fig)
        elif matplotlib.get_backend() == "agg":
            canvas = FigureCanvasAgg(fig)
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image = np.array(image)[..., :3]
            buf.close()
            return image

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            canvas.draw()

        if matplotlib.get_backend() == "Qt5Agg":
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        else:
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)

        width, height = fig.canvas.get_width_height()
        image = image.reshape((height, width, 3))
        return image
    except:
        log.warning("Could not convert figure to numpy array.")
        return np.array([])


class DummyTask:
    """Dummy task that returns the data passed to it.

    Used in ImageViewer to pass data to the ImageViewer without using threads.
    """

    def __init__(self, data):
        self.data = data

    def ready(self):
        """Always returns True."""
        return True

    def get(self):
        """Gets the data passed to the DummyTask."""
        return self.data


class ImageViewer(abc.ABC):
    """ImageViewer is an abstract class for displaying frames in a non-blocking way."""

    def __init__(
        self,
        get_frame: Callable[[], np.ndarray],
        window_name: Optional[str] = "frame",
        num_threads: Optional[int] = None,
        resizable_window: Optional[bool] = True,
        threading: Optional[bool] = True,
        collect_frames: Optional[bool] = False,
    ) -> None:
        """Initializes the ImageViewer object.
        Args:
            get_frame (function): A function that returns the current frame.
            window_name (str): The name of the window to display the frame.
            num_threads (int, optional): The number of threads to use for processing frames.
            resizable_window (bool, optional): Whether the window should be resizable or not.
            threading (bool, optional): Whether to use threading or not.
            collect_frames (bool, optional): Whether to collect frames or not.
        """
        self.get_frame_func = get_frame
        self.window_name = window_name
        self.num_threads = num_threads
        self.resizable_window = resizable_window
        self.threading = threading
        self.collect_frames = collect_frames

        if self.num_threads is None:
            self.num_threads = cv2.getNumberOfCPUs()

        if self.threading:
            self.pool = ThreadPool(processes=num_threads)

        self.pending = deque()
        self.frame_no = 0
        self.frames = []

    @abc.abstractmethod
    def show(self, *args, **kwargs) -> None:
        """Displays a frame.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        """
        self._add_task(*args, **kwargs)
        # show the frame

    def _add_task(self, *args, **kwargs):
        if self.threading:
            if len(self.pending) < self.num_threads:
                task = self.pool.apply_async(
                    self.get_frame_func, args=args, kwds=kwargs
                )
                self.pending.append(task)
        else:
            task = DummyTask(self.get_frame_func(*args, **kwargs))
            self.pending.append(task)

    def _get_frame(self):
        """Returns the most recent frame in the queue."""
        if len(self.pending) == 0:
            raise ValueError("No frames available.")
        if not self.pending[0].ready():
            raise ValueError("Frame not ready.")
        frame = self.pending.popleft().get()
        if self.collect_frames:
            self.frames.append(frame)
        return frame

    @property
    def frame_is_ready(self):
        """Returns True if a frame is ready to be displayed."""
        return len(self.pending) > 0 and self.pending[0].ready()


class ImageViewerOpenCV(ImageViewer):
    """ImageViewer displays frames using OpenCV's imshow function in a non-blocking way.

    Example:
        >>> import numpy as np
        >>> from usbmd.utils.io_lib import ImageViewerOpenCV
        >>> def generate_frame():
        >>>     return np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        >>> image_viewer = ImageViewerOpenCV(generate_frame, threading=True, num_threads=1)
        >>> while True:
        >>>     image_viewer.show()
        >>>     if cv2.waitKey(25) & 0xFF == ord("q"):
        >>>         break
    """

    def __init__(
        self,
        get_frame: Callable[[], np.ndarray],
        window_name: Optional[str] = "frame",
        num_threads: Optional[int] = None,
        resizable_window: Optional[bool] = True,
        threading: Optional[bool] = True,
        headless: Optional[bool] = False,
    ) -> None:
        """Initializes the ImageViewerOpenCV object."""
        super().__init__(
            get_frame, window_name, num_threads, resizable_window, threading
        )
        self.window = None
        self.headless = headless

    def _create_window(self):
        if self.resizable_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(self.window_name)
        self.window = True

    def show(self, *args, **kwargs) -> None:
        """Displays a frame using OpenCV's imshow function.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        Imshow is called asynchronously in a separate thread.
        """
        super().show(*args, **kwargs)
        while self.frame_is_ready:
            frame = self._get_frame()
            frame = np.array(frame, dtype=np.uint8)
            if self.headless:
                self.frame_no += 1
                continue

            if self.frame_no == 0:
                if self.window is None:
                    self._create_window()
                    cv2.resizeWindow(self.window_name, frame.shape[1], frame.shape[0])

            cv2.imshow(self.window_name, frame)
            self.frame_no += 1

    def close(self):
        """Closes the window."""
        if not self.window:
            return
        if self.has_been_closed():
            return
        cv2.destroyWindow(self.window_name)

    def has_been_closed(self):
        """Returns True if the window has been closed."""
        if self.window is None:
            return False
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1


class ImageViewerMatplotlib(ImageViewer):
    """ImageViewerMatplotlib displays frames using matplotlib's imshow function
    in a non-blocking way.

    Example:
        >>> import numpy as np
        >>> from usbmd.utils.io_lib import ImageViewerMatplotlib, plt_window_has_been_closed
        >>> def generate_frame():
        >>>     return np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        >>> image_viewer = ImageViewerMatplotlib(generate_frame, threading=True, num_threads=1)
        >>> while True:
        >>>     image_show.show()
        >>>     if cv2.waitKey(25) & plt_window_has_been_closed(image_viewer.fig):
        >>>         break
    """

    def __init__(
        self,
        get_frame: Callable[[], np.ndarray],
        window_name: Optional[str] = "frame",
        num_threads: Optional[int] = None,
        resizable_window: Optional[bool] = True,
        threading: Optional[bool] = True,
        imshow_kwargs: Optional[dict] = None,
        cax_kwargs: Optional[dict] = None,
    ) -> None:
        """Initializes the ImageViewerMatplotlib object."""
        super().__init__(
            get_frame, window_name, num_threads, resizable_window, threading
        )
        plt.ion()
        self.fig = None
        self.ax = None
        self.bg = None
        self.image_obj = None
        self.imshow_kwargs = imshow_kwargs
        self.cax_kwargs = cax_kwargs

        self.init_figure_props = True

    def _create_figure(self):
        # create figure but do not show it already
        self.fig = plt.figure(self.window_name)
        self.ax = self.fig.add_subplot(111)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # move to foreground
        raise_matplotlib_window(self.window_name)

        if not self.resizable_window:
            backend = matplotlib.get_backend()
            if backend == "Qt4Agg":
                win = self.fig.canvas.window()
                win.setFixedSize(win.size())
            else:
                log.warning(f"Backend {backend} does not support fixed size windows.")

    def show(self, *args, **kwargs) -> None:
        """Displays a frame using matplotlib's imshow function.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        """
        super().show(*args, **kwargs)

        while self.frame_is_ready:
            frame = self._get_frame()

            if self.image_obj is None:
                if self.fig is None:
                    self._create_figure()
                if self.bg is None:
                    self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

                if self.imshow_kwargs:
                    self.image_obj = self.ax.imshow(
                        frame, **self.imshow_kwargs, animated=True
                    )
                else:
                    self.image_obj = self.ax.imshow(frame, animated=True)

                # these only need to be set once
                if self.init_figure_props:
                    if self.cax_kwargs:
                        divider = make_axes_locatable(self.ax)
                        if "color" in self.cax_kwargs:
                            color = self.cax_kwargs.pop("color")
                            cax.yaxis.label.set_color(color)
                            cax.tick_params(axis="y", colors=color)
                            cax.title.set_color(color)
                        cax = divider.append_axes(**self.cax_kwargs)
                        plt.colorbar(self.image_obj, cax=cax)
                    self.fig.tight_layout()
                    self.init_figure_props = False

                self.fig.canvas.draw()
            else:
                self.image_obj.set_data(frame)
                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.draw()

            self.fig.canvas.flush_events()
            self.frame_no += 1

    def close(self):
        """Closes the window."""
        plt.close(self.fig)

    def has_been_closed(self):
        """Returns True if the window has been closed."""
        return plt_window_has_been_closed(self.fig)


async def start_async_app(app: Tk, *args, **kwargs):
    """
    Starts the asynchronous app.

    Args:
        app (Tk): The Tkinter app object.

    Raises:
        AssertionError: If the app is not an instance of Tk or does not have the "show" attribute.

    Returns:
        MyWindow: The instance of MyWindow.
    """
    my_app = app(asyncio.get_event_loop(), *args, **kwargs)
    await my_app.show()
