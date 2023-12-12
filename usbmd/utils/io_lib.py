"""Input/output functions

Use to quickly read and write files or interact with file system.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 12th, 2023
"""
import abc
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
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from pydicom.pixel_data_handlers import convert_color_space
from PyQt5.QtCore import QRect

_SUPPORTED_VID_TYPES = [".avi", ".mp4", ".gif", ""]
_SUPPORTED_IMG_TYPES = [".jpg", ".png", ".JPEG", ".PNG", ".jpeg"]


def running_in_notebook():
    """Check whether code is running in a Jupyter Notebook or not."""
    return "ipykernel_launcher" in sys.argv[0]


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


def load_image(filename, grayscale=True):
    """Load an image file and return a numpy array.

    Supported file types: jpg, png.

    Args:
        filename (str): The path to the image file.

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
    return image


def search_file_tree(directory, filetypes=None, write=True):
    """Lists all files in directory and sub-directories.

    If file_paths.txt is detected in the directory, that file is read and used.

    Args:
        directory (str): path to directory.
        filetypes (Tuple of strings, optional): filetypes.
            Defaults to image types (.png etc.).
        write (bool, optional): Whether to write to file. Useful has searching
            the tree takes quite a while. Defaults to True.

    Returns:
        file_paths (List): List with str to all file paths.

    """
    directory = Path(directory)
    if (directory / "file_paths.txt").is_file():
        print(
            "Using pregenerated txt file in the following directory for reading file paths: "
        )
        print(directory)
        with open(directory / "file_paths.txt", encoding="utf-8") as file:
            file_paths = [line.strip() for line in file]
        return file_paths

    # set default file type
    if filetypes is None:
        filetypes = _SUPPORTED_IMG_TYPES

    file_paths = []

    # Traverse file tree to index all files from filetypes
    for dirpath, _, filenames in tqdm.tqdm(
        os.walk(directory), desc="Searching file tree"
    ):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if Path(file).suffix in filetypes:
                file_paths.append(str(Path(dirpath) / Path(file)))

    assert len(file_paths) > 0, f"No image files were found in: {directory}"
    print(f"\nFound {len(file_paths)} image files in .\\{Path(directory)}\n")

    if write:
        with open(directory / "file_paths.txt", "w", encoding="utf-8") as file:
            _file_paths = [file + "\n" for file in file_paths]
            file.writelines(_file_paths)

    return file_paths


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
        warnings.warn(f"Could not get figure properties: {error}")
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
        warnings.warn("This function only works with a Qt or Tk graphics backend")

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

        canvas.draw()

        if matplotlib.get_backend() == "Qt5Agg":
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        else:
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)

        width, height = fig.canvas.get_width_height()
        image = image.reshape((height, width, 3))
        return image
    except:
        warnings.warn("Could not convert figure to numpy array.")
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
    def show(self) -> None:
        """Displays a frame.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        """
        self._add_task()
        # show the frame

    def _add_task(self):
        if self.threading:
            if len(self.pending) < self.num_threads:
                task = self.pool.apply_async(self.get_frame_func, ())
                self.pending.append(task)
        else:
            task = DummyTask(self.get_frame_func())
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
    ) -> None:
        """Initializes the ImageViewerOpenCV object."""
        super().__init__(
            get_frame, window_name, num_threads, resizable_window, threading
        )
        self.window = None

    def _create_window(self):
        if self.resizable_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(self.window_name)
        self.window = True

    def show(self) -> None:
        """Displays a frame using OpenCV's imshow function.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        Imshow is called asynchronously in a separate thread.
        """
        super().show()
        while self.frame_is_ready:
            frame = self._get_frame()
            frame = np.array(frame, dtype=np.uint8)
            if self.frame_no == 0:
                if self.window is None:
                    self._create_window()
                    cv2.resizeWindow(self.window_name, frame.shape[1], frame.shape[0])
            cv2.imshow(self.window_name, frame)
            self.frame_no += 1


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
        self.image_obj = None
        self.imshow_kwargs = imshow_kwargs
        self.cax_kwargs = cax_kwargs

        self.init_figure_props = True

    def _create_figure(self):
        # create figure but do not show it already
        self.fig = plt.figure(self.window_name)
        self.ax = self.fig.add_subplot(111)

        # move to foreground
        raise_matplotlib_window(self.window_name)

        if not self.resizable_window:
            backend = matplotlib.get_backend()
            if backend == "Qt4Agg":
                win = self.fig.canvas.window()
                win.setFixedSize(win.size())
            else:
                warnings.warn(f"Backend {backend} does not support fixed size windows.")

    def show(self) -> None:
        """Displays a frame using matplotlib's imshow function.
        Frame is generated using the get_frame function passed during initialization.

        This function is non-blocking, and will return immediately.
        """
        super().show()

        while self.frame_is_ready:
            frame = self._get_frame()

            if self.image_obj is None:
                if self.fig is None:
                    self._create_figure()

                if self.imshow_kwargs:
                    self.image_obj = self.ax.imshow(frame, **self.imshow_kwargs)
                else:
                    self.image_obj = self.ax.imshow(frame)

                # these only need to be set once
                if self.init_figure_props:
                    if self.cax_kwargs is not None:
                        divider = make_axes_locatable(self.ax)
                        if "color" in self.cax_kwargs:
                            color = self.cax_kwargs.pop("color")
                        cax = divider.append_axes(**self.cax_kwargs)
                        cax.yaxis.label.set_color(color)
                        cax.tick_params(axis="y", colors=color)
                        cax.title.set_color(color)
                        plt.colorbar(self.image_obj, cax=cax)
                    self.fig.tight_layout()
                    self.init_figure_props = False

                self.fig.canvas.draw_idle()
            else:
                self.image_obj.set_data(frame)
                self.fig.canvas.draw_idle()

            self.image_obj.figure.canvas.flush_events()
            self.frame_no += 1
