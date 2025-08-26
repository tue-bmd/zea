"""Functions for displaying images and videos using OpenCV or Matplotlib."""

import abc
import os
import sys
from collections import deque
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from zea import log


def running_in_notebook():
    """Check whether code is running in a Jupyter Notebook or not."""
    return "ipykernel_launcher" in sys.argv[0]


def plt_window_has_been_closed(fig):
    """Checks whether matplotlib plot window is closed"""
    return not plt.fignum_exists(fig.number)


def filename_from_window_dialog(window_name=None, filetypes=None, initialdir=None) -> Path:
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
            "Cannot run zea GUI on a server, unless a X11 server is properly setup"
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
    position, size = None, None
    try:
        manager = figure.canvas.manager
        window = getattr(manager, "window", None)
        if window is not None:
            # Try geometry() method (TkAgg, Qt)
            geom = getattr(window, "geometry", None)
            if callable(geom):
                g = geom()
                if isinstance(g, str):
                    # TkAgg: "widthxheight+X+Y"
                    size_str, *pos_str = g.split("+")
                    width, height = map(int, size_str.split("x"))
                    x, y = map(int, pos_str)
                    position, size = (x, y), (width, height)
                elif hasattr(g, "x") and hasattr(g, "y"):
                    # Qt: QRect
                    position, size = (g.x(), g.y()), (g.width(), g.height())
            # Try frameGeometry() method (MacOS, Qt)
            elif hasattr(window, "frameGeometry"):
                fg = window.frameGeometry()
                position, size = (fg.x(), fg.y()), (fg.width(), fg.height())
            # WXAgg
            elif hasattr(window, "GetPosition") and hasattr(window, "GetSize"):
                position, size = window.GetPosition(), window.GetSize()
    except Exception as error:
        log.warning(f"Could not get figure properties: {error}")

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
            # Default to the number of CPU cores if not specified
            self.num_threads = max(1, os.cpu_count() - 1)

        if self.threading:
            self.pool = ThreadPool(processes=self.num_threads)

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
                task = self.pool.apply_async(self.get_frame_func, args=args, kwds=kwargs)
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
        >>> from zea.internal.io_lib import ImageViewerOpenCV
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
        super().__init__(get_frame, window_name, num_threads, resizable_window, threading)
        self.window = None
        self.headless = headless

        try:
            import cv2

            self._cv2 = cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for ImageViewerOpenCV to work. "
                "Please install it with 'pip install opencv-python' or "
                "'pip install opencv-python-headless'."
            ) from exc

    def _create_window(self):
        if self.resizable_window:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
        else:
            self._cv2.namedWindow(self.window_name)
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
                    self._cv2.resizeWindow(self.window_name, frame.shape[1], frame.shape[0])

            self._cv2.imshow(self.window_name, frame)
            self.frame_no += 1

    def close(self):
        """Closes the window."""
        if not self.window:
            return
        if self.has_been_closed():
            return
        self._cv2.destroyWindow(self.window_name)

    def has_been_closed(self):
        """Returns True if the window has been closed."""
        if self.window is None:
            return False
        return self._cv2.getWindowProperty(self.window_name, self._cv2.WND_PROP_VISIBLE) < 1


class ImageViewerMatplotlib(ImageViewer):
    """ImageViewerMatplotlib displays frames using matplotlib's imshow function
    in a non-blocking way.

    Example:
        >>> import numpy as np
        >>> from zea.internal.io_lib import ImageViewerMatplotlib, plt_window_has_been_closed
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
        super().__init__(get_frame, window_name, num_threads, resizable_window, threading)
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
                    self.image_obj = self.ax.imshow(frame, **self.imshow_kwargs, animated=True)
                else:
                    self.image_obj = self.ax.imshow(frame, animated=True)

                # these only need to be set once
                if self.init_figure_props:
                    if self.cax_kwargs:
                        divider = make_axes_locatable(self.ax)
                        cax = divider.append_axes(**self.cax_kwargs)
                        if "color" in self.cax_kwargs:
                            color = self.cax_kwargs.pop("color")
                            cax.yaxis.label.set_color(color)
                            cax.tick_params(axis="y", colors=color)
                            cax.title.set_color(color)
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
