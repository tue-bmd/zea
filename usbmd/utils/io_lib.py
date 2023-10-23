"""Input/output functions

Use to quickly read and write files or interact with file system.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 12th, 2023
"""
import os
import warnings
from io import BytesIO
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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
from PIL import Image
from pydicom.pixel_data_handlers import convert_color_space
from PyQt5.QtCore import QRect

_SUPPORTED_VID_TYPES = ['.avi', '.mp4', '.gif', '']
_SUPPORTED_IMG_TYPES = ['.jpg', '.png', '.JPEG', '.PNG', '.jpeg']

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
    if "inline" in matplotlib.get_backend():
        root.wm_attributes("-topmost", False)
    # we don't want a full GUI, so keep the root window from appearing
    root.withdraw()
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(
        parent=root,
        title=window_name,
        filetypes=filetypes,
        initialdir=initialdir,
    )
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
    assert Path(filename).exists(), f'File {filename} does not exist'
    extension = filename.suffix
    assert extension in _SUPPORTED_VID_TYPES, f'File extension {extension} not supported'

    if extension in ['.avi', '.mp4']:
        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif extension == '.gif':
        frames = imageio.mimread(filename)
    elif extension == '':
        ds = pydicom.dcmread(filename)
        frames = convert_color_space(ds.pixel_array, "YBR_FULL", "RGB")
    else:
        raise ValueError('Unsupported file extension')
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
    assert Path(filename).exists(), f'File {filename} does not exist'
    extension = filename.suffix
    assert extension in _SUPPORTED_IMG_TYPES, f'File extension {extension} not supported'

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
            file_paths = file.read().splitlines()
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

    print(f"\nFound {len(file_paths)} image files in .\\{Path(directory)}\n")
    assert len(file_paths) > 0, "ERROR: No image files were found"

    if write:
        with open(directory / "file_paths.txt", "w", encoding="utf-8") as file:
            file_paths = [file + "\n" for file in file_paths]
            file.writelines(file_paths)

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

    if backend == 'TkAgg':
        figure.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == 'WXAgg':
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
        if backend_name == 'TkAgg':
            assert isinstance(geometry, str), \
                f"Unsupported geometry type: {type(geometry)} for backend: {backend_name}"
            # format: "widthxheight+X+Y"
            # Split the geometry string by '+' to extract size and position
            size_str, *pos_str = geometry.split('+')
            # Extract width and height from the size string
            size = map(int, size_str.split('x'))
            # Extract X and Y position values as integers
            position = map(int, pos_str)
        elif backend_name == 'QtAgg':
            assert isinstance(geometry, QRect), \
                f"Unsupported geometry type: {type(geometry)} for backend: {backend_name}"
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
    if backend in ['Qt5Agg', 'TkAgg']:
        warnings.warn("This function only works with a Qt or Tk graphics backend")

    if figname:
        plt.figure(figname)

    cfm = plt.get_current_fig_manager()

    if backend == 'Qt5Agg':
        cfm.window.activateWindow()
        cfm.window.raise_()
    elif backend == 'TkAgg':
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)

def matplotlib_figure_to_numpy(fig):
    """Convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.figure.Figure): figure to convert.

    Returns:
        np.ndarray: numpy array of figure.

    """
    try:
        if matplotlib.get_backend() == 'Qt5Agg':
            canvas = FigureCanvasQTAgg(fig)
        elif matplotlib.get_backend() == 'TkAgg':
            canvas = FigureCanvasTkAgg(fig)
        elif matplotlib.get_backend() == 'agg':
            canvas = FigureCanvasAgg(fig)
        else:
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image = np.array(image)[..., :3]
            buf.close()
            return image

        canvas.draw()

        if matplotlib.get_backend() == 'Qt5Agg':
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        else:
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)

        width, height = fig.canvas.get_width_height()
        image = image.reshape((height, width, 3))
        return image
    except:
        warnings.warn("Could not convert figure to numpy array.")
        return np.array([])
