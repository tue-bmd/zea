"""Helpers for file-open dialogs and matplotlib figure window management."""

import sys
from pathlib import Path

import matplotlib

from zea import log


def running_in_notebook():
    """Check whether code is running in a Jupyter Notebook or not."""
    return "ipykernel_launcher" in sys.argv[0]


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
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
    except ImportError as error:
        raise ImportError(
            "The file dialog window features in zea require Python's native Tkinter GUI toolkit. "
            "Tkinter was not found in your current Python environment. Please ensure that "
            "your Python distribution includes Tkinter or install your system's corresponding "
            "python-tk package."
        ) from error

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
