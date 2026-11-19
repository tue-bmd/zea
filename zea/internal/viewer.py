"""Helpers for matplotlib figure window management."""

import matplotlib

from zea import log


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
