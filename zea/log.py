"""Custom ``zea`` python logging module.

Wrapper around python logging module to provide a simple interface for logging both
to the console and to a file with color support.

Example usage
^^^^^^^^^^^^^^

.. testsetup::

    from zea import log

    log.info("This is an info message")
    path = "data/datafile.hdf5"
    log.info(f"Saved to {log.yellow(path)}")

"""

import contextlib
import contextvars
import inspect
import logging
import os
import re
import sys
import weakref
from pathlib import Path

from tqdm import tqdm as _tqdm_cls

# The logger to use
logger: logging.Logger
file_logger: logging.Logger | None = None

LOG_DIR = Path("log")

ZEA_LOG_LEVEL = os.getenv("ZEA_LOG_LEVEL", "DEBUG").upper()

DEPRECATED_LEVEL_NUM = logging.WARNING + 5
logging.addLevelName(DEPRECATED_LEVEL_NUM, "DEPRECATED")
logging.DEPRECATED = DEPRECATED_LEVEL_NUM  # ty: ignore[unresolved-attribute]


def get_format_fn(name_format):
    """Returns the format function for the given format name."""
    return {
        # Different consoles render these codes at different values
        "red": red,
        "green": green,
        "yellow": yellow,
        "blue": blue,
        "magenta": magenta,
        "cyan": cyan,
        "white": white,
        # Custom colors
        "purple": purple,
        "darkgreen": darkgreen,
        "orange": orange,
        # Formatting
        "bold": bold,
    }.get(name_format)


def red(string):
    """Adds ANSI escape codes to print a string in red around the string."""
    return "\033[31m" + str(string) + "\033[0m"


def green(string):
    """Adds ANSI escape codes to print a string in green around the string."""
    return "\033[32m" + str(string) + "\033[0m"


def yellow(string):
    """Adds ANSI escape codes to print a string in yellow around the string."""
    return "\033[33m" + str(string) + "\033[0m"


def blue(string):
    """Adds ANSI escape codes to print a string in blue around the string."""
    return "\033[34m" + str(string) + "\033[0m"


def magenta(string):
    """Adds ANSI escape codes to print a string in magenta around the string."""
    return "\033[35m" + str(string) + "\033[0m"


def cyan(string):
    """Adds ANSI escape codes to print a string in cyan around the string."""
    return "\033[36m" + str(string) + "\033[0m"


def white(string):
    """Adds ANSI escape codes to print a string in white around the string."""
    return "\033[37m" + str(string) + "\033[0m"


def purple(string):
    """Adds ANSI escape codes to print a string in purple around the string."""
    return "\033[38;5;93m" + str(string) + "\033[0m"


def darkgreen(string):
    """Adds ANSI escape codes to print a string in blue around the string."""
    return "\033[38;5;36m" + str(string) + "\033[0m"


def orange(string):
    """Adds ANSI escape codes to print a string in orange around the string."""
    return "\033[38;5;214m" + str(string) + "\033[0m"


def bold(string):
    """Adds ANSI escape codes to print a string in bold around the string."""
    return "\033[1m" + str(string) + "\033[0m"


# Progress bars (other than tqdm, which is handled separately below) that have
# asked to be redrawn whenever a log message is emitted while they are on screen.
# A WeakSet so a bar that never explicitly unregisters (e.g. a loop that
# `break`s before reaching its target) doesn't leak here forever - it drops out
# once nothing else references it.
# See `register_progress`/`unregister_progress` and `zea.utils.ProgressBar`.
_active_progress: "weakref.WeakSet" = weakref.WeakSet()


def register_progress(bar):
    """Registers a progress-bar-like object so log output doesn't corrupt its line.

    ``bar`` must implement a no-argument ``redraw()`` method that forces a fresh
    render of its current state. Registered bars are redrawn whenever a log
    message is emitted to the console while they are active.
    """
    _active_progress.add(bar)


def unregister_progress(bar):
    """Unregisters a progress bar previously passed to :func:`register_progress`."""
    _active_progress.discard(bar)


class _ProgressAwareStreamHandler(logging.StreamHandler):
    """A :class:`logging.StreamHandler` that doesn't corrupt an in-progress bar's line.

    - If any :mod:`tqdm` bars are currently active, the message is routed through
      ``tqdm.write()``, which knows how to clear and redraw every active bar.
    - Otherwise, if any bar registered via :func:`register_progress` is active
      (e.g. a :class:`zea.utils.ProgressBar`), the current unterminated line is
      cleared before the message is written, and each registered bar is asked
      to redraw itself.
    - If neither applies, this behaves like a plain ``StreamHandler``.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if _tqdm_cls._instances:
                _tqdm_cls.write(msg, file=stream)
                return
            if _active_progress:
                stream.write("\r\x1b[K")
            stream.write(msg + self.terminator)
            self.flush()
            for bar in list(_active_progress):
                bar.redraw()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class CustomFormatter(logging.Formatter):
    """Custom formatter to use different format strings for different log levels"""

    def __init__(self, name=None, color=True, name_color="darkgreen"):
        super().__init__()

        if name is None:
            name = ""
        else:
            if color:
                color_fn_name = get_format_fn(name_color)
                name = f"{bold(color_fn_name(name))}: "
            else:
                name = f"{name}: "

        orange_fn = orange if color else lambda x: x
        red_fn = red if color else lambda x: x
        yellow_fn = yellow if color else lambda x: x

        self.FORMATS = {
            logging.INFO: logging.Formatter(("".join([name, "%(message)s"]))),
            logging.WARNING: logging.Formatter(
                ("".join([name, orange_fn("%(levelname)s"), " %(message)s"]))
            ),
            logging.ERROR: logging.Formatter(
                ("".join([name, red_fn("%(levelname)s"), " %(message)s"]))
            ),
            logging.DEBUG: logging.Formatter(
                ("".join([name, yellow_fn("%(levelname)s"), " %(message)s"]))
            ),
            DEPRECATED_LEVEL_NUM: logging.Formatter(
                ("".join([name, orange_fn("%(levelname)s"), " %(message)s"]))
            ),
            "DEFAULT": logging.Formatter(
                ("".join([name, yellow_fn("%(levelname)s"), " %(message)s"]))
            ),
        }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"])
        return formatter.format(record)


def configure_console_logger(
    level="INFO", name=None, color=True, name_color="darkgreen"
) -> logging.Logger:
    """
    Configures a simple console logger with the givel level.
    A usecase is to change the formatting of the default handler of the root logger
    """
    assert level in [
        "DEBUG",
        "INFO",
        "DEPRECATED",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ], f"Invalid log level: {level}"

    # Create a logger
    new_logger = logging.getLogger("my_logger")
    new_logger.setLevel(level)

    formatter = CustomFormatter(name, color, name_color)

    # stdout stream handler if this logger doesn't already have one of its own
    if not new_logger.handlers:
        console = _ProgressAwareStreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        new_logger.addHandler(console)

    return new_logger


def configure_file_logger(level="INFO") -> logging.Logger:
    """
    Configures a simple console logger with the givel level.
    A usecase is to change the formatting of the default handler of the root logger
    """
    assert level in [
        "DEBUG",
        "INFO",
        "DEPRECATED",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ], f"Invalid log level: {level}"

    # Create a logger
    new_logger = logging.getLogger("file_logger")
    new_logger.setLevel("DEBUG")

    file_log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Set the date format
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(file_log_format, date_format)

    # File handler if this logger doesn't already have one of its own
    if not new_logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(Path(LOG_DIR, "log.log"), mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel("DEBUG")
        new_logger.addHandler(file_handler)

    return new_logger


def remove_color_escape_codes(text):
    """
    Removes ANSI color escape codes from the given string.
    """

    # ANSI escape code pattern (e.g., \x1b[31m for red)
    escape_code_pattern = re.compile(r"\x1b\[[0-9;]*m")

    return escape_code_pattern.sub("", text)


# Absolute path (with trailing separator) of the zea package directory, used to tell
# zea-internal call frames apart from external ("user") ones in `_caller_frame`.
_LOG_FILE = os.path.abspath(__file__)
_PACKAGE_DIR = os.path.dirname(_LOG_FILE) + os.sep


def _caller_frame(skip_package=True):
    """Finds the stack frame most relevant to attribute a log message to.

    Always skips frames inside this module (the ``log.*`` wrapper machinery
    itself, e.g. :func:`warning` or :func:`_log`). When ``skip_package`` is
    True (the default), also skips past any further frames inside the ``zea``
    package, returning the first external ("user") frame - usually the call
    site people actually want to see, even if the message was actually
    emitted deep inside some internal zea helper. If the whole stack is
    inside zea (e.g. triggered from zea's own tests/examples), falls back to
    the immediate caller instead of raising.

    Args:
        skip_package: If False, only skip this module's own frames and return
            the literal call site of the ``log.*`` call, even if that's
            inside zea itself.
    """
    stack = inspect.stack()[1:]
    frames = [f for f in stack if f.filename != _LOG_FILE]
    if skip_package:
        for frame in frames:
            if not frame.filename.startswith(_PACKAGE_DIR):
                return frame
    if frames:
        return frames[0]
    # Whole stack is inside this module: fall back to the immediate caller
    # rather than raising, as documented above.
    return stack[0]


# Track locations that have already emitted a once-only warning
_warned_locations: set = set()

# Call-scoped flag to suppress warnings. Implemented with a ContextVar so the
# suppression is local to the current thread / async task: setting it does not
# mutate the shared logger level, so concurrent callers are unaffected.
_warnings_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "zea_warnings_suppressed", default=False
)


@contextlib.contextmanager
def suppress_warnings():
    """Context manager to suppress ``warning``/``warning_once``/``deprecated`` output.

    Unlike :func:`set_level`, this does not mutate the shared logger level, so it is
    safe to use from one thread without suppressing warnings emitted by others.

    Yields:
        None
    """
    token = _warnings_suppressed.set(True)
    try:
        yield
    finally:
        _warnings_suppressed.reset(token)


def _log(level, message, *args, suppressible=False, location=False, raw_location=False, **kwargs):
    """Core implementation shared by all ``log.<level>`` functions.

    Args:
        level: The numeric logging level (e.g. ``logging.INFO``) to log at.
        message: The message to log.
        suppressible: If True, this call is skipped while
            :func:`suppress_warnings` is active.
        location: If True, prefixes the message with the ``file:line`` of the
            call site, similar to Python's :func:`warnings.warn`. By default
            this is the first call frame *outside* zea, so a message emitted
            deep inside an internal helper still points at the user's code.
        raw_location: If True (and ``location`` is True), show the literal
            call site of the ``log.*`` call instead - useful when the message
            is actually about something zea-internal.

    Returns:
        The (possibly location-prefixed) message that was logged.
    """
    if suppressible and _warnings_suppressed.get():
        return message
    if location:
        frame = _caller_frame(skip_package=not raw_location)
        message = f"{frame.filename}:{frame.lineno}: {message}"
    logger.log(level, message, *args, **kwargs)
    if file_logger:
        file_logger.log(level, remove_color_escape_codes(message), *args, **kwargs)
    return message


def warning(message, *args, **kwargs):
    """Prints a message with log level warning.

    Also accepts ``location``/``raw_location`` to prefix the message with its
    call site - see :func:`_log` for details.
    """
    return _log(logging.WARNING, message, *args, suppressible=True, **kwargs)


def warning_once(message, *args, key=None, **kwargs):
    """Prints a warning message only once for a dedupe key.

    By default, deduplication is per call location. A custom ``key`` can be
    provided to scope once-only behavior (for example, per object instance).

    Also accepts the same ``location``/``raw_location`` arguments as
    :func:`warning` (forwarded via ``**kwargs``).

    Args:
        message: The message to log.
        key: Optional dedupe key scoping the once-only behavior.
    """
    if _warnings_suppressed.get():
        return message
    frame = inspect.stack()[1]
    location_key = f"{frame.filename}:{frame.lineno}"
    dedupe_key = location_key if key is None else (location_key, key)
    if dedupe_key not in _warned_locations:
        _warned_locations.add(dedupe_key)
        warning(message, *args, **kwargs)
    return message


def deprecated(message, *args, **kwargs):
    """Prints a message with custom log level DEPRECATED."""
    return _log(DEPRECATED_LEVEL_NUM, message, *args, suppressible=True, **kwargs)


def error(message, *args, **kwargs):
    """Prints a message with log level error."""
    return _log(logging.ERROR, message, *args, **kwargs)


def debug(message, *args, **kwargs):
    """Prints a message with log level debug."""
    return _log(logging.DEBUG, message, *args, **kwargs)


def info(message, *args, **kwargs):
    """Prints a message with log level info."""
    return _log(logging.INFO, message, *args, **kwargs)


def critical(message, *args, **kwargs):
    """Prints a message with log level critical."""
    return _log(logging.CRITICAL, message, *args, **kwargs)


def success(message, *args, **kwargs):
    """Prints a message to the console in green."""
    _log(logging.INFO, green(message), *args, **kwargs)
    return message


def number_to_str(number, decimals=2):
    """Formats a number to a string with the given number of decimals."""
    if isinstance(number, (int, float)):
        return f"{number:.{decimals}f}"
    else:
        raise ValueError(f"Expected a number, got {type(number)}: {number}")


def set_file_logger_directory(directory):
    """Sets the log level of the logger."""
    global LOG_DIR, file_logger
    LOG_DIR = directory
    # Remove all handlers from the file logger
    if file_logger is None:
        raise RuntimeError("File logging not enabled; call enable_file_logging() first.")
    for handler in list(file_logger.handlers):
        file_logger.removeHandler(handler)

    # Add file handler
    file_logger = configure_file_logger(level="DEBUG")


def enable_file_logging():
    """Enables file logging"""
    global file_logger
    if not file_logger:
        file_logger = configure_file_logger(level="DEBUG")
        file_logger.propagate = False


@contextlib.contextmanager
def set_level(level):
    """Context manager to temporarily set the log level for the logger.

    Also sets the log level for the file logger if it exists.

    Args:
        level (str or int): The log level to set temporarily
            (e.g., "DEBUG", "INFO", logging.WARNING).

    Yields:
        None

    Example:
        .. doctest::

            >>> from zea import log
            >>> with log.set_level("ERROR"):
            ...     _ = log.info("Info messages will not be shown")
            ...     _ = log.error("Error messages will be shown")
    """
    prev_level = logger.level
    prev_file_level = file_logger.level if file_logger else None
    logger.setLevel(level)
    if file_logger:
        file_logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)
        if file_logger and prev_file_level is not None:
            file_logger.setLevel(prev_file_level)


logger = configure_console_logger(
    level=ZEA_LOG_LEVEL,
    name="zea",
    color=True,
    name_color="darkgreen",
)

# File logger is disabled by default
file_logger = None

# Do not propagate the log messages to the root logger
# Prevents double logging when using the logger in multiple modules
logger.propagate = False
