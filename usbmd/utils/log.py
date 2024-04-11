"""Wrapper around python logging module to provide a simple interface for logging both
to the console and to a file with color support.

## Example usage
```python
from usbmd.utils import log

log.info("This is an info message")
path = "data/datafile.hdf5"
log.info(f"Saved to {log.yellow(path)}")

"""

import logging
import re
import sys
from pathlib import Path

# The logger to use
logger = None
file_logger = None

LOG_DIR = Path("log")


DEPRECATED_LEVEL_NUM = logging.WARNING + 5
logging.addLevelName(DEPRECATED_LEVEL_NUM, "DEPRECATED")
logging.DEPRECATED = DEPRECATED_LEVEL_NUM


def get_format_fn(name_format):
    """Returns the format function for the given format name."""
    return {
        "red": red,
        "green": green,
        "yellow": yellow,
        "darkgreen": darkgreen,
        "orange": orange,
        "bold": bold,
    }.get(name_format)


def red(string):
    """Adds ANSI escape codes to print a string in red around the string."""
    return "\033[38;5;196m" + str(string) + "\033[0m"


def green(string):
    """Adds ANSI escape codes to print a string in green around the string."""
    return "\033[38;5;46m" + str(string) + "\033[0m"


def yellow(string):
    """Adds ANSI escape codes to print a string in yellow around the string."""
    return "\033[38;5;226m" + str(string) + "\033[0m"


def darkgreen(string):
    """Adds ANSI escape codes to print a string in blue around the string."""
    return "\033[38;5;36m" + str(string) + "\033[0m"


def orange(string):
    """Adds ANSI escape codes to print a string in orange around the string."""
    return "\033[38;5;214m" + str(string) + "\033[0m"


def bold(string):
    """Adds ANSI escape codes to print a string in bold around the string."""
    return "\033[1m" + str(string) + "\033[0m"


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
            logging.DEPRECATED: logging.Formatter(
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
):
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

    # stdout stream handler if no handler is configured
    if not new_logger.hasHandlers():
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        new_logger.addHandler(console)

    return new_logger


def configure_file_logger(level="INFO"):
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

    # stdout stream handler if no handler is configured
    if not new_logger.hasHandlers():
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


def success(message):
    """Prints a message to the console in green."""
    logger.info(green(message))
    file_logger.info(remove_color_escape_codes(message))
    return message


def warning(message, *args, **kwargs):
    """Prints a message with log level warning."""
    logger.warning(message, *args, **kwargs)
    file_logger.warning(remove_color_escape_codes(message), *args, **kwargs)
    return message


def deprecated(message, *args, **kwargs):
    """Prints a message with custom log level DEPRECATED."""
    logger.log(DEPRECATED_LEVEL_NUM, message, *args, **kwargs)
    file_logger.log(
        DEPRECATED_LEVEL_NUM, remove_color_escape_codes(message), *args, **kwargs
    )
    return message


def error(message, *args, **kwargs):
    """Prints a message with log level error."""
    logger.error(message, *args, **kwargs)
    file_logger.error(remove_color_escape_codes(message), *args, **kwargs)
    return message


def debug(message, *args, **kwargs):
    """Prints a message with log level debug."""
    logger.debug(message, *args, **kwargs)
    file_logger.debug(remove_color_escape_codes(message), *args, **kwargs)
    return message


def info(message, *args, **kwargs):
    """Prints a message with log level info."""
    logger.info(message, *args, **kwargs)
    file_logger.info(remove_color_escape_codes(message), *args, **kwargs)
    return message


def critical(message, *args, **kwargs):
    """Prints a message with log level critical."""
    logger.critical(message, *args, **kwargs)
    file_logger.critical(message, *args, **kwargs)
    return message


def set_level(level):
    """Sets the log level of the logger."""
    logger.setLevel(level)
    file_logger.setLevel(level)


def set_file_logger_directory(directory):
    """Sets the log level of the logger."""
    # Add pylint exception
    # pylint: disable=global-statement
    global LOG_DIR, file_logger
    LOG_DIR = directory
    # Remove all handlers from the file logger
    for handler in file_logger.handlers:
        file_logger.removeHandler(handler)

    # Add file handler
    file_logger = configure_file_logger(level="DEBUG")


logger = configure_console_logger(
    level="DEBUG",
    name="usbmd",
    color=True,
    name_color="darkgreen",
)
file_logger = configure_file_logger(level="DEBUG")
