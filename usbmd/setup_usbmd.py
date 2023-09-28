"""module summary
Author(s): Tristan Stevens
Date: 25/09/2023
"""
from pathlib import Path

from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.config_validation import check_config
from usbmd.utils.git_info import get_git_summary
from usbmd.utils.utils import filename_from_window_dialog


def setup_config(file=None):
    """Setup function for config. Retrieves config file and checks for validity.

    Args:
        file (str, optional): file path to config yaml. Defaults to None.
            if None, argparser is checked. If that is None as well, the window
            ui will pop up for choosing the config file manually.

    Returns:
        config (dict): config object / dict.

    """
    if file is None:
        # if no argument is provided resort to UI window
        filetype = 'yaml'
        try:
            file = filename_from_window_dialog(
                f'Choose .{filetype} file',
                filetypes=((filetype, '*.' + filetype),),
                initialdir='./configs',
            )
        except Exception as e:
            raise ValueError (
                'Please specify the path to a config file through --config flag ' \
                'if GUI is not working (usually on headless servers).') from e

    config = load_config_from_yaml(Path(file))
    print(f'Using config file: {file}')
    config = check_config(config)

    ## git
    cwd = Path.cwd().stem
    if cwd in ('Ultrasound-BMd', 'usbmd'):
        config['git'] = get_git_summary()

    return config
