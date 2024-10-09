"""Main entry point for USBMD

Run as `usbmd --config path/to/config.yaml` to start the USBMD GUI.
Or do not pass a config file to open a file dialog to choose a config file.

- **Author(s)**     : Tristan Stevens
- **Date**          : November 18th, 2021
"""

import argparse
import asyncio
import sys
from pathlib import Path

wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from usbmd.generate import GenerateDataSet
from usbmd.interface import Interface
from usbmd.setup_usbmd import setup
from usbmd.utils import keep_trying, log, strtobool
from usbmd.utils.checks import _DATA_TYPES
from usbmd.utils.gui import USBMDApp
from usbmd.utils.io_lib import start_async_app


def get_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(description="Process ultrasound data.")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="path to config file."
    )
    parser.add_argument(
        "-t",
        "--task",
        default="run",
        choices=["run", "generate"],
        type=str,
        help="which task to run",
    )
    # pylint: disable=no-member
    parser.add_argument("--gui", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def main():
    """main entrypoint for UI script USBMD"""
    args = get_args()
    config = setup(args.config)

    if args.task == "run":
        ui = Interface(config)

        log.info(f"Using {config.ml_library} backend")

        if args.gui:
            log.warning(
                "GUI is very much in beta, please report any bugs to "
                "https://github.com/tue-bmd/ultrasound-toolbox."
            )
            try:
                asyncio.run(
                    start_async_app(
                        USBMDApp,
                        title="USBMD GUI",
                        ui=ui,
                        resolution=(600, 300),
                        verbose=True,
                        config=config,
                    )
                )
            except RuntimeError as e:
                # probably a better way to handle this...
                if str(e) == "Event loop stopped before Future completed.":
                    log.info("GUI closed.")
                else:
                    raise e
        else:
            ui.run(plot=True)

    elif args.task == "generate":
        destination_folder = keep_trying(
            lambda: input(
                ">> Give destination folder path"
                + " (if relative path, will be relative to the original dataset): "
            )
        )
        to_dtype = keep_trying(
            lambda: input(f">> Specify data type \n{_DATA_TYPES}: "),
            required_set=_DATA_TYPES,
        )
        retain_folder_structure = keep_trying(
            lambda: strtobool(input(">> Retain folder structure? (Y/N): "))
        )
        if to_dtype in ["image", "image_sc"]:
            filetype = keep_trying(
                lambda: input(">> Filetype (hdf5, png): "), required_set=["hdf5", "png"]
            )
        else:
            filetype = "hdf5"

        generator = GenerateDataSet(
            config,
            to_dtype=to_dtype,
            destination_folder=destination_folder,
            retain_folder_structure=retain_folder_structure,
            filetype=filetype,
        )
        generator.generate()


if __name__ == "__main__":
    main()
