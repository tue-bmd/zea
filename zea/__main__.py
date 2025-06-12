"""Main entry point for zea

Run as `zea --config path/to/config.yaml` to start the zea interface.
Or do not pass a config file to open a file dialog to choose a config file.

"""

import argparse
import sys
from pathlib import Path

from zea import log
from zea.visualize import set_mpl_style


def get_args():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(description="Process ultrasound data.")
    parser.add_argument("-c", "--config", type=str, default=None, help="path to config file.")
    parser.add_argument(
        "-t",
        "--task",
        default="run",
        choices=["run", "generate"],
        type=str,
        help="which task to run",
    )
    parser.add_argument(
        "--backend",
        default=None,
        type=str,
        help=(
            "Keras backend to use. Default is the one set by the environment "
            "variable KERAS_BACKEND."
        ),
    )
    parser.add_argument(
        "--skip_validate_file",
        default=False,
        action="store_true",
        help="Skip zea file integrity checks. Use with caution.",
    )
    parser.add_argument("--gui", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def main():
    """main entrypoint for UI script USBMD"""
    args = get_args()

    set_mpl_style()

    if args.backend:
        from zea.internal.setup_zea import set_backend

        set_backend(args.backend)

    wd = Path(__file__).parent.resolve()
    sys.path.append(str(wd))

    import keras

    from zea.generate import GenerateDataSet
    from zea.interface import Interface
    from zea.internal.checks import _DATA_TYPES
    from zea.internal.setup_zea import setup
    from zea.utils import keep_trying, strtobool

    config = setup(args.config)

    if args.task == "run":
        ui = Interface(
            config,
            validate_file=not args.skip_validate_file,
        )

        log.info(f"Using {keras.backend.backend()} backend")
        ui.run(plot=True)

    elif args.task == "generate":
        destination_folder = keep_trying(lambda: input(">> Give absolute destination folder path"))
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
