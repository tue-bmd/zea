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
        default="view",
        choices=["view"],
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
    """main entrypoint for zea"""
    args = get_args()

    set_mpl_style()

    if args.backend:
        from zea.internal.setup_zea import set_backend

        set_backend(args.backend)

    wd = Path(__file__).parent.resolve()
    sys.path.append(str(wd))

    import keras

    from zea.interface import Interface
    from zea.internal.setup_zea import setup

    config = setup(args.config)

    if args.task == "view":
        cli = Interface(
            config,
            validate_file=not args.skip_validate_file,
        )

        log.info(f"Using {keras.backend.backend()} backend")
        cli.run(plot=True)
    else:
        raise ValueError(f"Unknown task {args.task}, see `zea --help` for available tasks.")


if __name__ == "__main__":
    main()
