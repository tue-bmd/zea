"""Main entry point for zea

Run as `zea --config path/to/config.yaml` to start the zea interface.
Or do not pass a config file to open a file dialog to choose a config file.

"""

import argparse
import sys
from pathlib import Path

from zea.visualize import set_mpl_style


def get_parser():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Load and process ultrasound data based on a configuration file."
    )
    parser.add_argument("-c", "--config", type=str, default=None, help="path to the config file.")
    parser.add_argument(
        "-t",
        "--task",
        default="view",
        choices=["view"],
        type=str,
        help="Which task to run. Currently only 'view' is supported.",
    )
    parser.add_argument(
        "--skip_validate_file",
        default=False,
        action="store_true",
        help="Skip zea file integrity checks. Use with caution.",
    )
    return parser


def main():
    """main entrypoint for zea"""
    args = get_parser().parse_args()

    set_mpl_style()

    wd = Path(__file__).parent.resolve()
    sys.path.append(str(wd))

    from zea.interface import Interface
    from zea.internal.setup_zea import setup

    config = setup(args.config)

    if args.task == "view":
        cli = Interface(
            config,
            validate_file=not args.skip_validate_file,
        )

        cli.run(plot=True)
    else:
        raise ValueError(f"Unknown task {args.task}, see `zea --help` for available tasks.")


if __name__ == "__main__":
    main()
