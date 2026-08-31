"""Entry point for the zea toolbox.

Usage::

    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer
    zea convert <dataset> <src> <dst> [options]                    # convert raw datasets to zea
    zea data <operation> [options]                                 # manipulate zea data files
    zea datapaths [--user-config users.yaml]                       # set up your local data paths
    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea tools select [files] [options]                             # annotate regions of interest

"""

import os
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

if "ZEA_LOG_LEVEL" not in os.environ:
    from zea import log

    log.set_level("WARNING")

from zea.cli_args import (
    AppArgs,
    ConvertArgs,
    DataArgs,
    DataPathsArgs,
    ProcessArgs,
    ToolsArgs,
)

# subcommands that don't require a device
_NO_DEVICE_FNS = [DataArgs, DataPathsArgs, ToolsArgs]


@dataclass
class CLI:
    """Top-level CLI with global arguments and subcommands."""

    subcommand: tyro.conf.OmitSubcommandPrefixes[
        Union[
            Annotated[AppArgs, tyro.conf.subcommand("app")],
            Annotated[ConvertArgs, tyro.conf.subcommand("convert")],
            Annotated[DataArgs, tyro.conf.subcommand("data")],
            Annotated[DataPathsArgs, tyro.conf.subcommand("datapaths")],
            Annotated[ProcessArgs, tyro.conf.subcommand("process")],
            Annotated[ToolsArgs, tyro.conf.subcommand("tools")],
        ]
    ]
    device: Annotated[
        tyro.conf.CascadeSubcommandArgs[str],
        tyro.conf.arg(help="Compute device passed to init_device (e.g. 'cpu', 'auto:1')."),
    ] = "auto:1"


def _check_if_device_needed(subcommand) -> bool:
    """Check if the subcommand requires a device."""
    if subcommand.__class__ in _NO_DEVICE_FNS:
        return False
    if hasattr(subcommand, "subcommand"):
        return _check_if_device_needed(subcommand.subcommand)

    return True


def main() -> None:
    """Dispatch to the requested subcommand using tyro for rich help output."""
    cli_args = tyro.cli(CLI)
    args = cli_args.subcommand

    # Check if device is needed for the subcommand
    if _check_if_device_needed(args):
        from zea.internal.device import init_device

        # Conversion runs should not preallocate the full GPU, mirroring the
        # standalone ``python -m zea.data.convert`` entry point.
        init_device(cli_args.device, allow_preallocate=not isinstance(args, ConvertArgs))

    # Each subcommand dataclass imports its implementation lazily and runs it.
    args.run()


if __name__ == "__main__":
    main()
