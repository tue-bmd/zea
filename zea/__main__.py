"""Entry point for the zea toolbox.

Usage::

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer
    zea data <operation> [options]                                 # manipulate zea data files

"""

import os
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

if "ZEA_LOG_LEVEL" not in os.environ:
    from zea import log

    log.set_level("WARNING")

from zea.cli_args import AppArgs, DataArgs, ProcessArgs

# subcommands that don't require a device
_NO_DEVICE_FNS = [DataArgs]


@dataclass
class CLI:
    """Top-level CLI with global arguments and subcommands."""

    subcommand: tyro.conf.OmitSubcommandPrefixes[
        Union[
            Annotated[ProcessArgs, tyro.conf.subcommand("process")],
            Annotated[AppArgs, tyro.conf.subcommand("app")],
            Annotated[DataArgs, tyro.conf.subcommand("data")],
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

        init_device(cli_args.device)

    if isinstance(args, ProcessArgs):
        from zea.data.process import run_processing

        run_processing(
            args.dataset,
            args.config,
            args.key,
            args.n_frames,
            args.save_dir,
            args.save_as,
            args.keep_keys,
            args.timings,
            args.num_threads,
            args.overwrite,
            args.keep_dynamic_range,
            args.revision,
            args.config_revision,
        )

    elif isinstance(args, AppArgs):
        try:
            import gradio as gr
        except ImportError as exc:
            raise ImportError(
                "gradio is required for the zea app. Install with: pip install 'zea[app]'"
            ) from exc

        from zea.data.app import CSS, build_interface

        demo = build_interface()
        demo.launch(
            share=args.share,
            server_port=args.server_port,
            theme=gr.themes.Soft(primary_hue="violet", secondary_hue="yellow"),
            css=CSS,
        )
    elif isinstance(args, DataArgs):
        args.run()
    else:
        raise ValueError(f"Unknown command: {args}")


if __name__ == "__main__":
    main()
