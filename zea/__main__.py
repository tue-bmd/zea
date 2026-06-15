"""Entry point for the zea toolbox.

Usage::

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer

"""

import argparse
import warnings
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

from zea.cli_args import ProcessArgs


@dataclass
class AppArgs:
    """Arguments for the interactive Gradio dataset visualizer."""

    share: bool = False
    server_port: int | None = None
    device: Annotated[
        str,
        tyro.conf.arg(help="Compute device passed to init_device (e.g. 'cpu', 'auto:1')."),
    ] = "auto:1"


def get_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser with ``process`` and ``app`` subcommands.

    Kept as plain argparse for ``sphinxcontrib-autoprogram`` doc generation and tests.
    The interactive ``main()`` uses :func:`tyro.cli` for richer help output.
    """
    parser = argparse.ArgumentParser(
        prog="zea",
        description="zea ultrasound toolbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # ── process ──────────────────────────────────────────────────────────────
    from zea.data.process import get_parser as _process_parser

    subparsers.add_parser(
        "process",
        help="Beamform a zea dataset using a pipeline YAML config.",
        parents=[_process_parser(add_help=False)],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── app ──────────────────────────────────────────────────────────────────
    app_p = subparsers.add_parser(
        "app",
        help="Launch the interactive Gradio dataset visualizer.",
    )
    app_p.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    app_p.add_argument(
        "--server-port",
        dest="server_port",
        type=int,
        default=None,
        help="Port for the Gradio server to listen on. Defaults to 7860.",
    )
    app_p.add_argument(
        "--device",
        type=str,
        default="auto:1",
        help="Compute device passed to init_device (e.g. 'cpu', 'auto:1').",
    )

    return parser


def main() -> None:
    """Dispatch to the requested subcommand using tyro for rich help output."""
    SubCmd = Union[
        Annotated[ProcessArgs, tyro.conf.subcommand("process")],
        Annotated[AppArgs, tyro.conf.subcommand("app")],
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        args = tyro.cli(SubCmd)

    from zea.internal.device import init_device

    init_device(args.device)

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


if __name__ == "__main__":
    main()
