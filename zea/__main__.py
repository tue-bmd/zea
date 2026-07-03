"""Entry point for the zea toolbox.

Usage::

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer

"""

import os
import warnings
from typing import Annotated, Optional, Sequence, Union

import zea

if "ZEA_LOG_LEVEL" not in os.environ:
    zea.log.set_level("WARNING")

import tyro

from zea.cli_args import AppArgs, ProcessArgs

# Top-level CLI: a union of subcommands, each tagged with its command name.
SubCmd = Union[
    Annotated[ProcessArgs, tyro.conf.subcommand("process")],
    Annotated[AppArgs, tyro.conf.subcommand("app")],
]


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse ``argv`` into a :class:`ProcessArgs` or :class:`AppArgs` instance.

    Passing ``argv=None`` reads from ``sys.argv`` (normal CLI usage). A ``--help``
    request or a parse error raises :class:`SystemExit`, matching tyro's behavior.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return tyro.cli(SubCmd, args=argv)  # ty: ignore[no-matching-overload]


def main() -> None:
    """Dispatch to the requested subcommand using tyro for rich help output."""
    args = parse_args()

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
