"""CLI for converting common open-source ultrasound datasets to the zea format.

The converters are available both as a subcommand of the ``zea`` command line tool
and as a standalone module:

    zea convert <dataset> <src> <dst> [options]        # preferred
    python -m zea.data.convert <dataset> <src> <dst>   # equivalent

Examples::

    zea convert camus ./raw ./output --download
    zea convert cetus ./raw ./output --download
    zea convert echonet ./raw ./output
    zea convert echoxflow ./raw ./output

Run ``zea convert --help`` (or ``python -m zea.data.convert --help``) for all options.

The CLI dataclasses themselves live in :mod:`zea.cli_args` (kept free of heavy
imports so ``zea --help`` renders without loading an ML backend); they are
re-exported here for backwards compatibility.
"""

import tyro

from zea.cli_args import ConvertArgs, ConvertDataset as Dataset
from zea.cli_args import (
    _Camus,
    _Cetus,
    _Echonet,
    _EchonetLVH,
    _EchoXFlow,
    _Picmus,
    _Verasonics,
)
from zea.internal.device import init_device

__all__ = [
    "ConvertArgs",
    "Dataset",
    "_Camus",
    "_Cetus",
    "_Echonet",
    "_EchonetLVH",
    "_EchoXFlow",
    "_Picmus",
    "_Verasonics",
    "main",
]


def main():
    """Parse command-line arguments and dispatch to the selected dataset conversion routine."""
    args = tyro.cli(Dataset)  # ty: ignore[no-matching-overload]
    args.run()


if __name__ == "__main__":
    init_device(allow_preallocate=False)
    main()
