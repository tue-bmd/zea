"""CLI for converting common open-source ultrasound datasets to the zea format.

Use the ``zea convert`` subcommand::

    zea convert <dataset> <src> <dst> [options]

Examples::

    zea convert camus ./raw ./output --download
    zea convert cetus ./raw ./output --download
    zea convert echonet ./raw ./output
    zea convert echoxflow ./raw ./output

Run ``zea convert --help`` for all options.

Running this module directly (``python -m zea.data.convert ...``) remains supported
for backwards compatibility. The CLI dataclasses live in :mod:`zea.cli_args` (kept
free of heavy imports so ``zea --help`` renders without loading an ML backend) and
are re-exported here.
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
