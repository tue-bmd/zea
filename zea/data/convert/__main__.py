"""CLI for converting common open-source ultrasound datasets to the zea format.

Use the ``zea convert`` subcommand::

    zea convert <dataset> <src> <dst> [options]

Examples::

    zea convert camus ./raw ./output --download
    zea convert cetus ./raw ./output --download
    zea convert echonet ./raw ./output
    zea convert echoxflow ./raw ./output
    zea convert us4us ./recording.pkl ./recording.hdf5 --mapping 0:image 1:raw_data

Run ``zea convert --help`` for all options.

Running this module directly (``python -m zea.data.convert ...``) remains supported
for backwards compatibility. The CLI dataclasses live in :mod:`zea.cli_args` (kept
free of heavy imports so ``zea --help`` renders without loading an ML backend) and
are re-exported here.
"""

import tyro

from zea.cli_args import (
    ConvertArgs,
    _Camus,
    _Cetus,
    _Echonet,
    _EchonetLVH,
    _EchoXFlow,
    _Picmus,
    _Us4us,
    _Verasonics,
)
from zea.cli_args import ConvertDataset as Dataset
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
    "_Us4us",
    "_Verasonics",
    "main",
]


def main():
    """Parse command-line arguments and dispatch to the selected dataset conversion routine."""
    args = tyro.cli(Dataset)
    args.run()


if __name__ == "__main__":
    init_device(allow_preallocate=False)
    main()
