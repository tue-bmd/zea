"""Command-line interface for copying a zea.Folder to a new location.

Usage:
    python -m zea.data --src <source> --dst <destination> --key <key>
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, Literal, Optional

import tyro

from zea.data.datasets import Folder


@dataclass
class FolderCopyArgs:
    """Arguments for copying a :class:`zea.Folder` to a new location."""

    src: Annotated[str, tyro.conf.arg(help="Source folder path.")]
    dst: Annotated[str, tyro.conf.arg(help="Destination folder path.")]
    key: Annotated[str, tyro.conf.arg(help="Key to access in the HDF5 files.")]
    # Default None lets Folder.copy auto-select the mode ('a' for a single key,
    # 'w' when --key is 'all'/'*'); passing 'a' would break the all/* case.
    mode: Optional[Literal["a", "w", "r+", "x"]] = None


def get_parser():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return tyro.extras.get_parser(FolderCopyArgs)


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        args = tyro.cli(FolderCopyArgs)

    src_folder = Folder(args.src, validate=False)
    src_folder.copy(args.dst, args.key, mode=args.mode)


if __name__ == "__main__":
    main()
