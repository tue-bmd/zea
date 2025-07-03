"""Command-line interface for copying a zea.Folder to a new location.

Usage:
    python -m zea.data <source_folder> <destination_folder> <key> [--all_keys]
"""

import argparse

from zea import Folder


def main():
    parser = argparse.ArgumentParser(description="Copy a zea.Folder to a new location.")
    parser.add_argument("src", help="Source folder path")
    parser.add_argument("dst", help="Destination folder path")
    parser.add_argument("key", help="Key to access in the hdf5 files")
    parser.add_argument(
        "--all_keys", action="store_true", help="Copy all keys from the source files"
    )
    parser.add_argument(
        "--mode",
        default="a",
        choices=["a", "w", "r+", "x"],
        help="Mode in which to open the destination files (default: 'a')",
    )

    args = parser.parse_args()

    src_folder = Folder(args.src, args.key, validate=False)
    src_folder.copy(args.dst, all_keys=args.all_keys, mode=args.mode)


if __name__ == "__main__":
    main()
