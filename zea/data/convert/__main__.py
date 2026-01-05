import argparse


def _add_parser_args_echonet(subparsers):
    """Add Echonet specific arguments to the parser."""
    echonet_parser = subparsers.add_parser("echonet", help="Convert Echonet dataset")
    echonet_parser.add_argument("src", type=str, help="Source folder path")
    echonet_parser.add_argument("dst", type=str, help="Destination folder path")
    echonet_parser.add_argument(
        "--split_path",
        type=str,
        help="Path to the split.yaml file containing the dataset split if a split should be copied",
    )
    echonet_parser.add_argument(
        "--no_hyperthreading",
        action="store_true",
        help="Disable hyperthreading for multiprocessing",
    )


def _add_parser_args_camus(subparsers):
    """Add CAMUS specific arguments to the parser."""
    camus_parser = subparsers.add_parser("camus", help="Convert CAMUS dataset")
    camus_parser.add_argument("src", type=str, help="Source folder path")
    camus_parser.add_argument("dst", type=str, help="Destination folder path")
    camus_parser.add_argument(
        "--no_hyperthreading",
        action="store_true",
        help="Disable hyperthreading for multiprocessing",
    )


def _add_parser_args_echonetlvh(subparsers):
    """Add EchonetLVH specific arguments to the parser."""
    echonetlvh_parser = subparsers.add_parser("echonetlvh", help="Convert EchonetLVH dataset")
    echonetlvh_parser.add_argument("src", type=str, help="Source folder path")
    echonetlvh_parser.add_argument("dst", type=str, help="Destination folder path")
    echonetlvh_parser.add_argument(
        "--no_rejection",
        action="store_true",
        help="Do not reject sequences in manual_rejections.txt",
    )
    echonetlvh_parser.add_argument(
        "--rejection_path",
        type=str,
        default=None,
        help="Path to custom rejection txt file (defaults to manual_rejections.txt)",
    )
    echonetlvh_parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Specify which BatchX directory to process, e.g. --batch=Batch2",
    )
    echonetlvh_parser.add_argument(
        "--convert_measurements",
        action="store_true",
        help="Only convert measurements CSV file",
    )
    echonetlvh_parser.add_argument(
        "--convert_images",
        action="store_true",
        help="Only convert image files",
    )
    echonetlvh_parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
    )
    echonetlvh_parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if parameters already exist",
    )
    echonetlvh_parser.add_argument(
        "--no_hyperthreading",
        action="store_true",
        help="Disable hyperthreading for multiprocessing",
    )


def _add_parser_args_picmus(subparsers):
    """Add PICMUS specific arguments to the parser."""
    picmus_parser = subparsers.add_parser("picmus", help="Convert PICMUS dataset")
    picmus_parser.add_argument("src", type=str, help="Source folder path")
    picmus_parser.add_argument("dst", type=str, help="Destination folder path")


def _add_parser_args_verasonics(subparsers):
    verasonics_parser = subparsers.add_parser(
        "verasonics", help="Convert Verasonics data to zea dataset"
    )
    verasonics_parser.add_argument("src", type=str, help="Source folder path")
    verasonics_parser.add_argument("dst", type=str, help="Destination folder path")
    verasonics_parser.add_argument(
        "--frames",
        default=["all"],
        type=str,
        nargs="+",
        help="The frames to add to the file. This can be a list of integers, a range "
        "of integers (e.g. 4-8), or 'all'.",
    )
    verasonics_parser.add_argument(
        "--allow_accumulate",
        action="store_true",
        help=(
            "Sometimes, some transmits are already accumulated on the Verasonics system "
            "(e.g. harmonic imaging through pulse inversion). In this case, the mode in the "
            "Receive structure is set to 1 (accumulate). If this flag is set, such files "
            "will be processed. Otherwise, an error is raised when such a mode is detected."
        ),
    )
    verasonics_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for conversion (e.g., 'cpu' or 'gpu:0').",
    )
    verasonics_parser.add_argument(
        "--no_compression",
        action="store_true",
        help="Disable compression when saving the zea dataset. By default, compression is "
        "enabled, which reduces disk space at the cost of increased conversion time.",
    )


def get_parser():
    """Build and parse command-line arguments for converting raw datasets to a zea dataset."""
    parser = argparse.ArgumentParser(description="Convert raw data to a zea dataset.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    _add_parser_args_echonet(subparsers)
    _add_parser_args_echonetlvh(subparsers)
    _add_parser_args_camus(subparsers)
    _add_parser_args_picmus(subparsers)
    _add_parser_args_verasonics(subparsers)
    return parser


def main():
    """
    Parse command-line arguments and dispatch to the selected dataset conversion routine.

    This function obtains CLI arguments via get_args() and calls the corresponding converter
    (convert_echonet, convert_echonetlvh, convert_camus, convert_picmus, or convert_verasonics)
    based on args.dataset.
    Raises a ValueError if args.dataset is not one of the supported choices.
    """
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == "echonet":
        from zea.data.convert.echonet import convert_echonet

        convert_echonet(args)
    elif args.dataset == "echonetlvh":
        from zea.data.convert.echonetlvh import convert_echonetlvh

        convert_echonetlvh(args)
    elif args.dataset == "camus":
        from zea.data.convert.camus import convert_camus

        convert_camus(args)
    elif args.dataset == "picmus":
        from zea.data.convert.picmus import convert_picmus

        convert_picmus(args)
    elif args.dataset == "verasonics":
        from zea.data.convert.verasonics import convert_verasonics

        convert_verasonics(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
