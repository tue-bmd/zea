import argparse

from .camus import convert_camus
from .matlab import convert_matlab
from .picmus import convert_picmus
from .echonet import convert_echonet
from .echonetlvh.convert_raw_to_usbmd import convert_echonetlvh


def get_args():
    """
    Build and parse command-line arguments for converting raw datasets to a USBMD dataset.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            dataset (str): One of "echonet", "echonetlvh", "camus", "picmus", "matlab".
            src (str): Source folder path.
            dst (str): Destination folder path.
            dst_npz (str|None): Optional alternate destination path for numpy output.
            split_path (str|None): Optional path to a split.yaml to copy dataset splits.
            no_hyperthreading (bool): Disable hyperthreading for multiprocessing.
            frames (list[str]): MATLAB frames spec (e.g., ["all"], integers, or ranges like "4-8").
            no_rejection (bool): EchonetLVH flag to skip manual_rejections.txt filtering.
            batch (str|None): EchonetLVH Batch directory to process (e.g., "Batch2").
            convert_measurements (bool): EchonetLVH flag to convert only measurements CSV.
            convert_images (bool): EchonetLVH flag to convert only image files.
            max_files (int|None): EchonetLVH maximum number of files to process.
            force (bool): EchonetLVH flag to force recomputation even if parameters exist.
    """
    parser = argparse.ArgumentParser(description="Convert raw data to a USBMD dataset.")
    parser.add_argument(
        "dataset",
        choices=["echonet", "echonetlvh", "camus", "picmus", "matlab"],
        help="Raw dataset to convert",
    )
    parser.add_argument("src", type=str, help="Source folder path")
    parser.add_argument("dst", type=str, help="Destination folder path")
    parser.add_argument(
        "--dst_npz",
        type=str,
        default=None,
        help="Additional destination folder path if also saving to numpy",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        help="Path to the split.yaml file containing the dataset split if a split should be copied",
    )
    parser.add_argument(
        "--no_hyperthreading",
        action="store_true",
        help="Disable hyperthreading for multiprocessing",
    )
    # Dataset specific arguments:

    # MATLAB
    parser.add_argument(
        "--frames",
        default=["all"],
        type=str,
        nargs="+",
        help="MATLAB: The frames to add to the file. This can be a list of integers, a range "
        "of integers (e.g. 4-8), or 'all'.",
    )
    # ECHONET_LVH
    parser.add_argument(
        "--no_rejection",
        action="store_true",
        help="EchonetLVH: Do not reject sequences in manual_rejections.txt",
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="EchonetLVH: Specify which BatchX directory to process, e.g. --batch=Batch2",
    )
    parser.add_argument(
        "--convert_measurements",
        action="store_true",
        help="EchonetLVH: Only convert measurements CSV file",
    )
    parser.add_argument(
        "--convert_images", action="store_true", help="EchonetLVH: Only convert image files"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="EchonetLVH: Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="EchonetLVH: Force recomputation even if parameters already exist",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Parse command-line arguments and dispatch to the selected dataset conversion routine.
    
    This function obtains CLI arguments via get_args() and calls the corresponding converter (convert_echonet, convert_echonetlvh, convert_camus, convert_picmus, or convert_matlab) based on args.dataset. Raises a ValueError if args.dataset is not one of the supported choices.
    """
    args = get_args()
    if args.dataset == "echonet":
        convert_echonet(args)
    elif args.dataset == "echonetlvh":
        convert_echonetlvh(args)
    elif args.dataset == "camus":
        convert_camus(args)
    elif args.dataset == "picmus":
        convert_picmus(args)
    elif args.dataset == "matlab":
        convert_matlab(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()