import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import keras
import numpy as np
from keras import ops

from zea import display, io_lib, log
from zea.backend import jit
from zea.config import Config
from zea.data.data_format import generate_zea_dataset
from zea.data.dataloader import Dataloader
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.internal.core import reduce_to_signature
from zea.internal.device import init_device
from zea.ops.pipeline import Pipeline
from zea.scan import Scan
from zea.utils import FunctionTimer

SUPPORTED_FORMATS = ["gif", "mp4", "hdf5"]

try:
    import SimpleITK as sitk

    SUPPORTED_FORMATS += ["nii.gz"]
except ImportError:
    sitk = None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a zea dataset to processed videos using a zea pipeline "
        + "defined in a config.yaml file. Will process the frames sequentially to allow "
        + "for temporal processing techniques.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path/URI to the zea dataset.",
    )
    parser.add_argument(
        "save_dir",
        type=Path,
        help="Directory where output videos are written.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml. Defaults to <dataset>/config.yaml.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="raw_data",
        help="Data type to load from each file.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Number of frames to load (all frames when omitted).",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="gif",
        help="Format to save output videos (e.g. 'gif' or 'mp4').",
    )
    parser.add_argument(
        "--keep_keys",
        nargs="+",
        default=["maxval"],
        help="Keys from the pipeline output to keep for the next iteration.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Whether to time dataloader and pipeline operations and save timings to a yaml file.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads to use for dataloader. Default is 8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files in `save_dir` on conflict. Default is False.",
    )
    return parser


def run_processing(
    dataset_path: str,
    config_path: str,
    data_type: str,
    n_frames: int | None,
    save_dir: Path,
    save_as: str = "gif",
    keep_keys=("maxval",),
    timings=False,
    num_threads=8,
    overwrite=False,
) -> None:
    if save_as == "nii.gz" and sitk is None:
        raise ValueError("SimpleITK is not installed, cannot save as nii.gz")

    if save_as not in SUPPORTED_FORMATS:
        raise ValueError(f"save_as must be one of {SUPPORTED_FORMATS}, got {save_as}")

    @jit
    def to_8bit(data, dynamic_range):
        # data = ops.convert_to_numpy(data)
        data = ops.nan_to_num(data, nan=dynamic_range[0])
        data = display.to_8bit(data, dynamic_range, to_numpy=False)
        return data

    # Load config and pipeline
    config = Config.from_path(config_path)
    pipeline = Pipeline.from_path(config_path, with_batch_dim=False)

    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Peek at the first file to determine which transmits the scan will select.
    # Passing these to the dataloader lets h5py read only those transmits from disk,
    # avoiding a large CPU-side slice on every frame.
    dataset_files = Dataset(dataset_path, validate=False)
    first_file_path = dataset_files.file_paths[0]
    dataset_files.close()
    with File(first_file_path) as first_file:
        first_scan = first_file.scan(**config.scan.as_dict())
    selected_transmits = [int(i) for i in first_scan.selected_transmits]
    # h5py requires strictly increasing indices. The scan also uses this list to index
    # its own per-transmit parameters, so re-ordering here would desync frame vs params.
    if any(b <= a for a, b in zip(selected_transmits, selected_transmits[1:])):
        raise ValueError(
            "selected_transmits must be strictly increasing to pre-filter on disk; "
            f"got {selected_transmits}"
        )

    # Set up dataloader. axis_selections pre-filters the transmits axis (axis 1 of
    # raw_data: frames, transmits, ...) at read time.
    dataloader = Dataloader(
        dataset_path,
        key=data_type,
        batch_size=None,
        shuffle=False,
        return_filename=True,
        limit_n_frames=n_frames,
        n_frames=1,
        num_threads=num_threads,
        insert_frame_axis=False,
        sort_files=True,
        axis_selections={1: selected_transmits},
    )
    iterator = iter(dataloader)
    total_batches = len(dataloader)

    # Define functions to time if timings flag is set
    get_data = lambda: next(iterator)
    prepare_parameters = pipeline.prepare_parameters
    pipeline_call = pipeline.__call__

    if timings:
        timer = FunctionTimer()
        get_data = timer(get_data, name="dataloader")
        prepare_parameters = timer(prepare_parameters, name="prepare_parameters")
        pipeline_call = timer(pipeline_call, name="pipeline")

    # Initialize variables for loop
    prev_file_path = None
    data_output = []
    filestem = None
    scan = None
    scan_dict = None
    _DEFAULT_FPS = 20
    fps = _DEFAULT_FPS

    def save_video_worker(video: np.ndarray, save_path: Path, fps: int, scan_dict: dict):
        if save_path.exists() and not overwrite:
            log.warning(
                f"The file {save_path} already exists. "
                "If you wish to overwrite, add the flag --overwrite."
            )

        if save_as in ["mp4", "gif"]:
            io_lib.save_video(video, save_path, fps=fps)
        elif save_as == "hdf5":
            kwargs = reduce_to_signature(generate_zea_dataset, scan_dict)
            kwargs.pop("probe_name", None)  # TODO: handle more gracefully
            generate_zea_dataset(
                save_path,
                image=video,
                cast_to_float=False,
                probe_name="generic",
                overwrite=overwrite,
                **kwargs,
            )
        elif save_as == "nii.gz":
            sitk.WriteImage(sitk.GetImageFromArray(video), save_path)
            log.info(f"sitk dataset written to {log.yellow(save_path)}")

    # Start iterating through all frames in dataset
    pbar = keras.utils.Progbar(total_batches)
    with ThreadPoolExecutor(max_workers=1) as executor:
        save_future = None
        for i in range(total_batches + 1):
            if i < total_batches:
                frame, metadata = get_data()
                file_path = metadata["fullpath"]
            else:
                # To trigger saving of the last video after the loop
                file_path = None

            # Save video and load next scan if we've moved to a new file
            if file_path != prev_file_path:
                if prev_file_path is not None:
                    data_output = ops.convert_to_numpy(data_output)
                    save_path = save_dir / f"{filestem}.{save_as}"
                    if save_future is not None:
                        save_future.result()
                    save_future = executor.submit(
                        save_video_worker, data_output, save_path, fps, scan_dict
                    )
                    data_output = []
                    if file_path is None:
                        # No more files to process, so we can
                        break

                prev_file_path = file_path

                # TODO: add prefetching for scan etc...
                with File(file_path) as file:
                    # Get original frame-rate before scan transmits are subsampled.
                    filestem = file.stem
                    scan_dict = file.get_scan_parameters()
                    scan: Scan = file.scan(**config.scan.as_dict())
                    fps = scan.frames_per_second or _DEFAULT_FPS

                    # The dataloader is pre-filtering transmits based on the first
                    # file's scan. Make sure this file agrees.
                    file_transmits = [int(i) for i in scan.selected_transmits]
                    if file_transmits != selected_transmits:
                        raise ValueError(
                            f"selected_transmits for {file_path} ({file_transmits}) "
                            f"differ from the first file ({selected_transmits}). "
                            "axis_selections requires a uniform transmit selection across files."
                        )

                    # Filter config scan to only include keys the pipeline needs.
                    # This avoids the warning that the pipeline would print otherwise.
                    to_pipeline_scan_dict = {
                        k: v for k, v in config.scan.as_dict().items() if pipeline.needs(k)
                    }
                    params = prepare_parameters(scan=scan, **to_pipeline_scan_dict)

            # Run the pipeline, store output frame, and forward keys to the next iteration
            output = pipeline_call(data=frame, **params)
            processed_frame = output["data"]
            processed_frame = to_8bit(processed_frame, scan.dynamic_range)
            data_output.append(processed_frame)
            pbar.add(1)
            for key in keep_keys:
                if key in output:
                    params[key] = output[key]

            # After each frame, append timings to yaml
            if timings:
                for tname in timer.timings.keys():
                    timer.append_to_yaml(save_dir / f"timings_{tname}.yaml", tname)

    if timings:
        timer.print()


def main() -> None:
    args = get_parser().parse_args()
    init_device()

    config_path = args.config or f"{args.dataset}/config.yaml"
    run_processing(
        args.dataset,
        config_path,
        args.data_type,
        args.n_frames,
        args.save_dir,
        args.save_as,
        args.keep_keys,
        args.timings,
        args.num_threads,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
