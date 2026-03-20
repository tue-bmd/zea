import argparse
from pathlib import Path

import keras
import numpy as np
from keras import ops

import zea
from zea.utils import FunctionTimer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a zea dataset to processed videos using a zea pipeline "
        + "defined in a config.yaml file.",
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
) -> None:
    def to_8bit(data, scan):
        data = ops.convert_to_numpy(data)
        data = np.nan_to_num(data, copy=False, nan=scan.dynamic_range[0])
        data = zea.display.to_8bit(data, scan.dynamic_range)
        return data

    def save_video_frames(filestem: str, frames: list[np.ndarray], fps: float) -> None:
        video = np.stack(frames, axis=0)
        save_path = save_dir / f"{filestem}.{save_as}"
        print("\n")
        zea.io_lib.save_video(video, save_path, fps=fps)
        print("\n")

    # Load config and pipeline
    config = zea.Config.from_path(config_path)
    pipeline = zea.Pipeline.from_path(config_path, with_batch_dim=False)

    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set up dataloader
    dataloader = zea.Dataloader(
        dataset_path,
        key=data_type,
        batch_size=None,
        shuffle=False,
        return_filename=True,
        limit_n_frames=n_frames,
        n_frames=1,
        num_threads=4,
        insert_frame_axis=False,
        sort_files=True,
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
    fps = None

    # Start iterating through all frames in dataset
    pbar = keras.utils.Progbar(total_batches)
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
                save_video_frames(filestem, data_output, fps)
                data_output = []
                if file_path is None:
                    # No more files to process, so we can
                    break

            prev_file_path = file_path
            with zea.File(file_path) as file:
                # Get original frame-rate before scan transmits are subsampled.
                filestem = file.stem
                scan: zea.Scan = file.scan(**config.scan.as_dict())
                fps = scan.frames_per_second
                params = prepare_parameters(scan=scan, **config.scan.as_dict())

        # Select the right transmits
        # TODO: this can be optimized by only loading the selected transmits from disk
        frame = frame[scan.selected_transmits]

        # Run the pipeline, store output frame, and forward keys to the next iteration
        output = pipeline_call(data=frame, **params)
        processed_frame = output["data"]
        processed_frame = to_8bit(processed_frame, scan)
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
    zea.init_device()

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
    )


if __name__ == "__main__":
    main()
