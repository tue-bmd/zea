"""CLI for beamforming a zea dataset with a pipeline defined in a YAML config file.

Usage:
    python -m zea.data.process --dataset <path> --config <config.yaml>
"""

import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from keras import ops

from zea import io_lib, log
from zea.backend import jit
from zea.cli_args import DEFAULT_DEVICE, DEVICE_HELP, SUPPORTED_FORMATS, ProcessArgs
from zea.config import Config
from zea.data.dataloader import Dataloader
from zea.data.datasets import Dataset
from zea.data.file import File, _GroupProxy
from zea.data.file_operations import output_blocked
from zea.data.spec import strip_track_prefix
from zea.func import translate
from zea.internal.checks import _NON_IMAGE_DATA_TYPES
from zea.internal.device import init_device
from zea.ops.pipeline import Pipeline
from zea.utils import FunctionTimer, ProgressBar


def _axis_selections_from_params(parameters) -> dict | None:
    """Return HDF5 axis_selections for transmit-axis pre-filtering, or None."""
    _tx = getattr(parameters, "selected_transmits", None)
    if _tx is not None:
        return {1: sorted(int(t) for t in _tx)}
    return None


sitk: types.ModuleType | None = None
try:
    import SimpleITK as _sitk

    sitk = _sitk
except ImportError:
    pass


def _get_config_parameters(config: Config) -> dict:
    """Return the config parameters dict, handling missing or empty sections."""
    params = getattr(config, "parameters", None)
    if params is None:
        return {}
    return params.as_dict() if hasattr(params, "as_dict") else dict(params)


def _key_requires_pipeline(key: str) -> bool:
    """Return True if ``key`` holds raw RF/pre-beamformed data that needs a pipeline.

    Normalizes the key so aliases like ``raw_data`` match ``_NON_IMAGE_DATA_TYPES``.
    """
    normalized = (key or "").strip()
    normalized = strip_track_prefix(normalized)
    normalized = normalized.removeprefix("data/").removesuffix("/values")
    return normalized in _NON_IMAGE_DATA_TYPES


def _resolve_track(file: File, key: str, track: str | None) -> tuple[int | None, str]:
    """Pick the track to process and return ``(track_index, data_key)``.

    ``track`` is matched against the file's labels first and read as an index
    only when no label matches, so a numeric label still wins. Files with a
    single track have nothing to address and return ``(None, key)`` unchanged.

    Raises:
        ValueError: If the file has several tracks and none was selected, or the
            requested track does not exist in this file.
    """
    labels = file.track_labels
    if file._n_tracks <= 1:
        return None, key
    if track is None:
        raise ValueError(
            f"{file.path} has {len(labels)} tracks {labels} but no track was selected. "
            "Pass --track <label|index> to pick one."
        )
    try:
        index = labels.index(track) if track in labels else int(track)
    except ValueError:
        index = -1  # neither a label nor a number: report it like a bad index
    if not 0 <= index < len(labels):
        raise ValueError(f"No track {track!r} in {file.path}. Available tracks: {labels}.")
    return index, f"tracks/track_{index}/data/{strip_track_prefix(key).removeprefix('data/')}"


def _load_track_parameters(file: File, track_index: int | None):
    """Load the acquisition parameters of ``track_index`` (or of the whole file)."""
    if track_index is None:
        return file.load_parameters()
    return file.tracks[track_index].load_parameters()


def _run_passthrough(
    dataset_path: str,
    key: str,
    n_frames: int | None,
    save_dir: Path,
    save_as: str,
    overwrite: bool,
    **hf_kwargs,
) -> None:
    """Save data frames directly without a beamforming pipeline."""
    if save_as not in ("gif", "mp4", "hdf5"):
        raise ValueError(f"Passthrough mode only supports gif/mp4/hdf5, got {save_as!r}")
    save_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(dataset_path, lazy=True, _suggest_lazy=False, **hf_kwargs) as ds:
        pbar = ProgressBar(len(ds))
        for i in range(len(ds)):
            f = ds[i]  # lazy download for hf:// paths; returns cached File handle
            data_key = f.format_key(key)
            _dset = f.dataset(data_key)
            # A map-style key such as "image" resolves to a group, not a dataset: read its
            # values child, the same way load_file does.
            if isinstance(_dset, _GroupProxy):
                _dset = _dset.values
            arr = np.asarray(_dset[:n_frames] if n_frames is not None else _dset[:])
            filestem = f.stem

            # Ensure (N, H, W) — squeeze any leading single-element dims
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr[np.newaxis]  # add frame axis

            if arr.dtype != np.uint8:
                lo, hi = float(arr.min()), float(arr.max())
                arr = (
                    ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                    if hi > lo
                    else np.zeros_like(arr, dtype=np.uint8)
                )

            save_path = save_dir / f"{filestem}.{save_as}"
            if output_blocked(save_path, overwrite):
                log.warning(f"File {save_path} already exists. Use --overwrite to replace it.")
            else:
                if save_as in ("gif", "mp4"):
                    io_lib.save_video(arr, save_path, fps=20)
                elif save_as == "hdf5":
                    File.create(save_path, data={"image": {"values": arr}}, overwrite=overwrite)
                log.info(f"Saved {log.yellow(save_path)}")

            pbar.add(1)


def run_processing(
    dataset_path: str,
    config_path: str,
    key: str,
    n_frames: int | None,
    save_dir: Path,
    save_as: str = "gif",
    keep_keys=("maxval",),
    timings=False,
    num_threads=16,
    overwrite=False,
    keep_dynamic_range=False,
    revision: str | None = None,
    config_revision: str | None = None,
    track: str | None = None,
) -> None:
    if keep_dynamic_range and save_as != "hdf5":
        raise ValueError("--keep_dynamic_range is only supported with --save_as hdf5.")
    if save_as == "nii.gz" and sitk is None:
        raise ValueError("SimpleITK is not installed; cannot save as nii.gz.")
    if save_as not in SUPPORTED_FORMATS:
        raise ValueError(f"save_as must be one of {SUPPORTED_FORMATS}, got {save_as!r}")

    dataset_hf_kwargs = {"revision": revision} if revision is not None else {}
    config_hf_kwargs = (
        {"revision": config_revision if config_revision is not None else revision}
        if (config_revision or revision)
        else {}
    )
    config = Config.from_path(config_path, **config_hf_kwargs)
    config_params = _get_config_parameters(config)

    # Peek at the first file for the track to read (the Dataloader takes one key for
    # the whole dataset, so ``data_key`` has to name it) and for selected_transmits,
    # letting the dataloader pre-filter the transmit axis at HDF5 read time. lazy=True
    # streams hf:// paths, fetching only the chunks the peek touches; validate=False
    # because the Dataloader validates these same files anyway.
    track_index: int | None = None
    data_key = key
    axis_selections: dict | None = None
    with Dataset(
        dataset_path, validate=False, lazy=True, _suggest_lazy=False, **dataset_hf_kwargs
    ) as _peek_ds:
        if len(_peek_ds) > 0:
            _peek_f = _peek_ds[0]
            track_index, data_key = _resolve_track(_peek_f, key, track)
            if _key_requires_pipeline(data_key):
                try:
                    _peek_params = _load_track_parameters(_peek_f, track_index)
                    _peek_params.update(config_params)
                    axis_selections = _axis_selections_from_params(_peek_params)
                except Exception:
                    pass  # fall back to runtime slicing if the peek fails

    try:
        pipeline = Pipeline.from_path(config_path, with_batch_dim=False, **config_hf_kwargs)
    except (ValueError, KeyError) as exc:
        if _key_requires_pipeline(data_key):
            raise
        log.warning(
            f"No pipeline found in config ({exc}). "
            f"Key '{data_key}' does not require beamforming — saving data as-is."
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        _run_passthrough(
            dataset_path, data_key, n_frames, save_dir, save_as, overwrite, **dataset_hf_kwargs
        )
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    dataloader = Dataloader(
        dataset_path,
        key=data_key,
        batch_size=None,
        shuffle=False,
        return_metadata=True,
        limit_n_frames=n_frames,
        n_frames=None,
        num_threads=num_threads,
        sort_files=True,
        dtype="float32",
        axis_selections=axis_selections,
        **dataset_hf_kwargs,
    )

    iterator = iter(dataloader)
    total_batches = len(dataloader)

    get_data = lambda: next(iterator)
    prepare_parameters = pipeline.prepare_parameters
    pipeline_call = pipeline.__call__

    if timings:
        timer = FunctionTimer()
        get_data = timer(get_data, name="dataloader")
        prepare_parameters = timer(prepare_parameters, name="prepare_parameters")
        pipeline_call = timer(pipeline_call, name="pipeline")

    _DEFAULT_FPS = 20

    prev_file_path = None
    data_output = []
    filestem = None
    parameters = None
    params = None
    fps = _DEFAULT_FPS

    def save_video_worker(
        video: np.ndarray,
        save_path: Path,
        parameters,
        fps: int,
    ):
        if output_blocked(save_path, overwrite):
            log.warning(f"File {save_path} already exists. Use --overwrite to replace it.")
            return
        if save_as in ["mp4", "gif"]:
            io_lib.save_video(video, save_path, fps=fps)
        elif save_as == "hdf5":
            scan_dict = parameters.to_scan_dict()
            probe_dict = parameters.to_probe_dict()
            File.create(
                save_path,
                data={"image": {"values": video}},
                scan=scan_dict or None,
                probe=probe_dict or None,
                overwrite=overwrite,
            )
        elif save_as == "nii.gz":
            assert sitk is not None, "SimpleITK must be installed to save as nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(video), str(save_path))
            log.info(f"Saved NIfTI to {log.yellow(save_path)}")

    pbar = ProgressBar(total_batches)

    @jit
    def to_8bit(image, dynamic_range):
        image = ops.nan_to_num(image, nan=dynamic_range[0])
        image = translate(image, dynamic_range, (0, 255))
        image = ops.clip(image, 0, 255)
        image = ops.cast(image, "uint8")
        return image

    with ThreadPoolExecutor(max_workers=1) as executor:
        save_future = None
        for i in range(total_batches + 1):
            if i < total_batches:
                frame, metadata = get_data()
                file_path = metadata["file"]["fullpath"]
            else:
                file_path = None  # sentinel to flush the last file

            if file_path != prev_file_path:
                if prev_file_path is not None:
                    video = ops.convert_to_numpy(data_output)
                    save_path = save_dir / f"{filestem}.{save_as}"
                    if save_future is not None:
                        save_future.result()
                    save_future = executor.submit(
                        save_video_worker, video, save_path, parameters, fps
                    )
                    data_output = []
                    if file_path is None:
                        break

                prev_file_path = file_path
                # Already validated by the Dataloader that handed us this path.
                with File(file_path, validate=False) as f:
                    filestem = f.stem
                    # Re-resolve rather than trust the peek: a file whose tracks are
                    # ordered differently would otherwise be read at the wrong index.
                    if _resolve_track(f, key, track)[0] != track_index:
                        raise ValueError(
                            f"Track {track!r} is not at index {track_index} in {file_path} as "
                            "it is in the first file; tracks must be ordered consistently."
                        )
                    parameters = _load_track_parameters(f, track_index)
                parameters.update(config_params)

                try:
                    fps = int(round(parameters.frames_per_second))
                except (ValueError, AttributeError):
                    fps = _DEFAULT_FPS

                params = prepare_parameters(parameters, **config_params)

            # Sentinel iteration (no more data — also covers an empty dataset
            # where total_batches == 0); nothing to process, so stop here.
            if file_path is None:
                break

            output = pipeline_call(data=frame, **params)  # ty: ignore[invalid-argument-type]
            processed_frame = output["data"]

            if not keep_dynamic_range:
                dr = getattr(parameters, "dynamic_range", None)
                dynamic_range = tuple(dr) if dr is not None else (-60, 0)
                processed_frame = to_8bit(processed_frame, dynamic_range)

            data_output.append(processed_frame)
            pbar.add(1)

            for keep_key in keep_keys:
                if keep_key in output:
                    params[keep_key] = output[keep_key]  # ty: ignore[invalid-assignment]

            if timings:
                for tname in timer.timings.keys():
                    timer.append_to_yaml(save_dir / f"timings_{tname}.yaml", tname)

        # Re-raise anything the last save hit. Every other save is awaited before
        # the next one is submitted, but the final one is not: leaving its result
        # unread swallows the exception, and a run that wrote nothing at all still
        # exits 0.
        if save_future is not None:
            save_future.result()

    if timings:
        timer.print()


@dataclass
class _StandaloneProcessArgs(ProcessArgs):
    """``ProcessArgs`` plus the ``--device`` flag that ``zea`` exposes globally."""

    device: Annotated[
        str,
        tyro.conf.arg(help=DEVICE_HELP),
    ] = DEFAULT_DEVICE


def main() -> None:
    """Entry point for ``python -m zea.data.process``, equivalent to ``zea process``."""
    args = tyro.cli(_StandaloneProcessArgs)
    init_device(args.device)
    args.run()


if __name__ == "__main__":
    main()
