"""CLI for beamforming a zea dataset with a pipeline defined in a YAML config file.

Usage:
    python -m zea.data.process --dataset <path> --config <config.yaml>
"""

import re
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import keras
import numpy as np
import tyro
from keras import ops

from zea import io_lib, log
from zea.backend import jit
from zea.cli_args import SUPPORTED_FORMATS, ProcessArgs
from zea.config import Config
from zea.data.dataloader import Dataloader
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.func import translate
from zea.internal.checks import _NON_IMAGE_DATA_TYPES
from zea.ops.pipeline import Pipeline
from zea.utils import FunctionTimer


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
    normalized = re.sub(r"^tracks/track_\d+/", "", normalized)
    normalized = normalized.removeprefix("data/").removesuffix("/values")
    return normalized in _NON_IMAGE_DATA_TYPES


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

    with Dataset(dataset_path, validate=False, lazy=True, _suggest_lazy=False, **hf_kwargs) as ds:
        pbar = keras.utils.Progbar(len(ds))
        for i in range(len(ds)):
            f = ds[i]  # lazy download for hf:// paths; returns cached File handle
            data_key = f.format_key(key)
            arr = np.asarray(f[data_key][:n_frames] if n_frames is not None else f[data_key][:])
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
            if save_path.exists() and not overwrite:
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
    try:
        pipeline = Pipeline.from_path(config_path, with_batch_dim=False, **config_hf_kwargs)
    except (ValueError, KeyError) as exc:
        if _key_requires_pipeline(key):
            raise
        log.warning(
            f"No pipeline found in config ({exc}). "
            f"Key '{key}' does not require beamforming — saving data as-is."
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        _run_passthrough(
            dataset_path, key, n_frames, save_dir, save_as, overwrite, **dataset_hf_kwargs
        )
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Peek at the first file to get selected_transmits so the dataloader can
    # pre-filter the transmit axis at HDF5 read time (avoids reading unused data).
    axis_selections: dict | None = None
    if _key_requires_pipeline(key):
        try:
            with Dataset(dataset_path, validate=False, **dataset_hf_kwargs) as _peek_ds:
                _first_path = _peek_ds.file_paths[0] if _peek_ds.file_paths else None
            if _first_path:
                with File(_first_path) as _peek_f:
                    _peek_params = _peek_f.load_parameters()
                _peek_params.update(config_params)
                axis_selections = _axis_selections_from_params(_peek_params)
        except Exception:
            pass  # fall back to runtime slicing if peek fails

    dataloader = Dataloader(
        dataset_path,
        key=key,
        batch_size=None,
        shuffle=False,
        return_filename=True,
        limit_n_frames=n_frames,
        n_frames=1,
        num_threads=num_threads,
        insert_frame_axis=False,
        sort_files=True,
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
        if save_path.exists() and not overwrite:
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

    pbar = keras.utils.Progbar(total_batches)

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
                file_path = metadata["fullpath"]
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
                with File(file_path) as f:
                    filestem = f.stem
                    parameters = f.load_parameters()
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

    if timings:
        timer.print()


def main() -> None:
    args = tyro.cli(ProcessArgs)
    run_processing(
        args.dataset,
        args.config,
        args.key,
        args.n_frames,
        args.save_dir,
        args.save_as,
        args.keep_keys,
        args.timings,
        args.num_threads,
        args.overwrite,
        args.keep_dynamic_range,
        args.revision,
        args.config_revision,
    )


if __name__ == "__main__":
    main()
