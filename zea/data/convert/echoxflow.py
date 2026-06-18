"""Convert EchoXFlow ``2d_brightness_mode`` recordings to the zea HDF5 format.

Finds all recordings with at least ``--min-frames`` B-mode frames and a frame rate
above ``--min-fps``, then writes each as a zea-format HDF5 file under ``dst``,
laid out as ``<dst>/<exam_id>/<recording_id>.hdf5``.

Usage::

    python -m zea.data.convert echoxflow <src> <dst>
    python -m zea.data.convert echoxflow <src> <dst> --limit 2

``src`` is the EchoXFlow data root (e.g. ``/data/EchoXFlow/data``) containing a
``croissant.json`` catalog. Reading EchoXFlow recordings requires the optional
``echoxflow`` package (https://github.com/Ahus-AIM/EchoXFlow), which is not
published on PyPI; install it directly from GitHub::

    pip install "echoxflow @ git+https://github.com/Ahus-AIM/EchoXFlow.git"

EchoXFlow declares ``requires-python = ">=3.13"``, but the reader also runs on
Python 3.11/3.12 (only pip's metadata check blocks it); there, append
``--ignore-requires-python`` to the command above.
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import log
from zea.beamform.pixelgrid import polar_pixel_grid
from zea.data.convert.utils import (
    check_output_dir_ownership,
    require_output_dir_ownership,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import File

MODALITY = "2d_brightness_mode"
_ECHOXFLOW_HF_REPO_ID = "zeahub/echoxflow"
_ECHOXFLOW_GIT = "git+https://github.com/Ahus-AIM/EchoXFlow.git"


def _import_echoxflow():
    """Import the optional third-party ``echoxflow`` package.

    Returns the ``(find_recordings, load_croissant, open_recording)`` callables.
    Raises a clear :class:`ImportError` if the package is not installed, so that
    ``python -m zea.data.convert --help`` keeps working without it.
    """
    try:
        from echoxflow import find_recordings, load_croissant, open_recording
    except ImportError as exc:
        command = f'pip install "echoxflow @ {_ECHOXFLOW_GIT}"'
        if sys.version_info < (3, 13):
            # EchoXFlow pins requires-python to >=3.13, but the reader runs fine on
            # 3.11/3.12 (zarr>=3.1 needs >=3.11); only pip's metadata check blocks it.
            command += " --ignore-requires-python"
        raise ImportError(
            "Converting EchoXFlow recordings requires the `echoxflow` package, "
            f"which is not published on PyPI. Install it from GitHub with:\n    {command}"
        ) from exc
    return find_recordings, load_croissant, open_recording


def sector_coordinates(geometry, height: int, width: int) -> np.ndarray | None:
    """Build a zea per-pixel Cartesian coordinate grid from EchoXFlow SectorGeometry.

    EchoXFlow B-mode frames are sampled on a polar (sector) grid whose rows span the
    depth (radial) axis and whose columns span the steering angle (polar) axis. zea's
    ``polar_pixel_grid`` maps that to Cartesian ``(x, y, z)`` positions in metres.

    Returns an array of shape ``(height, width, 3)`` (broadcast across frames), or
    ``None`` if the recording has no geometry.
    """
    if geometry is None:
        return None
    grid = polar_pixel_grid(
        polar_limits=(geometry.angle_start_rad, geometry.angle_end_rad),
        zlims=(geometry.depth_start_m, geometry.depth_end_m),
        num_radial_pixels=height,
        num_polar_pixels=width,
    )
    return np.asarray(grid, dtype=np.float32)


def build_metadata(record, bmode_stream, store) -> dict:
    """Collect EchoXFlow metadata that maps onto zea's MetadataSpec.

    - subject.id  <- exam_id (enables subject-wise splits)
    - ecg         <- the recording's ECG signal, if present (Signal1D)
    """
    metadata: dict = {"subject": {"id": record.exam_id, "type": "human"}}

    if record.has_array_path("data/ecg"):
        try:
            ecg = store.load_stream("ecg")
            samples = np.asarray(ecg.data, dtype=np.float32).reshape(-1)
            ecg_meta: dict = {"samples": samples}
            if ecg.sample_rate_hz is not None:
                start = float(ecg.timestamps[0]) if ecg.timestamps is not None else 0.0
                ecg_meta["sampling_frequency"] = np.float32(ecg.sample_rate_hz)
                ecg_meta["start_time_offset"] = np.float32(start)
            elif ecg.timestamps is not None and len(ecg.timestamps) >= 1:
                ts_ecg = np.asarray(ecg.timestamps, dtype=np.float32)
                ecg_meta["timestamps"] = ts_ecg - ts_ecg[0]
                ecg_meta["start_time_offset"] = np.float32(ts_ecg[0])
            if "sampling_frequency" in ecg_meta or "timestamps" in ecg_meta:
                metadata["ecg"] = ecg_meta
        except Exception:  # noqa: BLE001 - ECG is optional; skip if it won't load
            pass

    return metadata


def make_dataset_card(repo_id: str) -> str:
    """Build a HuggingFace dataset card (``README.md``) for the converted dataset.

    The ``zea_repo_id`` front-matter field is what the output-directory ownership
    checks key on, so it must match *repo_id*.
    """
    return f"""\
---
license: other
zea_repo_id: {repo_id}
task_categories:
  - image-classification
tags:
  - ultrasound
  - echocardiography
  - 2d
  - cardiac
  - medical
pretty_name: "EchoXFlow 2-D B-mode (zea format)"
---

# EchoXFlow - 2-D B-mode Ultrasound Dataset

This dataset is a **zea-format** (HDF5) conversion of the EchoXFlow
``{MODALITY}`` recordings, hosted at
[{repo_id}](https://huggingface.co/datasets/{repo_id}).

## Conversion

This dataset was converted to zea format and uploaded using the
[zea](https://github.com/tue-bmd/zea) data converter:

```bash
python -m zea.data.convert echoxflow <src> <dst>
```

## Dataset structure

```
<exam_id>/
  <recording_id>.hdf5
  ...
```

Each HDF5 file follows the [zea data format](https://github.com/tue-bmd/zea).
"""


def convert_echoxflow(args):
    """Convert EchoXFlow ``2d_brightness_mode`` recordings into zea HDF5 files.

    Each qualifying recording is written to ``<dst>/<exam_id>/<recording_id>.hdf5``.

    Usage::

        python -m zea.data.convert echoxflow <src> <dst>
        python -m zea.data.convert echoxflow <src> <dst> --limit 2

    Args:
        args (argparse.Namespace): An object with attributes:

            - src (str | Path): EchoXFlow data root containing ``croissant.json``.
            - dst (str | Path): Destination directory for zea HDF5 files.
            - croissant (str | None): Path to ``croissant.json``
              (default: ``<src>/croissant.json``).
            - min_frames (int): Minimum B-mode frame count.
            - min_fps (float): Minimum frame rate (Hz).
            - limit (int | None): Convert at most N recordings.
            - overwrite (bool): Overwrite existing output files.
            - upload (bool): Upload the converted dataset to HuggingFace Hub.
            - revision (str | None): Target branch on the Hub. Required when
              ``upload`` is set; upload to ``main`` is blocked.
            - hf_repo_id (str): HuggingFace repo id for ownership checks and
              optional upload. Defaults to ``zeahub/echoxflow`` when empty.
    """
    find_recordings, load_croissant, open_recording = _import_echoxflow()

    data_root = args.src
    croissant_path = args.croissant or f"{data_root}/croissant.json"
    dest_root = Path(args.dst)
    hf_repo_id = getattr(args, "hf_repo_id", "") or _ECHOXFLOW_HF_REPO_ID

    check_output_dir_ownership(dest_root, hf_repo_id)

    catalog = load_croissant(croissant_path)
    log.info(f"Catalog loaded: {len(catalog.recordings)} recordings")

    records = find_recordings(
        croissant=catalog,
        array_paths=(MODALITY,),
        require_all=True,
        min_frame_counts={MODALITY: args.min_frames},
        predicate=lambda r: (r.sample_rate_hz(MODALITY) or 0) > args.min_fps,
    )
    log.info(
        f"Found {len(records)} recordings with >={args.min_frames} frames and >{args.min_fps} fps"
    )

    if args.limit is not None:
        records = records[: args.limit]
        log.info(f"Limiting to first {len(records)} recordings")

    converted = skipped = failed = 0
    for record in tqdm(records, desc="Converting", unit="rec"):
        out_path = dest_root / record.exam_id / f"{record.recording_id}.hdf5"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            store = open_recording(record, root=data_root)
            stream = store.load_stream(MODALITY)
            frames = np.asarray(stream.data)  # (n_frames, H, W), uint8
            if frames.dtype != np.uint8:
                frames = frames.astype(np.uint8)

            height, width = int(frames.shape[1]), int(frames.shape[2])
            image: dict = {"values": frames, "unit": "-"}
            coords = sector_coordinates(stream.metadata.geometry, height, width)
            if coords is not None:
                image["coordinates"] = coords  # (H, W, 3), broadcast across frames
            if stream.timestamps is not None:
                ts = np.asarray(stream.timestamps, dtype=np.float32).reshape(-1)
                image["timestamps"] = ts - ts[0]
                image["start_time_offset"] = ts[0]

            metadata = build_metadata(record, stream, store)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            File.create(
                path=out_path,
                data={"image": image},
                metadata=metadata,
                probe={"name": "generic"},
                description=f"EchoXFlow {MODALITY} {record.recording_id} converted to zea format",
                overwrite=args.overwrite,
            )
            converted += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch going on a bad recording
            failed += 1
            log.warning(f"FAILED {record.exam_id}/{record.recording_id}: {exc}")

    log.info(f"Done. converted={converted} skipped={skipped} failed={failed}")

    # Skipped files already exist on disk from a previous run, so they still
    # count towards a complete dataset that can be carded and uploaded.
    if converted == 0 and skipped == 0:
        if getattr(args, "upload", False):
            log.error("No converted files exist; skipping upload.")
        return

    write_dataset_card(dest_root, make_dataset_card(hf_repo_id))

    if getattr(args, "upload", False):
        assert args.revision, "revision must be provided when --upload is set."
        upload_echoxflow(dest_root, revision=args.revision, repo_id=hf_repo_id)


def upload_echoxflow(
    output_folder: str | Path, revision: str, repo_id: str
) -> None:  # pragma: no cover
    """Upload a converted EchoXFlow dataset to a HuggingFace Hub revision branch.

    Only for zea maintainers with push access to the repository. Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_folder: Directory containing the converted HDF5 files.
        revision: Target branch name on the Hub (must not be ``"main"``).
        repo_id: Target HuggingFace repository ID.
    """
    require_output_dir_ownership(output_folder, repo_id)
    upload_dataset_to_hf(
        folder=output_folder,
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Upload EchoXFlow dataset (zea format) to {revision}",
    )
