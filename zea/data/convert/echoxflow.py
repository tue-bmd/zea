"""Convert the EchoXFlow dataset to the zea format.

`EchoXFlow <https://github.com/Ahus-AIM/EchoXFlow>`_ is a public echocardiography
dataset from Akershus University Hospital: 666 routine-care examinations with
35,377 recordings, stored in the scanner's native beamspace (polar) grid before
scan conversion. It covers 1-D, 2-D and 3-D B-mode, colour, tissue and spectral
Doppler, with a co-registered ECG and dense and sparse annotations.

This converter only takes the **single-plane 2-D B-mode** recordings and writes
each one to its own zea HDF5 file. Each file holds:

* the B-mode frames (``uint8``, still on the beamspace grid),
* the Cartesian ``(x, y, z)`` position of every pixel, from the sector geometry,
* the frame timestamps,
* the ECG, if the recording has one,
* the exam ID, stored as ``metadata/subject/id``. It groups recordings by
  examination, not by patient: one patient can have several exams, so do not
  use it for patient-disjoint splits. Use ``splits.csv`` from the original
  dataset for that.

.. admonition:: Limitations

   The result is a filtered subset of EchoXFlow, not the full dataset:

   * Other modalities (M-mode, multi-plane and 3-D B-mode, all Doppler) are skipped.
   * Recordings with fewer than ``--min-frames`` frames (default 10) or a frame rate
     of ``--min-fps`` or lower (default 30 Hz) are skipped. This drops
     probe-positioning sweeps and single-frame captures.
   * Annotations, most recording metadata, the patient-level splits
     (``splits.csv``) and the view labels are not converted. You can join them
     back from the original dataset using the exam and recording IDs in the
     output paths.

.. admonition:: License

   CC BY-NC-SA 4.0 - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

   Non-commercial use only. Converted data must be shared under the same
   license, with credit to the EchoXFlow authors.

.. citation:: stenhede2026echoxflow

.. rubric:: Links

* `Original dataset on Hugging Face <https://huggingface.co/datasets/Ahus-AIM/EchoXFlow>`_
* `EchoXFlow code <https://github.com/Ahus-AIM/EchoXFlow>`_
* `Converted dataset on Hugging Face <https://huggingface.co/datasets/zeahub/echoxflow>`_

.. rubric:: Usage

Download and extract EchoXFlow as described on its
`dataset page <https://huggingface.co/datasets/Ahus-AIM/EchoXFlow>`_. Then point
the converter at the data root, which is the folder containing ``croissant.json``.
The output is laid out as ``<dst>/data/<exam_id>/<recording_id>.hdf5``, next to
a dataset card (``README.md``) and the EchoXFlow ``LICENSE``.

.. code-block:: console

   zea convert echoxflow /data/EchoXFlow/data ./output
   zea convert echoxflow /data/EchoXFlow/data ./output --limit 2

.. note::

   Reading EchoXFlow needs the ``echoxflow`` package. It is not on PyPI, so
   install it from GitHub:

   .. code-block:: console

      pip install "echoxflow @ git+https://github.com/Ahus-AIM/EchoXFlow.git"

   The package declares Python >= 3.13, but the reader also works on 3.11 and
   3.12. On those versions, add ``--ignore-requires-python`` to the command.
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
    ``zea convert --help`` keeps working without it.
    """
    try:
        from echoxflow import (
            find_recordings,
            load_croissant,
            open_recording,
        )
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

    - subject.id  <- exam_id (groups by examination, *not* by patient; use the
      original ``splits.csv`` for patient-disjoint folds)
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


def make_dataset_card(
    repo_id: str,
    min_frames: int,
    min_fps: float,
    num_recordings: int,
    num_exams: int,
    size_gb: float,
) -> str:
    """Build a HuggingFace dataset card (``README.md``) for the converted dataset.

    The card credits the original EchoXFlow dataset, carries its CC BY-NC-SA 4.0
    license and lists what this subset leaves out. The counts, size and filter
    settings come from the conversion run, so the card matches the uploaded files.

    The ``zea_repo_id`` front-matter field is what the output-directory ownership
    checks key on, so it must match *repo_id*.
    """
    return f"""\
---
license: cc-by-nc-sa-4.0
zea_repo_id: {repo_id}
pretty_name: "EchoXFlow 2-D B-mode subset (zea format)"
tags:
  - ultrasound
  - echocardiography
  - cardiac
  - medical-imaging
  - beamspace
  - ecg
  - zea
size_categories:
  - 10K<n<100K
source_datasets:
  - Ahus-AIM/EchoXFlow
---

# EchoXFlow - 2-D B-mode subset (zea format)

> **This is a derived, partial copy of [EchoXFlow](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow)
> by Stenhede et al. (Akershus University Hospital).** All credit for the data goes to the original
> authors. Please cite their [paper](https://arxiv.org/abs/2605.05447) (see [Citation](#citation)) and
> use the [original dataset](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow) if you need the full data.

- **Paper:** [EchoXFlow: A Beamspace Echocardiography Dataset for Cardiac Motion, Flow, and Function](https://arxiv.org/abs/2605.05447) (NeurIPS 2026 Datasets and Benchmarks)
- **Original dataset:** [Ahus-AIM/EchoXFlow](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow)
- **Code:** [github.com/Ahus-AIM/EchoXFlow](https://github.com/Ahus-AIM/EchoXFlow)
- **License:** [CC BY-NC-SA 4.0](LICENSE), inherited from the original dataset

EchoXFlow contains 666 routine-care echocardiographic examinations (35,377 recordings) stored in the
scanner's native beamspace (polar) grid, with co-registered ECG, B-mode, Doppler and dense/sparse
annotations. This repository re-packages **only the single-plane 2-D B-mode recordings** into the
[zea](https://github.com/tue-bmd/zea) HDF5 format, so they can be loaded directly with zea.

## What is included

| | |
|---|---|
| **Recordings** | {num_recordings:,} single-plane 2-D B-mode recordings (of 26,356 in EchoXFlow) |
| **Examinations** | {num_exams:,} |
| **Selection** | frame rate > {min_fps:g} Hz and >= {min_frames} frames |
| **Per file** | B-mode frames (`uint8`, beamspace / polar grid, not scan-converted), per-pixel Cartesian coordinates, frame timestamps, ECG (when available), exam id |
| **Size** | ~{size_gb:.0f} GB |

```
data/
  <exam_id>/
    <recording_id>.hdf5
```

The exam id is stored as `metadata/subject/id`, but it identifies an **examination, not a patient**: one
patient can have several exams. Do not use it for patient-disjoint splits; use `splits.csv` from the
original dataset instead.

## Limitations and differences from the original

This is **not** the full EchoXFlow dataset. Compared to the original it is missing:

- All other modalities: M-mode, dual-plane, triplane and 3-D B-mode, colour/tissue Doppler, pulsed- and continuous-wave Doppler.
- Single-plane B-mode recordings with frame rate <= {min_fps:g} Hz or fewer than {min_frames} frames (e.g. probe-positioning or single-frame captures).
- All annotations (speckle-tracking contours, 3-D LV meshes, sample volumes, measurements, spectral traces, QRS triggers).
- Most recording metadata from the Zarr stores and `croissant.json` (e.g. `derived/` quantities, acquisition parameters beyond sector geometry, SHA-256 checksums).
- `splits.csv` (patient-level five-fold split) and `view_classifier_probabilities.csv` (view labels). Use the files from the
  [original repository](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow) with the `exam_id` / `recording_id` from the file paths.

The limitations of the original dataset also apply: single-centre, single-vendor data from a selected population. It is not
representative of all institutions, vendors, protocols or patient subgroups, and it is not validated for clinical diagnosis,
triage, treatment or prognosis.

## Usage

```python
import zea

path = "hf://{repo_id}/data/<exam_id>/<recording_id>.hdf5"
with zea.File(path) as f:
    image = f.tracks[0].data.image
    frames = image.values[:]       # (n_frames, n_depth, n_angles), uint8, beamspace
    coords = image.coordinates[:]  # (n_depth, n_angles, 3), Cartesian (x, y, z) in metres
    ecg = f.metadata.ecg.samples[:]
```

The files were created with the zea converter (requires the [`echoxflow`](https://github.com/Ahus-AIM/EchoXFlow) package):

```bash
zea convert echoxflow <echoxflow_data_root> <dst> --min-frames {min_frames} --min-fps {min_fps:g}
```

## License

[CC BY-NC-SA 4.0](LICENSE), the same license as the original EchoXFlow dataset. You may share and adapt the data for
**non-commercial** purposes, provided you give appropriate credit to the EchoXFlow authors, indicate changes, and
distribute derivatives under the same license. Changes made here: the subset selection and format conversion described above.

## Citation

If you use this data, cite the original EchoXFlow paper:

```bibtex
@inproceedings{{stenhede_2026_echoxflow,
  title = {{EchoXFlow: A Beamspace Echocardiography Dataset for Cardiac Motion, Flow, and Function}},
  author = {{Stenhede, Elias and Sulkowska, Joanna and Orstad, Eivind Bj{{\\o}}rkan and Schirmer, Henrik and Ranjbar, Arian}},
  booktitle = {{Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track}},
  year = {{2026}},
  eprint = {{2605.05447}},
  doi = {{10.48550/arXiv.2605.05447}},
  url = {{https://arxiv.org/abs/2605.05447}}
}}
```
"""  # noqa: E501


def _copy_license(src: str | Path, dest_root: Path) -> bool:
    """Copy the EchoXFlow ``LICENSE`` next to the converted data.

    The upstream download keeps ``LICENSE`` one level above the data root, so both
    *src* and its parent are checked.

    Returns:
        ``True`` if the license was copied, ``False`` if it was not found.
    """
    import shutil

    for candidate in (Path(src) / "LICENSE", Path(src).parent / "LICENSE"):
        if candidate.exists():
            shutil.copy2(candidate, dest_root / "LICENSE")
            return True
    log.warning(
        "EchoXFlow LICENSE not found in the data root or its parent; add it to "
        f"{dest_root} before uploading (CC BY-NC-SA 4.0 requires it)."
    )
    return False


def convert_echoxflow(args):
    """Convert EchoXFlow ``2d_brightness_mode`` recordings into zea HDF5 files.

    Each qualifying recording is written to ``<dst>/data/<exam_id>/<recording_id>.hdf5``.
    A dataset card (``README.md``) and the EchoXFlow ``LICENSE`` are written to ``<dst>``.

    Usage::

        zea convert echoxflow <src> <dst>
        zea convert echoxflow <src> <dst> --limit 2

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

    # Older versions wrote <dst>/<exam_id>/*.hdf5. Uploading the whole folder would
    # then publish both layouts, so refuse to mix them instead of migrating.
    data_dir = dest_root / "data"
    stray = [p for p in dest_root.rglob("*.hdf5") if data_dir not in p.parents]
    if stray:
        raise RuntimeError(
            f"Found {len(stray)} HDF5 files outside {data_dir} (e.g. {stray[0]}), from an "
            "older converter layout. Move them under data/ or use a fresh output folder."
        )

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
        out_path = dest_root / "data" / record.exam_id / f"{record.recording_id}.hdf5"
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

    hdf5_files = sorted(data_dir.rglob("*.hdf5"))
    card = make_dataset_card(
        hf_repo_id,
        min_frames=args.min_frames,
        min_fps=args.min_fps,
        num_recordings=len(hdf5_files),
        num_exams=len({f.parent.name for f in hdf5_files}),
        size_gb=sum(f.stat().st_size for f in hdf5_files) / 1e9,
    )
    write_dataset_card(dest_root, card)
    has_license = _copy_license(data_root, dest_root) or (dest_root / "LICENSE").exists()

    if getattr(args, "upload", False):
        assert args.revision, "revision must be provided when --upload is set."
        if not has_license:
            raise FileNotFoundError(
                f"Refusing to upload without the EchoXFlow LICENSE in {dest_root}; "
                "CC BY-NC-SA 4.0 requires it to accompany the data."
            )
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
