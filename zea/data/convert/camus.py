"""Convert the CAMUS dataset to the zea format.

.. note::

   Requires SimpleITK: ``pip install SimpleITK``.

CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) is a
public dataset containing 2-D echocardiographic sequences from 500 patients.
Sequences are stored in NIfTI (``.nii.gz``) format and include both 2-chamber
(2CH) and 4-chamber (4CH) apical views.

Dataset splits:

* **Train** - patients 1-400
* **Validation** - patients 401-450
* **Test** - patients 451-500

.. admonition:: License

   CC BY-NC-SA 4.0 - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

   The CAMUS dataset is available free of charge strictly for non-commercial
   scientific research purposes only.

.. admonition:: Reference

   S\\. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, F. Cervenansky, F. Espinosa,
   T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier, C. Lartizien,
   J. D'hooge, L. Lovstakken and O. Bernard.
   *Deep Learning for Segmentation Using an Open Large-Scale Dataset in
   2D Echocardiography.*
   IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, 2019.
   `DOI: 10.1109/TMI.2019.2900516 <https://doi.org/10.1109/TMI.2019.2900516>`_

.. rubric:: Links

* `Original dataset <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8>`_
* `Dataset on Hugging Face <https://huggingface.co/datasets/zeahub/camus>`_

.. rubric:: Usage


.. code-block:: console

   python -m zea.data.convert camus ./raw ./output --download

For testing purposes, you can also convert a reduced dataset containing only 6 half-sequence files:

.. code-block:: console

    python -m zea.data.convert camus ./raw ./output --download --reduced-dataset

"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import log
from zea.beamform.pixelgrid import polar_pixel_grid
from zea.data.convert.utils import (
    check_output_dir_ownership,
    download_from_girder,
    require_output_dir_ownership,
    sitk_load,
    unzip,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import File
from zea.func.tensor import translate

# Girder collection ID for the CAMUS dataset
_CAMUS_COLLECTION_ID = "6373703d73e9f0047faa1bc8"

# Segmentation label names — index 0 is the explicit 'unannotated' label.
# Frames that were not annotated will have only channel 0 set to True.
CAMUS_SEG_LABELS = np.array(["unannotated", "LV_endo", "LV_myo", "LA"], dtype=np.str_)

# ---------------------------------------------------------------------------
# Citation / license constants
# ---------------------------------------------------------------------------

CAMUS_CITATION = (
    "S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, F. Cervenansky, F. Espinosa, "
    "T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier, C. Lartizien, "
    "J. D'hooge, L. Lovstakken and O. Bernard. "
    '"Deep Learning for Segmentation Using an Open Large-Scale Dataset in '
    '2D Echocardiography." '
    "IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, 2019. "
    "https://doi.org/10.1109/TMI.2019.2900516"
)

CAMUS_LICENSE = "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)"

CAMUS_DESCRIPTION = (
    "CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) "
    "2D echocardiographic dataset converted to zea format. "
    f"License: {CAMUS_LICENSE}. "
    f"Citation: {CAMUS_CITATION}"
)

# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------


# Default HF repo for full dataset
_CAMUS_HF_REPO_ID = "zeahub/camus"
# HF repo for reduced/sample dataset
_CAMUS_SAMPLE_HF_REPO_ID = "zeahub/camus-sample"

# Hardcoded list of sample files for --reduced-dataset
_CAMUS_SAMPLE_FILES = [
    "train/patient0101/patient0101_2CH_half_sequence.hdf5",
    "train/patient0101/patient0101_4CH_half_sequence.hdf5",
    "val/patient0401/patient0401_2CH_half_sequence.hdf5",
    "val/patient0401/patient0401_4CH_half_sequence.hdf5",
    "test/patient0451/patient0451_2CH_half_sequence.hdf5",
    "test/patient0451/patient0451_4CH_half_sequence.hdf5",
]


def _parse_cfg(cfg_path: Path) -> dict:
    """Parse a CAMUS ``Info_*.cfg`` file into a plain dict.

    Each line has the form ``Key: value``.  Lines that cannot be parsed are
    silently ignored.

    Args:
        cfg_path: Path to the cfg file.

    Returns:
        Dictionary mapping field names to their raw string values.
    """
    result = {}
    for line in cfg_path.read_text().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def process_camus(source_path, output_path, overwrite=False):
    """Convert one CAMUS NIfTI half-sequence into the zea HDF5 format.

    Stores the scan-converted B-mode sequence (``data/image``),
    per-pixel Cartesian coordinates derived from the NIfTI voxel spacing,
    the full segmentation sequence (``data/segmentation``) with an explicit
    ``"unannotated"`` label channel for frames that lack manual annotations,
    and rich clinical metadata parsed from the accompanying ``Info_*.cfg``
    file.

    Args:
        source_path (str, pathlike): Path to a ``*_half_sequence.nii.gz`` file.
        output_path (str, pathlike): Destination HDF5 file path.
        overwrite (bool, optional): Overwrite existing output file.
            Defaults to False.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            log.warning("Output file %s already exists. Skipping.", log.yellow(output_path))
            return

    # ---- derive patient / view from filename --------------------------------
    # source_path.name  e.g.  patient0001_2CH_half_sequence.nii.gz
    stem = source_path.name.removesuffix(".nii.gz")  # patient0001_2CH_half_sequence
    parts = stem.split("_")  # [patient0001, 2CH, half, sequence]
    patient_name = parts[0]  # patient0001
    view = parts[1]  # 2CH | 4CH
    patient_dir = source_path.parent

    # ---- parse clinical metadata -------------------------------------------
    cfg = _parse_cfg(patient_dir / f"Info_{view}.cfg")
    # ED / ES are 1-indexed in the cfg file
    ed_idx = int(cfg["ED"]) - 1
    es_idx = int(cfg["ES"]) - 1
    n_frames = int(cfg["NbFrame"])
    sex = cfg.get("Sex", "").lower()  # "f" | "m"
    age = int(cfg.get("Age", 0))
    image_quality = cfg.get("ImageQuality", "")
    ef = cfg.get("EF", "")
    frame_rate = cfg.get("FrameRate", "")

    # ---- load image sequence ------------------------------------------------
    image_seq, meta = sitk_load(source_path)  # (n_frames, H, W), uint8
    image_seq = translate(
        image_seq.astype(np.float32), (0, 255), (-60, 0)
    )  # convert to dB, float32

    # ---- build pixel coordinates -------------------------------------------
    # sitk GetSpacing() order: (x_lateral, y_depth, z_frame) in mm
    spacing = meta["spacing"]  # (lateral_mm, depth_mm, 1.0)
    x_step = float(spacing[0]) / 1000  # metres per column
    z_step = float(spacing[1]) / 1000  # metres per row
    H, W = image_seq.shape[1], image_seq.shape[2]
    # x=0 at apex (centre column), z=0 at transducer surface — matches polar_pixel_grid convention
    cols = (np.arange(W, dtype=np.float32) - W / 2) * x_step
    rows = np.arange(H, dtype=np.float32) * z_step
    xx, zz = np.meshgrid(cols, rows)  # each (H, W)
    coordinates = np.stack([xx, np.zeros_like(xx), zz], axis=-1).astype(
        np.float32
    )  # (H, W, 3): [x_lateral, y=0, z_depth]

    # ---- polar image --------------------------------------------------------
    # coordinates are frame-agnostic so we grab them from the last iteration
    polar_values, polar_coords = _build_polar_image(image_seq[0], x_step, z_step, H, W)

    # ---- load segmentation --------------------------------------------------
    gt_path = patient_dir / f"{patient_name}_{view}_half_sequence_gt.nii.gz"
    gt_seq, _ = sitk_load(gt_path)  # (n_frames, H, W), uint8; labels 0-3

    # Build multi-label bool array with 4 channels:
    #   0 = unannotated  (True for frames without manual labels)
    #   1 = LV_endo      (label value 1 in the GT)
    #   2 = LV_myo       (label value 2)
    #   3 = LA           (label value 3)
    seg_values = np.zeros((n_frames, H, W, 4), dtype=np.bool_)
    annotated = np.zeros(n_frames, dtype=np.bool_)
    annotated[ed_idx] = True
    annotated[es_idx] = True

    seg_values[~annotated, :, :, 0] = True  # unannotated channel
    for label_idx, gt_val in enumerate([1, 2, 3], start=1):
        seg_values[annotated, :, :, label_idx] = gt_seq[annotated] == gt_val

    # ---- frame-level labels -------------------------------------------------
    frame_labels = np.array([""] * n_frames, dtype="<U2")
    frame_labels[ed_idx] = "ED"
    frame_labels[es_idx] = "ES"

    # ---- write HDF5 ---------------------------------------------------------
    text_report = f"EF: {ef}%  FrameRate: {frame_rate} fps  ImageQuality: {image_quality}"

    # ---- build full polar sequence by resampling each frame ----------------
    polar_seq = np.stack(
        [polar_values]
        + [_build_polar_image(image_seq[i], x_step, z_step, H, W)[0] for i in range(1, n_frames)],
        axis=0,
    )  # (n_frames, n_r, n_theta)

    File.create(
        path=output_path,
        data={
            "image": {"values": image_seq, "coordinates": coordinates},
            "image_polar": {"values": polar_seq, "coordinates": polar_coords},
            "segmentation": {
                "values": seg_values,
                "labels": CAMUS_SEG_LABELS,
                "coordinates": coordinates,
            },
        },
        probe={"name": "GE M5S"},
        metadata={
            "subject": {
                "id": patient_name,
                "type": "human",
                "sex": sex,
                "age": np.uint8(min(age, 255)),
            },
            "credit": CAMUS_CITATION,
            "text_report": text_report,
            "annotations": {
                "view": np.array([view] * n_frames, dtype=np.str_),
                "label": frame_labels,
                "image_quality": image_quality,
            },
        },
        description=CAMUS_DESCRIPTION,
    )


def _build_polar_image(
    scan_converted: np.ndarray,
    x_step: float,
    z_step: float,
    n_r: int,
    n_theta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one scan-converted frame onto a polar (depth × angle) grid.

    Uses :func:`~zea.beamform.pixelgrid.polar_pixel_grid` to build the sampling
    grid, then maps back to pixel coordinates in the scan-converted image and
    interpolates with ``scipy.ndimage.map_coordinates``.

    The transducer apex is assumed to be at (x=0, z=0) — i.e. the top-centre of
    the scan-converted image, consistent with the x-centered Cartesian coordinates
    stored in ``image``.  The sector half-angle and radius are inferred from the
    widest non-background row of the image rather than from the image dimensions,
    because the CAMUS scan-converted images are wider than the sector fan (the
    corners are background padding).

    Args:
        scan_converted: Scan-converted frame, shape ``(H, W)``, float32 dB.
        x_step: Lateral pixel spacing in metres.
        z_step: Axial pixel spacing in metres.
        n_r: Number of radial (depth) samples in the output.
        n_theta: Number of angular samples in the output.

    Returns:
        Tuple of:
            - ``polar_values``: ``(n_r, n_theta)`` float32, polar-resampled image.
            - ``polar_coords``: ``(n_r, n_theta, 3)`` float32, Cartesian [x, 0, z]
              positions in metres for each polar pixel (x=0 at apex centre).
    """
    from scipy.ndimage import map_coordinates

    H, W = scan_converted.shape

    # Detect the actual sector half-angle and radius from the image content.
    # The scan-converted image is wider than the fan; the image corners are
    # background padding.  The widest non-background row sits at the arc boundary
    # of the sector (r = R_max), giving the most accurate theta_max estimate.
    bg_val = float(scan_converted.min())
    fg = scan_converted > bg_val + 0.5
    row_widths = fg.sum(axis=1)
    widest_row = int(np.argmax(row_widths))
    fg_cols = np.where(fg[widest_row])[0]
    if fg_cols.size >= 2:
        x_half_m = ((fg_cols[-1] - fg_cols[0]) / 2) * x_step
        z_at_widest = widest_row * z_step
        theta_max = float(np.arctan2(x_half_m, z_at_widest))
        r_max = float(np.sqrt(x_half_m**2 + z_at_widest**2))
    else:
        x_half_m = (W / 2) * x_step
        r_max = H * z_step
        theta_max = float(np.arctan2(x_half_m, r_max))

    # polar_pixel_grid returns (n_r, n_theta, 3) Cartesian [x, y, z] with x=0 at apex
    polar_coords = polar_pixel_grid(
        polar_limits=(-theta_max, theta_max),
        zlims=(0.0, r_max),
        num_radial_pixels=n_r,
        num_polar_pixels=n_theta,
    ).astype(np.float32)  # (n_r, n_theta, 3)

    x_polar = polar_coords[:, :, 0]  # (n_r, n_theta), x=0 at apex centre
    z_polar = polar_coords[:, :, 2]  # (n_r, n_theta)

    # Map Cartesian coords back to pixel positions (col = (x + W/2*x_step)/x_step, row = z/z_step)
    col_coords = (x_polar + (W / 2) * x_step) / x_step
    row_coords = z_polar / z_step

    polar_values = map_coordinates(
        scan_converted,
        [row_coords, col_coords],
        order=1,
        mode="constant",
        cval=float(scan_converted.min()),
    ).astype(np.float32)

    return polar_values, polar_coords


splits = {"train": [1, 401], "val": [401, 451], "test": [451, 501]}


def get_split(patient_id: int) -> str:
    """Determine which dataset split a patient ID belongs to.

    Args:
        patient_id: Integer ID of the patient.

    Returns:
        The split name: "train", "val", or "test".

    Raises:
        ValueError: If the patient_id does not fall into any defined split range.
    """
    if splits["train"][0] <= patient_id < splits["train"][1]:
        return "train"
    elif splits["val"][0] <= patient_id < splits["val"][1]:
        return "val"
    elif splits["test"][0] <= patient_id < splits["test"][1]:
        return "test"
    else:
        raise ValueError(f"Did not find split for patient: {patient_id}")


def _process_task(task):
    """Unpack a task tuple and invoke process_camus in a worker process.

    Creates parent directories for the target outputs, calls process_camus
    with the unpacked paths, and logs then re-raises any exception raised by processing.

    Args:
        task (tuple): (source_file_str, output_file_str)

            - source_file_str: filesystem path to the source CAMUS file as a string.
            - output_file_str: filesystem path for the ZEA output file as a string.
    """
    source_file_str, output_file_str = task
    source_file = Path(source_file_str)
    output_file = Path(output_file_str)

    # Ensure destination directories exist (safe to call from multiple processes)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Call the real processing function (must be importable in the worker)
    # If process_camus lives in another module, import it there instead.
    try:
        process_camus(source_file, output_file, overwrite=False)
    except Exception:
        log.error("Error processing %s", log.yellow(source_file))
        raise


def download_camus(  # pragma: no cover
    destination: str | Path, patients: list[int] | None = None
) -> Path:
    """Download the CAMUS dataset from the Girder server.

    Downloads NIfTI files for each patient.

    Args:
        destination: Directory where the dataset will be downloaded.
        patients: List of patient IDs to download (1-500).
            If None, all patients are downloaded.

    Returns:
        Path to the downloaded dataset directory.
    """
    return download_from_girder(
        collection_id=_CAMUS_COLLECTION_ID,
        destination=destination,
        dataset_name="CAMUS",
        patients=patients,
        top_folder_name="database_nifti",
    )


def _resolve_path(src: str | Path) -> Path:
    src = Path(src)

    zip_name = "CAMUS_public.zip"
    folder_name = "CAMUS_public"
    unzip_dir = src / folder_name

    if (src / folder_name).exists():
        return unzip_dir

    # Girder download produces a database_nifti sub-folder,
    # or the user may have extracted patient* folders directly into src.
    if (src / "database_nifti").exists():
        log.info(f"Found database_nifti folder in {src}.")
        return src / "database_nifti"
    if any(src.glob("patient*")):
        log.info(f"Found patient folders directly in {src}.")
        return src

    zip_path = src / zip_name

    unzipped_path = unzip(zip_path, src)
    return unzipped_path / "database_nifti"


def convert_camus(args):
    """Convert the CAMUS dataset into zea HDF5 files across dataset splits.

    Processes files found under the CAMUS source folder (after unzipping or
    downloading if needed), assigns each patient to a train/val/test split,
    creates matching output paths, and executes per-file conversion tasks
    either serially or in parallel.

    Usage::

        python -m zea.data.convert camus <source_folder> <destination_folder>
        python -m zea.data.convert camus <source_folder> <destination_folder> --download

    Args:
        args (argparse.Namespace): An object with attributes:

            - src (str | Path): Path to the CAMUS archive or extracted folder,
              or a directory to download into when ``--download`` is set.
            - dst (str | Path): Root destination folder for ZEA HDF5 outputs;
              split subfolders will be created.
            - download (bool, optional): If True, download the dataset first from the
              Girder server.
            - no_hyperthreading (bool, optional): If True, run tasks serially instead
              of using a process pool.
    """

    camus_source_folder = Path(args.src)
    camus_output_folder = Path(args.dst)

    # Use sample repo if reduced-dataset flag is set
    is_reduced = getattr(args, "reduced_dataset", False)
    hf_repo_id = _CAMUS_SAMPLE_HF_REPO_ID if is_reduced else _CAMUS_HF_REPO_ID

    check_output_dir_ownership(camus_output_folder, hf_repo_id)

    # Optionally download the dataset
    if getattr(args, "download", False):
        camus_source_folder = download_camus(camus_source_folder)
    elif not camus_source_folder.exists():
        raise FileNotFoundError(
            f"Source folder does not exist: {camus_source_folder}. "
            "Use --download to download the CAMUS dataset automatically."
        )
    else:
        # Look for either CAMUS_public.zip or folders database_nifti, database_split
        camus_source_folder = _resolve_path(camus_source_folder)

    # check if output folders already exist
    for split in splits:
        split_dir = camus_output_folder / split
        if split_dir.exists():
            log.warning(
                "Output folder %s already exists. Existing files will be skipped.",
                log.yellow(split_dir),
            )

    # clone folder structure of source to output using pathlib

    tasks = []
    files = []
    if is_reduced:
        # Only process the hardcoded sample files
        for rel_path in _CAMUS_SAMPLE_FILES:
            split, patient, fname = rel_path.split("/")
            # Raw CAMUS source has no split subdirectory — patient folders sit
            # directly under camus_source_folder (e.g. raw-camus/patient0101/).
            nii_fname = fname.replace(".hdf5", ".nii.gz")
            source_file = camus_source_folder / patient / nii_fname
            output_file = camus_output_folder / split / patient / fname
            output_file.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(source_file), str(output_file)))
            files.append(source_file)
    else:
        files = sorted(camus_source_folder.glob("**/*_half_sequence.nii.gz"))
        for source_file in files:
            patient = source_file.name.removesuffix(".nii.gz").split("_")[0]
            patient_id = int(patient.removeprefix("patient"))
            split = get_split(patient_id)
            output_file = camus_output_folder / split / source_file.relative_to(camus_source_folder)
            output_file = output_file.with_suffix("").with_suffix(".hdf5")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(source_file), str(output_file)))
    if not tasks:
        log.info("No files found to process.")
        return

    if getattr(args, "no_hyperthreading", False):
        log.info("no_hyperthreading is True — running tasks serially (no ProcessPoolExecutor)")
        for t in tqdm(tasks, desc="Processing files (serial)"):
            try:
                _process_task(t)
            except Exception as e:
                log.error("Task processing failed: %s", e)
        log.info(
            "Conversion complete. %d files written to %s",
            len(tasks),
            log.yellow(camus_output_folder),
        )

        _copy_license(files, camus_output_folder)
        write_dataset_card(camus_output_folder, _CAMUS_DATASET_CARD)

        if getattr(args, "upload", False):
            upload_camus(camus_output_folder, revision=args.revision)
        return

    # Submit tasks to the process pool and track progress
    with ProcessPoolExecutor() as exe:
        for _ in tqdm(exe.map(_process_task, tasks), total=len(tasks), desc="Processing files"):
            pass
    log.info(
        "Conversion complete. %d files written to %s",
        len(tasks),
        log.yellow(camus_output_folder),
    )

    _copy_license(files, camus_output_folder)
    # Write special dataset card if reduced
    if is_reduced:
        write_dataset_card(camus_output_folder, _make_camus_sample_dataset_card())
    else:
        write_dataset_card(camus_output_folder, _CAMUS_DATASET_CARD)

    if getattr(args, "upload", False):
        upload_camus(camus_output_folder, revision=args.revision, repo_id=hf_repo_id)


def _copy_license(files: list[Path], output_folder: Path) -> None:
    """Copy ``MANDATORY_CITATION.md`` from the first patient directory to *output_folder*."""
    import shutil

    for f in files:
        candidate = f.parent / "MANDATORY_CITATION.md"
        if candidate.exists():
            shutil.copy2(candidate, output_folder / "MANDATORY_CITATION.md")
            log.info("Copied %s to %s", candidate.name, log.yellow(output_folder))
            return
    log.warning("MANDATORY_CITATION.md not found in any patient directory.")


def upload_camus(  # pragma: no cover
    output_folder: str | Path, revision: str, repo_id: str = _CAMUS_HF_REPO_ID
) -> None:
    """Upload the converted CAMUS dataset to a HuggingFace Hub revision branch.

    Only for zea maintainers with push access to the repository.  Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_folder: Root folder containing the train/val/test splits.
        revision: Target branch name on the Hub (must not be ``"main"``).
    """
    require_output_dir_ownership(output_folder, repo_id)
    upload_dataset_to_hf(
        folder=output_folder,
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Upload CAMUS dataset (zea format) to {revision}",
    )


_CAMUS_DATASET_CARD = (
    """\
---
license: cc-by-nc-sa-4.0
zea_repo_id: zeahub/camus
task_categories:
  - image-segmentation
tags:
  - ultrasound
  - echocardiography
  - 2d
  - cardiac
  - medical
pretty_name: "CAMUS: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation"
size_categories:
  - 1K<n<10K
---

# CAMUS - 2-D Echocardiographic Ultrasound Dataset

This dataset is a **zea-format** (HDF5) conversion of the
[CAMUS](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
dataset for multi-structure segmentation in 2-D echocardiography.

| Property | Value |
|---|---|
| **Modality** | 2-D transthoracic echocardiography |
| **Patients** | 500 |
| **Views** | 2-chamber (2CH) and 4-chamber (4CH) apical |
| **Splits** | train (1-400), val (401-450), test (451-500) |

## Conversion

This dataset was downloaded, converted to zea format, and uploaded using the
[zea](https://github.com/tue-bmd/zea) data converter:

```bash
python -m zea.data.convert camus <src> <dst> --download
```

## Dataset structure

```
train/
  patient0001/
    patient0001_2CH_half_sequence.hdf5
    patient0001_4CH_half_sequence.hdf5
  ...
val/
  patient0401/ ...
test/
  patient0451/ ...
```

Each HDF5 file follows the [zea data format](https://github.com/tue-bmd/zea) and contains:

- `data/image/values` — scan-converted B-mode sequence in dB, shape
  `(n_frames, H, W)`, float32; x=0 at apex centre
- `data/image/coordinates` — per-pixel Cartesian positions in metres,
  shape `(H, W, 3)` [x, y=0, z]
- `data/image_polar/values` — polar-resampled B-mode sequence,
  shape `(n_frames, n_r, n_theta)`, float32
- `data/image_polar/coordinates` — Cartesian [x, 0, z] positions of polar
  pixels in metres, shape `(n_r, n_theta, 3)`
- `data/segmentation/values` — multi-label bool segmentation, shape `(n_frames, H, W, 4)`
- `data/segmentation/labels` — `["unannotated", "LV_endo", "LV_myo", "LA"]`;
  unannotated frames have only the first channel set
- `data/segmentation/coordinates` — same grid as `image/coordinates`
- `metadata/subject` — patient ID, sex, age
- `metadata/credit` — full citation string
- `metadata/text_report` — ejection fraction, frame rate, image quality
- `metadata/annotations/view` — `"2CH"` or `"4CH"` repeated for all frames
- `metadata/annotations/label` — `"ED"` / `"ES"` for the corresponding frames, `""` otherwise
- `metadata/annotations/image_quality` — `"Good"` / `"Medium"` / `"Poor"`

## License

"""
    + CAMUS_LICENSE
    + """

The CAMUS dataset is available free of charge strictly for **non-commercial
scientific research purposes only**.

## Citation

If you use this dataset, please cite:

```bibtex
@article{leclerc2019deep,
  title   = {Deep Learning for Segmentation Using an Open Large-Scale Dataset in
             2D Echocardiography},
  author  = {Leclerc, Sarah and Smistad, Erik and Pedrosa, Joao and Ostvik, Andreas and
             Cervenansky, Frederic and Espinosa, Florian and Espeland, Torvald and
             Berg, Erik Andreas Rye and Jodoin, Pierre-Marc and Grenier, Thomas and
             Lartizien, Carole and D'hooge, Jan and Lovstakken, Lasse and
             Bernard, Olivier},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {38},
  number  = {9},
  pages   = {2198--2210},
  year    = {2019},
  doi     = {10.1109/TMI.2019.2900516}
}
```

## Links

- **Original dataset**: <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8>
- **zea toolkit**: <https://github.com/tue-bmd/zea>
"""
)


def _make_camus_sample_dataset_card() -> str:
    """Build the dataset card for the reduced sample subset from the full card.

    Derives the sample card from ``_CAMUS_DATASET_CARD`` by updating the YAML
    frontmatter fields that differ and prepending a notice that this is a
    sample subset.

    Returns:
        The dataset card string for the sample subset.
    """
    card = _CAMUS_DATASET_CARD
    card = card.replace(
        "zea_repo_id: zeahub/camus",
        "zea_repo_id: zeahub/camus-sample",
    )
    card = card.replace(
        'pretty_name: "CAMUS: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation"',
        'pretty_name: "CAMUS Sample: Cardiac Acquisitions for Multi-structure '
        'Ultrasound Segmentation (Sample)"',
    )
    card = card.replace(
        "size_categories:\n  - 1K<n<10K",
        "size_categories:\n  - n<10",
    )
    card = card.replace(
        "# CAMUS - 2-D Echocardiographic Ultrasound Dataset",
        "# CAMUS Sample - 2-D Echocardiographic Ultrasound Dataset",
    )
    sample_notice = (
        "\n> **This is a sample subset** of the full CAMUS dataset, provided for "
        "demonstration and testing purposes. It contains 6 files (1 patient per split). "
        "For the full dataset (500 patients), see: "
        "[zeahub/camus](https://huggingface.co/datasets/zeahub/camus).\n"
    )
    card = card.replace(
        "\n\nThis dataset is a **zea-format**",
        sample_notice + "\nThis dataset is a **zea-format**",
    )
    return card
