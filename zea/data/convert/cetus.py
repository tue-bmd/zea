"""Convert the CETUS dataset to the zea format.

.. note::

   Requires SimpleITK: ``pip install SimpleITK``.

CETUS (Challenge on Endocardial Three-dimensional Ultrasound Segmentation) is a
public MICCAI 2014 challenge dataset.  It contains 3-D echocardiographic volumes
from 45 patients with ground-truth left-ventricle segmentation masks at
end-diastole (ED) and end-systole (ES).  Volumes are stored as NIfTI
(``.nii.gz``) files with isotropic voxel spacing.

Dataset splits:

* **Train** - patients 1-30
* **Validation** - patients 31-38
* **Test** - patients 39-45

.. admonition:: License

   CC BY-NC-SA 4.0 - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

   The CETUS dataset is available free of charge strictly for non-commercial
   scientific research purposes only.

.. admonition:: Reference

   O. Bernard, et al.
   *Standardized Evaluation System for Left Ventricular Segmentation Algorithms
   in 3D Echocardiography.*
   IEEE Transactions on Medical Imaging, vol. 35, no. 4, pp. 967-977, April 2016.
   `DOI: 10.1109/tmi.2015.2503890 <https://doi.org/10.1109/tmi.2015.2503890>`_

.. rubric:: Links

* `MICCAI 2014 CETUS Challenge <https://www.creatis.insa-lyon.fr/Challenge/CETUS/>`_
* `Original dataset <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45>`_
* `Dataset on Hugging Face <https://huggingface.co/datasets/zeahub/cetus-miccai-2014>`_

.. rubric:: Usage

.. code-block:: console

   python -m zea.data.convert cetus ./raw ./output --download

"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import log
from zea.data.convert.utils import (
    download_from_girder,
    sitk_load,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import File

# Citation text for inclusion in every converted file
CETUS_CITATION = (
    'O. Bernard, et al. "Standardized Evaluation System for Left Ventricular '
    'Segmentation Algorithms in 3D Echocardiography" in IEEE Transactions on '
    "Medical Imaging, vol. 35, no. 4, pp. 967-977, April 2016. "
    "https://doi.org/10.1109/tmi.2015.2503890"
)

CETUS_LICENSE = "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)"

CETUS_DESCRIPTION = (
    "CETUS (Challenge on Endocardial Three-dimensional Ultrasound Segmentation) "
    "3D echocardiographic dataset converted to zea format. "
    "License: {license}. "
    "Citation: {citation}"
).format(license=CETUS_LICENSE, citation=CETUS_CITATION)

# Girder collection ID for the CETUS dataset
_CETUS_COLLECTION_ID = "62eb991b73e9f0048c3a6c45"

# Dataset splits: patient IDs 1-30 for training, 31-38 for validation, 39-45 for test
splits = {"train": [1, 31], "val": [31, 39], "test": [39, 46]}


def get_split(patient_id: int) -> str:
    """Determine which dataset split a patient ID belongs to.

    Args:
        patient_id: Integer ID of the patient (1-45).

    Returns:
        The split name: ``"train"``, ``"val"``, or ``"test"``.

    Raises:
        ValueError: If the patient_id does not fall into any defined split range.
    """
    for split_name, (start, end) in splits.items():
        if start <= patient_id < end:
            return split_name
    raise ValueError(f"Did not find split for patient: {patient_id}")


def _detect_background_level(volume: np.ndarray) -> float:
    """Detect the background padding value of a CETUS volume.

    The CETUS volumes are zero-padded outside the scanning cone, but the
    padding value is not exactly zero — it varies per file (e.g. 8 or 13 on a
    [0, 255] scale).  This function finds the mode of the integer-binned
    histogram which corresponds to the dominant background intensity.

    Args:
        volume: 3-D numpy array with values in [0, 255].

    Returns:
        The detected background intensity level.
    """
    # Use integer bins (0..255) — the padding value is always a single integer
    counts, bin_edges = np.histogram(volume.ravel(), bins=256, range=(0, 256))
    bg_level = float(bin_edges[np.argmax(counts)])
    return bg_level


def process_cetus(source_path, output_path, overwrite=False):
    """Convert a single CETUS patient time-point to a zea HDF5 file.

    Each file stores the 3D B-mode volume as ``image_sc`` (scan-converted image).
    If a corresponding ground truth segmentation file exists, it is stored as a
    ``Segmentation`` map under ``data/segmentation``. Both maps include a
    per-voxel coordinate grid (shape ``(1, D, H, W, 3)``) derived from the NIfTI
    voxel spacing.

    Patient ID and citation are stored in the ``metadata`` group.
    License information is embedded in the file description.

    Args:
        source_path (str or Path): Path to the source ``.nii.gz`` B-mode file.
        output_path (str or Path): Path to the output ``.hdf5`` file.
        overwrite (bool, optional): Whether to overwrite an existing output file.
            Defaults to False.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)

    # Check if output file already exists
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            log.info("Output file %s already exists. Skipping.", log.yellow(output_path))
            return

    # Load B-mode volume
    volume, metadata = sitk_load(source_path)
    # volume shape: (depth, height, width) — 3D

    # Voxel spacing in meters (NIfTI stores in mm-like units depending on header;
    # CETUS uses meters based on the spacing values ~0.0005763)
    voxel_spacing = np.array(metadata["spacing"], dtype=np.float64)

    # The CETUS volumes have a background padding value that is nonzero and varies per file.
    # Here we detect it from the histogram and create a binary mask so that
    # background voxels are mapped to exactly -60 dB (pure black).
    bg_level = int(_detect_background_level(volume))
    bg_mask = volume.astype(int) == bg_level

    # Convert B-mode intensity [0, 255] to dB range [-60, 0].
    volume_db = (volume / 255.0) * 60.0 - 60.0
    volume_db[bg_mask] = -60.0

    # Store as image_sc with shape (n_frames, depth, height, width).
    # For 3D volumes, n_frames=1 (single time point: ED or ES).
    image_sc = volume_db[np.newaxis, ...]  # (1, D, H, W)

    # Check for corresponding ground truth segmentation
    gt_path = source_path.with_name(source_path.name.replace(".nii.gz", "_gt.nii.gz"))

    # Extract patient and time-point info from filename
    stem = source_path.stem  # e.g. "patient01_ED.nii" -> stem is "patient01_ED"
    if stem.endswith(".nii"):
        stem = stem[:-4]  # remove .nii if present from double suffix
    time_point = stem.split("_")[-1]  # "ED" or "ES"
    patient_name = stem.split("_")[0]  # e.g. "patient01"

    # Build data dict
    D, H, W = volume.shape

    # Build per-voxel coordinate grid from voxel spacing.
    # Shape: (1, D, H, W, 3) — valid for both image_sc (1,D,H,W) and segmentation (1,D,H,W,1).
    d_range = np.arange(D, dtype=np.float32) * voxel_spacing[0]
    h_range = np.arange(H, dtype=np.float32) * voxel_spacing[1]
    w_range = np.arange(W, dtype=np.float32) * voxel_spacing[2]
    d_grid, h_grid, w_grid = np.meshgrid(d_range, h_range, w_range, indexing="ij")
    coordinates = np.stack([d_grid, h_grid, w_grid], axis=-1)[np.newaxis]  # (1, D, H, W, 3)

    data = {
        "image_sc": {
            "values": image_sc.astype(np.float32),
            "coordinates": coordinates,
        }
    }

    if gt_path.exists():
        gt_volume, _ = sitk_load(gt_path)
        # GT is binary: 0 or 255 -> bool mask, shape (1, D, H, W, 1)
        seg_mask = (gt_volume > 0)[np.newaxis, ..., np.newaxis]

        data["segmentation"] = {
            "values": seg_mask,
            "coordinates": coordinates,
            "labels": np.array(["endocardium"]),
        }

    # Build description for this file
    file_description = (
        f"CETUS dataset - {patient_name} {time_point} - "
        f"3D echocardiographic volume converted to zea format. "
        f"Voxel spacing: {voxel_spacing.tolist()} m. "
        f"License: {CETUS_LICENSE}. "
        f"Citation: {CETUS_CITATION}"
    )

    File.create(
        path=output_path,
        data=data,
        metadata={
            "subject": {"id": patient_name},
            "credit": CETUS_CITATION,
            "annotations": {"label": np.array([time_point])},
        },
        probe={"name": "Unspecified mix of GE 4V, Philips X5-1, Siemens 4Z1c"},
        us_machine="Unspecified mix of GE Vivid E9, Philips iE33, Siemens SC2000",
        # includes files from both:
        # - GE system refered as Vivid E9, using a 4V probe;
        # - Philips system refered as iE33, using a X5-1 probe;
        # - Siemens system refered as SC2000, using 4Z1c probe.
        description=file_description,
        overwrite=overwrite,
    )


def _process_task(task):
    """Unpack a task tuple and invoke process_cetus in a worker process.

    Args:
        task (tuple): ``(source_file_str, output_file_str)``
    """
    source_file_str, output_file_str = task
    source_file = Path(source_file_str)
    output_file = Path(output_file_str)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        process_cetus(source_file, output_file, overwrite=False)
    except Exception:
        log.error("Error processing %s", log.yellow(source_file))
        raise


def download_cetus(  # pragma: no cover
    destination: str | Path, patients: list[int] | None = None
) -> Path:
    """Download the CETUS dataset from the Girder server.

    Downloads NIfTI files for each patient (B-mode volumes and ground truth
    segmentations for ED and ES time points).

    Args:
        destination: Directory where the dataset will be downloaded.
        patients: List of patient IDs to download (1-45).
            If None, all 45 patients are downloaded.

    Returns:
        Path to the downloaded dataset directory.
    """
    return download_from_girder(
        collection_id=_CETUS_COLLECTION_ID,
        destination=destination,
        dataset_name="CETUS",
        patients=patients,
    )


def convert_cetus(args):
    """Convert the CETUS dataset into zea HDF5 files across dataset splits.

    Processes all NIfTI B-mode volumes found under the source folder, assigns
    each patient to a train/val/test split, and executes per-file conversion
    tasks either serially or in parallel.

    Usage::

        python -m zea.data.convert cetus <source_folder> <destination_folder> --download

    Args:
        args (argparse.Namespace): An object with attributes:

            - src (str | Path): Path to the folder containing CETUS patient subfolders,
              or a directory to download into when ``--download`` is set.
            - dst (str | Path): Root destination folder for zea HDF5 outputs;
              split subfolders (train/val/test) will be created.
            - download (bool, optional): If True, download the dataset first from the
              Girder server.
            - no_hyperthreading (bool, optional): If True, run tasks serially instead
              of using a process pool.
            - upload (bool, optional): If True, upload the converted dataset to
              HuggingFace Hub after conversion. Only for zea maintainers with push
              access to the repository.
    """
    cetus_source_folder = Path(args.src)
    cetus_output_folder = Path(args.dst)

    # Optionally download the dataset
    if getattr(args, "download", False):
        cetus_source_folder = download_cetus(cetus_source_folder)

    if not cetus_source_folder.exists():
        raise FileNotFoundError(
            f"Source folder does not exist: {cetus_source_folder}. "
            "Use --download to download the CETUS dataset automatically."
        )

    # Check if output folders already exist
    for split in splits:
        split_dir = cetus_output_folder / split
        if split_dir.exists():
            log.warning(
                "Output folder %s already exists. Existing files will be skipped.",
                log.yellow(split_dir),
            )

    # Find all B-mode NIfTI files (exclude ground truth files ending with _gt.nii.gz)
    files = sorted(cetus_source_folder.glob("**/*_ED.nii.gz")) + sorted(
        cetus_source_folder.glob("**/*_ES.nii.gz")
    )

    tasks = []
    for source_file in files:
        patient_name = source_file.stem.split("_")[0]  # e.g. "patient01"
        if source_file.stem.endswith(".nii"):
            # Handle double suffix: .nii.gz -> stem is "patient01_ED.nii"
            patient_name = source_file.name.split("_")[0]

        patient_id = int(patient_name.removeprefix("patient"))
        split = get_split(patient_id)

        # Build output filename
        output_name = source_file.name.replace(".nii.gz", ".hdf5")
        output_file = cetus_output_folder / split / patient_name / output_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        tasks.append((str(source_file), str(output_file)))

    if not tasks:
        log.info("No CETUS files found to process.")
        return

    log.info(f"Found {len(tasks)} files to convert.")

    if getattr(args, "no_hyperthreading", False):
        log.info("Running tasks serially (no ProcessPoolExecutor)")
        for t in tqdm(tasks, desc="Processing files (serial)"):
            try:
                _process_task(t)
            except Exception as exc:
                log.error("Failed to process %s: %s", log.yellow(t[0]), exc)
        log.info(
            "Conversion complete. %d files written to %s",
            len(tasks),
            log.yellow(cetus_output_folder),
        )

        write_dataset_card(cetus_output_folder, _DATASET_CARD)

        if getattr(args, "upload", False):
            upload_cetus(cetus_output_folder, revision=args.revision)
        return

    # Parallel processing
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(_process_task, t) for t in tasks]
        for future in tqdm(futures, desc="Processing files"):
            try:
                future.result()
            except Exception as exc:
                log.error("Failed to process a file: %s", exc)
    log.info(
        "Conversion complete. %d files written to %s",
        len(tasks),
        log.yellow(cetus_output_folder),
    )

    write_dataset_card(cetus_output_folder, _DATASET_CARD)

    if getattr(args, "upload", False):
        upload_cetus(cetus_output_folder, revision=args.revision)


def upload_cetus(output_folder: str | Path, revision: str) -> None:  # pragma: no cover
    """Upload the converted CETUS dataset to a HuggingFace Hub revision branch.

    Only for zea maintainers with push access to the repository.  Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_folder: Root folder containing the train/val/test splits.
        revision: Target branch name on the Hub (must not be ``"main"``).
    """
    upload_dataset_to_hf(
        folder=output_folder,
        repo_id=_HF_REPO_ID,
        revision=revision,
        commit_message=f"Upload CETUS dataset (zea format) to {revision}",
    )


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------

_HF_REPO_ID = "zeahub/cetus-miccai-2014"

_DATASET_CARD = """\
---
license: cc-by-nc-sa-4.0
task_categories:
  - image-segmentation
tags:
  - ultrasound
  - echocardiography
  - 3d
  - cardiac
  - medical
pretty_name: "CETUS: Challenge on Endocardial Three-dimensional Ultrasound Segmentation"
size_categories:
  - n<1K
---

# CETUS - 3-D Echocardiographic Ultrasound Dataset

This dataset is a **zea-format** (HDF5) conversion of the
[CETUS (MICCAI 2014)](https://www.creatis.insa-lyon.fr/Challenge/CETUS/)
challenge data for endocardial segmentation in 3-D echocardiography.

| Property | Value |
|---|---|
| **Modality** | 3-D transthoracic echocardiography |
| **Patients** | 45 |
| **Time points** | End-diastole (ED) and end-systole (ES) per patient |
| **Files** | 90 HDF5 volumes (45 patients x 2 time points) |
| **Voxel spacing** | Isotropic, ~0.576 mm (varies per patient) |
| **Segmentation** | Left-ventricle endocardial surface (binary) |
| **Splits** | train (1-30), val (31-38), test (39-45) |

## Conversion

This dataset was downloaded, converted to zea format, and uploaded using the
[zea](https://github.com/tue-bmd/zea) data converter:

```bash
python -m zea.data.convert cetus <src> <dst> --download
```

## Dataset structure

```
train/
  patient01/
    patient01_ED.hdf5
    patient01_ES.hdf5
  ...
val/
  patient31/ ...
test/
  patient39/ ...
```

Each HDF5 file follows the
[zea data format](https://github.com/tue-bmd/zea) and contains:

- `data/image_sc` - B-mode volume in dB, shape `(1, depth, height, width)`
- `non_standard_elements/segmentation` - binary LV mask, same shape
- `non_standard_elements/voxel_spacing` - `(x, y, z)` in metres
- `non_standard_elements/patient_id`, `time_point`, `citation`, `license`

## License

**CC BY-NC-SA 4.0** - <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>

The CETUS dataset is available free of charge strictly for **non-commercial
scientific research purposes only**.

## Citation

If you use this dataset, please cite the original CETUS paper:

```bibtex
@article{{bernard2016standardized,
  title   = {{Standardized Evaluation System for Left Ventricular Segmentation
              Algorithms in 3D Echocardiography}},
  author  = {{Bernard, Olivier and Bosch, Johan G. and Heyde, Brecht and
              Alessandrini, Martino and Barbosa, Daniel and Camarasu-Pop,
              Sorina and Cervenansky, Fr{{\\'e}}d{{\\'e}}ric and Valette,
              S{{\\'e}}bastien and Mirea, Oana and Berber, Merih and others}},
  journal = {{IEEE Transactions on Medical Imaging}},
  volume  = {{35}},
  number  = {{4}},
  pages   = {{967--977}},
  year    = {{2016}},
  doi     = {{10.1109/tmi.2015.2503890}}
}}
```

## Links

- **Original challenge**: <https://www.creatis.insa-lyon.fr/Challenge/CETUS/>
- **Original dataset**: <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45>
- **zea toolkit**: <https://github.com/tue-bmd/zea>

"""
