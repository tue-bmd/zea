"""Functionality to convert the CETUS dataset to the zea format.

.. note::
    Requires SimpleITK to be installed: ``pip install SimpleITK``.

The CETUS (Challenge on Endocardial Three-dimensional Ultrasound Segmentation)
dataset contains 3D echocardiographic volumes from 45 patients. Each patient has
end-diastolic (ED) and end-systolic (ES) B-mode volumes with corresponding
ground truth left ventricle segmentation masks. The volumes are stored in NIfTI
(.nii.gz) format with isotropic voxel spacing.

**License**: `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_

The CETUS dataset is available free of charge strictly for non-commercial
scientific research purposes only.

**Citation** (required for any use of the CETUS database):

    O. Bernard, et al.
    "Standardized Evaluation System for Left Ventricular Segmentation Algorithms
    in 3D Echocardiography"
    IEEE Transactions on Medical Imaging, vol. 35, no. 4, pp. 967-977, April 2016.
    `DOI: 10.1109/tmi.2015.2503890 <https://doi.org/10.1109/tmi.2015.2503890>`_

**Links**:

- `MICCAI 2014 CETUS Challenge <https://www.creatis.insa-lyon.fr/Challenge/CETUS/>`_
- `Original dataset <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45>`_

"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from zea import log
from zea.data.convert.utils import download_from_girder, sitk_load
from zea.data.data_format import DatasetElement, generate_zea_dataset

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
    If a corresponding ground truth segmentation file exists, it is stored as an
    additional element under ``non_standard_elements/segmentation``.

    The voxel spacing from the NIfTI header is stored as ``non_standard_elements/voxel_spacing``
    (in meters). License and citation information is embedded in the file description.

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
            os.remove(output_path)
        else:
            log.info(f"Output file {output_path} already exists. Skipping.")
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

    additional_elements = []

    # Store voxel spacing
    additional_elements.append(
        DatasetElement(
            dataset_name="voxel_spacing",
            data=voxel_spacing,
            description=(
                "Voxel spacing in meters for each dimension (x, y, z) "
                "as provided in the NIfTI header."
            ),
            unit="m",
        )
    )

    # Store citation
    additional_elements.append(
        DatasetElement(
            dataset_name="citation",
            data=np.array(CETUS_CITATION, dtype=h5py.string_dtype()),
            description="Required citation for any use of the CETUS database.",
            unit="unitless",
        )
    )

    # Store license
    additional_elements.append(
        DatasetElement(
            dataset_name="license",
            data=np.array(CETUS_LICENSE, dtype=h5py.string_dtype()),
            description="License of the CETUS dataset.",
            unit="unitless",
        )
    )

    # Store time point info (ED or ES)
    stem = source_path.stem  # e.g. "patient01_ED.nii" -> stem is "patient01_ED"
    if stem.endswith(".nii"):
        stem = stem[:-4]  # remove .nii if present from double suffix
    time_point = stem.split("_")[-1]  # "ED" or "ES"
    additional_elements.append(
        DatasetElement(
            dataset_name="time_point",
            data=np.array(time_point, dtype=h5py.string_dtype()),
            description="Cardiac time point: ED (end-diastole) or ES (end-systole).",
            unit="unitless",
        )
    )

    # Store patient ID
    patient_name = stem.split("_")[0]  # e.g. "patient01"
    patient_id = int(patient_name.removeprefix("patient"))
    additional_elements.append(
        DatasetElement(
            dataset_name="patient_id",
            data=np.array(patient_id, dtype=np.int64),
            description="Patient ID number.",
            unit="unitless",
        )
    )

    if gt_path.exists():
        gt_volume, _ = sitk_load(gt_path)
        # GT is binary: 0 or 255 -> normalize to 0/1
        gt_volume = (gt_volume > 0).astype(np.float32)
        gt_volume = gt_volume[np.newaxis, ...]  # (1, D, H, W)
        additional_elements.append(
            DatasetElement(
                dataset_name="segmentation",
                data=gt_volume,
                description=(
                    "Ground truth left ventricle segmentation mask. "
                    "Binary: 1 = endocardium, 0 = background. "
                    "Shape: (n_frames, depth, height, width)."
                ),
                unit="unitless",
            )
        )

    # Build description for this file
    file_description = (
        f"CETUS dataset - {patient_name} {time_point} - "
        f"3D echocardiographic volume converted to zea format. "
        f"Voxel spacing: {voxel_spacing.tolist()} m. "
        f"License: {CETUS_LICENSE}. "
        f"Citation: {CETUS_CITATION}"
    )

    generate_zea_dataset(
        path=output_path,
        image_sc=image_sc,
        probe_name="generic",
        description=file_description,
        additional_elements=additional_elements,
        cast_to_float=True,
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
        log.error("Error processing %s", source_file)
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
                f"Output folder {split_dir} already exists. Existing files will be skipped."
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
                log.error(f"Failed to process {t[0]}: {exc}")
        log.info(f"Processing finished for {len(tasks)} files (serial)")

        if getattr(args, "upload", False):
            upload_cetus(cetus_output_folder)
        return

    # Parallel processing
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(_process_task, t) for t in tasks]
        for future in tqdm(futures, desc="Processing files"):
            try:
                future.result()
            except Exception as exc:
                log.error(f"Failed to process a file: {exc}")
    log.info(f"Processing finished for {len(tasks)} files")

    if getattr(args, "upload", False):
        upload_cetus(cetus_output_folder)


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


def _write_dataset_card(folder: Path) -> Path:  # pragma: no cover
    """Write the HuggingFace dataset card (README.md) into *folder*."""
    card_path = folder / "README.md"
    card_path.write_text(_DATASET_CARD)
    return card_path


def upload_cetus(output_folder: str | Path) -> None:  # pragma: no cover
    """Upload the converted CETUS dataset to HuggingFace Hub.

    Only for zea maintainers with push access to the repository.

    Writes a dataset card, prints an upload summary, and asks for
    confirmation before pushing.

    Args:
        output_folder: Root folder containing the train/val/test splits.
    """
    from huggingface_hub import HfApi, login

    output_folder = Path(output_folder)

    # Collect files to upload
    hdf5_files = sorted(output_folder.rglob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {output_folder}")

    total_size_mb = sum(f.stat().st_size for f in hdf5_files) / 1e6
    split_counts = {}
    for f in hdf5_files:
        split = f.relative_to(output_folder).parts[0]
        split_counts[split] = split_counts.get(split, 0) + 1

    # Write dataset card
    _write_dataset_card(output_folder)

    # Print summary and ask for confirmation
    log.info("")
    log.info("=" * 60)
    log.info("  CETUS upload summary")
    log.info("=" * 60)
    log.info(f"  Repository : {_HF_REPO_ID}")
    log.info(f"  Source     : {output_folder}")
    log.info(f"  Files      : {len(hdf5_files)} HDF5 + README.md")
    for split, count in sorted(split_counts.items()):
        log.info(f"    {split:>5s}: {count} files")
    log.info(f"  Total size : {total_size_mb:.1f} MB")
    log.info(f"  License    : {CETUS_LICENSE}")
    log.info("=" * 60)
    log.info("")

    answer = input("Proceed with upload? [y/N] ").strip().lower()
    if answer != "y":
        log.info("Upload cancelled.")
        return

    login(new_session=False)
    api = HfApi()

    api.upload_folder(
        folder_path=str(output_folder),
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        commit_message="Upload CETUS dataset (zea format)",
    )

    log.info(f"Dataset uploaded to https://huggingface.co/datasets/{_HF_REPO_ID}")
