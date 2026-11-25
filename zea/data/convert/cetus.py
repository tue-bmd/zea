"""Convert the CETUS dataset to the zea format.

.. note::

   Requires SimpleITK: ``pip install SimpleITK``.

CETUS (Challenge on Endocardial Three-dimensional Ultrasound Segmentation) is a
public MICCAI 2014 challenge dataset.  It contains 3-D echocardiographic volumes
from 45 patients with ground-truth left-ventricle segmentation masks at
end-diastole (ED) and end-systole (ES).  Volumes are stored as NIfTI
(``.nii.gz``) files with isotropic voxel spacing.

Each converted file holds the scan-converted (Cartesian) volume and the same volume
resampled onto a polar ``(rho, theta, phi)`` grid (``image_polar``), which can be scan
converted back with :func:`zea.display.scan_convert_3d`. CETUS does not include the
acquisition geometry, so the scan frustum (apex, angle ranges and depth) is estimated
from every volume with :func:`zea.tools.fit_scan_cone.detect_frustum_parameters`.
The volumes come from three scanners. The sectors of 30 volumes (15 patients) are not a
frustum (a pyramid) but a fan of 2-D azimuth sectors tilted in elevation. For these the
polar grid is a frustum that encloses the whole sector, so parts of it lie outside the
sector, and ``custom/polar_geometry/sector_type`` is ``"fan"`` instead of ``"pyramid"``.
Sector corners that were cropped off the original Cartesian volumes are background in the
polar volume too. Pass ``--no-polar`` to store only the Cartesian volumes.

Dataset splits:

* **Train** - patients 1-30
* **Validation** - patients 31-38
* **Test** - patients 39-45

.. admonition:: License

   CC BY-NC-SA 4.0 - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

   The CETUS dataset is available free of charge strictly for non-commercial
   scientific research purposes only.

.. citation:: bernard2016standardized

.. rubric:: Links

* `MICCAI 2014 CETUS Challenge <https://www.creatis.insa-lyon.fr/Challenge/CETUS/>`_
* `Original dataset <https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45>`_
* `Dataset on Hugging Face <https://huggingface.co/datasets/zeahub/cetus-miccai-2014>`_

.. rubric:: Usage

.. code-block:: console

   zea convert cetus ./raw ./output --download

"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import log
from zea.data.convert.utils import (
    check_output_dir_ownership,
    download_from_girder,
    require_output_dir_ownership,
    sitk_load,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import CustomElement, File
from zea.tools.fit_scan_cone import detect_frustum_parameters

# Default shape (n_rho, n_theta, n_phi) of the polar volume. The radial step is then close
# to the 0.58 mm voxel size of the Cartesian volumes.
DEFAULT_POLAR_SHAPE = (256, 128, 128)

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


def cartesian_coordinates(shape, apex, spacing) -> np.ndarray:
    """Per-voxel ``[x, y, z]`` positions of a CETUS volume, with the apex at the origin.

    Args:
        shape: Volume shape ``(n_z, n_x, n_y)``.
        apex: Apex position ``(z, x, y)`` as a fractional voxel index.
        spacing: Voxel size along ``(z, x, y)`` in metres.

    Returns:
        Array of shape ``(n_z, n_x, n_y, 3)`` and type float32, in metres.
    """
    axes = [(np.arange(n) - a) * s for n, a, s in zip(shape, apex, spacing)]
    coordinates = np.empty((*shape, 3), dtype=np.float32)
    coordinates[..., 0] = axes[1][None, :, None]  # x: second axis
    coordinates[..., 1] = axes[2][None, None, :]  # y: third axis
    coordinates[..., 2] = axes[0][:, None, None]  # z: depth
    return coordinates


def polar_grid(frustum, polar_shape, spacing):
    """Sample positions of a regular ``(rho, theta, phi)`` grid spanning a scan frustum.

    Uses the frustum convention of :func:`zea.display.frustum_convert_rtp2xyz`, so a volume
    sampled on this grid can be scan converted back with :func:`zea.display.scan_convert_3d`
    using the frustum's ``rho_range``, ``theta_range`` and ``phi_range``.

    Args:
        frustum (dict): Output of :func:`~zea.tools.fit_scan_cone.detect_frustum_parameters`,
            with ``rho_range`` in metres.
        polar_shape (tuple): Number of samples ``(n_rho, n_theta, n_phi)``.
        spacing: Voxel size along ``(z, x, y)`` in metres.

    Returns:
        Tuple of:
            - ``indices``: ``(3, n_rho, n_theta, n_phi)`` fractional voxel indices ``(z, x, y)``
              of the samples in the Cartesian volume, for :func:`scipy.ndimage.map_coordinates`.
            - ``coordinates``: ``(n_rho, n_theta, n_phi, 3)`` float32 ``[x, y, z]`` positions of
              the samples in metres, with the apex at the origin.
    """
    n_rho, n_theta, n_phi = polar_shape
    rho = np.linspace(*frustum["rho_range"], n_rho)
    tan_theta = np.tan(np.linspace(*frustum["theta_range"], n_theta))
    tan_phi = np.tan(np.linspace(*frustum["phi_range"], n_phi))

    # Same as zea.display.frustum_convert_rtp2xyz, in NumPy so that conversion workers do not
    # initialise the ML backend.
    rho, tan_theta, tan_phi = np.meshgrid(rho, tan_theta, tan_phi, indexing="ij")
    z = rho / np.sqrt(1 + tan_theta**2 + tan_phi**2)
    x = z * tan_theta
    y = z * tan_phi

    apex = np.asarray(frustum["apex"], dtype=np.float64)
    indices = np.stack(
        [apex[0] + z / spacing[0], apex[1] + x / spacing[1], apex[2] + y / spacing[2]]
    )
    coordinates = np.stack([x, y, z], axis=-1).astype(np.float32)
    return indices, coordinates


def _polar_volume(volume_db, frustum, polar_shape, spacing, name):
    """Resample a CETUS volume onto the polar grid of its scan frustum.

    Args:
        volume_db (np.ndarray): ``(z, x, y)`` volume in dB, with background at -60 dB.
        frustum (dict): Output of :func:`~zea.tools.fit_scan_cone.detect_frustum_parameters`.
        polar_shape (tuple): Number of samples ``(n_rho, n_theta, n_phi)``.
        spacing: Voxel size along ``(z, x, y)`` in metres.
        name (str): File name, for log messages.

    Returns:
        Tuple of the ``image_polar`` data entry and the custom elements of the
        ``polar_geometry`` group.
    """
    from scipy.ndimage import map_coordinates

    indices, coordinates = polar_grid(frustum, polar_shape, spacing)
    values = map_coordinates(volume_db, indices, order=1, mode="constant", cval=-60.0)
    values = np.clip(values, -60.0, 0.0).astype(np.float32)
    image_polar = {
        "values": values[np.newaxis, ...],  # (1, n_rho, n_theta, n_phi)
        "coordinates": coordinates,
    }

    polar_geometry = {
        "rho_range": ("m", "Radial range of image_polar, measured from the apex."),
        "theta_range": ("rad", "Azimuth range of image_polar, along x (second axis of image)."),
        "phi_range": ("rad", "Elevation range of image_polar, along y (third axis of image)."),
        "apex": ("voxel", "Estimated apex as a fractional (z, x, y) voxel index into image."),
    }
    custom = [
        CustomElement(
            name=element,
            data=np.asarray(frustum[element], dtype=np.float32),
            description=description,
            unit=unit,
            group_name="polar_geometry",
        )
        for element, (unit, description) in polar_geometry.items()
    ]
    custom.append(
        CustomElement(
            name="sector_type",
            data=frustum["sector_type"],
            description=(
                "Shape of the imaging sector: 'pyramid' if it is the frustum of image_polar, "
                "'fan' if it is a fan of 2-D azimuth sectors tilted in elevation, which the "
                "frustum encloses, so image_polar has empty regions at its theta edges, or "
                "'unknown'."
            ),
            unit="",
            group_name="polar_geometry",
        )
    )
    if frustum["sector_type"] == "unknown":
        log.warning(
            "%s: the scan sector is neither a pyramid nor a fan (edge widening %s).",
            name,
            frustum["edge_widening"],
        )
    return image_polar, custom


def process_cetus(source_path, output_path, overwrite=False, polar_shape=DEFAULT_POLAR_SHAPE):
    """Convert a single CETUS patient time-point to a zea HDF5 file.

    Each file stores the 3D B-mode volume as ``image`` (scan-converted image) and the
    same volume resampled onto a polar ``(rho, theta, phi)`` grid as ``image_polar``.
    The polar grid spans the scan frustum detected with
    :func:`~zea.tools.fit_scan_cone.detect_frustum_parameters` and is sampled with
    trilinear interpolation. Its ranges are stored as custom elements in the
    ``polar_geometry`` group: ``rho_range`` (m), ``theta_range`` and ``phi_range`` (rad),
    and the ``apex`` as a fractional ``(z, x, y)`` voxel index into ``image``. The group
    also holds the ``sector_type`` string: ``"pyramid"`` if the sector is the frustum itself,
    ``"fan"`` if it is a fan of tilted 2-D sectors that the frustum encloses, so that
    ``image_polar`` has empty regions at its ``theta`` edges, or ``"unknown"``. With
    ``polar_shape=None`` only ``image`` is stored, without ``image_polar`` and
    ``polar_geometry``.

    If a corresponding ground truth segmentation file exists, it is stored as a
    ``Segmentation`` map under ``data/segmentation`` and shares the per-voxel coordinates
    of ``image``. All coordinates are ``[x, y, z]`` positions in metres with the estimated
    apex at the origin, also when no polar volume is stored, so the scan frustum is always
    detected.

    Patient ID and citation are stored in the ``metadata`` group.
    License information is embedded in the file description.

    Args:
        source_path (str or Path): Path to the source ``.nii.gz`` B-mode file.
        output_path (str or Path): Path to the output ``.hdf5`` file.
        overwrite (bool, optional): Whether to overwrite an existing output file.
            Defaults to False.
        polar_shape (tuple or None, optional): Shape ``(n_rho, n_theta, n_phi)`` of the polar
            volume, or None to store only the Cartesian volume. Defaults to
            :data:`DEFAULT_POLAR_SHAPE`.

    Raises:
        ValueError: If the scan frustum cannot be detected.
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
    # volume shape: (depth, height, width) = (z, x, y), with the apex at the top.

    # Voxel spacing in meters (CETUS stores it in metres, ~0.0005763). SimpleITK lists the
    # spacing in image (x, y, z) order, which is the reverse of the NumPy axes.
    voxel_spacing = np.array(metadata["spacing"][::-1], dtype=np.float64)

    # The CETUS volumes have a background padding value that is nonzero and varies per file.
    # Here we detect it from the histogram and create a binary mask so that
    # background voxels are mapped to exactly -60 dB (pure black).
    bg_level = int(_detect_background_level(volume))
    bg_mask = volume.astype(int) == bg_level

    # Convert B-mode intensity [0, 255] to dB range [-60, 0].
    volume_db = volume.astype(np.float32) * np.float32(60.0 / 255.0) - np.float32(60.0)
    volume_db[bg_mask] = -60.0

    # Estimate the scan frustum from the voxels that hold data. Its apex is the origin of the
    # coordinates, also without a polar volume. Besides the background level, the raw volumes
    # have zeros in the last plane(s) along each axis. Both are at the -60 dB floor, so only
    # voxels above it count as data. Otherwise the zero planes join the sector wherever it
    # touches those borders.
    frustum = detect_frustum_parameters(volume_db > -60.0, spacing=voxel_spacing)
    log.debug("%s: detected scan frustum %s", source_path.name, frustum)

    # Store as image with shape (n_frames, depth, height, width).
    # For 3D volumes, n_frames=1 (single time point: ED or ES).
    image_values = volume_db[np.newaxis, ...]  # (1, D, H, W)

    # Check for corresponding ground truth segmentation
    gt_path = source_path.with_name(source_path.name.replace(".nii.gz", "_gt.nii.gz"))

    # Extract patient and time-point info from filename
    stem = source_path.stem  # e.g. "patient01_ED.nii" -> stem is "patient01_ED"
    if stem.endswith(".nii"):
        stem = stem[:-4]  # remove .nii if present from double suffix
    time_point = stem.split("_")[-1]  # "ED" or "ES"
    patient_name = stem.split("_")[0]  # e.g. "patient01"

    # Per-voxel [x, y, z] positions, shared by image (1, D, H, W) and segmentation
    # (1, D, H, W, 1), in the same apex-centred frame as the polar coordinates.
    coordinates = cartesian_coordinates(volume.shape, frustum["apex"], voxel_spacing)

    data = {
        "image": {
            "values": image_values,
            "coordinates": coordinates,
        },
    }
    custom = []
    if polar_shape is not None:
        data["image_polar"], custom = _polar_volume(
            volume_db, frustum, polar_shape, voxel_spacing, source_path.name
        )

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
    polar_note = (
        ", with a polar resampling on an estimated scan frustum (image_polar)"
        if polar_shape is not None
        else ""
    )
    file_description = (
        f"CETUS dataset - {patient_name} {time_point} - "
        f"3D echocardiographic volume converted to zea format{polar_note}. "
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
        custom=custom,
        overwrite=overwrite,
        # image_polar is not a built-in data field, which File.create warns about.
        ignore_warnings=polar_shape is not None,
    )


def _process_task(task):
    """Unpack a task tuple and invoke process_cetus in a worker process.

    Args:
        task (tuple): ``(source_file_str, output_file_str, polar_shape)``, with
            ``polar_shape`` None to store only the Cartesian volume.
    """
    source_file_str, output_file_str, polar_shape = task
    source_file = Path(source_file_str)
    output_file = Path(output_file_str)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    process_cetus(source_file, output_file, overwrite=False, polar_shape=polar_shape)


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

        zea convert cetus <source_folder> <destination_folder> --download

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
            - polar_shape (tuple, optional): Shape ``(n_rho, n_theta, n_phi)`` of the
              polar volumes. Defaults to :data:`DEFAULT_POLAR_SHAPE`.
            - no_polar (bool, optional): If True, store only the Cartesian volumes,
              without ``image_polar``. Defaults to False.
            - upload (bool, optional): If True, upload the converted dataset to
              HuggingFace Hub after conversion. Only for zea maintainers with push
              access to the repository.

    Raises:
        ValueError: If ``polar_shape`` is invalid, or ``no_polar`` and ``upload`` are both
            set: the dataset on the Hub includes the polar volumes.
        RuntimeError: If any file failed to convert, after the other files are written and
            before anything is uploaded. Each failure is logged with its file name.
    """
    cetus_source_folder = Path(args.src)
    cetus_output_folder = Path(args.dst)
    if getattr(args, "no_polar", False):
        if getattr(args, "upload", False):
            raise ValueError(
                "--no-polar cannot be combined with --upload: the CETUS dataset on the Hub "
                "includes the polar volumes."
            )
        polar_shape = None
    else:
        polar_shape = tuple(getattr(args, "polar_shape", DEFAULT_POLAR_SHAPE))
        if len(polar_shape) != 3 or min(polar_shape) < 2:
            raise ValueError(f"polar_shape must be three sizes of at least 2, got {polar_shape}")

    check_output_dir_ownership(cetus_output_folder, _HF_REPO_ID)

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

        tasks.append((str(source_file), str(output_file), polar_shape))

    if not tasks:
        log.info("No CETUS files found to process.")
        return

    log.info(f"Found {len(tasks)} files to convert.")

    # Source file -> exception, for every file that failed to convert
    failures = {}
    if getattr(args, "no_hyperthreading", False):
        log.info("Running tasks serially (no ProcessPoolExecutor)")
        for task in tqdm(tasks, desc="Processing files (serial)"):
            try:
                _process_task(task)
            except Exception as exc:
                failures[task[0]] = exc
                log.error("Failed to convert %s: %s", log.yellow(task[0]), exc)
    else:
        # No more workers than files: the default is one per CPU, all forked up front.
        with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as exe:
            futures = {exe.submit(_process_task, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    future.result()
                except Exception as exc:
                    source = futures[future][0]
                    failures[source] = exc
                    log.error("Failed to convert %s: %s", log.yellow(source), exc)

    log.info(
        "Conversion complete. %d of %d files converted or already present in %s.",
        len(tasks) - len(failures),
        len(tasks),
        log.yellow(cetus_output_folder),
    )

    # Written even after failures, so that a rerun into this folder passes the ownership check.
    write_dataset_card(cetus_output_folder, _DATASET_CARD)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} CETUS files failed to convert: "
            f"{', '.join(Path(source).name for source in sorted(failures))}. "
            "See the errors above."
        )

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
    require_output_dir_ownership(output_folder, _HF_REPO_ID)
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
zea_repo_id: zeahub/cetus-miccai-2014
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
zea convert cetus <src> <dst> --download
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

- `data/image/values` - B-mode volume in dB, shape `(1, z, x, y)`, with depth `z` first
- `data/image/coordinates` - voxel positions `[x, y, z]` in metres with the estimated apex
  at the origin, shape `(z, x, y, 3)`
- `data/image_polar/values` - the B-mode volume resampled onto a polar grid, in dB, shape
  `(1, rho, theta, phi)`, `(1, 256, 128, 128)` by default (`--polar-shape`). This and the
  `polar_geometry` elements below are left out when converting with `--no-polar`.
- `data/image_polar/coordinates` - `[x, y, z]` position in metres of every polar sample, in
  the same frame as `data/image/coordinates`, shape `(rho, theta, phi, 3)`
- `custom/polar_geometry/rho_range`, `theta_range`, `phi_range` - extent of the polar grid
  (metres and radians), and `custom/polar_geometry/apex` - the apex as a `(z, x, y)` voxel
  index into `data/image`
- `custom/polar_geometry/sector_type` - shape of the imaging sector: `pyramid` or `fan` (see
  below)
- `data/segmentation/values` - binary LV endocardium mask, shape `(1, z, x, y, 1)`
- `data/segmentation/labels` - `["endocardium"]`
- `metadata/subject/id` - patient name (e.g. `patient01`)
- `metadata/annotations/label` - time point: `["ED"]` or `["ES"]`

## Polar volumes

CETUS ships only the scan-converted (Cartesian) volumes, without the acquisition geometry.
For every volume, the scan frustum (apex, azimuth `theta` and elevation `phi` ranges, and
depth) is estimated from the voxels that hold data, with
`zea.tools.fit_scan_cone.detect_frustum_parameters`. The volume is then resampled onto a
regular `(rho, theta, phi)` grid with trilinear interpolation, using the zea frustum
convention (`tan(theta) = x / z`, `tan(phi) = y / z`). It can be scan converted back with
`zea.display.scan_convert_3d` and the ranges in `custom/polar_geometry`.

This is an inverse scan conversion, so it adds an interpolation step and only approximates
the original beam geometry. The sector comes in two shapes, stored as
`custom/polar_geometry/sector_type`:

- `pyramid` (60 volumes, 30 patients): the sector is a frustum, which the polar grid fits.
- `fan` (30 volumes, 15 patients): the sector is a fan of 2-D azimuth sectors tilted in
  elevation, which is narrowest in azimuth at zero elevation. The polar grid is a frustum
  that encloses the whole sector, so about 5-10% of it lies outside the sector, in curved
  regions at its `theta` edges.

Samples outside the sector, including sector corners that were cropped off the original
Cartesian volumes, are -60 dB.

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
