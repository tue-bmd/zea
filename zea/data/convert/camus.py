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

"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from skimage.transform import resize
from tqdm import tqdm

from zea import log
from zea.data.convert.utils import (
    download_from_girder,
    sitk_load,
    unzip,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import File
from zea.func.tensor import translate
from zea.internal.utils import find_first_nonzero_index

# Girder collection ID for the CAMUS dataset
_CAMUS_COLLECTION_ID = "6373703d73e9f0047faa1bc8"

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

_CAMUS_HF_REPO_ID = "zeahub/camus"


def transform_sc_image_to_polar(image_sc, output_size=None, fit_outline=True):
    """
    Transform a scan converted input image (cone) into square
        using radial stretching and downsampling. Note that it assumes the background to be zero!
        Please verify if your results make sense, especially if the image contains black parts
        at the edges. This function is not perfect by any means, but it works for most cases.

    Args:
        image (numpy.ndarray): Input image as a 2D numpy array (height, width).
        output_size (tuple, optional): Output size of the image as a tuple.
            Defaults to image_sc.shape.
        fit_outline (bool, optional): Whether to fit a polynomial the outline of the image.
            Defaults to True. If this is set to False, and the ultrasound image contains
            some black parts at the edges, weird artifacts can occur, because the jagged outline
            is stretched to the desired width.

    Returns:
        numpy.ndarray: Squared image as a 2D numpy array (height, width).
    """
    assert len(image_sc.shape) == 2, "function only allows for 2D data"

    # Default output size is the input size
    if output_size is None:
        output_size = image_sc.shape

    # Initialize an empty target array for polar_image
    polar_image = np.zeros_like(image_sc)

    # Flip along the x axis (such that curve of image_sc is pointing up)
    flipped_image = np.flip(image_sc, axis=0)

    # Find index of first non zero element along y axis (for every vertical line)
    non_zeros_flipped = find_first_nonzero_index(flipped_image, 0)

    # Remove any black vertical lines (columns) that do not contain image data
    remove_vertical_lines = np.where(non_zeros_flipped == -1)[0]
    polar_image = np.delete(polar_image, remove_vertical_lines, axis=1)
    non_zeros_flipped = np.delete(non_zeros_flipped, remove_vertical_lines)

    if fit_outline:
        model_fitted_bottom = np.poly1d(
            np.polyfit(range(len(non_zeros_flipped)), non_zeros_flipped, 4)
        )
        non_zeros_flipped = model_fitted_bottom(range(len(non_zeros_flipped)))
        non_zeros_flipped = non_zeros_flipped.round().astype(np.int64)
        non_zeros_flipped = np.clip(non_zeros_flipped, 0, None)

    non_zeros = polar_image.shape[0] - non_zeros_flipped

    # Find the middle of the width of the image
    width = polar_image.shape[1]
    width_middle = round(width / 2)

    # For every vertical line in the image
    for x_i in range(width):
        # Move the flipped first non-zero element to the bottom of the image
        polar_image[non_zeros_flipped[x_i] :, x_i] = image_sc[: non_zeros[x_i], x_i]

    # Find indices of first and last non-zero element along x axis (for every horizontal line)
    non_zeros_left = find_first_nonzero_index(polar_image, 1)
    non_zeros_right = width - find_first_nonzero_index(np.flip(polar_image, 1), 1, width_middle)

    # Remove any black horizontal lines (rows) that do not contain image data
    remove_horizontal_lines = np.max(np.where(non_zeros_left == -1)) + 1
    polar_image = polar_image[remove_horizontal_lines:, :]
    non_zeros_left = non_zeros_left[remove_horizontal_lines:]
    non_zeros_right = non_zeros_right[remove_horizontal_lines:]

    if fit_outline:
        model_fitted_left = np.poly1d(np.polyfit(range(len(non_zeros_left)), non_zeros_left, 2))
        non_zeros_left = model_fitted_left(range(len(non_zeros_left)))
        non_zeros_left = non_zeros_left.round().astype(np.int64)

        model_fitted_right = np.poly1d(np.polyfit(range(len(non_zeros_right)), non_zeros_right, 2))
        non_zeros_right = model_fitted_right(range(len(non_zeros_right)))
        non_zeros_right = non_zeros_right.round().astype(np.int64)

    # For every horizontal line in the image
    for y_i in range(polar_image.shape[0]):
        small_array = polar_image[y_i, non_zeros_left[y_i] : non_zeros_right[y_i]]

        if len(small_array) <= 1:
            # If the array is too small for interpolation, set it to the middle value.
            polar_image[y_i, :] = polar_image[y_i, width_middle]
        else:
            # Perform linear interpolation to stretch the line to the desired width.
            array_interp = scipy.interpolate.interp1d(np.arange(small_array.size), small_array)
            polar_image[y_i, :] = array_interp(np.linspace(0, small_array.size - 1, width))

    # Resize image to output_size
    return resize(polar_image, output_size, preserve_range=True)


def process_camus(source_path, output_path, overwrite=False):
    """Converts the camus database to the zea format.

    Args:
        source_path (str, pathlike): The path to the original camus file.
        output_path (str, pathlike): The path to the output file.
        overwrite (bool, optional): Set to True to overwrite existing file.
            Defaults to False.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)

    # Check if output file already exists and remove
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            log.warning("Output file %s already exists. Skipping.", log.yellow(output_path))
            return

    # Open the file
    image_seq, _ = sitk_load(source_path)

    # Convert to polar coordinates
    image_seq_polar = []
    for image in image_seq:
        image_seq_polar.append(transform_sc_image_to_polar(image))
    image_seq_polar = np.stack(image_seq_polar, axis=0)

    # Change range to [-60, 0] dB — keep as float32, not uint8
    image_seq = translate(image_seq, (0, 255), (-60, 0))
    image_seq_polar = translate(image_seq_polar, (0, 255), (-60, 0))

    # Add y dimension (elevation) — CAMUS is 2D, so y=1
    image_seq_polar = np.expand_dims(image_seq_polar, axis=-1)

    File.create(
        path=output_path,
        data={"image_sc": {"values": image_seq}, "image": {"values": image_seq_polar}},
        probe={"name": "GE M5S"},
        description="camus dataset converted to zea format",
    )


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
        camus_source_folder = unzip(camus_source_folder, "camus")

    # check if output folders already exist
    for split in splits:
        assert not (camus_output_folder / split).exists(), (
            f"Output folder {camus_output_folder / split} exists. Exiting program."
        )

    # clone folder structure of source to output using pathlib
    files = list(camus_source_folder.glob("**/*_half_sequence.nii.gz"))
    tasks = []
    for source_file in files:
        patient = source_file.stem.split("_")[0]
        patient_id = int(patient.removeprefix("patient"))
        split = get_split(patient_id)

        output_file = camus_output_folder / split / source_file.relative_to(camus_source_folder)
        # Replace .nii.gz with .hdf5
        output_file = output_file.with_suffix("").with_suffix(".hdf5")
        # make sure folder exists
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

    write_dataset_card(camus_output_folder, _CAMUS_DATASET_CARD)

    if getattr(args, "upload", False):
        upload_camus(camus_output_folder, revision=args.revision)


def upload_camus(output_folder: str | Path, revision: str) -> None:  # pragma: no cover
    """Upload the converted CAMUS dataset to a HuggingFace Hub revision branch.

    Only for zea maintainers with push access to the repository.  Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_folder: Root folder containing the train/val/test splits.
        revision: Target branch name on the Hub (must not be ``"main"``).
    """
    upload_dataset_to_hf(
        folder=output_folder,
        repo_id=_CAMUS_HF_REPO_ID,
        revision=revision,
        commit_message=f"Upload CAMUS dataset (zea format) to {revision}",
    )


_CAMUS_DATASET_CARD = (
    """\
---
license: cc-by-nc-sa-4.0
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

- `data/image_sc` - scan-converted B-mode sequence, shape `(n_frames, H, W)`
- `data/image` - polar-coordinate B-mode sequence, shape `(n_frames, H, W, 1)`

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
