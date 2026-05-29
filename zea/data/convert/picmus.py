"""Convert the PICMUS database to the zea format.

PICMUS (Plane-wave Imaging Challenge in Medical UltraSound) is a public dataset
released for the IEEE International Ultrasonics Symposium 2016 (IUS 2016, Tours,
France).  It contains simulated, experimental, and in-vivo plane-wave ultrasound
acquisitions captured with a Verasonics Vantage system and L11-4v probe, designed
for benchmarking image reconstruction algorithms.

The dataset comprises three partitions:

* **Simulation** - Synthetic contrast-speckle and resolution phantoms (IQ data).
* **Experimental** - Physical phantom measurements (IQ and RF data).
* **In-vivo** - Carotid artery acquisitions (IQ data).

.. admonition:: License

   The datasets and code provided on PICMUS are free of use.
   The only request is to refer properly to PICMUS - The Plane Wave Imaging Challenge
   in Medical UltraSound and quote the proceeding paper.

.. admonition:: Reference

   H. Liebgott, A. Rodriguez-Molares, F. Cervenansky, J. D'hooge and O. Bernard.
   *Plane-Wave Imaging Challenge in Medical Ultrasound.*
   2016 IEEE International Ultrasonics Symposium (IUS), Tours, France, 2016, pp. 1-4.
   `DOI: 10.1109/ULTSYM.2016.7728908 <https://doi.org/10.1109/ULTSYM.2016.7728908>`_

.. rubric:: Links

* `PICMUS Challenge website <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>`_
* `Download page <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download>`_
* `Dataset on Hugging Face <https://huggingface.co/datasets/zeahub/picmus>`_
  (simulation and experimental partitions only; in-vivo is not yet uploaded)

.. rubric:: Usage

.. code-block:: console

   python -m zea.data.convert picmus ./raw ./output --download
   python -m zea.data.convert picmus ./raw ./output

"""

import zipfile
from pathlib import Path

import h5py
import numpy as np

from zea import log
from zea.beamform.delays import compute_t0_delays_planewave
from zea.data.convert.utils import download_file, unzip, upload_dataset_to_hf, write_dataset_card
from zea.data.file import File

# ---------------------------------------------------------------------------
# Citation / license constants
# ---------------------------------------------------------------------------

PICMUS_CITATION = (
    "H. Liebgott, A. Rodriguez-Molares, F. Cervenansky, J. D'hooge, O. Bernard. "
    '"Plane-Wave Imaging Challenge in Medical Ultrasound." '
    "2016 IEEE International Ultrasonics Symposium (IUS), Tours, France, 2016, pp. 1-4. "
    "https://doi.org/10.1109/ULTSYM.2016.7728908"
)

PICMUS_LICENSE = (
    "The datasets and code provided on PICMUS are free of use. "
    "The only request is to refer properly to PICMUS - The Plane Wave Imaging Challenge "
    "in Medical UltraSound and quote the proceeding paper."
)

PICMUS_DESCRIPTION = (
    "PICMUS (Plane-wave Imaging Challenge in Medical UltraSound) dataset "
    "converted to zea format. "
    f"License: {PICMUS_LICENSE}. "
    f"Citation: {PICMUS_CITATION}"
)

_L11_4V_PROBE = {
    "name": "verasonics_l11_4v",
    "type": "linear",
    "center_frequency": np.float32(5.1333e6),
    "bandwidth_percent": np.float32(67.0),
    "element_width": np.float32(0.270e-3),
    "element_height": np.float32(5e-3),
}

# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------

_PICMUS_HF_REPO_ID = "zeahub/picmus"

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------

_PICMUS_ARCHIVE_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
    "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/archive_to_download.zip"
)

_PICMUS_INVIVO_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
    "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/in_vivo.zip"
)


def _infer_subject_type(source_path: Path) -> str | None:
    """Infer the subject type from the source file path.

    Returns ``"simulation"`` for synthetic data, ``"phantom"`` for
    experimental phantom data, ``"human"`` for in-vivo acquisitions,
    or ``None`` if the type cannot be determined.
    """
    str_path = str(source_path).lower()
    if "simulation" in str_path:
        return "simulation"
    if "experiment" in str_path:
        return "phantom"
    if "in_vivo" in str_path or "invivo" in str_path:
        return "human"
    return None


def _extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract a zip archive and return the path to its top-level folder.

    If the zip contains a single top-level directory, that directory is
    returned.  Otherwise ``destination`` itself is returned.

    Args:
        zip_path: Path to the zip file.
        destination: Directory to extract into.

    Returns:
        Path to the extracted content root.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        top_levels = {name.split("/")[0] for name in names if name.split("/")[0]}
        z.extractall(destination)

    if len(top_levels) == 1:
        return destination / top_levels.pop()
    return destination


def download_picmus(destination: str | Path) -> tuple[Path, Path]:  # pragma: no cover
    """Download the PICMUS main dataset and in-vivo partition.

    Downloads ``archive_to_download.zip`` (simulation + experimental data)
    and ``in_vivo.zip`` from the PICMUS challenge website and extracts both
    archives into *destination*.

    Args:
        destination: Directory where the archives will be downloaded and
            extracted.  Will be created if it does not exist.

    Returns:
        A tuple ``(main_dir, invivo_dir)`` where *main_dir* is the path to
        the extracted main dataset and *invivo_dir* is the path to the
        extracted in-vivo dataset.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main dataset (simulation + experimental)
    # ------------------------------------------------------------------
    main_zip_path = destination / "archive_to_download.zip"
    main_zip = download_file(_PICMUS_ARCHIVE_URL, main_zip_path)

    main_top = destination / "archive_to_download"
    if not main_top.exists():
        log.info("Extracting %s ...", main_zip.name)
        main_top = _extract_zip(main_zip, destination)
    log.info("Main PICMUS data at %s", log.yellow(main_top))

    # ------------------------------------------------------------------
    # In-vivo dataset
    # ------------------------------------------------------------------
    invivo_zip_path = destination / "in_vivo.zip"
    invivo_zip = download_file(_PICMUS_INVIVO_URL, invivo_zip_path)

    # The zip may extract to 'in_vivo/' or a differently-named folder;
    # _extract_zip will detect the top-level directory automatically.
    invivo_top = destination / "in_vivo"
    if not invivo_top.exists():
        log.info("Extracting %s ...", invivo_zip.name)
        invivo_top = _extract_zip(invivo_zip, destination)
        # If extraction did not create a folder named 'in_vivo', use whatever was created
        if not invivo_top.exists():
            # Fallback: find any newly created directory that is not the main one
            candidates = [
                d
                for d in destination.iterdir()
                if d.is_dir()
                and d.name not in ("archive_to_download",)
                and not d.name.endswith(".zip")
            ]
            if candidates:
                invivo_top = candidates[0]
    log.info("In-vivo PICMUS data at %s", log.yellow(invivo_top))

    return main_top, invivo_top


def convert(source_path, output_path, overwrite=False):
    """Convert and write a single PICMUS HDF5 file to the zea format.

    The subject type (``"simulation"``, ``"phantom"``, or ``"human"``) is
    inferred automatically from the source file path.

    Args:
        source_path (str, pathlike): Path to the original PICMUS HDF5 file.
        output_path (str, pathlike): Path for the output zea HDF5 file.
        overwrite (bool, optional): If True, overwrite any existing output
            file.  Defaults to False.
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
    with h5py.File(source_path, "r") as hdf:
        # Get the group containing the dataset
        ds = hdf["US"]["US_DATASET0000"]

        if "data" not in ds:
            raise ValueError("The file does not contain the data group.")

        # Extract I- and Q-data (shape (tx, el, ax))
        i_data = ds["data"]["real"][:]
        q_data = ds["data"]["imag"][:]

        if np.abs(np.sum(q_data)) < 0.1:
            # Use only the I-data, add dummy dimension (shape (tx, el, ax, ch=1))
            raw_data = i_data[..., None]
        else:
            # Stack I- and Q-data (shape (tx, el, ax, 2))
            raw_data = np.stack([i_data, q_data], axis=-1)

        # Add dummy frame dimension (shape (frame=1, tx, el, ax, ch=1))
        raw_data = raw_data[None]

        raw_data = np.transpose(raw_data, (0, 1, 3, 2, 4))

        _, n_tx, _, n_el, _ = raw_data.shape

        center_frequency = int(ds["modulation_frequency"][:][0])
        # Fix a mistake in one of the PICMUS files
        if center_frequency == 0:
            center_frequency = 5.208e6
        sampling_frequency = int(ds["sampling_frequency"][:][0])
        probe_geometry = np.transpose(ds["probe_geometry"][:], (1, 0))

        sound_speed = float(ds["sound_speed"][:][0])
        focus_distances = np.zeros((n_tx,), dtype=np.float32)
        polar_angles = ds["angles"][:]
        azimuth_angles = np.zeros((n_tx,), dtype=np.float32)
        t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
        tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)

        initial_times = np.zeros((n_tx,))
        for n in range(n_tx):
            v = np.array([np.sin(polar_angles[n]), 0, np.cos(0)])
            initial_times[n] = -np.min(np.sum(probe_geometry * v[None], axis=1)) / sound_speed

            t0_delays[n] = compute_t0_delays_planewave(
                probe_geometry=probe_geometry,
                polar_angles=polar_angles[n],
                sound_speed=sound_speed,
            )

    # Build per-file description and metadata
    subject_type = _infer_subject_type(source_path)
    metadata = {"credit": PICMUS_CITATION}
    if subject_type is not None:
        metadata["subject"] = {"type": subject_type}

    File.create(
        path=output_path,
        data={"raw_data": raw_data.astype(np.float32)},
        scan={
            "center_frequency": center_frequency,
            "demodulation_frequency": center_frequency,
            "sampling_frequency": sampling_frequency,
            "initial_times": initial_times,
            "sound_speed": sound_speed,
            "t0_delays": t0_delays,
            "focus_distances": focus_distances,
            "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
            "polar_angles": polar_angles,
            "azimuth_angles": azimuth_angles,
            "tx_apodizations": tx_apodizations,
        },
        metadata=metadata,
        probe={**_L11_4V_PROBE, "probe_geometry": probe_geometry},
        description=PICMUS_DESCRIPTION,
    )


def convert_picmus(args):
    """Convert PICMUS HDF5 files to the zea dataset format.

    Processes all matching files found in the source directory (and optionally
    the in-vivo partition), preserving the relative directory structure under
    the destination.  Each converted file is placed in its own sub-directory
    named after the file stem, which is required by the zea dataset format.

    Usage::

        # Download and convert everything (main + in-vivo)
        python -m zea.data.convert picmus <src> <dst> --download

        # Convert from manually extracted archives
        python -m zea.data.convert picmus <src> <dst>

    Args:
        args (argparse.Namespace): An object with the following attributes.

            - src (str or Path): Working directory.  When ``--download`` is
              given, both archives are downloaded here.  Otherwise this must
              contain ``archive_to_download/`` (or ``picmus.zip``) and
              optionally an ``in_vivo/`` sub-directory.
            - dst (str or Path): Output directory.  Converted files are
              written here, preserving relative paths.  Created if absent;
              existing files are skipped.
            - download (bool, optional): If True, automatically download the
              main dataset and the in-vivo partition before converting.

    Note:
        - Only files whose names end in ``iq.hdf5`` or ``rf.hdf5`` (and do
          not contain ``img``) are converted; all others are skipped.
        - Without ``--download``, an ``in_vivo/`` sub-directory found next to
          the main data directory is automatically included.
    """
    base_dir = Path(args.src)
    dst = Path(args.dst)

    # ------------------------------------------------------------------
    # Determine source directories
    # ------------------------------------------------------------------
    # Each entry is (source_dir, relative_to_dir) where output paths are
    # computed as: dst / file.relative_to(relative_to_dir).
    source_entries: list[tuple[Path, Path]] = []

    if getattr(args, "download", False):
        main_dir, invivo_dir = download_picmus(base_dir)
        source_entries.append((main_dir, main_dir))
        source_entries.append((invivo_dir, invivo_dir.parent))
    else:
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Source directory {base_dir} does not exist. "
                "Use --download to download the PICMUS dataset automatically."
            )
        main_dir = unzip(base_dir, "picmus")
        source_entries.append((main_dir, main_dir))

        # Include in-vivo data when an 'in_vivo' sub-directory is present
        invivo_dir = base_dir / "in_vivo"
        if invivo_dir.exists():
            log.info("Found in-vivo data at %s", log.yellow(invivo_dir))
            source_entries.append((invivo_dir, invivo_dir.parent))

    dst.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convert all matching files
    # ------------------------------------------------------------------
    for source_dir, relative_to in source_entries:
        for file in source_dir.rglob("*.hdf5"):
            str_file = str(file)

            # Select only data files containing IQ or RF data
            is_data_file = str_file.endswith("iq.hdf5") or str_file.endswith("rf.hdf5")
            if not is_data_file or "img" in str_file:
                log.info("Skipping %s as it does not contain IQ or RF data", log.yellow(file.name))
                continue

            log.info("Converting %s", log.yellow(file.name))

            # Preserve relative directory structure under dst; each file gets
            # its own sub-directory so the folder can be used as a dataset.
            output_file = dst / file.relative_to(relative_to)
            output_file = output_file.parent / output_file.stem / f"{output_file.stem}.hdf5"

            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                convert(file, output_file, overwrite=True)
            except Exception:
                try:
                    output_file.parent.rmdir()
                except OSError:
                    pass
                log.error("Failed to convert %s", log.yellow(str_file))
                continue

    log.info(f"Finished converting PICMUS dataset. Output written to {log.yellow(dst)}")

    write_dataset_card(dst, _PICMUS_DATASET_CARD)

    if getattr(args, "upload", False):
        upload_picmus(dst, revision=args.revision)


def upload_picmus(output_folder: str | Path, revision: str) -> None:  # pragma: no cover
    """Upload the converted PICMUS dataset to a HuggingFace Hub revision branch.

    Only for zea maintainers with push access to the repository.  Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_folder: Root folder containing the converted HDF5 files.
        revision: Target branch name on the Hub (must not be ``"main"``).
    """
    upload_dataset_to_hf(
        folder=output_folder,
        repo_id=_PICMUS_HF_REPO_ID,
        revision=revision,
        commit_message=f"Upload PICMUS dataset (zea format) to {revision}",
    )


_PICMUS_DATASET_CARD = (
    """\
---
license: other
license_name: picmus-free-use
license_link: https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/
task_categories:
  - other
tags:
  - ultrasound
  - plane-wave
  - medical
  - beamforming
  - ius2016
pretty_name: "PICMUS: Plane-Wave Imaging Challenge in Medical UltraSound"
size_categories:
  - n<1K
---

# PICMUS - Plane-Wave Imaging Challenge in Medical UltraSound

This dataset is a **zea-format** (HDF5) conversion of the
[PICMUS (IEEE IUS 2016)](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)
challenge data for evaluating plane-wave image reconstruction quality.

| Property | Value |
|---|---|
| **Modality** | 2-D plane-wave ultrasound (IQ and RF) |
| **Probe** | Verasonics Vantage + L11-4v (128 elements, 5.208 MHz) |
| **Partitions** | Simulation, Experimental, In-vivo (carotid artery) |
| **Data types** | IQ (complex) and RF (real) channel data |
| **Splits** | simulation/, database/ (experimental), in_vivo/ |

## Conversion

This dataset was downloaded, converted to zea format, and uploaded using the
[zea](https://github.com/tue-bmd/zea) data converter:

```bash
python -m zea.data.convert picmus <src> <dst> --download
```

## Dataset structure

```
simulation/
  contrast_speckle/
    contrast_speckle_simu_dataset_iq/
      contrast_speckle_simu_dataset_iq.hdf5
  ...
database/
  experiments/
    contrast_speckle/
      contrast_speckle_expe_dataset_iq/...
      contrast_speckle_expe_dataset_rf/...
    ...
in_vivo/
  carotid_long/
    carotid_long_expe_dataset_iq/...
  carotid_cross/
    ...
```

Each HDF5 file follows the [zea data format](https://github.com/tue-bmd/zea) and contains:

- `data/raw_data` - channel data, shape `(1, n_tx, n_ax, n_el, n_ch)`
  where `n_ch=1` for RF data and `n_ch=2` (I/Q) for IQ data
- `scan/` - plane-wave acquisition parameters (angles, delays, geometry, …)
- `metadata/` - subject type and citation credit

## License

"""
    + PICMUS_LICENSE
    + """

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{liebgott2016picmus,
  title     = {Plane-Wave Imaging Challenge in Medical Ultrasound},
  author    = {Liebgott, Herve and Rodriguez-Molares, Alfonso and
               Cervenansky, Frederic and D'hooge, Jan and Bernard, Olivier},
  booktitle = {2016 IEEE International Ultrasonics Symposium (IUS)},
  pages     = {1--4},
  year      = {2016},
  doi       = {10.1109/ULTSYM.2016.7728908}
}
```

## Links

- **PICMUS challenge**: <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>
- **Download page**: <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download>
- **zea toolkit**: <https://github.com/tue-bmd/zea>
"""
)
