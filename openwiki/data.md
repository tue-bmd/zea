# Data & file format

The `zea` data layer (`zea/data/`) is built around one idea: **store raw ultrasound data
alongside every parameter needed to process it, plus metadata, in a single self-describing HDF5
file** (`docs/source/data-acquisition.rst`). This makes acquisitions portable and guarantees the
processing parameters are never separated from the data. The format is also designed for partial
access — you can read a single frame or transmit without loading the whole file, which supports
the cognitive-ultrasound loop (see [agent.md](agent.md)).

## The zea file format

Each acquisition is one HDF5 file following the schema defined in `zea/data/spec.py`. Parameters
are split into two groups (`docs/source/config.rst`):

- **`probe/`** — fixed for the whole acquisition: element geometry, center frequency, bandwidth,
  lens properties. Defined by `ProbeSpec`.
- **`scan/`** — the per-track transmit sequence: `t0_delays`, `tx_apodizations`, `initial_times`,
  `focus_distances`, angles, waveforms, sound speed. Defined by `ScanSpec`.

A file may contain multiple **tracks** (e.g. an RF `raw_data` product next to an IQ
`beamformed_data` product). `spec.py` enforces cross-product **dimension consistency** for named
dims (`n_frames`, `n_tx`, `n_ax`, `n_el`, `n_ch`), while channel counts stay local to each product
(`CONSISTENCY_DIMENSIONS` / `LOCAL_CONSISTENCY_DIMENSIONS` in `zea/data/spec.py`).

## Reading and writing files — `zea.File`

`zea.File` (`zea/data/file.py`) behaves like `h5py.File` but adds parameter parsing and schema
validation (`docs/source/data-acquisition.rst`):

```python
from zea import File
with File("my_acquisition.hdf5") as f:
    raw        = f.data.raw_data[:]      # lazy HDF5 slicing
    raw0       = f.data.raw_data[0]      # one frame
    parameters = f.load_parameters()     # merged probe + scan -> zea.Parameters
    scan       = f.scan                  # ScanSpec
    probe      = f.probe                 # zea.Probe
```

`File.create(...)` builds a validated file from NumPy arrays, checking everything against the full
schema before writing (`docs/source/data-acquisition.rst`). Remote `hf://…` paths are resolved
against the Hugging Face Hub transparently. `legacy_file.py` handles older on-disk formats.

## Parameters, probes

`File.load_parameters()` merges the probe and scan groups into a single `zea.Parameters` object
(`zea/parameters.py`) and adds derived quantities: `wavelength`, `n_tx`, `grid`, `xlims`/`zlims`,
`selected_transmits`. For multi-track files, use `f.tracks[0].load_parameters()`
(`docs/source/config.rst`).

`zea.Probe` (`zea/probes.py`) is a container for transducer parameters, subclassing `ProbeSpec`.
Concrete probes (e.g. `Verasonics_l11_4v`, `Verasonics_l11_5v`) register with `probe_registry`, so
a probe can be requested by name. Register a custom `Probe` subclass the same way.

## Datasets and dataloaders

For collections of files (`zea/data/datasets.py`, `zea/data/dataloader.py`):

- **`Dataset`** — manages many `File` objects over a folder, with an `H5FileHandleCache` to avoid
  reopening files; supports filtering datasets by file content (`git log`, `zea/data/datasets.py`).
- **`Folder`** — mostly internal; `Dataset` is the user-facing entry point.
- **`Dataloader`** — deep-learning-oriented batching/iteration over a dataset, backed by an
  `H5DataSource` and the [`grain`](https://github.com/google/grain) data pipeline library
  (`pyproject.toml` dependency, `zea/data/dataloader.py`).
- **`augmentations.py`** — data augmentation ops for training.

## Importing external datasets — `zea.data.convert`

`zea/data/convert/` converts common public ultrasound datasets into the zea format, with a
converter per source: `picmus.py`, `camus.py`, `cetus.py`, `echonet.py`, `echonetlvh/`,
`echoxflow.py`, `verasonics.py`, `images.py`. Run via the CLI:

```
python -m zea.data.convert <dataset> [options]
```

(see `docs/source/cli.rst` and [cli-and-config.md](cli-and-config.md)). `update_hf_dataset.py`
pushes/refreshes datasets on the Hugging Face Hub. To add a new source, add a converter module and
register the dataset in `dataset_registry` (recording `probe_used` / `scan_class`).

## Simulator

`zea/simulator.py` provides a frequency-domain (RFFT) RF simulator: it models RF data as a
superposition of scatterer responses, each with a location and magnitude. Call
`simulate_rf(...)` with transmit-scheme parameters and scatterers; loop over frames and stack for
sequences (`zea/simulator.py`). Useful for generating synthetic ground-truth data without hardware.

## Where to look / what to watch

- The schema is authoritative: changes to fields, dtypes, or dimension rules belong in
  `zea/data/spec.py` and are validated on both read and `File.create`.
- Relevant tests: `tests/test_io_lib.py`, `tests/test_object.py`, `tests/test_parameters.py`,
  `tests/test_probes.py`, `tests/test_configs.py`, and `tests/data/`.
