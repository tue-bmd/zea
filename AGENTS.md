# AGENTS.md

> Agent-facing documentation for the **zea** package — *A Toolbox for Cognitive Ultrasound Imaging.*

## Project overview

`zea` is a research-oriented Python toolbox for cognitive ultrasound imaging. It provides a flexible, modular, and differentiable image-formation pipeline built on modern deep-learning primitives so researchers can experiment with end-to-end learnable acquisition, reconstruction, and task-driven imaging strategies.

Key capabilities:
- Differentiable beamforming primitives (TOF correction, delay-and-sum variants, pressure-field weighting) and memory-efficient processing for large tensors.
- A composable `Operation`/`Pipeline` API for building custom processing graphs and serializing them from YAML/JSON or `Config` objects.
- Data loading and management for the project HDF5 format, with `File`/`Dataset` helpers and optional `hf://` (HuggingFace Hub) access for remote datasets.
- A preset-driven pretrained model system and model-presets for common ultrasound tasks (segmentation, quality scoring, generative models).
- Tools for cognitive/adaptive workflows: action-selection strategies, differentiable loss/AutoGrad tooling, simulators, and selection utilities.

Built on **Keras 3**, `zea` supports the main deep-learning backends (select via `KERAS_BACKEND`): PyTorch, TensorFlow, and JAX.

Quick references:
- **Paper / Citation:** Stevens et al., "zea: A Toolbox for Cognitive Ultrasound Imaging", arXiv:2512.01433 — https://arxiv.org/abs/2512.01433
- **Docs:** <https://zea.readthedocs.io>
- **Source:** <https://github.com/tue-bmd/zea>
- **PyPI:** `pip install zea`
- **Python:** ≥3.10
- **Status:** Beta / experimental

Top-level imports:

```python
from zea import Config, Probe, Scan, Pipeline, File, Dataset, Interface
```

---

## Setup commands

```bash
# Install with default backend (JAX)
pip install zea[jax]

# Or install with PyTorch / TensorFlow
pip install zea[torch]
pip install zea[tensorflow]

# Install for development (includes tests, docs, linting)
pip install -e ".[dev]"

# Set backend (before importing zea or keras)
export KERAS_BACKEND=jax      # or torch, tensorflow
```

The Dockerfile supports multi-backend installs:

```bash
docker build -t zea/all:latest --build-arg INSTALL_JAX=cpu --build-arg INSTALL_TORCH=cpu --build-arg INSTALL_TF=cpu .
```

---

## Testing

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_beamformer.py

# Run a single test
pytest tests/test_beamformer.py -k "test_name"

# Skip slow/heavy tests
pytest -m "not heavy and not performance"

# Run notebook tests (requires --notebook flag)
pytest tests/test_notebooks.py --notebook
```

Test markers: `performance`, `heavy`, `notebook`.

Test configuration is in `pytest.ini`. Test fixtures (HDF5 dummy data, backend workers, etc.) are in `tests/conftest.py`. Backend environment (`KERAS_BACKEND=tensorflow`, `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`) is set automatically in `tests/__init__.py`.

---

## Code style

- **Formatter/linter:** `ruff` (configured via `pyproject.toml`)
- **Pre-commit hooks:** configured (`.pre-commit-config.yaml`)
- **Docstrings:** Google-style (see `docs/example_google_docstrings.py`)
- **Type annotations:** used throughout; the project uses `typing` and modern union syntax (`X | None`)

---

## Architecture overview

### Multi-backend system (Keras 3)

All tensor operations go through Keras 3 ops (`keras.ops`). The backend is selected at import time via `KERAS_BACKEND`. The `zea.backend` module provides:

- `jit(func)` — unified JIT compilation dispatching to `jax.jit` or `tf.function`
- `AutoGrad` — backend-agnostic automatic differentiation (supports `torch.autograd`, `tf.GradientTape`, `jax.grad`)
- `on_device(device)` — context manager for device placement

When writing new code, always use `keras.ops` for tensor operations instead of calling backend-specific APIs directly.

### Internal registries

The codebase uses registries (`zea.internal.registry`) for operations, probes, metrics, and action selection strategies:

- `@ops_registry("name")` — register an Operation class
- `@probe_registry("name")` — register a Probe
- `@metrics_registry("name")` — register a metric function
- `@action_selection_registry("name")` — register an agent action selection strategy

---

## Core concepts

### Parameters (`zea.internal.parameters.Parameters`)

The `Parameters` base class provides:

- **Validated leaf parameters:** Subclasses define `VALID_PARAMS` dict mapping param names to `{"type": ..., "default": ...}`. Only leaf params can be set directly.
- **Lazy computed properties:** Decorated with `@cache_with_dependencies(*deps)`. Only computed on access; cached and invalidated when dependencies change.
- **Dependency tracking and cache invalidation:** Changing a leaf parameter automatically invalidates all downstream computed properties.
- **Efficient updates:** `update(force=False, **kwargs)` skips unchanged values and therefore does **not** invalidate/recompute the dependency tree for those keys. Changed values (or all values with `force=True`) are set via `setattr`, which triggers normal invalidation so dependent computed properties are recomputed on next access.
- **Optional dependency parameters:** A parameter can be both a leaf in `VALID_PARAMS` and a computed property. If explicitly set, that value is used; if unset (or set to `None`), it falls back to the computed value from dependencies.
- **Tensor conversion:** `to_tensor()` converts all parameters to ML tensors for use in pipelines.

Example interaction:

```python
scan = Scan(sound_speed=1540.0, center_frequency=5e6)
scan.wavelength  # computed property: sound_speed / center_frequency = 0.000308
scan.sound_speed = 1500.0  # invalidates wavelength cache
scan.wavelength  # recomputed automatically
```

```python
scan = Scan(sound_speed=1540.0, center_frequency=5e6)
_ = scan.wavelength # compute and cache dependency

scan.update(sound_speed=1540.0)  # unchanged: skips setattr, no invalidation/recompute
scan.update(sound_speed=1500.0)  # changed: invalidates dependents, recompute on next access
scan.update(force=True, sound_speed=1500.0)  # force-set: always invalidates dependents
```

Setting a computed property that is not a `VALID_PARAMS` leaf raises `ValueError` with guidance on which leaf params to modify.

### Scan (`zea.scan.Scan`)

`Scan` inherits from `Parameters` and represents a complete ultrasound scan configuration: pixel grid, coordinate system, and acquisition parameters. It has ~40 parameters in these categories:

- **Grid:** `grid_size_x`, `grid_size_z`, `xlims`, `zlims`, `pixels_per_wavelength`, `grid_type` ("cartesian"/"polar")
- **Acquisition:** `sound_speed` (default 1540.0 m/s), `sampling_frequency`, `center_frequency`, `n_el`, `n_tx`, `n_ax`, `n_ch`, `bandwidth_percent`, `f_number`
- **Probe/array:** `probe_geometry` (shape `(n_el, 3)`), `polar_angles`, `azimuth_angles`, `t0_delays`, `tx_apodizations`, `focus_distances`, `transmit_origins`
- **Scan conversion:** `theta_range`, `phi_range`, `rho_range`

Key computed properties: `grid` (shape `(nz, nx, 3)`), `flatgrid` (shape `(n_pix, 3)`), `wavelength`, `extent`.

Key method: `set_transmits(selection)` — accepts `"all"`, `"center"`, `"focused"`, `"diverging"`, `"plane"`, an int (evenly spaced), a list, a slice, or `np.ndarray`.

```python
from zea import Scan
scan = Scan(sound_speed=1540.0, center_frequency=5e6, grid_size_x=256, grid_size_z=256,
            xlims=[-0.02, 0.02], zlims=[0.005, 0.05])
grid = scan.grid  # shape (256, 256, 3) pixel positions in meters
scan.set_transmits("all")
```

### Probe (`zea.probes.Probe`)

Represents an ultrasound transducer probe.

```python
from zea import Probe
probe = Probe(probe_geometry=geometry, center_frequency=5e6, sampling_frequency=20e6)
# Or from a registered preset
probe = Probe.from_name("verasonics_l11_4v")
```

Registered probes: `generic`, `verasonics_l11_4v`, `verasonics_l11_5v`, `esaote_sll1543`.

### Config (`zea.config.Config`)

`Config` extends `dict` with dot-notation access, YAML/JSON serialization, and HuggingFace Hub integration.

```python
from zea import Config
config = Config.from_path("configs/config_picmus_rf.yaml")
# or from HuggingFace
config = Config.from_path("hf://zeahub/picmus/...")
config.data.dtype  # dot-access
config.scan.grid_size_x  # nested dot-access
```

Config YAML structure has three main sections: `data`, `scan`, and `pipeline`. See `configs/` for examples.

---

## Operations and Pipelines

### Operations (`zea.ops`)

All operations inherit from `Operation(keras.Operation)`. An operation reads from an input `key` (default `"data"`) and writes to an `output_key`.

**Ultrasound-specific operations** (`zea.ops.ultrasound`):

| Operation | Purpose |
|-----------|---------|
| `Demodulate` | RF → baseband IQ demodulation |
| `EnvelopeDetect` | Envelope detection of RF/IQ signals |
| `LogCompress` | Log compression with dynamic range clipping |
| `Normalize` | Normalize to a value range |
| `TOFCorrection` | Time-of-flight correction (delay calculation) |
| `PfieldWeighting` | Weight data with pressure field |
| `ReshapeGrid` | Reshape flat pixel data to grid shape |
| `ApplyWindow` | Apply window function to zero edges |
| `Downsample` | Downsample along axis |
| `BandPassFilter` | FIR band-pass filter |
| `ScanConvert` | Polar → Cartesian scan conversion |
| `Simulate` | Simulate RF data from scatterers |
| `Companding` | μ-law or A-law companding |
| `AnisotropicDiffusion` | SRAD speckle reduction |
| `CommonMidpointPhaseError` | Phase error for autofocusing |

**Tensor operations** (`zea.ops.tensor`): `Normalize`, `GaussianBlur`, `Pad`, `Threshold`.

**Keras ops** (`zea.ops.keras_ops`): All unary `keras.ops` functions are auto-wrapped as operations (e.g., `Squeeze`, `Abs`, `Reshape`, etc.).

**Composite operations:** `Beamform` (TOFCorrection + sum + ReshapeGrid), `DelayAndSum`, `DelayMultiplyAndSum`, `PatchedGrid` (memory-efficient grid processing), `Map` (batched/chunked processing via `vmap`).

**When to use `ReshapeGrid`:** Use it when your data has a flattened pixel axis (`n_pix`) and downstream operations expect image/grid layout (`grid_size_z x grid_size_x`, or polar-equivalent grid shape). This is typically needed after TOF-based beamforming steps that operate on `flatgrid`. It is usually not needed if your data is already in grid/image shape.

Stand-alone usage:

```python
from zea.ops import EnvelopeDetect
import numpy as np
data = np.random.randn(2000, 128, 1)
op = EnvelopeDetect(axis=-1)
result = op(data=data)
```

### Pipeline (`zea.ops.Pipeline`)

A `Pipeline` is an ordered sequence of `Operation` instances. It validates data type compatibility between consecutive operations.

**Creating pipelines:**

```python
from zea.ops import Pipeline, EnvelopeDetect, Normalize, LogCompress

# Default pipeline (beamform → envelope → normalize → log compress)
pipeline = Pipeline.from_default()

# Custom pipeline
pipeline = Pipeline([EnvelopeDetect(), Normalize(), LogCompress()])

# From config / YAML
pipeline = Pipeline.from_config(config)
pipeline = Pipeline.from_path("configs/config_picmus_rf.yaml")
pipeline = Pipeline.from_json('{"pipeline": {"operations": ["identity"]}}')
```

**JIT options:**
- `jit_options="ops"` (default) — JIT-compile each operation separately
- `jit_options="pipeline"` — compile the entire pipeline as one function (faster but no caching)
- `jit_options=None` — disable JIT

**Running a pipeline:**

```python
# Prepare scan/probe parameters as tensors and merge
params = pipeline.prepare_parameters(probe, scan, config.scan)
result = pipeline(return_numpy=True, **params)
```

**Pipeline methods:** `prepend()`, `append()`, `insert()`, `copy()`, `to_yaml()`, `to_json()`, `get_dict()`, `set_params()`, `get_params()`.

**Data types flow** (`zea.internal.core.DataTypes`): `RAW_DATA` → `ALIGNED_DATA` → `BEAMFORMED_DATA` → `ENVELOPE_DATA` → `IMAGE` → `IMAGE_SC`.

### YAML pipeline configuration

```yaml
pipeline:
  operations:
    - name: demodulate
    - name: downsample
      params:
        factor: 4
    - name: beamform
      params:
        beamformer: delay_and_sum
        enable_pfield: false
        num_patches: 100
    - name: envelope_detect
    - name: normalize
    - name: log_compress
```

---

## Beamforming

The core beamforming pipeline converts raw channel data into an ultrasound image:

1. **Demodulate** (optional) — RF → IQ
2. **TOFCorrection** — compute transmit/receive delays, interpolate channel data onto the pixel grid. Output shape: `(n_tx, n_pix, n_el, n_ch)`.
3. **Sum** — coherent summation across elements and transmits → beamformed image
4. **ReshapeGrid** — reshape flat pixel dimension to 2D grid

Key functions in `zea.beamform`:
- `tof_correction(data, flatgrid, t0_delays, ..., sound_speed, probe_geometry, ...)` — the main delay-and-sum kernel
- `calculate_delays(flatgrid, probe_geometry, ...)` → `(txdel, rxdel)`
- `compute_t0_delays_planewave(...)` / `compute_t0_delays_focused(...)` — transmit delay calculation
- `compute_pfield(...)` — pressure field for weighting

Beamforming supports: homogeneous and heterogeneous media, lens correction, F-number masking with window functions (rect, hann, tukey), phase rotation for IQ data, 2D and 3D grids, and patched grid processing for memory efficiency.

---

## Data loading

### File (`zea.data.File`)

Wraps `h5py.File` for zea's HDF5 data format. Supports `hf://` paths for HuggingFace Hub.

```python
from zea.data import File
f = File("path/to/data.hdf5")     # or "hf://zeahub/picmus/..."
data = f.load_data(dtype="raw_data")
scan = f.scan()        # → Scan object
probe = f.probe()      # → Probe object
f.summary()            # print file contents
```

### Dataset (`zea.data.Dataset`)

Manages collections of HDF5 files from a directory or HuggingFace Hub.

### Data format

HDF5 files follow this structure:
- `scan/` group — acquisition parameters (probe geometry, frequencies, delays, etc.)
- `data/` group — signal data (`raw_data`, `rf_data`, `iq_data`, `beamformed_data`, `image`, etc.)

---

## End-to-end image formation example

A typical beamforming workflow:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # set before any imports

from zea import Config, Pipeline
from zea.data import File

# Load config and data
config = Config.from_path("configs/config_picmus_rf.yaml")
f = File(config.data.dataset_folder + "/" + config.data.file_path)

# Get probe and scan from file
probe = f.probe()
scan = f.scan(**config.scan)

# Load raw data
data = f.load_data(dtype=config.data.dtype)

# Create pipeline and prepare parameters
pipeline = Pipeline.from_config(config)
params = pipeline.prepare_parameters(probe, scan, config.scan)

# Run the pipeline
image = pipeline(data=data, return_numpy=True, **params)
```

---

## Models

Pretrained models are loaded via the preset system:

```python
from zea.models import UNet, EchoNetDynamic, CarotidSegmenter
model = UNet.from_preset("unet-echonet-segmentation")
```

Available model families: `UNet`, `EchoNetDynamic`, `EchoNetLVH`, `CarotidSegmenter`, `LPIPS`, `TinyAutoencoder`, `DiffusionModel`, `HierarchicalVAE`, `GaussianMixtureModel`, `DeepLabV3`, `MobileNetv2RegionalQuality`, `AugmentedCamusSeg`.

Weights are stored on HuggingFace (`config.json` + `model.weights.h5`).

---

## Agent module (`zea.agent`)

Action selection strategies for cognitive (adaptive) ultrasound:

- `GreedyEntropy` — max entropy line selection
- `UniformRandomLines` — random sampling
- `EquispacedLines` — equispaced sweep
- `CovarianceSamplingLines` — correlation-based selection
- `TaskBasedLines` — task-driven information gain maximization (uses `AutoGrad`)

All inherit `LinesActionModel`. Key method: `sample(particles) → (lines, mask)`.

---

## Differentiable pipelines and AutoGrad

The pipeline is fully differentiable (all ops run through Keras). For gradient-based optimization:

```python
from zea.backend import AutoGrad

ag = AutoGrad()
ag.set_function(my_loss_fn)
grads = ag.gradient(variable, **kwargs)
grads, value = ag.gradient_and_value(variable, **kwargs)

# JIT-compiled gradient functions
grad_fn = ag.get_gradient_jit_fn()
```

---

## Other utilities

- **Metrics** (`zea.metrics`): `cnr`, `contrast`, `gcnr`, `fwhm`, `snr`, `wopt_mae`, `wopt_mse`, LPIPS
- **Simulator** (`zea.simulator`): `simulate_rf(scatterer_positions, scatterer_magnitudes, ...)` — frequency-domain RF simulation
- **Doppler** (`zea.doppler`): `color_doppler(data, ...)` — color Doppler from IQ packets
- **Tracking** (`zea.tracking`): `LucasKanadeTracker`, `SegmentationTracker` for speckle tracking
- **Display** (`zea.display`, `zea.visualize`): matplotlib and OpenCV viewers
- **CLI**: `python -m zea --config path/to/config.yaml --task view`

---

## Project structure

```
zea/
├── ops/           # Operations, Pipeline, beamform composites
├── beamform/      # Delay computation, TOF correction, pixel grids, pressure fields
├── data/          # HDF5 file I/O, datasets, dataloaders, augmentations
├── models/        # Pretrained model definitions and presets
├── agent/         # Action selection for cognitive ultrasound
├── backend/       # Multi-backend support (JAX, TF, PyTorch), AutoGrad, JIT
├── func/          # Low-level functional implementations (tensor, ultrasound)
├── internal/      # Registries, parameter system, config validation, caching
├── tracking/      # Speckle tracking (Lucas-Kanade, segmentation-based)
├── tools/         # HuggingFace utils, selection tool, W&B integration
├── scan.py        # Scan class (pixel grid, acquisition params)
├── probes.py      # Probe class (transducer definitions)
├── config.py      # Config class (YAML/JSON with dot-access)
├── interface.py   # High-level Interface for loading, processing, displaying
├── metrics.py     # Image quality metrics
├── simulator.py   # RF data simulator
├── doppler.py     # Color Doppler
├── io_lib.py      # I/O utilities
└── __main__.py    # CLI entry point
tests/             # pytest tests (mirrors zea/ structure)
configs/           # Example YAML configurations
docs/source/notebooks/  # Jupyter notebook examples
```

---

## Example notebooks

Notebooks in `docs/source/notebooks/` provide hands-on examples:

- **Pipeline:** `pipeline/zea_pipeline_example.ipynb` (basic beamforming), `pipeline/zea_sequence_example.ipynb` (sequence processing), `pipeline/polar_grid_example.ipynb`, `pipeline/3d_beamforming_example.ipynb`, `pipeline/doppler_example.ipynb`
- **Data:** `data/zea_data_example.ipynb` (loading HF data), `data/zea_local_data.ipynb`, `data/zea_simulation_example.ipynb`
- **Models:** `models/unet_example.ipynb`, `models/diffusion_model_example.ipynb`, `models/carotid_segmentation_example.ipynb`, `models/speckle_tracking_example.ipynb`
- **Agent:** `agent/agent_example.ipynb`, `agent/task_based_perception_action_loop.ipynb`
- **Metrics:** `metrics/lpips_example.ipynb`, `metrics/myocardial_quality_example.ipynb`

---

## Conventions and tips

- Always set `KERAS_BACKEND` **before** importing `zea` or `keras`.
- Use `keras.ops` for tensor operations, never call backend APIs directly.
- Operations use dictionary-based I/O: `key` selects the input, `output_key` selects where to write. Chain operations by matching keys.
- `Scan` and `Probe` objects must be converted to tensor dicts before passing to a pipeline — use `pipeline.prepare_parameters(probe, scan, config.scan)`.
- New operations should inherit from `Operation` and implement `call(**kwargs)`. Register with `@ops_registry("name")`.
- New probes should be registered with `@probe_registry("name")`.
- Config YAML files in `configs/` are auto-documented from `PARAMETER_DESCRIPTIONS`.
- Data files use HDF5 format. Remote data is accessed via `hf://` prefix (HuggingFace Hub).
- The `DataTypes` enum defines the pipeline data flow: `RAW_DATA → ALIGNED_DATA → BEAMFORMED_DATA → ENVELOPE_DATA → IMAGE → IMAGE_SC`.
- Parameters that are constant during JIT compilation should be listed in the operation's `STATIC_PARAMS` class attribute.
