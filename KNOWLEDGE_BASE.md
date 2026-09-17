# Knowledge Base: tue-bmd/zea

Generated: 2026-09-18

## Project Summary

`zea` is a Python library for cognitive ultrasound imaging that provides ultrasound signal processing, image reconstruction, and deep learning capabilities. Developed by the Biomedical Engineering department at TU Eindhoven, it serves researchers and developers working on ultrasound imaging applications. The library abstracts ultrasound processing pipelines across multiple deep learning backends (PyTorch, TensorFlow, JAX) via Keras 3, enabling framework-agnostic ultrasound research and development.

## Architecture

The codebase is organized into clear functional modules:

- **`zea/`** - Main package directory (~64k lines of Python)
  - **`ops/`** - Core operations including Pipeline, tensor operations, and ultrasound-specific operations
  - **`data/`** - Data loading, HDF5 file I/O, datasets, augmentations, and data specs
  - **`beamform/`** - Beamforming algorithms, delay calculations, and pixel grid management
  - **`models/`** - Pretrained models (ABLE, diffusion, segmentation, EchoNet variants, etc.)
  - **`agent/`** - Action selection functions for cognitive ultrasound (Gumbel softmax, masks)
  - **`backend/`** - Backend abstraction layer (autograd, optimizer, TF-to-JAX conversion)
  - **`func/`** - Functional utilities and tensor operations
  - **`tracking/`** - Tracking algorithms
  - **`tools/`** - Utilities including HuggingFace Hub integration and selection tools
  - **`internal/`** - Internal utilities (registry, caching, precision, device management, type definitions)
- **`tests/`** - Comprehensive test suite with unit, heavy (GPU), and notebook tests
- **`docs/`** - Sphinx documentation source (RST + Jupyter notebooks)
- **`configs/`** - YAML configuration files for datasets (PICMUS, CAMUS, EchoNet, etc.)
- **`scripts/`** - Utility scripts
- **`paper/`** - Research paper materials

## Key Files & Classes

**Core:**
- `zea/__init__.py` - Package entry point with backend bootstrapping logic
- `zea/config.py` - `Config` class for managing configuration with dot notation access, YAML/JSON serialization
- `zea/parameters.py` - `Parameters` class for ultrasound acquisition parameters with lazy evaluation and dependency tracking
- `zea/probes.py` - `Probe` class defining transducer geometry and properties
- `zea/ops/pipeline.py` - `Pipeline` class, the main beamforming/reconstruction pipeline
- `zea/ops/base.py` - `Operation` base class for all pipeline operations
- `zea/beamform/beamformer.py` - Core beamforming functions (TOF correction, delay calculations)

**Data:**
- `zea/data/file.py` - `File` class for HDF5-based zea data files
- `zea/data/datasets.py` - `Dataset` class and dataset registry
- `zea/data/dataloader.py` - `Dataloader` for batched data loading with metadata support
- `zea/data/metadata.py` - **Metadata handling utilities** for selective loading, slicing, and dimension-aware indexing
- `zea/data/spec.py` - Validation specs for data, metadata, probes, scans
- `zea/data/augmentations.py` - Data augmentation layers
- `zea/io_lib.py` - I/O utilities for various ultrasound data formats

**Models:**
- `zea/models/base.py` - Base model class
- `zea/models/able.py` - ABLE model for ultrasound beamforming
- `zea/models/echonet.py`, `echonetlvh.py` - EchoNet segmentation models
- `zea/models/diffusion.py` - Diffusion models

**Backend & Utilities:**
- `zea/backend/__init__.py` - Backend switching and device management
- `zea/backend/autograd.py` - `AutoGrad` class for automatic differentiation
- `zea/internal/device.py` - `init_device()` for GPU/CPU device configuration
- `zea/internal/registry.py` - Registration decorators for models, operations, datasets
- `zea/utils.py` - General utilities (progress bars, date handling, dict operations)
- `zea/log.py` - Logging infrastructure
- `zea/visualize.py`, `display.py` - Visualization and display utilities

**Tests:**
- `tests/conftest.py` - Pytest fixtures and test configuration
- `tests/backend_utils.py` - Backend testing utilities
- `tests/test_*.py` - Unit tests organized by module

**Config:**
- `pyproject.toml` - Project metadata, dependencies, tool configuration (ruff, pytest, coverage, ty)
- `.pre-commit-config.yaml` - Pre-commit hooks (ruff, ty, uv-lock, custom generators)
- `Dockerfile` - Multi-backend Docker image configuration

## Code Patterns

**Naming conventions:**
- Snake_case for functions, variables, modules (`compute_delays`, `probe_geometry`)
- PascalCase for classes (`Pipeline`, `Parameters`, `Probe`)
- Private internals prefixed with `_` (`_bootstrap_backend`, `_check_raw_data`)
- Internal modules in `zea/internal/`

**Import style:**
- Standard library → third-party → zea imports
- `TYPE_CHECKING` guards for circular import prevention
- Lazy imports for heavy dependencies (backends loaded on-demand)
- From-style imports for frequently used items: `from keras import ops`

**Type hints:**
- Pervasive use of type hints (enforced by `ty` type checker)
- Custom types in `zea/internal/typing.py`
- Runtime type validation via `spec.py` schemas

**Error handling:**
- Custom exception classes (e.g., `InvalidZeaFileError`)
- Warning utilities (`warning_once` for deduplicated warnings)
- Validation via `data/spec.py` with informative error messages

**Docstrings:**
- Google-style docstrings
- Extensive examples in docstrings with `.. doctest::` blocks
- RST formatting for cross-references (`:class:`, `:meth:`, `:attr:`)

**Base classes:**
- `Operation` - Base for all pipeline operations
- `MaskActionModel` / `LinesActionModel` - Agent action selection
- Keras `Layer` subclasses for models and data layers

**Backend abstraction:**
- `keras.ops` for tensor operations (backend-agnostic)
- `zea.backend.func_on_device` for device placement
- `zea.backend.jit` for JIT compilation across backends

## Development Setup

**Prerequisites:** Python ≥3.11

**Recommended (Docker):**
```bash
# Build image with all backends
docker build -t zea --build-arg DEV=true .
# Run with GPU support
docker run --gpus all -it zea
```

**Alternative (uv - recommended for local):**
```bash
git clone https://github.com/tue-bmd/zea.git
cd zea
uv sync --group jax-cpu  # or jax-gpu, torch-cpu, torch-gpu, tf-cpu, tf-gpu
uv run pre-commit install
```

**Alternative (pip):**
```bash
pip install -e . --group dev
pre-commit install
```

**Backend installation:** Choose one or more of JAX, PyTorch, or TensorFlow. Use `--group jax-cpu`, `--group torch-gpu`, etc. with `uv sync`.

## Testing

**Framework:** pytest

**Test groups:**
- `light` - Fast unit tests (default, runs on every PR)
- `heavy` - GPU-intensive tests (runs on self-hosted runners)
- `notebook` - End-to-end notebook tests (runs on self-hosted runners)

**Commands:**
```bash
# Light tests (local development)
pytest -m 'not heavy and not notebook'

# All unit tests with coverage
pytest --cov --cov-branch -m 'not heavy and not notebook'

# Heavy tests (requires GPU)
pytest -m 'heavy'

# Notebook tests
pytest -m 'notebook' --notebook-dir agent
```

**Test file naming:** `test_*.py` in `tests/` directory

**Fixtures:** Defined in `tests/conftest.py` (example dataset generation, backend setup)

**Coverage:** Codecov integration, branch coverage tracked, target coverage for PRs

## Build & CI

**Build system:** Hatchling (PEP 517)

**Package manager:** uv (lockfile: `uv.lock`)

**Linting & formatting:**
- **ruff** - Linter and formatter (line length: 100, target: py311)
- **ty** - Fast Python type checker
- Config in `pyproject.toml` under `[tool.ruff]` and `[tool.ty]`

**Pre-commit hooks:**
- ruff-check --fix
- ruff-format
- ty check
- check-yaml, check-added-large-files, detect-private-key
- uv-lock (keeps lockfile in sync)
- Custom: parameters_doc.py, spec_doc.py, generate-keras-ops, notebook-clean-and-check

**CI Workflows (.github/workflows/):**
- **tests.yaml** - Main gate: unit tests (ubuntu), minimum-deps check, heavy tests (self-hosted GPU), notebook tests (self-hosted GPU), docs build, Docker image build/smoke test
- **linter.yaml** - Ruff linting
- **precommits.yaml** - Pre-commit hook validation
- **publish.yaml** - PyPI publishing
- **docker.yaml** - Docker image builds to GHCR
- **sync-hf-configs.yaml** - Sync configs to HuggingFace Hub

**Documentation:** Sphinx (theme: Furo), hosted on ReadTheDocs

## Contribution Guidelines

**Process (from docs/source/contributing.rst):**
1. **Create an issue** - Discuss bug/feature before implementation
2. **Fork repository** - Use `gh repo fork tue-bmd/zea --clone`
3. **Setup environment** - Docker (recommended) or uv/pip with backend groups
4. **Test-driven development:**
   - Write test first
   - Implement functionality
   - Run `pytest -m 'not heavy and not notebook'` locally
5. **Commit & push** - Pre-commit hooks run automatically
6. **Open PR** - Reference issue number, provide clear description

**Code style:**
- Ruff-formatted (auto-fixed by pre-commit)
- Type hints required (checked by ty)
- Google-style docstrings with examples
- No trailing whitespace, Unix line endings

**Testing requirements:**
- New features require tests
- Maintain/improve coverage
- Tests must pass on at least one backend

**PR review:**
- Maintainers review for correctness, design, documentation
- CI must pass (tests, linting, docs build)
- Squash-merge preferred for clean history

**Commit messages:** Conventional style implied (fix:, feat:, docs:, etc.) based on GitHub actions

## Common Pitfalls

1. **Backend selection timing** - Backend must be set via `KERAS_BACKEND` env var or `keras.json` before importing keras. The `_bootstrap_backend()` function handles this.

2. **Circular imports** - Parameters, data specs, and pipeline can create circular dependencies. Use `TYPE_CHECKING` guards and lazy imports.

3. **Generated files** - `zea/ops/keras_ops.py` is auto-generated by `_generate_keras_ops.py`. Never edit it manually; regenerate via pre-commit hook.

4. **HDF5 file format** - Files must conform to `FileSpec` schema. Use `File.create()` to ensure compliance. Track naming has strict rules (see `_TRACK_RE` in spec.py).

5. **Device placement** - Use `func_on_device` wrapper for operations that need explicit device control. Don't assume GPU availability.

6. **Mixed precision** - Some operations use `signal_compute_dtype()` for numerical precision. Check `LOW_PRECISION_DTYPES` in `internal/precision.py`.

7. **Legacy compatibility** - `data/legacy_file.py` handles old format files. New code should use current specs.

8. **Notebooks in CI** - Notebooks run via papermill in CI. Must have reproducible outputs and be cleaned via `notebook_clean_and_check.py`.

9. **Test markers** - Always mark GPU tests with `@pytest.mark.gpu` and heavy tests with `@pytest.mark.heavy` to enable selective running.

10. **Docker cache** - Self-hosted runners use persistent volumes for caches (`/cache/zea`, `/cache/huggingface`). Local development doesn't have this.

11. **Metadata dimension alignment** - When using `additional_axes_iter` to iterate over non-frame axes, metadata fields sharing those dimensions must be indexed the same way. The `index_metadata_axes()` function handles this by:
    - Mapping axes from `additional_axes_iter` to dimension names via the file spec
    - Applying integer indexing to metadata fields carrying those dimensions
    - Dropping the indexed axes from metadata, matching h5py's integer indexing behavior
    - This prevents metadata from being misaligned with the sample data when dimensions are iterated over

    **Why this matters:** When `additional_axes_iter` uses integer indexing on an axis (e.g., `n_tx`), h5py drops that dimension from the loaded array. Without corresponding metadata indexing, a field like `scan.t0_delays` would still have its `n_tx` dimension while the data doesn't, breaking the alignment.

## Approachable Areas

**Good entry points for new contributors:**

1. **Data augmentations** (`zea/data/augmentations.py`) - Self-contained Keras layers, well-documented, easy to test

2. **Utility functions** (`zea/utils.py`) - Simple helpers, no complex dependencies

3. **Configuration examples** (`configs/`) - Adding new dataset configs requires only YAML knowledge

4. **Documentation improvements** (`docs/source/*.rst`) - RST files, docstring examples in source code

5. **Probe definitions** (`zea/probes.py`) - Adding new probe geometries is straightforward, follows clear pattern

6. **Tests** - Most test files are well-structured with clear fixtures. Adding test cases to existing test files is approachable.

7. **Visualization** (`zea/visualize.py`, `zea/display.py`) - Matplotlib-based, standard image processing

**Well-tested areas** (high confidence for modifications):
- Core beamforming (`zea/beamform/`) - Comprehensive test coverage
- Data loading (`zea/data/file.py`, `dataloader.py`) - Heavily tested with fixtures
- Parameter management (`zea/parameters.py`) - Extensive validation and tests
- Metadata handling (`zea/data/metadata.py`) - Per-frame slicing, axis selection, dimension-aware indexing all tested

**Areas requiring domain expertise:**
- Agent module (`zea/agent/`) - Requires understanding of cognitive ultrasound
- Models (`zea/models/`) - Deep learning architecture knowledge needed
- Backend implementation (`zea/backend/`) - Multi-framework compatibility is complex
- Beamformer core (`zea/beamform/beamformer.py`) - Ultrasound physics knowledge required
