# Testing & development

## Test suite

Tests live in `tests/` and are run with `pytest` (`pyproject.toml [tool.pytest.ini_options]`,
`testpaths = ["tests"]`). Coverage is high — there is roughly one `test_*.py` per subsystem
(`test_beamformer.py`, `test_operations.py`, `test_parameters.py`, `test_models.py`,
`test_agent.py`, `test_io_lib.py`, …).

### Multi-backend testing

The defining complexity is that every tensor path must work on **torch, tensorflow, and jax**.
`tests/conftest.py` and `tests/backend_utils.py` orchestrate this:

- At collection time it probes which backends are installed and which have working CUDA
  (`available_test_backends`, `missing_required_backends`, `backend_guard_skips`).
- A test can require specific backends via markers/params; a `backend` fixture parametrizes tests
  across backends, and unavailable ones are skipped (with `--skip-unavailable-backends`).
- Because importing a backend can lock Keras to it, backend-specific work is isolated in
  subprocesses; a JAX GPU warm-up runs while all GPUs are still visible (`tests/conftest.py`
  comments). `ZEA_TEST_DEVICE` restricts the device used.

### Markers

Custom pytest markers (`pyproject.toml`): `performance`, `heavy`, `notebook`, `gpu`. Deselect with
e.g. `pytest -m "not heavy and not performance"`. `test_notebooks.py` executes the example
notebooks (via `papermill`) so docs stay runnable. Coverage runs with branch + multiprocessing
concurrency and omits generated files (`keras_ops.py`, `app.py`).

## Linting, formatting, type checking

Configured in `pyproject.toml` and `.pre-commit-config.yaml`:

- **Ruff** (pinned `0.15.17`) for lint + format (`[tool.ruff]`, `ruff-check --fix`, `ruff-format`).
- **ty** (Astral's type checker, pinned) run as a **local** pre-commit hook so it sees zea's
  installed dependencies (the upstream isolated hook can't resolve keras/numpy/h5py — see the
  comment in `.pre-commit-config.yaml`).

Install hooks with `pre-commit install`; run all with `pre-commit run --all-files`.

## Continuous integration

GitHub Actions in `.github/workflows/`:

- `tests.yaml` — the test suite (CI forces CPU via `CUDA_VISIBLE_DEVICES=""`).
- `linter.yaml`, `precommits.yaml` — lint / pre-commit checks.
- `docker.yaml`, `build-image.yaml` — build container images (`Dockerfile`, `scripts/build_all_images.sh`,
  `scripts/resolve_backend_versions.sh`).
- `publish.yaml`, `set-tag.yaml` — release to PyPI on version tags.
- `sync-hf-configs.yaml` — push configs to the Hugging Face Hub.

## Local development

- Install dev extras: `pip install -e .[dev]` (or use `uv` — see `uv.lock`). Dependencies and dev
  tooling are declared in `pyproject.toml`.
- A `Dockerfile` and dev-container config provide a reproducible environment across the three
  backends.
- Full contributing guide: [zea.readthedocs.io/en/latest/contributing.html](https://zea.readthedocs.io/en/latest/contributing.html)
  (`CONTRIBUTING.md` points there). Docs are Sphinx-based under `docs/` and build on Read the Docs
  (`.readthedocs.yaml`).

## What to watch when changing code

- Run the suite on more than one backend before assuming a change is safe; a passing torch run does
  not prove jax/tf correctness. See [Architecture § Backend abstraction](architecture.md#backend-abstraction).
- New registrable classes (ops, models, probes, datasets, strategies) need their module imported so
  the registration runs — otherwise lookups by name fail. See [Architecture § registry pattern](architecture.md#the-registry-pattern).
- Keep changes JIT-safe where pipelines compile (`jit_options`), and validate against
  `zea/data/spec.py` when touching the file format.
