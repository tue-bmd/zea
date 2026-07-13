# zea — OpenWiki quickstart

`zea` is a Python toolbox for **cognitive ultrasound imaging**: ultrasound signal processing,
image reconstruction, and deep learning, all built on top of [Keras 3](https://keras.io/keras_3/)
so the same code runs on **JAX, TensorFlow, or PyTorch** (`README.md`, `pyproject.toml`). It is a
research library from the TU/e Biomedical Diagnostics group, published on
[PyPI](https://pypi.org/project/zea/) and [Hugging Face](https://huggingface.co/zeahub), and
documented at [zea.readthedocs.io](https://zea.readthedocs.io).

> Beta software under active development (`README.md`). The package version lives in
> `pyproject.toml` (`0.1.2` at time of writing).

## What zea gives you

Four pillars, mirrored by the package layout (`zea/__init__.py`, `docs/source/index.rst`):

1. **Data** — a self-describing HDF5 file format that stores raw ultrasound data *alongside* all
   acquisition parameters, plus loading tools designed for deep learning (`zea.File`,
   `zea.Dataset`, `zea.Dataloader`). See [Data & file format](data.md).
2. **Pipeline** — a composable, JIT-compilable sequence of `Operation`s that turns raw channel
   data into B-mode images (demodulate → beamform → envelope detect → log-compress → …). See
   [Pipeline & operations](pipeline.md).
3. **Models** — pretrained Keras models for ultrasound image/signal processing, loaded from the
   Hugging Face `zeahub` via a preset system. See [Models](models.md).
4. **Agent** — action-selection strategies for *active perception*: choosing which transmits to
   fire next given beliefs about the tissue. This is the "cognitive" part. See [Agent](agent.md).

## The core workflow

From `docs/source/getting-started.rst` — the canonical data flow is:

1. Load a `zea.File` and call `File.load_parameters()` to build a `zea.Parameters` object (merged
   probe + scan parameters, with derived quantities).
2. Optionally override parameters from a `config.yaml`.
3. Build a `zea.Pipeline` (from a config or in code).
4. Pass `data` + `parameters` through the pipeline.
5. Visualize with `zea.visualize` / `zea.display`.

```python
import zea

zea.init_device()                 # pick CPU/GPU
config = zea.Config.from_path("hf://zeahub/picmus/config_iq.yaml")
with zea.File("hf://zeahub/picmus/.../carotid_cross_expe_dataset_iq.hdf5") as f:
    data = f.data.raw_data[0]
    parameters = f.load_parameters()
pipeline = zea.Pipeline.from_config(config.pipeline)   # see pipeline.md for exact API
image = pipeline(data=data, **parameters.get_pipeline_inputs())
```

`hf://…` paths are resolved transparently against the Hugging Face Hub, so most examples run with
no local data (`docs/source/data-acquisition.rst`).

## Sections

- [Architecture](architecture.md) — package map, the Keras backend abstraction, and the registry
  pattern that ties config strings to classes.
- [Pipeline & operations](pipeline.md) — `zea.ops`, `zea.func`, beamforming, custom operations.
- [Data & file format](data.md) — the HDF5 spec, `File`/`Dataset`/`Dataloader`, `Parameters`,
  probes, dataset conversion, and the simulator.
- [Models](models.md) — `BaseModel`, presets, and adding a model.
- [Agent](agent.md) — cognitive ultrasound and action selection.
- [CLI & configuration](cli-and-config.md) — the `zea` command, YAML configs, environment
  variables.
- [Testing & development](testing.md) — multi-backend testing, linting, CI.

## Where to start when changing code

- New processing step → [Pipeline & operations](pipeline.md) (register an `Operation`).
- New data source / dataset → [Data & file format](data.md) (dataset conversion + registry).
- New neural network → [Models](models.md).
- New acquisition strategy → [Agent](agent.md).
- Anything touching tensors must stay **backend-agnostic** — read
  [Architecture § Backend abstraction](architecture.md#backend-abstraction) first.
