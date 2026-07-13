# Architecture

`zea` is a single Python package (`zea/`) organized by domain. The top-level `zea/__init__.py`
uses **lazy imports** so that importing `zea` does not eagerly pull in Keras or any ML backend —
the public API is declared under `TYPE_CHECKING` for IDEs, and a `_bootstrap_backend()` routine
checks that a supported backend (torch / tensorflow / jax) is installed before use
(`zea/__init__.py`).

## Package map

| Subpackage / module | Responsibility |
| --- | --- |
| `zea/ops/` | `Operation` and `Pipeline` — the composable processing graph. See [pipeline.md](pipeline.md). |
| `zea/func/` | Functional building blocks (`tensor.py`, `ultrasound.py`) that ops are built on. |
| `zea/beamform/` | Beamforming core: delays, `beamformer.py`, `pfield.py`, `pixelgrid.py`, lens correction, phantoms. |
| `zea/data/` | File format, `File`/`Dataset`/`Dataloader`, `spec.py` (schema), `convert/` (dataset importers), augmentations. See [data.md](data.md). |
| `zea/models/` | Pretrained Keras models + presets. See [models.md](models.md). |
| `zea/agent/` | Action-selection / active perception. See [agent.md](agent.md). |
| `zea/backend/` | Backend abstraction: `jit`, `device`, `AutoGrad`, per-backend subdirs (`jax/`, `torch/`, `tensorflow/`), `tf2jax.py`. |
| `zea/tracking/` | Motion/segmentation tracking (`lucas_kanade.py`, `segmentation.py`). |
| `zea/tools/` | Utilities: Hugging Face (`hf.py`), W&B (`wndb.py`), interactive selection, scan-cone fitting. |
| `zea/internal/` | Non-public infrastructure: `registry.py`, `core.py`, `device.py`, `checks.py`, ops codegen, caching. |
| `zea/config.py`, `zea/parameters.py`, `zea/probes.py`, `zea/simulator.py`, `zea/doppler.py`, `zea/display.py`, `zea/visualize.py`, `zea/metrics.py` | Top-level building blocks. |
| `zea/__main__.py`, `zea/cli_args.py` | CLI entry point (`zea …`). See [cli-and-config.md](cli-and-config.md). |

## Backend abstraction

The central design constraint: **all tensor code must run identically on JAX, TensorFlow, and
PyTorch.** In practice that means writing everything against `keras.ops` rather than any single
framework. `zea/backend/__init__.py` documents the rule and wraps only what Keras does *not*
expose directly:

- `jit` — unified JIT compilation (`jax.jit` for JAX, `tf.function` for TF, a no-op for torch).
- `device` — context manager pinning Keras ops to a device (re-exported as `zea.device`).
- `func_on_device` — run a callable with its tensors moved to a target device.
- `AutoGrad` — backend-agnostic automatic differentiation.

The backend is selected via the `KERAS_BACKEND` environment variable (`docs/source/environment.rst`).
When you touch tensors, prefer `keras.ops`; if you need framework-specific behavior, route it
through `zea.backend` so all three backends stay supported. `zea/backend/tf2jax.py` exists to
bridge TensorFlow-authored functions into JAX.

## The registry pattern

`zea` maps **config strings to classes** through registries so that pipelines, models, probes,
and datasets can be described entirely in YAML (`zea/internal/registry.py`). A registry is a
`RegisterDecorator`; classes opt in with a decorator that also records extra metadata:

```python
# zea/internal/registry.py docstring example
dataset_registry = RegisterDecorator(items_to_register=['probe_used', 'scan_class'])

@dataset_registry(name='picmus', probe_used='L11-5V', scan_class=PicmusScan)
class PICMUS(Dataset):
    ...
```

Lookups are then `dataset_registry['picmus']`. Key registries include `ops_registry`,
`beamformer_registry`, `action_selection_registry` (`zea/agent/selection.py`), and
`probe_registry` (`zea/probes.py`). `Operation` lookup additionally accepts a **dotted module
path** (e.g. `"my_pkg.my_mod.MyOp"`), which imports the module and locates the class by object
identity — this is what lets users register custom operations *outside* the `zea` source tree and
still round-trip them through YAML (`zea/ops/base.py` `get_ops`, `docs/source/pipeline.rst`).

**When adding any registrable class**, decorate it and make sure its defining module is imported
(usually via the subpackage `__init__.py`) so the registration side-effect runs.

## Cross-cutting conventions

- **Dict-of-tensors dataflow.** Operations receive and return a `dict` of tensors + parameters,
  each op reading only the keys it needs and passing the rest through (`zea/ops/__init__.py`).
- **Parameters are self-describing.** Acquisition parameters travel with the data (in the file and
  in `zea.Parameters`), not as scattered function arguments. See [data.md](data.md).
- **`zea/internal/` is private.** Public API is what `zea/__init__.py` exposes; treat `internal`
  as implementation detail that can change.
