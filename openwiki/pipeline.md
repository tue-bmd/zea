# Pipeline & operations

The processing core of `zea` lives in `zea/ops/`. It turns raw ultrasound channel data into
images (and other products) through a chain of small, composable, JIT-compilable steps.

## Operations

An **`Operation`** (`zea/ops/base.py`) is a callable unit of processing. Every op takes a `dict`
of tensors + parameters as input and returns a `dict`, reading only the keys it needs and passing
the rest through. Static arguments go in the constructor; dynamic arguments (including `data`) are
passed at call time (`zea/ops/__init__.py`):

```python
from zea.ops import EnvelopeDetect
op = EnvelopeDetect(axis=-1)              # static config
out = op(data=data)["data"]               # dynamic input, dict output
```

Operations come from three sources:

- **Ultrasound ops** (`zea/ops/ultrasound.py`): `Demodulate`, `TOFCorrection`, `ReceiveApodization`,
  `AlignedApodization`, `ApplyWindow`, `EnvelopeDetect`, `LogCompress`, `PfieldWeighting`,
  `ReshapeGrid`, … (see the import block in `zea/ops/pipeline.py`).
- **Tensor ops** (`zea/ops/tensor.py`): e.g. `Normalize`, plus generic tensor manipulation.
- **Keras ops** (`zea/ops/keras_ops.py`): every op in the [Keras Ops API](https://keras.io/api/ops/)
  is auto-exposed as a `zea` operation (generated code; see `zea/internal/_generate_keras_ops.py`).

Each op is registered under a name via `@ops_registry("name")`, so it can be referenced from YAML.

## Pipeline

A **`Pipeline`** (`zea/ops/pipeline.py`) chains operations so the output dict of one feeds the
next. Key construction options (`Pipeline.__init__`):

- `operations` — a list of `Operation`s (or nested `Pipeline`s).
- `with_batch_dim` — whether ops expect a leading batch dimension (default `True`).
- `jit_options` — `"ops"` (default; JIT each op, preserving Python control flow such as caching),
  `"pipeline"` (compile the whole chain as one function — faster but no Python control flow), or
  `None`.
- `device`, `timed`, `validate`, `name`.

A typical RF→B-mode pipeline is `demodulate → downsample → beamform → envelope_detect →
normalize → log_compress` (see the resolved config in `docs/source/config.rst`).

### Serialization

Pipelines round-trip through YAML: `Pipeline.to_yaml(path)` writes the op list + params, and
`Pipeline.from_path(path)` rebuilds an identical pipeline (`docs/source/pipeline.rst`). Because
ops are looked up through the registry (including dotted module paths), **custom operations
serialize and deserialize with no changes to `zea`**.

## Functional API

`zea/func/` holds the low-level functions that ops are built on (`func/tensor.py`,
`func/ultrasound.py`). For example, `EnvelopeDetect` wraps `zea.func.envelope_detect`. Use `func`
directly for a functional style; use `ops`/`Pipeline` when you want composition, JIT, and YAML
config (`zea/ops/__init__.py`).

## Beamforming

`zea/beamform/` implements the physics. `beamformer.py` provides the time-of-flight (TOF)
correction pipeline used by the `TOFCorrection` op, exposing lower-level building blocks (delay
computation, f-number masking, phase rotation) for standalone use (`zea/beamform/beamformer.py`).
Supporting modules:

- `delays.py` — transmit/receive delay computation (recent commits fixed transmit-delay
  discontinuity at the focus and added scanline beamforming — `git log`).
- `pfield.py` — pressure-field weighting; `pixelgrid.py` — output grid geometry; `lens_correction.py`
  — lens travel-time correction; `phantoms.py` — synthetic targets.

Beamformers are registered in `beamformer_registry` and selected by name (e.g.
`"delay_and_sum"`) in the config.

## Custom operations

To add a project-specific op without forking `zea` (`docs/source/pipeline.rst`):

```python
from zea.internal.registry import ops_registry
from zea.ops import Operation

@ops_registry("my_project.my_ops.MyScale")
class MyScale(Operation):
    def __init__(self, factor: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    def call(self, **kwargs):
        data = kwargs[self.key]
        return {self.output_key: data * self.factor}
```

It then works anywhere a built-in op works, including inside a `Pipeline` and in YAML. To add an
op *to* `zea` itself, follow the "adding a new operation" workflow in the docs and register it in
the appropriate `zea/ops/` module.

## Where to look / what to watch

- Backend-agnostic: write against `keras.ops`, not a single framework — see
  [Architecture § Backend abstraction](architecture.md#backend-abstraction).
- JIT interaction: `jit_options="pipeline"` drops Python control flow (e.g. caching). Prefer
  `"ops"` unless you have measured a win.
- Relevant tests: `tests/test_operations.py`, `tests/test_ops_infra.py`, `tests/test_beamformer.py`,
  `tests/test_pfield.py`, `tests/test_custom_ops.py`, `tests/test_keras_ops.py`.
