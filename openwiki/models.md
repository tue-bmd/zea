# Models

`zea/models/` is a collection of pretrained, backend-agnostic Keras models for ultrasound image
and signal processing, plus the infrastructure to load their weights from the Hugging Face
`zeahub` (`docs/source/models.rst`).

## BaseModel and preset system

All models inherit from `BaseModel` (`zea/models/base.py`), a `keras.models.Model` subclass whose
`from_config` override ensures the concrete subclass (not a vanilla `keras.Model`) is returned.

Weights are distributed via **presets** (`zea/models/presets.py`, `zea/models/preset_utils.py`):
each preset registers a named set of weights for a model architecture, so one architecture can
serve several checkpoints. The expected artifacts per preset are a `config.json` and a
`model.weights.h5`, hosted on [Hugging Face](https://huggingface.co/zeahub)
(`docs/source/models.rst`). Loading resolves the preset, downloads (and caches) the weights, and
builds the model.

## Available models

The module bundles a range of architectures (`zea/models/`), including:

- Generative models: `diffusion.py`, `flow_matching.py`, `dit.py`, `gmm.py`, `taesd.py`, and the
  `hvae/` package, layered on `generative.py` (`GenerativeModel` / `DeepGenerativeModel` base
  classes).
- Segmentation / task models: `carotid_segmenter.py`, `deeplabv3.py`, `lv_segmentation.py`,
  `echonet.py`, `echonetlvh.py`, `regional_quality.py`.
- Building blocks / utilities: `unet.py`, `dense.py`, `layers.py`, `lpips.py`, `speckle2self.py`.

## Adding a model

From `docs/source/models.rst`:

1. Add `zea/models/mymodel.py` with a class inheriting `BaseModel` (or `GenerativeModel` /
   `DeepGenerativeModel` for generative models); implement `call`.
2. Upload pretrained weights (`config.json` + `model.weights.h5`) to the `zeahub` Hugging Face.
   For non-standard saving, implement a `custom_load_weights` method (see `echonet.py`).
3. Register presets in `zea/models/presets.py`, then call `register_presets` from your model file.
4. Import your model in `zea/models/__init__.py` so it becomes part of the package.

### Porting PyTorch models

The recommended path is a **native Keras 3** reimplementation so you get all three backends and the
preset infrastructure for free. `docs/source/models.rst` documents the common PyTorch→Keras
gotchas to match numerics exactly: asymmetric `padding='same'` for stride-2 conv (use
`ZeroPadding2D(1) + Conv2D(padding='valid')`), `Conv2DTranspose` cropping, `InstanceNorm` via
`GroupNormalization`, weight-axis permutations, and NCHW↔NHWC transposes. Vendor the original
torch code behind a lazily-imported helper used only for weight conversion, not inference.

## Where to look / what to watch

- Keep everything backend-agnostic (`keras.ops`); a model that only works on one backend breaks the
  core promise — see [Architecture § Backend abstraction](architecture.md#backend-abstraction).
- Relevant tests: `tests/test_models.py`, `tests/test_model_loading.py`, `tests/test_generative.py`,
  `tests/test_flow_matching.py`.
