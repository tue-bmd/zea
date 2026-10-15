"""Fixtures shared by the model tests."""

import json

import numpy as np
import pytest

from .. import DEFAULT_TEST_SEED


@pytest.fixture
def rng():
    """Random number generator for reproducible tests."""
    return np.random.default_rng(DEFAULT_TEST_SEED)


@pytest.fixture
def local_preset(tmp_path):
    """Builds a local preset directory for a zea model class.

    A preset is just a directory with a ``config.json`` plus whatever asset files
    the model's ``custom_load_weights()`` asks for, so the loader is perfectly
    happy with a local one and no download is needed.

    Yields:
        Callable[[type, dict | None], str]: takes the model class the preset is for
        and an optional model config, and returns the preset directory path. Asset
        files are added by the caller, e.g.::

            preset = local_preset(MyModel)
            (Path(preset) / "model.onnx").write_bytes(b"...")
    """
    counter = iter(range(100))

    def _build(model_cls, config=None):
        preset_dir = tmp_path / f"preset-{next(counter)}"
        preset_dir.mkdir(parents=True, exist_ok=True)
        (preset_dir / "config.json").write_text(
            json.dumps(
                {
                    "module": model_cls.__module__,
                    "class_name": model_cls.__name__,
                    "registered_name": model_cls.__name__,
                    "config": config or {},
                    "build_config": None,
                }
            )
        )
        return str(preset_dir)

    return _build


class StubOnnxSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``.

    Records every input it is handed in ``seen``, and returns ``output`` once a
    test sets one -- zeros of the model's output shape until then.
    """

    def __init__(self, path, output_shape=()):
        self.path = path
        self.output_shape = output_shape
        self.output = None
        self.seen = []

    class _IO:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self._IO("input")]

    def get_outputs(self):
        return [self._IO("output")]

    def run(self, output_names, feed):
        self.seen.append(feed["input"])
        if self.output is not None:
            return [self.output]
        batch = feed["input"].shape[0]
        return [np.zeros((batch, *self.output_shape), dtype="float32")]


@pytest.fixture
def install_stub_onnx(monkeypatch):
    """Puts a :class:`StubOnnxSession` in place of the real ONNX Runtime session.

    Yields:
        Callable[[tuple], None]: takes the per-image output shape the stubbed
        session should return zeros of.
    """
    import onnxruntime

    def _install(output_shape):
        monkeypatch.setattr(
            onnxruntime,
            "InferenceSession",
            lambda path: StubOnnxSession(path, output_shape),
        )

    return _install
