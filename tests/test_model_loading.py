"""Tests for model loading infrastructure in zea.models.preset_utils.py."""

import json

import pytest

from zea.models.preset_utils import KerasPresetLoader


@pytest.fixture()
def dummy_loader(tmp_path):
    """A KerasPresetLoader backed by a minimal config.json in a temp directory."""
    config = {
        "class_name": "DummyModel",
        "registered_name": "DummyModel",
        "config": {},
        "module": "builtins",
        "build_config": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return KerasPresetLoader(str(tmp_path), config)


class DummyModelBase:
    """Minimal stand-in for a Keras model (no weights, already built)."""

    weights = []
    built = True


def test_load_model_custom_load_weights_with_load_weights_param(monkeypatch, dummy_loader):
    """custom_load_weights that accepts load_weights receives it forwarded."""
    calls = []

    class DummyModel(DummyModelBase):
        def custom_load_weights(self, preset, load_weights=True):
            calls.append(load_weights)

    dummy = DummyModel()
    monkeypatch.setattr("zea.models.preset_utils.load_serialized_object", lambda *a, **kw: dummy)

    result = dummy_loader.load_model(cls=object, load_weights=False)

    assert result is dummy
    assert calls == [False], f"load_weights should be forwarded as False, got {calls}"


def test_load_model_custom_load_weights_without_param_skipped_when_false(monkeypatch, dummy_loader):
    """custom_load_weights without load_weights param is NOT called when load_weights=False."""
    calls = []

    class DummyModel(DummyModelBase):
        def custom_load_weights(self, preset, **kwargs):
            calls.append(True)

    dummy = DummyModel()
    monkeypatch.setattr("zea.models.preset_utils.load_serialized_object", lambda *a, **kw: dummy)

    result = dummy_loader.load_model(cls=object, load_weights=False)

    assert result is dummy
    assert calls == [], "custom_load_weights should NOT be called when load_weights=False"


def test_load_model_custom_load_weights_without_param_called_when_true(monkeypatch, dummy_loader):
    """custom_load_weights without load_weights param IS called when load_weights=True."""
    calls = []

    class DummyModel(DummyModelBase):
        def custom_load_weights(self, preset, **kwargs):
            calls.append(True)

    dummy = DummyModel()
    monkeypatch.setattr("zea.models.preset_utils.load_serialized_object", lambda *a, **kw: dummy)

    result = dummy_loader.load_model(cls=object, load_weights=True)

    assert result is dummy
    assert calls == [True], "custom_load_weights should be called when load_weights=True"


def test_load_model_no_custom_load_weights_returns_model_when_false(monkeypatch, dummy_loader):
    """Model without custom_load_weights is returned as-is when load_weights=False."""
    dummy = DummyModelBase()
    monkeypatch.setattr("zea.models.preset_utils.load_serialized_object", lambda *a, **kw: dummy)

    result = dummy_loader.load_model(cls=object, load_weights=False)

    assert result is dummy
