"""Tests for custom-operation loading via :func:`~zea.ops.base.get_ops` and pipeline configs.

Covers:

* Plain registry-name lookup
* Module-path lookup where the registry key **equals** the lower-cased class name
  (the original shortname-fallback path)
* Module-path lookup where the registry key **differs** from the class name
  (requires the identity-based resolution added to ``get_ops``)
* Error paths (unknown name, unregistered class)
* End-to-end: custom op loaded into a :class:`~zea.ops.Pipeline` via a config dict,
  a YAML file, and via a pre-imported registry name
"""

import numpy as np
import pytest
import yaml

from zea.config import Config
from zea.ops.base import get_ops
from zea.ops.pipeline import Pipeline, pipeline_from_config

# ── get_ops: plain registry-name lookups ─────────────────────────────────────


def test_get_ops_by_registry_name():
    """get_ops with a known registry name returns the correct class."""
    from zea.ops.base import Identity

    cls = get_ops("identity")
    assert cls is Identity


def test_get_ops_unknown_plain_name_raises():
    """get_ops with an unknown plain name raises :exc:`KeyError`."""
    with pytest.raises(KeyError):
        get_ops("nonexistent_op_zzzzzz")


# ── get_ops: module-path lookups ─────────────────────────────────────────────


def test_get_ops_module_path_registry_key_differs_from_class_name():
    """Module-path lookup resolves by class identity when registry key ≠ class name.

    ``ScaleByFactorOp`` is registered as ``"fixture_scale_op"``.  The old
    shortname fallback (checking ``"ScaleByFactorOp"`` in the registry) would
    *not* find it; only the identity-based lookup introduced in the fix does.
    """
    from tests.fixtures.custom_ops import ScaleByFactorOp

    cls = get_ops("tests.fixtures.custom_ops.ScaleByFactorOp")
    assert cls is ScaleByFactorOp


def test_get_ops_module_path_registry_key_differs_by_underscore():
    """Module-path lookup resolves by identity when key differs only by underscore.

    ``Fixturepassthrough`` is registered as ``"fixture_passthrough"``; its
    lower-cased class name ``"fixturepassthrough"`` does not match the registry
    key (the underscore breaks the match), so identity-based lookup is required.
    """
    from tests.fixtures.custom_ops import Fixturepassthrough

    cls = get_ops("tests.fixtures.custom_ops.Fixturepassthrough")
    assert cls is Fixturepassthrough


def test_get_ops_module_path_unknown_class_raises():
    """Dotted path to a class that doesn't exist in the registry raises :exc:`ValueError`."""
    # The module exists and will be importable, but "NotARegisteredClass" is not there
    with pytest.raises(ValueError, match="not found in registry"):
        get_ops("tests.fixtures.custom_ops.NotARegisteredClass")


def test_get_ops_module_path_registered_under_full_dotted_key():
    """Covers path A: post-import ``ops_name in ops_registry`` check (line 68-69).

    ``DottedKeyOp`` is registered under its full module path as the key.
    ``dotted_key_ops`` is never imported elsewhere, so on the first call the
    module import triggers the decorator, and the second registry check fires.
    """
    full_key = "tests.fixtures.dotted_key_ops.DottedKeyOp"
    cls = get_ops(full_key)  # import happens here; path A check fires
    from tests.fixtures.dotted_key_ops import DottedKeyOp

    assert cls is DottedKeyOp


def test_get_ops_unregistered_class_in_module_raises():
    """Covers path B: ``except KeyError: pass`` when class exists but is not registered.

    ``UnregisteredOp`` is importable but has no ``@ops_registry`` decorator.
    ``get_name(cls)`` raises ``KeyError`` (path B caught); the shortname
    ``"UnregisteredOp"`` is also not a registry key, so ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="not found in registry"):
        get_ops("tests.fixtures.unregistered_ops.UnregisteredOp")


def test_get_ops_module_path_shortname_fallback():
    """Covers path C: class-name shortname fallback after path B.

    ``tests.fixtures.unregistered_ops.Identity`` is importable but unregistered,
    so ``get_name(cls)`` raises ``KeyError`` (path B caught).  The lowercased
    class name ``"identity"`` IS a registry key, so path C returns the built-in
    :class:`~zea.ops.base.Identity` class.
    """
    from zea.ops.base import Identity as BuiltinIdentity

    cls = get_ops("tests.fixtures.unregistered_ops.Identity")
    assert cls is BuiltinIdentity


def test_get_ops_module_path_already_imported_returns_same_object():
    """Calling get_ops with module path twice returns the identical class object."""
    cls_by_path = get_ops("tests.fixtures.custom_ops.ScaleByFactorOp")
    cls_by_name = get_ops("fixture_scale_op")
    assert cls_by_path is cls_by_name


# ── Pipeline: config dict with custom op by module path ──────────────────────


def test_pipeline_from_config_with_module_path():
    """Pipeline built from a config dict using a module-path op name works end-to-end."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    {"name": "tests.fixtures.custom_ops.ScaleByFactorOp"},
                ]
            }
        }
    )
    pipeline = pipeline_from_config(config, jit_options=None)
    assert len(pipeline.operations) == 1

    result = pipeline(data=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result["data"], np.array([3.0, 6.0, 9.0]))


def test_pipeline_from_config_with_module_path_and_params():
    """Custom op loaded by module path accepts constructor params from the config."""
    config = Config(
        {
            "pipeline": {
                "operations": [
                    {
                        "name": "tests.fixtures.custom_ops.ScaleByFactorOp",
                        "params": {"factor": 5.0},
                    }
                ]
            }
        }
    )
    pipeline = pipeline_from_config(config, jit_options=None)
    result = pipeline(data=np.array([2.0, 4.0]))
    np.testing.assert_allclose(result["data"], np.array([10.0, 20.0]))


def test_pipeline_from_config_custom_op_by_registry_name_after_import():
    """Once a custom module has been imported, its registry name works in config too."""
    # Ensure the module is imported (get_ops does the import)
    get_ops("tests.fixtures.custom_ops.ScaleByFactorOp")

    config = Config(
        {"pipeline": {"operations": [{"name": "fixture_scale_op", "params": {"factor": 2.0}}]}}
    )
    pipeline = pipeline_from_config(config, jit_options=None)
    result = pipeline(data=np.array([7.0]))
    np.testing.assert_allclose(result["data"], np.array([14.0]))


# ── Pipeline: YAML file with custom op by module path ────────────────────────


def test_pipeline_from_yaml_custom_op_by_module_path(tmp_path):
    """Custom op specified by module path in a YAML config is loaded correctly."""
    config_dict = {
        "pipeline": {
            "operations": [
                {
                    "name": "tests.fixtures.custom_ops.ScaleByFactorOp",
                    "params": {"factor": 4.0},
                }
            ]
        }
    }
    yaml_path = tmp_path / "custom_pipeline.yaml"
    yaml_path.write_text(yaml.dump(config_dict))

    pipeline = Pipeline.from_path(yaml_path, jit_options=None)
    result = pipeline(data=np.array([3.0]))
    np.testing.assert_allclose(result["data"], np.array([12.0]))


def test_pipeline_yaml_roundtrip_custom_op_registry_name(tmp_path):
    """Pipeline with a custom op round-trips through YAML using its registry name.

    After the module is imported, the op is available by registry name in the
    same Python session, so the saved YAML (which contains the registry name)
    loads successfully.
    """
    from tests.fixtures.custom_ops import ScaleByFactorOp

    pipeline = Pipeline(
        operations=[ScaleByFactorOp(factor=6.0)],
        jit_options=None,
    )

    yaml_path = tmp_path / "roundtrip.yaml"
    pipeline.to_yaml(yaml_path)

    with open(yaml_path) as f:
        content = yaml.safe_load(f)

    # The YAML stores the registry key, not the module path
    op_entry = content["pipeline"]["operations"][0]
    assert op_entry["name"] == "fixture_scale_op"
    assert op_entry["params"]["factor"] == 6.0

    # Loading back works because fixture_scale_op is registered in this session
    loaded = Pipeline.from_path(yaml_path, jit_options=None)
    result = loaded(data=np.array([2.0]))
    np.testing.assert_allclose(result["data"], np.array([12.0]))


# ── Pipeline: multi-op chain mixing built-in and custom ops ──────────────────


def test_pipeline_mixed_builtin_and_custom_ops():
    """Pipeline with both built-in and custom ops (by module path) runs correctly."""
    from zea.ops.base import Identity

    config = Config(
        {
            "pipeline": {
                "operations": [
                    {"name": "identity"},
                    {
                        "name": "tests.fixtures.custom_ops.ScaleByFactorOp",
                        "params": {"factor": 2.0},
                    },
                ]
            }
        }
    )
    pipeline = pipeline_from_config(config, jit_options=None)
    assert isinstance(pipeline.operations[0], Identity)

    result = pipeline(data=np.array([5.0]))
    np.testing.assert_allclose(result["data"], np.array([10.0]))
