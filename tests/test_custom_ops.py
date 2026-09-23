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

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from zea.config import Config
from zea.internal.registry import ops_registry
from zea.ops.base import Identity, get_ops
from zea.ops.imports import (
    OpsModuleImportError,
    RemoteCodeError,
    import_ops_module,
)
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


# ── imports: loading modules that define custom operations ───────────────────
#
# The ops registry is process-global and refuses a name that is already taken,
# so every module written here registers under a name unique to its test.

_OP_MODULE_TEMPLATE = """
from zea.internal.registry import ops_registry
from zea.ops.base import Operation


@ops_registry("{name}")
class TmpOp(Operation):
    def __init__(self, factor: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, data, **kwargs):
        return {{"data": data * self.factor}}
"""

_op_module_counter = itertools.count()


def write_op_module(directory, filename="tmp_ops.py"):
    """Write a module defining one operation under a name unique to this call.

    Returns:
        tuple[Path, str]: the module path and the registry name it uses.
    """
    name = f"fixture_tmp_op_{next(_op_module_counter)}"
    path = Path(directory) / filename
    path.write_text(_OP_MODULE_TEMPLATE.format(name=name), encoding="utf-8")
    return path, name


def test_import_ops_module_dotted_path():
    """A dotted module path is imported normally."""
    module = import_ops_module("tests.fixtures.custom_ops")
    assert module.ScaleByFactorOp is get_ops("fixture_scale_op")


def test_import_ops_module_absolute_file(tmp_path):
    """An absolute .py path is executed and its operation lands in the registry."""
    path, name = write_op_module(tmp_path)
    assert name not in ops_registry

    module = import_ops_module(str(path))

    assert get_ops(name) is module.TmpOp
    assert module.__file__ == str(path)


def test_import_ops_module_relative_to_base_path(tmp_path):
    """A relative path resolves against base_path, not the working directory."""
    path, name = write_op_module(tmp_path)

    module = import_ops_module(path.name, base_path=tmp_path)

    assert get_ops(name) is module.TmpOp


def test_import_ops_module_is_idempotent(tmp_path):
    """Loading the same file twice returns the first module instead of re-registering.

    Re-executing would re-run ``@ops_registry``, which rejects a name that is
    already taken — so without the cache the second call would raise.
    """
    path, _ = write_op_module(tmp_path)

    first = import_ops_module(str(path))
    second = import_ops_module(str(path))

    assert first is second


def test_import_ops_module_same_stem_different_files(tmp_path):
    """Two different files named the same get distinct sys.modules entries."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path, first_name = write_op_module(first_dir, "reconstruct.py")
    second_path, second_name = write_op_module(second_dir, "reconstruct.py")

    first = import_ops_module(str(first_path))
    second = import_ops_module(str(second_path))

    assert first is not second
    assert first.__name__ != second.__name__
    assert get_ops(first_name) is first.TmpOp
    assert get_ops(second_name) is second.TmpOp


def test_import_ops_module_missing_file_raises(tmp_path):
    """A path that does not exist names the resolved path in the error."""
    with pytest.raises(OpsModuleImportError, match="no such file"):
        import_ops_module(str(tmp_path / "nope.py"))


def test_import_ops_module_broken_module_raises(tmp_path):
    """A module that fails to execute is not left behind in sys.modules."""
    path = tmp_path / "broken_ops.py"
    path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(OpsModuleImportError, match="boom"):
        import_ops_module(str(path))

    assert "zea_modules.broken_ops" not in sys.modules


def test_import_ops_module_unknown_dotted_module_raises():
    with pytest.raises(OpsModuleImportError, match="PYTHONPATH"):
        import_ops_module("definitely_not_a_module_zzzzzz")


# ── imports: the remote-code gate ────────────────────────────────────────────


def test_remote_import_without_trust_raises(monkeypatch):
    """An hf:// module is refused by default, and the error names the flag."""
    monkeypatch.delenv("ZEA_TRUST_REMOTE_CODE", raising=False)

    with pytest.raises(RemoteCodeError) as excinfo:
        import_ops_module("hf://org/repo/my_ops.py")

    message = str(excinfo.value)
    assert "--trust-remote-code" in message
    assert "ZEA_TRUST_REMOTE_CODE" in message
    assert "hf://org/repo/my_ops.py" in message


def test_remote_import_with_trust_downloads_and_imports(tmp_path, monkeypatch):
    """With trust granted, the module is fetched and executed."""
    path, name = write_op_module(tmp_path)
    monkeypatch.setattr(
        "zea.internal.preset_utils._hf_resolve_path",
        lambda hf_path, **kwargs: path,
    )

    module = import_ops_module("hf://org/repo/my_ops.py", trust_remote_code=True)

    assert get_ops(name) is module.TmpOp


def test_remote_import_trust_via_env_var(tmp_path, monkeypatch):
    """ZEA_TRUST_REMOTE_CODE=1 grants the same consent as the flag."""
    path, name = write_op_module(tmp_path)
    monkeypatch.setenv("ZEA_TRUST_REMOTE_CODE", "1")
    monkeypatch.setattr(
        "zea.internal.preset_utils._hf_resolve_path",
        lambda hf_path, **kwargs: path,
    )

    module = import_ops_module("hf://org/repo/my_ops.py")

    assert get_ops(name) is module.TmpOp


def test_remote_import_forwards_revision(tmp_path, monkeypatch):
    """The config's revision is passed on, so the module matches the config commit."""
    path, _ = write_op_module(tmp_path)
    seen = {}

    def fake_resolve(hf_path, **kwargs):
        seen.update(hf_path=hf_path, **kwargs)
        return path

    monkeypatch.setattr("zea.internal.preset_utils._hf_resolve_path", fake_resolve)

    import_ops_module("hf://org/repo/my_ops.py", trust_remote_code=True, revision="abc123")

    assert seen == {"hf_path": "hf://org/repo/my_ops.py", "revision": "abc123"}


def test_http_import_is_rejected(monkeypatch):
    """Only hf:// is supported for remote modules; plain URLs are refused."""
    monkeypatch.setenv("ZEA_TRUST_REMOTE_CODE", "1")
    with pytest.raises(OpsModuleImportError, match="only 'hf://'"):
        import_ops_module("https://example.com/my_ops.py")


# ── imports: declared in a pipeline config ───────────────────────────────────


def test_pipeline_config_imports_sibling_module(tmp_path):
    """A config's `imports` entry is resolved relative to the config itself."""
    _, name = write_op_module(tmp_path)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "imports": ["tmp_ops.py"],
                    "operations": [{"name": name, "params": {"factor": 4.0}}],
                }
            }
        ),
        encoding="utf-8",
    )

    pipeline = Pipeline.from_path(str(config_path), jit_options=None)

    out = pipeline(data=np.ones((3,), dtype="float32"))["data"]
    np.testing.assert_allclose(np.asarray(out), np.full((3,), 4.0))


def test_pipeline_config_imports_from_other_directory(tmp_path):
    """The config is found from the working directory but its module is not."""
    module_dir = tmp_path / "code"
    module_dir.mkdir()
    _, name = write_op_module(module_dir)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump({"pipeline": {"imports": ["code/tmp_ops.py"], "operations": [name]}}),
        encoding="utf-8",
    )

    pipeline = Pipeline.from_path(str(config_path), jit_options=None)

    assert isinstance(pipeline.operations[0], get_ops(name))


def test_nested_pipeline_imports_are_loaded(tmp_path):
    """A module declared by a nested pipeline is imported before its ops resolve."""
    _, name = write_op_module(tmp_path)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "operations": [
                        {
                            "name": "pipeline",
                            "params": {"imports": ["tmp_ops.py"]},
                            "operations": [name],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    pipeline = Pipeline.from_path(str(config_path), jit_options=None)

    nested = pipeline.operations[0]
    assert isinstance(nested.operations[0], get_ops(name))


def test_pipeline_config_imports_missing_module_raises(tmp_path):
    """A declared module that is absent fails loudly, naming the resolved path."""
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump({"pipeline": {"imports": ["absent.py"], "operations": ["identity"]}}),
        encoding="utf-8",
    )

    with pytest.raises(OpsModuleImportError, match="absent.py"):
        Pipeline.from_path(str(config_path))


def test_pipeline_imports_roundtrip_through_yaml(tmp_path):
    """`imports` survives to_yaml → from_path, so a saved pipeline stays loadable."""
    path, name = write_op_module(tmp_path)
    import_ops_module(str(path))
    original = Pipeline(
        operations=[get_ops(name)(factor=5.0)],
        imports=[str(tmp_path / "tmp_ops.py")],
        jit_options=None,
    )
    out_path = tmp_path / "roundtrip.yaml"
    original.to_yaml(str(out_path))

    written = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert written["pipeline"]["imports"] == [str(tmp_path / "tmp_ops.py")]

    reloaded = Pipeline.from_path(str(out_path), jit_options=None)
    assert reloaded.imports == original.imports


def test_pipeline_without_imports_omits_the_key(tmp_path):
    """Pipelines that need no modules serialize exactly as they did before."""
    out_path = tmp_path / "plain.yaml"
    Pipeline(operations=[Identity()], jit_options=None).to_yaml(str(out_path))

    written = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert "imports" not in written["pipeline"]


def test_missing_custom_op_error_points_at_imports():
    """The failure a user actually hits explains how to declare the module."""
    with pytest.raises(KeyError) as excinfo:
        get_ops("apply_probe_pose")

    message = excinfo.value.args[0]
    assert "imports:" in message
    assert "--import" in message


def test_imported_op_is_picklable(tmp_path):
    """Ops from a file-loaded module survive pickling, for multiprocessing workers.

    ``zea_modules.<name>`` is not importable from a fresh interpreter, so pickling
    the class by reference would fail; it has to go by value instead.
    """
    cloudpickle = pytest.importorskip("cloudpickle")
    path, _ = write_op_module(tmp_path)

    module = import_ops_module(str(path))
    restored = cloudpickle.loads(cloudpickle.dumps(module.TmpOp))

    assert restored.__name__ == "TmpOp"


def test_failed_import_does_not_leave_operations_registered(tmp_path):
    """A module that registers an op and then raises can be fixed and retried.

    Without a rollback the registry keeps the half-registered name, and the retry
    reports "already registered" instead of the error the user has to fix.
    """
    name = f"fixture_rollback_op_{next(_op_module_counter)}"
    path = tmp_path / "half_broken_ops.py"
    body = _OP_MODULE_TEMPLATE.format(name=name)
    path.write_text(body + "\nraise RuntimeError('boom halfway')\n", encoding="utf-8")

    with pytest.raises(OpsModuleImportError, match="boom halfway"):
        import_ops_module(str(path))
    assert name not in ops_registry

    # The same error again, rather than a misleading one about the registry.
    with pytest.raises(OpsModuleImportError, match="boom halfway"):
        import_ops_module(str(path))

    # And once the module is corrected, it loads.
    path.write_text(body, encoding="utf-8")
    module = import_ops_module(str(path))
    assert get_ops(name) is module.TmpOp


def test_imports_on_a_plain_operation_are_not_loaded(tmp_path):
    """`imports` is a pipeline key, so an operation parameter of that name is ignored.

    A custom operation is free to take a parameter called ``imports``; it must not be
    mistaken for a list of modules to import.
    """
    _, name = write_op_module(tmp_path)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "imports": ["tmp_ops.py"],
                    "operations": [{"name": name, "params": {"imports": ["does_not_exist.py"]}}],
                }
            }
        ),
        encoding="utf-8",
    )

    # Reaches the operation itself (which has no such parameter) rather than
    # failing earlier on a missing module.
    with pytest.raises(TypeError):
        Pipeline.from_path(str(config_path), jit_options=None)


def test_nested_pipeline_imports_survive_a_config_roundtrip(tmp_path):
    """Nested `imports` still load after to_yaml → from_path.

    A Config round-trip turns the operations list into a numpy array, which the
    hand-written configs in the tests above do not exercise.
    """
    path, name = write_op_module(tmp_path)
    import_ops_module(str(path))
    inner = Pipeline(
        operations=[get_ops(name)()],
        imports=[str(path)],
        jit_options=None,
        name="inner",
    )
    out_path = tmp_path / "nested.yaml"
    Pipeline(operations=[inner], jit_options=None).to_yaml(str(out_path))

    written = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert written["pipeline"]["operations"][0]["params"]["imports"] == [str(path)]

    reloaded = Pipeline.from_path(str(out_path), jit_options=None)
    assert reloaded.operations[0].imports == [str(path)]
