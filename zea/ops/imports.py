"""Import modules that define custom operations, so a config can name them.

A :class:`~zea.ops.Pipeline` resolves operation names through
:data:`~zea.internal.registry.ops_registry`, which is populated by the
``@ops_registry(...)`` decorators that run when a module is imported. For
operations that are part of ``zea`` that happens automatically. For operations
defined elsewhere, something has to import the defining module first.

This module is that something. It takes a *source* -- a dotted module path, a
path to a ``.py`` file, or an ``hf://`` URI -- and imports it, so that a config
naming the operations inside can be loaded:

.. code-block:: yaml

    pipeline:
        imports:
          - reconstruct.py     # resolved relative to this config
        operations:
          - name: apply_probe_pose

See :ref:`custom-ops` for the full story.
"""

import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from zea import log

#: Package under which modules loaded from a file are registered in
#: ``sys.modules``, so they stay importable (and picklable) afterwards.
_MODULE_NAMESPACE = "zea_modules"

#: Schemes whose code is fetched from somewhere else, and therefore gated.
_REMOTE_PREFIXES = ("hf://", "http://", "https://")

_TRUST_ENV_VAR = "ZEA_TRUST_REMOTE_CODE"

# Resolved source (absolute file path or dotted name) -> module. Re-executing a
# module would re-run its ``@ops_registry`` decorators, and the registry rejects
# a name that is already taken, so loading has to be idempotent.
_LOADED: dict[str, ModuleType] = {}


class RemoteCodeError(RuntimeError):
    """Raised when a remote module would be executed without explicit consent.

    Kept distinct from :exc:`ValueError` / :exc:`KeyError` so that callers which
    catch pipeline-construction failures broadly do not swallow it.
    """


class OpsModuleImportError(ImportError):
    """Raised when a module declared in ``imports`` cannot be imported."""


def _is_remote(source: str) -> bool:
    return source.startswith(_REMOTE_PREFIXES)


def _is_dotted_module(source: str) -> bool:
    """True for ``my_project.my_ops``, false for ``./my_ops.py`` or a URI."""
    if _is_remote(source) or source.endswith(".py"):
        return False
    return "/" not in source and "\\" not in source


def _trust_remote_code(explicit: bool) -> bool:
    if explicit:
        return True
    return os.environ.get(_TRUST_ENV_VAR, "").lower() in ("1", "true", "yes")


def _resolve_remote(source: str, revision: str | None) -> Path:
    """Download a remote module and return its local path."""
    if not source.startswith("hf://"):
        raise OpsModuleImportError(
            f"Cannot import '{source}': only 'hf://' remote modules are supported. "
            "Download the file yourself and pass a local path instead."
        )
    # Imported lazily: huggingface_hub is heavy and zea.ops stays import-light.
    from zea.internal.preset_utils import _hf_resolve_path

    hf_kwargs = {"revision": revision} if revision else {}
    try:
        return Path(_hf_resolve_path(source, **hf_kwargs))
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise OpsModuleImportError(f"Could not download module '{source}': {exc}") from exc


def _join_base_path(source: str, base_path) -> str:
    """Resolve a relative ``source`` against the location of its config."""
    base = str(base_path)
    if base.startswith("hf://"):
        # Keep the hf:// scheme rather than the local cache dir, so the module is
        # fetched from the same repo (and revision) as the config that named it.
        from zea.tools.hf import HFPath

        return str(HFPath(base) / source)
    return str(Path(base) / source)


def parent_location(path) -> str:
    """Directory holding ``path``, preserving an ``hf://`` scheme.

    Used to turn the location of a config into the base path that its relative
    ``imports`` entries resolve against.
    """
    path = str(path)
    if _is_remote(path):
        # PurePath.parent would drop the scheme, so split the URI by hand.
        return path.rsplit("/", 1)[0] if "/" in path[len("hf://") :] else path
    return str(Path(path).expanduser().resolve().parent)


def _module_name_for(path: Path) -> str:
    """A ``sys.modules`` name for ``path``, unique across same-named files.

    Two datasets may each have a ``reconstruct.py``, so the bare stem is not
    enough; a digest of the full path is appended only when it has to be, so that
    the common case stays readable in tracebacks.
    """
    base = f"{_MODULE_NAMESPACE}.{path.stem}"
    existing = sys.modules.get(base)
    if existing is None or getattr(existing, "__file__", None) == str(path):
        return base
    return f"{base}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"


def _import_from_file(path: Path) -> ModuleType:
    """Execute a ``.py`` file as a module and return it."""
    module_name = _module_name_for(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OpsModuleImportError(f"Could not build an import spec for '{path}'.")

    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that the module resolves to this same instance
    # if it refers to itself, and so tracebacks and introspection name it properly.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        del sys.modules[module_name]
        raise OpsModuleImportError(f"Error while importing '{path}': {exc}") from exc

    _register_pickle_by_value(module)
    return module


def _register_pickle_by_value(module: ModuleType) -> None:
    """Make the module's classes picklable, for multiprocessing and jitted workers.

    Being in ``sys.modules`` normally means pickle stores a class by reference, but
    ``zea_modules.<name>`` is not importable from a fresh interpreter, so that
    reference cannot be resolved again and pickling fails outright. cloudpickle can
    serialize the module's contents by value instead.

    Best effort: cloudpickle is not a runtime dependency of zea, so this only applies
    when something else has already imported it.
    """
    cloudpickle = sys.modules.get("cloudpickle")
    if cloudpickle is None:
        return
    try:
        cloudpickle.register_pickle_by_value(module)
    except Exception:  # noqa: BLE001 - never let a pickling nicety break the import
        log.debug(f"Could not register {module.__name__} for pickling by value.")


def import_ops_module(
    source: str,
    *,
    base_path=None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> ModuleType:
    """Import a module so that the operations it defines enter the registry.

    Importing the same source twice in one process is a no-op: the first module
    object is returned again, rather than re-running its registrations.

    Args:
        source (str): What to import. One of

            * a dotted module path, e.g. ``"my_project.my_ops"`` -- imported
              normally, so it must be on ``sys.path``;
            * a path to a ``.py`` file, absolute or relative to ``base_path``;
            * an ``hf://org/repo/path/to/module.py`` URI, downloaded to the zea
              cache and then executed.

        base_path (str or Path, optional): Location a relative ``source`` is
            resolved against -- typically the directory holding the config that
            declared it. May itself be an ``hf://`` path. Defaults to ``None``
            (relative paths resolve against the working directory).
        trust_remote_code (bool, optional): Allow executing code fetched from a
            remote source. Also settable through ``ZEA_TRUST_REMOTE_CODE=1``.
            Defaults to ``False``.
        revision (str, optional): Hugging Face revision (branch, tag or commit)
            for ``hf://`` sources. Defaults to ``None``.

    Returns:
        ModuleType: The imported module.

    Raises:
        RemoteCodeError: If ``source`` is remote and remote code is not trusted.
        OpsModuleImportError: If the module cannot be found or fails to import.

    Examples:
        .. code-block:: python

            from zea.ops import import_ops_module

            import_ops_module("my_project.my_ops")
            import_ops_module("./my_ops.py")
    """
    source = str(source)

    if _is_dotted_module(source):
        if source in _LOADED:
            return _LOADED[source]
        try:
            module = importlib.import_module(source)
        except ImportError as exc:
            raise OpsModuleImportError(
                f"Could not import module '{source}': {exc}. Make sure it is installed "
                "or on your PYTHONPATH, or point to the .py file directly."
            ) from exc
        _LOADED[source] = module
        return module

    # A relative path is only meaningful next to the config that named it.
    if base_path is not None and not _is_remote(source) and not Path(source).is_absolute():
        source = _join_base_path(source, base_path)

    if _is_remote(source):
        if not _trust_remote_code(trust_remote_code):
            raise RemoteCodeError(
                f"The pipeline needs a module fetched from a remote location:\n"
                f"  {source}\n"
                "Loading it executes that code on your machine. If you trust the source, "
                "re-run with --trust-remote-code (or set "
                f"{_TRUST_ENV_VAR}=1)."
            )
        path = _resolve_remote(source, revision)
    else:
        path = Path(source).expanduser().resolve()

    key = str(path)
    if key in _LOADED:
        return _LOADED[key]

    if not path.is_file():
        raise OpsModuleImportError(
            f"Could not import '{source}': no such file '{path}'. Paths in "
            "'imports' are resolved relative to the config that declares them."
        )

    log.debug(f"Importing custom operations from {log.yellow(str(path))}")
    module = _import_from_file(path)
    _LOADED[key] = module
    return module


def import_ops_modules(
    sources,
    *,
    base_path=None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> list[ModuleType]:
    """Import several modules in order. See :func:`import_ops_module`."""
    return [
        import_ops_module(
            source,
            base_path=base_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        for source in (sources or [])
    ]
