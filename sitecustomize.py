"""Enable coverage measurement inside subprocesses.

Python imports ``sitecustomize`` automatically at interpreter startup whenever it
is importable, which it is here because the repo root is on ``sys.path`` for
subprocesses launched from it (e.g. ``python -m zea.data.convert`` as spawned by
the conversion tests). ``coverage.process_startup()`` is a no-op unless the
``COVERAGE_PROCESS_START`` environment variable points at a coverage config, so
this module has no effect outside of a coverage run. The test suite sets that
variable (see ``tests/conftest.py``) only while coverage is active, so the code
executed in those subprocesses is measured and combined into the report.
"""

try:
    import coverage

    coverage.process_startup()
except ImportError:  # pragma: no cover - coverage is always installed during a coverage run
    pass
