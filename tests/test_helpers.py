"""Regression tests for the backend test helpers in ``tests/helpers.py``.

These guard against the bug where every ``BackendEqualityCheck`` worker silently
ran in the tensorflow backend: ``tests/__init__.py`` called ``init_device``,
which imports tensorflow (and thus keras), locking the keras backend before the
spawned worker could set ``KERAS_BACKEND``. As a result
``keras.backend.backend()`` returned ``"tensorflow"`` in every worker, so the
"multi-backend" checks were all silently running the same backend.
"""

import pytest

from . import run_in_backend

BACKENDS = ["tensorflow", "torch", "jax"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_worker_runs_in_requested_backend(backend):
    """Each worker must actually run keras in the backend it was started for.

    If keras is imported (and its backend locked) before the worker sets
    ``KERAS_BACKEND``, ``keras.backend.backend()`` would return ``"tensorflow"``
    for every backend and this assertion would fail.
    """

    @run_in_backend(backend)
    def _active_backend():
        import keras

        return keras.backend.backend()

    result = str(_active_backend())
    assert result == backend, (
        f"Worker for {backend!r} actually ran in the {result!r} backend. "
        "keras was likely imported before KERAS_BACKEND could be set in the worker."
    )
