"""Utilities for backend-aware test collection and execution."""

from __future__ import annotations

import importlib.util
from collections import Counter

from zea import log

ML_BACKENDS = ("tensorflow", "torch", "jax")
DEFAULT_TEST_BACKEND = "tensorflow"
FALLBACK_TEST_BACKEND = "jax"

_backend_guard_skips = Counter()


def available_test_backends() -> tuple[str, ...]:
    """Return the installed ML backends available to the test suite."""
    return tuple(
        backend for backend in ML_BACKENDS if importlib.util.find_spec(backend) is not None
    )


def unavailable_test_backends() -> tuple[str, ...]:
    """Return the ML backends missing from the current environment."""
    available = set(available_test_backends())
    return tuple(backend for backend in ML_BACKENDS if backend not in available)


def all_test_backends_available() -> bool:
    """Return whether all supported ML backends are installed."""
    return len(unavailable_test_backends()) == 0


def get_test_backend() -> str:
    """Return the backend the test suite should use for regular imports."""
    available = available_test_backends()
    if DEFAULT_TEST_BACKEND in available:
        return DEFAULT_TEST_BACKEND
    if FALLBACK_TEST_BACKEND in available:
        return FALLBACK_TEST_BACKEND
    assert available, (
        "No supported ML back-end is available. Install at least one of tensorflow, torch, or jax."
    )
    return available[0]


def missing_required_backends(required_backends) -> tuple[str, ...]:
    """Return the missing backend subset for a test requirement list."""
    required = set(required_backends)
    available = set(available_test_backends())
    return tuple(
        backend for backend in ML_BACKENDS if backend in required and backend not in available
    )


def format_missing_backend_details() -> str:
    """Return a user-facing summary of available and unavailable backends."""
    available = ", ".join(available_test_backends()) or "none"
    unavailable = ", ".join(unavailable_test_backends()) or "none"
    return f"Available back-ends: {available}. Unavailable back-ends: {unavailable}."


def format_backend_skip_reason(missing_backends) -> str:
    """Return the standardized pytest skip reason for missing backends."""
    missing = ", ".join(missing_backends)
    return f"Skipping test because required back-end(s) are unavailable: {missing}."


def _record_backend_guard_skip(active_backend, required_backends, inclusive=True):
    if inclusive:
        msg = (
            f"Assert skipped. Only available on {required_backends}, "
            f"currently running on {active_backend}."
        )
    else:
        msg = (
            f"Assert skipped. Not available on {required_backends}, "
            f"currently running on {active_backend}."
        )
    log.warning(msg)
    _backend_guard_skips[(active_backend, tuple(required_backends))] += 1


def runs_on(*backends) -> bool:
    """Return whether the current test backend is one of the requested backends.
    Misses are counted so pytest can report how many backend-guarded blocks were skipped.
    """
    import keras

    active_backend = keras.backend.backend()
    if active_backend in backends:
        return True
    _record_backend_guard_skip(active_backend, backends, inclusive=True)
    return False


def runs_not_on(*backends) -> bool:
    """Return whether the current test backend is not one of the excluded backends.
    Misses are counted so pytest can report how many backend-guarded blocks were skipped.
    """
    import keras

    active_backend = keras.backend.backend()
    if active_backend not in backends:
        return True
    _record_backend_guard_skip(active_backend, backends, inclusive=False)
    return False


def backend_guard_skips():
    """Return counts for backend-guarded blocks that were not executed."""
    return dict(_backend_guard_skips)
