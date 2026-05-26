"""This file contains fixtures that are used by all tests in the tests directory."""

import os
import tempfile

import matplotlib.pyplot as plt
import pytest

from .backend_utils import (
    ML_BACKENDS,
    available_test_backends,
    backend_guard_skips,
    format_backend_skip_reason,
    format_missing_backend_details,
    missing_required_backends,
    get_test_backend,
)

_tmp_cache_dir = tempfile.TemporaryDirectory(prefix="zea_test_cache_")

os.environ["ZEA_CACHE_DIR"] = _tmp_cache_dir.name  # set before importing zea
os.environ["KERAS_BACKEND"] = get_test_backend()

plt.rcParams["backend"] = "agg"


def _skip_unavailable_backends_enabled(config):
    return bool(config.getoption("--skip-unavailable-backends"))


def _gpu_available():
    """Check whether any ML backend has CUDA available in this environment."""
    from zea.internal.device import backend_cuda_available

    return any(backend_cuda_available(backend) for backend in ML_BACKENDS)


def _required_backends_for_item(item):
    required_backends = [backend for backend in ML_BACKENDS if backend in item.keywords]
    required_backends.extend(
        backend
        for backend in getattr(item.obj, "_required_backends", ())
        if backend in ML_BACKENDS and backend not in required_backends
    )
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        backend = callspec.params.get("backend")
        if backend in ML_BACKENDS and backend not in required_backends:
            required_backends.append(backend)
    return tuple(required_backends)


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--notebook",
        action="store",
        default=None,
        help="Run only the notebook matching this name (e.g. --notebook dbua_example.ipynb)",
    )
    parser.addoption(
        "--skip-unavailable-backends",
        action="store_true",
        default=False,
        help="Skip tests that require ML backends unavailable in the current environment.",
    )


def pytest_configure(config):
    """Validate backend availability before importing backend-dependent test modules."""
    for backend in ML_BACKENDS:
        config.addinivalue_line("markers", f"{backend}: test requires the {backend} backend")

    os.environ["ZEA_SKIP_UNAVAILABLE_BACKENDS"] = (
        "1" if _skip_unavailable_backends_enabled(config) else "0"
    )
    available = available_test_backends()
    if not available:
        raise pytest.UsageError(
            "No supported ML back-end is available. Install at least one of tensorflow, "
            "torch, or jax before running the test suite."
        )
    if len(available) < len(ML_BACKENDS) and not _skip_unavailable_backends_enabled(config):
        raise pytest.UsageError(
            "Not all back-ends are available, meaning tests will fail.\n"
            "To skip tests that require unavailable back-ends, "
            "use pytest --skip-unavailable-backends.\n\n"
            f"{format_missing_backend_details()}"
        )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not _gpu_available():
            skip_gpu = pytest.mark.skip(reason="No CUDA GPU available at runtime")
            item.add_marker(skip_gpu)
        if _skip_unavailable_backends_enabled(config):
            missing_backends = missing_required_backends(_required_backends_for_item(item))
            if missing_backends:
                item.add_marker(
                    pytest.mark.skip(reason=format_backend_skip_reason(missing_backends))
                )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    guard_skip_counts = backend_guard_skips()
    guard_skips = sum(guard_skip_counts.values())
    if not guard_skips:
        return

    terminalreporter.write_sep("-", "backend-guarded assert skips")
    terminalreporter.write_line(
        f"Skipped {guard_skips} backend-guarded assert{'' if guard_skips == 1 else 's'}."
    )
    for (active_backend, required_backends), count in sorted(guard_skip_counts.items()):
        terminalreporter.write_line(f"{count} requiring {', '.join(required_backends)}")


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    try:
        from . import backend_workers
    except ImportError:
        return

    print("Stopping workers")
    backend_workers.stop_workers()


@pytest.fixture(scope="session", autouse=True)
def clean_cache_dir():
    """Fixture to clean the cache directory after all tests."""
    yield
    print("Cleaning cache directory")
    _tmp_cache_dir.cleanup()


@pytest.fixture
def dummy_file(tmp_path):
    """Fixture to create a temporary dataset"""
    from zea.data.data_format import generate_example_dataset

    from . import DUMMY_DATASET_GRID_SIZE_X, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_N_FRAMES

    temp_file = tmp_path / "test.hdf5"
    generate_example_dataset(
        temp_file,
        add_optional_dtypes=True,
        n_frames=DUMMY_DATASET_N_FRAMES,
        grid_size_z=DUMMY_DATASET_GRID_SIZE_Z,
        grid_size_x=DUMMY_DATASET_GRID_SIZE_X,
    )

    yield str(temp_file)


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """Fixture to create a temporary dataset"""
    from zea.data.data_format import generate_example_dataset

    from . import DUMMY_DATASET_GRID_SIZE_X, DUMMY_DATASET_GRID_SIZE_Z, DUMMY_DATASET_N_FRAMES

    for i in range(2):
        temp_file = tmp_path / "dummy_dataset_path" / f"test{i}.hdf5"
        generate_example_dataset(
            temp_file,
            add_optional_dtypes=True,
            n_frames=DUMMY_DATASET_N_FRAMES,
            grid_size_z=DUMMY_DATASET_GRID_SIZE_Z,
            grid_size_x=DUMMY_DATASET_GRID_SIZE_X,
        )

    yield str(temp_file.parent)
