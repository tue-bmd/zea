"""This file contains fixtures that are used by all tests in the tests directory."""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from zea.data.data_format import generate_example_dataset  # noqa: E402
from zea.internal.device import backend_cuda_available  # noqa: E402

# must be before importing anything that may call init_device()
_GPU_AVAILABLE = any(backend_cuda_available(b) for b in ["torch", "tensorflow", "jax"])

from .backend_utils import (  # noqa: E402
    ML_BACKENDS,
    available_test_backends,
    backend_guard_skips,
    format_backend_skip_reason,
    format_missing_backend_details,
    missing_required_backends,
    get_test_backend,
)

os.environ["KERAS_BACKEND"] = get_test_backend()

plt.rcParams["backend"] = "agg"


def _skip_unavailable_backends_enabled(config):
    return bool(config.getoption("--skip-unavailable-backends")) or bool(
        config.getoption("--torch-override")
    )


def _torch_override(config):
    return bool(config.getoption("--torch-override"))


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
    parser.addoption(
        "--torch-override",
        action="store_true",
        default=False,
        help="Run tests with the torch backend. "
        "Torch support is currently in alpha, and tests are expected to fail.",
    )
    parser.addoption(
        "--notebook-dir",
        action="append",
        default=None,
        help="Run only notebooks under this subfolder (e.g. --notebook-dir models)."
        " Can be repeated.",
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
    if len(available) == 1 and available[0] == "torch" and not _torch_override(config):
        raise pytest.UsageError(
            "Only the torch back-end is available, and torch support is currently alpha. "
            "Some tests are expected to fail. \n"
            "To run the test suite anyway, use pytest --torch-override."
        )


def pytest_sessionstart(session):
    notebooks_dir = Path("docs/source/notebooks")
    notebooks = list(notebooks_dir.rglob("*.ipynb"))
    if notebooks:
        print(f"📚 Preparing to test {len(notebooks)} notebooks from {notebooks_dir}")


def pytest_sessionfinish(session, exitstatus):
    from . import _notebook_timings

    if not _notebook_timings:
        return

    by_folder: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name, (folder, duration) in sorted(_notebook_timings.items()):
        by_folder[folder].append((name, duration))

    col_w = max(len(name) for name, _ in _notebook_timings.items()) + 2
    print("\n" + "=" * (col_w + 20))
    print("📊 Notebook run-time summary")
    print("=" * (col_w + 20))

    grand_total = 0.0
    for folder in sorted(by_folder):
        entries = sorted(by_folder[folder], key=lambda x: -x[1])
        folder_total = sum(d for _, d in entries)
        grand_total += folder_total
        print(f"\n  📁 {folder}  ({folder_total:.1f}s total)")
        for name, duration in entries:
            mins, secs = divmod(duration, 60)
            time_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
            print(f"    {name:<{col_w}}  {time_str:>8}")

    print("\n" + "-" * (col_w + 20))
    grand_mins, grand_secs = divmod(grand_total, 60)
    grand_str = f"{int(grand_mins)}m {grand_secs:.1f}s" if grand_mins else f"{grand_secs:.1f}s"
    print(f"  {'TOTAL':<{col_w}}  {grand_str:>8}")
    print("=" * (col_w + 20) + "\n")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not _GPU_AVAILABLE:
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


@pytest.fixture
def dummy_file(tmp_path):
    """Fixture to create a temporary dataset"""

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
