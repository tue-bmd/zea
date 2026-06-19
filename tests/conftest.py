"""This file contains fixtures that are used by all tests in the tests directory."""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from zea.internal.device import backend_cuda_available, init_device

# Device setup for the test session. Kept here (and not in tests/__init__.py) on purpose:
# init_device imports tensorflow -> keras, which locks the keras backend. The spawned
# BackendEqualityCheck workers re-import the `tests` package but never load conftest, so
# they remain free to select their own backend. See tests/__init__.py for details.
device = init_device(allow_preallocate=False)

_GPU_AVAILABLE = backend_cuda_available(os.environ.get("KERAS_BACKEND"))


from . import (  # noqa: E402
    DUMMY_DATASET_GRID_SIZE_X,
    DUMMY_DATASET_GRID_SIZE_Z,
    DUMMY_DATASET_N_FRAMES,
    _notebook_timings,
    backend_workers,
)
from .data import generate_example_dataset  # noqa: E402

plt.rcParams["backend"] = "agg"


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--notebook",
        action="store",
        default=None,
        help="Run only the notebook matching this name (e.g. --notebook dbua_example.ipynb)",
    )
    parser.addoption(
        "--notebook-dir",
        action="append",
        default=None,
        help="Run only notebooks under this subfolder (e.g. --notebook-dir models)."
        " Can be repeated.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``@pytest.mark.gpu`` tests when no CUDA GPU is accessible.
    Also announce notebook count only when notebook tests are actually collected.
    """
    has_notebooks = any("notebook" in item.nodeid for item in items)
    if has_notebooks:
        notebooks_dir = Path("docs/source/notebooks")
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        if notebooks:
            print(f"\n📚 Preparing to test {len(notebooks)} notebooks from {notebooks_dir}")

    if _GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available at runtime")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


def pytest_sessionfinish(session, exitstatus):
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


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    print("Stopping workers")
    backend_workers.stop_workers()


@pytest.fixture
def dummy_file(tmp_path):
    """Fixture to create a temporary dataset"""
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
