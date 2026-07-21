"""This file contains fixtures that are used by all tests in the tests directory."""

import importlib.util
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from .data import generate_example_dataset


def _gpu_available() -> bool:
    """Check in subprocesses so cpu tensorflow installs don't poison the CUDA state."""
    probe = (
        "import sys;"
        "from zea.internal.device import backend_cuda_available;"
        "sys.exit(0 if backend_cuda_available(sys.argv[1]) else 1)"
    )
    for backend in ("torch", "tensorflow", "jax"):
        # Skip subprocess call for backends that aren't installed
        if importlib.util.find_spec(backend) is None:
            continue

        try:
            result: subprocess.CompletedProcess[bytes] = subprocess.run(
                [sys.executable, "-c", probe, backend],
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue
        if result.returncode == 0:
            return True
    return False


# must be before importing anything that may call init_device()
_GPU_AVAILABLE = _gpu_available()

# Warm up jax's GPU backend while all GPUs are still visible. If the first init happens when
# a test has hidden the GPUs, jax permanently falls back to CPU.
if _GPU_AVAILABLE:
    try:
        import jax as _jax

        _jax.devices("gpu")
    except Exception:
        pass

# Device setup for the test session. Kept here (and not in tests/__init__.py) on purpose:
# init_device imports tensorflow -> keras, which locks the keras backend. The spawned
# BackendEqualityCheck workers re-import the `tests` package but never load conftest, so
# they remain free to select their own backend. See tests/__init__.py for details.
from zea.internal.device import init_device  # noqa: E402

device = os.environ.get("ZEA_TEST_DEVICE", "auto:1")
device = init_device(device=device, allow_preallocate=False)


from .backend_utils import (  # noqa: E402
    ML_BACKENDS,
    available_test_backends,
    backend_guard_skips,
    format_backend_skip_reason,
    format_missing_backend_details,
    missing_required_backends,
    unavailable_test_backends,
)

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


def _enable_subprocess_coverage():
    """Let coverage follow into subprocesses spawned by tests.

    Several tests exercise CLIs through ``subprocess`` (e.g.
    ``python -m zea.data.convert ...``); without this the code executed there is
    invisible to coverage. When the suite runs under ``pytest --cov`` we point
    ``COVERAGE_PROCESS_START`` at our config so the repo-root ``sitecustomize.py``
    starts coverage in each subprocess (parallel mode writes ``.coverage.*`` files
    that pytest-cov combines). Outside a coverage run this is a no-op.
    """
    try:
        import coverage
    except ImportError:  # pragma: no cover - coverage is always installed during a coverage run
        return
    if coverage.Coverage.current() is None:  # pragma: no cover - only when coverage is inactive
        return
    config_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    os.environ["COVERAGE_PROCESS_START"] = str(config_path)


def pytest_configure(config):
    """Validate backend availability before importing backend-dependent test modules."""
    _enable_subprocess_coverage()

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
    """Auto-skip ``@pytest.mark.gpu`` tests when no CUDA GPU is accessible.
    Also announce notebook count only when notebook tests are actually collected.
    """
    marker_expr = getattr(config.option, "markexpr", "") or ""
    if not marker_expr:
        notebooks_selected = True
    elif "not notebook" in marker_expr:
        notebooks_selected = False
    elif "notebook" in marker_expr:
        notebooks_selected = True
    else:
        notebooks_selected = False

    # Pre-count backend-skipped items so both announcements print together.
    backend_skip_count = 0
    if _skip_unavailable_backends_enabled(config):
        for item in items:
            if missing_required_backends(_required_backends_for_item(item)):
                backend_skip_count += 1

    # Announcements — printed together, before pytest's "collected N items" line.
    has_notebooks = any("notebook" in item.nodeid for item in items)
    if has_notebooks and notebooks_selected:
        notebooks_dir = Path("docs/source/notebooks")
        notebooks = list(notebooks_dir.rglob("*.ipynb"))
        notebook_filter = config.getoption("--notebook")
        notebook_dir_filter = config.getoption("--notebook-dir")
        if notebook_filter:
            notebooks = [nb for nb in notebooks if notebook_filter in nb.name]
        if notebook_dir_filter:
            notebooks = [nb for nb in notebooks if nb.parent.name in notebook_dir_filter]
        if notebooks:
            print(f"\n📚 Preparing to test {len(notebooks)} notebooks from {notebooks_dir}")

    if _skip_unavailable_backends_enabled(config):
        unavailable = unavailable_test_backends()
        if unavailable:
            missing_str = ", ".join(unavailable)
            n = backend_skip_count
            suffix = f" — {n} test(s) will be skipped." if n else ""
            print(f"\n⚠️  Missing backends: {missing_str}{suffix}")

    # Apply skip markers.
    for item in items:
        if "gpu" in item.keywords and not _GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="No CUDA GPU available at runtime"))
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
