"""Health check on the public zeahub datasets.

Asserts, per dataset, that the published files are not behind the file format and that
the dataset still has the shape we expect: the right number of HDF5 files and roughly
the right total size. Reads HDF5 headers only — no data transfer.

These are statements about the Hub, not about this repository. A failure means a
dataset needs attention, not that the code is broken.
"""

import warnings

import pytest

# Oldest zea version the published datasets may carry. Bump only after the datasets
# have actually been re-converted; deliberately not derived from zea.__version__, since
# a release that does not change the file format leaves them valid.
#
# 0.1.4 is the floor because v0.1.3 wrote HDF5 with libver="latest", which readers on
# HDF5 1.14.x reject (PR #520), and older files lack the chunked/Blosc layout.
MIN_DATASET_VERSION = (0, 1, 4)

# Public, maintained datasets: repo -> (number of HDF5 files, approximate total GB).
# Sizes drift a little when a dataset is
# re-converted, so they are checked loosely; the file count is not expected to change.
DATASETS = {
    "zeahub/camus": (1000, 24.8),
    "zeahub/camus-sample": (6, 0.16),
    "zeahub/cetus-miccai-2014": (90, 2.4),
    "zeahub/DehazingEcho2025": (1, 0.002),
    "zeahub/echoxflow": (13619, 275.7),
    "zeahub/phantoms": (2, 5.8),
    "zeahub/picmus": (12, 0.65),
    "zeahub/simulations": (1, 0.27),
    "zeahub/zea-cardiac-2026": (3, 5.2),
    "zeahub/zea-carotid-2023": (80, 435.4),
    "zeahub/zea-fat-layer-2025": (13, 2.0),
    "zeahub/zea-rotating-disk": (1, 0.75),
}

SIZE_TOLERANCE = 0.25

_INFRA_ERRORS = (
    "401",
    "403",
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "name resolution",
    "offline",
)


def _parse(version) -> tuple[int, ...]:
    """Numeric prefix of a version string: ``'0.1.0a1'`` -> ``(0, 1, 0)``."""
    parts = []
    for piece in str(version).split(".")[:3]:
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _is_infrastructure_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _INFRA_ERRORS)


@pytest.mark.heavy
@pytest.mark.parametrize("repo_id,expected", DATASETS.items(), ids=list(DATASETS))
def test_published_dataset_is_healthy(repo_id, expected):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")  # registers Blosc
    huggingface_hub = pytest.importorskip("huggingface_hub")

    expected_files, expected_gb = expected
    floor = ".".join(str(part) for part in MIN_DATASET_VERSION)

    try:
        api = huggingface_hub.HfApi()
        entries = [
            entry
            for entry in api.list_repo_tree(repo_id, recursive=True, repo_type="dataset")
            if hasattr(entry, "blob_id")
        ]
        hdf5 = [e for e in entries if e.path.endswith((".hdf5", ".h5"))]
        total_gb = sum(e.size or 0 for e in entries) / 1e9

        smallest = min(hdf5, key=lambda e: e.size or 0) if hdf5 else None
        version = None
        if smallest is not None:
            filesystem = huggingface_hub.HfFileSystem()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a stale file warns; we assert below
                with filesystem.open(f"datasets/{repo_id}/{smallest.path}", "rb") as handle:
                    with h5py.File(handle, "r") as h5:
                        version = h5.attrs.get("zea_version", None)
    except Exception as exc:  # noqa: BLE001 — classify, then skip or re-raise
        if _is_infrastructure_error(exc):
            pytest.skip(f"Hub unavailable for {repo_id}: {type(exc).__name__}: {exc}")
        raise

    assert len(hdf5) == expected_files, (
        f"On the Hub, {repo_id} has {len(hdf5)} HDF5 file(s), expected {expected_files}. "
        f"Files were added or removed on the dataset — update DATASETS if deliberate."
    )
    assert abs(total_gb - expected_gb) <= SIZE_TOLERANCE * expected_gb, (
        f"On the Hub, {repo_id} totals {total_gb:.2f} GB, expected about {expected_gb} GB. "
        f"Update DATASETS if the dataset was deliberately re-converted or changed."
    )
    assert version is not None, (
        f"On the Hub, {repo_id}/{smallest.path} has no 'zea_version' attribute, so it "
        f"predates zea 0.1.0. Re-convert the dataset and point its main branch at the result."
    )
    assert _parse(version) >= MIN_DATASET_VERSION, (
        f"On the Hub, {repo_id}/{smallest.path} was written by zea {version}, but the "
        f"published datasets must be at {floor} or newer. Re-convert it, or lower "
        f"MIN_DATASET_VERSION if the older format is genuinely still fine."
    )
