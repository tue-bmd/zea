"""Preset utils for zea datasets hosted on Hugging Face.

See https://huggingface.co/zea/
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, login
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
)

from zea.internal.cache import ZEA_CACHE_DIR

HF_DATASETS_DIR = ZEA_CACHE_DIR / "huggingface" / "datasets"
HF_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

HF_SCHEME = "hf"
HF_PREFIX = "hf://"


def _hf_parse_path(hf_path: str):
    """Parse hf://repo_id[/subpath] into (repo_id, subpath or None)."""
    if not hf_path.startswith(HF_PREFIX):
        raise ValueError(f"Invalid hf_path: {hf_path}. It must start with '{HF_PREFIX}'.")
    path = hf_path.removeprefix(HF_PREFIX)
    parts = path.split("/")
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subpath


def _hf_list_files(repo_id):
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except (RepositoryNotFoundError, HFValidationError, EntryNotFoundError):
        login(new_session=False)
        files = list_repo_files(repo_id, repo_type="dataset")
    return files


def _hf_download(repo_id, filename, cache_dir=HF_DATASETS_DIR):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type="dataset",
    )


def _hf_get_snapshot_dir(repo_id, cache_dir=HF_DATASETS_DIR):
    repo_cache_dir = Path(cache_dir) / f"datasets--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        # Try to trigger a download to populate the cache
        files = _hf_list_files(repo_id)
        # Pick the first file (prefer .h5/.hdf5 if possible)
        h5_files = [f for f in files if f.endswith(".h5") or f.endswith(".hdf5")]
        target_file = h5_files[0] if h5_files else files[0]
        _hf_download(repo_id, target_file, cache_dir)
        # Now try again
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            raise FileNotFoundError(
                f"No snapshots found in Hugging Face cache for {repo_id} after download attempt"
            )
    snapshot_hashes = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshot_hashes:
        raise FileNotFoundError(f"No snapshot found for {repo_id} in cache.")
    return snapshot_hashes[0]


def _hf_resolve_path(hf_path: str, cache_dir=HF_DATASETS_DIR):
    """Download a file or directory from Hugging Face Hub to a local cache directory.
    Returns the local path to the downloaded file or directory.
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    files = _hf_list_files(repo_id)
    snapshot_dir = _hf_get_snapshot_dir(repo_id, cache_dir)

    def is_h5(f):
        return f.endswith(".h5") or f.endswith(".hdf5")

    if subpath:
        # Directory
        if any(f.startswith(subpath + "/") for f in files):
            local_dir = snapshot_dir / subpath
            for f in files:
                if f.startswith(subpath + "/") and is_h5(f):
                    _hf_download(repo_id, f, cache_dir)
            if not local_dir.exists():
                raise FileNotFoundError(f"Directory {local_dir} not found after download.")
            return local_dir
        # File
        elif any(f == subpath for f in files) and is_h5(subpath):
            _hf_download(repo_id, subpath, cache_dir)
            local_file = snapshot_dir / subpath
            if not local_file.exists():
                raise FileNotFoundError(f"File {local_file} not found after download.")
            return local_file
        else:
            raise FileNotFoundError(f"{subpath} not found in {repo_id}")
    else:
        # All .h5/.hdf5 files in repo
        for f in files:
            if is_h5(f):
                _hf_download(repo_id, f, cache_dir)
        return snapshot_dir
