"""
zea.data.publish
================

Maintainer tooling to publish a zea dataset to the Hugging Face Hub in the
cloud-optimized layout, and to migrate the datasets that predate it.

Datasets published before zea 0.1.3 are lzf-compressed and chunked per
``(frame, transmit)``. Neither works for the virtual read path: Zarr cannot decode
lzf, and the fine chunking makes for many tiny range requests. :func:`publish_dataset`
runs the migration end to end:

1. **Resave** every file with the current defaults — Blosc(zstd) + one chunk per frame
   (:func:`zea.data.file_operations.resave`).
2. **Upload** the resaved files to the Hub.
3. **Virtualize** the *uploaded* files, pinning the chunk URLs to the commit just
   created, and publish the reference at ``virtual/index.json`` (plus its parameter
   sidecar). Building against the Hub, rather than the local copies, also verifies
   that what was published is readable at its final URLs.

Readers then get the cloud path for free::

    ds = zea.Dataset("hf://zeahub/my-dataset", lazy="virtual")
    ds.virtual["raw_data"][0, 0:4]

This is deliberately explicit, per-dataset tooling: zea ships it, but never migrates
anything on its own. It needs write access to the repo (``HF_TOKEN``, or ``hf auth
login``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from zea import log
from zea.data.virtual import VIRTUAL_INDEX_PATH, build_virtual_reference
from zea.internal.preset_utils import HF_PREFIX, _hf_login, _hf_resolve_path

# Directory in the repo holding the virtual reference and its parameter sidecar.
VIRTUAL_DIR = VIRTUAL_INDEX_PATH.rsplit("/", 1)[0]


def publish_dataset(
    input_path: str | Path,
    repo_id: str,
    resave: bool = True,
    branch: str | None = None,
    private: bool = False,
    workdir: str | Path | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> dict:
    """Publish a zea dataset to the Hugging Face Hub, with a virtual reference.

    Resaves the files with the current codec/chunking defaults (unless ``resave`` is
    False), uploads them, and publishes a virtual reference pinned to the resulting
    commit — so that ``Dataset(..., lazy="virtual")`` works on the published repo.

    Writes to a remote repository: it creates ``repo_id`` if it does not exist, and
    adds two commits to it (the data, then the reference).

    Args:
        input_path (str or Path): The dataset to publish: a local folder/file, or an
            ``hf://`` path (downloaded first — this is the migration case).
        repo_id (str): Target Hugging Face dataset repo, e.g. ``"zeahub/my-dataset"``.
            May be the same repo as ``input_path``, to migrate it in place.
        resave (bool, optional): Resave every file with the current defaults (Blosc +
            one chunk per frame) before uploading. Defaults to ``True``. Set to ``False``
            only when the files are known to already use them — the virtual reference
            cannot be built otherwise.
        branch (str, optional): Branch to commit to. Defaults to the repo's default
            branch. Use one (e.g. ``"virtual"``) to stage a migration before merging.
        private (bool, optional): Create the repo private, if it does not exist yet.
            Defaults to ``False``.
        workdir (str or Path, optional): Where the resaved files are written. Defaults
            to a temporary directory (removed afterwards). Pass one to keep them.
        token (str, optional): Hugging Face token. Defaults to the ambient credentials
            (``HF_TOKEN`` or a cached login).
        commit_message (str, optional): Message of the data commit.

    Returns:
        dict: ``{"repo_id", "data_commit", "virtual_commit", "n_files"}`` — the two
        commit hashes, with ``data_commit`` the revision the reference is pinned to.
    """
    from huggingface_hub import create_repo, upload_folder

    input_path = str(input_path)
    if input_path.startswith(HF_PREFIX):
        log.info(f"Downloading {log.yellow(input_path)} (needed to resave and republish)")
        source = Path(_hf_resolve_path(input_path))
    else:
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"No dataset at '{source}'.")

    _hf_login()

    with tempfile.TemporaryDirectory(prefix="zea_publish_") as scratch:
        upload_dir = Path(workdir) if workdir is not None else Path(scratch) / "data"

        if resave:
            from zea.data.file_operations import resave as resave_files

            log.info(f"Resaving {log.yellow(str(source))} → {log.yellow(str(upload_dir))}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            # ``resave`` mirrors a folder, but maps a single file to a single file.
            destination = upload_dir / source.name if source.is_file() else upload_dir
            resave_files(source, destination, overwrite=True)
        else:
            upload_dir = source if source.is_dir() else source.parent

        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

        log.info(f"Uploading data to {log.yellow(repo_id)}")
        data_commit = upload_folder(
            repo_id=repo_id,
            folder_path=str(upload_dir),
            repo_type="dataset",
            revision=branch,
            token=token,
            commit_message=commit_message or "Publish zea dataset (Blosc, per-frame chunks)",
            ignore_patterns=[f"{VIRTUAL_DIR}/*"],
        )

        # Virtualize what was actually published, pinned to the commit that published it:
        # the reference then cannot drift from its data, and building it against the Hub
        # verifies the uploaded files are readable at their final URLs.
        virtual_dir = Path(scratch) / VIRTUAL_DIR
        build_virtual_reference(
            f"{HF_PREFIX}{repo_id}",
            virtual_dir / "index.json",
            revision=data_commit.oid,
        )

        log.info(f"Uploading virtual reference to {log.yellow(f'{repo_id}/{VIRTUAL_DIR}')}")
        virtual_commit = upload_folder(
            repo_id=repo_id,
            folder_path=str(virtual_dir),
            path_in_repo=VIRTUAL_DIR,
            repo_type="dataset",
            revision=branch,
            token=token,
            commit_message=f"Virtual reference for {data_commit.oid[:7]}",
        )

        n_files = len(list(upload_dir.rglob("*.hdf5"))) + len(list(upload_dir.rglob("*.h5")))

    log.info(
        f"Published {n_files} file(s) to {log.green(repo_id)}. Read them with: "
        f'zea.Dataset("{HF_PREFIX}{repo_id}", lazy="virtual")'
    )
    return {
        "repo_id": repo_id,
        "data_commit": data_commit.oid,
        "virtual_commit": virtual_commit.oid,
        "n_files": n_files,
    }
