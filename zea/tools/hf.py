"""Huggingface hub (hf) tooling."""

from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, login, snapshot_download

from zea import log
from zea.data.preset_utils import _hf_list_files, _hf_parse_path

HF_PREFIX = "hf://"


def load_model_from_hf(repo_id, revision="main", verbose=True):
    """
    Load the model from a given repo_id using the Hugging Face library.

    Will download to a `model_dir` directory and return the path to it.
    Need your own load model logic to load the model from the `model_dir`.

    Args:
        repo_id (str): The ID of the repository.
        revision (str): The revision to download. Can be a branch, tag, or commit hash.
        verbose (bool): Whether to print the download message. Default is True.

    Returns:
        model_dir (Path): The path to the downloaded model directory.

    """
    login(new_session=False)

    model_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    api = HfApi()
    commit = api.list_repo_commits(repo_id, revision=revision)[0]
    commit_message = commit.title
    commit_time = commit.created_at.strftime("%B %d, %Y at %I:%M %p %Z")

    if verbose:
        log.info(
            log.yellow(
                f"Successfully loaded model {commit_message} from "
                f"'https://huggingface.co/{repo_id}'. Last updated on {commit_time}."
            )
        )

    return Path(model_dir)


def upload_folder_to_hf(
    local_dir,
    repo_id,
    commit_message=None,
    revision="main",
    tag=None,
    verbose=True,
):
    """
    Upload a local directory to Hugging Face Hub.

    Args:
        local_dir (str or Path): Path to the local directory to upload.
        repo_id (str): The ID of the repository to upload to.
        commit_message (str, optional): Commit message. Defaults to "Upload files".
        revision (str): The revision to upload to. Defaults to "main".
        tag (str, optional): Tag to create. Defaults to None.
        verbose (bool): Whether to print the upload message. Default is True.

    Returns:
        str: URL of the uploaded repository.
    """
    login(new_session=False)
    api = HfApi()

    local_dir = Path(local_dir)
    if not commit_message:
        commit_message = f"Upload files from {local_dir.name}"

    # create branch if it doesn't exist
    api.create_branch(repo_id, repo_type="model", branch=revision, exist_ok=True)

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        revision=revision,
    )

    if tag:
        api.create_tag(repo_id, repo_type="model", tag=tag)

    if verbose:
        msg = f"Uploaded files from '{local_dir}' to 'https://huggingface.co/{repo_id}'."
        if tag:
            msg += f" Tagged as {tag}."
        log.info(log.yellow(msg))

    return f"https://huggingface.co/{repo_id}"


class HFPath(PurePosixPath):
    """A path-like object that preserves the hf:// scheme and mimics Path API."""

    _scheme = HF_PREFIX

    def __new__(cls, *args):
        # Strip "hf://" from all arguments and normalize
        parts = []
        for arg in args:
            s = str(arg)
            if s.startswith(cls._scheme):
                s = s[len(cls._scheme) :]
            parts.append(s.strip("/"))
        combined = "/".join(parts)
        # Store path without scheme
        self = super().__new__(cls, combined)
        # Mark this as an HF path that needs a scheme when stringified
        self._needs_scheme = True
        return self

    def __str__(self):
        # Get the raw path string without any scheme
        path_str = PurePosixPath.__str__(self)

        # Remove any hf:/ prefix if it somehow got included
        if path_str.startswith("hf:/"):
            path_str = path_str[len("hf:/") :]

        # Add our scheme prefix if this is meant to be an HF path
        if getattr(self, "_needs_scheme", True):
            return f"{self._scheme}{path_str}"
        return path_str

    def __truediv__(self, key):
        return self.__class__(self, key)

    def joinpath(self, *args):
        """Join paths like Path.joinpath but preserve the hf:// scheme."""
        return self.__class__(self, *args)

    @property
    def repo_id(self):
        """Extract the repo ID (e.g., zeahub/camus-sample)."""
        parts = [p for p in self.parts if p and p != "hf:"]
        if len(parts) < 2:
            raise ValueError("Invalid HFPath: cannot extract repo_id")
        return f"{parts[0]}/{parts[1]}"

    @property
    def subpath(self):
        """Get path inside the repo."""
        return "/".join(self.parts[3:])

    def is_file(self):
        """Return True if this HFPath points to a file in the repo."""
        repo_id, subpath = _hf_parse_path(str(self))
        if not subpath:
            return False
        files = _hf_list_files(repo_id)
        return any(f == subpath for f in files)

    def is_dir(self):
        """Return True if this HFPath points to a directory in the repo."""
        repo_id, subpath = _hf_parse_path(str(self))
        files = _hf_list_files(repo_id)
        # If subpath is empty, it's the repo root, which is a directory
        if not subpath:
            return True
        # If any file starts with subpath + '/', it's a directory
        prefix = subpath.rstrip("/") + "/"
        return any(f.startswith(prefix) for f in files)
