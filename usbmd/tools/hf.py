"""Huggingface hub (hf) tooling."""

from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download

from usbmd import log


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
                f"Succesfully loaded model {commit_message} from "
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
        msg = (
            f"Uploaded files from '{local_dir}' to 'https://huggingface.co/{repo_id}'."
        )
        if tag:
            msg += f" Tagged as {tag}."
        log.info(log.yellow(msg))

    return f"https://huggingface.co/{repo_id}"
