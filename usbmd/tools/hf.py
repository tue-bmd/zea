"""Huggingface hub (hf) tooling."""

from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download
from zipp import Path

from usbmd.utils.log import yellow


def load_model_from_hf(repo_id, revision="main"):
    """
    Load the model from a given repo_id using the Hugging Face library.

    Will download to a `model_dir` directory and return the path to it.
    Need your own load model logic to load the model from the `model_dir`.

    Args:
        repo_id (str): The ID of the repository.
        revision (str): The revision to download. Can be a branch, tag, or commit hash.

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
    print(
        yellow(
            f"Succesfully loaded model {commit_message} from "
            f"'https://huggingface.co/{repo_id}'. Last updated on {commit_time}."
        )
    )

    return Path(model_dir)
