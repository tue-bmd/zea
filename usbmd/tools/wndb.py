"""Weights & Biases (wandb) tooling."""

from pathlib import Path

import wandb


def model_directory_from_wandb(workspace, name):
    """Get model directory from wandb name."""

    api = wandb.Api()
    runs = api.runs(workspace, filters={"display_name": name})
    if len(runs) == 0:
        raise ValueError(f"No runs found with name {name} in workspace {workspace}")
    if len(runs) > 1:
        raise ValueError(
            f"Multiple runs found with name {name} in workspace {workspace}"
        )
    run = runs[0]

    if run.config.get("run_dir") is None:
        raise ValueError(f"Run {name} does not have a 'run_dir' in its config")

    return Path(run.config["run_dir"])
