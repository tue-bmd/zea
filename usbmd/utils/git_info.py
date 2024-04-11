"""Git utilities.

Get git commit hash and branch name.

Author(s)     : Tristan Stevens
Data          : January 27th, 2023
"""

import subprocess
import sys

from usbmd.utils import log


def get_git_commit_hash():
    """Gets git commit hash of current branch."""
    return str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip(), "utf-8")


def get_git_branch():
    """Get current branch name."""
    return str(
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "utf-8",
    )


def get_git_summary(verbose=False):
    """Get summary of git info.
    Args:
        verbose (bool, optional): print git summary. Defaults to False.
    Returns:
        str: git summary string.
            contains branch name and commit hash.
    """
    try:
        git_summary = get_git_branch() + "=" + get_git_commit_hash()
        if verbose:
            log.info(f"Git branch and commit: {git_summary}")
        return git_summary
    except Exception:
        log.warning("Cannot find Git")


if __name__ == "__main__":
    get_git_summary()
    sys.stdout.flush()
