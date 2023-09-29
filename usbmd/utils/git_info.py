"""Git utilities

Author(s)     : Tristan Stevens, Iris Huijben
Data          : -
"""

import subprocess
import sys


def get_git_commit_hash():
    """Gets git commit hash of current branch."""
    return str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip(), "utf-8")


def get_git_branch():
    """Get current branch name."""
    return str(
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "utf-8",
    )


def get_git_summary():
    """Get summary of git info."""
    try:
        print("Git branch and commit: ")
        git_summary = get_git_branch() + "=" + get_git_commit_hash()
        print(git_summary)
    except Exception:
        print("Cannot find Git")
    return git_summary


if __name__ == "__main__":
    get_git_summary()
    sys.stdout.flush()
