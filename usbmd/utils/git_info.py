"""
Git utilities
"""

import subprocess
import sys


def get_git_commit_hash():
    """Gets git commit hash of current branch.
    """
    return str(
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip(), 'utf-8'
    )

def get_git_branch():
    """Get current branch name"""
    return str(
        subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).strip(), 'utf-8'
    )

if __name__ == "__main__":
    print(get_git_branch(), get_git_commit_hash())
    sys.stdout.flush()
