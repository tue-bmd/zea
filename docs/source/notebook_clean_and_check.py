#!/usr/bin/env python3
"""
Pre-commit hook to clean and validate Jupyter notebooks.

This script enforces consistency and quality standards for example notebooks. It:

1. **Execution order**
   Ensures code cells have sequential execution counts (`1..N`), verifying the notebook was
   executed in order before committing.

2. **Badges**
   Requires a markdown cell containing both:
   - An **Open in Colab** badge, linking to this notebook’s path on Colab.
   - A **View on GitHub** badge, linking to this notebook’s path in the repo.
   Also checks badge URLs for validity and common typos.

3. **Noisy outputs**
   Removes known transient/warning outputs from stderr streams, e.g.:
   - "computation placer already registered"
   - "Unable to register cuDNN factory"
   - "Unable to register cuBLAS factory"

4. **First cell requirement**
   The very first code cell must contain a `pip install zea` command
   (e.g. `%%capture\n%pip install zea`).

If any check fails:
- A clear error is printed.
- The script exits nonzero.
- If noisy outputs are cleaned, the notebook is rewritten, and the user must re-commit it.
"""

import json
import os
import re
import sys
from pathlib import Path

BADGE_COLAB = re.compile(
    r"\[!\[Open In Colab\]\(https://colab\.research\.google\.com/assets/colab-badge\.svg\)\]\(([^)]+)\)"
)
BADGE_GITHUB = re.compile(
    r"\[!\[View on GitHub\]\(https://img\.shields\.io/badge/GitHub-View%20Source-blue\?logo=github\)\]\(([^)]+)\)"
)


def error(msg, nb_path=None):
    prefix = f"[NOTEBOOK ERROR] {nb_path}: " if nb_path else "[NOTEBOOK ERROR] "
    print(f"{prefix}{msg}", file=sys.stderr)
    sys.exit(1)


def check_cell_size(nb, nb_path, max_cell_bytes=400_000):
    """Check that no cell is excessively large (e.g., due to embedded images/gifs)."""
    for idx, cell in enumerate(nb.get("cells", [])):
        # Check the size of the cell's JSON representation
        cell_bytes = len(json.dumps(cell, ensure_ascii=False).encode("utf-8"))
        if cell_bytes > max_cell_bytes:
            error(
                f"Cell {idx + 1} is too large (>{max_cell_bytes} bytes, got {cell_bytes}). "
                "This may be due to an embedded image or gif. Please save large media files "
                "separately and reference them in markdown instead of embedding.",
                nb_path,
            )


def check_execution_counts(nb, nb_path):
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    exec_counts = [c.get("execution_count") for c in code_cells]
    expected = list(range(1, len(exec_counts) + 1))
    if exec_counts != expected:
        error(
            f"Code cells must be executed sequentially (1..{len(exec_counts)}). "
            f"Found: {exec_counts}. Run all cells in order before committing.",
            nb_path,
        )


def check_badges(nb, nb_path):
    # validate badge URLs and detect common typos
    badge_pattern = re.compile(r"\[!\[[^\]]+\]\([^)]+\)\]\(([^)]+)\)")
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        for match in badge_pattern.finditer(src):
            url = match.group(1)
            if not url.startswith(("http://", "https://")):
                error(f"Badge with invalid URL: {url}", nb_path)
            if "shiedd" in url:  # common misspelling of shields.io
                error(f"Badge URL likely has a typo: {url}", nb_path)

    # check for required Colab + GitHub badges with correct path
    try:
        repo_root = Path(__file__).resolve().parents[2]
        nb_rel = str(Path(nb_path).resolve().relative_to(repo_root)).replace(os.sep, "/")
    except Exception:
        nb_rel = str(nb_path).replace(os.sep, "/")

    found_badge = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        m_colab = BADGE_COLAB.search(src)
        m_github = BADGE_GITHUB.search(src)
        if m_colab and m_github:
            if nb_rel in m_colab.group(1) and nb_rel in m_github.group(1):
                found_badge = True
                break
    if not found_badge:
        error(
            "Missing markdown cell with Colab + GitHub badges "
            f"linking to this notebook ({nb_rel}).",
            nb_path,
        )


def clean_outputs(nb, nb_path):
    """Remove known noisy stderr outputs. Returns True if notebook was modified."""
    changed = False
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs, new_outputs = cell.get("outputs", []), []
        for out in outputs:
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                print(
                    f"[NOTEBOOK CLEAN] {nb_path}: Removed stderr output from cell {idx + 1}.",
                    file=sys.stderr,
                )
                changed = True
                continue
            new_outputs.append(out)
        if new_outputs != outputs:
            cell["outputs"] = new_outputs
    return changed


def check_first_cell(nb, nb_path):
    """Ensure the first code cell contains 'pip install zea'."""
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        error("Notebook has no code cells.", nb_path)
    first_code = "".join(code_cells[0].get("source", []))
    if "pip install zea" not in first_code:
        error(
            "First code cell must contain 'pip install zea' (e.g. '%%capture\\n%pip install zea').",
            nb_path,
        )


def process_notebook(nb_path):
    nb_path = Path(nb_path)
    if not nb_path.is_file():
        error(f"Notebook not found: {nb_path}", nb_path)

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    check_cell_size(nb, nb_path)
    check_first_cell(nb, nb_path)
    check_execution_counts(nb, nb_path)
    check_badges(nb, nb_path)

    if clean_outputs(nb, nb_path):
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        error(
            "Notebook contained noisy outputs and was cleaned. Please commit the cleaned version.",
            nb_path,
        )

    print(f"[NOTEBOOK OK] {nb_path} passed all checks.")


def main():
    if len(sys.argv) < 2:
        repo_root = Path(__file__).resolve().parents[2]
        nb_dir = repo_root / "docs" / "source" / "notebooks"
        if not nb_dir.exists():
            print("Could not find docs/source/notebooks directory.", file=sys.stderr)
            sys.exit(2)
        nb_paths = list(map(str, nb_dir.rglob("*.ipynb")))
        if not nb_paths:
            print(f"No notebooks found in {nb_dir}", file=sys.stderr)
            sys.exit(0)
    else:
        nb_paths = sys.argv[1:]

    failed = False
    for nb_path in nb_paths:
        try:
            process_notebook(nb_path)
        except SystemExit:
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
