"""Tests for docs/source/notebook_clean_and_check.py's check_doc_links.

Hardcoded zea.readthedocs.io links can appear in a notebook markdown cell in
several Markdown forms (inline link, reference-style link, autolink, bare
URL); this guards that check_doc_links flags all of them, not just the inline
form.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "source" / "notebook_clean_and_check.py"
)
_spec = importlib.util.spec_from_file_location("notebook_clean_and_check", _MODULE_PATH)
notebook_clean_and_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = notebook_clean_and_check
_spec.loader.exec_module(notebook_clean_and_check)

check_doc_links = notebook_clean_and_check.check_doc_links


def _nb_with_markdown(source: str) -> dict:
    return {"cells": [{"cell_type": "markdown", "source": [source]}]}


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "See the [pipeline docs](https://zea.readthedocs.io/en/latest/pipeline.html).",
            id="inline-link-https",
        ),
        pytest.param(
            "See the [pipeline docs](http://zea.readthedocs.io/en/latest/pipeline.html).",
            id="inline-link-http",
        ),
        pytest.param(
            "See the [pipeline docs][1].\n\n[1]: https://zea.readthedocs.io/en/latest/pipeline.html",
            id="reference-style-link",
        ),
        pytest.param(
            "See <https://zea.readthedocs.io/en/latest/pipeline.html> for details.",
            id="autolink",
        ),
        pytest.param(
            "See https://zea.readthedocs.io/en/latest/pipeline.html for details.",
            id="bare-url",
        ),
    ],
)
def test_check_doc_links_rejects_hardcoded_readthedocs_links(source):
    nb = _nb_with_markdown(source)
    with pytest.raises(SystemExit):
        check_doc_links(nb, "dummy.ipynb")


def test_check_doc_links_allows_relative_links():
    nb = _nb_with_markdown("See the [pipeline docs](../../pipeline.rst) for details.")
    check_doc_links(nb, "dummy.ipynb")  # should not raise


def test_check_doc_links_ignores_code_cells():
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["# https://zea.readthedocs.io/en/latest/pipeline.html\n"],
            }
        ]
    }
    check_doc_links(nb, "dummy.ipynb")  # should not raise


# -- Examples page: tags and thumbnail ---------------------------------------

check_tags = notebook_clean_and_check.check_tags
check_thumbnail = notebook_clean_and_check.check_thumbnail


def _nb_with_tags(tags) -> dict:
    return {"cells": [], "metadata": {"zea": {"tags": tags}}}


def test_check_tags_accepts_lowercase_tags():
    check_tags(_nb_with_tags(["beamforming", "cardiac"]), "dummy.ipynb")


@pytest.mark.parametrize(
    "nb",
    [
        pytest.param({"cells": [], "metadata": {}}, id="missing"),
        pytest.param(_nb_with_tags([]), id="empty"),
        pytest.param(_nb_with_tags("beamforming"), id="not-a-list"),
        pytest.param(_nb_with_tags(["beamforming", ""]), id="empty-string"),
        pytest.param(_nb_with_tags(["Beamforming"]), id="uppercase"),
        pytest.param(_nb_with_tags([" cardiac"]), id="whitespace"),
        pytest.param(_nb_with_tags(["cardiac", "cardiac"]), id="duplicate"),
    ],
)
def test_check_tags_rejects_bad_tags(nb):
    with pytest.raises(SystemExit):
        check_tags(nb, "dummy.ipynb")


def test_check_thumbnail(tmp_path):
    with pytest.raises(SystemExit):
        check_thumbnail("notebooks/models/new_example.ipynb", thumbnail_dir=tmp_path)
    (tmp_path / "new_example.webp").write_bytes(b"")
    check_thumbnail("notebooks/models/new_example.ipynb", thumbnail_dir=tmp_path)


def test_every_notebook_has_tags_and_thumbnail():
    """Every example notebook in the repo is ready for the examples page."""
    notebooks = sorted((_MODULE_PATH.parent / "notebooks").rglob("*.ipynb"))
    assert notebooks
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        check_tags(nb, nb_path)
        check_thumbnail(nb_path)
