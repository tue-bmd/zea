"""Tests for docs/source/notebook_clean_and_check.py's check_doc_links.

Hardcoded zea.readthedocs.io links can appear in a notebook markdown cell in
several Markdown forms (inline link, reference-style link, autolink, bare
URL); this guards that check_doc_links flags all of them, not just the inline
form.
"""

import importlib.util
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
