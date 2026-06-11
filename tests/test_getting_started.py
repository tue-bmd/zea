"""Test the code examples in docs/source/getting-started.rst.

Not using doctest: these snippets require network access and a full ML
backend (HuggingFace data + pretrained weights), so they are marked
``heavy`` and must run on a self-hosted (localhost) runner.

Run with:

.. code-block:: bash

    pytest -s -m 'heavy' tests/test_getting_started.py
"""

import os
import re
from pathlib import Path

# Must be set before matplotlib.pyplot is imported anywhere in this process.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

RST_PATH = Path("docs/source/getting-started.rst")


def _extract_python_blocks(path):
    """Return all ``.. code-block:: python`` blocks from an RST file."""
    blocks = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"\s*\.\. code-block::\s+python", lines[i]):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            content_indent = None
            block = []
            while i < len(lines):
                if lines[i].strip() == "":
                    block.append("")
                    i += 1
                    continue
                indent = len(lines[i]) - len(lines[i].lstrip())
                if content_indent is None:
                    content_indent = indent
                if indent < content_indent:
                    break
                block.append(lines[i][content_indent:])
                i += 1
            while block and block[-1] == "":
                block.pop()
            blocks.append("\n".join(block))
        else:
            i += 1
    return blocks


@pytest.mark.heavy
def test_getting_started_snippets():
    """Execute the first two Python code blocks from getting-started.rst together."""
    blocks = _extract_python_blocks(RST_PATH)
    assert len(blocks) >= 2, "Expected at least two Python code blocks in getting-started.rst"
    combined = "\n\n".join(blocks[:2])
    assert combined.strip(), "Extracted Python snippets are empty"
    exec(combined, {})  # noqa: S102
