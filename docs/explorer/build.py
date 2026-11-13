"""Build the pixel grid explorer for the docs.

Embeds reference values computed by the installed zea into
``pixel_grid_explorer.html`` and writes a standalone page to
``docs/_static/pixel-grid-explorer.html``. The docs build (``docs/source/conf.py``) runs
this on every HTML build, so the page is not committed. To preview the page without
building the docs, run from the repository root::

    python docs/explorer/build.py
"""

import json
import sys
from pathlib import Path

import zea

HERE = Path(__file__).parent
TEMPLATE = HERE / "pixel_grid_explorer.html"
OUTPUT = HERE.parent / "_static" / "pixel-grid-explorer.html"

SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>[hidden]{display:none!important}body{margin:0}</style>
</head>
<body>
{page}
</body>
</html>
"""


def render(reference):
    """The page with ``reference`` embedded, without a document skeleton."""
    payload = json.dumps(reference, separators=(",", ":"))
    if "</script" in payload:
        raise ValueError("Reference values must not contain '</script'.")
    return TEMPLATE.read_text().replace("__FIXTURES__", payload)


def main():
    """Write the page to ``OUTPUT``, leaving the file untouched when nothing changed."""
    sys.path.insert(0, str(HERE))
    from make_reference import reference

    # Some reference cases deliberately build grids zea warns about.
    with zea.log.suppress_warnings():
        page = SKELETON.replace("{page}", render(reference()))
    if OUTPUT.exists() and OUTPUT.read_text() == page:
        return
    OUTPUT.write_text(page)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
