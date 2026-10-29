"""Build the pixel grid explorer for the docs.

Embeds reference values computed by the installed zea into
``pixel_grid_explorer.html`` and writes a standalone page to
``docs/_static/pixel-grid-explorer.html``. Run from the repository root after changing
the grid geometry of :class:`zea.Parameters` or the page itself::

    python docs/explorer/build.py
"""

import json
import sys
from pathlib import Path

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
    sys.path.insert(0, str(HERE))
    from make_reference import reference

    OUTPUT.write_text(SKELETON.replace("{page}", render(reference())))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
