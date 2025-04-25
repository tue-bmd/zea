import os
import sys

import usbmd

sys.path.insert(0, os.path.abspath("../usbmd"))

# -- Project information -----------------------------------------------------
project = "usbmd"
# get automatically the version from the package
release = usbmd.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/usbmd_logo_v3.svg"

# for index
modindex_common_prefix = ["usbmd."]
