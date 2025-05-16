"""Configuration file for the Sphinx documentation builder."""

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
    "sphinx_copybutton",  # for copy button in code blocks
    "sphinx.ext.viewcode",  # for links to source code
    "sphinx.ext.autosummary",  # for generating API documentation
    "sphinx.ext.intersphinx",  # for cross-project links
    "myst_parser",  # for markdown support
]

# autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "light_logo": "usbmd_logo_v3.svg",
    "dark_logo": "usbmd_logo_v3.svg",
}
html_static_path = ["../_static"]
# standard docs favicon
# html_favicon = "../_static/usbmd_logo_v3.svg"

# for index
modindex_common_prefix = ["usbmd."]
