"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import zea

# -- Project information -----------------------------------------------------
project = "zea"
# get automatically the version from the package
release = zea.__version__

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
    "sphinx.ext.doctest",  # for testing code snippets in the documentation
    "nbsphinx",  # for Jupyter notebook support
    "sphinx_design",  # for fancy code block selection
    "sphinxcontrib.bibtex",  # for bibliography support
    "sphinx_reredirects",  # for redirecting empty toc entries
    "sphinxarg.ext",  # for argparse support
    "sphinx.ext.mathjax",  # for rendering math in the documentation
]

autodoc_mock_imports = ["zea.backend.tf2jax"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_autosummary/zea.backend.tf2jax.rst",
    # Exclude internal implementation modules from documentation
    "_autosummary/zea.func.tensor.rst",
    "_autosummary/zea.func.ultrasound.rst",
    "_autosummary/zea.ops.base.rst",
    "_autosummary/zea.ops.tensor.rst",
    "_autosummary/zea.ops.ultrasound.rst",
    "_autosummary/zea.ops.pipeline.rst",
    "_autosummary/zea.tracking.base.rst",
    "_autosummary/zea.tracking.segmentation.rst",
    "_autosummary/zea.tracking.lucas_kanade.rst",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__call__",
}
autoclass_content = "both"  # include both class docstring and __init__ docstring

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    # "announcement": """
    #     <a style=\"text-decoration: none; color: white;\"
    #        href=\"https://github.com/tue-bmd/zea">
    #        <img src=\"_static/zea-logo.svg\"/> An example of an announcement!
    #     </a>
    # """,
    "sidebar_hide_name": True,
    "light_logo": "zea-logo.svg",
    "dark_logo": "zea-logo.svg",
}
html_static_path = ["../_static"]

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None
html_favicon = "../_static/zea-logo-fav-32px.png"

# for index
modindex_common_prefix = ["zea."]

# for bibtex
bibtex_bibfiles = ["../../paper/paper.bib"]

# for redirecting empty toc items to their parent
redirects = {
    f"notebooks/{page}.html": f"../examples.html#{page}"
    for page in ["data", "pipeline", "models", "metrics", "agent"]
}

# this will make sure that when an __all__ is defined in a module, the members
# listed in __all__ are the only ones included in the autosummary documentation
autosummary_ignore_module_all = False
