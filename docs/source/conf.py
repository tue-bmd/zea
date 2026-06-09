"""Configuration file for the Sphinx documentation builder."""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "zea"
release = str(get_version("zea"))

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
    "sphinxcontrib.autoprogram",  # for argparse support
    "sphinx.ext.mathjax",  # for rendering math in the documentation
]

autodoc_mock_imports = [
    "tensorflow",
    "torch",
    "zea.backend.tf2jax",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Included verbatim by data-acquisition.rst; excluded as a standalone
    # document so its labels (e.g. ``group-reference``) are not defined twice.
    "_spec_ref.rst",
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
    "_autosummary/zea.models.hvae.model.rst",
    "_autosummary/zea.models.hvae.utils.rst",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "exclude-members": "SCHEMA",
    "show-inheritance": True,
    "special-members": "__call__",
}
autoclass_content = "both"  # include both class docstring and __init__ docstring

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "announcement": (
        "<style>"
        "@media (max-width: 600px) {"
        "  .openh-rf-detail { display: none; }"
        "  .openh-rf-link { font-size: 0.85em; }"
        "}"
        "</style>"
        '<a class="openh-rf-link" style="text-decoration: none; color: inherit;" '
        'href="https://github.com/open-h/OpenH-RF" target="_blank">'
        "🧩 &nbsp; <code>zea</code> &nbsp; supports the OpenH-RF Initiative"
        '<span class="openh-rf-detail"> &mdash; a large-scale openly licensed ultrasound'
        " dataset by Stanford, TU/e &amp; NVIDIA</span>"
        ". Learn more &rarr;</a>"
    ),
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

# -- Top-level public API aliases --------------------------------------------
# A handful of classes are defined in submodules but re-exported at the package
# top level (``zea.Config`` etc.) as the encouraged import path. We document
# them *only* under their ``zea.X`` alias and hide the copy in the defining
# submodule, so each object has a single, unambiguous cross-reference target.
# autodoc still registers the canonical (submodule) name as an alias, so
# existing references like ``:class:`~zea.config.Config``` keep resolving.
#
# Maps the canonical ``module.Qualname`` to the public alias used in the docs.
TOPLEVEL_API_ALIASES = {
    "zea.config.Config": "zea.Config",
    "zea.data.dataloader.Dataloader": "zea.Dataloader",
    "zea.data.datasets.Dataset": "zea.Dataset",
    "zea.data.datasets.Folder": "zea.Folder",
    "zea.data.file.File": "zea.File",
    "zea.ops.pipeline.Pipeline": "zea.Pipeline",
    "zea.probes.Probe": "zea.Probe",
    "zea.scan.Parameters": "zea.Parameters",
}

_REEXPORTED_CANONICALS = set(TOPLEVEL_API_ALIASES)


def _skip_reexported_members(app, what, name, obj, skip, options):
    """Hide re-exported classes in their defining submodule.

    These classes are documented under their top-level ``zea.X`` alias instead
    (see ``TOPLEVEL_API_ALIASES``), which keeps a single documentation target
    per object and avoids "more than one target found" warnings.

    Returns ``None`` (rather than ``skip``) for everything else so the default
    filtering still applies. autosummary fires this event with ``skip=False``
    for every member, so echoing ``skip`` back would force private members into
    the generated summary tables.
    """
    if what == "module":
        canonical = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"
        if canonical in _REEXPORTED_CANONICALS:
            return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_reexported_members)
