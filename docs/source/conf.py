"""Configuration file for the Sphinx documentation builder."""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("_ext"))  # local Sphinx extensions

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
    "tyroprogram",  # local: auto-documents the tyro CLI (docs/source/_ext/tyroprogram.py)
]

autodoc_mock_imports = [
    "tensorflow",
    "torch",
    "zea.backend.tf2jax",
    "gradio",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Included verbatim by data-acquisition.rst; excluded as standalone
    # documents so their labels are not defined twice.
    "_spec_ref.rst",
    "_units_ref.rst",
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
    "_autosummary/zea.cli_args.rst",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "exclude-members": "SCHEMA,VALID_PARAMS",
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
html_css_files = ["custom.css"]
html_js_files = ["open-anchor-dropdown.js"]

# Strip >>> / ... prompts when copying doctest blocks; leave other blocks intact.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

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
    "zea.data.file.File": "zea.File",
    "zea.ops.pipeline.Pipeline": "zea.Pipeline",
    "zea.probes.Probe": "zea.Probe",
    "zea.parameters.Parameters": "zea.Parameters",
}

_REEXPORTED_CANONICALS = set(TOPLEVEL_API_ALIASES)

# Properties on zea.Parameters that are also fields on zea.data.spec.ScanSpec.
# Documenting both creates duplicate cross-reference targets and "more than one
# target found" warnings in Sphinx 7+.  The canonical documentation lives on
# ScanSpec; Parameters users are pointed there via the class docstring.
_PARAMETERS_SCANSPEC_ALIASES = frozenset({"focus_distances", "initial_times", "t0_delays"})

# VerasonicsProbe.type is a string label already documented by ProbeSpec.type.
# Excluding it from autodoc removes the duplicate cross-reference target.
_VERASONICS_PROBE_EXCLUDE = frozenset({"type"})


def _skip_reexported_members(app, what, name, obj, skip, options):
    """Hide re-exported classes and disambiguate duplicate attribute targets.

    1. Re-exported classes are documented under their top-level ``zea.X`` alias
       (see ``TOPLEVEL_API_ALIASES``), so the copy in the defining submodule is
       hidden to keep a single, unambiguous cross-reference target.

    2. ``zea.Parameters`` properties that duplicate ``ScanSpec`` fields are excluded
       so Sphinx does not register two targets for the same name.  These use
       ``@cache_with_dependencies`` which wraps the function in ``property()``, so
       ``obj`` is a ``property`` whose ``fget.__qualname__`` starts with
       ``"Parameters."``.  We use ``fget.__qualname__`` instead of ``name`` because
       Sphinx 9.x passes only the bare member name (e.g. ``"focus_distances"``),
       not the full dotted path that Sphinx 8.x used.

    3. ``VerasonicsProbe`` properties in ``_VERASONICS_PROBE_EXCLUDE`` are skipped
       so Sphinx does not register duplicate targets shared with ``ProbeSpec``/
       ``Subject``.  Same approach: check ``obj.fget.__qualname__``.

    Returns ``None`` (rather than ``skip``) for everything else so the default
    filtering still applies.
    """
    if what == "module":
        canonical = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"
        if canonical in _REEXPORTED_CANONICALS:
            return True

    # Resolve the bare attribute name from whatever form ``name`` takes.
    # Sphinx 8.x: full dotted path  (``zea.parameters.Parameters.focus_distances``)
    # Sphinx 9.x: bare member name  (``focus_distances``)
    attr_name = name.rsplit(".", 1)[-1]

    # For property objects, fget.__qualname__ encodes the owning class name and
    # works in both Sphinx versions without depending on how ``name`` is formatted.
    fget_qualname = getattr(getattr(obj, "fget", None), "__qualname__", "")

    # Skip Parameters properties that duplicate ScanSpec field names.
    if attr_name in _PARAMETERS_SCANSPEC_ALIASES:
        # Sphinx 9.x: use fget.__qualname__ (e.g. "Parameters.focus_distances")
        if fget_qualname.startswith("Parameters."):
            return True
        # Sphinx 8.x fallback: name contains the full class path
        if name.rsplit(".", 1)[0].endswith(".Parameters"):
            return True

    # Skip VerasonicsProbe properties that duplicate ProbeSpec/Subject targets.
    if attr_name in _VERASONICS_PROBE_EXCLUDE:
        # Sphinx 9.x: use fget.__qualname__ (e.g. "VerasonicsProbe.type")
        if "VerasonicsProbe" in fget_qualname:
            return True
        # Sphinx 8.x fallback: name contains the full class path
        if "VerasonicsProbe" in name:
            return True

    return None


doctest_global_setup = """
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
"""

# -- Options for linkcheck builder --------------------------------------------
# `make docs-linkcheck` also checks links in the example notebooks, which
# nbsphinx renders into the same doctree as the rest of the docs.
# Keep retries/timeout low; known-difficult links are ignored below instead.
linkcheck_retries = 1
linkcheck_timeout = 10
linkcheck_workers = 8
linkcheck_rate_limit_timeout = 30
linkcheck_ignore = [
    r"^https://colab\.research\.google\.com/",  # blocks non-browser requests
    r"^https://doi\.org/10\.1098/rsta\.2024\.0327$",  # royalsociety 403s bots
    r"^https://doi\.org/10\.1001/jamacardio\.2021\.6059$",  # jamanetwork 403s bots
    r"^https://doi\.org/10\.1016/j\.ultrasmedbio\.2024\.12\.008$",  # sciencedirect 403s bots
    r"^https://nl\.mathworks\.com/matlabcentral/",  # 403s bots
    r"^https://www\.creatis\.insa-lyon\.fr/",  # times out from CI
    r"^https://portal\.hdfgroup\.org/display/HDF5/H5P_SET_",  # 404, from an older h5py docstring
    # Self-repo file links (badges, README refs): correct by construction if
    # the build got this far, and hitting them over HTTP trips GitHub's
    # anonymous rate limit.
    r"^https://github\.com/tue-bmd/zea/blob/main/",
]
linkcheck_anchors_ignore_for_url = [
    r"^https://github\.com/[^/]+/[^/]+/blob/",  # line anchors are JS-rendered
    r"^https://humanheart-project\.creatis\.insa-lyon\.fr/",  # SPA, fragment is a client route
]


def setup(app):
    app.connect("autodoc-skip-member", _skip_reexported_members)
