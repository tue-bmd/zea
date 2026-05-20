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

# Prevent autosummary from overwriting existing stub files.
# The stubs in _autosummary/ are committed and maintained manually or via an
# initial generation. Setting this to False means only NEW items (that have no
# stub yet) get auto-generated; existing stubs (including any manual edits like
# :exclude-members:, :canonical:, or custom class templates) are preserved.
autosummary_generate_overwrite = False

# Cross-project link targets (numpy, scipy, Python stdlib, matplotlib).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Warn about all unresolved cross-references (catches typos like
# :class:`zea.models.echonet.EchoNet` where EchoNet doesn't exist).
# Add entries to nitpick_ignore_regex for intentional exceptions.
nitpicky = True

# Suppress "more than one target found" warnings: these arise because classes
# like `Probe` are deliberately accessible at both `zea.Probe` and
# `zea.probes.Probe`. Using the fully-qualified form in cross-references
# (e.g. :class:`zea.Probe`) avoids ambiguity; short-form uses in existing
# docstrings are acceptable.
# Also suppress toc.excluded: auto-generated stubs may include toctree entries
# for modules that are intentionally excluded from the rendered docs.
suppress_warnings = ["ref.python", "toc.excluded"]

# Suppress warnings for type annotations that are informal, backend-agnostic,
# or from internal / third-party modules not covered by intersphinx.
nitpick_ignore_regex = [
    # Informal / Google-style type annotations
    (r"py:.*", r"optional.*"),
    (r"py:.*", r"callable"),
    (r"py:.*", r"array.like"),
    # Backend-agnostic tensor / array types used across docstrings
    (r"py:class", r"Tensor"),
    (r"py:class", r"ops\.Tensor"),
    (r"py:class", r"tensor"),
    (r"py:class", r"ndarray"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"complex ndarray"),
    # Non-fully-qualified type shorthands
    (r"py:class", r"Path"),
    (r"py:class", r"SeedGenerator"),
    (r"py:class", r"ops\..*"),
    # Variable names used as types in Google-style return docstrings.
    # Lowercase snake_case identifiers are almost certainly variable names,
    # not actual class names, so suppress them all with one broad pattern.
    (r"py:class", r"[a-z_][a-z_0-9]*"),
    # Multi-word informal type descriptions
    (r"py:class", r"A tuple containing.*"),
    (r"py:class", r"dict with keys.*"),
    (r"py:class", r"str/int/list"),
    (r"py:class", r"int/list"),
    (r"py:class", r"2d array"),
    (r"py:class", r"The split name.*"),
    (r"py:class", r"The lens correction.*"),
    (r"py:class", r"Tuple of.*"),
    (r"py:class", r"Posterior samples.*"),
    (r"py:class", r"Dictionary containing.*"),
    (r"py:class", r"Defaults to.*"),
    (r"py:class", r"which is the.*"),
    (r"py:class", r"darkmode style.*"),
    # Informal names starting with - (e.g. -elbo, -)
    (r"py:class", r"-.*"),
    # Internal enum not part of the public API
    (r"py:class", r"DataTypes"),
    # Base classes from excluded internal modules
    (r"py:class", r"zea\.ops\.base\..*"),
    # Internal types not publicly documented
    (r"py:class", r"zea\.internal\..*"),
    # zea.Operation (ops re-export not at top level)
    (r"py:class", r"zea\.Operation"),
    # Third-party types without intersphinx
    (r"py:class", r"h5py\..*"),
    (r"py:class", r"grain\..*"),
    (r"py:class", r"yaml\..*"),
    (r"py:class", r"PIL\..*"),
    (r"py:class", r"jax\..*"),
    (r"py:class", r"keras\.src\..*"),
    (r"py:class", r"keras\.Model"),
    # torch types (torch not in intersphinx)
    (r"py:class", r"torch\..*"),
    # multiprocessing types not reliably in python intersphinx
    (r"py:class", r"multiprocessing\..*"),
    # keras module references (no intersphinx for keras)
    (r"py:mod", r"keras\..*"),
    # matplotlib sub-types not in main intersphinx (e.g. mpl_toolkits)
    (r"py:class", r"matplotlib\.axes\.Axes3DSubplot"),
    (r"py:class", r"plt\..*"),
    # Typing module references (fallback if intersphinx unavailable offline)
    (r"py:data", r"typing\..*"),
]
