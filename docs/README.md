# Building the Documentation

To build the documentation for the `zea` package, follow these steps:

## 1. Install dependencies
Install the `zea` package with documentation dependencies:

```sh
pip install -e .[docs]
```

We also need to install `pandoc` to build the documentation:

```sh
apt-get update && apt-get install pandoc
```

If you are using a clean Docker image, you may also need to set the locale to avoid issues with `make`:

```sh
apt-get install -y make
export LC_ALL=C.UTF-8
```

## 2. Build the HTML documentation

From the `docs` directory (`cd docs`), run:

```sh
make docs-clean && make docs-build
```

This will generate the HTML documentation in `docs/_build/html`.

## 3. View the documentation

Open the generated documentation in your browser:

```sh
docs/_build/html/index.html
```

## 4. Live preview with auto-reload

For a live preview that automatically reloads on changes, use:

```sh
make docs-clean && make docs-serve
```

This uses `sphinx-autobuild` to serve the docs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 5. Check for broken links

To check that all hyperlinks in the docs resolve (including links in the
example notebooks under `docs/source/notebooks`, which nbsphinx renders into
the same doctree as the rest of the docs), run:

```sh
make docs-linkcheck
```

When linking to another zea documentation page (an API reference or another
notebook) from a notebook, use a path **relative** to the notebook's
location, e.g. `[zea.Pipeline](../../_autosummary/zea.Pipeline.rst)` or
`[pipeline docs](../../pipeline.rst)`, instead of a hardcoded
`https://zea.readthedocs.io/...` URL. Relative links are checked for broken
targets by the doc build itself and keep working across doc versions, whereas
absolute links can silently rot; `notebook_clean_and_check.py` rejects
hardcoded `readthedocs.io` links for this reason.

---

For more information, see the [Sphinx documentation](https://www.sphinx-doc.org/).
