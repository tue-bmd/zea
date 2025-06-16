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
make docs-build
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
make docs-serve
```

This uses `sphinx-autobuild` to serve the docs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

For more information, see the [Sphinx documentation](https://www.sphinx-doc.org/).
