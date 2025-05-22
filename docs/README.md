# Building the Documentation

To build the documentation for the `usbmd` package, follow these steps:

## 1. Install dependencies

Assumed to have the Docker image `usbmd:latest` built and running.
Install additional dependencies for documentation:

```sh
pip install -r docs/docs-requirements.txt
```

If you are using a clean Docker image, you may also need:

```sh
apt-get install -y make
export LC_ALL=C.UTF-8
```

## 2. Build the HTML documentation

From the `docs` directory, run:

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
