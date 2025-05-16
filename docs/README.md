
# Building the Documentation

To build the documentation for the `usbmd` package, follow these steps:

## 1. Install dependencies

Assumed to have the Docker image `usbmd:latest` built and running.
Install additional dependencies for documentation:

```sh
pip install -r docs/docs-requirements.txt
```

Optionally, you need to install the following (necessary in clean Docker images):
```shell
apt-get install -y make
export LC_ALL=C.UTF-8
```

## 2. Build the HTML documentation (from the docs directory)

Use the Makefile targets to clean, build, and serve the documentation:

```sh
cd docs
make docs-build
```

This will:
- Clean old generated `.rst` files and build artifacts.
- Build the latest HTML documentation into the `docs/_build/html` directory.

## 3. View the documentation

To view the generated documentation, open the `index.html` in your browser:

```sh
docs/_build/html/index.html
```

Or, if you want a live preview:

## 4. (Optional) Live preview with auto-reload (from the project root)

If you'd like to preview changes live as you edit the docs, use the `docs-serve` target to spin up a local server:

```sh
make docs-serve
```

Then, open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the docs with live reload.

---

For more information, see the [Sphinx documentation](https://www.sphinx-doc.org/).
