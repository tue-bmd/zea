# Building the Documentation

To build the documentation for the `usbmd` package, follow these steps:

1. **Install dependencies**
   Assumed to have docker image `usbmd:latest` built and running.
   Install additional dependencies for documentation:
   ```sh
   pip install -r docs/docs-requirements.txt
   ```

2. **Build the HTML documentation (from project root)**
   ```sh
   sphinx-build -b html -c docs docs docs/_build/html
   ```

3. **View the documentation**
   Open the generated HTML in your browser:
   ```
   docs/_build/html/index.html
   ```

4. **(Optional) Live preview with auto-reload (from project root)**
   If you want to preview changes live as you edit:
   ```sh
   sphinx-autobuild docs docs/_build/html
   ```
   Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.


---

For more information, see the [Sphinx documentation](https://www.sphinx-doc.org/).