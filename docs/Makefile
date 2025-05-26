export LC_ALL=C.UTF-8
export KERAS_BACKEND=numpy
# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = _build
REMOVE_LIST   = ["usbmd.ops_dep.rst", "modules.rst"]

.PHONY: help Makefile docs-clean docs-build docs-serve docs-parameters

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate parameters.rst from PARAMETER_DESCRIPTIONS
docs-parameters:
	python $(SOURCEDIR)/parameters_doc.py

# Generate autosummary .rst files for API documentation
# force means that it will overwrite existing files
# separate means that it will create a separate file for each module
# we will first backup existing usbmd.rst as we don't want to overwrite it
docs-autosummary:
	@if [ -f $(SOURCEDIR)/_autosummary/usbmd.rst ]; then cp $(SOURCEDIR)/_autosummary/usbmd.rst $(SOURCEDIR)/_autosummary/usbmd.rst.bak; fi
	sphinx-apidoc -o $(SOURCEDIR)/_autosummary ../usbmd --force --separate
	@if [ -f $(SOURCEDIR)/_autosummary/usbmd.rst.bak ]; then mv $(SOURCEDIR)/_autosummary/usbmd.rst.bak $(SOURCEDIR)/_autosummary/usbmd.rst; fi
	@rm -f $(SOURCEDIR)/_autosummary/usbmd.ops_dep.rst $(SOURCEDIR)/_autosummary/modules.rst

# Build docs cleanly (parameters.rst and autosummary must be up to date)
docs-build: docs-parameters docs-autosummary
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Serve docs locally with sphinx-autobuild (auto-reloads on changes)
docs-serve: docs-parameters docs-autosummary
	@PORT=8000; \
	while lsof -i :$$PORT >/dev/null 2>&1; do PORT=$$((PORT+1)); done; \
	echo "Using port $$PORT"; \
	$(SPHINXAUTOBUILD) --port=$$PORT "$(SOURCEDIR)" "$(BUILDDIR)/html"

docs-test: docs-parameters
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)
