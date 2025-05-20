export LC_ALL=C.UTF-8
# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = _build

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
# We also copy the generated usbmd.rst file to the source directory
# as we have a custom template for it
docs-autosummary:
	sphinx-apidoc -o source/_autosummary ../usbmd --force --separate
	cp source/usbmd.rst source/_autosummary/usbmd.rst

# Build docs cleanly (parameters.rst and autosummary must be up to date)
docs-build: docs-parameters docs-autosummary
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Serve docs locally with sphinx-autobuild (auto-reloads on changes)
docs-serve: docs-parameters docs-autosummary
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html"

docs-test: docs-parameters
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)
