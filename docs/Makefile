export LC_ALL=C.UTF-8
export KERAS_BACKEND=numpy
# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = source
BUILDDIR      = _build
REMOVE_LIST   = ["modules.rst"]

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
# we will first backup existing manually maintained stubs before overwriting them
docs-autosummary:
	@if [ -f $(SOURCEDIR)/_autosummary/zea.rst ]; then cp $(SOURCEDIR)/_autosummary/zea.rst $(SOURCEDIR)/_autosummary/zea.rst.bak; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.ops.rst ]; then cp $(SOURCEDIR)/_autosummary/zea.ops.rst $(SOURCEDIR)/_autosummary/zea.ops.rst.bak; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.models.rst ]; then cp $(SOURCEDIR)/_autosummary/zea.models.rst $(SOURCEDIR)/_autosummary/zea.models.rst.bak; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.models.hvae.rst ]; then cp $(SOURCEDIR)/_autosummary/zea.models.hvae.rst $(SOURCEDIR)/_autosummary/zea.models.hvae.rst.bak; fi
	sphinx-apidoc -o $(SOURCEDIR)/_autosummary ../zea --force --separate
	@find $(SOURCEDIR)/_autosummary -name "*.rst" -exec sed -i 's/:maxdepth: 4/:maxdepth: 2/g' {} \;
	@if [ -f $(SOURCEDIR)/_autosummary/zea.rst.bak ]; then mv $(SOURCEDIR)/_autosummary/zea.rst.bak $(SOURCEDIR)/_autosummary/zea.rst; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.ops.rst.bak ]; then mv $(SOURCEDIR)/_autosummary/zea.ops.rst.bak $(SOURCEDIR)/_autosummary/zea.ops.rst; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.models.rst.bak ]; then mv $(SOURCEDIR)/_autosummary/zea.models.rst.bak $(SOURCEDIR)/_autosummary/zea.models.rst; fi
	@if [ -f $(SOURCEDIR)/_autosummary/zea.models.hvae.rst.bak ]; then mv $(SOURCEDIR)/_autosummary/zea.models.hvae.rst.bak $(SOURCEDIR)/_autosummary/zea.models.hvae.rst; fi
	@rm -f $(SOURCEDIR)/_autosummary/modules.rst

# Build docs cleanly (parameters.rst and autosummary must be up to date)
docs-build: docs-parameters docs-autosummary
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Serve docs locally with sphinx-autobuild (auto-reloads on changes)
docs-serve: docs-parameters docs-autosummary
	@PORT=8000; \
	while lsof -i :$$PORT >/dev/null 2>&1; do PORT=$$((PORT+1)); done; \
	echo "Using port $$PORT"; \
	$(SPHINXAUTOBUILD) --port=$$PORT "$(SOURCEDIR)" "$(BUILDDIR)/html"

docs-test: docs-parameters docs-autosummary
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)

docs-clean:
	rm -rf "$(BUILDDIR)"
	echo "Cleaned build directory: $(BUILDDIR)"
	find "$(SOURCEDIR)/_autosummary" -type f ! -name 'zea.rst' ! -name 'zea.ops.rst' ! -name 'zea.models.rst' ! -name 'zea.models.hvae.rst' -name '*.rst' -delete
	echo "Cleaned autosummary files in: $(SOURCEDIR)/_autosummary (except manually maintained stubs)"
