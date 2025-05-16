# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
# The docs folder where source .rst files are located
SOURCEDIR     = source
# The folder where the generated documentation will be placed
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile docs-clean docs-build docs-serve

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom commands

# Clean generated rst files (except index.rst) and built html
# Find all .rst files except index.rst and delete them
# Remove any generated folder and build files
docs-clean:
	find $(SOURCEDIR) -type f -name 'usbmd.*.rst' ! -name 'index.rst' ! -path '*/\.*' -exec rm -f {} \;
	rm -rf $(SOURCEDIR)/generated
	rm -rf $(BUILDDIR)


# Regenerate .rst files using sphinx-apidoc
docs-apidoc:
	sphinx-apidoc -o $(SOURCEDIR) ../usbmd

# Build docs cleanly
docs-build: docs-clean docs-apidoc
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Serve docs locally
docs-serve: docs-build
	cd $(BUILDDIR)/html && python3 -m http.server 8000
