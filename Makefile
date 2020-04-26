#!/usr/bin/env make -f

PYTHON=python3
DOWNLOAD=wget
ERFA_URL=https://github.com/liberfa/erfa/releases/download/v1.7.0/erfa-1.7.0.tar.gz


.PHONY: default build sdist wheel check clean distclean


default: build


build: cextern/erfa
	$(PYTHON) setup.py build_ext --inplace


sdist: cextern/erfa
	$(PYTHON) setup.py sdist


wheel: sdist
	$(PYTHON) -m pip wheel dist/pyerfa-*.tar.gz


check: build
	$(PYTHON) -m pytest


clean:
	$(RM) erfa/*.so
	$(RM) erfa/core.py erfa/ufunc.c  # generated
	$(RM) -r *.egg-info dist build
	find . -name __pycache__ -type d -exec rm -rf '{}' +


distclean: clean
	$(RM) -r cextern
	$(RM) $(shell basename $(ERFA_URL))
	$(RM) *.whl
	$(RM) -r .pytest_cache


cextern/erfa: cextern/$(shell basename $(ERFA_URL) .tar.gz)
	ln -fs $(shell basename $(ERFA_URL) .tar.gz)/src cextern/erfa


cextern/$(shell basename $(ERFA_URL) .tar.gz): $(shell basename $(ERFA_URL))
	mkdir -p cextern
	tar xvf $(shell basename $(ERFA_URL)) -C cextern


$(shell basename $(ERFA_URL)):
	$(DOWNLOAD) $(ERFA_URL)