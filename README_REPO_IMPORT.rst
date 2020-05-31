PyERFA repository import from Astropy
=====================================

:copyright: 2020 Antonio Valentino

Python wrapper for ERFA_, Essential Routines for Fundamental Astronomy.

The project is a split of the ``astropy._erfa`` sub-module, developed in the
context of Astropy_ project, into a standalone package.

See also https://github.com/astropy/astropy/issues/9802.

.. _ERFA: https://github.com/liberfa/erfa
.. _Astropy: https://www.astropy.org


The code has been imported form the git revision
`60ab27c <https://github.com/astropy/astropy/commit/60ab27c6da71aa289c7bf0e69121856a03fddc30>`_.

The following import script has been used::

    #!/bin/sh

    set -e

    if [ ! -d astropy ]
    then
      git clone https://github.com/astropy/astropy.git
    fi

    if [ ! -d pyerfa ]
    then
      git init pyerfa
    fi

    if [ ! -d git-filter-repo ]
    then
      git clone https://github.com/newren/git-filter-repo.git
    fi

    export PATH=${PATH}:${PWD}/git-filter-repo

    # if [ ! -d astropy/.git/filter-repo ]
    # then
    #   cd astropy
    #   git-filter-repo --analyze
    #   cd -
    # fi

    git-filter-repo \
    --source astropy \
    --target pyerfa \
    --path astropy/erfa \
    --path astropy/_erfa \
    --path .gitignore \
    --path licenses/ERFA.rst \
    --path licenses/LICENSE.rst \
    --path licenses/README.rst \
    --path-glob 'LICENSE*' \
    --path-rename astropy/_erfa:erfa \
    --path-rename astropy/erfa:erfa \
    --tag-rename v:astropy-v \
    --message-callback '
      return message.replace(b"pull request #", b"pull request astropy/astropy#")
    '

The full list of re-used files is the following:

=============================== ========================
Astropy                         PyERFA
=============================== ========================
astropy/_erfa/*                 erfa/*
astropy/_erfa/setup_package.py  setup.py
astropy/_erfa/erfa_generator.py erfa_generator.py
astropy/tests/helpers.py        erfa/tests/helpers.py
licenses/ERFA.rst               licenses/ERFA.rst
licenses/README.rst             licenses/README.rst
MANIFEST.in                     MANIFEST.in
pyproject.toml                  pyproject.toml
setup.cfg                       setup.cfg
=============================== ========================

Please note that some of the files have been slightly modified (e.g. import
statements have been updated) to allow the code to work properly as a
standalone package totally independent from Astropy_.

The ``erfa.tests.helpers`` module is duplicated form the original Astropy_
project, but all the code not strictly necessary to the PyERFA
package has been removed.

In addition:

* the ``ErfaError`` and ``ErfaWarning`` classes have been copied form
  ``astropy/utils/exceptions.py`` into ``erfa/core.py.templ``.
* the ``classproperty`` has been copied form ``astropy/utils/decorators.py``
  into ``erfa/helpers.py``.
