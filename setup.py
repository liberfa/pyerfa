# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import setuptools


ERFAPKGDIR = os.path.abspath(os.path.dirname(__file__))

ERFA_SRC = os.path.abspath(os.path.join(ERFAPKGDIR, 'liberfa', 'erfa', 'src'))


# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(setuptools.Extension):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        self.cython_directives = kwargs.pop('cython_directives', {})
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs


def get_extensions():
    if not os.path.exists(os.path.join('erfa', 'ufunc.c')):
        import sys
        import subprocess
        cmd = [
            sys.executable, os.path.join(ERFAPKGDIR, 'erfa_generator.py'),
            ERFA_SRC, '--quiet',
        ]
        subprocess.run(cmd, check=True)

    sources = [os.path.join(ERFAPKGDIR, 'erfa', fn)
               for fn in ("ufunc.c", "pav2pv.c", "pv2pav.c")]

    include_dirs = []

    libraries = []

    if (int(os.environ.get('PYERFA_USE_SYSTEM_ERFA', 0)) or
            int(os.environ.get('PYERFA_USE_SYSTEM_ALL', 0))):
        libraries.append('erfa')
    else:
        # get all of the .c files in the liberfa/erfa/src directory
        erfafns = os.listdir(ERFA_SRC)
        sources.extend([os.path.join(ERFA_SRC, fn)
                        for fn in erfafns
                        if fn.endswith('.c') and not fn.startswith('t_')])

        include_dirs.append(ERFA_SRC)

    erfa_ext = NumpyExtension(
        name="erfa.ufunc",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",)

    return [erfa_ext]


setuptools.setup(ext_modules=get_extensions())
