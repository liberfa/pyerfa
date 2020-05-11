# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import setuptools
import subprocess


ERFA_SRC = os.path.join('liberfa', 'erfa', 'src')


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
    cmd = [sys.executable, 'erfa_generator.py', ERFA_SRC, '--quiet']
    subprocess.run(cmd, check=True)

    sources = [os.path.join('erfa', fn)
               for fn in ("ufunc.c", "pav2pv.c", "pv2pav.c")]

    include_dirs = []

    libraries = []

    if int(os.environ.get('PYERFA_USE_SYSTEM_LIBERFA', 0)):
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
        language="c")

    return [erfa_ext]


VERSION_TEMPLATE = """
'''Wrapper, ERFA and SOFA version information.'''

# Set the version numbers a bit indirectly, so that Sphinx can pick up
# up the docstrings and list the values.
from . import ufunc


erfa_version = ufunc.erfa_version
'''Version of the C ERFA library that is wrapped.'''

sofa_version = ufunc.sofa_version
'''Version of the SOFA library the ERFA library is based on.'''

del ufunc

# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
    '''Version of the python wrappers.'''
except Exception:
    version = '{version}'
else:
    del get_version
""".lstrip()

setuptools.setup(use_scm_version={
    'write_to': os.path.join('erfa', 'version.py'),
    'write_to_template': VERSION_TEMPLATE},
      ext_modules=get_extensions())
