# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import setuptools
import subprocess
from warnings import warn


LIBERFADIR = os.path.join('liberfa', 'erfa')
ERFA_SRC = os.path.join(LIBERFADIR, 'src')


# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(setuptools.Extension):
    """Extension type that adds the NumPy include directory to include_dirs."""

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

    sources = [os.path.join('erfa', 'ufunc.c')]

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


def guess_next_dev(version):
    from setuptools_scm import git
    from setuptools_scm.version import guess_next_version

    erfa_version = git.parse(LIBERFADIR)
    if not erfa_version.exact:
        warn(f'liberfa/erfa not at a tagged release, but at {erfa_version}')

    erfa_tag = erfa_version.format_with("{tag}")
    version_string = str(version.tag)

    if version.exact:
        if not version_string.startswith(erfa_tag):
            warn(f'tag {version_string} does not start with liberfa/erfa tag {erfa_tag}')

        return version_string

    else:
        if erfa_tag > version_string:
            guessed = erfa_tag
        elif 'dev' in version_string or len(version_string.split('.')) > 3:
            return guess_next_version(version.tag)
        else:
            guessed = version_string.partition("+")[0] + '.1'
        return version.format_with("{guessed}.dev{distance}", guessed=guessed)


use_scm_version = {
    'write_to': os.path.join('erfa', 'version.py'),
    'write_to_template': VERSION_TEMPLATE,
    'version_scheme': guess_next_dev}

setuptools.setup(use_scm_version=use_scm_version,
                 ext_modules=get_extensions())
