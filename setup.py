# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re
import sys
import inspect
import textwrap
import functools
import setuptools
import subprocess
from warnings import warn
from distutils.dep_util import newer
import packaging.version


LIBERFADIR = os.path.join('liberfa', 'erfa')
ERFA_SRC = os.path.join(LIBERFADIR, 'src')
GEN_FILES = [
    os.path.join('erfa', 'core.py'),
    os.path.join('erfa', 'ufunc.c'),
]


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


def get_liberfa_versions(path=os.path.join(LIBERFADIR, 'configure.ac')):
    with open(path) as fd:
        s = fd.read()

    mobj = re.search(r'AC_INIT\(\[erfa\],\[(?P<version>[0-9.]+)\]\)', s)
    if not mobj:
        warn('unable to detect liberfa version')
        return []

    version = packaging.version.parse(mobj.group('version'))

    mobj = re.search(
        r'AC_DEFINE\(\[SOFA_VERSION\], \["(?P<version>\d{8})"\],', s)
    if not mobj:
        warn('unable to detect SOFA version')
        return []
    sofa_version = mobj.group('version')

    return [
        ('PACKAGE_VERSION', version.base_version),
        ('PACKAGE_VERSION_MAJOR', version.major),
        ('PACKAGE_VERSION_MINOR', version.minor),
        ('PACKAGE_VERSION_MICRO', version.micro),
        ('SOFA_VERSION', sofa_version),
    ]


def get_extensions():
    gen_files_exist = all(os.path.isfile(fn) for fn in GEN_FILES)
    gen_files_outdated = False
    if os.path.isdir(ERFA_SRC):
        # assume that 'erfaversion.c' is updated at each release at least
        src = os.path.join(ERFA_SRC, 'erfaversion.c')
        gen_files_outdated = any(newer(src, fn) for fn in GEN_FILES)
    elif not gen_files_exist:
        raise RuntimeError(
            'Missing "liberfa" source files, unable to generate '
            '"erfa/ufunc.c" and "erfa/core.py". '
            'Please check your source tree. '
            'Maybe "git submodule update" could help.')

    if not gen_files_exist or gen_files_outdated:
        print('Run "erfa_generator.py"')
        cmd = [sys.executable, 'erfa_generator.py', ERFA_SRC, '--quiet']
        subprocess.run(cmd, check=True)

    sources = [os.path.join('erfa', 'ufunc.c')]
    include_dirs = []
    libraries = []
    define_macros = []

    if int(os.environ.get('PYERFA_USE_SYSTEM_LIBERFA', 0)):
        libraries.append('erfa')
    else:
        # get all of the .c files in the liberfa/erfa/src directory
        erfafns = os.listdir(ERFA_SRC)
        sources.extend([os.path.join(ERFA_SRC, fn)
                        for fn in erfafns
                        if fn.endswith('.c') and not fn.startswith('t_')])

        include_dirs.append(ERFA_SRC)

        # liberfa configuration
        config_h = os.path.join(LIBERFADIR, 'config.h')
        if not os.path.exists(config_h):
            print('Configure liberfa')
            configure = os.path.join(LIBERFADIR, 'configure')
            try:
                if not os.path.exists(configure):
                    subprocess.run(
                        ['./bootstrap.sh'], check=True, cwd=LIBERFADIR)
                subprocess.run(['./configure'], check=True, cwd=LIBERFADIR)
            except (subprocess.SubprocessError, OSError) as exc:
                warn(f'unable to configure liberfa: {exc}')

        if not os.path.exists(config_h):
            liberfa_versions = get_liberfa_versions()
            if liberfa_versions:
                print('Configure liberfa ("configure.ac" scan)')
                lines = []
                for name, value in get_liberfa_versions():
                    lines.append(f'#define {name} "{value}"')
                with open(config_h, 'w') as fd:
                    fd.write('\n'.join(lines))
            else:
                warn('unable to get liberfa version')

        if os.path.exists(config_h):
            include_dirs.append(LIBERFADIR)
            define_macros.append(('HAVE_CONFIG_H', '1'))
        elif (not os.path.exists(configure) and 'sdist' in sys.argv and
              sys.platform != 'win32'):
            raise RuntimeError(
                'missing "configure" script in "liberfa/erfa"')

    erfa_ext = NumpyExtension(
        name="erfa.ufunc",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        define_macros=define_macros,
        language="c")

    return [erfa_ext]


def _guess_next_dev(version, liberfadir=None):
    from setuptools_scm import git
    from setuptools_scm.version import guess_next_version

    if liberfadir is None:
        import pathlib
        liberfadir = pathlib.Path(__file__).parent.parent / 'liberfa' / 'erfa'

    erfa_version = git.parse(liberfadir)
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


code = textwrap.indent(inspect.getsource(_guess_next_dev), '    ')
escaped_code = code.replace('{', '{{').replace('}', '}}')


VERSION_TEMPLATE = f"""
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
    from warnings import warn
    from setuptools_scm import get_version

{escaped_code}

    version = get_version(root='..', version_scheme=_guess_next_dev,
                          relative_to=__file__)
    '''Version of the python wrappers.'''
except Exception:
    version = '{{version}}'
else:
    del get_version, _guess_next_dev
""".lstrip()


guess_next_dev = functools.partial(_guess_next_dev, liberfadir=LIBERFADIR)


use_scm_version = {
    'write_to': os.path.join('erfa', 'version.py'),
    'write_to_template': VERSION_TEMPLATE,
    'version_scheme': guess_next_dev}


setuptools.setup(use_scm_version=use_scm_version,
                 ext_modules=get_extensions())
