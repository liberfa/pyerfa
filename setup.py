# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from warnings import warn

import packaging.version
import setuptools
from setuptools_scm import Configuration, git
from setuptools_scm.version import guess_next_version

LIBERFADIR = Path("liberfa", "erfa")
ERFA_SRC = LIBERFADIR / "src"
GEN_FILES = [Path("erfa", "core.py"), Path("erfa", "ufunc.c")]


# build with Py_LIMITED_API unless in freethreading build (which does not currently
# support the limited API in py313t)
USE_PY_LIMITED_API = not sysconfig.get_config_var("Py_GIL_DISABLED")

options = {}
if USE_PY_LIMITED_API:
    options["bdist_wheel"] = {"py_limited_api": "cp310"}


def newer(source, target):
    if not source.exists():
        raise FileNotFoundError(f"file '{source.resolve()}' does not exist")

    if not target.exists():
        return 1

    return source.stat().st_mtime > target.stat().st_mtime


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


def get_liberfa_versions(path=LIBERFADIR / "configure.ac"):
    s = path.read_text()

    mobj = re.search(r'AC_INIT\(\[erfa\],\[(?P<version>[0-9.]+)\]\)', s)
    if not mobj:
        warn('unable to detect liberfa version')
        return []

    version = packaging.version.parse(mobj.group('version'))

    mobj = re.search(
        r'AC_DEFINE\(\[SOFA_VERSION\], \["(?P<version>\d{8}(_\w)?)"\],', s)
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
    gen_files_exist = all(path.is_file() for path in GEN_FILES)
    gen_files_outdated = False
    if ERFA_SRC.is_dir():
        # assume that 'erfaversion.c' is updated at each release at least
        src = ERFA_SRC / "erfaversion.c"
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

    sources = [Path("erfa", "ufunc.c")]
    include_dirs = []
    libraries = []
    define_macros = []

    if int(os.environ.get('PYERFA_USE_SYSTEM_LIBERFA', 0)):
        print('Using system liberfa')
        libraries.append('erfa')
    else:
        # get all of the .c files in the liberfa/erfa/src directory
        sources.extend(
            file
            for file in ERFA_SRC.iterdir()
            if file.suffix == ".c" and not file.name.startswith("t_")
        )

        include_dirs.append(ERFA_SRC)

        # liberfa configuration
        config_h = LIBERFADIR / "config.h"
        if not config_h.exists():
            print('Configure liberfa')
            try:
                if not (LIBERFADIR / "configure").exists():
                    subprocess.run(
                        ['./bootstrap.sh'], check=True, cwd=LIBERFADIR)
                subprocess.run(['./configure'], check=True, cwd=LIBERFADIR)
            except (subprocess.SubprocessError, OSError) as exc:
                warn(f'unable to configure liberfa: {exc}')

        if not config_h.exists():
            liberfa_versions = get_liberfa_versions()
            if liberfa_versions:
                print('Configure liberfa ("configure.ac" scan)')
                lines = []
                for name, value in liberfa_versions:
                    # making sure strings are correctly quoted
                    lines.append(f'#define {name} {value!r}'.replace("'", '"'))
                config_h.write_text("\n".join(lines))
            else:
                warn('unable to get liberfa version')

        if config_h.exists():
            include_dirs.append(LIBERFADIR)
            define_macros.append(('HAVE_CONFIG_H', '1'))
        elif 'sdist' in sys.argv:
            raise RuntimeError('missing "configure" script in "liberfa/erfa"')

    if USE_PY_LIMITED_API:
        define_macros.append(("Py_LIMITED_API", "0x30900f0"))

    erfa_ext = NumpyExtension(
        name="erfa.ufunc",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        define_macros=define_macros,
        py_limited_api=USE_PY_LIMITED_API,
        language="c")

    return [erfa_ext]


def guess_next_dev(version):
    erfa_version = git.parse(LIBERFADIR, config=Configuration(root=LIBERFADIR))
    if not erfa_version.exact:
        warn(f"liberfa/erfa not at a tagged release, but at {erfa_version}")

    erfa_tag = erfa_version.format_with("{tag}")
    version_string = str(version.tag)

    if version.exact:
        if not version_string.startswith(erfa_tag):
            warn(
                f"tag {version_string} does not start with liberfa/erfa tag {erfa_tag}"
            )
        return version_string

    return (
        version.format_with("{guessed}.0.dev{distance}", guessed=erfa_tag)
        if erfa_tag > version_string
        else version.format_next_version(guess_next_version)
    )

use_scm_version = {
    'version_scheme': guess_next_dev,
}

setuptools.setup(
    use_scm_version=use_scm_version,
    ext_modules=get_extensions(),
    options=options,
)
