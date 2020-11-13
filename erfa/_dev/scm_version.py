# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.

import pathlib
import functools
import os.path as pth
from warnings import warn

try:
    from setuptools_scm import git, get_version as _get_version
    from setuptools_scm.version import guess_next_version

    def _guess_next_dev(version, liberfadir=None):
        if liberfadir is None:
            liberfadir = pathlib.Path(
                __file__).parent.parent.parent / 'liberfa' / 'erfa'

        erfa_version = git.parse(liberfadir)
        if not erfa_version.exact:
            warn(f'liberfa/erfa not at a tagged release, but at {erfa_version}')

        erfa_tag = erfa_version.format_with("{tag}")
        version_string = str(version.tag)

        if version.exact:
            if not version_string.startswith(erfa_tag):
                warn(f'tag {version_string} does not start with '
                     f'liberfa/erfa tag {erfa_tag}')

            return version_string

        else:
            if erfa_tag > version_string:
                guessed = erfa_tag
            elif 'dev' in version_string or len(version_string.split('.')) > 3:
                return guess_next_version(version.tag)
            else:
                guessed = version_string.partition("+")[0] + '.1'
            return version.format_with("{guessed}.dev{distance}",
                                       guessed=guessed)

    get_version = functools.partial(_get_version,
                                    root=pth.join('..', '..'),
                                    version_scheme=_guess_next_dev,
                                    relative_to=__file__)
except Exception as exc:
    raise ImportError('setuptools_scm broken or not installed') from exc
