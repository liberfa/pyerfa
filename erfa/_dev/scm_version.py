# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.

import functools
from pathlib import Path
from warnings import warn

try:
    from setuptools_scm import Configuration, git
    from setuptools_scm import get_version as _get_version
    from setuptools_scm.version import guess_next_version

    def _guess_next_dev(version, liberfadir=None):
        if liberfadir is None:
            liberfadir = Path(__file__).parent.parent.parent / "liberfa" / "erfa"

        config = Configuration(root=liberfadir)
        erfa_version = git.parse(liberfadir, config=config)
        if not erfa_version.exact:
            warn(f'liberfa/erfa not at a tagged release, but at {erfa_version}')

        erfa_tag = erfa_version.format_with("{tag}")
        version_string = str(version.tag)

        if version.exact:
            if not version_string.startswith(erfa_tag):
                warn(f'tag {version_string} does not start with '
                     f'liberfa/erfa tag {erfa_tag}')

            return version_string

        return (
            version.format_with("{guessed}.0.dev{distance}", guessed=erfa_tag)
            if erfa_tag > version_string
            else version.format_next_version(guess_next_version)
        )

    get_version = functools.partial(
        _get_version,
        root=Path("..", ".."),
        version_scheme=_guess_next_dev,
        relative_to=__file__,
    )
except Exception as exc:
    raise ImportError('setuptools_scm broken or not installed') from exc
