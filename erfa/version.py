"""Wrapper, ERFA and SOFA version information."""

# NOTE: First try _dev.scm_version if it exists and setuptools_scm is installed
# This file is not included in pyerfa wheels/tarballs, so otherwise it will
# fall back on the generated _version module.
try:
    try:
        from ._dev.scm_version import get_version as _get_version
        _version = _get_version()
        del _get_version
    except ImportError:
        from ._version import version as _version
except Exception:
    import warnings
    warnings.warn(
        f'could not determine {__name__.split(".")[0]} package version; '
        f'this indicates a broken installation')
    del warnings

    _version = '0.0.0'

# Set the version numbers a bit indirectly, so that Sphinx can pick up
# up the docstrings and list the values.
from . import ufunc

version = _version
'''Version of the python wrappers.'''

erfa_version = ufunc.erfa_version
'''Version of the C ERFA library that is wrapped.'''

sofa_version = ufunc.sofa_version
'''Version of the SOFA library the ERFA library is based on.'''

del ufunc, _version
