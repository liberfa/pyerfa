"""Wrapper, ERFA and SOFA version information."""

from ._version import version as _version

# Set the version numbers a bit indirectly, so that Sphinx can pick up
# up the docstrings and list the values.
try:
    from . import ufunc
except ImportError as exc:
    # If compiled to use a system liberfa, that library can be too old, and
    # miss functions that are available in newer liberfa. If so, we should
    # bail since nothing will work, but let's try to give a more informative
    # error message.
    try:
        from ctypes import CDLL, c_char_p, util
        liberfa = CDLL(util.find_library('erfa'))
        liberfa.eraVersion.restype = c_char_p
        erfa_version = liberfa.eraVersion().decode('ascii')
    except Exception:
        pass
    else:
        if erfa_version.split(".")[:2] < _version.split(".")[:2]:
            raise ImportError(
                f"liberfa {erfa_version} too old for PyERFA {_version}. "
                "This should only be possible if you are using a system liberfa; "
                "try installing using 'pip install pyerfa', with environment variable "
                "PYERFA_USE_SYSTEM_LIBERFA unset or 0.") from exc

    raise


version = _version
'''Version of the python wrappers.'''

erfa_version = ufunc.erfa_version
'''Version of the C ERFA library that is wrapped.'''

sofa_version = ufunc.sofa_version
'''Version of the SOFA library the ERFA library is based on.'''

del ufunc, _version
