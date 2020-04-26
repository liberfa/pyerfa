# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides the tools used to internally run the astropy test suite
from the installed astropy.  It makes use of the `pytest` testing framework.
"""
import sys
import types
import warnings
from distutils.version import LooseVersion


_deprecations_as_exceptions = False
_modules_to_ignore_on_import = set([
    r'compiler',  # A deprecated stdlib module used by py.test
    r'scipy',
    r'pygments',
    r'ipykernel',
    r'IPython',   # deprecation warnings for async and await
    r'setuptools'])
_warnings_to_ignore_entire_module = set([])
_warnings_to_ignore_by_pyver = {
    None: set([  # Python version agnostic
        # https://github.com/astropy/astropy/pull/7372
        (r"Importing from numpy\.testing\.decorators is deprecated, "
         r"import from numpy\.testing instead\.", DeprecationWarning),
        # inspect raises this slightly different warning on Python 3.6-3.7.
        # Keeping it since e.g. lxml as of 3.8.0 is still calling getargspec()
        (r"inspect\.getargspec\(\) is deprecated, use "
         r"inspect\.signature\(\) or inspect\.getfullargspec\(\)",
         DeprecationWarning),
        # https://github.com/astropy/pytest-doctestplus/issues/29
        (r"split\(\) requires a non-empty pattern match", FutureWarning),
        # Package resolution warning that we can do nothing about
        (r"can't resolve package from __spec__ or __package__, "
         r"falling back on __name__ and __path__", ImportWarning)]),
    (3, 7): set([
        # Deprecation warning for collections.abc, fixed in Astropy but still
        # used in lxml, and maybe others
        (r"Using or importing the ABCs from 'collections'",
         DeprecationWarning)])
}


def treat_deprecations_as_exceptions():
    """
    Turn all DeprecationWarnings (which indicate deprecated uses of
    Python itself or Numpy, but not within Astropy, where we use our
    own deprecation warning class) into exceptions so that we find
    out about them early.

    This completely resets the warning filters and any "already seen"
    warning state.
    """
    # First, totally reset the warning state. The modules may change during
    # this iteration thus we copy the original state to a list to iterate
    # on. See https://github.com/astropy/astropy/pull/5513.
    for module in list(sys.modules.values()):
        # We don't want to deal with six.MovedModules, only "real"
        # modules. FIXME: we no more use six, this should be useless ?
        if (isinstance(module, types.ModuleType) and
                hasattr(module, '__warningregistry__')):
            del module.__warningregistry__

    if not _deprecations_as_exceptions:
        return

    warnings.resetwarnings()

    # Hide the next couple of DeprecationWarnings
    warnings.simplefilter('ignore', DeprecationWarning)
    # Here's the wrinkle: a couple of our third-party dependencies
    # (py.test and scipy) are still using deprecated features
    # themselves, and we'd like to ignore those.  Fortunately, those
    # show up only at import time, so if we import those things *now*,
    # before we turn the warnings into exceptions, we're golden.
    for m in _modules_to_ignore_on_import:
        try:
            __import__(m)
        except ImportError:
            pass

    # Now, start over again with the warning filters
    warnings.resetwarnings()
    # Now, turn these warnings into exceptions
    _all_warns = [DeprecationWarning, FutureWarning, ImportWarning]

    # Only turn astropy deprecation warnings into exceptions if requested
    if _include_astropy_deprecations:
        _all_warns += [AstropyDeprecationWarning,
                       AstropyPendingDeprecationWarning]

    for w in _all_warns:
        warnings.filterwarnings("error", ".*", w)

    # This ignores all specified warnings from given module(s),
    # not just on import, for use of Astropy affiliated packages.
    for m in _warnings_to_ignore_entire_module:
        for w in _all_warns:
            warnings.filterwarnings('ignore', category=w, module=m)

    # This ignores only specified warnings by Python version, if applicable.
    for v in _warnings_to_ignore_by_pyver:
        if v is None or sys.version_info[:2] == v:
            for s in _warnings_to_ignore_by_pyver[v]:
                warnings.filterwarnings("ignore", s[0], s[1])

    # If using Matplotlib < 3, we should ignore the following warning since
    # this is beyond our control
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        if LooseVersion(matplotlib.__version__) < '3':
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                    module='numpy.lib.type_check')


class catch_warnings(warnings.catch_warnings):
    """
    A high-powered version of warnings.catch_warnings to use for testing
    and to make sure that there is no dependence on the order in which
    the tests are run.

    This completely blitzes any memory of any warnings that have
    appeared before so that all warnings will be caught and displayed.

    ``*args`` is a set of warning classes to collect.  If no arguments are
    provided, all warnings are collected.

    Use as follows::

        with catch_warnings(MyCustomWarning) as w:
            do.something.bad()
        assert len(w) > 0
    """

    def __init__(self, *classes):
        super().__init__(record=True)
        self.classes = classes

    def __enter__(self):
        warning_list = super().__enter__()
        treat_deprecations_as_exceptions()
        if len(self.classes) == 0:
            warnings.simplefilter('always')
        else:
            warnings.simplefilter('ignore')
            for cls in self.classes:
                warnings.simplefilter('always', cls)
        return warning_list

    def __exit__(self, type, value, traceback):
        treat_deprecations_as_exceptions()
