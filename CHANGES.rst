2.0.0 (17/05/2021)
==================

- Bundled liberfa version update to v2.0.0. This includes new functionality,
  and hence pyerfa 2.0.0 cannot run with previous versions of liberfa.
- ``erfa.dt_eraLDBODY`` has been corrected to ensure that the 'pv' entry is
  now of type ``erfa.dt_pv``, so that cross-assignments with that dtype work
  correctly. [gh-74]
- ``erfa_generator`` now also generates a ``test_ufunc.py`` file that
  runs all the C code tests on the ufuncs, thus verifying the code
  wrapping worked correctly. As part of that, the ability to give
  specific output file names has been removed, as it was not used.
  (Note: these changes have no effect on use of PyERFA.) [gh-71]

1.7.3 (25/04/2021)
==================

- Bundled liberfa version update to v1.7.3.
- Fixed a bug that caused the output of ``rx``, ``ry``, and ``rz`` to be
  boolean rather than float for some compilers/OS. [gh-72]

1.7.2 (25/01/2021)
==================

- Bundled liberfa version update to v1.7.2.
- The classproperty decorator is now thread-safe
  (backport https://github.com/astropy/astropy/pull/11224).


1.7.1.1 (18/11/2020)
====================

- Fix incorrect ``__version__`` value [gh-60].


1.7.1 (16/11/2020)
==================

- Bundled liberfa version update to v1.7.1.
- Now it is possible to build against system liberfa [gh-39].
- Improved the setup machinery to ensure a proper configuration of the
  embedded liberfa [gh-53] (see also https://github.com/liberfa/erfa/pull/73).
- Improve version testing [gh-53] (see issue gh-52 for rationale).
- Reworked version management [gh-57].
- Make pyerfa build reproducible [gh-46] (see issue gh-45 for rationale).
- Improved docstring titles [gh-47].
- Most of the CI has been moved to GitHub Actions.


1.7 (2020-05-31)
================

- Initial release, based on ERFA version 1.7.0, which in turn is based
  on SOFA version 20190722.
