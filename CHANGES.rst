2.0.1.1 (2023-10-19)
====================
- Ensured pyerfa works on PyPy too with the Python limited API. [gh-120]
- Ensure any non-contigous multi-dimensional inputs are recognized
  properly, so that, e.g., a non-contiguous matrix is copied as
  needed before input to the erfa functions. [gh-124]

2.0.1 (2023-10-13)
==================

- Bundled liberfa version update to v2.0.1, which is derived from SOFA
  version 19 (20231011), which has a few bug fixes. [gh-118]
- Fix dangling pointer leading to unexpected behavior with ``-O3``. [gh-104]
- PyERFA now only uses the Python limited API. [gh-94]
- Ensure things work under python 3.12 and recent setuptools-scm. [gh-113]

2.0.0.3 (2023-03-22)
====================

- Ensure minimum numpy version of 1.17 continues to work (for astropy LTS).

2.0.0.2 (2023-03-19)
====================

- Fix compatibility with numpy v1.24.
- Ensure documentation is correct also for functions without parameters.
- CI configuration updated.
- Switch building of wheels to GitHub Actions.
- Min ``setuptools_scm`` updated.
- Min numpy version in ``tox.ini`` updated to fix CI issues.

2.0.0.1 (2021-11-02)
====================

- The underlying universal functions in ``erfa.ufunc`` now work with an ``out``
  argument also if the required output is a structured array. [gh-76]

2.0.0 (2021-05-17)
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

1.7.3 (2021-04-25)
==================

- Bundled liberfa version update to v1.7.3.
- Fixed a bug that caused the output of ``rx``, ``ry``, and ``rz`` to be
  boolean rather than float for some compilers/OS. [gh-72]

1.7.2 (2021-01-25)
==================

- Bundled liberfa version update to v1.7.2.
- The classproperty decorator is now thread-safe
  (backport https://github.com/astropy/astropy/pull/11224).


1.7.1.1 (2020-11-18)
====================

- Fix incorrect ``__version__`` value [gh-60].


1.7.1 (2020-11-16)
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
