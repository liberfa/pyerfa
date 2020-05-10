======
PyERFA
======

PyERFA is the Python_ wrapper for the ERFA_ library (Essential Routines for
Fundamental Astronomy), a C library containing key algorithms for astronomy,
which is based on the SOFA library published by the International Astronomical
Union (IAU).  All C routines are wrapped as Numpy_ `universal functions
<https://numpy.org/devdocs/reference/ufuncs.html>`_, so that they can be
called with scalar or array inputs.

The project is a split of ``astropy._erfa`` module, developed in the
context of Astropy_ project, into a standalone package.  It contains
the ERFA_ C source code as a git submodule.

.. _Python: https://www.python.org/
.. _ERFA: https://github.com/liberfa/erfa
.. _Numpy: https://numpy.org/
.. _Astropy: https://www.astropy.org


Installation instructions
-------------------------

The package can be installed from the package directory using a simple::

  $ pip install .

and similarly a wheel_ can be created with::

  $ pip wheel .

.. note:: If you already have the C library ``liberfa`` on your
  system, you can use that by setting environment variable
  ``PYERFA_USE_SYSTEM_LIBERFA=1``.


.. _wheel: https://github.com/pypa/wheel


Testing
-------

For testing, one can install the packages together with its testing
dependencies and then test it with::

  $ pip install .[test]
  $ pytest

Alternatively, one can use ``tox``, which will set up a separate testing
environment for you, with::

  $ tox -e test

License
-------

PyERFA is licensed under a 3-clause BSD style license - see the
`LICENSE.rst <LICENSE.rst>`_ file.
