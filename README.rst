======
PyERFA
======

PyERFA is the Python_ wrapper for the ERFA_ library.

ERFA (Essential Routines for Fundamental Astronomy) is a C library containing
key algorithms for astronomy, and is based on the SOFA library published by
the International Astronomical Union (IAU).

The project is a split of `astropy._erfa` sub-module, developed in the
context of Astropy_ project, the into a standalone package.

See also https://github.com/astropy/astropy/issues/9802.

.. _Python: https://www.python.org/
.. _ERFA: https://github.com/liberfa/erfa
.. _Astropy: https://www.astropy.org


Setup the ERFA_ source code
---------------------------

The ERFA_ source code, necessary to build the `erfa` Python_ package, is
not included in the repository.
It can be downloaded and extracted into the proper directory (`cextern/erfa`)
using the following command::

  $ make cextern/erfa


Build instructions
------------------

Once the  ERFA_ source code is set-up the standard procedure to build
Python packages can be used::

  $ python3 setup.py sdist

produces the source distribution including the generated `C` wrapper code.

For Python_ wheel_ the procedure is the following::

  $ python3 -m pip wheel .


.. _wheel: https://github.com/pypa/wheel


Testing
-------

The following commands can be used to test the PyERFA package::

  $ python3 setup.py build_ext --inplace
  $ python3 -m pytest -v


License
-------

PyERFA is licensed under a 3-clause BSD style license - see the
`LICENSE.rst <LICENSE.rst>`_ file.
