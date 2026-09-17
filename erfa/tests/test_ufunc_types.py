# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from erfa import ufunc


def test_status_codes_scalar() -> None:
    _, _, scode = ufunc.eform(0)
    assert_type(scode, np.intc | NDArray[np.intc])
    assert isinstance(scode, np.intc)


def test_status_codes_array() -> None:
    _, _, scode = ufunc.tttcg([2453750.5], 0.892482639)
    assert_type(scode, np.intc | NDArray[np.intc])
    assert isinstance(scode, np.ndarray)
    assert scode.dtype == np.intc
