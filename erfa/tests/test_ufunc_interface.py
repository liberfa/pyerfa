# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Final

import numpy as np
from numpy.testing import assert_array_equal

from erfa import ufunc

SQRT2: Final = np.sqrt(2.0)


def test_positional_out() -> None:
    pos_args = ([-0.763, -0.608, -0.216], [2.104e-5, -8.910e-5, -3.863e-5], 0.9998, 1)
    result = ufunc.ab(*pos_args)
    arr = np.zeros_like(result)
    result_with_out = ufunc.ab(*pos_args, arr)
    assert result_with_out is arr
    assert_array_equal(arr, result)


def test_out_arr() -> None:
    result = ufunc.s2pv(np.pi / 2.0, np.pi / 4.0, 2.0, SQRT2 / 2.0, 0.0, 0.0)
    out = np.ones_like(result)
    result_with_out = ufunc.s2pv(
        np.pi / 2.0, np.pi / 4.0, 2.0, SQRT2 / 2.0, 0.0, 0.0, out=out
    )
    assert result_with_out is out
    assert_array_equal(out, result)


def test_out_tuple_of_arr() -> None:
    result = ufunc.zpv()
    out = np.ones_like(result)
    result_with_out = ufunc.zpv(out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_out_mixed_tuple() -> None:
    pos_args = ([2012, 2013], 12, 12)
    _, result, _ = ufunc.cal2jd(*pos_args)
    mjd = np.zeros(2)
    _, result_with_out, _ = ufunc.cal2jd(*pos_args, out=(None, mjd, None))
    assert result_with_out is mjd
    assert_array_equal(mjd, result)


def test_out_ellipsis() -> None:
    pos_args = (2451540, -7342.5)
    none_result = ufunc.epj(*pos_args, out=None)
    assert type(none_result) is np.float64
    ellipsis_result = ufunc.epj(*pos_args, out=...)
    assert type(ellipsis_result) is np.ndarray
    assert_array_equal(ellipsis_result, none_result)


def test_non_contiguous_output_matrix():
    # Fix for copy_from_double33 problem found by @devdanzin
    # Create non-contiguous output array (only reachable via ufunc interface).
    result = np.zeros((3, 4))[:, :3]
    out = ufunc.ltecm(2005.0, out=result)
    assert out is result
    expected = ufunc.ltecm(2005.0)
    assert_array_equal(expected, result)
