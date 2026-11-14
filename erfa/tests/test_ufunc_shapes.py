# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from erfa import ufunc

from .utils import assert_0d_array, assert_1d_array, assert_2d_array, assert_scalar


def test_ufunc_nin1_nout1_scalar() -> None:
    assert_scalar(ufunc.fal03(0.8), np.float64)


def test_ufunc_nin1_nout1_array() -> None:
    assert_1d_array(ufunc.falp03([0.8]), np.float64)


def test_ufunc_nin1_nout1_scalar_out_none_tuple() -> None:
    assert_scalar(ufunc.fama03(np.float64(0.8), out=(None,)), np.float64)


def test_ufunc_nin1_nout1_array_out_none_tuple() -> None:
    assert_1d_array(ufunc.fame03((1,), out=(None,)), np.float64)


def test_ufunc_nin1_nout1_scalar_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(ufunc.fane03(1, out=...), np.float64)


def test_ufunc_nin1_nout1_array_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(ufunc.cpv(ufunc.zpv(out=...), out=...), "ufunc.PVDType")


def test_ufunc_nin1_nout1_positional_out_array() -> None:
    assert_1d_array(ufunc.faom03(np.float16(0.8), np.zeros(2)), np.float64)


def test_ufunc_nin1_nout1_keyword_out_array() -> None:
    assert_2d_array(ufunc.fapa03(np.array([0.8]), out=np.empty((2, 1))), np.float64)
