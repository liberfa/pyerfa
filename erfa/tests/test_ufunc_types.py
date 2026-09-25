# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Union, assert_type

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


def test_dt_sign_scalar() -> None:
    sign, _ = ufunc.a2af(4, 2.345)
    assert_type(sign, Union["ufunc.SignDType", NDArray["ufunc.SignDType"]])
    assert isinstance(sign, np.void)
    assert sign.dtype == ufunc.dt_sign

    assert_type(sign[()], "ufunc.SignDType")
    assert sign[()].dtype == ufunc.dt_sign

    assert_type(sign[...], np.ndarray[tuple[()], np.dtype["ufunc.SignDType"]])
    assert type(sign[...]) is np.ndarray
    assert sign[...].ndim == 0
    assert sign[...].dtype == ufunc.dt_sign

    assert_type(sign[(...,)], np.ndarray[tuple[()], np.dtype["ufunc.SignDType"]])
    assert sign[(...,)].ndim == 0
    assert sign[...].dtype == ufunc.dt_sign

    assert_type(sign[None], np.ndarray[tuple[int], np.dtype["ufunc.SignDType"]])
    assert type(sign[None]) is np.ndarray
    assert sign[None].ndim == 1
    assert sign[None].dtype == ufunc.dt_sign

    assert_type(
        sign[(None, None)], np.ndarray[tuple[int, int], np.dtype["ufunc.SignDType"]]
    )
    assert type(sign[(None, None)]) is np.ndarray
    assert sign[(None, None)].ndim == 2
    assert sign[(None, None)].dtype == ufunc.dt_sign

    assert_type(sign["sign"], np.bytes_)
    assert type(sign["sign"]) is np.bytes_

    assert_type(sign[0], np.bytes_)
    assert type(sign[0]) is np.bytes_

    assert_type(sign[["sign"]], "ufunc.SignDType")
    assert sign[["sign"]].dtype == ufunc.dt_sign


def test_dt_sign_array() -> None:
    sign, _ = ufunc.a2tf(4, [-3.01234])
    assert_type(sign, Union["ufunc.SignDType", NDArray["ufunc.SignDType"]])
    assert isinstance(sign, np.ndarray)
    assert sign.dtype == ufunc.dt_sign
