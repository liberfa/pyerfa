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


def test_dt_dmsf_scalar() -> None:
    _, idmsf = ufunc.a2af(4, 2.345)
    assert_type(idmsf, Union["ufunc.DMSFDType", NDArray["ufunc.DMSFDType"]])
    assert isinstance(idmsf, np.void)
    assert idmsf.dtype == ufunc.dt_dmsf

    assert_type(idmsf["d"], np.intc)
    # Also a regression test for #343 - the degrees field was named "h".
    assert idmsf["d"].dtype == np.intc
    assert_type(idmsf["m"], np.intc)
    assert idmsf["m"].dtype == np.intc
    assert_type(idmsf["s"], np.intc)
    assert idmsf["s"].dtype == np.intc
    assert_type(idmsf["f"], np.intc)
    assert idmsf["f"].dtype == np.intc


def test_dt_dmsf_array() -> None:
    _, idmsf = ufunc.a2af([4], 2.345)
    assert_type(idmsf, Union["ufunc.DMSFDType", NDArray["ufunc.DMSFDType"]])
    assert type(idmsf) is np.ndarray
    assert idmsf.dtype == ufunc.dt_dmsf


def test_dt_hmsf_scalar() -> None:
    _, ihmsf = ufunc.d2tf(4, -0.987654321)
    assert_type(ihmsf, Union["ufunc.HMSFDType", NDArray["ufunc.HMSFDType"]])
    assert isinstance(ihmsf, np.void)
    assert ihmsf.dtype == ufunc.dt_hmsf

    assert_type(ihmsf["h"], np.intc)
    assert ihmsf["h"].dtype == np.intc
    assert_type(ihmsf["m"], np.intc)
    assert ihmsf["m"].dtype == np.intc
    assert_type(ihmsf["s"], np.intc)
    assert ihmsf["s"].dtype == np.intc
    assert_type(ihmsf["f"], np.intc)
    assert ihmsf["f"].dtype == np.intc


def test_dt_hmsf_array() -> None:
    _, _, _, ihmsf, _ = ufunc.d2dtf("UTC", [5], 2400000.5, 49533.99999)
    assert_type(ihmsf, Union["ufunc.HMSFDType", NDArray["ufunc.HMSFDType"]])
    assert type(ihmsf) is np.ndarray
    assert ihmsf.dtype == ufunc.dt_hmsf


def test_leap_seconds() -> None:
    leap_seconds = ufunc.get_leap_seconds()
    assert_type(leap_seconds, np.ndarray[tuple[int], np.dtype["ufunc.LeapSecondDType"]])
    assert isinstance(leap_seconds, np.ndarray)
    assert leap_seconds.ndim == 1
    assert leap_seconds.dtype == ufunc.dt_eraLEAPSECOND


def test_dt_pv_scalar() -> None:
    zpv = ufunc.zpv()
    assert_type(zpv, Union["ufunc.PVDType", NDArray["ufunc.PVDType"]])
    assert isinstance(zpv, np.void)
    assert zpv.dtype == ufunc.dt_pv

    assert_type(zpv["p"], np.ndarray[tuple[int], np.dtype[np.float64]])
    assert type(zpv["p"]) is np.ndarray
    assert zpv["p"].ndim == 1
    assert zpv["p"].dtype == np.float64

    assert_type(zpv["v"], np.ndarray[tuple[int], np.dtype[np.float64]])
    assert type(zpv["v"]) is np.ndarray
    assert zpv["v"].ndim == 1
    assert zpv["v"].dtype == np.float64


def test_dt_pv_array() -> None:
    pv = ufunc.p2pv([[1, 2, 3]])
    assert_type(pv, Union["ufunc.PVDType", NDArray["ufunc.PVDType"]])
    assert isinstance(pv, np.ndarray)
    assert pv.dtype == ufunc.dt_pv


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


def test_dt_ymdf_scalar() -> None:
    iydmf, _ = ufunc.jdcalf(4, 2400000.5, 50123.9999)
    assert_type(iydmf, Union["ufunc.YMDFDType", NDArray["ufunc.YMDFDType"]])
    assert isinstance(iydmf, np.void)
    assert iydmf.dtype == ufunc.dt_ymdf

    assert_type(iydmf["y"], np.intc)
    assert iydmf["y"].dtype == np.intc
    assert_type(iydmf["m"], np.intc)
    assert iydmf["m"].dtype == np.intc
    assert_type(iydmf["d"], np.intc)
    assert iydmf["d"].dtype == np.intc
    assert_type(iydmf["f"], np.intc)
    assert iydmf["f"].dtype == np.intc


def test_dt_ymdf_array() -> None:
    iydmf, _ = ufunc.jdcalf(4, [2400000.5], 50123.9999)
    assert_type(iydmf, Union["ufunc.YMDFDType", NDArray["ufunc.YMDFDType"]])
    assert type(iydmf) is np.ndarray
    assert iydmf.dtype == ufunc.dt_ymdf
