# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Union, assert_type

import numpy as np
from numpy.typing import NDArray

from erfa import ufunc
from erfa._typing import Array0D

from .utils import (
    assert_0d_array,
    assert_1d_array,
    assert_2d_array,
    assert_literal,
    assert_none,
    assert_scalar,
)


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

    # Also a regression test for #343 - the degrees field was named "h".
    assert_scalar(idmsf["d"], np.intc)

    assert_scalar(idmsf["m"], np.intc)
    assert_scalar(idmsf["s"], np.intc)
    assert_scalar(idmsf["f"], np.intc)


def test_dt_dmsf_array() -> None:
    _, idmsf = ufunc.a2af([4], 2.345)
    assert_type(idmsf, Union["ufunc.DMSFDType", NDArray["ufunc.DMSFDType"]])
    assert type(idmsf) is np.ndarray
    assert idmsf.dtype == ufunc.dt_dmsf


def test_float64_scalar() -> None:
    anp = ufunc.anp(1)
    assert_type(anp, np.float64 | NDArray[np.float64])
    assert isinstance(anp, np.float64)


def test_float64_array() -> None:
    anpm = ufunc.anpm([1])
    assert_type(anpm, np.float64 | NDArray[np.float64])
    assert isinstance(anpm, np.ndarray)
    assert anpm.dtype == np.float64


def test_float64_gufunc() -> None:
    r5h, s5h = ufunc.fk5hip()
    assert_type(r5h, NDArray[np.float64])
    assert isinstance(r5h, np.ndarray)
    assert r5h.dtype == np.float64
    assert_type(s5h, NDArray[np.float64])
    assert isinstance(s5h, np.ndarray)
    assert s5h.dtype == np.float64


def test_dt_hmsf_scalar() -> None:
    _, ihmsf = ufunc.d2tf(4, -0.987654321)
    assert_type(ihmsf, Union["ufunc.HMSFDType", NDArray["ufunc.HMSFDType"]])
    assert isinstance(ihmsf, np.void)
    assert ihmsf.dtype == ufunc.dt_hmsf

    assert_scalar(ihmsf["h"], np.intc)
    assert_scalar(ihmsf["m"], np.intc)
    assert_scalar(ihmsf["s"], np.intc)
    assert_scalar(ihmsf["f"], np.intc)


def test_dt_hmsf_array() -> None:
    _, _, _, ihmsf, _ = ufunc.d2dtf("UTC", [5], 2400000.5, 49533.99999)
    assert_type(ihmsf, Union["ufunc.HMSFDType", NDArray["ufunc.HMSFDType"]])
    assert type(ihmsf) is np.ndarray
    assert ihmsf.dtype == ufunc.dt_hmsf


def test_leap_seconds() -> None:
    assert_1d_array(ufunc.get_leap_seconds(), "ufunc.LeapSecondDType")


def test_dt_pv_scalar() -> None:
    zpv = ufunc.zpv()
    assert_scalar(zpv, "ufunc.PVDType")
    assert_1d_array(zpv["p"], np.float64)
    assert_1d_array(zpv["v"], np.float64)


def test_dt_pv_scalar_recarray_view() -> None:
    zpv_recarray = ufunc.zpv().view(np.recarray)
    assert_type(zpv_recarray, "ufunc.PVRecord")
    assert isinstance(zpv_recarray, np.record)
    assert zpv_recarray.dtype == ufunc.dt_pv

    assert_1d_array(zpv_recarray.p, np.float64)
    assert_1d_array(zpv_recarray.v, np.float64)
    assert_1d_array(zpv_recarray["p"], np.float64)
    assert_1d_array(zpv_recarray["v"], np.float64)


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

    assert_scalar(sign[()], "ufunc.SignDType")
    assert_0d_array(sign[...], "ufunc.SignDType")

    # Not using assert_0d_array here because of a bug in numpy < 2.6
    # (https://github.com/numpy/numpy/pull/32744)
    assert_type(sign[(...,)], Array0D["ufunc.SignDType"])
    assert sign[(...,)].ndim == 0
    assert sign[(...,)].dtype == ufunc.dt_sign

    assert_1d_array(sign[None], "ufunc.SignDType")
    assert_2d_array(sign[(None, None)], "ufunc.SignDType")
    assert_scalar(sign["sign"], np.bytes_)
    assert_scalar(sign[0], np.bytes_)
    assert_scalar(sign[["sign"]], "ufunc.SignDType")


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

    assert_scalar(iydmf["y"], np.intc)
    assert_scalar(iydmf["m"], np.intc)
    assert_scalar(iydmf["d"], np.intc)
    assert_scalar(iydmf["f"], np.intc)


def test_dt_ymdf_array() -> None:
    iydmf, _ = ufunc.jdcalf(4, [2400000.5], 50123.9999)
    assert_type(iydmf, Union["ufunc.YMDFDType", NDArray["ufunc.YMDFDType"]])
    assert type(iydmf) is np.ndarray
    assert iydmf.dtype == ufunc.dt_ymdf


def test_ufunc_nin0_nout1_attributes() -> None:
    assert_none(ufunc.zpv.identity)
    assert_literal(ufunc.zpv.nin, 0)
    assert_literal(ufunc.zpv.nout, 1)
    assert_literal(ufunc.zpv.nargs, 1)
    assert_literal(ufunc.zpv.ntypes, 0)
    assert_none(ufunc.zpv.signature)


def test_ufunc_nin0_attributes() -> None:
    assert_none(ufunc.bi00.identity)
    assert_literal(ufunc.bi00.nin, 0)
    assert_literal(ufunc.bi00.nout, 3)
    assert_literal(ufunc.bi00.nargs, 3)
    assert_literal(ufunc.bi00.ntypes, 1)
    assert_none(ufunc.bi00.signature)


def test_ufunc_nin1_nout1_attributes() -> None:
    assert_none(ufunc.fad03.identity)
    assert_literal(ufunc.fae03.nin, 1)
    assert_literal(ufunc.faf03.nout, 1)
    assert_literal(ufunc.faju03.nargs, 2)
    assert_literal(ufunc.falp03.ntypes, 1)
    assert_none(ufunc.fama03.signature)


def test_ufunc_nin2_nout1_attributes() -> None:
    assert_none(ufunc.epb.identity)
    assert_literal(ufunc.epj.nin, 2)
    assert_literal(ufunc.apcg13.nout, 1)
    assert_literal(ufunc.aper.nargs, 3)
    assert_literal(ufunc.moon98.ntypes, 0)
    assert_none(ufunc.eo06a.signature)


def test_ufunc_nout1_attributes() -> None:
    assert_none(ufunc.apcs13.identity)
    assert_literal(ufunc.s00.nin, 4)
    assert_literal(ufunc.gmst06.nout, 1)
    assert_literal(ufunc.hd2pa.nargs, 4)
    assert_literal(ufunc.pas.ntypes, 1)
    assert_none(ufunc.s2xpv.signature)


def test_ufunc_attributes() -> None:
    assert_none(ufunc.epb2jd.identity)
    assert_literal(ufunc.atci13.nin, 8)
    assert_literal(ufunc.refco.nout, 2)
    assert_literal(ufunc.nut00a.nargs, 4)
    assert_literal(ufunc.pb06.ntypes, 1)
    assert_none(ufunc.fk45z.signature)


def test_gufunc_nin0_nout1_attributes() -> None:
    assert_none(ufunc.ir.identity)
    assert_literal(ufunc.zp.nin, 0)
    assert_literal(ufunc.zr.nout, 1)
    assert_literal(ufunc.ir.nargs, 1)
    assert_literal(ufunc.zp.ntypes, 1)
    assert_literal(ufunc.zr.signature, "->(3, 3)")


def test_gufunc_nin0_attributes() -> None:
    assert_none(ufunc.fk5hip.identity)
    assert_literal(ufunc.fk5hip.nin, 0)
    assert_literal(ufunc.fk5hip.nout, 2)
    assert_literal(ufunc.fk5hip.nargs, 2)
    assert_literal(ufunc.fk5hip.ntypes, 1)
    assert_literal(ufunc.fk5hip.signature, "->(3, 3),(3)")


def test_gufunc_nin1_nout1_attributes() -> None:
    assert_none(ufunc.ltp.identity)
    assert_literal(ufunc.ltpb.nin, 1)
    assert_literal(ufunc.ltpecl.nout, 1)
    assert_literal(ufunc.ltpequ.nargs, 2)
    assert_literal(ufunc.ltecm.ntypes, 1)
    assert_literal(ufunc.cp.signature, "(3)->(3)")


def test_gufunc_nout1_attributes() -> None:
    assert_none(ufunc.ab.identity)
    assert_literal(ufunc.apcg.nin, 4)
    assert_literal(ufunc.apci.nout, 1)
    assert_literal(ufunc.apco.nargs, 17)
    assert_literal(ufunc.apcs.ntypes, 0)
    assert_literal(ufunc.ld.signature, "(),(3),(3),(3),(),()->(3)")


def test_gufunc_attributes() -> None:
    assert_none(ufunc.atciqn.identity)
    assert_literal(ufunc.aticqn.nin, 4)
    assert_literal(ufunc.bp00.nout, 3)
    assert_literal(ufunc.bp06.nargs, 5)
    assert_literal(ufunc.bpn2xy.ntypes, 1)
    assert_literal(
        ufunc.pn00.signature, "(),(),(),()->(),(3, 3),(3, 3),(3, 3),(3, 3),(3, 3)"
    )
