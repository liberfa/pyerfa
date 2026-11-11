# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Literal, Union, assert_type

import numpy as np
from numpy.typing import NDArray

from erfa import ufunc
from erfa._typing import Array0D

from .utils import assert_0d_array, assert_1d_array, assert_2d_array, assert_scalar


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
    assert_type(ufunc.zpv.identity, None)
    assert ufunc.zpv.identity is None
    assert_type(ufunc.zpv.nin, Literal[0])
    assert ufunc.zpv.nin == 0
    assert_type(ufunc.zpv.nout, Literal[1])
    assert ufunc.zpv.nout == 1
    assert_type(ufunc.zpv.nargs, Literal[1])
    assert ufunc.zpv.nargs == 1
    assert_type(ufunc.zpv.ntypes, Literal[0])
    assert ufunc.zpv.ntypes == 0
    assert_type(ufunc.zpv.signature, None)
    assert ufunc.zpv.signature is None


def test_ufunc_nin0_attributes() -> None:
    assert_type(ufunc.bi00.identity, None)
    assert ufunc.bi00.identity is None
    assert_type(ufunc.bi00.nin, Literal[0])
    assert ufunc.bi00.nin == 0
    assert_type(ufunc.bi00.nout, Literal[3])
    assert ufunc.bi00.nout == 3
    assert_type(ufunc.bi00.nargs, Literal[3])
    assert ufunc.bi00.nargs == 3
    assert_type(ufunc.bi00.ntypes, Literal[1])
    assert ufunc.bi00.ntypes == 1
    assert_type(ufunc.bi00.signature, None)
    assert ufunc.bi00.signature is None


def test_ufunc_nin1_nout1_attributes() -> None:
    assert_type(ufunc.fad03.identity, None)
    assert ufunc.fad03.identity is None
    assert_type(ufunc.fae03.nin, Literal[1])
    assert ufunc.fae03.nin == 1
    assert_type(ufunc.faf03.nout, Literal[1])
    assert ufunc.faf03.nout == 1
    assert_type(ufunc.faju03.nargs, Literal[2])
    assert ufunc.faju03.nargs == 2
    assert_type(ufunc.falp03.ntypes, Literal[1])
    assert ufunc.falp03.ntypes == 1
    assert_type(ufunc.fama03.signature, None)
    assert ufunc.fama03.signature is None


def test_ufunc_nin2_nout1_attributes() -> None:
    assert_type(ufunc.epb.identity, None)
    assert ufunc.epb.identity is None
    assert_type(ufunc.epj.nin, Literal[2])
    assert ufunc.epj.nin == 2
    assert_type(ufunc.apcg13.nout, Literal[1])
    assert ufunc.apcg13.nout == 1
    assert_type(ufunc.aper.nargs, Literal[3])
    assert ufunc.aper.nargs == 3
    assert_type(ufunc.moon98.ntypes, Literal[0])
    assert ufunc.moon98.ntypes == 0
    assert_type(ufunc.eo06a.signature, None)
    assert ufunc.eo06a.signature is None


def test_ufunc_nout1_attributes() -> None:
    assert_type(ufunc.apcs13.identity, None)
    assert ufunc.apcs13.identity is None
    assert_type(ufunc.s00.nin, Literal[4])
    assert ufunc.s00.nin == 4
    assert_type(ufunc.gmst06.nout, Literal[1])
    assert ufunc.gmst06.nout == 1
    assert_type(ufunc.hd2pa.nargs, Literal[4])
    assert ufunc.hd2pa.nargs == 4
    assert_type(ufunc.pas.ntypes, Literal[1])
    assert ufunc.pas.ntypes == 1
    assert_type(ufunc.s2xpv.signature, None)
    assert ufunc.s2xpv.signature is None


def test_ufunc_attributes() -> None:
    assert_type(ufunc.epb2jd.identity, None)
    assert ufunc.epb2jd.identity is None
    assert_type(ufunc.atci13.nin, Literal[8])
    assert ufunc.atci13.nin == 8
    assert_type(ufunc.refco.nout, Literal[2])
    assert ufunc.refco.nout == 2
    assert_type(ufunc.nut00a.nargs, Literal[4])
    assert ufunc.nut00a.nargs == 4
    assert_type(ufunc.pb06.ntypes, Literal[1])
    assert ufunc.pb06.ntypes == 1
    assert_type(ufunc.fk45z.signature, None)
    assert ufunc.fk45z.signature is None


def test_gufunc_nin0_nout1_attributes() -> None:
    assert_type(ufunc.ir.identity, None)
    assert ufunc.ir.identity is None
    assert_type(ufunc.zp.nin, Literal[0])
    assert ufunc.zp.nin == 0
    assert_type(ufunc.zr.nout, Literal[1])
    assert ufunc.zr.nout == 1
    assert_type(ufunc.ir.nargs, Literal[1])
    assert ufunc.ir.nargs == 1
    assert_type(ufunc.zp.ntypes, Literal[1])
    assert ufunc.zp.ntypes == 1
    assert_type(ufunc.zr.signature, Literal["->(3, 3)"])
    assert ufunc.zr.signature == "->(3, 3)"


def test_gufunc_nin0_attributes() -> None:
    assert_type(ufunc.fk5hip.identity, None)
    assert ufunc.fk5hip.identity is None
    assert_type(ufunc.fk5hip.nin, Literal[0])
    assert ufunc.fk5hip.nin == 0
    assert_type(ufunc.fk5hip.nout, Literal[2])
    assert ufunc.fk5hip.nout == 2
    assert_type(ufunc.fk5hip.nargs, Literal[2])
    assert ufunc.fk5hip.nargs == 2
    assert_type(ufunc.fk5hip.ntypes, Literal[1])
    assert ufunc.fk5hip.ntypes == 1
    assert_type(ufunc.fk5hip.signature, Literal["->(3, 3),(3)"])
    assert ufunc.fk5hip.signature == "->(3, 3),(3)"


def test_gufunc_nin1_nout1_attributes() -> None:
    assert_type(ufunc.ltp.identity, None)
    assert ufunc.ltp.identity is None
    assert_type(ufunc.ltpb.nin, Literal[1])
    assert ufunc.ltpb.nin == 1
    assert_type(ufunc.ltpecl.nout, Literal[1])
    assert ufunc.ltpecl.nout == 1
    assert_type(ufunc.ltpequ.nargs, Literal[2])
    assert ufunc.ltpequ.nargs == 2
    assert_type(ufunc.ltecm.ntypes, Literal[1])
    assert ufunc.ltecm.ntypes == 1
    assert_type(ufunc.cp.signature, Literal["(3)->(3)"])
    assert ufunc.cp.signature == "(3)->(3)"


def test_gufunc_nout1_attributes() -> None:
    assert_type(ufunc.ab.identity, None)
    assert ufunc.ab.identity is None
    assert_type(ufunc.apcg.nin, Literal[4])
    assert ufunc.apcg.nin == 4
    assert_type(ufunc.apci.nout, Literal[1])
    assert ufunc.apci.nout == 1
    assert_type(ufunc.apco.nargs, Literal[17])
    assert ufunc.apco.nargs == 17
    assert_type(ufunc.apcs.ntypes, Literal[0])
    assert ufunc.apcs.ntypes == 0
    assert_type(ufunc.ld.signature, Literal["(),(3),(3),(3),(),()->(3)"])
    assert ufunc.ld.signature == "(),(3),(3),(3),(),()->(3)"


def test_gufunc_attributes() -> None:
    assert_type(ufunc.atciqn.identity, None)
    assert ufunc.atciqn.identity is None
    assert_type(ufunc.aticqn.nin, Literal[4])
    assert ufunc.aticqn.nin == 4
    assert_type(ufunc.bp00.nout, Literal[3])
    assert ufunc.bp00.nout == 3
    assert_type(ufunc.bp06.nargs, Literal[5])
    assert ufunc.bp06.nargs == 5
    assert_type(ufunc.bpn2xy.ntypes, Literal[1])
    assert ufunc.bpn2xy.ntypes == 1
    assert_type(
        ufunc.pn00.signature,
        Literal["(),(),(),()->(),(3, 3),(3, 3),(3, 3),(3, 3),(3, 3)"],
    )
    assert ufunc.pn00.signature == "(),(),(),()->(),(3, 3),(3, 3),(3, 3),(3, 3),(3, 3)"
