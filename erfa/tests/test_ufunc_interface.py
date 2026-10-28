# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from erfa import ufunc


def test_erfa_parameters_are_positional_only() -> None:
    with pytest.raises(
        TypeError, match="takes from 2 to 3 positional arguments but 0 were given"
    ):
        ufunc.gmst82(dj1=2400000.5, dj2=53736.0)  # type: ignore[call-overload]


def test_positional_out_none_ufunc_nin0() -> None:
    result = ufunc.bi00()
    result_with_out = ufunc.bi00(None)
    assert_array_equal(result_with_out[0], result[0])
    assert_array_equal(result_with_out[1], result[1])
    assert_array_equal(result_with_out[2], result[2])


def test_positional_out_none_ufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.zpv(None), ufunc.zpv())


def test_positional_out_none_ufunc_nin1_nout1() -> None:
    t = 0.7
    assert_array_equal(ufunc.faom03(t, None), ufunc.faom03(t))


def test_positional_out_none_ufunc_nin2_nout1() -> None:
    args = (2415019.8135, 30103.18648)
    assert_array_equal(ufunc.epb(*args, None), ufunc.epb(*args))


def test_positional_out_none_ufunc_nout1() -> None:
    args = (2400000.5, 53736.0, 0.4090789763356509900, -0.9630909107115582393e-5)
    assert_array_equal(ufunc.ee00(*args, None), ufunc.ee00(*args))


def test_positional_out_none_ufunc_nout_many() -> None:
    args = (800.0, 10.0, 0.9, 0.4)
    result = ufunc.refco(*args)
    result_with_out = ufunc.refco(*args, None)
    assert_array_equal(result_with_out[0], result[0])
    assert_array_equal(result_with_out[1], result[1])


def test_positional_out_none_gufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.zr(None), ufunc.zr())


def test_positional_out_none_gufunc_nin1_nout1() -> None:
    r = np.arange(9).reshape((3, 3))
    assert_array_equal(ufunc.cr(r, None), ufunc.cr(r))


def test_keyword_out_none_ufunc_nin0() -> None:
    with pytest.raises(TypeError, match="'out' must be a tuple of arrays"):
        ufunc.bi00(out=None)  # type: ignore[call-overload]


def test_keyword_out_none_ufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.zpv(out=None), ufunc.zpv())


def test_keyword_out_none_ufunc_nin1_nout1() -> None:
    t = 0.7
    assert_array_equal(ufunc.faf03(t, out=None), ufunc.faf03(t))


def test_keyword_out_none_ufunc_nin2_nout1() -> None:
    args = (2415019.8135, 30103.18648)
    assert_array_equal(ufunc.epj(*args, out=None), ufunc.epj(*args))


def test_keyword_out_none_ufunc_nout1() -> None:
    args = (2400000.5, 53736.0, 2400000.5, 53736.0)
    assert_array_equal(ufunc.gmst00(*args, out=None), ufunc.gmst00(*args))


def test_keyword_out_none_ufunc_nout_many() -> None:
    with pytest.raises(TypeError, match="'out' must be a tuple of arrays"):
        ufunc.starpm(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, out=None)  # type: ignore[call-overload]


def test_keyword_out_none_gufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.zp(out=None), ufunc.zp())


def test_keyword_out_none_gufunc_nin1_nout1() -> None:
    p = np.array([0.2, 1.25, 3.0])
    assert_array_equal(ufunc.p2pv(p, out=None), ufunc.p2pv(p))


def test_positional_out_arr_ufunc_nin0() -> None:
    result = ufunc.bi00()
    out = (np.zeros(2), np.zeros(2), np.zeros(2))
    result_with_1_out = ufunc.bi00(out[0])
    assert result_with_1_out[0] is out[0]
    assert_array_equal(out[0], result[0])
    result_with_2_out = ufunc.bi00(out[0], out[1])
    assert result_with_2_out[0] is out[0]
    assert result_with_2_out[1] is out[1]
    assert_array_equal(out[0], result[0])
    assert_array_equal(out[1], result[1])
    result_with_3_out = ufunc.bi00(*out)
    assert result_with_3_out[0] is out[0]
    assert result_with_3_out[1] is out[1]
    assert result_with_3_out[2] is out[2]
    assert_array_equal(out[0], result[0])
    assert_array_equal(out[1], result[1])
    assert_array_equal(out[2], result[2])


def test_positional_out_arr_ufunc_nin0_nout1() -> None:
    result = ufunc.zpv()
    out = np.ones_like(result)
    result_with_out = ufunc.zpv(out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_arr_ufunc_nin1_nout1() -> None:
    a = -4.0
    result = ufunc.anpm(a)
    out = np.zeros_like(result)
    result_with_out = ufunc.anpm(a, out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_arr_ufunc_nin2_nout1() -> None:
    args = (2400000.5, 54388.0)
    result = ufunc.obl80(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.obl80(*args, out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_arr_ufunc_nout1() -> None:
    args = (2400000.5, 53736.0, 0.5791308486706011000e-3, 0.4020579816732961219e-4)
    result = ufunc.s06(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.s06(*args, out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_arr_ufunc_nout_many() -> None:
    args = (5.5850536063818546461558105, -0.7853981633974483096156608)
    result = ufunc.g2icrs(*args)
    out = (np.zeros(2), np.zeros(2))
    result_with_1_out = ufunc.g2icrs(*args, out[0])
    assert result_with_1_out[0] is out[0]
    assert_array_equal(out[0], result[0])
    result_with_2_out = ufunc.g2icrs(*args, *out)
    assert result_with_2_out[0] is out[0]
    assert result_with_2_out[1] is out[1]
    assert_array_equal(out[0], result[0])
    assert_array_equal(out[1], result[1])


def test_positional_out_arr_gufunc_nin0_nout1() -> None:
    result = ufunc.ir()
    out = np.zeros_like(result)
    result_with_out = ufunc.ir(out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_arr_gufunc_nin1_nout1() -> None:
    pv = np.void(None, ufunc.dt_pv)
    pv["p"] = [0.2, 1.3, -2.5]
    pv["v"] = [-0.4, 3.2, 0.8]
    result = ufunc.pv2p(pv)
    out = np.zeros_like(result)
    result_with_out = ufunc.pv2p(pv, out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_ufunc_nin0() -> None:
    with pytest.raises(TypeError, match="'out' must be a tuple of arrays"):
        ufunc.bi00(out=np.empty(1))  # type: ignore[call-overload]


def test_keyword_out_arr_ufunc_nin0_nout1() -> None:
    result = ufunc.zpv()
    out = np.ones_like(result)
    result_with_out = ufunc.zpv(out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_ufunc_nin1_nout1() -> None:
    a = -4.0
    result = ufunc.anp(a)
    out = np.zeros_like(result)
    result_with_out = ufunc.anp(a, out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_ufunc_nin2_nout1() -> None:
    args = (2400000.5, 54388.0)
    result = ufunc.obl06(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.obl06(*args, out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_ufunc_nout1() -> None:
    args = (2400000.5, 53736.0, 0.5791308486706011000e-3, 0.4020579816732961219e-4)
    result = ufunc.s00(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.s00(*args, out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_ufunc_nout_many() -> None:
    with pytest.raises(TypeError, match="'out' must be a tuple of arrays"):
        ufunc.icrs2g(5.585053606, -0.78539816, out=np.empty((2, 3)))  # type: ignore[call-overload]


def test_keyword_out_arr_gufunc_nin0_nout1() -> None:
    result = ufunc.zr()
    out = np.ones_like(result)
    result_with_out = ufunc.zr(out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_arr_gufunc_nin1_nout1() -> None:
    r = np.arange(0, 18, 2).reshape((3, 3))
    result = ufunc.tr(r)
    out = np.ones_like(result)
    result_with_out = ufunc.tr(r, out=out)
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_tuple_ufunc_nin0() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.bi00((np.empty(2), np.empty(2), np.empty(2)))  # type: ignore[call-overload]


def test_positional_out_tuple_ufunc_nin0_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.zpv((np.empty(2),))  # type: ignore[call-overload]


def test_positional_out_tuple_ufunc_nin1_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.falp03(0, (np.empty(2),))  # type: ignore[call-overload]


def test_positional_out_tuple_ufunc_nin2_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.ee00b(0, 0, (np.empty(2),))  # type: ignore[call-overload]


def test_positional_out_tuple_ufunc_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.dtdb(0, 0, 0, 0, 0, 0, (np.empty(2),))  # type: ignore[call-overload]


def test_positional_out_tuple_ufunc_nout_many() -> None:
    out = (np.empty(2), np.empty(2), np.empty(2), np.empty(2), np.empty(2), np.empty(2))
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.fk425(0, 0, 0, 0, 0, 0, out)  # type: ignore[call-overload]


def test_positional_out_tuple_gufunc_nin0_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.zr((np.empty(2),))  # type: ignore[call-overload]


def test_positional_out_tuple_gufunc_nin1_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.rm2v(np.zeros((3, 3)), (np.empty(2),))  # type: ignore[call-overload]


def test_keyword_out_tuple_of_arr_ufunc_nin0() -> None:
    result = ufunc.bi00()
    out = (np.zeros(2), np.zeros(2), np.zeros(2))
    result_with_3_out = ufunc.bi00(out=out)
    assert result_with_3_out[0] is out[0]
    assert result_with_3_out[1] is out[1]
    assert result_with_3_out[2] is out[2]
    assert_array_equal(out[0], result[0])
    assert_array_equal(out[1], result[1])
    assert_array_equal(out[2], result[2])


def test_keyword_out_tuple_of_arr_ufunc_nin0_nout1() -> None:
    result = ufunc.zpv()
    out = np.ones_like(result)
    result_with_out = ufunc.zpv(out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_tuple_of_arr_ufunc_nin1_nout1() -> None:
    t = 0.8
    result = ufunc.fama03(t)
    out = np.zeros_like(result)
    result_with_out = ufunc.fama03(t, out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_tuple_of_arr_ufunc_nin2_nout1() -> None:
    args = (2400000.5, 54388.0)
    result = ufunc.eect00(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.eect00(*args, out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_tuple_of_arr_ufunc_nout1() -> None:
    args = (1.1, 1.2, 0.3)
    result = ufunc.hd2pa(*args)
    out = np.zeros_like(result)
    result_with_out = ufunc.hd2pa(*args, out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_tuple_of_arr_ufunc_nout_many() -> None:
    args = (2453750.5, 0.892482639)
    result = ufunc.tttai(*args)
    out = (np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int32))
    result_with_out = ufunc.tttai(*args, out=out)
    assert result_with_out[0] is out[0]
    assert result_with_out[1] is out[1]
    assert result_with_out[2] is out[2]
    assert_array_equal(out[0], result[0])
    assert_array_equal(out[1], result[1])
    assert_array_equal(out[2], result[2])


def test_keyword_out_tuple_of_arr_gufunc_nin0_nout1() -> None:
    result = ufunc.zp()
    out = np.ones_like(result)
    result_with_out = ufunc.zp(out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_keyword_out_tuple_of_arr_gufunc_nin1_nout1() -> None:
    w = np.array([0.0, 1.41371669, -1.88495559])
    result = ufunc.rv2m(w)
    out = np.ones_like(result)
    result_with_out = ufunc.rv2m(w, out=(out,))
    assert result_with_out is out
    assert_array_equal(out, result)


def test_positional_out_tuple_with_none_ufunc_nin0() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.bi00((None, np.empty(2), None))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_ufunc_nin0_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.zpv((None,))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_ufunc_nin1_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.fal03(0, (None,))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_ufunc_nin2_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.ee06a(0, 0, (None,))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_ufunc_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.gst00a(0, 0, 0, 0, (None,))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_ufunc_nout_many() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.fk524(  # type: ignore[call-overload]
            0, 0, 0, 0, 0, 0, (None, None, np.empty(3), None, np.empty(3), None)
        )


def test_positional_out_tuple_with_none_gufunc_nin0_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.zr((None,))  # type: ignore[call-overload]


def test_positional_out_tuple_with_none_gufunc_nin1_nout1() -> None:
    with pytest.raises(TypeError, match="return arrays must be of ArrayType"):
        ufunc.pm(np.ones((3,)), (None,))  # type: ignore[call-overload]


def test_keyword_out_tuple_with_none_ufunc_nin0() -> None:
    result = ufunc.bi00()
    out = (np.zeros(2), np.zeros(2), None)
    result_with_out = ufunc.bi00(out=out)
    assert result_with_out[0] is out[0]
    assert_array_equal(out[0], result[0])
    assert result_with_out[1] is out[1]
    assert_array_equal(out[1], result[1])
    assert_array_equal(result_with_out[2], result[2])


def test_keyword_out_tuple_with_none_ufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.zpv(out=(None,)), ufunc.zpv())


def test_keyword_out_tuple_with_none_ufunc_nin1_nout1() -> None:
    t = 0.8
    assert_array_equal(ufunc.fasa03(t, out=(None,)), ufunc.fasa03(t))


def test_keyword_out_tuple_with_none_ufunc_nin2_nout1() -> None:
    args = (2400000.5, 54388.0)
    assert_array_equal(ufunc.eqeq94(*args, out=(None,)), ufunc.eqeq94(*args))


def test_keyword_out_tuple_with_none_ufunc_nout1() -> None:
    args = (1.0, 0.1, 0.2, -1.0)
    assert_array_equal(ufunc.pas(*args, out=(None,)), ufunc.pas(*args))


def test_keyword_out_tuple_with_none_ufunc_nout_many() -> None:
    args = (2453750.5, 0.892482639)
    result = ufunc.taitt(*args)
    out = (np.zeros(2), None, np.zeros(2, dtype=np.int32))
    result_with_out = ufunc.taitt(*args, out=out)
    assert result_with_out[0] is out[0]
    assert_array_equal(out[0], result[0])
    assert_array_equal(result_with_out[1], result[1])
    assert result_with_out[2] is out[2]
    assert_array_equal(out[2], result[2])


def test_keyword_out_tuple_with_none_gufunc_nin0_nout1() -> None:
    assert_array_equal(ufunc.ir(out=(None,)), ufunc.ir())


def test_keyword_out_tuple_with_none_gufunc_nin1_nout1() -> None:
    epj = -1500.0
    assert_array_equal(ufunc.ltpecl(epj, out=(None,)), ufunc.ltpecl(epj))


def test_out_ellipsis_ufunc_nin0() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    result = ufunc.bi00()
    result_with_out = ufunc.bi00(out=...)
    assert type(result_with_out[0]) is np.ndarray
    assert_array_equal(result_with_out[0], result[0])
    assert type(result_with_out[1]) is np.ndarray
    assert_array_equal(result_with_out[1], result[1])
    assert type(result_with_out[2]) is np.ndarray
    assert_array_equal(result_with_out[2], result[2])


def test_out_ellipsis_ufunc_nin0_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    result_with_out = ufunc.zpv(out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.zpv())


def test_out_ellipsis_ufunc_nin1_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    t = 0.8
    result_with_out = ufunc.fave03(t, out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.fave03(t))


def test_out_ellipsis_ufunc_nin2_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    args = (2400000.5, 54388.0)
    result_with_out = ufunc.gst00b(*args, out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.gst00b(*args))


def test_out_ellipsis_ufunc_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    args = (1.0, 0.1, 0.2, -1.0)
    result_with_out = ufunc.seps(*args, out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.seps(*args))


def test_out_ellipsis_ufunc_nout_many() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    args = (2400000.5, 50123.9999)
    result = ufunc.pfw06(*args)
    result_with_out = ufunc.pfw06(*args, out=...)
    assert type(result_with_out[0]) is np.ndarray
    assert_array_equal(result_with_out[0], result[0])
    assert type(result_with_out[1]) is np.ndarray
    assert_array_equal(result_with_out[1], result[1])
    assert type(result_with_out[2]) is np.ndarray
    assert_array_equal(result_with_out[2], result[2])
    assert type(result_with_out[3]) is np.ndarray
    assert_array_equal(result_with_out[3], result[3])


def test_out_ellipsis_gufunc_nin0_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    result_with_out = ufunc.zp(out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.zp())


def test_out_ellipsis_gufunc_nin1_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    epj = 1666.666
    result_with_out = ufunc.ltp(epj, out=...)
    assert type(result_with_out) is np.ndarray
    assert_array_equal(result_with_out, ufunc.ltp(epj))


def test_keyword_out_ellipsis_ufunc_nin0() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.bi00(...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_ufunc_nin0_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )

    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.zpv(...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_ufunc_nin1_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.fad03(0.7, ...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_ufunc_nin2_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.gst94(2400000.5, 54388.0, ...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_ufunc_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.seps(1.0, 0.1, 0.2, -1.0, ...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_ufunc_nout_many() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.pfw06(2400000.5, 50123.9999, ...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_gufunc_nin0_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.zp(...)  # type: ignore[call-overload]


def test_keyword_out_ellipsis_gufunc_nin1_nout1() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    with pytest.raises(
        TypeError, match=r"out=... is only allowed as a keyword argument."
    ):
        ufunc.ltpb(1654.0, ...)  # type: ignore[call-overload]


def test_non_contiguous_output_matrix() -> None:
    # Fix for copy_from_double33 problem found by @devdanzin
    # Create non-contiguous output array (only reachable via ufunc interface).
    result = np.zeros((3, 4))[:, :3]
    out = ufunc.ltecm(2005.0, out=result)
    assert out is result
    expected = ufunc.ltecm(2005.0)
    assert_array_equal(expected, result)


def test_at_unary() -> None:
    t = np.linspace(0, 1, 5)
    ufunc.faur03.at(t, [0, 4])
    assert t == pytest.approx([5.48129387, 0.25, 0.5, 0.75, 0.39308311])


def test_at_binary() -> None:
    s = np.array([2400000.5])
    ufunc.s00b.at(s, True, 52541.0)
    assert s[0] == pytest.approx(-0.1340695782951026584e-7, abs=1e-18)


def test_accumulate() -> None:
    dj1, dj2 = 2450123, 0.1
    era00 = ufunc.era00.accumulate([dj1, dj2])
    assert era00[0] == pytest.approx(dj1)
    assert era00[1] == pytest.approx(ufunc.era00(dj1, dj2))


def test_reduce() -> None:
    eect = ufunc.eect00.reduce([[2400000.5, 2400001], [0.5, 0]])
    assert eect.shape == (2,)
    assert eect == pytest.approx(-5.652677543542872e-09)


def test_reduceat() -> None:
    eqeq = ufunc.eqeq94.reduceat((2400000.5, 41234.0, 0), [0, 2])
    assert eqeq.shape == (2,)
    assert eqeq[0] == pytest.approx(0.5357758254609256894e-4, abs=1e-17)
    assert eqeq[1] == 0


def test_outer() -> None:
    spv = ufunc.sxpv.outer([5], [ufunc.zpv()])
    assert spv.shape == (1, 1)
    assert spv[0, 0] == ufunc.zpv()
