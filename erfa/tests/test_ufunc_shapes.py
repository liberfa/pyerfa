# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from erfa import ufunc

from .utils import (
    assert_0d_array,
    assert_0d_array_is_scalar_at_runtime,
    assert_1d_array,
    assert_2d_array,
    assert_scalar,
)


def test_ufunc_nin1_nout1_scalar() -> None:
    assert_scalar(ufunc.fal03(0.8), np.float64)


def test_ufunc_nin1_nout1_0d_array() -> None:
    assert_0d_array_is_scalar_at_runtime(ufunc.fal03(np.float32(0.8)[...]), np.float64)


def test_ufunc_nin1_nout1_array_like() -> None:
    assert_1d_array(ufunc.falp03([0.8]), np.float64)


def test_ufunc_nin1_nout1_array() -> None:
    assert_1d_array(ufunc.falp03(np.ones(1)), np.float64)


def test_ufunc_nin1_nout1_scalar_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(ufunc.fane03(1, out=...), np.float64)


def test_ufunc_nin1_nout1_array_like_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_1d_array(ufunc.cpv([ufunc.zpv()], out=...), "ufunc.PVDType")


def test_ufunc_nin1_nout1_array_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(ufunc.cpv(ufunc.zpv(out=...), out=...), "ufunc.PVDType")


def test_ufunc_nin1_nout1_positional_out_array() -> None:
    assert_1d_array(ufunc.faom03(np.float16(0.8), np.zeros(2)), np.float64)


def test_ufunc_nin1_nout1_keyword_out_array() -> None:
    assert_2d_array(ufunc.fapa03(np.array([0.8]), out=np.empty((2, 1))), np.float64)


def test_ufunc_nin2_nout1_scalars() -> None:
    assert_scalar(ufunc.sxpv(0.8, ufunc.zpv()), "ufunc.PVDType")


def test_ufunc_nin2_nout1_0d_array() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array_is_scalar_at_runtime(
        ufunc.pvxpv(ufunc.zpv(out=...), ufunc.zpv()), "ufunc.PVDType"
    )


def test_ufunc_nin2_nout1_array() -> None:
    assert_1d_array(ufunc.pvu(2, [ufunc.zpv()]), "ufunc.PVDType")
    assert_2d_array(
        ufunc.pvppv((ufunc.zpv(),), ufunc.zpv()[None, None]), "ufunc.PVDType"
    )


def test_ufunc_nin2_nout1_scalar_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(ufunc.pvmpv(ufunc.zpv(), ufunc.zpv(), out=...), "ufunc.PVDType")


def test_ufunc_nin2_nout1_array_out_ellipsis() -> None:
    pytest.importorskip(
        "numpy", minversion="2.3", reason="out=Ellipsis was implemented in numpy 2.3"
    )
    assert_0d_array(
        ufunc.pvdpv(ufunc.zpv(out=...), ufunc.zpv(), out=...), "ufunc.PVDPVDType"
    )
    assert_1d_array(
        ufunc.pvdpv(ufunc.zpv(), [ufunc.zpv()], out=...), "ufunc.PVDPVDType"
    )


def test_ufunc_nin2_nout1_positional_out_array() -> None:
    assert_1d_array(ufunc.gst94(np.float16(0.8), 0.8, np.zeros(2)), np.float64)


def test_ufunc_nin2_nout1_keyword_out_array() -> None:
    assert_2d_array(
        ufunc.gst00b(np.float32(0.8), np.array([0.8]), out=np.empty((2, 1))), np.float64
    )
