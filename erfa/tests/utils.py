# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections.abc import Mapping
from typing import Final, Literal, Union, overload

import numpy as np
import pytest
from numpy.typing import DTypeLike

from erfa import ufunc
from erfa._typing import Array0D, Array1D, Array2D

type DTypes = (
    Literal["ufunc.LeapSecondDType", "ufunc.PVDType", "ufunc.SignDType"]
    | type[np.bytes_ | np.float64 | np.intc]
)

DTYPES: Final[Mapping[DTypes, DTypeLike]] = {
    "ufunc.LeapSecondDType": ufunc.dt_eraLEAPSECOND,
    "ufunc.PVDType": ufunc.dt_pv,
    "ufunc.SignDType": ufunc.dt_sign,
    np.bytes_: "S1",
    np.float64: np.float64,
    np.intc: np.intc,
}


def _assert_array_type(arr: np.ndarray, ndim: int, dtype: DTypeLike) -> None:
    __tracebackhide__ = True
    if type(arr) is not np.ndarray:
        pytest.fail(f"received an instance of {type(arr)}, expected an np.ndarray")
    if arr.ndim != ndim:
        pytest.fail(f"{arr.ndim = }, expected {ndim}")
    if arr.dtype != dtype:
        pytest.fail(f"{arr.dtype = }, expected {dtype}")


@overload
def assert_0d_array(
    arr: Array0D["ufunc.LeapSecondDType"], dtype: Literal["ufunc.LeapSecondDType"]
) -> None: ...
@overload
def assert_0d_array(
    arr: Array0D["ufunc.PVDType"], dtype: Literal["ufunc.PVDType"]
) -> None: ...
@overload
def assert_0d_array(
    arr: Array0D["ufunc.SignDType"], dtype: Literal["ufunc.SignDType"]
) -> None: ...
@overload
def assert_0d_array(arr: Array0D[np.bytes_], dtype: type[np.bytes_]) -> None: ...
@overload
def assert_0d_array(arr: Array0D[np.float64], dtype: type[np.float64]) -> None: ...
@overload
def assert_0d_array(arr: Array0D[np.intc], dtype: type[np.intc]) -> None: ...
def assert_0d_array(
    arr: Array0D["ufunc.LeapSecondDType"]
    | Array0D["ufunc.PVDType"]
    | Array0D["ufunc.SignDType"]
    | Array0D[np.bytes_]
    | Array0D[np.float64]
    | Array0D[np.intc],
    dtype: DTypes,
) -> None:
    __tracebackhide__ = True
    _assert_array_type(arr, 0, DTYPES[dtype])


@overload
def assert_1d_array(
    arr: Array1D["ufunc.LeapSecondDType"], dtype: Literal["ufunc.LeapSecondDType"]
) -> None: ...
@overload
def assert_1d_array(
    arr: Array1D["ufunc.PVDType"], dtype: Literal["ufunc.PVDType"]
) -> None: ...
@overload
def assert_1d_array(
    arr: Array1D["ufunc.SignDType"], dtype: Literal["ufunc.SignDType"]
) -> None: ...
@overload
def assert_1d_array(arr: Array1D[np.bytes_], dtype: type[np.bytes_]) -> None: ...
@overload
def assert_1d_array(arr: Array1D[np.float64], dtype: type[np.float64]) -> None: ...
@overload
def assert_1d_array(arr: Array1D[np.intc], dtype: type[np.intc]) -> None: ...
def assert_1d_array(
    arr: Array1D["ufunc.LeapSecondDType"]
    | Array1D["ufunc.PVDType"]
    | Array1D["ufunc.SignDType"]
    | Array1D[np.bytes_]
    | Array1D[np.float64]
    | Array1D[np.intc],
    dtype: DTypes,
) -> None:
    __tracebackhide__ = True
    _assert_array_type(arr, 1, DTYPES[dtype])


@overload
def assert_2d_array(
    arr: Array2D["ufunc.LeapSecondDType"], dtype: Literal["ufunc.LeapSecondDType"]
) -> None: ...
@overload
def assert_2d_array(
    arr: Array2D["ufunc.PVDType"], dtype: Literal["ufunc.PVDType"]
) -> None: ...
@overload
def assert_2d_array(
    arr: Array2D["ufunc.SignDType"], dtype: Literal["ufunc.SignDType"]
) -> None: ...
@overload
def assert_2d_array(arr: Array2D[np.bytes_], dtype: type[np.bytes_]) -> None: ...
@overload
def assert_2d_array(arr: Array2D[np.float64], dtype: type[np.float64]) -> None: ...
@overload
def assert_2d_array(arr: Array2D[np.intc], dtype: type[np.intc]) -> None: ...
def assert_2d_array(
    arr: Array2D["ufunc.LeapSecondDType"]
    | Array2D["ufunc.PVDType"]
    | Array2D["ufunc.SignDType"]
    | Array2D[np.bytes_]
    | Array2D[np.float64]
    | Array2D[np.intc],
    dtype: DTypes,
) -> None:
    __tracebackhide__ = True
    _assert_array_type(arr, 2, DTYPES[dtype])


@overload
def assert_scalar(
    scalar: "ufunc.LeapSecondDType", dtype: Literal["ufunc.LeapSecondDType"]
) -> None: ...
@overload
def assert_scalar(scalar: "ufunc.PVDType", dtype: Literal["ufunc.PVDType"]) -> None: ...
@overload
def assert_scalar(
    scalar: "ufunc.SignDType", dtype: Literal["ufunc.SignDType"]
) -> None: ...
@overload
def assert_scalar(scalar: np.bytes_, dtype: type[np.bytes_]) -> None: ...
@overload
def assert_scalar(scalar: np.float64, dtype: type[np.float64]) -> None: ...
@overload
def assert_scalar(scalar: np.intc, dtype: type[np.intc]) -> None: ...
def assert_scalar(
    scalar: Union[
        "ufunc.LeapSecondDType",
        "ufunc.PVDType",
        "ufunc.SignDType",
        np.bytes_,
        np.float64,
        np.intc,
    ],
    dtype: DTypes,
) -> None:
    __tracebackhide__ = True
    if not isinstance(scalar, np.generic):
        pytest.fail(f"{scalar = } is not a scalar")
    if scalar.dtype != DTYPES[dtype]:
        pytest.fail(f"{scalar.dtype = }, expected {dtype}")


def assert_none(none: None) -> None:
    __tracebackhide__ = True
    if none is not None:
        pytest.fail(f"{none!r} is not 'None'")
