# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Protocol, runtime_checkable

import numpy as np

# numpy includes public type aliases for arrays with known (small) ndim, starting from
# numpy 2.6. The definitions here allow the aliases to be used with numpy 2.5 or older.
type Array0D[ST: np.generic] = np.ndarray[tuple[()], np.dtype[ST]]
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]
type Array3D[ST: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ST]]


# A re-implementation of the private `numpy._typing._DTypeLike` from numpy 2.5.4
@runtime_checkable
class _HasDType[DTypeT: np.dtype](Protocol):
    @property
    def dtype(self) -> DTypeT: ...


@runtime_checkable
class _HasNumPyDType[DTypeT: np.dtype](Protocol):
    @property
    def __numpy_dtype__(self, /) -> DTypeT: ...


type _SupportsDType[DTypeT: np.dtype] = _HasDType[DTypeT] | _HasNumPyDType[DTypeT]
type _DTypeLike[ScalarT: np.generic] = (  # noqa: PYI047
    type[ScalarT] | np.dtype[ScalarT] | _SupportsDType[np.dtype[ScalarT]]
)
