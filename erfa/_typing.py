# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

# numpy includes public type aliases for arrays with known (small) ndim, starting from
# numpy 2.6. The definitions here allow the aliases to be used with numpy 2.5 or older.
type Array0D[ST: np.generic] = np.ndarray[tuple[()], np.dtype[ST]]
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]
type Array3D[ST: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ST]]
