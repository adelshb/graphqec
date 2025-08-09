# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

__all__ = [
    "add_rows",
    "commutation_test",
    "compute_kernel",
    "find_pivots",
    "prep_matrix",
    "row_extended_check",
    "transpose_check",
]


def add_rows(row1: np.ndarray, row2: np.ndarray) -> np.ndarray:
    """
    Takes two binary arrays of equal length and returns their mod 2 sum.
    """
    assert row1.shape == row2.shape
    return (row1 + row2) % 2


def commutation_test(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    """
    Checks whether two binary matrices Hx and Hz commute (CSS constraint).
    """
    assert Hx.shape[1] == Hz.shape[1]
    prod = (Hx @ Hz.T) % 2
    return np.all(prod == 0)


def compute_kernel(check_matrix: np.ndarray) -> np.ndarray:
    """
    Computes an (overcomplete) basis for the kernel (nullspace) of a binary matrix.
    """
    nrows, ncols = check_matrix.shape
    pmatrix = prep_matrix(check_matrix)

    for col1 in range(ncols):
        col = pmatrix[col1]
        if 1 in col[:nrows]:
            mark = np.argmax(col[:nrows])
            for col2 in range(ncols):
                if col2 != col1 and pmatrix[col2][mark]:
                    pmatrix[col2] = add_rows(pmatrix[col2], pmatrix[col1])

    # Kernel is in the part of the rows where the first nrows elements are zero
    kernel = []
    for row in pmatrix:
        if np.sum(row[:nrows]) == 0:
            kernel.append(row[nrows:])

    return np.array(kernel, dtype=int)


def find_pivots(check_matrix: np.ndarray) -> list[int]:
    """
    Returns indices of pivot rows (linearly independent vectors) from a binary matrix.
    """
    ncols = check_matrix.shape[1]
    mtrans = transpose_check(check_matrix)
    pivots = []

    for col1 in range(ncols):
        col = mtrans[col1]
        if 1 in col:
            mark = np.argmax(col)
            pivots.append(mark)
            for col2 in range(ncols):
                if col2 != col1 and mtrans[col2][mark]:
                    mtrans[col2] = add_rows(mtrans[col2], mtrans[col1])

    return pivots


def prep_matrix(check_matrix: np.ndarray) -> np.ndarray:
    """
    Prepares an augmented matrix by appending identity and then transposing.
    """
    cmsextend = row_extended_check(check_matrix)
    return transpose_check(cmsextend)


def row_extended_check(check_matrix: np.ndarray) -> np.ndarray:
    """
    Extends the check matrix with an identity matrix below it.
    """
    ncols = check_matrix.shape[1]
    identity = np.eye(ncols, dtype=int)
    return np.vstack([check_matrix, identity])


def transpose_check(check_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the transpose of a binary matrix.
    """
    return check_matrix.T.copy()
