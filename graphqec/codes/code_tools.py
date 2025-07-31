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

__all__ = [
    "add_rows",
    "commutation_test",
    "compute_kernel",
    "find_pivots",
    "prep_matrix",
    "row_extended_check",
    "transpose_check",
]


def add_rows(row1: list[int], row2: list[int]) -> list[int]:

    assert len(row1) == len(row2)

    row1p2 = [(a + b) % 2 for (a, b) in zip(row1, row2)]

    return row1p2


def commutation_test(Hx: list[list[int]], Hz: list[list[int]]) -> bool:
    r"""
    function for taking two linear codes and determining if they satisfy
    the necessary constraints for defining a CSS code
    """

    assert len(Hx[0]) == len(Hz[0])

    return all(
        [(not sum([a * b for (a, b) in zip(hx, hz)]) % 2) for hx in Hx for hz in Hz]
    )


def compute_kernel(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function that takes in a check matrix as a list of binary lists
    and outputs an (overcomplete) list of operators in the kernel
    given by their supports on qubits.

    :param check_matrix: list of binary lists
    """
    nrows = len(check_matrix)
    ncols = len(check_matrix[0])
    pmatrix = prep_matrix(check_matrix)

    for col1 in range(ncols):
        if 1 in pmatrix[col1]:
            mark = pmatrix[col1].index(1)
            for col2 in range(ncols):
                if col2 != col1 and pmatrix[col2][mark]:
                    pmatrix[col2] = add_rows(pmatrix[col1], pmatrix[col2])

    kern = []

    for row in pmatrix:
        if not sum(row[:nrows]):
            kern.append([row[ii] for ii in range(nrows, len(row))])

    return kern


def find_pivots(check_matrix: list[list[int]]) -> int:
    r"""
    Function that computes the rank of a binary matrix given as a
    list of binary lists.

    :param check_matrix: List of binary lists
    :return: Binary rank of check_matrix given as an integer
    """
    ncols = len(check_matrix[0])
    mtrans = transpose_check(check_matrix)

    pivots = []

    for col1 in range(ncols):
        if 1 in mtrans[col1]:
            mark = mtrans[col1].index(1)
            pivots.append(mark)
            for col2 in range(ncols):
                if col2 != col1 and mtrans[col2][mark]:
                    mtrans[col2] = add_rows(mtrans[col1], mtrans[col2])

    return pivots


def prep_matrix(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function that takes in a check matrix given as a list of
    binary lists and outputs an extended matrix intended
    for use when computing the kernel.
    """

    cmsextend = row_extended_check(check_matrix)  # extend with the identity
    hf = transpose_check(cmsextend)  # transpose the list indices

    return hf


def row_extended_check(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function taking in a check matrix as a list of binary lists and
    outputs the matrix extended by the identity matrix with the same
    number of columns.

    :param check_matrix: list of binary lists
    """

    ncols = len(check_matrix[0])

    ch_ext = []

    for ii in range(ncols):
        ch_ext.append([0] * ii + [1] + [0] * (ncols - ii - 1))

    return check_matrix + ch_ext


def transpose_check(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function taking in a check matrix as a list of binary lists and
    outputs the matrix transpose as a list of binary lists

    :param check_matrix: list of binary lists
    """

    nrows = len(check_matrix)
    ncols = len(check_matrix[0])

    check_trans = [
        [check_matrix[row][col] for row in range(nrows)] for col in range(ncols)
    ]

    return check_trans
