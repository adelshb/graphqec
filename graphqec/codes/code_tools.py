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
    "commutation_test",
    "compare_rows",
    "row_sort",
    "leading_ones",
    "transpose_check",
    "row_extended_check",
]


def commutation_test(Hx: list[list[int]], Hz: list[list[int]]) -> bool:
    r"""
    function for taking two linear codes and determining if they satisfy
    the necessary constraints for defining a CSS code
    """

    assert len(Hx[0]) == len(Hz[0])

    return all(
        [(not sum([a * b for (a, b) in zip(hx, hz)]) % 2) for hx in Hx for hz in Hz]
    )


def compare_rows(row1: list[int], row2: list[int]) -> int:
    r"""
    A function that compares two rows based on the lowest unique non-zero element
    index.

    :param row1: binary list
    :param row2: binary list
    """

    assert len(row1) == len(row2)

    row1p2 = [(a + b) % 2 for (a, b) in zip(row1, row2)]

    if row1.index(1) == row1p2.index(1):
        return 0
    else:
        return 1


def row_sort(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function taking in a check matrix as a list of binary lists and outputs
    a sorted list of binary lists with the ordering determined by the
    compare_rows function.
    """

    num = len(check_matrix)
    if num < 2:
        return check_matrix
    else:
        first_half = row_sort(check_matrix[: int(num / 2)])
        second_half = row_sort(check_matrix[int(num / 2) :])
        sorted_full = []

        while len(first_half) > 0 and len(second_half) > 0:
            comp = compare_rows(first_half[0], second_half[0])
            if comp:
                sorted_full.append(second_half.pop(0))
            else:
                sorted_full.append(first_half.pop(0))

        return sorted_full + first_half + second_half


def leading_ones(check_matrix: list[list[int]]) -> dict[int, list[int]]:
    r"""
    Function taking in a check matrix as a list of binary lists and
    outputs a dictionary mapping each index to the list of rows for
    which that index is the leading one in the list.

    :param check_matrix: list of binary lists

    """

    leaders = {}

    for row_index in range(len(check_matrix)):
        if check_matrix[row_index].index(1) in leaders:
            leaders[check_matrix[row_index].index(1)].append(row_index)
        else:
            leaders[check_matrix[row_index].index(1)] = [row_index]

    return leaders


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
