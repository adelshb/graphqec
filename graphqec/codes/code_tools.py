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
    "row_reduced_echelon_form",
    "dependency_check",
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


def dependency_check(ref_matrix: list[list[int]], check_row: list[int]) -> bool:
    r"""
    function taking in a rref matrix and a vector and tests if the vector can
    be written as a linear combination of the rows of the matrix.
    """

    pass


def row_reduced_echelon_form(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    function taking in a binary check matrix and returns an
    equivalent check matrix in row reduced echelon form, that is, with all
    redundancies eliminated.
    """

    pass


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

    # # row1t2 = [a*b for (a,b) in zip(row1,row2)]

    # if row1t2.count(1) == row1.count(1) and row1t2.count(1) == row2.count(1):
    #     return (0,0)
    # else:
    #     if row1.index(1) == row1p2.index(1):
    #         return (0,1)
    #     else:
    #         return (1,0)


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
            # if comp == (0,0):
            #     sorted_full.append(first_half.pop(0))
            #     del second_half[0]
            if comp:
                sorted_full.append(second_half.pop(0))
            else:
                sorted_full.append(first_half.pop(0))

        return sorted_full + first_half + second_half
