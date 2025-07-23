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
    "row_sort_no_dupes",
    "row_reduced_echelon_form",
    "dependency_check",
    "row_order",
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


#     proto_ref_matrix = row_sort_no_dupes(check_matrix)
#     rindex = 0
#     startRow = proto_ref_matrix[rindex]
#     col_marker = startRow.index(1)

#     ref_matrix = [startRow]

#     while rindex < len(check_matrix):

#         while proto_ref_matrix[rindex][col_marker]:
#             row = [(a + b) % 2 for (a, b) in zip(proto_ref_matrix[rindex], startRow)]

#             if sum(row):
#                 ref_matrix.append(row)

#             rindex += 1

#         proto_ref_matrix_update = row_sort_no_dupes(check_matrix[rindex:])
#         startRow_update = proto_ref_matrix_update[0]
#         col_marker_update = startRow_update.index(1)


def row_sort_no_dupes(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    Function to sort and remove duplicate rows (but does not remove redundancies)
    """

    num = len(check_matrix)
    if num < 2:
        return check_matrix
    else:
        sorted_full = []

        halfs = [
            row_sort_no_dupes(check_matrix[: int(num / 2)]),
            row_sort_no_dupes(check_matrix[int(num / 2) :]),
        ]

        while min(len(halfs[0]), len(halfs[1])) > 0:
            if not sum([(a + b) % 2 for (a, b) in zip(halfs[0][0], halfs[1][0])]):
                sorted_full.append(halfs[0][0].pop(0))
                del halfs[1][0]
            else:
                firstRow = row_order(halfs[0][0], halfs[1][0])
                sorted_full.append(halfs[firstRow][0].pop(0))

        return sorted_full + halfs[0] + halfs[1]


def row_order(row1: list[int], row2: list[int]) -> int:
    r"""
    An ordering of two binary vectors based on the minimum unique instance of a 1 value.

    :input row1: list of 0's and 1's
    :input row2: list of 0's and 1's

    :output: binary value indicating whether row1 (0) or row2 (1) comes first
        in the ordering
    """

    assert len(row1) == len(row2)

    row1p2 = [(a + b) % 2 for (a, b) in zip(row1, row2)]
    assert sum(row1p2)

    firstOne = row1p2.index(1)

    return int(not row1[firstOne])
