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

__all__ = ["commutation_test", "row_sort", "row_echelon_form", "dependency_check"]


def commutation_test(Hx: list[list[int]], Hz: list[list[int]]) -> bool:
    r"""
    function for taking two linear codes and determining if they satisfy
    the necessary constraints for defining a CSS code
    """

    assert len(Hx[0]) == len(Hz[0])

    return all(
        [(not sum([a * b for (a, b) in zip(hx, hz)]) % 2) for hx in Hx for hz in Hz]
    )


def row_sort(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    A function sorting a list of check elements based on the
    minimum column index on which the checks act. This is
    done recursively with a merge-sort algorithm.
    """

    if len(check_matrix) == 1:
        # return [list(check_matrix[0])]
        return [check_matrix[0]]

    elif len(check_matrix) == 2:
        lr_sum = [not (a + b) % 2 for (a, b) in zip(check_matrix[0], check_matrix[1])]
        if all(lr_sum):
            return [check_matrix[0], check_matrix[1]]
            # return [list(check_matrix[0]),list(check_matrix[1])]
        else:
            first_one = lr_sum.index(False)
            if check_matrix[0][first_one]:
                return [check_matrix[0], check_matrix[1]]
                # return [list(check_matrix[0]),list(check_matrix[1])]
            else:
                return [check_matrix[1], check_matrix[0]]
                # return [list(check_matrix[1]),list(check_matrix[0])]
    else:
        print("length of list = " + str(len(check_matrix)))
        print(check_matrix[: int(len(check_matrix) / 2)])

        left_half = row_sort(check_matrix[: int((len(check_matrix) / 2)) + 1])
        right_half = row_sort(check_matrix[int((len(check_matrix) / 2)) + 1 :])

        row_sorted = []

        while len(left_half) > 0 and len(right_half) > 0:
            print(len(left_half), len(right_half))
            left_elem = left_half[0]
            righ_elem = right_half[0]
            # form the sum of the two check rows
            lr_sum = [a == b for (a, b) in zip(left_elem, righ_elem)]
            # identify the first index where the two elements disagree
            if all(lr_sum):
                return row_sorted.extend([left_half.pop(0), right_half.pop(0)])
            else:
                first_one = lr_sum.index(False)

                if left_elem[first_one]:
                    row_sorted.append(left_half.pop(0))
                else:
                    row_sorted.append(right_half.pop(0))

        return row_sorted.extend(right_half + left_half)


def row_echelon_form(check_matrix: list[list[int]]) -> list[list[int]]:
    r"""
    function taking in a binary check matrix and returns an
    equivalent check matrix in row reduced echelon form.
    """
    # zero_col = list(check_matrix[:][0])
    # pivot_row = zero_col.index(1)

    if len(check_matrix) == 1:
        return check_matrix
    else:
        recm = row_echelon_form(check_matrix[:-1])
        next_row = check_matrix[-1]
        if dependency_check(recm, next_row):
            return recm
        else:
            return recm.append(next_row)


def dependency_check(ref_matrix: list[list[int]], check_row: list[int]) -> bool:
    r"""
    function taking in a rref matrix and a vector and tests if the vector can
    be written as a linear combination of the rows of the matrix.
    """

    if len(ref_matrix) == 1:
        return check_row in ref_matrix

    else:

        return dependency_check(ref_matrix[:-1], check_row)
