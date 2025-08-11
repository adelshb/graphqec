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

from graphqec.codes.base_code import BaseCode
from graphqec.math import commutation_test, compute_kernel, find_pivots

__all__ = ["CssCode"]


class CssCode(BaseCode):
    r"""
    A class for a generic CSS code specified by a pair of linear codes subject
    to a constraint to ensure commutativity of the generators in the
    associated quantum code.
    """

    __slots__ = (
        "_Hx",
        "_Hz",
        "_L_z",
        "_L_x",
    )

    def __init__(
        self,
        Hx: list[list[int]],
        Hz: list[list[int]],
        name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the CSS code instance.
        """

        if not commutation_test(Hz, Hx):
            raise ValueError(
                "The input parity check operators do not commute with each other."
            )

        self._Hx = Hx
        self._Hz = Hz
        self._num_data_qubits = len(Hz[0])

        self._L_z = self.compute_logicals(logical="Z")
        self._L_x = self.compute_logicals(logical="X")

        self._num_logical_qubits = len(self._L_z)

        self._logic_check = {
            "Z": [np.where(row == 1)[0].tolist() for row in self.L_z],
            "X": [np.where(row == 1)[0].tolist() for row in self.L_x],
        }

        super().__init__(*args, **kwargs)

        self._distance = self.distance_upper_bound()
        if name is not None:
            self._name = f"{name} [[{self.num_data_qubits},{self.num_logical_qubits},{self.distance}]]"
        else:
            self._name = f"CSS [[{self.num_data_qubits},{self.num_logical_qubits},{self.distance}]]"

    @property
    def Hx(self) -> np.ndarray:
        r"""
        The X check matrix of the CSS code.
        """
        return self._Hx

    @property
    def Hz(self) -> np.ndarray:
        r"""
        The Z check matrix of the CSS code.
        """
        return self._Hz

    @property
    def L_z(self) -> np.ndarray:
        r"""
        The logical operators for Z.
        """
        return self._L_z

    @property
    def L_x(self) -> np.ndarray:
        r"""
        The logical operators for X.
        """
        return self._L_x

    def distance_upper_bound(self) -> int:
        r"""
        Compute the distance upper bound of the CSS code.
        The distance is defined as the minimum of the distances of the X and Z.
        """

        xdistance = min([sum(row) for row in self.L_x])
        zdistance = min([sum(row) for row in self.L_z])

        return min([xdistance, zdistance])

    def build_graph(self) -> None:
        r"""
        Build the Tanner graph for the CSS code.
        Qubit nodes labels: 0, 1,..., num_data_qubits-1
        Xcheck nodes labels: num_data_qubits, num_data_qubits+1,...,num_data_qubits+nxchecks-1
        Zcheck nodes labels: num_data_qubits+nxchecks,...,num_data_qubits+nxchecks+nzchecks-1

        Edges are added based on the X and Z check matrices.
        """

        self._graph.add_nodes_from(
            [
                (i, {"type": "data", "label": None, "label_id": i})
                for i in range(self.num_data_qubits)
            ]
        )
        self._graph.add_nodes_from(
            [
                (
                    i,
                    {"type": "check", "label": "X", "label_id": i},
                )
                for i in range(
                    self.num_data_qubits, self.num_data_qubits + len(self.Hx)
                )
            ]
        )
        self._graph.add_nodes_from(
            [
                (
                    i,
                    {"type": "check", "label": "Z", "label_id": i},
                )
                for i in range(
                    self.num_data_qubits + len(self.Hx),
                    self.num_data_qubits + len(self.Hx) + len(self.Hz),
                )
            ]
        )

        self._graph.add_weighted_edges_from(self.greedy_scheduler())

    def compute_logicals(self, logical: str = "Z") -> np.ndarray:
        r"""
        Function for computing a set of logical operators for the input
        parity check operators.
        Find the image and kernel for each linear code.

        :param logical: The type of logical operator to compute, either "X" or "Z".
        :return: A list of logical operators.
        """

        if logical == "Z":
            H1 = self.Hz
            H2 = self.Hx
        else:
            H1 = self.Hx
            H2 = self.Hz

        # First find the kernel of H2
        Ker_H2 = compute_kernel(H2)

        # Find elements of Ker(H2) that are independent of Im(H1)
        stack = np.vstack([H1, Ker_H2])
        pivots = find_pivots(stack)

        # Only choose rows introduced by Ker(H2)
        L = np.array([stack[ii, :] for ii in pivots if ii >= len(H1)])

        return L

    def greedy_scheduler(self) -> list[tuple[int, int, int]]:
        r"""
        Function returning a dictionary that specifies the schedule of CNOTs
        for the stabilizer measurement circuit.

        Schedule to constitute a full circuit for the syndrome extraction,
        specified by a coloration of the edges of the Tanner graph of the
        code.
        Using the structure of the Tanner graph, we focus on mapping
        qubits to the order of labelings for the checks.
        Default scheduler will measure all X stabilizers first
        then all Z stabilizers, using a greedy algorithm.
        """

        xboard = np.array(self.Hx, dtype=int)
        zboard = np.array(self.Hz, dtype=int)

        for xrow in range(len(xboard)):

            xcheck = xboard[xrow]

            for qcol in range(self.num_data_qubits):

                if xcheck[qcol]:
                    if not xrow:
                        current_qubit_slot = 0
                    else:
                        current_qubit_slot = max(xboard[:xrow, qcol])
                    if not qcol:
                        current_check_slot = 0
                    else:
                        current_check_slot = max(xboard[xrow, :qcol])

                    xboard[xrow][qcol] = (
                        max([current_check_slot, current_qubit_slot]) + 1
                    )

        for zrow in range(len(zboard)):

            zcheck = zboard[zrow]

            for qcol in range(self.num_data_qubits):

                if zcheck[qcol]:
                    if not zrow:
                        current_qubit_slot = max(xboard[:, qcol])
                    else:
                        current_qubit_slot = max(
                            max(xboard[:, qcol]), max(zboard[:zrow, qcol])
                        )
                    if not qcol:
                        current_check_slot = 0
                    else:
                        current_check_slot = max(zboard[zrow, :qcol])

                    zboard[zrow][qcol] = max(current_check_slot, current_qubit_slot) + 1

        schedule = [
            (ii, jj + self.num_data_qubits, xboard[jj][ii])
            for ii in range(self.num_data_qubits)
            for jj in range(len(xboard))
            if xboard[jj][ii]
        ]
        schedule.extend(
            [
                (ii, jj + self.num_data_qubits + len(xboard), zboard[jj][ii])
                for ii in range(self.num_data_qubits)
                for jj in range(len(zboard))
                if zboard[jj][ii]
            ]
        )

        return schedule
