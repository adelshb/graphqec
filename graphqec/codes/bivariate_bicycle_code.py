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

# See the original implementation by Sergey Bravyi:
# https://github.com/sbravyi/BivariateBicycleCodes.

from __future__ import annotations

import numpy as np

from graphqec.codes.base_code import BaseCode
from graphqec.codes.css_code import compute_logicals

__all__ = ["BivariateBicycleCode"]


class BivariateBicycleCode(BaseCode):
    r"""
    A class for the Bivariate Bicycle code.
    """

    def __init__(
        self,
        L1: int = 12,
        L2: int = 6,
        a1: int = 3,
        a2: int = 1,
        a3: int = 2,
        b1: int = 3,
        b2: int = 1,
        b3: int = 2,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the Bivariate Bicycle (BB) code instance.
        See https://arxiv.org/pdf/2308.07915.pdf for details.
        A and B that depends on two variables x and y such that:
        - x^L1 = 1
        - y^L2 = 1
        - A = x^{a_1} + y^{a_2} + y^{a_3}
        - B = y^{b_1} + x^{b_2} + x^{b_3}

        :param L1: Indicating the number of physical qubits on one axis.
        :param L2: Indicating the number of physical qubits on one axis.
        :param a1: Power in the polynomial.
        :param a2: Power in the polynomial.
        :param a3: Power in the polynomial.
        :param b1: Power in the polynomial.
        :param b2: Power in the polynomial.
        :param b3: Power in the polynomial.
        """

        self._name = "Bivariate Bicycle"
        self._logic_check = []
        self.L1 = L1
        self.L2 = L2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        # The following sequences are taken from Tab 5 in the reference
        self.compute_check_matrices()
        self.sequence_X = [
            (None, None),
            (self.A2, "L"),
            (self.B2, "R"),
            (self.B1, "R"),
            (self.B3, "R"),
            (self.A1, "L"),
            (self.A3, "L"),
        ]
        self.sequence_Z = [
            (self.A1.T, "R"),
            (self.A3.T, "R"),
            (self.B1.T, "L"),
            (self.B2.T, "L"),
            (self.B3.T, "L"),
            (self.A2.T, "R"),
            (None, None),
        ]

        super().__init__(*args, **kwargs)

        self.get_parameters()
        self._name = f"Bivariate Bicycle [[{self.num_data_qubits},{self.num_logical_qubits},{self.distance}]]"
        self._logic_check = {
            "Z": [np.where(row == 1)[0].tolist() for row in self.L_z],
            "X": [np.where(row == 1)[0].tolist() for row in self.L_x],
        }

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

    def compute_check_matrices(self) -> None:
        r""" """
        # Construct the cyclic shifts matrices
        I_ell = np.identity(self.L1, dtype=int)
        I_m = np.identity(self.L2, dtype=int)
        # I = np.identity(self.Lx * self.Ly, dtype=int)
        x = {}
        y = {}
        for i in range(self.L1):
            x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)

        for i in range(self.L2):
            y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))

        self.x = x
        self.y = y
        self.A = (x[self.a1] + y[self.a2] + y[self.a3]) % 2
        self.B = (y[self.b1] + x[self.b2] + x[self.b3]) % 2

        self.A1 = x[self.a1]
        self.A2 = y[self.a2]
        self.A3 = y[self.a3]
        self.B1 = y[self.b1]
        self.B2 = x[self.b2]
        self.B3 = x[self.b3]

    def get_parameters(self) -> None:
        r"""Compute the parameters [[n,k,d]] of the code."""

        AT = np.transpose(self.A)
        BT = np.transpose(self.B)

        self._Hx = np.hstack((self.A, self.B))
        self._Hz = np.hstack((BT, AT))

        self._L_z = compute_logicals(Hx=self.Hx, Hz=self.Hz, logical="Z")
        self._L_x = compute_logicals(Hx=self.Hx, Hz=self.Hz, logical="X")

        self._num_logical_qubits = len(self._L_z)
        self._num_data_qubits = len(self.Hz[0])

        xdistance = min([sum(row) for row in self.L_x])
        zdistance = min([sum(row) for row in self.L_z])
        self._distance = min([xdistance, zdistance])

    def build_graph(self) -> None:
        r"""
        Build the graph for the Bivariate Bicycle code
        """

        L_data = [
            (i, {"type": "data", "label": "L", "label_id": i})
            for i in range(self.L1 * self.L2)
        ]
        self._graph.add_nodes_from(L_data)

        R_data = [
            (
                i + self.L1 * self.L2,
                {"type": "data", "label": "R", "label_id": i},
            )
            for i in range(self.L1 * self.L2)
        ]
        self._graph.add_nodes_from(R_data)

        X_check = [
            (
                i + 2 * self.L1 * self.L2,
                {"type": "check", "label": "X", "label_id": i},
            )
            for i in range(self.L1 * self.L2)
        ]
        self._graph.add_nodes_from(X_check)

        Z_check = [
            (
                i + 3 * self.L1 * self.L2,
                {"type": "check", "label": "Z", "label_id": i},
            )
            for i in range(self.L1 * self.L2)
        ]
        self._graph.add_nodes_from(Z_check)

        # Add the ordered edges for the X checks
        x_edges = []
        for qx in X_check:

            node = (qx[0], self.graph.nodes(data=True)[qx[0]])
            ordered_neighbors = self.get_neighbor_qubits(node=node)

            for order, neighbor in enumerate(ordered_neighbors):
                if neighbor is not None:
                    x_edges.append((neighbor, qx[0], order + 1))
        self._graph.add_weighted_edges_from(x_edges)

        # Add the ordered edges for the Z checks
        z_edges = []
        for qz in Z_check:

            node = (qz[0], self.graph.nodes(data=True)[qz[0]])
            ordered_neighbors = self.get_neighbor_qubits(node=node)

            for order, neighbor in enumerate(ordered_neighbors):
                if neighbor is not None:
                    z_edges.append((neighbor, qz[0], order + 1))
        self._graph.add_weighted_edges_from(z_edges)

    def get_neighbor_qubits(self, node: tuple[int, dict[str, any]]) -> list[int]:
        r"""
        Return ordered neighbors list of the given node.

        :param node: a tuple containing the node number in the graph
            and a dictionnary of attributes.
        """

        neighbors = []

        if node[1]["label"] == "X":
            sequence = self.sequence_X
        elif node[1]["label"] == "Z":
            sequence = self.sequence_Z
        else:
            ValueError("The node label is not correct. Must be X or Z.")

        for item in sequence:

            mat, label = item

            if mat is None:

                neighbors.append(None)

            else:

                nb_label_id = np.nonzero(mat[node[1]["label_id"], :])[0][0]

                # Filter nodes based on the attributes
                filter_conditions = {"label": label, "label_id": nb_label_id}
                filtered_nodes = [
                    node
                    for node, attrs in self.graph.nodes(data=True)
                    if all(
                        attrs.get(key) == value
                        for key, value in filter_conditions.items()
                    )
                ]

                if len(filtered_nodes) != 1:
                    ValueError("There is mode than one node satisfying the condition.")
                else:
                    neighbors.append(filtered_nodes[0])

        return neighbors
