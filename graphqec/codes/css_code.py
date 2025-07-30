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

from graphqec.codes.base_code import BaseCode
from graphqec.codes.code_tools import binary_rank, commutation_test, compute_kernel

__all__ = ["CssCode"]


class CssCode(BaseCode):
    r"""
    A class for a generic CSS code specified by a pair of linear codes subject
    to a constraint to ensure commutativity of the generators in the
    associated quantum code.
    """

    def __init__(
        self,
        Hx: list[list[int]],
        Hz: list[list[int]],
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the CSS code instance.
        """

        assert commutation_test(Hx, Hz)

        self.Hx = Hx
        self.Hz = Hz

        self.nqubits = len(Hz[0])

        self._logic_check = [self.compute_x_logicals(), self.compute_z_logicals()]

        assert len(self._logic_check[0]) == len(self._logic_check[1])

        super().__init__(*args, **kwargs)

        self._name = (
            # f"CSS [[{self.nqubits},{len(self._logic_check)},{self.distance}]]"
            f"CSS [[{self.nqubits},{len(self._logic_check[0])}]]"
        )

    def build_graph(self) -> None:
        r"""
        Build the Tanner graph for the CSS code.
        Qubit nodes labels: 0, 1,..., nqubits-1
        Xcheck nodes labels: nqubits, nqubits+1,...,nqubits+nxchecks-1
        Zcheck nodes labels: nqubits+nxchecks,...,nqubits+nxchecks+nzchecks-1

        Edges are added based on the X and Z check matrices
        """

        self._graph.add_nodes_from(
            [
                (i, {"type": "data", "label": None, "label_id": i})
                for i in range(self.nqubits)
            ]
        )
        self._graph.add_nodes_from(
            [
                (
                    i,
                    {"type": "check", "label": "X", "label_id": i},
                )
                for i in range(self.nqubits, self.nqubits + len(self.Hx))
            ]
        )
        self._graph.add_nodes_from(
            [
                (
                    i,
                    {"type": "check", "label": "Z", "label_id": i},
                )
                for i in range(
                    self.nqubits + len(self.Hx),
                    self.nqubits + len(self.Hx) + len(self.Hz),
                )
            ]
        )

        for row_index in range(len(self.Hx)):

            xrow = self.Hx[row_index]
            suppx = []

            for qcol in range(self.nqubits):
                if xrow[qcol]:
                    suppx.append(qcol)

            self._graph.add_weighted_edges_from(
                [(suppx[i], self.nqubits + row_index, i) for i in range(len(suppx))]
            )

        for row_index in range(len(self.Hz)):

            zrow = self.Hz[row_index]
            suppz = []

            for qcol in range(self.nqubits):
                if zrow[qcol]:
                    suppz.append(qcol)

            self._graph.add_weighted_edges_from(
                [
                    (qcol, self.nqubits + len(self.Hx) + row_index, i)
                    for i in range(len(suppz))
                ]
            )

    def compute_x_logicals(self) -> list[list[int]]:
        r"""
        Function for computing a set of logical operators for the input
        parity check operators.
        Find the image and kernel for each linear code.
        """

        # First find the kernel of Hz

        zkern = compute_kernel(self.Hz)

        # now find elements of ker Hz that are independent of im Hx

        Hxw = [row for row in self.Hx]
        starting_rank = binary_rank(self.Hx)
        ranking_rank = starting_rank

        for row in zkern:
            new_rank = binary_rank(Hxw + [row])
            if new_rank > ranking_rank:  # If rank increases, new row is indept
                Hxw.append(row)  # Add new row
                ranking_rank = new_rank

        xlogs = Hxw[-(ranking_rank - starting_rank) :]

        return xlogs

    def compute_z_logicals(self) -> list[list[int]]:
        r"""
        Function for computing a set of logical operators for the input
        parity check operators.
        Find the image and kernel for each linear code.
        """

        # First find the kernel of Hz

        xkern = compute_kernel(self.Hx)

        # now find elements of ker Hz that are independent of im Hx

        Hzw = [row for row in self.Hz]
        starting_rank = binary_rank(self.Hz)
        ranking_rank = starting_rank

        for row in xkern:
            new_rank = binary_rank(Hzw + [row])
            if new_rank > ranking_rank:  # If rank increases, new row is indept
                Hzw.append(row)  # Add new row
                ranking_rank = new_rank

        zlogs = Hzw[-(ranking_rank - starting_rank) :]

        return zlogs
