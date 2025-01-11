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

__all__ = ["ShorCode"]


class ShorCode(BaseCode):
    r"""
    A class for the Shor code.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the Shor code instance.
        """

        self._name = "Shor"
        self._checks = ["Z-check", "X-check"]
        self._logic_check = []

        super().__init__(*args, **kwargs)

    def build_graph(self) -> None:
        r"""
        Build the graph for the Shor code
        """

        num_data_qubits = self.distance**2
        num_z_checks = num_data_qubits - self.distance

        # Add data qubits
        self._graph.add_nodes_from(
            [(i, {"type": "data", "coords": (i, i)}) for i in range(num_data_qubits)]
        )

        # Add Z checks
        self._graph.add_nodes_from(
            [
                (i + num_data_qubits - self.block(i), {"type": "Z-check", "coords": (i + 0.5, i + 0.5)})
                for i in range(num_data_qubits) if self.index_within_block(i) != self.distance - 1
            ]
        )

        # Add X checks
        self._graph.add_nodes_from(
            [
                (self.block(i) + num_data_qubits + num_z_checks, {"type": "X-check", "coords": (i + 2*self.distance - 1, i)})
                for i in range(num_data_qubits - self.distance) if self.index_within_block(i) == 0
            ]
        )

        # Add Z-check edges
        self._graph.add_weighted_edges_from(
            [(i, i + num_data_qubits - i // self.distance, 1)
             for i in range(num_data_qubits) if self.index_within_block(i) != (self.distance-1)]
        )
        self._graph.add_weighted_edges_from(
            [(i, i + num_data_qubits - i // self.distance - 1, 1)
             for i in range(num_data_qubits) if self.index_within_block(i) != 0]
        )

        # Add X-check edges
        self._graph.add_weighted_edges_from(
            [(i, self.block(i) + num_data_qubits + num_z_checks, 1) for i in range(num_data_qubits - self.distance) if self.block(i) != (self.distance - 1)]
        )
        self._graph.add_weighted_edges_from(
            [(i, self.block(i) + num_data_qubits + num_z_checks - 1, 1) for i in range(self.distance, num_data_qubits)]
        )

    def block(self, data_qubit_index):
        r"""
        Which block the data qubit is in
        """
        return data_qubit_index // self.distance
    
    def index_within_block(self, data_qubit_index):
        r"""
        Returns a data qubit's index within a block
        """
        return data_qubit_index % self.distance
