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

__all__ = ["RotatedSurfaceCode"]


class RotatedSurfaceCode(BaseCode):
    r"""
    A class for the Rotated Surface code.
    """

    def __init__(
        self,
        distance: int,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the Rotated Surface Code instance.
        """

        self._distance = distance
        self._num_data_qubits = self.distance**2
        self._num_logical_qubits = 1
        self._name = f"Rotated Surface [[{self.num_data_qubits},{self.num_logical_qubits},{self.distance}]]"
        self._checks = ["Z-check", "X-check"]

        super().__init__(*args, **kwargs)

        self._logic_check = {
            "Z": [i for i in range(self.distance)],
            "X": [i * self.distance for i in range(self.distance)],
        }

    def build_graph(self) -> None:
        r"""
        Build the 2D lattice of the rotated surface code.
        """

        # Add the nodes for the data qubits
        data_qubits_coords = [
            (col, row)
            for row in range(1, self.distance + 1)
            for col in range(1, self.distance + 1)
        ]
        data = [
            (i, {"type": "data", "label": None, "coords": data_qubits_coords[i]})
            for i in range(len(data_qubits_coords))
        ]
        self._graph.add_nodes_from(data)

        # Add the nodes the X check qubits.
        x_qubits_coords = [
            (col + (0.5 if row % 2 != 0 else -0.5), row - 0.5)
            for row in range(1, self.distance + 2)
            for col in range(2, self.distance, 2)
        ]
        x_check = [
            (
                i + len(data),
                {"type": "check", "label": "X", "coords": x_qubits_coords[i]},
            )
            for i in range(len(x_qubits_coords))
        ]
        self._graph.add_nodes_from(x_check)

        # Add the ordered edges for the X checks
        x_edges = []
        for qx in x_check:

            coords = qx[1]["coords"]
            ordered_neighbors = self.get_neighbor_qubits(
                coord=coords, index_order=[1, 3, 0, 2]
            )

            for order, neighbor in enumerate(ordered_neighbors):
                if neighbor is not None:
                    x_edges.append((neighbor, qx[0], order + 1))
        self._graph.add_weighted_edges_from(x_edges)

        # Add the nodes the Z check qubits.
        z_qubits_coords = [
            (col + (0.5 if row % 2 == 0 else -0.5), row + 0.5)
            for row in range(1, self.distance)
            for col in range(1, self.distance + 1, 2)
        ]
        z_check = [
            (
                i + len(data) + len(x_check),
                {"type": "check", "label": "Z", "coords": z_qubits_coords[i]},
            )
            for i in range(len(z_qubits_coords))
        ]
        self._graph.add_nodes_from(z_check)

        # Add the ordered edges for the Z checks
        z_edges = []
        for qz in z_check:

            coords = qz[1]["coords"]
            ordered_neighbors = self.get_neighbor_qubits(
                coord=coords, index_order=[1, 0, 3, 2]
            )

            for order, neighbor in enumerate(ordered_neighbors):
                if neighbor is not None:
                    z_edges.append((neighbor, qz[0], order + 1))
        self._graph.add_weighted_edges_from(z_edges)

    def get_neighbor_qubits(
        self, coord: tuple[float, float], index_order: list[int] | None = None
    ) -> list[int]:
        r"""
        Returns the four diagonal qubit, ordered as default:
        - top-left,
        - top-right,
        - bottom-left,
        - bottom-right.

        :param coords: The coordinates of the vertex we want to have the neighbors.
        :param index_order: The order in which the neighbors are
        """
        col, row = coord
        neighbors_coords = [
            (col + dx, row + dy) for dx in [-0.5, 0.5] for dy in [-0.5, 0.5]
        ]

        neighbors = []
        for coords in neighbors_coords:
            try:
                node = [
                    node
                    for node, data in self.graph.nodes(data=True)
                    if data.get("coords") == coords
                ][0]
                neighbors.append(node)
            except IndexError:
                neighbors.append(None)

        if index_order is None:
            return neighbors
        else:
            return [neighbors[i] for i in index_order]
