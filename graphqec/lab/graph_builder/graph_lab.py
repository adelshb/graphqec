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

from networkx import Graph

from graphqec.lab.graph_builder.utils import check_neighbor_attribute

__all__ = ["GraphLAB"]


class GraphLAB:
    r"""
    A class for the graph builder assistant.
    """

    __slots__ = "_graph"

    def __init__(
        graph: Graph,
        self,
    ) -> None:
        r"""
        Initialization of the Graph Builder Assistant.

        """

    @property
    def graph(self) -> Graph:
        r"""Return the graph."""
        return self._graph

    def simutaneous_cnot(self) -> bool:
        r"""Check whether qubit only have at most one CNOT at each time step."""

        for node in self.graph.nodes:
            edges = self.graph.edges(node, data=True)

            # Extract the weights
            weights = [data["weight"] for _, _, data in edges]

            # Check if all weights are unique
            if len(weights) != len(set(weights)):
                return True
        return False

    def data_illegal_connection(self) -> bool:
        r"""Check if data qubits are connected to each other illegally."""

        # Filter nodes based on the attributes
        filter_conditions = {"label": "data"}
        filtered_nodes = [
            node
            for node, attrs in self.graph.nodes(data=True)
            if all(attrs.get(key) == value for key, value in filter_conditions.items())
        ]

        res = False
        for node in filtered_nodes:

            res = check_neighbor_attribute(
                graph=self.graph,
                node=node,
                attribute_name="label",
                attribute_value="data",
            )
            if res:
                return True
        return False

    def check_illegal_connection(self) -> bool:
        r"""Check if check qubits are connected to each other illegally."""

        # Filter nodes based on the attributes
        filter_conditions = {"label": "check"}
        filtered_nodes = [
            node
            for node, attrs in self.graph.nodes(data=True)
            if all(attrs.get(key) == value for key, value in filter_conditions.items())
        ]

        res = False
        for node in filtered_nodes:

            res = check_neighbor_attribute(
                graph=self.graph,
                node=node,
                attribute_name="label",
                attribute_value="check",
            )
            if res:
                return True
        return False

    def determinism(self) -> bool:
        r"""
        The CNOT order must ensure stabilisers mutually
        commute so give deterministic measurement outcomes
        in the absence of errors. Practically,
        shared data qubits between two or more stabilisers
        must be interacted with in the same relative order
        by their shared stabilisers. That is, if one stabiliser’s
        interaction precedes another for any shared
        qubit, it must do so for all shared qubits.

        Rule #1 from https://arxiv.org/abs/2409.14765
        """

    def idling(self) -> bool:
        r"""
        Unnecessary idling errors can be avoided by ensuring
        that all CNOTs in a particular time step are
        physically parallel (aligned along the same axis).
        For instance, this implies that while
        all the X-type stabilisers are interacting with data
        qubits 0 or 2 (1 or 3), the Z-type stabilisers must
        also interact with either 0 or 2 (1 or 3), but not
        necessarily respectively.

        Rule #2 from https://arxiv.org/abs/2409.14765
        """

    def hook(self) -> bool:
        r"""
        CNOT order should avoid hook error.

        Rule #3 from https://arxiv.org/abs/2409.14765
        """

    def diagnostic(self, verbose: bool = True) -> list[bool]:
        r"""Run all the rules on the graph."""

        diag = []

        diag.append(self.simutaneous_cnot())
        diag.append(self.data_illegal_connection())
        diag.append(self.check_illegal_connection())
        # diag.append(self.determinism())
        # diag.append(self.idling())
        # diag.append(self.hook())

        return diag
