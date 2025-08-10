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

import networkx as nx
from networkx import Graph


def check_neighbor_attribute(
    graph: Graph, node: int, attribute_name: str, attribute_value: str
) -> bool:
    r"""
    Check if any neighbor of a given node has a specific attribute.

    :param graph: The graph over which we check the attributes.
    :param node: The node from which we check the neighbors.
    :param attribute_name: The attribute name.
    :param attribute_value: The attribute value.
    """

    # Get the neighbors of the node
    neighbors = graph.neighbors(node)

    # Check if any neighbor has the specified attribute value
    for neighbor in neighbors:
        if graph.nodes[neighbor].get(attribute_name) == attribute_value:
            return True
    return False


def find_cycle_with_attributes(
    graph: Graph, target_attributes: list[dict[str, str]], target_cycle_length: int
) -> list[list[int]]:
    r"""
    Find all cycles in the graph where nodes satisfy target attributes.

    :param graph: The graph over which we check the attributes.
    :param target_attributes: A list of dictionary for the attributes for each node.
    :param target_cycle: The length of the cycles we want to find.
    """

    # Find all simple cycles in the graph
    cycles = list(nx.simple_cycles(graph))

    filtered_cycles = []

    for cycle in cycles:
        if len(cycle) == target_cycle_length:

            # Get the relevant attributes for each node in the cycle
            cycle_attributes = [graph.nodes[node] for node in cycle]

            match = is_subset_of(target_attributes, cycle_attributes)
            if match:
                filtered_cycles.append(cycle)
    return filtered_cycles


def is_subset_of(sublist: list[dict], mainlist: list[dict]) -> bool:
    r"""
    Check if each dictionary in sublist is a subset of any dictionary in mainlist.

    :param sublist: A list of dictionnary we want to find in the other list.
    :param mainlist: A list of dictionnary we want to find a subset from.
    """
    for dict1 in sublist:
        match_found = False
        for dict2 in mainlist:
            if all(item in dict2.items() for item in dict1.items()):
                match_found = True
                break
        if not match_found:
            return False
    return True
