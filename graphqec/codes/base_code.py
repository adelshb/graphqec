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
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from stim import Circuit, target_rec

from graphqec.measurement import Measurement
from graphqec.stab import X_check, Z_check

__all__ = ["BaseCode"]


class BaseCode(ABC):
    r"""
    An abstract base class for quantum error correction codes.
    """

    __slots__ = (
        "_name",
        "_distance",
        "_memory_circuit",
        "_depolarize1_rate",
        "_depolarize2_rate",
        "_measurement",
        "_graph",
        "_checks",
        "_logic_check",
        "_length_sym_measurement",
    )

    def __init__(
        self,
        distance: int | None = None,
        depolarize1_rate: float = 0,
        depolarize2_rate: float = 0,
    ) -> None:
        r"""
        Initialization of the Base Code class.

        :param distance: Distance of the code.
        :param depolarize1_rate: Single qubit depolarization rate.
        :param depolarize2_rate: Two qubit depolarization rate.
        """

        self._distance = distance
        self._depolarize1_rate = depolarize1_rate
        self._depolarize2_rate = depolarize2_rate
        self._memory_circuit: Circuit
        self._measurement = Measurement()
        self._checks: list[str]
        self._logic_check: list[str]

        self._graph = nx.Graph()
        self.build_graph()
        self.get_check_label()

        self._length_sym_measurement = max(
            nx.get_edge_attributes(self.graph, "weight").values()
        )

    @property
    def name(self) -> str:
        r"""
        The name of the code.
        """
        return self._name

    @property
    def distance(self) -> int:
        r"""
        The distance of the code.
        """
        return self._distance

    @property
    def memory_circuit(self) -> Circuit:
        r"""
        The circuit for the memory.
        """
        return self._memory_circuit

    @property
    def depolarize1_rate(self) -> float:
        r"""
        The depolarization rate for single qubit gate.
        """
        return self._depolarize1_rate

    @property
    def depolarize2_rate(self) -> float:
        r"""
        The depolarization rate for two-qubit gate.
        """
        return self._depolarize2_rate

    @property
    def measurement(self) -> Measurement:
        r"""
        Return the measurement collection.
        """
        return self._measurement

    @property
    def register_count(self) -> int:
        r"""
        The number of outcome collected.
        """
        return self.measurement.register_count

    @property
    def graph(self) -> int:
        r"""
        The graph representing qubits network
        """
        return self._graph

    @property
    def checks(self) -> int:
        r"""
        The different checks in the QEC
        """
        return self._checks

    @property
    def logic_check(self) -> int:
        r"""
        Return logic check.
        """
        return self._logic_check

    def get_check_label(self) -> None:
        r"""
        Get the check labels.
        """
        node_categories = {}
        for node, data in self.graph.nodes(data=True):
            if data["type"] == "check":
                node_categories[node] = f"{data['label']}-check"
        self._checks = sorted(set(node_categories.values()))

    @abstractmethod
    def build_graph(self) -> None:
        r"""
        Build the graph representing the qubit network.
        """

    def build_memory_circuit(self, number_of_rounds: int) -> None:
        r"""
        Build and return a Stim Circuit object implementing a memory for the given time.

        :param number_of_rounds: The number of rounds in the memory.
        """

        all_qubits = [q for q in self.graph.nodes()]
        data_qubits = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("type") == "data"
        ]

        check_qubits = {}
        for check in self.checks:
            check_qubits[check] = [
                node
                for node, data in self.graph.nodes(data=True)
                if f"{data.get("label")}-{data.get("type")}" == check
            ]

        temp = [item for item in check_qubits.values()]
        all_check_qubits = [item for sublist in temp for item in sublist]

        # Initialization
        self._memory_circuit = Circuit()

        self._memory_circuit.append("R", all_qubits)
        self._memory_circuit.append("DEPOLARIZE1", all_qubits, self.depolarize1_rate)

        self.append_stab_circuit(
            round=0, data_qubits=data_qubits, check_qubits=check_qubits
        )

        for qz in check_qubits["Z-check"]:
            rec = self.get_target_rec(qubit=qz, round=0)
            self._memory_circuit.append("DETECTOR", [target_rec(rec)])

        # Body rounds
        for round in range(1, number_of_rounds):

            self.append_stab_circuit(
                round=round, data_qubits=data_qubits, check_qubits=check_qubits
            )

            for q in all_check_qubits:
                past_rec = self.get_target_rec(qubit=q, round=round - 1)
                current_rec = self.get_target_rec(qubit=q, round=round)
                self._memory_circuit.append(
                    "DETECTOR",
                    [target_rec(past_rec), target_rec(current_rec)],
                )

        # Finalization
        self._memory_circuit.append("DEPOLARIZE1", data_qubits, self.depolarize1_rate)
        self._memory_circuit.append("M", data_qubits)

        for i, q in enumerate(data_qubits):
            self.add_outcome(
                outcome=target_rec(-1 - i), qubit=q, round=number_of_rounds, type="data"
            )

        # Syndrome extraction grouping data qubits
        for qz in check_qubits["Z-check"]:

            qz_adjacent_data_qubits = list(self.graph.neighbors(qz))

            recs = [
                self.get_target_rec(qubit=qd, round=number_of_rounds)
                for qd in qz_adjacent_data_qubits
            ]
            recs += [self.get_target_rec(qubit=qz, round=number_of_rounds - 1)]

            self._memory_circuit.append("DETECTOR", [target_rec(r) for r in recs])

        # Adding the comparison with the expected state
        recs = [
            self.get_target_rec(qubit=q, round=number_of_rounds)
            for q in self.logic_check
        ]
        recs_str = " ".join(f"rec[{rec}]" for rec in recs)
        self._memory_circuit.append_from_stim_program_text(
            f"OBSERVABLE_INCLUDE(0) {recs_str}"
        )

    def append_stab_circuit(
        self, round: int, data_qubits: list[int], check_qubits: dict[str, list[int]]
    ) -> None:
        r"""
        Append the stabilizer circuit.

        :param round: The correction round.
        :param data_qubits: A list of data qubits index.
        :param check_qubits: A list of check qubits.
        """

        temp = [item for item in check_qubits.values()]
        all_check_qubits = [item for sublist in temp for item in sublist]

        if round > 0:
            self._memory_circuit.append(
                "DEPOLARIZE1", all_check_qubits, self.depolarize1_rate
            )

        if "X-check" in self.checks:
            self._memory_circuit.append("H", [q for q in check_qubits["X-check"]])
            self._memory_circuit.append(
                "DEPOLARIZE1",
                [q for q in check_qubits["X-check"]],
                self.depolarize1_rate,
            )

        # A flag to tell us if a data qubit was used this round
        measured = {qd: False for qd in data_qubits}

        # Perform CNOTs with specific order to avoid hook errors
        for order in range(1, self._length_sym_measurement + 1):
            for check in check_qubits.keys():
                for q in check_qubits[check]:
                    data = [
                        neighbor
                        for neighbor, attrs in self.graph[q].items()
                        if attrs.get("weight") == order
                    ]
                    if len(data) == 1:
                        data = data[0]
                        self.append_stab_element(
                            data_qubit=data, check_qubit=q, check=check
                        )
                        self._memory_circuit.append(
                            "DEPOLARIZE2", [data, q], self.depolarize2_rate
                        )
                        measured[data] = True

        # Apply depolarization channel to account for the time not being used
        not_measured = [key for key, value in measured.items() if value is False]
        self._memory_circuit.append("DEPOLARIZE1", not_measured, self.depolarize1_rate)

        if "X-check" in self.checks:
            self._memory_circuit.append("H", [q for q in check_qubits["X-check"]])
            self._memory_circuit.append(
                "DEPOLARIZE1",
                [q for q in check_qubits["X-check"]],
                self.depolarize1_rate,
            )

        self._memory_circuit.append(
            "DEPOLARIZE1", [q for q in all_check_qubits], self.depolarize1_rate
        )
        self._memory_circuit.append("MR", [q for q in all_check_qubits])
        for i, q in enumerate(all_check_qubits):
            self.add_outcome(
                outcome=target_rec(-1 - i), qubit=q, round=round, type="check"
            )

    def append_stab_element(
        self, data_qubit: any, check_qubit: any, check: str
    ) -> None:
        r"""
        Append a stabilizer circuit to the circuit.

        :param data_qubits: A list of data qubits index.
        :param check_qubits: A list of check qubits.
        :param check: A string indicating what stabilizer we are doing.
        """

        if check == "Z-check":
            Z_check(
                circ=self._memory_circuit,
                data_qubit=data_qubit,
                check_qubit=check_qubit,
            )
        elif check == "X-check":
            X_check(
                circ=self._memory_circuit,
                data_qubit=data_qubit,
                check_qubit=check_qubit,
            )
        elif check == "Y-check":
            ValueError("This check is not implemented.")
        else:
            ValueError("This check is not implemented.")

    def get_outcome(
        self,
        qubit: any,
        round: int,
    ) -> any:
        r"""
        Return the outcome for the qubit at the specified round or None.

        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        """
        return self._measurement.get_outcome(qubit=qubit, round=round)

    def add_outcome(
        self, outcome: any, qubit: any, round: int, type: str | None
    ) -> None:
        r"""
        Add an outcome to the collection.

        :param outcome: The outcome to store.
        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        :param type: The type of measurement.
        """
        self._measurement.add_outcome(
            outcome=outcome, qubit=qubit, round=round, type=type
        )

    def get_target_rec(self, qubit: any, round: int) -> int | None:
        r"""
        Return the rec of a specific measurement.

        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        """

        try:
            return (
                self.measurement.get_register_id(qubit=qubit, round=round)
                - self.measurement.register_count
            )
        except TypeError:
            return None

    def draw_graph(self) -> None:
        r"""
        Draw the graph.
        """

        # Extract qubit type for coloring
        node_categories = {}
        for node, data in self.graph.nodes(data=True):
            if data["label"] is not None:
                node_categories[node] = f"{data['label']}-{data['type']}"
            else:
                node_categories[node] = data["type"]

        # Get the unique types
        unique_categories = sorted(set(node_categories.values()))

        # Custom color palette
        custom_colors = {
            "data": "#D3D3D3",  # grey
            "L-data": "#FFCC99",  # orange
            "R-data": "#FFB6C1",  # pink
            "Z-check": "#d62728",  # red
            "X-check": "#1f77b4",  # blue
            "Y-check": "#2ca02c",  # green
            # Add more colors if you have more types
        }

        # Ensure that each type has a corresponding color
        node_colors = [
            custom_colors.get(
                node_categories[node], "#808080"
            )  # default to gray if type is not in custom_colors
            for node in self.graph.nodes()
        ]

        # Define a layout
        try:
            pos = {
                int(node[0]): (node[1]["coords"][0], node[1]["coords"][1])
                for node in self.graph.nodes(data=True)
            }
        except KeyError:
            pos = nx.spring_layout(self.graph)

        # Draw the graph
        plt.figure(figsize=(6, 6))

        # Draw the graph with node numbers and colors
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=400,
            node_color=node_colors,
            font_size=8,
            font_weight="bold",
            edge_color="gray",
            width=1,  # Edge width (adjust as needed)
        )

        # Add edge weights as labels
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=8, font_weight="bold"
        )

        # Create and display custom legend patches for each unique type
        category_legend = [
            mpatches.Patch(color=custom_colors[category], label=f"{category} qubit")
            for category in unique_categories
        ]

        # Display the graph
        plt.legend(
            handles=category_legend, loc="upper left", bbox_to_anchor=(1, 1), title=""
        )
        plt.title("")
        plt.show()
