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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import networkx as nx
from stim import Circuit, target_rec
import pymatching

from qec.measurement import Measurement
from qec.stab import X_check, Z_check

__all__ = ["BaseCode"]


class BaseCode(ABC):
    r"""
    An abstract base class for quantum error correction codes.
    """

    __slots__ = (
        "_distance",
        "_memory_circuit",
        "_depolarize1_rate",
        "_depolarize2_rate",
        "_sampler",
        "_measurement",
        "_graph" "_checks",
        "_logic_check",
    )

    def __init__(
        self,
        distance: int = 3,
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
    def sampler(self) -> any:
        r"""
        The sampler from the memory circuit
        """
        return self._sampler

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

    @abstractmethod
    def build_graph(self, number_of_rounds: int = 2) -> None:
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
                if data.get("type") == check
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

        for qz in check_qubits["Z-check"]:

            qz_adjacent_data_qubits = self.graph.neighbors(qz)

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
        for order in range(1, 5):
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

    def compute_logical_errors(self, num_shots: int) -> int:
        r"""
        Sample the memory circuit and return the number of errors.

        :param num_shots: The number of samples.
        """

        # Sample the memory circuit
        self._sampler = self.memory_circuit.compile_detector_sampler()
        detection_events, observable_flips = self.sampler.sample(
            num_shots, separate_observables=True
        )

        # Configure the decoder using the memory circuit then run the decoder
        detector_error_model = self.memory_circuit.detector_error_model(
            decompose_errors=False
        )

        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
        predictions = matcher.decode_batch(detection_events)

        # Count the number of errors
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors

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
        node_categories = nx.get_node_attributes(self.graph, "type")

        # Get the unique type and map them to a colormap
        unique_categories = sorted(
            set(node_categories.values())
        )  # Get unique categories
        num_categories = len(unique_categories)
        colors = cm.get_cmap("Set1", num_categories)

        # Map node colors based on their category
        node_colors = [
            colors(unique_categories.index(node_categories[node]))
            for node in self.graph.nodes()
        ]

        # Draw the graph
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(self.graph)

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
        )

        # Add edge weights as labels
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=8, font_weight="bold"
        )

        # Create and Display a custom legend patches for each unique type
        category_legend = [
            mpatches.Patch(color=colors(i), label=f"{unique_categories[i]} qubit")
            for i in range(num_categories)
        ]
        plt.legend(
            handles=category_legend, loc="upper left", bbox_to_anchor=(1, 1), title=""
        )

        # Display the graph
        plt.title("")
        plt.show()
