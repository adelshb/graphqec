# graphqec/codes/base_code.py

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

# NEW: Import your NoiseModel and a default DepolarizingNoiseModel
from graphqec.noise_models import NoiseModel, DepolarizingNoiseModel

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
        "_noise_model",  # NEW SLOT
    )

    def __init__(
        self,
        distance: int = 3,
        depolarize1_rate: float = 0,
        depolarize2_rate: float = 0,
        noise_model: NoiseModel = None,  # NEW PARAM
    ) -> None:
        r"""
        Initialization of the Base Code class.

        :param distance: Distance of the code.
        :param depolarize1_rate: Single qubit depolarization rate.
        :param depolarize2_rate: Two qubit depolarization rate.
        :param noise_model: An optional NoiseModel instance (defaults to DepolarizingNoiseModel).
        """

        self._name = self.__class__.__name__  # Or set this however you'd like
        self._distance = distance
        self._depolarize1_rate = depolarize1_rate
        self._depolarize2_rate = depolarize2_rate

        # If no noise model is given, default to DepolarizingNoiseModel
        self._noise_model = noise_model if noise_model else DepolarizingNoiseModel()

        self._memory_circuit: Circuit
        self._measurement = Measurement()
        self._checks: list[str] = []
        self._logic_check: list[str] = []
        self._graph = nx.Graph()

        self.build_graph()

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
        The depolarization rate for single-qubit gates.
        """
        return self._depolarize1_rate

    @property
    def depolarize2_rate(self) -> float:
        r"""
        The depolarization rate for two-qubit gates.
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
        The number of outcomes collected.
        """
        return self.measurement.register_count

    @property
    def graph(self) -> nx.Graph:
        r"""
        The graph representing the qubits' network.
        """
        return self._graph

    @property
    def checks(self) -> list[str]:
        r"""
        The different checks (e.g., 'Z-check', 'X-check') in the QEC.
        """
        return self._checks

    @property
    def logic_check(self) -> list[str]:
        r"""
        Return logic check qubits.
        """
        return self._logic_check

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
        all_qubits = list(self.graph.nodes())
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

        all_check_qubits = [q for sublist in check_qubits.values() for q in sublist]

        # Initialization
        self._memory_circuit = Circuit()
        self._memory_circuit.append("R", all_qubits)

        # Apply single-qubit noise to all qubits
        self._noise_model.apply_noise(
            circuit=self._memory_circuit,
            noise_type="single",
            qubits=all_qubits,
            rate=self.depolarize1_rate,
        )

        self.append_stab_circuit(
            round=0, data_qubits=data_qubits, check_qubits=check_qubits
        )

        # Add DETECTOR instructions for the first round
        if "Z-check" in check_qubits:
            for qz in check_qubits["Z-check"]:
                rec = self.get_target_rec(qubit=qz, round=0)
                self._memory_circuit.append("DETECTOR", [target_rec(rec)])

        # Body rounds
        for r in range(1, number_of_rounds):
            self.append_stab_circuit(
                round=r, data_qubits=data_qubits, check_qubits=check_qubits
            )

            for q in all_check_qubits:
                past_rec = self.get_target_rec(qubit=q, round=r - 1)
                current_rec = self.get_target_rec(qubit=q, round=r)
                self._memory_circuit.append(
                    "DETECTOR",
                    [target_rec(past_rec), target_rec(current_rec)],
                )

        # Finalization
        self._noise_model.apply_noise(
            circuit=self._memory_circuit,
            noise_type="single",
            qubits=data_qubits,
            rate=self.depolarize1_rate,
        )
        self._memory_circuit.append("M", data_qubits)

        for i, q in enumerate(data_qubits):
            self.add_outcome(
                outcome=target_rec(-1 - i), qubit=q, round=number_of_rounds, type="data"
            )

        # Syndrome extraction grouping data qubits
        if "Z-check" in check_qubits:
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
        if recs:
            recs_str = " ".join(f"rec[{rec}]" for rec in recs if rec is not None)
            self._memory_circuit.append_from_stim_program_text(
                f"OBSERVABLE_INCLUDE(0) {recs_str}"
            )

    def append_stab_circuit(
        self, round: int, data_qubits: list[int], check_qubits: dict[str, list[int]]
    ) -> None:
        r"""
        Append the stabilizer circuit for one round.
        """
        all_check_qubits = [q for sublist in check_qubits.values() for q in sublist]

        # Apply single-qubit noise to check qubits at the start of each round > 0
        if round > 0:
            self._noise_model.apply_noise(
                circuit=self._memory_circuit,
                noise_type="single",
                qubits=all_check_qubits,
                rate=self.depolarize1_rate,
            )

        # If there are X-check qubits, apply H and noise
        if "X-check" in check_qubits:
            x_checks = check_qubits["X-check"]
            self._memory_circuit.append("H", x_checks)
            self._noise_model.apply_noise(
                circuit=self._memory_circuit,
                noise_type="single",
                qubits=x_checks,
                rate=self.depolarize1_rate,
            )

        # Track which data qubits were used in this round
        measured = {qd: False for qd in data_qubits}

        # Perform CNOTs in specific weight order to avoid hook errors
        for order in range(1, 5):
            for check_type, qubits_list in check_qubits.items():
                for check_q in qubits_list:
                    data_neighbors = [
                        neighbor
                        for neighbor, attrs in self.graph[check_q].items()
                        if attrs.get("weight") == order
                    ]
                    if len(data_neighbors) == 1:
                        data = data_neighbors[0]
                        self.append_stab_element(
                            data_qubit=data,
                            check_qubit=check_q,
                            check=check_type,
                        )
                        # Apply two-qubit noise after each CNOT
                        self._noise_model.apply_noise(
                            circuit=self._memory_circuit,
                            noise_type="two",
                            qubits=[data, check_q],
                            rate=self.depolarize2_rate,
                        )
                        measured[data] = True

        # Apply single-qubit noise to any data qubits not used this round
        not_measured = [qd for qd, used in measured.items() if not used]
        self._noise_model.apply_noise(
            circuit=self._memory_circuit,
            noise_type="single",
            qubits=not_measured,
            rate=self.depolarize1_rate,
        )

        # If there are X-check qubits, apply another H and noise
        if "X-check" in check_qubits:
            x_checks = check_qubits["X-check"]
            self._memory_circuit.append("H", x_checks)
            self._noise_model.apply_noise(
                circuit=self._memory_circuit,
                noise_type="single",
                qubits=x_checks,
                rate=self.depolarize1_rate,
            )

        # Apply single-qubit noise to all check qubits before measurement
        self._noise_model.apply_noise(
            circuit=self._memory_circuit,
            noise_type="single",
            qubits=all_check_qubits,
            rate=self.depolarize1_rate,
        )

        # Measure check qubits
        self._memory_circuit.append("MR", all_check_qubits)
        for i, q in enumerate(all_check_qubits):
            self.add_outcome(
                outcome=target_rec(-1 - i), qubit=q, round=round, type="check"
            )

    def append_stab_element(self, data_qubit: int, check_qubit: int, check: str) -> None:
        """
        Append the appropriate stabilizer operation (Z_check or X_check).
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
            raise ValueError("Y-check is not implemented.")
        else:
            raise ValueError(f"Unknown check type '{check}'.")

    def get_outcome(self, qubit: int, round: int) -> any:
        r"""
        Return the outcome for the qubit at the specified round or None.
        """
        return self._measurement.get_outcome(qubit=qubit, round=round)

    def add_outcome(self, outcome: any, qubit: int, round: int, type: str | None) -> None:
        r"""
        Add an outcome to the collection.
        """
        self._measurement.add_outcome(
            outcome=outcome,
            qubit=qubit,
            round=round,
            type=type,
        )

    def get_target_rec(self, qubit: int, round: int) -> int | None:
        r"""
        Return the rec of a specific measurement.
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
        Draw the qubit graph.
        """
        # Extract qubit types for coloring
        node_categories = nx.get_node_attributes(self.graph, "type")
        unique_categories = sorted(set(node_categories.values()))

        # Define a custom color palette
        custom_colors = {
            "data": "#D3D3D3",   # grey
            "Z-check": "#d62728",  # red
            "X-check": "#1f77b4",  # blue
            "Y-check": "#2ca02c",  # green
        }

        # Map node types to colors
        node_colors = [
            custom_colors.get(node_categories[node], "#808080")
            for node in self.graph.nodes()
        ]

        # Define layout
        try:
            pos = {
                node: (data["coords"][0], data["coords"][1])
                for node, data in self.graph.nodes(data=True)
            }
        except KeyError:
            pos = nx.spring_layout(self.graph)

        # Draw the graph
        plt.figure(figsize=(6, 6))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=400,
            node_color=node_colors,
            font_size=8,
            font_weight="bold",
            edge_color="gray",
            width=1,
        )

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=8, font_weight="bold"
        )

        # Create legend
        category_legend = [
            mpatches.Patch(color=custom_colors[cat], label=f"{cat} qubit")
            for cat in unique_categories
        ]
        plt.legend(
            handles=category_legend,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Qubit Types",
        )
        plt.title(f"{self.name} Graph")
        plt.show()
