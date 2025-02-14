# tests/codes/test_base_code.py

import pytest
import stim

from graphqec.codes.base_code import BaseCode
from graphqec.noise_model import DepolarizingNoiseModel

class MinimalTestCode(BaseCode):
    def build_graph(self):
        """
        Build a trivial graph with:
         - 1 data qubit (node 0)
         - 1 Z-check qubit (node 1)
         - An edge connecting them with weight=1
        """
        self._graph.add_node(0, type="data")
        self._graph.add_node(1, type="Z-check")
        self._checks = ["Z-check"]
        self._logic_check = []
        self._graph.add_edge(0, 1, weight=1)

def test_base_code_noise_integration():
    code = MinimalTestCode(
        distance=3,
        depolarize1_rate=0.01,
        depolarize2_rate=0.02,
        noise_model=DepolarizingNoiseModel(),
    )
    code.build_memory_circuit(number_of_rounds=2)

    circuit = code.memory_circuit
    text_diagram = str(circuit)

    # Check for single-qubit noise
    assert "DEPOLARIZE1(0.01)" in text_diagram, (
        "Expected single-qubit noise not found in circuit."
    )
    # Check for two-qubit noise
    assert "DEPOLARIZE2(0.02)" in text_diagram, (
        "Expected two-qubit noise not found in circuit."
    )
