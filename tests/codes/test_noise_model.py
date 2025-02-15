# tests/codes/test_noise_model.py

import stim
import pytest
from graphqec.noise_model import DepolarizingNoiseModel

def test_depolarizing_noise_single_qubit():
    circuit = stim.Circuit()
    noise_model = DepolarizingNoiseModel()

    # Apply single-qubit noise
    noise_model.apply_noise(circuit, "single", [0], 0.01)

    # Convert circuit to text using str(...)
    text_diagram = str(circuit)

    # Check for the DEPOLARIZE1 instruction
    assert "DEPOLARIZE1(0.01) 0" in text_diagram, (
        "Expected single-qubit depolarizing noise instruction missing."
    )

def test_depolarizing_noise_two_qubit():
    circuit = stim.Circuit()
    noise_model = DepolarizingNoiseModel()

    # Apply two-qubit noise
    noise_model.apply_noise(circuit, "two", [0, 1], 0.02)
    text_diagram = str(circuit)

    # Check for the DEPOLARIZE2 instruction
    assert "DEPOLARIZE2(0.02) 0 1" in text_diagram, (
        "Expected two-qubit depolarizing noise instruction missing."
    )
