# tests/codes/test_noise_model.py

import stim
import pytest
from graphqec.noise_models import DepolarizingNoiseModel

def test_depolarizing_noise_single_qubit():
    """Check single-qubit depolarizing noise."""
    circuit = stim.Circuit()
    noise_model = DepolarizingNoiseModel()

    noise_model.apply_noise(circuit, "single", [0], 0.01)
    text_diagram = circuit.to_text_diagram()

    assert "DEPOLARIZE1(0.01) 0" in text_diagram, (
        "Expected single-qubit depolarizing noise instruction missing."
    )

def test_depolarizing_noise_two_qubit():
    """Check two-qubit depolarizing noise."""
    circuit = stim.Circuit()
    noise_model = DepolarizingNoiseModel()

    noise_model.apply_noise(circuit, "two", [0, 1], 0.02)
    text_diagram = circuit.to_text_diagram()

    assert "DEPOLARIZE2(0.02) 0 1" in text_diagram, (
        "Expected two-qubit depolarizing noise instruction missing."
    )
