# graphqec/noise_models.py
from abc import ABC, abstractmethod
from stim import Circuit

class NoiseModel(ABC):
    @abstractmethod
    def apply_noise(self, circuit: Circuit, noise_type: str, qubits: list[int], rate: float) -> None:
        """
        Apply noise to the given circuit.

        :param circuit: The Stim circuit to which noise is applied.
        :param noise_type: A string indicating the noise type (e.g., "single" or "two").
        :param qubits: The list of qubit indices on which to apply noise.
        :param rate: The noise rate (error probability).
        """
        pass


class DepolarizingNoiseModel(NoiseModel):
    def apply_noise(self, circuit: Circuit, noise_type: str, qubits: list[int], rate: float) -> None:
        """
        Concrete implementation of depolarizing noise using Stim instructions:
        - "DEPOLARIZE1" for single-qubit noise.
        - "DEPOLARIZE2" for two-qubit noise.
        """
        if not qubits or rate == 0:
            # No qubits or zero rate => no noise instruction needed
            return

        if noise_type == "single":
            circuit.append("DEPOLARIZE1", qubits, rate)
        elif noise_type == "two":
            circuit.append("DEPOLARIZE2", qubits, rate)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
