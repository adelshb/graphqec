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

import numpy as np
import matplotlib.pyplot as plt
import pymatching

# If your code parameter is actually a class (subclass of BaseCode),
# you may want to annotate it with `type[BaseCode]` instead of `BaseCode`.
from graphqec.codes.base_code import BaseCode

# Import the NoiseModel interface (and possibly other models) if needed
from graphqec.noise_models import NoiseModel

__all__ = ["ThresholdLAB"]


class ThresholdLAB:
    r"""
    A class for wrapping threshold calculation.
    """

    __slots__ = (
        "_distances",
        "_error_rates",
        "_code",
        "_collected_stats",
        "_code_name",
    )

    def __init__(
        self,
        code: BaseCode,        # or `type[BaseCode]` if `code` is a class, not an instance
        distances: list[int],
        error_rates: list[float]
    ) -> None:
        r"""
        :param code: The code or code class to use.
        :param distances: Distances for the code.
        :param error_rates: Error rates.
        """
        self._distances = distances
        self._code = code
        self._code_name = code().name  # Instantiating once to get the name
        self._error_rates = error_rates
        self._collected_stats = {}

    @property
    def distances(self) -> list[int]:
        r"""The distances of the code."""
        return self._distances

    @property
    def error_rates(self) -> list[float]:
        r"""The error rates."""
        return self._error_rates

    @property
    def code(self) -> BaseCode:
        r"""The code or code class."""
        return self._code

    @property
    def collected_stats(self) -> dict:
        r"""The collected stats during sampling."""
        return self._collected_stats

    @property
    def code_name(self) -> str:
        r"""The code name."""
        return self._code_name

    @staticmethod
    def compute_logical_errors(code: BaseCode, num_shots: int) -> int:
        r"""
        Sample the memory circuit and return the number of errors.

        :param code: The code instance to simulate.
        :param num_shots: The number of samples.
        """
        # Sample the memory circuit
        sampler = code.memory_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            num_shots, separate_observables=True
        )

        # Configure the decoder using the memory circuit, then run the decoder
        detector_error_model = code.memory_circuit.detector_error_model(
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

    def collect_stats(self, num_shots: int, noise_model: NoiseModel = None) -> None:
        r"""
        Collect sampling statistics over ranges of distance and errors.

        :param num_shots: Number of samples for each distance/error-rate pair.
        :param noise_model: Optional noise model to use; if None, the code's default is used.
        """

        # Loop over distance range
        for distance in self.distances:
            temp_logical_error_rate = []

            # Loop over physical error rates
            for prob_error in self.error_rates:
                # Instantiate the code with the chosen distance, error rates, and optional noise model
                code_instance = self.code(
                    distance=distance,
                    depolarize1_rate=prob_error,
                    depolarize2_rate=prob_error,
                    noise_model=noise_model
                )
                code_instance.build_memory_circuit(number_of_rounds=distance * 3)

                # Get the logical error rate
                num_errors_sampled = self.compute_logical_errors(
                    code=code_instance,
                    num_shots=num_shots
                )
                temp_logical_error_rate.append(num_errors_sampled / num_shots)

            self._collected_stats[distance] = temp_logical_error_rate

    def plot_stats(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        r"""
        Plot the collected data.

        :param x_min: Optional lower bound for the x-axis.
        :param x_max: Optional upper bound for the x-axis.
        :param y_min: Optional lower bound for the y-axis.
        :param y_max: Optional upper bound for the y-axis.
        """
        fig, ax = plt.subplots(1, 1)

        for distance, error_rates in self._collected_stats.items():
            ax.plot(
                self.error_rates,
                error_rates,
                label=f"d={distance}"
            )

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.loglog()
        ax.set_title(f"{self.code_name} Code Error Rates")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.grid(which="major")
        ax.grid(which="minor")
        ax.legend()
        fig.set_dpi(120)
        plt.show()
