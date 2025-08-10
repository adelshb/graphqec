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
import multiprocessing
import warnings

import sinter
import matplotlib.pyplot as plt
from stimbposd import SinterDecoder_BPOSD
from beliefmatching import BeliefMatchingSinterDecoder
from mwpf import SinterMWPFDecoder

from graphqec.codes.base_code import BaseCode

__all__ = ["ThresholdLAB"]

__available_decoders__ = {
    "pymatching": None,
    "fusion_blossom": None,
    "bposd": SinterDecoder_BPOSD,
    "mwpf": SinterMWPFDecoder,
    "beliefmatching": BeliefMatchingSinterDecoder,
}


class ThresholdLAB:
    r"""
    A class for wrapping threshold calculation
    """

    __slots__ = (
        "_configurations",
        "_error_rates",
        "_code",
        "_collected_stats",
        "_decoder",
        "_samples",
        "_code_name",
        "_logic_check",
    )

    def __init__(
        self,
        code: BaseCode,
        configurations: dict[str, any],
        error_rates: list[float],
        decoder: str = "pymatching",
        logic_check: str = "Z",
    ) -> None:
        r"""
        Initialization of the Base Code class.

        :param code: The code
        :param configurations: Distances for the code.
        :param error_rates: Error rate.
        :param decoder: The decoder
        """

        self._configurations = configurations
        self._code = code
        self._error_rates = error_rates

        if decoder in __available_decoders__.keys():
            self._decoder = decoder
        else:
            ValueError("This decoder is not available.")

        self._collected_stats = {}
        self._logic_check = logic_check

    @property
    def configurations(self) -> list[int]:
        r"""
        The distances of the code.
        """
        return self._configurations

    @property
    def error_rates(self) -> list[float]:
        r"""
        The error rates.
        """
        return self._error_rates

    @property
    def code(self) -> BaseCode:
        r"""
        The code.
        """
        return self._code

    @property
    def decoder(self) -> str:
        r"""Return the decoder."""
        return self._decoder

    @property
    def samples(self) -> dict[str, any]:
        r"""Return the collected samples."""
        return self._samples

    def logic_check(self) -> str:
        r"""
        The logic check type.
        """
        return self._logic_check

    def generate_sinter_tasks(self, logic_check: str = "Z") -> sinter.TaskGenerator:
        r"""Generates tasks using Stim's circuit generation."""

        # Loop over configurations
        for configuration in self.configurations:

            # Loop over physical errors
            for prob_error in self.error_rates:

                code = self.code(
                    **configuration,
                    depolarize1_rate=prob_error,
                    depolarize2_rate=prob_error,
                )

                if logic_check not in code.logic_check.keys():
                    raise ValueError(
                        f"Logic check {logic_check} is not supported by {code.name} code."
                    )

                code.build_memory_circuit(
                    number_of_rounds=code.distance, logic_check=logic_check
                )
                metadata = {
                    "name": code.name,
                    "error": prob_error,
                    "n": int(code.num_data_qubits),
                    "k": int(code.num_logical_qubits),
                    "d": int(code.distance),
                }
                yield sinter.Task(circuit=code.memory_circuit, json_metadata=metadata)

    def collect_stats(
        self,
        num_workers: int | None = None,
        max_shots: int = 10**4,
        max_errors: int = 1000,
        decoder_params: dict[str, any] | None = None,
        logic_check: str = "Z",
    ) -> None:
        r"""
        Collect sampling statistics over ranges of distance and errors.

        :param num_shots: The number of samples.
        :param max_shots: Maximum number of shots.
        :param max_errors: Maximum tolerated errors.
        :param decoder_params: The optional decoder parameters.
        """
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() - 1

        if __available_decoders__[self.decoder] is not None:

            # if decoder_params is None:
            #     custom_decoder = {self.decoder: __available_decoders__[self.decoder]()}
            # else:
            #     custom_decoder = {
            #         self.decoder: __available_decoders__[self.decoder](**decoder_params)
            #     }

            self._samples = sinter.collect(
                num_workers=num_workers,
                max_shots=max_shots,
                max_errors=max_errors,
                tasks=self.generate_sinter_tasks(logic_check=logic_check),
                decoders=[self.decoder],
                custom_decoders=decoder_params,
            )
        else:
            self._samples = sinter.collect(
                num_workers=num_workers,
                max_shots=max_shots,
                max_errors=max_errors,
                tasks=self.generate_sinter_tasks(logic_check=logic_check),
                decoders=[self.decoder],
            )

    def plot_stats(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        pseudo_threshold: bool = False,
    ) -> None:
        r"""
        Plot the collected data

        :param x_min: The x_min for the plot.
        :param x_max: The x_max for the plot.
        :param y_min: The y_min for the plot.
        :param y_max: The y_max for the plot.
        """

        # Render a matplotlib plot of the data.
        fig, ax = plt.subplots(1, 1)
        sinter.plot_error_rate(
            ax=ax,
            stats=self.samples,
            group_func=lambda stat: stat.json_metadata["name"],
            x_func=lambda stat: stat.json_metadata["error"],
        )

        if pseudo_threshold:
            if not self.check_is_family(self.samples):
                warnings.warn("Inconsistent sample families detected.")
            else:
                k = self.samples[0].json_metadata["k"]
                ax.plot(self.error_rates, k * self.error_rates)

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.loglog()
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.grid(which="major")
        ax.grid(which="minor")
        ax.legend()

        plt.show()

    @staticmethod
    def check_is_family(samples: list[sinter.TaskStats]) -> bool:
        r"""
        Check if the given samples belong to the same family.
        """
        first_name = samples[0].json_metadata["name"]
        return all(sample.json_metadata["name"] == first_name for sample in samples)
