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

from stim import Circuit, target_rec

from qec.codes.base_code import BaseCode

__all__ = ["ShorCode"]

class ShorCode(BaseCode):
    r"""
    A class for the Shor code.
    """

    __slots__ = ("_number_of_qubits")

    def __init__(
        self,
        *args, 
        **kwargs,
    ) -> None:
        r"""
        Initialize the Shor code instance.
        """
        super().__init__(*args, **kwargs)

        if self.distance % 3 != 0:
            raise ValueError("Distance must be a multiple of 3.")

        self._number_of_qubits = 3 * self.distance

    def build_memory_circuit(self, number_of_rounds: int = 2) -> None:
        r"""
        Build and return a Stim Circuit object implementing a memory for the given time.
        
        :param number_of_rounds: The number of rounds in the memory.
        """

        # Initialize the circuit
        self._memory_circuit = Circuit()