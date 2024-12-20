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
from abc import abstractmethod

from stim import Circuit, target_rec

from qec.codes.base_code import BaseCode

__all__ = ["TwoDLattice"]


class TwoDLattice(BaseCode):
    r"""
    A Parent class for 2D Lattice code.
    """
    
    __slots__ = ()
    
    def __init__(
        self,
        *args, 
        **kwargs,
    ) -> None:
        r"""
        Initialize the 2D Lattice code instance.
        """
        super().__init__(*args, **kwargs)
        
    def build_memory_circuit(self, number_of_rounds: int = 2) -> Circuit:        
        pass

    @abstractmethod
    def data_qubit_coords(self)->list[tuple]:
        r"""
        Return the coordinates of the data qubits.   
        """
        pass

    @abstractmethod
    def z_measure_coords(distance):
        r"""
        Return the coordinates of the Z measurments.
        """
        pass

    @abstractmethod
    def x_measure_coords(distance):
        r"""
        Return the coordinates of the X measurements.
        """
        pass
