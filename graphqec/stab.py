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

from stim import Circuit


def X_check(circ: Circuit, data_qubit: any, check_qubit: any) -> None:
    r"""
    Element of the X-check measurement
    """
    circ.append("CNOT", [check_qubit, data_qubit])


def Z_check(circ: Circuit, data_qubit: any, check_qubit: any) -> None:
    r"""
    Element of the Z-check measurement
    """
    circ.append("CNOT", [data_qubit, check_qubit])


# def Y_check(circ: Circuit, data_qubit: any, check_qubit: any)->None:
#     r"""
#     Element of the Y-check measurement
#     """
