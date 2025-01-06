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

__all__ = ["Measurement"]


class Measurement:
    r"""
    A class for collection of measurement outcomes.
    """

    __slots__ = ("_data", "_register_count")

    def __init__(self) -> None:

        self._data = {}
        self._register_count = 0

    @property
    def data(self) -> dict:
        r"""
        The collection of outcomes.
        """
        return self._data

    @property
    def register_count(self) -> int:
        r"""
        The number of outcome collected.
        """
        return self._register_count

    def get_outcome(
        self,
        qubit: any,
        round: int,
    ) -> any:
        r"""
        Return the outcome for the qubit at the specified round or return None.

        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        """

        try:
            return self.data[round][qubit]["outcome"]
        except KeyError:
            return None

    def get_register_id(
        self,
        qubit: any,
        round: int,
    ) -> any:
        r"""
        Return the register_id for the qubit at the specified round or return None.

        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        """

        try:
            return self.data[round][qubit]["register_id"]
        except KeyError:
            return None

    def add_outcome(
        self, outcome: any, qubit: any, round: int, type: str | None
    ) -> None:
        r"""
        Add an outcome to the collection.

        :param outcome: The outcome to store.
        :param qubit: The qubit on which the measurement is performed.
        :param round: The round during which the measurement is performed.
        :param type: The type of measurement.
        """

        if type not in ["check", "data", None]:
            ValueError("The value of type must be either 'check' or 'data'.")

        if round not in self.data.keys():
            self._data[round] = {}

        self._data[round][qubit] = {
            "outcome": outcome,
            "type": type,
            "register_id": self.register_count,
        }

        self._register_count += 1
