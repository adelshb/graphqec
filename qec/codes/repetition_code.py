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

from qec.codes.base_code import BaseCode

__all__ = ["RepetitionCode"]


class RepetitionCode(BaseCode):
    r"""
    A class for Repetition code.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the Repetition code instance.
        """

        self._checks = ["Z-check"]

        self._logic_check = [0]

        super().__init__(*args, **kwargs)

    def build_graph(self) -> None:
        r"""
        Build the graph for the repetition code
        """

        data = [(i, {"type": "data"}) for i in range(self.distance)]
        self._graph.add_nodes_from(data)

        check = [
            (i + self.distance, {"type": "Z-check"}) for i in range(self.distance - 1)
        ]
        self._graph.add_nodes_from(check)

        edges = [(i, i + self.distance, 1) for i in range(self.distance - 1)]
        self._graph.add_weighted_edges_from(edges)

        edges = [(i, i + self.distance - 1, 2) for i in range(1, self.distance)]
        self._graph.add_weighted_edges_from(edges)
