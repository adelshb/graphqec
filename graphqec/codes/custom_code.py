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

from networkx import Graph

from graphqec.codes.base_code import BaseCode

__all__ = ["CustomCode"]


class CustomCode(BaseCode):
    r"""
    A class for Custom code.
    """

    __slots__ = "_custom_graph"

    def __init__(
        self,
        graph: Graph,
        logic_check: list[int] = [0],
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the Custom code instance.
        """

        self._logic_check = logic_check

        self._custom_graph = graph
        super().__init__(*args, **kwargs)

        self._name = "CustomCode"

    @property
    def custom_graph(self) -> Graph:
        "Return the given graph."
        return self._custom_graph

    def build_graph(self) -> None:
        r"""
        Build the graph for the custom code
        """
        self._graph = self.custom_graph
