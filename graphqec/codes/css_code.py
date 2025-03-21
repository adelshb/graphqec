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

from graphqec.codes.base_code import BaseCode

__all__ = ["CssCode"]


def commutation_test(Hx: list[list[int]], Hz: list[list[int]]) -> bool:
    r"""
    function for taking two linear codes and determining if they satisfy
    the necessary constraints for defining a CSS code
    """

    return all(
        [(not sum([a * b for (a, b) in zip(hx, hz)]) % 2) for hx in Hx for hz in Hz]
    )


class CssCode(BaseCode):
    r"""
    A class for a generic CSS code specified by a pair of linear codes subject
    to a constraint to ensure commutativity of the generators in the
    associated quantum code.
    """

    def __init__(
        self,
        Hx: list[list[int]],
        Hz: list[list[int]],
        *args,
        **kwargs,
    ) -> None:
        r"""
        Initialize the CSS code instance.
        """

        # self._logic_check = [0]
        self.Hx = Hx
        self.Hz = Hz

        super().__init__(*args, **kwargs)

        self._name = (
            f"CSS [[{2*self.distance-1},{len(self._logic_check)},{self.distance}]]"
        )

    def build_graph(self) -> None:
        r"""
        Build the Tanner graph for the CSS code
        """

        pass

        # self._graph.add_nodes_from(
        #     [
        #         (i, {"type": "data", "label": None, "coords": (i, i)})
        #         for i in range(self.distance)
        #     ]
        # )
        # self._graph.add_nodes_from(
        #     [
        #         (
        #             i + self.distance,
        #             {"type": "check", "label": "Z", "coords": (i + 0.5, i + 0.5)},
        #         )
        #         for i in range(self.distance - 1)
        #     ]
        # )
        # self._graph.add_weighted_edges_from(
        #     [(i, i + self.distance, 1) for i in range(self.distance - 1)]
        # )
        # self._graph.add_weighted_edges_from(
        #     [(i, i + self.distance - 1, 2) for i in range(1, self.distance)]
        # )
