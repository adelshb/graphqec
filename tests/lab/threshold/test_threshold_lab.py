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

import pytest

import numpy as np

from graphqec import RepetitionCode, ThresholdLAB


class TestRepetitionCode:

    @pytest.fixture(autouse=True)
    def init(self) -> None:
        self.th = ThresholdLAB(
            configurations=[{"distance": d} for d in [3, 5]],
            code=RepetitionCode,
            error_rates=np.linspace(0, 0.1, 10),
        )

    def test_init(self):
        assert self.th.configurations == [{"distance": d} for d in [3, 5]]
        assert self.th.code == RepetitionCode
        assert (self.th.error_rates == np.linspace(0, 0.1, 10)).all()
        assert self.th.collected_stats == {}

    def test_compute_logical_errors(self):

        rep = RepetitionCode(distance=3, depolarize1_rate=0, depolarize2_rate=0)
        rep.build_memory_circuit(number_of_rounds=1)

        num_errors_sampled = self.th.compute_logical_errors(code=rep, num_shots=10)
        assert num_errors_sampled == 0

    def test_collect_states(self):
        self.th.collect_stats(num_shots=10)

        assert list(self.th.collected_stats.keys()) == [
            "Repetition [[5,1,3]]",
            "Repetition [[9,1,5]]",
        ]
        assert isinstance(self.th.collected_stats["Repetition [[5,1,3]]"], list)
        assert isinstance(self.th.collected_stats["Repetition [[5,1,3]]"][0], float)
        assert len(self.th.collected_stats["Repetition [[5,1,3]]"]) == 10
