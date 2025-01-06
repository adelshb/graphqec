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

import stim

from graphqec import RotatedSurfaceCode, Measurement


class TestRotatedSurfaceCode:

    @pytest.fixture(autouse=True)
    def init(self) -> None:
        self.code = RotatedSurfaceCode(
            distance=3, depolarize1_rate=0.01, depolarize2_rate=0
        )

    def test_build_memory(self):
        self.code.build_memory_circuit(number_of_rounds=2)
        assert type(self.code.memory_circuit) == stim.Circuit

    def test_measurement(self):
        assert isinstance(self.code.measurement, Measurement)
        assert isinstance(self.code.register_count, int)
        assert self.code.get_outcome(qubit=0, round=0) == None

        self.code.add_outcome(outcome="0", qubit=0, round=0, type="check")
        assert isinstance(self.code.get_target_rec(qubit=0, round=0), int)
        assert self.code.get_target_rec(qubit=1000, round=0) == None
