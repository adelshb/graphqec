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

from stim import Circuit

from graphqec import RepetitionCode


class TestRepetitionCode:

    @pytest.fixture(autouse=True)
    def init(self) -> None:
        self.code = RepetitionCode(
            distance=5, depolarize1_rate=0.01, depolarize2_rate=0
        )

    def test_init(self):
        assert isinstance(self.code, RepetitionCode)
        assert self.code.distance == 5
        assert self.code.depolarize1_rate == 0.01
        assert self.code.depolarize2_rate == 0
        assert self.code.name == "Repetition"

    def test_build_memory_circuit(self):
        self.code.build_memory_circuit(number_of_rounds=2)
        assert type(self.code.memory_circuit) == Circuit
