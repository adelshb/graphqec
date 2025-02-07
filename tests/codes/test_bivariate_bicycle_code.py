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

from graphqec import BivariateBicycleCode


class TestBivariateBicycleCode:

    @pytest.fixture(autouse=True)
    def init(self) -> None:
        self.code = BivariateBicycleCode(
            Lx=12,
            Ly=6,
            a1=3,
            a2=1,
            a3=2,
            b1=3,
            b2=1,
            b3=2,
            depolarize1_rate=0.01,
            depolarize2_rate=0,
        )

    def test_init(self):
        assert isinstance(self.code, BivariateBicycleCode)
        assert self.code.Lx == 12
        assert self.code.Ly == 6
        assert self.code.a1 == 3
        assert self.code.a2 == 1
        assert self.code.a3 == 2
        assert self.code.b1 == 3
        assert self.code.b2 == 1
        assert self.code.b3 == 2
        assert self.code.depolarize1_rate == 0.01
        assert self.code.depolarize2_rate == 0
        # assert self.code.name == "Bivariate Bicycle"

    def test_build_memory_circuit(self):
        self.code.build_memory_circuit(number_of_rounds=2)
        assert type(self.code.memory_circuit) == Circuit
