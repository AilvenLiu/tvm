# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest

from tvm.script import tir as T
from tvm.tir.analysis import verify_tirp_well_formed as verify


def test_exec_scope_verifier():
    # fmt: off
    @T.prim_func(tirp=True)
    def test() -> None:
        with T.thread():
            pass
    # fmt: on

    with pytest.raises(Exception, match="invalid exec_scope"):
        verify(test)


if __name__ == "__main__":
    test_exec_scope_verifier()
