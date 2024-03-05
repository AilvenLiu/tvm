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
import tvm
import tvm.testing
from tvm.script import tir as T, from_source

import pytest


def test_roundtrip():
    @T.prim_func
    def test():
        with T.world():
            kid = T.kernel_id(2)
            with T.kernel():
                bx, by, bz = T.block_id([32, 32, 1], parent="kernel")
                tx, ty, tz = T.thread_id([16, 8, 1], parent="block")
                warp_id = T.warp_id([4], parent="block")
                lane_id = T.thread_id([32], parent="warp")
                with T.block():
                    with T.warp():
                        with T.thread():
                            T.evaluate(0)
                    with T.thread():
                        T.evaluate(0)
                    with T.warp([warp_id], [T.Range(0, 2)]):
                        T.evaluate(0)
                    with T.thread([lane_id], [T.Range(0, 16)]):
                        T.evaluate(0)

    code = test.script()
    assert from_source(code).script() == code


if __name__ == "__main__":
    tvm.testing.main()
