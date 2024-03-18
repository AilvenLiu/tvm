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


def test_tensor_lowering():
    # fmt: off
    @T.prim_func
    def test(out_ptr: T.handle) -> None:
        out = T.match_buffer(out_ptr, (2), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.block_id([32, 32, 1], parent="kernel")
            tx, ty, tz = T.thread_id([16, 8, 1], parent="block")
            warp_id = T.warp_id([4], parent="block")
            lane_id = T.thread_id([32], parent="warp")            

            with T.block():
                A = T.alloc_buffer([2,], dtype="float16", scope="local")

                out[0] = A[0]
    # fmt: on

    print(tvm.lower(test))


if __name__ == "__main__":
    test_tensor_lowering()
