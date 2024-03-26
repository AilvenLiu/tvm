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
    def test(in_ptr: T.handle, out_ptr: T.handle) -> None:
        in_buf = T.match_buffer(in_ptr, (64), "float32", scope="global")
        out = T.match_buffer(out_ptr, (64), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.block_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="block")
            lane_id = T.thread_id([32], parent="warp")            

            with T.thread():
                A = T.alloc_buffer([2], dtype="float16", 
                                    scope="local", logical_scope="thread")
                with T.warp():
                    B = T.view(A, layout = None)
                    """
                    B = in_buf
                    out = B
                    """
                    with T.thread():
                        A_local = T.get(B, ...)
                        for i in T.vectorized(2):
                            A_local[i] = T.float32(in_buf[lane_id * 2 + i])
                        for i in T.vectorized(2):
                            out[lane_id * 2 + i] = T.float32(A_local[i])
    # fmt: on

    test.show(black_format=False)
    tvm.lower(test).show(black_format=False)


if __name__ == "__main__":
    test_tensor_lowering()
