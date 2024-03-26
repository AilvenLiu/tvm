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


def test_roundtrip_scopeid():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.block_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="block")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                with T.warp():
                    with T.thread():
                        A_local = T.alloc_buffer([1], dtype="float16", 
                                                 scope="local", logical_scope="thread")
                        for i in T.serial(2):
                            A_local[0] = A[lane_id * 2 + i]
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code


def test_roundtrip_exec_scope():
    @T.prim_func(tirp=True, check_well_formed=False)
    def test():
        with T.world():
            kid = T.kernel_id(2)
            with T.kernel():
                bx, by, bz = T.block_id([32, 32, 1], parent="kernel")
                tx, ty, tz = T.thread_id([16, 8, 1], parent="block")
                warp_id = T.warp_id([4], parent="block")
                lane_id = T.thread_id([32], parent="warp")
                with T.cta():
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


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @T.prim_func
    def test() -> None:
        with T.block():
            A = T.alloc_buffer([2], dtype="float16", scope="local", logical_scope="thread")
            with T.block():
                A_local = T.get(A, T.Buffer(2, dtype="float16", scope="local", logical_scope="thread"))
                A_local[0] = T.float16(0)
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def test(out_ptr: T.handle) -> None:
        out = T.match_buffer(out_ptr, (2), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.block_id([32, 32, 1], parent="kernel")
            tx, ty, tz = T.thread_id([16, 8, 1], parent="block")
            warp_id = T.warp_id([4], parent="block")
            lane_id = T.thread_id([32], parent="warp")

            with T.block():
                A = T.alloc_buffer([2,], dtype="float16", scope="local")
                B = T.alloc_buffer([8, 8], dtype="float16", 
                                   scope="local", logical_scope="warp", 
                                   layout=T.TileLayout([], []))
                C = T.view(A, T.TileLayout([], []), T.Buffer((2,), dtype="float16"))
                D = T.get(B, T.Buffer((2,), dtype="float16"))
                out[0] = A[0] + B[0, 0] + D[0]
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code


if __name__ == "__main__":
    test_roundtrip_scopeid()
    test_roundtrip_exec_scope()
    test_roundtrip_buffer_view_get1()
    test_roundtrip_buffer_view_get2()
