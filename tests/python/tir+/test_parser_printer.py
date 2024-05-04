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
import tvm.script
import tvm.testing
from tvm.script import tir as T
from tvm.ir import assert_structural_equal

import pytest


def from_source(code):
    return tvm.script.from_source(code, tirp=True)


def test_roundtrip_scopeid():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
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
    assert_structural_equal(test, from_source(code))


def test_roundtrip_exec_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.world():
            kid = T.kernel_id(2)
            with T.kernel():
                bx, by, bz = T.cta_id([32, 32, 1], parent="kernel")
                tx, ty, tz = T.thread_id([16, 8, 1], parent="cta")
                warp_id = T.warp_id([4], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                with T.cta():
                    with T.warp():
                        with T.thread():
                            T.evaluate(0)
                    with T.thread():
                        T.evaluate(0)
                    with T.warp([warp_id], [T.Range(0, 2)]):
                        with T.thread():
                            T.evaluate(0)
                    with T.thread([lane_id], [T.Range(0, 16)]):
                        T.evaluate(0)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code


def test_roundtrip_layout():
    def get_layout1():
        return T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (T.S(1), 2))),
            strides=((16, -1), (2, (-1, 1))),
            device=(8, 4),
            from_to=("thread", "warp"),
        )

    def get_layout2():
        return T.TileLayout.from_nested_tuple(
            data=((8, T.S(0)), (8, (4, 2))),
            strides=((64, -1), (8, (2, 1))),
            device=(8, 4),
            exclusive=[(1, 0)],
            from_to=("thread", "warp"),
        )

    def get_layout3():
        return T.TileLayout.from_nested_tuple(
            data=((8, 16), (8, 16)),
            strides=((1024, 16), (128, 1)),
        )

    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            
            C = T.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
            
            with T.cta():
                A_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
                B_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())
                with T.thread():
                    T.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0])
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @T.prim_func(tirp=True)
    def test() -> None:
        with T.kernel():
            with T.cta():
                A = T.alloc_buffer([2], dtype="float16", scope="local", logical_scope="thread")
                A_layout = T.TileLayout.from_nested_tuple((1, 2), (2, 1))
                A_warp_layout = T.TileLayout.shard(
                    (8, 8), (8, 4), "S0S1", inner=A_layout, from_to=("thread", "warp")
                )
                A_warp = T.view(A, layout=A_warp_layout)

                with T.thread():
                    A_local = T.get(A_warp)
                    A_local[0] = T.float16(0)

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(out_ptr: T.handle) -> None:
        out = T.match_buffer(out_ptr, (2), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([32, 32, 1], parent="kernel")
            tx, ty, tz = T.thread_id([16, 8, 1], parent="cta")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.cta():
                A = T.alloc_buffer([2,], dtype="float16", scope="local", logical_scope="thread")
                A_layout = T.TileLayout.from_nested_tuple((1, 2), (2, 1))
                B_layout = T.TileLayout.shard(
                    (8, 8), (8, 4), "S0S1", inner=A_layout, from_to=("thread", "warp")
                )
                B = T.view(A, layout=B_layout)
                D = T.get(B)

                with T.thread():
                    out[0] = A[0] + B[0, 0] + D[0]
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_alloc_buffer_default_logical_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test() -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                A = T.alloc_buffer([1], dtype="float16", scope="local")
                B = T.alloc_buffer([1], dtype="float16", scope="shared")
                C = T.alloc_buffer([1], dtype="float16", scope="global")
                with T.warp():
                    with T.thread():
                        pass
    # fmt: on

    body = test.body.block.body.block
    A = body.alloc_buffers[0]
    B = body.alloc_buffers[1]
    C = body.alloc_buffers[2]

    assert A.logical_scope() == "thread"
    assert B.logical_scope() == "cta"
    assert C.logical_scope() == "kernel"


if __name__ == "__main__":
    test_roundtrip_scopeid()
    test_roundtrip_exec_scope()
    test_roundtrip_layout()
    test_roundtrip_buffer_view_get1()
    test_roundtrip_buffer_view_get2()
    test_alloc_buffer_default_logical_scope()
