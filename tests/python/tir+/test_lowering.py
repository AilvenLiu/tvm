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
from tvm.script import tir as T
from tvm.tir.function import PrimFunc
from tvm.tir.transform import LowerTIRp


def compare(before, after, transform):
    if isinstance(before, PrimFunc):
        before = tvm.IRModule({"main": before})
    if isinstance(after, PrimFunc):
        after = tvm.IRModule({"main": after})
    assert isinstance(before, tvm.IRModule)
    assert isinstance(after, tvm.IRModule)
    tvm.ir.assert_structural_equal(transform()(before), after, map_free_vars=False)


def test_lowering1():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(in_ptr: T.handle, out_ptr: T.handle) -> None:
        in_buf = T.match_buffer(in_ptr, (64), "float32", scope="global")
        out = T.match_buffer(out_ptr, (64), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")            

            with T.thread():
                A = T.alloc_buffer([2], dtype="float16", scope="local", logical_scope="thread")
                A_layout = T.TileLayout.from_nested_tuple((2,), (1,))
                B_layout = T.TileLayout.shard((64,), (32,), "S0", inner=A_layout, from_to=("thread", "warp"))
                """
                B = in_buf
                """
                with T.warp():
                    T.reads(in_buf[:])
                    T.writes(A[:])
                    
                    B = T.view(A, layout=B_layout)
                    with T.thread():
                        A_local = T.get(B)
                        for i in T.vectorized(2):
                            A_local[i] = T.float32(in_buf[lane_id * 2 + i])
                """
                out_buf = B * 2
                """
                with T.warp():
                    T.reads(A[:])
                    T.writes(out[:])
                    
                    B = T.view(A, layout=B_layout)
                    with T.thread():
                        A_local = T.get(B)
                        for i in T.vectorized(2):
                            out[lane_id * 2 + i] = T.float32(A_local[i])

    @T.prim_func(private=True, tirp=True)
    def after(in_ptr: T.handle, out_ptr: T.handle) -> None:
        in_buf = T.match_buffer(in_ptr, (64), "float32", scope="global")
        out = T.match_buffer(out_ptr, (64), "float32", scope="global")

        with T.kernel():
            T.reads()
            T.writes()
            for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(32, thread="threadIdx.x"):
                    with T.thread():
                        T.reads()
                        T.writes()
                        A = T.alloc_buffer((2,), "float16", scope="local", logical_scope="thread", layout=None)
                        with T.warp():
                            T.reads(in_buf[:])
                            T.writes(A[:])
                            with T.thread():
                                T.reads()
                                T.writes()
                                for i in T.vectorized(2):
                                    A[i] = T.Cast("float16", in_buf[threadIdx % 32 * 2 + i])
                        with T.warp():
                            T.reads(A[:])
                            T.writes(out[:])
                            with T.thread():
                                T.reads()
                                T.writes()
                                for i in T.vectorized(2):
                                    out[threadIdx % 32 * 2 + i] = T.Cast("float32", A[i])

    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_scope_id():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before() -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")            

            with T.thread():
                T.evaluate(bx + by + bz)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def after() -> None:
        with T.kernel():
            for blockIdx in T.thread_binding(60, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(32, thread="threadIdx.x"):
                    with T.thread():
                        T.reads()
                        T.writes()
                        T.evaluate(blockIdx % 3 + blockIdx % 12 // 3 + blockIdx % 60 // 12)
    # fmt: on

    compare(before, after, LowerTIRp)


if __name__ == "__main__":
    test_lowering1()
    test_lower_scope_id()
