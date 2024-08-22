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
import numpy as np
import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.function import PrimFunc
from tvm.tir.transform import LowerTIRp


def compare(before, after, transform):
    if isinstance(before, PrimFunc):
        before = tvm.IRModule({"main": before})
    if isinstance(after, PrimFunc):
        after = tvm.IRModule({"main": after})
    assert isinstance(before, tvm.IRModule)
    assert isinstance(after, tvm.IRModule)

    with tvm.target.Target("cuda"):
        tvm.ir.assert_structural_equal(transform()(before), after, map_free_vars=False)


def test_lower_view_get():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before1(in_buf: T.Buffer(64, "float32"), out: T.Buffer(64, "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                A = T.alloc_buffer([2], dtype="float16", scope="local",
                                   layout=T.TileLayout.from_nested_tuple((2,), (1,)))
                B_layout = T.TileLayout.shard((64,), (32,), "S0", inner=A.layout, from_to=("thread", "warp"))
                """
                load in_buf into A
                """
                with T.warp():
                    # warp view of this load
                    B = T.view(A, layout=B_layout) # TODO(@bohan): consider making view API directly accepts shard parameters
                    # B[i] = in_buf[i]
                    with T.thread():
                        # done by each thread
                        A_local = T.get(B)
                        for i in T.vectorized(2):
                            A_local[i] = T.float32(in_buf[lane_id * 2 + i])
                """
                write A into out
                """
                with T.warp():
                    # warp view of this write
                    B = T.view(A, layout=B_layout)
                    # out[i] = B[i]
                    with T.thread():
                        # done by each thread
                        A_local = T.get(B)
                        for i in T.vectorized(2):
                            out[lane_id * 2 + i] = T.float32(A_local[i])

    @T.prim_func(private=True, tirp=True)
    def after1(in_buf: T.Buffer(64, "float32"), out: T.Buffer(64, "float32")) -> None:
        with T.kernel():
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
            threadIdx_x = T.launch_thread("threadIdx.x", 32)
            blockIdx_y = T.launch_thread("blockIdx.y", 1)
            blockIdx_z = T.launch_thread("blockIdx.z", 1)
            with T.thread():
                A = T.alloc_buffer((2,), "float16", scope="local")
                with T.warp():
                    with T.thread():
                        for i in T.vectorized(2):
                            A[i] = T.Cast("float16", in_buf[threadIdx_x * 2 + i])
                with T.warp():
                    with T.thread():
                        for i in T.vectorized(2):
                            out[threadIdx_x * 2 + i] = T.Cast("float32", A[i])
    # fmt: on

    compare(before1, after1, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before2(in_buf: T.Buffer((16, 16), "float32"), out: T.Buffer((16, 16), "float32")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                atom = T.TileLayout.from_nested_tuple((1, 2), (2, 1))
                tile = T.TileLayout.from_nested_tuple((2, 2), (2, 1))
                warp_atom = T.TileLayout.shard(
                    (8, 8), (8, 4), "S0S1", inner=atom, from_to=("thread", "warp")
                )
                A = T.alloc_buffer([4, 2], dtype="float32", scope="local",
                                   layout=T.TileLayout.tile(tile, atom))
                B_layout = T.TileLayout.tile(tile, warp_atom)
                """
                load in_buf into A
                """
                with T.warp():
                    # warp view of this load
                    B = T.view(A, layout=B_layout)
                    with T.thread():
                        # done by each thread
                        A_local = T.get(B)
                        for i in T.unroll(4):
                            for j in T.vectorized(2):
                                A_local[i, j] = in_buf[i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j]
                """
                write A into out
                """
                with T.warp():
                    # warp view of this write
                    B = T.view(A, layout=B_layout)
                    with T.thread():
                        # done by each thread
                        A_local = T.get(B)
                        for i in T.vectorized(2):
                            out[lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2] = A_local[i // 2, i % 2]

    @T.prim_func(private=True, tirp=True)
    def after2(in_buf: T.Buffer((16, 16), "float32"), out: T.Buffer((16, 16), "float32")):
        with T.kernel():
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
            threadIdx_x = T.launch_thread("threadIdx.x", 32)
            blockIdx_y = T.launch_thread("blockIdx.y", 1)
            blockIdx_z = T.launch_thread("blockIdx.z", 1)
            with T.thread():
                A = T.alloc_buffer((8,), scope="local", logical_scope="thread")
                with T.warp():
                    with T.thread():
                        for i in T.unroll(4):
                            for j in T.vectorized(2):
                                in_buf_1 = T.Buffer((256,), data=in_buf.data, logical_scope="kernel")
                                A[i * 2 + j] = in_buf_1[i // 2 * 128 + threadIdx_x % 32 // 4 * 16 + i % 2 * 8 + j + threadIdx_x % 4]
                with T.warp():
                    with T.thread():
                        for i in T.vectorized(2):
                            out_1 = T.Buffer((256,), data=out.data, logical_scope="kernel")
                            out_1[threadIdx_x % 32 // 4 * 128 + i // 2 * 128 + threadIdx_x % 4 * 18 + i % 2] = A[i]

    # fmt: on

    compare(before2, after2, LowerTIRp)


def test_lower_scope_id():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before1() -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([3, 4, 5], parent="kernel")
            tx = T.thread_id([32], parent="cta")

            with T.thread():
                T.evaluate(bx + by + bz + tx)

    @T.prim_func(private=True, tirp=True)
    def after1() -> None:
        with T.kernel():
            blockIdx_x = T.launch_thread("blockIdx.x", 3)
            threadIdx_x = T.launch_thread("threadIdx.x", 32)
            blockIdx_y = T.launch_thread("blockIdx.y", 4)
            blockIdx_z = T.launch_thread("blockIdx.z", 5)
            with T.thread():
                T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x)
    # fmt: on
    compare(before1, after1, LowerTIRp)

    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before2() -> None:
        with T.kernel():
            cbx, cby, cbz = T.cta_id([2, 2, 2], parent="cluster")
            bx, by, bz = T.cta_id([8, 8, 8], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.thread():
                T.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    @T.prim_func(private=True, tirp=True)
    def after2() -> None:
        with T.kernel():
            clusterCtaIdx_x = T.launch_thread("clusterCtaIdx.x", 2)
            clusterCtaIdx_y = T.launch_thread("clusterCtaIdx.y", 2)
            clusterCtaIdx_z = T.launch_thread("clusterCtaIdx.z", 2)
            blockIdx_x = T.launch_thread("blockIdx.x", 8)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            blockIdx_y = T.launch_thread("blockIdx.y", 8)
            blockIdx_z = T.launch_thread("blockIdx.z", 8)
            with T.thread():
                T.evaluate(blockIdx_x + blockIdx_y + blockIdx_z + threadIdx_x // 32 + threadIdx_x % 32 + clusterCtaIdx_x + clusterCtaIdx_y + clusterCtaIdx_z)

    # fmt: on
    compare(before2, after2, LowerTIRp)


def test_lower_layout():
    # fmt: off
    @T.prim_func(private=True, tirp=True)
    def before(A: T.Buffer((128, 32), "float16")) -> None:
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            tid = T.thread_id([128], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer([128, 32], dtype="float16", scope="shared",
                                        layout=T.SwizzleLayout(3, 3, 3))

                with T.thread():
                    thread_col = T.meta_var(4)
                    thread_row = T.meta_var(32)

                    for tile in T.serial(128 // thread_row):
                        row = T.meta_var(tile * thread_row + tid // thread_col)
                        col = T.meta_var(tid % thread_col * 8)
                        for vec in T.vectorized(8):
                            A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    @T.prim_func(private=True, tirp=True)
    def after(A: T.Buffer((128, 32), "float16")) -> None:
        with T.kernel():
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            blockIdx_y = T.launch_thread("blockIdx.y", 1)
            blockIdx_z = T.launch_thread("blockIdx.z", 1)
            with T.cta():
                A_smem = T.alloc_buffer((4096,), "float16", scope="shared", logical_scope="cta")
                with T.thread():
                    for tile in range(4):
                        for vec in T.vectorized(8):
                            A_1 = T.Buffer((4096,), "float16", data=A.data, logical_scope="kernel")
                            A_smem[T.shift_left(T.bitwise_xor(tile * 128 + threadIdx_x, T.shift_right(T.bitwise_and(tile * 128 + threadIdx_x, 56), 3)), 3) + vec] = A_1[tile * 1024 + threadIdx_x * 8 + vec]
    # fmt: on

    compare(before, after, LowerTIRp)


def test_lower_opcall_fail():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                A_smem = T.alloc_buffer([64], dtype="float32", scope="shared")

                Tp.copy(A[0:64], A_smem[0:64])
                for i in range(10):
                    Tp.fill(A_smem[0:64], T.float32(0))
                    Tp.gemm(A_smem, A_smem, A_smem, A_smem)
                Tp.copy(A_smem[0:64], A[0:64])
    # fmt: on

    with pytest.raises(Exception):
        LowerTIRp()(tvm.IRModule({"main": test}))


if __name__ == "__main__":
    test_lower_view_get()
    test_lower_scope_id()
    test_lower_layout()
    test_lower_opcall_fail()
