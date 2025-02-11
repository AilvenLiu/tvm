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
from tvm.script import tirp as Tp
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
                    with T.warp()[0:2]:
                        with T.thread():
                            T.evaluate(0)
                    with T.thread([128])[0:2]:
                        T.evaluate(0)
                    with T.thread([16, 8])[0:8, 0:4]:
                        T.evaluate(0)
                    with T.thread()[T.elect_sync(0xFFFFFFFF)]:
                        T.evaluate(0)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code


def test_roundtrip_layout():
    def get_layout1():
        return T.TileLayout.from_tuple(
            data=(8, T.S(0), 8, T.S(1), 2),
            strides=(6, -1, 2, -1, 1),
            device=(8, 4),
            from_to=("thread", "warp"),
        )

    def get_layout2():
        return T.TileLayout.from_tuple(
            data=(8, T.S(0), 8, 4, 2),
            strides=(64, -1, 8, 2, 1),
            device=(8, 4),
            exclusive=[(1, 0)],
            from_to=("thread", "warp"),
        )

    def get_layout3():
        return T.TileLayout.from_tuple(
            data=(8, 16, 8, 16),
            strides=(1024, 16, 128, 1),
        )

    def get_layout4():
        return T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([1], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            C = T.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
            D = T.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())

            with T.cta():
                A_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
                B_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())

                with T.thread():
                    T.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0])
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
                A_layout = T.TileLayout.from_tuple((1, 2), (2, 1))
                A_warp_layout = T.TileLayout.shard(
                    (8, 8), (8, 4), "S0S1", inner=A_layout, from_to=("thread", "warp")
                )
                A_warp = T.view(A, layout=A_warp_layout, shape=(8, 8))

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
                A_layout = T.TileLayout.from_tuple((1, 2), (2, 1))
                B_layout = T.TileLayout.shard(
                    (8, 8), (8, 4), "S0S1", inner=A_layout, from_to=("thread", "warp")
                )
                B = T.view(A, layout=B_layout, shape=(8, 8))
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


def test_roundtrip_op1():
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

                Tp.copy(A_smem, A)
                for i in range(10):
                    Tp.fill(A_smem, T.float32(0))
                    Tp.gemm(A_smem, A_smem, A_smem, A_smem)
                Tp.copy(A, A_smem)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op2():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, 128), "float16", scope="global")
        B = T.match_buffer(B_ptr, (128, 64), "float16", scope="global")
        C = T.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            with T.cta():
                A_smem = T.alloc_buffer([128, 32], dtype="float16", scope="shared")
                B_smem = T.alloc_buffer([32, 64], dtype="float16", scope="shared")

                C_local = T.alloc_buffer([128, 64], dtype="float32", scope="local")
                for k in range(4):
                    Tp.copy(A_smem, A[:, k * 32 : k * 32 + 32])
                    Tp.copy(B_smem, B[k * 32 : k * 32 + 32, 0:64])
                    Tp.gemm(C_local, A_smem, B_smem, C_local)
                Tp.copy(C, C_local)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op3():
    # fmt: off
    NUM_STAGES = 3
    K = 4096

    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, K), "float16", scope="global")
        B = T.match_buffer(B_ptr, (K, 64), "float16", scope="global")
        C = T.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.cta():
                A_smem = T.alloc_buffer([NUM_STAGES, 128, 32], dtype="float16", scope="shared")
                B_smem = T.alloc_buffer([NUM_STAGES, 32, 64], dtype="float16", scope="shared")

                C_local = T.alloc_buffer([128, 64], dtype="float32", scope="local")
                for i in range(NUM_STAGES - 1):
                    Tp.copy(A_smem[i, :, :], A[:, i * 32 : i * 32 + 32])
                    Tp.copy(B_smem[i, :, :], B[i * 32 : i * 32 + 32, :])

                for k in range(K // 32):
                    copy_k = T.meta_var(k + NUM_STAGES - 1)
                    gemm_stage = T.meta_var(k % NUM_STAGES)
                    copy_stage = T.meta_var(copy_k % NUM_STAGES)
                    Tp.copy(A_smem[copy_stage, :, :], A[:, copy_k * 32 : copy_k * 32 + 32])
                    Tp.copy(B_smem[copy_stage, :, :], B[copy_k * 32 : copy_k * 32 + 32, :])
                    Tp.gemm(C_local, A_smem[gemm_stage, :, :], B_smem[gemm_stage, :, :], C_local)

                Tp.copy(C, C_local)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_pipeline_no_specialize_async_no_depth():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (4096,), "float32", scope="global")
        
        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            warp_id = T.warp_id([4], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            
            with T.cta():
                A_smem = T.alloc_buffer([4096], dtype="float32", scope="shared")
                pipe = Tp.alloc_copy_pipeline(thread_scope="cta", depth=0, separate_pc=False)
                pipe.copy(A_smem[0:128], A[0:128])
                pipe.producer_commit()

                for i in range(1, 16):
                    pipe.copy(A_smem[i * 128 : (i + 1) * 128], A[i * 128 : (i + 1) * 128])
                    pipe.producer_commit()

                    pipe.consumer_wait(num_stages=1)
                    Tp.fill(A_smem[(i - 1) * 128 : i * 128], T.float32(0))
                
                pipe.consumer_wait(num_stages=0)

                Tp.fill(A_smem[15 * 128 : 4096], T.float32(0))
                Tp.copy(A, A_smem)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_pipeline_specialize_sync_depth():
    # fmt: off
    DEPTH = 3
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (32, 4096), "float32", scope="global")
        B = T.match_buffer(B_ptr, (32, 4096), "float32", scope="global")
        C = T.match_buffer(C_ptr, (32, 32), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            tid = T.thread_id([128], parent="cta")
            
            with T.cta():
                A_smem = T.alloc_buffer([DEPTH, 128, 32], dtype="float32", scope="shared")
                B_smem = T.alloc_buffer([DEPTH, 128, 32], dtype="float32", scope="shared")
                C_local = T.alloc_buffer([32, 32], dtype="float32", scope="local")

                pipe = Tp.alloc_copy_pipeline(thread_scope="cta", depth=DEPTH, separate_pc=True)
                
                with T.thread()[0:64]:
                    for i in range(32):
                        pipe.producer_acquire()
                        j = T.meta_var(i % DEPTH)
                        Tp.copy(A_smem[j, 0:32, 0:128], A[i*32, 0:128])
                        Tp.copy(B_smem[j, 0:32, 0:128], B[i*32, 0:128])
                        pipe.producer_commit()
                with T.thread()[64:128]:
                    for i in range(32):
                        pipe.consumer_wait()
                        j = T.meta_var(i % DEPTH)
                        Tp.gemm(A_smem[j, 0:32, 0:128], B_smem[j, 0:32, 0:128], C_local, C_local)
                        pipe.consumer_release()

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_tensormap():
    # fmt: off
    @T.prim_func(tirp=True)
    def func1(A_ptr: T.handle):
        T.func_attr({"global_symbol": "func"})
        A = T.match_buffer(A_ptr, [128], "float32")
    
        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.tensormap_init", A_map, A_ptr)
    # fmt: on
    code = func1.script()
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


if __name__ == "__main__":
    test_roundtrip_scopeid()
    test_roundtrip_exec_scope()
    test_roundtrip_layout()
    test_roundtrip_buffer_view_get1()
    test_roundtrip_buffer_view_get2()
    test_alloc_buffer_default_logical_scope()
    test_roundtrip_op1()
    test_roundtrip_op2()
    test_roundtrip_op3()
    test_roundtrip_pipeline_no_specialize_async_no_depth()
    test_roundtrip_pipeline_specialize_sync_depth()
    test_roundtrip_tensormap()
