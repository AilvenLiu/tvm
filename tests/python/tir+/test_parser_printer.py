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

import tvm
import tvm.script
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.event import EventImpl


def from_source(code):
    return tvm.script.from_source(code, tirp=True)


def test_roundtrip_scopeid1():
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
                                                 scope="local")
                        for i in T.serial(2):
                            A_local[0] = A[lane_id * 2 + i]
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid2():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        with T.kernel():
            bx, by, bz = T.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = T.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = T.cluster_id([4, 5, 12], parent="kernel")
            with T.cta():
                with T.warp():
                    with T.thread():
                        T.evaluate(bx + by + bz)
                        T.evaluate(cbx + cby + cbz)
                        T.evaluate(clx + cly + clz)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_exec_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.world():
            kid = T.kernel_id([2])
            with T.kernel():
                bx, by, bz = T.cta_id([32, 32, 1], parent="kernel")
                tx, ty, tz = T.thread_id([16, 8, 1], parent="cta")
                warp_id = T.warp_id([4], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                with T.cluster():
                    with T.cta():
                        with T.warpgroup():
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
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code


def test_roundtrip_layout():
    def get_layout1():
        return T.TileLayout(
            shard=([8, 8, 8, 4, 2], [6, (4, "laneid"), 2, (1, "laneid"), 1]),
        )

    def get_layout2():
        return T.TileLayout(shard=([8, 8, 8, 4, 2], [64, (4, "laneid"), 8, 2, 1]))

    def get_layout3():
        return T.TileLayout(
            shard=([8, 16, 8, 16], [1024, 16, 128, 1]),
        )

    def get_layout4():
        return T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    def get_layout5():
        return T.ComposeLayout(
            T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            T.TileLayout(shard=([64, 64, 4], [64, 1, 64 * 64])),
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
            D = T.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())

            with T.cta():
                A_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
                B_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())

                E = T.alloc_buffer([64, 256], dtype="float16", scope="shared", layout=get_layout5())

                with T.thread():
                    T.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0] + E[0, 0])
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


L_LANE = T.TileLayout(shard=([32], [(1, "laneid")]))


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @T.prim_func(tirp=True)
    def test() -> None:
        with T.kernel():
            with T.cta():
                A = T.alloc_buffer([2], dtype="float16", scope="local")
                A_layout = T.TileLayout(shard=([1, 2], [2, 1]))
                A_warp_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
                A_warp = T.view(A, layout=A_warp_layout, shape=(8, 8))

                with T.thread():
                    A_local = T.get(A_warp, shape=(2,))
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
                A = T.alloc_buffer([2,], dtype="float16", scope="local")
                A_layout = T.TileLayout(shard=([1, 2], [2, 1]))
                B_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
                B = T.view(A, layout=B_layout, shape=(8, 8))
                D = T.get(B, shape=(2,))

                with T.thread():
                    out[0] = A[0] + B[0, 0] + D[0]
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


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


def test_roundtrip_break_for():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        with T.kernel():
            with T.cta():
                for i in T.serial(10):
                    if i > 5:
                        break
                    A[i] = i
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_while():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")
        i = T.alloc_buffer((1,), dtype="int32", scope="local")

        with T.kernel():
            with T.cta():
                i[0] = 0
                while i[0] < 10:
                    A[i[0]] = i[0] * 2
                    if A[i[0]] > 10:
                        break
                    i[0] = i[0] + 1
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_nested():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (9,), "int32")

        with T.kernel():
            with T.cta():
                idx = T.alloc_buffer((1,), "int32", scope="local")
                idx[0] = 0
                for i in T.serial(3):
                    for j in T.serial(3):
                        A[idx[0]] = i * 10 + j
                        idx[0] += 1
                        if j == 1:
                            break
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_for():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        with T.kernel():
            with T.cta():
                for i in T.serial(10):
                    if (i % 2) == 0:
                        continue
                    A[i] = i
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_while():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")
        i = T.alloc_buffer((1,), "int32", scope="local")

        with T.kernel():
            with T.cta():
                i[0] = 0
                while i[0] < 10:
                    if (i[0] % 2) == 1:
                        i[0] += 1
                        continue
                    A[i[0]] = i[0]
                    i[0] += 1
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_nested():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (9,), "int32")

        with T.kernel():
            with T.cta():
                idx = T.alloc_buffer((1,), dtype="int32", scope="local")
                idx[0] = 0
                for i in T.serial(3):
                    for j in T.serial(3):
                        if j == 1:
                            continue
                        A[idx[0]] = i * 10 + j
                        idx[0] += 1
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_and_continue():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        with T.kernel():
            with T.cta():
                for i in T.serial(10):
                    if i == 2:
                        continue
                    if i == 7:
                        break
                    A[i] = i
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unreachable_after_break():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (5,), "int32")

        with T.kernel():
            with T.cta():
                for i in T.serial(5):
                    A[i] = i
                    break
                    # This line is never reached
                    A[i] = -1
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_allocated_addr():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            A = T.alloc_buffer([10], "float32", scope="trn.sbuf", allocated_addr=1024)
            for i in T.serial(2):
                Tp.memset(A[i*5:i*5+5], T.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_implicit_buffer_region():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10, 10, 10), "float32", layout=T.TileLayout((10, 10, 10)))
        with T.kernel():
            Tp.memset(A[0], T.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_alloc_under_any_scope():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            for i in T.serial(10):
                A = T.alloc_buffer([100], "float32", scope="trn.sbuf", allocated_addr=1024)
                Tp.memset(A[i*10:i*10+10], T.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            with Tp.compose_op():
                Tp.add(B, A, T.float32(1))
                Tp.add(C, B, T.float32(1))
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_workspace():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [10], "float32", scope="global")
        B = T.match_buffer(B_ptr, [10], "float32", scope="global")
        with T.kernel():
            smem = T.alloc_buffer([10], "float32", scope="shared")
            Tp.add(B, A, T.float32(1), workspace={"smem": smem})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_workspace():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            psum = T.alloc_buffer([10], "float32", scope="trn.psum")
            intermediate = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            with Tp.compose_op(workspace={"intermediate": intermediate}):
                Tp.add(B, A, T.float32(1))
                Tp.add(C, B, T.float32(1), workspace={"psum": psum})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_schedule_config():
    # fmt: off
    @T.prim_func(tirp=True)
    def test(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [10], "float32", scope="global")
        B = T.match_buffer(B_ptr, [10], "float32", scope="global")
        with T.kernel():
            Tp.add(B, A, T.float32(1), schedule_config={"schedule": "A"})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_schedule_config():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
            psum = T.alloc_buffer([10], "float32", scope="trn.psum")
            with Tp.compose_op(schedule_config={"schedule": "A"}):
                Tp.add(B, A, T.float32(1))
                Tp.add(C, B, T.float32(1), workspace={"psum": psum})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_predicate():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            A = T.alloc_buffer([10, 10], "float32")
            B = T.alloc_buffer([10, 10], "float32")
            Tp.select(B, A, 1.0, lambda i, j: i < j)
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_grid():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            with T.thread():
                for lvs in T.grid(10, (2, 12)):
                    T.evaluate(lvs[0] + lvs[1])
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_alloc_apis():
    # fmt: off
    class Test:
        def __init__(self, Ta, inner_pool):
            self.Ta = Ta
            self.inner_pool = inner_pool
            self.Tb = T.shared_cell("float16", "Tb")
            self.idx = T.local_cell("int32", "idx")
            self.inner_pool2 = T.decl_cell("float16", self.inner_pool.data, "shared.dyn", 5, "inner_pool2")

        @T.macro
        def init(self):
            self.Ta = self.Ta + T.float16(1)
            self.Tb = self.Tb + T.float16(2)
            self.idx.buffer[()] = T.int32(0)
            self.idx = self.idx + T.int32(1)
            self.inner_pool2 = self.inner_pool2 + T.float16(1)
            T.evaluate(T.address_of(self.Ta))
            T.evaluate(T.address_of(self.Tb))
            T.evaluate(T.address_of(self.idx))
            T.evaluate(T.address_of(self.inner_pool))
            T.evaluate(T.address_of(self.inner_pool2))

    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            # normal buffer
            A = T.alloc_shared([10], "float16")
            B = T.alloc_local([10], "float16")
            # cell buffer (alloc)
            C = T.shared_cell("float16")
            D = T.local_cell("float16")
            pool = T.alloc_buffer([10], "uint8", scope="shared.dyn")
            # cell buffer (decl)
            E = T.decl_cell("float16", pool.data, "shared.dyn", 0)
            # normal 0-dim buffer
            F = T.alloc_local((), "float16")
            with T.thread():
                Ta = T.local_cell("float16")
                inner_pool = T.decl_buffer(shape=[10], data=pool.data, dtype="uint8", scope="shared.dyn")
                test = T.meta_var(Test(Ta, inner_pool))
                test.init()
                A[0] = C
                A[0] = C + D
                A[1] = B[0] * C
                D.buffer[()] = D + T.float16(1)
                D = D + T.float16(1)
                C = D
                T.evaluate(E)
                E = E + T.float16(1)
                # normal 0-dim buffer can be assigned directly,
                # but not loaded directly
                F = F[()] + T.float16(1)
                C += D
                D += E + C + D
                T.evaluate(T.address_of(C))
                T.evaluate(C.buffer.access_ptr("rw", offset=0))
                T.evaluate(C.buffer.data)
                T.evaluate(D)
                T.evaluate(T.address_of(D))
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_macro():
    # fmt: off
    @T.macro(hygienic=True)
    def mul(x, c):
        T.evaluate(x * c)

    @T.prim_func(tirp=True, private=True)
    def test():
        with T.kernel():
            for x in range(10):

                @T.macro(hygienic=True)
                def add(c):
                    T.evaluate(x + c)

                @T.macro(hygienic=False)
                def two_add_and_mul(c):
                    add(c)
                    add(c + c)
                    mul(x, c)

                two_add_and_mul(1)
                two_add_and_mul(2)
                

    @T.prim_func(tirp=True, private=True)
    def expected():
        with T.kernel():
            for x in range(10):
                T.evaluate(x + 1)
                T.evaluate(x + 2)
                T.evaluate(x)
                T.evaluate(x + 2)
                T.evaluate(x + 4)
                T.evaluate(x * 2)
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(test, expected)


def test_macro_recursive():
    # fmt: off
    @T.prim_func(tirp=True, private=True)
    def test():
        with T.kernel():
            for x in T.serial(10):

                @T.macro
                def add(x, c):
                    if c > 0:
                        add(x, c - 1)
                    T.evaluate(x)

                add(x, 5)

    @T.prim_func(private=True, tirp=True)
    def expected():
        with T.kernel():
            for x in range(10):
                T.evaluate(x)
                T.evaluate(x)
                T.evaluate(x)
                T.evaluate(x)
                T.evaluate(x)
                T.evaluate(x)
    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(expected, from_source(code))


def test_list_comprehension():
    # fmt: off
    @T.prim_func(tirp=True, private=True)
    def test():
        with T.kernel():
            with T.thread():
                acc = T.alloc_local([10], "bool")
                regs = T.meta_var([acc[_] for _ in range(10)])
                T.evaluate(regs[0])
                T.evaluate(tvm.tir.all(*regs))
                T.evaluate(tvm.tir.all(*[acc[_] for _ in range(10)]))
                T.evaluate(tvm.tir.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)])))
    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_event():
    # fmt: off
    @T.prim_func(tirp=True)
    def test():
        with T.kernel():
            with T.cta():
                event_b = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
                event_s_tensor = Tp.alloc_semaphore_event_tensor(EventImpl.kTMALoad, [], [10])

                event_b.commit()
                event_b.wait()
                event_b.wait(1)

                event_s_tensor.init(1)
                event_s_tensor[0].commit()
                event_s_tensor[0].wait()

                A = T.alloc_buffer([10], "float32", scope="shared")
                B = T.alloc_buffer([10], "float32", scope="shared")
                Tp.copy_async(A[:], B[:], event_b)
                Tp.copy_async(A[:], B[:], event_s_tensor[0])
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_range():
    # fmt: off
    @T.prim_func(tirp=True, private=True)
    def test():
        l = T.meta_var([i for i in range(10)])
        T.evaluate(l[3])
        
    @T.prim_func(tirp=True, private=True)
    def expected():
        T.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm.ir.assert_structural_equal(test, expected)


def test_buffer():
    # fmt: off
    @T.prim_func(tirp=True, private=True)
    def test(
        A: T.Buffer((10, 11), "float32", layout=None),
        B: T.Buffer((10, 11), "float32", scope="global"),
        C: T.Buffer((10, 11), "float32", layout="default"),
        D: T.Buffer((10, 11), "float32", layout=T.TileLayout(([10, 11], [1, 10]))),
        E_ptr: T.handle,
        F_ptr: T.handle,
        G_ptr: T.handle,
        H_ptr: T.handle,
    ):
        E = T.match_buffer(E_ptr, [10, 11], "float16", layout=None)
        F = T.match_buffer(F_ptr, [10, 11], "float16", scope="global")
        G = T.match_buffer(G_ptr, [10, 11], "float16", layout="default")
        H = T.match_buffer(H_ptr, [10, 11], "float16", layout=T.TileLayout(([10, 11], [1, 10])))

        A0 = T.decl_buffer((10, 11), "float32", data=A.data, layout=None)
        B0 = T.decl_buffer((10, 11), "float32", data=B.data, scope="global")
        C0 = T.decl_buffer((10, 11), "float32", data=C.data, layout="default")
        D0 = T.decl_buffer((10, 11), "float32", data=D.data, layout=T.TileLayout(([10, 11], [1, 10])))

        with T.kernel():
            A1 = T.alloc_buffer((10, 11), "float32", layout=None)
            B1 = T.alloc_buffer((10, 11), "float32", scope="global")
            C1 = T.alloc_buffer((10, 11), "float32", layout="default")
            D1 = T.alloc_buffer((10, 11), "float32", layout=T.TileLayout(([10, 11], [1, 10])))

            pass
    # fmt: on
    code = test.script()
    assert_structural_equal(test, from_source(code))


if __name__ == "__main__":
    tvm.testing.main()
