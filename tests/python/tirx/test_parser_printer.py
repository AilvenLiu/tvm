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
from tvm.tir.layout import laneid
from tvm.script import tirx as Tx


def from_source(code):
    return tvm.script.from_source(code, tirx=True)


def test_roundtrip_scopeid1():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        A_local = Tx.alloc_buffer([1], dtype="float16", scope="local")
                        for i in Tx.serial(2):
                            A_local[0] = A[lane_id * 2 + i]
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid2():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12], parent="kernel")
            cbx, cby, cbz = Tx.cta_id([2, 2, 1], parent="cluster")
            clx, cly, clz = Tx.cluster_id([4, 5, 12], parent="kernel")
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_exec_scope():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.world():
            kid = Tx.kernel_id([2])
            with Tx.kernel():
                bx, by, bz = Tx.cta_id([32, 32, 1], parent="kernel")
                tx, ty, tz = Tx.thread_id([16, 8, 1], parent="cta")
                warp_id = Tx.warp_id([4], parent="cta")
                lane_id = Tx.thread_id([32], parent="warp")
                with Tx.cluster():
                    with Tx.cta():
                        with Tx.warpgroup():
                            with Tx.warp():
                                with Tx.thread():
                                    Tx.evaluate(0)
                        with Tx.thread():
                            Tx.evaluate(0)
                        with Tx.warp()[0:2]:
                            with Tx.thread():
                                Tx.evaluate(0)
                        with Tx.thread([128])[0:2]:
                            Tx.evaluate(0)
                        with Tx.thread([16, 8])[0:8, 0:4]:
                            Tx.evaluate(0)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_layout():
    def get_layout1():
        return Tx.TileLayout(
            shard=([8, 8, 8, 4, 2], [6, 4 @ laneid, 2, 1 @ laneid, 1]),
        )

    def get_layout2():
        return Tx.TileLayout(shard=([8, 8, 8, 4, 2], [64, 4 @ laneid, 8, 2, 1]))

    def get_layout3():
        return Tx.TileLayout(
            shard=([8, 16, 8, 16], [1024, 16, 128, 1]),
        )

    def get_layout4():
        return Tx.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    def get_layout5():
        return Tx.ComposeLayout(
            Tx.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            Tx.TileLayout(shard=([64, 64, 4], [64, 1, 64 * 64])),
        )

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            C = Tx.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
            D = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())

            with Tx.cta():
                A_warp = Tx.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
                B_warp = Tx.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())

                E = Tx.alloc_buffer([64, 256], dtype="float16", scope="shared", layout=get_layout5())

                with Tx.thread():
                    Tx.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0] + E[0, 0])
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_print_kwargs_schedule_op_full_code():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        A = Tx.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], Tx.float32(1.25), dispatch="v10", bar=7, foo=42)
    # fmt: on

    expected = (
        "# from tvm.script import tirx as Tx\n\n"
        "@Tx.prim_func(tirx=True)\n"
        "def test():\n"
        "    A = Tx.alloc_buffer((16,))\n"
        '    Tx.memset(A[0:16], Tx.float32(1.25), dispatch="v10", bar=7, foo=42)'
    )
    code = test.script()
    assert code == expected
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


L_LANE = Tx.TileLayout(shard=([32], [1 @ laneid]))


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test() -> None:
        with Tx.kernel():
            with Tx.cta():
                A = Tx.alloc_buffer([2], dtype="float16", scope="local")
                A_layout = Tx.TileLayout(shard=([1, 2], [2, 1]))
                A_warp_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
                A_warp = A.view(8, 8, layout=A_warp_layout)

                with Tx.thread():
                    A_local = A_warp.storage(2)
                    A_local[0] = Tx.float16(0)

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(out_ptr: Tx.handle) -> None:
        out = Tx.match_buffer(out_ptr, (2), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([32, 32, 1], parent="kernel")
            tx, ty, tz = Tx.thread_id([16, 8, 1], parent="cta")
            warp_id = Tx.warp_id([4], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                A = Tx.alloc_buffer([2,], dtype="float16", scope="local")
                A_layout = Tx.TileLayout(shard=([1, 2], [2, 1]))
                B_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
                B = A.view(8, 8, layout=B_layout)
                D = B.storage(2)

                with Tx.thread():
                    out[0] = A[0] + B[0, 0] + D[0]
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get3():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test() -> None:
        with Tx.kernel():
            with Tx.cta():
                A = Tx.alloc_buffer([8, 8], dtype="float32", scope="local")
                A_f16 = A.view("float16")
                A_f64 = A.view("float64")

                with Tx.thread():
                    A_f16[0, 0] = Tx.float16(0)
                    A_f64[0, 0] = Tx.float64(0)

    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op1():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([1], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                A_smem = Tx.alloc_buffer([64], dtype="float32", scope="shared")

                Tx.copy(A_smem, A)
                for i in range(10):
                    Tx.fill(A_smem, Tx.float32(0))
                    Tx.gemm(A_smem, A_smem, A_smem, A_smem)
                Tx.copy(A, A_smem)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op2():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, 128), "float16", scope="global")
        B = Tx.match_buffer(B_ptr, (128, 64), "float16", scope="global")
        C = Tx.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.cta():
                A_smem = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared")
                B_smem = Tx.alloc_buffer([32, 64], dtype="float16", scope="shared")

                C_local = Tx.alloc_buffer([128, 64], dtype="float32", scope="local")
                for k in range(4):
                    Tx.copy(A_smem, A[:, k * 32 : k * 32 + 32])
                    Tx.copy(B_smem, B[k * 32 : k * 32 + 32, 0:64])
                    Tx.gemm(C_local, A_smem, B_smem, C_local)
                Tx.copy(C, C_local)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op3():
    # fmt: off
    NUM_STAGES = 3
    K = 4096

    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, K), "float16", scope="global")
        B = Tx.match_buffer(B_ptr, (K, 64), "float16", scope="global")
        C = Tx.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            warp_id = Tx.warp_id([4], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                A_smem = Tx.alloc_buffer([NUM_STAGES, 128, 32], dtype="float16", scope="shared")
                B_smem = Tx.alloc_buffer([NUM_STAGES, 32, 64], dtype="float16", scope="shared")

                C_local = Tx.alloc_buffer([128, 64], dtype="float32", scope="local")
                for i in range(NUM_STAGES - 1):
                    Tx.copy(A_smem[i, :, :], A[:, i * 32 : i * 32 + 32])
                    Tx.copy(B_smem[i, :, :], B[i * 32 : i * 32 + 32, :])

                for k in range(K // 32):
                    copy_k = Tx.meta_var(k + NUM_STAGES - 1)
                    gemm_stage = Tx.meta_var(k % NUM_STAGES)
                    copy_stage = Tx.meta_var(copy_k % NUM_STAGES)
                    Tx.copy(A_smem[copy_stage, :, :], A[:, copy_k * 32 : copy_k * 32 + 32])
                    Tx.copy(B_smem[copy_stage, :, :], B[copy_k * 32 : copy_k * 32 + 32, :])
                    Tx.gemm(C_local, A_smem[gemm_stage, :, :], B_smem[gemm_stage, :, :], C_local)

                Tx.copy(C, C_local)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_tensormap():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def func1(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "func"})
        A = Tx.match_buffer(A_ptr, [128], "float32")

        A_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.tensormap_init", A_map, A_ptr)
    # fmt: on
    code = func1.script()
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_break_for():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            with Tx.cta():
                for i in Tx.serial(10):
                    if i > 5:
                        break
                    A[i] = i
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_while():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            with Tx.cta():
                i = Tx.alloc_buffer((1,), "int32", scope="local")
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
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (9,), "int32")

        with Tx.kernel():
            with Tx.cta():
                idx = Tx.alloc_buffer((1,), "int32", scope="local")
                idx[0] = 0
                for i in Tx.serial(3):
                    for j in Tx.serial(3):
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
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            with Tx.cta():
                for i in Tx.serial(10):
                    if (i % 2) == 0:
                        continue
                    A[i] = i
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_while():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            with Tx.cta():
                i = Tx.alloc_buffer((1,), "int32", scope="local")
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
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (9,), "int32")

        with Tx.kernel():
            with Tx.cta():
                idx = Tx.alloc_buffer((1,), dtype="int32", scope="local")
                idx[0] = 0
                for i in Tx.serial(3):
                    for j in Tx.serial(3):
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
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            with Tx.cta():
                for i in Tx.serial(10):
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
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (5,), "int32")

        with Tx.kernel():
            with Tx.cta():
                for i in Tx.serial(5):
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
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf", allocated_addr=1024)
            for i in Tx.serial(2):
                Tx.memset(A[i*5:i*5+5], Tx.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_implicit_buffer_region():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10, 10, 10), "float32", layout=Tx.TileLayout((10, 10, 10)))
        with Tx.kernel():
            Tx.memset(A[0], Tx.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_alloc_under_any_scope():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            for i in Tx.serial(10):
                A = Tx.alloc_buffer([100], "float32", scope="trn.sbuf", allocated_addr=1024)
                Tx.memset(A[i*10:i*10+10], Tx.float32(0.0))

    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            with Tx.compose_op():
                Tx.add(B, A, Tx.float32(1))
                Tx.add(C, B, Tx.float32(1))
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_workspace():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, [10], "float32", scope="global")
        B = Tx.match_buffer(B_ptr, [10], "float32", scope="global")
        with Tx.kernel():
            smem = Tx.alloc_buffer([10], "float32", scope="shared")
            Tx.add(B, A, Tx.float32(1), workspace={"smem": smem})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_workspace():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            psum = Tx.alloc_buffer([10], "float32", scope="trn.psum")
            intermediate = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            with Tx.compose_op(workspace={"intermediate": intermediate}):
                Tx.add(B, A, Tx.float32(1))
                Tx.add(C, B, Tx.float32(1), workspace={"psum": psum})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_config():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, [10], "float32", scope="global")
        B = Tx.match_buffer(B_ptr, [10], "float32", scope="global")
        with Tx.kernel():
            Tx.add(B, A, Tx.float32(1), schedule="A")
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_config():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
            psum = Tx.alloc_buffer([10], "float32", scope="trn.psum")
            with Tx.compose_op( schedule="A"):
                Tx.add(B, A, Tx.float32(1))
                Tx.add(C, B, Tx.float32(1), workspace={"psum": psum})
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_predicate():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            A = Tx.alloc_buffer([10, 10], "float32")
            B = Tx.alloc_buffer([10, 10], "float32")
            Tx.select(B, A, 1.0, lambda i, j: i < j)
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_grid():
    # fmt: off
    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            with Tx.thread():
                for lvs in Tx.grid(10, (2, 12)):
                    Tx.evaluate(lvs[0] + lvs[1])
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
            self.Tb = Tx.shared_cell("float16", "Tb")
            self.idx = Tx.local_cell("int32", "idx")
            self.inner_pool2 = Tx.decl_cell("float16", self.inner_pool.data, "shared.dyn", 5, name="inner_pool2")

        @Tx.macro
        def init(self):
            self.Ta = self.Ta + Tx.float16(1)
            self.Tb = self.Tb + Tx.float16(2)
            self.idx.buffer[0] = Tx.int32(0)
            self.idx = self.idx + Tx.int32(1)
            self.inner_pool2 = self.inner_pool2 + Tx.float16(1)
            Tx.evaluate(Tx.address_of(self.Ta))
            Tx.evaluate(Tx.address_of(self.Tb))
            Tx.evaluate(Tx.address_of(self.idx))
            Tx.evaluate(Tx.address_of(self.inner_pool))
            Tx.evaluate(Tx.address_of(self.inner_pool2))

    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            # normal buffer
            A = Tx.alloc_shared([10], "float16")
            B = Tx.alloc_local([10], "float16")
            # cell buffer (alloc)
            C = Tx.shared_cell("float16")
            D = Tx.local_cell("float16")
            pool = Tx.alloc_buffer([10], "uint8", scope="shared.dyn")
            # cell buffer (decl)
            E = Tx.decl_cell("float16", pool.data, "shared.dyn", 0)
            # normal 1-dim buffer with shape (1,)
            F = Tx.alloc_local((1,), "float16")
            with Tx.thread():
                Ta = Tx.local_cell("float16")
                inner_pool = Tx.decl_buffer(shape=[10], data=pool.data, dtype="uint8", scope="shared.dyn")
                test = Tx.meta_var(Test(Ta, inner_pool))
                test.init()
                A[0] = C
                A[0] = C + D
                A[1] = B[0] * C
                D.buffer[0] = D + Tx.float16(1)
                D = D + Tx.float16(1)
                C = D
                Tx.evaluate(E)
                E = E + Tx.float16(1)
                # normal 1-dim buffer with shape (1,) can be assigned directly,
                # but not loaded directly
                F = F[0] + Tx.float16(1)
                C += D
                D += E + C + D
                Tx.evaluate(Tx.address_of(C))
                Tx.evaluate(C.buffer.access_ptr("rw", offset=0))
                Tx.evaluate(C.buffer.data)
                Tx.evaluate(D)
                Tx.evaluate(Tx.address_of(D))
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_macro():
    # fmt: off
    @Tx.macro(hygienic=True)
    def mul(x, c):
        Tx.evaluate(x * c)

    @Tx.prim_func(tirx=True, private=True)
    def test():
        with Tx.kernel():
            for x in range(10):

                @Tx.macro(hygienic=True)
                def add(c):
                    Tx.evaluate(x + c)

                @Tx.macro(hygienic=False)
                def two_add_and_mul(c):
                    add(c)
                    add(c + c)
                    mul(x, c)

                two_add_and_mul(1)
                two_add_and_mul(2)


    @Tx.prim_func(tirx=True, private=True)
    def expected():
        with Tx.kernel():
            for x in range(10):
                Tx.evaluate(x + 1)
                Tx.evaluate(x + 2)
                Tx.evaluate(x)
                Tx.evaluate(x + 2)
                Tx.evaluate(x + 4)
                Tx.evaluate(x * 2)
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(test, expected)


def test_macro_recursive():
    # fmt: off
    @Tx.prim_func(tirx=True, private=True)
    def test():
        with Tx.kernel():
            for x in Tx.serial(10):

                @Tx.macro
                def add(x, c):
                    if c > 0:
                        add(x, c - 1)
                    Tx.evaluate(x)

                add(x, 5)

    @Tx.prim_func(private=True, tirx=True)
    def expected():
        with Tx.kernel():
            for x in range(10):
                Tx.evaluate(x)
                Tx.evaluate(x)
                Tx.evaluate(x)
                Tx.evaluate(x)
                Tx.evaluate(x)
                Tx.evaluate(x)
    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(expected, from_source(code))


def test_list_comprehension():
    # fmt: off
    @Tx.prim_func(tirx=True, private=True)
    def test():
        with Tx.kernel():
            with Tx.thread():
                acc = Tx.alloc_local([10], "bool")
                regs = Tx.meta_var([acc[_] for _ in range(10)])
                Tx.evaluate(regs[0])
                Tx.evaluate(tvm.tir.all(*regs))
                Tx.evaluate(tvm.tir.all(*[acc[_] for _ in range(10)]))
                Tx.evaluate(tvm.tir.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)])))
    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_range():
    # fmt: off
    @Tx.prim_func(tirx=True, private=True)
    def test():
        l = Tx.meta_var([i for i in range(10)])
        Tx.evaluate(l[3])

    @Tx.prim_func(tirx=True, private=True)
    def expected():
        Tx.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm.ir.assert_structural_equal(test, expected)


def test_buffer():
    # fmt: off
    @Tx.prim_func(tirx=True, private=True)
    def test(
        A: Tx.Buffer((10, 11), "float32", layout=None),
        B: Tx.Buffer((10, 11), "float32", scope="global"),
        C: Tx.Buffer((10, 11), "float32", layout="default"),
        D: Tx.Buffer((10, 11), "float32", layout=Tx.TileLayout(([10, 11], [1, 10]))),
        E_ptr: Tx.handle,
        F_ptr: Tx.handle,
        G_ptr: Tx.handle,
        H_ptr: Tx.handle,
    ):
        E = Tx.match_buffer(E_ptr, [10, 11], "float16", layout=None)
        F = Tx.match_buffer(F_ptr, [10, 11], "float16", scope="global")
        G = Tx.match_buffer(G_ptr, [10, 11], "float16", layout="default")
        H = Tx.match_buffer(H_ptr, [10, 11], "float16", layout=Tx.TileLayout(([10, 11], [1, 10])))

        A0 = Tx.decl_buffer((10, 11), "float32", data=A.data, layout=None)
        B0 = Tx.decl_buffer((10, 11), "float32", data=B.data, scope="global")
        C0 = Tx.decl_buffer((10, 11), "float32", data=C.data, layout="default")
        D0 = Tx.decl_buffer((10, 11), "float32", data=D.data, layout=Tx.TileLayout(([10, 11], [1, 10])))

        with Tx.kernel():
            A1 = Tx.alloc_buffer((10, 11), "float32", layout=None)
            B1 = Tx.alloc_buffer((10, 11), "float32", scope="global")
            C1 = Tx.alloc_buffer((10, 11), "float32", layout="default")
            D1 = Tx.alloc_buffer((10, 11), "float32", layout=Tx.TileLayout(([10, 11], [1, 10])))

            pass
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_kwargs_op_call():
    # fmt: off
    @Tx.prim_func(tirx=True, private=True)
    def test(A: Tx.Buffer((10, 10), "float32"), B: Tx.Buffer((10, 10), "float32")):
        with Tx.kernel():
            kwargs = Tx.meta_var({"dispatch": "tma", "cta_group": 2})
            Tx.copy_async(A[:, :], B[:, :], **kwargs)
    # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_workspace_default_none():
    """Regression: TIRX op IR builder functions (binary_reduce, unary_reduce,
    binary_chain, reduce_negate) should handle workspace=None (the default)
    without error. Previously these functions were missing the
    ``if workspace is None: workspace = {}`` guard."""
    from tvm.script.ir_builder.tir import tirx as tirx_builder
    from tvm.tir import BufferRegion, Buffer

    A_buf = tvm.tir.decl_buffer((128, 128), "float16", name="A")
    B_buf = tvm.tir.decl_buffer((128, 128), "float16", name="B")
    C_buf = tvm.tir.decl_buffer((128,), "float16", name="C")
    A = BufferRegion(A_buf, [tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)])
    B = BufferRegion(B_buf, [tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)])
    C = BufferRegion(C_buf, [tvm.ir.Range(0, 128)])

    # These should not crash when workspace is not provided (defaults to None)
    from tvm.tirx.operator import op as tirx_op

    op_br = tirx_op.BinaryReduce(
        B, C, A, B, tirx_op.get_tirx_op("add"), tirx_op.get_tirx_op("max"), (-1,)
    )
    assert len(op_br.workspace) == 0

    op_ur = tirx_op.UnaryReduce(
        B, C, A, tirx_op.get_tirx_op("sqrt"), tirx_op.get_tirx_op("sum"), None, None, (-1,)
    )
    assert len(op_ur.workspace) == 0

    op_bc = tirx_op.BinaryChain(
        B, A, A, A, tirx_op.get_tirx_op("add"), tirx_op.get_tirx_op("mul"), False
    )
    assert len(op_bc.workspace) == 0

    op_rn = tirx_op.ReduceNegate(C, A, (-1,), False, tirx_op.get_tirx_op("sum"))
    assert len(op_rn.workspace) == 0


def test_cell_assign_in_macro():
    """Regression: the parser's cell-assignment sugar (cell = PrimExpr) must
    work in macro context via self.attr.

    The parser narrowed ``except Exception: pass`` around the cell-detection
    path. This test verifies that PrimExpr assignment to a cell attribute in
    a macro still goes through buffer_store correctly.

    The full integration regression for the TypeError fallthrough path
    (meta_var assigned to a cell variable) is covered by
    test_hgemm::test_hgemm (tile_scheduler.m_idx pattern)."""

    # fmt: off
    class State:
        def __init__(self, counter):
            self.counter = counter

        @Tx.macro
        def add_one(self):
            # PrimExpr assigned to cell via self.attr → buffer_store succeeds
            self.counter = self.counter + Tx.int32(1)

    @Tx.prim_func(tirx=True)
    def test():
        with Tx.kernel():
            with Tx.thread([128]):
                counter = Tx.local_cell("int32")
                state = Tx.meta_var(State(counter))
                state.add_one()
                Tx.evaluate(state.counter)
    # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_cell_assign_error_not_swallowed():
    """Regression: genuine errors (non-TypeError) from buffer_store during
    cell-assignment sugar must propagate, not be silently swallowed.

    Before the fix, both eval_expr and buffer_store were wrapped in a single
    broad ``except Exception: pass``, so any error from buffer_store would be
    swallowed and the assignment would silently fall through to eval_assign."""
    from unittest.mock import patch

    original = tvm.script.ir_builder.tir.buffer_store

    def bomb(*args, **kwargs):
        # Intercept only the cell-assignment path (indices == [0])
        if args[2] == [0]:
            raise ValueError("boom")
        return original(*args, **kwargs)

    src = """
# from tvm.script import tirx as Tx

@Tx.prim_func(tirx=True)
def func():
    with Tx.kernel():
        with Tx.thread([128]):
            cell = Tx.local_cell("int32")
            cell = cell + Tx.int32(1)
"""
    # The ValueError propagates through the parser framework which wraps it
    # into a DiagnosticError.  Before the fix the broad ``except Exception``
    # would silently swallow it and fall through to eval_assign.
    with patch("tvm.script.ir_builder.tir.buffer_store", side_effect=bomb):
        with pytest.raises(tvm.error.DiagnosticError):
            from_source(src)


if __name__ == "__main__":
    tvm.testing.main()
