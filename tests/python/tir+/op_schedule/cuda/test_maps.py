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
from tvm.tir.layout import TileLayout
import numpy as np
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32,), # g_shape
            (0, 0,), # st_a
            (0, 0,), # st_res
            (32, 32), # extent_a
            (32, 32), # extent_res
            64, # thread_cnt
            tvm.cuda(0), # dev
        ),
        ######### offset test #########
        (
            (32, 8, 12), # g_shape
            (10, 0, 3), # st_a
            (20, 0, 2), # st_res
            (5, 6, 7), # extent_a
            (5, 6, 7), # extent_res
            64, # thread_cnt
            tvm.cuda(0), # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["zero", "sqrt"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op(input, op_type, dtype):
    g_shape, st_a, st_res, ext_a, ext_res, thread_cnt, dev = input
    s_shape = g_shape
    g_layout = s_layout = TileLayout.from_tuple(g_shape)

    copy_slice = list(slice(None) for _ in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    # fmt: off
    @T.prim_func(tirp=True)
    def unary_op(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=s_layout)
                Tp.copy(A_smem[*copy_slice], A[*copy_slice])
                if op_type == "zero":
                    Tp.zero(A_smem[*map_slice_res], A_smem[*map_slice_a])
                elif op_type == "sqrt":
                    Tp.sqrt(A_smem[*map_slice_res], A_smem[*map_slice_a])
                Tp.copy(A[*copy_slice], A_smem[*copy_slice])
    # fmt: on

    def get_ref(A_np):
        A_ref = A_np.copy()
        if op_type == "zero":
            A_ref[*map_slice_res] = 0.0
        elif op_type == "sqrt":
            A_ref[*map_slice_res] = np.sqrt(A_np[*map_slice_a])

        return A_ref

    target = tvm.target.Target.from_device(dev)
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.nd.array(A_np, dev)

        mod = tvm.IRModule({"main": unary_op})
        mod = tvm.build(mod, target=target, pipeline="tirp")
        print(f"compiled source code: {mod.imported_modules[0].get_source()}")
        mod(A)

        A_ref = get_ref(A_np)
        tvm.testing.assert_allclose(A_ref, A.asnumpy(), atol=1e-8)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32,), # g_shape
            (0, 0,), # st_a
            (0, 0,), # st_b
            (0, 0,), # st_res
            (32, 32), # extent_a
            (32, 32), # extent_b
            (32, 32), # extent_res
            64, # thread_cnt
            tvm.cuda(0), # dev
        ),
        ######### offset test #########
        (
            (32, 8, 12), # g_shape
            (10, 0, 3), # st_a
            (14, 1, 4), # st_b
            (20, 0, 2), # st_res
            (5, 6, 7), # extent_a
            (5, 6, 7), # extent_b
            (5, 6, 7), # extent_res
            64, # thread_cnt
            tvm.cuda(0), # dev
        ),
        ######### broadcast test #########
        (
            (32, 8, 12), # g_shape
            (10, 0, 3), # st_a
            (14, 1, 4), # st_b
            (20, 0, 2), # st_res
            (5, 6, 7), # extent_a
            (1, 6, 1), # extent_b
            (5, 6, 7), # extent_res
            64, # thread_cnt
            tvm.cuda(0), # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("operands_type", ["region_region", "const_region", "region_const"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op(input, op_type, operands_type, dtype):
    # skip test
    if op_type in ["sub", "fdiv"] and operands_type == "const_region":
        return

    g_shape, st_a, st_b, st_res, ext_a, ext_b, ext_res, thread_cnt, dev = input
    s_shape = g_shape
    g_layout = s_layout = TileLayout.from_tuple(g_shape)

    copy_slice = list(slice(None) for i in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_b = list(slice(st_b[i], st_b[i] + ext_b[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    const = T.float16(3.0) if dtype == "float16" else T.float32(3.0)

    # fmt: off
    @T.prim_func(tirp=True)
    def binary_op_region_region(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)
                B_smem = T.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tp.copy(A_smem[*copy_slice], A[*copy_slice])
                Tp.copy(B_smem[*copy_slice], B[*copy_slice])
                if op_type == "add":
                    Tp.add(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "sub":
                    Tp.sub(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "mul":
                    Tp.mul(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "fdiv":
                    Tp.fdiv(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                Tp.copy(A[*copy_slice], A_smem[*copy_slice])

    @T.prim_func(tirp=True)
    def binary_op_const_region_or_region_const(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tp.copy(A_smem[*copy_slice], A[*copy_slice])
                if op_type == "add":
                    if operands_type == "const_region":
                        Tp.add(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tp.add(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "sub":
                    if operands_type == "const_region":
                        Tp.sub(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tp.sub(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "mul":
                    if operands_type == "const_region":
                        Tp.mul(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tp.mul(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "fdiv":
                    if operands_type == "const_region":
                        Tp.fdiv(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tp.fdiv(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                Tp.copy(A[*copy_slice], A_smem[*copy_slice])
    # fmt: on

    def get_prim_func(operands_type):
        if operands_type == "region_region":
            return binary_op_region_region
        elif operands_type in ["const_region", "region_const"]:
            return binary_op_const_region_or_region_const
        raise ValueError(f"operands_type={operands_type} is not supported")

    def get_ref(A_np, B_np):
        A_ref = A_np.copy()
        if op_type == "add":
            if operands_type == "region_region":
                A_ref[*map_slice_res] = A_np[*map_slice_a] + B_np[*map_slice_b]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[*map_slice_res] = A_np[*map_slice_a] + 3.0
        elif op_type == "sub":
            if operands_type == "region_region":
                A_ref[*map_slice_res] = A_np[*map_slice_a] - B_np[*map_slice_b]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[*map_slice_res] = A_np[*map_slice_a] - 3.0
        elif op_type == "mul":
            if operands_type == "region_region":
                A_ref[*map_slice_res] = A_np[*map_slice_a] * B_np[*map_slice_b]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[*map_slice_res] = A_np[*map_slice_a] * 3.0
        elif op_type == "fdiv":
            if operands_type == "region_region":
                A_ref[*map_slice_res] = A_np[*map_slice_a] / B_np[*map_slice_b]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[*map_slice_res] = A_np[*map_slice_a] / 3.0

        return A_ref

    target = tvm.target.Target.from_device(dev)
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func(operands_type)})
        mod = tvm.build(mod, target=target, pipeline="tirp")
        print(f"compiled source code: {mod.imported_modules[0].get_source()}")
        mod(A, B)

        A_ref = get_ref(A_np, B_np)
        atol = 1e-3 if op_type == "fdiv" else 1e-8
        tvm.testing.assert_allclose(A_ref, A.asnumpy(), atol=atol)


if __name__ == "__main__":
    tvm.testing.main()
