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
import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TileLayout, laneid, tid_in_wg, tx, warpid
from tvm.tirx.op_schedule.cuda.cast import _cast_layout_supported_for_local


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["zero", "sqrt"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [("float16", "float16"), ("float32", "float16"), ("float32", "bfloat16")],
)
def test_unary_op_shared(input, op_type, src_dtype, dst_dtype):
    g_shape, st_a, st_res, ext_a, ext_res, thread_cnt, dev = input
    s_shape = g_shape
    g_layout = s_layout = TileLayout(S[g_shape])
    in_place = src_dtype == dst_dtype

    copy_slice = list(slice(None) for _ in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    if in_place:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def unary_op(A_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, src_dtype, layout=g_layout)

            with Tx.kernel():
                _bx = Tx.cta_id([1], parent="kernel")
                _tx = Tx.thread_id([thread_cnt], parent="cta")

                with Tx.cta():
                    A_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                    if op_type == "zero":
                        Tx.zero(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)])
                    elif op_type == "sqrt":
                        Tx.sqrt(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)])
                    Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def unary_op(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, src_dtype, layout=g_layout)
            B = Tx.match_buffer(B_ptr, g_shape, dst_dtype, layout=g_layout)

            with Tx.kernel():
                _bx = Tx.cta_id([1], parent="kernel")
                _tx = Tx.thread_id([thread_cnt], parent="cta")

                with Tx.cta():
                    A_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    B_smem = Tx.alloc_buffer(s_shape, dst_dtype, scope="shared", layout=s_layout)
                    Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                    if op_type == "zero":
                        Tx.zero(B_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)])
                    elif op_type == "sqrt":
                        Tx.sqrt(B_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)])
                    Tx.copy(B[tuple(copy_slice)], B_smem[tuple(copy_slice)])
        # fmt: on

    def get_ref(A_np):
        if in_place:
            A_ref = A_np.copy()
            if op_type == "zero":
                A_ref[tuple(map_slice_res)] = 0.0
            elif op_type == "sqrt":
                A_ref[tuple(map_slice_res)] = np.sqrt(A_np[tuple(map_slice_a)])
            return A_ref
        else:
            B_ref = np.zeros(g_shape, dtype=dst_dtype)
            if op_type == "zero":
                B_ref[tuple(map_slice_res)] = 0.0
            elif op_type == "sqrt":
                B_ref[tuple(map_slice_res)] = np.sqrt(A_np[tuple(map_slice_a)]).astype(dst_dtype)
            return B_ref

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.abs(np.random.rand(*g_shape).astype(src_dtype)) + 0.1
        A = tvm.runtime.tensor(A_np, dev)

        mod = tvm.IRModule({"main": unary_op})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        if in_place:
            mod(A)
            A_ref = get_ref(A_np)
            tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)
        else:
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=dst_dtype), dev)
            mod(A, B)
            B_ref = get_ref(A_np)
            tvm.testing.assert_allclose(B_ref, B.numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("exec_scope", ["warp", "warpgroup"])
def test_unary_op_shared_subcta_scope(exec_scope):
    dtype = "float16"
    n_warps = 4 if exec_scope == "warpgroup" else 1
    g_shape = (n_warps * 32, 8)
    dev = tvm.cuda(0)

    @Tx.prim_func(tirx=True)
    def unary_op_subcta(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=TileLayout(S[g_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            _tid = Tx.thread_id([256], parent="cta")
            with Tx.cta():
                A_smem = Tx.alloc_buffer(
                    g_shape, dtype, scope="shared", layout=TileLayout(S[g_shape])
                )
                Tx.copy(A_smem, A)
                if exec_scope == "warp":
                    with Tx.warp()[5:6]:
                        Tx.zero(A_smem, A_smem)
                elif exec_scope == "warpgroup":
                    with Tx.warpgroup()[1:2]:
                        Tx.zero(A_smem, A_smem)
                Tx.cuda.cta_sync()
                Tx.copy(A, A_smem)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        mod = tvm.IRModule({"main": unary_op_subcta})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A)
        tvm.testing.assert_allclose(A.numpy(), np.zeros_like(A_np), atol=1e-3)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
@pytest.mark.parametrize("bias_type", ["const", "region"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("float16", "float16"),
        ("float32", "float32"),
        ("float32", "float16"),
        ("float32", "bfloat16"),
    ],
)
def test_unary_op_shared_with_bias_scale(input, op_type, bias_type, src_dtype, dst_dtype):
    g_shape, st_a, st_res, ext_a, ext_res, thread_cnt, dev = input
    s_shape = g_shape
    g_layout = s_layout = TileLayout(S[g_shape])
    in_place = src_dtype == dst_dtype

    copy_slice = list(slice(None) for _ in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    # scale and bias in compute_dtype (= src_dtype)
    scale = Tx.FloatImm(src_dtype, 1.5)
    const_bias = Tx.FloatImm(src_dtype, 0.88)

    if in_place:

        @Tx.prim_func(tirx=True)
        def unary_op_with_bias(A_ptr: Tx.handle, bias_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, src_dtype, layout=g_layout)
            bias = Tx.match_buffer(bias_ptr, g_shape, src_dtype, layout=g_layout)

            with Tx.kernel():
                _bx = Tx.cta_id([1], parent="kernel")
                _tx = Tx.thread_id([thread_cnt], parent="cta")

                with Tx.cta():
                    A_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    bias_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                    Tx.copy(bias_smem[tuple(copy_slice)], bias[tuple(copy_slice)])
                    if bias_type == "const":
                        if op_type == "sqrt":
                            Tx.sqrt(
                                A_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                const_bias,
                                scale,
                            )
                        elif op_type == "exp":
                            Tx.exp(
                                A_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                const_bias,
                                scale,
                            )
                    elif bias_type == "region":
                        if op_type == "sqrt":
                            Tx.sqrt(
                                A_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                bias_smem[tuple(map_slice_a)],
                                scale,
                            )
                        elif op_type == "exp":
                            Tx.exp(
                                A_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                bias_smem[tuple(map_slice_a)],
                                scale,
                            )
                    Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])
    else:

        @Tx.prim_func(tirx=True)
        def unary_op_with_bias(A_ptr: Tx.handle, B_ptr: Tx.handle, bias_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, src_dtype, layout=g_layout)
            B = Tx.match_buffer(B_ptr, g_shape, dst_dtype, layout=g_layout)
            bias = Tx.match_buffer(bias_ptr, g_shape, src_dtype, layout=g_layout)

            with Tx.kernel():
                _bx = Tx.cta_id([1], parent="kernel")
                _tx = Tx.thread_id([thread_cnt], parent="cta")

                with Tx.cta():
                    A_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    B_smem = Tx.alloc_buffer(s_shape, dst_dtype, scope="shared", layout=s_layout)
                    bias_smem = Tx.alloc_buffer(s_shape, src_dtype, scope="shared", layout=s_layout)
                    Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                    Tx.copy(bias_smem[tuple(copy_slice)], bias[tuple(copy_slice)])
                    if bias_type == "const":
                        if op_type == "sqrt":
                            Tx.sqrt(
                                B_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                const_bias,
                                scale,
                            )
                        elif op_type == "exp":
                            Tx.exp(
                                B_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                const_bias,
                                scale,
                            )
                    elif bias_type == "region":
                        if op_type == "sqrt":
                            Tx.sqrt(
                                B_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                bias_smem[tuple(map_slice_a)],
                                scale,
                            )
                        elif op_type == "exp":
                            Tx.exp(
                                B_smem[tuple(map_slice_res)],
                                A_smem[tuple(map_slice_a)],
                                bias_smem[tuple(map_slice_a)],
                                scale,
                            )
                    Tx.copy(B[tuple(copy_slice)], B_smem[tuple(copy_slice)])

    def get_ref(A_np, bias_np):
        if in_place:
            A_ref = A_np.copy()
            if bias_type == "region":
                if op_type == "sqrt":
                    A_ref[tuple(map_slice_res)] = np.sqrt(
                        A_np[tuple(map_slice_a)] * scale.value + bias_np[tuple(map_slice_a)]
                    )
                elif op_type == "exp":
                    A_ref[tuple(map_slice_res)] = np.exp(
                        A_np[tuple(map_slice_a)] * scale.value + bias_np[tuple(map_slice_a)]
                    )
            elif bias_type == "const":
                if op_type == "sqrt":
                    A_ref[tuple(map_slice_res)] = np.sqrt(
                        A_np[tuple(map_slice_a)] * scale.value + const_bias.value
                    )
                elif op_type == "exp":
                    A_ref[tuple(map_slice_res)] = np.exp(
                        A_np[tuple(map_slice_a)] * scale.value + const_bias.value
                    )
            else:
                raise ValueError(f"bias_type={bias_type} is not supported")
            return A_ref
        else:
            B_ref = np.zeros(g_shape, dtype=dst_dtype)
            if bias_type == "region":
                if op_type == "sqrt":
                    B_ref[tuple(map_slice_res)] = np.sqrt(
                        A_np[tuple(map_slice_a)] * scale.value + bias_np[tuple(map_slice_a)]
                    ).astype(dst_dtype)
                elif op_type == "exp":
                    B_ref[tuple(map_slice_res)] = np.exp(
                        A_np[tuple(map_slice_a)] * scale.value + bias_np[tuple(map_slice_a)]
                    ).astype(dst_dtype)
            elif bias_type == "const":
                if op_type == "sqrt":
                    B_ref[tuple(map_slice_res)] = np.sqrt(
                        A_np[tuple(map_slice_a)] * scale.value + const_bias.value
                    ).astype(dst_dtype)
                elif op_type == "exp":
                    B_ref[tuple(map_slice_res)] = np.exp(
                        A_np[tuple(map_slice_a)] * scale.value + const_bias.value
                    ).astype(dst_dtype)
            else:
                raise ValueError(f"bias_type={bias_type} is not supported")
            return B_ref

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.abs(np.random.rand(*g_shape).astype(src_dtype)) + 0.1
        bias_np = np.random.rand(*g_shape).astype(src_dtype)
        A = tvm.runtime.tensor(A_np, dev)
        bias = tvm.runtime.tensor(bias_np, dev)

        mod = tvm.IRModule({"main": unary_op_with_bias})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        if in_place:
            mod(A, bias)
            A_ref = get_ref(A_np, bias_np)
            atol = (
                1e-1
                if src_dtype == "float16" and op_type == "exp"
                else (1e-2 if src_dtype == "float16" else 1e-3)
            )
            tvm.testing.assert_allclose(A_ref, A.numpy(), atol=atol)
        else:
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=dst_dtype), dev)
            mod(A, B, bias)
            B_ref = get_ref(A_np, bias_np)
            tvm.testing.assert_allclose(B_ref, B.numpy(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_b
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_b
            (32, 32),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (14, 1, 4),  # st_b
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_b
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### broadcast test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (14, 1, 4),  # st_b
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (1, 6, 1),  # extent_b
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("operands_type", ["region_region", "region_const", "const_region"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_shared(input, op_type, operands_type, dtype):
    # skip test
    if op_type in ["sub", "fdiv"] and operands_type == "const_region":
        return

    g_shape, st_a, st_b, st_res, ext_a, ext_b, ext_res, thread_cnt, dev = input
    g_layout = s_layout = TileLayout(S[g_shape])

    copy_slice = list(slice(None) for i in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_b = list(slice(st_b[i], st_b[i] + ext_b[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    const = Tx.float16(3.0) if dtype == "float16" else Tx.float32(3.0)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def binary_op_region_region(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)
                B_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                Tx.copy(B_smem[tuple(copy_slice)], B[tuple(copy_slice)])
                if op_type == "add":
                    Tx.add(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "sub":
                    Tx.sub(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "mul":
                    Tx.mul(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "fdiv":
                    Tx.fdiv(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])

    @Tx.prim_func(tirx=True)
    def binary_op_const_region_or_region_const(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        _B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                if op_type == "add":
                    if operands_type == "const_region":
                        Tx.add(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.add(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "sub":
                    if operands_type == "const_region":
                        Tx.sub(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.sub(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "mul":
                    if operands_type == "const_region":
                        Tx.mul(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.mul(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "fdiv":
                    if operands_type == "const_region":
                        Tx.fdiv(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.fdiv(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])
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
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] + B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] + 3.0
        elif op_type == "sub":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] - B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] - 3.0
        elif op_type == "mul":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] * B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] * 3.0
        elif op_type == "fdiv":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] / B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] / 3.0

        return A_ref

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func(operands_type)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B)

        A_ref = get_ref(A_np, B_np)
        atol = 1e-3
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=atol)


@pytest.mark.parametrize("op_type", ["sub", "fdiv"])
def test_binary_non_commutative_const_lhs_rejected(op_type):
    dtype = "float16"
    shape = (16, 16)
    layout = TileLayout(S[shape])
    const = Tx.float16(3.0)

    with pytest.raises(Exception):

        @Tx.prim_func(tirx=True)
        def bad_kernel() -> None:
            with Tx.kernel():
                _bx = Tx.cta_id([1], parent="kernel")
                _tid = Tx.thread_id([64], parent="cta")
                with Tx.cta():
                    A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=layout)
                    if op_type == "sub":
                        Tx.sub(A_smem, const, A_smem)
                    elif op_type == "fdiv":
                        Tx.fdiv(A_smem, const, A_smem)

        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": bad_kernel})
            tvm.compile(mod, target=target, tir_pipeline="tirx")


@pytest.mark.parametrize("exec_scope", ["warp", "warpgroup"])
@pytest.mark.parametrize("op_type", ["add", "mul"])
def test_binary_op_shared_subcta_scope(exec_scope, op_type):
    """Test binary ops in warp/warpgroup scope with shared memory."""
    dtype = "float16"
    n_warps = 4 if exec_scope == "warpgroup" else 1
    g_shape = (n_warps * 32, 8)
    dev = tvm.cuda(0)
    tx_op = {"add": Tx.add, "mul": Tx.mul}[op_type]

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=TileLayout(S[g_shape]))
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=TileLayout(S[g_shape]))
        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            _tid = Tx.thread_id([256], parent="cta")
            with Tx.cta():
                A_smem = Tx.alloc_buffer(
                    g_shape, dtype, scope="shared", layout=TileLayout(S[g_shape])
                )
                B_smem = Tx.alloc_buffer(
                    g_shape, dtype, scope="shared", layout=TileLayout(S[g_shape])
                )
                Tx.copy(A_smem, A)
                Tx.copy(B_smem, B)
                if exec_scope == "warp":
                    with Tx.warp()[5:6]:
                        tx_op(A_smem, A_smem, B_smem)
                elif exec_scope == "warpgroup":
                    with Tx.warpgroup()[1:2]:
                        tx_op(A_smem, A_smem, B_smem)
                Tx.cuda.cta_sync()
                Tx.copy(A, A_smem)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        np_op = {"add": np.add, "mul": np.multiply}[op_type]
        A_ref = np_op(A_np, B_np).astype(dtype)
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)


@pytest.mark.parametrize(
    "input",
    [
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            1,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            4,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            2,  # N_GROUPS
            8,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["reciprocal", "exp", "exp2"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [("float16", "float16"), ("float32", "float16"), ("float32", "bfloat16")],
)
def test_unary_op_local(input, op_type, src_dtype, dst_dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = TileLayout(S[g_shape_a])
    acc_shape = red_shape = (16, NUM_COL)

    @Tx.prim_func(tirx=True)
    def test_unary(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, src_dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dst_dtype, layout=g_layout_b)

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([N_GROUPS], parent="cta")
            warp_id_in_wg = Tx.warp_id([N_WARPS // N_GROUPS], parent="warpgroup")
            lane_id = Tx.thread_id([thread_cnt], parent="warp")

            with Tx.thread():
                # acc layout
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = Tx.TileLayout(Tx.S[(2, NUM_COL // 8) : (1, 2)])
                acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                acc = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=src_dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )
                res = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dst_dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # load A into acc
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                acc[j, i * 2 + vec] = A[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ]

                # unary op
                with Tx.warp():
                    acc_view = acc.view(*acc_shape, layout=acc_layout)
                    res_view = res.view(*red_shape, layout=acc_layout)
                    if op_type == "reciprocal":
                        Tx.reciprocal(res_view, acc_view)
                    elif op_type == "exp":
                        Tx.exp(res_view, acc_view)
                    elif op_type == "exp2":
                        Tx.exp2(res_view, acc_view)

                # write res into B
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                B[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ] = res[j, i * 2 + vec]

    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_unary})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.abs(np.random.rand(*g_shape_a).astype(src_dtype)) + 0.1
        B_np = np.zeros(g_shape_b, dtype=dst_dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B)

        # find ref result
        if op_type == "reciprocal":
            B_ref = (1 / A_np).astype(dst_dtype)
        elif op_type == "exp":
            B_ref = np.exp(A_np).astype(dst_dtype)
        elif op_type == "exp2":
            B_ref = np.exp2(A_np).astype(dst_dtype)
        else:
            raise ValueError(f"op_type={op_type} is not supported")
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "input",
    [
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            1,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            4,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            2,  # N_GROUPS
            8,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
@pytest.mark.parametrize("bias_type", ["const", "region"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [("float32", "float32"), ("float32", "float16"), ("float32", "bfloat16")],
)
def test_unary_op_local_with_bias_scale(input, op_type, bias_type, src_dtype, dst_dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = g_shape_bias = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = g_layout_bias = TileLayout(S[g_shape_a])
    acc_shape = red_shape = bias_shape = (16, NUM_COL)

    scale = Tx.float16(1.5) if src_dtype == "float16" else Tx.float32(1.5)
    const_bias = Tx.float16(0.88) if src_dtype == "float16" else Tx.float32(0.88)

    @Tx.prim_func(tirx=True)
    def test_unary_with_bias(A_ptr: Tx.handle, B_ptr: Tx.handle, bias_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, src_dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dst_dtype, layout=g_layout_b)
        bias = Tx.match_buffer(bias_ptr, g_shape_bias, src_dtype, layout=g_layout_bias)

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([N_GROUPS], parent="cta")
            warp_id_in_wg = Tx.warp_id([N_WARPS // N_GROUPS], parent="warpgroup")
            lane_id = Tx.thread_id([thread_cnt], parent="warp")

            with Tx.thread():
                # acc layout
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = Tx.TileLayout(Tx.S[(2, NUM_COL // 8) : (1, 2)])
                acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                acc = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=src_dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )
                bias_local = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=src_dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )
                res = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dst_dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # load A into acc
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                acc[j, i * 2 + vec] = A[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ]
                # load bias into bias_local
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                bias_local[j, i * 2 + vec] = bias[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ]

                # unary op
                with Tx.warp():
                    acc_view = acc.view(*acc_shape, layout=acc_layout)
                    res_view = res.view(*red_shape, layout=acc_layout)
                    bias_view = bias_local.view(*bias_shape, layout=acc_layout)
                    if bias_type == "const":
                        if op_type == "sqrt":
                            Tx.sqrt(res_view, acc_view, const_bias, scale)
                        elif op_type == "exp":
                            Tx.exp(res_view, acc_view, const_bias, scale)
                    elif bias_type == "region":
                        if op_type == "sqrt":
                            Tx.sqrt(res_view, acc_view, bias_view, scale)
                        elif op_type == "exp":
                            Tx.exp(res_view, acc_view, bias_view, scale)

                # write res into B
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                B[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ] = res[j, i * 2 + vec]

    def get_ref(A_np, bias_np):
        A_ref = A_np.copy()
        if bias_type == "region":
            if op_type == "sqrt":
                A_ref = np.sqrt(A_np * scale.value + bias_np)
            elif op_type == "exp":
                A_ref = np.exp(A_np * scale.value + bias_np)
        elif bias_type == "const":
            if op_type == "sqrt":
                A_ref = np.sqrt(A_np * scale.value + const_bias.value)
            elif op_type == "exp":
                A_ref = np.exp(A_np * scale.value + const_bias.value)
        else:
            raise ValueError(f"bias_type={bias_type} is not supported")
        return A_ref.astype(dst_dtype)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(src_dtype)
        bias_np = np.random.rand(*g_shape_bias).astype(src_dtype)
        B_np = np.zeros(g_shape_b, dtype=dst_dtype)
        A = tvm.runtime.tensor(A_np, dev)
        bias = tvm.runtime.tensor(bias_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": test_unary_with_bias})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B, bias)

        B_ref = get_ref(A_np, bias_np)
        atol = 1e-3 if src_dtype == dst_dtype else 2e-2
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)


@pytest.mark.parametrize(
    "input",
    [
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            1,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            4,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        (
            "wgmma",  # layout
            2,  # N_GROUPS
            8,  # N_WARPS
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_local(input, op_type, dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a, g_shape_b = (16 * N_WARPS, NUM_COL), (16 * N_WARPS, 4)
    g_layout_a, g_layout_b = TileLayout(S[g_shape_a]), TileLayout(S[g_shape_b])
    A_shape, B_shape = (16, NUM_COL), (16, 4)
    const = Tx.float16(3.0) if dtype == "float16" else Tx.float32(3.0)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_broadcast_and_apply_const(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([N_GROUPS], parent="cta")
            warp_id_in_wg = Tx.warp_id([N_WARPS // N_GROUPS], parent="warpgroup")
            lane_id = Tx.thread_id([thread_cnt], parent="warp")

            with Tx.thread():
                # A layout
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = Tx.TileLayout(Tx.S[(2, NUM_COL // 8) : (1, 2)])
                A_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                A_buffer = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # B layout
                B_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
                B_warp_atom = B_atom.tile(warp_layout, (8, 4), (1, 1))
                B_tile = Tx.TileLayout(Tx.S[(2, 1) : (1, 1)])
                B_layout = B_warp_atom.tile(B_tile, (2, 1), (8, 4))
                B_buffer = Tx.alloc_buffer(
                    [2,],
                    dtype=dtype,
                    scope="local",
                    layout=B_atom.tile(B_tile, (2, 1), (1, 1)),
                )

                # load A into A_buffer
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                A_buffer[j, i * 2 + vec] = A[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec]  # noqa: E501

                # load B into B_buffer
                with Tx.thread():
                    for i in Tx.unroll(2):
                        B_buffer[i] = B[wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4]  # noqa: E501

                # binary op
                with Tx.warp():
                    A_view = A_buffer.view(*A_shape, layout=A_layout)
                    B_view = B_buffer.view(*B_shape, layout=B_layout)
                    if op_type == "add":
                        Tx.add(A_view, A_view, B_view)
                        Tx.add(A_view, A_view, const)
                    elif op_type == "sub":
                        Tx.sub(A_view, A_view, B_view)
                        Tx.sub(A_view, A_view, const)
                    elif op_type == "mul":
                        Tx.mul(A_view, A_view, B_view)
                        Tx.mul(A_view, A_view, const)
                    elif op_type == "fdiv":
                        Tx.fdiv(A_view, A_view, B_view)
                        Tx.fdiv(A_view, A_view, const)

                # write A_buffer back to A
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                A[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec] = A_buffer[j, i * 2 + vec]  # noqa: E501

    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_broadcast_and_apply_const})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.random.rand(*g_shape_b).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # find ref result
        val = np.tile(np.repeat(B_np, 2, axis=1), NUM_COL // 8)
        if op_type == "add":
            B_ref = (A_np + val) + 3.0
        elif op_type == "sub":
            B_ref = (A_np - val) - 3.0
        elif op_type == "mul":
            B_ref = (A_np * val) * 3.0
        elif op_type == "fdiv":
            B_ref = (A_np / val) / 3.0
        else:
            raise ValueError(f"op_type={op_type} is not supported")
        atol = 1e-2 if op_type == "fdiv" else 1e-3
        tvm.testing.assert_allclose(B_ref, A.numpy(), atol=atol)


@pytest.mark.parametrize("exec_scope", ["cta", "warpgroup", "warp"])
@pytest.mark.parametrize("rhs_kind", ["region", "broadcast", "const"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
def test_binary_op_local_subcta_trivial(exec_scope, rhs_kind, op_type):
    dtype = "float16"
    m, n = 4, 8
    n_threads = 256 if exec_scope == "cta" else (128 if exec_scope == "warpgroup" else 32)
    # in this test, use warp3/warpgroup1 to test
    thr_str = 0 if exec_scope == "cta" else (128 if exec_scope == "warpgroup" else 32 * 3)
    a_shape = (n_threads, m, n)
    b_shape = (n_threads, m, n if rhs_kind == "region" else 1)
    c_shape = a_shape
    const = Tx.float16(1.25)
    dev = tvm.cuda(0)
    tx_op = {"add": Tx.add, "sub": Tx.sub, "mul": Tx.mul, "fdiv": Tx.fdiv}[op_type]

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))
        C = Tx.match_buffer(C_ptr, c_shape, dtype, layout=TileLayout(S[c_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            _tid = Tx.thread_id([256], parent="cta")
            tid_in_scope = Tx.thread_id([n_threads], parent=exec_scope)

            with Tx.cta():
                b_n = Tx.meta_var(n if rhs_kind == "region" else 1)
                A_local = Tx.alloc_buffer(
                    (m, n), dtype, scope="local", layout=TileLayout(S[(m, n)])
                )
                C_local = Tx.alloc_buffer(
                    (m, n), dtype, scope="local", layout=TileLayout(S[(m, n)])
                )
                B_local = Tx.alloc_buffer(
                    (m, b_n), dtype, scope="local", layout=TileLayout(S[(m, b_n)])
                )

                with Tx.thread()[thr_str : thr_str + n_threads]:
                    for i in Tx.serial(m):
                        for j in Tx.serial(n):
                            A_local[i, j] = A[tid_in_scope, i, j]
                    if rhs_kind != "const":
                        for i in Tx.serial(m):
                            for j in Tx.serial(b_n):
                                B_local[i, j] = B[tid_in_scope, i, j]
                # Tx.cuda.cta_sync()

                if exec_scope == "cta":
                    with Tx.cta():
                        if rhs_kind == "const":
                            tx_op(C_local, A_local, const)
                        else:
                            tx_op(C_local, A_local, B_local)
                elif exec_scope == "warpgroup":
                    with Tx.warpgroup()[1:2]:
                        if rhs_kind == "const":
                            tx_op(C_local, A_local, const)
                        else:
                            tx_op(C_local, A_local, B_local)
                else:
                    with Tx.warp()[3:4]:
                        if rhs_kind == "const":
                            tx_op(C_local, A_local, const)
                        else:
                            tx_op(C_local, A_local, B_local)
                # Tx.cuda.cta_sync()

                with Tx.thread()[thr_str : thr_str + n_threads]:
                    for i in Tx.serial(m):
                        for j in Tx.serial(n):
                            C[tid_in_scope, i, j] = C_local[i, j]

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        C_np = np.zeros(c_shape, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        C = tvm.runtime.tensor(C_np, dev)

        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B, C)

        np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[op_type]
        if rhs_kind == "region":
            C_ref = np_op(A_np, B_np)
        elif rhs_kind == "broadcast":
            C_ref = np_op(A_np, np.repeat(B_np, n, axis=2))
        else:
            C_ref = np_op(A_np, const.value)
        atol = 1e-2 if op_type == "fdiv" else 1e-3
        tvm.testing.assert_allclose(C_ref, C.numpy(), atol=atol)


@pytest.mark.parametrize("shape", [(128, 8), (128, 4, 16), (128, 5, 5)])
@pytest.mark.parametrize("op_type", ["fill"])
@pytest.mark.parametrize("exec_scope", ["thread", "cta"])
@pytest.mark.parametrize("storage_scope", ["local", "shared"])
def test_unary_op_vectorized(shape, op_type, exec_scope, storage_scope):
    if storage_scope == "local" and exec_scope == "cta":
        return  # skip unsupported case
    dev = tvm.cuda(0)
    dtype = "float16"
    A_ref = np.random.rand(*shape).astype(dtype)
    A = tvm.runtime.tensor(A_ref, dev)
    value = Tx.float16(7.89) if dtype == "float16" else Tx.float32(7.89)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_unary_thread(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, dtype, layout=TileLayout(S[shape]))
        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([128], parent="cta")
            with Tx.thread():
                if storage_scope == "shared":
                    a_smem = Tx.alloc_buffer(
                        shape, dtype=dtype, layout=TileLayout(S[shape]), scope="shared"
                    )
                    Tx.fill(a_smem[tx], value)
                    Tx.copy(A[tx], a_smem[tx])
                elif storage_scope == "local":
                    a_local = Tx.alloc_buffer(
                        shape[1:], dtype=dtype, layout=TileLayout(S[shape[1:]]), scope="local"
                    )
                    Tx.fill(a_local, value)
                    Tx.copy(A[tx], a_local)

    @Tx.prim_func(tirx=True)
    def test_unary_cta(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, dtype, layout=TileLayout(S[shape]))
        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            _tid = Tx.thread_id([128], parent="cta")
            with Tx.cta():
                if storage_scope == "shared":
                    a_smem = Tx.alloc_buffer(
                        shape, dtype=dtype, layout=TileLayout(S[shape]), scope="shared"
                    )
                    Tx.fill(a_smem, value)
                    Tx.copy(A, a_smem)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule(
            {"main": test_unary_thread if exec_scope == "thread" else test_unary_cta}
        )
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A)
        print(mod.mod.imports[0].inspect_source())
        tvm.testing.assert_allclose(A.numpy(), np.full(shape, value.value), atol=1e-2)


@pytest.mark.parametrize("op_type", ["zero", "sqrt", "reciprocal", "exp"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op_local_thread_wise(op_type, dtype):
    """Test unary ops in thread scope with local buffers (trivial layout)."""
    shape = (64, 32)
    local_shape = shape[1:]
    dev = tvm.cuda(0)

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, dtype, layout=TileLayout(S[shape]))
        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tid = Tx.thread_id([64], parent="cta")
            with Tx.thread():
                a_local = Tx.alloc_buffer(
                    local_shape, dtype, scope="local", layout=TileLayout(S[local_shape])
                )
                Tx.copy(a_local, A[tid])
                if op_type == "zero":
                    Tx.zero(a_local, a_local)
                elif op_type == "sqrt":
                    Tx.sqrt(a_local, a_local)
                elif op_type == "reciprocal":
                    Tx.reciprocal(a_local, a_local)
                elif op_type == "exp":
                    Tx.exp(a_local, a_local)
                Tx.copy(A[tid], a_local)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.abs(np.random.rand(*shape).astype(dtype)) + 0.1
        A = tvm.runtime.tensor(A_np, dev)
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A)
        if op_type == "zero":
            A_ref = np.zeros_like(A_np)
        elif op_type == "sqrt":
            A_ref = np.sqrt(A_np)
        elif op_type == "reciprocal":
            A_ref = (1.0 / A_np).astype(dtype)
        elif op_type == "exp":
            A_ref = np.exp(A_np)
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (64, 32),  # a_shape
            (64, 32),  # b_shape
            (64, 32),  # res_shape
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### broadcast test #########
        (
            (16, 5, 4),  # a_shape
            (16, 1, 4),  # b_shape
            (16, 5, 4),  # res_shape
            16,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("storage_scope", ["shared", "local"])
@pytest.mark.parametrize("exec_scope", ["cta", "thread"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_vectorized(input, storage_scope, exec_scope, op_type, dtype):
    a_shape, b_shape, res_shape, thread_cnt, dev = input
    tx_op = {"add": Tx.add, "sub": Tx.sub, "mul": Tx.mul, "fdiv": Tx.fdiv}[op_type]

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_binary_cta(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")
            with Tx.cta():
                if storage_scope == "shared":
                    A_smem = Tx.alloc_buffer(
                        a_shape, dtype, scope="shared", layout=TileLayout(S[a_shape])
                    )
                    B_smem = Tx.alloc_buffer(
                        b_shape, dtype, scope="shared", layout=TileLayout(S[b_shape])
                    )
                    Tx.copy(A_smem, A)
                    Tx.copy(B_smem, B)
                    tx_op(A_smem, A_smem, B_smem)
                    Tx.copy(A, A_smem)
            with Tx.thread():
                if storage_scope == "local":
                    A_local = Tx.alloc_buffer(
                        a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                    )
                    B_local = Tx.alloc_buffer(
                        b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                    )
                    Tx.copy(A_local, A[tx])
                    Tx.copy(B_local, B[tx])
                    with Tx.cta():
                        tx_op(A_local, A_local, B_local)
                    Tx.copy(A[tx], A_local)

    @Tx.prim_func(tirx=True)
    def test_binary_thread(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                if storage_scope == "shared":
                    A_smem = Tx.alloc_buffer(
                        a_shape, dtype, scope="shared", layout=TileLayout(S[a_shape])
                    )
                    B_smem = Tx.alloc_buffer(
                        b_shape, dtype, scope="shared", layout=TileLayout(S[b_shape])
                    )
                    Tx.copy(A_smem, A)
                    Tx.copy(B_smem, B)
                    tx_op(A_smem, A_smem, B_smem)
                    Tx.copy(A, A_smem)
                elif storage_scope == "local":
                    A_local = Tx.alloc_buffer(
                        a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                    )
                    B_local = Tx.alloc_buffer(
                        b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                    )
                    Tx.copy(A_local, A[tx])
                    Tx.copy(B_local, B[tx])
                    tx_op(A_local, A_local, B_local)
                    Tx.copy(A[tx], A_local)
    # fmt: on

    def get_prim_func():
        if exec_scope == "cta":
            return test_binary_cta
        elif exec_scope == "thread":
            return test_binary_thread
        else:
            raise ValueError(f"exec_scope={exec_scope} is not supported")

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B)

        np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[op_type]
        A_ref = np_op(A_np, B_np)
        atol = 1e-2 if op_type == "fdiv" else 1e-3
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=atol)


@pytest.mark.parametrize("op_type", ["add", "sub", "mul"])
def test_binary_op_packed_f32x2_auto_dispatch(op_type):
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits:
        pytest.skip(f"cannot parse target arch: {arch}")
    sm_version = int(sm_digits)
    if sm_version < 100:
        pytest.skip(f"packed_f32x2 auto-dispatch requires sm_100+, got {arch}")

    a_shape, b_shape = (64, 32), (64, 32)
    dtype = "float32"
    dev = tvm.cuda(0)

    @Tx.prim_func(tirx=True)
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([64], parent="cta")
            with Tx.thread():
                A_local = Tx.alloc_buffer(
                    a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                )
                B_local = Tx.alloc_buffer(
                    b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                )
                Tx.copy(A_local, A[tx])
                Tx.copy(B_local, B[tx])
                if op_type == "add":
                    Tx.add(A_local, A_local, B_local)
                elif op_type == "sub":
                    Tx.sub(A_local, A_local, B_local)
                elif op_type == "mul":
                    Tx.mul(A_local, A_local, B_local)
                Tx.copy(A[tx], A_local)

    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        ptx_pat = {
            "add": r"add\.[a-z]+\.ftz\.f32x2",
            "sub": r"sub\.[a-z]+\.ftz\.f32x2",
            "mul": r"mul\.[a-z]+\.ftz\.f32x2",
        }[op_type]
        builtin_pat = {
            "add": r"tvm_builtin_ptx_add_packed_",
            "sub": r"tvm_builtin_ptx_sub_packed_",
            "mul": r"tvm_builtin_ptx_mul_packed_",
        }[op_type]
        assert re.search(ptx_pat, src) or re.search(builtin_pat, src), src
        mod(A, B)

        if op_type == "add":
            A_ref = A_np + B_np
        elif op_type == "sub":
            A_ref = A_np - B_np
        elif op_type == "mul":
            A_ref = A_np * B_np
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)


@pytest.mark.parametrize("shape", [(8,), (16, 16), (5, 5)])
@pytest.mark.parametrize("A_dtype", ["float16", "float32"])
@pytest.mark.parametrize("B_dtype", ["float16", "float32"])
def test_cast_thread_local(shape, A_dtype, B_dtype):
    if A_dtype == B_dtype:
        return

    dev = tvm.cuda(0)
    A_ref = np.random.rand(*shape).astype(A_dtype)
    B_ref = np.random.rand(*shape).astype(B_dtype)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(B_ref, dev)

    B_ref = A_ref.astype(B_dtype)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_cast(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, A_dtype, layout=TileLayout(S[shape]))
        B = Tx.match_buffer(B_ptr, shape, B_dtype, layout=TileLayout(S[shape]))

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([1], parent="cta")

            with Tx.thread():
                A_local = Tx.alloc_local(shape, dtype=A_dtype, layout=TileLayout(S[shape]))
                B_local = Tx.alloc_local(shape, dtype=B_dtype, layout=TileLayout(S[shape]))
                Tx.copy(A_local, A)
                Tx.cast(B_local, A_local)
                Tx.copy(B, B_local)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        print(mod.mod.imports[0].inspect_source())
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)


@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_warpgroup_local_view(A_dtype, B_dtype):
    """Tx.cast in warpgroup scope with offset (tid_in_wg + layout offset). Covers offset/tid_in_wg/warpgroup scope."""  # noqa: E501
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    use_offset = True
    if use_offset:
        from tvm.tir.layout import Axis, Iter

        m_axis = Axis.get("m")
        shard = [Iter(N_THREADS, 1, tid_in_wg), Iter(LOCAL_LEN, 1, m_axis)]
        cast_layout = TileLayout.from_iters(shard, [], {m_axis: 0})
    else:
        cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tid_in_wg, 1)])

    dev = tvm.cuda(0)
    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(B_ref, dev)
    B_ref = A_ref.astype(B_dtype)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_cast(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, A_dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, B_dtype, layout=g_layout)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            tid_in_wg = Tx.thread_id([N_THREADS], parent="warpgroup")

            with Tx.thread():
                reg_src = Tx.alloc_buffer((LOCAL_LEN,), A_dtype, scope="local")
                reg_dst = Tx.alloc_buffer((LOCAL_LEN,), B_dtype, scope="local")
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        reg_src[i] = A[tid_in_wg, i]
                with Tx.warpgroup():
                    reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    Tx.cast(reg_dst_view, reg_src_view)
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        B[tid_in_wg, i] = reg_dst[i]
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        print(mod.mod.imports[0].inspect_source())
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)


@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_cta_local_view(A_dtype, B_dtype):
    """Tx.cast with view+layout in CTA scope (128 threads, register->register)."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    dev = tvm.cuda(0)
    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(B_ref, dev)
    B_ref = A_ref.astype(B_dtype)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_cast(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, A_dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, B_dtype, layout=g_layout)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            tx_var = Tx.thread_id([N_THREADS], parent="cta")

            with Tx.thread():
                reg_src = Tx.alloc_buffer((LOCAL_LEN,), A_dtype, scope="local")
                reg_dst = Tx.alloc_buffer((LOCAL_LEN,), B_dtype, scope="local")
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        reg_src[i] = A[tx_var, i]
                with Tx.cta():
                    reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    Tx.cast(reg_dst_view, reg_src_view)
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        B[tx_var, i] = reg_dst[i]
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        print(mod.mod.imports[0].inspect_source())
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)


@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
@pytest.mark.parametrize("slice_start,slice_end", [(0, 4), (2, 6), (4, 8)])
def test_cast_local_view_sliced(A_dtype, B_dtype, slice_start, slice_end):
    """Tx.cast with sliced view in CTA scope — exercises _emit_cast_local_view_sliced."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    dev = tvm.cuda(0)
    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
    B_ref[:, slice_start:slice_end] = A_ref[:, slice_start:slice_end].astype(B_dtype)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, A_dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, B_dtype, layout=g_layout)
        with Tx.kernel():
            _bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([N_THREADS], parent="cta")
            with Tx.thread():
                reg_src = Tx.alloc_buffer((LOCAL_LEN,), A_dtype, scope="local")
                reg_dst = Tx.alloc_buffer((LOCAL_LEN,), B_dtype, scope="local")
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        reg_src[i] = A[tx, i]
                with Tx.cta():
                    reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
                    Tx.cast(
                        reg_dst_view[0:N_THREADS, slice_start:slice_end],
                        reg_src_view[0:N_THREADS, slice_start:slice_end],
                    )
                with Tx.thread():
                    for i in Tx.serial(LOCAL_LEN):
                        B[tx, i] = reg_dst[i]
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
    tvm.testing.assert_allclose(
        B.numpy()[:, slice_start:slice_end],
        B_ref[:, slice_start:slice_end],
        atol=1e-2,
    )


def test_cast_layout_partition_and_validation():
    """Partition table (simplified): partition structure and _cast_layout_supported_for_local."""
    from tvm.tir.layout import Axis, Iter
    from tvm.tirx.op_schedule.cuda.cast import _get_layout_thread_local_partition

    m_axis = Axis.get("m")

    # (layout, expected_supported, optional check: part -> None or assert)
    cases = [
        # Supported: single tx, tid_in_wg, thread in middle (from_iters), mixed warpid+laneid
        (
            TileLayout(S[(128, 8) : (1 @ tx, 1)]),
            True,
            lambda p: p[0].get(tx) == ([0], [128]) and p[1] == [1] and p[2] == [8],
        ),
        (
            TileLayout(S[(128, 8) : (1 @ tid_in_wg, 1)]),
            True,
            lambda p: p[0].get(tid_in_wg) == ([0], [128]),
        ),
        (
            TileLayout.from_iters([Iter(4, 16, "m"), Iter(8, 2, tx), Iter(2, 1, "m")], [], {}),
            True,
            lambda p: p[0].get(tx) == ([1], [8]) and p[1] == [0, 2],
        ),
        (
            TileLayout(S[(2, 8, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)]),
            True,
            lambda p: warpid in p[0] and laneid in p[0] and p[1] == [3] and p[2] == [2],
        ),
        # Rejected: no thread, no local, thread in replica
        (TileLayout(S[(64, 8) : (1, 1)]), False, None),
        (TileLayout(S[(8, 8) : (1 @ tx, 1 @ laneid)]), False, None),
        (
            TileLayout.from_iters([Iter(128, 1, tx), Iter(8, 1, m_axis)], [Iter(2, 1, laneid)], {}),
            False,
            None,
        ),
    ]

    for layout, expected_supported, check in cases:
        part = _get_layout_thread_local_partition(layout)
        supported = _cast_layout_supported_for_local(layout)
        assert supported is expected_supported, f"layout={layout}"
        if expected_supported and check:
            assert part is not None
            check(part)


@pytest.mark.parametrize("slice_start,slice_end", [(0, 2), (2, 4)])
def test_cast_mixed_axes_and_subregion(slice_start, slice_end):
    """Test cast with mixed axes and subregion."""

    N_WARPS, LANES = 2, 32
    LOCAL_LEN = 4
    full_shape = (8, N_WARPS, 4, LOCAL_LEN)
    g_layout = TileLayout(S[full_shape])
    cast_layout = TileLayout(S[full_shape : (4 @ laneid, 2 @ warpid, 1 @ laneid, 1)])

    A_ref = np.zeros(full_shape, dtype="float32")
    for j in range(full_shape[0]):
        for w in range(full_shape[1]):
            for k in range(full_shape[2]):
                for i in range(full_shape[3]):
                    A_ref[j, w, k, i] = float(j * 1000 + w * 100 + k * 10 + i)
    B_ref = np.zeros(full_shape, dtype="float16")
    B_ref[:, :, :, slice_start:slice_end] = A_ref[:, :, :, slice_start:slice_end].astype("float16")

    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(np.zeros(full_shape, dtype="float16"), dev)

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, full_shape, "float32", layout=g_layout)
        B = Tx.match_buffer(B_ptr, full_shape, "float16", layout=g_layout)
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            warp_id = Tx.warp_id([N_WARPS], parent="cta")
            lane_id = Tx.thread_id([LANES], parent="warp")
            with Tx.thread():
                reg_src = Tx.alloc_buffer((LOCAL_LEN,), "float32", scope="local")
                reg_dst = Tx.alloc_buffer((LOCAL_LEN,), "float16", scope="local")
                with Tx.thread():
                    j, k = lane_id // 4, lane_id % 4
                    for i in Tx.serial(LOCAL_LEN):
                        reg_src[i] = A[j, warp_id, k, i]
                with Tx.cta():
                    reg_src_view = reg_src.view(*full_shape, layout=cast_layout)
                    reg_dst_view = reg_dst.view(*full_shape, layout=cast_layout)
                    Tx.cast(
                        reg_dst_view[0:8, 0:N_WARPS, 0:4, slice_start:slice_end],
                        reg_src_view[0:8, 0:N_WARPS, 0:4, slice_start:slice_end],
                    )
                with Tx.thread():
                    j, k = lane_id // 4, lane_id % 4
                    for i in Tx.serial(LOCAL_LEN):
                        B[j, warp_id, k, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
    tvm.testing.assert_allclose(
        B.numpy()[:, :, :, slice_start:slice_end],
        B_ref[:, :, :, slice_start:slice_end],
        atol=1e-2,
        rtol=0,
    )


def test_cast_joint_decomposition_extents_order():
    """Test joint decomposition uses thread dims in layout order with correct extents."""
    from tvm.tirx.op_schedule.cuda.cast import _get_layout_thread_local_partition

    layout = TileLayout(S[(2, 32, 4) : (2 @ warpid, 32 @ laneid, 1)])
    part = _get_layout_thread_local_partition(layout)
    assert part is not None
    thread_groups, local_dims, local_extents = part
    assert warpid in thread_groups and laneid in thread_groups
    assert thread_groups[warpid] == ([0], [2])
    assert thread_groups[laneid] == ([1], [32])
    assert local_dims == [2]
    assert local_extents == [4]

    thread_dims_ordered = []
    for _axis, (dim_indices, extents) in thread_groups.items():
        for i, dim_idx in enumerate(dim_indices):
            thread_dims_ordered.append((dim_idx, extents[i]))
    thread_dims_ordered.sort(key=lambda x: x[0])
    # Region extent = layout extent for full region
    shape = [2, 32, 4]
    joint_all_extents = [shape[dim_idx] for dim_idx, _ in thread_dims_ordered]
    assert thread_dims_ordered == [(0, 2), (1, 32)], thread_dims_ordered
    assert joint_all_extents == [2, 32], joint_all_extents


def test_cast_validate_extent_mismatch_rejected():
    """Validation rejects when src and dst layouts have same thread positions but different extents."""  # noqa: E501

    view_shape = (2, 8, 4, 8)
    g_layout = TileLayout(S[view_shape])
    src_layout = TileLayout(S[view_shape : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)])
    dst_layout = TileLayout(
        S[view_shape : (2 @ warpid, 8 @ laneid, 1 @ laneid, 1)]
    )  # dim1 extent 8 != 4

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, view_shape, "float32", layout=g_layout)
        B = Tx.match_buffer(B_ptr, view_shape, "float16", layout=g_layout)
        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            warp_id = Tx.warp_id([2], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")
            with Tx.thread():
                reg_src = Tx.alloc_buffer((8,), "float32", scope="local")
                reg_dst = Tx.alloc_buffer((8,), "float16", scope="local")
                with Tx.thread():
                    j, k = lane_id // 4, lane_id % 4
                    for i in Tx.serial(8):
                        reg_src[i] = A[warp_id, j, k, i]
                with Tx.cta():
                    reg_src_view = reg_src.view(*view_shape, layout=src_layout)
                    reg_dst_view = reg_dst.view(*view_shape, layout=dst_layout)
                    Tx.cast(reg_dst_view, reg_src_view)
                with Tx.thread():
                    j, k = lane_id // 4, lane_id % 4
                    for i in Tx.serial(8):
                        B[warp_id, j, k, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(Exception, match="validate_cast_local_view failed"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


if __name__ == "__main__":
    tvm.testing.main()
