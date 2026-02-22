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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import laneid, tid_in_wg, tx, warpid
from tvm.tir.layout import TileLayout, S

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
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op_shared(input, op_type, dtype):
    g_shape, st_a, st_res, ext_a, ext_res, thread_cnt, dev = input
    s_shape = g_shape
    g_layout = s_layout = TileLayout(S[g_shape])

    copy_slice = list(slice(None) for _ in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    # fmt: off
    @Tx.prim_func(tirx=True)
    def unary_op(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(s_shape, dtype, scope="shared", layout=s_layout)
                Tx.copy(A_smem[*copy_slice], A[*copy_slice])
                if op_type == "zero":
                    Tx.zero(A_smem[*map_slice_res], A_smem[*map_slice_a])
                elif op_type == "sqrt":
                    Tx.sqrt(A_smem[*map_slice_res], A_smem[*map_slice_a])
                Tx.copy(A[*copy_slice], A_smem[*copy_slice])
    # fmt: on

    def get_ref(A_np):
        A_ref = A_np.copy()
        if op_type == "zero":
            A_ref[*map_slice_res] = 0.0
        elif op_type == "sqrt":
            A_ref[*map_slice_res] = np.sqrt(A_np[*map_slice_a])

        return A_ref

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)

        mod = tvm.IRModule({"main": unary_op})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A)

        A_ref = get_ref(A_np)
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)


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
@pytest.mark.parametrize("op_type", ["add", "fdiv"])
@pytest.mark.parametrize("operands_type", ["region_region", "region_const"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_shared(input, op_type, operands_type, dtype):
    # skip test
    if op_type in ["sub", "fdiv"] and operands_type == "const_region":
        return

    g_shape, st_a, st_b, st_res, ext_a, ext_b, ext_res, thread_cnt, dev = input
    s_shape = g_shape
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
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)
                B_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[*copy_slice], A[*copy_slice])
                Tx.copy(B_smem[*copy_slice], B[*copy_slice])
                if op_type == "add":
                    Tx.add(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "sub":
                    Tx.sub(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "mul":
                    Tx.mul(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                elif op_type == "fdiv":
                    Tx.fdiv(A_smem[*map_slice_res], A_smem[*map_slice_a], B_smem[*map_slice_b])
                Tx.copy(A[*copy_slice], A_smem[*copy_slice])

    @Tx.prim_func(tirx=True)
    def binary_op_const_region_or_region_const(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[*copy_slice], A[*copy_slice])
                if op_type == "add":
                    if operands_type == "const_region":
                        Tx.add(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tx.add(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "sub":
                    if operands_type == "const_region":
                        Tx.sub(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tx.sub(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "mul":
                    if operands_type == "const_region":
                        Tx.mul(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tx.mul(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                elif op_type == "fdiv":
                    if operands_type == "const_region":
                        Tx.fdiv(A_smem[*map_slice_res], const, A_smem[*map_slice_a])
                    elif operands_type == "region_const":
                        Tx.fdiv(A_smem[*map_slice_res], A_smem[*map_slice_a], const)
                Tx.copy(A[*copy_slice], A_smem[*copy_slice])
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
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op_local(input, op_type, dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = TileLayout(S[g_shape_a])
    acc_shape = red_shape = (16, NUM_COL)

    @Tx.prim_func(tirx=True)
    def test_unary(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

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
                    dtype=dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )
                res = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
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
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # find ref result
        if op_type == "reciprocal":
            B_ref = 1 / A_np
        elif op_type == "exp":
            B_ref = np.exp(A_np)
        elif op_type == "exp2":
            B_ref = np.exp2(A_np)
        else:
            raise ValueError(f"op_type={op_type} is not supported")
        # exp (e^x) is not a native GPU instruction and has higher fp16 error than exp2
        atol = 5e-3 if op_type == "exp" else 1e-3
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
@pytest.mark.parametrize("op_type", ["sub", "mul"])
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
                                A_buffer[j, i * 2 + vec] = A[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec]

                # load B into B_buffer
                with Tx.thread():
                    for i in Tx.unroll(2):
                        B_buffer[i] = B[wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4]

                # binary op
                with Tx.warp():
                    A_view = A_buffer.view(*A_shape, layout=A_layout)
                    B_view = B_buffer.view(*B_shape, layout=B_layout)
                    if op_type == "sub":
                        Tx.sub(A_view, A_view, B_view)
                        Tx.sub(A_view, A_view, const)
                    elif op_type == "mul":
                        Tx.mul(A_view, A_view, B_view)
                        Tx.mul(A_view, A_view, const)

                # write A_buffer back to A
                with Tx.thread():
                    for i in Tx.serial(NUM_COL // 8):
                        for j in Tx.unroll(2):
                            for vec in Tx.vectorized(2):
                                A[wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4, i * 8 + lane_id % 4 * 2 + vec] = A_buffer[j, i * 2 + vec]

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
        if op_type == "sub":
            B_ref = (A_np - val) - 3.0
        elif op_type == "mul":
            B_ref = (A_np * val) * 3.0
        else:
            raise ValueError(f"op_type={op_type} is not supported")
        atol = 1e-3
        tvm.testing.assert_allclose(B_ref, A.numpy(), atol=atol)


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
            bx = Tx.cta_id([1], parent="kernel")
            tx = Tx.thread_id([1], parent="cta")

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
    """Tx.cast in warpgroup scope with offset (tid_in_wg + layout offset). Covers offset/tid_in_wg/warpgroup scope."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    use_offset = True
    if use_offset:
        from tvm.tir.layout import Iter, Axis
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
            bx = Tx.cta_id([1], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
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
            bx = Tx.cta_id([1], parent="kernel")
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


def test_cast_layout_partition_and_validation():
    """Partition table (simplified): partition structure and _cast_layout_supported_for_local."""
    from tvm.tir.layout import Iter, Axis, warpid
    from tvm.tirx.op_schedule.cuda.cast import _get_layout_thread_local_partition

    m_axis = Axis.get("m")

    # (layout, expected_supported, optional check: part -> None or assert)
    cases = [
        # Supported: single tx, tid_in_wg, thread in middle (from_iters), mixed warpid+laneid
        (TileLayout(S[(128, 8) : (1 @ tx, 1)]), True,
         lambda p: p[0].get(tx) == ([0], [128]) and p[1] == [1] and p[2] == [8]),
        (TileLayout(S[(128, 8) : (1 @ tid_in_wg, 1)]), True,
         lambda p: p[0].get(tid_in_wg) == ([0], [128])),
        (TileLayout.from_iters(
            [Iter(4, 16, "m"), Iter(8, 2, tx), Iter(2, 1, "m")], [], {}), True,
         lambda p: p[0].get(tx) == ([1], [8]) and p[1] == [0, 2]),
        (TileLayout(S[(2, 8, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)]), True,
         lambda p: warpid in p[0] and laneid in p[0] and p[1] == [3] and p[2] == [2]),
        # Rejected: no thread, no local, thread in replica
        (TileLayout(S[(64, 8) : (1, 1)]), False, None),
        (TileLayout(S[(8, 8) : (1 @ tx, 1 @ laneid)]), False, None),
        (TileLayout.from_iters(
            [Iter(128, 1, tx), Iter(8, 1, m_axis)], [Iter(2, 1, laneid)], {}), False, None),
    ]

    for layout, expected_supported, check in cases:
        part = _get_layout_thread_local_partition(layout)
        supported = _cast_layout_supported_for_local(layout)
        assert supported is expected_supported, f"layout={layout}"
        if expected_supported and check:
            assert part is not None
            check(part)


def test_cast_mixed_axes_and_subregion():
    """Test cast with mixed axes and subregion."""
    from tvm.tir.layout import warpid

    N_WARPS, LANES = 2, 32
    LOCAL_LEN = 4
    SLICE_START, SLICE_END = 0, 2
    full_shape = (8, N_WARPS, 4, LOCAL_LEN)
    g_layout = TileLayout(S[full_shape])
    cast_layout = TileLayout(S[full_shape : (4 @ laneid, 2 @ warpid, 1 @ laneid, 1)])

    A_ref = np.zeros(full_shape, dtype="float32")
    for j in range(full_shape[0]):
        for w in range(full_shape[1]):
            for k in range(full_shape[2]):
                for i in range(full_shape[3]):
                    A_ref[j, w, k, i] = float(j * 1000 + w * 100 + k * 10 + i)
    # Kernel only casts the subregion; only that part is defined in output
    B_ref = np.zeros(full_shape, dtype="float16")
    B_ref[:, :, :, SLICE_START:SLICE_END] = A_ref[:, :, :, SLICE_START:SLICE_END].astype("float16")

    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_ref, dev)
    B = tvm.runtime.tensor(np.zeros(full_shape, dtype="float16"), dev)

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, full_shape, "float32", layout=g_layout)
        B = Tx.match_buffer(B_ptr, full_shape, "float16", layout=g_layout)
        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
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
                        reg_dst_view[0:8, 0:N_WARPS, 0:4, SLICE_START:SLICE_END],
                        reg_src_view[0:8, 0:N_WARPS, 0:4, SLICE_START:SLICE_END],
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
    # Only the cast subregion is written; rest of B may be uninitialized
    tvm.testing.assert_allclose(
        B.numpy()[:, :, :, SLICE_START:SLICE_END],
        B_ref[:, :, :, SLICE_START:SLICE_END],
        atol=1e-2, rtol=0,
    )


def test_cast_joint_decomposition_extents_order():
    """Test joint decomposition uses thread dims in layout order with correct extents."""
    from tvm.tirx.op_schedule.cuda.cast import _get_layout_thread_local_partition
    from tvm.tir.layout import warpid

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
    """Validation rejects when src and dst layouts have same thread positions but different extents."""
    from tvm.tir.layout import warpid

    view_shape = (2, 8, 4, 8)
    g_layout = TileLayout(S[view_shape])
    src_layout = TileLayout(S[view_shape : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)])
    dst_layout = TileLayout(S[view_shape : (2 @ warpid, 8 @ laneid, 1 @ laneid, 1)])  # dim1 extent 8 != 4

    @Tx.prim_func(tirx=True)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, view_shape, "float32", layout=g_layout)
        B = Tx.match_buffer(B_ptr, view_shape, "float16", layout=g_layout)
        with Tx.kernel():
            bx = Tx.cta_id([1], parent="kernel")
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
