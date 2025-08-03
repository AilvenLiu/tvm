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
from tvm.tir.layout import TileLayout
from tvm.script import tir as T
from tvm.script import tirp as Tp


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
    g_layout = s_layout = TileLayout(g_shape)

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

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.nd.array(A_np, dev)

        mod = tvm.IRModule({"main": unary_op})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(f"compiled source code: {mod.mod.imported_modules[0].get_source()}")
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
    g_layout = s_layout = TileLayout(g_shape)

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

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func(operands_type)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(f"compiled source code: {mod.mod.imported_modules[0].get_source()}")
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
@pytest.mark.parametrize("op_type", ["reciprocal", "exp"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op_local(input, op_type, dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = TileLayout(g_shape_a)
    acc_shape = red_shape = (16, NUM_COL)

    @T.prim_func(tirp=True)
    def test_unary(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = T.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            wg_id = T.warpgroup_id([N_GROUPS], parent="cta")
            warp_id_in_wg = T.warp_id([N_WARPS // N_GROUPS], parent="warpgroup")
            lane_id = T.thread_id([thread_cnt], parent="warp")

            with T.thread():
                # acc layout
                atom = T.TileLayout(shard=([1, 2], [2, 1]))
                warp_layout = T.TileLayout(shard=([8, 4], [(4, "laneid"), (1, "laneid")]))
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = T.TileLayout(shard=([2, NUM_COL // 8], [1, 2]))
                acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                acc = T.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
                    scope="local",
                    logical_scope="thread",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )
                res = T.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
                    scope="local",
                    logical_scope="thread",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # load A into acc
                with T.thread():
                    for i in T.serial(NUM_COL // 8):
                        for j in T.unroll(2):
                            for vec in T.vectorized(2):
                                acc[j, i * 2 + vec] = A[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ]

                # unary op
                with T.warp():
                    acc_view = T.view(acc, layout=acc_layout, shape=acc_shape)
                    res_view = T.view(res, layout=acc_layout, shape=red_shape)
                    if op_type == "reciprocal":
                        Tp.reciprocal(res_view, acc_view)
                    elif op_type == "exp":
                        Tp.exp(res_view, acc_view)

                # write res into B
                with T.thread():
                    for i in T.serial(NUM_COL // 8):
                        for j in T.unroll(2):
                            for vec in T.vectorized(2):
                                B[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ] = res[j, i * 2 + vec]

    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_unary})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        # find ref result
        if op_type == "reciprocal":
            B_ref = 1 / A_np
        elif op_type == "exp":
            B_ref = np.exp2(A_np)
        else:
            raise ValueError(f"op_type={op_type} is not supported")
        atol = 1e-3
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
    g_layout_a, g_layout_b = TileLayout(g_shape_a), TileLayout(g_shape_b)
    A_shape, B_shape = (16, NUM_COL), (16, 4)
    const = T.float16(3.0) if dtype == "float16" else T.float32(3.0)

    @T.prim_func(tirp=True)
    def test_broadcast_and_apply_const(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = T.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

        with T.kernel():
            bx, by, bz = T.cta_id([1, 1, 1], parent="kernel")
            wg_id = T.warpgroup_id([N_GROUPS], parent="cta")
            warp_id_in_wg = T.warp_id([N_WARPS // N_GROUPS], parent="warpgroup")
            lane_id = T.thread_id([thread_cnt], parent="warp")

            with T.thread():
                # A layout
                atom = T.TileLayout(shard=([1, 2], [2, 1]))
                warp_layout = T.TileLayout(shard=([8, 4], [(4, "laneid"), (1, "laneid")]))
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = T.TileLayout(shard=([2, NUM_COL // 8], [1, 2]))
                A_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                A_buffer = T.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
                    scope="local",
                    logical_scope="thread",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # B layout
                B_atom = T.TileLayout(shard=([1, 1], [1, 1]))
                B_warp_atom = B_atom.tile(warp_layout, (8, 4), (1, 1))
                B_tile = T.TileLayout(shard=([2, 1], [1, 1]))
                B_layout = B_warp_atom.tile(B_tile, (2, 1), (8, 4))
                B_buffer = T.alloc_buffer(
                    [
                        2,
                    ],
                    dtype=dtype,
                    scope="local",
                    logical_scope="thread",
                    layout=B_atom.tile(B_tile, (2, 1), (1, 1)),
                )

                # load A into A_buffer
                with T.thread():
                    for i in T.serial(NUM_COL // 8):
                        for j in T.unroll(2):
                            for vec in T.vectorized(2):
                                A_buffer[j, i * 2 + vec] = A[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ]

                # load B into B_buffer
                with T.thread():
                    for i in T.unroll(2):
                        B_buffer[i] = B[
                            wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4
                        ]

                # binary op
                with T.warp():
                    A_view = T.view(A_buffer, layout=A_layout, shape=A_shape)
                    B_view = T.view(B_buffer, layout=B_layout, shape=B_shape)
                    if op_type == "sub":
                        Tp.sub(A_view, A_view, B_view)
                        Tp.sub(A_view, A_view, const)
                    elif op_type == "mul":
                        Tp.mul(A_view, A_view, B_view)
                        Tp.mul(A_view, A_view, const)

                # write A_buffer back to A
                with T.thread():
                    for i in T.serial(NUM_COL // 8):
                        for j in T.unroll(2):
                            for vec in T.vectorized(2):
                                A[
                                    wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                    i * 8 + lane_id % 4 * 2 + vec,
                                ] = A_buffer[j, i * 2 + vec]

    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_broadcast_and_apply_const})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.random.rand(*g_shape_b).astype(dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
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
    A = tvm.nd.array(A_ref, dev)
    B = tvm.nd.array(B_ref, dev)

    B_ref = A_ref.astype(B_dtype)

    # fmt: off
    @T.prim_func(tirp=True)
    def test_cast(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, A_dtype, layout=TileLayout(shape))
        B = T.match_buffer(B_ptr, shape, B_dtype, layout=TileLayout(shape))

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([1], parent="cta")

            with T.thread():
                A_local = T.alloc_local(shape, dtype=A_dtype, layout=TileLayout(shape))
                B_local = T.alloc_local(shape, dtype=B_dtype, layout=TileLayout(shape))
                Tp.copy(A_local, A)
                Tp.cast(B_local, A_local)
                Tp.copy(B, B_local)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        mod(A, B)
        print(mod.mod.imported_modules[0].get_source())
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
