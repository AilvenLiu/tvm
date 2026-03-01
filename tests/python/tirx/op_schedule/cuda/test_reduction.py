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
from tvm.tir.layout import S, TileLayout, laneid


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape_a
            (32,),  # g_shape_b
            (0, 0),  # st_a
            (0,),  # st_b
            (32, 32),  # extent_a
            (32,),  # extent_b
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### large size #########
        (
            (8, 16, 2, 22),  # g_shape_a
            (8, 16),  # g_shape_b
            (0, 0, 0, 0),  # st_a
            (0, 0),  # st_b
            (8, 16, 2, 22),  # extent_a
            (8, 16),  # extent_b
            128,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        # ######### small size #########
        (
            (32, 7),  # g_shape_a
            (32,),  # g_shape_b
            (0, 0),  # st_a
            (0,),  # st_b
            (32, 7),  # extent_a
            (32,),  # extent_b
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### offset test #########
        (
            (32, 32),  # g_shape_a
            (32,),  # g_shape_b
            (1, 1),  # st_a
            (2,),  # st_b
            (5, 8),  # extent_a
            (5,),  # extent_b
            32,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["sum", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_reduction_op_shared(input, op_type, dtype):
    g_shape_a, g_shape_b, st_a, st_b, extent_a, extent_b, thread_cnt, dev = input

    s_shape_a = g_shape_a
    s_shape_b = g_shape_b
    copy_slice_a = list(slice(None) for i in range(len(g_shape_a)))
    copy_slice_b = list(slice(None) for i in range(len(g_shape_b)))
    reduce_slice_a = list(slice(st_a[i], st_a[i] + extent_a[i]) for i in range(len(g_shape_a)))
    reduce_slice_b = list(slice(st_b[i], st_b[i] + extent_b[i]) for i in range(len(g_shape_b)))
    g_layout_a = s_layout_a = TileLayout(S[g_shape_a])
    g_layout_b = s_layout_b = TileLayout(S[g_shape_b])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_reduction(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(s_shape_a, dtype, scope="shared", layout=s_layout_a)
                B_smem = Tx.alloc_buffer(s_shape_b, dtype, scope="shared", layout=s_layout_b)

                Tx.copy(A_smem[tuple(copy_slice_a)], A[tuple(copy_slice_a)])
                if op_type == "sum":
                    Tx.sum(B_smem[tuple(reduce_slice_b)], A_smem[tuple(reduce_slice_a)])
                elif op_type == "max":
                    Tx.max(B_smem[tuple(reduce_slice_b)], A_smem[tuple(reduce_slice_a)])
                Tx.copy(B[tuple(copy_slice_b)], B_smem[tuple(copy_slice_b)])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_reduction})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # find ref result
        D = len(A.shape) - len(B.shape)
        if op_type == "sum":
            B_ref = A.numpy()[tuple(reduce_slice_a)].sum(axis=tuple(range(-D, 0)))
        elif op_type == "max":
            B_ref = A.numpy()[tuple(reduce_slice_a)].max(axis=tuple(range(-D, 0)))
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")

        atol = 1e-5 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(B_ref, B.numpy()[tuple(reduce_slice_b)], atol=atol)


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
@pytest.mark.parametrize("op_type", ["sum", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_reduction_op_local(input, op_type, dtype, shuffle):
    layout, N_GROUPS, N_WARPS, thread_cnt, dev = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a, g_shape_b = (16 * N_WARPS, NUM_COL), (16 * N_WARPS, 4)
    g_layout_a, g_layout_b = TileLayout(S[g_shape_a]), TileLayout(S[g_shape_b])
    acc_shape, red_shape = (16, NUM_COL), (16, 4)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_reduction(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
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
                warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4@laneid, 1@laneid)])
                warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
                tile = Tx.TileLayout(Tx.S[(2, NUM_COL // 8) : (1, 2)])
                acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
                acc = Tx.alloc_buffer(
                    [2, NUM_COL // 4],
                    dtype=dtype,
                    scope="local",
                    layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                )

                # red layout
                red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
                red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))
                red_tile = Tx.TileLayout(Tx.S[(2, 1) : (1, 1)])
                red_layout = red_warp_atom.tile(red_tile, (2, 1), (8, 4))
                red = Tx.alloc_buffer(
                    [
                        2,
                    ],
                    dtype=dtype,
                    scope="local",
                    layout=red_atom.tile(red_tile, (2, 1), (1, 1)),
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

                # reduce
                with Tx.warp():
                    acc_view = acc.view(*acc_shape, layout=acc_layout)
                    red_view = red.view(*red_shape, layout=red_layout)
                    if op_type == "sum":
                        Tx.sum(red_view, acc_view, thread_reduce=shuffle)
                    elif op_type == "max":
                        Tx.max(red_view, acc_view, thread_reduce=shuffle)
                    # perform an additional shuffle step if not shuffled above
                    if not shuffle:
                        if op_type == "sum":
                            Tx.sum(red_view, red_view, thread_reduce=True)
                        elif op_type == "max":
                            Tx.max(red_view, red_view, thread_reduce=True)

                # write red into B
                with Tx.thread():
                    for i in Tx.unroll(2):
                        B[wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4] = (
                            red[i]
                        )

    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_reduction})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # find ref result
        if op_type == "sum":
            B_ref = A.numpy().sum(axis=-1)
        elif op_type == "max":
            B_ref = A.numpy().max(axis=-1)
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")
        atol = 1e-5 if dtype == "float32" else 1e-1
        B_ref = np.tile(B_ref[:, np.newaxis], (1, 4))
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)


@pytest.mark.parametrize(
    "reduction_len",
    [
        8,  # minimum for optimized path
        16,
        64,
        128,  # typical FA4 size
        256,
        7,  # fallback path (not multiple of 8)
        10,
        15,
        100,
    ],
)
@pytest.mark.parametrize("op_type", ["max", "min"])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_op_local_thread_3input_maxmin(reduction_len, op_type, accum):
    """Test thread-level local buffer reduction with 3-input max/min PTX intrinsics."""
    dev = tvm.cuda(0)
    dtype = "float32"

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [reduction_len], dtype, layout=TileLayout(S[reduction_len]))
        B = Tx.match_buffer(B_ptr, [1], dtype, layout=TileLayout(S[1]))

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([1], parent="cta")

            with Tx.thread():
                A_local = Tx.alloc_buffer([reduction_len], dtype, scope="local")
                B_local = Tx.alloc_buffer([1], dtype, scope="local")

                # Load from global to local
                for i in Tx.serial(reduction_len):
                    A_local[i] = A[i]

                # Initialize B_local for accum test
                if accum:
                    B_local[0] = B[0]

                # Thread-level reduction
                if op_type == "max":
                    Tx.max(B_local, A_local, accum=accum)
                elif op_type == "min":
                    Tx.min(B_local, A_local, accum=accum)

                # Store result to global
                B[0] = B_local[0]
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(reduction_len).astype(dtype)

        if accum:
            B_np = np.array([0.5], dtype=dtype)  # Initial value for accumulation
        else:
            B_np = np.zeros(1, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        if op_type == "max":
            if accum:
                B_ref = max(A_np.max(), 0.5)
            else:
                B_ref = A_np.max()
        elif op_type == "min":
            if accum:
                B_ref = min(A_np.min(), 0.5)
            else:
                B_ref = A_np.min()

        tvm.testing.assert_allclose(B_ref, B.numpy()[0], atol=1e-5)


@pytest.mark.parametrize(
    "reduction_len",
    [
        8,  # minimum for optimized path
        16,
        64,
        128,
        256,
        9,  # not divisible by 8, tests remainder handling
        17,
        63,
        65,
        100,
    ],
)
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_op_local_thread_packed_add_sum(reduction_len, accum):
    """Test thread-level sum reduction using packed add with add.f32x2 PTX instruction."""
    dev = tvm.cuda(0)
    dtype = "float32"

    # fmt: off
    @Tx.prim_func(tirx=True)
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [reduction_len], dtype, layout=TileLayout(S[reduction_len]))
        B = Tx.match_buffer(B_ptr, [1], dtype, layout=TileLayout(S[1]))

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([1], parent="cta")

            with Tx.thread():
                A_local = Tx.alloc_buffer([reduction_len], dtype, scope="local")
                B_local = Tx.alloc_buffer([1], dtype, scope="local")

                # Load from global to local
                for i in Tx.serial(reduction_len):
                    A_local[i] = A[i]

                # Initialize B_local for accum test
                if accum:
                    B_local[0] = B[0]

                # Thread-level sum reduction
                Tx.sum(B_local, A_local, accum=accum)

                # Store result to global
                B[0] = B_local[0]
    # fmt: on

    # Use sm_100a target for packed add sum dispatch
    target = tvm.target.Target("cuda -arch=sm_100a")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(reduction_len).astype(dtype)

        if accum:
            B_np = np.array([0.5], dtype=dtype)  # Initial value for accumulation
        else:
            B_np = np.zeros(1, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        if accum:
            B_ref = A_np.sum() + 0.5
        else:
            B_ref = A_np.sum()

        # Use larger tolerance due to rounding differences from packed add (add.rz.ftz.f32x2)
        tvm.testing.assert_allclose(B_ref, B.numpy()[0], atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
