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
    g_layout_a = s_layout_a = TileLayout(g_shape_a)
    g_layout_b = s_layout_b = TileLayout(g_shape_b)

    # fmt: off
    @T.prim_func(tirp=True)
    def test_reduction(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = T.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)
        
        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.cta():
                A_smem = T.alloc_buffer(s_shape_a, dtype, scope="shared", layout=s_layout_a)
                B_smem = T.alloc_buffer(s_shape_b, dtype, scope="shared", layout=s_layout_b)

                Tp.copy(A_smem[*copy_slice_a], A[*copy_slice_a])
                if op_type == "sum":
                    Tp.sum(B_smem[*reduce_slice_b], A_smem[*reduce_slice_a])
                elif op_type == "max":
                    Tp.max(B_smem[*reduce_slice_b], A_smem[*reduce_slice_a])
                Tp.copy(B[*copy_slice_b], B_smem[*copy_slice_b])
    # fmt: on

    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.IRModule({"main": test_reduction})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
        mod(A, B)

        # find ref result
        D = len(A.shape) - len(B.shape)
        if op_type == "sum":
            B_ref = A.numpy()[*reduce_slice_a].sum(axis=tuple(range(-D, 0)))
        elif op_type == "max":
            B_ref = A.numpy()[*reduce_slice_a].max(axis=tuple(range(-D, 0)))
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")

        atol = 1e-5 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(B_ref, B.numpy()[*reduce_slice_b], atol=atol)


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
    g_layout_a, g_layout_b = TileLayout(g_shape_a), TileLayout(g_shape_b)
    acc_shape, red_shape = (16, NUM_COL), (16, 4)

    # fmt: off
    @T.prim_func(tirp=True)
    def test_reduction(A_ptr: T.handle, B_ptr: T.handle) -> None:
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

                # red layout
                red_atom = T.TileLayout(shard=([1, 1], [1, 1]))
                red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))
                red_tile = T.TileLayout(shard=([2, 1], [1, 1]))
                red_layout = red_warp_atom.tile(red_tile, (2, 1), (8, 4))
                red = T.alloc_buffer(
                    [
                        2,
                    ],
                    dtype=dtype,
                    scope="local",
                    logical_scope="thread",
                    layout=red_atom.tile(red_tile, (2, 1), (1, 1)),
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

                # reduce
                with T.warp():
                    acc_view = T.view(acc, layout=acc_layout, shape=acc_shape)
                    red_view = T.view(red, layout=red_layout, shape=red_shape)
                    if op_type == "sum":
                        Tp.sum(red_view, acc_view, schedule_config={"thread_reduce": shuffle})
                    elif op_type == "max":
                        Tp.max(red_view, acc_view, schedule_config={"thread_reduce": shuffle})
                    # perform an additional shuffle step if not shuffled above
                    if not shuffle:
                        if op_type == "sum":
                            Tp.sum(red_view, red_view, schedule_config={"thread_reduce": True})
                        elif op_type == "max":
                            Tp.max(red_view, red_view, schedule_config={"thread_reduce": True})

                # write red into B
                with T.thread():
                    for i in T.unroll(2):
                        B[wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4] = (
                            red[i]
                        )

    # fmt: on

    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.IRModule({"main": test_reduction})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.nd.array(A_np, dev)
        B = tvm.nd.array(B_np, dev)
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


if __name__ == "__main__":
    tvm.testing.main()
