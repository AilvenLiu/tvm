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
# pylint: disable=invalid-name, missing-function-docstring
"""Tests for the TMEM copy_async dispatch (tcgen05-based tmem<->reg and smem<->tmem)."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg
from tvm.tirx.operator.tile_primitive_dispatch.cuda.common import next_power_of_2
from tvm.tirx.operator.tile_primitive_dispatch.cuda.tma_utils import (
    SwizzleMode,
    mma_atom_shape,
    mma_shared_layout,
)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
def test_copy_tmem2reg_async(dtype, width_32b):
    """Test async tmem<->local copy using copy_async instead of copy.

    This tests the new copy_async dispatch for tmem<->local that doesn't
    immediately wait after the operation, allowing for pipelining.
    """

    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async_test(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            warp_id = Tx.warp_id([(128) // 32])
            cta_id = Tx.cta_id([2])
            wg_id = Tx.warpgroup_id([1])
            warp_id_in_wg = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            tid_in_wg = Tx.thread_id([128])

            tmem_addr = Tx.alloc_shared([1], "uint32")

            if Tx.filter(wg_id, 0, 1):
                with Tx.warpgroup():
                    if Tx.filter(warp_id, 0, 1):
                        with Tx.warp():
                            Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

                    Tx.tvm_storage_sync("shared")

                    tmem = Tx.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                         layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

                    A_reg = Tx.alloc_local((WIDTH), dtype)
                    B_reg = Tx.alloc_local((WIDTH), dtype)
                    A_local = A_reg.view(128, WIDTH, layout=local_view)
                    B_local = B_reg.view(128, WIDTH, layout=local_view)

                    # A -> A_local
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()

                    # A_local -> tmem (async)
                    Tx.copy_async(tmem[:, :], A_local[:, :])
                    Tx.ptx.tcgen05.wait.st()  # explicit wait
                    Tx.cuda.cta_sync()

                    # tmem -> B_local (async)
                    Tx.copy_async(B_local[:, :], tmem[:, :])
                    Tx.ptx.tcgen05.wait.ld()  # explicit wait
                    Tx.cuda.cta_sync()

                    # B_local -> B
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

                    if Tx.filter(warp_id, 0, 1):
                        with Tx.warp():
                            Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                            Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


################################################################################
# SMEM->TMEM async copy tests (consolidated)
################################################################################


def _smem2tmem_async_task(
    dtype, width_32b, *, swizzle_mode=0, smem_rows=32, cta_group=1, partial=None, multicast=""
):
    """Create a task dict for smem2tmem async tests."""
    return dict(
        dtype=dtype,
        width_32b=width_32b,
        swizzle_mode=swizzle_mode,
        smem_rows=smem_rows,
        cta_group=cta_group,
        partial=partial,
        multicast=multicast,
    )


SMEM2TMEM_ASYNC_TASKS = [
    # --- basic (no swizzle) ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w), id=f"basic-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [2, 4, 8, 16, 32, 64, 128]
    ],
    # --- swizzle modes ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w, swizzle_mode=sm), id=f"swizzle{sm}-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [4, 8, 16, 128]
        for sm in [1, 2, 3]
    ],
    # --- cta_group=2 (cluster copy) ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w, cta_group=2), id=f"cta_group2-{dt}-{w}")
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
    ],
    # --- partial region ---
    *[
        pytest.param(
            _smem2tmem_async_task(dt, w, partial=(sr, sc, tc)),
            id=f"partial-{dt}-{w}-{sr}-{sc}-{tc}",
        )
        for dt in ["float16", "float32"]
        for w in [8, 16]
        for sr, sc, tc in [(32, 0, 4), (0, 4, 16), (64, 8, 12)]
    ],
    # --- shape / multicast ---
    *[
        pytest.param(
            _smem2tmem_async_task(dt, w, smem_rows=int(sh.split("x")[0]), multicast=mc),
            id=f"shape-{sh}-mc{mc or 'none'}-{dt}-{w}",
        )
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
        for sh, mc in [("128x256b", ""), ("128x128b", ""), ("32x128b", "warpx4")]
    ],
]


@pytest.mark.parametrize("task", SMEM2TMEM_ASYNC_TASKS)
def test_smem2tmem_async(task):
    """Consolidated SMEM->TMEM async copy test covering basic, swizzle, cta_group,
    partial region, and shape/multicast variants."""
    dtype = task["dtype"]
    width_32b = task["width_32b"]
    swizzle_mode = task["swizzle_mode"]
    smem_rows = task["smem_rows"]
    cta_group = task["cta_group"]
    partial = task["partial"]
    multicast = task["multicast"]

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")
    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    # Swizzle compatibility check
    if swizzle_mode > 0:
        atom_shape = mma_atom_shape(dtype, SwizzleMode(swizzle_mode))
        atom_cols = atom_shape[-1]
        if WIDTH < atom_cols or WIDTH % atom_cols != 0:
            pytest.skip(f"WIDTH {WIDTH} not compatible with swizzle mode {swizzle_mode}")

    # Compute dimensions and layouts
    if partial:
        smem_row_start, smem_col_start_32b, tmem_col_offset_32b = partial
        SMEM_COL_START = smem_col_start_32b * (32 // bits)
        TMEM_COL_OFFSET = tmem_col_offset_32b * (32 // bits)
        A_ROWS = 128
        A_WIDTH = ((WIDTH + SMEM_COL_START + VEC_LEN - 1) // VEC_LEN) * VEC_LEN
        TMEM_WIDTH = TMEM_COL_OFFSET + WIDTH
        smem_shape = (A_ROWS, A_WIDTH)
        a_shape = smem_shape
        tmem_n_cols = max(32, next_power_of_2(tmem_col_offset_32b + width_32b))
    else:
        SMEM_COL_START = 0
        TMEM_COL_OFFSET = 0
        A_ROWS = smem_rows
        A_WIDTH = WIDTH
        TMEM_WIDTH = WIDTH
        smem_shape = (smem_rows, WIDTH)
        a_shape = (128, WIDTH)
        tmem_n_cols = max(32, next_power_of_2(width_32b))

    if swizzle_mode > 0:
        smem_layout = mma_shared_layout(dtype, SwizzleMode(swizzle_mode), smem_shape)
    else:
        smem_layout = TileLayout(S[smem_rows, VEC_LEN]).tile_to(smem_shape, (smem_rows, VEC_LEN))
    local_view = TileLayout(S[(128, WIDTH) : (1 @ tid_in_wg, 1)])
    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])

    is_partial = partial is not None
    is_cluster = cta_group == 2
    B_ROWS = 256 if is_cluster else 128
    CTA_STRIDE = 128 * WIDTH

    if is_cluster:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def smem2tmem_async_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                warp_id = Tx.warp_id([(128) // 32])
                cbx, cby = Tx.cta_id_in_cluster([2, 1])
                bx = Tx.cta_id([2])
                wg_id = Tx.warpgroup_id([1])
                warp_id_in_wg = Tx.warp_id_in_wg([4])
                lane_id = Tx.lane_id([32])
                tid_in_wg = Tx.thread_id([128])
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                if Tx.filter(warp_id, 0, 1):
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
                Tx.cuda.cta_sync()
                Tx.cuda.cluster_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                if Tx.filter(tid_in_wg, 0, 1):
                    with Tx.thread():
                        Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.mbarrier_init()
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                if Tx.filter(wg_id, 0, 1):
                    with Tx.warpgroup():
                        with Tx.cta():
                            Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                            for i in range(WIDTH):
                                B_reg[i] = Tx.cast(0, dtype)
                        Tx.cuda.cta_sync()
                        Tx.cuda.cluster_sync()
                        if cbx == 0:
                            if Tx.filter(tid_in_wg, 0, 1):
                                with Tx.thread():
                                    Tx.copy_async(tmem[:, 0:WIDTH], A_smem[:, :],
                                            mbar=cp_mbar.ptr_to([0]), cta_group=cta_group, cta_mask=3)  # noqa: E501
                        Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.copy_async(B_local[:, :], tmem[:, 0:WIDTH])
                        Tx.ptx.tcgen05.wait.ld()
                        Tx.cuda.cta_sync()
                        cta_row_offset = Tx.meta_var(cbx * CTA_STRIDE)
                        with Tx.thread():
                            for i in range(WIDTH // VEC_LEN):
                                g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                                Tx.copy(B_flat[cta_row_offset + g_offset: cta_row_offset + g_offset + VEC_LEN],  # noqa: E501
                                        B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                        if Tx.filter(warp_id, 0, 1):
                            with Tx.warp():
                                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def smem2tmem_async_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                warp_id = Tx.warp_id([(128) // 32])
                bx = Tx.cta_id([1])
                wg_id = Tx.warpgroup_id([1])
                warp_id_in_wg = Tx.warp_id_in_wg([4])
                lane_id = Tx.lane_id([32])
                tid_in_wg = Tx.thread_id([128])
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                if Tx.filter(warp_id, 0, 1):
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=1)  # noqa: E501
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                if Tx.filter(tid_in_wg, 0, 1):
                    with Tx.thread():
                        Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                if Tx.filter(wg_id, 0, 1):
                    with Tx.warpgroup():
                        with Tx.cta():
                            if is_partial:
                                Tx.copy(A_smem[:, :], A[:, :])
                            else:
                                Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                            for i in range(WIDTH):
                                B_reg[i] = Tx.cast(0, dtype)
                        Tx.cuda.cta_sync()
                        if Tx.filter(tid_in_wg, 0, 1):
                            with Tx.thread():
                                if is_partial:
                                    Tx.copy_async(tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH],
                                            A_smem[smem_row_start:smem_row_start + 32, SMEM_COL_START:SMEM_COL_START + WIDTH],  # noqa: E501
                                            mbar=cp_mbar.ptr_to([0]), cta_group=1)
                                else:
                                    Tx.copy_async(tmem[:, 0:WIDTH], A_smem[:, :],
                                            mbar=cp_mbar.ptr_to([0]), cta_group=1)
                        Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.copy_async(B_local[:, :], tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH])  # noqa: E501
                        Tx.ptx.tcgen05.wait.ld()
                        Tx.cuda.cta_sync()
                        with Tx.thread():
                            for i in range(WIDTH // VEC_LEN):
                                g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN],
                                        B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                        if Tx.filter(warp_id, 0, 1):
                            with Tx.warp():
                                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=1)  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": smem2tmem_async_kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, a_shape)
        B_np = np.zeros((B_ROWS, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)

        B_result = B.numpy()
        if partial:
            expected = A_np[
                smem_row_start : smem_row_start + 32, SMEM_COL_START : SMEM_COL_START + WIDTH
            ]
        elif multicast == "warpx4":
            expected = A_np[0:32, :]
        elif smem_rows < 128:
            expected = A_np[0:32, :]
        else:
            expected = A_np[0:32, :]

        n_warps = 8 if is_cluster else 4
        if not partial and multicast != "warpx4" and smem_rows >= 128:
            np.testing.assert_allclose(B_result, A_np[:B_ROWS, :], err_msg="data mismatch")
        else:
            for warp_idx in range(n_warps):
                np.testing.assert_allclose(
                    B_result[warp_idx * 32 : (warp_idx + 1) * 32, :],
                    expected,
                    err_msg=f"Warp {warp_idx} data mismatch",
                )


if __name__ == "__main__":
    tvm.testing.main()
