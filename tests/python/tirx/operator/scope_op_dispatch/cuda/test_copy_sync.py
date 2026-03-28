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
# pylint: disable=missing-function-docstring
import ml_dtypes
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.operator.scope_op_dispatch.cuda.common import next_power_of_2
from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import (
    SwizzleMode,
    tma_atom_shape,
    tma_shared_layout,
)

ml_dtypes_dict = {
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "bfloat16": ml_dtypes.bfloat16,
    "int4": ml_dtypes.int4,
}


@pytest.mark.parametrize(
    "task",
    [
        ################################################################################ vectorized copy  # noqa: E501
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            ((0, 8), (0, 8)),  # g_region
            8,  # thread_cnt
            TileLayout(S[16, 16]),  # layoutA
            TileLayout(S[16, 16]),  # layoutB
            TileLayout(S[8, 8]),  # layoutS
            tvm.cuda(0),
        ),
        # A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32]
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            ((0, 128), (0, 32)),  # g_region
            32,  # thread_cnt
            TileLayout(S[128, 32]),  # layoutA
            TileLayout(S[128, 32]),  # layoutB
            TileLayout(S[128, 32]),  # layoutS
            tvm.cuda(0),
        ),
        # A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64]
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            ((32, 64), (32, 64)),  # g_region
            32,  # thread_cnt
            TileLayout(S[64, 64]),  # layoutA
            TileLayout(S[64, 64]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
            tvm.cuda(0),
        ),
        # A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32]
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            ((0, 1), (0, 32), (0, 32)),  # g_region
            32,  # thread_cnt
            TileLayout(S[4, 32, 32]),  # layoutA
            TileLayout(S[4, 32, 32]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
            tvm.cuda(0),
        ),
        ############################################################################### default
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            ((0, 8), (0, 8)),  # g_region
            32,  # thread_cnt
            TileLayout(S[16, 16]),  # layoutA
            TileLayout(S[16, 16]),  # layoutB
            TileLayout(S[8, 64]),  # layoutS
            tvm.cuda(0),
        ),
        # A[32:96, 256:512] -> A_smem[0:32, 0:256] -> B[32:96, 256:512]
        (
            (96, 512),  # g_shape
            (32, 256),  # s_shape
            ((16, 48), (256, 512)),  # g_region
            32,  # thread_cnt
            TileLayout(S[96, 512]),  # layoutA
            TileLayout(S[96, 512]),  # layoutB
            ComposeLayout(SwizzleLayout(3, 3, 3), TileLayout(S[8, 64]))
            .tile_to((16, 128), (8, 64))
            .tile_to((32, 256), (16, 128)),  # layoutS
            tvm.cuda(0),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("scope", ["cta", "thread"])
def test_copy_g2s_s2g(task, dtype, scope):
    g_shape, s_shape, g_region, thread_cnt, layoutA, layoutB, layoutS, dev = task

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    if scope == "cta":
        scoper = Tx.cta
    elif scope == "thread":
        scoper = Tx.thread
        thread_cnt = 1

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_sync(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with scoper():
                A_smem = Tx.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                Tx.copy(A_smem[tuple(r_smem)], A[tuple(r_gmem)])
                Tx.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.parametrize(
    "task",
    [
        ################################################################################ vectorized copy  # noqa: E501
        # A[0:8, 0:8] -> A_local[0:8, 0:8] -> B[0:8, 0:8]
        (
            (4, 16, 16),  # g_shape
            (8, 8),  # l_shape
            ((3, 4), (8, 16), (8, 16)),  # g_region
            1,  # thread_cnt
            TileLayout(S[4, 16, 16]),  # layoutA
            TileLayout(S[4, 16, 16]),  # layoutB
            TileLayout(S[8, 8]),  # layoutLocal
            tvm.cuda(0),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2l_l2g_vec_load(task, dtype):
    g_shape, l_shape, g_region, thread_cnt, layoutA, layoutB, layoutLocal, dev = task

    r_lmem = list(slice(None) for i in range(len(l_shape)))
    r_gmem = list(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_sync(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                A_local = Tx.alloc_buffer(l_shape, dtype, scope="local", layout=layoutLocal)

                Tx.copy(A_local[tuple(r_lmem)], A[tuple(r_gmem)])
                Tx.copy(B[tuple(r_gmem)], A_local[tuple(r_lmem)])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("width_32b", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("offset_32b", [0, 3, 10])
def test_copy_tmem2reg(dtype, width_32b, offset_32b):
    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    OFFSET = offset_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ tid_in_wg, 1)])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_sync(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            Tx.warp_id([4], parent="warpgroup")
            Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="cta")

            tmem_addr = Tx.alloc_shared([1], "uint32")

            with Tx.warpgroup()[0:1]:
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)  # noqa: E501

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer((128, OFFSET + WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                     layout=TileLayout(S[(128, OFFSET + WIDTH) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

                A_reg = Tx.alloc_local((WIDTH), dtype)
                B_reg = Tx.alloc_local((WIDTH), dtype)
                A_local = A_reg.view(128, WIDTH, layout=local_view) # collective view of the whole warpgroup  # noqa: E501
                B_local = B_reg.view(128, WIDTH, layout=local_view) # collective view of the whole warpgroup  # noqa: E501

                # A -> A_local
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
                    for i in range(WIDTH):
                        B_reg[i] = Tx.cast(0, dtype)
                Tx.cuda.cta_sync()

                # A_local -> tmem
                Tx.copy(tmem[:, OFFSET: OFFSET + WIDTH], A_local[:, :])
                Tx.cuda.cta_sync()

                # tmem -> B_local
                Tx.copy(B_local[:, :], tmem[:, OFFSET: OFFSET + WIDTH])
                Tx.cuda.cta_sync()

                # B_local -> B
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)  # noqa: E501
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
@pytest.mark.parametrize("local_offset_32b", [0, 2, 4])
def test_copy_tmem2reg_sliced_local(dtype, width_32b, local_offset_32b):
    """Test tmem<->local copy with sliced local buffer region.

    This tests the fix for handling non-zero local buffer start offset:
    - Using local_region.region[1].extent instead of local_buf.shape[1]
    - Correctly indexing with local_st[1] offset
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
    LOCAL_OFFSET = local_offset_32b * (32 // bits)
    TOTAL_LOCAL_WIDTH = WIDTH + LOCAL_OFFSET
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0 or TOTAL_LOCAL_WIDTH % VEC_LEN != 0:
        pytest.skip(
            f"dtype {dtype} + width {width_32b} + offset {local_offset_32b} is not supported"
        )

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, TOTAL_LOCAL_WIDTH) : (1 @ tid_in_wg, 1)])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_sync(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            Tx.warp_id([4], parent="warpgroup")
            Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="cta")

            tmem_addr = Tx.alloc_shared([1], "uint32")

            with Tx.warpgroup()[0:1]:
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                     layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

                # Allocate larger local buffer, but only use a slice
                A_reg = Tx.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
                B_reg = Tx.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
                A_local = A_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)
                B_local = B_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)

                # A -> A_local (only the slice we care about)
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(A_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
                    for i in range(TOTAL_LOCAL_WIDTH):
                        B_reg[i] = Tx.cast(0, dtype)
                Tx.cuda.cta_sync()

                # A_local[sliced] -> tmem (use sliced region)
                Tx.copy(tmem[:, 0:WIDTH], A_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH])
                Tx.cuda.cta_sync()

                # tmem -> B_local[sliced] (use sliced region)
                Tx.copy(B_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH], tmem[:, 0:WIDTH])
                Tx.cuda.cta_sync()

                # B_local -> B
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN])  # noqa: E501

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


################################################################################
# SMEM->TMEM copy tests (consolidated)
################################################################################


def _smem2tmem_task(
    dtype, width_32b, *, swizzle_mode=0, smem_rows=32, cta_group=1, partial=None, multicast=""
):
    """Create a task dict for smem2tmem tests."""
    return dict(
        dtype=dtype,
        width_32b=width_32b,
        swizzle_mode=swizzle_mode,
        smem_rows=smem_rows,
        cta_group=cta_group,
        partial=partial,
        multicast=multicast,
    )


SMEM2TMEM_TASKS = [
    # --- basic (no swizzle) ---
    *[
        pytest.param(_smem2tmem_task(dt, w), id=f"basic-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [2, 4, 8, 16, 32, 64, 128]
    ],
    # --- swizzle modes ---
    *[
        pytest.param(_smem2tmem_task(dt, w, swizzle_mode=sm), id=f"swizzle{sm}-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [4, 8, 16, 128]
        for sm in [1, 2, 3]
    ],
    # --- cta_group=2 (cluster copy) ---
    *[
        pytest.param(_smem2tmem_task(dt, w, cta_group=2), id=f"cta_group2-{dt}-{w}")
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
    ],
    # --- partial region ---
    *[
        pytest.param(
            _smem2tmem_task(dt, w, partial=(sr, sc, tc)), id=f"partial-{dt}-{w}-{sr}-{sc}-{tc}"
        )
        for dt in ["float16", "float32"]
        for w in [8, 16]
        for sr, sc, tc in [(32, 0, 4), (0, 4, 16), (64, 8, 12)]
    ],
    # --- shape / multicast ---
    *[
        pytest.param(
            _smem2tmem_task(dt, w, smem_rows=int(sh.split("x")[0]), multicast=mc),
            id=f"shape-{sh}-mc{mc or 'none'}-{dt}-{w}",
        )
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
        for sh, mc in [("128x256b", ""), ("128x128b", ""), ("32x128b", "warpx4")]
    ],
    # --- 128x256b with swizzle (regression: desc_offset_16B within-atom scaling) ---
    # Tests the fix where within-atom offset must be multiplied by bytes_per_copy//16.
    # Only triggered when smem_rows=128 + col_bits%256==0 (→128x256b) + swizzle>0.
    pytest.param(
        _smem2tmem_task("float16", 32, swizzle_mode=3, smem_rows=128), id="128x256b-swz3-f16-32"
    ),
    pytest.param(
        _smem2tmem_task("float16", 64, swizzle_mode=3, smem_rows=128), id="128x256b-swz3-f16-64"
    ),
    pytest.param(
        _smem2tmem_task("float16", 16, swizzle_mode=2, smem_rows=128), id="128x256b-swz2-f16-16"
    ),
    pytest.param(
        _smem2tmem_task("float16", 32, swizzle_mode=2, smem_rows=128), id="128x256b-swz2-f16-32"
    ),
    pytest.param(
        _smem2tmem_task("float32", 32, swizzle_mode=3, smem_rows=128), id="128x256b-swz3-f32-32"
    ),
    pytest.param(
        _smem2tmem_task("float32", 16, swizzle_mode=2, smem_rows=128), id="128x256b-swz2-f32-16"
    ),
]


@pytest.mark.parametrize("task", SMEM2TMEM_TASKS)
def test_smem2tmem(task):
    """Consolidated SMEM->TMEM copy test covering basic, swizzle, cta_group,
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
        atom_shape = tma_atom_shape(dtype, SwizzleMode(swizzle_mode))
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
        smem_layout = tma_shared_layout(dtype, SwizzleMode(swizzle_mode), smem_shape)
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
        def smem2tmem_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                cbx, cby = Tx.cta_id([2, 1], parent="cluster")
                bx = Tx.cta_id([2], parent="kernel")  # noqa: F841
                wg_id = Tx.warpgroup_id([1], parent="cta")  # noqa: F841
                warp_id = Tx.warp_id([4], parent="warpgroup")  # noqa: F841
                lane_id = Tx.thread_id([32], parent="warp")  # noqa: F841
                tid_in_wg = Tx.thread_id([128], parent="cta")
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
                Tx.cuda.cta_sync()
                Tx.cuda.cluster_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.mbarrier_init()
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.warpgroup()[0:1]:
                    with Tx.cta():
                        Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()
                    Tx.cuda.cluster_sync()
                    if cbx == 0:
                        with Tx.thread()[0:1]:
                            Tx.copy(tmem[:, 0:WIDTH], A_smem[:, :],
                                    mbar=cp_mbar.ptr_to([0]), cta_group=cta_group, cta_mask=3)
                    Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    Tx.copy(B_local[:, :], tmem[:, 0:WIDTH])
                    Tx.cuda.cta_sync()
                    cta_row_offset = Tx.meta_var(cbx * CTA_STRIDE)
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[cta_row_offset + g_offset: cta_row_offset + g_offset + VEC_LEN],  # noqa: E501
                                    B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def smem2tmem_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                bx = Tx.cta_id([1], parent="kernel")  # noqa: F841
                wg_id = Tx.warpgroup_id([1], parent="cta")  # noqa: F841
                warp_id = Tx.warp_id([4], parent="warpgroup")  # noqa: F841
                lane_id = Tx.thread_id([32], parent="warp")  # noqa: F841
                tid_in_wg = Tx.thread_id([128], parent="cta")
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=1)
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.warpgroup()[0:1]:
                    with Tx.cta():
                        if is_partial:
                            Tx.copy(A_smem[:, :], A[:, :])
                        else:
                            Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()
                    with Tx.thread()[0:1]:
                        if is_partial:
                            Tx.copy(tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH],
                                    A_smem[smem_row_start:smem_row_start + 32, SMEM_COL_START:SMEM_COL_START + WIDTH],  # noqa: E501
                                    mbar=cp_mbar.ptr_to([0]), cta_group=1)
                        else:
                            Tx.copy(tmem[:, 0:WIDTH], A_smem[:, :],
                                    mbar=cp_mbar.ptr_to([0]), cta_group=1)
                    Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    Tx.copy(B_local[:, :], tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH])
                    Tx.cuda.cta_sync()
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[g_offset: g_offset + VEC_LEN],
                                    B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=1)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": smem2tmem_kernel})
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
