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
import pytest
import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.layout import TileLayout, SwizzleLayout, ComposeLayout

ml_dtypes_dict = {
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "bfloat16": ml_dtypes.bfloat16,
    "int4": ml_dtypes.int4,
}


@pytest.mark.parametrize(
    "task",
    [
        ################################################################################ vectorized copy
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            ((0, 8), (0, 8)),  # g_region
            8,  # thread_cnt
            TileLayout([16, 16]),  # layoutA
            TileLayout([16, 16]),  # layoutB
            TileLayout([8, 8]),  # layoutS
            tvm.cuda(0),
        ),
        # A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32]
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            ((0, 128), (0, 32)),  # g_region
            32,  # thread_cnt
            TileLayout([128, 32]),  # layoutA
            TileLayout([128, 32]),  # layoutB
            TileLayout([128, 32]),  # layoutS
            tvm.cuda(0),
        ),
        # A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64]
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            ((32, 64), (32, 64)),  # g_region
            32,  # thread_cnt
            TileLayout([64, 64]),  # layoutA
            TileLayout([64, 64]),  # layoutB
            TileLayout([32, 32]),  # layoutS
            tvm.cuda(0),
        ),
        # A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32]
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            ((0, 1), (0, 32), (0, 32)),  # g_region
            32,  # thread_cnt
            TileLayout([4, 32, 32]),  # layoutA
            TileLayout([4, 32, 32]),  # layoutB
            TileLayout([32, 32]),  # layoutS
            tvm.cuda(0),
        ),
        ############################################################################### default
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            ((0, 8), (0, 8)),  # g_region
            32,  # thread_cnt
            TileLayout([16, 16]),  # layoutA
            TileLayout([16, 16]),  # layoutB
            TileLayout([8, 64]),  # layoutS
            tvm.cuda(0),
        ),
        # A[32:96, 256:512] -> A_smem[0:32, 0:256] -> B[32:96, 256:512]
        (
            (96, 512),  # g_shape
            (32, 256),  # s_shape
            ((16, 48), (256, 512)),  # g_region
            32,  # thread_cnt
            TileLayout([96, 512]),  # layoutA
            TileLayout([96, 512]),  # layoutB
            ComposeLayout(SwizzleLayout(3, 3, 3), TileLayout([8, 64]))
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
        scoper = T.cta
    elif scope == "thread":
        scoper = T.thread
        thread_cnt = 1

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with scoper():
                A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                Tp.copy(A_smem[*r_smem], A[*r_gmem])
                Tp.copy(B[*r_gmem], A_smem[*r_smem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[*r_gmem] = A_np[*r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.parametrize(
    "task",
    [
        ################################################################################ vectorized copy
        # A[0:8, 0:8] -> A_local[0:8, 0:8] -> B[0:8, 0:8]
        (
            (4, 16, 16),  # g_shape
            (8, 8),  # l_shape
            ((3, 4), (8, 16), (8, 16)),  # g_region
            1,  # thread_cnt
            TileLayout([4, 16, 16]),  # layoutA
            TileLayout([4, 16, 16]),  # layoutB
            TileLayout([8, 8]),  # layoutLocal
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
    @T.prim_func(tirp=True)
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            tx = T.thread_id([thread_cnt], parent="cta")

            with T.thread():
                A_local = T.alloc_buffer(l_shape, dtype, scope="local", layout=layoutLocal)

                Tp.copy(A_local[*r_lmem], A[*r_gmem])
                Tp.copy(B[*r_gmem], A_local[*r_lmem])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[*r_gmem] = A_np[*r_gmem]
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

    g_layout = TileLayout(shard=([128, WIDTH // VEC_LEN, VEC_LEN], [WIDTH, VEC_LEN, 1]))
    local_view = TileLayout(shard=([128, WIDTH], [(1, "tid_in_wg"), (1, "m")]))

    # fmt: off
    @T.prim_func(tirp=True)
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)
        
        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with T.kernel():
            bx = T.cta_id([1], parent="kernel")
            wg_id = T.warpgroup_id([1], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid_in_wg = T.thread_id([128], parent="cta")

            tmem_addr = T.alloc_shared([1], "uint32")

            with T.warpgroup()[0:1]:
                with T.warp()[0:1]:
                    T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)

                T.tvm_storage_sync("shared")

                tmem = T.decl_buffer((128, OFFSET + WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                     layout=TileLayout(([128, OFFSET + WIDTH], [(1, "TCol"), (1, "TLane")])))

                A_reg = T.alloc_local((WIDTH), dtype)
                B_reg = T.alloc_local((WIDTH), dtype)
                A_local = A_reg.view(128, WIDTH, layout=local_view) # collective view of the whole warpgroup
                B_local = B_reg.view(128, WIDTH, layout=local_view) # collective view of the whole warpgroup

                # A -> A_local
                with T.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tp.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])
                    for i in range(WIDTH):
                        B_reg[i] = T.cast(0, dtype)
                T.cuda.cta_sync()

                # A_local -> tmem
                Tp.copy(tmem[:, OFFSET: OFFSET + WIDTH], A_local[:, :])
                T.cuda.cta_sync()

                # tmem -> B_local
                Tp.copy(B_local[:, :], tmem[:, OFFSET: OFFSET + WIDTH])
                T.cuda.cta_sync()

                # B_local -> B
                with T.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tp.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])

                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source())
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


if __name__ == "__main__":
    tvm.testing.main()
