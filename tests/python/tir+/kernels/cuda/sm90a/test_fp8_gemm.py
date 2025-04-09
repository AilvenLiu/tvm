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
import ml_dtypes

import tvm
import tvm.testing
from tvm.script.ir_builder import IRBuilder
from tvm.script import tir as T
from ..utils import bench, ProtonContext


@tvm.testing.requires_cuda_compute_version(9)
def test_fp8_gemm_hopper_no_ws():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")

    f8_bytes = 1
    f16_bytes = 2
    SM_COUNT = DEV.multi_processor_count
    M, N, K = 8192, 8192, 8192
    BLK_M, BLK_N, BLK_K = 128, 256, 128
    WGMMA_M, WGMMA_N, WGMMA_K = 64, 256, 32

    swizzleA = swizzleB = 3  # 128B swizzle
    swizzleC = 3  # 128B swizzle
    GROUP_SIZE = 16
    STAGES_TMA = 4
    STAGES_WGMMA = 2
    STAGES_EPI = 2
    WG_SIZE = 128
    TMA_BYTES = BLK_M * BLK_K * f8_bytes + BLK_K * BLK_N * f8_bytes

    class TileScheduler:
        m_blocks = (M + BLK_M - 1) // BLK_M
        n_blocks = (N + BLK_N - 1) // BLK_N

        def __init__(self, prefix: str):
            self.m_idx = int_var()
            self.n_idx = int_var()
            self.linear_idx = int_var()
            IRBuilder.current().name(prefix + "_m_idx", self.m_idx)
            IRBuilder.current().name(prefix + "_n_idx", self.n_idx)
            IRBuilder.current().name(prefix + "_linear_idx", self.linear_idx)

        def get_current_m_n_idx(self, linear_idx):
            group_row_outer = linear_idx // (GROUP_SIZE * self.n_blocks)
            group_row_inner = linear_idx % GROUP_SIZE
            row = group_row_outer * GROUP_SIZE + group_row_inner
            col = linear_idx // GROUP_SIZE % self.n_blocks
            return row, col

        @T.macro
        def init(self, linear_init):
            self.linear_idx[0] = linear_init
            self.m_idx[0] = self.get_current_m_n_idx(linear_init)[0]
            self.n_idx[0] = self.get_current_m_n_idx(linear_init)[1]

        @T.macro
        def next_tile(self):
            self.linear_idx[0] = self.linear_idx[0] + SM_COUNT
            self.m_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[0]
            self.n_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[1]

        def valid(self):
            return self.linear_idx[0] < self.m_blocks * self.n_blocks

    def int_var():
        return T.alloc_buffer([1], "int32", scope="local", align=4)

    # fmt: off
    @T.macro
    def tma_load(tid, m_idx, n_idx, k_tile, A_smem: tvm.tir.Buffer, B_smem: tvm.tir.Buffer, A_map, B_map, bars: tvm.tir.Buffer):
        with T.thread()[tid == 0]:
            stage = T.meta_var(k_tile % STAGES_TMA)
            T.ptx.mbarrier.arrive.expect_tx(bars.access_ptr("rw", offset=stage), TMA_BYTES)
            T.ptx.cp_async.bulk.tensor.g2c(
                2, A_smem.access_ptr("w", offset=A_smem.offset_of_p([stage, 0])),
                bars.access_ptr("rw", offset=stage), A_map, k_tile * BLK_K, m_idx * BLK_M,
            )
            T.ptx.cp_async.bulk.tensor.g2c(
                2, B_smem.access_ptr("w", offset=B_smem.offset_of_p([stage, 0])),
                bars.access_ptr("rw", offset=stage), B_map, k_tile * BLK_K, n_idx * BLK_N,
            )

    def get_accum_list(C, C_elems):
        return [C[i] for i in range(C_elems)]

    @T.macro
    def mma_compute(wg_id, k_tile, A_smem: tvm.tir.Buffer, B_smem: tvm.tir.Buffer, accum, bars: tvm.tir.Buffer, descA, descB):
        stage = T.meta_var(k_tile % STAGES_TMA)
        parity = T.meta_var((k_tile // STAGES_TMA) % 2)
        T.ptx.mbarrier.try_wait(bars.access_ptr("rw", offset=stage), parity)
        for inner_k in T.serial(BLK_K // WGMMA_K):
            A_offset = T.meta_var(wg_id * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
            B_offset = T.meta_var(inner_k * WGMMA_K)
            T.ptx.encode_matrix_descriptor(descA.data, A_smem.access_ptr("r", offset=A_smem.offset_of_p([stage, A_offset])), 1, 64, swizzle=3)
            T.ptx.encode_matrix_descriptor(descB.data, B_smem.access_ptr("r", offset=B_smem.offset_of_p([stage, B_offset])), 1, 64, swizzle=3)
            T.ptx.wgmma.mma_async.ss(WGMMA_M, WGMMA_N, WGMMA_K, "float8_e4m3fn", "float32", False, False, 1.0, 1.0, True,
                                     descA[0], descB[0], *get_accum_list(accum, 128))
        T.ptx.wgmma.commit_group()

    @T.macro
    def r2S(warp_id, lane_id, C_smem: tvm.tir.Buffer, accum, accum_half, n_tile):
        T.tvm_storage_sync("shared")
        for st_tile in T.serial(4):
            for i in T.serial(8):
                accum_half[i] = accum[n_tile * 32 + st_tile * 8 + i]
            col_noswizzle = T.meta_var(st_tile * 2 + lane_id // 16)
            col = T.meta_var((lane_id % 8) ^ col_noswizzle)
            row = T.meta_var(warp_id * 16 + lane_id % 16)
            T.ptx.stmatrix(4, False, C_smem.access_ptr("w", offset=C_smem.offset_of_p([n_tile % STAGES_EPI, row, col * 8])),
                           accum_half[0], accum_half[1], accum_half[2], accum_half[3],
                           accum_half[4], accum_half[5], accum_half[6], accum_half[7])

    @T.macro
    def s2G(warp_id, lane_id, C_smem: tvm.tir.Buffer, C_map, m_idx, n_idx, n_tile):
        T.ptx.fence.proxy("shared")
        T.tvm_storage_sync("shared")
        with T.thread()[warp_id == 0 and lane_id == 0]:
            T.ptx.cp_async.bulk.tensor.s2g(2, C_smem.access_ptr("r", offset=C_smem.offset_of_p([n_tile % STAGES_EPI, 0, 0])),
                                           C_map, n_idx * BLK_N + n_tile * 64, m_idx * BLK_M)
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(1, read=True)

    @T.macro
    def write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem: tvm.tir.Buffer, C_map, accum, accum_half):
        T.tvm_storage_sync("shared")
        for n_tile in T.serial(BLK_N // 64):
            if n_tile != 0:
                # s2G for the previous stage
                s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, n_tile - 1)
            # r2S for the current stage
            r2S(warp_id, lane_id, C_smem, accum, accum_half, n_tile)
        # s2G for the last stage
        s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, BLK_N // 64 - 1)
        T.ptx.cp_async.bulk.wait_group(0, read=False)

    @T.prim_func(tirp=True)
    def manual(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), "float8_e4m3fn", scope="global", layout=T.TileLayout.from_tuple((M, K)))
        B = T.match_buffer(B_ptr, (N, K), "float8_e4m3fn", scope="global", layout=T.TileLayout.from_tuple((N, K)))
        C = T.match_buffer(C_ptr, (M, N), "float16", scope="global", layout=T.TileLayout.from_tuple((M, N)))

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float8_e4m3fn", 2, A.data, K, M, f8_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float8_e4m3fn", 2, B.data, K, N, f8_bytes * K, BLK_K, BLK_N, 1, 1, 0, swizzleB, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", C_map, "float16", 2, C.data, N, M, f16_bytes * N, 64, BLK_M, 1, 1, 0, swizzleC, 0, 0)

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            tid = T.thread_id([128 * 2], parent="cta")
            warp_id = T.warp_id([8], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            wg_id = T.warpgroup_id([2], parent="cta")

            with T.cta():
                # tensor stroage
                A_smem = T.alloc_buffer([STAGES_TMA, BLK_M * BLK_K], "float8_e4m3fn", scope="shared.dyn", align=128)
                B_smem = T.alloc_buffer([STAGES_TMA, BLK_K * BLK_N], "float8_e4m3fn", scope="shared.dyn", align=128)
                C_smem = T.alloc_buffer([STAGES_EPI, BLK_M, 64], "float16", scope="shared.dyn", align=1024)
                # barriers
                bars = T.alloc_buffer([STAGES_TMA], "uint64", scope="shared.dyn", align=8)
                desc_A = T.alloc_buffer([1], "uint64", scope="local", align=8)
                desc_B = T.alloc_buffer([1], "uint64", scope="local", align=8)

                with T.thread():
                    # index
                    tma_index = int_var()
                    mma_index = int_var()
                    # acuumulators
                    accum = T.alloc_buffer([128], "float32", scope="local")
                    accum_half = T.alloc_buffer([8], "float16", scope="local")
                    # tile scheduler
                    tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))

                    # initialize the tile scheduler
                    tile_scheduler.init(bx)

                    while (tile_scheduler.valid()):
                        # initialize the barriers
                        with T.thread()[tid == 0]:
                            for i in range(STAGES_TMA):
                                T.ptx.mbarrier.init(bars.access_ptr("rw", offset=i), 1)
                        T.tvm_storage_sync("shared")
                        # initialize the index
                        tma_index[0] = 0
                        mma_index[0] = 0
                        # initialize the accumulators
                        for i in range(128):
                            accum[i] = 0
                        T.tvm_storage_sync("shared")

                        m_idx = T.meta_var(tile_scheduler.m_idx[0])
                        n_idx = T.meta_var(tile_scheduler.n_idx[0])
                        # prelogue
                        for _ in range(STAGES_TMA):
                            tma_load(tid, m_idx, n_idx, tma_index[0], A_smem, B_smem, A_map, B_map, bars)
                            tma_index[0] = tma_index[0] + 1
                        for _ in range(STAGES_WGMMA - 1):
                            mma_compute(wg_id, mma_index[0], A_smem, B_smem, accum, bars, desc_A, desc_B)
                            mma_index[0] = mma_index[0] + 1

                        # mainloop
                        k_tile_count = T.meta_var((K + BLK_K - 1) // BLK_K)
                        for _ in range(k_tile_count):
                            if mma_index[0] < k_tile_count:
                                mma_compute(wg_id, mma_index[0], A_smem, B_smem, accum, bars, desc_A, desc_B)
                                mma_index[0] = mma_index[0] + 1
                            # wait for oldest one stage to finish
                            if _ == k_tile_count - 1:
                                T.ptx.wgmma.wait_group(0)
                            else:
                                T.ptx.wgmma.wait_group(STAGES_WGMMA - 1)
                            T.tvm_storage_sync("shared")
                            # load the next tile
                            if tma_index[0] < k_tile_count:
                                tma_load(tid, m_idx, n_idx, tma_index[0], A_smem, B_smem, A_map, B_map, bars)
                                tma_index[0] = tma_index[0] + 1

                        # epilogue
                        write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem, C_map, accum, accum_half)

                        # move to the next tile
                        tile_scheduler.next_tile()
    # fmt: on

    A_np = np.random.randn(M, K).astype(ml_dtypes.float8_e4m3fn)
    B_np = np.random.randn(N, K).astype(ml_dtypes.float8_e4m3fn)
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)

    def flops(ms):
        return M * N * K * 2 / ms / 1e9

    def cublas_gemm():
        C_np = np.empty((M, N)).astype(np.float16)

        gemm_func = tvm.get_global_func("cublaslt.fp8_gemm")

        C_tvm = tvm.nd.array(C_np, device=DEV)
        workspace = tvm.nd.empty((32 * 1024 * 1024,), dtype="uint8", device=DEV)
        scale = tvm.nd.array(np.array([1.0], dtype="float32"), device=DEV)
        func = lambda: gemm_func(A_tvm, B_tvm, workspace, scale, C_tvm)
        ms = bench(func, warmup=0, repeat=10, proton_name="cublas")
        print(f"CUBLAS flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        return C_tvm

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float16)
        C_tvm = tvm.nd.array(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        return C_tvm

    with ProtonContext("matmul"):
        C_cublas = cublas_gemm()
        C_tir = tir_gemm()

    tvm.testing.assert_allclose(C_tir.numpy(), C_cublas.numpy(), rtol=2e-2, atol=1e-4)


if __name__ == "__main__":
    test_fp8_gemm_hopper_no_ws()
