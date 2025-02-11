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

import tvm
import tvm.testing
from tvm import te
from tvm.script.ir_builder import IRBuilder
from tvm.script import tir as T
from tvm.tir.transform import LowerTIRp
from tvm.contrib import cublas
from ..utils import bench, ProtonContext


@tvm.testing.requires_cuda_compute_version(9)
def test_hgemm_hopper_ws_cooperative():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")

    f16_bytes = 2
    f32_bytes = 4
    SM_COUNT = DEV.multi_processor_count
    M, N, K = 8192, 8192, 8192
    BLK_M, BLK_N, BLK_K = 128, 256, 64
    WGMMA_M, WGMMA_N, WGMMA_K = 64, 256, 16
    CLUSTER_M, CLUSTER_N, CLUSTER_K = 2, 1, 1
    swizzleA = swizzleB = swizzleC = 3  # 128B swizzle
    GROUP_SIZE = 8
    STAGES_MMA = 4
    STAGES_EPI = 2
    WG_SIZE = 128
    TMA_BYTES = BLK_M * BLK_K * f16_bytes + BLK_K * BLK_N * f16_bytes

    np.random.seed(0)

    # replicate of cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_void_f32_128x256x64_2x1x1_0_ttt_align8_warpspecialized_cooperative_epi_tma
    def ceildiv(a, b):
        return (a + b - 1) // b

    def int_var():
        return T.alloc_buffer([1], "int32", scope="local", align=4)

    class TileScheduler:
        m_blocks = ceildiv(ceildiv(M, BLK_M), CLUSTER_M) * CLUSTER_M
        n_blocks = ceildiv(ceildiv(N, BLK_N), CLUSTER_N) * CLUSTER_N

        def __init__(self, prefix: str):
            self.m_idx = int_var()
            self.n_idx = int_var()
            self.linear_idx = int_var()
            IRBuilder.current().name(prefix + "_m_idx", self.m_idx)
            IRBuilder.current().name(prefix + "_n_idx", self.n_idx)
            IRBuilder.current().name(prefix + "_linear_idx", self.linear_idx)

        def get_current_m_n_idx(self, linear_idx):
            clusters_per_row = self.n_blocks // CLUSTER_N

            div_cluster_x = linear_idx // CLUSTER_M
            mod_cluster_x = linear_idx % CLUSTER_M
            div_cluster_xy = div_cluster_x // CLUSTER_N
            mod_cluster_xy = div_cluster_x % CLUSTER_N

            group_row_outer = div_cluster_xy // (GROUP_SIZE * clusters_per_row)
            group_row_inner = div_cluster_xy % GROUP_SIZE
            cluster_row = group_row_outer * GROUP_SIZE + group_row_inner
            cluster_col = div_cluster_xy // GROUP_SIZE % clusters_per_row
            return cluster_row * CLUSTER_M + mod_cluster_x, cluster_col * CLUSTER_N + mod_cluster_xy

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

    class PipelineState:
        def __init__(self, prefix: str):
            self.index = int_var()
            self.phase = int_var()
            IRBuilder.current().name(prefix + "_index", self.index)
            IRBuilder.current().name(prefix + "_phase", self.phase)

        @T.macro
        def init(self, index, phase):
            self.index[0] = index
            self.phase[0] = phase

        @T.macro
        def copy(self, other):
            self.index[0] = other.index[0]
            self.phase[0] = other.phase[0]

        @T.macro
        def advance(self):
            self.index[0] = self.index[0] + 1
            if self.index[0] == STAGES_MMA:
                self.index[0] = 0
                self.phase[0] = self.phase[0] ^ 1

    class WarpGroupRole:
        PRODUCER = 0
        CONSUMER0 = 1
        CONSUMER1 = 2

    def get_accum_list(C, C_elems):
        return [C[i] for i in range(C_elems)]

    @T.macro
    def wgmma_fence_operand(accum):
        for i in T.serial(128):
            T.wgmma_fence_operand(accum[i])

    @T.macro
    def tma_store(n_tile, C_smem: tvm.tir.Buffer, C_map, m_glb_offset, n_glb_offset, tid):
        # make sure smem write is visible to TMA
        T.cuda_fence_proxy_async("shared")
        T.named_barrier_sync(0, 256)
        if tid == 128:
            # only 1 thread in 2 consumers write to TMA
            T.cp_async_bulk_tensor_shared_to_global(
                2,
                C_smem.access_ptr("r", offset=C_smem.offset_of_p([n_tile % STAGES_EPI, 0])),
                C_map,
                n_glb_offset + n_tile * 32,
                m_glb_offset,
            )
            T.cp_async_bulk_tensor_commit_group()
            T.cp_async_bulk_tensor_wait_group(n=1, read=False)

    # fmt: off
    @T.prim_func(tirp=True)
    def manual(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), "float16", scope="global", layout=T.TileLayout.from_tuple((M, K)))
        B = T.match_buffer(B_ptr, (K, N), "float16", scope="global", layout=T.TileLayout.from_tuple((K, N)))
        C = T.match_buffer(C_ptr, (M, N), "float32", scope="global", layout=T.TileLayout.from_tuple((M, N)))

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        T.call_packed("runtime.cuTensorMapInit", A_map, "float16", 2, A.data, K, M, f16_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)
        T.call_packed("runtime.cuTensorMapInit", B_map, "float16", 2, B.data, N, K, f16_bytes * N, 64, BLK_K // 2, 1, 1, 0, swizzleB, 0, 0) # multicast
        T.call_packed("runtime.cuTensorMapInit", C_map, "float32", 2, C.data, N, M, f32_bytes * N, 32, BLK_M, 1, 1, 0, swizzleC, 0, 0)

        with T.kernel():
            cbx, cby = T.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx, by = T.cta_id([CLUSTER_M, SM_COUNT // CLUSTER_M], parent="kernel")
            tid = T.thread_id([128 * 3], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.cta():
                # tensor stroage
                A_smem = T.alloc_buffer([STAGES_MMA, BLK_M * BLK_K], "float16", scope="shared.dyn", align=128)
                B_smem = T.alloc_buffer([STAGES_MMA, BLK_K * BLK_N], "float16", scope="shared.dyn", align=128)
                C_smem = T.alloc_buffer([STAGES_EPI, BLK_M * 32], "float32", scope="shared.dyn", align=1024)
                # mainloop pipeline barriers, note that these are cluster-wide barriers
                full = T.alloc_buffer([STAGES_MMA], "uint64", scope="shared.dyn", layout=T.TileLayout.from_tuple(STAGES_MMA), align=8)
                empty = T.alloc_buffer([STAGES_MMA], "uint64", scope="shared.dyn", layout=T.TileLayout.from_tuple(STAGES_MMA), align=8)

                with T.thread():
                    wg_id = int_var()
                    tid_in_wg = int_var()
                    warp_id_in_wg = int_var()
                    # work tile info
                    tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))
                    # produer pipelinen states
                    producer = T.meta_var(PipelineState("producer"))
                    # consumer pipeline states
                    consumer_read = T.meta_var(PipelineState("consumer_read"))
                    consumer_release = T.meta_var(PipelineState("consumer_release"))
                    # consumer signals
                    is_signal_thread = int_var()
                    dst_block_ID = int_var()
                    # smem desc_A, desc_B
                    desc_A = T.alloc_buffer([1], "uint64", scope="local", align=8)
                    desc_B = T.alloc_buffer([1], "uint64", scope="local", align=8)
                    # accumulators
                    accum = T.alloc_buffer([128], "float32", scope="local")
                    # epilogue
                    col_swizzle = int_var()
                    smem_offset = int_var()

                    ############################################################################## INITIALIZATION
                    # threading
                    wg_id[0] = tid // WG_SIZE
                    tid_in_wg[0] = tid % WG_SIZE
                    warp_id_in_wg[0] = tid_in_wg[0] // 32
                    # initialize work tile info
                    tile_scheduler.init(bx + by * CLUSTER_M)
                    # initialize producer pipeline states
                    producer.init(0, 1)
                    # initialize consumer pipeline states
                    consumer_read.init(0, 0)
                    # initialize consumer signals, each consumer WG signals 2 CTAs
                    is_signal_thread[0] = T.Select(tid_in_wg[0] % 8 == 0, 1, 0)
                    dst_block_ID[0] = (warp_id_in_wg[0] % 4) * 4 + (tid_in_wg[0] // 8) % 4
                    is_signal_thread[0] = is_signal_thread[0] & (dst_block_ID[0] < 2)
                    is_signal_thread[0] = is_signal_thread[0] & (dst_block_ID[0] % 2 == cbx or dst_block_ID[0] // 2 == cby)
                    # initialize mainloop pipeline barriers per CTA
                    if (tid // 32 == 0 and T.elect_sync(0xFFFFFFFF) > 0):
                        for i in range(STAGES_MMA):
                            T.mbarrier_init(full.access_ptr("rw", offset=i), 1) # 1 producer per CTA
                            T.mbarrier_init(empty.access_ptr("rw", offset=i), 4) # 2 CTA, 2 consumers per CTA
                    # fence the barrier init, the memory ordering is visble across the cluster
                    T.fence_mbarrier_init_release_cluster()
                    # cluster synchronization
                    T.barrier_cluster_arrive("relaxed", aligned=True)
                    T.barrier_cluster_wait(aligned=True)

                    k_tile_count = T.meta_var((K + BLK_K - 1) // BLK_K)
                    if (wg_id[0] == WarpGroupRole.PRODUCER):
                        ############################################################################## PRODUCER
                        # producer WG
                        if (warp_id_in_wg[0] == 0):
                            stage = T.meta_var(producer.index[0])
                            cur_full = T.meta_var(full.access_ptr("rw", offset=stage))
                            cur_empty = T.meta_var(empty.access_ptr("rw", offset=stage))
                            # mainloop producer warp
                            while (tile_scheduler.valid()):
                                if (T.elect_sync(0xFFFFFFFF)):
                                    # only the leader thread does the TMA load
                                    for i in range(k_tile_count):
                                        # producer acquire the slot
                                        T.mbarrier_wait(cur_empty, producer.phase[0])
                                        T.mbarrier_arrive_expect_tx(cur_full, TMA_BYTES)
                                        # issue TMA loads for A
                                        T.cp_async_bulk_tensor_global_to_cluster(
                                            2, A_smem.access_ptr("w", offset=A_smem.offset_of_p([stage, 0])),
                                            cur_full, A_map, i * BLK_K, tile_scheduler.m_idx[0] * BLK_M
                                        )
                                        # issue TMA loads for B
                                        for n_tile in range(4):
                                            multicast_stride_b = T.meta_var(BLK_K // CLUSTER_M)
                                            T.cp_async_bulk_tensor_global_to_cluster(
                                                2, B_smem.access_ptr("w", offset=B_smem.offset_of_p([stage, cbx * multicast_stride_b * 64 + n_tile * BLK_K * 64])),
                                                cur_full, B_map, tile_scheduler.n_idx[0] * BLK_N + n_tile * 64, i * BLK_K + cbx * multicast_stride_b,
                                                cta_mask=0x3
                                            )
                                        # move to the next stage
                                        producer.advance()
                                # move to the next tile
                                tile_scheduler.next_tile()
                            # producer needs to wait for consumers to finish to prevent early exit
                            # early exit can cause conusmers signaling CTAs that are already finished
                            if (T.elect_sync(0xFFFFFFFF)):
                                for _ in range(STAGES_MMA):
                                    # producer acquire the slot
                                    T.mbarrier_wait(cur_empty, producer.phase[0])
                                    # move to the next stage
                                    producer.advance()
                    elif (wg_id[0] == WarpGroupRole.CONSUMER0 or wg_id[0] == WarpGroupRole.CONSUMER1):
                        ############################################################################## CONSUMER
                        # consumer WG
                        while (tile_scheduler.valid()):
                            ####################################### WGMMA
                            # initialize the accumulators
                            for i in range(128):
                                accum[i] = 0
                                T.wgmma_fence_operand(accum[i])
                            read_stage = T.meta_var(consumer_read.index[0])
                            release_stage = T.meta_var(consumer_release.index[0])
                            full_read = T.meta_var(full.access_ptr("rw", offset=read_stage))
                            empty_release = T.meta_var(empty.access_ptr("rw", offset=release_stage))
                            for k_iter in range(k_tile_count):
                                # consumer acquire the slot
                                T.mbarrier_wait(full_read, consumer_read.phase[0])
                                # issue WGMMA for the current stage
                                wgmma_fence_operand(accum)
                                T.wgmma_arrive()
                                for inner_k in range(BLK_K // WGMMA_K):
                                    A_offset = T.meta_var((wg_id[0] - 1) * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
                                    B_offset = T.meta_var(inner_k * WGMMA_K * 64)
                                    T.encode_matrix_descriptor(desc_A.data, A_smem.access_ptr("r", offset=A_smem.offset_of_p([read_stage, A_offset])), 1, 64, swizzle=3)
                                    T.encode_matrix_descriptor(desc_B.data, B_smem.access_ptr("r", offset=B_smem.offset_of_p([read_stage, B_offset])), 512, 64, swizzle=3)
                                    T.wgmma_mma_async_ss(WGMMA_M, WGMMA_N, WGMMA_K, "float16", "float32", False, True, 1.0, 1.0, True,
                                                        desc_A[0], desc_B[0], *get_accum_list(accum, 128))
                                T.wgmma_commit_group()
                                if k_iter > 0:
                                    # wait for the previous stage to finish
                                    T.wgmma_wait_group(n = 1)
                                    wgmma_fence_operand(accum)
                                    # release the previous stage, send the signal to the producer
                                    T.mbarrier_arrive(empty_release, dst_block_ID[0], is_signal_thread[0])
                                # move to the next stage
                                consumer_release.copy(consumer_read)
                                consumer_read.advance()
                            wgmma_fence_operand(accum)
                            # wait for the last stage to finish
                            T.wgmma_wait_group(0)
                            wgmma_fence_operand(accum)
                            T.mbarrier_arrive(empty_release, dst_block_ID[0], is_signal_thread[0])
                            # ####################################### Epilogue
                            m_glb_offset = T.meta_var(tile_scheduler.m_idx[0] * BLK_M)
                            n_glb_offset = T.meta_var(tile_scheduler.n_idx[0] * BLK_N)
                            T.named_barrier_sync(0, 256)
                            epi_tile_count = T.meta_var(BLK_N // 32)
                            for i in T.serial(epi_tile_count):
                                if (i != 0):
                                    # s2G for the previous stage
                                    tma_store(i - 1, C_smem, C_map, m_glb_offset, n_glb_offset, tid)
                                quad_id = T.meta_var(lane_id // 4)
                                quad_lane = T.meta_var(lane_id % 4)
                                smem_offset[0] = ((wg_id[0] - 1) * BLK_M // 2 + 16 * warp_id_in_wg[0] + quad_id) * 32
                                r2S_stage = T.meta_var(i % STAGES_EPI)
                                T.named_barrier_sync(0, 256)
                                for reg in T.serial(4):
                                    col_id = T.meta_var(quad_lane // 2 + reg * 2)
                                    col_swizzle[0] = (quad_id ^ col_id) * 4 + quad_lane % 2 * 2
                                    C_smem[r2S_stage, smem_offset[0] + col_swizzle[0]] = accum[16 * i + reg * 4]
                                    C_smem[r2S_stage, smem_offset[0] + col_swizzle[0] + 1] = accum[16 * i + reg * 4 + 1]
                                    C_smem[r2S_stage, smem_offset[0] + 8*32 + col_swizzle[0]] = accum[16 * i + reg * 4 + 2]
                                    C_smem[r2S_stage, smem_offset[0] + 8*32 + col_swizzle[0] + 1] = accum[16 * i + reg * 4 + 3]
                            tma_store(epi_tile_count - 1, C_smem, C_map, m_glb_offset, n_glb_offset, tid)
                            T.cp_async_bulk_tensor_wait_group(n = 0, read=False)

                            # move to the next tile
                            tile_scheduler.next_tile()
    # fmt: on

    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(K, N).astype(np.float16)
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float32)
        C_tvm = tvm.nd.array(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = LowerTIRp()(mod)
            mod = tvm.build(mod, target=target)
            mod(A_tvm, B_tvm, C_tvm)
            timer = mod.time_evaluator(mod.entry_name, DEV, number=100, repeat=5)
            res = timer(A_tvm, B_tvm, C_tvm)
            print("tvm tir time: ")
            print(res)

        return C_tvm

    def cublas_gemm():
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((K, N), name="B", dtype="float16")
        C = cublas.matmul(A, B, transb=False, dtype="float32")
        s = te.create_schedule(C.op)

        C_np = np.zeros((M, N), dtype=np.float32)
        C_tvm = tvm.nd.array(C_np, device=DEV)
        mod_cublaslt = tvm.build(s, [A, B, C], target)
        mod_cublaslt(A_tvm, B_tvm, C_tvm)

        timer = mod_cublaslt.time_evaluator(mod_cublaslt.entry_name, DEV, number=100, repeat=5)
        res = timer(A_tvm, B_tvm, C_tvm)
        print("cublas time: ")
        print(res)

        return C_tvm

    C_tvm = tir_gemm()
    C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm.asnumpy(), C_cublas.asnumpy(), rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(9)
def test_hgemm_hopper_no_ws():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")

    f16_bytes = 2
    SM_COUNT = DEV.multi_processor_count
    M, N, K = 8192, 8192, 8192
    BLK_M, BLK_N, BLK_K = 128, 256, 64
    WGMMA_M, WGMMA_N, WGMMA_K = 64, 256, 16

    swizzleA = swizzleB = 3  # 128B swizzle
    swizzleC = 3  # 128B swizzle
    GROUP_SIZE = 16
    STAGES_TMA = 4
    STAGES_WGMMA = 2
    STAGES_EPI = 2
    WG_SIZE = 128
    TMA_BYTES = BLK_M * BLK_K * f16_bytes + BLK_K * BLK_N * f16_bytes

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
        if tid == 0:
            stage = T.meta_var(k_tile % STAGES_TMA)
            T.mbarrier_arrive_expect_tx(bars.access_ptr("rw", offset=stage), TMA_BYTES)
            T.cp_async_bulk_tensor_global_to_cluster(
                2, A_smem.access_ptr("w", offset=A_smem.offset_of_p([stage, 0])),
                bars.access_ptr("rw", offset=stage), A_map, k_tile * BLK_K, m_idx * BLK_M,
            )
            T.cp_async_bulk_tensor_global_to_cluster(
                2, B_smem.access_ptr("w", offset=B_smem.offset_of_p([stage, 0])),
                bars.access_ptr("rw", offset=stage), B_map, k_tile * BLK_K, n_idx * BLK_N,
            )

    def get_accum_list(C, C_elems):
        return [C[i] for i in range(C_elems)]

    @T.macro
    def mma_compute(wg_id, k_tile, A_smem: tvm.tir.Buffer, B_smem: tvm.tir.Buffer, accum, bars: tvm.tir.Buffer, descA, descB):
        stage = T.meta_var(k_tile % STAGES_TMA)
        parity = T.meta_var((k_tile // STAGES_TMA) % 2)
        T.mbarrier_wait(bars.access_ptr("rw", offset=stage), parity)
        for inner_k in T.serial(BLK_K // WGMMA_K):
            A_offset = T.meta_var(wg_id * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
            B_offset = T.meta_var(inner_k * WGMMA_K)
            T.encode_matrix_descriptor(descA.data, A_smem.access_ptr("r", offset=A_smem.offset_of_p([stage, A_offset])), 1, 64, swizzle=3)
            T.encode_matrix_descriptor(descB.data, B_smem.access_ptr("r", offset=B_smem.offset_of_p([stage, B_offset])), 1, 64, swizzle=3)
            T.wgmma_mma_async_ss(WGMMA_M, WGMMA_N, WGMMA_K, "float16", "float32", False, False, 1.0, 1.0, True,
                                descA[0], descB[0], *get_accum_list(accum, 128))
        T.wgmma_commit_group()

    @T.macro
    def r2S(warp_id, lane_id, C_smem: tvm.tir.Buffer, accum, accum_half, n_tile):
        T.tvm_storage_sync("shared")
        for st_tile in T.serial(4):
            for i in T.serial(8):
                accum_half[i] = accum[n_tile * 32 + st_tile * 8 + i]
            col_noswizzle = T.meta_var(st_tile * 2 + lane_id // 16)
            col = T.meta_var((lane_id % 8) ^ col_noswizzle)
            row = T.meta_var(warp_id * 16 + lane_id % 16)
            T.stmatrix_sync_aligned(4, False, C_smem.access_ptr("w", offset=C_smem.offset_of_p([n_tile % STAGES_EPI, row, col * 8])),
                                    accum_half[0], accum_half[1], accum_half[2], accum_half[3],
                                    accum_half[4], accum_half[5], accum_half[6], accum_half[7])

    @T.macro
    def s2G(warp_id, lane_id, C_smem: tvm.tir.Buffer, C_map, m_idx, n_idx, n_tile):
        T.cuda_fence_proxy_async("shared")
        T.tvm_storage_sync("shared")
        if warp_id == 0 and lane_id == 0:
            T.cp_async_bulk_tensor_shared_to_global(2, C_smem.access_ptr("r", offset=C_smem.offset_of_p([n_tile % STAGES_EPI, 0, 0])),
                                                    C_map, n_idx * BLK_N + n_tile * 64, m_idx * BLK_M)
            T.cp_async_bulk_tensor_commit_group()
            T.cp_async_bulk_tensor_wait_group(1, read=True)

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
        T.cp_async_bulk_tensor_wait_group(0, read=False)

    @T.prim_func(tirp=True)
    def manual(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), "float16", scope="global", layout=T.TileLayout.from_tuple((M, K)))
        B = T.match_buffer(B_ptr, (N, K), "float16", scope="global", layout=T.TileLayout.from_tuple((N, K)))
        C = T.match_buffer(C_ptr, (M, N), "float16", scope="global", layout=T.TileLayout.from_tuple((M, N)))

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        C_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        T.call_packed("runtime.cuTensorMapInit", A_map, "float16", 2, A.data, K, M, f16_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)
        T.call_packed("runtime.cuTensorMapInit", B_map, "float16", 2, B.data, K, N, f16_bytes * K, BLK_K, BLK_N, 1, 1, 0, swizzleB, 0, 0)
        T.call_packed("runtime.cuTensorMapInit", C_map, "float16", 2, C.data, N, M, f16_bytes * N, 64, BLK_M, 1, 1, 0, swizzleC, 0, 0)

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            tid = T.thread_id([128 * 2], parent="cta")
            warp_id = T.warp_id([8], parent="cta")
            lane_id = T.thread_id([32], parent="warp")

            with T.cta():
                # tensor stroage
                A_smem = T.alloc_buffer([STAGES_TMA, BLK_M * BLK_K], "float16", scope="shared.dyn", align=128)
                B_smem = T.alloc_buffer([STAGES_TMA, BLK_K * BLK_N], "float16", scope="shared.dyn", align=128)
                C_smem = T.alloc_buffer([STAGES_EPI, BLK_M, 64], "float16", scope="shared.dyn", align=1024)
                # barriers
                bars = T.alloc_buffer([STAGES_TMA], "uint64", scope="shared.dyn", align=8)
                desc_A = T.alloc_buffer([1], "uint64", scope="local", align=8)
                desc_B = T.alloc_buffer([1], "uint64", scope="local", align=8)

                with T.thread():
                    # index
                    wg_id = int_var()
                    tma_index = int_var()
                    mma_index = int_var()
                    # acuumulators
                    accum = T.alloc_buffer([128], "float32", scope="local")
                    # temp accum
                    accum_half = T.alloc_buffer([8], "float16", scope="local")
                    # tile scheduler
                    tile_scheduler = T.meta_var(TileScheduler("tile_scheduler"))

                    wg_id[0] = tid // WG_SIZE
                    # initialize the tile scheduler
                    tile_scheduler.init(bx)

                    while (tile_scheduler.valid()):
                        # initialize the barriers
                        if (tid == 0):
                            for i in range(STAGES_TMA):
                                T.mbarrier_init(bars.access_ptr("rw", offset=i), 1)
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
                            mma_compute(wg_id[0], mma_index[0], A_smem, B_smem, accum, bars, desc_A, desc_B)
                            mma_index[0] = mma_index[0] + 1

                        # mainloop
                        k_tile_count = T.meta_var((K + BLK_K - 1) // BLK_K)
                        for _ in range(k_tile_count):
                            if mma_index[0] < k_tile_count:
                                mma_compute(wg_id[0], mma_index[0], A_smem, B_smem, accum, bars, desc_A, desc_B)
                                mma_index[0] = mma_index[0] + 1
                            # wait for oldest one stage to finish
                            if _ == k_tile_count - 1:
                                T.wgmma_wait_group(0)
                            else:
                                T.wgmma_wait_group(STAGES_WGMMA - 1)
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

    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(N, K).astype(np.float16)
    A_tvm = tvm.nd.array(A_np, device=DEV)
    B_tvm = tvm.nd.array(B_np, device=DEV)

    def flops(ms):
        return M * N * K * 2 / ms / 1e9

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float16)
        C_tvm = tvm.nd.array(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = LowerTIRp()(mod)
            mod = tvm.build(mod, target=target)
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        return C_tvm.asnumpy()

    def cublas_gemm():
        import torch

        torch_dev = torch.device("cuda")
        A_torch = torch.tensor(A_np, device=torch_dev)
        B_torch = torch.tensor(B_np, device=torch_dev)
        C_torch = torch.zeros((M, N), device=torch_dev)
        func = lambda: torch.matmul(A_torch, B_torch.T)
        ms = bench(func, warmup=0, repeat=10, proton_name="cublas")
        print(f"CUBLAS flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    with ProtonContext("matmul"):
        C_tvm = tir_gemm()
        C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm, C_cublas, rtol=2e-2, atol=1e-4)


if __name__ == "__main__":
    test_hgemm_hopper_ws_cooperative()
    test_hgemm_hopper_no_ws()
