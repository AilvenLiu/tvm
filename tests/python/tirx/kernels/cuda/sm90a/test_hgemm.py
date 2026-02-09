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
from tvm.script import tirx as Tx
import tvm.testing
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.bench.utils import ProtonContext, bench
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D


@tvm.testing.requires_cuda_compute_version(9, exact=True)
def test_hgemm_hopper_ws_cooperative():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

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

    class PipelineState:
        def __init__(self, prefix: str):
            self.index = Tx.local_cell("int32", name=prefix + "_index")
            self.phase = Tx.local_cell("int32", name=prefix + "_phase")

        @Tx.macro
        def init(self, index, phase):
            self.index = index
            self.phase = phase

        @Tx.macro
        def copy(self, other):
            self.index = other.index
            self.phase = other.phase

        @Tx.macro
        def advance(self):
            self.index = self.index + 1
            if self.index == STAGES_MMA:
                self.index = 0
                self.phase = self.phase ^ 1

    class WarpGroupRole:
        PRODUCER = 0
        CONSUMER0 = 1
        CONSUMER1 = 2

    @Tx.macro
    def ptx_wgmma_noop_barrier(accum):
        for i in Tx.serial(128):
            Tx.ptx.wgmma.noop_barrier(accum[i])

    @Tx.macro
    def tma_store(n_tile, C_smem: tvm.tir.Buffer, C_map, m_glb_offset, n_glb_offset, tid):
        # make sure smem write is visible to TMA
        Tx.ptx.fence.proxy("shared")
        Tx.ptx.bar.sync(0, 256)
        with Tx.thread()[tid == 128]:
            # only 1 thread in 2 consumers write to TMA
            Tx.ptx.cp_async.bulk.tensor.s2g(
                2,
                C_smem.ptr_to([n_tile % STAGES_EPI, 0]),
                C_map,
                n_glb_offset + n_tile * 32,
                m_glb_offset,
            )
            Tx.ptx.cp_async.bulk.commit_group()
            Tx.ptx.cp_async.bulk.wait_group(n=1, read=False)

    # fmt: off
    @Tx.prim_func(tirx=True)
    def manual(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (M, K), "float16", scope="global", layout=Tx.TileLayout((M, K)))
        B = Tx.match_buffer(B_ptr, (K, N), "float16", scope="global", layout=Tx.TileLayout((K, N)))
        C = Tx.match_buffer(C_ptr, (M, N), "float32", scope="global", layout=Tx.TileLayout((M, N)))


        A_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        B_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        C_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)

        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float16", 2, A.data, K, M, f16_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float16", 2, B.data, N, K, f16_bytes * N, 64, BLK_K // 2, 1, 1, 0, swizzleB, 0, 0) # multicast
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", C_map, "float32", 2, C.data, N, M, f32_bytes * N, 32, BLK_M, 1, 1, 0, swizzleC, 0, 0)

        with Tx.kernel():
            cbx, cby = Tx.cta_id([CLUSTER_M, CLUSTER_N], parent="cluster")
            bx, by = Tx.cta_id([CLUSTER_M, SM_COUNT // CLUSTER_M], parent="kernel")
            wg_id = Tx.warpgroup_id([3], parent="cta")
            tid = Tx.thread_id([128 * 3], parent="cta")
            warp_id_in_wg = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")

            with Tx.cta():
                # tensor stroage
                A_smem = Tx.alloc_buffer([STAGES_MMA, BLK_M * BLK_K], "float16", scope="shared.dyn", align=128)
                B_smem = Tx.alloc_buffer([STAGES_MMA, BLK_K * BLK_N], "float16", scope="shared.dyn", align=128)
                C_smem = Tx.alloc_buffer([STAGES_EPI, BLK_M * 32], "float32", scope="shared.dyn", align=1024)
                # mainloop pipeline barriers, note that these are cluster-wide barriers
                full = Tx.alloc_buffer([STAGES_MMA], "uint64", scope="shared.dyn", layout=Tx.TileLayout([STAGES_MMA]), align=8)
                empty = Tx.alloc_buffer([STAGES_MMA], "uint64", scope="shared.dyn", layout=Tx.TileLayout([STAGES_MMA]), align=8)

                with Tx.thread():
                    # work tile info
                    m_blocks = ceildiv(ceildiv(M, BLK_M), CLUSTER_M) * CLUSTER_M
                    n_blocks = ceildiv(ceildiv(N, BLK_N), CLUSTER_N) * CLUSTER_N
                    tile_scheduler = Tx.meta_var(
                        ClusterPersistentScheduler2D(
                            "tile_scheduler",
                            num_m_tiles=m_blocks,
                            num_n_tiles=n_blocks,
                            num_clusters=SM_COUNT,
                            l2_group_size=GROUP_SIZE,
                            cluster_m=CLUSTER_M,
                            cluster_n=CLUSTER_N,
                        )
                    )
                    # produer pipelinen states
                    producer = Tx.meta_var(PipelineState("producer"))
                    # consumer pipeline states
                    consumer_read = Tx.meta_var(PipelineState("consumer_read"))
                    consumer_release = Tx.meta_var(PipelineState("consumer_release"))
                    # consumer signals
                    is_signal_thread = Tx.local_cell("int32")
                    dst_block_ID = Tx.local_cell("int32")
                    # smem desc_A, desc_B
                    desc_A = Tx.local_cell("uint64")
                    desc_B = Tx.local_cell("uint64")
                    # accumulators
                    accum = Tx.alloc_buffer([128], "float32", scope="local")
                    # epilogue
                    col_swizzle = Tx.local_cell("int32")
                    smem_offset = Tx.local_cell("int32")

                    ############################################################################## INITIALIZATION
                    # initialize work tile info
                    tile_scheduler.init(bx + by * CLUSTER_M)
                    # initialize producer pipeline states
                    producer.init(0, 1)
                    # initialize consumer pipeline states
                    consumer_read.init(0, 0)
                    # initialize consumer signals, each consumer WG signals 2 CTAs
                    is_signal_thread = Tx.Select(tid_in_wg % 8 == 0, 1, 0)
                    dst_block_ID = (warp_id_in_wg % 4) * 4 + (tid_in_wg // 8) % 4
                    is_signal_thread = is_signal_thread & (dst_block_ID < 2)
                    is_signal_thread = is_signal_thread & (dst_block_ID % 2 == cbx or dst_block_ID // 2 == cby)
                    # initialize mainloop pipeline barriers per CTA
                    if (tid // 32 == 0 and Tx.ptx.elect_sync() > 0):
                        for i in range(STAGES_MMA):
                            Tx.ptx.mbarrier.init(full.ptr_to([i]), 1) # 1 producer per CTA
                            Tx.ptx.mbarrier.init(empty.ptr_to([i]), 4) # 2 CTA, 2 consumers per CTA
                    # fence the barrier init, the memory ordering is visble across the cluster
                    Tx.ptx.fence.mbarrier_init()
                    # cluster synchronization
                    Tx.cuda.cluster_sync()

                    k_tile_count = Tx.meta_var((K + BLK_K - 1) // BLK_K)
                    with Tx.warpgroup()[0:1]:
                        ############################################################################## PRODUCER
                        # producer WG
                        with Tx.warp(parent="warpgroup")[0:1]:
                            stage = Tx.meta_var(producer.index)
                            cur_full = Tx.meta_var(full.ptr_to([stage]))
                            cur_empty = Tx.meta_var(empty.ptr_to([stage]))
                            # mainloop producer warp
                            while (tile_scheduler.valid()):
                                with Tx.thread()[Tx.ptx.elect_sync()]:
                                    # only the leader thread does the TMA load
                                    for i in range(k_tile_count):
                                        # producer acquire the slot
                                        Tx.ptx.mbarrier.try_wait(cur_empty, producer.phase)
                                        Tx.ptx.mbarrier.arrive.expect_tx(cur_full, TMA_BYTES)
                                        # issue TMA loads for A
                                        Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([stage, 0]),cur_full, A_map, i * BLK_K, tile_scheduler.m_idx * BLK_M)
                                        # issue TMA loads for B
                                        for n_tile in range(4):
                                            multicast_stride_b = Tx.meta_var(BLK_K // CLUSTER_M)
                                            Tx.ptx.cp_async.bulk.tensor.g2c(
                                                2, B_smem.ptr_to([stage, cbx * multicast_stride_b * 64 + n_tile * BLK_K * 64]),
                                                cur_full, B_map, tile_scheduler.n_idx * BLK_N + n_tile * 64, i * BLK_K + cbx * multicast_stride_b,
                                                cta_mask=0x3
                                            )
                                        # move to the next stage
                                        producer.advance()
                                # move to the next tile
                                tile_scheduler.next_tile()
                            # producer needs to wait for consumers to finish to prevent early exit
                            # early exit can cause conusmers signaling CTAs that are already finished
                            with Tx.thread()[Tx.ptx.elect_sync()]:
                                for _ in range(STAGES_MMA):
                                    # producer acquire the slot
                                    Tx.ptx.mbarrier.try_wait(cur_empty, producer.phase)
                                    # move to the next stage
                                    producer.advance()
                    with Tx.warpgroup()[1:3]:
                        ############################################################################## CONSUMER
                        # consumer WG
                        while (tile_scheduler.valid()):
                            ####################################### WGMMA
                            # initialize the accumulators
                            for i in range(128):
                                accum[i] = 0
                                Tx.ptx.wgmma.noop_barrier(accum[i])
                            read_stage = Tx.meta_var(consumer_read.index)
                            release_stage = Tx.meta_var(consumer_release.index)
                            full_read = Tx.meta_var(full.ptr_to([read_stage]))
                            empty_release = Tx.meta_var(empty.ptr_to([release_stage]))
                            for k_iter in range(k_tile_count):
                                # consumer acquire the slot
                                Tx.ptx.mbarrier.try_wait(full_read, consumer_read.phase)
                                # issue WGMMA for the current stage
                                ptx_wgmma_noop_barrier(accum)
                                Tx.ptx.wgmma.fence()
                                for inner_k in range(BLK_K // WGMMA_K):
                                    A_offset = Tx.meta_var((wg_id - 1) * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
                                    B_offset = Tx.meta_var(inner_k * WGMMA_K * 64)
                                    Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(desc_A), A_smem.ptr_to([read_stage, A_offset]), 1, 64, swizzle=3)
                                    Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(desc_B), B_smem.ptr_to([read_stage, B_offset]), 512, 64, swizzle=3)
                                    Tx.ptx.wgmma.mma_async.ss(WGMMA_M, WGMMA_N, WGMMA_K, "float16", "float32", False, True, 1.0, 1.0, True,
                                                             desc_A, desc_B, *[accum[i] for i in range(128)])
                                Tx.ptx.wgmma.commit_group()
                                if k_iter > 0:
                                    # wait for the previous stage to finish
                                    Tx.ptx.wgmma.wait_group(n = 1)
                                    ptx_wgmma_noop_barrier(accum)
                                    # release the previous stage, send the signal to the producer
                                    Tx.ptx.mbarrier.arrive(empty_release, dst_block_ID, is_signal_thread)
                                # move to the next stage
                                consumer_release.copy(consumer_read)
                                consumer_read.advance()
                            ptx_wgmma_noop_barrier(accum)
                            # wait for the last stage to finish
                            Tx.ptx.wgmma.wait_group(0)
                            ptx_wgmma_noop_barrier(accum)
                            Tx.ptx.mbarrier.arrive(empty_release, dst_block_ID, is_signal_thread)
                            # ####################################### Epilogue
                            m_glb_offset = Tx.meta_var(tile_scheduler.m_idx * BLK_M)
                            n_glb_offset = Tx.meta_var(tile_scheduler.n_idx * BLK_N)
                            Tx.ptx.bar.sync(0, 256)
                            epi_tile_count = Tx.meta_var(BLK_N // 32)
                            for i in Tx.serial(epi_tile_count):
                                if (i != 0):
                                    # s2G for the previous stage
                                    tma_store(i - 1, C_smem, C_map, m_glb_offset, n_glb_offset, tid)
                                quad_id = Tx.meta_var(lane_id // 4)
                                quad_lane = Tx.meta_var(lane_id % 4)
                                smem_offset = ((wg_id - 1) * BLK_M // 2 + 16 * warp_id_in_wg + quad_id) * 32
                                r2S_stage = Tx.meta_var(i % STAGES_EPI)
                                Tx.ptx.bar.sync(0, 256)
                                for reg in Tx.serial(4):
                                    col_id = Tx.meta_var(quad_lane // 2 + reg * 2)
                                    col_swizzle = (quad_id ^ col_id) * 4 + quad_lane % 2 * 2
                                    C_smem[r2S_stage, smem_offset + col_swizzle] = accum[16 * i + reg * 4]
                                    C_smem[r2S_stage, smem_offset + col_swizzle + 1] = accum[16 * i + reg * 4 + 1]
                                    C_smem[r2S_stage, smem_offset + 8*32 + col_swizzle] = accum[16 * i + reg * 4 + 2]
                                    C_smem[r2S_stage, smem_offset + 8*32 + col_swizzle + 1] = accum[16 * i + reg * 4 + 3]
                            tma_store(epi_tile_count - 1, C_smem, C_map, m_glb_offset, n_glb_offset, tid)
                            Tx.ptx.cp_async.bulk.wait_group(n = 0, read=False)

                            # move to the next tile
                            tile_scheduler.next_tile()
    # fmt: on

    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(K, N).astype(np.float16)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)

    def flops(ms):
        return M * N * K * 2 / ms / 1e9

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float32)
        C_tvm = tvm.runtime.tensor(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        return C_tvm.numpy()

    def cublas_gemm():
        import torch

        torch_dev = torch.device("cuda")
        A_torch = torch.tensor(A_np, device=torch_dev)
        B_torch = torch.tensor(B_np, device=torch_dev)
        C_torch = torch.zeros((M, N), device=torch_dev)
        func = lambda: torch.matmul(A_torch, B_torch)
        ms = bench(func, warmup=0, repeat=10, proton_name="cublas")
        print(f"CUBLAS flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    with ProtonContext("hopper_gemm_ws"):
        C_tvm = tir_gemm()
        C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm, C_cublas, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(9, exact=True)
def test_hgemm_hopper_no_ws():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

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

    # fmt: off
    @Tx.macro
    def tma_load(tid, m_idx, n_idx, k_tile, A_smem: tvm.tir.Buffer, B_smem: tvm.tir.Buffer, A_map, B_map, bars: tvm.tir.Buffer):
        with Tx.thread()[tid == 0]:
            stage = Tx.meta_var(k_tile % STAGES_TMA)
            Tx.ptx.mbarrier.arrive.expect_tx(bars.ptr_to([stage]), TMA_BYTES)
            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([stage, 0]), bars.ptr_to([stage]), A_map, k_tile * BLK_K, m_idx * BLK_M)
            Tx.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([stage, 0]), bars.ptr_to([stage]), B_map, k_tile * BLK_K, n_idx * BLK_N)

    def get_accum_list(C, C_elems):
        return [C[i] for i in range(C_elems)]

    @Tx.macro
    def mma_compute(wg_id, k_tile, A_smem: tvm.tir.Buffer, B_smem: tvm.tir.Buffer, accum, bars: tvm.tir.Buffer, descA, descB):
        stage = Tx.meta_var(k_tile % STAGES_TMA)
        parity = Tx.meta_var((k_tile // STAGES_TMA) % 2)
        Tx.ptx.mbarrier.try_wait(bars.ptr_to([stage]), parity)
        for inner_k in Tx.serial(BLK_K // WGMMA_K):
            A_offset = Tx.meta_var(wg_id * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
            B_offset = Tx.meta_var(inner_k * WGMMA_K)
            Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descA), A_smem.ptr_to([stage, A_offset]), 1, 64, swizzle=3)
            Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descB), B_smem.ptr_to([stage, B_offset]), 1, 64, swizzle=3)
            Tx.ptx.wgmma.mma_async.ss(WGMMA_M, WGMMA_N, WGMMA_K, "float16", "float32", False, False, 1.0, 1.0, True,
                                     descA, descB, *get_accum_list(accum, 128))
        Tx.ptx.wgmma.commit_group()

    @Tx.macro
    def r2S(warp_id, lane_id, C_smem: tvm.tir.Buffer, accum, accum_half, n_tile):
        Tx.cuda.cta_sync()
        for st_tile in Tx.serial(4):
            for i in Tx.serial(8):
                accum_half[i] = accum[n_tile * 32 + st_tile * 8 + i]
            col_noswizzle = Tx.meta_var(st_tile * 2 + lane_id // 16)
            col = Tx.meta_var((lane_id % 8) ^ col_noswizzle)
            row = Tx.meta_var(warp_id * 16 + lane_id % 16)
            Tx.ptx.stmatrix(4, False, C_smem.ptr_to([n_tile % STAGES_EPI, row, col * 8]), accum_half.ptr_to([0]))

    @Tx.macro
    def s2G(warp_id, lane_id, C_smem: tvm.tir.Buffer, C_map, m_idx, n_idx, n_tile):
        Tx.ptx.fence.proxy("shared")
        Tx.cuda.cta_sync()
        with Tx.thread()[warp_id == 0 and lane_id == 0]:
            Tx.ptx.cp_async.bulk.tensor.s2g(2, C_smem.ptr_to([n_tile % STAGES_EPI, 0, 0]), C_map, n_idx * BLK_N + n_tile * 64, m_idx * BLK_M)
            Tx.ptx.cp_async.bulk.commit_group()
            Tx.ptx.cp_async.bulk.wait_group(1, read=True)

    @Tx.macro
    def write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem: tvm.tir.Buffer, C_map, accum, accum_half):
        Tx.cuda.cta_sync()
        for n_tile in Tx.serial(BLK_N // 64):
            if n_tile != 0:
                # s2G for the previous stage
                s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, n_tile - 1)
            # r2S for the current stage
            r2S(warp_id, lane_id, C_smem, accum, accum_half, n_tile)
        # s2G for the last stage
        s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, BLK_N // 64 - 1)
        Tx.ptx.cp_async.bulk.wait_group(0, read=False)

    @Tx.prim_func(tirx=True)
    def manual(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (M, K), "float16", scope="global", layout=Tx.TileLayout([M, K]))
        B = Tx.match_buffer(B_ptr, (N, K), "float16", scope="global", layout=Tx.TileLayout([N, K]))
        C = Tx.match_buffer(C_ptr, (M, N), "float16", scope="global", layout=Tx.TileLayout([M, N]))

        A_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        B_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)
        C_map: Tx.handle("tensormap") = Tx.tvm_stack_alloca("tensormap", 1)

        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float16", 2, A.data, K, M, f16_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float16", 2, B.data, K, N, f16_bytes * K, BLK_K, BLK_N, 1, 1, 0, swizzleB, 0, 0)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", C_map, "float16", 2, C.data, N, M, f16_bytes * N, 64, BLK_M, 1, 1, 0, swizzleC, 0, 0)

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([2], parent="cta")
            tid = Tx.thread_id([128 * 2], parent="cta")
            warp_id = Tx.warp_id([8], parent="cta")
            lane_id = Tx.thread_id([32], parent="warp")

            with Tx.cta():
                # tensor stroage
                A_smem = Tx.alloc_buffer([STAGES_TMA, BLK_M * BLK_K], "float16", scope="shared.dyn", align=128)
                B_smem = Tx.alloc_buffer([STAGES_TMA, BLK_K * BLK_N], "float16", scope="shared.dyn", align=128)
                C_smem = Tx.alloc_buffer([STAGES_EPI, BLK_M, 64], "float16", scope="shared.dyn", align=1024)
                # barriers
                bars = Tx.alloc_buffer([STAGES_TMA], "uint64", scope="shared.dyn", align=8)
                desc_A = Tx.local_cell("uint64")
                desc_B = Tx.local_cell("uint64")

                with Tx.thread():
                    # index
                    tma_index = Tx.local_cell("int32")
                    mma_index = Tx.local_cell("int32")
                    # acuumulators
                    accum = Tx.alloc_buffer([128], "float32", scope="local")
                    # temp accum
                    accum_half = Tx.alloc_buffer([8], "float16", scope="local")
                    # tile scheduler
                    m_blocks = ceildiv(ceildiv(M, BLK_M), CLUSTER_M) * CLUSTER_M
                    n_blocks = ceildiv(ceildiv(N, BLK_N), CLUSTER_N) * CLUSTER_N
                    tile_scheduler = Tx.meta_var(
                        ClusterPersistentScheduler2D(
                            "tile_scheduler",
                            num_m_tiles=m_blocks,
                            num_n_tiles=n_blocks,
                            num_clusters=SM_COUNT,
                            l2_group_size=GROUP_SIZE,
                            cluster_m=CLUSTER_M,
                            cluster_n=CLUSTER_N,
                        )
                    )

                    # initialize the tile scheduler
                    tile_scheduler.init(bx)

                    while (tile_scheduler.valid()):
                        # initialize the barriers
                        with Tx.thread()[tid == 0]:
                            for i in range(STAGES_TMA):
                                Tx.ptx.mbarrier.init(bars.ptr_to([i]), 1)
                        Tx.cuda.cta_sync()
                        # initialize the index
                        tma_index = 0
                        mma_index = 0
                        # initialize the accumulators
                        for i in range(128):
                            accum[i] = 0
                        Tx.cuda.cta_sync()

                        m_idx = Tx.meta_var(tile_scheduler.m_idx)
                        n_idx = Tx.meta_var(tile_scheduler.n_idx)
                        # prelogue
                        for _ in range(STAGES_TMA):
                            tma_load(tid, m_idx, n_idx, tma_index, A_smem, B_smem, A_map, B_map, bars)
                            tma_index = tma_index + 1
                        for _ in range(STAGES_WGMMA - 1):
                            mma_compute(wg_id, mma_index, A_smem, B_smem, accum, bars, desc_A, desc_B)
                            mma_index = mma_index + 1

                        # mainloop
                        k_tile_count = Tx.meta_var((K + BLK_K - 1) // BLK_K)
                        for _ in range(k_tile_count):
                            if mma_index < k_tile_count:
                                mma_compute(wg_id, mma_index, A_smem, B_smem, accum, bars, desc_A, desc_B)
                                mma_index = mma_index + 1
                            # wait for oldest one stage to finish
                            if _ == k_tile_count - 1:
                                Tx.ptx.wgmma.wait_group(0)
                            else:
                                Tx.ptx.wgmma.wait_group(STAGES_WGMMA - 1)
                            Tx.cuda.cta_sync()
                            # load the next tile
                            if tma_index < k_tile_count:
                                tma_load(tid, m_idx, n_idx, tma_index, A_smem, B_smem, A_map, B_map, bars)
                                tma_index = tma_index + 1

                        # epilogue
                        write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem, C_map, accum, accum_half)

                        # move to the next tile
                        tile_scheduler.next_tile()
    # fmt: on

    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(N, K).astype(np.float16)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)

    def flops(ms):
        return M * N * K * 2 / ms / 1e9

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float16)
        C_tvm = tvm.runtime.tensor(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
            func = lambda: mod(A_tvm, B_tvm, C_tvm)
            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        return C_tvm.numpy()

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

    with ProtonContext("hopper_hgemm_no_ws"):
        C_tvm = tir_gemm()
        C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm, C_cublas, rtol=2e-2, atol=1e-4)


if __name__ == "__main__":
    test_hgemm_hopper_ws_cooperative()
    test_hgemm_hopper_no_ws()
