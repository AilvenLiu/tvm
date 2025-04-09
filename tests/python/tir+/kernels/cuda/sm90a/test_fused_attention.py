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
import math

import tvm
import tvm.testing
from tvm.script.ir_builder import IRBuilder
from tvm.script import tir as T
from ..utils import bench, ProtonContext, ProfileEventType, export_to_perfetto_trace


@tvm.testing.requires_cuda_compute_version(9)
def test_fp16_fused_attn():
    def ceildiv(a, b):
        return (a + b - 1) // b

    np.random.seed(0)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-h100")

    F16_BYTES = 2
    SM_COUNT = DEV.multi_processor_count
    # problem size
    DIM = 2048
    BATCH_SIZE = 2
    QO_LEN = 4096
    KV_LEN = 4096
    HEAD_DIM = 128
    NHEADS = DIM // HEAD_DIM
    QO_SHAPE = BATCH_SIZE, QO_LEN, NHEADS, HEAD_DIM
    KV_SHAPE = BATCH_SIZE, KV_LEN, NHEADS, HEAD_DIM
    # kernel config
    CAUSAL = True
    BLK_Q = 128
    if HEAD_DIM == 256:
        BLK_KV = 64
    elif HEAD_DIM == 128:
        BLK_KV = 128
    else:
        raise ValueError(f"HEAD_DIM {HEAD_DIM} not supported")

    INF = 5e4
    total_qtiles = BATCH_SIZE * NHEADS * ceildiv(QO_LEN, BLK_Q)
    KV_STAGES = 2
    STAGES_EPI = 2
    NUM_WARPS = 12
    WG_SIZE = 128

    SMEM_SIZE = 1024 + BLK_Q * HEAD_DIM * F16_BYTES + 2 * KV_STAGES * BLK_KV * HEAD_DIM * F16_BYTES
    TMA_TILE = 64
    TMA_BYTES_Q = BLK_Q * HEAD_DIM * F16_BYTES
    TMA_BYTES_K = BLK_KV * HEAD_DIM * F16_BYTES
    TMA_BYTES_V = BLK_KV * HEAD_DIM * F16_BYTES
    WGMMA_QK_M, WGMMA_QK_N, WGMMA_QK_K = 64, BLK_KV, 16
    WGMMA_PV_M, WGMMA_PV_N, WGMMA_PV_K = 64, HEAD_DIM, 16
    S_REG_COUNT = BLK_KV // 2
    O_REG_COUNT = HEAD_DIM // 2
    NUM_WGS = NUM_WARPS // 4
    CONSUMER_WGS = NUM_WGS - 1
    MMA_THREADS = CONSUMER_WGS * WG_SIZE
    assert WGMMA_QK_M * CONSUMER_WGS == BLK_Q
    assert BLK_KV % 16 == 0
    SWIZZLE = 3

    # profiling
    PROFILER_BUFFER_SIZE = 1048576
    PROFILER_WRITE_STRIDE = SM_COUNT * NUM_WARPS // 4

    def flops(ms):
        if CAUSAL:
            return BATCH_SIZE * QO_LEN * QO_LEN * NHEADS * HEAD_DIM * 2 / ms / 1e9
        else:
            return BATCH_SIZE * QO_LEN * QO_LEN * NHEADS * HEAD_DIM * 4 / ms / 1e9

    # fmt: off
    def int_var():
        return T.alloc_buffer([1], "int32", scope="local", align=4)

    class WarpGroupRole:
        PRODUCER = 0
        CONSUMER0 = 1
        CONSUMER1 = 2

    class NameBarrier:
        Q_EMPTY = 0
        V_LOAD_READY = 1
        O_LOAD_READY = 2
        EPILOGUE = 3

    class TileScheduler:
        def __init__(self, prefix: str, b_indices, h_indices, q_indices, tiles_indptr):
            self.linear_idx = int_var()
            self.linear_lim = int_var()
            self.b_indices = b_indices
            self.h_indices = h_indices
            self.q_indices = q_indices
            self.tiles_indptr = tiles_indptr
            self.q_idx = int_var()
            self.h_idx = int_var()
            self.b_idx = int_var()
            IRBuilder.current().name(prefix + "_linear_idx", self.linear_idx)
            IRBuilder.current().name(prefix + "_linear_lim", self.linear_lim)
            IRBuilder.current().name(prefix + "_b_indices", self.b_indices)
            IRBuilder.current().name(prefix + "_h_indices", self.h_indices)
            IRBuilder.current().name(prefix + "_q_indices", self.q_indices)
            IRBuilder.current().name(prefix + "_tiles_indptr", self.tiles_indptr)
            IRBuilder.current().name(prefix + "_q_idx", self.q_idx)
            IRBuilder.current().name(prefix + "_h_idx", self.h_idx)
            IRBuilder.current().name(prefix + "_b_idx", self.b_idx)

        @T.macro
        def get_block_coord(self):
            self.q_idx[0] = self.q_indices[self.linear_idx[0]]
            self.h_idx[0] = self.h_indices[self.linear_idx[0]]
            self.b_idx[0] = self.b_indices[self.linear_idx[0]]

        @T.macro
        def init(self, sm):
            self.linear_idx[0] = self.tiles_indptr[sm]
            self.linear_lim[0] = self.tiles_indptr[sm + 1]
            self.get_block_coord()

        @T.macro
        def next_tile(self):
            self.linear_idx[0] = self.linear_idx[0] + 1
            self.get_block_coord()
            
        def valid(self):
            return self.linear_idx[0] < self.linear_lim[0]

    class Pipeline:
        def __init__(self, full, empty, bytes):
            self.full = full
            self.empty = empty
            self.bytes = bytes

        @T.macro
        def init(self, tid):
            with T.thread()[tid == 0]:
                for i in T.serial(KV_STAGES):
                    T.ptx.mbarrier.init(self.full.access_ptr("rw", offset=i), 1)
                    T.ptx.mbarrier.init(self.empty.access_ptr("rw", offset=i), 2 * WG_SIZE)  # 2 consumers
            # fence the barrier init, the memory ordering is visible across the whole block
            T.ptx.fence.mbarrier_init()

        @T.macro
        def producer_acquire(self, state, profiler_buffer, profiler_tag, profiler_write_offset):
            stage = T.meta_var(state.index[0])
            cur_empty = T.meta_var(self.empty.access_ptr("rw", offset=stage))
            cur_full = T.meta_var(self.full.access_ptr("rw", offset=stage))
            T.ptx.mbarrier.try_wait(cur_empty, state.phase[0])
            T.timer_start_cuda(ProfileEventType.IssueLoadKV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
            T.ptx.mbarrier.arrive.expect_tx(cur_full, self.bytes)

        @T.macro
        def producer_tail(self, state):
            stage = T.meta_var(state.index[0])
            cur_empty = T.meta_var(self.empty.access_ptr("rw", offset=stage))
            for _ in T.serial(KV_STAGES):
                T.ptx.mbarrier.try_wait(cur_empty, state.phase[0])
                state.advance()

        @T.macro
        def consumer_wait(self, state):
            stage = T.meta_var(state.index[0])
            cur_full = T.meta_var(self.full.access_ptr("rw", offset=stage))
            T.ptx.mbarrier.try_wait(cur_full, state.phase[0])

        @T.macro
        def consumer_release(self, state):
            stage = T.meta_var(state.index[0])
            cur_empty = T.meta_var(self.empty.access_ptr("rw", offset=stage))
            T.ptx.mbarrier.arrive(cur_empty)

        @T.macro
        def copy(self, state, smem, tmap, *coord):
            # copy [HEAD_DIM, BLK_KV/BLK_Q] from global [HAED_DIM, hidx, BLK_KV/BLK_Q * idx, bidx] to smem
            # copy at most [TMA_TILE, BLK_KV/BLK_Q] elements per iteration due to the restriction of swizzle
            stage = T.meta_var(state.index[0])
            cur_full = T.meta_var(self.full.access_ptr("rw", offset=stage))
            for tma_tile in T.serial(HEAD_DIM // TMA_TILE):
                T.ptx.cp_async.bulk.tensor.g2c(
                    4, smem.access_ptr("w", offset=smem.offset_of_p([stage, tma_tile * TMA_TILE * BLK_KV])), cur_full,
                    tmap, coord[0] + tma_tile * TMA_TILE, coord[1], coord[2], coord[3],
                )

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
            if self.index[0] == KV_STAGES:
                self.index[0] = 0
                self.phase[0] = self.phase[0] ^ 1

    @T.macro
    def ptx_wgmma_noop_barrier(accum, accum_count):
        for i in T.serial(accum_count):
            T.ptx.wgmma.noop_barrier(accum[i])

    def get_elems_list(arr, count, start=0):
        return [arr[start + i] for i in range(count)]

    @T.macro
    def mask(S_reg, consumer_id, warp_id_in_wg, lane_id, q_row_base, kv_col_base, QO_LEN, KV_LEN):
        quad_id = T.meta_var(lane_id // 4)
        quad_lane = T.meta_var(lane_id % 4)
        with T.thread():
            row = int_var()
            col = int_var()
            row[0] = q_row_base + consumer_id * 64 + warp_id_in_wg * 16 + quad_id
            col[0] = kv_col_base + quad_lane * 2 - (KV_LEN - QO_LEN)
            if not CAUSAL:
                for i in T.serial(S_REG_COUNT // 4):
                    if col[0] >= KV_LEN:
                        S_reg[i * 4] = -INF
                        S_reg[i * 4 + 2] = -INF
                    if col[0] + 1 >= KV_LEN:
                        S_reg[i * 4 + 1] = -INF
                        S_reg[i * 4 + 3] = -INF
                    col[0] = col[0] + 8
            else:
                for i in T.serial(S_REG_COUNT // 4):
                    if (row[0] < col[0]):# or (col[0] - QO_LEN >= 0):
                        S_reg[i * 4] = -INF
                    if (row[0] < col[0] + 1):# or (col[0] + 1 - QO_LEN >= 0):
                        S_reg[i * 4 + 1] = -INF
                    if (row[0] + 8 < col[0]):# or (col[0] - QO_LEN >= 0):
                        S_reg[i * 4 + 2] = -INF
                    if (row[0] + 8 < col[0] + 1):# or (col[0] + 1 - QO_LEN >= 0):
                        S_reg[i * 4 + 3] = -INF
                    col[0] = col[0] + 8

    class Softmax:
        sm_scale_log2 = 1 / (HEAD_DIM**0.5) * math.log2(math.exp(1))
        row = 4 * BLK_Q // ((NUM_WARPS // 4 - 1) * 128)

        def __init__(self, prefix: str):
            self.row_max = T.alloc_buffer((self.row), "float32", scope="local") # m
            self.row_max_old = T.alloc_buffer((self.row), "float32", scope="local") # m_old
            self.row_sum = T.alloc_buffer((self.row), "float32", scope="local") # l
            self.scores_scale = T.alloc_buffer((self.row), "float32", scope="local") # e^*(m_old - m_new)
            IRBuilder.current().name(prefix + "_row_max", self.row_max)
            IRBuilder.current().name(prefix + "_row_max_old", self.row_max_old)
            IRBuilder.current().name(prefix + "_row_sum", self.row_sum)
            IRBuilder.current().name(prefix + "_scores_scale", self.scores_scale)

        @T.macro
        def init(self):
            for i in T.serial(self.row):
                self.row_max[i] = -INF
                self.row_sum[i] = 0
                self.scores_scale[i] = 1.0

        @T.macro
        def scale_o(self, O_regs):
            for i in T.serial(self.row):
                scale = T.meta_var(self.scores_scale[i])
                for j in T.serial(HEAD_DIM // 8):
                    O_regs[i * 2 + j * 4] = O_regs[i * 2 + j * 4] * scale
                    O_regs[i * 2 + j * 4 + 1] = O_regs[i * 2 + j * 4 + 1] * scale

        @T.macro
        def reduce_m(self, S_regs):
            for i in T.serial(self.row):
                row_max = T.meta_var(self.row_max[i])
                # thread reduce
                for j in T.serial(BLK_KV // 8):
                    self.row_max[i] = T.max(row_max, S_regs[i * 2 + j * 4])
                    self.row_max[i] = T.max(row_max, S_regs[i * 2 + j * 4 + 1])
                # quad reduce, (T0, T1, T2, T3), (T4, T5, T6, T7) ...
                self.row_max[i] = T.max(row_max, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_max, 2, 32, 32))
                self.row_max[i] = T.max(row_max, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_max, 1, 32, 32))

        @T.macro
        def scale_apply_exp2(self, S_regs):
            for i in T.serial(self.row):
                row_max = T.meta_var(self.row_max[i])
                for j in T.serial(BLK_KV // 8):
                    S_regs[i * 2 + j * 4] = T.exp2(S_regs[i * 2 + j * 4] * self.sm_scale_log2 - row_max * self.sm_scale_log2)
                    S_regs[i * 2 + j * 4 + 1] = T.exp2(S_regs[i * 2 + j * 4 + 1] * self.sm_scale_log2 - row_max * self.sm_scale_log2)

        @T.macro
        def reduce_l(self, S_regs):
            for i in T.serial(self.row):
                row_sum = T.meta_var(self.row_sum[i])
                self.row_sum[i] *= self.scores_scale[i]
                # thread reduce
                for j in T.serial(BLK_KV // 8):
                    self.row_sum[i] = row_sum + S_regs[i * 2 + j * 4]
                    self.row_sum[i] = row_sum + S_regs[i * 2 + j * 4 + 1]

        @T.macro
        def init_m_P_l(self, S_regs):
            self.reduce_m(S_regs)
            self.scale_apply_exp2(S_regs)
            self.reduce_l(S_regs)

        @T.macro
        def update_m_P_l(self, S_regs):
            for i in T.serial(self.row):
                self.row_max_old[i] = self.row_max[i]
            self.reduce_m(S_regs)
            for i in T.serial(self.row):
                self.scores_scale[i] = T.exp2((self.row_max_old[i] - self.row_max[i]) * self.sm_scale_log2)
            self.scale_apply_exp2(S_regs)
            self.reduce_l(S_regs)

        @T.macro
        def finalize(self):
            # quad reduce, (T0, T1, T2, T3), (T4, T5, T6, T7) ...
            for i in T.serial(self.row):
                self.row_sum[i] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, self.row_sum[i], 2, 32, 32)
                self.row_sum[i] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, self.row_sum[i], 1, 32, 32)
                self.scores_scale[i] = 1.0 / self.row_sum[i]

    @T.macro
    def gemm_QK(S_reg, smem_q, smem_k, desc_Q, desc_K, consumer_id, consumer_k):
        ptx_wgmma_noop_barrier(S_reg, S_REG_COUNT)
        T.ptx.wgmma.fence()

        k_stage = T.meta_var(consumer_k.index[0])
        with T.thread():
            scaleD = int_var()
            scaleD[0] = 0
            T.ptx.encode_matrix_descriptor(desc_Q.data, smem_q.access_ptr("r", offset=consumer_id * BLK_Q // 2 * TMA_TILE),
                                       BLK_Q * 8, 64, SWIZZLE)
            T.ptx.encode_matrix_descriptor(desc_K.data, smem_k.access_ptr("r", offset=smem_k.offset_of_p([k_stage, 0])),
                                       BLK_KV * 8, 64, SWIZZLE)
            for tma_tile in T.serial(HEAD_DIM // TMA_TILE):
                for wgmma_tile in T.serial(TMA_TILE // WGMMA_QK_K):
                    T.ptx.wgmma.mma_async.ss(
                        WGMMA_QK_M, WGMMA_QK_N, WGMMA_QK_K, "float16", "float32", False, False,
                        1.0, 1.0, scaleD[0], desc_Q[0], desc_K[0], *get_elems_list(S_reg, S_REG_COUNT)
                    )
                    scaleD[0] = 1
                    desc_Q[0] = desc_Q[0] + (2 * (WGMMA_QK_K) >> 4)
                    desc_K[0] = desc_K[0] + (2 * (WGMMA_QK_K) >> 4)
                desc_Q[0] = desc_Q[0] + (2 * (-TMA_TILE + BLK_Q * TMA_TILE) >> 4)
                desc_K[0] = desc_K[0] + (2 * (-TMA_TILE + BLK_KV * TMA_TILE) >> 4)
                
        T.ptx.wgmma.commit_group()
        ptx_wgmma_noop_barrier(S_reg, S_REG_COUNT)

    @T.macro
    def gemm_PV(O_reg, P_reg, smem_v, desc_V, consumer_v):
        ptx_wgmma_noop_barrier(P_reg, S_REG_COUNT // 2)
        ptx_wgmma_noop_barrier(O_reg, O_REG_COUNT)
        T.ptx.wgmma.fence()

        v_stage = T.meta_var(consumer_v.index[0])
        T.ptx.encode_matrix_descriptor(desc_V.data, smem_v.access_ptr("r", offset=smem_v.offset_of_p([v_stage, 0])),
                                   BLK_KV * 8, 64, SWIZZLE)
        for wgmma_tile in T.serial(BLK_KV // WGMMA_PV_K):
            P_offset = T.meta_var(wgmma_tile * WGMMA_PV_K // 4)
            T.ptx.wgmma.mma_async.rs(
                WGMMA_PV_M, WGMMA_PV_N, WGMMA_PV_K, "float16", "float32", False, True,
                1.0, 1.0, True, desc_V[0], *(get_elems_list(P_reg, WGMMA_PV_K // 4, P_offset) + get_elems_list(O_reg, O_REG_COUNT, 0))
            )
            desc_V[0] = desc_V[0] + (2 * (WGMMA_PV_K * TMA_TILE) >> 4)

        T.ptx.wgmma.commit_group()
        ptx_wgmma_noop_barrier(P_reg, S_REG_COUNT // 2)
        ptx_wgmma_noop_barrier(O_reg, O_REG_COUNT)

    # TODO(@bohan): try hygienic=True
    @T.macro
    def consumer_body(
        do_masking, pipeline_k, pipeline_v, consumer_k, consumer_v, softmax, smem_q, smem_k, smem_v, desc_Q, desc_K, desc_V, S_reg, P_reg, P_reg_fp16, O_reg,
        wg_id, warp_id_in_wg, lane_id, q_idx, kv_tile_idx_read, profiler_buffer, profiler_tag, profiler_write_offset
    ):
        pipeline_k.consumer_wait(consumer_k)
        T.timer_start_cuda(ProfileEventType.GemmQK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # commit S_i = Q^TK_i but do not wait
        gemm_QK(S_reg, smem_q, smem_k, desc_Q, desc_K, wg_id - 1, consumer_k)
        T.timer_start_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # scale O_{i-1} with scale_{i-1}
        softmax.scale_o(O_reg)
        T.timer_end_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # wait V_{i-1} to be loaded
        pipeline_v.consumer_wait(consumer_v)
        T.timer_start_cuda(ProfileEventType.GemmPV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # commit O_{i-1} += P_{i-1}V_{i-1} but do not wait
        gemm_PV(O_reg, P_reg, smem_v, desc_V, consumer_v)
        # wait for the gemm result of S_i = Q^TK_i
        T.ptx.wgmma.wait_group(1)
        T.timer_end_cuda(ProfileEventType.GemmQK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # release the current K tile
        pipeline_k.consumer_release(consumer_k)
        T.timer_start_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # mask S_i
        if do_masking:
            mask(S_reg, wg_id - 1, warp_id_in_wg, lane_id, q_idx * BLK_Q, kv_tile_idx_read[0] * BLK_KV, QO_LEN, KV_LEN)
        # update m, P, l for the current tile
        softmax.update_m_P_l(S_reg)
        T.timer_end_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # wait for P_{i-1}V_{i-1} to be computed
        T.ptx.wgmma.wait_group(0)
        T.timer_end_cuda(ProfileEventType.GemmPV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # release the previous V tile
        pipeline_v.consumer_release(consumer_v)
        consumer_k.advance()
        consumer_v.advance()
        T.timer_start_cuda(ProfileEventType.WritePReg, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
        # copy to P_{i-1} and downcast to float16
        for i in T.serial(S_REG_COUNT):
            P_reg_fp16[i] = T.Cast("float16", S_reg[i])
        T.timer_end_cuda(ProfileEventType.WritePReg, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)

    @T.macro
    def r2S(warp_id, lane_id, smem_o, O_reg, n_tile):
        T.ptx.bar.sync(NameBarrier.EPILOGUE, MMA_THREADS)
        with T.thread():
            O_half = T.alloc_buffer([8], "float32", scope="local")
            for st_tile in T.serial(4):
                for i in T.serial(8):
                    O_half[i] = O_reg[n_tile * 32 + st_tile * 8 + i]
                col_noswizzle = T.meta_var(st_tile * 2 + lane_id // 16)
                col = T.meta_var((lane_id % 8) ^ col_noswizzle)
                row = T.meta_var(warp_id * 16 + lane_id % 16)
                T.ptx.stmatrix(4, False, smem_o.access_ptr("w", offset=smem_o.offset_of_p([n_tile % STAGES_EPI, row, col * 8])),
                               O_half[0], O_half[1], O_half[2], O_half[3], O_half[4], O_half[5], O_half[6], O_half[7])

    @T.macro
    def s2G(warp_id, lane_id, smem_o, O_map, q_idx, h_idx, b_idx, n_tile):
        T.ptx.fence.proxy("shared")
        T.ptx.bar.sync(NameBarrier.EPILOGUE, MMA_THREADS)
        with T.thread()[warp_id == 0 and lane_id == 0]:
            T.ptx.cp_async.bulk.tensor.s2g(4, smem_o.access_ptr("r", offset=smem_o.offset_of_p([n_tile % STAGES_EPI, 0, 0])),
                                           O_map, n_tile * 64, h_idx, q_idx * BLK_Q, b_idx)
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(1, read=True)

    @T.macro
    def write_epilogue(warp_id, lane_id, q_idx, h_idx, b_idx, smem_o, O_map, O_reg):
        for n_tile in T.serial(HEAD_DIM // TMA_TILE):
            if n_tile != 0:
                # s2G for the previous stage
                s2G(warp_id, lane_id, smem_o, O_map, q_idx, h_idx, b_idx, n_tile - 1)
            # r2S for the current stage
            r2S(warp_id, lane_id, smem_o, O_reg, n_tile)
        # s2G for the last stage
        s2G(warp_id, lane_id, smem_o, O_map, q_idx, h_idx, b_idx, HEAD_DIM // TMA_TILE - 1)

    @T.prim_func(tirp=True)
    def manual(Q_ptr: T.handle, K_ptr: T.handle, V_ptr: T.handle, b_indices_ptr: T.handle, h_indices_ptr: T.handle,
               q_indices_ptr: T.handle, tiles_indptr_ptr: T.handle, O_ptr: T.handle, profiler_buffer_ptr: T.handle) -> None:
        qo_layout = T.meta_var(T.TileLayout.from_tuple(QO_SHAPE))
        kv_layout = T.meta_var(T.TileLayout.from_tuple(KV_SHAPE))
        Q = T.match_buffer(Q_ptr, QO_SHAPE, "float16", scope="global", layout=qo_layout)
        K = T.match_buffer(K_ptr, KV_SHAPE, "float16", scope="global", layout=kv_layout)
        V = T.match_buffer(V_ptr, KV_SHAPE, "float16", scope="global", layout=kv_layout)
        b_indices = T.match_buffer(b_indices_ptr, (total_qtiles,), "int32", scope="global")
        h_indices = T.match_buffer(h_indices_ptr, (total_qtiles,), "int32", scope="global")
        q_indices = T.match_buffer(q_indices_ptr, (total_qtiles,), "int32", scope="global")
        tiles_indptr = T.match_buffer(tiles_indptr_ptr, (SM_COUNT + 1,), "int32", scope="global")
        O = T.match_buffer(O_ptr, QO_SHAPE, "float16", scope="global", layout=qo_layout)
        profiler_buffer = T.match_buffer(profiler_buffer_ptr, (PROFILER_BUFFER_SIZE,), "uint64", scope="global")

        Q_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        K_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        V_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        O_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        GSHAPE_QO = T.meta_var((HEAD_DIM, NHEADS, QO_LEN, BATCH_SIZE))
        GSTRIDES_QO = T.meta_var((F16_BYTES * HEAD_DIM, F16_BYTES * HEAD_DIM * NHEADS, F16_BYTES * HEAD_DIM * NHEADS * QO_LEN))
        SSHAPE_QO = T.meta_var((TMA_TILE, 1, BLK_Q, 1))

        GSHAPE_KV = T.meta_var((HEAD_DIM, NHEADS, KV_LEN, BATCH_SIZE))
        GSTRIDES_KV = T.meta_var((F16_BYTES * HEAD_DIM, F16_BYTES * HEAD_DIM * NHEADS, F16_BYTES * HEAD_DIM * NHEADS * KV_LEN))
        SSHAPE_KV = T.meta_var((TMA_TILE, 1, BLK_KV, 1))

        COMMMON = T.meta_var((1, 1, 1, 1, 0, SWIZZLE, 0, 0))

        T.call_packed("runtime.cuTensorMapEncodeTiled", Q_map, "float16", 4, Q.data, *GSHAPE_QO, *GSTRIDES_QO, *SSHAPE_QO, *COMMMON)
        T.call_packed("runtime.cuTensorMapEncodeTiled", K_map, "float16", 4, K.data, *GSHAPE_KV, *GSTRIDES_KV, *SSHAPE_KV, *COMMMON)
        T.call_packed("runtime.cuTensorMapEncodeTiled", V_map, "float16", 4, V.data, *GSHAPE_KV, *GSTRIDES_KV, *SSHAPE_KV, *COMMMON)
        T.call_packed("runtime.cuTensorMapEncodeTiled", O_map, "float16", 4, O.data, *GSHAPE_QO, *GSTRIDES_QO, *SSHAPE_QO, *COMMMON)

        with T.kernel():
            bx = T.cta_id([SM_COUNT], parent="kernel")
            warp_id = T.warp_id([NUM_WARPS], parent="cta")
            tid = T.thread_id([NUM_WARPS * 32], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            wg_id = T.warpgroup_id([NUM_WARPS // 4], parent="cta")
            warp_id_in_wg = T.warp_id([4], parent="warpgroup")

            with T.cta():
                # dyn smem buffer
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                # smem storage
                bar_Q = T.decl_buffer([1], "uint64", buf.data, elem_offset=0)
                bar_O = T.decl_buffer([1], "uint64", buf.data, elem_offset=1)
                full_k = T.decl_buffer([KV_STAGES], "uint64", buf.data, elem_offset=2)
                empty_k = T.decl_buffer([KV_STAGES], "uint64", buf.data, elem_offset=2 + KV_STAGES)
                full_v = T.decl_buffer([KV_STAGES], "uint64", buf.data, elem_offset=2 + 2 * KV_STAGES)
                empty_v = T.decl_buffer([KV_STAGES], "uint64", buf.data, elem_offset=2 + 3 * KV_STAGES)
                smem_q = T.decl_buffer([BLK_Q * HEAD_DIM], "float16", buf.data, elem_offset=512) # 1024B
                smem_k = T.decl_buffer([KV_STAGES, BLK_KV * HEAD_DIM], "float16", buf.data, elem_offset=512 + BLK_Q * HEAD_DIM)
                # v and o share the same smem region for reuse
                smem_v = T.decl_buffer([KV_STAGES, BLK_KV * HEAD_DIM], "float16", buf.data, elem_offset=512 + (BLK_Q + BLK_KV * KV_STAGES) * HEAD_DIM)
                smem_o = T.decl_buffer([STAGES_EPI, BLK_Q, TMA_TILE], "float16", buf.data, elem_offset=512 + (BLK_Q + BLK_KV * KV_STAGES) * HEAD_DIM)

                with T.thread():
                    # tile scheduler
                    tile_scheduler = T.meta_var(TileScheduler("tile_scheduler", b_indices, h_indices, q_indices, tiles_indptr))
                    # pipeline
                    pipeline_k = T.meta_var(Pipeline(full_k, empty_k, TMA_BYTES_K))
                    pipeline_v = T.meta_var(Pipeline(full_v, empty_v, TMA_BYTES_V))
                    producer_k = T.meta_var(PipelineState("producer_k"))
                    producer_v = T.meta_var(PipelineState("producer_v"))
                    consumer_k = T.meta_var(PipelineState("consumer_k"))
                    consumer_v = T.meta_var(PipelineState("consumer_v"))
                    q_phase = int_var()
                    # producer WG regs
                    kv_tile_idx_load = int_var()
                    kv_tile_idx_read = int_var()
                    # smem desc_Q, desc_K, desc_V
                    desc_Q = T.alloc_buffer([1], "uint64", scope="local", align=8)
                    desc_K = T.alloc_buffer([1], "uint64", scope="local", align=8)
                    desc_V = T.alloc_buffer([1], "uint64", scope="local", align=8)
                    # accums
                    S_reg = T.alloc_buffer([S_REG_COUNT], "float32", scope="local")
                    P_reg = T.alloc_buffer([S_REG_COUNT // 2], "uint32", scope="local")
                    O_reg = T.alloc_buffer([O_REG_COUNT], "float32", scope="local")

                    # profiler
                    profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                    profiler_tag = T.alloc_buffer([1], "uint32", scope="local", align=8)
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data)

                    # softmax
                    softmax = T.meta_var(Softmax("softmax"))
                    ############################################################################## INITIALIZATION
                    # tile scheduler
                    tile_scheduler.init(bx)
                    # barriers
                    with T.thread()[tid == 0]:
                        T.ptx.mbarrier.init(bar_Q.access_ptr("rw", offset=0), 1)
                        T.ptx.mbarrier.init(bar_O.access_ptr("rw", offset=0), 1)
                    T.ptx.fence.mbarrier_init()
                    # pipelines
                    pipeline_k.init(tid)
                    pipeline_v.init(tid)
                    # pipeline states
                    producer_k.init(0, 1)
                    producer_v.init(0, 1)
                    consumer_k.init(0, 0)
                    consumer_v.init(0, 0)
                    q_phase[0] = 0
                    # sync to make sure everything is initialized and visible
                    T.tvm_storage_sync("shared")

                    bar_Q_ptr = T.meta_var(bar_Q.access_ptr("rw", offset=0))
                    q_idx = T.meta_var(tile_scheduler.q_idx[0])
                    h_idx = T.meta_var(tile_scheduler.h_idx[0])
                    b_idx = T.meta_var(tile_scheduler.b_idx[0])
                    attend_kv_len = T.meta_var(KV_LEN - QO_LEN + (q_idx + 1) * BLK_Q if CAUSAL else KV_LEN) # the kvlen to attend of the current q tile
                    num_kv_tiles = T.meta_var(ceildiv(attend_kv_len, BLK_KV)) # number of kv tiles to attend
                    is_leader = T.meta_var(T.ptx.elect_sync(0xFFFFFFFF))

                    P_reg_fp16 = T.decl_buffer([S_REG_COUNT], "float16", data=P_reg.data, elem_offset=0)
                    with T.cta():
                        T.block_attr({"tirp.scope_partition": True})
                        with T.warpgroup()[0:1]:
                            ############################################################################## PRODUCER
                            # deallocate registers
                            T.ptx.setmaxnreg(False, 24)
                            # only 1 warp is responsible for the producer
                            with T.warp()[0:1]:
                                while (tile_scheduler.valid()):
                                    if q_idx * BLK_Q >= QO_LEN:
                                        break
                                    kv_tile_idx_load[0] = num_kv_tiles - 1
                                    # copy a tile of K first
                                    with T.thread()[is_leader]:
                                        pipeline_k.producer_acquire(producer_k, profiler_buffer, profiler_tag, profiler_write_offset)
                                        pipeline_k.copy(producer_k, smem_k, K_map, 0, h_idx, kv_tile_idx_load[0] * BLK_KV, b_idx)
                                        T.timer_end_cuda(ProfileEventType.IssueLoadKV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                        producer_k.advance()
                                    # wait for consumers to finish the previous Q tile, then load the current Q tile
                                    T.ptx.bar.sync(NameBarrier.Q_EMPTY, 32 + MMA_THREADS) # 32 threads in producer, 256 threads in consumer
                                    T.timer_start_cuda(ProfileEventType.IssueLoadQ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                    with T.thread()[is_leader]:
                                        T.ptx.mbarrier.arrive.expect_tx(bar_Q_ptr, TMA_BYTES_Q)
                                        for tma_tile in T.serial(HEAD_DIM // TMA_TILE):
                                            T.ptx.cp_async.bulk.tensor.g2c(
                                                4, smem_q.access_ptr("w", offset=tma_tile * TMA_TILE * BLK_Q), bar_Q_ptr, Q_map,
                                                tma_tile * TMA_TILE, h_idx, q_idx * BLK_Q, b_idx
                                            )
                                    T.timer_end_cuda(ProfileEventType.IssueLoadQ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                    # wait for the consumers to finish writing last O_tile to gmem to reuse for Vsmem
                                    T.ptx.bar.sync(NameBarrier.V_LOAD_READY, 32 + MMA_THREADS)
                                    with T.thread()[is_leader]:
                                        while kv_tile_idx_load[0] > 0:
                                            # load k tiles
                                            pipeline_k.producer_acquire(producer_k, profiler_buffer, profiler_tag, profiler_write_offset)
                                            pipeline_k.copy(producer_k, smem_k, K_map, 0, h_idx, (kv_tile_idx_load[0] - 1) * BLK_KV, b_idx)
                                            T.timer_end_cuda(ProfileEventType.IssueLoadKV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                            producer_k.advance()
                                            # load v tiles
                                            pipeline_v.producer_acquire(producer_v, profiler_buffer, profiler_tag, profiler_write_offset)
                                            pipeline_v.copy(producer_v, smem_v, V_map, 0, h_idx, kv_tile_idx_load[0] * BLK_KV, b_idx)
                                            T.timer_end_cuda(ProfileEventType.IssueLoadKV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                            producer_v.advance()
                                            kv_tile_idx_load[0] = kv_tile_idx_load[0] - 1
                                        # load the last v tile
                                        pipeline_v.producer_acquire(producer_v, profiler_buffer, profiler_tag, profiler_write_offset)
                                        pipeline_v.copy(producer_v, smem_v, V_map, 0, h_idx, 0, b_idx)
                                        T.timer_end_cuda(ProfileEventType.IssueLoadKV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                        producer_v.advance()
                                # move to the next tile
                                    tile_scheduler.next_tile()
                                # wait until all the consumers finish
                                # in our case, I think it's fine to let producers exit early since there's no cluster coordination
                                with T.thread()[is_leader]:
                                    pipeline_k.producer_tail(producer_k)
                                    pipeline_v.producer_tail(producer_v)
                        with T.warpgroup()[1:3]:
                            ############################################################################## CONSUMER
                            # allocate registers
                            T.ptx.setmaxnreg(True, 240)
                            # inform the producer to Q is ready to be loaded
                            T.ptx.bar.arrive(NameBarrier.Q_EMPTY, 32 + MMA_THREADS)
                            while (tile_scheduler.valid()):
                                if q_idx * BLK_Q >= QO_LEN:
                                    break
                                kv_tile_idx_read[0] = num_kv_tiles - 1
                                # wait Q to be loaded
                                T.ptx.mbarrier.try_wait(bar_Q_ptr, q_phase[0])
                                q_phase[0] = q_phase[0] ^ 1
                                # wait first K tile to be loaded
                                pipeline_k.consumer_wait(consumer_k)
                                # initialize the S and O reg
                                for i in T.serial(S_REG_COUNT):
                                    S_reg[i] = T.float32(0)
                                for i in T.serial(O_REG_COUNT):
                                    O_reg[i] = T.float16(0)
                                # initialize the softmax (m=-INF, l=0)
                                softmax.init()
                                T.timer_start_cuda(ProfileEventType.GemmQK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # calculate S_0 = Q^TK_0
                                gemm_QK(S_reg, smem_q, smem_k, desc_Q, desc_K, wg_id - 1, consumer_k)
                                # wait for O of last tile to be written back to gmem, notifify the producer to load V
                                T.ptx.cp_async.bulk.wait_group(0)
                                T.ptx.bar.arrive(NameBarrier.V_LOAD_READY, 32 + MMA_THREADS)
                                # wait for the gemm result of S_0 = Q^TK_0
                                T.ptx.wgmma.wait_group(0)
                                T.timer_end_cuda(ProfileEventType.GemmQK, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # release the first K tile
                                pipeline_k.consumer_release(consumer_k)
                                consumer_k.advance()
                                T.timer_start_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # mask S_0
                                mask(S_reg, wg_id - 1, warp_id_in_wg, lane_id, q_idx * BLK_Q, kv_tile_idx_read[0] * BLK_KV, QO_LEN, KV_LEN)
                                # softmax, initialize m, P, l for the first tile
                                softmax.init_m_P_l(S_reg)
                                T.timer_end_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                T.timer_start_cuda(ProfileEventType.WritePReg, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # copy to P_0 and downcast to float16
                                for i in T.serial(S_REG_COUNT):
                                    P_reg_fp16[i] = T.Cast("float16", S_reg[i])
                                T.timer_end_cuda(ProfileEventType.WritePReg, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                with T.thread():
                                    masking_step = int_var()
                                    n_masking_steps = int_var()
                                    masking_step[0] = 0
                                    n_masking_steps[0] = 0 if not CAUSAL else ceildiv(BLK_Q, BLK_KV)
                                    kv_tile_idx_read[0] = kv_tile_idx_read[0] - 1
                                    # split out tiles of K and V that require masking
                                    while (masking_step[0] < n_masking_steps[0] and kv_tile_idx_read[0] >= 0):
                                        consumer_body(
                                            True, pipeline_k, pipeline_v, consumer_k, consumer_v, softmax, smem_q, smem_k, smem_v, desc_Q, desc_K, desc_V, S_reg, P_reg, P_reg_fp16, O_reg,
                                            wg_id, warp_id_in_wg, lane_id, q_idx, kv_tile_idx_read, profiler_buffer, profiler_tag, profiler_write_offset
                                        )
                                        masking_step[0] = masking_step[0] + 1
                                        kv_tile_idx_read[0] = kv_tile_idx_read[0] - 1
                                    # no masking
                                    while kv_tile_idx_read[0] >= 0:
                                        consumer_body(
                                            False, pipeline_k, pipeline_v, consumer_k, consumer_v, softmax, smem_q, smem_k, smem_v, desc_Q, desc_K, desc_V, S_reg, P_reg, P_reg_fp16, O_reg,
                                            wg_id, warp_id_in_wg, lane_id, q_idx, kv_tile_idx_read, profiler_buffer, profiler_tag, profiler_write_offset
                                        )
                                        kv_tile_idx_read[0] = kv_tile_idx_read[0] - 1
                                # notify the producer to load the next Q tile
                                T.ptx.bar.arrive(NameBarrier.Q_EMPTY, 32 + MMA_THREADS)
                                T.timer_start_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # scale O for the last tile
                                softmax.scale_o(O_reg)
                                T.timer_end_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # wait for the last V tile to be loaded
                                pipeline_v.consumer_wait(consumer_v)
                                T.timer_start_cuda(ProfileEventType.GemmPV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # commit the last O += PV
                                gemm_PV(O_reg, P_reg, smem_v, desc_V, consumer_v)
                                T.timer_start_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # get final l, do quad reduce and l^(-1)
                                softmax.finalize()
                                T.timer_end_cuda(ProfileEventType.SoftmaxUpdate, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # wait for the last O += PV to be computed
                                T.ptx.wgmma.wait_group(0)
                                T.timer_end_cuda(ProfileEventType.GemmPV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # release the last V tile
                                pipeline_v.consumer_release(consumer_v)
                                consumer_v.advance()
                                T.timer_start_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # final scale o using diagonal l^(-1)
                                softmax.scale_o(O_reg)
                                T.timer_end_cuda(ProfileEventType.ScaleO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                ####### Epilogue, write O back to gmem
                                # make sure all consumer threads finish V consumption, so Vsmem can be reused for Osmem
                                T.ptx.bar.sync(NameBarrier.O_LOAD_READY, MMA_THREADS)
                                T.timer_start_cuda(ProfileEventType.WriteO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)
                                # write O back to gmem
                                write_epilogue(warp_id - 4, lane_id, q_idx, h_idx, b_idx, smem_o, O_map, O_reg)
                                T.timer_end_cuda(ProfileEventType.WriteO, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE)

                                # move to the next tile
                                tile_scheduler.next_tile()
    # fmt: on

    q = np.random.randn(*QO_SHAPE).astype(np.float16)
    k = np.random.randn(*KV_SHAPE).astype(np.float16)
    v = np.random.randn(*KV_SHAPE).astype(np.float16)

    np.set_printoptions(precision=3, suppress=True)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    def fa3():
        from flash_attn_interface import flash_attn_func
        import torch

        q_torch = torch.from_numpy(q).cuda()
        k_torch = torch.from_numpy(k).cuda()
        v_torch = torch.from_numpy(v).cuda()

        func = lambda: flash_attn_func(q_torch, k_torch, v_torch, causal=CAUSAL)[0]
        ms = bench(func, warmup=0, repeat=10, proton_name="fa3")
        print(f"FA3 flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        o_torch = func()
        return o_torch.cpu().numpy()

    def tir():
        q_tvm = tvm.nd.array(q, DEV)
        k_tvm = tvm.nd.array(k, DEV)
        v_tvm = tvm.nd.array(v, DEV)
        o_tvm = tvm.nd.array(np.zeros(QO_SHAPE, dtype=np.float16), DEV)

        # profiler
        profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
        profiler_buffer_tvm = tvm.nd.array(profiler_buffer, DEV)

        from heapq import heappush, heappop

        def cost(qo_len, kv_len):
            return 2 * qo_len + kv_len

        q_indices = [[] for _ in range(SM_COUNT)]
        h_indices = [[] for _ in range(SM_COUNT)]
        b_indices = [[] for _ in range(SM_COUNT)]

        heap = []
        for cta in range(SM_COUNT):
            heappush(heap, (0.0, cta))

        num_q_tiles = ceildiv(QO_LEN, BLK_Q)

        for h_idx in range(NHEADS):
            for q_idx in reversed(range(num_q_tiles)):
                for b_idx in range(BATCH_SIZE):
                    cur_cost, cta = heappop(heap)
                    heappush(
                        heap,
                        (
                            cur_cost
                            + cost(
                                BLK_Q,
                                (
                                    KV_LEN
                                    if not CAUSAL
                                    else KV_LEN - (num_q_tiles - q_idx - 1) * BLK_Q
                                ),
                            ),
                            cta,
                        ),
                    )
                    q_indices[cta].append(q_idx)
                    h_indices[cta].append(h_idx)
                    b_indices[cta].append(b_idx)

        tiles_indptr = np.cumsum([0] + [len(b) for b in b_indices]).astype(np.int32)
        q_indices = np.concatenate(q_indices).astype(np.int32)
        h_indices = np.concatenate(h_indices).astype(np.int32)
        b_indices = np.concatenate(b_indices).astype(np.int32)
        tiles_indptr_tvm = tvm.nd.array(tiles_indptr, DEV)
        q_indices_tvm = tvm.nd.array(q_indices, DEV)
        h_indices_tvm = tvm.nd.array(h_indices, DEV)
        b_indices_tvm = tvm.nd.array(b_indices, DEV)

        with target:
            with tvm.transform.PassContext(
                config={"tir.RemoveNoOp": {"ignore_profiler_call": True}}
            ):
                mod = tvm.IRModule({"main": manual})
                mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
                func = lambda: mod(
                    q_tvm,
                    k_tvm,
                    v_tvm,
                    b_indices_tvm,
                    h_indices_tvm,
                    q_indices_tvm,
                    tiles_indptr_tvm,
                    o_tvm,
                    profiler_buffer_tvm,
                )
                ms = bench(func, warmup=0, repeat=10, proton_name="tir")
                print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        export_to_perfetto_trace(
            profiler_buffer_tvm.numpy(),
            f"mla-{BATCH_SIZE}-{QO_LEN}-{NHEADS}.perfetto-trace",
        )

        return o_tvm.numpy()

    def flashinfer():
        import flashinfer
        import torch

        q_torch = torch.from_numpy(q).cuda()
        k_torch = torch.from_numpy(k).cuda()
        v_torch = torch.from_numpy(v).cuda()

        q_torch = q_torch.reshape(-1, NHEADS, HEAD_DIM)
        k_torch = k_torch.reshape(-1, NHEADS, HEAD_DIM)
        v_torch = v_torch.reshape(-1, NHEADS, HEAD_DIM)

        sm90_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"),
            kv_layout="NHD",
            backend="fa3",
        )

        qo_indptr = torch.arange(0, BATCH_SIZE * QO_LEN + 1, QO_LEN).int()
        kv_indptr = torch.arange(0, BATCH_SIZE * KV_LEN + 1, KV_LEN).int()

        sm90_wrapper.plan(
            qo_indptr,
            kv_indptr,
            NHEADS,
            NHEADS,
            HEAD_DIM,
            causal=CAUSAL,
        )

        func = lambda: sm90_wrapper.run(q_torch, k_torch, v_torch)
        ms = bench(func, warmup=0, repeat=10, proton_name="flashinfer")
        print(f"FlashInfer flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        o_torch = func()
        return o_torch.reshape(BATCH_SIZE, QO_LEN, NHEADS, HEAD_DIM).cpu().numpy()

    with ProtonContext("fused_attn"):
        O_fa3 = fa3()
        O_tir = tir()
        O_flashinfer = flashinfer()

    np.testing.assert_allclose(O_fa3, O_tir, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(O_fa3, O_flashinfer, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_fp16_fused_attn()
