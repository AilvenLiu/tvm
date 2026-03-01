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

import math
from enum import Enum

import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tir.layout import S, TCol, TileLayout, TLane
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg
from tvm.tirx.bench.utils import CudaProfiler, ProtonContext, bench, export_to_perfetto_trace
from tvm.tirx.tile_scheduler import FlashAttentionLinearScheduler, FlashAttentionLPTScheduler

M_CLUSTER = 1
N_CLUSTER = 1
SM_NUMBER = 148

NUM_GROUPS = 6
PROFILER_BUFFER_SIZE = int(2e6)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
PROFILER_ON = False


class ProfileEventType(Enum):
    IssueTMA_Q = 0
    IssueTMA_K = 1
    IssueTMA_V = 2
    IssueMMA_QK = 3
    IssueMMA_PV = 4
    Softmax_MAX = 5
    Softmax_FMA = 6
    Softmax_EXP2 = 7
    Softmax_TMEM_ST = 8
    Softmax_SUM = 9
    Correction = 10
    EpiLDTMEM = 11
    TMAStore = 12


event_type_names = [
    "issue-tma-q",
    "issue-tma-k",
    "issue-tma-v",
    "issue-mma-qk",
    "issue-mma-pv",
    "softmax-max",
    "softmax-fma",
    "softmax-exp2",
    "softmax-tmem-st",
    "softmax-sum",
    "correction",
    "epi-ld-tmem",
    "tma-store",
]

WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER

N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3

BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False


MMA_M = 128
MMA_N = 128
MMA_K = 16

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = tvm.DataType("float16")
b_type_qk = tvm.DataType("float16")
d_type_qk = tvm.DataType("float32")
a_type_pv = tvm.DataType("float16")
b_type_pv = tvm.DataType("float16")
d_type_pv = tvm.DataType("float32")


# fmt: off
def get_flash_attention4_kernel(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False):  # noqa: E501

    BATCH_SIZE = batch_size
    SEQ_LEN_Q = seq_len_q
    SEQ_LEN_KV = seq_len_kv
    NUM_QO_HEADS = num_qo_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim

    # GQA parameters
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # e.g., 4 for num_qo_heads=32, num_kv_heads=8
    SEQ_Q_PER_TILE = BLK_M // GQA_RATIO       # e.g., 32 sequence positions per tile

    HEAD_DIM // MMA_K
    NUM_MMA_PV = BLK_N // MMA_K
    NUM_BLK_K = HEAD_DIM // BLK_K
    NUM_EPI_TILE = HEAD_DIM // EPI_TILE
    CTA_GROUP = 1
    SWIZZLE = 3

    # Block info for causal masking (following flash_attn/cute/block_info.py)
    def get_n_block_max(m_block_idx, causal):
        """Maximum KV block index (exclusive) for this Q block."""
        n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
        if not causal:
            return n_block_max
        # For causal: only process KV blocks up to diagonal
        # SEQ_Q_PER_TILE is already BLK_M // GQA_RATIO, so already in sequence coordinates
        m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
        return Tx.min(n_block_max, ceildiv(n_idx, BLK_N))

    def get_n_block_min_causal_mask(m_block_idx):
        """KV block index where causal masking stops being needed.
        Blocks with index < this value don't need causal masking.
        """
        # SEQ_Q_PER_TILE is already in sequence coordinates (BLK_M // GQA_RATIO)
        m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
        return Tx.max(0, n_idx // BLK_N)


    # L2 cache optimization for LPT scheduling (causal attention)
    L2_SIZE = 50 * 1024 * 1024  # 50MB L2 cache
    SIZE_ONE_KV_HEAD = SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES  # K+V size per head
    L2_SWIZZLE = 1 if L2_SIZE < SIZE_ONE_KV_HEAD else (1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD)))  # noqa: E501

    SSCALE_TOTAL_SIZE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M
    assert TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"

    def ceildiv(a, b):
        return (a + b - 1) // b

    def combine_int_frac_ex2(x_rounded, frac_ex2):
        func_name = "combine_int_frac_ex2"
        source_code = f"""
__device__ __forceinline__ float {func_name}(float x_rounded, float frac_ex2) {{
  float out;
  asm volatile(
    "{{\\n\\t"
    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\\n\\t"
    "mov.b32 x_rounded_i, %1;\\n\\t"
    "mov.b32 frac_ex_i, %2;\\n\\t"
    "shl.b32 x_rounded_e, x_rounded_i, 23;\\n\\t"
    "add.s32 out_i, x_rounded_e, frac_ex_i;\\n\\t"
    "mov.b32 %0, out_i;\\n\\t"
    "}}\\n"
    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));
  return out;
}}
"""
        return Tx.cuda.func_call(
            func_name, x_rounded, frac_ex2, source_code=source_code, return_type="float32"
        )

    @Tx.inline
    def ex2_emulation_2(out, idx, x, y):
        # Polynomial coefficients for exp2 approximation (degree 3)
        poly_ex2_deg3 = Tx.meta_var(
            (
                1.0,
                0.695146143436431884765625,
                0.227564394474029541015625,
                0.077119089663028717041015625,
            )
        )
        fp32_round_int = Tx.meta_var(float(2**23 + 2**22))

        # Clamp inputs to avoid overflow (we assume x, y <= 127.0)
        xy_clamped = Tx.alloc_local([2], "float32")
        xy_clamped[0] = Tx.max(x, -127.0)
        xy_clamped[1] = Tx.max(y, -127.0)

        # Round down to get integer part (stored as float with integer in lower bits)
        xy_rounded = Tx.alloc_local([2], "float32")
        Tx.ptx.add_packed_f32x2(xy_clamped[0], xy_clamped[1], fp32_round_int, fp32_round_int, Tx.address_of(xy_rounded[0]), rounding_mode="rm")  # noqa: E501

        # Subtract to get the rounded-back value (round to nearest even)
        xy_rounded_back = Tx.alloc_local([2], "float32")
        Tx.ptx.sub_packed_f32x2(xy_rounded[0], xy_rounded[1], fp32_round_int, fp32_round_int, Tx.address_of(xy_rounded_back[0]), rounding_mode="rn")  # noqa: E501

        # Compute fractional part: xy_frac = xy_clamped - xy_rounded_back
        xy_frac = Tx.alloc_local([2], "float32")
        Tx.ptx.sub_packed_f32x2(xy_clamped[0], xy_clamped[1], xy_rounded_back[0], xy_rounded_back[1], Tx.address_of(xy_frac[0]), rounding_mode="rn")  # noqa: E501

        # Evaluate polynomial using Horner's method: ((poly[3] * x + poly[2]) * x + poly[1]) * x + poly[0]  # noqa: E501
        xy_frac_ex2 = Tx.alloc_local([2], "float32")
        xy_frac_ex2[0] = poly_ex2_deg3[3]
        xy_frac_ex2[1] = poly_ex2_deg3[3]
        Tx.ptx.fma_packed_f32x2(xy_frac_ex2[0], xy_frac_ex2[1], xy_frac[0], xy_frac[1], poly_ex2_deg3[2], poly_ex2_deg3[2], Tx.address_of(xy_frac_ex2[0]))  # noqa: E501
        Tx.ptx.fma_packed_f32x2(xy_frac_ex2[0], xy_frac_ex2[1], xy_frac[0], xy_frac[1], poly_ex2_deg3[1], poly_ex2_deg3[1], Tx.address_of(xy_frac_ex2[0]))  # noqa: E501
        Tx.ptx.fma_packed_f32x2(xy_frac_ex2[0], xy_frac_ex2[1], xy_frac[0], xy_frac[1], poly_ex2_deg3[0], poly_ex2_deg3[0], Tx.address_of(xy_frac_ex2[0]))  # noqa: E501

        # Combine integer and fractional parts: shift integer left by 23 bits and add to fractional exp2  # noqa: E501
        out[idx] = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
        out[idx + 1] = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])

    @Tx.meta_class
    class Barriers:

        def __init__(self, pool_allocator, pipe_depth, is_p2c):
            self.mbar = pool_allocator.alloc([pipe_depth], "uint64")
            self.init_phase = 0 if is_p2c else 1
            self.pipe_depth = pipe_depth

        @Tx.inline
        def init(self, threads_num_wait):
            with Tx.thread()[0:1]:
                for i in Tx.serial(self.pipe_depth):
                    Tx.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

        @Tx.inline
        def wait(self, idx, phase):
            Tx.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)

    class BarrierWithCommit(Barriers):
        @Tx.inline
        def arrive(self, idx):
            if CTA_GROUP == 1:
                if Tx.ptx.elect_sync():
                    Tx.ptx.tcgen05.commit(self.mbar.ptr_to([idx]))

    class BarrierWithArrive(Barriers):
        @Tx.inline
        def arrive(self, idx):
            Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))

    class BarrierWithExpectTx(Barriers):
        @Tx.inline
        def arrive(self, idx, expected_bytes=None):
            if expected_bytes is not None:
                Tx.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)
            else:
                Tx.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))

    @Tx.meta_class
    class SmemDescriptor:
        def __init__(self, prefix: str):
            self.desc = Tx.local_scalar("uint64", name=prefix + "sdesc")

        @Tx.inline
        def init(self, smem_ptr, ldo, sdo, swizzle):
            Tx.ptx.tcgen05.encode_matrix_descriptor(
                Tx.address_of(self.desc), smem_ptr, ldo, sdo, swizzle
            )
            # Make the lo part (containing start_address) warp-uniform in-place
            self._make_lo_uniform()

        def _make_lo_uniform(self):
            """Shuffle the lower 32 bits of the descriptor to ensure warp-uniformity."""
            func_name = "smem_desc_make_lo_uniform"
            source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc) {{
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}}
"""
            return Tx.cuda.func_call(
                func_name, Tx.address_of(self.desc),
                source_code=source_code, return_type="void"
            )

        def add_16B_offset(self, offset):
            """Add 16B-aligned offset to lower 32 bits only."""
            func_name = "tvm_builtin_smem_desc_add_16B_offset"
            source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
            return Tx.cuda.func_call(
                func_name, self.desc, offset, source_code=source_code, return_type="uint64"
            )

    Q_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH_Q, BLK_M, NUM_BLK_K, BLK_K) : (BLK_M * HEAD_DIM, BLK_K, BLK_M * BLK_K, 1)]))  # noqa: E501
    K_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH_KV, BLK_N, NUM_BLK_K, BLK_K) : (BLK_N * HEAD_DIM, BLK_K, BLK_N * BLK_K, 1)]))  # noqa: E501
    O_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, NUM_EPI_TILE, EPI_TILE) : (BLK_M * HEAD_DIM, EPI_TILE, BLK_M * EPI_TILE, 1)]))  # noqa: E501

    @Tx.prim_func(tirx=True)
    def flash_attention4(
        Q: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
        K: Tx.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        V: Tx.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        O: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),  # noqa: E741
        profiler_buffer: Tx.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
    ):
        # For GQA: each tile processes SEQ_Q_PER_TILE seq positions (not BLK_M)
        num_q_blocks_total = Tx.meta_var(ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE))
        num_q_blocks_per_cta = Tx.meta_var(SMEM_PIPE_DEPTH_Q)
        num_q_blocks = Tx.meta_var(ceildiv(num_q_blocks_total, num_q_blocks_per_cta))

        # Task scheduling
        num_total_tasks = Tx.meta_var(BATCH_SIZE * NUM_KV_HEADS * num_q_blocks)

        # use non-persistent kernel for causal attention
        max_ctas: Tx.let = 148
        cta_count: Tx.let = Tx.min(max_ctas, num_total_tasks) if not is_causal else num_total_tasks

        with Tx.kernel():
            bx = Tx.cta_id([cta_count], parent="kernel")
            wg_id = Tx.warpgroup_id([4], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            Tx.attr(0, "tirx.persistent_kernel", True)
            with Tx.cta():
                pool = Tx.PoolAllocator()
                # Allocate Q buffer with alignment
                Q_smem = pool.alloc((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16", layout=Q_layout, align=1024)  # noqa: E501
                # Allocate K and V buffers (they share the same offset)
                K_smem = pool.alloc((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16", layout=K_layout, align=1024)  # noqa: E501
                V_smem = K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM)
                # Allocate O buffer
                O_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16", layout=O_layout, align=1024)  # noqa: E501
                # Allocate sScale buffer (ACC_SCALE/ROW_SUM shared + ROW_MAX)
                sScale = pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
                tmem_addr = pool.alloc([1], "uint32")

                ACC_SCALE_BASE: Tx.let = 0
                ROW_SUM_BASE: Tx.let = 0  # Shares with ACC_SCALE


                # Allocate phase buffers using PoolAllocator
                phase_kv = Tx.alloc_local([1], "int32")

                phase_q = Tx.alloc_local([1], "int32")

                phase_s_full = Tx.alloc_local([1], "int32")

                phase_tmem = Tx.alloc_local([1], "int32")

                phase_s0_s1 = Tx.alloc_local([1], "int32")

                phase_q_load = Tx.alloc_local([1], "int32")

                stage_kv = Tx.alloc_local([1], "int32")

                bar_load_q_full = BarrierWithExpectTx(pool, SMEM_PIPE_DEPTH_Q, True)
                bar_load_q_empty = BarrierWithCommit(pool, SMEM_PIPE_DEPTH_Q, False)  # init_phase = 1  # noqa: E501

                bar_load_kv_full = BarrierWithExpectTx(pool, SMEM_PIPE_DEPTH_KV, True)
                bar_load_kv_empty = BarrierWithCommit(pool, SMEM_PIPE_DEPTH_KV, False)

                bar_p_full_o_rescaled = BarrierWithArrive(pool, 2, True)

                bar_s_full = BarrierWithCommit(pool, 2, True)

                bar_o_full = BarrierWithCommit(pool, 2, True)

                bar_softmax_corr_full = BarrierWithArrive(pool, 2, True)
                bar_softmax_corr_empty = BarrierWithArrive(pool, 2, False)

                bar_corr_epi_full = BarrierWithArrive(pool, TMEM_PIPE_DEPTH, True)
                bar_corr_epi_empty = BarrierWithArrive(pool, TMEM_PIPE_DEPTH, False)
                bar_p_full_2 = BarrierWithArrive(pool, 2, True)

                bar_s0_s1_sequence = BarrierWithArrive(pool, 8, True)

                bar_tmem_dealloc = BarrierWithArrive(pool, 1, True)
                pool.commit()

                profiler = CudaProfiler(profiler_buffer, write_stride=PROFILER_WRITE_STRIDE, num_groups=NUM_GROUPS, profiler_enabled=PROFILER_ON)  # noqa: E501

                if wg_id == 0 and warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr[0]), n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)  # noqa: E501
                    Tx.cuda.trap_when_assert_failed(tmem_addr[0] == Tx.uint32(0))

                tmem = Tx.decl_buffer((128, N_COLS_TMEM), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS_TMEM) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
                tmem_as_f16 = Tx.decl_buffer((128, N_COLS_TMEM * 2), "float16", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS_TMEM * 2) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

                # Create appropriate scheduler based on causal mode
                scheduler = (
                    FlashAttentionLPTScheduler(
                        "fa_scheduler",
                        num_batches=BATCH_SIZE,
                        num_heads=NUM_KV_HEADS,
                        num_m_blocks=num_q_blocks,
                        l2_swizzle=L2_SWIZZLE,
                    ) if is_causal else FlashAttentionLinearScheduler(
                        "fa_scheduler",
                        num_batches=BATCH_SIZE,
                        num_heads=NUM_KV_HEADS,
                        num_m_blocks=num_q_blocks,
                        num_ctas=cta_count,
                    )
                )

                scheduler.init(bx)  # Initialize with CTA ID

                if wg_id == 3 and warp_id == 1:
                    profiler.init(0)
                elif wg_id == 3 and warp_id == 2:
                    profiler.init(1)
                elif wg_id == 3 and warp_id == 0:
                    profiler.init(2)
                elif wg_id <= 1:
                    profiler.init(3 + wg_id)
                elif wg_id == 2:
                    profiler.init(5)

                stage_kv[0] = 0
                phase_q[0] = 0
                phase_kv[0] = 0
                phase_tmem[0] = 0
                phase_s_full[0] = 0
                if USE_S0_S1_BARRIER:
                    phase_s0_s1[0] = Tx.if_then_else(wg_id == 1, 0, 1)
                phase_q_load[0] = 0

                with Tx.thread()[0:1]:
                    bar_load_q_full.init(1)
                    bar_load_q_empty.init(1)
                    bar_load_kv_full.init(1)
                    bar_load_kv_empty.init(1)
                    bar_p_full_o_rescaled.init(256)
                    bar_p_full_2.init(128)
                    bar_s_full.init(1)
                    bar_o_full.init(1)
                    bar_softmax_corr_full.init(128)
                    bar_softmax_corr_empty.init(128)
                    bar_corr_epi_full.init(128)
                    bar_corr_epi_empty.init(32)
                    bar_s0_s1_sequence.init(32)
                    bar_tmem_dealloc.init(1)

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cta_sync()
                if wg_id == 2:
                    for i_q in Tx.unroll(2):
                        bar_p_full_o_rescaled.arrive(i_q)

                @Tx.inline
                def advance_kv_stage():
                    stage_kv[0] = stage_kv[0] + 1
                    if stage_kv[0] == SMEM_PIPE_DEPTH_KV:
                        stage_kv[0] = 0
                        phase_kv[0] ^= 1

                num_kv_blocks: Tx.let = ceildiv(SEQ_LEN_KV, BLK_N)
                tmem_s_base: Tx.let = 0
                tmem_o_base: Tx.let = 256
                tmem_p_base: Tx.let = 64
                tmem_offset: Tx.let = 128

                while scheduler.valid():
                    # Extract indices from scheduler
                    m_block_idx = Tx.meta_var(scheduler.m_block_idx)
                    batch_idx = Tx.meta_var(scheduler.batch_idx)
                    kv_head_idx = Tx.meta_var(scheduler.head_idx)
                    # m_start refers to SEQ_Q positions (not BLK_M rows)
                    m_start = Tx.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q)
                    with Tx.cta():
                        # Tx.attr({"tirx.scope_partition": True})

                        if wg_id == 3:
                            Tx.ptx.setmaxnreg(False, 48)
                            if warp_id == 1:
                                with Tx.warp():

                                    @Tx.inline
                                    def load_q(i_q):
                                        # Use phase_q_load for Q prefetch barrier synchronization
                                        bar_load_q_empty.wait(i_q, phase_q_load[0])
                                        # stage_q[0] ->  0 -> 1 -> 0 -> 1 -> ...

                                        tma_copy_q = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_q_full.mbar.ptr_to([i_q]), "cta_group": CTA_GROUP})  # noqa: E501
                                        # GQA: Load each qo_head with 2D TMA copy
                                        # SMEM layout: row i corresponds to (seq = i // GQA_RATIO, head = i % GQA_RATIO)  # noqa: E501
                                        profiler.start(ProfileEventType.IssueTMA_Q, lane_id == 0)
                                        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)  # noqa: E501
                                        with Tx.thread()[Tx.ptx.elect_sync()]:
                                            Tx.copy_async(
                                                Q_smem_3d[i_q, :, :, :], Q[batch_idx, m_start + i_q * SEQ_Q_PER_TILE : m_start + (i_q + 1) * SEQ_Q_PER_TILE, kv_head_idx * GQA_RATIO: (kv_head_idx + 1) * GQA_RATIO, :],  # noqa: E501
                                                **tma_copy_q,
                                            )
                                            bar_load_q_full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES)  # ar(0,x)  # noqa: E501
                                        profiler.end(ProfileEventType.IssueTMA_Q, lane_id == 0)

                                    @Tx.inline
                                    def load_k(i_kv):
                                        bar_load_kv_empty.wait(stage_kv[0], phase_kv[0])
                                        tma_copy_k = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_kv_full.mbar.ptr_to([stage_kv[0]]), "cta_group": CTA_GROUP})  # noqa: E501
                                        profiler.start(ProfileEventType.IssueTMA_K, lane_id == 0)
                                        with Tx.thread()[Tx.ptx.elect_sync()]:
                                            Tx.copy_async(K_smem[stage_kv[0], :, :], K[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],  # noqa: E501
                                                **tma_copy_k,
                                            )
                                            bar_load_kv_full.arrive(stage_kv[0], CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)  # noqa: E501
                                        profiler.end(ProfileEventType.IssueTMA_K, lane_id == 0)
                                        advance_kv_stage()

                                    @Tx.inline
                                    def load_v(i_kv):
                                        bar_load_kv_empty.wait(stage_kv[0], phase_kv[0])
                                        tma_copy_v = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_kv_full.mbar.ptr_to([stage_kv[0]]), "cta_group": CTA_GROUP})  # noqa: E501
                                        profiler.start(ProfileEventType.IssueTMA_V, lane_id == 0)
                                        with Tx.thread()[Tx.ptx.elect_sync()]:
                                            Tx.copy_async(
                                                V_smem[stage_kv[0], :, :],
                                                V[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],  # noqa: E501
                                                **tma_copy_v,
                                            )
                                            bar_load_kv_full.arrive(stage_kv[0], CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)  # noqa: E501
                                        profiler.end(ProfileEventType.IssueTMA_V, lane_id == 0)
                                        advance_kv_stage()

                                    # For causal, compute reduced trip count for loads
                                    load_trip_count: Tx.int32
                                    load_trip_count = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks  # noqa: E501

                                    load_q(0)
                                    load_k(load_trip_count - 1)
                                    load_q(1)
                                    # Flip phase_q_load after Q stages complete (for persistent kernel)  # noqa: E501
                                    phase_q_load[0] ^= 1
                                    load_v(load_trip_count - 1)
                                    for _i in Tx.serial(load_trip_count - 1, annotations={"disable_unroll": True}):  # noqa: E501
                                        i_kv: Tx.let = load_trip_count - 2 - _i
                                        load_k(i_kv)
                                        load_v(i_kv)

                            elif warp_id == 2:
                                with Tx.warp():
                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):  # stage=0,1
                                        bar_corr_epi_full.wait(i_q, phase_tmem[0])
                                        if i_q == 0:
                                            profiler.start(ProfileEventType.TMAStore, lane_id == 0)
                                        # GQA: m_start_global refers to SEQ_Q positions
                                        m_start_global = Tx.meta_var(m_start + i_q * SEQ_Q_PER_TILE)
                                        # TMA O store: Store each qo_head with 2D TMA copy
                                        # SMEM layout: row i corresponds to (seq = i // GQA_RATIO, head = i % GQA_RATIO)  # noqa: E501
                                        O_smem_3d = O_smem.view(TMEM_PIPE_DEPTH, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)  # noqa: E501
                                        with Tx.thread()[Tx.ptx.elect_sync()]:
                                            Tx.copy_async(
                                                O[batch_idx, m_start_global : m_start_global + SEQ_Q_PER_TILE, kv_head_idx * GQA_RATIO: (kv_head_idx + 1) * GQA_RATIO, :],  # noqa: E501
                                                O_smem_3d[i_q, :, :, :],
                                                dispatch="tma",
                                            )
                                        Tx.ptx.cp_async.bulk.commit_group()
                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        Tx.ptx.cp_async.bulk.wait_group(1 - i_q)
                                        bar_corr_epi_empty.arrive(i_q)
                                    profiler.end(ProfileEventType.TMAStore, lane_id == 0)
                                    phase_tmem[0] ^= 1

                            elif warp_id == 0:
                                with Tx.warp():
                                    acc: Tx.int32
                                    acc = 0

                                    # Encode instruction descriptors once
                                    descI_qk: Tx.uint32
                                    Tx.ptx.tcgen05.encode_instr_descriptor(
                                        Tx.address_of(descI_qk), "float32", "float16", "float16",  # noqa: F821
                                        MMA_M, MMA_N, MMA_K,
                                        trans_a=False, trans_b=False, n_cta_groups=CTA_GROUP,
                                    )
                                    descI_pv: Tx.uint32
                                    Tx.ptx.tcgen05.encode_instr_descriptor(
                                        Tx.address_of(descI_pv), "float32", "float16", "float16",  # noqa: F821
                                        MMA_M, MMA_N, MMA_K,
                                        trans_a=False, trans_b=True, n_cta_groups=CTA_GROUP,
                                    )

                                    @Tx.inline
                                    def gemm_qk(q_stage, kv_stage, tmem_col_s, bar_s_full):
                                        descQ = SmemDescriptor("Q")
                                        descK = SmemDescriptor("K")
                                        # All threads: encode base descriptors ONCE
                                        descQ.init(Q_smem.ptr_to([q_stage, 0, 0]),
                                                   ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)  # noqa: E501
                                        descK.init(K_smem.ptr_to([kv_stage, 0, 0]),
                                                   ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE)  # noqa: E501

                                        for blk_k in Tx.unroll(NUM_BLK_K):
                                            for ki in Tx.unroll(BLK_K // MMA_K):
                                                k: Tx.let = blk_k * (BLK_K // MMA_K) + ki
                                                offset = Tx.meta_var(
                                                    (blk_k * BLK_M * BLK_K + ki * MMA_K) * F16_BYTES >> 4  # noqa: E501
                                                )
                                                if Tx.ptx.elect_sync():
                                                    Tx.ptx.tcgen05.mma(
                                                        "float32", "float16", "float16",
                                                        Tx.cuda.get_tmem_addr(0, 0, tmem_col_s),
                                                        descQ.add_16B_offset(offset),
                                                        descK.add_16B_offset(offset),
                                                        descI_qk,  # noqa: F821
                                                        use_a_tmem=False,
                                                        cta_group=CTA_GROUP,
                                                        enable_input_d=(k > 0),
                                                    )
                                        bar_s_full.arrive(q_stage)

                                    @Tx.inline
                                    def gemm_pv(i_q, kv_stage, tmem_col_o, tmem_col_p, should_accumulate, bar_p_full_2):  # noqa: E501
                                        descV = SmemDescriptor("V")
                                        # All threads: encode V descriptor ONCE
                                        descV.init(V_smem.ptr_to([kv_stage, 0, 0]),
                                                   ldo=BLK_N * BLK_K * F16_BYTES // F128_BYTES,
                                                   sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                                   swizzle=SWIZZLE)

                                        # First MMA (k=0): runtime accumulate flag
                                        if Tx.ptx.elect_sync():
                                            Tx.ptx.tcgen05.mma(
                                                "float32", "float16", "float16",
                                                Tx.cuda.get_tmem_addr(0, 0, tmem_col_o),
                                                Tx.cuda.get_tmem_addr(0, 0, tmem_col_p),
                                                descV.add_16B_offset(0), descI_pv,  # noqa: F821
                                                use_a_tmem=True, cta_group=CTA_GROUP,
                                                enable_input_d=should_accumulate,
                                            )


                                        # MMAs k=1..5 (before barrier wait)
                                        for k in Tx.unroll(1, 6):
                                            v_offset = Tx.meta_var((k * MMA_K * BLK_K * F16_BYTES) >> 4)  # noqa: E501
                                            if Tx.ptx.elect_sync():
                                                Tx.ptx.tcgen05.mma(
                                                    "float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(0, 0, tmem_col_o),
                                                    Tx.cuda.get_tmem_addr(0, 0, tmem_col_p + k * MMA_K // 2),  # noqa: E501
                                                    descV.add_16B_offset(v_offset), descI_pv,  # noqa: F821
                                                    use_a_tmem=True, cta_group=CTA_GROUP,
                                                    enable_input_d=True,
                                                )

                                        # Barrier wait (all threads)
                                        Tx.ptx.mbarrier.try_wait(bar_p_full_2.mbar.ptr_to([i_q]), phase_tmem[0])  # noqa: E501

                                        # MMAs k=6..NUM_MMA_PV-1 (after barrier wait)
                                        for k in Tx.unroll(6, NUM_MMA_PV):
                                            v_offset = Tx.meta_var((k * MMA_K * BLK_K * F16_BYTES) >> 4)  # noqa: E501
                                            if Tx.ptx.elect_sync():
                                                Tx.ptx.tcgen05.mma(
                                                    "float32", "float16", "float16",
                                                    Tx.cuda.get_tmem_addr(0, 0, tmem_col_o),
                                                    Tx.cuda.get_tmem_addr(0, 0, tmem_col_p + k * MMA_K // 2),  # noqa: E501
                                                    descV.add_16B_offset(v_offset), descI_pv,  # noqa: F821
                                                    use_a_tmem=True, cta_group=CTA_GROUP,
                                                    enable_input_d=True,
                                                )

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        tmem_col_s: Tx.let = tmem_s_base + i_q * tmem_offset
                                        bar_load_q_full.wait(i_q, phase_q_load[0])
                                        if i_q == 0:
                                            # for 2 q, confirm k is loaded
                                            bar_load_kv_full.wait(stage_kv[0], phase_kv[0])
                                        gemm_qk(i_q, stage_kv[0], tmem_col_s, bar_s_full)
                                        if i_q == 1:
                                            # finish twice qk mma
                                            bar_load_kv_empty.arrive(stage_kv[0])
                                    advance_kv_stage()

                                    # For causal, compute reduced trip count
                                    mma_trip_count: Tx.int32
                                    mma_trip_count = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks  # noqa: E501

                                    for i_kv in Tx.serial(
                                        mma_trip_count - 1, annotations={"disable_unroll": True}
                                    ):
                                        stage_v: Tx.let = stage_kv[0]
                                        phase_v: Tx.let = phase_kv[0]
                                        advance_kv_stage()
                                        stage_k = Tx.meta_var(stage_kv[0])
                                        phase_k = Tx.meta_var(phase_kv[0])

                                        for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                            tmem_col_s: Tx.let = tmem_s_base + i_q * tmem_offset
                                            tmem_col_p: Tx.let = tmem_p_base + i_q * tmem_offset
                                            tmem_col_o: Tx.let = tmem_o_base + i_q * tmem_offset
                                            if i_q == 0:
                                                # wait for v is loaded
                                                bar_load_kv_full.wait(stage_v, phase_v)
                                            # wait for o_full to be ready
                                            bar_p_full_o_rescaled.wait(i_q, phase_tmem[0])
                                            gemm_pv(i_q, stage_v, tmem_col_o, tmem_col_p, acc, bar_p_full_2)  # noqa: E501
                                            if i_q == 1:
                                                # finish twice pv mma
                                                bar_load_kv_empty.arrive(stage_v)
                                            if i_q == 0:
                                                # for 2 q, confirm k is loaded
                                                bar_load_kv_full.wait(stage_k, phase_k)
                                            gemm_qk(i_q, stage_k, tmem_col_s, bar_s_full)
                                            if i_q == 1:
                                                # finish twice qk mma
                                                bar_load_kv_empty.arrive(stage_k)
                                        acc = 1
                                        advance_kv_stage()
                                        phase_tmem[0] ^= 1

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        tmem_col_p: Tx.let = tmem_p_base + i_q * tmem_offset
                                        tmem_col_o: Tx.let = tmem_o_base + i_q * tmem_offset
                                        if i_q == 0:
                                            # wait for v is loaded
                                            bar_load_kv_full.wait(stage_kv[0], phase_kv[0])
                                        # wait for o_full to be ready
                                        bar_p_full_o_rescaled.wait(i_q, phase_tmem[0])
                                        gemm_pv(i_q, stage_kv[0], tmem_col_o, tmem_col_p, acc, bar_p_full_2)  # noqa: E501
                                        if i_q == 1:
                                            # finish twice pv mma
                                            bar_load_kv_empty.arrive(stage_kv[0])
                                        bar_o_full.arrive(i_q)
                                    advance_kv_stage()
                                    phase_tmem[0] ^= 1

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        bar_load_q_empty.arrive(i_q)

                                    # Flip phase_q_load after Q stages complete (for persistent kernel)  # noqa: E501
                                    phase_q_load[0] ^= 1

                        elif wg_id < 2:
                            with Tx.warpgroup():
                                # here phase_q and stage_q represent phase_tmem and stage_tmem

                                Tx.ptx.setmaxnreg(True, 200)

                                scale_log2 = Tx.meta_var(math.log2(math.e) / math.sqrt(HEAD_DIM))
                                rescale_threshold = Tx.meta_var(8.0)

                                row_max = Tx.alloc_local([1], "float32")
                                row_sum = Tx.alloc_local([1], "float32")

                                @Tx.inline
                                def mask_r2p(s_chunk_buf, col_limit, ncol: Tx.int32):
                                    """Apply mask using R2P-style bit manipulation.

                                    Optimizes: for j in range(N): buf[j] = -inf if j >= col_limit else buf[j]
                                    Into: bitmask operations that compile to R2P PTX instruction.

                                    Following flash_attn/cute/mask.py mask_r2p() lines 13-40:
                                    Process in 24-element chunks because shift by 31+ bits is problematic.
                                    For ncol=128: chunks 0-4 have 24 elements, chunk 5 has 8 elements.

                                    The bit test `mask & (1 << i)` compiles to the R2P (Register to Predicate)
                                    PTX instruction, which is more efficient than per-column comparisons.
                                    """  # noqa: E501
                                    ncol = Tx.meta_var(ncol)
                                    CHUNK_SIZE: Tx.let = 24  # Max safe shift amount (< 32)
                                    num_chunks: Tx.let = ceildiv(ncol, CHUNK_SIZE)

                                    for s in Tx.unroll(num_chunks):
                                        # Compute col_limit for this chunk (clamped to [0, chunk_cols])  # noqa: E501
                                        col_limit_s: Tx.let = Tx.max(col_limit - s * CHUNK_SIZE, 0)
                                        mask: Tx.uint32
                                        # Create bitmask: col_limit=5 -> 0b11111 (bits 0-4 set)
                                        mask = Tx.shift_left(Tx.int32(1), col_limit_s) - 1

                                        # Apply mask to each column in this chunk
                                        for i in Tx.unroll(CHUNK_SIZE):
                                            if i < ncol - s * CHUNK_SIZE:
                                                c: Tx.let = s * CHUNK_SIZE + i
                                                in_bound: Tx.let = Tx.bitwise_and(mask, Tx.shift_left(Tx.int32(1), i))  # noqa: E501
                                                s_chunk_buf[c] = Tx.Select(Tx.cast(in_bound, "bool"), s_chunk_buf[c], Tx.float32(-float("inf")))  # noqa: E501

                                @Tx.inline
                                def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                                    """Apply causal mask to attention scores.

                                    Following flash_attn/cute/mask.py apply_mask_sm100() lines 384-400:
                                    causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q
                                    row_idx = thread_row + m_block * tile_m
                                    col_limit_right = row_idx + causal_row_offset
                                    Mask if col >= col_limit_right

                                    Coordinate Mapping:
                                    - BLK_M = 128 packed rows per tile
                                    - SEQ_Q_PER_TILE = BLK_M // GQA_RATIO (e.g., 32 for GQA_RATIO=4)
                                    - Each warpgroup handles one Q stage with SEQ_Q_PER_TILE sequence positions
                                    - tid_in_wg (0-127) maps to packed rows: (seq_pos, head) = (tid//GQA_RATIO, tid%GQA_RATIO)
                                    """  # noqa: E501
                                    # Convert thread index to sequence position within warpgroup
                                    seq_pos_in_wg: Tx.let = tid_in_wg // GQA_RATIO

                                    # Global sequence position
                                    # wg_id 0/1 handles different Q stages (each stage has SEQ_Q_PER_TILE positions)  # noqa: E501
                                    # m_block covers SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q sequence positions  # noqa: E501
                                    row_idx: Tx.let = (m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q +  # noqa: E501
                                               wg_id * SEQ_Q_PER_TILE +
                                               seq_pos_in_wg)

                                    # Causal row offset (from mask.py:385)
                                    # For seq_len_q == seq_len_kv: causal_row_offset = 1 - n_block * BLK_N  # noqa: E501
                                    causal_row_offset: Tx.let = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q  # noqa: E501

                                    # Column limit: mask if col >= col_limit_right
                                    col_limit_right: Tx.let = row_idx + causal_row_offset

                                    # Use R2P-style masking instead of per-column comparison
                                    mask_r2p(s_chunk_buf, col_limit_right, BLK_N)

                                @Tx.inline
                                def softmax_step(i_kv, apply_mask=False, is_first=False):
                                    s_chunk_buf = Tx.alloc_local([BLK_N], "float32")
                                    s_chunk = s_chunk_buf.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501

                                    p_chunk_buf_f32 = Tx.alloc_local([BLK_N // 2], "float32")
                                    p_chunk_buf = Tx.decl_buffer((BLK_N,), dtype="float16", data=p_chunk_buf_f32.data)  # noqa: E501
                                    p_chunk = p_chunk_buf.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501

                                    tmem_col_s = Tx.meta_var(tmem_s_base + wg_id * tmem_offset)
                                    tmem_col_p = Tx.meta_var(tmem_p_base + wg_id * tmem_offset)

                                    bar_s_full.wait(wg_id, phase_s_full[0])
                                    profiler.start(ProfileEventType.Softmax_MAX, tid_in_wg == 0)
                                    tile_max = Tx.alloc_local([1], "float32")
                                    for chunk_idx in Tx.unroll(BLK_N // SOFTMAX_LD_CHUNK):
                                        Tx.copy_async(s_chunk[:, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK], tmem[:, tmem_col_s + chunk_idx * SOFTMAX_LD_CHUNK : tmem_col_s + chunk_idx * SOFTMAX_LD_CHUNK + SOFTMAX_LD_CHUNK])  # noqa: E501

                                    # Apply causal mask if needed
                                    if apply_mask:
                                        apply_causal_mask(s_chunk_buf, m_block_idx, i_kv)

                                    row_max_old = Tx.alloc_local([1], "float32")
                                    row_max_old[0] = row_max[0]
                                    with Tx.thread():
                                        if is_first:
                                            Tx.max(tile_max, s_chunk_buf)
                                        else:
                                            tile_max[0] = row_max_old[0]
                                            Tx.max(tile_max, s_chunk_buf, accum=True)
                                    row_max_new = Tx.alloc_local([1], "float32")
                                    acc_scale = Tx.alloc_local([1], "float32")
                                    acc_scale_ = Tx.alloc_local([1], "float32")  # For slack check
                                    row_max_safe = Tx.alloc_local([1], "float32")
                                    row_max_new[0] = tile_max[0]
                                    row_max_safe[0] = Tx.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])  # noqa: E501

                                    if is_first:
                                        acc_scale[0] = Tx.float32(1.0)
                                    else:
                                        acc_scale_[0] = (row_max_old[0] - row_max_safe[0]) * scale_log2  # noqa: E501

                                        # if the difference is too small, don't rescale
                                        if acc_scale_[0] >= -rescale_threshold:
                                            row_max_new[0] = row_max_old[0]
                                            row_max_safe[0] = row_max_old[0]
                                            acc_scale[0] = Tx.float32(1.0)
                                        else:
                                            acc_scale[0] = Tx.ptx.exp2(acc_scale_[0])

                                    # row_max is the max value of the tile
                                    # and row_max_scaled is the max value of the tile after scaled
                                    # scale_log2 is the log2 of the scale factor
                                    row_max[0] = row_max_new[0]
                                    row_max_scaled: Tx.let = row_max_safe[0] * scale_log2
                                    profiler.end(ProfileEventType.Softmax_MAX, tid_in_wg == 0)

                                    # Write acc_scale to sScale and arrive immediately (no wait here)  # noqa: E501
                                    if tid_in_wg < BLK_M and not is_first:
                                        sScale_idx: Tx.let = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M  # noqa: E501
                                        sScale[sScale_idx] = acc_scale[0]
                                    bar_softmax_corr_full.arrive(wg_id)
                                    profiler.start(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                                    for i in Tx.unroll(BLK_N // 2):
                                        Tx.ptx.fma_packed_f32x2(s_chunk_buf[2 * i], s_chunk_buf[2 * i + 1], scale_log2, scale_log2, -row_max_scaled, -row_max_scaled, Tx.address_of(s_chunk_buf[2 * i]))  # noqa: E501
                                    profiler.end(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                                    if USE_S0_S1_BARRIER:
                                        bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1[0])
                                    profiler.start(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                                    for frag_idx in Tx.unroll(4):
                                        for i in Tx.unroll(BLK_N // 4 // 2):
                                            idx = Tx.meta_var(frag_idx * BLK_N // 4 + 2 * i)
                                            if i * 2 % 16 < 16 - 4 or frag_idx >= 4 - 1 or apply_mask:  # noqa: E501
                                                s_chunk_buf[idx] = Tx.ptx.exp2(s_chunk_buf[idx])
                                                s_chunk_buf[idx + 1] = Tx.ptx.exp2(s_chunk_buf[idx + 1])  # noqa: E501
                                            else:
                                                ex2_emulation_2(s_chunk_buf, idx, s_chunk_buf[idx], s_chunk_buf[idx + 1])  # noqa: E501
                                        with Tx.thread():
                                            Tx.cast(p_chunk_buf[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4], s_chunk_buf[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4])  # noqa: E501
                                    if USE_S0_S1_BARRIER:
                                        bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id)
                                    profiler.end(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                                    profiler.start(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)
                                    for i in Tx.unroll(3):
                                        Tx.copy_async(tmem_as_f16[:, tmem_col_p * 2 + i * BLK_N // 4 : tmem_col_p * 2 + (i + 1) * BLK_N // 4], p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4])  # noqa: E501
                                    Tx.ptx.tcgen05.wait.st()
                                    bar_p_full_o_rescaled.arrive(wg_id)
                                    Tx.copy_async(tmem_as_f16[:, tmem_col_p * 2 + 3 * BLK_N // 4 : tmem_col_p * 2 + BLK_N], p_chunk[:, 3 * BLK_N // 4 : BLK_N])  # noqa: E501
                                    Tx.ptx.tcgen05.wait.st()
                                    bar_p_full_2.arrive(wg_id)

                                    profiler.end(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)

                                    # Wait for correction warp to finish reading previous acc_scale
                                    bar_softmax_corr_empty.wait(wg_id, phase_q[0])

                                    profiler.start(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                                    phase_s_full[0] ^= 1
                                    phase_q[0] ^= 1
                                    with Tx.thread():
                                        if is_first:
                                            Tx.sum(row_sum, s_chunk_buf)
                                        else:
                                            row_sum[0] = row_sum[0] * acc_scale[0]
                                            Tx.sum(row_sum, s_chunk_buf, accum=True)
                                    profiler.end(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                                    if USE_S0_S1_BARRIER:
                                        phase_s0_s1[0] ^= 1

                                bar_softmax_corr_empty.wait(wg_id, phase_q[0])
                                phase_q[0] ^= 1
                                # Compute block ranges for this Q block
                                n_block_max: Tx.let = get_n_block_max(m_block_idx, is_causal)
                                n_block_min_causal: Tx.let = get_n_block_min_causal_mask(m_block_idx) if is_causal else n_block_max  # noqa: E501

                                # Phase 1: Last KV block (n_block_max - 1) with causal mask
                                # This block may have both seqlen boundary AND causal masking
                                softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)

                                # Update n_block_max after Phase 1
                                n_block_max_after_p1: Tx.let = n_block_max - 1

                                # Phase 2: Blocks with partial causal masking
                                # These are blocks in [n_block_min_causal, n_block_max - 1)
                                num_phase2_blocks: Tx.let = Tx.max(n_block_max_after_p1 - n_block_min_causal, 0)  # noqa: E501
                                for i in Tx.serial(num_phase2_blocks, annotations={"disable_unroll": True}):  # noqa: E501
                                    n_block: Tx.let = n_block_max_after_p1 - 1 - i
                                    softmax_step(n_block, apply_mask=True)

                                # Update n_block_max after Phase 2
                                n_block_max_after_p2: Tx.let = Tx.min(n_block_max_after_p1, n_block_min_causal)  # noqa: E501

                                # Phase 3: Unmasked blocks (no causal mask overhead)
                                # These are blocks in [0, n_block_min_causal)
                                for i in Tx.serial(n_block_max_after_p2, annotations={"disable_unroll": True}):  # noqa: E501
                                    n_block: Tx.let = n_block_max_after_p2 - 1 - i
                                    softmax_step(n_block, apply_mask=False)
                                if tid_in_wg < BLK_M:
                                    sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                                bar_softmax_corr_full.arrive(wg_id)
                        elif wg_id == 2:
                            with Tx.warpgroup():
                                Tx.ptx.setmaxnreg(False, 64)

                                bar_softmax_corr_full.wait(0, phase_q[0])
                                bar_softmax_corr_empty.arrive(0)
                                bar_softmax_corr_full.wait(1, phase_q[0])
                                phase_q[0] ^= 1

                                # For causal, compute reduced trip count for correction warp
                                corr_trip_count: Tx.let = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks  # noqa: E501

                                for i_kv in Tx.serial(corr_trip_count - 1, annotations={"disable_unroll": True}):  # noqa: E501
                                    for i_q in Tx.unroll(2):
                                        bar_softmax_corr_full.wait(i_q, phase_q[0])
                                        profiler.start(ProfileEventType.Correction, tid_in_wg == 0)
                                        acc_scale = Tx.alloc_local([1], "float32")
                                        should_rescale = Tx.alloc_local([1], "int32")

                                        if tid_in_wg < BLK_M:
                                            acc_scale[0] = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]  # noqa: E501
                                            should_rescale[0] = Tx.Select(acc_scale[0] < Tx.float32(1.0), 1, 0)  # noqa: E501
                                        else:
                                            should_rescale[0] = 0

                                        any_needs_rescale: Tx.let = Tx.ptx.any_sync(0xFFFFFFFF, should_rescale[0])  # noqa: E501
                                        if any_needs_rescale != 0:
                                            if tid_in_wg < BLK_M:
                                                tmem_col_o_stage: Tx.let = tmem_o_base + i_q * tmem_offset  # noqa: E501
                                                RESCALE_TILE: Tx.let = 16

                                                o_row_buf = Tx.alloc_buffer((16,), "float32", scope="local")  # noqa: E501
                                                o_row_wg = o_row_buf.view(128, 16, layout=TileLayout(S[(128, 16) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501

                                                for d_tile in Tx.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):  # noqa: E501
                                                    d_start: Tx.let = d_tile * RESCALE_TILE
                                                    if d_start < HEAD_DIM:
                                                        Tx.copy_async(o_row_wg, tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16])  # noqa: E501
                                                        for d in Tx.unroll(8):
                                                            Tx.ptx.mul_packed_f32x2(o_row_buf[d * 2], o_row_buf[d * 2 + 1], acc_scale[0], acc_scale[0], o_row_buf.ptr_to([d * 2]))  # noqa: E501
                                                        Tx.copy_async(tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16], o_row_wg[:, 0:16])  # noqa: E501
                                                Tx.ptx.tcgen05.wait.st()

                                        bar_p_full_o_rescaled.arrive(i_q)
                                        bar_softmax_corr_empty.arrive(1 - i_q)
                                        profiler.end(ProfileEventType.Correction, tid_in_wg == 0)
                                    # flip epi producer phase
                                    phase_q[0] ^= 1
                                bar_softmax_corr_empty.arrive(1)

                                for i_q in Tx.unroll(2):
                                    # 1. Wait for softmax to signal row_sum is ready
                                    bar_softmax_corr_full.wait(i_q, phase_q[0])

                                    # 2. Read row_sum and release softmax_corr_empty immediately
                                    row_sum: Tx.let = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                                    bar_softmax_corr_empty.arrive(i_q)

                                    # 3. Wait for O_full and epi_empty (after releasing softmax)
                                    bar_o_full.wait(i_q, phase_tmem[0])
                                    bar_corr_epi_empty.wait(i_q, phase_tmem[0])

                                    profiler.start(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                                    acc_O_mn_row_is_zero_or_nan: Tx.let = tvm.tir.any(row_sum == Tx.float32(0.0), row_sum != row_sum)  # noqa: E501
                                    norm_scale: Tx.let = Tx.ptx.rcp(Tx.Select(acc_O_mn_row_is_zero_or_nan, Tx.float32(1.0), row_sum))  # noqa: E501
                                    tmem_col_o_stage: Tx.let = tmem_o_base + i_q * tmem_offset
                                    o_row_f32_buf = Tx.alloc_buffer((TMEM_EPI_LD_SIZE,), "float32", scope="local")  # noqa: E501
                                    o_row_f32_wg = o_row_f32_buf.view(128, TMEM_EPI_LD_SIZE, layout=TileLayout(S[(128, TMEM_EPI_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501
                                    o_row_f16 = Tx.alloc_local([TMEM_EPI_LD_SIZE], "float16")

                                    for d_tile in Tx.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                                        d_start: Tx.let = d_tile * TMEM_EPI_LD_SIZE
                                        if d_start < HEAD_DIM:
                                            Tx.copy_async(o_row_f32_wg, tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + TMEM_EPI_LD_SIZE])  # noqa: E501
                                            for d in Tx.unroll(TMEM_EPI_LD_SIZE // 2):
                                                Tx.ptx.mul_packed_f32x2(o_row_f32_buf[d * 2], o_row_f32_buf[d * 2 + 1], norm_scale, norm_scale, o_row_f32_buf.ptr_to([d * 2]))  # noqa: E501
                                            with Tx.thread():
                                                Tx.cast(o_row_f16, o_row_f32_buf)
                                                Tx.copy(O_smem[i_q, tid_in_wg, d_tile * TMEM_EPI_LD_SIZE : d_tile * TMEM_EPI_LD_SIZE + TMEM_EPI_LD_SIZE], o_row_f16, vec_len=8)  # noqa: E501

                                        profiler.end(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                                    Tx.ptx.fence.proxy_async("shared::cta")

                                    # arrive epi_full
                                    bar_corr_epi_full.arrive(i_q)
                                    # Signal for the next work tile that O buffers in tmem are already read  # noqa: E501
                                    bar_p_full_o_rescaled.arrive(i_q)
                                phase_tmem[0] ^= 1
                                phase_q[0] ^= 1

                    scheduler.next_tile()

                # Deallocate TMEM after all tasks complete
                if wg_id == 0 and warp_id == 0:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(0, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()

    return flash_attention4
# fmt: on


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)  # noqa: E741

    return Q, K, V, O


@pytest.mark.parametrize("seq_len", [8192, 4096, 2048, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [4, 8, 16, 32])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention4(seq_len, num_qo_heads, num_kv_heads, is_causal):
    BATCH = 1
    SEQ_Q = seq_len
    SEQ_KV = seq_len
    NUM_QO_HEADS = num_qo_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = 128
    DEBUG = False

    def flops(ms):
        """Calculate FLOPS for Flash Attention: Q@K^T + P@V = 2 * B * H * S_q * S_k * D"""
        # For causal, effective ops is approximately half
        effective_factor = 0.5 if is_causal else 1.0
        return 4 * BATCH * NUM_QO_HEADS * SEQ_Q * SEQ_KV * HEAD_DIM * effective_factor / (ms * 1e-3)

    Q, K, V, _ = prepare_data(BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM)

    def get_source(func):
        target = tvm.target.Target("cuda")
        mod = tvm.IRModule({"main": func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source(), flush=True)
        return mod

    def tir_attn(Q, K, V):
        Q_tir, K_tir, V_tir = Q, K, V
        O_tir = torch.zeros_like(Q)

        prim_func = get_flash_attention4_kernel(
            BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, is_causal=is_causal
        )
        mod = get_source(prim_func)

        dev = tvm.cuda(0)
        Q_tvm = tvm.runtime.tensor(Q_tir.cpu().numpy(), device=dev)
        K_tvm = tvm.runtime.tensor(K_tir.cpu().numpy(), device=dev)
        V_tvm = tvm.runtime.tensor(V_tir.cpu().numpy(), device=dev)
        O_tvm = tvm.runtime.tensor(O_tir.cpu().numpy(), device=dev)
        profiler_buffer = np.zeros((PROFILER_BUFFER_SIZE,), dtype=np.uint64)
        profiler_buffer_tvm = tvm.runtime.tensor(profiler_buffer, dev)

        def func():
            return mod(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buffer_tvm)

        ms = bench(func, warmup=100, repeat=300, proton_name="tir_fa4", debug=DEBUG)
        print(f"TIR FA4: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True)
        if PROFILER_ON:
            export_to_perfetto_trace(
                profiler_buffer_tvm.numpy(),
                f"fa4-{BATCH}-{SEQ_Q}-{SEQ_KV}-{NUM_QO_HEADS}-{NUM_KV_HEADS}-{HEAD_DIM}.perfetto-trace",
                event_type_names,
            )

        mod(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buffer_tvm)
        torch.cuda.synchronize()

        O_res = O_tvm.numpy()
        return O_res

    def cutedsl_attn(Q, K, V):
        """CuTeDSL Blackwell FMHA baseline"""
        try:
            import math
            import os
            import sys

            current_dir = os.path.dirname(os.path.abspath(__file__))
            tvm_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
            blackwell_path = os.path.join(
                tvm_root, "3rdparty/cutlass/examples/python/CuTeDSL/blackwell"
            )
            sys.path.insert(0, blackwell_path)
            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch
            from fmha import BlackwellFusedMultiHeadAttentionForward, MaskType
        except ImportError as e:
            print(f"CuTeDSL Blackwell FMHA not available: {e}, skipping baseline")
            return None

        Q_cute = Q.cuda()
        K_cute = K.cuda()
        V_cute = V.cuda()
        O_cute = torch.zeros_like(Q_cute)

        q_tensor, q_torch = cutlass_torch.cute_tensor_like(
            Q_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        k_tensor, k_torch = cutlass_torch.cute_tensor_like(
            K_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        v_tensor, v_torch = cutlass_torch.cute_tensor_like(
            V_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        o_tensor, o_torch = cutlass_torch.cute_tensor_like(
            O_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )

        q_torch.copy_(Q_cute)
        k_torch.copy_(K_cute)
        v_torch.copy_(V_cute)

        mma_tiler = (128, 128, HEAD_DIM)
        fmha = BlackwellFusedMultiHeadAttentionForward(
            cutlass.Float32,
            cutlass.Float32,
            mma_tiler,
            is_persistent=True,
            mask_type=MaskType.NO_MASK if not is_causal else MaskType.CAUSAL_MASK,
        )

        current_stream = cutlass_torch.default_stream()

        scale_softmax = 1.0 / math.sqrt(HEAD_DIM)
        log2_e = math.log2(math.exp(1.0))
        scale_softmax_log2 = scale_softmax * log2_e
        scale_output = 1.0

        problem_size = (BATCH, SEQ_Q, SEQ_KV, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM)
        cum_seqlen_q = None
        cum_seqlen_k = None

        compiled_fmha = cute.compile(
            fmha,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            current_stream,
        )

        def run_fmha():
            compiled_fmha(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                problem_size,
                cum_seqlen_q,
                cum_seqlen_k,
                scale_softmax_log2,
                scale_output,
                current_stream,
            )

        ms = bench(run_fmha, warmup=10, repeat=30, proton_name="cutedsl_fa4", debug=DEBUG)
        print(f"CuTeDSL FA: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True)

        # Run once for result
        run_fmha()
        torch.cuda.synchronize()

        return o_torch.cpu().numpy()

    def flashattn_sm100(Q, K, V):
        """Flash-Attention SM100 implementation from installed flash-attn package

        Note: Requires flash-attn to be installed with:
            pip install flash-attn --no-build-isolation
        """
        try:
            import math

            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch
            from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
        except ImportError as e:
            print(f"Flash-Attention SM100 not available: {e}")
            print("Install with: pip install flash-attn --no-build-isolation")
            print("Note: CuTeDSL baseline uses the same CUTLASS implementation")
            return None
        except Exception as e:
            print(f"Unexpected error loading Flash-Attention SM100: {e}")
            return None

        Q_fa = Q.cuda()
        K_fa = K.cuda()
        V_fa = V.cuda()
        O_fa = torch.zeros_like(Q_fa)

        q_tensor, q_torch = cutlass_torch.cute_tensor_like(
            Q_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        k_tensor, k_torch = cutlass_torch.cute_tensor_like(
            K_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        v_tensor, v_torch = cutlass_torch.cute_tensor_like(
            V_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        o_tensor, o_torch = cutlass_torch.cute_tensor_like(
            O_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )

        q_torch.copy_(Q_fa)
        k_torch.copy_(K_fa)
        v_torch.copy_(V_fa)

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=HEAD_DIM,
            head_dim_v=HEAD_DIM,
            qhead_per_kvhead=NUM_QO_HEADS // NUM_KV_HEADS,  # GQA
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
        )

        current_stream = cutlass_torch.default_stream()

        scale_softmax = 1.0 / math.sqrt(HEAD_DIM)

        compiled_fa = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            None,  # lse_tensor
            scale_softmax,
            current_stream,
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            None,  # softcap
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
        )

        def run_fa():
            compiled_fa(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                None,  # lse_tensor
                scale_softmax,
                current_stream,
                None,  # mCuSeqlensQ
                None,  # mCuSeqlensK
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                None,  # softcap
                None,  # window_size_left
                None,  # window_size_right
                None,  # learnable_sink
            )

        ms = bench(run_fa, warmup=100, repeat=300, proton_name="flashattn_sm100", debug=DEBUG)
        print(
            f"Flash-Attention SM100: {flops(ms) / 1e12:.2f} TFLOPS, time: {ms:.3f} ms", flush=True
        )

        run_fa()
        torch.cuda.synchronize()

        return o_torch.cpu().numpy()

    def flashinfer(Q, K, V):
        import flashinfer

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD", backend="cutlass"
        )
        qo_indptr = torch.tensor([0, SEQ_Q], device="cuda:0", dtype=torch.int32)
        kv_indptr = torch.tensor([0, SEQ_KV], device="cuda:0", dtype=torch.int32)
        prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_qk=HEAD_DIM,
        )
        q_torch = Q.clone().reshape(-1, NUM_QO_HEADS, HEAD_DIM).cuda()
        k_torch = K.clone().reshape(-1, NUM_KV_HEADS, HEAD_DIM).cuda()
        v_torch = V.clone().reshape(-1, NUM_KV_HEADS, HEAD_DIM).cuda()

        def run_flashinfer():
            o_torch = prefill_wrapper.run(q_torch, k_torch, v_torch)
            return o_torch

        bench(run_flashinfer, warmup=100, repeat=300, proton_name="flashinfer", debug=DEBUG)
        o_torch = run_flashinfer()
        return o_torch.cpu().numpy().reshape(BATCH, SEQ_Q, NUM_QO_HEADS, HEAD_DIM)

    with ProtonContext("blackwell_fa4", debug=DEBUG):
        print("\nRunning CuTeDSL FA4 baseline...")
        O_cutedsl = cutedsl_attn(Q, K, V)

        print("\nRunning Flash-Attention SM100 baseline...")
        O_flashattn = flashattn_sm100(Q, K, V)

        print("Running TIR Flash Attention...")
        O_tir = tir_attn(Q, K, V)

        print("\nRunning FlashInfer FA4 baseline...")
        # O_flashinfer = flashinfer(Q, K, V)
        O_flashinfer = None

    # Compare with CuTeDSL FA
    if O_cutedsl is not None:
        print("\n=== TIR vs CuTeDSL FA4 ===")
        diff_cute = np.abs(O_tir - O_cutedsl)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_cutedsl)
        valid_mask = abs_ref > atol
        rel_diff_cute = np.zeros_like(diff_cute)
        if np.any(valid_mask):
            rel_diff_cute[valid_mask] = diff_cute[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_cute) if np.any(valid_mask) else 0.0
        mismatch_mask_cute = (diff_cute > atol) & (rel_diff_cute > rtol)
        num_mismatches_cute = np.sum(mismatch_mask_cute)

        print(
            f"max_abs_err={np.max(diff_cute):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_cute}/{O_tir.size} ({100.0 * num_mismatches_cute / O_tir.size:.2f}%)"  # noqa: E501
        )

        np.testing.assert_allclose(O_tir, O_cutedsl, rtol=rtol, atol=atol)
        print("\nVerification passed!")
    else:
        print("\nCuTeDSL FA4 baseline not available, skipping comparison")

    # Compare with Flash-Attention4 SM100
    if O_flashattn is not None:
        print("\n=== TIR vs Flash-Attention4 SM100 ===")
        diff_fa = np.abs(O_tir - O_flashattn)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_flashattn)
        valid_mask = abs_ref > atol
        rel_diff_fa = np.zeros_like(diff_fa)
        if np.any(valid_mask):
            rel_diff_fa[valid_mask] = diff_fa[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_fa) if np.any(valid_mask) else 0.0
        mismatch_mask_fa = (diff_fa > atol) & (rel_diff_fa > rtol)
        num_mismatches_fa = np.sum(mismatch_mask_fa)

        print(
            f"max_abs_err={np.max(diff_fa):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_fa}/{O_tir.size} ({100.0 * num_mismatches_fa / O_tir.size:.2f}%)"  # noqa: E501
        )

        np.testing.assert_allclose(O_tir, O_flashattn, rtol=rtol, atol=atol)
        print("\nVerification vs Flash-Attention SM100 passed!")
    else:
        print("\nFlash-Attention SM100 baseline not available, skipping comparison")

    # Compare with FlashInfer FA4
    if O_flashinfer is not None:
        print("\n=== TIR vs FlashInfer FA4 ===")
        diff_flashinfer = np.abs(O_tir - O_flashinfer)
        rtol, atol = 1e-2, 1e-2

        abs_ref = np.abs(O_flashinfer)
        valid_mask = abs_ref > atol
        rel_diff_flashinfer = np.zeros_like(diff_flashinfer)
        if np.any(valid_mask):
            rel_diff_flashinfer[valid_mask] = diff_flashinfer[valid_mask] / abs_ref[valid_mask]

        max_rel_err = np.max(rel_diff_flashinfer) if np.any(valid_mask) else 0.0
        mismatch_mask_flashinfer = (diff_flashinfer > atol) & (rel_diff_flashinfer > rtol)
        num_mismatches_flashinfer = np.sum(mismatch_mask_flashinfer)

        print(
            f"max_abs_err={np.max(diff_flashinfer):.6f}, max_rel_err={max_rel_err:.6f}, "
            f"mismatches={num_mismatches_flashinfer}/{O_tir.size} ({100.0 * num_mismatches_flashinfer / O_tir.size:.2f}%)"  # noqa: E501
        )


if __name__ == "__main__":
    test_flash_attention4(8192, 32, 8, is_causal=False)
    test_flash_attention4(8192, 32, 8, is_causal=True)
    # TODO: causal attention for non-GQA kernel is still 10% slower than FA4.
    # likely due to register pressure issue
    test_flash_attention4(8192, 32, 32, is_causal=True)
    test_flash_attention4(8192, 32, 32, is_causal=False)
