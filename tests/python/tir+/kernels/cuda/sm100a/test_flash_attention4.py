import os
from functools import partial
import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.layout import TileLayout
from tvm.tirp.bench.utils import ProtonContext, bench
M_CLUSTER = 1
N_CLUSTER = 1
SM_NUMBER = 148
DEBUG_BARRIER = False

WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER

N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3

BLK_M = 128
BLK_N = 128
CORR_CHUNK = 32

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


def get_flash_attention4_kernel(batch_size, seq_len_q, seq_len_kv, num_heads, head_dim):

    BATCH_SIZE = batch_size
    SEQ_LEN_Q = seq_len_q
    SEQ_LEN_KV = seq_len_kv
    NUM_HEADS = num_heads
    HEAD_DIM = head_dim

    NUM_MMA_QK = HEAD_DIM // MMA_K
    NUM_MMA_PV = BLK_N // MMA_K
    CTA_GROUP = 1
    SWIZZLE = 3

    SMEM_SIZE_Q_BYTES = SMEM_PIPE_DEPTH_Q * BLK_M * HEAD_DIM * F16_BYTES
    SMEM_SIZE_KV_BYTES = SMEM_PIPE_DEPTH_KV * BLK_N * HEAD_DIM * F16_BYTES
    SMEM_SIZE_O_BYTES = TMEM_PIPE_DEPTH * BLK_M * HEAD_DIM * F16_BYTES
    SMEM_SIZE_SCALE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M * F32_BYTES
    SMEM_SIZE_MBAR = 35 * 8

    SMEM_SIZE = 229376
    assert (
        SMEM_SIZE <= 229376
    ), f"SMEM size {SMEM_SIZE} exceeds limit (Q:{SMEM_SIZE_Q_BYTES}, KV:{SMEM_SIZE_KV_BYTES}, O:{SMEM_SIZE_O_BYTES}, Scale:{SMEM_SIZE_SCALE}, Mbar:{SMEM_SIZE_MBAR}, Total:{SMEM_SIZE})"
    assert TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"

    def ceildiv(a, b):
        return (a + b - 1) // b

    def ptx_exp2(x):

        func_name = "tvm_builtin_ptx_exp2"
        source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
        return T.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")

    def ptx_rcp(x):

        func_name = "tvm_builtin_ptx_rcp"
        source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
        return T.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")

    def any_sync(mask, pred):
        return T.cuda.func_call(
            "any_sync",
            mask,
            pred,
            source_code=f"""
__forceinline__ __device__ int any_sync(unsigned mask, int pred) {{
  return __any_sync(mask, pred);
}}
""",
            return_type="int32",
        )

    def get_sm_scale():

        func_name = "get_sm_scale"
        source_code = f"""
__device__ __forceinline__ float {func_name}() {{
  return 1.44269504088896340736 / sqrtf({HEAD_DIM});
}}
"""
        return T.cuda.func_call(func_name, source_code=source_code, return_type="float32")

    class Barriers:

        def __init__(self, pool_allocator, pipe_depth, is_p2c):
            self.mbar = pool_allocator.alloc([pipe_depth], "uint64").buffer
            self.init_phase = 0 if is_p2c else 1
            self.pipe_depth = pipe_depth

        @T.macro
        def init(self, threads_num_wait):
            with T.thread()[0:1]:
                for i in T.serial(self.pipe_depth):
                    T.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

        @T.macro
        def wait(self, idx, phase):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)

    class BarrierWithCommit(Barriers):
        @T.macro
        def arrive(self, idx):
            if CTA_GROUP == 1:
                T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]))

    class BarrierWithArrive(Barriers):
        @T.macro
        def arrive(self, idx):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))

    class BarrierWithExpectTx(Barriers):
        @T.macro
        def arrive(self, idx, expected_bytes=None):
            if expected_bytes is not None:
                T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)
            else:
                T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))

    Q_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), (BLK_M * HEAD_DIM, HEAD_DIM, 1))),
    )
    # K/V: 3-stage pipeline
    K_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), (BLK_N * HEAD_DIM, HEAD_DIM, 1))),
    )
    V_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), (BLK_N * HEAD_DIM, HEAD_DIM, 1))),
    )
    O_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(shard=((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), (BLK_M * HEAD_DIM, HEAD_DIM, 1))),
    )

    @T.prim_func(tirp=True)
    def flash_attention4(
        Q: T.Buffer((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), "float16"),
        K: T.Buffer((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), "float16"),
        V: T.Buffer((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), "float16"),
        O: T.Buffer((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM), "float16"),
    ):

        num_q_blocks_total = T.meta_var(ceildiv(SEQ_LEN_Q, BLK_M))
        num_q_blocks_per_cta = T.meta_var(SMEM_PIPE_DEPTH_Q)
        num_q_blocks = T.meta_var(ceildiv(num_q_blocks_total, num_q_blocks_per_cta))
        
        # Persistent kernel: limit CTA count to SM number
        num_total_tasks = T.meta_var(BATCH_SIZE * NUM_HEADS * num_q_blocks)
        max_ctas = 148 
        cta_count = T.min(max_ctas, num_total_tasks)

        with T.kernel():
            bx = T.cta_id([cta_count], parent="kernel")
            wg_id = T.warpgroup_id([4], parent="cta")
            warp_id = T.warp_id([4], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            tid_in_wg = T.thread_id([128], parent="warpgroup")

            with T.cta():
                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))
                # Allocate Q buffer with alignment
                Q_smem = pool.alloc(
                    (SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16", layout=Q_layout, align=1024
                )

                # Allocate K and V buffers (they share the same offset)
                kv_offset_bytes = pool.offset
                K_smem = pool.alloc(
                    (SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16", layout=K_layout, align=1024
                )
                # V shares the same offset as K
                # pool.move_base_to(kv_offset_bytes)
                V_smem = pool.alloc(
                    (SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16", layout=V_layout, align=1024
                )

                # Allocate O buffer
                O_smem = pool.alloc(
                    (TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16", layout=O_layout, align=1024
                )
                # Allocate sScale buffer
                sScale_total_size = (
                    2 * SMEM_PIPE_DEPTH_Q * BLK_M
                )  # ACC_SCALE/ROW_SUM (shared) + ROW_MAX
                sScale = pool.alloc((sScale_total_size,), "float32", align=1024)
                tmem_addr = pool.alloc([1], "uint32")

                ACC_SCALE_BASE = 0
                ROW_SUM_BASE = 0  # Shares with ACC_SCALE

                TMEM_LD_SIZE = 16

                # Allocate phase buffers using PoolAllocator
                phase_kv = T.alloc_local([1], "int32")

                phase_q = T.alloc_local([1], "int32")

                phase_tmem = T.alloc_local([1], "int32")

                stage_kv = T.alloc_local([1], "int32")

                stage_q = T.alloc_local([1], "int32")

                stage_tmem = T.alloc_local([1], "int32")

                bar_load_q_full = T.meta_var(
                    BarrierWithExpectTx(pool, SMEM_PIPE_DEPTH_Q, True)
                )
                bar_load_q_empty = T.meta_var(
                    BarrierWithCommit(pool, SMEM_PIPE_DEPTH_Q, False)  # init_phase = 1
                )

                bar_load_k_full = T.meta_var(
                    BarrierWithExpectTx(pool, SMEM_PIPE_DEPTH_KV, True)
                )
                bar_load_k_empty = T.meta_var(
                    BarrierWithCommit(pool, SMEM_PIPE_DEPTH_KV, False)
                )

                bar_load_v_full = T.meta_var(
                    BarrierWithExpectTx(pool, SMEM_PIPE_DEPTH_KV, True)
                )
                bar_load_v_empty = T.meta_var(
                    BarrierWithCommit(pool, SMEM_PIPE_DEPTH_KV, False)
                )

                bar_p_full_o_rescaled = T.meta_var(BarrierWithArrive(pool, 2, True))

                bar_s_full = T.meta_var(BarrierWithCommit(pool, 2, True))

                bar_o_full = T.meta_var(BarrierWithCommit(pool, 2, True))

                bar_softmax_corr_full = T.meta_var(BarrierWithArrive(pool, 2, True))
                bar_softmax_corr_empty = T.meta_var(BarrierWithArrive(pool, 2, False))

                bar_corr_epi_full = T.meta_var(
                    BarrierWithArrive(pool, TMEM_PIPE_DEPTH, True)
                )
                bar_corr_epi_empty = T.meta_var(
                    BarrierWithArrive(pool, TMEM_PIPE_DEPTH, False)
                )

                bar_s0_s1_sequence = T.meta_var(BarrierWithArrive(pool, 8, True))

                bar_tmem_dealloc = T.meta_var(BarrierWithArrive(pool, 1, True))

                bar_p_empty = T.meta_var(BarrierWithCommit(pool, 2, False))


                with T.warp()[0:1]:

                    T.ptx.tcgen05.alloc(
                        T.address_of(tmem_addr[0]), n_cols=N_COLS_TMEM, cta_group=CTA_GROUP
                    )
                    T.cuda.trap_when_assert_failed(tmem_addr[0] == T.uint32(0))

                T.ptx.fence.proxy("shared")
                T.ptx.fence.mbarrier_init()
                T.cuda.cta_sync()

                tmem = T.decl_buffer(
                    (128, N_COLS_TMEM),
                    "float32",
                    scope="tmem",
                    allocated_addr=0,
                    layout=TileLayout(([128, N_COLS_TMEM], [(1, "TLane"), (1, "TCol")])),
                )
                tmem_as_f16 = T.decl_buffer(
                    (128, N_COLS_TMEM * 2),
                    "float16",
                    scope="tmem",
                    allocated_addr=0,
                    layout=TileLayout(([128, N_COLS_TMEM * 2], [(1, "TLane"), (1, "TCol")])),
                )

                task_idx = T.alloc_local([1], "int32")
                task_idx[0] = bx  # Start from CTA ID
                
                while task_idx[0] < num_total_tasks:
                    T.cuda.cta_sync()

                    stage_q[0] = 0
                    stage_kv[0] = 0
                    stage_tmem[0] = 0
                    phase_q[0] = 0
                    phase_kv[0] = 0
                    phase_tmem[0] = 0

                    with T.thread()[0:1]:
                        bar_load_q_full.init(1);      bar_load_q_empty.init(1)
                        bar_load_k_full.init(1);      bar_load_k_empty.init(1)
                        bar_load_v_full.init(1);      bar_load_v_empty.init(1)
                        bar_p_full_o_rescaled.init(256)
                        bar_s_full.init(1);           bar_o_full.init(1)
                        bar_softmax_corr_full.init(128);  bar_softmax_corr_empty.init(128)
                        bar_corr_epi_full.init(1);    bar_corr_epi_empty.init(32)
                        bar_s0_s1_sequence.init(32);  bar_tmem_dealloc.init(1)
                        bar_p_empty.init(1)

                    T.ptx.fence.proxy("shared")
                    T.ptx.fence.mbarrier_init()
                    T.cuda.cta_sync()

                    # Decode task index into batch/head/q_block
                    batch_idx = T.meta_var(task_idx[0] // (num_q_blocks * NUM_HEADS))
                    head_idx = T.meta_var((task_idx[0] % (num_q_blocks * NUM_HEADS)) // num_q_blocks)
                    m_block_idx = T.meta_var(task_idx[0] % num_q_blocks)
                    m_start = T.meta_var(m_block_idx * BLK_M * SMEM_PIPE_DEPTH_Q)
                    num_kv_blocks = T.meta_var(ceildiv(SEQ_LEN_KV, BLK_N))
                    sm_scale = get_sm_scale()
                    tmem_s_base = 0
                    tmem_o_base = 256
                    tmem_p_base = 0
                    tmem_s_to_p_offset = 0  # 64 for head_dim=64
                    tmem_offset = 128

                    with T.cta():
                        T.block_attr({"tirp.scope_partition": True})
    
                        with T.warpgroup()[wg_id == 3]:
                            T.ptx.setmaxnreg(False, 48)
                            if warp_id == 1:
                                with T.thread()[T.ptx.elect_sync()]:
                                    for i_q in T.serial(SMEM_PIPE_DEPTH_Q):
                                        # Use phase=0 for Q prefetch (not tied to phase_q which is for Softmax sync)
                                        bar_load_q_empty.wait(stage_q[0], 0)
                                        # stage_q[0] ->  0 -> 1 -> 0 -> 1 -> ...
                                        # NOTE: phase_q is NOT used for Q prefetch, always use phase=0

                                        tma_copy_q = T.meta_var(
                                            {
                                                "dispatch": "tma",
                                                "mbar": bar_load_q_full.mbar.ptr_to([stage_q[0]]),
                                                "cta_group": CTA_GROUP,
                                            }
                                        )
                                        Tp.copy_async(
                                            Q_smem[stage_q[0], :, :],
                                            Q[
                                                batch_idx,
                                                head_idx,
                                                m_start + i_q * BLK_M : m_start + (i_q + 1) * BLK_M,
                                                0:HEAD_DIM,
                                            ],
                                            **tma_copy_q,
                                        )

                                        bar_load_q_full.arrive(
                                            stage_q[0], CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES
                                        )  # ar(0,x)
                                        stage_q[0] = stage_q[0] + 1
                                        if stage_q[0] == SMEM_PIPE_DEPTH_Q:
                                            stage_q[0] = 0
                                            # NOTE: Do NOT flip phase_q here! Q prefetch is one-time,
                                            # phase_q is only for Softmax<->Correction sync
    
                                    for i_kv in T.serial(num_kv_blocks):
                                        bar_load_k_empty.wait(stage_kv[0], phase_kv[0])
                                        tma_copy_k = T.meta_var(
                                            {
                                                "dispatch": "tma",
                                                "mbar": bar_load_k_full.mbar.ptr_to([stage_kv[0]]),
                                                "cta_group": CTA_GROUP,
                                            }
                                        )
                                        Tp.copy_async(
                                            K_smem[stage_kv[0], :, :],
                                            K[
                                                batch_idx,
                                                head_idx,
                                                i_kv * BLK_N : (i_kv + 1) * BLK_N,
                                                0:HEAD_DIM,
                                            ],
                                            **tma_copy_k,
                                        )
    
                                        bar_load_k_full.arrive(
                                            stage_kv[0], CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES
                                        )
    
                                        bar_load_v_empty.wait(stage_kv[0], phase_kv[0])
                                        tma_copy_v = T.meta_var(
                                            {
                                                "dispatch": "tma",
                                                "mbar": bar_load_v_full.mbar.ptr_to([stage_kv[0]]),
                                                "cta_group": CTA_GROUP,
                                            }
                                        )
                                        Tp.copy_async(
                                            V_smem[stage_kv[0], :, :],
                                            V[
                                                batch_idx,
                                                head_idx,
                                                i_kv * BLK_N : (i_kv + 1) * BLK_N,
                                                0:HEAD_DIM,
                                            ],
                                            **tma_copy_v,
                                        )
    
                                        bar_load_v_full.arrive(
                                            stage_kv[0], CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES
                                        )
                                        stage_kv[0] = stage_kv[0] + 1
                                        if stage_kv[0] == SMEM_PIPE_DEPTH_KV:
                                            stage_kv[0] = 0
                                            phase_kv[0] ^= 1
    
                            elif warp_id == 2:
                                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):  # stage=0,1
                                    bar_corr_epi_full.wait(i_q, phase_tmem[0])
                                    # TMA O store
                                    with T.thread()[T.ptx.elect_sync()]:
                                        m_start_global = m_start + i_q * BLK_M
                                        if m_start_global < SEQ_LEN_Q:
                                            Tp.copy_async(
                                                O[
                                                    batch_idx,
                                                    head_idx,
                                                    m_start_global : m_start_global + BLK_M,
                                                    0:HEAD_DIM,
                                                ],
                                                O_smem[i_q, :, :],
                                                dispatch="tma",
                                            )
                                    T.ptx.cp_async.bulk.commit_group()
                                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                    T.ptx.cp_async.bulk.wait_group(1 - i_q)
                                    bar_corr_epi_empty.arrive(i_q)
                                phase_tmem[0] ^= 1
    
                            elif warp_id == 0:

                                descQ = T.local_cell("uint64")
                                descK = T.local_cell("uint64")
                                descI = T.local_cell("uint32")
                                T.ptx.tcgen05.encode_instr_descriptor(
                                    T.address_of(descI),
                                    d_type_qk,
                                    a_type_qk,
                                    b_type_qk,
                                    MMA_M,
                                    MMA_N,
                                    MMA_K,
                                    False,
                                    False,
                                    CTA_GROUP,
                                )
    
                                descV = T.local_cell("uint64")
                                descI_pv = T.local_cell("uint32")
                                T.ptx.tcgen05.encode_instr_descriptor(
                                    T.address_of(descI_pv),
                                    d_type_pv,
                                    a_type_pv,
                                    b_type_pv,
                                    MMA_M,
                                    MMA_N,
                                    MMA_K,
                                    False,
                                    True,
                                    CTA_GROUP,
                                )
    
                                if T.ptx.elect_sync():
                                    T.ptx.tcgen05.fence.after_thread_sync()

                                    @T.macro
                                    def gemm_qk(q_stage, stage, tmem_col_s, zero_init):
                                        if NUM_MMA_QK > 0:
                                            T.ptx.tcgen05.encode_matrix_descriptor(
                                                T.address_of(descQ),
                                                Q_smem.ptr_to([q_stage, 0, 0]),
                                                ldo=1,
                                                sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                swizzle=SWIZZLE,
                                            )
                                            T.ptx.tcgen05.encode_matrix_descriptor(
                                                T.address_of(descK),
                                                K_smem.ptr_to([stage, 0, 0]),
                                                ldo=1,
                                                sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                swizzle=SWIZZLE,
                                            )
                                            T.ptx.tcgen05.mma(
                                                d_type_qk,
                                                a_type_qk,
                                                b_type_qk,
                                                tmem_col_s,
                                                descQ,
                                                descK,
                                                descI,
                                                False,
                                                CTA_GROUP,
                                                zero_init,
                                            )
                                            if NUM_MMA_QK > 1:
                                                for ki in T.serial(NUM_MMA_QK - 1):
                                                    base = (ki + 1) * MMA_K
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descQ),
                                                        Q_smem.ptr_to([q_stage, 0, base]),
                                                        ldo=1,
                                                        sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                        swizzle=SWIZZLE,
                                                    )
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descK),
                                                        K_smem.ptr_to([stage, 0, base]),
                                                        ldo=1,
                                                        sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                        swizzle=SWIZZLE,
                                                    )
                                                    T.ptx.tcgen05.mma(
                                                        d_type_qk,
                                                        a_type_qk,
                                                        b_type_qk,
                                                        tmem_col_s,
                                                        descQ,
                                                        descK,
                                                        descI,
                                                        False,
                                                        CTA_GROUP,
                                                        True,
                                                    )
    
                                    @T.macro
                                    def gemm_pv(
                                        q_stage, stage, tmem_col_o, tmem_col_p, should_accumulate
                                    ):
                                        if NUM_MMA_PV > 0:
                                            T.ptx.tcgen05.encode_matrix_descriptor(
                                                T.address_of(descV),
                                                V_smem.ptr_to([stage, 0, 0]),
                                                ldo=8 * F16_BYTES // F128_BYTES,
                                                sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                swizzle=SWIZZLE,
                                            )
                                            T.ptx.tcgen05.mma(
                                                "float32",
                                                "float16",
                                                "float16",
                                                tmem_col_o,
                                                tmem_col_p,
                                                descV,
                                                descI_pv,
                                                True,
                                                CTA_GROUP,
                                                should_accumulate,
                                            )
                                            if NUM_MMA_PV > 1:
                                                for ki in T.serial(NUM_MMA_PV - 1):
                                                    base_v = (ki + 1) * MMA_K
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descV),
                                                        V_smem.ptr_to([stage, base_v, 0]),
                                                        ldo=8 * F16_BYTES // F128_BYTES,
                                                        sdo=8 * HEAD_DIM * F16_BYTES // F128_BYTES,
                                                        swizzle=SWIZZLE,
                                                    )
                                                    T.ptx.tcgen05.mma(
                                                        "float32",
                                                        "float16",
                                                        "float16",
                                                        tmem_col_o,
                                                        tmem_col_p + (ki + 1) * MMA_K // 2,
                                                        descV,
                                                        descI_pv,
                                                        True,
                                                        CTA_GROUP,
                                                        True,
                                                    )
    
                                    for i_kv in T.serial(num_kv_blocks):
                                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                            q_stage = i_q
                                            tmem_col_s = tmem_s_base + i_q * tmem_offset
                                            tmem_col_p = tmem_s_base + i_q * tmem_offset
                                            tmem_col_o = tmem_o_base + i_q * tmem_offset
                                            # Only wait for Q on first KV block (Q is prefetched once)
                                            if i_kv == 0:
                                                bar_load_q_full.wait(q_stage, 0)
                                            if i_q == 0:
                                                # for 2 q, confirm k is loaded
                                                bar_load_k_full.wait(stage_kv[0], phase_kv[0])
                                            T.ptx.tcgen05.fence.after_thread_sync()
                                            gemm_qk(q_stage, stage_kv[0], tmem_col_s, False)
                                            bar_s_full.arrive(i_q)
                                            if i_q == 1:
                                                # finish twice qk mma
                                                bar_load_k_empty.arrive(stage_kv[0])
                                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                            q_stage = i_q
                                            tmem_col_s = tmem_s_base + i_q * tmem_offset
                                            tmem_col_p = tmem_p_base + i_q * tmem_offset
                                            tmem_col_o = tmem_o_base + i_q * tmem_offset
                                            if i_q == 0:
                                                # wait for v is loaded
                                                bar_load_v_full.wait(stage_kv[0], phase_kv[0])
                                            T.ptx.tcgen05.fence.after_thread_sync()
                                            # wait for o_full to be ready
                                            bar_p_full_o_rescaled.wait(i_q, phase_tmem[0])
                                            if i_kv == 0:
                                                gemm_pv(q_stage, stage_kv[0], tmem_col_o, tmem_col_p, False)
                                            else:
                                                gemm_pv(q_stage, stage_kv[0], tmem_col_o, tmem_col_p, True)
                                            if i_q == 1:
                                                # finish twice pv mma
                                                bar_load_v_empty.arrive(stage_kv[0])
                                            bar_p_empty.arrive(i_q)
                                            if i_kv == num_kv_blocks - 1:
                                                bar_o_full.arrive(i_q)
                                        stage_kv[0] = stage_kv[0] + 1
                                        if stage_kv[0] == SMEM_PIPE_DEPTH_KV:
                                            stage_kv[0] = 0
                                            phase_kv[0] ^= 1
                                        phase_tmem[0] ^= 1
    
                                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                        bar_load_q_empty.arrive(i_q)
    
                        with T.warpgroup()[0:2]:
                            # here phase_q and stage_q represent phase_tmem and stage_tmem

                            T.ptx.setmaxnreg(True, 200)

                            scale_log2 = get_sm_scale()
                            # TODO: set to 8.0 to enable conditional rescale
                            rescale_threshold = T.float32(-8.0)
    
                            row_max = T.alloc_local([1], "float32")
                            row_sum = T.alloc_local([1], "float32")
                            row_max[0] = T.float32(-1e5)
                            row_sum[0] = T.float32(0.0)
    
                            s_chunk_buf = T.alloc_local([CORR_CHUNK], "float32")
                            s_chunk = s_chunk_buf.view(
                                128,
                                CORR_CHUNK,
                                layout=TileLayout(([128, CORR_CHUNK], [(1, "tid_in_wg"), (1, "m")])),
                            )
    
                            p_chunk_buf = T.alloc_local([CORR_CHUNK], "float16")
                            p_chunk = p_chunk_buf.view(
                                128,
                                CORR_CHUNK,
                                layout=TileLayout(([128, CORR_CHUNK], [(1, "tid_in_wg"), (1, "m")])),
                            )
    
                            for i_kv in T.serial(num_kv_blocks):
                                is_first = i_kv == 0

                                tmem_col_s = tmem_s_base + wg_id * tmem_offset
                                # P shares the same region as S, starting at the middle (offset 64)
                                tmem_col_p = tmem_s_base + wg_id * tmem_offset
                                tmem_col_o = tmem_o_base + wg_id * tmem_offset

                                bar_s_full.wait(wg_id, phase_q[0])
                                T.ptx.tcgen05.fence.after_thread_sync()
    
                                tile_max = T.alloc_local([1], "float32")
                                tile_max[0] = T.float32(-1e5)
    
                                for chunk_idx in T.unroll(BLK_N // CORR_CHUNK):
                                    Tp.copy(
                                        s_chunk[:, 0:CORR_CHUNK],
                                        tmem[
                                            :,
                                            tmem_col_s
                                            + chunk_idx * CORR_CHUNK : tmem_col_s
                                            + chunk_idx * CORR_CHUNK
                                            + CORR_CHUNK,
                                        ],
                                    )
                                    for col in T.serial(CORR_CHUNK):
                                        tile_max[0] = T.max(tile_max[0], s_chunk_buf[col])
    
                                row_max_old = row_max[0]
                                row_max_new = T.alloc_local([1], "float32")
                                acc_scale = T.alloc_local([1], "float32")
                                acc_scale_ = T.alloc_local([1], "float32")  # For slack check
    
                                if is_first:
                                    row_max_new[0] = tile_max[0]
                                    acc_scale[0] = T.float32(1.0)
                                else:
                                    row_max_new[0] = T.max(row_max_old, tile_max[0])
                                    acc_scale_[0] = (row_max_old - row_max_new[0]) * scale_log2

                                    # if the difference is too small, don't rescale
                                    if acc_scale_[0] >= -rescale_threshold:
                                        row_max_new[0] = row_max_old
                                        acc_scale[0] = T.float32(1.0)
                                    else:
                                        acc_scale[0] = ptx_exp2(acc_scale_[0])
    
                                # row_max is the max value of the tile
                                # and row_max_scaled is the max value of the tile after scaled
                                # scale_log2 is the log2 of the scale factor
                                row_max[0] = row_max_new[0]
                                row_max_scaled = row_max_new[0] * scale_log2
    
                                bar_softmax_corr_empty.wait(wg_id, phase_q[0])
                                if tid_in_wg < BLK_M and not is_first:
                                    sScale_idx = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                                    sScale[sScale_idx] = acc_scale[0]
                                bar_softmax_corr_full.arrive(wg_id)
    
                                tile_sum = T.alloc_local([1], "float32")
                                tile_sum[0] = T.float32(0.0)
                                for chunk_idx in T.unroll(4):
                                    Tp.copy(
                                        s_chunk[:, 0:CORR_CHUNK],
                                        tmem[
                                            :,
                                            tmem_col_s
                                            + chunk_idx * CORR_CHUNK : tmem_col_s
                                            + chunk_idx * CORR_CHUNK
                                            + CORR_CHUNK,
                                        ],
                                    )
                                    for col in T.serial(CORR_CHUNK):
                                        s_chunk_buf[col] = (
                                            s_chunk_buf[col] * scale_log2 - row_max_scaled
                                        )
                                        s_chunk_buf[col] = ptx_exp2(s_chunk_buf[col])
                                        tile_sum[0] = tile_sum[0] + s_chunk_buf[col]
    
                                    with T.thread():
                                        Tp.cast(p_chunk_buf, s_chunk_buf)
                                    if chunk_idx == 0:
                                        bar_p_empty.wait(wg_id, phase_q[0])
                                    Tp.copy(
                                        tmem_as_f16[
                                            :,
                                            tmem_col_p * 2
                                            + chunk_idx * CORR_CHUNK : tmem_col_p * 2
                                            + chunk_idx * CORR_CHUNK
                                            + CORR_CHUNK,
                                        ],
                                        p_chunk[:, 0:CORR_CHUNK],
                                    )
    
                                T.ptx.tcgen05.fence.after_thread_sync()
                                bar_p_full_o_rescaled.arrive(wg_id)
                                phase_q[0] ^= 1
    
                                if is_first:
                                    row_sum[0] = tile_sum[0]
                                else:
                                    row_sum[0] = row_sum[0] * acc_scale[0] + tile_sum[0]
                            bar_softmax_corr_empty.wait(wg_id, phase_q[0])
                            if tid_in_wg < BLK_M:
                                sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                            bar_softmax_corr_full.arrive(wg_id)
                            phase_q[0] ^= 1
    
                        with T.warpgroup()[wg_id == 2]:
                            T.ptx.setmaxnreg(False, 64)

                            for i_q in T.unroll(2): # kv block 0 no need to rescale
                                bar_p_full_o_rescaled.arrive(i_q)
                                bar_softmax_corr_full.wait(i_q, phase_q[0])
                                bar_softmax_corr_empty.arrive(i_q)
                            phase_q[0] ^= 1
                            for i_kv in T.serial(num_kv_blocks - 1):
                                for i_q in T.serial(2):
    
                                    bar_softmax_corr_full.wait(i_q, phase_q[0])
    
                                    acc_scale = T.alloc_local([1], "float32")
                                    should_rescale = T.alloc_local([1], "int32")
    
                                    if tid_in_wg < BLK_M:
                                        acc_scale[0] = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]
                                        should_rescale[0] = T.if_then_else(
                                            acc_scale[0] < T.float32(1.0), 1, 0
                                        )
                                        should_rescale[0] = 1
                                    else:
                                        should_rescale[0] = 0
    
                                    any_needs_rescale = any_sync(0xFFFFFFFF, should_rescale[0])
    
                                    if any_needs_rescale != 0:
                                        if tid_in_wg < BLK_M:
                                            tmem_col_o_stage = tmem_o_base + i_q * tmem_offset
                                            RESCALE_TILE = 16
    
                                            o_row_buf = T.alloc_buffer((16,), "float32", scope="local")
                                            o_row_wg = o_row_buf.view(
                                                128,
                                                16,
                                                layout=TileLayout(
                                                    ([128, 16], [(1, "tid_in_wg"), (1, "m")])
                                                ),
                                            )
    
                                            for d_tile in T.serial(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                                d_start = d_tile * RESCALE_TILE
                                                if d_start < HEAD_DIM:
    
                                                    Tp.copy(
                                                        o_row_wg,
                                                        tmem[
                                                            :,
                                                            tmem_col_o_stage
                                                            + d_start : tmem_col_o_stage
                                                            + d_start
                                                            + 16,
                                                        ],
                                                    )
                                                    if should_rescale[0] != 0:
                                                        for d in T.serial(8):
                                                            d_idx = d * 2
                                                            o_row_buf[d_idx] *= acc_scale[0]
                                                            o_row_buf[d_idx + 1] *= acc_scale[0]
    
                                                    Tp.copy(
                                                        tmem[
                                                            :,
                                                            tmem_col_o_stage
                                                            + d_start : tmem_col_o_stage
                                                            + d_start
                                                            + 16,
                                                        ],
                                                        o_row_wg[:, 0:16],
                                                    )
    
                                                T.ptx.tcgen05.fence.after_thread_sync()
    
                                    bar_p_full_o_rescaled.arrive(i_q)
                                    bar_softmax_corr_empty.arrive(i_q)
    
                                # flip epi producer phase
                                phase_q[0] ^= 1
    
                            for i_q in T.serial(2):
                                # wait O_full
                                bar_softmax_corr_full.wait(i_q, phase_q[0])
                                bar_o_full.wait(i_q, phase_tmem[0])
                                
                                # wait epi_empty (from wg3)
                                bar_corr_epi_empty.wait(i_q, phase_tmem[0])
    
                                if tid_in_wg < BLK_M:
                                    row_sum = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                                    bar_softmax_corr_full.arrive(i_q)
    
                                    acc_O_mn_row_is_zero_or_nan = tvm.tir.any(
                                        row_sum == T.float32(0.0), row_sum != row_sum
                                    )
                                    norm_scale = ptx_rcp(
                                        T.if_then_else(
                                            acc_O_mn_row_is_zero_or_nan, T.float32(1.0), row_sum
                                        )
                                    )
    
                                    tmem_col_o_stage = tmem_o_base + i_q * tmem_offset
                                    NORM_TILE = 32
    
                                    o_row_f32_buf = T.alloc_buffer((32,), "float32", scope="local")
                                    o_row_f32_wg = o_row_f32_buf.view(
                                        128,
                                        32,
                                        layout=TileLayout(([128, 32], [(1, "tid_in_wg"), (1, "m")])),
                                    )
                                    o_row_f16 = T.alloc_local([NORM_TILE], "float16")
    
                                    for d_tile in T.serial(ceildiv(HEAD_DIM, NORM_TILE)):
                                        d_start = d_tile * NORM_TILE
                                        if d_start < HEAD_DIM:
    
                                            Tp.copy(
                                                o_row_f32_wg[:, 0:32],
                                                tmem[
                                                    :,
                                                    tmem_col_o_stage
                                                    + d_start : tmem_col_o_stage
                                                    + d_start
                                                    + 32,
                                                ],
                                            )
    
                                            T.ptx.tcgen05.fence.after_thread_sync()
    
                                            for d in T.serial(16):
                                                d_idx = d * 2
                                                o_row_f32_buf[d_idx] *= norm_scale
                                                o_row_f32_buf[d_idx + 1] *= norm_scale
    
                                            with T.thread():
                                                Tp.cast(o_row_f16[0:32], o_row_f32_buf[0:32])
    
                                            for d in T.serial(32):
                                                if d_start + d < HEAD_DIM:
                                                    O_smem[i_q, tid_in_wg, d_start + d] = o_row_f16[d]
    
                                    T.ptx.fence.proxy("shared")
    
                                # arrive epi_full
                                T.cuda.warpgroup_sync(14)
                                if tid_in_wg == 0:
                                    bar_corr_epi_full.arrive(i_q)
                            phase_tmem[0] ^= 1
                            phase_q[0] ^= 1
    
                    task_idx[0] = task_idx[0] + cta_count
                
                # Deallocate TMEM after all tasks complete
                with T.warp()[0:1]:
                    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    T.ptx.tcgen05.dealloc(0, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)

                T.cuda.cta_sync()

    return flash_attention4


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_heads, head_dim):

    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_heads, head_dim), dtype=torch.float16)


    return Q, K, V, O


def test_flash_attention4():
    BATCH = 1
    SEQ_Q = 8192
    SEQ_KV = 8192
    HEADS = 12
    HEAD_DIM = 64


    Q, K, V, _ = prepare_data(BATCH, SEQ_Q, SEQ_KV, HEADS, HEAD_DIM)

    def flops(ms):
        return 4 * BATCH * HEADS * SEQ_Q * SEQ_KV * HEAD_DIM / (ms * 1e-3)

    def get_source(func):
        target = tvm.target.Target("cuda")
        mod = tvm.IRModule({"main": func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirp")
        print(mod.mod.imports[0].inspect_source(), flush=True)
        return mod

    def tir_attn(Q, K, V):
        # (B, S, H, D) -> (B, H, S, D) - keep 4D for kernel
        Q_tir = Q.permute(0, 2, 1, 3).contiguous().cuda()  # (B, H, S, D)
        K_tir = K.permute(0, 2, 1, 3).contiguous().cuda()  # (B, H, S, D)
        V_tir = V.permute(0, 2, 1, 3).contiguous().cuda()  # (B, H, S, D)
        O_tir = torch.zeros_like(Q_tir)  # (B, H, S, D)

        prim_func = get_flash_attention4_kernel(BATCH, SEQ_Q, SEQ_KV, HEADS, HEAD_DIM)
        mod = get_source(prim_func)

        dev = tvm.cuda(0)
        Q_tvm = tvm.runtime.tensor(Q_tir.cpu().numpy(), device=dev)
        K_tvm = tvm.runtime.tensor(K_tir.cpu().numpy(), device=dev)
        V_tvm = tvm.runtime.tensor(V_tir.cpu().numpy(), device=dev)
        O_tvm = tvm.runtime.tensor(O_tir.cpu().numpy(), device=dev)

        func = lambda: mod(Q_tvm, K_tvm, V_tvm, O_tvm)
        ms = bench(func, warmup=10, repeat=30, proton_name="tir_attn")
        print(f"TIR flops: {flops(ms) / 1e12:.3f} TFLOPS, time: {ms:.3f} ms")

        # Convert back to (B, S, H, D) for comparison
        O_res = np.transpose(O_tvm.numpy(), (0, 2, 1, 3))  # (B, H, S, D) -> (B, S, H, D)
        return O_res


    def pytorch_attn(Q, K, V):
        Q_pt = Q.permute(0, 2, 1, 3).cuda()
        K_pt = K.permute(0, 2, 1, 3).cuda()
        V_pt = V.permute(0, 2, 1, 3).cuda()
        func = lambda: torch.nn.functional.scaled_dot_product_attention(Q_pt, K_pt, V_pt, is_causal=False)
        ms = bench(func, warmup=10, repeat=30, proton_name="pt_attn")
        print(f"PyTorch flops: {flops(ms) / 1e12:.3f} TFLOPS, time: {ms:.3f} ms")
        O_pt = func()
        O_res = np.transpose(O_pt.cpu().numpy(), (0, 2, 1, 3))  # (B, H, S, D) -> (B, S, H, D)
        return O_res

    try:
        with ProtonContext("flash_attention4", debug=False):
            O_tir = tir_attn(Q, K, V)
            O_pt = pytorch_attn(Q, K, V)
    except RuntimeError as e:
        if "cuptiSubscribe" in str(e) or "cupti" in str(e).lower():
            print("Warning: Proton profiling failed (likely due to compute-sanitizer conflict), running without profiling")
            O_tir = tir_attn(Q, K, V)
            O_pt = pytorch_attn(Q, K, V)
        else:
            raise

    # Compare
    diff = np.abs(O_tir - O_pt)
    rel_diff = np.abs(diff / (np.abs(O_pt) + 1e-8))
    rtol, atol = 1e-2, 1e-2
    mismatch_mask = (diff > atol) & (rel_diff > rtol)
    num_mismatches = np.sum(mismatch_mask)

    print(f"\nComparison: max_abs_err={np.max(diff):.6f}, max_rel_err={np.max(rel_diff):.6f}, "
          f"mismatches={num_mismatches}/{O_tir.size} ({100.0*num_mismatches/O_tir.size:.2f}%)")

    np.testing.assert_allclose(O_tir, O_pt, rtol=rtol, atol=atol)
    print("Verification passed!")


if __name__ == "__main__":
    test_flash_attention4()