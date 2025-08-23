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
import flashinfer
import numpy as np
import pytest
import torch

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
from tvm.tirp.bench.utils import ProtonContext, bench


def ceildiv(a, b):
    return (a + b - 1) // b


def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1


# Paged kv-cache config
KV_LAYOUT = "HND"

# HW config
SM_COUNT = 148

# Other
F16_BYTE = 2
F32_BYTE = 4

MAX_TOTAL_NUM_WORKERS = 65536
MAX_NUM_KV_SPLITS = 4 * SM_COUNT * 2 * (128 + 16)

DEBUG_PRINT = False
DEBUG_BX_EXPECTED = 2


def perpare_data(batch_size, qo_heads, kv_heads, seq_len, head_dim, page_size, max_page_num):
    PAGE_SIZE = page_size
    MAX_PAGE_NUM = max_page_num
    import torch

    page_last_len = PAGE_SIZE if seq_len % PAGE_SIZE == 0 else seq_len % PAGE_SIZE
    page_num = ceildiv(seq_len, PAGE_SIZE)
    total_page_num = page_num * batch_size
    assert total_page_num <= MAX_PAGE_NUM

    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32).int()
    for i in range(batch_size + 1):
        kv_indptr[i] = i * page_num
    kv_last_page_len = torch.empty(batch_size, dtype=torch.int32).int()
    for i in range(batch_size):
        kv_last_page_len[i] = page_last_len
    kv_indices = torch.arange(MAX_PAGE_NUM, dtype=torch.int32).int()
    kv_indices = kv_indices[torch.randperm(MAX_PAGE_NUM)]
    kv_indices = kv_indices[:total_page_num]

    q = torch.randn([batch_size, qo_heads, head_dim]).half()
    kv_data = torch.randn([MAX_PAGE_NUM, 2, kv_heads, PAGE_SIZE, head_dim]).half()

    # q = torch.ones([batch_size, qo_heads, head_dim]).half()
    # kv_data = torch.ones([MAX_PAGE_NUM, 2, kv_heads, PAGE_SIZE, head_dim]).half()

    # generate q, k, if head_dim is odd, then odd 1, even 0
    # q = torch.zeros([batch_size, qo_heads, head_dim]).half()
    # kv_data = torch.zeros([MAX_PAGE_NUM, 2, kv_heads, PAGE_SIZE, head_dim]).half()
    # q[:, :, ::2] = 1
    # kv_data[:, :, :, ::2, ::2] = 1

    return q, kv_data, kv_indptr, kv_last_page_len, kv_indices


# void flashinfer::PersistentKernelTemplate<
#     flashinfer::BlockBatchPagedAttentionPersistent<
#         flashinfer::KernelTraits<(flashinfer::MaskMode)0, 128u, 2u, 4u, 8u, 8u, 4u, 1u,
#                                  (flashinfer::PosEncodingMode)0, __half, __half, __half, float, int,
#                                  StandardAttention<false> >,
#         PersistentParams>,
#     flashinfer::BlockBatchPagedAttentionPersistent<
#         flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u,
#                                  (flashinfer::PosEncodingMode)0, __half, __half, __half, float, int,
#                                  StandardAttention<false> >,
#         PersistentParams>,
#     flashinfer::BlockBatchReductionPersistent<
#         flashinfer::StateReductionKernelTraits<128u, 4u, 128u, __half, __half, int> > >(
#     flashinfer::BlockBatchPagedAttentionPersistent<
#         flashinfer::KernelTraits<(flashinfer::MaskMode)0, 128u, 2u, 4u, 8u, 8u, 4u, 1u,
#                                  (flashinfer::PosEncodingMode)0, __half, __half, __half, float, int,
#                                  StandardAttention<false> >,
#         PersistentParams>::Params,
#     flashinfer::BlockBatchPagedAttentionPersistent<
#         flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u,
#                                  (flashinfer::PosEncodingMode)0, __half, __half, __half, float, int,
#                                  StandardAttention<false> >,
#         PersistentParams>::Params)

# CTA_TILE_Q_1: 128
# CTA_TILE_Q_2: 16
# HEAD_DIM_QK: 128
# HEAD_DIM_VO: 128
# MASK_MODE: 0
# >>>>>
# NUM_WARPS_Q_1: 4
# NUM_WARPS_KV_1: 1
# NUM_MMA_Q_1: 2
# NUM_MMA_KV_1: 4
# NUM_MMA_D_QK: 8
# NUM_MMA_D_VO: 8
# NUM_MMA_KV_1 * KTraits1::KV_THR_LAYOUT_COL / 2 / KTraits1::NUM_WARPS_Q_1: 4
# KernelTraits1::CTA_TILE_KV: 64
# >>>>>
# NUM_WARPS_Q_2: 1
# NUM_WARPS_KV_2: 4
# NUM_MMA_Q_2: 1
# NUM_MMA_KV_2: 2
# NUM_MMA_D_QK: 8
# NUM_MMA_D_VO: 8
# NUM_MMA_KV_2 * KTraits2::KV_THR_LAYOUT_COL / 2 / KTraits2::NUM_WARPS_Q_2: 8
# KernelTraits2::CTA_TILE_KV: 128
# NUM_THREADS: 128
# smem_size: 69632
# num_blks_x: 1
# num_blks_y: 296
# ReductionKTraits::SMEM_SIZE: 8704
# ReductionKTraits::NUM_THREADS: 128
# ReductionKTraits::NUM_WARPS: 4
# ReductionKTraits::NUM_SMEM_STAGES: 4
# ReductionKTraits::bdx: 16
# ReductionKTraits::bdy: 2
# ReductionKTraits::vec_size: 8
# ReductionKTraits::head_dim: 128


def upcast_size(dtype):
    return 128 // tvm.DataType(dtype).bits


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


def ptx_log2(x):
    func_name = "tvm_builtin_ptx_log2"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
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


def half_to_float(x):
    func_name = "tvm_builtin_half_to_float"
    source_code = f"""
__device__ __forceinline__ float {func_name}(half x) {{
  return __half2float(x);
}}
"""
    return T.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def fdivdef(x, y):
    func_name = "tvm_builtin_fdivdef"
    source_code = f"""
__device__ __forceinline__ float {func_name}(float x, float y) {{
  return __fdividef(x, y);
}}
"""
    return T.cuda.func_call(func_name, x, y, source_code=source_code, return_type="float32")


@T.macro
def cast_load(v, vec_len, buf, *indices):
    with T.thread():
        v_tmp = T.alloc_local([vec_len], buf.dtype, layout="default")
        for i in T.vectorized(vec_len):
            buffer_load = T.meta_var(T.BufferLoad(buf, indices[:-1] + (indices[-1] + i,)))
            v_tmp[i] = buffer_load
        Tp.cast(v[:], v_tmp[:])


@T.macro
def cast_store(v, vec_len, buf, *indices):
    with T.thread():
        v_tmp = T.alloc_local([vec_len], buf.dtype, layout="default")
        Tp.cast(v_tmp[:], v[:])
        for i in T.vectorized(vec_len):
            T.buffer_store(buf, v_tmp[i], indices[:-1] + (indices[-1] + i,))


def get_batch_attention_kernel(qo_heads, kv_heads, head_dim, page_size):
    QO_HEADS = qo_heads
    KV_HEADS = kv_heads
    HEAD_DIM = head_dim
    PAGE_SIZE = page_size
    GQA_GROUP_SIZE = QO_HEADS // KV_HEADS
    INF = 5e4

    def int_var(val=None):
        buf = T.alloc_local([1], "int32", layout="default")
        if val is not None:
            T.buffer_store(buf, val, 0)
        return buf

    def float_var(val=None):
        buf = T.alloc_local([1], "float32", layout="default")
        if val is not None:
            T.buffer_store(buf, val, 0)
        return buf

    def ceildiv(a, b):
        return (a + b - 1) // b

    def size_of(dtype):
        return tvm.DataType(dtype).bits // 8

    def craft_batch_attention_kernel():
        NUM_MMA_Q = 1
        NUM_MMA_KV = 2
        NUM_MMA_D_QK = head_dim // 16
        NUM_MMA_D_VO = head_dim // 16
        UPCAST_STRIDE_Q = HEAD_DIM // upcast_size("float16")
        UPCAST_STRIDE_K = HEAD_DIM // upcast_size("float16")
        UPCAST_STRIDE_V = HEAD_DIM // upcast_size("float16")
        UPCAST_STRIDE_O = HEAD_DIM // upcast_size("float16")
        NUM_WARPS_Q = 1
        NUM_WARPS_KV = 4
        NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV
        CTA_TILE_Q = 16
        CTA_TILE_KV = 128
        KV_THR_LAYOUT_COL = 8
        KV_THR_LAYOUT_ROW = 4
        SMEM_SIZE = 69632
        NUM_STAGES = 1
        assert NUM_WARPS == 4  # total warps in a cta
        assert NUM_MMA_KV * 4 % NUM_WARPS_Q == 0
        assert (
            NUM_MMA_KV * KV_THR_LAYOUT_COL // 2 // NUM_WARPS_Q
            == CTA_TILE_KV // 4 // KV_THR_LAYOUT_ROW
            == NUM_MMA_KV * 4 // NUM_WARPS_Q
            == NUM_MMA_KV * NUM_WARPS_KV
        )
        assert NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM == NUM_WARPS * NUM_MMA_Q * NUM_MMA_D_VO * 32 * 8
        assert NUM_WARPS_KV * CTA_TILE_Q * 2 == NUM_WARPS * NUM_MMA_Q * 16 * 2

        def get_permuted_offset(stride, i, j):
            return i * stride + (j ^ (i % 8))

        def get_warp_idx_q(tid):
            if NUM_WARPS_Q == 1:
                return 0
            else:
                return tid[1]

        def get_warp_idx_kv(tid):
            if NUM_WARPS_KV == 1:
                return 0
            else:
                return tid[2]

        def advance_offset_by_column(step_size: int, offset, step_idx: int):
            if not (step_size == 2 or step_size == 4 or step_size % 8 == 0):
                raise ValueError(f"Unsupported step_size {step_size} for K128B mode")

            if step_size == 2:
                return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + ((step_idx % 4 == 3) * 8)
            elif step_size == 4:
                return (offset ^ 0x4) + ((step_idx % 2 == 1) * 8)
            else:  # This condition implies step_size % 8 == 0
                return offset + step_size

        def advance_offset_by_row(step_size: int, row_stride: int, offset):
            if not (step_size == 4 or step_size % 8 == 0):
                raise ValueError(
                    f"Unsupported step_size: {step_size}. Must be 4 or a multiple of 8."
                )
            if step_size % 8 == 0:
                return offset + step_size * row_stride
            return (offset ^ 0x4) + step_size * row_stride

        def scope_sync():
            return T.tvm_storage_sync("shared")

        def m16k16_row_sum_f16f16f32(C_ptr, A_ptr):
            func_name = "m16k16_rowsum_f16f16f32"
            source_code = f"""
__device__ __forceinline__ void {func_name}(float* d, half* s) {{
  uint32_t* s_u32 = (uint32_t*)(s);
  asm volatile(
    "{{\\n"
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{{%0,  _,  %1,  _}}, "
    "{{%2,  %3,  %4,  %5}}, "
    "{{%6,  %7}}, "
    "{{%8,  0.,  %9,  0.}};\\n"
    "}}\\n"
    : "=f"(d[0]), "=f"(d[1])
    : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(0x3C003C00),
        "r"(0x3C003C00), "f"(d[0]), "f"(d[1]));
}}
"""
            return T.cuda.func_call(func_name, C_ptr, A_ptr, source_code=source_code)

        def store_128b(dst_ptr, src_ptr):
            func_name = "store_128b"
            source_code = f"""
__device__ __forceinline__ void {func_name}(void* dst_ptr, void* src_ptr) {{
  using b128_t = uint4;
  b128_t* dst_ptr_b128 = reinterpret_cast<b128_t*>(dst_ptr);
  b128_t* src_ptr_b128 = reinterpret_cast<b128_t*>(src_ptr);
  *dst_ptr_b128 = *src_ptr_b128;
}}
"""
            return T.cuda.func_call(func_name, dst_ptr, src_ptr, source_code=source_code)

        def get_sm_scale():
            func_name = "get_sm_scale"
            source_code = f"""
__device__ __forceinline__ float {func_name}() {{
  return 1.44269504088896340736 * 1 / sqrtf({HEAD_DIM});
}}
"""
            return T.cuda.func_call(func_name, source_code=source_code, return_type="float32")

        # fmt: off
        @T.prim_func(tirp=True)
        def batch_attention(
            q_ptr: T.handle,
            kv_ptr: T.handle,
            q_indptr_ptr: T.handle,
            kv_indptr_ptr: T.handle,
            partial_indptr_ptr: T.handle,
            kv_indices_ptr: T.handle,
            q_len_ptr: T.handle,
            kv_len_ptr: T.handle,
            q_start_ptr: T.handle,
            kv_start_ptr: T.handle,
            kv_end_ptr: T.handle,
            kv_head_idx_ptr: T.handle,
            work_indptr_ptr: T.handle,
            len_kv_chunk_ptr: T.handle,
            o_ptr: T.handle,
            partial_o_ptr: T.handle,
            partial_lse_ptr: T.handle,
        ):
            batch_size = T.int32()
            max_page_num = T.int64()
            total_page_num = T.int32()

            q_buf = T.match_buffer(q_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16", layout="default")
            kv_buf = T.match_buffer(kv_ptr, [max_page_num, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM], "float16", layout="default")
            q_indptr_buf = T.match_buffer(q_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_indptr_buf = T.match_buffer(kv_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            partial_indptr_buf = T.match_buffer(partial_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_indices_buf = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", layout="default", offset_factor=1)
            q_len_buf = T.match_buffer(q_len_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_len_buf = T.match_buffer(kv_len_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            q_start_buf = T.match_buffer(q_start_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_start_buf = T.match_buffer(kv_start_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_end_buf = T.match_buffer(kv_end_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            kv_head_idx_buf = T.match_buffer(kv_head_idx_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            work_indptr_buf = T.match_buffer(work_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            len_kv_chunk_buf = T.match_buffer(len_kv_chunk_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", layout="default", offset_factor=1)
            o_buf = T.match_buffer(o_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16", layout="default")
            partial_o_buf = T.match_buffer(partial_o_ptr, [2 * MAX_NUM_KV_SPLITS * head_dim], "float16", layout="default", offset_factor=1)
            partial_lse_buf = T.match_buffer(partial_lse_ptr, [2 * MAX_TOTAL_NUM_WORKERS], "float32", layout="default", offset_factor=1)
            
            with T.kernel():
                bx = T.cta_id([SM_COUNT * 2], parent="kernel")
                warp_id = T.warp_id([NUM_WARPS_Q * NUM_WARPS_KV], parent="cta")
                lane_id = T.thread_id([32], parent="warp")

                buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))
                q_smem = pool.alloc([CTA_TILE_Q * HEAD_DIM], "float16", layout="default", align=16)
                k_smem = pool.alloc([CTA_TILE_KV * HEAD_DIM], "float16", layout="default", align=16)
                v_smem = pool.alloc([CTA_TILE_KV * HEAD_DIM], "float16", layout="default", align=16)
                pool.move_base_to(0)
                cta_sync_o_smem = pool.alloc([1] if NUM_WARPS_KV == 1 else [NUM_WARPS, NUM_MMA_Q, NUM_MMA_D_VO, 32, 8], "float32", layout="default", align=16)
                cta_sync_md_smem = pool.alloc([1] if NUM_WARPS_KV == 1 else [NUM_WARPS, NUM_MMA_Q, 16, 2], "float32", layout="default", align=16)
                pool.move_base_to(0)
                smem_o = pool.alloc([CTA_TILE_Q * HEAD_DIM], "float16", layout="default", align=16)

                with T.thread():
                    s_frag = T.alloc_local([NUM_MMA_Q, NUM_MMA_KV, 8], "float32", align=0, layout="default")
                    o_frag = T.alloc_local([NUM_MMA_Q, NUM_MMA_D_VO, 8], "float32", align=16, layout="default")
                    m = T.alloc_local([NUM_MMA_Q, 2], "float32")
                    d = T.alloc_local([NUM_MMA_Q, 2], "float32")
                    tid = T.alloc_local([3], "int32")
                            
                    @T.macro
                    def debug_print_q(kv_tile_idx):
                        if DEBUG_PRINT:
                            scope_sync()
                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                T.cuda.printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR Q: kv_tile_idx = %d\n", kv_tile_idx[0])
                                for i in range(CTA_TILE_Q):
                                    T.cuda.printf("i = %d\n", i)
                                    for j in range(HEAD_DIM):
                                        T.cuda.printf("%f ", half_to_float(q_smem[i * HEAD_DIM + j]))
                                    T.cuda.printf("\n")
                            scope_sync()
                    @T.macro
                    def debug_print_k(kv_tile_idx):
                        if DEBUG_PRINT:
                            scope_sync()
                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                T.cuda.printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR K: kv_tile_idx = %d\n", kv_tile_idx[0])
                                for i in range(CTA_TILE_KV):
                                    T.cuda.printf("i = %d\n", i)
                                    for j in range(HEAD_DIM):
                                        T.cuda.printf("%f ", half_to_float(k_smem[i * HEAD_DIM + j]))
                                    T.cuda.printf("\n")
                            scope_sync()
                    @T.macro
                    def debug_print_v(kv_tile_idx):
                        if DEBUG_PRINT:
                            scope_sync()
                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                T.cuda.printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR V: kv_tile_idx = %d\n", kv_tile_idx[0])
                                for i in range(CTA_TILE_KV):
                                    T.cuda.printf("i = %d\n", i)
                                    for j in range(HEAD_DIM):
                                        T.cuda.printf("%f ", half_to_float(v_smem[i * HEAD_DIM + j]))
                                    T.cuda.printf("\n")
                            scope_sync()
                    @T.macro
                    def debug_print_somd_frag(kv_tile_idx, msg):
                        if DEBUG_PRINT:
                            scope_sync()
                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                T.cuda.printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR SOMD_FRAG: kv_tile_idx = %d, msg = %s\n", kv_tile_idx[0], msg)
                            scope_sync()
                            for warp_id_ in range(NUM_WARPS):
                                for lane_id_ in range(32):
                                    scope_sync()
                                    if bx == DEBUG_BX_EXPECTED and warp_id == warp_id_ and lane_id == lane_id_:
                                        T.cuda.printf("warp_id = %d, lane_id = %d\n", warp_id_, lane_id_)
                                        T.cuda.printf("s: ")
                                        for mma_q in range(NUM_MMA_Q):
                                            for mma_kv in range(NUM_MMA_KV):
                                                for j in range(8):
                                                    T.cuda.printf("%f ", s_frag[mma_q, mma_kv, j])
                                        T.cuda.printf("\n")
                                        T.cuda.printf("o: ")
                                        for mma_q in range(NUM_MMA_Q):
                                            for mma_d in range(NUM_MMA_D_VO):
                                                for j in range(8):
                                                    T.cuda.printf("%f ", o_frag[mma_q, mma_d, j])
                                        T.cuda.printf("\n")
                                        T.cuda.printf("m: ")
                                        for mma_q in range(NUM_MMA_Q):
                                            for j in range(2):
                                                T.cuda.printf("%f ", m[mma_q, j])
                                        T.cuda.printf("\n")
                                        T.cuda.printf("d: ")
                                        for mma_q in range(NUM_MMA_Q):
                                            for j in range(2):
                                                T.cuda.printf("%f ", d[mma_q, j])
                                        T.cuda.printf("\n")
                                    scope_sync()
                            scope_sync()

                    tid[0] = lane_id
                    tid[1] = warp_id % NUM_WARPS_Q
                    tid[2] = warp_id // NUM_WARPS_Q

                    q_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_Q, get_warp_idx_q(tid) * NUM_MMA_Q * 16 + lane_id % 16, lane_id // 16))
                    k_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_K, get_warp_idx_kv(tid) * NUM_MMA_KV * 16 + 8 * (lane_id // 16) + lane_id % 8, (lane_id % 16 // 8)))
                    v_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_V, get_warp_idx_kv(tid) * NUM_MMA_KV * 16 + lane_id % 16, lane_id // 16))
                    k_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_K, warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL, lane_id % KV_THR_LAYOUT_COL))
                    v_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_V, warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL, lane_id % KV_THR_LAYOUT_COL))
                    thr_local_kv_offset = T.alloc_local([NUM_MMA_KV * KV_THR_LAYOUT_COL // 2 // NUM_WARPS_Q], "int64")

                    @T.macro
                    def debug_print_thr_local_kv_offset(kv_tile_idx):
                        if DEBUG_PRINT:
                            scope_sync()
                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                T.cuda.printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR THR_LOCAL_KV_OFFSET: kv_tile_idx = %d\n", kv_tile_idx[0])
                            scope_sync()
                            for warp_id_ in range(NUM_WARPS):
                                for lane_id_ in range(32):
                                    scope_sync()
                                    if bx == DEBUG_BX_EXPECTED and warp_id == warp_id_ and lane_id == lane_id_:
                                        T.cuda.printf("warp_id = %d, lane_id = %d\n", warp_id_, lane_id_)
                                        for i in range(NUM_MMA_KV * KV_THR_LAYOUT_COL // 2 // NUM_WARPS_Q):
                                            T.cuda.printf("%d ", thr_local_kv_offset[i])
                                        T.cuda.printf("\n")
                                    scope_sync()
                            scope_sync()

                    work_idx = int_var()
                    work_idx[0] = work_indptr_buf[bx]
                    while work_idx[0] < work_indptr_buf[bx + 1]:
                        with T.thread():
                            # get_block_coord
                            q_indptr = int_var(q_indptr_buf[work_idx[0]])
                            kv_indptr = int_var(kv_indptr_buf[work_idx[0]])
                            o_indptr = int_var(partial_indptr_buf[work_idx[0]])
                            q_len = int_var(q_len_buf[work_idx[0]])
                            kv_len = int_var(kv_len_buf[work_idx[0]])
                            packed_qo_start = int_var(q_start_buf[work_idx[0]])
                            kv_start = int_var(kv_start_buf[work_idx[0]])
                            kv_end = int_var(kv_end_buf[work_idx[0]])
                            kv_head_idx = int_var(kv_head_idx_buf[work_idx[0]])
                            len_kv_chunk = int_var(len_kv_chunk_buf[work_idx[0]])
                            
                            kv_chunk_idx = int_var(ceildiv(kv_start[0], len_kv_chunk[0]))
                            num_kv_chunks = int_var(ceildiv(kv_len[0], len_kv_chunk[0]))
                            qo_packed_idx_base = int_var(packed_qo_start[0] + get_warp_idx_q(tid) * NUM_MMA_Q * 16)
                            qo_upperbound = int_var(T.min(q_len[0], ceildiv(qo_packed_idx_base[0] + CTA_TILE_Q, GQA_GROUP_SIZE)))

                            @T.macro
                            def init_states():
                                for i0, i1, i2 in T.grid(NUM_MMA_Q, NUM_MMA_D_VO, 8):
                                    o_frag[i0, i1, i2] = T.float32(0)
                                for i0, i1 in T.grid(NUM_MMA_Q, 2):
                                    m[i0, i1] = T.float32(-INF)
                                    d[i0, i1] = T.float32(0)
                            
                            init_states()

                            @T.macro
                            def load_q_global_smem():
                                if get_warp_idx_kv(tid) == 0:
                                    q_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_Q, get_warp_idx_q(tid) * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))
                                    # unroll
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for j in T.unroll(4):
                                            with T.thread():
                                                qo_packed_id = T.meta_var(qo_packed_idx_base[0] + lane_id // 8 + mma_q * 16 + j * 4)
                                                q = int_var(T.floordiv(qo_packed_id, GQA_GROUP_SIZE))
                                                r = int_var(T.floormod(qo_packed_id, GQA_GROUP_SIZE))
                                                for mma_do in T.unroll(NUM_MMA_D_QK // 4):
                                                    T.ptx.cp_async(q_smem.ptr_to([q_smem_offset_w[0] * upcast_size("float16")]),
                                                                   # TODO: optimize the addr computation here
                                                                   q_buf.ptr_to([q_indptr[0] + q[0], kv_head_idx[0] * GQA_GROUP_SIZE + r[0], (lane_id % 8 + mma_do * 8) * upcast_size("float16")]), cp_size=16, prefetch_size=128, 
                                                                   predicate=q[0] < qo_upperbound[0])
                                                    q_smem_offset_w[0] = advance_offset_by_column(8, q_smem_offset_w[0], mma_do)
                                            q_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_Q, q_smem_offset_w[0]) - 2 * NUM_MMA_D_QK
                            
                            load_q_global_smem()

                            kv_tile_idx = int_var(ceildiv(kv_end[0], CTA_TILE_KV) - 1 - (kv_start[0] // CTA_TILE_KV))
                            mast_tile_idx = int_var(kv_end[0] // CTA_TILE_KV - (kv_start[0] // CTA_TILE_KV))
                            
                            block_iter_base = int_var(kv_indptr[0] * PAGE_SIZE + kv_start[0])
                            scope_sync()
                            packed_kv_bound = int_var(kv_indptr[0] * PAGE_SIZE + kv_len[0])

                            #########################################################
                            @T.macro
                            def debug_print_work():
                                if DEBUG_PRINT:
                                    scope_sync()
                                    if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                        T.cuda.printf("q_indptr: %d\n", q_indptr[0])
                                        T.cuda.printf("kv_indptr: %d\n", kv_indptr[0])
                                        T.cuda.printf("q_len: %d\n", q_len[0])
                                        T.cuda.printf("kv_len: %d\n", kv_len[0])
                                        T.cuda.printf("packed_qo_start: %d\n", packed_qo_start[0])
                                        T.cuda.printf("kv_start: %d\n", kv_start[0])
                                        T.cuda.printf("kv_end: %d\n", kv_end[0])
                                        T.cuda.printf("kv_head_idx: %d\n", kv_head_idx[0])
                                        T.cuda.printf("len_kv_chunk: %d\n", len_kv_chunk[0])
                                        T.cuda.printf("kv_chunk_idx: %d\n", kv_chunk_idx[0])
                                        T.cuda.printf("num_kv_chunks: %d\n", num_kv_chunks[0])
                                        T.cuda.printf("qo_packed_idx_base: %d\n", qo_packed_idx_base[0])
                                        T.cuda.printf("qo_upperbound: %d\n", qo_upperbound[0])
                                        T.cuda.printf("block_iter_base: %d\n", block_iter_base[0])
                                        T.cuda.printf("packed_kv_bound: %d\n", packed_kv_bound[0])
                                        T.cuda.printf("kv_tile_idx: %d\n", kv_tile_idx[0])
                                        T.cuda.printf("mast_tile_idx: %d\n", mast_tile_idx[0])
                                    scope_sync()

                            debug_print_work()
                            #########################################################
                            
                            @T.macro
                            def prefetch_offset(packed_block_iter_base_in):
                                with T.thread():
                                    packed_block_iter_base = int_var(packed_block_iter_base_in)
                                    for i in T.unroll(NUM_MMA_KV * 4 // NUM_WARPS_Q):
                                        packed_block_iter = int_var(packed_block_iter_base[0] + warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL 
                                                                    + KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i)
                                        page_iter = int_var(T.floordiv(packed_block_iter[0], PAGE_SIZE))
                                        entry_idx = int_var(T.floormod(packed_block_iter[0], PAGE_SIZE))
                                        mapped_page = T.meta_var(T.if_then_else(packed_block_iter[0] < packed_kv_bound[0], kv_indices_buf[page_iter[0]], 0))
                                        thr_local_kv_offset[i] = kv_buf.offset_of_p([mapped_page, 0, kv_head_idx[0], entry_idx[0], (lane_id % KV_THR_LAYOUT_COL) * upcast_size("float16")])

                            @T.macro
                            def page_produce_kv(produce_v: bool, kv_idx_base_in, smem_offset, smem):
                                v_offset = KV_HEADS * PAGE_SIZE * HEAD_DIM if produce_v else 0
                                fill_mode = T.meta_var("zero" if produce_v else "")
                                NUM_MMA_D = T.meta_var(NUM_MMA_D_QK if produce_v else NUM_MMA_D_VO)
                                UPCAST_STRIDE = T.meta_var(UPCAST_STRIDE_V if produce_v else UPCAST_STRIDE_K)
                                with T.thread():
                                    kv_idx_base = int_var(kv_idx_base_in)
                                    kv_idx = int_var(kv_idx_base[0] + warp_id * 4 + lane_id // 8)
                                    kv_buf_1d = Tp.reshape(kv_buf, [-1])
                                    # unroll
                                    for i in T.unroll(NUM_MMA_KV * 4 // NUM_WARPS_Q):
                                        for j in T.unroll(NUM_MMA_D // (8 // size_of("float16"))):
                                            if DEBUG_PRINT:
                                                if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                                    T.cuda.printf("i, j, gptr - kv_ptr, smem_offset: %d, %d, %p, %d, thr_local_kv_offset: %d\n", i, j, kv_buf_1d.ptr_to([v_offset + thr_local_kv_offset[i] + 8 * j * upcast_size("float16")]) - kv_buf_1d.ptr_to([v_offset]), smem_offset[0], thr_local_kv_offset[i])
                                            T.ptx.cp_async(
                                                smem.ptr_to([smem_offset[0] * upcast_size("float16")]),
                                                # TODO: optimize the addr computation here
                                                kv_buf_1d.ptr_to([v_offset + thr_local_kv_offset[i] + 8 * j * upcast_size("float16")]), cp_size=16, prefetch_size=128, fill_mode=fill_mode,
                                                predicate=kv_idx[0] < kv_len[0]
                                            )
                                            smem_offset[0] = advance_offset_by_column(8, smem_offset[0], j)
                                        kv_idx[0] += NUM_WARPS * 4
                                        smem_offset[0] = advance_offset_by_row(NUM_WARPS * 4, UPCAST_STRIDE, smem_offset[0]) - size_of("float16") * NUM_MMA_D
                                    smem_offset[0] -= CTA_TILE_KV * UPCAST_STRIDE

                            @T.macro
                            def mma_sync_m16n16k16_row_col_f16f16f32(C_in, c_offset, A_in, a_offset, B_in, b_offset, init: bool):
                                with T.thread():
                                    C_mma = T.decl_buffer([8], dtype="float32", data=C_in.data, elem_offset=c_offset)
                                    A_mma = T.decl_buffer([4], dtype="uint32", data=A_in.data, elem_offset=a_offset)
                                    B_mma = T.decl_buffer([4], dtype="uint32", data=B_in.data, elem_offset=b_offset)
                                    if init:
                                        T.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                                                C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]))
                                        T.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                                                C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]))
                                    else:
                                        T.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                                                C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]), C_mma.ptr_to([0]))
                                        T.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                                                C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]), C_mma.ptr_to([4]))

                            @T.macro
                            def compute_qk():
                                with T.thread():
                                    a_frag = T.alloc_local([NUM_MMA_Q, 4], "uint32")
                                    b_frag = T.alloc_local([4], "uint32")

                                    # unroll
                                    for mma_d in T.unroll(NUM_MMA_D_QK):
                                        for mma_q in T.unroll(NUM_MMA_Q):
                                            T.ptx.ldmatrix(False, 4, ".b16", a_frag.ptr_to([mma_q, 0]), q_smem.ptr_to([q_smem_offset_r[0] * upcast_size("float16")]))
                                            q_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_Q, q_smem_offset_r[0])
                                        q_smem_offset_r[0] = advance_offset_by_column(2, q_smem_offset_r[0], mma_d) - NUM_MMA_Q * 16 * UPCAST_STRIDE_Q
                                        for mma_kv in T.unroll(NUM_MMA_KV):
                                            T.ptx.ldmatrix(False, 4, ".b16", b_frag.ptr_to([0]), k_smem.ptr_to([k_smem_offset_r[0] * upcast_size("float16")]))
                                            k_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_K, k_smem_offset_r[0])
                                            for mma_q in T.unroll(NUM_MMA_Q):
                                                if mma_d == 0:
                                                    mma_sync_m16n16k16_row_col_f16f16f32(s_frag, s_frag.offset_of_p([mma_q, mma_kv, 0]),
                                                                                         a_frag, a_frag.offset_of_p([mma_q, 0]),
                                                                                         b_frag, b_frag.offset_of_p([0]), True)
                                                else:
                                                    mma_sync_m16n16k16_row_col_f16f16f32(s_frag, s_frag.offset_of_p([mma_q, mma_kv, 0]),
                                                                                         a_frag, a_frag.offset_of_p([mma_q, 0]),
                                                                                         b_frag, b_frag.offset_of_p([0]), False)
                                        k_smem_offset_r[0] = advance_offset_by_column(2, k_smem_offset_r[0], mma_d) - NUM_MMA_KV * 16 * UPCAST_STRIDE_K
                                    q_smem_offset_r[0] -= NUM_MMA_D_QK * 2
                                    k_smem_offset_r[0] -= NUM_MMA_D_QK * size_of("float16")

                            @T.macro
                            def logits_mask():
                                chunk_end = T.meta_var(kv_end[0])
                                with T.thread():
                                    # unroll
                                    kv_idx_base = int_var(kv_start[0] + (kv_tile_idx[0] * NUM_WARPS_KV + get_warp_idx_kv(tid)) * NUM_MMA_KV * 16)
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for mma_kv in T.unroll(NUM_MMA_KV):
                                            for reg_id in T.unroll(8):
                                                with T.thread():
                                                    kv_idx = int_var(kv_idx_base[0] + mma_kv * 16 + 2 * (lane_id % 4) + 8 * (reg_id // 4) + reg_id % 2)
                                                    s_frag[mma_q, mma_kv, reg_id] = T.if_then_else(T.Not(kv_idx[0] >= chunk_end), s_frag[mma_q, mma_kv, reg_id], T.float32(-INF))

                            @T.macro
                            def update_mdo_states():
                                WARP_MASK = T.meta_var(0xFFFFFFFF)
                                with T.thread():
                                    sm_scale = float_var(get_sm_scale())
                                    if DEBUG_PRINT:
                                        if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                            T.cuda.printf("sm_scale: %f\n", sm_scale[0])
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for j in T.unroll(2):
                                            m_prev = float_var(m[mma_q, j])
                                            for mma_kv in T.unroll(NUM_MMA_KV):
                                                m_local = float_var(T.max(T.max(s_frag[mma_q, mma_kv, j * 2 + 0], s_frag[mma_q, mma_kv, j * 2 + 1]),
                                                                          T.max(s_frag[mma_q, mma_kv, j * 2 + 4], s_frag[mma_q, mma_kv, j * 2 + 5])))
                                                m[mma_q, j] = T.max(m[mma_q, j], m_local[0])
                                            m[mma_q, j] = T.max(m[mma_q, j], T.tvm_warp_shuffle_xor(WARP_MASK, m[mma_q, j], 0x2, 32, 32))
                                            m[mma_q, j] = T.max(m[mma_q, j], T.tvm_warp_shuffle_xor(WARP_MASK, m[mma_q, j], 0x1, 32, 32))

                                            o_scale = float_var(ptx_exp2(m_prev[0] * sm_scale[0] - m[mma_q, j] * sm_scale[0]))
                                            d[mma_q, j] *= o_scale[0]
                                            # unroll
                                            for mma_d in T.unroll(NUM_MMA_D_VO):
                                                o_frag[mma_q, mma_d, j * 2 + 0] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 1] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 4] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 5] *= o_scale[0]
                                            # unroll
                                            for mma_kv in T.unroll(NUM_MMA_KV):
                                                s_frag[mma_q, mma_kv, j * 2 + 0] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 0] * sm_scale[0] - m[mma_q, j] * sm_scale[0])
                                                s_frag[mma_q, mma_kv, j * 2 + 1] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 1] * sm_scale[0] - m[mma_q, j] * sm_scale[0])
                                                s_frag[mma_q, mma_kv, j * 2 + 4] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 4] * sm_scale[0] - m[mma_q, j] * sm_scale[0])
                                                s_frag[mma_q, mma_kv, j * 2 + 5] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 5] * sm_scale[0] - m[mma_q, j] * sm_scale[0])

                            @T.macro
                            def compute_sfm_v():
                                with T.thread():
                                    s_frag_f16 = T.alloc_local([NUM_MMA_Q, NUM_MMA_KV, 8], "float16", layout="default")
                                    Tp.cast(s_frag_f16[:, :, :], s_frag[:, :, :])
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for mma_kv in T.unroll(NUM_MMA_KV):
                                            m16k16_row_sum_f16f16f32(d.ptr_to([mma_q, 0]), s_frag_f16.ptr_to([mma_q, mma_kv, 0]))
                                    for mma_kv in T.unroll(NUM_MMA_KV):
                                        for mma_d in T.unroll(NUM_MMA_D_VO):
                                            with T.thread():
                                                b_frag = T.alloc_local([4], "uint32")
                                                T.ptx.ldmatrix(True, 4, ".b16", b_frag.ptr_to([0]), v_smem.ptr_to([v_smem_offset_r[0] * upcast_size("float16")]))
                                                for mma_q in T.unroll(NUM_MMA_Q):
                                                    s_frag_f16x2 = T.decl_buffer([NUM_MMA_Q, NUM_MMA_KV, 4], "uint32", data=s_frag_f16.data)
                                                    b_frag_f16 = T.decl_buffer([8], "float16", data=b_frag.data)
                                                    if DEBUG_PRINT:
                                                        if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == 0:
                                                            T.cuda.printf("mma_q = %d, mma_d = %d, mma_kv = %d\n", mma_q, mma_d, mma_kv)
                                                        for lane_id_ in range(32):
                                                            if bx == DEBUG_BX_EXPECTED and warp_id == 0 and lane_id == lane_id_:
                                                                T.cuda.printf("lane_id = %d\n", lane_id_)
                                                                for reg_id in range(8):
                                                                    T.cuda.printf("%f ", o_frag[mma_q, mma_d, reg_id])
                                                                T.cuda.printf("\n")
                                                                for reg_id in range(4):
                                                                    T.cuda.printf("%d ", b_frag[reg_id])
                                                                T.cuda.printf("\n")
                                                                for reg_id in range(8):
                                                                    T.cuda.printf("%f ", s_frag[mma_q, mma_kv, reg_id])
                                                                T.cuda.printf("\n")
                                                                for reg_id in range(8):
                                                                    T.cuda.printf("%f|%A ", half_to_float(s_frag_f16[mma_q, mma_kv, reg_id]), s_frag_f16[mma_q, mma_kv, reg_id])
                                                                T.cuda.printf("\n")
                                                                for reg_id in range(8):
                                                                    T.cuda.printf("%f ", half_to_float(b_frag_f16[reg_id]))
                                                                T.cuda.printf("\n")
                                                    mma_sync_m16n16k16_row_col_f16f16f32(
                                                        o_frag, o_frag.offset_of_p([mma_q, mma_d, 0]),
                                                        s_frag_f16x2, s_frag_f16x2.offset_of_p([mma_q, mma_kv, 0]),
                                                        b_frag, b_frag.offset_of_p([0]), False
                                                    )
                                                v_smem_offset_r[0] = advance_offset_by_column(2, v_smem_offset_r[0], mma_d)
                                        v_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_V, v_smem_offset_r[0]) - size_of("float16") * NUM_MMA_D_VO
                                    v_smem_offset_r[0] -= 16 * NUM_MMA_KV * UPCAST_STRIDE_V

                            @T.macro
                            def loop_body(WITH_MASK: bool):
                                prefetch_offset(block_iter_base[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV)
                                T.ptx.cp_async.wait_group(1)
                                scope_sync()
                                
                                compute_qk()
                                if WITH_MASK:
                                    logits_mask()
                                update_mdo_states()

                                scope_sync()
                                page_produce_kv(False, kv_start[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV, k_smem_offset_w, k_smem)
                                T.ptx.cp_async.commit_group()
                                T.ptx.cp_async.wait_group(1)
                                
                                scope_sync()
                                compute_sfm_v()
                                scope_sync()

                                page_produce_kv(True, kv_start[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV, v_smem_offset_w, v_smem)
                                T.ptx.cp_async.commit_group()
                            
                            prefetch_offset(block_iter_base[0] + kv_tile_idx[0] * CTA_TILE_KV)
                            debug_print_thr_local_kv_offset(kv_tile_idx)
                            page_produce_kv(False, kv_start[0] + kv_tile_idx[0] * CTA_TILE_KV, k_smem_offset_w, k_smem)
                            T.ptx.cp_async.commit_group()
                            page_produce_kv(True, kv_start[0] + kv_tile_idx[0] * CTA_TILE_KV, v_smem_offset_w, v_smem)
                            T.ptx.cp_async.commit_group()

                            while kv_tile_idx[0] >= mast_tile_idx[0] and kv_tile_idx[0] > 0:
                                loop_body(True)
                                kv_tile_idx[0] -= 1
                            
                            while kv_tile_idx[0] + 1 > NUM_STAGES:
                                loop_body(False)
                                kv_tile_idx[0] -= 1
                            
                            T.ptx.cp_async.wait_group(0)
                            scope_sync()


                            #########################################################
                            debug_print_q(kv_tile_idx)
                            #########################################################
                            debug_print_k(kv_tile_idx)
                            #########################################################
                            debug_print_v(kv_tile_idx)
                            #########################################################

                            while kv_tile_idx[0] >= 0:
                                compute_qk()
                                
                                #########################################################
                                debug_print_somd_frag(kv_tile_idx, "after compute_qk")
                                #########################################################

                                logits_mask()

                                #########################################################
                                debug_print_somd_frag(kv_tile_idx, "after logits_mask")
                                #########################################################

                                update_mdo_states()

                                #########################################################
                                debug_print_somd_frag(kv_tile_idx, "after update_mdo_states")
                                #########################################################

                                compute_sfm_v()

                                #########################################################
                                debug_print_somd_frag(kv_tile_idx, "after compute_sfm_v")
                                #########################################################

                                kv_tile_idx[0] -= 1
                            
                            scope_sync()

                            @T.macro
                            def finalize_m():
                                with T.thread():
                                    sm_scale = float_var(get_sm_scale())
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for j in T.unroll(2):
                                            if m[mma_q, j] != -INF:
                                                m[mma_q, j] *= sm_scale[0]
                            
                            @T.macro
                            def threadblock_sync_mdo_states():
                                for mma_q in T.unroll(NUM_MMA_Q):
                                    for mma_d in T.unroll(NUM_MMA_D_VO):
                                        Tp.copy(cta_sync_o_smem[warp_id, mma_q, mma_d, lane_id, :], o_frag[mma_q, mma_d, :])

                                for mma_q in T.unroll(NUM_MMA_Q):
                                    for j in T.unroll(2):
                                        with T.thread():
                                            md = T.alloc_local([2], "float32", layout="default")
                                            md[0] = m[mma_q, j]
                                            md[1] = d[mma_q, j]
                                            Tp.copy(cta_sync_md_smem[warp_id, mma_q, j * 8 + lane_id // 4, :], md[:])
                                scope_sync()

                                for mma_q in T.unroll(NUM_MMA_Q):
                                    with T.thread():
                                        o_scale = T.alloc_local([2, NUM_WARPS_KV], "float32")
                                        for j in T.unroll(2):
                                            with T.thread():
                                                m_new = float_var(-INF)
                                                d_new = float_var(1.0)
                                                for i in T.unroll(NUM_WARPS_KV):
                                                    with T.thread():
                                                        md = T.alloc_local([2], "float32", layout="default") 
                                                        Tp.copy(md[:], cta_sync_md_smem[i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, j * 8 + lane_id // 4, :])
                                                        m_prev = float_var(m_new[0])
                                                        d_prev = float_var(d_new[0])
                                                        m_new[0] = T.max(m_new[0], md[0])
                                                        d_new[0] = d_prev[0] * ptx_exp2(m_prev[0] - m_new[0]) + md[1] * ptx_exp2(md[0] - m_new[0])
                                                for i in T.unroll(NUM_WARPS_KV):
                                                    with T.thread():
                                                        md = T.alloc_local([2], "float32", layout="default")
                                                        Tp.copy(md[:], cta_sync_md_smem[i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, j * 8 + lane_id // 4, :])
                                                        o_scale[j, i] = ptx_exp2(md[0] - m_new[0])
                                                m[mma_q, j] = m_new[0]
                                                d[mma_q, j] = d_new[0]
                                        for mma_d in T.unroll(NUM_MMA_D_VO):
                                            with T.thread():
                                                o_new = T.alloc_local([8], "float32", layout="default")
                                                for i in T.unroll(8):
                                                    o_new[i] = 0.0
                                                for i in T.unroll(NUM_WARPS_KV):
                                                    with T.thread():
                                                        o_i = T.alloc_local([8], "float32", layout="default")
                                                        Tp.copy(o_i[:], cta_sync_o_smem[i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, mma_d, lane_id, :])
                                                        for reg_id in T.unroll(8):
                                                            o_new[reg_id] += o_i[reg_id] * o_scale[(reg_id % 4) // 2, i]
                                                Tp.copy(o_frag[mma_q, mma_d, :], o_new[:])

                            @T.macro
                            def normalize_d():
                                with T.thread():
                                    d_rcp = T.alloc_local([NUM_MMA_Q, 2], "float32", layout="default")
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for j in T.unroll(2):
                                            d_rcp[mma_q, j] = T.if_then_else(m[mma_q, j] != -INF, ptx_rcp(d[mma_q, j]), 0.0)
                                    for mma_q in T.unroll(NUM_MMA_Q):
                                        for mma_d in T.unroll(NUM_MMA_D_VO):
                                            for reg_id in T.unroll(8):
                                                o_frag[mma_q, mma_d, reg_id] *= d_rcp[mma_q, (reg_id >> 1) & 1]

                            @T.macro
                            def store_o_to_smem(o_smem):
                                for mma_q in T.unroll(NUM_MMA_Q):
                                    for mma_d in T.unroll(NUM_MMA_D_VO):
                                        with T.thread():
                                            o_frag_f16 = T.alloc_local([8], "float16", layout="default")
                                            Tp.cast(o_frag_f16[:], o_frag[mma_q, mma_d, :])
                                            o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, (get_warp_idx_q(tid) * NUM_MMA_Q + mma_q) * 16 + lane_id % 16, mma_d * 2 + lane_id // 16))
                                            T.ptx.stmatrix(4, False, o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]), o_frag_f16.ptr_to([0]))

                            @T.macro
                            def write_partial_o(o_ptr_base_offset, o_stride_n):
                                o_packed_idx_base_warp = T.meta_var(qo_packed_idx_base[0])
                                o_packed_idx_base_cta = T.meta_var(packed_qo_start[0])
                                o_smem = T.meta_var(smem_o)
                                warp_id_x = int_var(get_warp_idx_q(tid))
                                warp_id_z = int_var(get_warp_idx_kv(tid))
                                
                                if warp_id_z[0] == 0:
                                    with T.thread():
                                        store_o_to_smem(o_smem)
                                        o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, warp_id_x[0] * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))
                                        for mma_q in T.unroll(NUM_MMA_Q):
                                            for j in T.unroll(4):
                                                with T.thread():
                                                    o_packed_idx = int_var(o_packed_idx_base_warp + lane_id // 8 + mma_q * 16 + j * 4)
                                                    q = int_var(T.floordiv(o_packed_idx[0], GQA_GROUP_SIZE))
                                                    r = int_var(T.floormod(o_packed_idx[0], GQA_GROUP_SIZE))
                                                    o_ptr_offset = int_var(o_ptr_base_offset + (o_packed_idx[0] - o_packed_idx_base_cta) * o_stride_n + (lane_id % 8) * upcast_size("float16"))
                                                    for mma_do in T.unroll(NUM_MMA_D_VO // 4):
                                                        if q[0] < qo_upperbound[0]:
                                                            store_128b(partial_o_buf.ptr_to([o_ptr_offset[0]]), o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]))
                                                        o_ptr_offset[0] += 8 * upcast_size("float16")
                                                        o_smem_offset_w[0] = advance_offset_by_column(8, o_smem_offset_w[0], mma_do)
                                                    o_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_O, o_smem_offset_w[0]) - 2 * NUM_MMA_D_VO

                            @T.macro
                            def write_final_o(o_ptr_base_offset):
                                o_packed_idx_base = T.meta_var(qo_packed_idx_base[0])
                                o_smem = T.meta_var(smem_o)
                                warp_id_x = int_var(get_warp_idx_q(tid))
                                warp_id_z = int_var(get_warp_idx_kv(tid))

                                if warp_id_z[0] == 0:
                                    with T.thread():
                                        store_o_to_smem(o_smem)
                                        o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, warp_id_x[0] * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))
                                        for mma_q in T.unroll(NUM_MMA_Q):
                                            for j in T.unroll(4):
                                                with T.thread():
                                                    o_packed_idx = int_var(o_packed_idx_base + lane_id // 8 + mma_q * 16 + j * 4)
                                                    q = int_var(T.floordiv(o_packed_idx[0], GQA_GROUP_SIZE))
                                                    r = int_var(T.floormod(o_packed_idx[0], GQA_GROUP_SIZE))
                                                    o_ptr_offset = int_var(o_ptr_base_offset + o_buf.offset_of_p([q[0], r[0], (lane_id % 8) * upcast_size("float16")]))
                                                    for mma_do in T.unroll(NUM_MMA_D_VO // 4):
                                                        if q[0] < qo_upperbound[0]:
                                                            o_buf_1d = Tp.reshape(o_buf, [-1])
                                                            store_128b(o_buf_1d.ptr_to([o_ptr_offset[0]]), o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]))
                                                        o_ptr_offset[0] += 8 * upcast_size("float16")
                                                        o_smem_offset_w[0] = advance_offset_by_column(8, o_smem_offset_w[0], mma_do)
                                                    o_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_O, o_smem_offset_w[0]) - 2 * NUM_MMA_D_VO

                            @T.macro
                            def write_partial_lse():
                                if num_kv_chunks[0] > 1:
                                    if get_warp_idx_kv(tid) == 0:
                                        for mma_q in T.unroll(NUM_MMA_Q):
                                            for j in T.unroll(2):
                                                with T.thread():
                                                    packed_qo_idx = int_var(qo_packed_idx_base[0] + lane_id // 4 + j * 8 + mma_q * 16)
                                                    q = int_var(T.floordiv(packed_qo_idx[0], GQA_GROUP_SIZE))
                                                    r = int_var(T.floormod(packed_qo_idx[0], GQA_GROUP_SIZE))
                                                    if q[0] < qo_upperbound[0]:                                                    
                                                        partial_lse_buf_offset = T.meta_var((o_indptr[0] + (packed_qo_idx[0] - packed_qo_start[0]) * num_kv_chunks[0] + kv_chunk_idx[0]) * KV_HEADS + kv_head_idx[0])
                                                        partial_lse_buf[partial_lse_buf_offset] = ptx_log2(d[mma_q, j]) + T.cast(m[mma_q, j], "float32")

                            finalize_m()
                            debug_print_somd_frag(kv_tile_idx, "after finalize_m")
                            threadblock_sync_mdo_states()
                            debug_print_somd_frag(kv_tile_idx, "after threadblock_sync_mdo_states")
                            normalize_d()
                            debug_print_somd_frag(kv_tile_idx, "after normalize_d")

                            if num_kv_chunks[0] > 1:
                                # reuse q, k, v's smem
                                with T.thread():
                                    o_ptr_base_offset = int_var(((o_indptr[0] + kv_chunk_idx[0]) * KV_HEADS + kv_head_idx[0]) * HEAD_DIM)
                                    write_partial_o(o_ptr_base_offset[0], num_kv_chunks[0] * KV_HEADS * HEAD_DIM)
                            else:
                                with T.thread():
                                    o_ptr_base_offset = int_var(o_buf.offset_of_p([q_indptr[0], kv_head_idx[0] * GQA_GROUP_SIZE, 0]))
                                    write_final_o(o_ptr_base_offset[0])

                            write_partial_lse()

                            work_idx[0] += 1
        # fmt: on
        return batch_attention

    def craft_batch_attention_merge_kernel():
        NUM_THREADS = 128
        NUM_WARPS = NUM_THREADS // 32
        NUM_SMEM_STAGES = 4
        VEC_SIZE = max(16 // size_of("float16"), HEAD_DIM // 32)
        BDX = HEAD_DIM // VEC_SIZE
        assert NUM_THREADS % BDX == 0
        BDY = 32 // BDX
        SMEM_SIZE = NUM_WARPS * NUM_SMEM_STAGES * BDY * HEAD_DIM * size_of(
            "float16"
        ) + NUM_THREADS * size_of("float32")
        NUM_WORKERS = SM_COUNT * 2 * NUM_WARPS

        class State:
            def __init__(self, vec_size):
                self.vec_size = vec_size
                self.o = T.alloc_local([vec_size], "float32", layout="default")
                self.m = T.alloc_local([1], "float32", layout="default")
                self.d = T.alloc_local([1], "float32", layout="default")
                IRBuilder.name("state_o", self.o)
                IRBuilder.name("state_m", self.m)
                IRBuilder.name("state_d", self.d)

            @T.macro
            def init(self):
                with T.thread():
                    for i in T.unroll(self.vec_size):
                        self.o[i] = 0.0
                    self.m[0] = -INF
                    self.d[0] = 1.0

            # fmt: off
            @T.macro
            def merge(self, other_o, other_m, other_d):
                with T.thread():
                    m_prev = float_var(self.m[0])
                    d_prev = float_var(self.d[0])
                    self.m[0] = T.max(m_prev[0], other_m)
                    self.d[0] = d_prev[0] * ptx_exp2(m_prev[0] - self.m[0]) + other_d * ptx_exp2(other_m - self.m[0])
                    for i in T.unroll(self.vec_size):
                        self.o[i] = self.o[i] * ptx_exp2(m_prev[0] - self.m[0]) + other_o[i] * ptx_exp2(other_m - self.m[0])
            # fmt: on
            
            # fmt: off
            @T.macro
            def normalize(self):
                with T.thread():
                    for i in T.unroll(self.vec_size):
                        self.o[i] = fdivdef(self.o[i], self.d[0])
            # fmt: on

            def get_lse(self):
                return self.m[0] + ptx_log2(self.d[0])

        def warp_sync():
            func_name = "tvm_builtin_warp_sync"
            source_code = f"""
__device__ __forceinline__ void {func_name}() {{
    __syncwarp();
}}
"""
            return T.cuda.func_call(func_name, source_code=source_code)

        # fmt: off
        @T.prim_func(tirp=True)
        def batch_attention_merge(
            partial_o_ptr: T.handle,
            final_o_ptr: T.handle,
            partial_lse_ptr: T.handle,
            num_qo_len_ptr: T.handle,
            merge_indptr_ptr: T.handle,
            merge_o_indices_ptr: T.handle,
        ):
            batch_size = T.int32()

            partial_o_buf = T.match_buffer(partial_o_ptr, [2 * MAX_NUM_KV_SPLITS * head_dim], "float16", layout="default", offset_factor=1)
            final_o_buf = T.match_buffer(final_o_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16", layout="default")
            partial_lse_buf = T.match_buffer(partial_lse_ptr, [2 * MAX_TOTAL_NUM_WORKERS], "float32", layout="default", offset_factor=1)
            num_qo_len_buf = T.match_buffer(num_qo_len_ptr, [1], "int32", layout="default", offset_factor=1)
            merge_indptr_buf = T.match_buffer(merge_indptr_ptr, [MAX_NUM_KV_SPLITS], "int32", layout="default", offset_factor=1)
            merge_o_indices_buf = T.match_buffer(merge_o_indices_ptr, [MAX_NUM_KV_SPLITS], "int32", layout="default", offset_factor=1)
            with T.kernel():
                bx = T.cta_id([SM_COUNT * 2], parent="kernel")
                warp_id = T.warp_id([NUM_THREADS // 32], parent="cta")
                lane_id = T.thread_id([32], parent="warp")

                with T.thread():
                    tx = int_var(lane_id % BDX)
                    ty = int_var(lane_id // BDX)
                    worker_id = int_var(bx * NUM_WARPS + warp_id)

                    buf = T.alloc_shared([SMEM_SIZE], "uint8", layout="default")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    v_smem = pool.alloc([NUM_WARPS, NUM_SMEM_STAGES, BDY, HEAD_DIM], "float16", layout="default")
                    s_smem = pool.alloc([NUM_WARPS, 32], "float32", layout="default")

                    num_qo_len_local = int_var(num_qo_len_buf[0])
                    
                    i = int_var(worker_id[0])
                    while i[0] < num_qo_len_local[0] * KV_HEADS:
                        with T.thread():
                            warp_sync()
                            packed_qo_idx = int_var(T.floordiv(i[0], KV_HEADS))
                            kv_head_idx = int_var(T.floormod(i[0], KV_HEADS))
                            qo_head_idx = int_var(T.floormod(packed_qo_idx[0], GQA_GROUP_SIZE))

                            partial_idx_to_offset = T.meta_var(lambda off: (merge_indptr_buf[packed_qo_idx[0]] + off) * KV_HEADS + kv_head_idx[0])
                            merge_idx_to_offset = T.meta_var((merge_o_indices_buf[packed_qo_idx[0]] * KV_HEADS + kv_head_idx[0]) * GQA_GROUP_SIZE + qo_head_idx[0])

                            state = T.meta_var(State(VEC_SIZE))
                            state.init()

                            num_index_sets = int_var(merge_indptr_buf[packed_qo_idx[0] + 1] - merge_indptr_buf[packed_qo_idx[0]])

                            # prelogue
                            for it in range(NUM_SMEM_STAGES):
                                with T.thread():
                                    T.ptx.cp_async(
                                        v_smem.ptr_to([warp_id, it, ty[0], tx[0] * VEC_SIZE]),
                                        partial_o_buf.ptr_to([partial_idx_to_offset(it * BDY + ty[0]) * HEAD_DIM + tx[0] * VEC_SIZE]),
                                        cp_size=16, prefetch_size=128,
                                        predicate=it * BDY + ty[0] < num_index_sets[0]
                                    )
                                    T.ptx.cp_async.commit_group()

                            for it in T.serial(ceildiv(num_index_sets[0], BDY)):
                                with T.thread():
                                    if it % BDX == 0:
                                        s_smem[warp_id, ty[0] * BDX + tx[0]] = T.if_then_else(
                                            it * BDY + (ty[0] * BDX + tx[0]) < num_index_sets[0],
                                            partial_lse_buf[partial_idx_to_offset(it * BDY + ty[0] * BDX + tx[0])],
                                            T.float32(0)
                                        )
                                        warp_sync()
                                    T.ptx.cp_async.wait_group(NUM_SMEM_STAGES - 1)
                                    warp_sync()

                                    v = T.alloc_local([VEC_SIZE], "float32", layout="default")
                                    cast_load(v, VEC_SIZE, v_smem, warp_id, it % NUM_SMEM_STAGES, ty[0], tx[0] * VEC_SIZE)
                                    
                                    if it * BDY + ty[0] < num_index_sets[0]:
                                        s = float_var(s_smem[warp_id, (it % BDX) * BDY + ty[0]])
                                        state.merge(v, s[0], 1)
                                    warp_sync()

                                    T.ptx.cp_async(
                                        v_smem.ptr_to([warp_id, (it % NUM_SMEM_STAGES), ty[0], tx[0] * VEC_SIZE]),
                                        partial_o_buf.ptr_to([partial_idx_to_offset((it + NUM_SMEM_STAGES) * BDY + ty[0]) * HEAD_DIM + tx[0] * VEC_SIZE]),
                                        cp_size=16, prefetch_size=128,
                                        predicate=(it + NUM_SMEM_STAGES) * BDY + ty[0] < num_index_sets[0]
                                    )
                                    T.ptx.cp_async.commit_group()

                            T.ptx.cp_async.wait_group(0)
                            warp_sync()

                            @T.macro
                            def warp_sync_state():
                                cast_store(state.o, VEC_SIZE, v_smem, warp_id, 0, ty[0], tx[0] * VEC_SIZE)
                                s_smem[warp_id, ty[0]] = state.get_lse()
                                state.init()
                                warp_sync()

                                for it in T.unroll(BDY):
                                    with T.thread():
                                        s = float_var(s_smem[warp_id, it])
                                        v = T.alloc_local([VEC_SIZE], "float32", layout="default")
                                        cast_load(v, VEC_SIZE, v_smem, warp_id, 0, it, tx[0] * VEC_SIZE)
                                        state.merge(v, s[0], 1)

                            state.normalize()
                            if (BDY > 1):
                                warp_sync_state()
                                state.normalize()

                            final_o_buf_2d = Tp.reshape(final_o_buf, [-1, HEAD_DIM])
                            cast_store(state.o, VEC_SIZE, final_o_buf_2d, merge_idx_to_offset, tx[0] * VEC_SIZE)

                            i[0] += NUM_WORKERS
        # fmt: on

        return batch_attention_merge

    return craft_batch_attention_kernel(), craft_batch_attention_merge_kernel()


@pytest.mark.parametrize("num_heads", [(64, 8)])
@pytest.mark.parametrize("seq_len", [512, 2077, 4033])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 11, 25, 128])
@pytest.mark.parametrize("seed", [42])
def test(num_heads, seq_len, head_dim, batch_size, seed):
    PAGE_SIZE = 16
    MAX_PAGE_NUM = 32768
    qo_heads, kv_heads = num_heads

    torch.manual_seed(seed)

    Q, KV_data, KV_indptr, KV_last_page_len, KV_indices = perpare_data(
        batch_size, qo_heads, kv_heads, seq_len, head_dim, PAGE_SIZE, MAX_PAGE_NUM
    )
    kv_indptr_f = KV_indptr.to(0)
    kv_indices_f = KV_indices.to(0)
    qo_indptr_f = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
    wrapper = flashinfer.BatchAttention(KV_LAYOUT)
    wrapper.plan(
        qo_indptr_f,
        kv_indptr_f,
        kv_indices_f,
        torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
        qo_heads,
        kv_heads,
        head_dim,
        head_dim,
        PAGE_SIZE,
        kv_data_type=torch.float16,
        q_data_type=torch.float16,
    )
    plan_info = wrapper._plan_info

    def get_id(i):
        return plan_info[i].item()

    def tensor_from_bytes(
        byte_tensor: torch.Tensor, offset: int, shape, data_type: torch.dtype
    ) -> torch.Tensor:
        if byte_tensor.dtype != torch.uint8 or byte_tensor.dim() != 1:
            raise ValueError("Input must be a 1D torch.uint8 tensor.")

        num_elements = shape
        element_byte_size = torch.tensor([], dtype=data_type).element_size()
        required_bytes = num_elements * element_byte_size

        if offset + required_bytes > byte_tensor.numel():
            raise ValueError("The requested offset and shape are out of bounds.")

        byte_slice = byte_tensor[offset : offset + required_bytes]

        return byte_slice.view(data_type)

    def get_tensor(offset, shape, data_type):
        if data_type == torch.int32:
            return tensor_from_bytes(wrapper.int_workspace_buffer, offset, shape, data_type)
        elif data_type in [torch.float16, torch.float32]:
            return tensor_from_bytes(wrapper.float_workspace_buffer, offset, shape, data_type)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    for i in range(0, 2):
        q_indptr = get_tensor(get_id(i * 11 + 2), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_indptr = get_tensor(get_id(i * 11 + 3), MAX_TOTAL_NUM_WORKERS, torch.int32)
        partial_indptr = get_tensor(get_id(i * 11 + 4), MAX_TOTAL_NUM_WORKERS, torch.int32)
        q_len = get_tensor(get_id(i * 11 + 5), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_len = get_tensor(get_id(i * 11 + 6), MAX_TOTAL_NUM_WORKERS, torch.int32)
        q_start = get_tensor(get_id(i * 11 + 7), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_start = get_tensor(get_id(i * 11 + 8), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_end = get_tensor(get_id(i * 11 + 9), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_head_idx = get_tensor(get_id(i * 11 + 10), MAX_TOTAL_NUM_WORKERS, torch.int32)
        work_indptr = get_tensor(get_id(i * 11 + 11), MAX_TOTAL_NUM_WORKERS, torch.int32)
        len_kv_chunk = get_tensor(get_id(i * 11 + 12), MAX_TOTAL_NUM_WORKERS, torch.int32)
    partial_o = get_tensor(get_id(1 * 11 + 13), 2 * MAX_NUM_KV_SPLITS * head_dim, torch.float16)
    partial_lse = get_tensor(get_id(1 * 11 + 14), 2 * MAX_TOTAL_NUM_WORKERS, torch.float32)
    merge_indptr = get_tensor(get_id(1 * 11 + 15), MAX_NUM_KV_SPLITS, torch.int32)
    merge_o_indices = get_tensor(get_id(1 * 11 + 16), MAX_NUM_KV_SPLITS, torch.int32)
    num_qo_len = get_tensor(get_id(1 * 11 + 17), 1, torch.int32)

    out_f = torch.zeros(batch_size, qo_heads, head_dim, dtype=torch.float16, device="cuda")
    lse_f = torch.zeros(batch_size, qo_heads, dtype=torch.float32, device="cuda")

    def print_work(work_idx):
        print("q_indptr", q_indptr[work_idx].cpu().numpy())
        print("kv_indptr", kv_indptr[work_idx].cpu().numpy())
        print("partial_indptr", partial_indptr[work_idx].cpu().numpy())
        print("q_len", q_len[work_idx].cpu().numpy())
        print("kv_len", kv_len[work_idx].cpu().numpy())
        print("q_start", q_start[work_idx].cpu().numpy())
        print("kv_start", kv_start[work_idx].cpu().numpy())
        print("kv_end", kv_end[work_idx].cpu().numpy())
        print("kv_head_idx", kv_head_idx[work_idx].cpu().numpy())
        print("len_kv_chunk", len_kv_chunk[work_idx].cpu().numpy())

    def flashinfer_batch_attention():
        q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
        kv_data_f = KV_data.to(0)
        func = lambda: wrapper.run(q_f, kv_data_f, out_f, lse_f)
        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_attention")
        func()
        print(f"FlashInfer BatchAttention time: {ms:.3f} ms")

        return partial_o.cpu().numpy(), partial_lse.cpu().numpy(), out_f.cpu().numpy()

    def tir():
        def torch_to_tvm(tensor):
            return tvm.ffi.from_dlpack(torch.to_dlpack(tensor))

        DEV = tvm.cuda(0)
        q_tvm = tvm.nd.array(Q, DEV)
        kv_data_tvm = tvm.nd.array(KV_data, DEV)
        q_indptr_tvm = torch_to_tvm(q_indptr)
        kv_indptr_tvm = torch_to_tvm(kv_indptr)
        partial_indptr_tvm = torch_to_tvm(partial_indptr)
        kv_indices_tvm = tvm.nd.array(KV_indices, DEV)
        q_len_tvm = torch_to_tvm(q_len)
        kv_len_tvm = torch_to_tvm(kv_len)
        q_start_tvm = torch_to_tvm(q_start)
        kv_start_tvm = torch_to_tvm(kv_start)
        kv_end_tvm = torch_to_tvm(kv_end)
        kv_head_idx_tvm = torch_to_tvm(kv_head_idx)
        work_indptr_tvm = torch_to_tvm(work_indptr)
        len_kv_chunk_tvm = torch_to_tvm(len_kv_chunk)
        o_tvm = torch_to_tvm(
            torch.zeros(batch_size, qo_heads, head_dim, dtype=torch.float16, device="cuda")
        )
        lse_tvm = torch_to_tvm(
            torch.zeros(batch_size, qo_heads, dtype=torch.float32, device="cuda")
        )
        partial_o_tvm = torch_to_tvm(partial_o)
        partial_lse_tvm = torch_to_tvm(partial_lse)
        merge_indptr_tvm = torch_to_tvm(merge_indptr)
        merge_o_indices_tvm = torch_to_tvm(merge_o_indices)
        num_qo_len_tvm = torch_to_tvm(num_qo_len)

        attention, merge = get_batch_attention_kernel(qo_heads, kv_heads, head_dim, PAGE_SIZE)
        mod_attn = tvm.IRModule({"main": attention})
        mod_merge = tvm.IRModule({"main": merge})
        target = tvm.target.Target("cuda")
        with target:
            mod_attn = tvm.compile(mod_attn, target=target, tir_pipeline="tirp")
            mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirp")

        def func():
            mod_attn(
                q_tvm,
                kv_data_tvm,
                q_indptr_tvm,
                kv_indptr_tvm,
                partial_indptr_tvm,
                kv_indices_tvm,
                q_len_tvm,
                kv_len_tvm,
                q_start_tvm,
                kv_start_tvm,
                kv_end_tvm,
                kv_head_idx_tvm,
                work_indptr_tvm,
                len_kv_chunk_tvm,
                o_tvm,
                partial_o_tvm,
                partial_lse_tvm,
            )
            mod_merge(
                partial_o_tvm,
                o_tvm,
                partial_lse_tvm,
                num_qo_len_tvm,
                merge_indptr_tvm,
                merge_o_indices_tvm,
            )

        ms = bench(func, warmup=10, repeat=30, proton_name="tir")
        func()
        print(f"TIR time: {ms:.3f} ms")

        return partial_o_tvm.numpy(), partial_lse_tvm.numpy(), o_tvm.numpy()

    with ProtonContext("batch_attention"):
        print(
            f"qo_heads: {qo_heads}, kv_heads: {kv_heads}, seq_len: {seq_len}, head_dim: {head_dim}, batch_size: {batch_size}, seed: {seed}"
        )
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Flashinfer Kernel Start",
            flush=True,
        )
        partial_o_flashinfer, partial_lse_flashinfer, O_flashinfer = flashinfer_batch_attention()
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TIR Kernel Start",
            flush=True,
        )
        partial_o_tir, partial_lse_tir, O_tir = tir()

    np.testing.assert_allclose(partial_o_tir, partial_o_flashinfer, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(partial_lse_tir, partial_lse_flashinfer, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(O_tir, O_flashinfer, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    import itertools

    num_heads_list = [(64, 8)]
    seq_len_list = [512, 2077, 4033]
    head_dim_list = [128]
    batch_size_list = [1, 11, 25, 128]
    seed_list = [42]

    for num_heads, seq_len, head_dim, batch_size, seed in itertools.product(
        num_heads_list, seq_len_list, head_dim_list, batch_size_list, seed_list
    ):
        test(num_heads, seq_len, head_dim, batch_size, seed)
