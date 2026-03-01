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
import tvm_ffi

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.bench.utils import ProtonContext, bench


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

MAX_TOTAL_NUM_WORKERS = 1025
MAX_NUM_KV_SPLITS = 4 * SM_COUNT * 2 * 16


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
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def ptx_log2(x):
    func_name = "tvm_builtin_ptx_log2"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def ptx_rcp(x):
    func_name = "tvm_builtin_ptx_rcp"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def half_to_float(x):
    func_name = "tvm_builtin_half_to_float"
    source_code = f"""
__device__ __forceinline__ float {func_name}(half x) {{
  return __half2float(x);
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def fdivdef(x, y):
    func_name = "tvm_builtin_fdivdef"
    source_code = f"""
__device__ __forceinline__ float {func_name}(float x, float y) {{
  return __fdividef(x, y);
}}
"""
    return Tx.cuda.func_call(func_name, x, y, source_code=source_code, return_type="float32")


@Tx.inline
def cast_load(v, vec_len, buf, *indices):
    with Tx.thread():
        v_tmp = Tx.alloc_local([vec_len], buf.dtype)
        for i in Tx.vectorized(vec_len):
            buffer_load = Tx.meta_var(Tx.BufferLoad(buf, (*indices[:-1], indices[-1] + i)))
            v_tmp[i] = buffer_load
        Tx.cast(v[:], v_tmp[:])


@Tx.inline
def cast_store(v, vec_len, buf, *indices):
    with Tx.thread():
        v_tmp = Tx.alloc_local([vec_len], buf.dtype)
        Tx.cast(v_tmp[:], v[:])
        for i in Tx.vectorized(vec_len):
            Tx.buffer_store(buf, v_tmp[i], (*indices[:-1], indices[-1] + i))


def get_batch_attention_kernel(qo_heads, kv_heads, head_dim, page_size):
    QO_HEADS = qo_heads
    KV_HEADS = kv_heads
    HEAD_DIM = head_dim
    PAGE_SIZE = page_size
    GQA_GROUP_SIZE = QO_HEADS // KV_HEADS
    INF = 5e4

    def int_var(val=None):
        buf = Tx.alloc_local([1], "int32")
        if val is not None:
            Tx.buffer_store(buf, val, 0)
        return buf

    def float_var(val=None):
        buf = Tx.alloc_local([1], "float32")
        if val is not None:
            Tx.buffer_store(buf, val, 0)
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
        NUM_STAGES = 1
        WG_COUNT = 2
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

        def scope_sync(wg_id):
            return Tx.cuda.warpgroup_sync(wg_id)
            # return Tx.cuda.cta_sync()

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
            return Tx.cuda.func_call(func_name, C_ptr, A_ptr, source_code=source_code)

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
            return Tx.cuda.func_call(func_name, dst_ptr, src_ptr, source_code=source_code)

        def get_sm_scale():
            func_name = "get_sm_scale"
            source_code = f"""
__device__ __forceinline__ float {func_name}() {{
  return 1.44269504088896340736 * 1 / sqrtf({HEAD_DIM});
}}
"""
            return Tx.cuda.func_call(func_name, source_code=source_code, return_type="float32")

        # fmt: off
        @Tx.prim_func(tirx=True)
        def batch_attention(
            q_ptr: Tx.handle,
            kv_ptr: Tx.handle,
            q_indptr_ptr: Tx.handle,
            kv_indptr_ptr: Tx.handle,
            partial_indptr_ptr: Tx.handle,
            kv_indices_ptr: Tx.handle,
            q_len_ptr: Tx.handle,
            kv_len_ptr: Tx.handle,
            q_start_ptr: Tx.handle,
            kv_start_ptr: Tx.handle,
            kv_end_ptr: Tx.handle,
            kv_head_idx_ptr: Tx.handle,
            work_indptr_ptr: Tx.handle,
            len_kv_chunk_ptr: Tx.handle,
            o_ptr: Tx.handle,
            partial_o_ptr: Tx.handle,
            partial_lse_ptr: Tx.handle,
        ):
            batch_size = Tx.int32()
            max_page_num = Tx.int64()
            total_page_num = Tx.int32()

            q_buf = Tx.match_buffer(q_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16")
            kv_buf = Tx.match_buffer(kv_ptr, [max_page_num, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM], "float16")  # noqa: E501
            q_indptr_buf = Tx.match_buffer(q_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_indptr_buf = Tx.match_buffer(kv_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            partial_indptr_buf = Tx.match_buffer(partial_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_indices_buf = Tx.match_buffer(kv_indices_ptr, [total_page_num], "int32", offset_factor=1)  # noqa: E501
            q_len_buf = Tx.match_buffer(q_len_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_len_buf = Tx.match_buffer(kv_len_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            q_start_buf = Tx.match_buffer(q_start_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_start_buf = Tx.match_buffer(kv_start_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_end_buf = Tx.match_buffer(kv_end_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            kv_head_idx_buf = Tx.match_buffer(kv_head_idx_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            work_indptr_buf = Tx.match_buffer(work_indptr_ptr, [MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)  # noqa: E501
            len_kv_chunk_buf = Tx.match_buffer(len_kv_chunk_ptr, [2], "int32", offset_factor=1)
            o_buf = Tx.match_buffer(o_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16")
            partial_o_buf = Tx.match_buffer(partial_o_ptr, [MAX_NUM_KV_SPLITS * head_dim * kv_heads], "float16", offset_factor=1)  # noqa: E501
            partial_lse_buf = Tx.match_buffer(partial_lse_ptr, [MAX_NUM_KV_SPLITS * kv_heads], "float32", offset_factor=1)  # noqa: E501

            with Tx.kernel():
                bx = Tx.cta_id([SM_COUNT * 2 // WG_COUNT], parent="kernel")
                wg_id = Tx.warpgroup_id([m], parent="cta")  # noqa: F821
                warp_id = Tx.warp_id([NUM_WARPS_Q * NUM_WARPS_KV], parent="warpgroup")
                lane_id = Tx.thread_id([32], parent="warp")

                pool = Tx.PoolAllocator()
                q_smem = pool.alloc([WG_COUNT * CTA_TILE_Q * HEAD_DIM], "float16", align=16)
                k_smem = pool.alloc([WG_COUNT * CTA_TILE_KV * HEAD_DIM], "float16", align=16)
                v_smem = pool.alloc([WG_COUNT * CTA_TILE_KV * HEAD_DIM], "float16", align=16)
                # pool.move_base_to(0)
                cta_sync_o_smem = pool.alloc([WG_COUNT, 1] if NUM_WARPS_KV == 1 else [WG_COUNT, NUM_WARPS, NUM_MMA_Q, NUM_MMA_D_VO, 32, 8], "float32", align=16)  # noqa: E501
                cta_sync_md_smem = pool.alloc([WG_COUNT, 1] if NUM_WARPS_KV == 1 else [WG_COUNT, NUM_WARPS, NUM_MMA_Q, 16, 2], "float32", align=16)  # noqa: E501
                # pool.move_base_to(0)
                smem_o = pool.alloc([WG_COUNT * CTA_TILE_Q * HEAD_DIM], "float16", align=16)
                pool.commit()
                print(pool.offset)

                with Tx.thread():
                    s_frag = Tx.alloc_local([NUM_MMA_Q, NUM_MMA_KV, 8], "float32", align=0)
                    o_frag = Tx.alloc_local([NUM_MMA_Q, NUM_MMA_D_VO, 8], "float32", align=16)
                    m = Tx.alloc_local([NUM_MMA_Q, 2], "float32")
                    d = Tx.alloc_local([NUM_MMA_Q, 2], "float32")
                    tid = Tx.alloc_local([3], "int32")

                    tid[0] = lane_id
                    tid[1] = warp_id % NUM_WARPS_Q
                    tid[2] = warp_id // NUM_WARPS_Q

                    q_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_Q, wg_id * CTA_TILE_Q + get_warp_idx_q(tid) * NUM_MMA_Q * 16 + lane_id % 16, lane_id // 16))  # noqa: E501
                    k_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_K, wg_id * CTA_TILE_KV + get_warp_idx_kv(tid) * NUM_MMA_KV * 16 + 8 * (lane_id // 16) + lane_id % 8, (lane_id % 16 // 8)))  # noqa: E501
                    v_smem_offset_r = int_var(get_permuted_offset(UPCAST_STRIDE_V, wg_id * CTA_TILE_KV + get_warp_idx_kv(tid) * NUM_MMA_KV * 16 + lane_id % 16, lane_id // 16))  # noqa: E501
                    k_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_K, wg_id * CTA_TILE_KV + warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL, lane_id % KV_THR_LAYOUT_COL))  # noqa: E501
                    v_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_V, wg_id * CTA_TILE_KV + warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL, lane_id % KV_THR_LAYOUT_COL))  # noqa: E501
                    thr_local_kv_offset = Tx.alloc_local([NUM_MMA_KV * KV_THR_LAYOUT_COL // 2 // NUM_WARPS_Q], "int64")  # noqa: E501

                    work_idx = int_var()
                    work_idx[0] = work_indptr_buf[bx * WG_COUNT + wg_id]
                    while work_idx[0] < work_indptr_buf[bx * WG_COUNT + wg_id + 1]:
                        with Tx.thread():
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
                            len_kv_chunk = int_var(len_kv_chunk_buf[1])

                            kv_chunk_idx = int_var(ceildiv(kv_start[0], len_kv_chunk[0]))
                            num_kv_chunks = int_var(ceildiv(kv_len[0], len_kv_chunk[0]))
                            qo_packed_idx_base = int_var(packed_qo_start[0] + get_warp_idx_q(tid) * NUM_MMA_Q * 16)  # noqa: E501
                            qo_upperbound = int_var(Tx.min(q_len[0], ceildiv(qo_packed_idx_base[0] + CTA_TILE_Q, GQA_GROUP_SIZE)))  # noqa: E501

                            @Tx.inline
                            def init_states():
                                for i0, i1, i2 in Tx.grid(NUM_MMA_Q, NUM_MMA_D_VO, 8):
                                    o_frag[i0, i1, i2] = Tx.float32(0)
                                for i0, i1 in Tx.grid(NUM_MMA_Q, 2):
                                    m[i0, i1] = Tx.float32(-INF)
                                    d[i0, i1] = Tx.float32(0)

                            init_states()

                            @Tx.inline
                            def load_q_global_smem():
                                if get_warp_idx_kv(tid) == 0:
                                    q_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_Q, wg_id * CTA_TILE_Q + get_warp_idx_q(tid) * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))  # noqa: E501
                                    # unroll
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for j in Tx.unroll(4):
                                            with Tx.thread():
                                                qo_packed_id = Tx.meta_var(qo_packed_idx_base[0] + lane_id // 8 + mma_q * 16 + j * 4)  # noqa: E501
                                                q = int_var(Tx.floordiv(qo_packed_id, GQA_GROUP_SIZE))  # noqa: E501
                                                r = int_var(Tx.floormod(qo_packed_id, GQA_GROUP_SIZE))  # noqa: E501
                                                for mma_do in Tx.unroll(NUM_MMA_D_QK // 4):
                                                    Tx.ptx.cp_async(q_smem.ptr_to([q_smem_offset_w[0] * upcast_size("float16")]),  # noqa: E501
                                                                   # TODO: optimize the addr computation here  # noqa: E501
                                                                   q_buf.ptr_to([q_indptr[0] + q[0], kv_head_idx[0] * GQA_GROUP_SIZE + r[0], (lane_id % 8 + mma_do * 8) * upcast_size("float16")]), cp_size=16, prefetch_size=128,  # noqa: E501
                                                                   predicate=q[0] < qo_upperbound[0])  # noqa: E501
                                                    q_smem_offset_w[0] = advance_offset_by_column(8, q_smem_offset_w[0], mma_do)  # noqa: E501
                                            q_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_Q, q_smem_offset_w[0]) - 2 * NUM_MMA_D_QK  # noqa: E501

                            load_q_global_smem()

                            kv_tile_idx = int_var(ceildiv(kv_end[0], CTA_TILE_KV) - 1 - (kv_start[0] // CTA_TILE_KV))  # noqa: E501
                            mast_tile_idx = int_var(kv_end[0] // CTA_TILE_KV - (kv_start[0] // CTA_TILE_KV))  # noqa: E501

                            block_iter_base = int_var(kv_indptr[0] * PAGE_SIZE + kv_start[0])
                            scope_sync(wg_id)
                            packed_kv_bound = int_var(kv_indptr[0] * PAGE_SIZE + kv_len[0])

                            @Tx.inline
                            def prefetch_offset(packed_block_iter_base_in):
                                with Tx.thread():
                                    packed_block_iter_base = int_var(packed_block_iter_base_in)
                                    for i in Tx.unroll(NUM_MMA_KV * 4 // NUM_WARPS_Q):
                                        packed_block_iter = int_var(packed_block_iter_base[0] + warp_id * KV_THR_LAYOUT_ROW + lane_id // KV_THR_LAYOUT_COL  # noqa: E501
                                                                    + KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i)  # noqa: E501
                                        page_iter = int_var(Tx.floordiv(packed_block_iter[0], PAGE_SIZE))  # noqa: E501
                                        entry_idx = int_var(Tx.floormod(packed_block_iter[0], PAGE_SIZE))  # noqa: E501
                                        mapped_page = Tx.meta_var(Tx.if_then_else(packed_block_iter[0] < packed_kv_bound[0], kv_indices_buf[page_iter[0]], 0))  # noqa: E501
                                        thr_local_kv_offset[i] = kv_buf.elem_offset_of([mapped_page, 0, kv_head_idx[0], entry_idx[0], (lane_id % KV_THR_LAYOUT_COL) * upcast_size("float16")])  # noqa: E501

                            @Tx.inline
                            def page_produce_kv(produce_v: bool, kv_idx_base_in, smem_offset, smem):
                                v_offset = KV_HEADS * PAGE_SIZE * HEAD_DIM if produce_v else 0
                                fill_mode = Tx.meta_var("zero" if produce_v else "")
                                NUM_MMA_D = Tx.meta_var(NUM_MMA_D_QK if produce_v else NUM_MMA_D_VO)
                                UPCAST_STRIDE = Tx.meta_var(UPCAST_STRIDE_V if produce_v else UPCAST_STRIDE_K)  # noqa: E501
                                with Tx.thread():
                                    kv_idx_base = int_var(kv_idx_base_in)
                                    kv_idx = int_var(kv_idx_base[0] + warp_id * 4 + lane_id // 8)
                                    kv_buf_1d = kv_buf.view(-1)
                                    # unroll
                                    for i in Tx.unroll(NUM_MMA_KV * 4 // NUM_WARPS_Q):
                                        for j in Tx.unroll(NUM_MMA_D // (8 // size_of("float16"))):
                                            Tx.ptx.cp_async(
                                                smem.ptr_to([smem_offset[0] * upcast_size("float16")]),  # noqa: E501
                                                # TODO: optimize the addr computation here
                                                kv_buf_1d.ptr_to([v_offset + thr_local_kv_offset[i] + 8 * j * upcast_size("float16")]), cp_size=16, prefetch_size=128, fill_mode=fill_mode,  # noqa: E501
                                                predicate=kv_idx[0] < kv_len[0]
                                            )
                                            smem_offset[0] = advance_offset_by_column(8, smem_offset[0], j)  # noqa: E501
                                        kv_idx[0] += NUM_WARPS * 4
                                        smem_offset[0] = advance_offset_by_row(NUM_WARPS * 4, UPCAST_STRIDE, smem_offset[0]) - size_of("float16") * NUM_MMA_D  # noqa: E501
                                    smem_offset[0] -= CTA_TILE_KV * UPCAST_STRIDE

                            @Tx.inline
                            def mma_sync_m16n16k16_row_col_f16f16f32(C_in, c_offset, A_in, a_offset, B_in, b_offset, init: bool):  # noqa: E501
                                with Tx.thread():
                                    C_mma = Tx.decl_buffer([8], dtype="float32", data=C_in.data, byte_offset=c_offset)  # noqa: E501
                                    A_mma = Tx.decl_buffer([4], dtype="uint32", data=A_in.data, byte_offset=a_offset)  # noqa: E501
                                    B_mma = Tx.decl_buffer([4], dtype="uint32", data=B_in.data, byte_offset=b_offset)  # noqa: E501
                                    if init:
                                        Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",  # noqa: E501
                                                C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]))  # noqa: E501
                                        Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",  # noqa: E501
                                                C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]))  # noqa: E501
                                    else:
                                        Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",  # noqa: E501
                                                C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]), C_mma.ptr_to([0]))  # noqa: E501
                                        Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",  # noqa: E501
                                                C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]), C_mma.ptr_to([4]))  # noqa: E501

                            @Tx.inline
                            def compute_qk():
                                with Tx.thread():
                                    a_frag = Tx.alloc_local([NUM_MMA_Q, 4], "uint32")
                                    b_frag = Tx.alloc_local([4], "uint32")

                                    # unroll
                                    for mma_d in Tx.unroll(NUM_MMA_D_QK):
                                        for mma_q in Tx.unroll(NUM_MMA_Q):
                                            Tx.ptx.ldmatrix(False, 4, ".b16", a_frag.ptr_to([mma_q, 0]), q_smem.ptr_to([q_smem_offset_r[0] * upcast_size("float16")]))  # noqa: E501
                                            q_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_Q, q_smem_offset_r[0])  # noqa: E501
                                        q_smem_offset_r[0] = advance_offset_by_column(2, q_smem_offset_r[0], mma_d) - NUM_MMA_Q * 16 * UPCAST_STRIDE_Q  # noqa: E501
                                        for mma_kv in Tx.unroll(NUM_MMA_KV):
                                            Tx.ptx.ldmatrix(False, 4, ".b16", b_frag.ptr_to([0]), k_smem.ptr_to([k_smem_offset_r[0] * upcast_size("float16")]))  # noqa: E501
                                            k_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_K, k_smem_offset_r[0])  # noqa: E501
                                            for mma_q in Tx.unroll(NUM_MMA_Q):
                                                if mma_d == 0:
                                                    mma_sync_m16n16k16_row_col_f16f16f32(s_frag, s_frag.byte_offset_of([mma_q, mma_kv, 0]),  # noqa: E501
                                                                                         a_frag, a_frag.byte_offset_of([mma_q, 0]),  # noqa: E501
                                                                                         b_frag, b_frag.byte_offset_of([0]), True)  # noqa: E501
                                                else:
                                                    mma_sync_m16n16k16_row_col_f16f16f32(s_frag, s_frag.byte_offset_of([mma_q, mma_kv, 0]),  # noqa: E501
                                                                                         a_frag, a_frag.byte_offset_of([mma_q, 0]),  # noqa: E501
                                                                                         b_frag, b_frag.byte_offset_of([0]), False)  # noqa: E501
                                        k_smem_offset_r[0] = advance_offset_by_column(2, k_smem_offset_r[0], mma_d) - NUM_MMA_KV * 16 * UPCAST_STRIDE_K  # noqa: E501
                                    q_smem_offset_r[0] -= NUM_MMA_D_QK * 2
                                    k_smem_offset_r[0] -= NUM_MMA_D_QK * size_of("float16")

                            @Tx.inline
                            def logits_mask():
                                chunk_end = Tx.meta_var(kv_end[0])
                                with Tx.thread():
                                    # unroll
                                    kv_idx_base = int_var(kv_start[0] + (kv_tile_idx[0] * NUM_WARPS_KV + get_warp_idx_kv(tid)) * NUM_MMA_KV * 16)  # noqa: E501
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for mma_kv in Tx.unroll(NUM_MMA_KV):
                                            for reg_id in Tx.unroll(8):
                                                with Tx.thread():
                                                    kv_idx = int_var(kv_idx_base[0] + mma_kv * 16 + 2 * (lane_id % 4) + 8 * (reg_id // 4) + reg_id % 2)  # noqa: E501
                                                    s_frag[mma_q, mma_kv, reg_id] = Tx.if_then_else(Tx.Not(kv_idx[0] >= chunk_end), s_frag[mma_q, mma_kv, reg_id], Tx.float32(-INF))  # noqa: E501

                            @Tx.inline
                            def update_mdo_states():
                                WARP_MASK = Tx.meta_var(0xFFFFFFFF)
                                with Tx.thread():
                                    sm_scale = float_var(get_sm_scale())
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for j in Tx.unroll(2):
                                            m_prev = float_var(m[mma_q, j])
                                            for mma_kv in Tx.unroll(NUM_MMA_KV):
                                                m_local = float_var(Tx.max(Tx.max(s_frag[mma_q, mma_kv, j * 2 + 0], s_frag[mma_q, mma_kv, j * 2 + 1]),  # noqa: E501
                                                                          Tx.max(s_frag[mma_q, mma_kv, j * 2 + 4], s_frag[mma_q, mma_kv, j * 2 + 5])))  # noqa: E501
                                                m[mma_q, j] = Tx.max(m[mma_q, j], m_local[0])
                                            m[mma_q, j] = Tx.max(m[mma_q, j], Tx.tvm_warp_shuffle_xor(WARP_MASK, m[mma_q, j], 0x2, 32, 32))  # noqa: E501
                                            m[mma_q, j] = Tx.max(m[mma_q, j], Tx.tvm_warp_shuffle_xor(WARP_MASK, m[mma_q, j], 0x1, 32, 32))  # noqa: E501

                                            o_scale = float_var(ptx_exp2(m_prev[0] * sm_scale[0] - m[mma_q, j] * sm_scale[0]))  # noqa: E501
                                            d[mma_q, j] *= o_scale[0]
                                            # unroll
                                            for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                                o_frag[mma_q, mma_d, j * 2 + 0] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 1] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 4] *= o_scale[0]
                                                o_frag[mma_q, mma_d, j * 2 + 5] *= o_scale[0]
                                            # unroll
                                            for mma_kv in Tx.unroll(NUM_MMA_KV):
                                                s_frag[mma_q, mma_kv, j * 2 + 0] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 0] * sm_scale[0] - m[mma_q, j] * sm_scale[0])  # noqa: E501
                                                s_frag[mma_q, mma_kv, j * 2 + 1] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 1] * sm_scale[0] - m[mma_q, j] * sm_scale[0])  # noqa: E501
                                                s_frag[mma_q, mma_kv, j * 2 + 4] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 4] * sm_scale[0] - m[mma_q, j] * sm_scale[0])  # noqa: E501
                                                s_frag[mma_q, mma_kv, j * 2 + 5] = ptx_exp2(s_frag[mma_q, mma_kv, j * 2 + 5] * sm_scale[0] - m[mma_q, j] * sm_scale[0])  # noqa: E501

                            @Tx.inline
                            def compute_sfm_v():
                                with Tx.thread():
                                    s_frag_f16 = Tx.alloc_local([NUM_MMA_Q, NUM_MMA_KV, 8], "float16")  # noqa: E501
                                    Tx.cast(s_frag_f16[:, :, :], s_frag[:, :, :])
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for mma_kv in Tx.unroll(NUM_MMA_KV):
                                            m16k16_row_sum_f16f16f32(d.ptr_to([mma_q, 0]), s_frag_f16.ptr_to([mma_q, mma_kv, 0]))  # noqa: E501
                                    for mma_kv in Tx.unroll(NUM_MMA_KV):
                                        for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                            with Tx.thread():
                                                b_frag = Tx.alloc_local([4], "uint32")
                                                Tx.ptx.ldmatrix(True, 4, ".b16", b_frag.ptr_to([0]), v_smem.ptr_to([v_smem_offset_r[0] * upcast_size("float16")]))  # noqa: E501
                                                for mma_q in Tx.unroll(NUM_MMA_Q):
                                                    mma_sync_m16n16k16_row_col_f16f16f32(
                                                        o_frag, o_frag.byte_offset_of([mma_q, mma_d, 0]),  # noqa: E501
                                                        s_frag_f16, s_frag_f16.byte_offset_of([mma_q, mma_kv, 0]),  # noqa: E501
                                                        b_frag, b_frag.byte_offset_of([0]), False
                                                    )
                                                v_smem_offset_r[0] = advance_offset_by_column(2, v_smem_offset_r[0], mma_d)  # noqa: E501
                                        v_smem_offset_r[0] = advance_offset_by_row(16, UPCAST_STRIDE_V, v_smem_offset_r[0]) - size_of("float16") * NUM_MMA_D_VO  # noqa: E501
                                    v_smem_offset_r[0] -= 16 * NUM_MMA_KV * UPCAST_STRIDE_V

                            @Tx.inline
                            def loop_body(WITH_MASK: bool):
                                prefetch_offset(block_iter_base[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV)  # noqa: E501
                                Tx.ptx.cp_async.wait_group(1)
                                scope_sync(wg_id)

                                compute_qk()
                                if WITH_MASK:
                                    logits_mask()
                                update_mdo_states()

                                scope_sync(wg_id)
                                page_produce_kv(False, kv_start[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV, k_smem_offset_w, k_smem)  # noqa: E501
                                Tx.ptx.cp_async.commit_group()
                                Tx.ptx.cp_async.wait_group(1)

                                scope_sync(wg_id)
                                compute_sfm_v()
                                scope_sync(wg_id)

                                page_produce_kv(True, kv_start[0] + (kv_tile_idx[0] - 1) * CTA_TILE_KV, v_smem_offset_w, v_smem)  # noqa: E501
                                Tx.ptx.cp_async.commit_group()

                            prefetch_offset(block_iter_base[0] + kv_tile_idx[0] * CTA_TILE_KV)
                            page_produce_kv(False, kv_start[0] + kv_tile_idx[0] * CTA_TILE_KV, k_smem_offset_w, k_smem)  # noqa: E501
                            Tx.ptx.cp_async.commit_group()
                            page_produce_kv(True, kv_start[0] + kv_tile_idx[0] * CTA_TILE_KV, v_smem_offset_w, v_smem)  # noqa: E501
                            Tx.ptx.cp_async.commit_group()

                            while kv_tile_idx[0] >= mast_tile_idx[0] and kv_tile_idx[0] > 0:
                                loop_body(True)
                                kv_tile_idx[0] -= 1

                            while kv_tile_idx[0] + 1 > NUM_STAGES:
                                loop_body(False)
                                kv_tile_idx[0] -= 1

                            Tx.ptx.cp_async.wait_group(0)
                            scope_sync(wg_id)

                            while kv_tile_idx[0] >= 0:
                                compute_qk()
                                logits_mask()
                                update_mdo_states()
                                compute_sfm_v()
                                kv_tile_idx[0] -= 1

                            scope_sync(wg_id)

                            @Tx.inline
                            def finalize_m():
                                with Tx.thread():
                                    sm_scale = float_var(get_sm_scale())
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for j in Tx.unroll(2):
                                            if m[mma_q, j] != -INF:
                                                m[mma_q, j] *= sm_scale[0]

                            @Tx.inline
                            def threadblock_sync_mdo_states():
                                for mma_q in Tx.unroll(NUM_MMA_Q):
                                    for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                        Tx.copy(cta_sync_o_smem[wg_id, warp_id, mma_q, mma_d, lane_id, :], o_frag[mma_q, mma_d, :])  # noqa: E501

                                for mma_q in Tx.unroll(NUM_MMA_Q):
                                    for j in Tx.unroll(2):
                                        with Tx.thread():
                                            md = Tx.alloc_local([2], "float32")
                                            md[0] = m[mma_q, j]
                                            md[1] = d[mma_q, j]
                                            Tx.copy(cta_sync_md_smem[wg_id, warp_id, mma_q, j * 8 + lane_id // 4, :], md[:])  # noqa: E501
                                scope_sync(wg_id)

                                for mma_q in Tx.unroll(NUM_MMA_Q):
                                    with Tx.thread():
                                        o_scale = Tx.alloc_local([2, NUM_WARPS_KV], "float32")
                                        for j in Tx.unroll(2):
                                            with Tx.thread():
                                                m_new = float_var(-INF)
                                                d_new = float_var(1.0)
                                                for i in Tx.unroll(NUM_WARPS_KV):
                                                    with Tx.thread():
                                                        md = Tx.alloc_local([2], "float32")
                                                        Tx.copy(md[:], cta_sync_md_smem[wg_id, i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, j * 8 + lane_id // 4, :])  # noqa: E501
                                                        m_prev = float_var(m_new[0])
                                                        d_prev = float_var(d_new[0])
                                                        m_new[0] = Tx.max(m_new[0], md[0])
                                                        d_new[0] = d_prev[0] * ptx_exp2(m_prev[0] - m_new[0]) + md[1] * ptx_exp2(md[0] - m_new[0])  # noqa: E501
                                                for i in Tx.unroll(NUM_WARPS_KV):
                                                    with Tx.thread():
                                                        md = Tx.alloc_local([2], "float32")
                                                        Tx.copy(md[:], cta_sync_md_smem[wg_id, i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, j * 8 + lane_id // 4, :])  # noqa: E501
                                                        o_scale[j, i] = ptx_exp2(md[0] - m_new[0])
                                                m[mma_q, j] = m_new[0]
                                                d[mma_q, j] = d_new[0]
                                        for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                            with Tx.thread():
                                                o_new = Tx.alloc_local([8], "float32")
                                                for i in Tx.unroll(8):
                                                    o_new[i] = 0.0
                                                for i in Tx.unroll(NUM_WARPS_KV):
                                                    with Tx.thread():
                                                        o_i = Tx.alloc_local([8], "float32")
                                                        Tx.copy(o_i[:], cta_sync_o_smem[wg_id, i * NUM_WARPS_Q + get_warp_idx_q(tid), mma_q, mma_d, lane_id, :])  # noqa: E501
                                                        for reg_id in Tx.unroll(8):
                                                            o_new[reg_id] += o_i[reg_id] * o_scale[(reg_id % 4) // 2, i]  # noqa: E501
                                                Tx.copy(o_frag[mma_q, mma_d, :], o_new[:])

                            @Tx.inline
                            def normalize_d():
                                with Tx.thread():
                                    d_rcp = Tx.alloc_local([NUM_MMA_Q, 2], "float32")
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for j in Tx.unroll(2):
                                            d_rcp[mma_q, j] = Tx.if_then_else(m[mma_q, j] != -INF, ptx_rcp(d[mma_q, j]), 0.0)  # noqa: E501
                                    for mma_q in Tx.unroll(NUM_MMA_Q):
                                        for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                            for reg_id in Tx.unroll(8):
                                                o_frag[mma_q, mma_d, reg_id] *= d_rcp[mma_q, (reg_id >> 1) & 1]  # noqa: E501

                            @Tx.inline
                            def store_o_to_smem(o_smem):
                                for mma_q in Tx.unroll(NUM_MMA_Q):
                                    for mma_d in Tx.unroll(NUM_MMA_D_VO):
                                        with Tx.thread():
                                            o_frag_f16 = Tx.alloc_local([8], "float16")
                                            Tx.cast(o_frag_f16[:], o_frag[mma_q, mma_d, :])
                                            o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, wg_id * CTA_TILE_Q + (get_warp_idx_q(tid) * NUM_MMA_Q + mma_q) * 16 + lane_id % 16, mma_d * 2 + lane_id // 16))  # noqa: E501
                                            Tx.ptx.stmatrix(4, False, o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]), o_frag_f16.ptr_to([0]))  # noqa: E501

                            @Tx.inline
                            def write_partial_o(o_ptr_base_offset, o_stride_n):
                                o_packed_idx_base_warp = Tx.meta_var(qo_packed_idx_base[0])
                                o_packed_idx_base_cta = Tx.meta_var(packed_qo_start[0])
                                o_smem = Tx.meta_var(smem_o)
                                warp_id_x = int_var(get_warp_idx_q(tid))
                                warp_id_z = int_var(get_warp_idx_kv(tid))

                                if warp_id_z[0] == 0:
                                    with Tx.thread():
                                        store_o_to_smem(o_smem)
                                        o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, wg_id * CTA_TILE_Q + warp_id_x[0] * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))  # noqa: E501
                                        for mma_q in Tx.unroll(NUM_MMA_Q):
                                            for j in Tx.unroll(4):
                                                with Tx.thread():
                                                    o_packed_idx = int_var(o_packed_idx_base_warp + lane_id // 8 + mma_q * 16 + j * 4)  # noqa: E501
                                                    q = int_var(Tx.floordiv(o_packed_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    int_var(Tx.floormod(o_packed_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    o_ptr_offset = int_var(o_ptr_base_offset + (o_packed_idx[0] - o_packed_idx_base_cta) * o_stride_n + (lane_id % 8) * upcast_size("float16"))  # noqa: E501
                                                    for mma_do in Tx.unroll(NUM_MMA_D_VO // 4):
                                                        if q[0] < qo_upperbound[0]:
                                                            store_128b(partial_o_buf.ptr_to([o_ptr_offset[0]]), o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]))  # noqa: E501
                                                        o_ptr_offset[0] += 8 * upcast_size("float16")  # noqa: E501
                                                        o_smem_offset_w[0] = advance_offset_by_column(8, o_smem_offset_w[0], mma_do)  # noqa: E501
                                                    o_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_O, o_smem_offset_w[0]) - 2 * NUM_MMA_D_VO  # noqa: E501

                            @Tx.inline
                            def write_final_o(o_ptr_base_offset):
                                o_packed_idx_base = Tx.meta_var(qo_packed_idx_base[0])
                                o_smem = Tx.meta_var(smem_o)
                                warp_id_x = int_var(get_warp_idx_q(tid))
                                warp_id_z = int_var(get_warp_idx_kv(tid))

                                if warp_id_z[0] == 0:
                                    with Tx.thread():
                                        store_o_to_smem(o_smem)
                                        o_smem_offset_w = int_var(get_permuted_offset(UPCAST_STRIDE_O, wg_id * CTA_TILE_Q + warp_id_x[0] * NUM_MMA_Q * 16 + lane_id // 8, lane_id % 8))  # noqa: E501
                                        for mma_q in Tx.unroll(NUM_MMA_Q):
                                            for j in Tx.unroll(4):
                                                with Tx.thread():
                                                    o_packed_idx = int_var(o_packed_idx_base + lane_id // 8 + mma_q * 16 + j * 4)  # noqa: E501
                                                    q = int_var(Tx.floordiv(o_packed_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    r = int_var(Tx.floormod(o_packed_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    o_ptr_offset = int_var(o_ptr_base_offset + o_buf.elem_offset_of([q[0], r[0], (lane_id % 8) * upcast_size("float16")]))  # noqa: E501
                                                    for mma_do in Tx.unroll(NUM_MMA_D_VO // 4):
                                                        if q[0] < qo_upperbound[0]:
                                                            o_buf_1d = o_buf.view(-1)
                                                            store_128b(o_buf_1d.ptr_to([o_ptr_offset[0]]), o_smem.ptr_to([o_smem_offset_w[0] * upcast_size("float16")]))  # noqa: E501
                                                        o_ptr_offset[0] += 8 * upcast_size("float16")  # noqa: E501
                                                        o_smem_offset_w[0] = advance_offset_by_column(8, o_smem_offset_w[0], mma_do)  # noqa: E501
                                                    o_smem_offset_w[0] = advance_offset_by_row(4, UPCAST_STRIDE_O, o_smem_offset_w[0]) - 2 * NUM_MMA_D_VO  # noqa: E501

                            @Tx.inline
                            def write_partial_lse():
                                if num_kv_chunks[0] > 1:
                                    if get_warp_idx_kv(tid) == 0:
                                        for mma_q in Tx.unroll(NUM_MMA_Q):
                                            for j in Tx.unroll(2):
                                                with Tx.thread():
                                                    packed_qo_idx = int_var(qo_packed_idx_base[0] + lane_id // 4 + j * 8 + mma_q * 16)  # noqa: E501
                                                    q = int_var(Tx.floordiv(packed_qo_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    int_var(Tx.floormod(packed_qo_idx[0], GQA_GROUP_SIZE))  # noqa: E501
                                                    if q[0] < qo_upperbound[0]:
                                                        partial_lse_buf_offset = Tx.meta_var((o_indptr[0] + (packed_qo_idx[0] - packed_qo_start[0]) * num_kv_chunks[0] + kv_chunk_idx[0]) * KV_HEADS + kv_head_idx[0])  # noqa: E501
                                                        partial_lse_buf[partial_lse_buf_offset] = ptx_log2(d[mma_q, j]) + Tx.cast(m[mma_q, j], "float32")  # noqa: E501

                            finalize_m()
                            threadblock_sync_mdo_states()
                            normalize_d()

                            if num_kv_chunks[0] > 1:
                                # reuse q, k, v's smem
                                with Tx.thread():
                                    o_ptr_base_offset = int_var(((o_indptr[0] + kv_chunk_idx[0]) * KV_HEADS + kv_head_idx[0]) * HEAD_DIM)  # noqa: E501
                                    write_partial_o(o_ptr_base_offset[0], num_kv_chunks[0] * KV_HEADS * HEAD_DIM)  # noqa: E501
                            else:
                                with Tx.thread():
                                    o_ptr_base_offset = int_var(o_buf.elem_offset_of([q_indptr[0], kv_head_idx[0] * GQA_GROUP_SIZE, 0]))  # noqa: E501
                                    write_final_o(o_ptr_base_offset[0])

                            write_partial_lse()
                            scope_sync(wg_id)

                            work_idx[0] += 1
        # fmt: on
        return batch_attention

    def craft_batch_attention_merge_kernel():
        NUM_THREADS = 256
        NUM_WARPS = NUM_THREADS // 32
        NUM_SMEM_STAGES = 1
        VEC_SIZE = max(16 // size_of("float16"), HEAD_DIM // 32)
        BDX = HEAD_DIM // VEC_SIZE
        assert NUM_THREADS % BDX == 0
        BDY = 32 // BDX
        SMEM_SIZE = NUM_WARPS * NUM_SMEM_STAGES * BDY * HEAD_DIM * size_of(
            "float16"
        ) + NUM_THREADS * size_of("float32")
        NUM_WORKERS = SM_COUNT * NUM_WARPS

        @Tx.meta_class
        class State:
            def __init__(self, vec_size):
                self.vec_size = vec_size
                self.o = Tx.alloc_local([vec_size], "float32", name="state_o")
                self.m = Tx.alloc_local([1], "float32", name="state_m")
                self.d = Tx.alloc_local([1], "float32", name="state_d")

            @Tx.inline
            def init(self):
                with Tx.thread():
                    for i in Tx.unroll(self.vec_size):
                        self.o[i] = 0.0
                    self.m[0] = -INF
                    self.d[0] = 1.0

            # fmt: off
            @Tx.inline
            def merge(self, other_o, other_m, other_d):
                with Tx.thread():
                    m_prev = float_var(self.m[0])
                    d_prev = float_var(self.d[0])
                    self.m[0] = Tx.max(m_prev[0], other_m)
                    self.d[0] = d_prev[0] * ptx_exp2(m_prev[0] - self.m[0]) + other_d * ptx_exp2(other_m - self.m[0])  # noqa: E501
                    for i in Tx.unroll(self.vec_size):
                        self.o[i] = self.o[i] * ptx_exp2(m_prev[0] - self.m[0]) + other_o[i] * ptx_exp2(other_m - self.m[0])  # noqa: E501
            # fmt: on

            # fmt: off
            @Tx.inline
            def normalize(self):
                with Tx.thread():
                    for i in Tx.unroll(self.vec_size):
                        self.o[i] = fdivdef(self.o[i], self.d[0])
            # fmt: on

            def get_lse(self):
                return self.m[0] + ptx_log2(self.d[0])

        # fmt: off
        @Tx.prim_func(tirx=True)
        def batch_attention_merge(
            partial_o_ptr: Tx.handle,
            final_o_ptr: Tx.handle,
            partial_lse_ptr: Tx.handle,
            num_qo_len_ptr: Tx.handle,
            merge_indptr_ptr: Tx.handle,
            merge_o_indices_ptr: Tx.handle,
        ):
            batch_size = Tx.int32()

            partial_o_buf = Tx.match_buffer(partial_o_ptr, [MAX_NUM_KV_SPLITS * head_dim * kv_heads], "float16", offset_factor=1)  # noqa: E501
            final_o_buf = Tx.match_buffer(final_o_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16")
            partial_lse_buf = Tx.match_buffer(partial_lse_ptr, [MAX_NUM_KV_SPLITS * kv_heads], "float32", offset_factor=1)  # noqa: E501
            num_qo_len_buf = Tx.match_buffer(num_qo_len_ptr, [1], "int32", offset_factor=1)
            merge_indptr_buf = Tx.match_buffer(merge_indptr_ptr, [MAX_NUM_KV_SPLITS], "int32", offset_factor=1)  # noqa: E501
            merge_o_indices_buf = Tx.match_buffer(merge_o_indices_ptr, [MAX_NUM_KV_SPLITS], "int32", offset_factor=1)  # noqa: E501
            with Tx.kernel():
                bx = Tx.cta_id([SM_COUNT], parent="kernel")
                warp_id = Tx.warp_id([NUM_THREADS // 32], parent="cta")
                lane_id = Tx.thread_id([32], parent="warp")

                with Tx.thread():
                    tx = int_var(lane_id % BDX)
                    ty = int_var(lane_id // BDX)
                    worker_id = int_var(bx * NUM_WARPS + warp_id)

                    buf = Tx.alloc_shared([SMEM_SIZE], "uint8")
                    pool = Tx.PoolAllocator(buf.data)
                    v_smem = pool.alloc([NUM_WARPS, NUM_SMEM_STAGES, BDY, HEAD_DIM], "float16")
                    s_smem = pool.alloc([NUM_WARPS, 32], "float32")

                    num_qo_len_local = int_var(num_qo_len_buf[0])

                    i = int_var(worker_id[0])
                    while i[0] < num_qo_len_local[0] * KV_HEADS:
                        with Tx.thread():
                            Tx.cuda.warp_sync()
                            packed_qo_idx = int_var(Tx.floordiv(i[0], KV_HEADS))
                            kv_head_idx = int_var(Tx.floormod(i[0], KV_HEADS))
                            qo_head_idx = int_var(Tx.floormod(packed_qo_idx[0], GQA_GROUP_SIZE))

                            partial_idx_to_offset = Tx.meta_var(lambda off: (merge_indptr_buf[packed_qo_idx[0]] + off) * KV_HEADS + kv_head_idx[0])  # noqa: E501
                            merge_idx_to_offset = Tx.meta_var((merge_o_indices_buf[packed_qo_idx[0]] * KV_HEADS + kv_head_idx[0]) * GQA_GROUP_SIZE + qo_head_idx[0])  # noqa: E501

                            state = State(VEC_SIZE)
                            state.init()

                            num_index_sets = int_var(merge_indptr_buf[packed_qo_idx[0] + 1] - merge_indptr_buf[packed_qo_idx[0]])  # noqa: E501

                            # prelogue
                            for it in range(NUM_SMEM_STAGES):
                                with Tx.thread():
                                    Tx.ptx.cp_async(
                                        v_smem.ptr_to([warp_id, it, ty[0], tx[0] * VEC_SIZE]),
                                        partial_o_buf.ptr_to([partial_idx_to_offset(it * BDY + ty[0]) * HEAD_DIM + tx[0] * VEC_SIZE]),  # noqa: E501
                                        cp_size=16, prefetch_size=128,
                                        predicate=it * BDY + ty[0] < num_index_sets[0]
                                    )
                                    Tx.ptx.cp_async.commit_group()

                            for it in Tx.serial(ceildiv(num_index_sets[0], BDY)):
                                with Tx.thread():
                                    if it % BDX == 0:
                                        s_smem[warp_id, ty[0] * BDX + tx[0]] = Tx.if_then_else(
                                            it * BDY + (ty[0] * BDX + tx[0]) < num_index_sets[0],
                                            partial_lse_buf[partial_idx_to_offset(it * BDY + ty[0] * BDX + tx[0])],  # noqa: E501
                                            Tx.float32(0)
                                        )
                                        Tx.cuda.warp_sync()
                                    Tx.ptx.cp_async.wait_group(NUM_SMEM_STAGES - 1)
                                    Tx.cuda.warp_sync()

                                    v = Tx.alloc_local([VEC_SIZE], "float32")
                                    cast_load(v, VEC_SIZE, v_smem, warp_id, it % NUM_SMEM_STAGES, ty[0], tx[0] * VEC_SIZE)  # noqa: E501

                                    if it * BDY + ty[0] < num_index_sets[0]:
                                        s = float_var(s_smem[warp_id, (it % BDX) * BDY + ty[0]])
                                        state.merge(v, s[0], 1)
                                    Tx.cuda.warp_sync()

                                    Tx.ptx.cp_async(
                                        v_smem.ptr_to([warp_id, (it % NUM_SMEM_STAGES), ty[0], tx[0] * VEC_SIZE]),  # noqa: E501
                                        partial_o_buf.ptr_to([partial_idx_to_offset((it + NUM_SMEM_STAGES) * BDY + ty[0]) * HEAD_DIM + tx[0] * VEC_SIZE]),  # noqa: E501
                                        cp_size=16, prefetch_size=128,
                                        predicate=(it + NUM_SMEM_STAGES) * BDY + ty[0] < num_index_sets[0]  # noqa: E501
                                    )
                                    Tx.ptx.cp_async.commit_group()

                            Tx.ptx.cp_async.wait_group(0)
                            Tx.cuda.warp_sync()

                            @Tx.inline
                            def warp_sync_state():
                                cast_store(state.o, VEC_SIZE, v_smem, warp_id, 0, ty[0], tx[0] * VEC_SIZE)  # noqa: E501
                                s_smem[warp_id, ty[0]] = state.get_lse()
                                state.init()
                                Tx.cuda.warp_sync()

                                for it in Tx.unroll(BDY):
                                    with Tx.thread():
                                        s = float_var(s_smem[warp_id, it])
                                        v = Tx.alloc_local([VEC_SIZE], "float32")
                                        cast_load(v, VEC_SIZE, v_smem, warp_id, 0, it, tx[0] * VEC_SIZE)  # noqa: E501
                                        state.merge(v, s[0], 1)

                            state.normalize()
                            if (BDY > 1):
                                warp_sync_state()
                                state.normalize()

                            final_o_buf_2d = final_o_buf.view(-1, HEAD_DIM)
                            cast_store(state.o, VEC_SIZE, final_o_buf_2d, merge_idx_to_offset, tx[0] * VEC_SIZE)  # noqa: E501

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

    NUM_TASK_ARGS = 10
    NUM_TASKS = 2
    print(f"MAX_TOTAL_NUM_WORKERS: {MAX_TOTAL_NUM_WORKERS}")
    for i in range(0, NUM_TASKS):
        q_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 2), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 3), MAX_TOTAL_NUM_WORKERS, torch.int32)
        partial_indptr = get_tensor(
            get_id(i * NUM_TASK_ARGS + 4), MAX_TOTAL_NUM_WORKERS, torch.int32
        )
        q_len = get_tensor(get_id(i * NUM_TASK_ARGS + 5), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_len = get_tensor(get_id(i * NUM_TASK_ARGS + 6), MAX_TOTAL_NUM_WORKERS, torch.int32)
        q_start = get_tensor(get_id(i * NUM_TASK_ARGS + 7), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_start = get_tensor(get_id(i * NUM_TASK_ARGS + 8), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_end = get_tensor(get_id(i * NUM_TASK_ARGS + 9), MAX_TOTAL_NUM_WORKERS, torch.int32)
        kv_head_idx = get_tensor(get_id(i * NUM_TASK_ARGS + 10), MAX_TOTAL_NUM_WORKERS, torch.int32)
        work_indptr = get_tensor(get_id(i * NUM_TASK_ARGS + 11), MAX_TOTAL_NUM_WORKERS, torch.int32)

    len_kv_chunk = get_tensor(get_id(1 * NUM_TASK_ARGS + 12), NUM_TASKS, torch.int32)
    partial_o = get_tensor(
        get_id(1 * NUM_TASK_ARGS + 13), MAX_NUM_KV_SPLITS * head_dim * kv_heads, torch.float16
    )
    partial_lse = get_tensor(
        get_id(1 * NUM_TASK_ARGS + 14), MAX_NUM_KV_SPLITS * kv_heads, torch.float32
    )
    merge_indptr = get_tensor(get_id(1 * NUM_TASK_ARGS + 15), MAX_NUM_KV_SPLITS, torch.int32)
    merge_o_indices = get_tensor(get_id(1 * NUM_TASK_ARGS + 16), MAX_NUM_KV_SPLITS, torch.int32)
    num_qo_len = get_tensor(get_id(1 * NUM_TASK_ARGS + 17), 1, torch.int32)

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

        def func():
            return wrapper.run(q_f, kv_data_f, out_f, lse_f)

        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_attention")
        func()
        print(f"FlashInfer BatchAttention time: {ms:.3f} ms")

        # return partial_o.cpu().numpy(), partial_lse.cpu().numpy(), out_f.cpu().numpy()
        return out_f.cpu().numpy()

    def flashinfer_batch_prefill():
        q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
        kv_data_f = KV_data.to(0)
        kv_indptr_f = KV_indptr.to(0)
        kv_last_page_len_f = KV_last_page_len.to(0)
        kv_indices_f = KV_indices.to(0)

        qo_indptr_f = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
        workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
        wrapper.plan(
            qo_indptr_f,
            kv_indptr_f,
            kv_indices_f,
            kv_last_page_len_f,
            qo_heads,
            kv_heads,
            head_dim,
            PAGE_SIZE,
            pos_encoding_mode="NONE",
            kv_data_type=torch.float16,
            q_data_type=torch.float16,
        )
        o, lse = wrapper.run_return_lse(q_f, kv_data_f)

        def func():
            return wrapper.run(q_f, kv_data_f)

        ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_prefill")
        func()
        print(f"FlashInfer BatchPrefillWithPagedKVCacheWrapper time: {ms:.3f} ms")

        return o.reshape(batch_size, qo_heads, head_dim).cpu().numpy()

    def tir():
        def torch_to_tvm(tensor):
            return tvm_ffi.from_dlpack(torch.to_dlpack(tensor))

        DEV = tvm.cuda(0)
        q_tvm = tvm.runtime.tensor(Q, DEV)
        kv_data_tvm = tvm.runtime.tensor(KV_data, DEV)
        q_indptr_tvm = torch_to_tvm(q_indptr)
        kv_indptr_tvm = torch_to_tvm(kv_indptr)
        partial_indptr_tvm = torch_to_tvm(partial_indptr)
        kv_indices_tvm = tvm.runtime.tensor(KV_indices, DEV)
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
            mod_attn = tvm.compile(mod_attn, target=target, tir_pipeline="tirx")
            mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirx")

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

        # return partial_o_tvm.numpy(), partial_lse_tvm.numpy(), o_tvm.numpy()
        return o_tvm.numpy()

    with ProtonContext("batch_attention"):
        print(
            f"qo_heads: {qo_heads}, kv_heads: {kv_heads}, seq_len: {seq_len}, head_dim: {head_dim}, batch_size: {batch_size}, seed: {seed}"  # noqa: E501
        )
        print("Flashinfer BatchAttention Start", flush=True)
        O_flashinfer_attention = flashinfer_batch_attention()
        print("Flashinfer BatchPrefill Start", flush=True)
        O_flashinfer_prefill = flashinfer_batch_prefill()
        print("TIR BatchAttention Start", flush=True)
        O_tir = tir()

    np.testing.assert_allclose(O_flashinfer_prefill, O_flashinfer_attention, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(O_tir, O_flashinfer_attention, rtol=1e-3, atol=1e-3)


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
