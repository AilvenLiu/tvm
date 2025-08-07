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
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tir.event import EventImpl
from ..utils import bench, ProtonContext

def ceildiv(a, b):
    return (a + b - 1) // b

def find_power_of_two(n):
    assert n > 0 and (n & (n - 1)) == 0
    return n.bit_length() - 1

# Paged kv-cache config
KV_LAYOUT = "HND"
PAGE_SIZE = 16
MAX_PAGE_NUM = 8192 + 2048

# HW config
SM_COUNT = 148
SMEM_SIZE = 232448

# Other
F16_BYTE = 2
F32_BYTE = 4


def perpare_data(batch_size, num_heads, seq_len, head_dim):
    import torch
    torch.manual_seed(42)

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
    q = torch.randn([batch_size, num_heads, head_dim]).half()
    kv_data = torch.randn([MAX_PAGE_NUM, 2, num_heads, PAGE_SIZE, head_dim]).half()

    return q, kv_data, kv_indptr, kv_last_page_len, kv_indices


class PlanInfo:
    def __init__(self, num_heads, head_dim):
        # static info
        self.max_blk_per_sm = 1
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vec_size_d = max(8, head_dim // 32) # ensure cp_async_size >= 128bit && bdx <= 32
        self.vec_size_m = max(4, head_dim // 32) # ensure cp_async_size >= 128bit && bdx <= 32
        bdx_d = self.head_dim // self.vec_size_d
        self.num_threads_d = max(256, bdx_d)
        bdy_d = self.num_threads_d // bdx_d
        self.bd_d = (bdx_d, bdy_d)
        bdx_m = self.head_dim // self.vec_size_m
        bdy_m = 128 // bdx_m
        self.num_threads_m = bdx_m * bdy_m
        self.bd_m = (bdx_m, bdy_m)
        self.pipe_d = 2
        self.pipe_m = 2
        self.tile_per_bdx = 4
        self.sm_scale = (1 / 0.6931471805599453) * (1 / head_dim ** 0.5)

        # dynamic info
        self.batch_size = 0
        self.new_batch_size = 0
        self.split_kv = False
        self.max_chunk_size = None
        self.gd_d = (0, 0)
        self.gd_m = (0,)
        self.smem_d = 0
        self.smem_m = 0
        self.request_indices_tvm = None
        self.kv_tile_indices_tvm = None
        self.o_indptr_tvm = None
        self.o_tvm = None
        self.lse_tvm = None
        self.tmp_o_tvm = None
        self.tmp_lse_tvm = None

    def plan(self, batch_size, kv_indptr_h):

        DEV = tvm.cuda(0)
        self.batch_size = batch_size

        # kernel dim config for decode kernel: [gdx, gdy] x [bdx, bdy]
        gdx_d = batch_size
        gdy_d = self.num_heads
        bdx_d, bdy_d = self.bd_d
        smem_size_d = max(2 * self.pipe_d * self.tile_per_bdx * bdy_d * self.head_dim * F16_BYTE + self.tile_per_bdx * bdy_d * bdx_d * F32_BYTE,
                            bdx_d * bdy_d * self.vec_size_d * F32_BYTE + bdy_d * 2 * F32_BYTE)
        assert smem_size_d <= SMEM_SIZE
        assert self.pipe_d <= bdx_d
        self.gd_d = (gdx_d, gdy_d)
        self.smem_d = smem_size_d

        # balance the workload (split-kv)
        if gdx_d * gdy_d >= SM_COUNT * self.max_blk_per_sm:
            split_kv = False
            max_page_num = 1
            for idx in range(batch_size):
                max_page_num = max(max_page_num, kv_indptr_h[idx + 1] - kv_indptr_h[idx])
            new_batch_size = batch_size
        else:
            page_num_list = [kv_indptr_h[idx + 1] - kv_indptr_h[idx] for idx in range(batch_size)]
            new_batch_size = batch_size
            low = max(128 // PAGE_SIZE, 1) # avoid having the sequence fragmented too much
            high = max(page_num_list)
            while low < high:
                mid = (low + high) // 2
                new_batch_size = 0
                for page_num in page_num_list:
                    new_batch_size += ceildiv(page_num, mid)
                if new_batch_size * gdy_d > SM_COUNT * self.max_blk_per_sm:
                    low = mid + 1
                else:
                    high = mid
            max_page_num = low
            new_batch_size = 0
            for page_num in page_num_list:
                new_batch_size += ceildiv(page_num, max_page_num)
            split_kv = new_batch_size != batch_size

        self.split_kv = split_kv
        self.new_batch_size = new_batch_size
        self.max_chunk_size = tvm.nd.array(np.array([max_page_num * PAGE_SIZE], dtype=np.int32), device=DEV)
        
        # kernel config for merge kernel when split-kv
        if split_kv:
            bdx_m, bdy_m = self.bd_m
            num_blk_per_sm = min(self.max_blk_per_sm, ceildiv(batch_size * self.num_heads, SM_COUNT))
            gdx_m = num_blk_per_sm * SM_COUNT
            
            smem_size_m = max(self.pipe_m * bdy_m * self.head_dim * F32_BYTE +  bdy_m * bdx_m * F32_BYTE,
                                bdy_m * self.head_dim * F32_BYTE + bdy_d * F32_BYTE)
            assert smem_size_m <= SMEM_SIZE
            assert self.pipe_m <= bdx_m
            self.gd_m = (gdx_m,)
            self.smem_m = smem_size_m

        # generate the necessary tvm arrays
        request_indices = []
        kv_tile_indices = []
        o_indptr = [0]
        for idx in range(batch_size):
            num_tiles_kv = ceildiv(kv_indptr_h[idx + 1] - kv_indptr_h[idx], max_page_num)
            for tile_idx in range(num_tiles_kv):
                request_indices.append(idx)
                kv_tile_indices.append(tile_idx)
            o_indptr.append(o_indptr[-1] + num_tiles_kv)  
        assert len(request_indices) == len(kv_tile_indices) == new_batch_size
        
        self.request_indices_tvm = tvm.nd.array(np.array(request_indices, dtype=np.int32), DEV)
        self.kv_tile_indices_tvm = tvm.nd.array(np.array(kv_tile_indices, dtype=np.int32), DEV)
        self.o_indptr_tvm = tvm.nd.array(np.array(o_indptr, dtype=np.int32), DEV)
        self.o_tvm = tvm.nd.array(np.zeros([batch_size, self.num_heads, self.head_dim], dtype=np.float16), DEV)
        self.lse_tvm = tvm.nd.array(np.zeros([batch_size, self.num_heads], dtype=np.float32), DEV)
        if split_kv:
            self.tmp_o_tvm = tvm.nd.array(np.zeros([new_batch_size, self.num_heads, self.head_dim], dtype=np.float32), DEV) 
            self.tmp_lse_tvm = tvm.nd.array(np.zeros([new_batch_size, self.num_heads], dtype=np.float32), DEV) 

def test(num_heads, seq_len, head_dim, batch_size_list):

    plan_info = PlanInfo(num_heads, head_dim)

    def decode(SPLIT_KV):

        NUM_HEADS = plan_info.num_heads
        HEAD_DIM = plan_info.head_dim
        VEC_SIZE = plan_info.vec_size_d
        PIPE_DEPTH = plan_info.pipe_d
        TILE_PER_BDX = plan_info.tile_per_bdx
        SM_SCALE = plan_info.sm_scale
        O_TYPE = "float32" if SPLIT_KV else "float16"
        BDX, BDY = plan_info.bd_d

        @T.prim_func(tirp=True)
        def decode_kernel(q_ptr: T.handle, kv_ptr: T.handle, o_ptr: T.handle, lse_ptr: T.handle,
                        kv_indptr: T.handle, kv_last_page_len: T.handle, kv_indices: T.handle,
                        request_indices: T.handle, kv_tile_indices: T.handle, max_chunk_size: T.handle):
            
            batch_size = T.int32()
            total_page_num = T.int32()
            new_batch_size = T.int32()
            
            q_global = T.match_buffer(q_ptr, [batch_size, NUM_HEADS, HEAD_DIM], "float16", scope="global", layout="default")
            kv_global = T.match_buffer(kv_ptr, [MAX_PAGE_NUM, 2, NUM_HEADS, PAGE_SIZE, HEAD_DIM], "float16", scope="global", layout="default")
            kv_indptr_global = T.match_buffer(kv_indptr, [batch_size + 1], "int32", scope="global", layout="default")
            kv_last_page_len_global = T.match_buffer(kv_last_page_len, [batch_size], "int32", scope="global", layout="default")
            kv_indices_global = T.match_buffer(kv_indices, [total_page_num], "int32", scope="global", layout="default")
            request_indices_global = T.match_buffer(request_indices, [new_batch_size], "int32", scope="global", layout="default")
            kv_tile_indices_global = T.match_buffer(kv_tile_indices, [new_batch_size], "int32", scope="global", layout="default")
            max_chunk_size_global = T.match_buffer(max_chunk_size, [1], "int32", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [new_batch_size, NUM_HEADS, HEAD_DIM], O_TYPE, scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [new_batch_size, NUM_HEADS], "float32", scope="global", layout="default")

            with T.kernel():
                bx = T.cta_id([SM_COUNT], parent="kernel")
                tx, ty = T.thread_id([BDX, BDY], parent="cta")

                kv_global_1d = Tp.reshape(kv_global, (-1,))

                with T.cta():
                    # allocate the memory
                    buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    k_smem = pool.alloc([PIPE_DEPTH, BDY, TILE_PER_BDX, HEAD_DIM], "float16", layout="default")
                    v_smem = pool.alloc([PIPE_DEPTH, BDY, TILE_PER_BDX, HEAD_DIM], "float16", layout="default")
                    kv_offset_load = pool.alloc([TILE_PER_BDX, BDY, BDX], "int32", layout="default")
                    kv_offset_use = Tp.reshape(kv_offset_load, [BDX, BDY, TILE_PER_BDX])
                    pool.move_base_to(0)
                    epi_o = pool.alloc([BDY, BDX, VEC_SIZE], "float32", layout="default")
                    epi_md = pool.alloc([BDY, 2], "float32", layout="default")

                    with T.thread():

                        # allocate the reg
                        idx = T.alloc_local([1], "int32", layout="default")
                        tmp = T.alloc_local([VEC_SIZE], "float16", layout="default")
                        q = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        k = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        v = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        s = T.alloc_local([TILE_PER_BDX], "float32", layout="default")
                        batch_idx = T.alloc_local([1], "int32", layout="default")
                        chunk_start_logical = T.alloc_local([1], "int32", layout="default")
                        chunk_end_logical = T.alloc_local([1], "int32", layout="default")
                        chunk_size = T.alloc_local([1], "int32", layout="default")
                        indices = T.alloc_local([1], "int32", layout="default")
                        kv_offset_cp = T.alloc_local([TILE_PER_BDX], "int32", layout="default")
                        o = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        m = T.alloc_local([2], "float32", layout="default")
                        d = T.alloc_local([2], "float32", layout="default")
                        m_tmp = T.alloc_local([1], "float32", layout="default")
                        d_tmp = T.alloc_local([1], "float32", layout="default")
                        o_tmp = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        cur = T.alloc_local([1], "int32", layout="default")

                        evt = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
                        tx_start = T.meta_var(tx * VEC_SIZE)
                        new_batch_id = T.meta_var(cur[0] // NUM_HEADS)
                        head_id = T.meta_var(cur[0] % NUM_HEADS)


                        @T.macro
                        def fetch_kv_offset(kt, offset):
                            token_id = T.meta_var(chunk_start_logical[0] + offset)
                            if token_id < chunk_end_logical[0]:
                                p = T.meta_var(token_id // PAGE_SIZE)
                                r = T.meta_var(token_id % PAGE_SIZE)
                                indices[0] = T.cuda.ldg(kv_indices_global.ptr_to([p]), "int32")
                                kv_offset_load[kt, ty, tx] = indices[0] * 2 * NUM_HEADS * PAGE_SIZE * HEAD_DIM + head_id* PAGE_SIZE * HEAD_DIM + r * HEAD_DIM
                            
                        cur[0] = bx
                        while cur[0] < new_batch_size * NUM_HEADS:
                            # fetch q
                            batch_idx[0] = request_indices_global[new_batch_id]
                            Tp.copy(tmp[:], q_global[batch_idx[0], head_id, tx_start:tx_start + VEC_SIZE])
                            Tp.cast(q[:], tmp[:])

                            # get chunk size info
                            chunk_start_logical[0] = kv_indptr_global[batch_idx[0]] * PAGE_SIZE
                            chunk_end_logical[0] = chunk_start_logical[0]
                            if SPLIT_KV:
                                chunk_start_logical[0] += max_chunk_size_global[0] * kv_tile_indices_global[new_batch_id]
                                chunk_end_logical[0] = T.min(chunk_start_logical[0] + max_chunk_size_global[0], 
                                                            chunk_end_logical[0] + (kv_indptr_global[batch_idx[0] + 1] - kv_indptr_global[batch_idx[0]] - 1) * PAGE_SIZE 
                                                            + kv_last_page_len_global[batch_idx[0]])
                            else:
                                chunk_end_logical[0] += (kv_indptr_global[batch_idx[0] + 1] - kv_indptr_global[batch_idx[0]] - 1) * PAGE_SIZE + kv_last_page_len_global[batch_idx[0]]
                            chunk_size[0] = chunk_end_logical[0] - chunk_start_logical[0]
                            
                            # fetch kv-offset
                            for kt in T.unroll(TILE_PER_BDX):
                                fetch_kv_offset(kt, (kt * BDY + ty) * BDX + tx)
                            T.ptx.fence.proxy("shared")
                            T.ptx.bar.sync(1, BDX * BDY)

                            for kp in T.unroll(PIPE_DEPTH):
                                # get kv-offset used in cp
                                for kt in T.unroll(TILE_PER_BDX):
                                    kv_offset_cp[kt] = kv_offset_use[kp, ty, kt] + tx * VEC_SIZE

                                # fetch K
                                for kt in T.unroll(TILE_PER_BDX):
                                    if (kp * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        g_st = T.meta_var(kv_offset_cp[kt])
                                        Tp.copy_async(k_smem[kp, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], evt,
                                                        schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()

                                # fetch V
                                for kt in T.unroll(TILE_PER_BDX):
                                    if (kp * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        g_st = T.meta_var(NUM_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt])
                                        Tp.copy_async(v_smem[kp, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], evt,
                                                        schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()

                            # initilize the value
                            idx[0] = 0
                            for kv in T.serial(VEC_SIZE):
                                o[kv] = 0.0
                            m[0] = T.float32('-inf')
                            d[0] = 1.0
                            # pipeline
                            for ki in T.serial(ceildiv(chunk_size[0], (TILE_PER_BDX * BDY))):
                                # fetch new kv-offset
                                if ((ki + PIPE_DEPTH) % BDX == 0):
                                    for kt in T.unroll(TILE_PER_BDX):
                                        fetch_kv_offset(kt, (ki + PIPE_DEPTH) * TILE_PER_BDX * BDY + (kt * BDY + ty) * BDX + tx)
                                    T.ptx.fence.proxy("shared")
                                
                                # compute qk
                                # T.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1)
                                evt.wait(2 * PIPE_DEPTH - 1) # wait for K
                                T.ptx.bar.sync(1, BDX * BDY)
                                m[1] = m[0]
                                for kt in T.unroll(TILE_PER_BDX):
                                    # cast k to f32
                                    Tp.cast(k[:], k_smem[idx[0], ty, kt, tx_start:tx_start + VEC_SIZE])
                                    s[kt] = 0.0
                                    # local gemm
                                    for kv in T.unroll(VEC_SIZE):
                                        s[kt] += q[kv] * k[kv]
                                    # reduce from other tx's sum
                                    for kr in T.unroll(find_power_of_two(BDX // 2) + 1):
                                        s[kt] = s[kt] + T.tvm_warp_shuffle_xor(0xFFFFFFFF, s[kt], (BDX // 2) >> kr, 32, 32)
                                    s[kt] *= SM_SCALE
                                    if (ki * BDY + ty) * TILE_PER_BDX + kt >= chunk_size[0]:
                                        s[kt] = T.float32('-inf')
                                    # update max value
                                    m[0] = T.max(m[0], s[kt])
                                
                                # update the sum for softmax
                                o_scale = T.meta_var(T.exp2(m[1] - m[0]))
                                d[0] *= o_scale
                                for kt in T.unroll(TILE_PER_BDX):
                                    s[kt] = T.exp2(s[kt] - m[0])
                                    d[0] += s[kt]
                                for kv in T.unroll(VEC_SIZE):
                                    o[kv] = o[kv] * o_scale
                                T.ptx.bar.sync(1, BDX * BDY)

                                # get kv-offset used in cp
                                for kt in T.unroll(TILE_PER_BDX):
                                    kv_offset_cp[kt] = kv_offset_use[(ki + PIPE_DEPTH) % BDX, ty, kt] + tx * VEC_SIZE

                                # fetch K
                                for kt in T.unroll(TILE_PER_BDX):
                                    if ((ki + PIPE_DEPTH) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        g_st = T.meta_var(kv_offset_cp[kt])
                                        Tp.copy_async(k_smem[idx[0], ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], evt,
                                                        schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()

                                # calculate softmax(qk)v
                                T.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1) # wait for V
                                T.ptx.bar.sync(1, BDX * BDY)
                                for kt in T.unroll(TILE_PER_BDX):
                                    Tp.cast(v[:], v_smem[idx[0], ty, kt, tx_start:tx_start + VEC_SIZE])
                                    for kv in T.unroll(VEC_SIZE):
                                        o[kv] += s[kt] * v[kv]
                                T.ptx.bar.sync(1, BDX * BDY)

                                # fetch V
                                for kt in T.unroll(TILE_PER_BDX):
                                    if ((ki + PIPE_DEPTH) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        g_st = T.meta_var(NUM_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt])
                                        Tp.copy_async(v_smem[idx[0], ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], evt,
                                                        schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()
                                idx[0] = (idx[0] + 1) % PIPE_DEPTH

                            evt.wait(0)
                            # T.ptx.cp_async.wait_group(0)
                            T.ptx.bar.sync(1, BDX * BDY)

                            # prepare o,m,d in smem for merging
                            for kv in T.unroll(VEC_SIZE):
                                epi_o[ty, tx, kv] = o[kv]
                            if tx == 0:
                                epi_md[ty, 0] = m[0]
                                epi_md[ty, 1] = d[0]
                            T.ptx.fence.proxy("shared")
                            T.ptx.bar.sync(1, BDX * BDY)
                            # merge o through different ty
                            if ty == 0:
                                m[0] = T.float32('-inf')
                                d[0] = 1.0
                                for kv in T.unroll(VEC_SIZE):
                                    o[kv] = 0.0
                                for ky in T.unroll(BDY):
                                    m_tmp[0] = epi_md[ky, 0]
                                    d_tmp[0] = epi_md[ky, 1]
                                    for kv in T.unroll(VEC_SIZE):
                                        o_tmp[kv] = epi_o[ky, tx, kv]
                                    m[1] = m[0]
                                    d[1] = d[0]
                                    m[0] = T.max(m[1], m_tmp[0])
                                    d[0] = d[1] * T.exp2(m[1] - m[0]) + d_tmp[0] * T.exp2(m_tmp[0] - m[0])
                                    for kv in T.unroll(VEC_SIZE):
                                        o[kv] = o[kv] * T.exp2(m[1] - m[0]) + o_tmp[kv] * T.exp2(m_tmp[0] - m[0])
                                # normalize
                                for kv in T.unroll(VEC_SIZE):
                                    o[kv] = o[kv] / d[0]
                                # store to global mem
                                if SPLIT_KV:
                                    Tp.copy(o_global[new_batch_id, head_id, tx_start:tx_start + VEC_SIZE], o[:])
                                else:
                                    Tp.cast(tmp[:], o[:])
                                    Tp.copy(o_global[new_batch_id, head_id, tx_start:tx_start + VEC_SIZE], tmp[:])
                                if tx == 0:
                                    lse_global[new_batch_id, head_id] = m[0] + T.log2(d[0])
                                
                            T.ptx.bar.sync(1, BDX * BDY)
                            cur[0] += SM_COUNT
        
        return decode_kernel

    def merge():
        NUM_HEADS = plan_info.num_heads
        HEAD_DIM = plan_info.head_dim
        VEC_SIZE = plan_info.vec_size_m
        PIPE_DEPTH = plan_info.pipe_m
        BDX, BDY = plan_info.bd_m

        @T.prim_func(tirp=True)
        def merge_kernel(o_tmp_ptr: T.handle, o_indptr: T.handle, o_ptr: T.handle, lse_tmp_ptr: T.handle, lse_ptr: T.handle):
            batch_size = T.int32()
            new_batch_size = T.int32()
            o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_HEADS, HEAD_DIM], "float32", scope="global", layout="default")
            o_indptr_global = T.match_buffer(o_indptr, [batch_size + 1], "int32", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [batch_size, NUM_HEADS, HEAD_DIM], "float16", scope="global", layout="default")
            lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_HEADS], "float32", scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_HEADS], "float32", scope="global", layout="default")

            with T.kernel():
                bx = T.cta_id([SM_COUNT], parent="kernel")
                tx, ty = T.thread_id([BDX, BDY], parent="cta")

                with T.cta():
                    # allocate the memory
                    buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    o_tmp_smem = pool.alloc([PIPE_DEPTH, BDY, HEAD_DIM], "float32", layout="default")
                    lse_tmp_smem_load = pool.alloc([BDY, BDX], "float32", layout="default")
                    lse_tmp_smem_use = Tp.reshape(lse_tmp_smem_load, [BDX, BDY])
                    pool.move_base_to(0)
                    o_epi_smem = pool.alloc([BDY, HEAD_DIM], "float32", layout="default")
                    lse_epi_smem = pool.alloc([BDY], "float32", layout="default")

                    with T.thread():
                        idx = T.alloc_local([1], "int32", layout="default")
                        head_idx = T.alloc_local([1], "int32", layout="default")
                        batch_idx = T.alloc_local([1], "int32", layout="default")
                        new_beg_batch_idx = T.alloc_local([1], "int32", layout="default")
                        num = T.alloc_local([1], "int32", layout="default")
                        tmp = T.alloc_local([VEC_SIZE], "float16", layout="default")
                        o = T.alloc_local([VEC_SIZE], "float32", layout="default")
                        m = T.alloc_local([2], "float32", layout="default")
                        d = T.alloc_local([2], "float32", layout="default")
                        m_tmp = T.alloc_local([1], "float32", layout="default")
                        o_tmp = T.alloc_local([VEC_SIZE], "float32", layout="default")

                        evt = Tp.alloc_bulk_group_event(EventImpl.kCpAsync)
                        tx_start = T.meta_var(tx * VEC_SIZE)

                        idx[0] = bx
                        while idx[0] < batch_size * NUM_HEADS:
                            head_idx[0] = idx[0] % NUM_HEADS
                            batch_idx[0] = idx[0] // NUM_HEADS
                            new_beg_batch_idx[0] = o_indptr_global[batch_idx[0]]
                            num[0] = o_indptr_global[batch_idx[0] + 1] - new_beg_batch_idx[0]

                            if num[0] == 1:
                                if ty == 0:
                                    Tp.copy(o[:], o_tmp_global[new_beg_batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE])
                                    Tp.cast(tmp[:], o[:])
                                    Tp.copy(o_global[batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE], tmp[:])
                                    lse_global[batch_idx[0], head_idx[0]] = lse_tmp_global[new_beg_batch_idx[0], head_idx[0]]
                                continue
                                
                            # pipeline
                            for kp in T.unroll(PIPE_DEPTH):
                                if kp * BDY + ty < num[0]:
                                    Tp.copy_async(o_tmp_smem[kp, ty, tx_start:tx_start + VEC_SIZE], 
                                                  o_tmp_global[new_beg_batch_idx[0] + kp * BDY + ty, head_idx[0], tx_start:tx_start + VEC_SIZE],
                                                  evt, schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()
                            
                            # initialize the value
                            m[0] = T.float32('-inf')
                            d[0] = 1.0
                            for kv in T.unroll(VEC_SIZE):
                                o[kv] = 0.0

                            for ki in T.serial(ceildiv(num[0], BDY)):
                                if ki % BDX == 0:
                                    # load lse
                                    if ki * BDY + ty * BDX + tx < num[0]:
                                        lse_tmp_smem_load[ty, tx] = lse_tmp_global[new_beg_batch_idx[0] + ki * BDY + ty * BDX + tx, head_idx[0]]
                                    else:
                                        lse_tmp_smem_load[ty, tx] = 0.0
                                    T.ptx.fence.proxy("shared")
                                    T.ptx.bar.sync(2, BDX * BDY)

                                # T.ptx.cp_async.wait_group(PIPE_DEPTH - 1)
                                evt.wait(PIPE_DEPTH - 1)
                                T.ptx.bar.sync(2, BDX * BDY)
                                T.ptx.fence.proxy("shared")

                                for kv in T.serial(VEC_SIZE):
                                    o_tmp[kv] = o_tmp_smem[ki % PIPE_DEPTH, ty, tx * VEC_SIZE + kv]
                                if ki * BDY + ty < num[0]:
                                    m_tmp[0] = lse_tmp_smem_use[ki % BDX, ty] 
                                    m[1] = m[0]
                                    d[1] = d[0]
                                    m[0] = T.max(m[1], m_tmp[0])
                                    d[0] = d[1] * T.exp2(m[1] - m[0]) + T.exp2(m_tmp[0] - m[0])
                                    for kv in T.unroll(VEC_SIZE):
                                        o[kv] = o[kv] * T.exp2(m[1] - m[0]) + o_tmp[kv] * T.exp2(m_tmp[0] - m[0])
                                T.ptx.bar.sync(2, BDX * BDY)
                                if (PIPE_DEPTH + ki) * BDY + ty < num[0]:
                                    Tp.copy_async(o_tmp_smem[ki % PIPE_DEPTH, ty, tx_start:tx_start + VEC_SIZE], 
                                                  o_tmp_global[new_beg_batch_idx[0] + (ki + PIPE_DEPTH) * BDY + ty, head_idx[0], tx_start:tx_start + VEC_SIZE],
                                                  evt, schedule_config={"vec_len": VEC_SIZE})
                                evt.commit()
                            evt.wait(0)
                            # T.ptx.cp_async.wait_group(0)
                            T.ptx.bar.sync(2, BDX * BDY)
                            # normalize
                            for kv in T.unroll(VEC_SIZE):
                                o[kv] = o[kv] / d[0]

                            # reduce
                            for kv in T.serial(VEC_SIZE):
                                o_epi_smem[ty, tx * VEC_SIZE + kv] = o[kv]
                            lse_epi_smem[ty] = m[0] + T.log2(d[0])
                            m[0] = T.float32('-inf')
                            d[0] = 1.0
                            for kv in T.serial(VEC_SIZE):
                                o[kv] = 0.0
                            T.ptx.fence.proxy("shared")
                            T.ptx.bar.sync(2, BDX * BDY)
                            if ty == 0:
                                for ky in T.serial(BDY):
                                    m_tmp[0] = lse_epi_smem[ky]
                                    for kv in T.serial(VEC_SIZE):
                                        o_tmp[kv] = o_epi_smem[ky, tx * VEC_SIZE + kv]
                                    m[1] = m[0]
                                    d[1] = d[0]
                                    m[0] = T.max(m[1], m_tmp[0])
                                    d[0] = d[1] * T.exp2(m[1] - m[0]) + T.exp2(m_tmp[0] - m[0])
                                    for kv in T.unroll(VEC_SIZE):
                                        o[kv] = o[kv] * T.exp2(m[1] - m[0]) + o_tmp[kv] * T.exp2(m_tmp[0] - m[0])
                                          
                                for kv in T.unroll(VEC_SIZE):
                                    o[kv] = o[kv] / d[0]
                                
                                # store to global mem
                                Tp.cast(tmp[:], o[:])
                                Tp.copy(o_global[batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE], tmp[:])
                                if tx == 0:
                                    lse_global[batch_idx[0], head_idx[0]] = m[0] + T.log2(d[0])
                            idx[0] += SM_COUNT
                        T.ptx.bar.sync(2, BDX * BDY)  
        
        return merge_kernel

    def test_dynamic_batch_size(batch_size, mod_decode_no_split_kv, mod_decode_split_kv, mod_merge):
        Q, KV_data, KV_indptr, KV_last_page_len, KV_indices = perpare_data(batch_size, num_heads, seq_len, head_dim)

        def tir():

            DEV = tvm.cuda(0)
            q_tvm = tvm.nd.array(Q, DEV)
            kv_data_tvm = tvm.nd.array(KV_data, DEV)
            kv_indptr_tvm = tvm.nd.array(KV_indptr, DEV)
            kv_last_page_len_tvm = tvm.nd.array(KV_last_page_len, DEV)
            kv_indices_tvm = tvm.nd.array(KV_indices, DEV)
            plan_info.plan(batch_size, KV_indptr)

            def func():
                if plan_info.split_kv:
                    mod_decode_split_kv(q_tvm, kv_data_tvm, plan_info.tmp_o_tvm, plan_info.tmp_lse_tvm, 
                                        kv_indptr_tvm, kv_last_page_len_tvm, kv_indices_tvm, plan_info.request_indices_tvm, 
                                        plan_info.kv_tile_indices_tvm, plan_info.max_chunk_size) 
                    mod_merge(plan_info.tmp_o_tvm, plan_info.o_indptr_tvm, plan_info.o_tvm, plan_info.tmp_lse_tvm, plan_info.lse_tvm)
                else:
                    mod_decode_no_split_kv(q_tvm, kv_data_tvm, plan_info.o_tvm, plan_info.lse_tvm, 
                                           kv_indptr_tvm, kv_last_page_len_tvm, kv_indices_tvm, plan_info.request_indices_tvm, 
                                           plan_info.kv_tile_indices_tvm, plan_info.max_chunk_size)
            
            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            func()
            print(f"TIR time: {ms:.3f} ms")

            return plan_info.o_tvm.numpy(), plan_info.lse_tvm.numpy()

        def flashinfer():
            import flashinfer
            import torch

            q_f = Q.to(0)
            kv_data_f = KV_data.to(0)
            kv_indptr_f = KV_indptr.to(0)
            kv_last_page_len_f = KV_last_page_len.to(0)
            kv_indices_f = KV_indices.to(0)

            workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, KV_LAYOUT)
            wrapper.plan(
                kv_indptr_f,
                kv_indices_f,
                kv_last_page_len_f,
                num_heads,
                num_heads,
                head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                data_type=torch.float16,
                q_data_type=torch.float16,
            )
            o, lse = wrapper.run_return_lse(q_f, kv_data_f)

            wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, KV_LAYOUT, use_tensor_cores=True
            )
            wrapper_tensor_cores.plan(
                kv_indptr_f,
                kv_indices_f,
                kv_last_page_len_f,
                num_heads,
                num_heads,
                head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                data_type=torch.float16,
                q_data_type=torch.float16,
            )
            o_tc, lse_tc = wrapper_tensor_cores.run_return_lse(q_f, kv_data_f)

            torch.testing.assert_close(o, o_tc, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(lse, lse_tc, rtol=1e-3, atol=1e-3)

            func = lambda: wrapper.run(q_f, kv_data_f)
            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer")
            func()
            print(f"FlashInfer time: {ms:.3f} ms")

            return o.reshape(batch_size, num_heads, head_dim).cpu().numpy(), lse.reshape(batch_size, num_heads).cpu().numpy()

        with ProtonContext("batch_decode"):
            print(f"Testing (B,H,N,D) = ({batch_size},{num_heads},{seq_len},{head_dim})")
            O_flashinfer, lse_flashinfer = flashinfer()
            O_tir, lse_tir = tir()
            np.testing.assert_allclose(O_tir, O_flashinfer, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lse_tir, lse_flashinfer, rtol=1e-3, atol=1e-3)

    # compile tir kernel
    target = tvm.target.Target("cuda")
    with target:
        mod_decode_no_split_kv = tvm.IRModule({"main": decode(False)})
        mod_decode_no_split_kv = tvm.compile(mod_decode_no_split_kv, target=target, tir_pipeline="tirp")
        # src = mod_decode_no_split_kv.mod.imported_modules[0].get_source()
        # print(src)
        mod_decode_split_kv = tvm.IRModule({"main": decode(True)})
        mod_decode_split_kv = tvm.compile(mod_decode_split_kv, target=target, tir_pipeline="tirp")
        mod_merge = tvm.IRModule({"main": merge()})
        mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirp")

    for batch_size in batch_size_list:
        test_dynamic_batch_size(batch_size, mod_decode_no_split_kv, mod_decode_split_kv, mod_merge)     

if __name__ == "__main__":
    import itertools

    num_heads_list = [8]
    seq_len_list = [2048]
    head_dim_list = [128]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64]
    
    for (num_heads, seq_len, head_dim) in itertools.product(num_heads_list, seq_len_list, head_dim_list):
        test(num_heads, seq_len, head_dim, batch_size_list)