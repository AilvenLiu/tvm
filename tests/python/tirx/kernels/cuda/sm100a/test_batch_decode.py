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
import pytest

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
SMEM_SIZE = 232448

# Other
F16_BYTE = 2
F32_BYTE = 4


def perpare_data(batch_size, qo_heads, kv_heads, seq_len, head_dim, page_size, max_page_num):
    PAGE_SIZE = page_size
    MAX_PAGE_NUM = max_page_num
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
    q = torch.randn([batch_size, qo_heads, head_dim]).half()
    kv_data = torch.randn([MAX_PAGE_NUM, 2, kv_heads, PAGE_SIZE, head_dim]).half()

    return q, kv_data, kv_indptr, kv_last_page_len, kv_indices


class PlanInfo:
    def __init__(self, qo_heads, kv_heads, head_dim, enforce_no_split_kv=False):
        # static info
        self.max_blk_per_sm = 8 if not enforce_no_split_kv else 0
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        assert qo_heads % kv_heads == 0
        self.head_dim = head_dim
        self.vec_size_d = max(8, head_dim // 32)  # ensure cp_async_size >= 128bit && bdx <= 32
        self.vec_size_m = max(4, head_dim // 32)  # ensure cp_async_size >= 128bit && bdx <= 32
        bdx_d = self.head_dim // self.vec_size_d
        bdy_d = qo_heads // kv_heads
        self.num_threads_d = max(512, bdx_d * bdy_d)
        self.head_per_cta = 1
        bdz_d = self.num_threads_d // (bdx_d * bdy_d)
        self.bd_d = (bdx_d, bdy_d, bdz_d)
        bdx_m = self.head_dim // self.vec_size_m
        self.num_threads_m = max(128, bdx_m)
        bdy_m = self.num_threads_m // bdx_m
        self.bd_m = (bdx_m, bdy_m)
        self.pipe_d = 1
        self.pipe_m = 4
        self.tile_per_bdx = 4
        self.sm_scale = (1 / 0.6931471805599453) * (1 / head_dim**0.5)

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

    def plan(self, batch_size, kv_indptr_h, page_size, max_page_num):
        if isinstance(kv_indptr_h, tvm.runtime.Tensor):
            kv_indptr_h = kv_indptr_h.numpy().tolist()

        PAGE_SIZE = page_size
        MAX_PAGE_NUM = max_page_num

        DEV = tvm.cuda(0)
        self.batch_size = batch_size

        # kernel dim config for decode kernel
        bdx_d, bdy_d, bdz_d = self.bd_d
        smem_size_d = (
            2
            * self.pipe_d
            * self.head_per_cta
            * bdz_d
            * bdy_d
            * self.tile_per_bdx
            * self.head_dim
            * F16_BYTE
            + self.num_threads_d * self.tile_per_bdx * F32_BYTE
            + self.num_threads_d * self.head_per_cta * self.vec_size_d * F32_BYTE
            + bdz_d * bdy_d * self.head_per_cta * 2 * F32_BYTE
        )
        assert smem_size_d <= SMEM_SIZE
        assert self.pipe_d <= bdx_d
        self.smem_d = smem_size_d

        # balance the workload (split-kv)
        if batch_size * self.qo_heads >= SM_COUNT * self.max_blk_per_sm:
            split_kv = False
            max_page_num = 1
            for idx in range(batch_size):
                max_page_num = max(max_page_num, kv_indptr_h[idx + 1] - kv_indptr_h[idx])
            new_batch_size = batch_size
        else:
            page_num_list = [kv_indptr_h[idx + 1] - kv_indptr_h[idx] for idx in range(batch_size)]
            new_batch_size = batch_size
            low = max(1, 64 // PAGE_SIZE)
            high = max(page_num_list)
            while low < high:
                mid = (low + high) // 2
                new_batch_size = 0
                for page_num in page_num_list:
                    new_batch_size += ceildiv(page_num, mid)
                if new_batch_size * self.qo_heads > SM_COUNT * self.max_blk_per_sm:
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
        self.max_chunk_size = tvm.runtime.tensor(
            np.array([max_page_num * PAGE_SIZE], dtype=np.int32), device=DEV
        )

        # kernel config for merge kernel when split-kv
        if split_kv:
            bdx_m, bdy_m = self.bd_m
            smem_size_m = max(
                self.pipe_m * bdy_m * self.head_dim * F32_BYTE + bdy_m * bdx_m * F32_BYTE,
                bdy_m * self.head_dim * F32_BYTE + bdy_d * F32_BYTE,
            )
            assert smem_size_m <= SMEM_SIZE
            assert self.pipe_m <= bdx_m
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

        self.request_indices_tvm = tvm.runtime.tensor(
            np.array(request_indices, dtype=np.int32), DEV
        )
        self.kv_tile_indices_tvm = tvm.runtime.tensor(
            np.array(kv_tile_indices, dtype=np.int32), DEV
        )
        self.o_indptr_tvm = tvm.runtime.tensor(np.array(o_indptr, dtype=np.int32), DEV)
        self.o_tvm = tvm.runtime.tensor(
            np.zeros([batch_size, self.qo_heads, self.head_dim], dtype=np.float16), DEV
        )
        self.lse_tvm = tvm.runtime.tensor(
            np.zeros([batch_size, self.qo_heads], dtype=np.float32), DEV
        )
        if split_kv:
            self.tmp_o_tvm = tvm.runtime.tensor(
                np.zeros([new_batch_size, self.qo_heads, self.head_dim], dtype=np.float32), DEV
            )
            self.tmp_lse_tvm = tvm.runtime.tensor(
                np.zeros([new_batch_size, self.qo_heads], dtype=np.float32), DEV
            )


def decode_attn_plan(
    qo_heads, kv_heads, head_dim, batch_size, kv_indptr_h, page_size, max_page_num
):
    plan_info = PlanInfo(qo_heads, kv_heads, head_dim, enforce_no_split_kv=True)
    plan_info.plan(batch_size, kv_indptr_h, page_size, max_page_num)
    return (
        plan_info.lse_tvm,
        plan_info.request_indices_tvm,
        plan_info.kv_tile_indices_tvm,
        plan_info.max_chunk_size,
    )


def get_decode_kernel(plan_info: PlanInfo, page_size):
    PAGE_SIZE = page_size

    def decode(SPLIT_KV):
        QO_HEADS = plan_info.qo_heads
        KV_HEADS = plan_info.kv_heads
        HEAD_DIM = plan_info.head_dim
        VEC_SIZE = plan_info.vec_size_d
        PIPE_DEPTH = plan_info.pipe_d
        TILE_PER_BDX = plan_info.tile_per_bdx
        SM_SCALE = plan_info.sm_scale
        O_TYPE = "float32" if SPLIT_KV else "float16"
        BDX, BDY, BDZ = plan_info.bd_d
        HEAD_PER_CTA = plan_info.head_per_cta

        # fmt: off
        @Tx.prim_func(tirx=True)
        def decode_kernel(q_ptr: Tx.handle, kv_ptr: Tx.handle, lse_ptr: Tx.handle, kv_indptr: Tx.handle, kv_last_page_len: Tx.handle,
                          kv_indices: Tx.handle, request_indices: Tx.handle, kv_tile_indices: Tx.handle, max_chunk_size: Tx.handle, o_ptr: Tx.handle):

            batch_size = Tx.int32()
            max_page_num = Tx.int32()
            total_page_num = Tx.int32()
            new_batch_size = Tx.int32()
            request_indices_global_elem_offset = Tx.int32()
            kv_tile_indices_global_elem_offset = Tx.int32()
            max_chunk_size_global_elem_offset = Tx.int32()

            q_global = Tx.match_buffer(q_ptr, [batch_size, QO_HEADS, HEAD_DIM], "float16", scope="global")
            kv_global = Tx.match_buffer(kv_ptr, [max_page_num, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM], "float16", scope="global")
            kv_indptr_global = Tx.match_buffer(kv_indptr, [batch_size + 1], "int32", scope="global", offset_factor=1)
            kv_last_page_len_global = Tx.match_buffer(kv_last_page_len, [batch_size], "int32", scope="global", offset_factor=1)
            kv_indices_global = Tx.match_buffer(kv_indices, [total_page_num], "int32", scope="global", offset_factor=1)
            request_indices_global = Tx.match_buffer(request_indices, [new_batch_size], "int32", scope="global", elem_offset=request_indices_global_elem_offset)
            kv_tile_indices_global = Tx.match_buffer(kv_tile_indices, [new_batch_size], "int32", scope="global", elem_offset=kv_tile_indices_global_elem_offset)
            max_chunk_size_global = Tx.match_buffer(max_chunk_size, [1], "int32", scope="global", elem_offset=max_chunk_size_global_elem_offset)
            o_global = Tx.match_buffer(o_ptr, [new_batch_size, QO_HEADS, HEAD_DIM], O_TYPE, scope="global")
            lse_global = Tx.match_buffer(lse_ptr, [new_batch_size, QO_HEADS], "float32", scope="global")

            with Tx.kernel():
                bx = Tx.cta_id([SM_COUNT], parent="kernel")
                tx, ty, tz = Tx.thread_id([BDX, BDY, BDZ], parent="cta")
                kv_global_1d = kv_global.view(-1)

                with Tx.cta():
                    # allocate the memory
                    pool = Tx.PoolAllocator()
                    k_smem = pool.alloc([PIPE_DEPTH, HEAD_PER_CTA, BDZ, BDY, TILE_PER_BDX, HEAD_DIM], "float16")
                    v_smem = pool.alloc([PIPE_DEPTH, HEAD_PER_CTA, BDZ, BDY, TILE_PER_BDX, HEAD_DIM], "float16")
                    kv_offset = pool.alloc([BDZ, BDX, BDY, TILE_PER_BDX], "int32")
                    epi_o = pool.alloc([BDZ, BDY, BDX, HEAD_PER_CTA, VEC_SIZE], "float32")
                    epi_md = pool.alloc([BDZ, BDY, HEAD_PER_CTA, 2], "float32")
                    pool.commit()

                    with Tx.thread():

                        # allocate the reg
                        idx = Tx.alloc_local([1], "int32")
                        tmp = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float16")
                        q = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float32")
                        k = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float32")
                        v = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float32")
                        s = Tx.alloc_local([HEAD_PER_CTA, TILE_PER_BDX * BDY], "float32")
                        batch_idx = Tx.alloc_local([1], "int32")
                        chunk_start_logical = Tx.alloc_local([1], "int32")
                        chunk_end_logical = Tx.alloc_local([1], "int32")
                        chunk_size = Tx.alloc_local([1], "int32")
                        indices = Tx.alloc_local([1], "int32")
                        kv_offset_cp = Tx.alloc_local([TILE_PER_BDX], "int32")
                        o = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float32")
                        m = Tx.alloc_local([HEAD_PER_CTA, 2], "float32")
                        d = Tx.alloc_local([HEAD_PER_CTA, 2], "float32")
                        m_tmp = Tx.alloc_local([HEAD_PER_CTA, 1], "float32")
                        d_tmp = Tx.alloc_local([HEAD_PER_CTA, 1], "float32")
                        o_tmp = Tx.alloc_local([HEAD_PER_CTA, VEC_SIZE], "float32")
                        cur = Tx.alloc_local([1], "int32")
                        tx_start = Tx.meta_var(tx * VEC_SIZE)

                        @Tx.macro
                        def fetch_kv_offset(kt, kv_head_id_beg, offset):
                            token_id = Tx.meta_var(chunk_start_logical[0] + offset)
                            if token_id < chunk_end_logical[0]:
                                p = Tx.meta_var(token_id // PAGE_SIZE)
                                r = Tx.meta_var(token_id % PAGE_SIZE)
                                indices[0] = Tx.cuda.ldg(kv_indices_global.ptr_to([p]), "int32")
                                kv_offset[tz, tx, ty, kt] = (indices[0] * 2 * KV_HEADS * PAGE_SIZE * HEAD_DIM
                                                                  + kv_head_id_beg * PAGE_SIZE * HEAD_DIM + r * HEAD_DIM)
                        @Tx.macro
                        def sync_blk():
                            if BDZ <= 4:
                                Tx.ptx.bar.sync(1 + tz, BDX * BDY)
                            else:
                                Tx.ptx.bar.sync(1, BDX * BDY * BDZ)

                        cur[0] = bx
                        while cur[0] < new_batch_size * KV_HEADS // HEAD_PER_CTA:
                            new_batch_id = Tx.meta_var(cur[0] * HEAD_PER_CTA // KV_HEADS)
                            kv_head_id_beg = Tx.meta_var((cur[0] * HEAD_PER_CTA) % KV_HEADS)

                            # fetch q
                            batch_idx[0] = request_indices_global[new_batch_id]
                            for kb in Tx.unroll(HEAD_PER_CTA):
                                Tx.copy(tmp[kb, :], q_global[batch_idx[0], (kv_head_id_beg + kb) * BDY + ty, tx_start:tx_start + VEC_SIZE])
                                Tx.cast(q[kb, :], tmp[kb, :])

                            # get chunk size info
                            chunk_start_logical[0] = kv_indptr_global[batch_idx[0]] * PAGE_SIZE
                            chunk_end_logical[0] = chunk_start_logical[0]
                            if SPLIT_KV:
                                chunk_start_logical[0] += max_chunk_size_global[0] * kv_tile_indices_global[new_batch_id]
                                chunk_end_logical[0] = Tx.min(chunk_start_logical[0] + max_chunk_size_global[0],
                                                            chunk_end_logical[0] + (kv_indptr_global[batch_idx[0] + 1] - kv_indptr_global[batch_idx[0]] - 1) * PAGE_SIZE
                                                            + kv_last_page_len_global[batch_idx[0]])
                            else:
                                chunk_end_logical[0] += (kv_indptr_global[batch_idx[0] + 1] - kv_indptr_global[batch_idx[0]] - 1) * PAGE_SIZE + kv_last_page_len_global[batch_idx[0]]
                            chunk_size[0] = chunk_end_logical[0] - chunk_start_logical[0]

                            # fetch kv-offset
                            for kt in Tx.unroll(TILE_PER_BDX):
                                fetch_kv_offset(kt, kv_head_id_beg, ((tx * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt)
                            Tx.ptx.fence.proxy("shared")
                            sync_blk()

                            for kp in Tx.unroll(PIPE_DEPTH):
                                # get kv-offset used in cp
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    kv_offset_cp[kt] = kv_offset[tz, kp, ty, kt] + tx * VEC_SIZE

                                # fetch K
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    if ((kp * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        for kb in Tx.unroll(HEAD_PER_CTA):
                                            g_st = Tx.meta_var(kv_offset_cp[kt] + kb * PAGE_SIZE * HEAD_DIM)
                                            Tx.copy_async(k_smem[kp, kb, tz, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], dispatch="non-bulk-copy",
                                                            vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()

                                # fetch V
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    if ((kp * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        for kb in Tx.unroll(HEAD_PER_CTA):
                                            g_st = Tx.meta_var(KV_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt] + kb * PAGE_SIZE * HEAD_DIM)
                                            Tx.copy_async(v_smem[kp, kb, tz, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], dispatch="non-bulk-copy",
                                                            vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()

                            # initilize the value
                            idx[0] = 0
                            for kb in Tx.unroll(HEAD_PER_CTA):
                                for kv in Tx.unroll(VEC_SIZE):
                                    o[kb, kv] = 0.0
                                m[kb, 0] = Tx.float32('-inf')
                                d[kb, 0] = 1.0
                            # pipeline
                            for ki in Tx.serial(ceildiv(chunk_size[0], (TILE_PER_BDX * BDY * BDZ))):
                                # fetch new kv-offset
                                if ((ki + PIPE_DEPTH) % BDX == 0):
                                    for kt in Tx.unroll(TILE_PER_BDX):
                                        fetch_kv_offset(kt, kv_head_id_beg,
                                                        (ki + PIPE_DEPTH) * TILE_PER_BDX * BDY * BDZ  + ((tx * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt)
                                    Tx.ptx.fence.proxy("shared")

                                # compute qk
                                Tx.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1) # wait for K
                                sync_blk()
                                for kb in Tx.unroll(HEAD_PER_CTA):
                                    m[kb, 1] = m[kb, 0]
                                for kt in Tx.unroll(TILE_PER_BDX * BDY):
                                    for kb in Tx.unroll(HEAD_PER_CTA):
                                        # cast k to f32
                                        Tx.cast(k[kb, :], k_smem[idx[0], kb, tz, kt // TILE_PER_BDX, kt % TILE_PER_BDX, tx_start:tx_start + VEC_SIZE])
                                        s[kb, kt] = 0.0
                                        # local gemm
                                        for kv in Tx.unroll(VEC_SIZE):
                                            s[kb, kt] += q[kb, kv] * k[kb, kv]
                                        # reduce from other tx's sum
                                        for kr in Tx.unroll(find_power_of_two(BDX // 2) + 1):
                                            s[kb, kt] = s[kb, kt] + Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, s[kb, kt], (BDX // 2) >> kr, 32, 32)
                                        s[kb, kt] *= SM_SCALE
                                        if (ki * BDZ + tz) * BDY * TILE_PER_BDX + kt >= chunk_size[0]:
                                            s[kb, kt] = Tx.float32('-inf')
                                        # update max value
                                        m[kb, 0] = Tx.max(m[kb, 0], s[kb, kt])

                                # update the sum for softmax
                                if TILE_PER_BDX * BDY * tz < chunk_size[0]:
                                    for kb in Tx.unroll(HEAD_PER_CTA):
                                        o_scale = Tx.meta_var(Tx.exp2(m[kb, 1] - m[kb, 0]))
                                        d[kb, 0] *= o_scale
                                        for kt in Tx.unroll(TILE_PER_BDX * BDY):
                                            s[kb, kt] = Tx.exp2(s[kb, kt] - m[kb, 0])
                                            d[kb, 0] += s[kb, kt]
                                        for kv in Tx.unroll(VEC_SIZE):
                                            o[kb, kv] = o[kb, kv] * o_scale
                                sync_blk()

                                # get kv-offset used in cp
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    kv_offset_cp[kt] = kv_offset[tz, (ki + PIPE_DEPTH) % BDX, ty, kt] + tx * VEC_SIZE

                                # fetch K
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    if (((ki + PIPE_DEPTH) * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        for kb in Tx.unroll(HEAD_PER_CTA):
                                            g_st = Tx.meta_var(kv_offset_cp[kt] + kb * PAGE_SIZE * HEAD_DIM)
                                            Tx.copy_async(k_smem[idx[0], kb, tz, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], dispatch="non-bulk-copy",
                                                            vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()

                                # calculate softmax(qk)v
                                Tx.ptx.cp_async.wait_group(2 * PIPE_DEPTH - 1) # wait for V
                                sync_blk()
                                for kt in Tx.unroll(TILE_PER_BDX * BDY):
                                    for kb in Tx.unroll(HEAD_PER_CTA):
                                        if (ki * BDZ + tz) * BDY * TILE_PER_BDX + kt < chunk_size[0]:
                                            Tx.cast(v[kb, :], v_smem[idx[0], kb, tz, kt // TILE_PER_BDX, kt % TILE_PER_BDX, tx_start:tx_start + VEC_SIZE])
                                            for kv in Tx.unroll(VEC_SIZE):
                                                o[kb, kv] += s[kb, kt] * v[kb, kv]
                                sync_blk()

                                # fetch V
                                for kt in Tx.unroll(TILE_PER_BDX):
                                    if (((ki + PIPE_DEPTH) * BDZ + tz) * BDY + ty) * TILE_PER_BDX + kt < chunk_size[0]:
                                        for kb in Tx.unroll(HEAD_PER_CTA):
                                            g_st = Tx.meta_var(KV_HEADS * PAGE_SIZE * HEAD_DIM + kv_offset_cp[kt] + kb * PAGE_SIZE * HEAD_DIM)
                                            Tx.copy_async(v_smem[idx[0], kb, tz, ty, kt, tx_start:tx_start + VEC_SIZE], kv_global_1d[g_st:g_st + VEC_SIZE], dispatch="non-bulk-copy",
                                                            vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()
                                idx[0] = (idx[0] + 1) % PIPE_DEPTH

                            Tx.ptx.cp_async.wait_group(0)

                            # prepare o,m,d in smem for merging
                            for kb in Tx.unroll(HEAD_PER_CTA):
                                for kv in Tx.unroll(VEC_SIZE):
                                    epi_o[tz, ty, tx, kb, kv] = o[kb, kv]
                            if tx == 0:
                                for kb in Tx.unroll(HEAD_PER_CTA):
                                    epi_md[tz, ty, kb, 0] = m[kb, 0]
                                    epi_md[tz, ty, kb, 1] = d[kb, 0]
                            Tx.ptx.fence.proxy("shared")
                            Tx.cuda.cta_sync()
                            # merge o through different tz
                            if tz == 0:
                                for kb in Tx.unroll(HEAD_PER_CTA):
                                    m[kb, 0] = Tx.float32('-inf')
                                    d[kb, 0] = 1.0
                                    for kv in Tx.unroll(VEC_SIZE):
                                        o[kb, kv] = 0.0
                                    for kz in Tx.unroll(BDZ):
                                        if TILE_PER_BDX * BDY * kz < chunk_size[0]:
                                            m_tmp[kb, 0] = epi_md[kz, ty, kb, 0]
                                            d_tmp[kb, 0] = epi_md[kz, ty, kb, 1]
                                            for kv in Tx.unroll(VEC_SIZE):
                                                o_tmp[kb, kv] = epi_o[kz, ty, tx, kb, kv]
                                            m[kb, 1] = m[kb, 0]
                                            d[kb, 1] = d[kb, 0]
                                            m[kb, 0] = Tx.max(m[kb, 1], m_tmp[kb, 0])
                                            d[kb, 0] = d[kb, 1] * Tx.exp2(m[kb, 1] - m[kb, 0]) + d_tmp[kb, 0] * Tx.exp2(m_tmp[kb, 0] - m[kb, 0])
                                            for kv in Tx.unroll(VEC_SIZE):
                                                o[kb, kv] = o[kb, kv] * Tx.exp2(m[kb, 1] - m[kb, 0]) + o_tmp[kb, kv] * Tx.exp2(m_tmp[kb, 0] - m[kb, 0])
                                    # normalize
                                    for kv in Tx.unroll(VEC_SIZE):
                                        o[kb, kv] = o[kb, kv] / d[kb, 0]
                                # store to global mem
                                for kb in Tx.unroll(HEAD_PER_CTA):
                                    qo_head_id = Tx.meta_var((kv_head_id_beg + kb) * BDY + ty)
                                    if SPLIT_KV:
                                        Tx.copy(o_global[new_batch_id, qo_head_id, tx_start:tx_start + VEC_SIZE], o[kb, :])
                                    else:
                                        Tx.cast(tmp[kb, :], o[kb, :])
                                        Tx.copy(o_global[new_batch_id, qo_head_id, tx_start:tx_start + VEC_SIZE], tmp[kb, :])
                                    if tx == 0:
                                        lse_global[new_batch_id, qo_head_id] = m[kb, 0] + Tx.log2(d[kb, 0])

                            cur[0] += SM_COUNT
        # fmt: on
        return decode_kernel

    def merge():
        NUM_HEADS = plan_info.qo_heads
        HEAD_DIM = plan_info.head_dim
        VEC_SIZE = plan_info.vec_size_m
        PIPE_DEPTH = plan_info.pipe_m
        BDX, BDY = plan_info.bd_m

        # fmt: off
        @Tx.prim_func(tirx=True)
        def merge_kernel(o_tmp_ptr: Tx.handle, o_indptr: Tx.handle, o_ptr: Tx.handle, lse_tmp_ptr: Tx.handle, lse_ptr: Tx.handle):
            batch_size = Tx.int32()
            new_batch_size = Tx.int32()
            o_tmp_global = Tx.match_buffer(o_tmp_ptr, [new_batch_size, NUM_HEADS, HEAD_DIM], "float32", scope="global")
            o_indptr_global = Tx.match_buffer(o_indptr, [batch_size + 1], "int32", scope="global")
            o_global = Tx.match_buffer(o_ptr, [batch_size, NUM_HEADS, HEAD_DIM], "float16", scope="global")
            lse_tmp_global = Tx.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_HEADS], "float32", scope="global")
            lse_global = Tx.match_buffer(lse_ptr, [batch_size, NUM_HEADS], "float32", scope="global")

            with Tx.kernel():
                bx = Tx.cta_id([SM_COUNT], parent="kernel")
                tx, ty = Tx.thread_id([BDX, BDY], parent="cta")

                with Tx.cta():
                    # allocate the memory
                    pool = Tx.PoolAllocator()
                    o_tmp_smem = pool.alloc([PIPE_DEPTH, BDY, HEAD_DIM], "float32")
                    lse_tmp_smem_load = pool.alloc([BDY, BDX], "float32")
                    lse_tmp_smem_use = lse_tmp_smem_load.view(BDX, BDY)
                    pool.move_base_to(0)
                    o_epi_smem = pool.alloc([BDY, HEAD_DIM], "float32")
                    lse_epi_smem = pool.alloc([BDY], "float32")
                    pool.commit()

                    with Tx.thread():
                        idx = Tx.alloc_local([1], "int32")
                        head_idx = Tx.alloc_local([1], "int32")
                        batch_idx = Tx.alloc_local([1], "int32")
                        new_beg_batch_idx = Tx.alloc_local([1], "int32")
                        num = Tx.alloc_local([1], "int32")
                        tmp = Tx.alloc_local([VEC_SIZE], "float16")
                        o = Tx.alloc_local([VEC_SIZE], "float32")
                        m = Tx.alloc_local([2], "float32")
                        d = Tx.alloc_local([2], "float32")
                        m_tmp = Tx.alloc_local([1], "float32")
                        o_tmp = Tx.alloc_local([VEC_SIZE], "float32")

                        tx_start = Tx.meta_var(tx * VEC_SIZE)

                        idx[0] = bx
                        while idx[0] < batch_size * NUM_HEADS:
                            head_idx[0] = idx[0] % NUM_HEADS
                            batch_idx[0] = idx[0] // NUM_HEADS
                            new_beg_batch_idx[0] = o_indptr_global[batch_idx[0]]
                            num[0] = o_indptr_global[batch_idx[0] + 1] - new_beg_batch_idx[0]

                            if num[0] == 1:
                                if ty == 0:
                                    Tx.copy(o[:], o_tmp_global[new_beg_batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE])
                                    Tx.cast(tmp[:], o[:])
                                    Tx.copy(o_global[batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE], tmp[:])
                                    lse_global[batch_idx[0], head_idx[0]] = lse_tmp_global[new_beg_batch_idx[0], head_idx[0]]
                                continue

                            # pipeline
                            for kp in Tx.unroll(PIPE_DEPTH):
                                if kp * BDY + ty < num[0]:
                                    Tx.copy_async(o_tmp_smem[kp, ty, tx_start:tx_start + VEC_SIZE],
                                                  o_tmp_global[new_beg_batch_idx[0] + kp * BDY + ty, head_idx[0], tx_start:tx_start + VEC_SIZE],
                                                  dispatch="non-bulk-copy", vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()

                            # initialize the value
                            m[0] = Tx.float32('-inf')
                            d[0] = 1.0
                            for kv in Tx.unroll(VEC_SIZE):
                                o[kv] = 0.0

                            for ki in Tx.serial(ceildiv(num[0], BDY)):
                                if ki % BDX == 0:
                                    # load lse
                                    if ki * BDY + ty * BDX + tx < num[0]:
                                        lse_tmp_smem_load[ty, tx] = lse_tmp_global[new_beg_batch_idx[0] + ki * BDY + ty * BDX + tx, head_idx[0]]
                                    else:
                                        lse_tmp_smem_load[ty, tx] = 0.0
                                    Tx.ptx.fence.proxy("shared")
                                    Tx.ptx.bar.sync(2, BDX * BDY)

                                Tx.ptx.cp_async.wait_group(PIPE_DEPTH - 1)
                                Tx.ptx.bar.sync(2, BDX * BDY)
                                Tx.ptx.fence.proxy("shared")

                                for kv in Tx.serial(VEC_SIZE):
                                    o_tmp[kv] = o_tmp_smem[ki % PIPE_DEPTH, ty, tx * VEC_SIZE + kv]
                                if ki * BDY + ty < num[0]:
                                    m_tmp[0] = lse_tmp_smem_use[ki % BDX, ty]
                                    m[1] = m[0]
                                    d[1] = d[0]
                                    m[0] = Tx.max(m[1], m_tmp[0])
                                    d[0] = d[1] * Tx.exp2(m[1] - m[0]) + Tx.exp2(m_tmp[0] - m[0])
                                    for kv in Tx.unroll(VEC_SIZE):
                                        o[kv] = o[kv] * Tx.exp2(m[1] - m[0]) + o_tmp[kv] * Tx.exp2(m_tmp[0] - m[0])
                                Tx.ptx.bar.sync(2, BDX * BDY)
                                if (PIPE_DEPTH + ki) * BDY + ty < num[0]:
                                    Tx.copy_async(o_tmp_smem[ki % PIPE_DEPTH, ty, tx_start:tx_start + VEC_SIZE],
                                                  o_tmp_global[new_beg_batch_idx[0] + (ki + PIPE_DEPTH) * BDY + ty, head_idx[0], tx_start:tx_start + VEC_SIZE],
                                                  dispatch="non-bulk-copy", vec_len=VEC_SIZE)
                                Tx.ptx.cp_async.commit_group()
                            Tx.ptx.cp_async.wait_group(0)
                            Tx.ptx.bar.sync(2, BDX * BDY)
                            # normalize
                            for kv in Tx.unroll(VEC_SIZE):
                                o[kv] = o[kv] / d[0]

                            # reduce
                            for kv in Tx.serial(VEC_SIZE):
                                o_epi_smem[ty, tx * VEC_SIZE + kv] = o[kv]
                            lse_epi_smem[ty] = m[0] + Tx.log2(d[0])
                            m[0] = Tx.float32('-inf')
                            d[0] = 1.0
                            for kv in Tx.serial(VEC_SIZE):
                                o[kv] = 0.0
                            Tx.ptx.fence.proxy("shared")
                            Tx.ptx.bar.sync(2, BDX * BDY)
                            if ty == 0:
                                for ky in Tx.serial(BDY):
                                    m_tmp[0] = lse_epi_smem[ky]
                                    for kv in Tx.serial(VEC_SIZE):
                                        o_tmp[kv] = o_epi_smem[ky, tx * VEC_SIZE + kv]
                                    m[1] = m[0]
                                    d[1] = d[0]
                                    m[0] = Tx.max(m[1], m_tmp[0])
                                    d[0] = d[1] * Tx.exp2(m[1] - m[0]) + Tx.exp2(m_tmp[0] - m[0])
                                    for kv in Tx.unroll(VEC_SIZE):
                                        o[kv] = o[kv] * Tx.exp2(m[1] - m[0]) + o_tmp[kv] * Tx.exp2(m_tmp[0] - m[0])

                                for kv in Tx.unroll(VEC_SIZE):
                                    o[kv] = o[kv] / d[0]

                                # store to global mem
                                Tx.cast(tmp[:], o[:])
                                Tx.copy(o_global[batch_idx[0], head_idx[0], tx_start:tx_start + VEC_SIZE], tmp[:])
                                if tx == 0:
                                    lse_global[batch_idx[0], head_idx[0]] = m[0] + Tx.log2(d[0])
                            idx[0] += SM_COUNT
        # fmt: on
        return merge_kernel

    return decode(plan_info.split_kv), merge()


@pytest.mark.parametrize("num_heads", [(64, 8)])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test(num_heads, seq_len, head_dim, batch_size):
    PAGE_SIZE = 16
    MAX_PAGE_NUM = 32768
    qo_heads, kv_heads = num_heads
    plan_info = PlanInfo(qo_heads, kv_heads, head_dim)

    def test_dynamic_batch_size(batch_size):
        Q, KV_data, KV_indptr, KV_last_page_len, KV_indices = perpare_data(
            batch_size, qo_heads, kv_heads, seq_len, head_dim, PAGE_SIZE, MAX_PAGE_NUM
        )

        def tir():
            DEV = tvm.cuda(0)
            q_tvm = tvm.runtime.tensor(Q, DEV)
            kv_data_tvm = tvm.runtime.tensor(KV_data, DEV)
            kv_indptr_tvm = tvm.runtime.tensor(KV_indptr, DEV)
            kv_last_page_len_tvm = tvm.runtime.tensor(KV_last_page_len, DEV)
            kv_indices_tvm = tvm.runtime.tensor(KV_indices, DEV)
            plan_info.plan(batch_size, KV_indptr.numpy().tolist(), PAGE_SIZE, MAX_PAGE_NUM)

            decode, merge = get_decode_kernel(plan_info, PAGE_SIZE)
            mod_decode = tvm.IRModule({"main": decode})
            mod_merge = tvm.IRModule({"main": merge})
            target = tvm.target.Target("cuda")
            with target:
                mod_decode = tvm.compile(mod_decode, target=target, tir_pipeline="tirx")
                mod_merge = tvm.compile(mod_merge, target=target, tir_pipeline="tirx")

            def func():
                if plan_info.split_kv:
                    mod_decode(
                        q_tvm,
                        kv_data_tvm,
                        plan_info.tmp_lse_tvm,
                        kv_indptr_tvm,
                        kv_last_page_len_tvm,
                        kv_indices_tvm,
                        plan_info.request_indices_tvm,
                        plan_info.kv_tile_indices_tvm,
                        plan_info.max_chunk_size,
                        plan_info.tmp_o_tvm,
                    )
                    mod_merge(
                        plan_info.tmp_o_tvm,
                        plan_info.o_indptr_tvm,
                        plan_info.o_tvm,
                        plan_info.tmp_lse_tvm,
                        plan_info.lse_tvm,
                    )
                else:
                    mod_decode(
                        q_tvm,
                        kv_data_tvm,
                        plan_info.lse_tvm,
                        kv_indptr_tvm,
                        kv_last_page_len_tvm,
                        kv_indices_tvm,
                        plan_info.request_indices_tvm,
                        plan_info.kv_tile_indices_tvm,
                        plan_info.max_chunk_size,
                        plan_info.o_tvm,
                    )

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            func()
            print(f"TIR time: {ms:.3f} ms")

            return plan_info.o_tvm.numpy(), plan_info.lse_tvm.numpy()

        def flashinfer_batch_decode():
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
                qo_heads,
                kv_heads,
                head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                kv_data_type=torch.float16,
                q_data_type=torch.float16,
            )
            o, lse = wrapper.run_return_lse(q_f, kv_data_f)

            func = lambda: wrapper.run(q_f, kv_data_f)
            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_decode")
            func()
            print(f"FlashInfer BatchDecodeWithPagedKVCacheWrapper time: {ms:.3f} ms")

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_decode_tensor_cores():
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
                qo_heads,
                kv_heads,
                head_dim,
                PAGE_SIZE,
                pos_encoding_mode="NONE",
                kv_data_type=torch.float16,
                q_data_type=torch.float16,
            )
            o, lse = wrapper.run_return_lse(q_f, kv_data_f)

            func = lambda: wrapper.run(q_f, kv_data_f)
            ms = bench(
                func, warmup=10, repeat=30, proton_name="flashinfer_batch_decode_tensor_cores"
            )
            func()
            print(
                f"FlashInfer BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) time: {ms:.3f} ms"
            )

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_prefill():
            import flashinfer
            import torch

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

            func = lambda: wrapper.run(q_f, kv_data_f)
            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_prefill")
            func()
            print(f"FlashInfer BatchPrefillWithPagedKVCacheWrapper time: {ms:.3f} ms")

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        def flashinfer_batch_attention():
            import flashinfer
            import torch

            q_f = Q.to(0).reshape(batch_size, qo_heads, head_dim)
            kv_data_f = KV_data.to(0)
            kv_indptr_f = KV_indptr.to(0)
            kv_last_page_len_f = KV_last_page_len.to(0)
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
            o, lse = wrapper.run(q_f, kv_data_f)

            func = lambda: wrapper.run(q_f, kv_data_f)
            ms = bench(func, warmup=10, repeat=30, proton_name="flashinfer_batch_attention")
            func()
            print(f"FlashInfer BatchAttention time: {ms:.3f} ms")

            return (
                o.reshape(batch_size, qo_heads, head_dim).cpu().numpy(),
                lse.reshape(batch_size, qo_heads).cpu().numpy(),
            )

        with ProtonContext("batch_decode"):
            print(
                f">>>>>>>>>>>>>>>>>>>>>>>>> Testing (B,(H_qo,H_kv),N,D) = ({batch_size},({qo_heads},{kv_heads}),{seq_len},{head_dim})"
            )
            O_flashinfer_batch_decode, lse_flashinfer_batch_decode = flashinfer_batch_decode()
            O_flashinfer_batch_decode_tensor_cores, lse_flashinfer_batch_decode_tensor_cores = (
                flashinfer_batch_decode_tensor_cores()
            )
            O_flashinfer_batch_prefill, lse_flashinfer_batch_prefill = flashinfer_batch_prefill()
            O_flashinfer_batch_attention, lse_flashinfer_batch_attention = (
                flashinfer_batch_attention()
            )
            O_tir, lse_tir = tir()

            np.testing.assert_allclose(O_tir, O_flashinfer_batch_decode, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lse_tir, lse_flashinfer_batch_decode, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(
                O_tir, O_flashinfer_batch_decode_tensor_cores, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                lse_tir, lse_flashinfer_batch_decode_tensor_cores, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(O_tir, O_flashinfer_batch_prefill, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lse_tir, lse_flashinfer_batch_prefill, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(O_tir, O_flashinfer_batch_attention, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(
                lse_tir, lse_flashinfer_batch_attention, rtol=1e-3, atol=1e-3
            )

    test_dynamic_batch_size(batch_size)


if __name__ == "__main__":
    import itertools

    num_heads_list = [(32, 8)]
    seq_len_list = [512, 3456]
    head_dim_list = [128]
    batch_size_list = [1, 128]

    for num_heads, seq_len, head_dim, batch_size in itertools.product(
        num_heads_list, seq_len_list, head_dim_list, batch_size_list
    ):
        test(num_heads, seq_len, head_dim, batch_size)
