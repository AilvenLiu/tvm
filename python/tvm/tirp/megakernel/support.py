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
"""Plan info for attention kernel."""
import numpy as np
import torch

import tvm
from tvm.tirp.megakernel.allreduce import AllreduceTile
from tvm.tirp.megakernel.common import KernelConfig
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile
from tvm.tirp.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirp.megakernel.decode_merge import DecodeMergeTile

# Paged kv-cache config
KV_LAYOUT = "HND"

# HW config
SM_COUNT = 148
SMEM_SIZE = 232448

# Other
F16_BYTE = 2
F32_BYTE = 4


def ceildiv(a, b):
    return (a + b - 1) // b


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
        if isinstance(kv_indptr_h, tvm.nd.NDArray):
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
        self.max_chunk_size_tvm = tvm.nd.array(
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

        self.request_indices_tvm = tvm.nd.array(np.array(request_indices, dtype=np.int32), DEV)
        self.kv_tile_indices_tvm = tvm.nd.array(np.array(kv_tile_indices, dtype=np.int32), DEV)
        self.o_indptr_tvm = tvm.nd.array(np.array(o_indptr, dtype=np.int32), DEV)


@tvm.register_func("megakernel.decode_attn_plan")
def decode_attn_plan(
    qo_heads, kv_heads, head_dim, batch_size, kv_indptr_h, page_size, max_page_num
):
    plan_info = PlanInfo(qo_heads, kv_heads, head_dim, enforce_no_split_kv=True)
    plan_info.plan(batch_size, kv_indptr_h, page_size, max_page_num)
    return (
        plan_info.request_indices_tvm,
        plan_info.kv_tile_indices_tvm,
        plan_info.max_chunk_size_tvm,
        plan_info.o_indptr_tvm,
    )


SPLIT_QKV_PROJECT = {
    1: 3,
    8: 4,
}
SPLIT_O_PROJRCT = {
    1: 3,
    8: 2,
}
DOWN_PROJ_SPLIT_K_FACTOR = {
    1: 10,
    8: 3,
}


VOCAB_SIZE = 151936
MAX_POSITION_EMBEDDINGS = 40960
HIDDEN_SIZE = 5120
FULL_INTERMEDIATE_SIZE = 25600
NUM_HIDDEN_LAYERS = 64
FULL_NUM_ATTENTION_HEADS = 64
FULL_NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000
MAX_PAGE_NUM = 8192
PAGE_SIZE = 16


def generate_event_tensor(batch_size, o_indptr, split_kv, WORLD_SIZE):
    """The event tensor generation function for layer testing use."""
    INTERMEDIATE_SIZE = FULL_INTERMEDIATE_SIZE // WORLD_SIZE
    NUM_ATTENTION_HEADS = FULL_NUM_ATTENTION_HEADS // WORLD_SIZE
    NUM_KEY_VALUE_HEADS = FULL_NUM_KEY_VALUE_HEADS // WORLD_SIZE
    DEV = tvm.cuda(0)
    qkv_h_d = (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM
    qk_h_d = (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) * HEAD_DIM
    q_h_d = NUM_ATTENTION_HEADS * HEAD_DIM
    k_h_d = NUM_KEY_VALUE_HEADS * HEAD_DIM
    etensor_qkv_partial = tvm.nd.array(
        np.zeros(ceildiv(qkv_h_d, SplitKReduceTile.N_UNIT), dtype=np.int32), device=DEV
    )
    etensor_q_reduce = tvm.nd.array(
        np.zeros((batch_size, ceildiv(q_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
    )
    etensor_k_reduce = tvm.nd.array(
        np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
    )
    etensor_v_reduce = tvm.nd.array(
        np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
    )
    etensor_decode = tvm.nd.array(
        np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
    )
    etensor_decode_merge = np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32)
    o_indptr = o_indptr.numpy()
    for b in range(batch_size):
        num_merge = o_indptr[b + 1] - o_indptr[b]
        for j in range(NUM_KEY_VALUE_HEADS):
            etensor_decode_merge[b, j] = num_merge
    etensor_decode_merge = tvm.nd.array(etensor_decode_merge, device=DEV)
    etensor_o_proj = np.zeros(SPLIT_O_PROJRCT[WORLD_SIZE], dtype=np.int32)
    o_proj_tile_k = (
        ceildiv(
            ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SPLIT_O_PROJRCT[WORLD_SIZE]), GemmTile.BLK_K
        )
        * GemmTile.BLK_K
    )
    if split_kv:
        for h in range(NUM_ATTENTION_HEADS // DecodeMergeTile.bdz):
            range_start = h * DecodeMergeTile.bdz * HEAD_DIM // o_proj_tile_k
            range_end = ((h + 1) * DecodeMergeTile.bdz * HEAD_DIM - 1) // o_proj_tile_k
            for i in range(range_start, range_end + 1):
                etensor_o_proj[i] += 1
        etensor_o_proj *= batch_size
        etensor_o_proj = tvm.nd.array(etensor_o_proj, device=DEV)
    else:
        for h in range(NUM_KEY_VALUE_HEADS):
            range_start = h * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile_k
            range_end = (
                (h + 1) * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM - 1
            ) // o_proj_tile_k
            for i in range(range_start, range_end + 1):
                etensor_o_proj[i] += 1
        etensor_o_proj *= batch_size
        etensor_o_proj = tvm.nd.array(etensor_o_proj, device=DEV)
    etensor_o_partial = tvm.nd.array(
        np.zeros(ceildiv(HIDDEN_SIZE, GemmTile.BLK_N), dtype=np.int32), device=DEV
    )
    etensor_o_allreduce = tvm.nd.array(
        np.zeros(HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE, dtype=np.int32), device=DEV
    )
    etensor_attn_add_rms_norm = tvm.nd.array(np.zeros(batch_size, dtype=np.int32), device=DEV)
    etensor_attn_mlp = tvm.nd.array(np.zeros(1, dtype=np.int32), device=DEV)
    etensor_gate_up_proj = tvm.nd.array(
        np.zeros(INTERMEDIATE_SIZE // GemmTile.BLK_N, dtype=np.int32), device=DEV
    )
    etensor_down_proj = np.zeros(DOWN_PROJ_SPLIT_K_FACTOR[WORLD_SIZE], dtype=np.int32)
    down_proj_tile_k = ceildiv(ceildiv(INTERMEDIATE_SIZE, DOWN_PROJ_SPLIT_K_FACTOR[WORLD_SIZE]), GemmTile.BLK_K) * GemmTile.BLK_K
    for m in range(INTERMEDIATE_SIZE // SiluMultiplyTile.TILE_SIZE):
        range_start = m * SiluMultiplyTile.TILE_SIZE // down_proj_tile_k
        range_end = ((m + 1) * SiluMultiplyTile.TILE_SIZE - 1) // down_proj_tile_k
        for i in range(range_start, range_end + 1):
            etensor_down_proj[i] += 1
    etensor_down_proj = tvm.nd.array(etensor_down_proj, device=DEV)
    etensor_down_proj_reduce = tvm.nd.array(
        np.zeros(HIDDEN_SIZE // GemmTile.BLK_N, dtype=np.int32), device=DEV
    )
    etensor_down_proj_allreduce = tvm.nd.array(
        np.zeros(HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE, dtype=np.int32), device=DEV
    )
    etensor_mlp_add_rms_norm = tvm.nd.array(np.zeros((batch_size,), dtype=np.int32), device=DEV)

    etensor_end = tvm.nd.array(np.zeros(1, dtype=np.int32), device=DEV)
    return (
        etensor_qkv_partial,
        etensor_q_reduce,
        etensor_k_reduce,
        etensor_v_reduce,
        etensor_decode,
        etensor_decode_merge,
        etensor_o_proj,
        etensor_o_partial,
        etensor_o_allreduce,
        etensor_attn_add_rms_norm,
        etensor_attn_mlp,
        etensor_gate_up_proj,
        etensor_down_proj,
        etensor_down_proj_reduce,
        etensor_down_proj_allreduce,
        etensor_mlp_add_rms_norm,
        etensor_end,
    )
