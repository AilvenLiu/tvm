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
import threading
from typing import List, Literal
import torch
from pathlib import Path

import tvm_ffi

import tvm
from tvm.script import tir as T
from tvm.tirx.megakernel.allreduce import AllreduceTile
from tvm.tirx.megakernel.common import KernelConfig, pack_into_32bit
from tvm.tirx.megakernel.gemm import GemmTile
from tvm.tirx.megakernel.gemm_splitk_reduce import SplitKReduceTile, MOETopKReduceTile
from tvm.tirx.megakernel.split_silu_multiply import SiluMultiplyTile, SiluMultiplyMOETile
from tvm.tirx.megakernel.gate_up_silu import GateUpSiluTile
from tvm.tirx.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirx.megakernel.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tvm.tirx.megakernel.group_gemm_sm100 import GroupGEMMTile
from tvm.tirx.megakernel.common import event_type_names
from tvm.tirx.bench.utils import export_to_perfetto_trace

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


# plan for batch decode
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
        self.max_chunk_size_tvm = tvm.runtime.tensor(
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


@tvm_ffi.register_global_func("megakernel.decode_attn_plan")
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


def get_inverse_plan_info(batch_size, kv_head_num, q_indptr, kv_head_idx, attn_task_num):
    """For layer testing use."""
    MAX_TOTAL_NUM_WORKERS = 1025
    inverse_info = [[] for _ in range(batch_size * kv_head_num)]
    for m in range(attn_task_num):
        bs_idx = q_indptr[m]
        kv_idx = kv_head_idx[m]
        inverse_info[kv_idx * batch_size + bs_idx].append((m // KernelConfig.WG_NUMBER) % KernelConfig.SM_NUMBER)
    for m in range(attn_task_num, ceildiv(attn_task_num, KernelConfig.WG_NUMBER) * KernelConfig.WG_NUMBER):
        inverse_info[(m - attn_task_num) % (batch_size * kv_head_num)].append((m // KernelConfig.WG_NUMBER) % KernelConfig.SM_NUMBER) # align attn_task_num
    inverse_indptr = [0 for _ in range(MAX_TOTAL_NUM_WORKERS)]
    inverse_indices = [0 for _ in range(MAX_TOTAL_NUM_WORKERS)]
    for i in range(batch_size * kv_head_num):
        inverse_indptr[i + 1] = inverse_indptr[i] + len(inverse_info[i])
        for j in range(len(inverse_info[i])):
            inverse_indices[inverse_indptr[i] + j] = inverse_info[i][j]
    assert inverse_indptr[batch_size * kv_head_num] == ceildiv(attn_task_num, KernelConfig.WG_NUMBER) * KernelConfig.WG_NUMBER

    DEV = tvm.cuda(0)
    return (
        tvm.runtime.tensor(np.array(inverse_indptr, dtype=np.int32), DEV),
        tvm.runtime.tensor(np.array(inverse_indices, dtype=np.int32), DEV)
    )


def push_moe_tasks(central_queue, batch_size, config, insert_wait_etensor_init=False):
    MOE_BLK_M = 128
    gating_blk_m = 128
    for m_idx in range(ceildiv(batch_size, gating_blk_m)):
        for k_idx in range(config["GATING_SPLIT_K_FACTOR"]):
            central_queue.append((m_idx, 0, k_idx, JobType.MOE_GATING.value))
    if insert_wait_etensor_init:
        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))
    for m_idx in range(KernelConfig.SM_NUMBER):
        central_queue.append((m_idx, 0, 0, JobType.MOE_TOPK_SOFTMAX.value))
    central_queue.append((0, 0, 0, JobType.MOE_ALIGN.value))
    for m_idx in range(KernelConfig.SM_NUMBER):
        central_queue.append((m_idx, 0, 0, JobType.MOE_COUNT_AND_SORT.value))
    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, config["NUM_EXPERTS_PER_TOK"], config["NUM_EXPERTS"], MOE_BLK_M
    )
    for m_idx in range(max_num_tokens_padded // MOE_BLK_M):
        for n_idx in range(config["INTERMEDIATE_SIZE"] * 2 // GroupGEMMTile.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value))
    for m_idx in range(max_num_tokens_padded // MOE_BLK_M):
        for n_idx in range(config["HIDDEN_SIZE"] // GroupGEMMTile.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_DOWN.value))


def generate_exec_queue(batch_size, attn_task_num, config, WORLD_SIZE, etensor_num, scheduler: Literal["static", "dynamic"]):
    """The execution queue generation function for layer testing use."""
    INTERMEDIATE_SIZE = config["INTERMEDIATE_SIZE"] // WORLD_SIZE
    NUM_ATTENTION_HEADS = config["NUM_ATTENTION_HEADS"] // WORLD_SIZE
    NUM_KEY_VALUE_HEADS = config["NUM_KEY_VALUE_HEADS"] // WORLD_SIZE
    is_moe = "NUM_EXPERTS" in config
    torch.cuda.nvtx.range_push("generate_exec_queue")
    if scheduler == "static":
        exec_queue = np.zeros(
            (KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32
        )
        central_queue = []

        # init etensor
        for i in range(etensor_num):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))

        # qkv projection
        for n_idx in range(
            ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * config["HEAD_DIM"], GemmTile.BLK_N)
        ):
            for k_idx in range(config["SPLIT_QKV_PROJECT_DICT"][WORLD_SIZE]):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_QKV_PROJ.value))
        
        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))

        m_split = ceildiv(KernelConfig.SM_NUMBER, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS)
        m_tile = ceildiv(batch_size, m_split)
        m_split = ceildiv(batch_size, m_tile)

        # q reduce + rmsnorm + rope
        for m_idx in range(m_split):
            for n_idx in range(NUM_ATTENTION_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.Q_REDUCE_RMS_ROPE.value))

        # k reduce + rmsnorm + rope + append
        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.K_REDUCE_RMS_ROPE_APPEND.value))

        # v reduce + append
        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.V_REDUCE_APPEND.value))

        # attention
        attn_tile_num = ceildiv(attn_task_num, KernelConfig.WG_NUMBER)
        for m_idx in range(min(KernelConfig.SM_NUMBER, attn_tile_num)):
            central_queue.append((m_idx, -1, -1, JobType.BATCH_ATTENTION.value))

        if attn_task_num > NUM_KEY_VALUE_HEADS * batch_size:
            # merge
            for m_idx in range(
                ceildiv(
                    batch_size * NUM_ATTENTION_HEADS, KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
                )
            ):
                central_queue.append((m_idx, -1, -1, JobType.BATCH_ATTENTION_MERGE.value))

        # o projection
        for n_idx in range(ceildiv(config["HIDDEN_SIZE"], GemmTile.BLK_N)):
            for k_idx in range(config["SPLIT_O_PROJECT_DICT"][WORLD_SIZE]):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_O_PROJ.value))

        if WORLD_SIZE > 1:
            # o reduction
            m_split_o_proj_reduce = min(
                batch_size, KernelConfig.SM_NUMBER // (config["HIDDEN_SIZE"] // SplitKReduceTile.N_UNIT)
            )
            n_tile_o_proj_reduce = (
                ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split_o_proj_reduce))
                * SplitKReduceTile.N_UNIT
            )
            m_tile_o_proj_reduce = ceildiv(batch_size, m_split_o_proj_reduce)
            m_split_o_proj_reduce = ceildiv(batch_size, m_tile_o_proj_reduce)
            for n_idx in range(ceildiv(config["HIDDEN_SIZE"], n_tile_o_proj_reduce)):
                for m_idx in range(m_split_o_proj_reduce):
                    central_queue.append((m_idx, n_idx, 0, JobType.GEMM_O_REDUCE.value))

            # o allreduce
            for m_idx in range(ceildiv(batch_size, AllreduceTile.M_TILE)):
                for n_idx in range(ceildiv(config["HIDDEN_SIZE"] // WORLD_SIZE, AllreduceTile.N_TILE)):
                    central_queue.append((m_idx, n_idx, 0, JobType.O_ALLREDUCE.value))

        # add rmsnorm
        for m_idx in range(batch_size):
            central_queue.append((m_idx, -1, -1, JobType.ATTN_ADD_RMS_NORM.value))

        if is_moe:
            push_moe_tasks(central_queue, batch_size, config)
        else:
            if config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE] == 1:
                # gate_up_silu
                assert INTERMEDIATE_SIZE % GateUpSiluTile.BLK_N == 0
                for n_idx in range(INTERMEDIATE_SIZE * 2 // GateUpSiluTile.BLK_N):
                    central_queue.append((0, n_idx, 0, JobType.GATE_UP_SILU.value))
            else:
                # gate_up_proj
                assert INTERMEDIATE_SIZE % GemmTile.BLK_N == 0
                for n_idx in range(INTERMEDIATE_SIZE // GemmTile.BLK_N):
                    for k_idx in range(config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE]):
                        central_queue.append((0, n_idx, k_idx, JobType.GEMM_GATE_UP_PROJ.value))
                        central_queue.append((0, n_idx + INTERMEDIATE_SIZE // GemmTile.BLK_N, k_idx, JobType.GEMM_GATE_UP_PROJ.value))

                # gate_up reduce
                m_split_gate_up_proj_reduce = min(
                    batch_size,
                    KernelConfig.SM_NUMBER // (INTERMEDIATE_SIZE * 2 // SplitKReduceTile.N_UNIT),
                )
                m_tile_gate_up_proj_reduce = ceildiv(batch_size, m_split_gate_up_proj_reduce)
                m_split_gate_up_proj_reduce = ceildiv(batch_size, m_tile_gate_up_proj_reduce)
                for m_idx in range(m_split_gate_up_proj_reduce):
                    for n_idx in range(ceildiv(INTERMEDIATE_SIZE * 2, SplitKReduceTile.N_UNIT)):
                        central_queue.append((m_idx, n_idx, 0, JobType.GATE_UP_PROJ_REDUCE.value))

                # split_silu_multiply
                for m_idx in range(ceildiv(INTERMEDIATE_SIZE, SiluMultiplyTile.TILE_SIZE)):
                    central_queue.append((m_idx, 0, 0, JobType.SPLIT_SILU_MULTIPLY.value))

            # gemm_down_proj
            for n_idx in range(ceildiv(config["HIDDEN_SIZE"], GemmTile.BLK_N)):
                for k_idx in range(config["DOWN_PROJ_SPLIT_K_FACTOR_DICT"][WORLD_SIZE]):
                    central_queue.append((0, n_idx, k_idx, JobType.GEMM_DOWN_PROJ.value))

            if WORLD_SIZE > 1:
                # down_proj_reduce
                m_split_down_proj_reduce = min(
                    KernelConfig.SM_NUMBER // (config["HIDDEN_SIZE"] // SplitKReduceTile.N_UNIT), batch_size
                )
                n_tile = (
                    ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split_down_proj_reduce))
                    * SplitKReduceTile.N_UNIT
                )
                m_tile_down_proj_reduce = ceildiv(batch_size, m_split_down_proj_reduce)
                m_split_down_proj_reduce = ceildiv(batch_size, m_tile_down_proj_reduce)
                for m_idx in range(m_split_down_proj_reduce):
                    for n_idx in range(ceildiv(config["HIDDEN_SIZE"], n_tile)):
                        central_queue.append((m_idx, n_idx, 0, JobType.DOWN_PROJ_REDUCE.value))

                # down_proj_allreduce
                for m_idx in range(ceildiv(batch_size, AllreduceTile.M_TILE)):
                    for n_idx in range(ceildiv(config["HIDDEN_SIZE"] // WORLD_SIZE, AllreduceTile.N_TILE)):
                        central_queue.append((m_idx, n_idx, 0, JobType.DOWN_PROJ_ALLREDUCE.value))

        # add_rms_norm
        for m_idx in range(batch_size):
            central_queue.append((m_idx, 0, 0, JobType.MLP_ADD_RMS_NORM.value))

        tile_idx = 0
        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0))
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        DEV = tvm.cuda(0)
        ret = tvm.runtime.tensor(exec_queue, device=DEV)
        torch.cuda.nvtx.range_pop()
        return ret
    elif scheduler == "dynamic":
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        # init etensor
        for i in range(etensor_num):
            exec_queue.enqueue(JobType.INIT_ETENSOR.value, i, 0, 0)
        for n in reversed(range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * config["HEAD_DIM"], GemmTile.BLK_N))):
            for k in range(config["SPLIT_QKV_PROJECT_DICT"][WORLD_SIZE]):
                exec_queue.enqueue(JobType.GEMM_QKV_PROJ.value, 0, n, k)
        torch.cuda.nvtx.range_pop()
        return exec_queue
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")


@tvm_ffi.register_global_func("tirx.megakernel.get_max_num_tokens_padded")
def get_max_num_tokens_padded(batch_size, topk, num_experts, moe_blk_m):
    if isinstance(batch_size, int):
        if batch_size * topk < num_experts:
            return batch_size * topk * moe_blk_m
        else:
            return (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m
    else:
        return T.if_then_else(batch_size * topk < num_experts, batch_size * topk * moe_blk_m, (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m)


def get_max_blocks_padded_relaxed(batch_size, topk, num_experts, moe_blk_m):
    return batch_size * topk // moe_blk_m + (num_experts + 1)

def generate_etensor_unmatched_dim(i, dim_len, in_par_size, out_par_size):
    start_out_par = i * out_par_size
    end_out_par = min(dim_len, (i + 1) * out_par_size)
    return (end_out_par - 1) // in_par_size - start_out_par // in_par_size + 1


def generate_exec_queue_moe(
    batch_size, config, etensor_num, scheduler: Literal["static", "dynamic"]
):
    torch.cuda.nvtx.range_push("generate_exec_queue")
    if scheduler == "static":
        exec_queue = np.zeros(
                (KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32
            )
        central_queue = []
        # init etensor
        for i in range(etensor_num):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))
        push_moe_tasks(central_queue, batch_size, config, insert_wait_etensor_init=True)
        tile_idx = 0

        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0))
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        DEV = tvm.cuda(0)
        ret = tvm.runtime.tensor(exec_queue, device=DEV)
        torch.cuda.nvtx.range_pop()
        return ret
    elif scheduler == "dynamic":
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        gating_blk_m = 128
        # init etensor
        for i in range(etensor_num):
            exec_queue.enqueue(JobType.INIT_ETENSOR.value, i, 0, 0)
        for m in range(ceildiv(batch_size, gating_blk_m)):
            for k in range(config["GATING_SPLIT_K_FACTOR"]):
                exec_queue.enqueue(JobType.MOE_GATING.value, m, 0, k)
        torch.cuda.nvtx.range_pop()
        return exec_queue





class ProfilerHandler:
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
        # TODO: make this configurable if needed
        self.trigger_count = 167
        self.profiler_layer_id = [3, 47]
        self.dir_path = Path("~/qwen3-mg-debug").expanduser()
        self.file_name = "qwen3-model-mega"

    def export_trace(self, profiler_buffer, rank):
        with self.lock:
            self.counter += 1
            current_run = self.counter

        if current_run == self.trigger_count:
            for layer_id in self.profiler_layer_id:
                if rank == -1:
                    file_name = f"{self.dir_path}/{self.file_name}-layer{layer_id}.perfetto-trace"
                else:
                    file_name = f"{self.dir_path}/{self.file_name}-layer{layer_id}-rank{rank}.perfetto-trace"
                export_to_perfetto_trace(profiler_buffer[layer_id].numpy(), file_name, event_type_names)
                print(f"Exported layer {layer_id} to {file_name}")

profiler_handler = ProfilerHandler()

@tvm_ffi.register_global_func("megakernel.export_trace")
def export_trace(profiler_buffer, rank):
    profiler_handler.export_trace(profiler_buffer, rank)
