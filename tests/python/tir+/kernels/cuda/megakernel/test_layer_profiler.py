from enum import Enum
import numpy as np
import pytest

import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
import tvm.testing
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.static_scheduler import StaticTileScheduler, JobType
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.rms_norm import RMSnormTile
from tvm.tirp.megakernel.rope import RopeTile
from tvm.tirp.megakernel.append_kv import AppendKVTile
from tvm.tirp.megakernel.batch_decode import DecodeTile
from tvm.tirp.megakernel.decode_merge import DecodeMergeTile
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile

from tvm.tirp.bench.utils import ProtonContext, bench

# model configs
VOCAB_SIZE = 151936
MAX_POSITION_EMBEDDINGS = 40960
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600
NUM_HIDDEN_LAYERS = 64
NUM_ATTENTION_HEADS = 64
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000
MAX_PAGE_NUM = 8192
PAGE_SIZE = 16
SEQ_LEN = 511

problem_config = {
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
    "hidden_size": HIDDEN_SIZE,
    "intermediate_size": INTERMEDIATE_SIZE,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_attention_heads": NUM_ATTENTION_HEADS,
    "num_key_value_heads": NUM_KEY_VALUE_HEADS,
    "head_dim": HEAD_DIM,
    "rms_norm_eps": RMS_NORM_EPS,
    "rope_theta": ROPE_THETA,
    "max_page_num": 128,
    "page_size": 16,
    "seq_len": SEQ_LEN
}

# profiling
NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
PROFILER_BUFFER_SIZE = int(1e6)
PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS
PROFILER_ON = False


SPLIT_QKV_PROJECT = 3
SPLIT_O_PROJRCT = 3
DOWN_PROJ_SPLIT_K_FACTOR = 10

class MegaKernel:
    class_list = [GemmTile, SplitKReduceTile,RMSnormTile, RopeTile, AppendKVTile,
                    DecodeTile, DecodeMergeTile, AddRMSNormTile, SiluMultiplyTile]    

    def __init__(self):
        self.tile_list = []

    def host_init_all(self):
        for tile in self.tile_list:
            tile.host_init()

    def class_init_all(self, pool_allocator):
        for cls in self.class_list:
            cls.class_init(pool_allocator)

    def class_finalize_all(self):
        for cls in self.class_list:
            cls.class_finalize()

    def device_init_all(self, pool_allocator):
        offset = pool_allocator.offset
        max_offset = 0
        for tile in self.tile_list:
            pool_allocator.move_base_to(offset)
            tile.init(pool_allocator)
            max_offset = max(max_offset, pool_allocator.offset)
        pool_allocator.move_base_to(max_offset)

    def get_func_static(self):
        from .static_scheduler import Semaphore
        @T.prim_func(tirp=True)
        def mega_kernel(hidden_state_ptr: T.handle, 
                        qkv_proj_ptr: T.handle, 
                        partital_qkv_ptr: T.handle, 
                        qkv_ptr: T.handle, 
                        rms_weight_ptr: T.handle, 
                        cos_sin_cache_ptr: T.handle, 
                        kv_cache_ptr: T.handle,
                        kv_indptr_ptr: T.handle, 
                        kv_indices_ptr: T.handle, 
                        kv_last_page_len_ptr: T.handle,
                        pos_map_ptr: T.handle, 
                        o_ptr: T.handle, 
                        lse_ptr: T.handle, 
                        o_tmp_ptr: T.handle, 
                        lse_tmp_ptr: T.handle,
                        o_indptr_ptr: T.handle, 
                        request_indices_ptr: T.handle, 
                        kv_tile_indices_ptr: T.handle, 
                        max_chunk_size_ptr: T.handle, 
                        partial_o_ptr: T.handle, 
                        o_proj_ptr: T.handle, 
                        hidden_state_out_ptr: T.handle,
                        attn_add_rms_weight_ptr: T.handle, 
                        residual_ptr: T.handle,
                        w_gate_up_ptr: T.handle,
                        out_gate_up_proj_ptr: T.handle,
                        out_silu_multiply_ptr: T.handle,
                        w_down_proj_ptr: T.handle,
                        partial_sum_down_proj_ptr: T.handle,
                        out_down_proj_ptr: T.handle,
                        mlp_add_rms_weight_ptr: T.handle,
                        etensor_qkv_partial_ptr: T.handle, 
                        etensor_q_reduce_ptr: T.handle, 
                        etensor_k_reduce_ptr: T.handle, 
                        etensor_v_reduce_ptr: T.handle, 
                        etensor_rms_rope_ptr: T.handle, 
                        etensor_k_rope_append_ptr: T.handle, 
                        etensor_decode_ptr: T.handle, 
                        etensor_decode_merge_ptr: T.handle,
                        etensor_o_proj_ptr: T.handle, 
                        etensor_o_partial_ptr: T.handle, 
                        etensor_attn_add_rms_ptr: T.handle,
                        etensor_attn_mlp_ptr: T.handle,
                        etensor_gate_up_proj_ptr: T.handle,
                        etensor_down_proj_ptr: T.handle,
                        etensor_down_proj_reduce_ptr: T.handle,
                        etensor_mlp_add_rms_ptr: T.handle,
                        exec_queue_ptr: T.handle, 
                        profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
            # match buffer
            batch_size = T.int32()
            new_batch_size = T.int32()
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            qkv_proj_global = T.match_buffer(qkv_proj_ptr, [(NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            partital_qkv_global = T.match_buffer(partital_qkv_ptr, [SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            rms_weight_global = T.match_buffer(rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default")
            cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [1, HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            kv_cache_global = T.match_buffer(kv_cache_ptr, [MAX_PAGE_NUM, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            kv_indptr_global = T.match_buffer(kv_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default")
            total_page_num = T.int32()
            kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", scope="global", layout="default")
            kv_last_page_len_global = T.match_buffer(kv_last_page_len_ptr, [batch_size], "int32", scope="global", layout="default")
            pos_map_global = T.match_buffer(pos_map_ptr, [batch_size], "int32", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_ATTENTION_HEADS], 
                                                            "float32", scope="global", layout="default")
            o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS], 
                                                            "float32", scope="global", layout="default")
            o_indptr_global = T.match_buffer(o_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default")
            request_indices_global = T.match_buffer(request_indices_ptr, [new_batch_size], "int32", scope="global", layout="default")
            kv_tile_indices_global = T.match_buffer(kv_tile_indices_ptr, [new_batch_size], "int32", scope="global", layout="default")
            max_chunk_size_global = T.match_buffer(max_chunk_size_ptr, [1], "int32", scope="global", layout="default")
            partial_o_global = T.match_buffer(partial_o_ptr, [SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], 
                                                            "float32", scope="global", layout="default")
            o_proj_global = T.match_buffer(o_proj_ptr, [HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            hidden_state_out_global = T.match_buffer(hidden_state_out_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")
            residual_global = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            w_gate_up_global = T.match_buffer(w_gate_up_ptr, [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE], 
                                              "float16", scope="global", layout="default")
            out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE * 2], 
                                                     "float16", scope="global", layout="default")
            out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], 
                                                      "float16", scope="global", layout="default")
            w_down_proj_global = T.match_buffer(w_down_proj_ptr, [HIDDEN_SIZE, INTERMEDIATE_SIZE], 
                                                "float16", scope="global", layout="default")
            partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], 
                                                          "float32", layout="default")
            out_down_proj_global = T.match_buffer(out_down_proj_ptr, [batch_size, HIDDEN_SIZE], 
                                                  "float16", layout="default")
            mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [HIDDEN_SIZE], 
                                                       "float16", scope="global", layout="default")
            etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_rms_rope_global = T.match_buffer(etensor_rms_rope_ptr, [batch_size, NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS],
                                                            "int32", scope="global", layout="default") 
            etensor_k_rope_append_global = T.match_buffer(etensor_k_rope_append_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_decode_global = T.match_buffer(etensor_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", layout="default")
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)], 
                                                            "int32", scope="global", layout="default") 
            etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", layout="default")
            etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", layout="default")
            etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [INTERMEDIATE_SIZE // GemmTile.BLK_N], 
                                                            "int32", scope="global", layout="default")
            etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR],
                                                            "int32", scope="global", layout="default")
            etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [HIDDEN_SIZE // GemmTile.BLK_N],
                                                            "int32", scope="global", layout="default")
            etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", layout="default")

            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4], 
                                        "int32", scope="global", layout="default")

            A_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

            # initialize tile
            qkv_proj_tile = T.meta_var(GemmTile(hidden_state_global, qkv_proj_global, partital_qkv_global, split_k_factor=SPLIT_QKV_PROJECT))
            qkv_reduce_tile = T.meta_var(SplitKReduceTile(partital_qkv_global, 
                                                          Tp.reshape(qkv_global, [-1, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM]).buffer))
            rmsnorm_tile = T.meta_var(RMSnormTile(rms_weight_global, qkv_global))
            rope_tile = T.meta_var(RopeTile(qkv_global, cos_sin_cache_global))
            append_kv_tile = T.meta_var(AppendKVTile(kv_cache_global, qkv_global, kv_indptr_global, kv_indices_global, 
                                                     kv_last_page_len_global, pos_map_global))
            decode_tile = T.meta_var(DecodeTile(qkv_global, kv_cache_global, o_global, lse_global, o_tmp_global, lse_tmp_global,
                                                kv_indptr_global, kv_last_page_len_global,
                                                kv_indices_global, request_indices_global, kv_tile_indices_global, max_chunk_size_global))
            decode_merge_tile = T.meta_var(DecodeMergeTile(o_indptr_global, o_tmp_global, o_global, lse_tmp_global, lse_global))
            o_proj_tile = T.meta_var(GemmTile(Tp.reshape(o_global, [-1, NUM_ATTENTION_HEADS * HEAD_DIM]).buffer, o_proj_global, partial_o_global, split_k_factor=SPLIT_O_PROJRCT))
            o_reduce_tile = T.meta_var(SplitKReduceTile(partial_o_global, hidden_state_out_global))
            attn_add_rms_tile = T.meta_var(AddRMSNormTile(hidden_state_out_global, residual_global, attn_add_rms_weight_global))

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(hidden_state_out_global, w_gate_up_global, out_gate_up_proj_global, split_k_factor=1))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj_global, out_silu_multiply_global))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply_global, w_down_proj_global, partial_sum_down_proj_global, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj_global, out_down_proj_global))
            mlp_add_rms_norm_tile = T.meta_var(AddRMSNormTile(out_down_proj_global, residual_global, mlp_add_rms_weight_global))

            self.tile_list.append(qkv_proj_tile)
            self.tile_list.append(qkv_reduce_tile)
            self.tile_list.append(rmsnorm_tile)
            self.tile_list.append(rope_tile)
            self.tile_list.append(append_kv_tile)
            self.tile_list.append(decode_tile)
            self.tile_list.append(decode_merge_tile)
            self.tile_list.append(o_proj_tile)
            self.tile_list.append(o_reduce_tile)
            self.tile_list.append(attn_add_rms_tile)
            self.tile_list.append(gemm_gate_up_proj_tile)
            self.tile_list.append(silu_multiply_tile)
            self.tile_list.append(gemm_down_proj_tile)
            self.tile_list.append(down_proj_reduce_tile)
            self.tile_list.append(mlp_add_rms_norm_tile)

            qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj)
            o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj)
            gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
            gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)

            self.host_init_all()

            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                if PROFILER_ON:
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, warp_id)
                with T.cta():
                    wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                    tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                    buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    self.device_init_all(pool)
                    self.class_init_all(pool)

                    # initialize event tensors
                    evt_qkv_partial = T.meta_var(Semaphore(SPLIT_QKV_PROJECT * (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), 
                                                             etensor_qkv_partial_global))
                    evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global))
                    evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global))
                    evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global))
                    evt_rms_rope = T.meta_var(Semaphore(1, etensor_rms_rope_global))
                    evt_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS+2, etensor_decode_global))
                    evt_k_rope_append = T.meta_var(Semaphore(1, etensor_k_rope_append_global))
                    evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global))
                    evt_o_proj = T.meta_var(Semaphore(-1, etensor_o_proj_global))
                    evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global))
                    evt_attn_add_rms = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, o_reduce_tile.N_TILE), etensor_attn_add_rms_global))
                    evt_attn_mlp = T.meta_var(Semaphore(batch_size, etensor_attn_mlp_global))
                    evt_gate_up_proj = T.meta_var(Semaphore(2, etensor_gate_up_proj_global))
                    evt_down_proj = T.meta_var(Semaphore(INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR, etensor_down_proj_global))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                                                etensor_down_proj_reduce_global))
                    evt_add_rms_norm = T.meta_var(Semaphore(HIDDEN_SIZE // down_proj_reduce_tile.N_TILE, etensor_mlp_add_rms_global))

                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(StaticTileScheduler(prefix="attn", exec_queue=exec_queue, pool_allocator=pool))
                    tile_scheduler.init(bx, tid)
                    while tile_scheduler.valid():
                        if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_qkv_partial.semaphore_notify(tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
                            evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            qkv_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile:
                                    evt_q_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                                elif tile_scheduler.n_idx < (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile:
                                    evt_k_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile)
                                else:
                                    evt_v_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.Q_RMSNORM_ROPE.value:
                            evt_q_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.K_RMSNORM_ROPE_APPEND_KV.value:
                            evt_k_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rmsnorm_tile.h_tile, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rope_tile.h_tile, tile_scheduler.k_idx)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.V_APPEND_KV.value:
                            evt_v_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 1)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
                            evt_decode.semaphore_wait(tile_scheduler.m_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            evt_decode.semaphore_wait(tile_scheduler.m_idx // append_kv_tile.m_tile, tile_scheduler.n_idx // append_kv_tile.h_tile) # wait for append kv
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                                range_end = T.meta_var((tile_scheduler.n_idx+1) * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                                evt_o_proj.semaphore_notify(range_start)
                                if range_end != range_start:
                                    evt_o_proj.semaphore_notify(range_end)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
                            batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
                            evt_decode.semaphore_wait(batch_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            evt_decode.semaphore_wait(batch_idx // append_kv_tile.m_tile, tile_scheduler.n_idx // append_kv_tile.h_tile) # wait for append kv
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode_merge.semaphore_notify(batch_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
                            evt_decode_merge.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                                range_end = T.meta_var((tile_scheduler.n_idx+1) * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                                evt_o_proj.semaphore_notify(range_start)
                                if range_end != range_start:
                                    evt_o_proj.semaphore_notify(range_end)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                            evt_o_proj.semaphore_wait(tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_o_partial.semaphore_notify(tile_scheduler.n_idx // (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                            evt_o_partial.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_attn_add_rms.semaphore_notify(tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                            evt_attn_add_rms.semaphore_wait(tile_scheduler.m_idx // o_reduce_tile.M_TILE)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            attn_add_rms_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_attn_mlp.semaphore_notify(0)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            evt_attn_mlp.semaphore_wait(0)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx - INTERMEDIATE_SIZE // GemmTile.BLK_N)
                                    else:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            evt_gate_up_proj.semaphore_wait(tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_down_proj.semaphore_notify(tile_scheduler.m_idx // (INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                            evt_down_proj.semaphore_wait(tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_down_proj_reduce.semaphore_notify(tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            evt_down_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_add_rms_norm.semaphore_notify(tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                            evt_add_rms_norm.semaphore_wait(tile_scheduler.m_idx//(batch_size//down_proj_reduce_tile.M_split))
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            mlp_add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))

                        tile_scheduler.next_tile()
                    self.class_finalize_all()
        return mega_kernel

    def get_func_dynamic(self):
        from .dynamic_scheduler import Semaphore

        @T.prim_func(tirp=True)
        def mega_kernel(hidden_state_ptr: T.handle, 
                        qkv_proj_ptr: T.handle, 
                        partital_qkv_ptr: T.handle, 
                        qkv_ptr: T.handle, 
                        rms_weight_ptr: T.handle, 
                        cos_sin_cache_ptr: T.handle, 
                        kv_cache_ptr: T.handle,
                        kv_indptr_ptr: T.handle, 
                        kv_indices_ptr: T.handle, 
                        kv_last_page_len_ptr: T.handle,
                        pos_map_ptr: T.handle, 
                        o_ptr: T.handle, 
                        lse_ptr: T.handle, 
                        o_tmp_ptr: T.handle, 
                        lse_tmp_ptr: T.handle,
                        o_indptr_ptr: T.handle, 
                        request_indices_ptr: T.handle, 
                        kv_tile_indices_ptr: T.handle, 
                        max_chunk_size_ptr: T.handle, 
                        partial_o_ptr: T.handle, 
                        o_proj_ptr: T.handle, 
                        hidden_state_out_ptr: T.handle,
                        attn_add_rms_weight_ptr: T.handle, 
                        residual_ptr: T.handle,
                        w_gate_up_ptr: T.handle,
                        out_gate_up_proj_ptr: T.handle,
                        out_silu_multiply_ptr: T.handle,
                        w_down_proj_ptr: T.handle,
                        partial_sum_down_proj_ptr: T.handle,
                        out_down_proj_ptr: T.handle,
                        mlp_add_rms_weight_ptr: T.handle,
                        etensor_qkv_partial_ptr: T.handle, 
                        etensor_q_reduce_ptr: T.handle, 
                        etensor_k_reduce_ptr: T.handle, 
                        etensor_v_reduce_ptr: T.handle, 
                        etensor_rms_rope_ptr: T.handle, 
                        etensor_k_rope_append_ptr: T.handle, 
                        etensor_decode_ptr: T.handle, 
                        etensor_decode_merge_ptr: T.handle,
                        etensor_o_proj_ptr: T.handle, 
                        etensor_o_partial_ptr: T.handle, 
                        etensor_attn_add_rms_ptr: T.handle,
                        etensor_attn_mlp_ptr: T.handle,
                        etensor_gate_up_proj_ptr: T.handle,
                        etensor_down_proj_ptr: T.handle,
                        etensor_down_proj_reduce_ptr: T.handle,
                        etensor_mlp_add_rms_ptr: T.handle,
                        etensor_end: T.Buffer((1, ), "int32"),
                        queue_tasks: T.Buffer((DynamicTileScheduler.MAX_TASKS, 4), "int32"),
                        queue_head: T.Buffer((1,), "int32"),
                        queue_tail: T.Buffer((1,), "int32"),
                        profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
            # match buffer
            batch_size = T.int32()
            new_batch_size = T.int32()
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            qkv_proj_global = T.match_buffer(qkv_proj_ptr, [(NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            partital_qkv_global = T.match_buffer(partital_qkv_ptr, [SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            rms_weight_global = T.match_buffer(rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default")
            cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [1, HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            kv_cache_global = T.match_buffer(kv_cache_ptr, [MAX_PAGE_NUM, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            kv_indptr_global = T.match_buffer(kv_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default")
            total_page_num = T.int32()
            kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", scope="global", layout="default")
            kv_last_page_len_global = T.match_buffer(kv_last_page_len_ptr, [batch_size], "int32", scope="global", layout="default")
            pos_map_global = T.match_buffer(pos_map_ptr, [batch_size], "int32", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_ATTENTION_HEADS], 
                                                            "float32", scope="global", layout="default")
            o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                                            "float32", scope="global", layout="default")
            lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS], 
                                                            "float32", scope="global", layout="default")
            o_indptr_global = T.match_buffer(o_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default")
            request_indices_global = T.match_buffer(request_indices_ptr, [new_batch_size], "int32", scope="global", layout="default")
            kv_tile_indices_global = T.match_buffer(kv_tile_indices_ptr, [new_batch_size], "int32", scope="global", layout="default")
            max_chunk_size_global = T.match_buffer(max_chunk_size_ptr, [1], "int32", scope="global", layout="default")
            partial_o_global = T.match_buffer(partial_o_ptr, [SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], 
                                                            "float32", scope="global", layout="default")
            o_proj_global = T.match_buffer(o_proj_ptr, [HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM], 
                                                            "float16", scope="global", layout="default")
            hidden_state_out_global = T.match_buffer(hidden_state_out_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")
            residual_global = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], 
                                                            "float16", scope="global", layout="default")
            w_gate_up_global = T.match_buffer(w_gate_up_ptr, [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE], 
                                              "float16", scope="global", layout="default")
            out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE * 2], 
                                                     "float16", scope="global", layout="default")
            out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], 
                                                      "float16", scope="global", layout="default")
            w_down_proj_global = T.match_buffer(w_down_proj_ptr, [HIDDEN_SIZE, INTERMEDIATE_SIZE], 
                                                "float16", scope="global", layout="default")
            partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], 
                                                          "float32", layout="default")
            out_down_proj_global = T.match_buffer(out_down_proj_ptr, [batch_size, HIDDEN_SIZE], 
                                                  "float16", layout="default")
            mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [HIDDEN_SIZE], 
                                                       "float16", scope="global", layout="default")
            etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                                            "int32", scope="global", layout="default")
            etensor_rms_rope_global = T.match_buffer(etensor_rms_rope_ptr, [batch_size, NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS],
                                                            "int32", scope="global", layout="default") 
            etensor_decode_global = T.match_buffer(etensor_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_k_rope_append_global = T.match_buffer(etensor_k_rope_append_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", layout="default")
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)], 
                                                            "int32", scope="global", layout="default") 
            etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", layout="default")
            etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", layout="default")
            etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [INTERMEDIATE_SIZE // GemmTile.BLK_N], 
                                                            "int32", scope="global", layout="default")
            etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR],
                                                            "int32", scope="global", layout="default")
            etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [HIDDEN_SIZE // GemmTile.BLK_N],
                                                            "int32", scope="global", layout="default")
            etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", layout="default")

            A_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

            # initialize tile
            qkv_proj_tile = T.meta_var(GemmTile(hidden_state_global, qkv_proj_global, partital_qkv_global, split_k_factor=SPLIT_QKV_PROJECT))
            qkv_reduce_tile = T.meta_var(SplitKReduceTile(partital_qkv_global, 
                                                          Tp.reshape(qkv_global, [-1, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM]).buffer))
            rmsnorm_tile = T.meta_var(RMSnormTile(rms_weight_global, qkv_global))
            rope_tile = T.meta_var(RopeTile(qkv_global, cos_sin_cache_global))
            append_kv_tile = T.meta_var(AppendKVTile(kv_cache_global, qkv_global, kv_indptr_global, kv_indices_global, 
                                                     kv_last_page_len_global, pos_map_global))
            decode_tile = T.meta_var(DecodeTile(qkv_global, kv_cache_global, o_global, lse_global, o_tmp_global, lse_tmp_global,
                                                kv_indptr_global, kv_last_page_len_global,
                                                kv_indices_global, request_indices_global, kv_tile_indices_global, max_chunk_size_global))
            decode_merge_tile = T.meta_var(DecodeMergeTile(o_indptr_global, o_tmp_global, o_global, lse_tmp_global, lse_global))
            o_proj_tile = T.meta_var(GemmTile(Tp.reshape(o_global, [-1, NUM_ATTENTION_HEADS * HEAD_DIM]).buffer, o_proj_global, partial_o_global, split_k_factor=SPLIT_O_PROJRCT))
            o_reduce_tile = T.meta_var(SplitKReduceTile(partial_o_global, hidden_state_out_global))
            attn_add_rms_tile = T.meta_var(AddRMSNormTile(hidden_state_out_global, residual_global, attn_add_rms_weight_global))

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(hidden_state_out_global, w_gate_up_global, out_gate_up_proj_global, split_k_factor=1))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj_global, out_silu_multiply_global))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply_global, w_down_proj_global, partial_sum_down_proj_global, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj_global, out_down_proj_global))
            mlp_add_rms_norm_tile = T.meta_var(AddRMSNormTile(out_down_proj_global, residual_global, mlp_add_rms_weight_global))

            self.tile_list.append(qkv_proj_tile)
            self.tile_list.append(qkv_reduce_tile)
            self.tile_list.append(rmsnorm_tile)
            self.tile_list.append(rope_tile)
            self.tile_list.append(append_kv_tile)
            self.tile_list.append(decode_tile)
            self.tile_list.append(decode_merge_tile)
            self.tile_list.append(o_proj_tile)
            self.tile_list.append(o_reduce_tile)
            self.tile_list.append(attn_add_rms_tile)
            self.tile_list.append(gemm_gate_up_proj_tile)
            self.tile_list.append(silu_multiply_tile)
            self.tile_list.append(gemm_down_proj_tile)
            self.tile_list.append(down_proj_reduce_tile)
            self.tile_list.append(mlp_add_rms_norm_tile)

            qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj)
            o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj)
            gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
            gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)

            self.host_init_all()

            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                lane_id = T.thread_id([32], parent="warp")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                if PROFILER_ON:
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, warp_id)
                with T.cta():
                    wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                    tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                    buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    self.device_init_all(pool)
                    self.class_init_all(pool)

                    # initialize event tensors
                    evt_qkv_partial = T.meta_var(Semaphore(SPLIT_QKV_PROJECT * (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), 
                                                             etensor_qkv_partial_global))
                    evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global))
                    evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global))
                    evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global))
                    evt_rms_rope = T.meta_var(Semaphore(1, etensor_rms_rope_global))
                    evt_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS + 2, etensor_decode_global))
                    evt_k_rope_append = T.meta_var(Semaphore(1, etensor_k_rope_append_global))
                    evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global, decrement=True))
                    evt_o_proj = T.meta_var(Semaphore(-1, etensor_o_proj_global, decrement=True))
                    evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global))
                    evt_attn_add_rms = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, o_reduce_tile.N_TILE), etensor_attn_add_rms_global))
                    evt_attn_mlp = T.meta_var(Semaphore(batch_size, etensor_attn_mlp_global))
                    evt_gate_up_proj = T.meta_var(Semaphore(2, etensor_gate_up_proj_global))
                    evt_down_proj = T.meta_var(Semaphore(INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR, etensor_down_proj_global))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                                                etensor_down_proj_reduce_global))
                    evt_add_rms_norm = T.meta_var(Semaphore(HIDDEN_SIZE // down_proj_reduce_tile.N_TILE, etensor_mlp_add_rms_global))
                    evt_end = T.meta_var(Semaphore(batch_size, etensor_end))

                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(DynamicTileScheduler(queue_tasks, queue_head, queue_tail, pool_allocator=pool))
                    tile_scheduler.init(warp_id)
                    while tile_scheduler.valid():
                        if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_QKV_REDUCE.value,
                                0,
                                tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // GemmTile.BLK_N),
                                0,
                                qkv_reduce_tile.M_split,
                                0,
                                warp_id,
                                lane_id,
                                evt_qkv_partial,
                                tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // GemmTile.BLK_N),
                                use_barrier=False,
                            )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            qkv_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile:
                                tile_scheduler.push_task(
                                    JobType.Q_RMSNORM_ROPE.value,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                    0,
                                    warp_id,
                                    evt_q_reduce,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            elif tile_scheduler.n_idx < (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile:
                                tile_scheduler.push_task(
                                    JobType.K_RMSNORM_ROPE_APPEND_KV.value,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile,
                                    0,
                                    warp_id,
                                    evt_k_reduce,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile,
                                )
                            else:
                                tile_scheduler.push_task(
                                    JobType.V_APPEND_KV.value,
                                    tile_scheduler.m_idx, 
                                    tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile,
                                    1,
                                    warp_id,
                                    evt_v_reduce,
                                    tile_scheduler.m_idx, 
                                    tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile,
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.Q_RMSNORM_ROPE.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if new_batch_size == batch_size: # no split kv
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile, # FIXME: this is dangerous when m_tile not divisible by M. 
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                    0,
                                    append_kv_tile.m_tile,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[(tile_scheduler.m_idx+1) * append_kv_tile.m_tile])
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_SPLIT.value,
                                    range_start,
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                    0,
                                    range_end-range_start,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.K_RMSNORM_ROPE_APPEND_KV.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rmsnorm_tile.h_tile, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rope_tile.h_tile, tile_scheduler.k_idx)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0)
                            if batch_size == new_batch_size:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile,
                                    tile_scheduler.n_idx,
                                    0,
                                    append_kv_tile.m_tile,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[(tile_scheduler.m_idx+1) * append_kv_tile.m_tile])
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_SPLIT.value,
                                    range_start,
                                    tile_scheduler.n_idx,
                                    0,
                                    range_end-range_start,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.V_APPEND_KV.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 1)
                            if batch_size == new_batch_size:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile,
                                    tile_scheduler.n_idx,
                                    0,
                                    append_kv_tile.m_tile,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[(tile_scheduler.m_idx+1) * append_kv_tile.m_tile])
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_SPLIT.value,
                                    range_start,
                                    tile_scheduler.n_idx,
                                    0,
                                    range_end-range_start,
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
                            T.tvm_storage_sync("shared")
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
                            range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                            range_end = T.meta_var((tile_scheduler.n_idx+1) * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_O_PROJ.value,
                                0,
                                0,
                                range_start,
                                ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                1,
                                warp_id,
                                lane_id,
                                evt_o_proj,
                                range_start
                            )
                            if range_end != range_start:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.GEMM_O_PROJ.value,
                                    0,
                                    0,
                                    range_end,
                                    ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                    1,
                                    warp_id,
                                    lane_id,
                                    evt_o_proj,
                                    range_end,
                                    use_barrier=False,
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
                            T.tvm_storage_sync("shared")
                            batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
                            tile_scheduler.push_task(
                                JobType.DECODE_MERGE.value,
                                batch_idx, 
                                tile_scheduler.n_idx,
                                0,
                                warp_id,
                                evt_decode_merge,
                                batch_idx,
                                tile_scheduler.n_idx,
                            )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
                            T.tvm_storage_sync("shared")
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                            range_end = T.meta_var((tile_scheduler.n_idx+1) * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_O_PROJ.value,
                                0,
                                0,
                                range_start,
                                ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                1,
                                warp_id,
                                lane_id,
                                evt_o_proj,
                                range_start,
                            )
                            if range_end != range_start:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.GEMM_O_PROJ.value,
                                    0,
                                    0,
                                    range_end,
                                    ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                    1,
                                    warp_id,
                                    lane_id,
                                    evt_o_proj,
                                    range_end,
                                    use_barrier=False,
                                )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_O_REDUCE.value,
                                0,
                                tile_scheduler.n_idx,
                                0,
                                o_reduce_tile.M_split,
                                0,
                                warp_id,
                                lane_id,
                                evt_o_partial,
                                tile_scheduler.n_idx // (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),
                                use_barrier=False
                            )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)

                            tile_scheduler.push_tasks_along_dim(
                                JobType.ATTN_ADD_RMS_NORM.value,
                                tile_scheduler.m_idx * o_reduce_tile.M_TILE,
                                0,
                                0,
                                o_reduce_tile.M_TILE,
                                0,
                                warp_id,
                                lane_id,
                                evt_attn_add_rms,
                                tile_scheduler.m_idx,
                            )                            
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                            T.tvm_storage_sync("shared")
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            attn_add_rms_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_GATE_UP_PROJ.value,
                                0,
                                0,
                                0,
                                ceildiv(INTERMEDIATE_SIZE *2 , GemmTile.BLK_N),
                                1,
                                warp_id,
                                lane_id,
                                evt_attn_mlp,
                                0,
                            )
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                offset = T.meta_var(tile_scheduler.n_idx-INTERMEDIATE_SIZE//GemmTile.BLK_N)
                                tile_scheduler.push_task(JobType.SPLIT_SILU_MULTIPLY.value, offset, 0, 0, warp_id, evt_gate_up_proj, offset, use_barrier=False)
                            else:
                                tile_scheduler.push_task(JobType.SPLIT_SILU_MULTIPLY.value, tile_scheduler.n_idx, 0, 0, warp_id, evt_gate_up_proj, tile_scheduler.n_idx, use_barrier=False)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            offset = T.meta_var(tile_scheduler.m_idx // (INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR))
                            tile_scheduler.push_tasks_along_dim(JobType.GEMM_DOWN_PROJ.value, 0, 0, offset, HIDDEN_SIZE // GemmTile.BLK_N, 1, warp_id, lane_id, evt_down_proj, offset)

                        elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(
                                    ProfileEventType.GEMM_DOWN_PROJ,
                                    profiler_buffer.data,
                                    profiler_tag.data,
                                    profiler_write_offset.data,
                                    PROFILER_WRITE_STRIDE,
                                    lane_id == 0 and (warp_id ==3 or warp_id == 7),
                                )
                            gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(JobType.DOWN_PROJ_REDUCE.value, 0, tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 0, down_proj_reduce_tile.M_split, 0, warp_id, lane_id, evt_down_proj_reduce, tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), use_barrier=False)
                            if PROFILER_ON:
                                T.timer_end_cuda(
                                    ProfileEventType.GEMM_DOWN_PROJ,
                                    profiler_buffer.data,
                                    profiler_tag.data,
                                    profiler_write_offset.data,
                                    PROFILER_WRITE_STRIDE,
                                    lane_id == 0 and (warp_id ==3 or warp_id == 7),
                                )
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(JobType.MLP_ADD_RMS_NORM.value, tile_scheduler.m_idx * (batch_size //down_proj_reduce_tile.M_split), 0, 0, batch_size //down_proj_reduce_tile.M_split, 0, warp_id, lane_id, evt_add_rms_norm, tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                            T.tvm_storage_sync("shared")
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                            mlp_add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(JobType.END.value, 0, 0, 0, KernelConfig.SM_NUMBER, 0, warp_id, lane_id, evt_end, 0)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))

                        if PROFILER_ON:
                            T.timer_start_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                        tile_scheduler.next_tile(warp_id)
                        if PROFILER_ON:
                            T.timer_end_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0 and (warp_id ==3 or warp_id == 7))
                    self.class_finalize_all()

        return mega_kernel


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static, mega_kernel_dynamic):

    def generate_exec_queue(batch_size, new_batch_size, split_kv):
        exec_queue = np.zeros((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4), dtype=np.int32)
        central_queue = []

        # qkv projection
        for n_idx in range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, GemmTile.BLK_N)):
            for k_idx in range(SPLIT_QKV_PROJECT):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_QKV_PROJ.value))

        # qkv reduction
        m_split = min(batch_size, ceildiv(KernelConfig.SM_NUMBER, 
                                                           (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM // SplitKReduceTile.N_UNIT))
        n_tile_qkv_proj_reduce = ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split)) * SplitKReduceTile.N_UNIT
        for m_idx in range(m_split):
            for n_idx in range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, n_tile_qkv_proj_reduce)):
                central_queue.append((m_idx, n_idx, 0, JobType.GEMM_QKV_REDUCE.value))

        for m_idx in range(m_split):
            for n_idx in range(NUM_ATTENTION_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.Q_RMSNORM_ROPE.value))

        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.K_RMSNORM_ROPE_APPEND_KV.value))

        for m_idx in range(m_split):
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                central_queue.append((m_idx, n_idx, -1, JobType.V_APPEND_KV.value))
        # decode
        if split_kv:
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                for m_idx in range(new_batch_size):
                    central_queue.append(((m_idx, n_idx, -1, JobType.BATCH_DECODE_SPLIT.value)))
        else:
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                for m_idx in range(batch_size):
                    central_queue.append(((m_idx, n_idx, -1, JobType.BATCH_DECODE_NO_SPLIT.value)))

        # merge of decode
        if split_kv:
            for n_idx in range(NUM_KEY_VALUE_HEADS):
                for m_idx in range(batch_size):
                    central_queue.append(((m_idx, n_idx, -1, JobType.DECODE_MERGE.value)))

        # o projection
        for n_idx in range(ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)):
            for k_idx in range(SPLIT_O_PROJRCT):
                central_queue.append((0, n_idx, k_idx, JobType.GEMM_O_PROJ.value))

        # o reduction
        m_split_o_proj_reduce = min(batch_size, ceildiv(KernelConfig.SM_NUMBER, HIDDEN_SIZE // SplitKReduceTile.N_UNIT))
        n_tile_o_proj_reduce = ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, m_split_o_proj_reduce)) * SplitKReduceTile.N_UNIT
        for n_idx in range(ceildiv(HIDDEN_SIZE, n_tile_o_proj_reduce)):
            for m_idx in range(m_split_o_proj_reduce):
                central_queue.append((m_idx, n_idx, 0, JobType.GEMM_O_REDUCE.value))

        # add rmsnorm
        for m_idx in range(batch_size):
            central_queue.append((m_idx, -1, -1, JobType.ATTN_ADD_RMS_NORM.value))

        # gate_up_proj
        for n in range(INTERMEDIATE_SIZE*2 // GemmTile.BLK_N):
            central_queue.append((0, n, 0, JobType.GEMM_GATE_UP_PROJ.value))

        # split_silu_multiply
        for n in range(INTERMEDIATE_SIZE // SiluMultiplyTile.TILE_SIZE):
            central_queue.append((n, 0, 0, JobType.SPLIT_SILU_MULTIPLY.value))

        # gemm_down_proj
        for n in range(HIDDEN_SIZE // GemmTile.BLK_N):
            for k in range(DOWN_PROJ_SPLIT_K_FACTOR):
                central_queue.append((0, n, k, JobType.GEMM_DOWN_PROJ.value))

        # down_proj_reduce
        M_split_down_proj_reduce = min(
            ceildiv(KernelConfig.SM_NUMBER, HIDDEN_SIZE // SplitKReduceTile.N_UNIT), batch_size
        )
        N_tile = ceildiv(SplitKReduceTile.N_REPEAT, ceildiv(batch_size, M_split_down_proj_reduce)) * SplitKReduceTile.N_UNIT
        for m in range(M_split_down_proj_reduce):
            for n in range(HIDDEN_SIZE // N_tile):
                central_queue.append((m, n, 0, JobType.DOWN_PROJ_REDUCE.value))

        # add_rms_norm
        for m in range(batch_size):
            central_queue.append((m, 0, 0, JobType.MLP_ADD_RMS_NORM.value))

        tile_idx = 0
        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    m, n, k, c = central_queue.pop(0)
                    exec_queue[bx, tile_idx, 0] = m
                    exec_queue[bx, tile_idx, 1] = n
                    exec_queue[bx, tile_idx, 2] = k
                    exec_queue[bx, tile_idx, 3] = c
                else:
                    exec_queue[bx, tile_idx, 0] = -1
                    exec_queue[bx, tile_idx, 1] = -1
                    exec_queue[bx, tile_idx, 2] = -1
                    exec_queue[bx, tile_idx, 3] = JobType.END.value
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx, 0] = -1
            exec_queue[bx, tile_idx, 1] = -1
            exec_queue[bx, tile_idx, 2] = -1
            exec_queue[bx, tile_idx, 3] = JobType.END.value
        return exec_queue

    def generate_exec_queue_dynamic():
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        for n in reversed(range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, GemmTile.BLK_N))):
            for k in range(SPLIT_QKV_PROJECT):
                exec_queue.enqueue(JobType.GEMM_QKV_PROJ.value, 0, n, k)
        return exec_queue

    def prepare_data(batch_size):
        import torch
        torch.manual_seed(42)
        arg_dict = {}

        # input
        arg_dict["hidden_state"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["residual"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)

        # qkv projection
        arg_dict["qkv_proj"] = torch.randn(((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE), dtype=torch.float16)
        torch.nn.init.xavier_normal_(arg_dict["qkv_proj"], gain=1.0)

        # rms
        arg_dict["rms_wight"] = torch.randn((HEAD_DIM), dtype=torch.float16)

        # qkv
        arg_dict["qkv"] = torch.randn((batch_size, NUM_KEY_VALUE_HEADS * 2 + NUM_ATTENTION_HEADS, HEAD_DIM), dtype=torch.float16)

        # rope cos_sin_cache
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float, device="cuda") / HEAD_DIM)
        )
        pos = SEQ_LEN
        assert pos < MAX_POSITION_EMBEDDINGS
        t = torch.full((1,), pos, dtype=torch.float, device="cuda")
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        arg_dict["cos_sin_cache"] = torch.cat((cos, sin), dim=-1).reshape(-1, HEAD_DIM).cpu()

        # paged kv-cache
        page_last_len = PAGE_SIZE if (SEQ_LEN + 1) % PAGE_SIZE == 0 else (SEQ_LEN + 1) % PAGE_SIZE # +1 since need to append new kv
        page_num = ceildiv(SEQ_LEN + 1, PAGE_SIZE)
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
        pos_map = torch.empty(batch_size, dtype=torch.int32).int()
        for i in range(batch_size):
            pos_map[i] = SEQ_LEN
        arg_dict["kv_cache"] = torch.randn((MAX_PAGE_NUM, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM), dtype=torch.float16).cpu()
        arg_dict["kv_indptr"] = kv_indptr.cpu() 
        arg_dict["kv_last_page_len"] = kv_last_page_len.cpu()
        arg_dict["kv_indices"] = kv_indices.cpu()
        arg_dict["pos_map"] = pos_map.cpu()

        # output
        arg_dict["o"] = torch.zeros((batch_size, NUM_ATTENTION_HEADS, HEAD_DIM), dtype=torch.float16)
        arg_dict["lse"] = torch.zeros((batch_size, NUM_ATTENTION_HEADS), dtype=torch.float32)
        arg_dict["o_proj"] = torch.randn((HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM), dtype=torch.float16)
        torch.nn.init.xavier_normal_(arg_dict["o_proj"], gain=1.0)
        arg_dict["hidden_state_out"] = torch.zeros((batch_size, HIDDEN_SIZE), dtype=torch.float16)

        # add rms
        arg_dict["attn_add_rms_weight"] = torch.randn((HIDDEN_SIZE), dtype=torch.float16)

        # mlp
        arg_dict["w_gate_up"] = torch.zeros((INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=torch.float16)
        torch.nn.init.xavier_normal_(arg_dict["w_gate_up"], gain=1.0)
        arg_dict["out_gate_up_proj"] = torch.zeros((batch_size, INTERMEDIATE_SIZE * 2), dtype=torch.float16)
        arg_dict["out_silu_multiply"] = torch.zeros((batch_size, INTERMEDIATE_SIZE), dtype=torch.float16)
        arg_dict["w_down_proj"] = torch.zeros((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.float16) * 0.01
        torch.nn.init.xavier_normal_(arg_dict["w_down_proj"], gain=1.0)
        arg_dict["partial_sum_down_proj"] = torch.zeros((DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE), dtype=torch.float32)
        arg_dict["out_down_proj"] = torch.zeros((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["mlp_add_rms_weight"] = torch.randn((HIDDEN_SIZE, ), dtype=torch.float16)

        # profiler
        arg_dict["profiler_buffer"] = torch.zeros((PROFILER_BUFFER_SIZE, ), dtype=torch.uint64)

        return arg_dict

    def prepare_tvm_arg_dict(arg_dict, REPEAT=100, use_dynamic=False):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        # plan for decoding
        smem_size_decode = (2 * DecodeTile.pipe_depth * DecodeTile.loop_inner * DecodeTile.bdz 
                            * DecodeTile.bdy * DecodeTile.tile_per_bdx * DecodeTile.head_dim * F16_BYTES
                            + KernelConfig.NUM_THREADS * DecodeTile.tile_per_bdx * F32_BYTES
                            + KernelConfig.NUM_THREADS * DecodeTile.loop_inner * DecodeTile.vec_size * F32_BYTES
                        )
        assert smem_size_decode <= KernelConfig.MAX_SMEM_SIZE
        assert DecodeTile.pipe_depth <= DecodeTile.bdx

        # balance the workload (split-kv)
        kv_indptr_h = arg_dict["kv_indptr"].cpu().numpy()
        if batch_size * DecodeTile.qo_heads >= KernelConfig.SM_NUMBER * DecodeTile.max_blk_per_sm:
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
                if new_batch_size * DecodeTile.qo_heads > KernelConfig.SM_NUMBER * DecodeTile.max_blk_per_sm:
                    low = mid + 1
                else:
                    high = mid
            max_page_num = low
            new_batch_size = 0
            for page_num in page_num_list:
                new_batch_size += ceildiv(page_num, max_page_num)
            split_kv = new_batch_size != batch_size
        tvm_arg_dict["split_kv"] = split_kv
        tvm_arg_dict["new_batch_size"] = new_batch_size
        tvm_arg_dict["max_chunk_size"] = tvm.nd.array(np.array([max_page_num * PAGE_SIZE], dtype=np.int32), device=DEV)

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

        tvm_arg_dict["request_indices"] = tvm.nd.array(np.array(request_indices, dtype=np.int32), DEV)
        tvm_arg_dict["kv_tile_indices"] = tvm.nd.array(np.array(kv_tile_indices, dtype=np.int32), DEV)
        tvm_arg_dict["o_indptr"] = tvm.nd.array(np.array(o_indptr, dtype=np.int32), DEV)
        tvm_arg_dict["o_tmp"]= tvm.nd.array(np.zeros([new_batch_size, DecodeTile.qo_heads, DecodeTile.head_dim], dtype=np.float32), DEV)
        tvm_arg_dict["lse_tmp"] = tvm.nd.array(np.zeros([new_batch_size, DecodeTile.qo_heads], dtype=np.float32), DEV)
        tvm_arg_dict["partial_qkv"] = tvm.nd.array(np.zeros([SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM], dtype=np.float32), DEV)
        tvm_arg_dict["partial_o"] = tvm.nd.array(np.zeros([SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], dtype=np.float32), DEV)

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.nd.array(value, device=DEV)

        for i in range(REPEAT):
            # generate event tensor
            qkv_h_d = (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM
            qk_h_d = (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) * HEAD_DIM
            q_h_d = NUM_ATTENTION_HEADS * HEAD_DIM
            k_h_d = NUM_KEY_VALUE_HEADS * HEAD_DIM
            tvm_arg_dict[f"residual_{i}"] = tvm.nd.array(arg_dict["residual"], device=DEV)
            tvm_arg_dict[f"etensor_qkv_partial_{i}"] = tvm.nd.array(
                np.zeros(ceildiv(qkv_h_d, SplitKReduceTile.N_UNIT), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_q_reduce_{i}"] = tvm.nd.array(
                np.zeros((batch_size, ceildiv(q_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_k_reduce_{i}"] = tvm.nd.array(
                np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_v_reduce_{i}"] = tvm.nd.array(
                np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_rms_rope_{i}"] = tvm.nd.array(
                np.zeros((batch_size, NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_k_rope_append_{i}"] = tvm.nd.array(
                np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_decode_{i}"] = tvm.nd.array(
                np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
            )
            etensor_decode_merge = np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32)
            for b in range(batch_size):
                num_merge = o_indptr[b + 1] - o_indptr[b]
                for j in range(NUM_KEY_VALUE_HEADS):
                    etensor_decode_merge[b, j] = num_merge
            tvm_arg_dict[f"etensor_decode_merge_{i}"] = tvm.nd.array(etensor_decode_merge, device=DEV)
            etensor_o_proj = np.zeros(SPLIT_O_PROJRCT, dtype=np.int32)
            o_proj_tile_k = ceildiv(ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SPLIT_O_PROJRCT), GemmTile.BLK_K) * GemmTile.BLK_K
            for h in range(NUM_KEY_VALUE_HEADS):
                range_start = h * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile_k
                range_end = (h + 1) * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile_k
                if range_end != range_start:
                    etensor_o_proj[range_start] += 1
                    etensor_o_proj[range_end] += 1
                else:
                    etensor_o_proj[range_start] += 1
            etensor_o_proj *= batch_size
            tvm_arg_dict[f"etensor_o_proj_{i}"] = tvm.nd.array(
                etensor_o_proj, device=DEV
            )
            tvm_arg_dict[f"etensor_o_partial_{i}"] = tvm.nd.array(
                np.zeros(ceildiv(HIDDEN_SIZE, GemmTile.BLK_N), dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_attn_add_rms_norm_{i}"] = tvm.nd.array(
                np.zeros(batch_size, dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_attn_mlp_{i}"] = tvm.nd.array(
                np.zeros(1, dtype=np.int32), device=DEV    
            )
            tvm_arg_dict[f"etensor_gate_up_proj_{i}"] = tvm.nd.array(
                np.zeros(INTERMEDIATE_SIZE // GemmTile.BLK_N, dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_down_proj_{i}"] = tvm.nd.array(
                np.zeros(DOWN_PROJ_SPLIT_K_FACTOR, dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_down_proj_reduce_{i}"] = tvm.nd.array(
                np.zeros(HIDDEN_SIZE // GemmTile.BLK_N, dtype=np.int32), device=DEV
            )
            tvm_arg_dict[f"etensor_mlp_add_rms_norm_{i}"] = tvm.nd.array(
                np.zeros((batch_size, ), dtype=np.int32), device=DEV
            )
            if use_dynamic:
                exec_queue = generate_exec_queue_dynamic()
                tvm_arg_dict[f"etensor_end_{i}"] = tvm.nd.array(
                    np.zeros(1, dtype=np.int32), device=DEV
                )
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.nd.array(exec_queue.tasks, DEV)   
                tvm_arg_dict[f"queue_head_{i}"] = tvm.nd.array(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.nd.array(exec_queue.tail, DEV)
            else:
                exec_queue = generate_exec_queue(batch_size, tvm_arg_dict["new_batch_size"], tvm_arg_dict["split_kv"])
                tvm_arg_dict["exec_queue"] = tvm.nd.array(exec_queue, DEV)
        return tvm_arg_dict

    arg_dict = prepare_data(batch_size)

    def tir_static(arg_dict):
        tvm_arg_dict = prepare_tvm_arg_dict(arg_dict, REPEAT=100, use_dynamic=False)
        # static schedule
        iter = 0            
        def func():
            if PROFILER_ON:
                tvm_arg_dict["profiler_buffer"] = tvm.nd.array(arg_dict["profiler_buffer"], device=tvm.cuda(0))
            nonlocal iter
            mega_kernel_static(
                tvm_arg_dict["hidden_state"], 
                tvm_arg_dict["qkv_proj"], 
                tvm_arg_dict["partial_qkv"],
                tvm_arg_dict["qkv"], 
                tvm_arg_dict["rms_wight"], 
                tvm_arg_dict["cos_sin_cache"], 
                tvm_arg_dict["kv_cache"],
                tvm_arg_dict["kv_indptr"], 
                tvm_arg_dict["kv_indices"], 
                tvm_arg_dict["kv_last_page_len"], 
                tvm_arg_dict["pos_map"],
                tvm_arg_dict["o"], 
                tvm_arg_dict["lse"], 
                tvm_arg_dict["o_tmp"], 
                tvm_arg_dict["lse_tmp"], 
                tvm_arg_dict["o_indptr"],
                tvm_arg_dict["request_indices"], 
                tvm_arg_dict["kv_tile_indices"], 
                tvm_arg_dict["max_chunk_size"], 
                tvm_arg_dict["partial_o"], 
                tvm_arg_dict["o_proj"], 
                tvm_arg_dict["hidden_state_out"],
                tvm_arg_dict["attn_add_rms_weight"], 
                tvm_arg_dict[f"residual_{iter}"],
                tvm_arg_dict["w_gate_up"],
                tvm_arg_dict["out_gate_up_proj"],
                tvm_arg_dict["out_silu_multiply"],
                tvm_arg_dict["w_down_proj"],
                tvm_arg_dict["partial_sum_down_proj"],
                tvm_arg_dict["out_down_proj"],
                tvm_arg_dict["mlp_add_rms_weight"],
                tvm_arg_dict[f"etensor_qkv_partial_{iter}"], 
                tvm_arg_dict[f"etensor_q_reduce_{iter}"], 
                tvm_arg_dict[f"etensor_k_reduce_{iter}"], 
                tvm_arg_dict[f"etensor_v_reduce_{iter}"], 
                tvm_arg_dict[f"etensor_rms_rope_{iter}"], 
                tvm_arg_dict[f"etensor_k_rope_append_{iter}"], 
                tvm_arg_dict[f"etensor_decode_{iter}"], 
                tvm_arg_dict[f"etensor_decode_merge_{iter}"],
                tvm_arg_dict[f"etensor_o_proj_{iter}"], 
                tvm_arg_dict[f"etensor_o_partial_{iter}"], 
                tvm_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                tvm_arg_dict[f"etensor_attn_mlp_{iter}"],
                tvm_arg_dict[f"etensor_gate_up_proj_{iter}"],
                tvm_arg_dict[f"etensor_down_proj_{iter}"],
                tvm_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                tvm_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                tvm_arg_dict["exec_queue"], 
                tvm_arg_dict["profiler_buffer"]
            )
            iter+=1

        ms = bench(func, warmup=1, repeat=10, proton_name="tir-static")
        print(f"TIR time: {ms:.3f} ms")
        if PROFILER_ON:
            export_to_perfetto_trace(
                tvm_arg_dict["profiler_buffer"].numpy(),
                f"megakernel-layer-static.perfetto-trace",
                event_type_names
            )
        return tvm_arg_dict["out_down_proj"].numpy(), tvm_arg_dict["residual_0"].numpy()

    def tir_dynamic(arg_dict):
        target = tvm.target.Target("cuda")
        tvm_arg_dict = prepare_tvm_arg_dict(arg_dict, REPEAT=100, use_dynamic=True)
        iter = 0
        with target:
            def func():
                if PROFILER_ON:
                    tvm_arg_dict["profiler_buffer"] = tvm.nd.array(arg_dict["profiler_buffer"], device=tvm.cuda(0))
                nonlocal iter
                mega_kernel_dynamic(
                    tvm_arg_dict["hidden_state"], 
                    tvm_arg_dict["qkv_proj"], 
                    tvm_arg_dict["partial_qkv"],
                    tvm_arg_dict["qkv"], 
                    tvm_arg_dict["rms_wight"], 
                    tvm_arg_dict["cos_sin_cache"], 
                    tvm_arg_dict["kv_cache"],
                    tvm_arg_dict["kv_indptr"], 
                    tvm_arg_dict["kv_indices"], 
                    tvm_arg_dict["kv_last_page_len"], 
                    tvm_arg_dict["pos_map"],
                    tvm_arg_dict["o"], 
                    tvm_arg_dict["lse"], 
                    tvm_arg_dict["o_tmp"], 
                    tvm_arg_dict["lse_tmp"], 
                    tvm_arg_dict["o_indptr"],
                    tvm_arg_dict["request_indices"], 
                    tvm_arg_dict["kv_tile_indices"], 
                    tvm_arg_dict["max_chunk_size"], 
                    tvm_arg_dict["partial_o"], 
                    tvm_arg_dict["o_proj"], 
                    tvm_arg_dict["hidden_state_out"],
                    tvm_arg_dict["attn_add_rms_weight"], 
                    tvm_arg_dict[f"residual_{iter}"],
                    tvm_arg_dict["w_gate_up"],
                    tvm_arg_dict["out_gate_up_proj"],
                    tvm_arg_dict["out_silu_multiply"],
                    tvm_arg_dict["w_down_proj"],
                    tvm_arg_dict["partial_sum_down_proj"],
                    tvm_arg_dict["out_down_proj"],
                    tvm_arg_dict["mlp_add_rms_weight"],
                    tvm_arg_dict[f"etensor_qkv_partial_{iter}"], 
                    tvm_arg_dict[f"etensor_q_reduce_{iter}"], 
                    tvm_arg_dict[f"etensor_k_reduce_{iter}"], 
                    tvm_arg_dict[f"etensor_v_reduce_{iter}"], 
                    tvm_arg_dict[f"etensor_rms_rope_{iter}"], 
                    tvm_arg_dict[f"etensor_k_rope_append_{iter}"], 
                    tvm_arg_dict[f"etensor_decode_{iter}"], 
                    tvm_arg_dict[f"etensor_decode_merge_{iter}"],
                    tvm_arg_dict[f"etensor_o_proj_{iter}"], 
                    tvm_arg_dict[f"etensor_o_partial_{iter}"], 
                    tvm_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                    tvm_arg_dict[f"etensor_attn_mlp_{iter}"],
                    tvm_arg_dict[f"etensor_gate_up_proj_{iter}"],
                    tvm_arg_dict[f"etensor_down_proj_{iter}"],
                    tvm_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                    tvm_arg_dict[f"etensor_end_{iter}"],
                    tvm_arg_dict[f"queue_tasks_{iter}"],
                    tvm_arg_dict[f"queue_head_{iter}"],
                    tvm_arg_dict[f"queue_tail_{iter}"],
                    tvm_arg_dict["profiler_buffer"]
                )
                iter+=1
        bench(func, warmup=1, repeat=10, proton_name="tir-dynamic")
        if PROFILER_ON:
            export_to_perfetto_trace(
                tvm_arg_dict["profiler_buffer"].numpy(),
                f"megakernel-layer-dynamic.perfetto-trace",
                event_type_names
            )
        return tvm_arg_dict["out_down_proj"].numpy(), tvm_arg_dict["residual_0"].numpy()

    def std(arg_dict):
        import torch
        import flashinfer
        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def get_silu_multiply_std_impl():
            BDX = 32
            BDY = KernelConfig.NUM_THREADS // BDX
            VEC_SIZE = 8
            SEQ_LEN = batch_size

            @T.prim_func(tirp=True)
            def fused_split_silu_multiply(input_cat_ptr: T.handle, output_ptr: T.handle):
                input_cat_global = T.match_buffer(
                    input_cat_ptr,
                    [batch_size, INTERMEDIATE_SIZE * 2],
                    "float16",
                    scope="global",
                    layout="default",
                )
                output_global = T.match_buffer(
                    output_ptr, [batch_size, INTERMEDIATE_SIZE], "float16", scope="global", layout="default"
                )

                with T.kernel():
                    bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                    tx, ty = T.thread_id([BDX, BDY], parent="cta")
                    thread_id = T.meta_var(ty * BDX + tx)

                    with T.thread():
                        idx = T.alloc_local([1], "int32", layout="default")
                        vec1 = T.alloc_local([VEC_SIZE], "float16", layout="default")
                        vec2 = T.alloc_local([VEC_SIZE], "float16", layout="default")

                        idx[0] = bx
                        while idx[0] < ceildiv(SEQ_LEN * INTERMEDIATE_SIZE, KernelConfig.NUM_THREADS * VEC_SIZE):
                            real_idx = T.meta_var((idx[0] * KernelConfig.NUM_THREADS + thread_id) * VEC_SIZE)
                            token_idx = T.meta_var(real_idx // INTERMEDIATE_SIZE)
                            offset_imme = T.meta_var((real_idx % INTERMEDIATE_SIZE) % INTERMEDIATE_SIZE)
                            for kv in T.serial(VEC_SIZE):
                                vec1[kv] = input_cat_global[token_idx, offset_imme + kv]
                            for kv in T.serial(VEC_SIZE):
                                vec2[kv] = input_cat_global[token_idx, INTERMEDIATE_SIZE + offset_imme + kv]
                            for kv in T.serial(VEC_SIZE):
                                vec1[kv] = vec1[kv] * T.sigmoid(vec1[kv]) * vec2[kv]
                            for kv in T.serial(VEC_SIZE):
                                output_global[token_idx, offset_imme + kv] = vec1[kv]
                            idx[0] += KernelConfig.SM_NUMBER
            return fused_split_silu_multiply
        _, mod_fused_split_silu_multiply = get_source(get_silu_multiply_std_impl())

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)

            workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
            wrapper.plan(
                indptr=std_arg_dict["kv_indptr"],
                indices=std_arg_dict["kv_indices"],
                last_page_len=std_arg_dict["kv_last_page_len"],
                num_qo_heads=NUM_ATTENTION_HEADS,
                num_kv_heads=NUM_KEY_VALUE_HEADS,
                head_dim=HEAD_DIM,
                page_size=PAGE_SIZE,
                pos_encoding_mode="NONE",
                data_type=torch.float16,
                q_data_type=torch.float16,
            )

            qkv = torch.matmul(std_arg_dict["hidden_state"], std_arg_dict["qkv_proj"].T).reshape(batch_size, -1, HEAD_DIM)
            q, k, v = torch.split(qkv, [NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_HEADS], dim=1)
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, HEAD_DIM), 
                weight=std_arg_dict["rms_wight"],
                eps=RMS_NORM_EPS, enable_pdl=False
            ).reshape(batch_size, NUM_ATTENTION_HEADS, HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, HEAD_DIM), 
                weight=std_arg_dict["rms_wight"],
                eps=RMS_NORM_EPS, enable_pdl=False
            ).reshape(batch_size, NUM_KEY_VALUE_HEADS, HEAD_DIM)
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=torch.full((1,), 0, dtype=torch.int32, device=torch_dev).repeat(batch_size),
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, NUM_KEY_VALUE_HEADS, HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["pos_map"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["kv_indices"],
                kv_indptr=std_arg_dict["kv_indptr"],
                kv_last_page_len=std_arg_dict["kv_last_page_len"],
                kv_layout="HND",
            )
            o, lse = wrapper.run_return_lse(q.reshape(batch_size, NUM_ATTENTION_HEADS, HEAD_DIM), std_arg_dict["kv_cache"])
            hidden_state_out = torch.matmul(o.reshape(batch_size, NUM_ATTENTION_HEADS * HEAD_DIM), std_arg_dict["o_proj"].T)
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_out,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=RMS_NORM_EPS,
                enable_pdl=False
            )

            out_gate_up_proj = torch.matmul(hidden_state_out, std_arg_dict["w_gate_up"].T)
            out_gate_up_proj_tvm = tvm.nd.array(out_gate_up_proj.cpu(), device=tvm.cuda(0))
            out_silu_multiply_tvm = tvm.nd.array(torch.zeros((batch_size, INTERMEDIATE_SIZE), dtype=torch.float16), device=tvm.cuda(0))
            mod_fused_split_silu_multiply(out_gate_up_proj_tvm, out_silu_multiply_tvm)
            out_silu_multiply = torch.from_numpy(out_silu_multiply_tvm.numpy()).to(torch_dev)
            out_down_proj = torch.matmul(out_silu_multiply, std_arg_dict["w_down_proj"].T)
            flashinfer.norm.fused_add_rmsnorm(
                input=out_down_proj,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["mlp_add_rms_weight"],
                eps=AddRMSNormTile.EPS,
                enable_pdl=False
            )
            return out_down_proj.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()
        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"std time: {ms:.3f} ms")
        return output

    with ProtonContext("blackwell_attn"):
        output_tir_static, residual_tir_static = tir_static(arg_dict)
        output_tir_dynamic, residual_tir_dynamic = tir_dynamic(arg_dict)
        output_std, residual_std = std(arg_dict)

    np.testing.assert_allclose(output_tir_static, output_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_tir_static, residual_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(output_tir_dynamic, output_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_tir_dynamic, residual_std, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    def init_tile_class_config():
        AppendKVTile.class_config_init(problem_config)
        RMSnormTile.class_config_init(problem_config)
        RopeTile.class_config_init(problem_config)
        DecodeTile.class_config_init(problem_config)
        DecodeMergeTile.class_config_init(problem_config)

    init_tile_class_config()
    mega_kernel_static = MegaKernel().get_func_static()
    src, mod_static = get_source(mega_kernel_static)
    mega_kernel_dynamic = MegaKernel().get_func_dynamic()
    src, mod_dynamic = get_source(mega_kernel_dynamic)

    for batch_size in [128, 64, 32, 16, 8, 4, 2, 1]:

        print(f"batch_size: {batch_size}", flush=True)
        test(batch_size, mod_static, mod_dynamic)
