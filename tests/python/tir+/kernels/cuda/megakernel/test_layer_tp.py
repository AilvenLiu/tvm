import math
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile
from tvm.tirp.megakernel.allreduce import AllreduceTile
from tvm.tirp.megakernel.append_kv import AppendKVTile
from tvm.tirp.megakernel.batch_decode import DecodeTile
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.decode_merge import DecodeMergeTile
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.rms_norm import RMSnormTile
from tvm.tirp.megakernel.rope import RopeTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile
from tvm.tirp.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirp.megakernel.support import generate_event_tensor

# model configs
WORLD_SIZE = 8
VOCAB_SIZE = 151936
MAX_POSITION_EMBEDDINGS = 40960
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600 // WORLD_SIZE
NUM_HIDDEN_LAYERS = 64
NUM_ATTENTION_HEADS = 64 // WORLD_SIZE
NUM_KEY_VALUE_HEADS = 8 // WORLD_SIZE
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
    "seq_len": SEQ_LEN,
}

SPLIT_QKV_PROJECT = 4
SPLIT_O_PROJRCT = 2
GATE_UP_PROJ_SPLIT_K_FACTOR = 2
DOWN_PROJ_SPLIT_K_FACTOR = 3

NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
PROFILER_BUFFER_SIZE = int(1e6)
PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS
PROFILER_ON = False


class MegaKernel:
    class_list = [
        GemmTile,
        SplitKReduceTile,
        RMSnormTile,
        RopeTile,
        AppendKVTile,
        DecodeTile,
        DecodeMergeTile,
        AddRMSNormTile,
        SiluMultiplyTile,
        AllreduceTile,
    ]

    def __init__(self, problem_config, profiler_on):
        self.tile_list = []
        self.class_config_init(problem_config)
        self.profiler_on = profiler_on

    def class_config_init(self, config):
        for cls in self.class_list:
            cls.class_config_init(config)

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
        from tvm.tirp.megakernel.static_scheduler import Semaphore

        # fmt: off
        @T.prim_func(tirp=True)
        def mega_kernel(
                # input and output
                hidden_state_ptr: T.handle, # input: read-only
                residual_ptr: T.handle, # input & output: inplace update
                output_ptr: T.handle, # output

                # weight
                qkv_proj_weight_ptr: T.handle, # read-only
                o_proj_weight_ptr: T.handle, # read-only
                q_rms_weight_ptr: T.handle, # read-only
                k_rms_weight_ptr: T.handle, # read-only
                gate_up_weight_ptr: T.handle, # read-only
                down_weight_ptr: T.handle, # read-only
                attn_add_rms_weight_ptr: T.handle, # read-only
                mlp_add_rms_weight_ptr: T.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: T.handle, # read-only
                rope_pos_ptr: T.handle, # read-only
                kv_cache_ptr: T.handle, # inplace update
                kv_indptr_ptr: T.handle, # read-only
                kv_indices_ptr: T.handle, # read-only
                kv_last_page_len_ptr: T.handle, # read-only
                append_pos_ptr: T.handle, # read-only
                request_indices_ptr: T.handle, # read-only
                kv_tile_indices_ptr: T.handle, # read-only
                max_chunk_size_ptr: T.handle, # read-only
                o_indptr_ptr: T.handle, # read-only

                # intermediate buffer
                partital_qkv_ptr: T.handle, # intermediate
                qkv_ptr: T.handle,  # intermediate
                o_ptr: T.handle, # intermediate
                lse_ptr: T.handle, # intermediate
                o_tmp_ptr: T.handle, # intermediate
                lse_tmp_ptr: T.handle, # intermediate
                partial_o_ptr: T.handle, # intermediate
                before_o_allreduce_ptr: T.handle, # intermediate
                hidden_state_attn_mlp_ptr: T.handle, # intermediate
                partial_out_gate_up_proj_ptr: T.handle, # intermediate
                out_gate_up_proj_ptr: T.handle, # intermediate
                out_silu_multiply_ptr: T.handle, # intermediate
                partial_sum_down_proj_ptr: T.handle, # intermediate
                before_down_proj_allreduce_ptr: T.handle, # intermediate
                
                # event tensor
                etensor_qkv_partial_ptr: T.handle, 
                etensor_q_reduce_ptr: T.handle, 
                etensor_k_reduce_ptr: T.handle, 
                etensor_v_reduce_ptr: T.handle, 
                etensor_decode_ptr: T.handle, 
                etensor_decode_merge_ptr: T.handle,
                etensor_o_proj_ptr: T.handle, 
                etensor_o_partial_ptr: T.handle, 
                etensor_o_allreduce_ptr: T.handle,
                etensor_attn_add_rms_ptr: T.handle,
                etensor_attn_mlp_ptr: T.handle,
                etensor_gate_up_proj_reduce_ptr: T.handle,
                etensor_gate_up_proj_ptr: T.handle,
                etensor_down_proj_ptr: T.handle,
                etensor_down_proj_reduce_ptr: T.handle,
                etensor_down_proj_allreduce_ptr: T.handle,
                etensor_mlp_add_rms_ptr: T.handle,

                # execution queue
                exec_queue_ptr: T.handle,
                profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):

            # match buffer
            batch_size = T.int32()
            new_batch_size = T.int32()
            cos_sin_cache_len = T.int32()
            max_page_num = T.int32()
            total_page_num = T.int32()

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global", layout="default")
            residual_global = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global", layout="default")
            output_global = T.match_buffer(output_ptr, [batch_size, HIDDEN_SIZE], "float16", layout="default")

            # weight
            qkv_proj_weight_global = T.match_buffer(qkv_proj_weight_ptr, [(NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE], 
                                                    "float16", scope="global", layout="default")
            o_proj_weight_global = T.match_buffer(o_proj_weight_ptr, [HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM], 
                                                  "float16", scope="global", layout="default")
            q_rms_weight_global = T.match_buffer(q_rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default")
            k_rms_weight_global = T.match_buffer(k_rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default") 
            gate_up_weight_global = T.match_buffer(gate_up_weight_ptr, [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE], 
                                                   "float16", scope="global", layout="default")
            down_weight_global = T.match_buffer(down_weight_ptr, [HIDDEN_SIZE, INTERMEDIATE_SIZE], "float16", scope="global", layout="default")
            attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")
            mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")

            # page cache, kv cache and plan info
            cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, HEAD_DIM], "float32", scope="global", layout="default")
            rope_pos_global = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            kv_cache_global = T.match_buffer(kv_cache_ptr, [max_page_num, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM], 
                                             "float16", scope="global", layout="default")
            kv_indptr_global = T.match_buffer(kv_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default", offset_factor=1)
            kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", scope="global", layout="default", offset_factor=1)
            kv_last_page_len_global = T.match_buffer(kv_last_page_len_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            append_pos_global = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            request_indices_global = T.match_buffer(request_indices_ptr, [new_batch_size], "int32", scope="global", layout="default", offset_factor=1)
            kv_tile_indices_global = T.match_buffer(kv_tile_indices_ptr, [new_batch_size], "int32", scope="global", layout="default", offset_factor=1)
            max_chunk_size_global = T.match_buffer(max_chunk_size_ptr, [1], "int32", scope="global", layout="default", offset_factor=1)
            o_indptr_global = T.match_buffer(o_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default", offset_factor=1)

            # intermediate buffer
            partital_qkv_global = T.match_buffer(partital_qkv_ptr, [SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM], 
                                    "float32", scope="global", layout="default")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS, HEAD_DIM], 
                                    "float16", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                    "float16", scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_ATTENTION_HEADS], 
                                    "float32", scope="global", layout="default")
            o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                    "float32", scope="global", layout="default")
            lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS], 
                                    "float32", scope="global", layout="default")
            partial_o_global = T.match_buffer(partial_o_ptr, [SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], 
                                    "float32", scope="global", layout="default")
            before_o_allreduce_global = T.match_buffer(before_o_allreduce_ptr, [batch_size, HIDDEN_SIZE], 
                                    "float16", scope="global", layout="default")
            hidden_state_attn_mlp_global = T.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, HIDDEN_SIZE], 
                                    "float16", scope="global", layout="default")
            partial_out_gate_up_proj_global = T.match_buffer(partial_out_gate_up_proj_ptr, [GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, INTERMEDIATE_SIZE * 2], 
                                    "float32", scope="global", layout="default")
            out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE * 2], 
                                 "float16", scope="global", layout="default")
            out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], 
                                  "float16", scope="global", layout="default")
            partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], 
                                  "float32", layout="default")
            before_down_proj_allreduce_global = T.match_buffer(before_down_proj_allreduce_ptr, [batch_size, HIDDEN_SIZE], 
                                  "float16", scope="global", layout="default")

            # event tensor
            etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_decode_global = T.match_buffer(etensor_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_allreduce_global = T.match_buffer(etensor_o_allreduce_ptr, [HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE], "int32", scope="global", layout="default", offset_factor=1)
            etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", layout="default", offset_factor=1)
            etensor_gate_up_proj_reduce_global = T.match_buffer(etensor_gate_up_proj_reduce_ptr, [INTERMEDIATE_SIZE * 2 // GemmTile.BLK_N], 
                                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [INTERMEDIATE_SIZE // GemmTile.BLK_N], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR],
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [HIDDEN_SIZE // GemmTile.BLK_N],
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_allreduce_global = T.match_buffer(etensor_down_proj_allreduce_ptr, [HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE], "int32", scope="global", layout="default", offset_factor=1)
            etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)

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
            qkv_proj_tile = T.meta_var(GemmTile(hidden_state_global, qkv_proj_weight_global, partital_qkv_global, split_k_factor=SPLIT_QKV_PROJECT))
            qkv_reduce_tile = T.meta_var(SplitKReduceTile(partital_qkv_global, 
                                                          Tp.reshape(qkv_global, [-1, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM]).buffer))
            rmsnorm_tile = T.meta_var(RMSnormTile(q_rms_weight_global, k_rms_weight_global, qkv_global))
            rope_tile = T.meta_var(RopeTile(qkv_global, cos_sin_cache_global, rope_pos_global))
            append_kv_tile = T.meta_var(AppendKVTile(kv_cache_global, qkv_global, append_pos_global))
            decode_tile = T.meta_var(DecodeTile(qkv_global, kv_cache_global, o_global, lse_global, o_tmp_global, lse_tmp_global,
                                                kv_indptr_global, kv_last_page_len_global,
                                                kv_indices_global, request_indices_global, kv_tile_indices_global, max_chunk_size_global))
            decode_merge_tile = T.meta_var(DecodeMergeTile(o_indptr_global, o_tmp_global, o_global, lse_tmp_global, lse_global))
            o_proj_tile = T.meta_var(GemmTile(Tp.reshape(o_global, [-1, NUM_ATTENTION_HEADS * HEAD_DIM]).buffer, o_proj_weight_global, 
                                              partial_o_global, split_k_factor=SPLIT_O_PROJRCT))
            o_reduce_tile = T.meta_var(SplitKReduceTile(partial_o_global, before_o_allreduce_global))
            o_allreduce_tile = T.meta_var(AllreduceTile(before_o_allreduce_global, hidden_state_attn_mlp_global, WORLD_SIZE))
            attn_add_rms_tile = T.meta_var(AddRMSNormTile(hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global))

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(hidden_state_attn_mlp_global, gate_up_weight_global, partial_out_gate_up_proj_global, split_k_factor=GATE_UP_PROJ_SPLIT_K_FACTOR))
            gemm_gate_up_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_out_gate_up_proj_global, out_gate_up_proj_global))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj_global, out_silu_multiply_global))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply_global, down_weight_global, partial_sum_down_proj_global, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj_global, before_down_proj_allreduce_global))
            down_proj_allreduce_tile = T.meta_var(AllreduceTile(before_down_proj_allreduce_global, output_global, WORLD_SIZE))
            mlp_add_rms_norm_tile = T.meta_var(AddRMSNormTile(output_global, residual_global, mlp_add_rms_weight_global))

            self.tile_list.append(qkv_proj_tile)
            self.tile_list.append(qkv_reduce_tile)
            self.tile_list.append(rmsnorm_tile)
            self.tile_list.append(rope_tile)
            self.tile_list.append(append_kv_tile)
            self.tile_list.append(decode_tile)
            self.tile_list.append(decode_merge_tile)
            self.tile_list.append(o_proj_tile)
            self.tile_list.append(o_reduce_tile)
            self.tile_list.append(o_allreduce_tile)
            self.tile_list.append(attn_add_rms_tile)
            self.tile_list.append(gemm_gate_up_proj_tile)
            self.tile_list.append(gemm_gate_up_proj_reduce_tile)
            self.tile_list.append(silu_multiply_tile)
            self.tile_list.append(gemm_down_proj_tile)
            self.tile_list.append(down_proj_reduce_tile)
            self.tile_list.append(down_proj_allreduce_tile)
            self.tile_list.append(mlp_add_rms_norm_tile)

            qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj)
            o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj)
            gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
            gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)

            self.host_init_all()

            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                if self.profiler_on:
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
                                                             etensor_qkv_partial_global, use_nvshmem=True))
                    evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global, use_nvshmem=True))
                    evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global, use_nvshmem=True))
                    evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global, use_nvshmem=True))
                    evt_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS+2, etensor_decode_global, use_nvshmem=True))
                    evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global, use_nvshmem=True))
                    evt_o_proj = T.meta_var(Semaphore(-1, etensor_o_proj_global, use_nvshmem=True))
                    evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global, use_nvshmem=True))
                    evt_o_allreduce = T.meta_var(Semaphore(WORLD_SIZE * o_reduce_tile.M_split, etensor_o_allreduce_global, use_nvshmem=True))
                    evt_attn_add_rms = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, o_allreduce_tile.N_TILE), etensor_attn_add_rms_global, use_nvshmem=True))
                    evt_attn_mlp = T.meta_var(Semaphore(batch_size, etensor_attn_mlp_global, use_nvshmem=True))
                    evt_gate_up_proj_reduce = T.meta_var(Semaphore(GATE_UP_PROJ_SPLIT_K_FACTOR, etensor_gate_up_proj_reduce_global, use_nvshmem=True))
                    evt_gate_up_proj = T.meta_var(Semaphore(2 * gemm_gate_up_proj_reduce_tile.M_split, etensor_gate_up_proj_global, use_nvshmem=True))
                    evt_down_proj = T.meta_var(Semaphore(-1, etensor_down_proj_global, use_nvshmem=True))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                                                etensor_down_proj_reduce_global, use_nvshmem=True))
                    evt_down_proj_allreduce = T.meta_var(Semaphore(WORLD_SIZE * down_proj_reduce_tile.M_split, etensor_down_proj_allreduce_global, use_nvshmem=True))
                    evt_add_rms_norm = T.meta_var(Semaphore(HIDDEN_SIZE// down_proj_allreduce_tile.N_TILE, etensor_mlp_add_rms_global, use_nvshmem=True))

                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(StaticTileScheduler(prefix="attn", exec_queue=exec_queue, pool_allocator=pool))
                    tile_scheduler.init(bx, tid)
                    while tile_scheduler.valid():
                        if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_qkv_partial.semaphore_notify(tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
                            evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            qkv_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile:
                                    evt_q_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                                elif tile_scheduler.n_idx < (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile:
                                    evt_k_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile)
                                else:
                                    evt_v_reduce.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.Q_RMSNORM_ROPE.value:
                            evt_q_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS))
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.K_RMSNORM_ROPE_APPEND_KV.value:
                            evt_k_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rmsnorm_tile.h_tile, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rope_tile.h_tile, tile_scheduler.k_idx)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.V_APPEND_KV.value:
                            evt_v_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 1)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
                            evt_decode.semaphore_wait(tile_scheduler.m_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                                range_end = T.meta_var(((tile_scheduler.n_idx+1) * (NUM_ATTENTION_HEADS //NUM_KEY_VALUE_HEADS) * HEAD_DIM - 1) // o_proj_tile.TILE_K)
                                for i in T.serial(0, range_end + 1 - range_start):
                                    evt_o_proj.semaphore_notify(i + range_start)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
                            batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
                            evt_decode.semaphore_wait(batch_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode_merge.semaphore_notify(batch_idx, tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
                            evt_decode_merge.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx * DecodeMergeTile.bdz // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS))
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                range_start = T.meta_var(tile_scheduler.n_idx * DecodeMergeTile.bdz * HEAD_DIM // o_proj_tile.TILE_K)
                                range_end = T.meta_var(((tile_scheduler.n_idx + 1) * DecodeMergeTile.bdz * HEAD_DIM - 1) // o_proj_tile.TILE_K)
                                for kr in T.serial(range_end - range_start + 1):
                                    evt_o_proj.semaphore_notify(range_start + kr)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                            evt_o_proj.semaphore_wait(tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_o_partial.semaphore_notify(tile_scheduler.n_idx // (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                            evt_o_partial.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_o_allreduce.semaphore_notify(tile_scheduler.n_idx // WORLD_SIZE, rank=tile_scheduler.n_idx % WORLD_SIZE)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.O_ALLREDUCE.value:
                            evt_o_allreduce.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.O_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            o_allreduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid < WORLD_SIZE:    
                                evt_attn_add_rms.semaphore_notify(tile_scheduler.m_idx, rank=tid)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.O_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                            evt_attn_add_rms.semaphore_wait(tile_scheduler.m_idx // o_allreduce_tile.M_TILE)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            attn_add_rms_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_attn_mlp.semaphore_notify(0)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            evt_attn_mlp.semaphore_wait(0)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_gate_up_proj_reduce.semaphore_notify(tile_scheduler.n_idx)
                        elif tile_scheduler.task_type == JobType.GATE_UP_PROJ_REDUCE.value:
                            evt_gate_up_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GATE_UP_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            gemm_gate_up_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                    evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx - INTERMEDIATE_SIZE // GemmTile.BLK_N)
                                else:
                                    evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GATE_UP_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            evt_gate_up_proj.semaphore_wait(tile_scheduler.m_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                range_start = T.meta_var(tile_scheduler.m_idx * SiluMultiplyTile.TILE_SIZE // gemm_down_proj_tile.TILE_K)
                                range_end = T.meta_var(((tile_scheduler.m_idx + 1) * SiluMultiplyTile.TILE_SIZE - 1) // gemm_down_proj_tile.TILE_K)
                                for i in T.serial(0, range_end + 1 - range_start):
                                    evt_down_proj.semaphore_notify(i + range_start)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                            evt_down_proj.semaphore_wait(tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_down_proj_reduce.semaphore_notify(tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N))
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            evt_down_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_down_proj_allreduce.semaphore_notify(tile_scheduler.n_idx // WORLD_SIZE, rank=tile_scheduler.n_idx % WORLD_SIZE)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_ALLREDUCE.value:
                            evt_down_proj_allreduce.semaphore_wait(tile_scheduler.n_idx)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            down_proj_allreduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid < WORLD_SIZE:
                                evt_add_rms_norm.semaphore_notify(tile_scheduler.m_idx, rank=tid)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                            evt_add_rms_norm.semaphore_wait(tile_scheduler.m_idx // down_proj_allreduce_tile.M_TILE)
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            mlp_add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)

                        tile_scheduler.next_tile()
                    self.class_finalize_all()
        # fmt: on

        return mega_kernel

    def get_func_dynamic(self):
        from tvm.tirp.megakernel.dynamic_scheduler import Semaphore

        def get_rank_map(idx):
            return idx

        # fmt: off
        @T.prim_func(tirp=True)
        def mega_kernel(
            # input and output
            hidden_state_ptr: T.handle, # input: read-only
            residual_ptr: T.handle, # input & output: inplace update
            output_ptr: T.handle, # output

            # weight
            qkv_proj_weight_ptr: T.handle, # read-only
            o_proj_weight_ptr: T.handle, # read-only
            q_rms_weight_ptr: T.handle, # read-only
            k_rms_weight_ptr: T.handle, # read-only
            gate_up_weight_ptr: T.handle, # read-only
            down_weight_ptr: T.handle, # read-only
            attn_add_rms_weight_ptr: T.handle, # read-only
            mlp_add_rms_weight_ptr: T.handle, # read-only

            # page cache, cos_sin cache and plan info
            cos_sin_cache_ptr: T.handle, # read-only
            rope_pos_ptr: T.handle, # read-only
            kv_cache_ptr: T.handle, # inplace update
            kv_indptr_ptr: T.handle, # read-only
            kv_indices_ptr: T.handle, # read-only
            kv_last_page_len_ptr: T.handle, # read-only
            append_pos_ptr: T.handle, # read-only
            request_indices_ptr: T.handle, # read-only
            kv_tile_indices_ptr: T.handle, # read-only
            max_chunk_size_ptr: T.handle, # read-only
            o_indptr_ptr: T.handle, # read-only

            # intermediate buffer
            partital_qkv_ptr: T.handle, # intermediate
            qkv_ptr: T.handle,  # intermediate
            o_ptr: T.handle, # intermediate
            lse_ptr: T.handle, # intermediate
            o_tmp_ptr: T.handle, # intermediate
            lse_tmp_ptr: T.handle, # intermediate
            partial_o_ptr: T.handle, # intermediate
            before_o_allreduce_ptr: T.handle, # intermediate
            hidden_state_attn_mlp_ptr: T.handle, # intermediate
            partial_out_gate_up_proj_ptr: T.handle, # intermediate
            out_gate_up_proj_ptr: T.handle, # intermediate
            out_silu_multiply_ptr: T.handle, # intermediate
            partial_sum_down_proj_ptr: T.handle, # intermediate
            before_down_proj_allreduce_ptr: T.handle, # intermediate
            
            # event tensor
            etensor_qkv_partial_ptr: T.handle, 
            etensor_q_reduce_ptr: T.handle, 
            etensor_k_reduce_ptr: T.handle, 
            etensor_v_reduce_ptr: T.handle, 
            etensor_decode_ptr: T.handle, 
            etensor_decode_merge_ptr: T.handle,
            etensor_o_proj_ptr: T.handle, 
            etensor_o_partial_ptr: T.handle, 
            etensor_o_allreduce_ptr: T.handle,
            etensor_attn_add_rms_ptr: T.handle,
            etensor_attn_mlp_ptr: T.handle,
            etensor_gate_up_proj_reduce_ptr: T.handle,
            etensor_gate_up_proj_ptr: T.handle,
            etensor_down_proj_ptr: T.handle,
            etensor_down_proj_reduce_ptr: T.handle,
            etensor_down_proj_allreduce_ptr: T.handle,
            etensor_mlp_add_rms_ptr: T.handle,
            etensor_end: T.Buffer((1, ), "int32", offset_factor=1),

            # execution queue
            queue_tasks: T.Buffer((DynamicTileScheduler.MAX_TASKS, 4), "int32", offset_factor=1),
            queue_head: T.Buffer((1,), "int32", offset_factor=1),
            queue_tail: T.Buffer((1,), "int32", offset_factor=1),
            profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
        ):

            # match buffer
            batch_size = T.int32()
            new_batch_size = T.int32()
            cos_sin_cache_len = T.int32()
            max_page_num = T.int32()
            total_page_num = T.int32()

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global", layout="default")
            residual_global = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global", layout="default")
            output_global = T.match_buffer(output_ptr, [batch_size, HIDDEN_SIZE], "float16", layout="default")

            # weight
            qkv_proj_weight_global = T.match_buffer(qkv_proj_weight_ptr, [(NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE], 
                                                    "float16", scope="global", layout="default")
            o_proj_weight_global = T.match_buffer(o_proj_weight_ptr, [HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM], 
                                                  "float16", scope="global", layout="default")
            q_rms_weight_global = T.match_buffer(q_rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default")
            k_rms_weight_global = T.match_buffer(k_rms_weight_ptr, [HEAD_DIM], "float16", scope="global", layout="default") 
            gate_up_weight_global = T.match_buffer(gate_up_weight_ptr, [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE], 
                                                   "float16", scope="global", layout="default")
            down_weight_global = T.match_buffer(down_weight_ptr, [HIDDEN_SIZE, INTERMEDIATE_SIZE], "float16", scope="global", layout="default")
            attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")
            mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global", layout="default")

            # page cache, kv cache and plan info
            cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, HEAD_DIM], "float32", scope="global", layout="default")
            rope_pos_global = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            kv_cache_global = T.match_buffer(kv_cache_ptr, [max_page_num, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM], 
                                             "float16", scope="global", layout="default")
            kv_indptr_global = T.match_buffer(kv_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default", offset_factor=1)
            kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", scope="global", layout="default", offset_factor=1)
            kv_last_page_len_global = T.match_buffer(kv_last_page_len_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            append_pos_global = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            request_indices_global = T.match_buffer(request_indices_ptr, [new_batch_size], "int32", scope="global", layout="default", offset_factor=1)
            kv_tile_indices_global = T.match_buffer(kv_tile_indices_ptr, [new_batch_size], "int32", scope="global", layout="default", offset_factor=1)
            max_chunk_size_global = T.match_buffer(max_chunk_size_ptr, [1], "int32", scope="global", layout="default", offset_factor=1)
            o_indptr_global = T.match_buffer(o_indptr_ptr, [batch_size + 1], "int32", scope="global", layout="default", offset_factor=1)

            # intermediate buffer
            partital_qkv_global = T.match_buffer(partital_qkv_ptr, [SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM], 
                                    "float32", scope="global", layout="default")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS, HEAD_DIM], 
                                    "float16", scope="global", layout="default")
            o_global = T.match_buffer(o_ptr, [batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                    "float16", scope="global", layout="default")
            lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_ATTENTION_HEADS], 
                                    "float32", scope="global", layout="default")
            o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS, HEAD_DIM], 
                                    "float32", scope="global", layout="default")
            lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS], 
                                    "float32", scope="global", layout="default")
            partial_o_global = T.match_buffer(partial_o_ptr, [SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], 
                                    "float32", scope="global", layout="default")
            before_o_allreduce_global = T.match_buffer(before_o_allreduce_ptr, [batch_size, HIDDEN_SIZE], 
                                    "float16", scope="global", layout="default")
            hidden_state_attn_mlp_global = T.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, HIDDEN_SIZE], 
                                    "float16", scope="global", layout="default")
            partial_out_gate_up_proj_global = T.match_buffer(partial_out_gate_up_proj_ptr, [GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, INTERMEDIATE_SIZE * 2], 
                                    "float32", scope="global", layout="default")
            out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE * 2], 
                                 "float16", scope="global", layout="default")
            out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], 
                                  "float16", scope="global", layout="default")
            partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], 
                                  "float32", layout="default")
            before_down_proj_allreduce_global = T.match_buffer(before_down_proj_allreduce_ptr, [batch_size, HIDDEN_SIZE], 
                                  "float16", scope="global", layout="default")

            # event tensor
            etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1  )
            etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_decode_global = T.match_buffer(etensor_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_o_allreduce_global = T.match_buffer(etensor_o_allreduce_ptr, [HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE], 
                                                        "int32", scope="global", layout="default", offset_factor=1)
            etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)
            etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", layout="default", offset_factor=1)
            etensor_gate_up_proj_reduce_global = T.match_buffer(etensor_gate_up_proj_reduce_ptr, [INTERMEDIATE_SIZE * 2 // GemmTile.BLK_N], 
                                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [INTERMEDIATE_SIZE // GemmTile.BLK_N], 
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR],
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [HIDDEN_SIZE // GemmTile.BLK_N],
                                    "int32", scope="global", layout="default", offset_factor=1)
            etensor_down_proj_allreduce_global = T.match_buffer(etensor_down_proj_allreduce_ptr, [HIDDEN_SIZE // WORLD_SIZE // AllreduceTile.N_TILE], 
                                                                "int32", scope="global", layout="default", offset_factor=1)
            etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", layout="default", offset_factor=1)


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
            qkv_proj_tile = T.meta_var(GemmTile(hidden_state_global, qkv_proj_weight_global, partital_qkv_global, split_k_factor=SPLIT_QKV_PROJECT))
            qkv_reduce_tile = T.meta_var(SplitKReduceTile(partital_qkv_global, 
                                                          Tp.reshape(qkv_global, [-1, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM]).buffer))
            rmsnorm_tile = T.meta_var(RMSnormTile(q_rms_weight_global, k_rms_weight_global, qkv_global))
            rope_tile = T.meta_var(RopeTile(qkv_global, cos_sin_cache_global, rope_pos_global))
            append_kv_tile = T.meta_var(AppendKVTile(kv_cache_global, qkv_global, append_pos_global))
            decode_tile = T.meta_var(DecodeTile(qkv_global, kv_cache_global, o_global, lse_global, o_tmp_global, lse_tmp_global,
                                                kv_indptr_global, kv_last_page_len_global,
                                                kv_indices_global, request_indices_global, kv_tile_indices_global, max_chunk_size_global))
            decode_merge_tile = T.meta_var(DecodeMergeTile(o_indptr_global, o_tmp_global, o_global, lse_tmp_global, lse_global))
            o_proj_tile = T.meta_var(GemmTile(Tp.reshape(o_global, [-1, NUM_ATTENTION_HEADS * HEAD_DIM]).buffer, o_proj_weight_global, 
                                              partial_o_global, split_k_factor=SPLIT_O_PROJRCT))
            o_reduce_tile = T.meta_var(SplitKReduceTile(partial_o_global, before_o_allreduce_global))
            o_allreduce_tile = T.meta_var(AllreduceTile(before_o_allreduce_global, hidden_state_attn_mlp_global, WORLD_SIZE))
            attn_add_rms_tile = T.meta_var(AddRMSNormTile(hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global))

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(hidden_state_attn_mlp_global, gate_up_weight_global, partial_out_gate_up_proj_global, split_k_factor=GATE_UP_PROJ_SPLIT_K_FACTOR))
            gemm_gate_up_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_out_gate_up_proj_global, out_gate_up_proj_global))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj_global, out_silu_multiply_global))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply_global, down_weight_global, partial_sum_down_proj_global, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj_global, before_down_proj_allreduce_global))
            down_proj_allreduce_tile = T.meta_var(AllreduceTile(before_down_proj_allreduce_global, output_global, WORLD_SIZE))
            mlp_add_rms_norm_tile = T.meta_var(AddRMSNormTile(output_global, residual_global, mlp_add_rms_weight_global))

            self.tile_list.append(qkv_proj_tile)
            self.tile_list.append(qkv_reduce_tile)
            self.tile_list.append(rmsnorm_tile)
            self.tile_list.append(rope_tile)
            self.tile_list.append(append_kv_tile)
            self.tile_list.append(decode_tile)
            self.tile_list.append(decode_merge_tile)
            self.tile_list.append(o_proj_tile)
            self.tile_list.append(o_reduce_tile)
            self.tile_list.append(o_allreduce_tile)
            self.tile_list.append(attn_add_rms_tile)
            self.tile_list.append(gemm_gate_up_proj_tile)
            self.tile_list.append(gemm_gate_up_proj_reduce_tile)
            self.tile_list.append(silu_multiply_tile)
            self.tile_list.append(gemm_down_proj_tile)
            self.tile_list.append(down_proj_reduce_tile)
            self.tile_list.append(down_proj_allreduce_tile)
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
                if self.profiler_on:
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
                                                             etensor_qkv_partial_global, use_nvshmem=True))
                    evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global, use_nvshmem=True))
                    evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global, use_nvshmem=True))
                    evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global, use_nvshmem=True))
                    evt_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS + 2, etensor_decode_global, use_nvshmem=True))
                    evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global, decrement=True, use_nvshmem=True))
                    evt_o_proj = T.meta_var(Semaphore(-1, etensor_o_proj_global, decrement=True, use_nvshmem=True))
                    evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global, use_nvshmem=True))
                    evt_o_allreduce = T.meta_var(Semaphore(WORLD_SIZE * o_reduce_tile.M_split, etensor_o_allreduce_global, use_nvshmem=True))
                    evt_attn_add_rms = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, o_allreduce_tile.N_TILE), etensor_attn_add_rms_global, use_nvshmem=True))
                    evt_attn_mlp = T.meta_var(Semaphore(batch_size, etensor_attn_mlp_global, use_nvshmem=True))
                    evt_gate_up_proj_reduce = T.meta_var(Semaphore(GATE_UP_PROJ_SPLIT_K_FACTOR, etensor_gate_up_proj_reduce_global, use_nvshmem=True))
                    evt_gate_up_proj = T.meta_var(Semaphore(2 * gemm_gate_up_proj_reduce_tile.M_split, etensor_gate_up_proj_global, use_nvshmem=True))
                    evt_down_proj = T.meta_var(Semaphore(-1, etensor_down_proj_global, decrement=True, use_nvshmem=True))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                                                etensor_down_proj_reduce_global, use_nvshmem=True))
                    evt_down_proj_allreduce = T.meta_var(Semaphore(WORLD_SIZE * down_proj_reduce_tile.M_split, etensor_down_proj_allreduce_global, use_nvshmem=True))
                    evt_add_rms_norm = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, down_proj_allreduce_tile.N_TILE), etensor_mlp_add_rms_global, use_nvshmem=True))
                    evt_end = T.meta_var(Semaphore(batch_size, etensor_end, use_nvshmem=True))

                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(DynamicTileScheduler(queue_tasks, queue_head, queue_tail, pool_allocator=pool, use_nvshmem=True))
                    tile_scheduler.init(warp_id)
                    
                    while tile_scheduler.valid():
                        if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
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
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            qkv_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
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
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.Q_RMSNORM_ROPE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            if new_batch_size == batch_size: # no split kv
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile,
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                    0,
                                    T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx + 1) * append_kv_tile.m_tile, batch_size)])
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
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.K_RMSNORM_ROPE_APPEND_KV.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rmsnorm_tile.h_tile, tile_scheduler.k_idx)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rope_tile.h_tile, tile_scheduler.k_idx)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            if batch_size == new_batch_size:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile,
                                    tile_scheduler.n_idx,
                                    0,
                                    T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx+1) * append_kv_tile.m_tile, batch_size)])
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
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.V_APPEND_KV.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 1)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            if batch_size == new_batch_size:
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.BATCH_DECODE_NO_SPLIT.value,
                                    tile_scheduler.m_idx * append_kv_tile.m_tile,
                                    tile_scheduler.n_idx,
                                    0,
                                    T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
                                    0,
                                    warp_id,
                                    lane_id,
                                    evt_decode,
                                    tile_scheduler.m_idx,
                                    tile_scheduler.n_idx,
                                )
                            else:
                                range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
                                range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx+1) * append_kv_tile.m_tile, batch_size)])
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
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
                            T.tvm_storage_sync("shared")
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
                            range_end = T.meta_var(((tile_scheduler.n_idx + 1) * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM - 1) // o_proj_tile.TILE_K)
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
                            for kr in T.serial(range_end - range_start):
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.GEMM_O_PROJ.value,
                                    0,
                                    0,
                                    range_start + 1 + kr,
                                    ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                    1,
                                    warp_id,
                                    lane_id,
                                    evt_o_proj,
                                    range_start + 1 + kr,
                                    use_barrier=False
                                )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
                            T.tvm_storage_sync("shared")
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.DECODE_MERGE.value,
                                batch_idx, 
                                tile_scheduler.n_idx * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) // DecodeMergeTile.bdz,
                                0,
                                (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) // DecodeMergeTile.bdz,
                                1,
                                warp_id,
                                lane_id,
                                evt_decode_merge,
                                batch_idx,
                                tile_scheduler.n_idx,
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
                            T.tvm_storage_sync("shared")
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            range_start = T.meta_var(tile_scheduler.n_idx * DecodeMergeTile.bdz * HEAD_DIM // o_proj_tile.TILE_K)
                            range_end = T.meta_var(((tile_scheduler.n_idx + 1) * DecodeMergeTile.bdz * HEAD_DIM - 1) // o_proj_tile.TILE_K)
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
                            for kr in T.serial(range_end - range_start):
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.GEMM_O_PROJ.value,
                                    0,
                                    0,
                                    range_start + 1 + kr,
                                    ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
                                    1,
                                    warp_id,
                                    lane_id,
                                    evt_o_proj,
                                    range_start + 1 + kr,
                                    use_barrier=False,
                                )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
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
                                use_barrier=False,
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.O_ALLREDUCE.value,
                                0,
                                tile_scheduler.n_idx // WORLD_SIZE,
                                0,
                                ceildiv(batch_size, o_allreduce_tile.M_TILE),
                                0,
                                warp_id,
                                lane_id,
                                evt_o_allreduce,
                                tile_scheduler.n_idx // WORLD_SIZE,
                                rank=tile_scheduler.n_idx % WORLD_SIZE
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.O_ALLREDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.O_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            o_allreduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.O_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim_wg_with_extend_rank(
                                JobType.ATTN_ADD_RMS_NORM.value,
                                tile_scheduler.m_idx * o_allreduce_tile.M_TILE,
                                0,
                                0,
                                T.min(o_allreduce_tile.M_TILE, batch_size - tile_scheduler.m_idx * o_allreduce_tile.M_TILE),
                                0,
                                warp_id,
                                lane_id,
                                evt_attn_add_rms,
                                get_rank_map,
                                WORLD_SIZE,
                                tile_scheduler.m_idx,
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            attn_add_rms_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks(
                                lambda idx: (JobType.GEMM_GATE_UP_PROJ.value, 0, idx // GATE_UP_PROJ_SPLIT_K_FACTOR, idx % GATE_UP_PROJ_SPLIT_K_FACTOR),
                                ceildiv(INTERMEDIATE_SIZE *2 , GemmTile.BLK_N) * GATE_UP_PROJ_SPLIT_K_FACTOR,
                                warp_id,
                                lane_id,
                                evt_attn_mlp,
                                0,
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GATE_UP_PROJ_REDUCE.value,
                                0,
                                tile_scheduler.n_idx,
                                0,
                                gemm_gate_up_proj_reduce_tile.M_split,
                                0,
                                warp_id,
                                lane_id,
                                evt_gate_up_proj_reduce,
                                tile_scheduler.n_idx,
                                use_barrier=False,
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GATE_UP_PROJ_REDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GATE_UP_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            gemm_gate_up_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GATE_UP_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                offset = T.meta_var(tile_scheduler.n_idx-INTERMEDIATE_SIZE//GemmTile.BLK_N)
                                tile_scheduler.push_task(
                                    JobType.SPLIT_SILU_MULTIPLY.value, 
                                    offset, 
                                    0, 
                                    0, 
                                    warp_id, 
                                    evt_gate_up_proj, 
                                    offset, 
                                )
                            else:
                                tile_scheduler.push_task(
                                    JobType.SPLIT_SILU_MULTIPLY.value, 
                                    tile_scheduler.n_idx, 
                                    0, 
                                    0, 
                                    warp_id, 
                                    evt_gate_up_proj, 
                                    tile_scheduler.n_idx
                                )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            range_start = T.meta_var(tile_scheduler.m_idx * SiluMultiplyTile.TILE_SIZE // gemm_down_proj_tile.TILE_K)
                            range_end = T.meta_var(((tile_scheduler.m_idx + 1) * SiluMultiplyTile.TILE_SIZE - 1) // gemm_down_proj_tile.TILE_K)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.GEMM_DOWN_PROJ.value, 
                                0, 
                                0, 
                                range_start, 
                                HIDDEN_SIZE // GemmTile.BLK_N, 
                                1, 
                                warp_id, 
                                lane_id, 
                                evt_down_proj, 
                                range_start
                            )
                            for kr in T.serial(range_end - range_start):
                                tile_scheduler.push_tasks_along_dim(
                                    JobType.GEMM_DOWN_PROJ.value, 
                                    0, 
                                    0, 
                                    range_start + kr + 1, 
                                    HIDDEN_SIZE // GemmTile.BLK_N, 
                                    1, 
                                    warp_id, 
                                    lane_id, 
                                    evt_down_proj, 
                                    range_start + kr + 1,
                                    use_barrier=False
                                )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.DOWN_PROJ_REDUCE.value, 
                                0, 
                                tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                0, 
                                down_proj_reduce_tile.M_split, 
                                0, 
                                warp_id, 
                                lane_id, 
                                evt_down_proj_reduce, 
                                tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                use_barrier=False
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.DOWN_PROJ_ALLREDUCE.value,
                                0,
                                tile_scheduler.n_idx // WORLD_SIZE,
                                0,
                                ceildiv(batch_size, down_proj_allreduce_tile.M_TILE),
                                0,
                                warp_id,
                                lane_id,
                                evt_down_proj_allreduce,
                                tile_scheduler.n_idx // WORLD_SIZE,
                                rank=tile_scheduler.n_idx % WORLD_SIZE
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_ALLREDUCE.value:
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            down_proj_allreduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_ALLREDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim_wg_with_extend_rank(
                                JobType.MLP_ADD_RMS_NORM.value, 
                                tile_scheduler.m_idx * down_proj_allreduce_tile.M_TILE, 
                                0, 
                                0,
                                T.min(down_proj_allreduce_tile.M_TILE, batch_size - tile_scheduler.m_idx * down_proj_allreduce_tile.M_TILE), 
                                0, 
                                warp_id, 
                                lane_id, 
                                evt_add_rms_norm, 
                                get_rank_map, 
                                WORLD_SIZE,
                                tile_scheduler.m_idx
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                            T.tvm_storage_sync("shared")
                            if self.profiler_on:
                                T.timer_start_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            mlp_add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                                T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                            tile_scheduler.push_tasks_along_dim(
                                JobType.END.value, 
                                0, 
                                0, 
                                0, 
                                KernelConfig.SM_NUMBER, 
                                0, 
                                warp_id, 
                                lane_id, 
                                evt_end, 
                                0
                            )
                            if self.profiler_on:
                                T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        if self.profiler_on:
                            T.timer_start_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                        tile_scheduler.next_tile(warp_id)
                        if self.profiler_on:
                            T.timer_end_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
                    self.class_finalize_all()
        # fmt: on

        return mega_kernel


def generate_exec_queue_dynamic():
    exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
    for n in reversed(
        range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, GemmTile.BLK_N))
    ):
        for k in range(SPLIT_QKV_PROJECT):
            exec_queue.enqueue(JobType.GEMM_QKV_PROJ.value, 0, n, k)
    return exec_queue


def decode_plan(batch_size, kv_indptr):
    # plan for decoding
    smem_size_decode = (
        2
        * DecodeTile.pipe_depth
        * DecodeTile.loop_inner
        * DecodeTile.bdz
        * DecodeTile.bdy
        * DecodeTile.tile_per_bdx
        * DecodeTile.head_dim
        * F16_BYTES
        + KernelConfig.NUM_THREADS * DecodeTile.tile_per_bdx * F32_BYTES
        + KernelConfig.NUM_THREADS * DecodeTile.loop_inner * DecodeTile.vec_size * F32_BYTES
    )
    assert smem_size_decode <= KernelConfig.MAX_SMEM_SIZE
    assert DecodeTile.pipe_depth <= DecodeTile.bdx

    DEV = tvm.cuda(0)

    # balance the workload (split-kv)
    kv_indptr_h = kv_indptr.cpu().numpy()
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
            if (
                new_batch_size * DecodeTile.qo_heads
                > KernelConfig.SM_NUMBER * DecodeTile.max_blk_per_sm
            ):
                low = mid + 1
            else:
                high = mid
        max_page_num = low
        new_batch_size = 0
        for page_num in page_num_list:
            new_batch_size += ceildiv(page_num, max_page_num)
        split_kv = new_batch_size != batch_size

    max_chunk_size = tvm.nd.array(np.array([max_page_num * PAGE_SIZE], dtype=np.int32), device=DEV)

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

    request_indices = tvm.nd.array(np.array(request_indices, dtype=np.int32), DEV)
    kv_tile_indices = tvm.nd.array(np.array(kv_tile_indices, dtype=np.int32), DEV)
    o_indptr = tvm.nd.array(np.array(o_indptr, dtype=np.int32), DEV)

    return (
        split_kv,
        new_batch_size,
        max_chunk_size,
        request_indices,
        kv_tile_indices,
        o_indptr,
    )


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static, mega_kernel_dynamic, sess):

    def prepare_data(batch_size):
        import torch

        torch.manual_seed(42)
        arg_dict = {}

        # input
        arg_dict["hidden_state"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["residual"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)

        # qkv projection
        arg_dict["qkv_proj_weight"] = torch.randn(
            (WORLD_SIZE, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE),
            dtype=torch.float16,
        )
        torch.nn.init.xavier_normal_(arg_dict["qkv_proj_weight"], gain=1.0)

        # rms
        arg_dict["q_rms_wight"] = torch.randn((HEAD_DIM), dtype=torch.float16)
        arg_dict["k_rms_wight"] = torch.randn((HEAD_DIM), dtype=torch.float16)

        # qkv
        arg_dict["qkv"] = torch.randn(
            (batch_size, NUM_KEY_VALUE_HEADS * 2 + NUM_ATTENTION_HEADS, HEAD_DIM),
            dtype=torch.float16,
        )

        # rope cos_sin_cache
        inv_freq = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float, device="cuda") / HEAD_DIM)
        )
        pos = SEQ_LEN
        assert pos < 4096  # for faster test
        arg_dict["rope_pos"] = torch.full((batch_size,), pos, dtype=torch.int32)
        t = torch.arange(4096, dtype=torch.float, device="cuda")
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        arg_dict["cos_sin_cache"] = torch.cat((cos, sin), dim=-1).reshape(-1, HEAD_DIM).cpu()

        # paged kv-cache
        page_last_len = (
            PAGE_SIZE if (SEQ_LEN + 1) % PAGE_SIZE == 0 else (SEQ_LEN + 1) % PAGE_SIZE
        )  # +1 since need to append new kv
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
        append_pos = torch.empty(batch_size, dtype=torch.int32).int()
        for i in range(batch_size):
            append_pos[i] = SEQ_LEN
        arg_dict["kv_cache"] = torch.randn(
            (WORLD_SIZE, MAX_PAGE_NUM, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM),
            dtype=torch.float16,
        ).cpu()
        arg_dict["kv_indptr"] = kv_indptr.cpu()
        arg_dict["kv_last_page_len"] = kv_last_page_len.cpu()
        arg_dict["kv_indices"] = kv_indices.cpu()
        arg_dict["append_pos"] = append_pos.cpu()

        # output
        arg_dict["o"] = torch.zeros(
            (batch_size, NUM_ATTENTION_HEADS, HEAD_DIM), dtype=torch.float16
        )
        arg_dict["lse"] = torch.zeros((batch_size, NUM_ATTENTION_HEADS), dtype=torch.float32)
        arg_dict["o_proj_weight"] = torch.randn(
            (WORLD_SIZE, HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM), dtype=torch.float16
        )
        torch.nn.init.xavier_normal_(arg_dict["o_proj_weight"], gain=1.0)
        arg_dict["hidden_state_attn_mlp"] = torch.zeros(
            (batch_size, HIDDEN_SIZE), dtype=torch.float16
        )

        # add rms
        arg_dict["attn_add_rms_weight"] = torch.randn((HIDDEN_SIZE), dtype=torch.float16)

        # mlp
        arg_dict["gate_up_weight"] = torch.zeros(
            (WORLD_SIZE, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=torch.float16
        )
        torch.nn.init.xavier_normal_(arg_dict["gate_up_weight"], gain=1.0)
        arg_dict["out_gate_up_proj"] = torch.zeros(
            (batch_size, INTERMEDIATE_SIZE * 2), dtype=torch.float16
        )
        arg_dict["partial_out_gate_up_proj"] = torch.zeros(
            (GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, INTERMEDIATE_SIZE * 2), dtype=torch.float32
        )
        arg_dict["out_silu_multiply"] = torch.zeros(
            (batch_size, INTERMEDIATE_SIZE), dtype=torch.float16
        )
        arg_dict["down_weight"] = (
            torch.zeros((WORLD_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.float16) * 0.01
        )
        torch.nn.init.xavier_normal_(arg_dict["down_weight"], gain=1.0)
        arg_dict["partial_sum_down_proj"] = torch.zeros(
            (DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE), dtype=torch.float32
        )
        arg_dict["output"] = torch.zeros((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["mlp_add_rms_weight"] = torch.randn((HIDDEN_SIZE,), dtype=torch.float16)

        return arg_dict

    arg_dict = prepare_data(batch_size)

    def tir_static(arg_dict):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        (
            tvm_arg_dict["split_kv"],
            tvm_arg_dict["new_batch_size"],
            tvm_arg_dict["max_chunk_size"],
            tvm_arg_dict["request_indices"],
            tvm_arg_dict["kv_tile_indices"],
            tvm_arg_dict["o_indptr"],
        ) = decode_plan(batch_size, arg_dict["kv_indptr"])

        tvm_arg_dict["o_tmp"] = tvm.nd.array(
            np.zeros(
                [tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads, DecodeTile.head_dim],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["lse_tmp"] = tvm.nd.array(
            np.zeros([tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads], dtype=np.float32), DEV
        )
        tvm_arg_dict["partial_qkv"] = tvm.nd.array(
            np.zeros(
                [
                    SPLIT_QKV_PROJECT,
                    batch_size,
                    (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM,
                ],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["partial_o"] = tvm.nd.array(
            np.zeros([SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], dtype=np.float32), DEV
        )

        # static schedule
        generate_exec_queue = tvm.get_global_func("megakernel.generate_exec_queue")
        exec_queue = generate_exec_queue(
            batch_size,
            tvm_arg_dict["new_batch_size"],
            WORLD_SIZE,
            NUM_ATTENTION_HEADS,
            NUM_KEY_VALUE_HEADS,
            HEAD_DIM,
            DEV,
            tvm.cpu(),
        )
        tvm_arg_dict["exec_queue"] = tvm.nd.array(exec_queue, device=DEV)

        # append_pos here is different from flashinfer
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["kv_indices"][
                    (arg_dict["kv_indptr"][b] * PAGE_SIZE + append_pos[b]) // PAGE_SIZE
                ]
                * PAGE_SIZE
                + append_pos[b] % PAGE_SIZE
            )
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.nd.array(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.nd.array(append_pos, device=DEV)
        REPEAT = 10
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.nd.array(arg_dict["residual"], device=DEV)
            # generate event tensor
            (
                tvm_arg_dict[f"etensor_qkv_partial_{i}"],
                tvm_arg_dict[f"etensor_q_reduce_{i}"],
                tvm_arg_dict[f"etensor_k_reduce_{i}"],
                tvm_arg_dict[f"etensor_v_reduce_{i}"],
                tvm_arg_dict[f"etensor_decode_{i}"],
                tvm_arg_dict[f"etensor_decode_merge_{i}"],
                tvm_arg_dict[f"etensor_o_proj_{i}"],
                tvm_arg_dict[f"etensor_o_partial_{i}"],
                tvm_arg_dict[f"etensor_o_allreduce_{i}"],
                tvm_arg_dict[f"etensor_attn_add_rms_norm_{i}"],
                tvm_arg_dict[f"etensor_attn_mlp_{i}"],
                tvm_arg_dict[f"etensor_gate_up_proj_reduce_{i}"],
                tvm_arg_dict[f"etensor_gate_up_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_reduce_{i}"],
                tvm_arg_dict[f"etensor_down_proj_allreduce_{i}"],
                tvm_arg_dict[f"etensor_mlp_add_rms_norm_{i}"],
                _,
            ) = generate_event_tensor(batch_size, tvm_arg_dict["o_indptr"], batch_size != tvm_arg_dict["new_batch_size"], WORLD_SIZE)
            tvm_arg_dict[f"profiler_buffer_{i}"] = tvm.nd.array(np.zeros([PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV)

        nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")

        # prepare test data
        print(f"begin preparing test data ...")
        np.random.seed(42)
        # prepare disco args
        tensor_to_gather = [
            "qkv_proj_weight",
            "o_proj_weight",
            "gate_up_weight",
            "down_weight",
            "kv_cache",
        ]
        disco_arg_dict = {}
        for key, value in tvm_arg_dict.items():
            if not isinstance(value, tvm.nd.NDArray):
                continue
            if key in tensor_to_gather:
                disco_arg_dict[key] = sess.empty(value.shape[1:], value.dtype)
            elif "etensor" in key:
                disco_arg_dict[key] = nvshmem_malloc_hook(
                    ShapeTuple([*value.shape]), str(value.dtype), None
                )
            else:
                disco_arg_dict[key] = sess.empty(value.shape, value.dtype)
        disco_arg_dict["before_o_allreduce"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["hidden_state_attn_mlp"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["before_down_proj_allreduce"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["output"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )

        res_dict = {
            "output_host": tvm.nd.empty((batch_size, HIDDEN_SIZE), "float16", device=DEV),
            "residual_host": tvm.nd.empty((batch_size, HIDDEN_SIZE), "float16", device=DEV),
            "hidden_state_attn_mlp_host": tvm.nd.empty(
                (WORLD_SIZE, batch_size, HIDDEN_SIZE), "float16", device=DEV
            ),
            "hidden_state_attn_mlp_res": sess.empty(
                (WORLD_SIZE, batch_size, HIDDEN_SIZE), "float16", worker0_only=True
            ),
            "profiler_buffer_host": tvm.nd.empty(
                (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
            ),
            "profiler_buffer_res": sess.empty(
                (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
            ),
        }
        # init disco weight/input args
        gathered_arg_dict = {}
        for key, value in tvm_arg_dict.items():
            if key in tensor_to_gather:
                gathered_arg_dict[key] = sess.empty(value.shape, value.dtype, worker0_only=True)
                sess.copy_to_worker_0(value, gathered_arg_dict[key])
                sess.scatter_from_worker0(gathered_arg_dict[key], disco_arg_dict[key])
            elif key in disco_arg_dict:
                sess.broadcast(value, disco_arg_dict[key])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.so"
            mega_kernel_static.export_library(path)
            rt_mod = sess.load_vm_module(path)
            sess._sync_all()

        with target:
            iter = 0

            def func():
                nonlocal iter
                rt_mod["mega_kernel"](
                    # input and output
                    disco_arg_dict["hidden_state"],
                    disco_arg_dict[f"residual_{iter}"],
                    disco_arg_dict["output"],
                    # weight
                    disco_arg_dict["qkv_proj_weight"],
                    disco_arg_dict["o_proj_weight"],
                    disco_arg_dict["q_rms_wight"],
                    disco_arg_dict["k_rms_wight"],
                    disco_arg_dict["gate_up_weight"],
                    disco_arg_dict["down_weight"],
                    disco_arg_dict["attn_add_rms_weight"],
                    disco_arg_dict["mlp_add_rms_weight"],
                    # page cache, cos_sin cache and plan info
                    disco_arg_dict["cos_sin_cache"],
                    disco_arg_dict["rope_pos"],
                    disco_arg_dict["kv_cache"],
                    disco_arg_dict["kv_indptr"],
                    disco_arg_dict["kv_indices"],
                    disco_arg_dict["kv_last_page_len"],
                    disco_arg_dict["append_pos"],
                    disco_arg_dict["request_indices"],
                    disco_arg_dict["kv_tile_indices"],
                    disco_arg_dict["max_chunk_size"],
                    disco_arg_dict["o_indptr"],
                    # intermediate buffer
                    disco_arg_dict["partial_qkv"],
                    disco_arg_dict["qkv"],
                    disco_arg_dict["o"],
                    disco_arg_dict["lse"],
                    disco_arg_dict["o_tmp"],
                    disco_arg_dict["lse_tmp"],
                    disco_arg_dict["partial_o"],
                    disco_arg_dict["before_o_allreduce"],
                    disco_arg_dict["hidden_state_attn_mlp"],
                    disco_arg_dict["partial_out_gate_up_proj"],
                    disco_arg_dict["out_gate_up_proj"],
                    disco_arg_dict["out_silu_multiply"],
                    disco_arg_dict["partial_sum_down_proj"],
                    disco_arg_dict["before_down_proj_allreduce"],
                    # event tensor
                    disco_arg_dict[f"etensor_qkv_partial_{iter}"],
                    disco_arg_dict[f"etensor_q_reduce_{iter}"],
                    disco_arg_dict[f"etensor_k_reduce_{iter}"],
                    disco_arg_dict[f"etensor_v_reduce_{iter}"],
                    disco_arg_dict[f"etensor_decode_{iter}"],
                    disco_arg_dict[f"etensor_decode_merge_{iter}"],
                    disco_arg_dict[f"etensor_o_proj_{iter}"],
                    disco_arg_dict[f"etensor_o_partial_{iter}"],
                    disco_arg_dict[f"etensor_o_allreduce_{iter}"],
                    disco_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                    disco_arg_dict[f"etensor_attn_mlp_{iter}"],
                    disco_arg_dict[f"etensor_gate_up_proj_reduce_{iter}"],
                    disco_arg_dict[f"etensor_gate_up_proj_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_allreduce_{iter}"],
                    disco_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                    # exec queue
                    disco_arg_dict["exec_queue"],
                    disco_arg_dict[f"profiler_buffer_{iter}"],
                )
                iter += 1

            # ms = bench(func, warmup=1, repeat=10, proton_name="tir-static")
            for i in range(REPEAT):
                func()
            sess._sync_all()
            sess.copy_from_worker_0(res_dict["output_host"], disco_arg_dict["output"])
            sess.copy_from_worker_0(res_dict["residual_host"], disco_arg_dict[f"residual_0"])
            # sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], disco_arg_dict["hidden_state_attn_mlp"])
            sess.gather_to_worker0(
                disco_arg_dict["hidden_state_attn_mlp"], res_dict["hidden_state_attn_mlp_res"]
            )
            sess.copy_from_worker_0(
                res_dict["hidden_state_attn_mlp_host"], res_dict["hidden_state_attn_mlp_res"]
            )
            sess.gather_to_worker0(
                disco_arg_dict[f"profiler_buffer_{iter-1}"], res_dict["profiler_buffer_res"]
            )
            sess.copy_from_worker_0(
                res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"]
            )
            sess._sync_all()

        if PROFILER_ON:
            for r in range(WORLD_SIZE):
                export_to_perfetto_trace(
                    res_dict["profiler_buffer_host"].numpy()[r],
                    f"static-schedule-layer-tp-8-rank-{r}.perfetto-trace",
                    event_type_names,
                )

        return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def tir_dynamic(arg_dict):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        (
            tvm_arg_dict["split_kv"],
            tvm_arg_dict["new_batch_size"],
            tvm_arg_dict["max_chunk_size"],
            tvm_arg_dict["request_indices"],
            tvm_arg_dict["kv_tile_indices"],
            tvm_arg_dict["o_indptr"],
        ) = decode_plan(batch_size, arg_dict["kv_indptr"])

        tvm_arg_dict["o_tmp"] = tvm.nd.array(
            np.zeros(
                [tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads, DecodeTile.head_dim],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["lse_tmp"] = tvm.nd.array(
            np.zeros([tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads], dtype=np.float32), DEV
        )
        tvm_arg_dict["partial_qkv"] = tvm.nd.array(
            np.zeros(
                [
                    SPLIT_QKV_PROJECT,
                    batch_size,
                    (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM,
                ],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["partial_o"] = tvm.nd.array(
            np.zeros([SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE], dtype=np.float32), DEV
        )

        # static schedule
        exec_queue = generate_exec_queue_dynamic()

        # append_pos here is different from flashinfer
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["kv_indices"][
                    (arg_dict["kv_indptr"][b] * PAGE_SIZE + append_pos[b]) // PAGE_SIZE
                ]
                * PAGE_SIZE
                + append_pos[b] % PAGE_SIZE
            )
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.nd.array(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.nd.array(append_pos, device=DEV)
        REPEAT = 10
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.nd.array(arg_dict["residual"], device=DEV)
            # generate event tensor
            (
                tvm_arg_dict[f"etensor_qkv_partial_{i}"],
                tvm_arg_dict[f"etensor_q_reduce_{i}"],
                tvm_arg_dict[f"etensor_k_reduce_{i}"],
                tvm_arg_dict[f"etensor_v_reduce_{i}"],
                tvm_arg_dict[f"etensor_decode_{i}"],
                tvm_arg_dict[f"etensor_decode_merge_{i}"],
                tvm_arg_dict[f"etensor_o_proj_{i}"],
                tvm_arg_dict[f"etensor_o_partial_{i}"],
                tvm_arg_dict[f"etensor_o_allreduce_{i}"],
                tvm_arg_dict[f"etensor_attn_add_rms_norm_{i}"],
                tvm_arg_dict[f"etensor_attn_mlp_{i}"],
                tvm_arg_dict[f"etensor_gate_up_proj_reduce_{i}"],
                tvm_arg_dict[f"etensor_gate_up_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_reduce_{i}"],
                tvm_arg_dict[f"etensor_down_proj_allreduce_{i}"],
                tvm_arg_dict[f"etensor_mlp_add_rms_norm_{i}"],
                tvm_arg_dict[f"etensor_end_{i}"],
            ) = generate_event_tensor(batch_size, tvm_arg_dict["o_indptr"], batch_size != tvm_arg_dict["new_batch_size"], WORLD_SIZE)
            tvm_arg_dict[f"queue_tasks_{i}"] = tvm.nd.array(exec_queue.tasks, DEV)
            tvm_arg_dict[f"queue_head_{i}"] = tvm.nd.array(exec_queue.head, DEV)
            tvm_arg_dict[f"queue_tail_{i}"] = tvm.nd.array(exec_queue.tail, DEV)
            tvm_arg_dict[f"profiler_buffer_{i}"] = tvm.nd.array(np.zeros([PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV)

        nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")

        # prepare test data
        print(f"begin preparing test data ...")
        np.random.seed(42)
        # prepare disco args
        tensor_to_gather = [
            "qkv_proj_weight",
            "o_proj_weight",
            "gate_up_weight",
            "down_weight",
            "kv_cache",
        ]
        disco_arg_dict = {}
        for key, value in tvm_arg_dict.items():
            if not isinstance(value, tvm.nd.NDArray):
                continue
            if key in tensor_to_gather:
                disco_arg_dict[key] = sess.empty(value.shape[1:], value.dtype)
            elif "etensor" in key or "queue" in key:
                disco_arg_dict[key] = nvshmem_malloc_hook(
                    ShapeTuple([*value.shape]), str(value.dtype), None
                )
            else:
                disco_arg_dict[key] = sess.empty(value.shape, value.dtype)
        disco_arg_dict["before_o_allreduce"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["hidden_state_attn_mlp"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["before_down_proj_allreduce"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )
        disco_arg_dict["output"] = nvshmem_malloc_hook(
            ShapeTuple((batch_size, HIDDEN_SIZE)), "float16", None
        )

        res_dict = {
            "output_host": tvm.nd.empty((batch_size, HIDDEN_SIZE), "float16", device=DEV),
            "residual_host": tvm.nd.empty((batch_size, HIDDEN_SIZE), "float16", device=DEV),
            "hidden_state_attn_mlp_host": tvm.nd.empty(
                (WORLD_SIZE, batch_size, HIDDEN_SIZE), "float16", device=DEV
            ),
            "hidden_state_attn_mlp_res": sess.empty(
                (WORLD_SIZE, batch_size, HIDDEN_SIZE), "float16", worker0_only=True
            ),
            "profiler_buffer_host": tvm.nd.empty(
                (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", device=DEV
            ),
            "profiler_buffer_res": sess.empty(
                (WORLD_SIZE, PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
            ),
        }
        # init disco weight/input args
        gathered_arg_dict = {}
        for key, value in tvm_arg_dict.items():
            if key in tensor_to_gather:
                gathered_arg_dict[key] = sess.empty(value.shape, value.dtype, worker0_only=True)
                sess.copy_to_worker_0(value, gathered_arg_dict[key])
                sess.scatter_from_worker0(gathered_arg_dict[key], disco_arg_dict[key])
            elif key in disco_arg_dict:
                sess.broadcast(value, disco_arg_dict[key])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.so"
            mega_kernel_dynamic.export_library(path)
            rt_mod = sess.load_vm_module(path)
            sess._sync_all()

        with target:
            iter = 0

            def func():
                nonlocal iter
                rt_mod["mega_kernel"](
                    # input and output
                    disco_arg_dict["hidden_state"],
                    disco_arg_dict[f"residual_{iter}"],
                    disco_arg_dict["output"],
                    # weight
                    disco_arg_dict["qkv_proj_weight"],
                    disco_arg_dict["o_proj_weight"],
                    disco_arg_dict["q_rms_wight"],
                    disco_arg_dict["k_rms_wight"],
                    disco_arg_dict["gate_up_weight"],
                    disco_arg_dict["down_weight"],
                    disco_arg_dict["attn_add_rms_weight"],
                    disco_arg_dict["mlp_add_rms_weight"],
                    # page cache, cos_sin cache and plan info
                    disco_arg_dict["cos_sin_cache"],
                    disco_arg_dict["rope_pos"],
                    disco_arg_dict["kv_cache"],
                    disco_arg_dict["kv_indptr"],
                    disco_arg_dict["kv_indices"],
                    disco_arg_dict["kv_last_page_len"],
                    disco_arg_dict["append_pos"],
                    disco_arg_dict["request_indices"],
                    disco_arg_dict["kv_tile_indices"],
                    disco_arg_dict["max_chunk_size"],
                    disco_arg_dict["o_indptr"],
                    # intermediate buffer
                    disco_arg_dict["partial_qkv"],
                    disco_arg_dict["qkv"],
                    disco_arg_dict["o"],
                    disco_arg_dict["lse"],
                    disco_arg_dict["o_tmp"],
                    disco_arg_dict["lse_tmp"],
                    disco_arg_dict["partial_o"],
                    disco_arg_dict["before_o_allreduce"],
                    disco_arg_dict["hidden_state_attn_mlp"],
                    disco_arg_dict["partial_out_gate_up_proj"],
                    disco_arg_dict["out_gate_up_proj"],
                    disco_arg_dict["out_silu_multiply"],
                    disco_arg_dict["partial_sum_down_proj"],
                    disco_arg_dict["before_down_proj_allreduce"],
                    # event tensor
                    disco_arg_dict[f"etensor_qkv_partial_{iter}"],
                    disco_arg_dict[f"etensor_q_reduce_{iter}"],
                    disco_arg_dict[f"etensor_k_reduce_{iter}"],
                    disco_arg_dict[f"etensor_v_reduce_{iter}"],
                    disco_arg_dict[f"etensor_decode_{iter}"],
                    disco_arg_dict[f"etensor_decode_merge_{iter}"],
                    disco_arg_dict[f"etensor_o_proj_{iter}"],
                    disco_arg_dict[f"etensor_o_partial_{iter}"],
                    disco_arg_dict[f"etensor_o_allreduce_{iter}"],
                    disco_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                    disco_arg_dict[f"etensor_attn_mlp_{iter}"],
                    disco_arg_dict[f"etensor_gate_up_proj_reduce_{iter}"],
                    disco_arg_dict[f"etensor_gate_up_proj_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                    disco_arg_dict[f"etensor_down_proj_allreduce_{iter}"],
                    disco_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                    disco_arg_dict[f"etensor_end_{iter}"],
                    # exec queue
                    disco_arg_dict[f"queue_tasks_{iter}"],
                    disco_arg_dict[f"queue_head_{iter}"],
                    disco_arg_dict[f"queue_tail_{iter}"],
                    disco_arg_dict[f"profiler_buffer_{iter}"],
                )
                iter += 1

            for i in range(REPEAT):
                func()
            sess._sync_all()
            sess.copy_from_worker_0(res_dict["output_host"], disco_arg_dict["output"])
            sess.copy_from_worker_0(res_dict["residual_host"], disco_arg_dict[f"residual_0"])
            # sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], disco_arg_dict["hidden_state_attn_mlp"])
            sess.gather_to_worker0(
                disco_arg_dict["hidden_state_attn_mlp"], res_dict["hidden_state_attn_mlp_res"]
            )
            sess.copy_from_worker_0(
                res_dict["hidden_state_attn_mlp_host"], res_dict["hidden_state_attn_mlp_res"]
            )
            sess.gather_to_worker0(
                disco_arg_dict[f"profiler_buffer_{iter-1}"], res_dict["profiler_buffer_res"]
            )
            sess.copy_from_worker_0(
                res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"]
            )
            sess._sync_all()

        if PROFILER_ON:
            for r in range(WORLD_SIZE):
                export_to_perfetto_trace(
                    res_dict["profiler_buffer_host"].numpy()[r],
                    f"dynamic-schedule-layer-tp-8-rank-{r}.perfetto-trace",
                    event_type_names,
                )

        return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def std(arg_dict):
        import flashinfer
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        FULL_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE * WORLD_SIZE
        FULL_NUM_ATTENTION_HEADS = NUM_ATTENTION_HEADS * WORLD_SIZE
        FULL_NUM_KEY_VALUE_HEADS = NUM_KEY_VALUE_HEADS * WORLD_SIZE

        def func():

            for key, value in arg_dict.items():
                if key == "qkv_proj_weight":
                    split_sizes = [NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_HEADS]
                    value = value.reshape(WORLD_SIZE, -1, HEAD_DIM, HIDDEN_SIZE)
                    q_weight, k_weight, v_weight = torch.split(value, split_sizes, dim=1)
                    q_weight = q_weight.reshape(-1, HIDDEN_SIZE)
                    k_weight = k_weight.reshape(-1, HIDDEN_SIZE)
                    v_weight = v_weight.reshape(-1, HIDDEN_SIZE)
                    value = torch.cat([q_weight, k_weight, v_weight], dim=0)
                elif key == "gate_up_weight":
                    value = value.reshape(-1, *value.shape[2:])
                elif key == "o_proj_weight" or key == "down_weight":
                    value = value.transpose(0, 1)
                    value = value.reshape(value.shape[0], -1)
                elif key == "kv_cache":
                    value = value.movedim(0, 2)
                    value = value.reshape(value.shape[0], value.shape[1], -1, *value.shape[4:])

                std_arg_dict[key] = value.clone().to(torch_dev)

            workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
            wrapper.plan(
                indptr=std_arg_dict["kv_indptr"],
                indices=std_arg_dict["kv_indices"],
                last_page_len=std_arg_dict["kv_last_page_len"],
                num_qo_heads=FULL_NUM_ATTENTION_HEADS,
                num_kv_heads=FULL_NUM_KEY_VALUE_HEADS,
                head_dim=HEAD_DIM,
                page_size=PAGE_SIZE,
                pos_encoding_mode="NONE",
                data_type=torch.float16,
                q_data_type=torch.float16,
            )

            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, HEAD_DIM)
            q, k, v = torch.split(
                qkv,
                [FULL_NUM_ATTENTION_HEADS, FULL_NUM_KEY_VALUE_HEADS, FULL_NUM_KEY_VALUE_HEADS],
                dim=1,
            )
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, HEAD_DIM),
                weight=std_arg_dict["q_rms_wight"],
                eps=RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_ATTENTION_HEADS, HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, HEAD_DIM),
                weight=std_arg_dict["k_rms_wight"],
                eps=RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, HEAD_DIM)
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["append_pos"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["kv_indices"],
                kv_indptr=std_arg_dict["kv_indptr"],
                kv_last_page_len=std_arg_dict["kv_last_page_len"],
                kv_layout="HND",
            )
            o, lse = wrapper.run_return_lse(
                q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, HEAD_DIM), std_arg_dict["kv_cache"]
            )
            hidden_state_attn_mlp = torch.matmul(
                o.reshape(batch_size, FULL_NUM_ATTENTION_HEADS * HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=RMS_NORM_EPS,
                enable_pdl=False,
            )
            out_gate_up_proj = torch.matmul(hidden_state_attn_mlp, std_arg_dict["gate_up_weight"].T)
            out_silu_multiply = flashinfer.activation.silu_and_mul(
                input=out_gate_up_proj,
            )
            output = torch.matmul(out_silu_multiply, std_arg_dict["down_weight"].T)
            flashinfer.norm.fused_add_rmsnorm(
                input=output,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["mlp_add_rms_weight"],
                eps=AddRMSNormTile.EPS,
                enable_pdl=False,
            )
            return output.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()
            # return hidden_state_attn_mlp.cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"std time: {ms:.3f} ms")
        return output

    # with ProtonContext("blackwell_attn"):
    output_tir_static, residual_tir_static = tir_static(arg_dict)
    print("static tir pass", flush=True)
    output_tir_dynamic, residual_tir_dynamic = tir_dynamic(arg_dict)
    print("dynamic tir pass", flush=True)
    output_std, residual_std = std(arg_dict)

    try:
        np.testing.assert_allclose(output_tir_static, output_std, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(residual_tir_static, residual_std, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(output_tir_dynamic, output_std, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(residual_tir_dynamic, residual_std, rtol=1e-3, atol=1e-2)
    except Exception as e:
        print(e)



if __name__ == "__main__":

    mega_kernel_wrapper_static = MegaKernel(problem_config, profiler_on=PROFILER_ON)
    mega_kernel_wrapper_dynamic = MegaKernel(problem_config, profiler_on=PROFILER_ON)
    mega_kernel_static = mega_kernel_wrapper_static.get_func_static()
    mega_kernel_dynamic = mega_kernel_wrapper_dynamic.get_func_dynamic()
    src, mod_static = get_source(mega_kernel_static)
    src, mod_dynamic = get_source(mega_kernel_dynamic)
    devices = list(np.arange(WORLD_SIZE))
    sess = di.ProcessSession(num_workers=WORLD_SIZE)
    sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, WORLD_SIZE, 0)
    sess.sync_worker_0()

    for batch_size in [1, 3, 5, 7, 15, 31, 63, 127]:

        print(f"batch_size: {batch_size}", flush=True)
        test(batch_size, mod_static, mod_dynamic, sess)

    sess.shutdown()