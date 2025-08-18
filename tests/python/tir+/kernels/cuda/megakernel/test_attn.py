from enum import Enum
import numpy as np
import pytest
import tvm
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.script.ir_builder import IRBuilder
import tvm.testing
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.static_scheduler import StaticTileScheduler, JobType, Semaphore
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.rms_norm import RMSnormTile
from tvm.tirp.megakernel.rope import RopeTile
from tvm.tirp.megakernel.append_kv import AppendKVTile
from tvm.tirp.megakernel.batch_decode import DecodeTile
from tvm.tirp.megakernel.decode_merge import DecodeMergeTile

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
NUM_GROUPS = 1
PROFILER_BUFFER_SIZE = int(1e6)
PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS
PROFILER_ON = True


SPLIT_QKV_PROJECT = 5
SPLIT_O_PROJRCT = 4

class MegaKernel:
    class_list = [GemmTile, SplitKReduceTile,RMSnormTile, RopeTile, AppendKVTile, DecodeTile, DecodeMergeTile]    
    
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

    def get_func(self):

        @T.prim_func(tirp=True)
        def mega_kernel(hidden_state_ptr: T.handle, qkv_proj_ptr: T.handle, partital_qkv_ptr: T.handle, qkv_ptr: T.handle, 
                        rms_weight_ptr: T.handle, cos_sin_cache_ptr: T.handle, kv_cache_ptr: T.handle,
                        kv_indptr_ptr: T.handle, kv_indices_ptr: T.handle, kv_last_page_len_ptr: T.handle,
                        pos_map_ptr: T.handle, o_ptr: T.handle, lse_ptr: T.handle, o_tmp_ptr: T.handle, lse_tmp_ptr: T.handle,
                        o_indptr_ptr: T.handle, request_indices_ptr: T.handle, kv_tile_indices_ptr: T.handle, 
                        max_chunk_size_ptr: T.handle, partial_o_ptr: T.handle, o_proj_ptr: T.handle, hidden_state_out_ptr: T.handle,
                        etensor_qkv_partial_ptr: T.handle, etensor_q_reduce_ptr: T.handle, etensor_k_reduce_ptr: T.handle, 
                        etensor_v_reduce_ptr: T.handle, 
                        etensor_rms_rope_ptr: T.handle, 
                        etensor_q_rope_decode_ptr: T.handle, etensor_k_rope_append_ptr: T.handle, 
                        etensor_append_decode_ptr: T.handle, etensor_decode_merge_ptr: T.handle,
                        etensor_o_proj_ptr: T.handle, etensor_o_partial_ptr: T.handle,
                        exec_queue_ptr: T.handle, profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
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
            etensor_q_rope_decode_global = T.match_buffer(etensor_q_rope_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_k_rope_append_global = T.match_buffer(etensor_k_rope_append_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_append_decode_global = T.match_buffer(etensor_append_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS], 
                                                            "int32", scope="global", layout="default")
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", layout="default")
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)], 
                                                            "int32", scope="global", layout="default") 
            
            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4], 
                                        "int32", scope="global", layout="default")
            
            A_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            

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
            self.tile_list.append(qkv_proj_tile)
            self.tile_list.append(qkv_reduce_tile)
            self.tile_list.append(rmsnorm_tile)
            self.tile_list.append(rope_tile)
            self.tile_list.append(append_kv_tile)
            self.tile_list.append(decode_tile)
            self.tile_list.append(decode_merge_tile)
            self.tile_list.append(o_proj_tile)
            self.tile_list.append(o_reduce_tile)
            qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj)
            o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj)
            self.host_init_all()

            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                if PROFILER_ON:
                    T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 0)
                with T.cta():
                    wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                    tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                    buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    self.class_init_all(pool)
                    self.device_init_all(pool)

                    # initialize event tensors
                    evt_qkv_partial = T.meta_var(Semaphore(SPLIT_QKV_PROJECT * (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), 
                                                             etensor_qkv_partial_global))
                    evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global))
                    evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global))
                    evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global))
                    evt_rms_rope = T.meta_var(Semaphore(1, etensor_rms_rope_global))
                    evt_q_rope_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS, etensor_q_rope_decode_global))
                    evt_k_rope_append = T.meta_var(Semaphore(1, etensor_k_rope_append_global))
                    evt_append_decode = T.meta_var(Semaphore(2, etensor_append_decode_global)) 
                    evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global))
                    evt_o_proj = T.meta_var(Semaphore(batch_size * NUM_KEY_VALUE_HEADS // SPLIT_O_PROJRCT, etensor_o_proj_global))
                    evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global))

                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(StaticTileScheduler(prefix="attn", exec_queue=exec_queue, pool_allocator=pool))
                    tile_scheduler.init(bx, tid)
                    while tile_scheduler.valid():
                        # T.cuda.printf("%d, tile_scheduler.task_type: %d\n", bx, tile_scheduler.task_type)
                        if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_qkv_partial.semaphore_notify(tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
                            evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
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
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.RMSNORM.value:
                            if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile:
                                evt_q_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            else:
                                evt_k_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.RMSNORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_rms_rope.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.RMSNORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.ROPE.value:
                            evt_rms_rope.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rope_tile.h_tile:
                                    evt_q_rope_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS))
                                else:
                                    evt_k_rope_append.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rope_tile.h_tile)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.APPEND_KV.value:
                            if tile_scheduler.k_idx == 0:
                                evt_k_rope_append.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            else:
                                evt_v_reduce.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_append_decode.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
                            evt_q_rope_decode.semaphore_wait(tile_scheduler.m_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            evt_append_decode.semaphore_wait(tile_scheduler.m_idx // append_kv_tile.m_tile, tile_scheduler.n_idx // append_kv_tile.h_tile) # wait for append kv
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_o_proj.semaphore_notify(tile_scheduler.n_idx // (NUM_KEY_VALUE_HEADS // SPLIT_O_PROJRCT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
                            batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
                            evt_q_rope_decode.semaphore_wait(batch_idx // rope_tile.m_tile, tile_scheduler.n_idx // rope_tile.h_tile) # wait for q rope
                            evt_append_decode.semaphore_wait(batch_idx // append_kv_tile.m_tile, tile_scheduler.n_idx // append_kv_tile.h_tile) # wait for append kv
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_decode_merge.semaphore_notify(batch_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
                            evt_decode_merge.semaphore_wait(tile_scheduler.m_idx, tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_o_proj.semaphore_notify(tile_scheduler.n_idx // (NUM_KEY_VALUE_HEADS // SPLIT_O_PROJRCT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                            evt_o_proj.semaphore_wait(tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_o_partial.semaphore_notify(tile_scheduler.n_idx // (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                            evt_o_partial.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)

                        tile_scheduler.next_tile()
                    self.class_finalize_all()

        return mega_kernel


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static):

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

        # # append v to kv cache
        h_tile = ceildiv(AppendKVTile.min_bdy, ceildiv(batch_size, m_split))
        for m_idx in range(m_split):
            for n_idx in range(ceildiv(NUM_KEY_VALUE_HEADS, h_tile)):
                central_queue.append((m_idx, n_idx, 1, JobType.APPEND_KV.value))

        # rmsnorm
        for m_idx in range(m_split):
            for n_idx in range(ceildiv(NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS, h_tile)):
                central_queue.append((m_idx, n_idx, -1, JobType.RMSNORM.value))

        # rope
        for m_idx in range(m_split):
            for n_idx in range(ceildiv(NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS, h_tile)):
                central_queue.append((m_idx, n_idx, -1, JobType.ROPE.value))

        # append k to kv cache
        for m_idx in range(m_split):
            for n_idx in range(ceildiv(NUM_KEY_VALUE_HEADS, h_tile)):
                central_queue.append((m_idx, n_idx, 0, JobType.APPEND_KV.value))

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

    def prepare_data(batch_size):
        import torch
        torch.manual_seed(42)
        arg_dict = {}

        # input
        arg_dict["hidden_state"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)

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

        # profiler
        arg_dict["profiler_buffer"] = torch.zeros((PROFILER_BUFFER_SIZE, ), dtype=torch.uint64)

        return arg_dict

    arg_dict = prepare_data(batch_size)

    def tir(arg_dict):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

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

        # static schedule
        exec_queue = generate_exec_queue(batch_size, tvm_arg_dict["new_batch_size"], tvm_arg_dict["split_kv"])
        exec_queue_tvm = tvm.nd.array(exec_queue, DEV)

        with target:
            def func():

                for key, value in arg_dict.items():
                    tvm_arg_dict[key] = tvm.nd.array(value, device=DEV)

                # generate event tensor
                qkv_h_d = (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM
                qk_h_d = (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) * HEAD_DIM
                q_h_d = NUM_ATTENTION_HEADS * HEAD_DIM
                k_h_d = NUM_KEY_VALUE_HEADS * HEAD_DIM
                tvm_arg_dict["etensor_qkv_partial"] = tvm.nd.array(
                    np.zeros(ceildiv(qkv_h_d, SplitKReduceTile.N_UNIT), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_q_reduce"] = tvm.nd.array(
                    np.zeros((batch_size, ceildiv(q_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_k_reduce"] = tvm.nd.array(
                    np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_v_reduce"] = tvm.nd.array(
                    np.zeros((batch_size, ceildiv(k_h_d, SplitKReduceTile.N_UNIT)), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_rms_rope"] = tvm.nd.array(
                    np.zeros((batch_size, NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_q_rope_decode"] = tvm.nd.array(
                    np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_k_rope_append"] = tvm.nd.array(
                    np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_append_decode"] = tvm.nd.array(
                    np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV
                )
                tvm_arg_dict["etensor_decode_merge"] = tvm.nd.array(np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32), device=DEV)
                etensor_decode_merge = np.zeros((batch_size, NUM_KEY_VALUE_HEADS), dtype=np.int32)
                for i in range(batch_size):
                    num_merge = o_indptr[i + 1] - o_indptr[i]
                    for j in range(NUM_KEY_VALUE_HEADS):
                        etensor_decode_merge[i, j] = num_merge
                tvm_arg_dict["etensor_decode_merge"] = tvm.nd.array(etensor_decode_merge, device=DEV)
                tvm_arg_dict["etensor_o_proj"] = tvm.nd.array(np.zeros(SPLIT_O_PROJRCT, dtype=np.int32), device=DEV)
                o_partial_evt_shape = [ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)]
                tvm_arg_dict["etensor_o_partial"] = tvm.nd.array(np.zeros(o_partial_evt_shape, dtype=np.int32), device=DEV)

                mega_kernel_static(tvm_arg_dict["hidden_state"], tvm_arg_dict["qkv_proj"], tvm_arg_dict["partial_qkv"],
                    tvm_arg_dict["qkv"], tvm_arg_dict["rms_wight"], tvm_arg_dict["cos_sin_cache"], tvm_arg_dict["kv_cache"],
                    tvm_arg_dict["kv_indptr"], tvm_arg_dict["kv_indices"], tvm_arg_dict["kv_last_page_len"], tvm_arg_dict["pos_map"],
                    tvm_arg_dict["o"], tvm_arg_dict["lse"], tvm_arg_dict["o_tmp"], tvm_arg_dict["lse_tmp"], tvm_arg_dict["o_indptr"],
                    tvm_arg_dict["request_indices"], tvm_arg_dict["kv_tile_indices"], tvm_arg_dict["max_chunk_size"], 
                    tvm_arg_dict["partial_o"], tvm_arg_dict["o_proj"], tvm_arg_dict["hidden_state_out"],
                    tvm_arg_dict["etensor_qkv_partial"], tvm_arg_dict["etensor_q_reduce"], tvm_arg_dict["etensor_k_reduce"], 
                    tvm_arg_dict["etensor_v_reduce"], tvm_arg_dict["etensor_rms_rope"], tvm_arg_dict["etensor_q_rope_decode"], 
                    tvm_arg_dict["etensor_k_rope_append"], tvm_arg_dict["etensor_append_decode"], tvm_arg_dict["etensor_decode_merge"],
                    tvm_arg_dict["etensor_o_proj"], tvm_arg_dict["etensor_o_partial"],
                    exec_queue_tvm, tvm_arg_dict["profiler_buffer"])

            ms = bench(func, warmup=10, repeat=30, proton_name="tir")
            print(f"TIR time: {ms:.3f} ms")
            func()
            if PROFILER_ON:
                export_to_perfetto_trace(
                    tvm_arg_dict["profiler_buffer"].numpy(),
                    f"megakernel-attn.perfetto-trace",
                    event_type_names
                )
        return (
            tvm_arg_dict["qkv"].numpy()[:, :NUM_ATTENTION_HEADS, :],
            tvm_arg_dict["qkv"].numpy()[:, NUM_ATTENTION_HEADS:NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS, :],
            tvm_arg_dict["qkv"].numpy()[:, NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS:, :],
            tvm_arg_dict["o"].numpy(),
        )

    def std(arg_dict):
        import torch
        import flashinfer
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        def func():
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
            return (
                q.cpu().numpy(),
                k.cpu().numpy(),
                v.cpu().numpy(),
                o.cpu().numpy(),
            )

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"std time: {ms:.3f} ms")
        return output

    with ProtonContext("blackwell_attn"):
        q_tir, k_tir, v_tir, o_tir = tir(arg_dict)
        q_std, k_std, v_std, o_std = std(arg_dict)

    np.testing.assert_allclose(q_tir.reshape(batch_size, -1), q_std.reshape(batch_size, -1), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(k_tir.reshape(batch_size, -1), k_std.reshape(batch_size, -1), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(v_tir.reshape(batch_size, -1), v_std.reshape(batch_size, -1), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(o_tir.reshape(batch_size, -1), o_std.reshape(batch_size, -1), rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    def init_tile_class_config():
        AppendKVTile.class_config_init(problem_config)
        RMSnormTile.class_config_init(problem_config)
        RopeTile.class_config_init(problem_config)
        DecodeTile.class_config_init(problem_config)
        DecodeMergeTile.class_config_init(problem_config)

    init_tile_class_config()
    mega_kernel_static = MegaKernel().get_func()
    src, mod_static = get_source(mega_kernel_static)
    for batch_size in [128, 64, 32, 16, 8, 4, 2, 1]:
        print(f"batch_size: {batch_size}", flush=True)
        test(batch_size, mod_static)
