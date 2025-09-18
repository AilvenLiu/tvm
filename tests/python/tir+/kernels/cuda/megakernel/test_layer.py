import math

import flashinfer
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile
from tvm.tirp.megakernel.allreduce import AllreduceTile
from tvm.tirp.megakernel.append_kv import AppendKVTile
from tvm.tirp.megakernel.batch_decode import DecodeTile
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.decode_merge import DecodeMergeTile
from tvm.tirp.megakernel.batch_attn import BatchAttnTile
from tvm.tirp.megakernel.batch_merge import BatchMergeTile
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.rms_norm import RMSnormTile
from tvm.tirp.megakernel.rope import RopeTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile
from tvm.tirp.megakernel.reduce_rms_norm_rope_q import SplitKReduceRMSnormRopeQTile
from tvm.tirp.megakernel.reduce_rms_norm_rope_append_k import SplitKReduceRMSnormRopeAppendKTile
from tvm.tirp.megakernel.reduce_append_v import SplitKReduceAppendVTile
from tvm.tirp.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirp.megakernel import static_scheduler
from tvm.tirp.megakernel.support import generate_event_tensor, generate_exec_queue

# TODO: fix abnormal slowness of batch-attn on the first tile


class MegaKernel:

    # model configs
    VOCAB_SIZE = 151936
    MAX_POSITION_EMBEDDINGS = 40960
    HIDDEN_SIZE = 5120
    INTERMEDIATE_SIZE_TP1 = 25600
    NUM_HIDDEN_LAYERS = 64
    NUM_ATTENTION_HEADS_TP1 = 64
    NUM_KEY_VALUE_HEADS_TP1 = 8
    HEAD_DIM = 128
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 1000000
    MAX_PAGE_NUM = 8192
    PAGE_SIZE = 16
    SEQ_LEN = 511

    # FIXME: the config for TP 4 is not tuned
    SPLIT_QKV_PROJECT_DICT = {1: 3, 4: 4, 8: 4}
    SPLIT_O_PROJRCT_DICT = {1: 3, 4: 2, 8: 2}
    GATE_UP_PROJ_SPLIT_K_FACTOR_DICT = {1: 1, 4: 2, 8: 2}
    DOWN_PROJ_SPLIT_K_FACTOR_DICT = {1: 10, 4: 3, 8: 3}
    NUM_TASK_ARGS = 10
    MAX_TOTAL_NUM_WORKERS = 65536
    MAX_NUM_KV_SPLITS = 4 * KernelConfig.SM_NUMBER * 2 * (128 + 16)

    NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
    PROFILER_BUFFER_SIZE = int(1e7)
    PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS

    def __init__(self, world_size, profiler_on):
        self.world_size = world_size
        self.INTERMEDIATE_SIZE = self.INTERMEDIATE_SIZE_TP1 // world_size
        self.NUM_ATTENTION_HEADS = self.NUM_ATTENTION_HEADS_TP1 // world_size
        self.NUM_KEY_VALUE_HEADS = self.NUM_KEY_VALUE_HEADS_TP1 // world_size
        self.SPLIT_QKV_PROJECT = self.SPLIT_QKV_PROJECT_DICT[world_size]
        self.SPLIT_O_PROJRCT = self.SPLIT_O_PROJRCT_DICT[world_size]
        self.GATE_UP_PROJ_SPLIT_K_FACTOR = self.GATE_UP_PROJ_SPLIT_K_FACTOR_DICT[world_size]
        self.DOWN_PROJ_SPLIT_K_FACTOR = self.DOWN_PROJ_SPLIT_K_FACTOR_DICT[world_size]
        self.tile_attr = {}
        self.class_list = set()
        self.profiler_on = profiler_on

    def set_tiles(self, batch_size, BLK_M):
        self.tile_attr = {}
        self.class_list = set()
        self.qkv_proj_tile = self._add_tile(GemmTile((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, self.HIDDEN_SIZE,
                                                "float16", "float16", self.SPLIT_QKV_PROJECT, BLK_M, BLK_M), ProfileEventType.GEMM_QKV_PROJ)
        self.qkv_reduce_tile = self._add_tile(SplitKReduceTile(batch_size,(self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, 
                                                        "float16", self.SPLIT_QKV_PROJECT), ProfileEventType.GEMM_QKV_REDUCE)
        self.attn_tile = self._add_tile(BatchAttnTile(self.PAGE_SIZE, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM), ProfileEventType.BATCH_ATTENTION)
        self.merge_tile = self._add_tile(BatchMergeTile(self.HEAD_DIM, self.NUM_KEY_VALUE_HEADS, self.NUM_ATTENTION_HEADS), ProfileEventType.BATCH_ATTENTION_MERGE)
        self.o_proj_tile = self._add_tile(GemmTile(self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM, "float16", "float16", self.SPLIT_O_PROJRCT, BLK_M, BLK_M, prefetch_on=True), ProfileEventType.GEMM_O_PROJ)
        self.o_allreduce_tile = self._add_tile(AllreduceTile(self.world_size), ProfileEventType.O_ALLREDUCE)
        self.o_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.SPLIT_O_PROJRCT), ProfileEventType.GEMM_O_REDUCE)
        self.attn_add_rms_tile = self._add_tile(AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE), ProfileEventType.ATTN_ADD_RMS_NORM)
        self.gemm_gate_up_proj_tile = self._add_tile(GemmTile(self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE, "float16", "float16", self.GATE_UP_PROJ_SPLIT_K_FACTOR, BLK_M, BLK_M, prefetch_on=True), ProfileEventType.GEMM_GATE_UP_PROJ)
        self.silu_multiply_tile = self._add_tile(SiluMultiplyTile(batch_size, self.INTERMEDIATE_SIZE, "float16"), ProfileEventType.SPLIT_SILU_MULTIPLY)
        self.gemm_down_proj_tile = self._add_tile(GemmTile(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, "float16", "float16", self.DOWN_PROJ_SPLIT_K_FACTOR, BLK_M, BLK_M, prefetch_on=True), ProfileEventType.GEMM_DOWN_PROJ)
        self.gemm_gate_up_proj_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.INTERMEDIATE_SIZE * 2, "float16", self.GATE_UP_PROJ_SPLIT_K_FACTOR), ProfileEventType.GATE_UP_PROJ_REDUCE)
        self.down_proj_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.DOWN_PROJ_SPLIT_K_FACTOR), ProfileEventType.DOWN_PROJ_REDUCE)
        self.down_proj_allreduce_tile = self._add_tile(AllreduceTile(self.world_size), ProfileEventType.DOWN_PROJ_ALLREDUCE)
        self.mlp_add_rms_norm_tile = self._add_tile(AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE), ProfileEventType.MLP_ADD_RMS_NORM)
        self.reduce_rms_rope_q_tile = self._add_tile(SplitKReduceRMSnormRopeQTile(batch_size, self.RMS_NORM_EPS, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT), ProfileEventType.Q_REDUCE_RMSNORM_ROPE)
        self.reduce_rms_rope_append_k_tile = self._add_tile(SplitKReduceRMSnormRopeAppendKTile(batch_size, self.RMS_NORM_EPS, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT, self.PAGE_SIZE), ProfileEventType.K_REDUCE_RMSNORM_ROPE_APPEND)
        self.reduce_append_v_tile = self._add_tile(SplitKReduceAppendVTile(batch_size, self.NUM_KEY_VALUE_HEADS, self.NUM_ATTENTION_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT, self.PAGE_SIZE), ProfileEventType.V_REDUCE_APPEND)

    def _init_profiler(self, profiler_buffer, profiler_tag, profiler_write_offset):
        self.profiler_buffer = profiler_buffer
        self.profiler_tag = profiler_tag
        self.profiler_write_offset = profiler_write_offset

    @T.macro
    def init_profiler(self, profiler_buffer, profiler_tag, profiler_write_offset):
        self._init_profiler(profiler_buffer, profiler_tag, profiler_write_offset)
        with T.cta():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            if self.profiler_on:
                T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, self.NUM_GROUPS, warp_id)

    @T.macro
    def run_tile(self, tile, *args):
        event_type = T.meta_var(self.tile_attr[tile])
        with T.cta():
            lane_id = T.thread_id([32], parent="warp")
            if self.profiler_on:
                T.timer_start_cuda(event_type, self.profiler_buffer.data, self.profiler_tag.data, self.profiler_write_offset.data, self.PROFILER_WRITE_STRIDE, lane_id == 0)
            tile.run(*args)
            if self.profiler_on:
                T.timer_end_cuda(event_type, self.profiler_buffer.data, self.profiler_tag.data, self.profiler_write_offset.data, self.PROFILER_WRITE_STRIDE, lane_id == 0)

    @T.macro
    def run_tile_prefetch(self, tile, *args):
        with T.cta():
            lane_id = T.thread_id([32], parent="warp")
            if self.profiler_on:
                T.timer_start_cuda(ProfileEventType.PREFETCH_SMEM, self.profiler_buffer.data, self.profiler_tag.data, self.profiler_write_offset.data, self.PROFILER_WRITE_STRIDE, lane_id == 0)
            tile.prefetch(*args)
            if self.profiler_on:
                T.timer_end_cuda(ProfileEventType.PREFETCH_SMEM, self.profiler_buffer.data, self.profiler_tag.data, self.profiler_write_offset.data, self.PROFILER_WRITE_STRIDE, lane_id == 0)

    def _add_tile(self, tile, profiler_event_type):
        self.tile_attr[tile] = profiler_event_type
        self.class_list.add(tile.__class__)
        return tile

    def host_init_all(self):
        for tile in self.tile_attr.keys():
            tile.host_init()

    def class_init_all(self, smem_manager: SmemManager):
        for cls in self.class_list:
            smem_manager.set_tile(cls)
            cls.class_init(smem_manager)

    def class_finalize_all(self):
        for cls in self.class_list:
            cls.class_finalize()

    def device_init_all(self, smem_manager: SmemManager):
        offset = smem_manager.pool_allocator.offset
        max_offset = 0
        for tile in self.tile_attr.keys():
            smem_manager.pool_allocator.move_base_to(offset)
            smem_manager.set_tile(tile)
            tile.init(smem_manager)
            max_offset = max(max_offset, smem_manager.pool_allocator.offset)
        smem_manager.pool_allocator.move_base_to(max_offset)

    # fmt: off
    @T.macro
    def static_fused_body(
        self,
        batch_size,
        hidden_state_global,
        residual_global,
        output_global,
        qkv_proj_weight_global,
        o_proj_weight_global,
        q_rms_weight_global,
        k_rms_weight_global,
        gate_up_weight_global,
        down_weight_global,
        attn_add_rms_weight_global,
        mlp_add_rms_weight_global,
        cos_sin_cache_global,
        rope_pos_global,
        kv_cache_global,
        append_pos_global,
        q_indptr_global,
        kv_indptr_global,
        partial_indptr_global,
        kv_indices_global,
        q_len_global,
        kv_len_global,
        q_start_global,
        kv_start_global,
        kv_end_global,
        kv_head_idx_global,
        work_indptr_global,
        len_kv_chunk_global,
        num_qo_len_global,
        merge_indptr_global,
        merge_o_indices_global,
        partital_qkv_global,
        qkv_global,
        o_global,
        o_partial_attn_global,
        lse_partial_attn_global,
        partial_o_global,
        before_o_allreduce_global,
        hidden_state_attn_mlp_global,
        partial_out_gate_up_proj_global,
        out_gate_up_proj_global,
        out_silu_multiply_global,
        partial_sum_down_proj_global,
        before_down_proj_allreduce_global,
        etensor_qkv_partial_global,
        etensor_attn_global,
        etensor_attn_merge_global,
        etensor_o_proj_global,
        etensor_o_partial_global,
        etensor_o_allreduce_global,
        etensor_attn_add_rms_global,
        etensor_attn_mlp_global,
        etensor_gate_up_proj_reduce_global,
        etensor_gate_up_proj_global,
        etensor_down_proj_global,
        etensor_down_proj_reduce_global,
        etensor_down_proj_allreduce_global,
        etensor_mlp_add_rms_global,
        profiler_buffer,
        exec_queue,
        BLK_M,
    ):
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
        self.set_tiles(batch_size, BLK_M)
        
        self.qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj, hidden_state_global, qkv_proj_weight_global, partital_qkv_global)
        self.o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj, o_global.view(-1,self.NUM_ATTENTION_HEADS * self.HEAD_DIM).buffer, o_proj_weight_global, partial_o_global)
        self.gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj, hidden_state_attn_mlp_global, gate_up_weight_global, out_gate_up_proj_global if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1 else partial_out_gate_up_proj_global)
        self.gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj, out_silu_multiply_global, down_weight_global, partial_sum_down_proj_global)

        self.host_init_all()

        with T.kernel():
            bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
            profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
            self.init_profiler(profiler_buffer, profiler_tag, profiler_write_offset)
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                pool = T.meta_var(Tp.PoolAllocator(buf.data))
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, pool))
                self.device_init_all(smem_manager)
                self.class_init_all(smem_manager)

                # initialize event tensors
                evt_qkv_partial = T.meta_var(static_scheduler.Semaphore(self.SPLIT_QKV_PROJECT * (self.qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), 
                                                        etensor_qkv_partial_global, use_nvshmem=self.world_size > 1))
                evt_attn = T.meta_var(static_scheduler.Semaphore(self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS + 2, etensor_attn_global, use_nvshmem=self.world_size > 1))
                evt_attn_merge = T.meta_var(static_scheduler.Semaphore(-1, etensor_attn_merge_global, use_nvshmem=self.world_size > 1))
                evt_o_proj = T.meta_var(static_scheduler.Semaphore(-1, etensor_o_proj_global, use_nvshmem=self.world_size > 1))
                evt_o_partial = T.meta_var(static_scheduler.Semaphore(self.SPLIT_O_PROJRCT * (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global, use_nvshmem=self.world_size > 1))
                evt_o_allreduce = T.meta_var(static_scheduler.Semaphore(self.world_size * self.o_reduce_tile.M_split, etensor_o_allreduce_global, use_nvshmem=self.world_size > 1))
                evt_attn_add_rms = T.meta_var(static_scheduler.Semaphore(ceildiv(self.HIDDEN_SIZE, (self.o_reduce_tile.N_TILE if self.world_size == 1 else self.o_allreduce_tile.N_TILE)), etensor_attn_add_rms_global, use_nvshmem=self.world_size > 1))
                evt_attn_mlp = T.meta_var(static_scheduler.Semaphore(batch_size, etensor_attn_mlp_global, use_nvshmem=self.world_size > 1))
                evt_gate_up_proj_reduce = T.meta_var(static_scheduler.Semaphore(self.GATE_UP_PROJ_SPLIT_K_FACTOR, etensor_gate_up_proj_reduce_global, use_nvshmem=self.world_size > 1))
                evt_gate_up_proj = T.meta_var(static_scheduler.Semaphore(2 if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1 else 2 * self.gemm_gate_up_proj_reduce_tile.M_split, etensor_gate_up_proj_global, use_nvshmem=self.world_size > 1))
                evt_down_proj = T.meta_var(static_scheduler.Semaphore(-1, etensor_down_proj_global, use_nvshmem=self.world_size > 1))
                evt_down_proj_reduce = T.meta_var(static_scheduler.Semaphore(self.DOWN_PROJ_SPLIT_K_FACTOR * (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 
                                                            etensor_down_proj_reduce_global, use_nvshmem=self.world_size > 1))
                evt_down_proj_allreduce = T.meta_var(static_scheduler.Semaphore(self.world_size * self.down_proj_reduce_tile.M_split, etensor_down_proj_allreduce_global, use_nvshmem=self.world_size > 1))
                evt_add_rms_norm = T.meta_var(static_scheduler.Semaphore(self.HIDDEN_SIZE // (self.down_proj_reduce_tile.N_TILE if self.world_size == 1 else self.down_proj_allreduce_tile.N_TILE), etensor_mlp_add_rms_global, use_nvshmem=self.world_size > 1))

                # initialize tile scheduler
                tile_scheduler = T.meta_var(StaticTileScheduler(prefix="attn", exec_queue=exec_queue, pool_allocator=pool))
                tile_scheduler.init(bx, tid)
                smem_manager.init()

                while tile_scheduler.valid():
                    if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                        self.run_tile(self.qkv_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        if wg_id == 0:
                            T.ptx.bar.sync(1, 128)
                            if tid == 0:
                                evt_qkv_partial.semaphore_notify(tile_scheduler.n_idx // (self.qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                    elif tile_scheduler.task_type == JobType.Q_REDUCE_RMS_ROPE.value:
                        evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx)
                        self.run_tile(self.reduce_rms_rope_q_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partital_qkv_global, qkv_global, q_rms_weight_global, rope_pos_global, cos_sin_cache_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            evt_attn.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                    elif tile_scheduler.task_type == JobType.K_REDUCE_RMS_ROPE_APPEND.value:
                        evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx + self.NUM_ATTENTION_HEADS // SplitKReduceRMSnormRopeAppendKTile.h_tile)
                        self.run_tile(self.reduce_rms_rope_append_k_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partital_qkv_global, k_rms_weight_global, rope_pos_global, cos_sin_cache_global, append_pos_global, kv_cache_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            evt_attn.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                    elif tile_scheduler.task_type == JobType.V_REDUCE_APPEND.value:
                        evt_qkv_partial.semaphore_wait(tile_scheduler.n_idx + (self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS) // SplitKReduceAppendVTile.H_TILE)
                        self.run_tile(self.reduce_append_v_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partital_qkv_global, kv_cache_global, append_pos_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            evt_attn.semaphore_notify(tile_scheduler.m_idx, tile_scheduler.n_idx)
                    elif tile_scheduler.task_type == JobType.BATCH_ATTENTION.value:
                        worker_idx = tile_scheduler.m_idx * KernelConfig.WG_NUMBER + wg_id
                        attn_task_num = work_indptr_global[KernelConfig.SM_NUMBER * KernelConfig.WG_NUMBER]  
                        batch_idx = T.meta_var(q_indptr_global[worker_idx])
                        kv_idx = T.meta_var(kv_head_idx_global[worker_idx])

                        # TODO: Now sync cta for simple, need to finegrain in the future
                        if warp_id == 0 and wg_id == 0:
                            smem_manager.wait_all(lane_id)
                        T.tvm_storage_sync("shared")   

                        if worker_idx < attn_task_num:
                            self.run_tile_prefetch(self.attn_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, qkv_global, kv_cache_global, q_indptr_global, kv_indptr_global, partial_indptr_global,
                                                kv_indices_global, q_len_global, kv_len_global, q_start_global, kv_start_global,
                                                kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global,
                                                o_global, o_partial_attn_global, lse_partial_attn_global, profiler_buffer, profiler_tag, profiler_write_offset)
                            evt_attn.semaphore_wait_warp(batch_idx // self.reduce_rms_rope_q_tile.m_tile, kv_idx)
                            self.run_tile(self.attn_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, qkv_global, kv_cache_global, q_indptr_global, kv_indptr_global, partial_indptr_global,
                                            kv_indices_global, q_len_global, kv_len_global, q_start_global, kv_start_global,
                                            kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global,
                                            o_global, o_partial_attn_global, lse_partial_attn_global, profiler_buffer, profiler_tag, profiler_write_offset)

                            if work_indptr_global[KernelConfig.SM_NUMBER * KernelConfig.WG_NUMBER] > batch_size * self.NUM_KEY_VALUE_HEADS:
                                if tid % (KernelConfig.WARP_NUMBER * 32) == 0:
                                    evt_attn_merge.semaphore_notify(batch_idx, kv_idx)
                            else:
                                range_start = T.meta_var(kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM // self.o_proj_tile.TILE_K)
                                range_end = T.meta_var(((kv_idx + 1) * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM - 1) // self.o_proj_tile.TILE_K)
                                if tid % (KernelConfig.WARP_NUMBER * 32) <= range_end - range_start:
                                    evt_o_proj.semaphore_notify(range_start + tid % (KernelConfig.WARP_NUMBER * 32))
                                    # TODO: Now sync cta for simple, need to finegrain in the future
                        T.tvm_storage_sync("shared")
                        if warp_id == 0 and wg_id == 0:
                            smem_manager.arrive_all(lane_id)
                        smem_manager.advance()

                    elif tile_scheduler.task_type == JobType.BATCH_ATTENTION_MERGE.value:
                        worker_id = T.meta_var(tile_scheduler.m_idx * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER)
                        qo_idx = T.meta_var(worker_id % (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                        kv_idx = T.meta_var(worker_id // (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)))
                        batch_idx = T.meta_var((worker_id % (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))) // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                        evt_attn_merge.semaphore_wait(batch_idx, kv_idx)
                        self.run_tile(self.merge_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, o_partial_attn_global, o_global, lse_partial_attn_global, num_qo_len_global,
                                            merge_indptr_global, merge_o_indices_global)
                        range_start = T.meta_var((kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) + qo_idx) * self.HEAD_DIM // self.o_proj_tile.TILE_K)
                        range_end = T.meta_var(((kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS) + qo_idx + KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER) * self.HEAD_DIM - 1) // self.o_proj_tile.TILE_K)
                        T.tvm_storage_sync("shared")
                        if tid <= range_end - range_start:
                            evt_o_proj.semaphore_notify(range_start + tid)
                    elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                        self.run_tile_prefetch(self.o_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        evt_o_proj.semaphore_wait_warp(tile_scheduler.k_idx)
                        self.run_tile(self.o_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        if wg_id == 0:
                            T.ptx.bar.sync(1, 128)
                            if tid == 0:
                                evt_o_partial.semaphore_notify(tile_scheduler.n_idx // (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT))
                    elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                        evt_o_partial.semaphore_wait(tile_scheduler.n_idx)
                        if self.world_size == 1:
                            self.run_tile(self.o_reduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partial_o_global, hidden_state_attn_mlp_global)
                        else:
                            self.run_tile(self.o_reduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partial_o_global, before_o_allreduce_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            if self.world_size == 1:
                                evt_attn_add_rms.semaphore_notify(tile_scheduler.m_idx)
                            else:
                                evt_o_allreduce.semaphore_notify(tile_scheduler.n_idx // self.world_size, rank=tile_scheduler.n_idx % self.world_size)
                    elif self.world_size > 1 and tile_scheduler.task_type == JobType.O_ALLREDUCE.value:
                            evt_o_allreduce.semaphore_wait(tile_scheduler.n_idx)
                            # T.cuda.nano_sleep(1000 * 10)
                            self.run_tile(self.o_allreduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, before_o_allreduce_global, hidden_state_attn_mlp_global)
                            T.tvm_storage_sync("shared")
                            if tid < self.world_size:    
                                evt_attn_add_rms.semaphore_notify(tile_scheduler.m_idx, rank=tid)
                    elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                        if self.world_size == 1:
                            evt_attn_add_rms.semaphore_wait(tile_scheduler.m_idx // self.o_reduce_tile.M_TILE)
                        else:
                            evt_attn_add_rms.semaphore_wait(tile_scheduler.m_idx // self.o_allreduce_tile.M_TILE)
                        self.run_tile(self.attn_add_rms_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            evt_attn_mlp.semaphore_notify(0)
                    elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                        self.run_tile_prefetch(self.gemm_gate_up_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        evt_attn_mlp.semaphore_wait_warp(0)
                        self.run_tile(self.gemm_gate_up_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        if wg_id == 0:
                            T.ptx.bar.sync(1, 128)
                            if tid == 0:
                                if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
                                    if tile_scheduler.n_idx >= self.INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx - self.INTERMEDIATE_SIZE // GemmTile.BLK_N)
                                    else:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx)
                                else:
                                    evt_gate_up_proj_reduce.semaphore_notify(tile_scheduler.n_idx)
                    elif self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1 and tile_scheduler.task_type == JobType.GATE_UP_PROJ_REDUCE.value:
                            evt_gate_up_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                            self.run_tile(self.gemm_gate_up_proj_reduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partial_out_gate_up_proj_global, out_gate_up_proj_global)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                if tile_scheduler.n_idx >= self.INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                    evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx - self.INTERMEDIATE_SIZE // GemmTile.BLK_N)
                                else:
                                    evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx)
                    elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                        evt_gate_up_proj.semaphore_wait_warp(tile_scheduler.m_idx)
                        self.run_tile(self.silu_multiply_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, out_gate_up_proj_global, out_silu_multiply_global, tile_scheduler)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            range_start = T.meta_var(tile_scheduler.m_idx * SiluMultiplyTile.TILE_SIZE // self.gemm_down_proj_tile.TILE_K)
                            range_end = T.meta_var(((tile_scheduler.m_idx + 1) * SiluMultiplyTile.TILE_SIZE - 1) // self.gemm_down_proj_tile.TILE_K)
                            for i in T.serial(0, range_end + 1 - range_start):
                                evt_down_proj.semaphore_notify(i + range_start)
                    elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                        self.run_tile_prefetch(self.gemm_down_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        evt_down_proj.semaphore_wait_warp(tile_scheduler.k_idx)
                        self.run_tile(self.gemm_down_proj_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, profiler_buffer, profiler_tag, profiler_write_offset)
                        if wg_id == 0:
                            T.ptx.bar.sync(1, 128)
                            if tid == 0:
                                evt_down_proj_reduce.semaphore_notify(tile_scheduler.n_idx // (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N))
                    elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                        evt_down_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                        if self.world_size == 1:
                            self.run_tile(self.down_proj_reduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partial_sum_down_proj_global, output_global)
                        else:
                            self.run_tile(self.down_proj_reduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, partial_sum_down_proj_global, before_down_proj_allreduce_global)
                        T.tvm_storage_sync("shared")
                        if tid == 0:
                            if self.world_size == 1:
                                evt_add_rms_norm.semaphore_notify(tile_scheduler.m_idx)
                            else:
                                evt_down_proj_allreduce.semaphore_notify(tile_scheduler.n_idx // self.world_size, rank=tile_scheduler.n_idx % self.world_size)
                    elif self.world_size > 1 and tile_scheduler.task_type == JobType.DOWN_PROJ_ALLREDUCE.value:
                        evt_down_proj_allreduce.semaphore_wait(tile_scheduler.n_idx)
                        self.run_tile(self.down_proj_allreduce_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, before_down_proj_allreduce_global, output_global)
                        T.tvm_storage_sync("shared")
                        if tid < self.world_size:
                            evt_add_rms_norm.semaphore_notify(tile_scheduler.m_idx, rank=tid)
                    elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                        if self.world_size == 1:
                            evt_add_rms_norm.semaphore_wait(tile_scheduler.m_idx // self.down_proj_reduce_tile.M_TILE)
                        else:
                            evt_add_rms_norm.semaphore_wait(tile_scheduler.m_idx // self.down_proj_allreduce_tile.M_TILE)
                        self.run_tile(self.mlp_add_rms_norm_tile, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, output_global, residual_global, mlp_add_rms_weight_global)

                    tile_scheduler.next_tile()

                if self.profiler_on:
                    T.timer_finalize_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, self.PROFILER_WRITE_STRIDE, lane_id == 0)

                self.class_finalize_all()

    # fmt: on

    # FIXME: change offset_factor to 0 can make performance better
    #       but it requires change on engine side
    def get_func_static(self):
        # fmt: off
        @T.prim_func(tirp=True)
        def main(
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
            append_pos_ptr: T.handle, # read-only
            q_indptr_ptr : T.handle, # read-only
            kv_indptr_ptr : T.handle, # read-only
            partial_indptr_ptr : T.handle, # read-only
            kv_indices_ptr : T.handle, # read-only
            q_len_ptr : T.handle, # read-only
            kv_len_ptr : T.handle, # read-only
            q_start_ptr : T.handle, # read-only
            kv_start_ptr : T.handle, # read-only
            kv_end_ptr : T.handle, # read-only
            kv_head_idx_ptr : T.handle, # read-only
            work_indptr_ptr : T.handle, # read-only
            len_kv_chunk_ptr : T.handle, # read-only
            num_qo_len_ptr: T.handle, # read-only
            merge_indptr_ptr: T.handle, # read-only
            merge_o_indices_ptr: T.handle, # read-only

            # intermediate buffer
            partital_qkv_ptr: T.handle, # intermediate
            qkv_ptr: T.handle,  # intermediate
            o_ptr: T.handle, # intermediate
            o_partial_attn_ptr: T.handle, # intermediate
            lse_partial_attn_ptr: T.handle, # intermediate
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
            etensor_attn_ptr: T.handle, 
            etensor_attn_merge_ptr: T.handle,
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
            profiler_buffer: T.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            T.func_attr(
                {"global_symbol": "main", "target": T.target("cuda")}
            )

            # match buffer
            batch_size = T.int32()
            cos_sin_cache_len = T.int32()
            max_page_num = T.int32()
            total_page_num = T.int32()

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = T.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")

            # weight
            qkv_proj_weight_global = T.match_buffer(qkv_proj_weight_ptr, [(self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, self.HIDDEN_SIZE], 
                                                    "float16", scope="global")
            o_proj_weight_global = T.match_buffer(o_proj_weight_ptr, [self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM], 
                                                "float16", scope="global")
            q_rms_weight_global = T.match_buffer(q_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global")
            k_rms_weight_global = T.match_buffer(k_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global") 
            gate_up_weight_global = T.match_buffer(gate_up_weight_ptr, [self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], 
                                                "float16", scope="global")
            down_weight_global = T.match_buffer(down_weight_ptr, [self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")
            attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")
            mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")

            # page cache, kv cache and plan info
            cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, self.HEAD_DIM], "float32", scope="global")
            rope_pos_global = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            kv_cache_global = T.match_buffer(kv_cache_ptr, [max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], 
                                            "float16", scope="global")
            append_pos_global = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            q_indptr_global = T.match_buffer(q_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indptr_global = T.match_buffer(kv_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            partial_indptr_global = T.match_buffer(partial_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", offset_factor=1)
            q_len_global = T.match_buffer(q_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_len_global = T.match_buffer(kv_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            q_start_global = T.match_buffer(q_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_start_global = T.match_buffer(kv_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_end_global = T.match_buffer(kv_end_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_head_idx_global = T.match_buffer(kv_head_idx_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            work_indptr_global = T.match_buffer(work_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            len_kv_chunk_global = T.match_buffer(len_kv_chunk_ptr, [2], "int32", offset_factor=1)
            num_qo_len_global = T.match_buffer(num_qo_len_ptr, [1], "int32", offset_factor=1)
            merge_indptr_global = T.match_buffer(merge_indptr_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            merge_o_indices_global = T.match_buffer(merge_o_indices_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            
            # intermediate buffer
            partital_qkv_global = T.match_buffer(partital_qkv_ptr, [self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM], 
                                    "float32", scope="global")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM], 
                                    "float16", scope="global")
            o_global = T.match_buffer(o_ptr, [batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM], 
                                    "float16", scope="global")
            o_partial_attn_global = T.match_buffer(o_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM], 
                                    "float32", scope="global")
            lse_partial_attn_global = T.match_buffer(lse_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS], 
                                    "float32", scope="global")
            partial_o_global = T.match_buffer(partial_o_ptr, [self.SPLIT_O_PROJRCT, batch_size, self.HIDDEN_SIZE], 
                                    "float32", scope="global")
            before_o_allreduce_global = T.match_buffer(before_o_allreduce_ptr, [batch_size, self.HIDDEN_SIZE], 
                                    "float16", scope="global")
            hidden_state_attn_mlp_global = T.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, self.HIDDEN_SIZE], 
                                    "float16", scope="global")
            partial_out_gate_up_proj_global = T.match_buffer(partial_out_gate_up_proj_ptr, [self.GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, self.INTERMEDIATE_SIZE * 2], 
                                    "float32", scope="global")
            out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, self.INTERMEDIATE_SIZE * 2], 
                                "float16", scope="global")
            out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, self.INTERMEDIATE_SIZE], 
                                "float16", scope="global")
            partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [self.DOWN_PROJ_SPLIT_K_FACTOR, batch_size, self.HIDDEN_SIZE], 
                                "float32")
            before_down_proj_allreduce_global = T.match_buffer(before_down_proj_allreduce_ptr, [batch_size, self.HIDDEN_SIZE], 
                                  "float16", scope="global")

            # event tensor
            etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", offset_factor=1)
            etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(self.NUM_ATTENTION_HEADS * self.HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", offset_factor=1)
            etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", offset_factor=1)
            etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM, SplitKReduceTile.N_UNIT)], 
                                    "int32", scope="global", offset_factor=1)
            etensor_attn_global = T.match_buffer(etensor_attn_ptr, [batch_size, self.NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", offset_factor=1)
            etensor_attn_merge_global = T.match_buffer(etensor_attn_merge_ptr, [batch_size, self.NUM_KEY_VALUE_HEADS], 
                                    "int32", scope="global", offset_factor=1)
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [self.SPLIT_O_PROJRCT], "int32", scope="global", offset_factor=1)
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N)], 
                                    "int32", scope="global", offset_factor=1)
            etensor_o_allreduce_global = T.match_buffer(etensor_o_allreduce_ptr, [self.HIDDEN_SIZE // self.world_size // AllreduceTile.N_TILE], "int32", scope="global", offset_factor=1)
            etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", offset_factor=1)
            etensor_gate_up_proj_reduce_global = T.match_buffer(etensor_gate_up_proj_reduce_ptr, [self.INTERMEDIATE_SIZE * 2 // GemmTile.BLK_N], 
                                                    "int32", scope="global", offset_factor=1)
            etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [self.INTERMEDIATE_SIZE // GemmTile.BLK_N], 
                                    "int32", scope="global", offset_factor=1)
            etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [self.DOWN_PROJ_SPLIT_K_FACTOR],
                                    "int32", scope="global", offset_factor=1)
            etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [self.HIDDEN_SIZE // GemmTile.BLK_N],
                                    "int32", scope="global", offset_factor=1)
            etensor_down_proj_allreduce_global = T.match_buffer(etensor_down_proj_allreduce_ptr, [self.HIDDEN_SIZE // self.world_size // AllreduceTile.N_TILE], "int32", scope="global", offset_factor=1)
            etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", offset_factor=1)

            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4], 
                        "int32", scope="global")
            
            @T.macro
            def run(BLK_M):
                self.static_fused_body(batch_size,hidden_state_global, residual_global, output_global, 
                                       qkv_proj_weight_global, o_proj_weight_global, q_rms_weight_global, k_rms_weight_global, gate_up_weight_global, down_weight_global, 
                                       attn_add_rms_weight_global, mlp_add_rms_weight_global, cos_sin_cache_global, rope_pos_global, kv_cache_global, append_pos_global, 
                                       q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global, q_len_global, kv_len_global, q_start_global, 
                                       kv_start_global, kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global, num_qo_len_global, merge_indptr_global, merge_o_indices_global, 
                                       partital_qkv_global, qkv_global, o_global, o_partial_attn_global, lse_partial_attn_global, partial_o_global, before_o_allreduce_global, hidden_state_attn_mlp_global, partial_out_gate_up_proj_global, out_gate_up_proj_global, 
                                       out_silu_multiply_global, partial_sum_down_proj_global, before_down_proj_allreduce_global, etensor_qkv_partial_global, 
                                       etensor_attn_global, etensor_attn_merge_global, etensor_o_proj_global, etensor_o_partial_global, etensor_o_allreduce_global, etensor_attn_add_rms_global, etensor_attn_mlp_global, 
                                       etensor_gate_up_proj_reduce_global, etensor_gate_up_proj_global, etensor_down_proj_global, etensor_down_proj_reduce_global, etensor_down_proj_allreduce_global, etensor_mlp_add_rms_global, profiler_buffer, exec_queue, BLK_M)
                
            if batch_size <= 32:
                run(32)
            elif batch_size <= 64:
                run(64)
            else:
                run(128)
            # fmt: on
        return main

    def get_module_static(self):

        @I.ir_module(tirp=True)
        class StaticModule:

            @T.prim_func(tirp=True)
            def main():
                pass

        module: tvm.IRModule = StaticModule
        module.update_func(module.get_global_var("main"), self.get_func_static())
        return module

    # def get_func_dynamic(self):
    #     from tvm.tirp.megakernel.dynamic_scheduler import Semaphore

    #     # fmt: off
    #     @T.prim_func(tirp=True)
    #     def mega_kernel(
    #         # input and output
    #         hidden_state_ptr: T.handle, # input: read-only
    #         residual_ptr: T.handle, # input & output: inplace update
    #         output_ptr: T.handle, # output

    #         # weight
    #         qkv_proj_weight_ptr: T.handle, # read-only
    #         o_proj_weight_ptr: T.handle, # read-only
    #         q_rms_weight_ptr: T.handle, # read-only
    #         k_rms_weight_ptr: T.handle, # read-only
    #         gate_up_weight_ptr: T.handle, # read-only
    #         down_weight_ptr: T.handle, # read-only
    #         attn_add_rms_weight_ptr: T.handle, # read-only
    #         mlp_add_rms_weight_ptr: T.handle, # read-only

    #         # page cache, cos_sin cache and plan info
    #         cos_sin_cache_ptr: T.handle, # read-only
    #         rope_pos_ptr: T.handle, # read-only
    #         kv_cache_ptr: T.handle, # inplace update
    #         kv_indptr_ptr: T.handle, # read-only
    #         kv_indices_ptr: T.handle, # read-only
    #         kv_last_page_len_ptr: T.handle, # read-only
    #         append_pos_ptr: T.handle, # read-only
    #         request_indices_ptr: T.handle, # read-only
    #         kv_tile_indices_ptr: T.handle, # read-only
    #         max_chunk_size_ptr: T.handle, # read-only
    #         o_indptr_ptr: T.handle, # read-only

    #         # intermediate buffer
    #         partital_qkv_ptr: T.handle, # intermediate
    #         qkv_ptr: T.handle,  # intermediate
    #         o_ptr: T.handle, # intermediate
    #         lse_ptr: T.handle, # intermediate
    #         o_tmp_ptr: T.handle, # intermediate
    #         lse_tmp_ptr: T.handle, # intermediate
    #         partial_o_ptr: T.handle, # intermediate
    #         hidden_state_attn_mlp_ptr: T.handle, # intermediate
    #         out_gate_up_proj_ptr: T.handle, # intermediate
    #         out_silu_multiply_ptr: T.handle, # intermediate
    #         partial_sum_down_proj_ptr: T.handle, # intermediate

    #         # event tensor
    #         etensor_qkv_partial_ptr: T.handle,
    #         etensor_q_reduce_ptr: T.handle,
    #         etensor_k_reduce_ptr: T.handle,
    #         etensor_v_reduce_ptr: T.handle,
    #         etensor_decode_ptr: T.handle,
    #         etensor_decode_merge_ptr: T.handle,
    #         etensor_o_proj_ptr: T.handle,
    #         etensor_o_partial_ptr: T.handle,
    #         etensor_attn_add_rms_ptr: T.handle,
    #         etensor_attn_mlp_ptr: T.handle,
    #         etensor_gate_up_proj_ptr: T.handle,
    #         etensor_down_proj_ptr: T.handle,
    #         etensor_down_proj_reduce_ptr: T.handle,
    #         etensor_mlp_add_rms_ptr: T.handle,
    #         etensor_end: T.Buffer((1, ), "int32", offset_factor=1),

    #         # execution queue
    #         queue_tasks: T.Buffer((DynamicTileScheduler.MAX_TASKS, 4), "int32", offset_factor=1),
    #         queue_head: T.Buffer((1,), "int32", offset_factor=1),
    #         queue_tail: T.Buffer((1,), "int32", offset_factor=1),
    #         profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
    #     ):

    #         # match buffer
    #         batch_size = T.int32()
    #         new_batch_size = T.int32()
    #         cos_sin_cache_len = T.int32()
    #         max_page_num = T.int32()
    #         total_page_num = T.int32()
    #         request_indices_global_elem_offset = T.int32()
    #         kv_tile_indices_global_elem_offset = T.int32()
    #         max_chunk_size_global_elem_offset = T.int32()
    #         o_indptr_global_elem_offset = T.int32()

    #         # input and output
    #         hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global")
    #         residual_global = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], "float16", scope="global")
    #         output_global = T.match_buffer(output_ptr, [batch_size, HIDDEN_SIZE], "float16")

    #         # weight
    #         qkv_proj_weight_global = T.match_buffer(qkv_proj_weight_ptr, [(NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, HIDDEN_SIZE],
    #                                                 "float16", scope="global")
    #         o_proj_weight_global = T.match_buffer(o_proj_weight_ptr, [HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM],
    #                                               "float16", scope="global")
    #         q_rms_weight_global = T.match_buffer(q_rms_weight_ptr, [HEAD_DIM], "float16", scope="global")
    #         k_rms_weight_global = T.match_buffer(k_rms_weight_ptr, [HEAD_DIM], "float16", scope="global")
    #         gate_up_weight_global = T.match_buffer(gate_up_weight_ptr, [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE],
    #                                                "float16", scope="global")
    #         down_weight_global = T.match_buffer(down_weight_ptr, [HIDDEN_SIZE, INTERMEDIATE_SIZE], "float16", scope="global")
    #         attn_add_rms_weight_global = T.match_buffer(attn_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global")
    #         mlp_add_rms_weight_global = T.match_buffer(mlp_add_rms_weight_ptr, [HIDDEN_SIZE], "float16", scope="global")

    #         # page cache, kv cache and plan info
    #         cos_sin_cache_global = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, HEAD_DIM], "float32", scope="global")
    #         rope_pos_global = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
    #         kv_cache_global = T.match_buffer(kv_cache_ptr, [max_page_num, 2, NUM_KEY_VALUE_HEADS, PAGE_SIZE, HEAD_DIM],
    #                                          "float16", scope="global")
    #         kv_indptr_global = T.match_buffer(kv_indptr_ptr, [batch_size + 1], "int32", scope="global", offset_factor=1)
    #         kv_indices_global = T.match_buffer(kv_indices_ptr, [total_page_num], "int32", scope="global", offset_factor=1)
    #         kv_last_page_len_global = T.match_buffer(kv_last_page_len_ptr, [batch_size], "int32", scope="global", offset_factor=1)
    #         append_pos_global = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
    #         request_indices_global = T.match_buffer(request_indices_ptr, [new_batch_size], "int32", scope="global", elem_offset=request_indices_global_elem_offset)
    #         kv_tile_indices_global = T.match_buffer(kv_tile_indices_ptr, [new_batch_size], "int32", scope="global", elem_offset=kv_tile_indices_global_elem_offset)
    #         max_chunk_size_global = T.match_buffer(max_chunk_size_ptr, [1], "int32", scope="global", elem_offset=max_chunk_size_global_elem_offset)
    #         o_indptr_global = T.match_buffer(o_indptr_ptr, [batch_size + 1], "int32", scope="global", elem_offset=o_indptr_global_elem_offset)

    #         # intermediate buffer
    #         partital_qkv_global = T.match_buffer(partital_qkv_ptr, [SPLIT_QKV_PROJECT, batch_size, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM],
    #                                 "float32", scope="global")
    #         qkv_global = T.match_buffer(qkv_ptr, [batch_size, NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS, HEAD_DIM],
    #                                 "float16", scope="global")
    #         o_global = T.match_buffer(o_ptr, [batch_size, NUM_ATTENTION_HEADS, HEAD_DIM],
    #                                 "float16", scope="global")
    #         lse_global = T.match_buffer(lse_ptr, [batch_size, NUM_ATTENTION_HEADS],
    #                                 "float32", scope="global")
    #         o_tmp_global = T.match_buffer(o_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS, HEAD_DIM],
    #                                 "float32", scope="global")
    #         lse_tmp_global = T.match_buffer(lse_tmp_ptr, [new_batch_size, NUM_ATTENTION_HEADS],
    #                                 "float32", scope="global")
    #         partial_o_global = T.match_buffer(partial_o_ptr, [SPLIT_O_PROJRCT, batch_size, HIDDEN_SIZE],
    #                                 "float32", scope="global")
    #         hidden_state_attn_mlp_global = T.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, HIDDEN_SIZE],
    #                                 "float16", scope="global")
    #         out_gate_up_proj_global = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE * 2],
    #                              "float16", scope="global")
    #         out_silu_multiply_global = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE],
    #                               "float16", scope="global")
    #         partial_sum_down_proj_global = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE],
    #                               "float32")

    #         # event tensor
    #         etensor_qkv_partial_global = T.match_buffer(etensor_qkv_partial_ptr, [ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, SplitKReduceTile.N_UNIT)],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_q_reduce_global = T.match_buffer(etensor_q_reduce_ptr, [batch_size, ceildiv(NUM_ATTENTION_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_k_reduce_global = T.match_buffer(etensor_k_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_v_reduce_global = T.match_buffer(etensor_v_reduce_ptr, [batch_size, ceildiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, SplitKReduceTile.N_UNIT)],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_decode_global = T.match_buffer(etensor_decode_ptr, [batch_size, NUM_KEY_VALUE_HEADS],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_decode_merge_global = T.match_buffer(etensor_decode_merge_ptr, [batch_size, NUM_KEY_VALUE_HEADS],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [SPLIT_O_PROJRCT], "int32", scope="global", offset_factor=1)
    #         etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [T.ceildiv(HIDDEN_SIZE, GemmTile.BLK_N)],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_attn_add_rms_global = T.match_buffer(etensor_attn_add_rms_ptr, [batch_size], "int32", scope="global", offset_factor=1)
    #         etensor_attn_mlp_global = T.match_buffer(etensor_attn_mlp_ptr, [1], "int32", scope="global", offset_factor=1)
    #         etensor_gate_up_proj_global = T.match_buffer(etensor_gate_up_proj_ptr, [INTERMEDIATE_SIZE // GemmTile.BLK_N],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_down_proj_global = T.match_buffer(etensor_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_down_proj_reduce_global = T.match_buffer(etensor_down_proj_reduce_ptr, [HIDDEN_SIZE // GemmTile.BLK_N],
    #                                 "int32", scope="global", offset_factor=1)
    #         etensor_mlp_add_rms_global = T.match_buffer(etensor_mlp_add_rms_ptr, [batch_size], "int32", scope="global", offset_factor=1)

    #         A_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         B_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         D_tensor_map_qkv_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         A_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         B_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         D_tensor_map_o_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         A_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         B_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         D_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         A_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         B_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
    #         D_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

    #         # initialize tile
    #         qkv_proj_tile = T.meta_var(GemmTile(hidden_state_global, qkv_proj_weight_global, partital_qkv_global, split_k_factor=SPLIT_QKV_PROJECT))
    #         qkv_reduce_tile = T.meta_var(SplitKReduceTile(partital_qkv_global,
    #                                                       Tp.reshape(qkv_global, [-1, (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM]).buffer))
    #         rmsnorm_tile = T.meta_var(RMSnormTile(q_rms_weight_global, k_rms_weight_global, qkv_global))
    #         rope_tile = T.meta_var(RopeTile(qkv_global, cos_sin_cache_global, rope_pos_global))
    #         append_kv_tile = T.meta_var(AppendKVTile(kv_cache_global, qkv_global, append_pos_global))
    #         decode_tile = T.meta_var(DecodeTile(qkv_global, kv_cache_global, o_global, lse_global, o_tmp_global, lse_tmp_global,
    #                                             kv_indptr_global, kv_last_page_len_global,
    #                                             kv_indices_global, request_indices_global, kv_tile_indices_global, max_chunk_size_global))
    #         decode_merge_tile = T.meta_var(DecodeMergeTile(o_indptr_global, o_tmp_global, o_global, lse_tmp_global, lse_global))
    #         o_proj_tile = T.meta_var(GemmTile(Tp.reshape(o_global, [-1, NUM_ATTENTION_HEADS * HEAD_DIM]).buffer, o_proj_weight_global,
    #                                           partial_o_global, split_k_factor=SPLIT_O_PROJRCT))
    #         o_reduce_tile = T.meta_var(SplitKReduceTile(partial_o_global, hidden_state_attn_mlp_global))
    #         attn_add_rms_tile = T.meta_var(AddRMSNormTile(hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global))

    #         gemm_gate_up_proj_tile = T.meta_var(GemmTile(hidden_state_attn_mlp_global, gate_up_weight_global, out_gate_up_proj_global, split_k_factor=1))
    #         silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj_global, out_silu_multiply_global))
    #         gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply_global, down_weight_global, partial_sum_down_proj_global, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
    #         down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj_global, output_global))
    #         mlp_add_rms_norm_tile = T.meta_var(AddRMSNormTile(output_global, residual_global, mlp_add_rms_weight_global))

    #         self.tile_list.append(qkv_proj_tile)
    #         self.tile_list.append(qkv_reduce_tile)
    #         self.tile_list.append(rmsnorm_tile)
    #         self.tile_list.append(rope_tile)
    #         self.tile_list.append(append_kv_tile)
    #         self.tile_list.append(decode_tile)
    #         self.tile_list.append(decode_merge_tile)
    #         self.tile_list.append(o_proj_tile)
    #         self.tile_list.append(o_reduce_tile)
    #         self.tile_list.append(attn_add_rms_tile)
    #         self.tile_list.append(gemm_gate_up_proj_tile)
    #         self.tile_list.append(silu_multiply_tile)
    #         self.tile_list.append(gemm_down_proj_tile)
    #         self.tile_list.append(down_proj_reduce_tile)
    #         self.tile_list.append(mlp_add_rms_norm_tile)

    #         qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj)
    #         o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj)
    #         gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
    #         gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)

    #         self.host_init_all()

    #         with T.kernel():
    #             bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
    #             warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
    #             lane_id = T.thread_id([32], parent="warp")
    #             profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
    #             profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
    #             if self.profiler_on:
    #                 T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, 0)

    #             with T.cta():
    #                 wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
    #                 tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
    #                 buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
    #                 pool = T.meta_var(Tp.PoolAllocator(buf.data))
    #                 self.device_init_all(pool)
    #                 self.class_init_all(pool)

    #                 # initialize event tensors
    #                 evt_qkv_partial = T.meta_var(Semaphore(SPLIT_QKV_PROJECT * (qkv_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),
    #                                                          etensor_qkv_partial_global))
    #                 evt_q_reduce = T.meta_var(Semaphore(1, etensor_q_reduce_global))
    #                 evt_k_reduce = T.meta_var(Semaphore(1, etensor_k_reduce_global))
    #                 evt_v_reduce = T.meta_var(Semaphore(1, etensor_v_reduce_global))
    #                 evt_decode = T.meta_var(Semaphore(NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS + 2, etensor_decode_global))
    #                 evt_decode_merge = T.meta_var(Semaphore(-1, etensor_decode_merge_global, decrement=True))
    #                 evt_o_proj = T.meta_var(Semaphore(-1, etensor_o_proj_global, decrement=True))
    #                 evt_o_partial = T.meta_var(Semaphore(SPLIT_O_PROJRCT * (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), etensor_o_partial_global))
    #                 evt_attn_add_rms = T.meta_var(Semaphore(ceildiv(HIDDEN_SIZE, o_reduce_tile.N_TILE), etensor_attn_add_rms_global))
    #                 evt_attn_mlp = T.meta_var(Semaphore(batch_size, etensor_attn_mlp_global))
    #                 evt_gate_up_proj = T.meta_var(Semaphore(2, etensor_gate_up_proj_global))
    #                 evt_down_proj = T.meta_var(Semaphore(-1, etensor_down_proj_global, decrement=True))
    #                 evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),
    #                                                             etensor_down_proj_reduce_global))
    #                 evt_add_rms_norm = T.meta_var(Semaphore(HIDDEN_SIZE // down_proj_reduce_tile.N_TILE, etensor_mlp_add_rms_global))
    #                 evt_end = T.meta_var(Semaphore(batch_size, etensor_end))

    #                 # initialize tile scheduler
    #                 tile_scheduler = T.meta_var(DynamicTileScheduler(queue_tasks, queue_head, queue_tail, pool_allocator=pool))
    #                 tile_scheduler.init(warp_id)

    #                 while tile_scheduler.valid():
    #                     if tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         qkv_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_QKV_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.GEMM_QKV_REDUCE.value,
    #                             0,
    #                             tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // GemmTile.BLK_N),
    #                             0,
    #                             qkv_reduce_tile.M_split,
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_qkv_partial,
    #                             tile_scheduler.n_idx // (qkv_reduce_tile.N_TILE // GemmTile.BLK_N),
    #                             use_barrier=False,
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.GEMM_QKV_REDUCE.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         qkv_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_QKV_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         if tile_scheduler.n_idx < NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile:
    #                             tile_scheduler.push_task(
    #                                 JobType.Q_RMSNORM_ROPE.value,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 warp_id,
    #                                 evt_q_reduce,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                             )
    #                         elif tile_scheduler.n_idx < (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile:
    #                             tile_scheduler.push_task(
    #                                 JobType.K_RMSNORM_ROPE_APPEND_KV.value,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile,
    #                                 0,
    #                                 warp_id,
    #                                 evt_k_reduce,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx - NUM_ATTENTION_HEADS // rmsnorm_tile.h_tile,
    #                             )
    #                         else:
    #                             tile_scheduler.push_task(
    #                                 JobType.V_APPEND_KV.value,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile,
    #                                 1,
    #                                 warp_id,
    #                                 evt_v_reduce,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx - (NUM_ATTENTION_HEADS + NUM_KEY_VALUE_HEADS) // rmsnorm_tile.h_tile,
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.Q_RMSNORM_ROPE.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.Q_RMSNORM_ROPE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         if new_batch_size == batch_size: # no split kv
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_NO_SPLIT.value,
    #                                 tile_scheduler.m_idx * append_kv_tile.m_tile,
    #                                 tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
    #                                 0,
    #                                 T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
    #                             )
    #                         else:
    #                             range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
    #                             range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx+1) * append_kv_tile.m_tile, batch_size)])
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_SPLIT.value,
    #                                 range_start,
    #                                 tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
    #                                 0,
    #                                 range_end-range_start,
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS),
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.K_RMSNORM_ROPE_APPEND_KV.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         rmsnorm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rmsnorm_tile.h_tile, tile_scheduler.k_idx)
    #                         rope_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx+NUM_ATTENTION_HEADS//rope_tile.h_tile, tile_scheduler.k_idx)
    #                         append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 0)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.K_RMSNORM_ROPE_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         if batch_size == new_batch_size:
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_NO_SPLIT.value,
    #                                 tile_scheduler.m_idx * append_kv_tile.m_tile,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                             )
    #                         else:
    #                             range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
    #                             range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx+1) * append_kv_tile.m_tile, batch_size)])
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_SPLIT.value,
    #                                 range_start,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 range_end-range_start,
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.V_APPEND_KV.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         append_kv_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, 1)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.V_APPEND_KV, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         if batch_size == new_batch_size:
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_NO_SPLIT.value,
    #                                 tile_scheduler.m_idx * append_kv_tile.m_tile,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 T.min(append_kv_tile.m_tile, batch_size-tile_scheduler.m_idx*append_kv_tile.m_tile),
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                             )
    #                         else:
    #                             range_start = T.meta_var(o_indptr_global[tile_scheduler.m_idx * append_kv_tile.m_tile])
    #                             range_end = T.meta_var(o_indptr_global[T.min((tile_scheduler.m_idx+1) * append_kv_tile.m_tile, batch_size)])
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.BATCH_DECODE_SPLIT.value,
    #                                 range_start,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 range_end-range_start,
    #                                 0,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_decode,
    #                                 tile_scheduler.m_idx,
    #                                 tile_scheduler.n_idx,
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.BATCH_DECODE_NO_SPLIT.value:
    #                         T.tvm_storage_sync("shared")
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=False)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.BATCH_DECODE_NO_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         range_start = T.meta_var(tile_scheduler.n_idx * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM // o_proj_tile.TILE_K)
    #                         range_end = T.meta_var(((tile_scheduler.n_idx + 1) * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) * HEAD_DIM - 1) // o_proj_tile.TILE_K)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.GEMM_O_PROJ.value,
    #                             0,
    #                             0,
    #                             range_start,
    #                             ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
    #                             1,
    #                             warp_id,
    #                             lane_id,
    #                             evt_o_proj,
    #                             range_start,
    #                         )
    #                         for kr in T.serial(range_end - range_start):
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.GEMM_O_PROJ.value,
    #                                 0,
    #                                 0,
    #                                 range_start + 1 + kr,
    #                                 ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
    #                                 1,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_o_proj,
    #                                 range_start + 1 + kr,
    #                                 use_barrier=False
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.BATCH_DECODE_SPLIT.value:
    #                         T.tvm_storage_sync("shared")
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         batch_idx = T.meta_var(request_indices_global[tile_scheduler.m_idx]) # original batch_idx
    #                         decode_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, split_kv=True)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.BATCH_DECODE_SPLIT, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.DECODE_MERGE.value,
    #                             batch_idx,
    #                             tile_scheduler.n_idx * (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) // DecodeMergeTile.bdz,
    #                             0,
    #                             (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS) // DecodeMergeTile.bdz,
    #                             1,
    #                             warp_id,
    #                             lane_id,
    #                             evt_decode_merge,
    #                             batch_idx,
    #                             tile_scheduler.n_idx,
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.DECODE_MERGE.value:
    #                         T.tvm_storage_sync("shared")
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         decode_merge_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.DECODE_MERGE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         range_start = T.meta_var(tile_scheduler.n_idx * DecodeMergeTile.bdz * HEAD_DIM // o_proj_tile.TILE_K)
    #                         range_end = T.meta_var(((tile_scheduler.n_idx + 1) * DecodeMergeTile.bdz * HEAD_DIM - 1) // o_proj_tile.TILE_K)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.GEMM_O_PROJ.value,
    #                             0,
    #                             0,
    #                             range_start,
    #                             ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
    #                             1,
    #                             warp_id,
    #                             lane_id,
    #                             evt_o_proj,
    #                             range_start,
    #                         )
    #                         for kr in T.serial(range_end - range_start):
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.GEMM_O_PROJ.value,
    #                                 0,
    #                                 0,
    #                                 range_start + 1 + kr,
    #                                 ceildiv(HIDDEN_SIZE, GemmTile.BLK_N),
    #                                 1,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_o_proj,
    #                                 range_start + 1 + kr,
    #                                 use_barrier=False,
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         o_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_O_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.GEMM_O_REDUCE.value,
    #                             0,
    #                             tile_scheduler.n_idx,
    #                             0,
    #                             o_reduce_tile.M_split,
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_o_partial,
    #                             tile_scheduler.n_idx // (o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),
    #                             use_barrier=False,
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         o_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_O_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.ATTN_ADD_RMS_NORM.value,
    #                             tile_scheduler.m_idx * o_reduce_tile.M_TILE,
    #                             0,
    #                             0,
    #                             T.min(o_reduce_tile.M_TILE, batch_size-tile_scheduler.m_idx*o_reduce_tile.M_TILE),
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_attn_add_rms,
    #                             tile_scheduler.m_idx,
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         attn_add_rms_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.ATTN_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.GEMM_GATE_UP_PROJ.value,
    #                             0,
    #                             0,
    #                             0,
    #                             ceildiv(INTERMEDIATE_SIZE *2 , GemmTile.BLK_N),
    #                             1,
    #                             warp_id,
    #                             lane_id,
    #                             evt_attn_mlp,
    #                             0,
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
    #                             offset = T.meta_var(tile_scheduler.n_idx-INTERMEDIATE_SIZE//GemmTile.BLK_N)
    #                             tile_scheduler.push_task(
    #                                 JobType.SPLIT_SILU_MULTIPLY.value,
    #                                 offset,
    #                                 0,
    #                                 0,
    #                                 warp_id,
    #                                 evt_gate_up_proj,
    #                                 offset,
    #                                 use_barrier=False
    #                             )
    #                         else:
    #                             tile_scheduler.push_task(
    #                                 JobType.SPLIT_SILU_MULTIPLY.value,
    #                                 tile_scheduler.n_idx,
    #                                 0,
    #                                 0,
    #                                 warp_id,
    #                                 evt_gate_up_proj,
    #                                 tile_scheduler.n_idx,
    #                                 use_barrier=False
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         range_start = T.meta_var(tile_scheduler.m_idx * SiluMultiplyTile.TILE_SIZE // gemm_down_proj_tile.TILE_K)
    #                         range_end = T.meta_var(((tile_scheduler.m_idx + 1) * SiluMultiplyTile.TILE_SIZE - 1) // gemm_down_proj_tile.TILE_K)
    #                         for i in T.serial(0, range_end + 1 - range_start):
    #                             tile_scheduler.push_tasks_along_dim(
    #                                 JobType.GEMM_DOWN_PROJ.value,
    #                                 0,
    #                                 0,
    #                                 range_start + i,
    #                                 HIDDEN_SIZE // GemmTile.BLK_N,
    #                                 1,
    #                                 warp_id,
    #                                 lane_id,
    #                                 evt_down_proj,
    #                                 range_start + i
    #                             )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.DOWN_PROJ_REDUCE.value,
    #                             0,
    #                             tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),
    #                             0,
    #                             down_proj_reduce_tile.M_split,
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_down_proj_reduce,
    #                             tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),
    #                             use_barrier=False
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.MLP_ADD_RMS_NORM.value,
    #                             tile_scheduler.m_idx * down_proj_reduce_tile.M_TILE,
    #                             0,
    #                             0,
    #                             T.min(down_proj_reduce_tile.M_TILE, batch_size-tile_scheduler.m_idx * down_proj_reduce_tile.M_TILE),
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_add_rms_norm,
    #                             tile_scheduler.m_idx
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     elif tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
    #                         T.tvm_storage_sync("shared")
    #                         if self.profiler_on:
    #                             T.timer_start_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         mlp_add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.MLP_ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                             T.timer_start_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                         tile_scheduler.push_tasks_along_dim(
    #                             JobType.END.value,
    #                             0,
    #                             0,
    #                             0,
    #                             KernelConfig.SM_NUMBER,
    #                             0,
    #                             warp_id,
    #                             lane_id,
    #                             evt_end,
    #                             0
    #                         )
    #                         if self.profiler_on:
    #                             T.timer_end_cuda(ProfileEventType.PUSH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     if self.profiler_on:
    #                         T.timer_start_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                     tile_scheduler.next_tile(warp_id)
    #                     if self.profiler_on:
    #                         T.timer_end_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0 or tid == 96 or tid == 224)
    #                 self.class_finalize_all()
    #     # fmt: on

    #     return mega_kernel

    # def generate_exec_queue_dynamic():
    #     exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
    #     for n in reversed(
    #         range(ceildiv((NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM, GemmTile.BLK_N))
    #     ):
    #         for k in range(SPLIT_QKV_PROJECT):
    #             exec_queue.enqueue(JobType.GEMM_QKV_PROJ.value, 0, n, k)
    #     return exec_queue

arg_dict = {}
def prepare_data(batch_size, mk: MegaKernel):
    global arg_dict
    import torch

    def _correct_weight_tensor_view(tensor):
        if mk.world_size == 1:
            return tensor.view(*tensor.shape[1:])
        return tensor

    torch.manual_seed(42)

    # input
    arg_dict["hidden_state"] = torch.randn(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )
    arg_dict["residual"] = torch.randn(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # rms
    arg_dict["q_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)
    arg_dict["k_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)

    # qkv
    arg_dict["qkv"] = torch.randn(
        (
            batch_size,
            mk.NUM_KEY_VALUE_HEADS * 2 + mk.NUM_ATTENTION_HEADS,
            mk.HEAD_DIM,
        ),
        dtype=torch.float16,
    )

    # rope cos_sin_cache
    inv_freq = 1.0 / (
        mk.ROPE_THETA
        ** (
            torch.arange(0, mk.HEAD_DIM, 2, dtype=torch.float, device="cuda")
            / mk.HEAD_DIM
        )
    )
    pos = mk.SEQ_LEN
    assert pos < 4096  # for faster test
    arg_dict["rope_pos"] = torch.full((batch_size,), pos, dtype=torch.int32)
    t = torch.arange(4096, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    arg_dict["cos_sin_cache"] = (
        torch.cat((cos, sin), dim=-1).reshape(-1, mk.HEAD_DIM).cpu()
    )

    # paged kv-cache
    page_last_len = (
        mk.PAGE_SIZE
        if (mk.SEQ_LEN + 1) % mk.PAGE_SIZE == 0
        else (mk.SEQ_LEN + 1) % mk.PAGE_SIZE
    )  # +1 since need to append new kv
    page_num = ceildiv(mk.SEQ_LEN + 1, mk.PAGE_SIZE)
    total_page_num = page_num * batch_size
    assert total_page_num <= mk.MAX_PAGE_NUM
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32).int()
    for i in range(batch_size + 1):
        kv_indptr[i] = i * page_num
    kv_last_page_len = torch.empty(batch_size, dtype=torch.int32).int()
    for i in range(batch_size):
        kv_last_page_len[i] = page_last_len
    kv_indices = torch.arange(mk.MAX_PAGE_NUM, dtype=torch.int32).int()
    kv_indices = kv_indices[torch.randperm(mk.MAX_PAGE_NUM)]
    kv_indices = kv_indices[:total_page_num]
    append_pos = torch.empty(batch_size, dtype=torch.int32).int()
    for i in range(batch_size):
        append_pos[i] = mk.SEQ_LEN
    arg_dict["page_kv_indptr"] = kv_indptr.cpu()
    arg_dict["page_kv_last_page_len"] = kv_last_page_len.cpu()
    arg_dict["page_kv_indices"] = kv_indices.cpu()
    arg_dict["append_pos"] = append_pos.cpu()

    # output
    arg_dict["o"] = torch.zeros(
        (batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM), dtype=torch.float16
    )
    arg_dict["lse"] = torch.zeros(
        (batch_size, mk.NUM_ATTENTION_HEADS), dtype=torch.float32
    )
    arg_dict["hidden_state_attn_mlp"] = torch.zeros(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # add rms
    arg_dict["attn_add_rms_weight"] = torch.randn(
        (mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # mlp
    arg_dict["out_gate_up_proj"] = torch.zeros(
        (batch_size, mk.INTERMEDIATE_SIZE * 2), dtype=torch.float16
    )
    arg_dict["out_silu_multiply"] = torch.zeros(
        (batch_size, mk.INTERMEDIATE_SIZE), dtype=torch.float16
    )
    arg_dict["partial_sum_down_proj"] = torch.zeros(
        (mk.DOWN_PROJ_SPLIT_K_FACTOR, batch_size, mk.HIDDEN_SIZE),
        dtype=torch.float32,
    )
    arg_dict["output"] = torch.zeros((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["mlp_add_rms_weight"] = torch.randn(
        (mk.HIDDEN_SIZE,), dtype=torch.float16
    )

    # plan info
    wrapper = flashinfer.BatchAttention("HND")
    wrapper.plan(
        torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
        arg_dict["page_kv_indptr"].to(0),
        arg_dict["page_kv_indices"].to(0),
        torch.tensor([mk.SEQ_LEN + 1] * batch_size, dtype=torch.int32).to(0),
        mk.NUM_ATTENTION_HEADS,
        mk.NUM_KEY_VALUE_HEADS,
        mk.HEAD_DIM,
        mk.HEAD_DIM,
        mk.PAGE_SIZE,
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

    arg_dict["q_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 2), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 3), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["partial_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 4), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["q_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 5), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 6), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["q_start"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 7), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_start"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 8), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_end"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 9), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["kv_head_idx"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 10), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["work_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 11), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    ).cpu()
    arg_dict["attn_task_num"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 11), mk.MAX_TOTAL_NUM_WORKERS, torch.int32
    )[2 * KernelConfig.SM_NUMBER].cpu()
    arg_dict["len_kv_chunk"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 12), 2, torch.int32
    ).cpu()
    arg_dict["merge_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 15), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["merge_o_indices"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 16), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["num_qo_len"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 17), 1, torch.int32
    ).cpu()

    # weight initialization
    if not hasattr(prepare_data, "weight_initialized"):
        prepare_data.weight_initialized = True
    else:
        return arg_dict
    arg_dict["kv_cache"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            mk.MAX_PAGE_NUM,
            2,
            mk.NUM_KEY_VALUE_HEADS,
            mk.PAGE_SIZE,
            mk.HEAD_DIM,
        ),
        dtype=torch.float16,
    )).cpu()
    arg_dict["qkv_proj_weight"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            (mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS)
            * mk.HEAD_DIM,
            mk.HIDDEN_SIZE,
        ),
        dtype=torch.float16,
    ))
    torch.nn.init.xavier_normal_(arg_dict["qkv_proj_weight"], gain=1.0)
    arg_dict["o_proj_weight"] = _correct_weight_tensor_view(torch.randn(
        (
            mk.world_size,
            mk.HIDDEN_SIZE,
            mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM
        ),
        dtype=torch.float16,
    ))
    torch.nn.init.xavier_normal_(arg_dict["o_proj_weight"], gain=1.0)
    arg_dict["gate_up_weight"] = _correct_weight_tensor_view(
        torch.zeros((mk.world_size, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype=torch.float16)
    )
    torch.nn.init.xavier_normal_(arg_dict["gate_up_weight"], gain=1.0)
    arg_dict["down_weight"] = _correct_weight_tensor_view(torch.zeros(
        (
            mk.world_size,
            mk.HIDDEN_SIZE,
            mk.INTERMEDIATE_SIZE
        ),
        dtype=torch.float16
    ))
    torch.nn.init.xavier_normal_(arg_dict["down_weight"], gain=1.0)
    return arg_dict


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper):
    arg_dict = prepare_data(batch_size, mega_kernel_wrapper)

    def tir_static(arg_dict, mk: MegaKernel):
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        tvm_arg_dict["o_partial_attn"] = tvm.runtime.tensor(
            np.zeros([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS * mk.HEAD_DIM], dtype=np.float32),
            DEV,
        )
        tvm_arg_dict["lse_partial"] = tvm.runtime.tensor(
            np.zeros([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS], dtype=np.float32), DEV
        )
        tvm_arg_dict["partial_qkv"] = tvm.runtime.tensor(
            np.zeros(
                [
                    mk.SPLIT_QKV_PROJECT,
                    batch_size,
                    (mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM,
                ],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["partial_o"] = tvm.runtime.tensor(
            np.zeros([mk.SPLIT_O_PROJRCT, batch_size, mk.HIDDEN_SIZE], dtype=np.float32), DEV
        )
        tvm_arg_dict["before_o_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )
        tvm_arg_dict["partial_out_gate_up_proj"] = tvm.runtime.tensor(
            np.zeros(
                [
                    mk.GATE_UP_PROJ_SPLIT_K_FACTOR,
                    batch_size,
                    mk.INTERMEDIATE_SIZE * 2,
                ],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["before_down_proj_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )
        # static schedule
        # generate_exec_queue = tvm.get_global_func("megakernel.generate_exec_queue")
        # exec_queue = generate_exec_queue(
        #     batch_size,
        #     arg_dict["attn_task_num"].item(),
        #     1,
        #     NUM_ATTENTION_HEADS,
        #     NUM_KEY_VALUE_HEADS,
        #     HEAD_DIM,
        #     DEV,
        #     tvm.cpu(),
        # )
        exec_queue = generate_exec_queue(batch_size, arg_dict["attn_task_num"].item(), 1)

        # append_pos here is different from flashinfer
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["page_kv_indices"][
                    (arg_dict["page_kv_indptr"][b] * mk.PAGE_SIZE + append_pos[b]) // mk.PAGE_SIZE
                ]
                * mk.PAGE_SIZE
                + append_pos[b] % mk.PAGE_SIZE
            )
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.runtime.tensor(append_pos, device=DEV)
        REPEAT = 100
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
            # generate event tensor
            (
                tvm_arg_dict[f"etensor_qkv_partial_{i}"],
                tvm_arg_dict[f"etensor_q_reduce_{i}"],
                tvm_arg_dict[f"etensor_k_reduce_{i}"],
                tvm_arg_dict[f"etensor_v_reduce_{i}"],
                tvm_arg_dict[f"etensor_attn_{i}"],
                tvm_arg_dict[f"etensor_attn_merge_{i}"],
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
            ) = generate_event_tensor(
                batch_size,
                arg_dict["attn_task_num"],
                tvm_arg_dict["kv_head_idx"],
                tvm_arg_dict["q_indptr"],
                1,
            )
            tvm_arg_dict[f"profiler_buffer_{i}"] = tvm.runtime.tensor(
                np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
            )

        with target:
            iter = 0

            def func():
                nonlocal iter
                mega_kernel_static(
                    # input and output
                    tvm_arg_dict["hidden_state"],
                    tvm_arg_dict[f"residual_{iter}"],
                    tvm_arg_dict["output"],
                    # weight
                    tvm_arg_dict["qkv_proj_weight"],
                    tvm_arg_dict["o_proj_weight"],
                    tvm_arg_dict["q_rms_wight"],
                    tvm_arg_dict["k_rms_wight"],
                    tvm_arg_dict["gate_up_weight"],
                    tvm_arg_dict["down_weight"],
                    tvm_arg_dict["attn_add_rms_weight"],
                    tvm_arg_dict["mlp_add_rms_weight"],
                    # page cache, cos_sin cache and plan info
                    tvm_arg_dict["cos_sin_cache"],
                    tvm_arg_dict["rope_pos"],
                    tvm_arg_dict["kv_cache"],
                    tvm_arg_dict["append_pos"],
                    tvm_arg_dict["q_indptr"],
                    tvm_arg_dict["kv_indptr"],
                    tvm_arg_dict["partial_indptr"],
                    tvm_arg_dict["page_kv_indices"],
                    tvm_arg_dict["q_len"],
                    tvm_arg_dict["kv_len"],
                    tvm_arg_dict["q_start"],
                    tvm_arg_dict["kv_start"],
                    tvm_arg_dict["kv_end"],
                    tvm_arg_dict["kv_head_idx"],
                    tvm_arg_dict["work_indptr"],
                    tvm_arg_dict["len_kv_chunk"],
                    tvm_arg_dict["num_qo_len"],
                    tvm_arg_dict["merge_indptr"],
                    tvm_arg_dict["merge_o_indices"],
                    # intermediate buffer
                    tvm_arg_dict["partial_qkv"],
                    tvm_arg_dict["qkv"],
                    tvm_arg_dict["o"],
                    tvm_arg_dict["o_partial_attn"],
                    tvm_arg_dict["lse_partial"],
                    tvm_arg_dict["partial_o"],
                    tvm_arg_dict["before_o_allreduce"],
                    tvm_arg_dict["hidden_state_attn_mlp"],
                    tvm_arg_dict["partial_out_gate_up_proj"],
                    tvm_arg_dict["out_gate_up_proj"],
                    tvm_arg_dict["out_silu_multiply"],
                    tvm_arg_dict["partial_sum_down_proj"],
                    tvm_arg_dict["before_down_proj_allreduce"],
                    # event tensor
                    tvm_arg_dict[f"etensor_qkv_partial_{iter}"],
                    tvm_arg_dict[f"etensor_q_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_k_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_v_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_attn_{iter}"],
                    tvm_arg_dict[f"etensor_attn_merge_{iter}"],
                    tvm_arg_dict[f"etensor_o_proj_{iter}"],
                    tvm_arg_dict[f"etensor_o_partial_{iter}"],
                    tvm_arg_dict[f"etensor_o_allreduce_{iter}"],
                    tvm_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                    tvm_arg_dict[f"etensor_attn_mlp_{iter}"],
                    tvm_arg_dict[f"etensor_gate_up_proj_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_gate_up_proj_{iter}"],
                    tvm_arg_dict[f"etensor_down_proj_{iter}"],
                    tvm_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_down_proj_allreduce_{iter}"],
                    tvm_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                    # exec queue
                    exec_queue,
                    tvm_arg_dict[f"profiler_buffer_{iter}"],
                )
                iter += 1

            ms = bench(func, warmup=1, repeat=10, proton_name="tir-static")
            print(f"TIR time: {ms:.3f} ms")

            if mk.profiler_on:
                export_to_perfetto_trace(
                    tvm_arg_dict[f"profiler_buffer_{iter - 1}"].numpy(),
                    f"static-schedule-layer.perfetto-trace",
                    event_type_names,
                )
        return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy()

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

        tvm_arg_dict["o_tmp"] = tvm.runtime.tensor(
            np.zeros(
                [tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads, DecodeTile.head_dim],
                dtype=np.float32,
            ),
            DEV,
        )
        tvm_arg_dict["lse_tmp"] = tvm.runtime.tensor(
            np.zeros([tvm_arg_dict["new_batch_size"], DecodeTile.qo_heads], dtype=np.float32), DEV
        )
        tvm_arg_dict["partial_qkv"] = tvm.runtime.tensor(
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
        tvm_arg_dict["partial_o"] = tvm.runtime.tensor(
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
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.runtime.tensor(append_pos, device=DEV)
        REPEAT = 100
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
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
                _,
                tvm_arg_dict[f"etensor_attn_add_rms_norm_{i}"],
                tvm_arg_dict[f"etensor_attn_mlp_{i}"],
                _,
                tvm_arg_dict[f"etensor_gate_up_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_{i}"],
                tvm_arg_dict[f"etensor_down_proj_reduce_{i}"],
                _,
                tvm_arg_dict[f"etensor_mlp_add_rms_norm_{i}"],
                tvm_arg_dict[f"etensor_end_{i}"],
            ) = generate_event_tensor(
                batch_size,
                tvm_arg_dict["o_indptr"],
                batch_size != tvm_arg_dict["new_batch_size"],
                1,
            )
            tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
            tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
            tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)
            tvm_arg_dict[f"profiler_buffer_{i}"] = tvm.runtime.tensor(
                np.zeros([PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
            )

        with target:
            iter = 0

            def func():
                nonlocal iter
                mega_kernel_dynamic(
                    # input and output
                    tvm_arg_dict["hidden_state"],
                    tvm_arg_dict[f"residual_{iter}"],
                    tvm_arg_dict["output"],
                    # weight
                    tvm_arg_dict["qkv_proj_weight"],
                    tvm_arg_dict["o_proj_weight"],
                    tvm_arg_dict["q_rms_wight"],
                    tvm_arg_dict["k_rms_wight"],
                    tvm_arg_dict["gate_up_weight"],
                    tvm_arg_dict["down_weight"],
                    tvm_arg_dict["attn_add_rms_weight"],
                    tvm_arg_dict["mlp_add_rms_weight"],
                    # page cache, cos_sin cache and plan info
                    tvm_arg_dict["cos_sin_cache"],
                    tvm_arg_dict["rope_pos"],
                    tvm_arg_dict["kv_cache"],
                    tvm_arg_dict["kv_indptr"],
                    tvm_arg_dict["kv_indices"],
                    tvm_arg_dict["kv_last_page_len"],
                    tvm_arg_dict["append_pos"],
                    tvm_arg_dict["request_indices"],
                    tvm_arg_dict["kv_tile_indices"],
                    tvm_arg_dict["max_chunk_size"],
                    tvm_arg_dict["o_indptr"],
                    # intermediate buffer
                    tvm_arg_dict["partial_qkv"],
                    tvm_arg_dict["qkv"],
                    tvm_arg_dict["o"],
                    tvm_arg_dict["lse"],
                    tvm_arg_dict["o_tmp"],
                    tvm_arg_dict["lse_tmp"],
                    tvm_arg_dict["partial_o"],
                    tvm_arg_dict["hidden_state_attn_mlp"],
                    tvm_arg_dict["out_gate_up_proj"],
                    tvm_arg_dict["out_silu_multiply"],
                    tvm_arg_dict["partial_sum_down_proj"],
                    # event tensor
                    tvm_arg_dict[f"etensor_qkv_partial_{iter}"],
                    tvm_arg_dict[f"etensor_q_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_k_reduce_{iter}"],
                    tvm_arg_dict[f"etensor_v_reduce_{iter}"],
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
                    # exec queue
                    tvm_arg_dict[f"queue_tasks_{iter}"],
                    tvm_arg_dict[f"queue_head_{iter}"],
                    tvm_arg_dict[f"queue_tail_{iter}"],
                    tvm_arg_dict[f"profiler_buffer_{iter}"],
                )
                iter += 1

            ms = bench(func, warmup=1, repeat=10, proton_name="tir-dynamic")
            print(f"TIR time: {ms:.3f} ms")

            if PROFILER_ON:
                export_to_perfetto_trace(
                    tvm_arg_dict[f"profiler_buffer_{iter - 1}"].numpy(),
                    f"dynamic-schedule-layer.perfetto-trace",
                    event_type_names,
                )
        return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy()

    def std(arg_dict, use_prefill, mk: MegaKernel):
        import flashinfer
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)
            out_f = torch.zeros(
                batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM, dtype=torch.float16, device="cuda"
            )
            lse_f = torch.zeros(batch_size, mk.NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda")
            if use_prefill:
                workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    std_arg_dict["page_kv_last_page_len"],
                    mk.NUM_ATTENTION_HEADS,
                    mk.NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    pos_encoding_mode="NONE",
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            else:
                wrapper = flashinfer.BatchAttention("HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    torch.tensor([mk.SEQ_LEN + 1] * batch_size, dtype=torch.int32).to(0),
                    mk.NUM_ATTENTION_HEADS,
                    mk.NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )
            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, mk.HEAD_DIM)
            q, k, v = torch.split(
                qkv, [mk.NUM_ATTENTION_HEADS, mk.NUM_KEY_VALUE_HEADS, mk.NUM_KEY_VALUE_HEADS], dim=1
            )
            q = flashinfer.norm.rmsnorm(
                input=q.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["q_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM)
            k = flashinfer.norm.rmsnorm(
                input=k.reshape(-1, mk.HEAD_DIM),
                weight=std_arg_dict["k_rms_wight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            ).reshape(batch_size, mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM)
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=mk.HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                append_value=v,
                batch_indices=torch.arange(batch_size, dtype=torch.int32, device=torch_dev),
                positions=std_arg_dict["append_pos"],
                paged_kv_cache=std_arg_dict["kv_cache"],
                kv_indices=std_arg_dict["page_kv_indices"],
                kv_indptr=std_arg_dict["page_kv_indptr"],
                kv_last_page_len=std_arg_dict["page_kv_last_page_len"],
                kv_layout="HND",
            )
            if use_prefill:
                out_f = wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM), std_arg_dict["kv_cache"]
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
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
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )
            return output.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std-use_prefill={use_prefill}")
        print(f"std time: {ms:.3f} ms")
        return output

    with ProtonContext("blackwell_attn"):
        output_tir_static, residual_tir_static = tir_static(arg_dict, mega_kernel_wrapper)
        print("static tir pass", flush=True)
        #     output_tir_dynamic, residual_tir_dynamic = tir_dynamic(arg_dict)
        #     print("dynamic tir pass", flush=True)
        output_std1, residual_std1 = std(arg_dict, use_prefill=True, mk=mega_kernel_wrapper)
        output_std2, residual_std2 = std(arg_dict, use_prefill=False, mk=mega_kernel_wrapper)

    np.testing.assert_allclose(output_std1, output_std2, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_std1, residual_std2, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(output_tir_static, output_std1, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_tir_static, residual_std1, rtol=1e-3, atol=1e-2)
    # np.testing.assert_allclose(output_tir_dynamic, output_std, rtol=1e-3, atol=1e-2)
    # np.testing.assert_allclose(residual_tir_dynamic, residual_std, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":

    mega_kernel_wrapper_static = MegaKernel(world_size=1, profiler_on=False)
    # mega_kernel_wrapper_dynamic = MegaKernel(problem_config, profiler_on=PROFILER_ON)
    mega_static_module = mega_kernel_wrapper_static.get_module_static()
    # mega_kernel_dynamic = mega_kernel_wrapper_dynamic.get_func_dynamic()
    src, lib_static = get_source(mega_static_module)
    print(src)
    # src, mod_dynamic = get_source(mega_kernel_dynamic)

    for batch_size in [1, 3, 5, 7, 15, 31, 63, 127, 128]:

        print(f"batch_size: {batch_size}", flush=True)
        test(batch_size, lib_static["main"], None, mega_kernel_wrapper_static)
