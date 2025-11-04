import argparse
import math
import tempfile

import flashinfer
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile, RMSNormTile
from tvm.tirp.megakernel.allreduce import AllreduceTile
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.batch_attn import BatchAttnTile
from tvm.tirp.megakernel.batch_merge import BatchMergeTile
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile
from tvm.tirp.megakernel.gate_up_silu import GateUpSiluTile
from tvm.tirp.megakernel.reduce_rms_norm_rope_q import SplitKReduceRMSnormRopeQTile
from tvm.tirp.megakernel.reduce_rms_norm_rope_append_k import SplitKReduceRMSnormRopeAppendKTile
from tvm.tirp.megakernel.reduce_append_v import SplitKReduceAppendVTile
from tvm.tirp.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirp.megakernel import static_scheduler, dynamic_scheduler
from tvm.tirp.megakernel.wrapper import MegaKernelWrapper
from tvm.tirp.megakernel.model_config import get_model_config
from tvm.tirp.megakernel.support import generate_event_tensor, generate_exec_queue, get_inverse_plan_info


class MegaKernelDenseLayer(MegaKernelWrapper):

    def __init__(self, config, world_size, profiler_on):
        super().__init__(config, world_size, profiler_on)
        self.world_size = world_size

    def _set_tiles(self, batch_size, BLK_M):
        self.qkv_proj_tile = self._add_tile(GemmTile((self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, self.HIDDEN_SIZE,
                                                    "float16", "float16", self.SPLIT_QKV_PROJECT, BLK_M, BLK_M), ProfileEventType.GEMM_QKV_PROJ)
        self.reduce_rms_rope_q_tile = self._add_tile(SplitKReduceRMSnormRopeQTile(batch_size, self.RMS_NORM_EPS, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT, use_rms_norm=(self.MODEL_NAME != "llama3_1b")), ProfileEventType.Q_REDUCE_RMSNORM_ROPE)
        self.reduce_rms_rope_append_k_tile = self._add_tile(SplitKReduceRMSnormRopeAppendKTile(batch_size, self.RMS_NORM_EPS, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT, self.PAGE_SIZE, use_rms_norm=(self.MODEL_NAME != "llama3_1b")), ProfileEventType.K_REDUCE_RMSNORM_ROPE_APPEND)
        self.reduce_append_v_tile = self._add_tile(SplitKReduceAppendVTile(batch_size, self.NUM_KEY_VALUE_HEADS, self.NUM_ATTENTION_HEADS, self.HEAD_DIM, self.SPLIT_QKV_PROJECT, self.PAGE_SIZE), ProfileEventType.V_REDUCE_APPEND)
        self.attn_tile = self._add_tile(BatchAttnTile(self.PAGE_SIZE, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM, prefetch_on=True), ProfileEventType.BATCH_ATTENTION)
        self.merge_tile = self._add_tile(BatchMergeTile(self.HEAD_DIM, self.NUM_KEY_VALUE_HEADS, self.NUM_ATTENTION_HEADS), ProfileEventType.BATCH_ATTENTION_MERGE)
        self.o_proj_tile = self._add_tile(GemmTile(self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM, "float16", "float16", self.SPLIT_O_PROJECT, BLK_M, BLK_M, use_tma_reduce=self.world_size == 1, prefetch_on=True), ProfileEventType.GEMM_O_PROJ)
        self.o_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.SPLIT_O_PROJECT), ProfileEventType.GEMM_O_REDUCE, predicate=self.world_size > 1)
        self.o_allreduce_tile = self._add_tile(AllreduceTile(self.world_size), ProfileEventType.O_ALLREDUCE, predicate=self.world_size > 1)
        self.attn_add_rms_tile = self._add_tile(RMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE) if self.world_size == 1 else AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE), ProfileEventType.ATTN_ADD_RMS_NORM)
        self.gate_up_silu_tile = self._add_tile(GateUpSiluTile(self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE, "float16", "float16", self.GATE_UP_PROJ_SPLIT_K_FACTOR, BLK_M, BLK_M, prefetch_on=True), ProfileEventType.GATE_UP_SILU, predicate=self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1)
        self.gemm_gate_up_proj_tile = self._add_tile(GemmTile(self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE, "float16", "float16", self.GATE_UP_PROJ_SPLIT_K_FACTOR, BLK_M, BLK_M, prefetch_on=True), ProfileEventType.GEMM_GATE_UP_PROJ, predicate=self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1)
        self.silu_multiply_tile = self._add_tile(SiluMultiplyTile(batch_size, self.INTERMEDIATE_SIZE, "float16"), ProfileEventType.SPLIT_SILU_MULTIPLY, predicate=self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1)
        self.gemm_gate_up_proj_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.INTERMEDIATE_SIZE * 2, "float16", self.GATE_UP_PROJ_SPLIT_K_FACTOR), ProfileEventType.GATE_UP_PROJ_REDUCE, predicate=self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1)
        self.gemm_down_proj_tile = self._add_tile(GemmTile(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, "float16", "float16", self.DOWN_PROJ_SPLIT_K_FACTOR, BLK_M, BLK_M, use_tma_reduce=self.world_size == 1, prefetch_on=True), ProfileEventType.GEMM_DOWN_PROJ)
        self.down_proj_reduce_tile = self._add_tile(SplitKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.DOWN_PROJ_SPLIT_K_FACTOR), ProfileEventType.DOWN_PROJ_REDUCE, predicate=self.world_size > 1)
        self.down_proj_allreduce_tile = self._add_tile(AllreduceTile(self.world_size), ProfileEventType.DOWN_PROJ_ALLREDUCE, predicate=self.world_size > 1)
        self.mlp_add_rms_norm_tile = self._add_tile(RMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE) if self.world_size == 1 else AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE), ProfileEventType.MLP_ADD_RMS_NORM)

    def set_tiles(self, batch_size, BLK_M):
        self.tile_attr = {}
        self.class_list = set()
        self._set_tiles(batch_size, BLK_M)

    def set_events(
        self,
        batch_size,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        etensor_qkv_partial_global,
        etensor_notify_attn_global,
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
        etensor_end_global,
    ):
        self.evt_qkv_partial = Semaphore(
            self.SPLIT_QKV_PROJECT, etensor_qkv_partial_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_qkv_partial_global is not None else None
        self.evt_attn_merge = Semaphore(
            -1, etensor_attn_merge_global, decrement=True, use_nvshmem=self.world_size > 1
        ) if etensor_attn_merge_global is not None else None
        self.evt_o_proj = Semaphore(
            -1, etensor_o_proj_global, decrement=True, use_nvshmem=self.world_size > 1
        ) if etensor_o_proj_global is not None else None
        self.evt_o_partial = Semaphore(
            self.SPLIT_O_PROJECT * (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),
            etensor_o_partial_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_o_partial_global is not None else None
        self.evt_o_allreduce = Semaphore(
            self.world_size * self.o_reduce_tile.M_split,
            etensor_o_allreduce_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_o_allreduce_global is not None else None   
        self.evt_attn_add_rms = Semaphore(
            self.SPLIT_O_PROJECT * (self.HIDDEN_SIZE // GemmTile.BLK_N) if self.world_size == 1 else
            self.HIDDEN_SIZE // self.o_allreduce_tile.N_TILE,
            etensor_attn_add_rms_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_attn_mlp_global is not None else None
        self.evt_attn_mlp = Semaphore(
            batch_size, etensor_attn_mlp_global, use_nvshmem=self.world_size > 1
        )
        self.evt_gate_up_proj_reduce = Semaphore(
            self.GATE_UP_PROJ_SPLIT_K_FACTOR,
            etensor_gate_up_proj_reduce_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_gate_up_proj_reduce_global is not None else None
        self.evt_gate_up_proj = Semaphore(
            2 if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1 else 2 * self.gemm_gate_up_proj_reduce_tile.M_split,
            etensor_gate_up_proj_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_gate_up_proj_global is not None else None
        self.evt_down_proj = Semaphore(
            -1, etensor_down_proj_global, decrement=True, use_nvshmem=self.world_size > 1
        ) if etensor_down_proj_global is not None else None
        self.evt_down_proj_reduce = Semaphore(
            self.DOWN_PROJ_SPLIT_K_FACTOR * (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),
            etensor_down_proj_reduce_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_down_proj_reduce_global is not None else None
        self.evt_down_proj_allreduce = Semaphore(
            self.world_size * self.down_proj_reduce_tile.M_split,
            etensor_down_proj_allreduce_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_down_proj_allreduce_global is not None else None
        self.evt_mlp_add_rms = Semaphore(
            self.DOWN_PROJ_SPLIT_K_FACTOR * (self.HIDDEN_SIZE // GemmTile.BLK_N) if self.world_size == 1 else
            self.HIDDEN_SIZE // self.down_proj_allreduce_tile.N_TILE,
            etensor_mlp_add_rms_global,
            use_nvshmem=self.world_size > 1,
        ) if etensor_mlp_add_rms_global is not None else None
        self.evt_notify_attn = Semaphore(
            -1, etensor_notify_attn_global, decrement=True, use_nvshmem=self.world_size > 1
        ) if etensor_notify_attn_global is not None else None
        self.evt_end = Semaphore(batch_size, etensor_end_global, use_nvshmem=self.world_size > 1) if etensor_end_global is not None else None

    @T.macro
    def task_impl_gemm_qkv_proj(self, is_dynamic_sch):
        with T.cta():
            if is_dynamic_sch:
                h_tile = T.meta_var(GemmTile.BLK_N // self.HEAD_DIM)
                if self.tile_scheduler.n_idx * h_tile < self.NUM_ATTENTION_HEADS // self.reduce_rms_rope_append_k_tile.h_tile:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_qkv_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx),
                        lambda trigger_idx: (
                            -1, self.reduce_rms_rope_q_tile.m_split * h_tile, 
                            lambda push_idx: (JobType.Q_REDUCE_RMS_ROPE.value, push_idx // h_tile, self.tile_scheduler.n_idx * h_tile + push_idx % h_tile, 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
                elif self.tile_scheduler.n_idx * h_tile < (self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS) // self.reduce_rms_rope_append_k_tile.h_tile:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_qkv_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx),
                        lambda trigger_idx: (
                            -1, self.reduce_rms_rope_append_k_tile.m_split * h_tile, 
                            lambda push_idx: (JobType.K_REDUCE_RMS_ROPE_APPEND.value, push_idx // h_tile, self.tile_scheduler.n_idx * h_tile - self.NUM_ATTENTION_HEADS // self.reduce_rms_rope_append_k_tile.h_tile + push_idx % h_tile, 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
                else:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_qkv_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx),
                        lambda trigger_idx: (
                            -1, self.reduce_append_v_tile.m_split * h_tile, 
                            lambda push_idx: (JobType.V_REDUCE_APPEND.value, push_idx // h_tile, self.tile_scheduler.n_idx * h_tile - (self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS) // self.reduce_append_v_tile.h_tile + push_idx % h_tile, 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
            self.run_tile(self.qkv_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
            self.tile_scheduler.notify(self.evt_qkv_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_q_reduce_rms_rope(
        self,
        batch_size,
        inverse_indptr_global,
        inverse_indices_global,
        partial_qkv_global,
        qkv_global,
        q_rms_weight_global,
        rope_pos_global,
        cos_sin_cache_global,
        is_dynamic_sch,
    ):
        with T.cta():
            # TODO: assume h_tile=1 here
            m_tile = T.meta_var(self.reduce_rms_rope_q_tile.m_tile)
            kv_idx = self.tile_scheduler.n_idx // (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)
            beg_idx = kv_idx * batch_size + self.tile_scheduler.m_idx * m_tile
            end_idx = kv_idx * batch_size + T.min((self.tile_scheduler.m_idx + 1) * m_tile, batch_size)
            beg = inverse_indptr_global[beg_idx]
            end = inverse_indptr_global[end_idx]
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],),
                    lambda trigger_idx: (
                        -1, 1, 
                        lambda push_idx: (JobType.BATCH_ATTENTION.value, inverse_indices_global[beg + trigger_idx], 0, 0)
                    ), "thread", "warpgroup", scope_id=1
                )
            self.tile_scheduler.wait(self.evt_qkv_partial, self.tile_scheduler.n_idx // (GemmTile.BLK_N // self.HEAD_DIM))
            self.run_tile(self.reduce_rms_rope_q_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_qkv_global, qkv_global, q_rms_weight_global, rope_pos_global, cos_sin_cache_global)
            self.tile_scheduler.notify(self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],), scope="cta")            

    @T.macro
    def task_impl_k_reduce_rms_rope_append(
        self,
        batch_size,
        inverse_indptr_global,
        inverse_indices_global,
        partial_qkv_global,
        k_rms_weight_global,
        rope_pos_global,
        cos_sin_cache_global,
        append_pos_global,
        kv_cache_global,
        is_dynamic_sch,
    ):
        with T.cta():
            # TODO: assume h_tile=1 here
            m_tile = T.meta_var(self.reduce_rms_rope_append_k_tile.m_tile)
            beg_idx = self.tile_scheduler.n_idx * batch_size + self.tile_scheduler.m_idx * m_tile
            end_idx = self.tile_scheduler.n_idx * batch_size + T.min((self.tile_scheduler.m_idx + 1) * m_tile, batch_size)
            beg = inverse_indptr_global[beg_idx]
            end = inverse_indptr_global[end_idx]
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],),
                    lambda trigger_idx: (
                        -1, 1, 
                        lambda push_idx: (JobType.BATCH_ATTENTION.value, inverse_indices_global[beg + trigger_idx], 0, 0)
                    ), "thread", "warpgroup", scope_id=1
                )
            self.tile_scheduler.wait(self.evt_qkv_partial, (self.tile_scheduler.n_idx + self.NUM_ATTENTION_HEADS) // (GemmTile.BLK_N // self.HEAD_DIM))
            self.run_tile(self.reduce_rms_rope_append_k_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_qkv_global, k_rms_weight_global, rope_pos_global, cos_sin_cache_global, append_pos_global, kv_cache_global)
            self.tile_scheduler.notify(self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],), scope="cta")

    @T.macro
    def task_impl_v_reduce_append(
        self,
        batch_size,
        inverse_indptr_global,
        inverse_indices_global,
        partial_qkv_global,
        kv_cache_global,
        append_pos_global,
        is_dynamic_sch,
    ):
        with T.cta():
            # TODO: assume h_tile=1 here
            m_tile = T.meta_var(self.reduce_append_v_tile.m_tile)
            beg_idx = self.tile_scheduler.n_idx * batch_size + self.tile_scheduler.m_idx * m_tile
            end_idx = self.tile_scheduler.n_idx * batch_size + T.min((self.tile_scheduler.m_idx + 1) * m_tile, batch_size)
            beg = inverse_indptr_global[beg_idx]
            end = inverse_indptr_global[end_idx]
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],),
                    lambda trigger_idx: (
                        -1, 1, 
                        lambda push_idx: (JobType.BATCH_ATTENTION.value, inverse_indices_global[beg + trigger_idx], 0, 0)
                    ), "thread", "warpgroup", scope_id=1
                )
            self.tile_scheduler.wait(self.evt_qkv_partial, (self.tile_scheduler.n_idx + self.NUM_ATTENTION_HEADS + self.NUM_KEY_VALUE_HEADS) // (GemmTile.BLK_N // self.HEAD_DIM))
            self.run_tile(self.reduce_append_v_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_qkv_global, kv_cache_global, append_pos_global)
            self.tile_scheduler.notify(self.evt_notify_attn, end - beg, lambda notify_idx: (-1, inverse_indices_global[beg + notify_idx],), scope="cta")

    @T.macro
    def task_impl_batch_attention(
        self,
        smem_manager,
        qkv_global,
        kv_cache_global,
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
        o_global,
        o_partial_attn_global,
        lse_partial_attn_global,
        num_qo_len_global,
        merge_indptr_global,
        merge_o_indices_global,
        batch_size,
        is_dynamic_sch,
    ):
        with T.cta():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            # TODO: Here now cannot handle tha cases that each tile will be allocated with more than 8 tasks
            attn_task_num = work_indptr_global[KernelConfig.SM_NUMBER * KernelConfig.WG_NUMBER]
            self.run_tile_prefetch(self.attn_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, qkv_global, kv_cache_global, q_indptr_global, kv_indptr_global, partial_indptr_global,
                                    kv_indices_global, q_len_global, kv_len_global, q_start_global, kv_start_global,
                                    kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global,
                                    o_global, o_partial_attn_global, lse_partial_attn_global, self.profiler)
            if is_dynamic_sch:
                if attn_task_num > batch_size * self.NUM_KEY_VALUE_HEADS: # split kv
                    # notes: assume that gqa_size is no greater than warp number in cta here
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_attn_merge, 1, lambda notify_idx: (-1, 0),
                        lambda trigger_idx: ( 
                            -1, ceildiv(batch_size * self.NUM_ATTENTION_HEADS, KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER), 
                            lambda push_idx: (JobType.BATCH_ATTENTION_MERGE.value, push_idx, 0, 0)
                        ), "cta", "cta", scope_id=-1
                    )
                else: # no split kv    
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_o_proj, self.SPLIT_O_PROJECT, lambda notify_idx: (-1, notify_idx,),
                        lambda trigger_idx: ( 
                            -1, ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N), 
                            lambda push_idx: (JobType.GEMM_O_PROJ.value, 0, push_idx, trigger_idx)
                        ), "warp", "cta", scope_id=-1 
                    )
            self.tile_scheduler.wait(self.evt_notify_attn, self.tile_scheduler.m_idx, wait_level="warp")
            self.run_tile(self.attn_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, qkv_global, kv_cache_global, q_indptr_global, kv_indptr_global, partial_indptr_global,
                            kv_indices_global, q_len_global, kv_len_global, q_start_global, kv_start_global,
                            kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global,
                            o_global, o_partial_attn_global, lse_partial_attn_global, self.profiler)                           
            if attn_task_num > batch_size * self.NUM_KEY_VALUE_HEADS: # split kv
                self.tile_scheduler.notify(self.evt_attn_merge, 1, lambda notify_idx: (-1, 0), scope="cta", scope_id=-1)
            else: # no split kv     
                self.tile_scheduler.notify(self.evt_o_proj, self.SPLIT_O_PROJECT, lambda notify_idx: (-1, notify_idx,), scope="cta", scope_id=-1)

    @T.macro
    def task_impl_batch_attention_merge(
        self,
        batch_size,
        o_partial_attn_global,
        o_global,
        lse_partial_attn_global,
        num_qo_len_global,
        merge_indptr_global,
        merge_o_indices_global,
        is_dynamic_sch,
    ):
        if self.MODEL_NAME == "qwen3_32b" or self.MODEL_NAME == "qwen3_30b_a3b":
            with T.cta():
                # here we leverage the fact that each cta processes the gqa_num(=8) heads
                worker_id = self.tile_scheduler.m_idx * KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER 
                # logical shape: [kv_head_num, batch_size, gqa_num]
                kv_idx = worker_id // (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                range_start = (kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM // self.o_proj_tile.TILE_K
                range_end = (((kv_idx + 1) * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM - 1) // self.o_proj_tile.TILE_K
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_o_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,),
                        lambda trigger_idx: (
                            -1, ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N),
                            lambda push_idx: (JobType.GEMM_O_PROJ.value, 0, push_idx, range_start + trigger_idx)
                        ), "warp", "cta"
                    )
                self.tile_scheduler.wait(self.evt_attn_merge, 0)
                self.run_tile(self.merge_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, o_partial_attn_global, o_global, lse_partial_attn_global, num_qo_len_global,
                                    merge_indptr_global, merge_o_indices_global)
                self.tile_scheduler.notify(self.evt_o_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,), scope="cta")
        elif self.MODEL_NAME == "llama3_1b":
            with T.cta():
                # here we leverage the fact that each warpgroup processes the gqa_num(=4) heads
                wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                worker_id = (self.tile_scheduler.m_idx * KernelConfig.WG_NUMBER + wg_id) * KernelConfig.WARP_NUMBER 
                # logical shape: [kv_head_num, batch_size, gqa_num]
                kv_idx = worker_id // (batch_size * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS))
                range_start = (kv_idx * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM // self.o_proj_tile.TILE_K
                range_end = (((kv_idx + 1) * (self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS)) * self.HEAD_DIM - 1) // self.o_proj_tile.TILE_K
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_o_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,),
                        lambda trigger_idx: (
                            -1, ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N),
                            lambda push_idx: (JobType.GEMM_O_PROJ.value, 0, push_idx, range_start + trigger_idx)
                        ), "warp", "warpgroup", scope_id=-1
                    )
                self.tile_scheduler.wait(self.evt_attn_merge, 0)
                self.run_tile(self.merge_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, o_partial_attn_global, o_global, lse_partial_attn_global, num_qo_len_global,
                                    merge_indptr_global, merge_o_indices_global)
                self.tile_scheduler.notify(self.evt_o_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,), scope="warpgroup", scope_id=-1)
        else:
            assert False

    @T.macro
    def task_impl_gemm_o_proj(self, batch_size, is_dynamic_sch):
        with T.cta():
            self.run_tile_prefetch(self.o_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
            if is_dynamic_sch:
                if self.world_size == 1:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_attn_add_rms, 1, lambda notify_idx: (-1, 0,),
                        lambda trigger_idx: (
                            -1, batch_size,
                            lambda push_idx: (JobType.ATTN_ADD_RMS_NORM.value, push_idx, 0, 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
                else:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_o_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx // (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),),
                        lambda trigger_idx: (
                            -1, self.o_reduce_tile.M_split,
                            lambda push_idx: (JobType.GEMM_O_REDUCE.value, push_idx, self.tile_scheduler.n_idx // (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT), 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
            self.tile_scheduler.wait(self.evt_o_proj, self.tile_scheduler.k_idx, wait_level="warp")
            self.run_tile(self.o_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
            if self.world_size == 1:
                self.tile_scheduler.notify(self.evt_attn_add_rms, 1, lambda notify_idx: (-1, 0), scope="warpgroup", scope_id=0)
            else:
                self.tile_scheduler.notify(self.evt_o_partial, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx // (self.o_reduce_tile.N_TILE // SplitKReduceTile.N_UNIT),), scope="warpgroup", scope_id=0)
                

    @T.macro
    def task_impl_gemm_o_reduce(
        self,
        batch_size,
        partial_o_global,
        hidden_state_attn_mlp_global,
        before_o_allreduce_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.world_size > 1:
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_o_allreduce, 1, lambda notify_idx: (self.tile_scheduler.n_idx % self.world_size, self.tile_scheduler.n_idx // self.world_size,),
                        lambda trigger_idx: (
                            self.tile_scheduler.n_idx % self.world_size, ceildiv(batch_size, self.o_allreduce_tile.M_TILE),
                            lambda push_idx: (JobType.O_ALLREDUCE.value, push_idx, self.tile_scheduler.n_idx // self.world_size, 0)
                        ), "warpgroup", "warpgroup", scope_id=1
                    )
                self.tile_scheduler.wait(self.evt_o_partial, self.tile_scheduler.n_idx, wait_level="warp")
                self.run_tile(self.o_reduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_o_global, before_o_allreduce_global)
                self.tile_scheduler.notify(self.evt_o_allreduce, 1, lambda notify_idx: (self.tile_scheduler.n_idx % self.world_size, self.tile_scheduler.n_idx // self.world_size,), scope="cta")

    @T.macro
    def task_impl_o_allreduce(
        self,
        batch_size,
        before_o_allreduce_global,
        hidden_state_attn_mlp_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.world_size > 1:
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_attn_add_rms, self.world_size, lambda notify_idx: (notify_idx, self.tile_scheduler.m_idx,),
                        lambda trigger_idx: (
                            trigger_idx, T.min(self.o_allreduce_tile.M_TILE, batch_size - self.tile_scheduler.m_idx * self.o_allreduce_tile.M_TILE),
                            lambda push_idx: (JobType.ATTN_ADD_RMS_NORM.value, self.tile_scheduler.m_idx * self.o_allreduce_tile.M_TILE + push_idx, 0, 0)
                        ), "warp", "cta", scope_id=0
                    )
                self.tile_scheduler.wait(self.evt_o_allreduce, self.tile_scheduler.n_idx)
                self.run_tile(self.o_allreduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, before_o_allreduce_global, hidden_state_attn_mlp_global)
                self.tile_scheduler.notify(self.evt_attn_add_rms, self.world_size, lambda notify_idx: (notify_idx, self.tile_scheduler.m_idx,), scope="cta")

    @T.macro
    def task_impl_attn_add_rms_norm(
        self,
        hidden_state_attn_mlp_global,
        residual_global,
        attn_add_rms_weight_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if is_dynamic_sch:
                if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_attn_mlp, 1, lambda notify_idx: (-1, 0,),
                        lambda trigger_idx: (
                            -1, self.INTERMEDIATE_SIZE * 2 // GemmTile.BLK_N,
                            lambda push_idx: (JobType.GATE_UP_SILU.value, 0, push_idx, 0)
                        ), "cta", "cta"
                    )  
                else:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_attn_mlp, 1, lambda notify_idx: (-1, 0,),
                        lambda trigger_idx: (
                            -1, self.INTERMEDIATE_SIZE * 2 // GemmTile.BLK_N * self.GATE_UP_PROJ_SPLIT_K_FACTOR,
                            lambda push_idx: (JobType.GEMM_GATE_UP_PROJ.value, 0, ((push_idx // 2) // self.GATE_UP_PROJ_SPLIT_K_FACTOR 
                                                                                    + (push_idx % 2) * self.INTERMEDIATE_SIZE // GemmTile.BLK_N),
                                                                                    (push_idx // 2) % self.GATE_UP_PROJ_SPLIT_K_FACTOR)
                        ), "cta", "cta"
                    )  
            if self.world_size == 1:
                self.tile_scheduler.wait(self.evt_attn_add_rms, 0)
                T.cuda.thread_fence() # ensure previous tma-reduce are visible
            else:
                self.tile_scheduler.wait(self.evt_attn_add_rms, self.tile_scheduler.m_idx // self.o_allreduce_tile.M_TILE)
            self.run_tile(self.attn_add_rms_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global)
            self.tile_scheduler.notify(self.evt_attn_mlp, 1, lambda notify_idx: (-1, 0,), scope="cta")
            
    @T.macro
    def task_impl_gemm_gate_up_proj(self, is_dynamic_sch):
        with T.cta():
            if self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1:
                self.run_tile_prefetch(self.gemm_gate_up_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_gate_up_proj_reduce, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx,),
                        lambda trigger_idx: (
                            -1, self.gemm_gate_up_proj_reduce_tile.M_split,
                            lambda push_idx: (JobType.GATE_UP_PROJ_REDUCE.value, push_idx, self.tile_scheduler.n_idx, 0)
                        ), "warpgroup", "warpgroup"
                    )
                self.tile_scheduler.wait(self.evt_attn_mlp, 0, wait_level="warp")
                self.run_tile(self.gemm_gate_up_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
                self.tile_scheduler.notify(self.evt_gate_up_proj_reduce, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx,), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_gate_up_silu(self, is_dynamic_sch):
        with T.cta():
            if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
                range_start = self.tile_scheduler.n_idx * GateUpSiluTile.BLK_N // 2 // self.gemm_down_proj_tile.TILE_K
                range_end = ((self.tile_scheduler.n_idx + 1) * GateUpSiluTile.BLK_N // 2 - 1) // self.gemm_down_proj_tile.TILE_K
                self.run_tile_prefetch(self.gate_up_silu_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_down_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,),
                        lambda trigger_idx: (
                            -1, ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N),
                            lambda push_idx: (JobType.GEMM_DOWN_PROJ.value, 0, push_idx, range_start + trigger_idx)
                        ), "warp", "warpgroup"
                    )
                self.tile_scheduler.wait(self.evt_attn_mlp, 0, wait_level="warp")
                self.run_tile(self.gate_up_silu_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
                self.tile_scheduler.notify(self.evt_down_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_gate_up_proj_reduce(
        self,
        partial_out_gate_up_proj_global,
        out_gate_up_proj_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1:
                if is_dynamic_sch:
                    if self.tile_scheduler.n_idx >= self.INTERMEDIATE_SIZE // GemmTile.BLK_N:
                        self.tile_scheduler.pre_notify_and_push(
                            self.evt_gate_up_proj, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx - self.INTERMEDIATE_SIZE // GemmTile.BLK_N,),
                            lambda trigger_idx: (
                                -1, 1,
                                lambda push_idx: (JobType.SPLIT_SILU_MULTIPLY.value, self.tile_scheduler.n_idx - self.INTERMEDIATE_SIZE // GemmTile.BLK_N, 0, 0)
                            ), "thread", "thread"
                        )
                    else:
                        self.tile_scheduler.pre_notify_and_push(
                            self.evt_gate_up_proj, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx,),
                            lambda trigger_idx: (
                                -1, 1,
                                lambda push_idx: (JobType.SPLIT_SILU_MULTIPLY.value, self.tile_scheduler.n_idx, 0, 0)
                            ), "thread", "thread"
                        )
                self.tile_scheduler.wait(self.evt_gate_up_proj_reduce, self.tile_scheduler.n_idx)
                self.run_tile(self.gemm_gate_up_proj_reduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_out_gate_up_proj_global, out_gate_up_proj_global)
                if self.tile_scheduler.n_idx >= self.INTERMEDIATE_SIZE // GemmTile.BLK_N:
                    self.tile_scheduler.notify(self.evt_gate_up_proj, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx - self.INTERMEDIATE_SIZE // GemmTile.BLK_N,), scope="cta")
                else:
                    self.tile_scheduler.notify(self.evt_gate_up_proj, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx,), scope="cta")

    @T.macro
    def task_impl_split_silu_multiply(
        self,
        out_gate_up_proj_global,
        out_silu_multiply_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.GATE_UP_PROJ_SPLIT_K_FACTOR > 1:
                range_start = self.tile_scheduler.m_idx * SiluMultiplyTile.TILE_SIZE // self.gemm_down_proj_tile.TILE_K
                range_end = ((self.tile_scheduler.m_idx + 1) * SiluMultiplyTile.TILE_SIZE - 1) // self.gemm_down_proj_tile.TILE_K
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_down_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,),
                        lambda trigger_idx: (
                            -1, ceildiv(self.HIDDEN_SIZE, GemmTile.BLK_N),
                            lambda push_idx: (JobType.GEMM_DOWN_PROJ.value, 0, push_idx, range_start + trigger_idx)
                        ), "warp", "cta"
                    )
                self.tile_scheduler.wait(self.evt_gate_up_proj, self.tile_scheduler.m_idx, wait_level="warp")
                self.run_tile(self.silu_multiply_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, out_gate_up_proj_global, out_silu_multiply_global, self.tile_scheduler)
                self.tile_scheduler.notify(self.evt_down_proj, range_end - range_start + 1, lambda notify_idx: (-1, range_start + notify_idx,), scope="cta")

    @T.macro
    def task_impl_gemm_down_proj(self, batch_size, is_dynamic_sch):
        with T.cta():
            self.run_tile_prefetch(self.gemm_down_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
            if is_dynamic_sch:
                if self.world_size == 1:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_mlp_add_rms, 1, lambda notify_idx: (-1, 0,),
                        lambda trigger_idx: (
                            -1, batch_size,
                            lambda push_idx: (JobType.MLP_ADD_RMS_NORM.value, push_idx, 0, 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    ) 
                else:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_down_proj_reduce, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx // (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),),
                        lambda trigger_idx: (
                            -1, self.down_proj_reduce_tile.M_split,
                            lambda push_idx: (JobType.DOWN_PROJ_REDUCE.value, push_idx, self.tile_scheduler.n_idx // (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), 0)
                        ), "warpgroup", "warpgroup", scope_id=0
                    )
            self.tile_scheduler.wait(self.evt_down_proj, self.tile_scheduler.k_idx, wait_level="warp")
            self.run_tile(self.gemm_down_proj_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
            if self.world_size == 1:
                self.tile_scheduler.notify(self.evt_mlp_add_rms, 1, lambda notify_idx: (-1, 0), scope="warpgroup", scope_id=0)
            else:
                self.tile_scheduler.notify(self.evt_down_proj_reduce, 1, lambda notify_idx: (-1, self.tile_scheduler.n_idx // (self.down_proj_reduce_tile.N_TILE // GemmTile.BLK_N),), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_down_proj_reduce(
        self,
        batch_size,
        partial_sum_down_proj_global,
        output_global,
        before_down_proj_allreduce_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.world_size > 1:
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_down_proj_allreduce, 1, lambda notify_idx: (self.tile_scheduler.n_idx % self.world_size, self.tile_scheduler.n_idx // self.world_size,),
                        lambda trigger_idx: (
                            self.tile_scheduler.n_idx % self.world_size, ceildiv(batch_size, self.down_proj_allreduce_tile.M_TILE),
                            lambda push_idx: (JobType.DOWN_PROJ_ALLREDUCE.value, push_idx, self.tile_scheduler.n_idx // self.world_size, 0)
                        ), "warpgroup", "warpgroup", scope_id=1
                    )
                self.tile_scheduler.wait(self.evt_down_proj_reduce, self.tile_scheduler.n_idx, wait_level="warp")
                self.run_tile(self.down_proj_reduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, partial_sum_down_proj_global, before_down_proj_allreduce_global)
                self.tile_scheduler.notify(self.evt_down_proj_allreduce, 1, lambda notify_idx: (self.tile_scheduler.n_idx % self.world_size, self.tile_scheduler.n_idx // self.world_size,), scope="cta")

    @T.macro
    def task_impl_down_proj_allreduce(
        self,
        batch_size,
        before_down_proj_allreduce_global,
        output_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if self.world_size > 1:
                if is_dynamic_sch:
                    self.tile_scheduler.pre_notify_and_push(
                        self.evt_mlp_add_rms, self.world_size, lambda notify_idx: (notify_idx, self.tile_scheduler.m_idx,),
                        lambda trigger_idx: (
                            trigger_idx, T.min(self.down_proj_allreduce_tile.M_TILE, batch_size - self.tile_scheduler.m_idx * self.down_proj_allreduce_tile.M_TILE),
                            lambda push_idx: (JobType.MLP_ADD_RMS_NORM.value, self.tile_scheduler.m_idx * self.down_proj_allreduce_tile.M_TILE + push_idx, 0, 0)
                        ), "warp", "cta"
                    )
                self.tile_scheduler.wait(self.evt_down_proj_allreduce, self.tile_scheduler.n_idx)
                self.run_tile(self.down_proj_allreduce_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, before_down_proj_allreduce_global, output_global)
                self.tile_scheduler.notify(self.evt_mlp_add_rms, self.world_size, lambda notify_idx: (notify_idx, self.tile_scheduler.m_idx,), scope="cta")
                
    @T.macro
    def task_impl_mlp_add_rms_norm(
        self,
        output_global,
        residual_global,
        mlp_add_rms_weight_global,
        is_dynamic_sch,
    ):
        with T.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_end, 1, lambda notify_idx: (-1, 0,),
                    lambda trigger_idx: (
                        -1, KernelConfig.SM_NUMBER,
                        lambda push_idx: (JobType.END.value, 0, 0, 0)
                    ), "cta", "cta"
                )
            if self.world_size == 1:
                self.tile_scheduler.wait(self.evt_mlp_add_rms, 0)
                T.cuda.thread_fence() # ensure previous tma-reduce are visible
            else:
                self.tile_scheduler.wait(self.evt_mlp_add_rms, self.tile_scheduler.m_idx // self.down_proj_allreduce_tile.M_TILE)
            self.run_tile(self.mlp_add_rms_norm_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, output_global, residual_global, mlp_add_rms_weight_global)
            

    # fmt: off
    @T.macro
    def fused_body(
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
        inverse_indptr_global,
        inverse_indices_global,
        partial_qkv_global,
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
        etensor_notify_attn_global,
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
        etensor_end_global,
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        BLK_M,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        Scheduler: Type[Union[static_scheduler.StaticTileScheduler, dynamic_scheduler.DynamicTileScheduler]],
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

        self.qkv_proj_tile.set_tensor_map(A_tensor_map_qkv_proj, B_tensor_map_qkv_proj, D_tensor_map_qkv_proj, hidden_state_global, qkv_proj_weight_global, partial_qkv_global)
        self.o_proj_tile.set_tensor_map(A_tensor_map_o_proj, B_tensor_map_o_proj, D_tensor_map_o_proj, o_global.view(-1, self.NUM_ATTENTION_HEADS * self.HEAD_DIM).buffer, 
                                        o_proj_weight_global, residual_global if self.world_size == 1 else partial_o_global)
        if self.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            self.gate_up_silu_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj, hidden_state_attn_mlp_global, gate_up_weight_global, out_silu_multiply_global)
        else:
            self.gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj, hidden_state_attn_mlp_global, gate_up_weight_global, partial_out_gate_up_proj_global)
        self.gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj, out_silu_multiply_global, down_weight_global, 
                                                residual_global if self.world_size == 1 else partial_sum_down_proj_global)

        self.host_init_all()

        with T.kernel():
            bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tid_in_wg = T.thread_id([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            self.init_profiler(profiler_buffer)
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data))
                self.device_init_all(smem_manager)
                self.class_init_all(smem_manager)

                # initialize event tensors
                self.set_events(batch_size, Semaphore, etensor_qkv_partial_global, etensor_notify_attn_global, etensor_attn_merge_global, etensor_o_proj_global, etensor_o_partial_global, etensor_o_allreduce_global, etensor_attn_add_rms_global, etensor_attn_mlp_global, etensor_gate_up_proj_reduce_global, etensor_gate_up_proj_global, etensor_down_proj_global, etensor_down_proj_reduce_global, etensor_down_proj_allreduce_global, etensor_mlp_add_rms_global, etensor_end_global)

                # initialize tile scheduler and smem_manager
                self.init_tile_scheduler(issubclass(Scheduler, StaticTileScheduler), smem_manager, exec_queue, exec_task, exec_head, exec_tail)
                smem_manager.init()

                while self.tile_scheduler.valid():
                    is_dynamic_sch = T.meta_var(issubclass(Scheduler, DynamicTileScheduler))
                    if self.tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                        self.task_impl_gemm_qkv_proj(is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.Q_REDUCE_RMS_ROPE.value:
                        self.task_impl_q_reduce_rms_rope(batch_size, inverse_indptr_global, inverse_indices_global, partial_qkv_global, qkv_global, q_rms_weight_global, rope_pos_global, cos_sin_cache_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.K_REDUCE_RMS_ROPE_APPEND.value:
                        self.task_impl_k_reduce_rms_rope_append(batch_size, inverse_indptr_global, inverse_indices_global, partial_qkv_global, k_rms_weight_global, rope_pos_global, cos_sin_cache_global, append_pos_global, kv_cache_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.V_REDUCE_APPEND.value:
                        self.task_impl_v_reduce_append(batch_size, inverse_indptr_global, inverse_indices_global, partial_qkv_global, kv_cache_global, append_pos_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.BATCH_ATTENTION.value:
                        self.task_impl_batch_attention(smem_manager, qkv_global, kv_cache_global, q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global, q_len_global, kv_len_global, q_start_global, kv_start_global, kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global, o_global, o_partial_attn_global, lse_partial_attn_global, num_qo_len_global, merge_indptr_global, merge_o_indices_global, batch_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.BATCH_ATTENTION_MERGE.value:
                        self.task_impl_batch_attention_merge(batch_size, o_partial_attn_global, o_global, lse_partial_attn_global, num_qo_len_global, merge_indptr_global, merge_o_indices_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                        self.task_impl_gemm_o_proj(batch_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                        self.task_impl_gemm_o_reduce(batch_size, partial_o_global, hidden_state_attn_mlp_global, before_o_allreduce_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.O_ALLREDUCE.value:
                        self.task_impl_o_allreduce(batch_size, before_o_allreduce_global, hidden_state_attn_mlp_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                        self.task_impl_attn_add_rms_norm(hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                        self.task_impl_gemm_gate_up_proj(is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GATE_UP_SILU.value:
                        self.task_impl_gate_up_silu(is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GATE_UP_PROJ_REDUCE.value:
                        self.task_impl_gate_up_proj_reduce(partial_out_gate_up_proj_global, out_gate_up_proj_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                        self.task_impl_split_silu_multiply(out_gate_up_proj_global, out_silu_multiply_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                        self.task_impl_gemm_down_proj(batch_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                        self.task_impl_down_proj_reduce(batch_size, partial_sum_down_proj_global, output_global, before_down_proj_allreduce_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.DOWN_PROJ_ALLREDUCE.value:
                        self.task_impl_down_proj_allreduce(batch_size, before_down_proj_allreduce_global, output_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                        self.task_impl_mlp_add_rms_norm(output_global, residual_global, mlp_add_rms_weight_global, is_dynamic_sch)
                    else:
                        trap_when_assert_failed(False)
                    smem_manager.exit_tile_runtime()
                    self.tile_scheduler.next_tile()
                if self.profiler_on:
                    self.profiler.finalize(lane_id == 0)
                self.class_finalize_all()

    # fmt: on

    # FIXME: change offset_factor to 0 can make performance better
    #       but it requires change on engine side
    def _get_func_static(self):
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
            inverse_indptr_ptr: T.handle, # read-only
            inverse_indices_ptr: T.handle, # read-only

            # intermediate buffer
            partial_qkv_ptr: T.handle, # intermediate
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
            etensor_notify_attn_ptr: T.handle,
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
            etensor_end_ptr: T.handle,

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
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float32" if self.world_size == 1 else "float16", scope="global")
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
            inverse_indptr_global = T.match_buffer(inverse_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            inverse_indices_global = T.match_buffer(inverse_indices_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)

            # intermediate buffer
            partial_qkv_global = T.match_buffer(partial_qkv_ptr, [self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM],
                                    "float32", scope="global")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM],
                                    "float16", scope="global")
            o_global = T.match_buffer(o_ptr, [batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM],
                                    "float16", scope="global")
            o_partial_attn_global = T.match_buffer(o_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM],
                                    "float32", scope="global")
            lse_partial_attn_global = T.match_buffer(lse_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS],
                                    "float32", scope="global")
            partial_o_global = T.match_buffer(partial_o_ptr, [self.SPLIT_O_PROJECT, batch_size, self.HIDDEN_SIZE],
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
            etensor_notify_attn_global = T.match_buffer(etensor_notify_attn_ptr, [KernelConfig.SM_NUMBER], "int32", scope="global", offset_factor=1)
            etensor_attn_merge_global = T.match_buffer(etensor_attn_merge_ptr, [batch_size * self.NUM_KEY_VALUE_HEADS],
                                    "int32", scope="global", offset_factor=1)
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [self.SPLIT_O_PROJECT], "int32", scope="global", offset_factor=1)
            etensor_o_partial_global = T.match_buffer(etensor_o_partial_ptr, [self.HIDDEN_SIZE// GemmTile.BLK_N], 
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
            etensor_end_global = T.match_buffer(etensor_end_ptr, [1], "int32", scope="global", offset_factor=1)

            # exec queue
            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")

            @T.macro
            def run(BLK_M):
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, qkv_proj_weight_global, o_proj_weight_global,
                    q_rms_weight_global, k_rms_weight_global, gate_up_weight_global, down_weight_global, attn_add_rms_weight_global,
                    mlp_add_rms_weight_global, cos_sin_cache_global, rope_pos_global, kv_cache_global, append_pos_global,
                    q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global, q_len_global, kv_len_global, q_start_global,
                    kv_start_global, kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global, num_qo_len_global,
                    merge_indptr_global, merge_o_indices_global, inverse_indptr_global, inverse_indices_global, partial_qkv_global,
                    qkv_global, o_global, o_partial_attn_global, lse_partial_attn_global, partial_o_global, before_o_allreduce_global,
                    hidden_state_attn_mlp_global, partial_out_gate_up_proj_global, out_gate_up_proj_global, out_silu_multiply_global,
                    partial_sum_down_proj_global, before_down_proj_allreduce_global, etensor_qkv_partial_global, etensor_notify_attn_global,
                    etensor_attn_merge_global, etensor_o_proj_global, etensor_o_partial_global, etensor_o_allreduce_global, etensor_attn_add_rms_global,
                    etensor_attn_mlp_global, etensor_gate_up_proj_reduce_global, etensor_gate_up_proj_global, etensor_down_proj_global,
                    etensor_down_proj_reduce_global, etensor_down_proj_allreduce_global, etensor_mlp_add_rms_global, etensor_end_global,
                    profiler_buffer, exec_queue, None, None, None, BLK_M,
                    static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
                )

            if batch_size <= 32:
                run(32)
            elif batch_size <= 64:
                run(64)
            else:
                run(128)
            # fmt: on
        return main
    
    def _get_func_dynamic(self):
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
            inverse_indptr_ptr: T.handle, # read-only
            inverse_indices_ptr: T.handle, # read-only

            # intermediate buffer
            partial_qkv_ptr: T.handle, # intermediate
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
            etensor_notify_attn_ptr: T.handle,
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
            etensor_end_ptr: T.handle,

            # execution queue
            queue_tasks_ptr: T.handle,
            queue_head_ptr: T.handle,
            queue_tail_ptr: T.handle,
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
            attn_tile_num = T.int32()

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float32" if self.world_size == 1 else "float16", scope="global")
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
            inverse_indptr_global = T.match_buffer(inverse_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            inverse_indices_global = T.match_buffer(inverse_indices_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)

            # intermediate buffer
            partial_qkv_global = T.match_buffer(partial_qkv_ptr, [self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM],
                                    "float32", scope="global")
            qkv_global = T.match_buffer(qkv_ptr, [batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM],
                                    "float16", scope="global")
            o_global = T.match_buffer(o_ptr, [batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM],
                                    "float16", scope="global")
            o_partial_attn_global = T.match_buffer(o_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM],
                                    "float32", scope="global")
            lse_partial_attn_global = T.match_buffer(lse_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS],
                                    "float32", scope="global")
            partial_o_global = T.match_buffer(partial_o_ptr, [self.SPLIT_O_PROJECT, batch_size, self.HIDDEN_SIZE],
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
            etensor_notify_attn_global = T.match_buffer(etensor_notify_attn_ptr, [attn_tile_num], "int32", scope="global", offset_factor=1)
            etensor_attn_merge_global = T.match_buffer(etensor_attn_merge_ptr, [batch_size * self.NUM_KEY_VALUE_HEADS],
                                    "int32", scope="global", offset_factor=1)
            etensor_o_proj_global = T.match_buffer(etensor_o_proj_ptr, [self.SPLIT_O_PROJECT], "int32", scope="global", offset_factor=1)
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
            etensor_end_global = T.match_buffer(etensor_end_ptr, [1], "int32", scope="global", offset_factor=1)

            # exec queue
            queue_tasks_global = T.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = T.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = T.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)

            @T.macro
            def run(BLK_M):
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, qkv_proj_weight_global, o_proj_weight_global,
                    q_rms_weight_global, k_rms_weight_global, gate_up_weight_global, down_weight_global, attn_add_rms_weight_global,
                    mlp_add_rms_weight_global, cos_sin_cache_global, rope_pos_global, kv_cache_global, append_pos_global,
                    q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global, q_len_global, kv_len_global, q_start_global,
                    kv_start_global, kv_end_global, kv_head_idx_global, work_indptr_global, len_kv_chunk_global, num_qo_len_global,
                    merge_indptr_global, merge_o_indices_global, inverse_indptr_global, inverse_indices_global, partial_qkv_global,
                    qkv_global, o_global, o_partial_attn_global, lse_partial_attn_global, partial_o_global, before_o_allreduce_global,
                    hidden_state_attn_mlp_global, partial_out_gate_up_proj_global, out_gate_up_proj_global, out_silu_multiply_global,
                    partial_sum_down_proj_global, before_down_proj_allreduce_global, etensor_qkv_partial_global, etensor_notify_attn_global,
                    etensor_attn_merge_global, etensor_o_proj_global, etensor_o_partial_global, etensor_o_allreduce_global, etensor_attn_add_rms_global,
                    etensor_attn_mlp_global, etensor_gate_up_proj_reduce_global, etensor_gate_up_proj_global, etensor_down_proj_global,
                    etensor_down_proj_reduce_global, etensor_down_proj_allreduce_global, etensor_mlp_add_rms_global, etensor_end_global,
                    profiler_buffer, None, queue_tasks_global, queue_head_global, queue_tail_global, BLK_M,
                    dynamic_scheduler.Semaphore, dynamic_scheduler.DynamicTileScheduler
                )

            if batch_size <= 32:
                run(32)
            elif batch_size <= 64:
                run(64)
            else:
                run(128)
            # fmt: on
        return main

    def get_func(self, scheduler: Literal["static", "dynamic"]):
        if scheduler == "static":
            return self._get_func_static()
        elif scheduler == "dynamic":
            return self._get_func_dynamic()
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    def get_module(self, scheduler: Literal["static", "dynamic"]):

        @I.ir_module(tirp=True)
        class Module:

            @T.prim_func(tirp=True)
            def main():
                pass

        module: tvm.IRModule = Module
        if scheduler == "static":
            module.update_func(module.get_global_var("main"), self._get_func_static())
        elif scheduler == "dynamic":
            module.update_func(module.get_global_var("main"), self._get_func_dynamic())
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        return module


arg_dict = {}
def prepare_data(batch_size, seq_len, mk: MegaKernelDenseLayer):
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
    
    pos = seq_len - 1
    assert pos < 4096  # for faster test
    arg_dict["rope_pos"] = torch.full((batch_size,), pos, dtype=torch.int32)

    # rope cos_sin_cache
    inv_freq = 1.0 / (mk.ROPE_THETA ** (torch.arange(0, mk.HEAD_DIM, 2, dtype=torch.float, device="cuda") / mk.HEAD_DIM))
    if mk.ROPE_SCALING is not None and mk.ROPE_SCALING["ROPE_TYPE"] == "llama3":
        # llama3 rope
        factor = mk.ROPE_SCALING["FACTOR"]
        low_freq_factor = mk.ROPE_SCALING["LOW_FREQ_FACTOR"]
        high_freq_factor = mk.ROPE_SCALING["HIGH_FREQ_FACTOR"]
        old_context_len = mk.ROPE_SCALING["ORIGINAL_MAX_POSITION_EMBEDDINGS"]
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    t = torch.arange(4096, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    arg_dict["cos_sin_cache"] = (
        torch.cat((cos, sin), dim=-1).reshape(-1, mk.HEAD_DIM).cpu()
    )

    # paged kv-cache
    page_last_len = mk.PAGE_SIZE if seq_len % mk.PAGE_SIZE == 0 else seq_len % mk.PAGE_SIZE
    page_num = ceildiv(seq_len, mk.PAGE_SIZE)
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
        append_pos[i] = seq_len - 1
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
        torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
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
def test(batch_size, seq_len, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper, sess):
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelDenseLayer, scheduler: Literal["static", "dynamic"]):
        import torch
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        # preprocess the buffer used in the tir kernel
        if mk.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            # reorder the gate_up_weight for the fusion of gate_up projection and silu
            new_order_indices = np.stack(
                (
                    np.arange(mk.INTERMEDIATE_SIZE).reshape(-1, 16),
                    np.arange(mk.INTERMEDIATE_SIZE, mk.INTERMEDIATE_SIZE * 2).reshape(-1, 16)
                ), axis=1
            ).reshape(-1)
            if mk.world_size > 1:
                gate_up_weight = arg_dict["gate_up_weight"][:, new_order_indices, :]
            else:
                gate_up_weight = arg_dict["gate_up_weight"][new_order_indices, :]
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
            np.zeros([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE], dtype=np.float32), DEV
        )
        tvm_arg_dict["before_o_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )
        tvm_arg_dict["partial_out_gate_up_proj"] = tvm.runtime.tensor(
            np.zeros([mk.GATE_UP_PROJ_SPLIT_K_FACTOR, batch_size, mk.INTERMEDIATE_SIZE * 2], dtype=np.float32), DEV
        )
        tvm_arg_dict["before_down_proj_allreduce"] = tvm.runtime.tensor(
            np.zeros([batch_size, mk.HIDDEN_SIZE], dtype=np.float16), DEV
        )
        res = get_inverse_plan_info(batch_size, mk.NUM_KEY_VALUE_HEADS, arg_dict["q_indptr"], arg_dict["kv_head_idx"], arg_dict["attn_task_num"].item())
        tvm_arg_dict["inverse_indptr"], tvm_arg_dict["inverse_indices"] = res

        if scheduler == "static":
            # static schedule
            # generate_exec_queue_static = tvm.get_global_func("megakernel.generate_exec_queue_static")
            # exec_queue = generate_exec_queue_static(
            #     batch_size,
            #     arg_dict["attn_task_num"].item(),
            #     mk.world_size,
            #     mk.NUM_ATTENTION_HEADS,
            #     mk.NUM_KEY_VALUE_HEADS,
            #     mk.HEAD_DIM,
            #     DEV,
            #     tvm.cpu(),
            # )
            exec_queue = generate_exec_queue(batch_size, arg_dict["attn_task_num"].item(), mk.config, mk.world_size, "static")
            tvm_arg_dict[f"exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            exec_queue = generate_exec_queue(None, None, mk.config, mk.world_size, "dynamic")
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

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
        if mk.GATE_UP_PROJ_SPLIT_K_FACTOR == 1:
            tvm_arg_dict["gate_up_weight"] = tvm.runtime.tensor(gate_up_weight, device=DEV)
        for i in range(REPEAT):
            if mk.world_size == 1:
                tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=DEV)
            else:
                tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
            # generate event tensor
            (
                tvm_arg_dict[f"etensor_qkv_partial_{i}"],
                tvm_arg_dict[f"etensor_notify_attn_{i}"],
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
                tvm_arg_dict[f"etensor_end_{i}"],
            ) = generate_event_tensor(
                batch_size,
                arg_dict["attn_task_num"].item(),
                tvm_arg_dict["kv_head_idx"],
                tvm_arg_dict["q_indptr"],
                mk.config,
                mk.world_size,
            )
        tvm_arg_dict[f"profiler_buffer"] = tvm.runtime.tensor(np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV)

        if mk.world_size > 1:
            nvshmem_malloc_hook = sess.get_global_func("runtime.disco.nvshmem.empty")
            tensor_to_gather = [
                "qkv_proj_weight",
                "o_proj_weight",
                "gate_up_weight",
                "down_weight",
                "kv_cache",
            ]
            disco_arg_dict = {}
            for key, value in tvm_arg_dict.items():
                if not isinstance(value, tvm.runtime.Tensor):
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
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["hidden_state_attn_mlp"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["before_down_proj_allreduce"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )
            disco_arg_dict["output"] = nvshmem_malloc_hook(
                ShapeTuple((batch_size, mk.HIDDEN_SIZE)), "float16", None
            )

            res_dict = {
                "output_host": tvm.runtime.empty((batch_size, mk.HIDDEN_SIZE), "float16", device=DEV),
                "residual_host": tvm.runtime.empty((batch_size, mk.HIDDEN_SIZE), "float16", device=DEV),
                "hidden_state_attn_mlp_host": tvm.runtime.empty(
                    (mk.world_size, batch_size, mk.HIDDEN_SIZE), "float16", device=DEV
                ),
                "hidden_state_attn_mlp_res": sess.empty(
                    (mk.world_size, batch_size, mk.HIDDEN_SIZE), "float16", worker0_only=True
                ),
                "profiler_buffer_host": tvm.runtime.empty(
                    (mk.world_size, mk.PROFILER_BUFFER_SIZE), "uint64", device=DEV
                ),
                "profiler_buffer_res": sess.empty(
                    (mk.world_size, mk.PROFILER_BUFFER_SIZE), "uint64", worker0_only=True
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
                (mega_kernel_static if scheduler == "static" else mega_kernel_dynamic).export_library(path)
                rt_mod = sess.load_vm_module(path)
                sess._sync_all()

        # run
        with target:
            iter = 0

            if scheduler == "static":
                kernel = mega_kernel_static["main"] if mk.world_size == 1 else rt_mod["main"]
                work_arg_dict = tvm_arg_dict if mk.world_size == 1 else disco_arg_dict
                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["gate_up_weight"],
                        work_arg_dict["down_weight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # page cache, cos_sin cache and plan info
                        work_arg_dict["cos_sin_cache"],
                        work_arg_dict["rope_pos"],
                        work_arg_dict["kv_cache"],
                        work_arg_dict["append_pos"],
                        work_arg_dict["q_indptr"],
                        work_arg_dict["kv_indptr"],
                        work_arg_dict["partial_indptr"],
                        work_arg_dict["page_kv_indices"],
                        work_arg_dict["q_len"],
                        work_arg_dict["kv_len"],
                        work_arg_dict["q_start"],
                        work_arg_dict["kv_start"],
                        work_arg_dict["kv_end"],
                        work_arg_dict["kv_head_idx"],
                        work_arg_dict["work_indptr"],
                        work_arg_dict["len_kv_chunk"],
                        work_arg_dict["num_qo_len"],
                        work_arg_dict["merge_indptr"],
                        work_arg_dict["merge_o_indices"],
                        work_arg_dict["inverse_indptr"],
                        work_arg_dict["inverse_indices"],
                        # intermediate buffer
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        work_arg_dict["partial_out_gate_up_proj"],
                        work_arg_dict["out_gate_up_proj"],
                        work_arg_dict["out_silu_multiply"],
                        work_arg_dict["partial_sum_down_proj"],
                        work_arg_dict["before_down_proj_allreduce"],
                        # event tensor
                        work_arg_dict[f"etensor_qkv_partial_{iter}"],
                        work_arg_dict[f"etensor_notify_attn_{iter}"],
                        work_arg_dict[f"etensor_attn_merge_{iter}"],
                        work_arg_dict[f"etensor_o_proj_{iter}"],
                        work_arg_dict[f"etensor_o_partial_{iter}"],
                        work_arg_dict[f"etensor_o_allreduce_{iter}"],
                        work_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                        work_arg_dict[f"etensor_attn_mlp_{iter}"],
                        work_arg_dict[f"etensor_gate_up_proj_reduce_{iter}"],
                        work_arg_dict[f"etensor_gate_up_proj_{iter}"],
                        work_arg_dict[f"etensor_down_proj_{iter}"],
                        work_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                        work_arg_dict[f"etensor_down_proj_allreduce_{iter}"],
                        work_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                        work_arg_dict[f"etensor_end_{iter}"],
                        # exec queue
                        work_arg_dict["exec_queue"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1
            else:
                kernel = mega_kernel_dynamic["main"] if mk.world_size == 1 else rt_mod["main"]
                work_arg_dict = tvm_arg_dict if mk.world_size == 1 else disco_arg_dict
                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["gate_up_weight"],
                        work_arg_dict["down_weight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # page cache, cos_sin cache and plan info
                        work_arg_dict["cos_sin_cache"],
                        work_arg_dict["rope_pos"],
                        work_arg_dict["kv_cache"],
                        work_arg_dict["append_pos"],
                        work_arg_dict["q_indptr"],
                        work_arg_dict["kv_indptr"],
                        work_arg_dict["partial_indptr"],
                        work_arg_dict["page_kv_indices"],
                        work_arg_dict["q_len"],
                        work_arg_dict["kv_len"],
                        work_arg_dict["q_start"],
                        work_arg_dict["kv_start"],
                        work_arg_dict["kv_end"],
                        work_arg_dict["kv_head_idx"],
                        work_arg_dict["work_indptr"],
                        work_arg_dict["len_kv_chunk"],
                        work_arg_dict["num_qo_len"],
                        work_arg_dict["merge_indptr"],
                        work_arg_dict["merge_o_indices"],
                        work_arg_dict["inverse_indptr"],
                        work_arg_dict["inverse_indices"],
                        # intermediate buffer
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        work_arg_dict["partial_out_gate_up_proj"],
                        work_arg_dict["out_gate_up_proj"],
                        work_arg_dict["out_silu_multiply"],
                        work_arg_dict["partial_sum_down_proj"],
                        work_arg_dict["before_down_proj_allreduce"],
                        # event tensor
                        work_arg_dict[f"etensor_qkv_partial_{iter}"],
                        work_arg_dict[f"etensor_notify_attn_{iter}"],
                        work_arg_dict[f"etensor_attn_merge_{iter}"],
                        work_arg_dict[f"etensor_o_proj_{iter}"],
                        work_arg_dict[f"etensor_o_partial_{iter}"],
                        work_arg_dict[f"etensor_o_allreduce_{iter}"],
                        work_arg_dict[f"etensor_attn_add_rms_norm_{iter}"],
                        work_arg_dict[f"etensor_attn_mlp_{iter}"],
                        work_arg_dict[f"etensor_gate_up_proj_reduce_{iter}"],
                        work_arg_dict[f"etensor_gate_up_proj_{iter}"],
                        work_arg_dict[f"etensor_down_proj_{iter}"],
                        work_arg_dict[f"etensor_down_proj_reduce_{iter}"],
                        work_arg_dict[f"etensor_down_proj_allreduce_{iter}"],
                        work_arg_dict[f"etensor_mlp_add_rms_norm_{iter}"],
                        work_arg_dict[f"etensor_end_{iter}"],
                        # exec queue
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            # post process
            if mk.world_size == 1:
                ms = bench(func, warmup=1, repeat=3, proton_name=f"tir-{scheduler}")
                print(f"TIR time: {ms:.3f} ms")
                if mk.profiler_on:
                    export_to_perfetto_trace(
                        tvm_arg_dict[f"profiler_buffer"].numpy(),
                        f"{scheduler}-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                        event_type_names,
                    )
                return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy().astype(np.float16)
            else:
                for i in range(REPEAT):
                    func()
                sess._sync_all()
                sess.copy_from_worker_0(res_dict["output_host"], disco_arg_dict["output"])
                sess.copy_from_worker_0(res_dict["residual_host"], disco_arg_dict[f"residual_0"])
                # sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], disco_arg_dict["hidden_state_attn_mlp"])
                sess.gather_to_worker0(disco_arg_dict["hidden_state_attn_mlp"], res_dict["hidden_state_attn_mlp_res"])
                sess.copy_from_worker_0(res_dict["hidden_state_attn_mlp_host"], res_dict["hidden_state_attn_mlp_res"])
                sess.gather_to_worker0(disco_arg_dict[f"profiler_buffer"], res_dict["profiler_buffer_res"])
                sess.copy_from_worker_0(res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"])
                sess._sync_all()
                if mk.profiler_on:
                    for r in range(mk.world_size):
                        export_to_perfetto_trace(
                            res_dict["profiler_buffer_host"].numpy()[r],
                            f"{scheduler}-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                            event_type_names,
                        )
                return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def std(arg_dict, use_prefill, mk: MegaKernelDenseLayer):
        import flashinfer
        import torch

        FULL_INTERMEDIATE_SIZE = mk.INTERMEDIATE_SIZE * mk.world_size
        FULL_NUM_ATTENTION_HEADS = mk.NUM_ATTENTION_HEADS * mk.world_size
        FULL_NUM_KEY_VALUE_HEADS = mk.NUM_KEY_VALUE_HEADS * mk.world_size

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                if mk.world_size > 1:
                    if key == "qkv_proj_weight":
                        split_sizes = [mk.NUM_ATTENTION_HEADS, mk.NUM_KEY_VALUE_HEADS, mk.NUM_KEY_VALUE_HEADS]
                        value = value.reshape(mk.world_size, -1, mk.HEAD_DIM, mk.HIDDEN_SIZE)
                        q_weight, k_weight, v_weight = torch.split(value, split_sizes, dim=1)
                        q_weight = q_weight.reshape(-1, mk.HIDDEN_SIZE)
                        k_weight = k_weight.reshape(-1, mk.HIDDEN_SIZE)
                        v_weight = v_weight.reshape(-1, mk.HIDDEN_SIZE)
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
            out_f = torch.zeros(
                batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM, dtype=torch.float16, device="cuda"
            )
            lse_f = torch.zeros(batch_size, FULL_NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda")
            if use_prefill:
                workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.int8).to(0)
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "HND")
                wrapper.plan(
                    torch.arange(0, batch_size + 1, dtype=torch.int32).to(0),
                    std_arg_dict["page_kv_indptr"],
                    std_arg_dict["page_kv_indices"],
                    std_arg_dict["page_kv_last_page_len"],
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
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
                    torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
                    FULL_NUM_ATTENTION_HEADS,
                    FULL_NUM_KEY_VALUE_HEADS,
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
                qkv, [FULL_NUM_ATTENTION_HEADS, FULL_NUM_KEY_VALUE_HEADS, FULL_NUM_KEY_VALUE_HEADS], dim=1
            )
            if mk.MODEL_NAME == "qwen3_32b":
                q = flashinfer.norm.rmsnorm(
                    input=q.reshape(-1, mk.HEAD_DIM),
                    weight=std_arg_dict["q_rms_wight"],
                    eps=mk.RMS_NORM_EPS,
                    enable_pdl=False,
                ).reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM)
                k = flashinfer.norm.rmsnorm(
                    input=k.reshape(-1, mk.HEAD_DIM),
                    weight=std_arg_dict["k_rms_wight"],
                    eps=mk.RMS_NORM_EPS,
                    enable_pdl=False,
                ).reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM)
                q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                    positions=std_arg_dict["rope_pos"],
                    query=q.reshape(batch_size, -1),
                    key=k.reshape(batch_size, -1),
                    head_size=mk.HEAD_DIM,
                    cos_sin_cache=std_arg_dict["cos_sin_cache"],
                    is_neox=True,
                )
            elif mk.MODEL_NAME == "llama3_1b":
                q, k = flashinfer.rope.apply_llama31_rope_pos_ids(
                    q=q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    k=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
                    pos_ids=std_arg_dict["rope_pos"],
                    rope_scale=mk.ROPE_SCALING["FACTOR"],
                    rope_theta=mk.ROPE_THETA,
                    low_freq_factor=mk.ROPE_SCALING["LOW_FREQ_FACTOR"],
                    high_freq_factor=mk.ROPE_SCALING["HIGH_FREQ_FACTOR"],
                    old_context_len=mk.ROPE_SCALING["ORIGINAL_MAX_POSITION_EMBEDDINGS"],
                )
            else:
                raise ValueError(f"Unsupported model name: {mk.MODEL_NAME}")
            flashinfer.page.append_paged_kv_cache(
                append_key=k.reshape(batch_size, FULL_NUM_KEY_VALUE_HEADS, mk.HEAD_DIM),
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
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM), std_arg_dict["kv_cache"]
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, FULL_NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, FULL_NUM_ATTENTION_HEADS * mk.HEAD_DIM),
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

    def run():
        if mega_kernel_static["main"] is not None:
            output_tir_static, residual_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("static tir finish", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            output_tir_dynamic, residual_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("dynamic tir finish", flush=True)
        output_std1, residual_std1 = std(arg_dict, use_prefill=True, mk=mega_kernel_wrapper)
        output_std2, residual_std2 = std(arg_dict, use_prefill=False, mk=mega_kernel_wrapper)

        # this assert will fail on latest flashinfer version
        # np.testing.assert_allclose(output_std1, output_std2, rtol=1e-3, atol=1e-2)
        # np.testing.assert_allclose(residual_std1, residual_std2, rtol=1e-3, atol=1e-2)
        if mega_kernel_static["main"] is not None:
            np.testing.assert_allclose(output_tir_static, output_std1, rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(residual_tir_static, residual_std1, rtol=1e-3, atol=1e-2)
            print("static pass", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            np.testing.assert_allclose(output_tir_dynamic, output_std1, rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(residual_tir_dynamic, residual_std1, rtol=1e-3, atol=1e-2)
            print("dynamic pass", flush=True)

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("blackwell_layer"):
            run()
    else:
        run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument("--model", type=str, default="qwen3_32b", choices=["qwen3_32b", "llama3_1b"],
                        help="The supporting model.")
    parser.add_argument("--scheduler", type=str, nargs='+', default=["static", "dynamic"],
                        choices=["static", "dynamic", "none"],
                        help="A list of test methods to run: 'static' or 'dynamic'.")
    parser.add_argument("--world-size", type=int, default=1,
                        help="The number of devices for the world size.")
    parser.add_argument("--batch-size", type=int, nargs='+',
                        default=[1, 3, 7, 15, 31, 63, 127, 128],
                        help="A list of batch sizes to test.")
    parser.add_argument("--seq-len", type=int, nargs='+', default=[512],
                        help="A list of sequence lengths to test.")
    parser.add_argument("--profiler-on", action="store_true",
                        help="Enable the profiler.")
    args = parser.parse_args()

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelDenseLayer(config=get_model_config(args.model), world_size=args.world_size, profiler_on=args.profiler_on)
    if "static" in testing_scheduler:
        mega_static_module = mega_kernel_wrapper.get_module("static")
        src, lib_static = get_source(mega_static_module)
        print(src)
    else:
        lib_static = {"main": None}
    if "dynamic" in testing_scheduler:
        mega_dynamic_module = mega_kernel_wrapper.get_module("dynamic")
        src, lib_dynamic = get_source(mega_dynamic_module)
        print(src)
    else:
        lib_dynamic = {"main": None}
    if mega_kernel_wrapper.world_size > 1:
        devices = list(np.arange(mega_kernel_wrapper.world_size))
        sess = di.ProcessSession(num_workers=mega_kernel_wrapper.world_size)
        sess.init_ccl(tvm.get_global_func("runtime.disco.compiled_ccl")(), *devices)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, mega_kernel_wrapper.world_size, 0)
        sess.sync_worker_0()
    else:
        sess = None
    for batch_size in args.batch_size:
        print(f"batch_size: {batch_size}", flush=True)
        for seq_len in args.seq_len:
            print(f"seq_len: {seq_len}", flush=True)
            test(batch_size, seq_len, lib_static, lib_dynamic, mega_kernel_wrapper, sess)