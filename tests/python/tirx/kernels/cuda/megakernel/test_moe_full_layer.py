import argparse
import math
import tempfile
import functools
from typing import Literal, Type, Union

import flashinfer
import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as Tx
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler

from tvm.tirx.megakernel.utils.config import ProfileEventType, KernelConfig, qwen3_30b_a3b_config, JobType, event_type_names
from tvm.tirx.megakernel.utils.base import MegaKernelWrapper, SmemManager
from tvm.tirx.megakernel.utils.utils import get_source, ceildiv
from tvm.tirx.megakernel.utils import static_scheduler, dynamic_scheduler
from tvm.tirx.megakernel.utils.static_scheduler import StaticTileScheduler
from tvm.tirx.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tvm.tirx.megakernel.kernels import (
    AddRMSNormTile, RMSNormTile, AllreduceTile, BatchAttnTile, BatchMergeTile,
    GemmTile, SplitKReduceTile, SplitKReduceRMSnormRopeQTile,
    SplitKReduceRMSnormRopeAppendKTile, SplitKReduceAppendVTile
)
from tvm.tirx.megakernel.utils.support import (
    generate_exec_queue,
    get_inverse_plan_info,
    get_max_num_tokens_padded,
)

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sgl_kernel import topk_softmax
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as fused_moe_triton
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe import MoeRunnerConfig
import flashinfer.fused_moe as fused_moe
from triton import language as tl

# Import parent classes
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from test_layer import MegaKernelDenseLayer
from test_moe import MegaKernelMOE


class MegaKernelMOEFullLayer(MegaKernelDenseLayer, MegaKernelMOE):
    """
    Full transformer layer combining attention and MoE.
    Inherits from both MegaKernelDenseLayer and MegaKernelMOE to reuse implementations.
    """

    def __init__(self, config, world_size, profiler_on):
        assert world_size == 1, "Currently only support world_size=1"
        # Only initialize MegaKernelWrapper once through MegaKernelDenseLayer
        MegaKernelDenseLayer.__init__(self, config, world_size, profiler_on)
        # Set MOE_M_PAD_SIZE from MegaKernelMOE
        self.MOE_M_PAD_SIZE = MegaKernelMOE.MOE_M_PAD_SIZE

    def set_tiles(self, batch_size, BLK_M, low_batch=None):
        """
        Combine tiles from both attention and MoE.
        Keep attention tiles from MegaKernelDenseLayer and add MoE tiles.
        """
        # Initialize tile tracking
        self.reset()
        self.qkv_proj_tile = self._add_tile(
            GemmTile(
                (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM,
                self.HIDDEN_SIZE,
                "float16",
                "float16",
                self.SPLIT_QKV_PROJECT,
                BLK_M,
                BLK_M,
            ),
            ProfileEventType.GEMM_QKV_PROJ,
        )
        self.reduce_rms_rope_q_tile = self._add_tile(
            SplitKReduceRMSnormRopeQTile(
                batch_size,
                self.RMS_NORM_EPS,
                self.NUM_ATTENTION_HEADS,
                self.NUM_KEY_VALUE_HEADS,
                self.HEAD_DIM,
                self.SPLIT_QKV_PROJECT,
            ),
            ProfileEventType.Q_REDUCE_RMSNORM_ROPE,
        )
        self.reduce_rms_rope_append_k_tile = self._add_tile(
            SplitKReduceRMSnormRopeAppendKTile(
                batch_size,
                self.RMS_NORM_EPS,
                self.NUM_ATTENTION_HEADS,
                self.NUM_KEY_VALUE_HEADS,
                self.HEAD_DIM,
                self.SPLIT_QKV_PROJECT,
                self.PAGE_SIZE,
            ),
            ProfileEventType.K_REDUCE_RMSNORM_ROPE_APPEND,
        )
        self.reduce_append_v_tile = self._add_tile(
            SplitKReduceAppendVTile(
                batch_size,
                self.NUM_KEY_VALUE_HEADS,
                self.NUM_ATTENTION_HEADS,
                self.HEAD_DIM,
                self.SPLIT_QKV_PROJECT,
                self.PAGE_SIZE,
            ),
            ProfileEventType.V_REDUCE_APPEND,
        )
        self.attn_tile = self._add_tile(
            BatchAttnTile(
                self.PAGE_SIZE, self.NUM_ATTENTION_HEADS, self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM
            ),
            ProfileEventType.BATCH_ATTENTION,
        )
        self.merge_tile = self._add_tile(
            BatchMergeTile(self.HEAD_DIM, self.NUM_KEY_VALUE_HEADS, self.NUM_ATTENTION_HEADS),
            ProfileEventType.BATCH_ATTENTION_MERGE,
        )
        self.o_proj_tile = self._add_tile(
            GemmTile(
                self.HIDDEN_SIZE,
                self.NUM_ATTENTION_HEADS * self.HEAD_DIM,
                "float16",
                "float16",
                self.SPLIT_O_PROJECT,
                BLK_M,
                BLK_M,
                prefetch_on=True,
                use_tma_reduce=self.world_size == 1,
            ),
            ProfileEventType.GEMM_O_PROJ,
        )
        self.o_reduce_tile = self._add_tile(
            SplitKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.SPLIT_O_PROJECT),
            ProfileEventType.GEMM_O_REDUCE,
            predicate=self.world_size > 1,
        )
        self.o_allreduce_tile = self._add_tile(
            AllreduceTile(self.world_size),
            ProfileEventType.O_ALLREDUCE,
            predicate=self.world_size > 1,
        )
        self.attn_add_rms_tile = self._add_tile(
            AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE) if self.world_size > 1 
            else RMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE), 
            ProfileEventType.ATTN_ADD_RMS_NORM
        )
        self.mlp_add_rms_norm_tile = self._add_tile(
            AddRMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE) if self.world_size > 1
            else RMSNormTile(self.RMS_NORM_EPS, self.HIDDEN_SIZE),
            ProfileEventType.MLP_ADD_RMS_NORM
        )
        MegaKernelMOE._set_tiles(self, batch_size, low_batch)

    def set_events(
        self,
        is_dynamic_sch,
        batch_size,
        attn_task_num,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        etensor_workspace_global,
    ):
        """Combine events from both attention and MoE"""
        # Attention events (similar to MegaKernelDenseLayer.set_events)
        MegaKernelDenseLayer._set_events(
            self,
            batch_size,
            attn_task_num,
            Semaphore,
            etensor_workspace_global,
            ignore_mlp_part=True,
        )
        MegaKernelMOE._set_events(
            self,
            batch_size,
            Semaphore,
            etensor_workspace_global,
        )
        self.set_events_complete(is_dynamic_sch, Semaphore, etensor_workspace_global)
        self.num_etensors[is_dynamic_sch] = len(self.etensor_and_f_init_pairs)

    # Override attn_add_rms_norm to trigger MoE instead of MLP
    @Tx.macro
    def task_impl_attn_add_rms_norm(
        self,
        batch_size,
        hidden_state_attn_mlp_global,
        residual_global,
        attn_add_rms_weight_global,
        gating_output_global,
        output_global,
        is_dynamic_sch,
    ):
        """
        Override from MegaKernelDenseLayer to trigger MoE gating instead of gate_up_silu/gate_up_proj.
        """
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_attn_mlp, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (
                            JobType.MOE_GATING.value,
                            ceildiv(batch_size, 128) * self.GATING_SPLIT_K_FACTOR,
                            0,
                            push_idx // self.GATING_SPLIT_K_FACTOR,
                            push_idx % self.GATING_SPLIT_K_FACTOR,
                        )
                    ), "warpgroup", "warpgroup"
                )
            if self.world_size == 1:
                self.tile_scheduler.wait(self.evt_attn_add_rms, 0, wait_level="cta")
                Tx.cuda.thread_fence() # ensure previous tma-reduce are visible
            else:
                self.tile_scheduler.wait(self.evt_attn_add_rms, self.tile_scheduler.m_idx // self.o_allreduce_tile.M_TILE, wait_level="cta")
            self.run_tile(self.attn_add_rms_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx,
                            hidden_state_attn_mlp_global, residual_global, attn_add_rms_weight_global)
            # FIXME: this only works for num experts <= 1024
            if tid * 4 < self.NUM_EXPERTS:
                for vec in Tx.vectorized(4):
                    gating_output_global[self.tile_scheduler.m_idx, tid * 4 + vec] = 0
            # FIXME: this only works for hidden size <= 2048
            if tid * 8 < self.HIDDEN_SIZE:
                for vec in Tx.vectorized(8):
                    output_global[self.tile_scheduler.m_idx, tid * 8 + vec] = 0
            self.tile_scheduler.notify(self.evt_attn_mlp, lambda notify_idx: (1, -1, 0), scope="cta")

    @Tx.macro
    def task_impl_moe_gating(self, A, B, output, is_dynamic_sch):
        with Tx.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_gating, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (
                            JobType.MOE_TOPK_SOFTMAX.value, self.topk_softmax.PERSISTENT_SM_NUMBER, push_idx, 0, 0
                        )
                    ), "warpgroup", "warpgroup", scope_id=0
                )
            self.tile_scheduler.wait(self.evt_attn_mlp, 0, wait_level="warp")
            self.run_tile(
                self.gate,
                self.tile_scheduler.m_idx,
                self.tile_scheduler.n_idx,
                self.tile_scheduler.k_idx,
                A, B, output,
                self.profiler,
            )
            self.tile_scheduler.notify(self.evt_gating, lambda notify_idx: (1, -1, 0), scope="warpgroup", scope_id=0)

    @Tx.macro
    def task_impl_moe_group_gemm_down(
        self,
        A, B, output,
        batch_size,
        expert_ids_global,
        topk_weights_flattened,
        sorted_token_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        unfused,
        down_proj_task_size,
        is_dynamic_sch,
    ):
        with Tx.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_group_gemm_down, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MLP_ADD_RMS_NORM.value, batch_size, push_idx, 0, 0)
                    ), "warpgroup", "warpgroup"
                )
            wait_idx = Tx.meta_var(self.tile_scheduler.m_idx if not unfused else 0)
            self.tile_scheduler.wait(self.evt_group_gemm_gate_up, wait_idx, wait_level="warp")
            if (
                is_dynamic_sch
                or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE
            ):
                for i in range(down_proj_task_size):
                    self.run_tile(
                        self.group_gemm_down,
                        self.tile_scheduler.m_idx,
                        self.tile_scheduler.n_idx * down_proj_task_size + i,
                        self.tile_scheduler.k_idx,
                        A, B, output,
                        expert_ids_global,
                        topk_weights_flattened,
                        sorted_token_ids_global,
                        num_valid_tokens_global,
                        self.profiler,
                    )
            self.tile_scheduler.notify(self.evt_group_gemm_down, lambda notify_idx: (1, -1, 0), scope="warpgroup", scope_id=0)

    @Tx.macro
    def task_impl_mlp_add_rms_norm(
        self,
        output_global,
        residual_global,
        mlp_add_rms_weight_global,
        is_dynamic_sch,
    ):
        with Tx.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_end, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.END.value, KernelConfig.SM_NUMBER, 0, 0, 0)
                    ), "cta", "cta"
                )
            self.tile_scheduler.wait(self.evt_group_gemm_down, 0, wait_level="cta")
            self.run_tile(
                self.mlp_add_rms_norm_tile,
                self.tile_scheduler.m_idx,
                self.tile_scheduler.n_idx,
                self.tile_scheduler.k_idx,
                output_global,
                residual_global,
                mlp_add_rms_weight_global,
            )

    # The rest of the task implementations are inherited from parent classes
    # - Attention tasks from MegaKernelDenseLayer
    # - MoE tasks from MegaKernelMOE

    # We need to implement the combined fused_body
    @Tx.macro
    def fused_body(
        self,
        batch_size,
        hidden_state_global,
        residual_global,
        output_global,
        # Attention weights
        qkv_proj_weight_global,
        o_proj_weight_global,
        q_rms_weight_global,
        k_rms_weight_global,
        attn_add_rms_weight_global,
        mlp_add_rms_weight_global,
        # MoE weights
        gate_weight_global,
        grp_gate_up_weight_global,
        grp_down_weight_global,
        # Attention cache and plan
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
        # Attention intermediate buffers
        partial_qkv_global,
        qkv_global,
        o_global,
        o_partial_attn_global,
        lse_partial_attn_global,
        partial_o_global,
        before_o_allreduce_global,
        hidden_state_attn_mlp_global,
        # MoE intermediate buffers
        gating_output_global,
        topk_weights_global,
        topk_indices_global,
        sorted_token_ids_global,
        expert_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        cumsum_buffer_global,
        reordered_hidden_state_global,
        gate_up_output_global,
        silu_mul_output_global,
        # event tensors
        etensor_workspace_global,
        # Profiler and execution queue
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        BLK_M,
        down_proj_task_size,
        low_batch,
        unfused,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        Scheduler: Type[
            Union[static_scheduler.StaticTileScheduler, dynamic_scheduler.DynamicTileScheduler]
        ],
    ):
 
        # Initialize tiles
        self.set_tiles(batch_size, BLK_M, low_batch)
        self.host_init_all()

        with Tx.kernel():
            bx = Tx.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            tid = Tx.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tid_in_wg = Tx.thread_id(
                [KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup"
            )
            lane_id = Tx.thread_id([32], parent="warp")
            self.init_profiler(profiler_buffer)

            with Tx.cta():
                buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                # initialize smem manager
                self.set_smem_manager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data)

                # initialize device
                self.device_init_all(self.smem_manager)
                self.class_init_all(self.smem_manager)

                # Initialize event tensors
                attn_task_num = Tx.meta_var(work_indptr_global[KernelConfig.SM_NUMBER * KernelConfig.WG_NUMBER])
                self.set_events(
                    issubclass(Scheduler, DynamicTileScheduler),
                    batch_size,
                    attn_task_num,
                    Semaphore,
                    etensor_workspace_global,
                )
                
                # initialize tile scheduler and smem_manager
                if issubclass(Scheduler, static_scheduler.StaticTileScheduler):
                    self.init_tile_scheduler(False, Scheduler, "layer", exec_queue, self.smem_manager)
                else:
                    self.init_tile_scheduler(True, Scheduler, exec_task, exec_head, exec_tail, self.smem_manager, self.profiler)
                self.smem_manager.init()

                topk_ids_flattened = topk_indices_global.view(-1)
                topk_weights_flattened = topk_weights_global.view(-1)

                while self.tile_scheduler.valid():
                    is_dynamic_sch = Tx.meta_var(issubclass(Scheduler, DynamicTileScheduler))

                    # Attention tasks (inherited from MegaKernelDenseLayer)
                    if self.tile_scheduler.task_type == JobType.GEMM_QKV_PROJ.value:
                        self.task_impl_gemm_qkv_proj(
                            hidden_state_global,
                            qkv_proj_weight_global,
                            partial_qkv_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.Q_REDUCE_RMS_ROPE.value:
                        self.task_impl_q_reduce_rms_rope(
                            batch_size,
                            inverse_indptr_global,
                            inverse_indices_global,
                            partial_qkv_global,
                            qkv_global,
                            q_rms_weight_global,
                            rope_pos_global,
                            cos_sin_cache_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.K_REDUCE_RMS_ROPE_APPEND.value:
                        self.task_impl_k_reduce_rms_rope_append(
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
                        )
                    elif self.tile_scheduler.task_type == JobType.V_REDUCE_APPEND.value:
                        self.task_impl_v_reduce_append(
                            batch_size,
                            inverse_indptr_global,
                            inverse_indices_global,
                            partial_qkv_global,
                            kv_cache_global,
                            append_pos_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.BATCH_ATTENTION.value:
                        self.task_impl_batch_attention(
                            self.smem_manager,
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
                        )
                    elif self.tile_scheduler.task_type == JobType.BATCH_ATTENTION_MERGE.value:
                        self.task_impl_batch_attention_merge(
                            batch_size,
                            o_partial_attn_global,
                            o_global,
                            lse_partial_attn_global,
                            num_qo_len_global,
                            merge_indptr_global,
                            merge_o_indices_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.GEMM_O_PROJ.value:
                        self.task_impl_gemm_o_proj(
                            batch_size,
                            o_global.view(-1, self.NUM_ATTENTION_HEADS * self.HEAD_DIM),
                            o_proj_weight_global,
                            partial_o_global if self.world_size > 1 else residual_global,
                            is_dynamic_sch
                        )
                    elif self.tile_scheduler.task_type == JobType.GEMM_O_REDUCE.value:
                        self.task_impl_gemm_o_reduce(
                            batch_size,
                            partial_o_global,
                            hidden_state_attn_mlp_global,
                            before_o_allreduce_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.ATTN_ADD_RMS_NORM.value:
                        self.task_impl_attn_add_rms_norm(
                            batch_size,
                            hidden_state_attn_mlp_global,
                            residual_global,
                            attn_add_rms_weight_global,
                            gating_output_global,
                            output_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.INIT_ETENSOR.value:
                        self.task_impl_init_etensor(is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.WAIT_ETENSOR_INITx.value:
                        self.task_impl_wait_etensor_init_complete(is_dynamic_sch)
                    else:
                        break
                    self.tile_scheduler.next_tile()

                while self.tile_scheduler.valid():
                    is_dynamic_sch = Tx.meta_var(issubclass(Scheduler, DynamicTileScheduler))
                    # MoE tasks (inherited from MegaKernelMOE)
                    if self.tile_scheduler.task_type == JobType.MOE_GATING.value:
                        self.task_impl_moe_gating(
                            hidden_state_attn_mlp_global,
                            gate_weight_global,
                            gating_output_global,
                            is_dynamic_sch
                        )
                    elif self.tile_scheduler.task_type == JobType.MOE_TOPK_SOFTMAX.value:
                        self.task_impl_moe_topk_softmax(
                            gating_output_global,
                            topk_weights_global,
                            topk_indices_global,
                            is_dynamic_sch,
                            renormalize=True
                        )
                    elif self.tile_scheduler.task_type == JobType.MOE_ALIGN.value:
                        self.task_impl_moe_align(
                            topk_ids_flattened,
                            sorted_token_ids_global,
                            expert_ids_global,
                            num_tokens_post_pad_global,
                            cumsum_buffer_global,
                            num_valid_tokens_global,
                            down_proj_task_size,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.MOE_COUNT_AND_SORTx.value:
                        self.task_impl_moe_count_and_sort(
                            topk_ids_flattened,
                            sorted_token_ids_global,
                            cumsum_buffer_global,
                            hidden_state_attn_mlp_global,
                            reordered_hidden_state_global,
                            num_tokens_post_pad_global,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value:
                        self.task_impl_moe_group_gemm_gate_up_silu(
                            reordered_hidden_state_global,
                            grp_gate_up_weight_global,
                            silu_mul_output_global,
                            topk_weights_flattened,
                            sorted_token_ids_global,
                            expert_ids_global,
                            num_valid_tokens_global,
                            num_tokens_post_pad_global,
                            unfused,
                            down_proj_task_size,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_DOWN.value:
                        self.task_impl_moe_group_gemm_down(
                            silu_mul_output_global,
                            grp_down_weight_global,
                            output_global if self.world_size > 1 else residual_global,
                            batch_size,
                            expert_ids_global,
                            topk_weights_flattened,
                            sorted_token_ids_global,
                            num_valid_tokens_global,
                            num_tokens_post_pad_global,
                            unfused,
                            down_proj_task_size,
                            is_dynamic_sch,
                        )
                    elif self.tile_scheduler.task_type == JobType.MLP_ADD_RMS_NORM.value:
                        self.task_impl_mlp_add_rms_norm(
                            output_global,
                            residual_global,
                            mlp_add_rms_weight_global,
                            is_dynamic_sch,
                        )
                    else:
                        Tx.cuda.trap_when_assert_failed(False)

                    self.smem_manager.exit_tile_runtime()
                    self.tile_scheduler.next_tile()

                if self.profiler_on:
                    self.profiler.finalize(lane_id == 0)
                self.class_finalize_all()

    def get_func_dynamic(self):
        """Dynamic scheduler function combining attention and MoE"""
        # fmt: off
        @Tx.prim_func(tirx=True)
        def main(
            # Input and output
            hidden_state_ptr: Tx.handle,
            residual_ptr: Tx.handle,
            output_ptr: Tx.handle,
            
            # Attention weights
            qkv_proj_weight_ptr: Tx.handle,
            o_proj_weight_ptr: Tx.handle,
            q_rms_weight_ptr: Tx.handle,
            k_rms_weight_ptr: Tx.handle,
            attn_add_rms_weight_ptr: Tx.handle,
            mlp_add_rms_weight_ptr: Tx.handle,
            
            # MoE weights
            gate_weight_ptr: Tx.handle,
            grp_gate_up_weight_ptr: Tx.handle,
            grp_down_weight_ptr: Tx.handle,
            
            # Attention: page cache, cos_sin cache and plan info
            cos_sin_cache_ptr: Tx.handle,
            rope_pos_ptr: Tx.handle,
            kv_cache_ptr: Tx.handle,
            append_pos_ptr: Tx.handle,
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
            num_qo_len_ptr: Tx.handle,
            merge_indptr_ptr: Tx.handle,
            merge_o_indices_ptr: Tx.handle,
            inverse_indptr_ptr: Tx.handle,
            inverse_indices_ptr: Tx.handle,
            
            # Attention intermediate buffer
            partial_qkv_ptr: Tx.handle,
            qkv_ptr: Tx.handle,
            o_ptr: Tx.handle,
            o_partial_attn_ptr: Tx.handle,
            lse_partial_attn_ptr: Tx.handle,
            partial_o_ptr: Tx.handle,
            before_o_allreduce_ptr: Tx.handle,
            hidden_state_attn_mlp_ptr: Tx.handle,
            
            # MoE intermediate buffer
            gating_output_ptr: Tx.handle,
            topk_weights_ptr: Tx.handle,
            topk_indices_ptr: Tx.handle,
            sorted_token_ids_ptr: Tx.handle,
            expert_ids_ptr: Tx.handle,
            num_valid_tokens_ptr: Tx.handle,
            num_tokens_post_pad_ptr: Tx.handle,
            cumsum_buffer_ptr: Tx.handle,
            reordered_hidden_state_ptr: Tx.handle,
            gate_up_output_ptr: Tx.handle,
            silu_mul_output_ptr: Tx.handle,
            
            # event tensors
            etensor_workspace_ptr: Tx.handle,
            # Execution queue
            queue_tasks_ptr: Tx.handle,
            queue_head_ptr: Tx.handle,
            queue_tail_ptr: Tx.handle,
            profiler_buffer: Tx.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            Tx.func_attr({"global_symbol": "main", "target": Tx.target("cuda")})
            
            # Match buffer
            batch_size = Tx.int32()
            cos_sin_cache_len = Tx.int32()
            max_page_num = Tx.int32()
            total_page_num = Tx.int32()
            attn_tile_num = Tx.int32()
            
            # Input and output
            hidden_state_global = Tx.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = Tx.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16" if self.world_size > 1 else "float32", scope="global")
            output_global = Tx.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")
            
            # Attention weights
            qkv_proj_weight_global = Tx.match_buffer(qkv_proj_weight_ptr, [(self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, self.HIDDEN_SIZE], "float16", scope="global")
            o_proj_weight_global = Tx.match_buffer(o_proj_weight_ptr, [self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM], "float16", scope="global")
            q_rms_weight_global = Tx.match_buffer(q_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global")
            k_rms_weight_global = Tx.match_buffer(k_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global")
            attn_add_rms_weight_global = Tx.match_buffer(attn_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")
            mlp_add_rms_weight_global = Tx.match_buffer(mlp_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")
            
            # MoE weights
            gate_weight_global = Tx.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = Tx.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = Tx.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")
            
            # Attention: page cache, kv cache and plan info
            cos_sin_cache_global = Tx.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, self.HEAD_DIM], "float32", scope="global")
            rope_pos_global = Tx.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            kv_cache_global = Tx.match_buffer(kv_cache_ptr, [max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16", scope="global")
            append_pos_global = Tx.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            q_indptr_global = Tx.match_buffer(q_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indptr_global = Tx.match_buffer(kv_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            partial_indptr_global = Tx.match_buffer(partial_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indices_global = Tx.match_buffer(kv_indices_ptr, [total_page_num], "int32", offset_factor=1)
            q_len_global = Tx.match_buffer(q_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_len_global = Tx.match_buffer(kv_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            q_start_global = Tx.match_buffer(q_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_start_global = Tx.match_buffer(kv_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_end_global = Tx.match_buffer(kv_end_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_head_idx_global = Tx.match_buffer(kv_head_idx_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            work_indptr_global = Tx.match_buffer(work_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            len_kv_chunk_global = Tx.match_buffer(len_kv_chunk_ptr, [2], "int32", offset_factor=1)
            num_qo_len_global = Tx.match_buffer(num_qo_len_ptr, [1], "int32", offset_factor=1)
            merge_indptr_global = Tx.match_buffer(merge_indptr_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            merge_o_indices_global = Tx.match_buffer(merge_o_indices_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            inverse_indptr_global = Tx.match_buffer(inverse_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            inverse_indices_global = Tx.match_buffer(inverse_indices_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            
            # Attention intermediate buffer
            partial_qkv_global = Tx.match_buffer(partial_qkv_ptr, [self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM], "float32", scope="global")
            qkv_global = Tx.match_buffer(qkv_ptr, [batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM], "float16", scope="global")
            o_global = Tx.match_buffer(o_ptr, [batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM], "float16", scope="global")
            o_partial_attn_global = Tx.match_buffer(o_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM], "float32", scope="global")
            lse_partial_attn_global = Tx.match_buffer(lse_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS], "float32", scope="global")
            partial_o_global = Tx.match_buffer(partial_o_ptr, [self.SPLIT_O_PROJECT, batch_size, self.HIDDEN_SIZE], "float32", scope="global")
            before_o_allreduce_global = Tx.match_buffer(before_o_allreduce_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            hidden_state_attn_mlp_global = Tx.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            
            # MoE intermediate buffer
            gating_output_global = Tx.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = Tx.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = Tx.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = Tx.int32()
            sorted_token_ids_global = Tx.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = Tx.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = Tx.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = Tx.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = Tx.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = Tx.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = Tx.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = Tx.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            
            # event tensor
            etensor_workspace_size = Tx.int32()
            etensor_workspace_global = Tx.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")
            
            # Execution queue
            queue_tasks_global = Tx.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = Tx.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = Tx.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)
            
            @Tx.macro
            def run(BLK_M, low_batch, down_proj_task_size):
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global,
                    qkv_proj_weight_global, o_proj_weight_global, q_rms_weight_global, k_rms_weight_global,
                    attn_add_rms_weight_global, mlp_add_rms_weight_global,
                    gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    cos_sin_cache_global, rope_pos_global, kv_cache_global, append_pos_global,
                    q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global,
                    q_len_global, kv_len_global, q_start_global, kv_start_global, kv_end_global,
                    kv_head_idx_global, work_indptr_global, len_kv_chunk_global, num_qo_len_global,
                    merge_indptr_global, merge_o_indices_global, inverse_indptr_global, inverse_indices_global,
                    partial_qkv_global, qkv_global, o_global, o_partial_attn_global, lse_partial_attn_global,
                    partial_o_global, before_o_allreduce_global, hidden_state_attn_mlp_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global,
                    expert_ids_global, None, num_tokens_post_pad_global, cumsum_buffer_global, # no num_valid_tokens_global 
                    reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global,
                    etensor_workspace_global,
                    profiler_buffer, None, queue_tasks_global, queue_head_global, queue_tail_global,
                    BLK_M, down_proj_task_size, low_batch, False,
                    dynamic_scheduler.Semaphore, dynamic_scheduler.DynamicTileScheduler
                )
            if batch_size <= 4:
                run(32, True, 1)
            elif batch_size <= 32:
                run(32, True, 4)
            elif batch_size <= 64:
                run(64, True, 4)
            else:
                run(128, True, 4)
        # fmt: on
        return main

    def get_func_static(self, unfused=False):
        """Static scheduler function combining attention and MoE"""
        # fmt: off
        @Tx.prim_func(tirx=True)
        def main(
            # Input and output
            hidden_state_ptr: Tx.handle,
            residual_ptr: Tx.handle,
            output_ptr: Tx.handle,
            
            # Attention weights
            qkv_proj_weight_ptr: Tx.handle,
            o_proj_weight_ptr: Tx.handle,
            q_rms_weight_ptr: Tx.handle,
            k_rms_weight_ptr: Tx.handle,
            attn_add_rms_weight_ptr: Tx.handle,
            mlp_add_rms_weight_ptr: Tx.handle,
            
            # MoE weights
            gate_weight_ptr: Tx.handle,
            grp_gate_up_weight_ptr: Tx.handle,
            grp_down_weight_ptr: Tx.handle,
            
            # Attention: page cache, cos_sin cache and plan info
            cos_sin_cache_ptr: Tx.handle,
            rope_pos_ptr: Tx.handle,
            kv_cache_ptr: Tx.handle,
            append_pos_ptr: Tx.handle,
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
            num_qo_len_ptr: Tx.handle,
            merge_indptr_ptr: Tx.handle,
            merge_o_indices_ptr: Tx.handle,
            inverse_indptr_ptr: Tx.handle,
            inverse_indices_ptr: Tx.handle,
            
            # Attention intermediate buffer
            partial_qkv_ptr: Tx.handle,
            qkv_ptr: Tx.handle,
            o_ptr: Tx.handle,
            o_partial_attn_ptr: Tx.handle,
            lse_partial_attn_ptr: Tx.handle,
            partial_o_ptr: Tx.handle,
            before_o_allreduce_ptr: Tx.handle,
            hidden_state_attn_mlp_ptr: Tx.handle,
            
            # MoE intermediate buffer
            gating_output_ptr: Tx.handle,
            topk_weights_ptr: Tx.handle,
            topk_indices_ptr: Tx.handle,
            sorted_token_ids_ptr: Tx.handle,
            expert_ids_ptr: Tx.handle,
            num_valid_tokens_ptr: Tx.handle,
            num_tokens_post_pad_ptr: Tx.handle,
            cumsum_buffer_ptr: Tx.handle,
            reordered_hidden_state_ptr: Tx.handle,
            gate_up_output_ptr: Tx.handle,
            silu_mul_output_ptr: Tx.handle,
            
            # event tensors
            etensor_workspace_ptr: Tx.handle,
            # Execution queue
            exec_queue_ptr: Tx.handle,
            profiler_buffer: Tx.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            Tx.func_attr({"global_symbol": "main", "target": Tx.target("cuda")})
            
            # Match buffer
            batch_size = Tx.int32()
            cos_sin_cache_len = Tx.int32()
            max_page_num = Tx.int32()
            total_page_num = Tx.int32()
            
            # Input and output
            hidden_state_global = Tx.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = Tx.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16" if self.world_size > 1 else "float32", scope="global")
            output_global = Tx.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")
            
            # Attention weights
            qkv_proj_weight_global = Tx.match_buffer(qkv_proj_weight_ptr, [(self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM, self.HIDDEN_SIZE], "float16", scope="global")
            o_proj_weight_global = Tx.match_buffer(o_proj_weight_ptr, [self.HIDDEN_SIZE, self.NUM_ATTENTION_HEADS * self.HEAD_DIM], "float16", scope="global")
            q_rms_weight_global = Tx.match_buffer(q_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global")
            k_rms_weight_global = Tx.match_buffer(k_rms_weight_ptr, [self.HEAD_DIM], "float16", scope="global")
            attn_add_rms_weight_global = Tx.match_buffer(attn_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")
            mlp_add_rms_weight_global = Tx.match_buffer(mlp_add_rms_weight_ptr, [self.HIDDEN_SIZE], "float16", scope="global")
            
            # MoE weights
            gate_weight_global = Tx.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = Tx.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = Tx.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")
            
            # Attention: page cache, kv cache and plan info
            cos_sin_cache_global = Tx.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, self.HEAD_DIM], "float32", scope="global")
            rope_pos_global = Tx.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            kv_cache_global = Tx.match_buffer(kv_cache_ptr, [max_page_num, 2, self.NUM_KEY_VALUE_HEADS, self.PAGE_SIZE, self.HEAD_DIM], "float16", scope="global")
            append_pos_global = Tx.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            q_indptr_global = Tx.match_buffer(q_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indptr_global = Tx.match_buffer(kv_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            partial_indptr_global = Tx.match_buffer(partial_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_indices_global = Tx.match_buffer(kv_indices_ptr, [total_page_num], "int32", offset_factor=1)
            q_len_global = Tx.match_buffer(q_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_len_global = Tx.match_buffer(kv_len_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            q_start_global = Tx.match_buffer(q_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_start_global = Tx.match_buffer(kv_start_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_end_global = Tx.match_buffer(kv_end_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            kv_head_idx_global = Tx.match_buffer(kv_head_idx_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            work_indptr_global = Tx.match_buffer(work_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            len_kv_chunk_global = Tx.match_buffer(len_kv_chunk_ptr, [2], "int32", offset_factor=1)
            num_qo_len_global = Tx.match_buffer(num_qo_len_ptr, [1], "int32", offset_factor=1)
            merge_indptr_global = Tx.match_buffer(merge_indptr_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            merge_o_indices_global = Tx.match_buffer(merge_o_indices_ptr, [self.MAX_NUM_KV_SPLITS], "int32", offset_factor=1)
            inverse_indptr_global = Tx.match_buffer(inverse_indptr_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            inverse_indices_global = Tx.match_buffer(inverse_indices_ptr, [self.MAX_TOTAL_NUM_WORKERS], "int32", offset_factor=1)
            
            # Attention intermediate buffer
            partial_qkv_global = Tx.match_buffer(partial_qkv_ptr, [self.SPLIT_QKV_PROJECT, batch_size, (self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS) * self.HEAD_DIM], "float32", scope="global")
            qkv_global = Tx.match_buffer(qkv_ptr, [batch_size, self.NUM_ATTENTION_HEADS + 2 * self.NUM_KEY_VALUE_HEADS, self.HEAD_DIM], "float16", scope="global")
            o_global = Tx.match_buffer(o_ptr, [batch_size, self.NUM_ATTENTION_HEADS, self.HEAD_DIM], "float16", scope="global")
            o_partial_attn_global = Tx.match_buffer(o_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS * self.HEAD_DIM], "float32", scope="global")
            lse_partial_attn_global = Tx.match_buffer(lse_partial_attn_ptr, [self.MAX_NUM_KV_SPLITS * self.NUM_KEY_VALUE_HEADS], "float32", scope="global")
            partial_o_global = Tx.match_buffer(partial_o_ptr, [self.SPLIT_O_PROJECT, batch_size, self.HIDDEN_SIZE], "float32", scope="global")
            before_o_allreduce_global = Tx.match_buffer(before_o_allreduce_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            hidden_state_attn_mlp_global = Tx.match_buffer(hidden_state_attn_mlp_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            
            # MoE intermediate buffer
            gating_output_global = Tx.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = Tx.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = Tx.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = Tx.int32()
            sorted_token_ids_global = Tx.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = Tx.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = Tx.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = Tx.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = Tx.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = Tx.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = Tx.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = Tx.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            
            # event tensor
            etensor_workspace_size = Tx.int32()
            etensor_workspace_global = Tx.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")
            
            # Execution queue
            exec_queue = Tx.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")
            
            @Tx.macro
            def run(BLK_M, low_batch):
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global,
                    qkv_proj_weight_global, o_proj_weight_global, q_rms_weight_global, k_rms_weight_global,
                    attn_add_rms_weight_global, mlp_add_rms_weight_global,
                    gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    cos_sin_cache_global, rope_pos_global, kv_cache_global, append_pos_global,
                    q_indptr_global, kv_indptr_global, partial_indptr_global, kv_indices_global,
                    q_len_global, kv_len_global, q_start_global, kv_start_global, kv_end_global,
                    kv_head_idx_global, work_indptr_global, len_kv_chunk_global, num_qo_len_global,
                    merge_indptr_global, merge_o_indices_global, inverse_indptr_global, inverse_indices_global,
                    partial_qkv_global, qkv_global, o_global, o_partial_attn_global, lse_partial_attn_global,
                    partial_o_global, before_o_allreduce_global, hidden_state_attn_mlp_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global,
                    expert_ids_global, None, num_tokens_post_pad_global, cumsum_buffer_global, # no num_valid_tokens_global 
                    reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global,
                    etensor_workspace_global,
                    profiler_buffer, exec_queue, None, None, None,
                    BLK_M, 1, low_batch, False,
                    static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
                )
            
            if batch_size <= 32:
                run(32, True)
            elif batch_size <= 64:
                run(64, True)
            else:
                run(128, True)
        # fmt: on
        return main


# Module-level helper functions

arg_dict = {}


def prepare_data(batch_size, seq_len, mk: MegaKernelMOEFullLayer):
    """Prepare data combining attention and MoE requirements"""
    global arg_dict
    print("Preparing data for MoE full layer...", flush=True)

    def _correct_weight_tensor_view(tensor):
        if mk.world_size == 1:
            return tensor.view(*tensor.shape[1:])
        return tensor

    torch.manual_seed(42)

    # ===== Attention data (from test_layer.py) =====
    # Input
    arg_dict["hidden_state"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["residual"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)

    # RMS weights
    arg_dict["q_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)
    arg_dict["k_rms_wight"] = torch.randn((mk.HEAD_DIM), dtype=torch.float16)
    arg_dict["attn_add_rms_weight"] = torch.randn((mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["mlp_add_rms_weight"] = torch.randn((mk.HIDDEN_SIZE,), dtype=torch.float16)

    # QKV
    arg_dict["qkv"] = torch.randn(
        (batch_size, mk.NUM_KEY_VALUE_HEADS * 2 + mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
        dtype=torch.float16,
    )

    # RoPE cos_sin cache
    inv_freq = 1.0 / (
        mk.ROPE_THETA
        ** (torch.arange(0, mk.HEAD_DIM, 2, dtype=torch.float, device="cuda") / mk.HEAD_DIM)
    )
    pos = seq_len - 1
    assert pos < 4096  # for faster test
    arg_dict["rope_pos"] = torch.full((batch_size,), pos, dtype=torch.int32)
    t = torch.arange(4096, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    arg_dict["cos_sin_cache"] = torch.cat((cos, sin), dim=-1).reshape(-1, mk.HEAD_DIM).cpu()

    # Paged kv-cache
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

    # Attention output
    arg_dict["o"] = torch.zeros(
        (batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM), dtype=torch.float16
    )
    arg_dict["lse"] = torch.zeros((batch_size, mk.NUM_ATTENTION_HEADS), dtype=torch.float32)
    arg_dict["hidden_state_attn_mlp"] = torch.zeros(
        (batch_size, mk.HIDDEN_SIZE), dtype=torch.float16
    )

    # ===== MoE data (from test_moe.py) =====
    # MoE intermediate buffers
    arg_dict["gating_output"] = torch.zeros((batch_size, mk.NUM_EXPERTS), dtype=torch.float32)
    arg_dict["topk_weights"] = torch.zeros(
        (batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.float32
    )
    arg_dict["topk_indices"] = torch.zeros((batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.int32)
    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, mk.NUM_EXPERTS_PER_TOK, mk.NUM_EXPERTS, mk.MOE_M_PAD_SIZE
    )
    arg_dict["sorted_token_ids"] = torch.zeros((max_num_tokens_padded,), dtype=torch.int32)
    arg_dict["expert_ids"] = torch.zeros(
        (max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32
    )
    arg_dict["num_valid_tokens"] = torch.zeros(
        (max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32
    )
    arg_dict["num_tokens_post_pad"] = torch.zeros((1,), dtype=torch.int32)
    arg_dict["cumsum_buffer"] = torch.zeros((mk.NUM_EXPERTS + 1,), dtype=torch.int32)
    arg_dict["reordered_hidden_state"] = torch.zeros(
        (max_num_tokens_padded, mk.HIDDEN_SIZE), dtype=torch.float16
    )
    arg_dict["gate_up_output"] = torch.zeros(
        (max_num_tokens_padded, mk.INTERMEDIATE_SIZE * 2), dtype=torch.float16
    )
    arg_dict["silu_mul_output"] = torch.zeros(
        (max_num_tokens_padded, mk.INTERMEDIATE_SIZE), dtype=torch.float16
    )
    arg_dict["output"] = torch.zeros((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)

    # Plan info using flashinfer
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

    def tensor_from_bytes(byte_tensor: torch.Tensor, offset: int, shape, data_type: torch.dtype):
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
    arg_dict["len_kv_chunk"] = get_tensor(get_id(mk.NUM_TASK_ARGS + 12), 2, torch.int32).cpu()
    arg_dict["merge_indptr"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 15), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["merge_o_indices"] = get_tensor(
        get_id(mk.NUM_TASK_ARGS + 16), mk.MAX_NUM_KV_SPLITS, torch.int32
    ).cpu()
    arg_dict["num_qo_len"] = get_tensor(get_id(mk.NUM_TASK_ARGS + 17), 1, torch.int32).cpu()

    # Weight initialization (do only once)
    if not hasattr(prepare_data, "weight_initialized"):
        prepare_data.weight_initialized = True
    else:
        return arg_dict

    # Attention weights
    arg_dict["kv_cache"] = _correct_weight_tensor_view(
        torch.randn(
            (mk.world_size, mk.MAX_PAGE_NUM, 2, mk.NUM_KEY_VALUE_HEADS, mk.PAGE_SIZE, mk.HEAD_DIM),
            dtype=torch.float16,
        )
    ).cpu()
    arg_dict["qkv_proj_weight"] = _correct_weight_tensor_view(
        torch.randn(
            (
                mk.world_size,
                (mk.NUM_ATTENTION_HEADS + 2 * mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM,
                mk.HIDDEN_SIZE,
            ),
            dtype=torch.float16,
        )
    )
    torch.nn.init.xavier_normal_(arg_dict["qkv_proj_weight"], gain=1.0)
    arg_dict["o_proj_weight"] = _correct_weight_tensor_view(
        torch.randn(
            (mk.world_size, mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM),
            dtype=torch.float16,
        )
    )
    torch.nn.init.xavier_normal_(arg_dict["o_proj_weight"], gain=1.0)

    # MoE weights
    arg_dict["gate_weight"] = _correct_weight_tensor_view(
        torch.zeros((mk.world_size, mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype=torch.float16).cuda()
    )
    torch.nn.init.xavier_normal_(arg_dict["gate_weight"], gain=1.0)
    arg_dict["gate_weight"] = arg_dict["gate_weight"].cpu()

    arg_dict["grp_gate_up_weight"] = _correct_weight_tensor_view(
        torch.zeros(
            (mk.world_size, mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE),
            dtype=torch.float16,
        ).cuda()
    )
    for i in range(mk.NUM_EXPERTS):
        torch.nn.init.xavier_normal_(arg_dict["grp_gate_up_weight"][i], gain=1.0)
    arg_dict["grp_gate_up_weight"] = arg_dict["grp_gate_up_weight"].cpu()
    w1 = arg_dict["grp_gate_up_weight"]
    arg_dict["grp_up_gate_weight"] = torch.cat(
        (w1[:, mk.INTERMEDIATE_SIZE :, :], w1[:, : mk.INTERMEDIATE_SIZE, :]), dim=1
    ).contiguous()
    new_order_indices = np.stack(
        (
            np.arange(mk.INTERMEDIATE_SIZE).reshape(-1, 16),
            np.arange(mk.INTERMEDIATE_SIZE, mk.INTERMEDIATE_SIZE * 2).reshape(-1, 16),
        ),
        axis=1,
    ).reshape(-1)
    arg_dict["shuffled_grp_gate_up_weight"] = arg_dict["grp_gate_up_weight"][:, new_order_indices, :]

    arg_dict["grp_down_weight"] = _correct_weight_tensor_view(
        torch.zeros(
            (mk.world_size, mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE),
            dtype=torch.float16,
        ).cuda()
    )
    for i in range(mk.NUM_EXPERTS):
        torch.nn.init.xavier_normal_(arg_dict["grp_down_weight"][i], gain=1.0)
    arg_dict["grp_down_weight"] = arg_dict["grp_down_weight"].cpu()

    print("Data preparation complete", flush=True)
    return arg_dict


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, seq_len, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper, sess):
    """Test function combining attention and MoE"""
    arg_dict = prepare_data(batch_size, seq_len, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelMOEFullLayer, scheduler: Literal["static", "dynamic"]):
        """Run TIR kernel (supports both static and dynamic scheduler)"""
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")

        # Prepare execution queue based on scheduler type
        if scheduler == "static":
            # Static scheduler needs exec queue with attention task info
            exec_queue = generate_exec_queue(
                batch_size,
                arg_dict["attn_task_num"].item(),
                mk.config,
                mk.world_size,
                mk.num_etensors[False],
                "static",
            )
            tvm_arg_dict[f"exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            # Dynamic scheduler
            exec_queue = generate_exec_queue(None, None, mk.config, mk.world_size, mk.num_etensors[True],  "dynamic")
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        # Prepare attention intermediate buffers
        tvm_arg_dict["o_partial_attn"] = tvm.runtime.tensor(
            np.zeros(
                [mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS * mk.HEAD_DIM], dtype=np.float32
            ),
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

        # Get inverse plan info
        res = get_inverse_plan_info(
            batch_size,
            mk.NUM_KEY_VALUE_HEADS,
            arg_dict["q_indptr"],
            arg_dict["kv_head_idx"],
            arg_dict["attn_task_num"].item(),
        )
        tvm_arg_dict["inverse_indptr"], tvm_arg_dict["inverse_indices"] = res

        # Process append_pos (different from flashinfer)
        append_pos = arg_dict["append_pos"].clone()
        for b in range(batch_size):
            append_pos[b] = (
                arg_dict["page_kv_indices"][
                    (arg_dict["page_kv_indptr"][b] * mk.PAGE_SIZE + append_pos[b]) // mk.PAGE_SIZE
                ]
                * mk.PAGE_SIZE
                + append_pos[b] % mk.PAGE_SIZE
            )

        # Copy all arg_dict tensors to device
        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)
        tvm_arg_dict["append_pos"] = tvm.runtime.tensor(append_pos, device=DEV)

        tvm_arg_dict["etensor_workspace"] = tvm.runtime.tensor(np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), device=DEV)

        # Prepare per-iteration buffers
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"].to(torch.float32), device=DEV)

        tvm_arg_dict[f"profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        # Run kernel
        with target:
            iter = 0
            if scheduler == "static":
                kernel = mega_kernel_static["main"]
            else:
                kernel = mega_kernel_dynamic["main"]
            work_arg_dict = tvm_arg_dict

            if scheduler == "static":

                def func():
                    nonlocal iter
                    kernel(
                        # Input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # Attention weights
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # MoE weights
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # Attention cache and plan
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
                        # Attention intermediate
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        # MoE intermediate
                        work_arg_dict["gating_output"],
                        work_arg_dict["topk_weights"],
                        work_arg_dict["topk_indices"],
                        work_arg_dict["sorted_token_ids"],
                        work_arg_dict["expert_ids"],
                        work_arg_dict["num_valid_tokens"],
                        work_arg_dict["num_tokens_post_pad"],
                        work_arg_dict["cumsum_buffer"],
                        work_arg_dict["reordered_hidden_state"],
                        work_arg_dict["gate_up_output"],
                        work_arg_dict["silu_mul_output"],
                        # Event tensors
                        work_arg_dict["etensor_workspace"],
                        # Execution queue (static)
                        work_arg_dict["exec_queue"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            else:

                def func():
                    nonlocal iter
                    kernel(
                        # Input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # Attention weights
                        work_arg_dict["qkv_proj_weight"],
                        work_arg_dict["o_proj_weight"],
                        work_arg_dict["q_rms_wight"],
                        work_arg_dict["k_rms_wight"],
                        work_arg_dict["attn_add_rms_weight"],
                        work_arg_dict["mlp_add_rms_weight"],
                        # MoE weights
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # Attention cache and plan
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
                        # Attention intermediate
                        work_arg_dict["partial_qkv"],
                        work_arg_dict["qkv"],
                        work_arg_dict["o"],
                        work_arg_dict["o_partial_attn"],
                        work_arg_dict["lse_partial"],
                        work_arg_dict["partial_o"],
                        work_arg_dict["before_o_allreduce"],
                        work_arg_dict["hidden_state_attn_mlp"],
                        # MoE intermediate
                        work_arg_dict["gating_output"],
                        work_arg_dict["topk_weights"],
                        work_arg_dict["topk_indices"],
                        work_arg_dict["sorted_token_ids"],
                        work_arg_dict["expert_ids"],
                        work_arg_dict["num_valid_tokens"],
                        work_arg_dict["num_tokens_post_pad"],
                        work_arg_dict["cumsum_buffer"],
                        work_arg_dict["reordered_hidden_state"],
                        work_arg_dict["gate_up_output"],
                        work_arg_dict["silu_mul_output"],
                        # Event tensors
                        work_arg_dict["etensor_workspace"],
                        # Execution queue (dynamic)
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            ms = bench(func, warmup=1, repeat=7, proton_name=f"tir-{scheduler}")
            print(f"TIR ({scheduler}) time: {ms:.3f} ms")
            if mk.profiler_on:
                export_to_perfetto_trace(
                    tvm_arg_dict[f"profiler_buffer"].numpy(),
                    f"{scheduler}-moe-full-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                    event_type_names,
                )
            return tvm_arg_dict["output"].numpy(), tvm_arg_dict["residual_0"].numpy().astype(np.float16)

    def std(arg_dict, use_prefill, mk: MegaKernelMOEFullLayer):
        """Standard reference implementation combining attention and MoE"""
        import flashinfer

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)

            # ===== Attention part (from test_layer.py) =====
            out_f = torch.zeros(
                batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM, dtype=torch.float16, device="cuda"
            )
            lse_f = torch.zeros(
                batch_size, mk.NUM_ATTENTION_HEADS, dtype=torch.float32, device="cuda"
            )

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
                    torch.tensor([seq_len] * batch_size, dtype=torch.int32).to(0),
                    mk.NUM_ATTENTION_HEADS,
                    mk.NUM_KEY_VALUE_HEADS,
                    mk.HEAD_DIM,
                    mk.HEAD_DIM,
                    mk.PAGE_SIZE,
                    kv_data_type=torch.float16,
                    q_data_type=torch.float16,
                )

            # QKV projection
            qkv = torch.matmul(
                std_arg_dict["hidden_state"], std_arg_dict["qkv_proj_weight"].T
            ).reshape(batch_size, -1, mk.HEAD_DIM)
            q, k, v = torch.split(
                qkv,
                [mk.NUM_ATTENTION_HEADS, mk.NUM_KEY_VALUE_HEADS, mk.NUM_KEY_VALUE_HEADS],
                dim=1,
            )

            # RMS norm
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

            # RoPE
            q, k = flashinfer.rope.apply_rope_with_cos_sin_cache(
                positions=std_arg_dict["rope_pos"],
                query=q.reshape(batch_size, -1),
                key=k.reshape(batch_size, -1),
                head_size=mk.HEAD_DIM,
                cos_sin_cache=std_arg_dict["cos_sin_cache"],
                is_neox=True,
            )

            # Append KV cache
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

            # Attention
            if use_prefill:
                out_f = wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                )
            else:
                wrapper.run(
                    q.reshape(batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM),
                    std_arg_dict["kv_cache"],
                    out_f,
                    lse_f,
                )

            # O projection
            hidden_state_attn_mlp = torch.matmul(
                out_f.reshape(batch_size, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM),
                std_arg_dict["o_proj_weight"].T,
            )

            # Add residual + RMS norm
            flashinfer.norm.fused_add_rmsnorm(
                input=hidden_state_attn_mlp,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["attn_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )

            # ===== MoE part (from test_moe.py) =====
            gating_output = hidden_state_attn_mlp @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=True,
            )

            # Prepare MoE outputs
            out1 = torch.empty(
                (batch_size, mk.NUM_EXPERTS_PER_TOK, 2 * mk.INTERMEDIATE_SIZE),
                dtype=torch.float16,
                device="cuda",
            )
            out2 = torch.empty(
                (batch_size, mk.NUM_EXPERTS_PER_TOK, mk.HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

            def get_config(batch_size):
                get_config_func = functools.partial(
                    try_get_optimal_moe_config,
                    std_arg_dict["grp_gate_up_weight"].shape,
                    std_arg_dict["grp_down_weight"].shape,
                    std_arg_dict["topk_indices"].shape[1],
                    "float16",
                    block_shape=None,
                )
                return get_config_func(batch_size)

            sgL_config = get_config(batch_size)
            sorted_ids_std, expert_ids_std, num_tokens_post_pad_std = moe_align_block_size(
                std_arg_dict["topk_indices"],
                sgL_config["BLOCK_SIZE_M"],
                mk.NUM_EXPERTS,
            )

            # Gate-up projection
            invoke_fused_moe_kernel(
                hidden_state_attn_mlp,
                std_arg_dict["grp_gate_up_weight"],
                None,  # bias
                out1,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                sorted_token_ids=sorted_ids_std,
                expert_ids=expert_ids_std,
                num_tokens_post_padded=num_tokens_post_pad_std,
                mul_routed_weight=False,
                top_k=mk.NUM_EXPERTS_PER_TOK,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

            # SiLU + multiply
            silu_mul_out = flashinfer.activation.silu_and_mul(
                out1.view(-1, 2 * mk.INTERMEDIATE_SIZE)
            )

            # Down projection
            invoke_fused_moe_kernel(
                silu_mul_out,
                std_arg_dict["grp_down_weight"],
                None,  # bias
                out2,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                sorted_token_ids=sorted_ids_std,
                expert_ids=expert_ids_std,
                num_tokens_post_padded=num_tokens_post_pad_std,
                mul_routed_weight=True,
                top_k=1,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

            # Reduce topk outputs
            output = out2.sum(dim=1)

            # Final add residual + RMS norm
            flashinfer.norm.fused_add_rmsnorm(
                input=output,
                residual=std_arg_dict["residual"],
                weight=std_arg_dict["mlp_add_rms_weight"],
                eps=mk.RMS_NORM_EPS,
                enable_pdl=False,
            )

            return  output.cpu().numpy(), std_arg_dict["residual"].cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std-use_prefill={use_prefill}")
        print(f"Standard (use_prefill={use_prefill}) time: {ms:.3f} ms")
        return output

    def run():
        if mega_kernel_dynamic["main"] is not None:
            output_tir_dynamic, residual_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("Dynamic TIR finish", flush=True)
        if mega_kernel_static["main"] is not None:
            output_tir_static, residual_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("Static TIR finish", flush=True)
        output_std1, residual_std1 = std(arg_dict, use_prefill=True, mk=mega_kernel_wrapper)
        output_std2, residual_std2 = std(arg_dict, use_prefill=False, mk=mega_kernel_wrapper)

        if mega_kernel_dynamic["main"] is not None:
            try:
                np.testing.assert_allclose(output_tir_dynamic, output_std1, rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(
                    residual_tir_dynamic, residual_std1, rtol=1e-3, atol=1e-2
                )
                print("✓ Dynamic scheduler PASSED", flush=True)
            except Exception as e:
                print(f"✗ Dynamic scheduler FAILED: {e}")
        if mega_kernel_static["main"] is not None:
            try:
                np.testing.assert_allclose(output_tir_static, output_std1, rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(residual_tir_static, residual_std1, rtol=1e-3, atol=1e-2)
                print("✓ Static scheduler PASSED", flush=True)
            except Exception as e:
                print(f"✗ Static scheduler FAILED: {e}")

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("moe_full_layer"):
            run()
    else:
        run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MegaKernel MoE Full Layer testing script.")
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static", "dynamic"],
        choices=["static", "dynamic", "none"],
        help="Scheduler to test (static, dynamic, or none).",
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="Number of devices (only 1 supported)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="List of batch sizes to test.",
    )
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[512], help="List of sequence lengths to test."
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    assert args.world_size == 1, "Currently only world_size=1 is supported"

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelMOEFullLayer(
        config=qwen3_30b_a3b_config, world_size=args.world_size, profiler_on=args.profiler_on
    )
    if "static" in testing_scheduler:
        print("\nCompiling static scheduler module...")
        mega_static_module = mega_kernel_wrapper.get_module("static")
        src, lib_static = get_source(mega_static_module)
        print(src)
        print("Compilation complete")
    else:
        lib_static = {"main": None}

    if "dynamic" in testing_scheduler:
        print("\nCompiling dynamic scheduler module...")
        mega_dynamic_module = mega_kernel_wrapper.get_module("dynamic")
        from tvm.tirx.megakernel.utils.common import get_source

        src, lib_dynamic = get_source(mega_dynamic_module)
        print(src)
        print("Compilation complete")
    else:
        lib_dynamic = {"main": None}

    sess = None  # No multi-GPU support yet

    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            print(f"\n{'='*60}")
            print(f"Testing: batch_size={batch_size}, seq_len={seq_len}")
            print(f"{'='*60}")
            test(batch_size, seq_len, lib_static, lib_dynamic, mega_kernel_wrapper, sess)
