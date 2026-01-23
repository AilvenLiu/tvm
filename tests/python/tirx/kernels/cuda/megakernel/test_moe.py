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
from tvm.script import tir as T
from tvm.script import tirx as Tx
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirx.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler

from tvm.tirx.megakernel.utils.config import ProfileEventType, KernelConfig, JobType, event_type_names, qwen3_30b_a3b_config
from tvm.tirx.megakernel.utils.base import SmemManager, MegaKernelWrapper
from tvm.tirx.megakernel.utils.utils import get_source, ceildiv, f_init_const
from tvm.tirx.megakernel.utils import static_scheduler, dynamic_scheduler
from tvm.tirx.megakernel.kernels import (
    GemmTile, MOETopKReduceTile, TopkSoftmaxTile, MOEAlignTile, 
    CountAndSortExpertTokens, GroupGEMMTileSM100, GroupGEMMSiluTile
)
from tvm.tirx.megakernel.utils.static_scheduler import StaticTileScheduler
from tvm.tirx.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tvm.tirx.megakernel.utils.support import (
    generate_exec_queue_moe,
    get_max_num_tokens_padded,
    get_max_blocks_padded_relaxed,
)

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sgl_kernel import topk_softmax
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_triton,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe import MoeRunnerConfig
import flashinfer.fused_moe as fused_moe
from triton import language as tl

# TODO: fix abnormal slowness of batch-attn on the first tile

class MegaKernelMOE(MegaKernelWrapper):

    MOE_M_PAD_SIZE = 128
    GATING_BLK_M = 128

    def __init__(self, config, world_size, profiler_on):
        super().__init__(config, 1, profiler_on)
        self.world_size = world_size
        self.MODEL_NAME = config.get("MODEL_NAME", None)
        self.HIDDEN_SIZE = config.get("HIDDEN_SIZE", None)
        self.INTERMEDIATE_SIZE = config.get("INTERMEDIATE_SIZE", None)
        self.HEAD_DIM = config.get("HEAD_DIM", None)
        self.NUM_EXPERTS = config.get("NUM_EXPERTS", None)
        self.NUM_EXPERTS_PER_TOK = config.get("NUM_EXPERTS_PER_TOK", None)
        self.GATING_SPLIT_K_FACTOR = config.get("GATING_SPLIT_K_FACTOR", None)
      
    def _set_tiles(self, batch_size, low_batch):
        self.gate = self._add_tile(
            GemmTile(
                self.NUM_EXPERTS,
                self.HIDDEN_SIZE,
                "float16",
                "float16",
                self.GATING_SPLIT_K_FACTOR,
                self.GATING_BLK_M,
                self.GATING_BLK_M,
                use_tma_reduce=True,
            ),
            ProfileEventType.MOE_GATING,
        )
        self.topk_softmax = self._add_tile(
            TopkSoftmaxTile(
                self.NUM_EXPERTS, batch_size, self.NUM_EXPERTS_PER_TOK, dtype="float32"
            ),
            ProfileEventType.TOPK_SOFTMAX,
        )
        numel = self.NUM_EXPERTS_PER_TOK * batch_size
        self.align = self._add_tile(
            MOEAlignTile(self.NUM_EXPERTS, numel, self.MOE_M_PAD_SIZE, pad_sorted_token_ids=True),
            ProfileEventType.MOE_ALIGN,
        )
        self.count_and_sort_expert_tokens = self._add_tile(
            CountAndSortExpertTokens(numel, self.HIDDEN_SIZE, self.NUM_EXPERTS_PER_TOK),
            ProfileEventType.COUNT_AND_SORT,
        )
        self.group_gemm_gate_up_silu = self._add_tile(
            GroupGEMMSiluTile(
                self.INTERMEDIATE_SIZE * 2,
                self.HIDDEN_SIZE,
                self.NUM_EXPERTS,
                self.NUM_EXPERTS_PER_TOK,
                numel,
                "float16",
                "float16",
                low_batch=low_batch,
            ),
            ProfileEventType.GROUP_GEMM_GATE_UP_SILU,
        )
        self.group_gemm_down = self._add_tile(
            GroupGEMMTileSM100(
                self.HIDDEN_SIZE,
                self.INTERMEDIATE_SIZE,
                self.NUM_EXPERTS,
                self.NUM_EXPERTS_PER_TOK,
                numel,
                "float16",
                "float16",
                acc_output=True,
                low_batch=low_batch,
            ),
            ProfileEventType.GROUP_GEMM_DOWN,
        )
        self.topk_reduce = self._add_tile(
            MOETopKReduceTile(batch_size, self.HIDDEN_SIZE, "float16", self.NUM_EXPERTS_PER_TOK),
            ProfileEventType.TOPK_REDUCE,
        )

    def set_tiles(self, batch_size, low_batch):
        self.reset()
        self._set_tiles(batch_size, low_batch)

    def _set_events(self, batch_size, Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]], etensor_workspace_global, unfused=False):
        self.evt_gating = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(self.GATING_SPLIT_K_FACTOR * ceildiv(batch_size, self.GATING_BLK_M)))
        self.evt_topk_softmax = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(KernelConfig.SM_NUMBER))
        self.evt_moe_align = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(1))
        self.evt_count_and_sort = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(KernelConfig.SM_NUMBER))
        max_num_tokens_padded = get_max_num_tokens_padded(batch_size, self.NUM_EXPERTS_PER_TOK, self.NUM_EXPERTS, self.MOE_M_PAD_SIZE)
        max_blocks_padded_relaxed = get_max_blocks_padded_relaxed(batch_size, self.NUM_EXPERTS_PER_TOK, self.NUM_EXPERTS, self.MOE_M_PAD_SIZE)
        if unfused:
            self.evt_group_gemm_gate_up = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(max_num_tokens_padded // self.MOE_M_PAD_SIZE * self.INTERMEDIATE_SIZE * 2 // GroupGEMMTileSM100.BLK_N))
        else:
            self.evt_group_gemm_gate_up = self.add_etensor(Semaphore, etensor_workspace_global, shape=[max_blocks_padded_relaxed], f_init=f_init_const(self.INTERMEDIATE_SIZE * 2 // GroupGEMMTileSM100.BLK_N))
        f_init_group_gemm_down = f_init_const(
            max_num_tokens_padded // self.MOE_M_PAD_SIZE * self.HIDDEN_SIZE // GroupGEMMTileSM100.BLK_N
        ) if issubclass(Semaphore, static_scheduler.Semaphore) else None
        self.evt_group_gemm_down = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_group_gemm_down)
        
    def set_events(self, is_dynamic_sch, batch_size, Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]], etensor_workspace_global, unfused=False):
        self._set_events(batch_size, Semaphore, etensor_workspace_global, unfused=unfused)
        self.set_events_complete(is_dynamic_sch, Semaphore, etensor_workspace_global)
        self.num_etensors[is_dynamic_sch] = len(self.etensor_and_f_init_pairs)

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        subclass = GroupGEMMTileSM100 if isinstance(tile, GemmTile) else tile.__class__
        self.class_list.add(subclass)
        return tile

    @T.macro
    def task_impl_moe_gating(self, A, B, output, is_dynamic_sch):
        with T.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_gating, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MOE_TOPK_SOFTMAX.value, self.topk_softmax.PERSISTENT_SM_NUMBER, push_idx, 0, 0)
                    ), "warpgroup", "warpgroup", scope_id=0
                )
            self.run_tile(self.gate, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, A, B, output, self.profiler)
            self.tile_scheduler.notify(self.evt_gating, lambda notify_idx: (1, -1, 0), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_moe_topk_softmax(self, gating_output_global, topk_weights_global, topk_indices_global, is_dynamic_sch, renormalize=True):
        with T.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_topk_softmax, lambda notify_idx: (1, -1, 0),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MOE_ALIGN.value, 1, 0, 0, 0)
                    ), "thread", "thread"
                )
            self.tile_scheduler.wait(self.evt_gating, 0, wait_level="cta")
            self.run_tile(self.topk_softmax, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, gating_output_global, topk_weights_global, topk_indices_global, renormalize=renormalize)
            self.tile_scheduler.notify(self.evt_topk_softmax, lambda notify_idx: (1, -1, 0), scope="cta")

    @T.macro
    def task_impl_moe_align(self, topk_ids_flattened, sorted_token_ids_global, expert_ids_global, num_tokens_post_pad_global, cumsum_buffer_global, num_valid_tokens_global, down_proj_task_size, is_dynamic_sch):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_moe_align, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MOE_COUNT_AND_SORT.value, KernelConfig.SM_NUMBER, push_idx, 0, 0)
                    ), "cta", "cta"
                )
            self.tile_scheduler.wait(self.evt_topk_softmax, 0, wait_level="cta")
            self.run_tile(self.align, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, topk_ids_flattened, sorted_token_ids_global, expert_ids_global, num_tokens_post_pad_global, cumsum_buffer_global, num_valid_tokens_global)
            T.cuda.cta_sync()
            if tid == 0:
                # TODO: make this etensor initialization a task
                if is_dynamic_sch:
                    self.evt_group_gemm_down.sem[0] = (self.evt_group_gemm_down.base + 1) * (num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE) * (self.HIDDEN_SIZE // GroupGEMMTileSM100.BLK_N // down_proj_task_size)
            self.tile_scheduler.notify(self.evt_moe_align, lambda notify_idx: (1, -1, 0), scope="thread")

    @T.macro
    def task_impl_moe_count_and_sort(self, topk_ids_flattened, sorted_token_ids_global, cumsum_buffer_global, hidden_state_global, reordered_hidden_state_global, num_tokens_post_pad_global, is_dynamic_sch):
        with T.cta():
            if is_dynamic_sch:
                n_axis_len = T.meta_var(self.INTERMEDIATE_SIZE * 2 // GroupGEMMTileSM100.BLK_N)
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_count_and_sort, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value, num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE * n_axis_len, push_idx // n_axis_len, push_idx % n_axis_len, 0)
                    ), "cta", "cta"
                )
            self.tile_scheduler.wait(self.evt_moe_align, 0, wait_level="cta")
            self.run_tile(self.count_and_sort_expert_tokens, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, topk_ids_flattened, sorted_token_ids_global, cumsum_buffer_global, hidden_state_global, reordered_hidden_state_global)
            self.tile_scheduler.notify(self.evt_count_and_sort, lambda notify_idx: (1, -1, 0), scope="cta")

    @T.macro
    def task_impl_moe_group_gemm_gate_up_silu(self, A, B, output, topk_weights_flattened, sorted_token_ids_global, expert_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch):
        with T.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_group_gemm_gate_up, lambda notify_idx: (1, -1, self.tile_scheduler.m_idx),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.MOE_GROUP_GEMM_DOWN.value, self.HIDDEN_SIZE // GroupGEMMTileSM100.BLK_N // down_proj_task_size, self.tile_scheduler.m_idx, push_idx, 0)
                    ), "warp", "warp"
                )
            self.tile_scheduler.wait(self.evt_count_and_sort, 0, wait_level="warp")
            if is_dynamic_sch or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE:
                self.run_tile(self.group_gemm_gate_up_silu, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, A, B, output, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, self.profiler)    
            idx = T.meta_var(self.tile_scheduler.m_idx if not unfused else 0)
            self.tile_scheduler.notify(self.evt_group_gemm_gate_up, lambda notify_idx: (1, -1, idx), scope="warpgroup", scope_id=0)

    @T.macro
    def task_impl_moe_group_gemm_down(self, A, B, output, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch):
        with T.cta():
            if is_dynamic_sch:
                self.tile_scheduler.pre_notify_and_push(
                    self.evt_group_gemm_down, lambda notify_idx: (1, -1, 0,),
                    lambda trigger_idx: (
                        lambda push_idx: (JobType.END.value, KernelConfig.SM_NUMBER, 0, 0, 0)
                    ), "warp", "warp"
                )
            wait_idx = T.meta_var(self.tile_scheduler.m_idx if not unfused else 0)
            self.tile_scheduler.wait(self.evt_group_gemm_gate_up, wait_idx, wait_level="warp")
            if is_dynamic_sch or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE:
                for i in range(down_proj_task_size):
                    self.run_tile(self.group_gemm_down, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx * down_proj_task_size + i, self.tile_scheduler.k_idx, A, B, output, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, self.profiler)

    # fmt: off
    @T.macro
    def fused_body(
        self,
        batch_size,
        hidden_state_global,
        residual_global,
        output_global,
        gate_weight_global,
        grp_gate_up_weight_global,
        grp_down_weight_global,
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
        topk_reduce_output_global,
        etensor_workspace_global,
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        down_proj_task_size, # to amortize dynamic scheduling overhead
        low_batch,
        unfused,
        is_dynamic_sch,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        Scheduler: Type[Union[static_scheduler.StaticTileScheduler, dynamic_scheduler.DynamicTileScheduler]],
    ):
        # initialize tile
        self.set_tiles(batch_size, low_batch)
        self.host_init_all()

        with T.kernel():
            bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tid_in_wg = T.thread_id([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
            profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
            self.init_profiler(profiler_buffer)
            with T.cta():
                buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                # initialize smem manager
                self.set_smem_manager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data)

                # initialize device
                self.device_init_all(self.smem_manager)
                self.class_init_all(self.smem_manager)

                # initialize event tensors
                self.set_events(is_dynamic_sch, batch_size, Semaphore, etensor_workspace_global, unfused)
                
                # initialize tile scheduler and smem_manager
                if not is_dynamic_sch:
                    self.init_tile_scheduler(False, Scheduler, "layer", exec_queue, self.smem_manager)
                else:
                    self.init_tile_scheduler(True, Scheduler, exec_task, exec_head, exec_tail, self.smem_manager, self.profiler)
                self.smem_manager.init()

                topk_ids_flattened = topk_indices_global.view(-1)
                topk_weights_flattened = topk_weights_global.view(-1)
                while self.tile_scheduler.valid():
                    if self.tile_scheduler.task_type == JobType.MOE_GATING.value:
                        self.task_impl_moe_gating(hidden_state_global, gate_weight_global, gating_output_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.MOE_TOPK_SOFTMAX.value:
                        self.task_impl_moe_topk_softmax(gating_output_global, topk_weights_global, topk_indices_global, is_dynamic_sch, renormalize=False)
                    elif self.tile_scheduler.task_type == JobType.MOE_ALIGN.value:
                        self.task_impl_moe_align(topk_ids_flattened, sorted_token_ids_global, expert_ids_global, num_tokens_post_pad_global, cumsum_buffer_global, num_valid_tokens_global, down_proj_task_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.MOE_COUNT_AND_SORT.value:
                        self.task_impl_moe_count_and_sort(topk_ids_flattened, sorted_token_ids_global, cumsum_buffer_global, hidden_state_global, reordered_hidden_state_global, num_tokens_post_pad_global, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value:
                        self.task_impl_moe_group_gemm_gate_up_silu(reordered_hidden_state_global, grp_gate_up_weight_global, silu_mul_output_global, topk_weights_flattened, sorted_token_ids_global, expert_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_DOWN.value:
                        self.task_impl_moe_group_gemm_down(silu_mul_output_global, grp_down_weight_global, topk_reduce_output_global, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.INIT_ETENSOR.value:
                        self.task_impl_init_etensor(is_dynamic_sch)
                    elif self.tile_scheduler.task_type == JobType.WAIT_ETENSOR_INIT.value:
                        self.task_impl_wait_etensor_init_complete(is_dynamic_sch)
                    else:
                        T.cuda.trap_when_assert_failed(False)
                    self.smem_manager.exit_tile_runtime()
                    self.tile_scheduler.next_tile()
                if self.profiler_on:
                    self.profiler.finalize(lane_id == 0)
                self.class_finalize_all()

    # fmt: on

    # FIXME: change offset_factor to 0 can make performance better
    #       but it requires change on engine side
    def get_func_static(self, unfused=False):
        # fmt: off
        @T.prim_func(tirx=True)
        def main(
            # input and output
            hidden_state_ptr: T.handle, # input: read-only
            residual_ptr: T.handle, # input & output: inplace update
            output_ptr: T.handle, # output

            # weight
            gate_weight_ptr: T.handle, # read-only
            grp_gate_up_weight_ptr: T.handle, # read-only
            grp_down_weight_ptr: T.handle, # read-only

            # intermediate buffer
            gating_output_ptr: T.handle, # intermediate
            topk_weights_ptr: T.handle, # intermediate
            topk_indices_ptr: T.handle, # intermediate
            sorted_token_ids_ptr: T.handle, # intermediate
            expert_ids_ptr: T.handle, # intermediate
            num_valid_tokens_ptr: T.handle, # intermediate
            num_tokens_post_pad_ptr: T.handle, # intermediate
            cumsum_buffer_ptr: T.handle, # intermediate
            reordered_hidden_state_ptr: T.handle, # intermediate
            gate_up_output_ptr: T.handle, # intermediate
            silu_mul_output_ptr: T.handle, # intermediate
            topk_reduce_output_ptr: T.handle, # intermediate


            # event tensor
            etensor_workspace_ptr: T.handle, # not required to reset. Must be 0 before launch.

            # execution queue
            exec_queue_ptr: T.handle,
            profiler_buffer: T.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            T.func_attr(
                {"global_symbol": "main", "target": T.target("cuda")}
            )

            # match buffer
            batch_size = T.int32()

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = T.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")

            # weight
            gate_weight_global = T.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = T.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = T.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")

            # intermediate buffer
            gating_output_global = T.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = T.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = T.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = T.int32()
            sorted_token_ids_global = T.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = T.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = T.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = T.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = T.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = T.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = T.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = T.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = T.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")

            # event tensor
            etensor_workspace_size = T.int32()
            etensor_workspace_global = T.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")
            
            # exec queue
            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")

            @T.macro
            def run(low_batch, dynamic_gemm_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global,
                    profiler_buffer, exec_queue, None, None, None, 1, low_batch, unfused, 
                    False, static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
                )

            if batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True)
            elif batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True)
            else:
                run(low_batch=True, dynamic_gemm_size=False)
            # fmt: on
        return main

    def get_func_dynamic(self):
        # fmt: off
        @T.prim_func(tirx=True)
        def main(
            # input and output
            hidden_state_ptr: T.handle, # input: read-only
            residual_ptr: T.handle, # input & output: inplace update
            output_ptr: T.handle, # output

            # weight
            gate_weight_ptr: T.handle, # read-only
            grp_gate_up_weight_ptr: T.handle, # read-only
            grp_down_weight_ptr: T.handle, # read-only

            # intermediate buffer
            gating_output_ptr: T.handle, # intermediate
            topk_weights_ptr: T.handle, # intermediate
            topk_indices_ptr: T.handle, # intermediate
            sorted_token_ids_ptr: T.handle, # intermediate
            expert_ids_ptr: T.handle, # intermediate
            num_valid_tokens_ptr: T.handle, # intermediate
            num_tokens_post_pad_ptr: T.handle, # intermediate
            cumsum_buffer_ptr: T.handle, # intermediate
            reordered_hidden_state_ptr: T.handle, # intermediate
            gate_up_output_ptr: T.handle, # intermediate
            silu_mul_output_ptr: T.handle, # intermediate
            topk_reduce_output_ptr: T.handle, # intermediate


            # event tensor
            etensor_workspace_ptr: T.handle, # not required to reset. Must be 0 before launch.

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

            # input and output
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = T.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")

            # weight
            gate_weight_global = T.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = T.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = T.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")

            # intermediate buffer
            gating_output_global = T.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = T.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = T.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = T.int32()
            sorted_token_ids_global = T.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = T.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = T.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = T.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = T.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = T.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = T.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = T.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = T.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")

            # event tensor
            etensor_workspace_size = T.int32()
            etensor_workspace_global = T.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")

            # exec queue
            queue_tasks_global = T.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = T.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = T.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)

            @T.macro
            def run(low_batch, dynamic_gemm_size, down_proj_task_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global,
                    profiler_buffer, None, queue_tasks_global, queue_head_global, queue_tail_global, down_proj_task_size, low_batch, False, 
                    True, dynamic_scheduler.Semaphore, dynamic_scheduler.DynamicTileScheduler
                )

            if batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True, down_proj_task_size=4)
            elif batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True, down_proj_task_size=4)
            elif batch_size >= 4:
                run(low_batch=True, dynamic_gemm_size=False, down_proj_task_size=4)
            else:
                run(low_batch=True, dynamic_gemm_size=False, down_proj_task_size=1)
            # fmt: on
        return main


def fused_moe_sglang(
    hidden_states,
    w13,
    w2,
    router_logits,
    routing_weights,
    selected_experts,
):
    topk_output = StandardTopKOutput(
        topk_weights=routing_weights,
        topk_ids=selected_experts,
        router_logits=router_logits,
    )
    moe_config = MoeRunnerConfig(inplace=False)
    return fused_moe_triton(hidden_states, w13, w2, topk_output, moe_config)


arg_dict = {}


def prepare_data(batch_size, mk: MegaKernelMOE):
    print("start prepare data", flush=True)
    global arg_dict
    import torch

    def _correct_weight_tensor_view(tensor):
        if mk.world_size == 1:
            return tensor.view(*tensor.shape[1:])
        return tensor

    torch.manual_seed(42)

    # input
    arg_dict["hidden_state"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["residual"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    # intermediate buffer
    arg_dict["gating_output"] = torch.zeros((batch_size, mk.NUM_EXPERTS), dtype=torch.float32)
    arg_dict["topk_weights"] = torch.zeros(
        (batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.float32
    )
    arg_dict["topk_indices"] = torch.zeros((batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.int32)
    max_num_tokens_padded = get_max_num_tokens_padded(batch_size, mk.NUM_EXPERTS_PER_TOK, mk.NUM_EXPERTS, mk.MOE_M_PAD_SIZE)
    arg_dict["sorted_token_ids"] = torch.zeros((max_num_tokens_padded,), dtype=torch.int32)
    arg_dict["expert_ids"] = torch.zeros(
        (max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32)
    arg_dict["num_valid_tokens"] = torch.zeros((max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32
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
    arg_dict["topk_reduce_output"] = torch.zeros((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)

    # weight initialization
    if not hasattr(prepare_data, "weight_initialized"):
        prepare_data.weight_initialized = True
    else:
        return arg_dict
    arg_dict["gate_weight"] = _correct_weight_tensor_view(
        torch.zeros(
            (mk.world_size, mk.NUM_EXPERTS, mk.HIDDEN_SIZE),
            dtype=torch.float16,
        ).cuda()
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
    print("end prepare data", flush=True)
    return arg_dict


@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mega_kernel_static, mega_kernel_dynamic, mega_kernel_unfused, mega_kernel_wrapper, sess):
    arg_dict = prepare_data(batch_size, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernelMOE, scheduler: Literal["static", "dynamic", "unfused"]):
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")
        if scheduler == "static" or scheduler == "unfused":
            # static schedule
            exec_queue = generate_exec_queue_moe(batch_size, mk.config, mk.num_etensors[False], "static")
            tvm_arg_dict[f"exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            exec_queue = generate_exec_queue_moe(batch_size, mk.config, mk.num_etensors[True], "dynamic")
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)

        tvm_arg_dict["output"] = tvm.runtime.tensor(
            np.zeros((batch_size, mk.HIDDEN_SIZE), dtype=np.float16), device=DEV
        )
        tvm_arg_dict["etensor_workspace"] = tvm.runtime.tensor(np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), device=DEV)
        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
            # initial tensor must be 0
            tvm_arg_dict[f"gating_output_{i}"] = tvm.runtime.tensor(
                arg_dict["gating_output"], device=DEV
            )
            tvm_arg_dict[f"topk_reduce_output_{i}"] = tvm.runtime.tensor(
                arg_dict["topk_reduce_output"], device=DEV
            )
        tvm_arg_dict[f"profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        if mk.world_size > 1:
            raise ValueError(f"Unsupported world size: {mk.world_size}")
        with target:
            iter = 0

            if scheduler == "static" or scheduler == "unfused":
                kernel = mega_kernel_static["main"] if scheduler == "static" else mega_kernel_unfused["main"]
                work_arg_dict = tvm_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # intermediate buffer
                        work_arg_dict[f"gating_output_{iter}"],
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
                        work_arg_dict[f"topk_reduce_output_{iter}"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict["exec_queue"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            else:
                kernel = mega_kernel_dynamic["main"]
                work_arg_dict = tvm_arg_dict

                def func():
                    nonlocal iter
                    kernel(
                        # input and output
                        work_arg_dict["hidden_state"],
                        work_arg_dict[f"residual_{iter}"],
                        work_arg_dict["output"],
                        # weight
                        work_arg_dict["gate_weight"],
                        work_arg_dict["shuffled_grp_gate_up_weight"],
                        work_arg_dict["grp_down_weight"],
                        # intermediate buffer
                        work_arg_dict[f"gating_output_{iter}"],
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
                        work_arg_dict[f"topk_reduce_output_{iter}"],
                        # event tensor
                        work_arg_dict["etensor_workspace"],
                        # exec queue
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            if mk.world_size == 1:
                ms = bench(func, warmup=1, repeat=5, proton_name=f"tir-{scheduler}")
                print(f"TIR time: {ms:.3f} ms")
                if mk.profiler_on:
                    export_to_perfetto_trace(
                        tvm_arg_dict[f"profiler_buffer"].numpy(),
                        f"{scheduler}-moe-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                        event_type_names,
                    )
                # def scatter_out(out_tvm, *shape):
                #     tmp_out = torch.from_numpy(out_tvm.numpy())
                #     out = torch.zeros(shape, dtype=torch.float16)
                #     sorted_token_ids = tvm_arg_dict["sorted_token_ids"].numpy()
                #     num_tokens_post_pad = tvm_arg_dict["num_tokens_post_pad"].numpy()
                #     index_for_scatter = sorted_token_ids[:num_tokens_post_pad[0]]
                #     for i in range(num_tokens_post_pad[0]):
                #         if index_for_scatter[i] >= 0 and index_for_scatter[i] < shape[0]:
                #             out[index_for_scatter[i]] = tmp_out[i]
                #     return out
                # out1 = scatter_out(
                #     tvm_arg_dict["silu_mul_output"],
                #     batch_size * mk.NUM_EXPERTS_PER_TOK,
                #     mk.INTERMEDIATE_SIZE,
                # )
                return tvm_arg_dict["topk_reduce_output_0"].numpy()
            else:
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
                    disco_arg_dict[f"profiler_buffer"], res_dict["profiler_buffer_res"]
                )
                sess.copy_from_worker_0(
                    res_dict["profiler_buffer_host"], res_dict["profiler_buffer_res"]
                )
                sess._sync_all()
                if mk.profiler_on:
                    for r in range(mk.world_size):
                        export_to_perfetto_trace(
                            res_dict["profiler_buffer_host"].numpy()[r],
                            f"{scheduler}-moe-layer-bs{batch_size}-tp{mk.world_size}.perfetto-trace",
                            event_type_names,
                        )
                return res_dict["output_host"].numpy(), res_dict["residual_host"].numpy()

    def std(arg_dict, mk: MegaKernelMOE):
        import flashinfer
        import torch

        torch_dev = torch.device("cuda")
        std_arg_dict = {}

        def func():
            for key, value in arg_dict.items():
                std_arg_dict[key] = value.clone().to(torch_dev)
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )

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
            print(f"sgL_config: {sgL_config}")
            invoke_fused_moe_kernel(
                std_arg_dict["hidden_state"],
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
            silu_mul_out = flashinfer.activation.silu_and_mul(
                out1.view(-1, 2 * mk.INTERMEDIATE_SIZE)
            )
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
            ret = out2.sum(dim=1)
            return ret.cpu().numpy()

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"std")
        print(f"std time: {ms:.3f} ms")
        return output

    def flashinfer(arg_dict):
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        output = torch.zeros_like(std_arg_dict["hidden_state"])
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        def flashinfer_func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )
            return fused_moe.cutlass_fused_moe(
                std_arg_dict["hidden_state"],
                std_arg_dict["topk_indices"].to(torch.int),
                std_arg_dict["topk_weights"],
                std_arg_dict["grp_up_gate_weight"],
                std_arg_dict["grp_down_weight"],
                std_arg_dict["hidden_state"].dtype,
                quant_scales=[],
                output=output,
            )

        for _ in range(10):
            flashinfer_func()
        with torch.cuda.graph(graph, stream=stream):
            flashinfer_func()
        torch.cuda.synchronize()

        def func():
            nonlocal graph
            graph.replay()

        ms = bench(func, warmup=10, repeat=30, proton_name=f"flashinfer")
        print(f"flashinfer time: {ms:.3f} ms")
        return output.cpu().numpy()

    def sglang_fused(arg_dict):
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        output = torch.zeros_like(std_arg_dict["hidden_state"])
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        def sglang_fused_func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
                renormalize=False,
            )
            out = fused_moe_sglang(
                std_arg_dict["hidden_state"],
                std_arg_dict["grp_gate_up_weight"],
                std_arg_dict["grp_down_weight"],
                gating_output,
                std_arg_dict["topk_weights"],
                std_arg_dict["topk_indices"].to(torch.int),
            )
            return out

        for _ in range(10):
            out = sglang_fused_func()
        with torch.cuda.graph(graph, stream=stream):
            sglang_fused_func()
        torch.cuda.synchronize()

        def func():
            nonlocal graph
            graph.replay()

        ms = bench(func, warmup=10, repeat=30, proton_name=f"sglang_fused")
        print(f"sglang_fused time: {ms:.3f} ms")
        return out.cpu().numpy()

    def run():
        if mega_kernel_static["main"] is not None:
            output1_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("static tir finish", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            output1_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("dynamic tir finish", flush=True)
        if mega_kernel_unfused["main"] is not None:
            output1_tir_unfused = tir(arg_dict, mega_kernel_wrapper, "unfused")
            print("unfused tir finish", flush=True)
        output1_std = std(arg_dict, mk=mega_kernel_wrapper)
        # output1_flashinfer = flashinfer(arg_dict)
        # np.testing.assert_allclose(output1_flashinfer, output1_std, rtol=1e-3, atol=1e-2)
        output1_sglang_fused = sglang_fused(arg_dict)
        np.testing.assert_allclose(output1_sglang_fused, output1_std, rtol=1e-3, atol=1e-2)
        if mega_kernel_static["main"] is not None:
            # std and tir might choose different experts because slight difference in gating output
            try:
                np.testing.assert_allclose(output1_tir_static, output1_std, rtol=1e-3, atol=1e-2)
                print("static pass", flush=True)
            except Exception as e:
                print(e)
        if mega_kernel_dynamic["main"] is not None:
            try:
                np.testing.assert_allclose(output1_tir_dynamic, output1_std, rtol=1e-3, atol=1e-2)
                print("dynamic pass", flush=True)
            except Exception as e:
                print(e)
        if mega_kernel_unfused["main"] is not None:
            try:
                np.testing.assert_allclose(output1_tir_unfused, output1_std, rtol=1e-3, atol=1e-2)
                print("unfused pass", flush=True)
            except Exception as e:
                print(e)

    if mega_kernel_wrapper.world_size == 1:
        with ProtonContext("blackwell_moe"):
            run()
    else:
        run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MegaKernel testing script.")
    parser.add_argument(
        "--scheduler",
        type=str,
        nargs="+",
        default=["static", "dynamic", "unfused"],
        choices=["static", "dynamic", "unfused", "none"],
        help="A list of test methods to run: 'static' or 'dynamic'.",
    )
    parser.add_argument(
        "--world-size", type=int, default=1, help="The number of devices for the world size."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31, 63, 127, 128],
        help="A list of batch sizes to test.",
    )
    parser.add_argument("--profiler-on", action="store_true", help="Enable the profiler.")
    args = parser.parse_args()

    testing_scheduler = set(args.scheduler)
    mega_kernel_wrapper = MegaKernelMOE(config=qwen3_30b_a3b_config, world_size=args.world_size, profiler_on=args.profiler_on)
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
    if "unfused" in testing_scheduler:
        mega_unfused_module = mega_kernel_wrapper.get_module("unfused")
        src, lib_unfused = get_source(mega_unfused_module)
        print(src)
    else:
        lib_unfused = {"main": None}
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
        test(batch_size, lib_static, lib_dynamic, lib_unfused, mega_kernel_wrapper, sess)
