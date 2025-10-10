import argparse
import math
import tempfile
import functools

import flashinfer
import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.gemm_splitk_reduce import MOETopKReduceTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyMOETile
from tvm.tirp.megakernel.static_scheduler import JobType, StaticTileScheduler
from tvm.tirp.megakernel.topk_softmax import TopkSoftmaxTile
from tvm.tirp.megakernel.moe_align import MOEAlignTile, CountAndSortExpertTokens
from tvm.tirp.megakernel.group_gemm_sm100 import GroupGEMMTile
from tvm.tirp.megakernel import static_scheduler, dynamic_scheduler
from tvm.tirp.megakernel.support import (
    generate_event_tensor_moe,
    generate_exec_queue_moe,
    get_inverse_plan_info,
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


class MegaKernel:

    # model configs
    VOCAB_SIZE = 151936
    MAX_POSITION_EMBEDDINGS = 40960
    HIDDEN_SIZE = 2048
    INTERMEDIATE_SIZE_TP1 = 768
    NUM_HIDDEN_LAYERS = 64
    NUM_ATTENTION_HEADS_TP1 = 32
    NUM_KEY_VALUE_HEADS_TP1 = 4
    HEAD_DIM = 128
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 1000000
    MAX_PAGE_NUM = 8192
    PAGE_SIZE = 16
    NUM_EXPERTS = 128
    NUM_EXPERTS_PER_TOK = 8

    GATING_SPLIT_K_FACTOR = 4
    NUM_TASK_ARGS = 10
    MAX_TOTAL_NUM_WORKERS = 65536
    MAX_NUM_KV_SPLITS = 4 * KernelConfig.SM_NUMBER * 2 * (128 + 16)
    MOE_M_PAD_SIZE = 128

    NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
    PROFILER_BUFFER_SIZE = int(1e7)
    PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS

    def __init__(self, world_size, profiler_on):
        self.world_size = world_size
        self.INTERMEDIATE_SIZE = self.INTERMEDIATE_SIZE_TP1 // world_size
        self.NUM_ATTENTION_HEADS = self.NUM_ATTENTION_HEADS_TP1 // world_size
        self.NUM_KEY_VALUE_HEADS = self.NUM_KEY_VALUE_HEADS_TP1 // world_size
        self.tile_attr = {}
        self.class_list = set()
        self.profiler_on = profiler_on

    def set_tiles(self, batch_size, low_batch):
        self.tile_attr = {}
        self.class_list = set()
        self.gate = self._add_tile(
            GemmTile(
                self.NUM_EXPERTS,
                self.HIDDEN_SIZE,
                "float16",
                "float16",
                self.GATING_SPLIT_K_FACTOR,
                128,
                128,
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
        self.group_gemm_gate_up = self._add_tile(
            GroupGEMMTile(
                self.INTERMEDIATE_SIZE * 2,
                self.HIDDEN_SIZE,
                self.NUM_EXPERTS,
                self.NUM_EXPERTS_PER_TOK,
                numel,
                "float16",
                "float16",
                low_batch=low_batch,
            ),
            ProfileEventType.GROUP_GEMM_GATE_UP,
        )
        self.silu_mul = self._add_tile(
            SiluMultiplyMOETile(
                batch_size, self.INTERMEDIATE_SIZE, numel, self.MOE_M_PAD_SIZE, "float16"
            ),
            ProfileEventType.SILU_MUL,
        )  # TODO: check if this is correct
        self.group_gemm_down = self._add_tile(
            GroupGEMMTile(
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

    def _init_profiler(self, profiler_buffer):
        if self.profiler_on:
            self.profiler = CudaProfiler(
                profiler_buffer, write_stride=self.PROFILER_WRITE_STRIDE, num_groups=self.NUM_GROUPS
            )
        else:
            self.profiler = None

    @T.macro
    def init_profiler(self, profiler_buffer):
        self._init_profiler(profiler_buffer)
        with T.cta():
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            if self.profiler_on:
                self.profiler.init(warp_id)

    def _init_tile_scheduler(
        self, is_static_scheduler, smem_manager, exec_queue, exec_tasks, exec_head, exec_tail
    ):
        self.smem_manager = smem_manager
        if is_static_scheduler:
            self.tile_scheduler = StaticTileScheduler("layer", exec_queue, smem_manager)
        else:
            self.tile_scheduler = DynamicTileScheduler(
                exec_tasks,
                exec_head,
                exec_tail,
                smem_manager,
                use_nvshmem=self.world_size > 1,
                profiler=self.profiler,
            )

    @T.macro
    def init_tile_scheduler(
        self, is_static_scheduler, smem_manager, exec_queue, exec_tasks, exec_head, exec_tail
    ):
        self._init_tile_scheduler(
            is_static_scheduler, smem_manager, exec_queue, exec_tasks, exec_head, exec_tail
        )
        with T.cta():
            bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            if is_static_scheduler:
                self.tile_scheduler.init(bx, tid)
            else:
                self.tile_scheduler.init()

    @T.macro
    def run_tile(self, tile, *args):
        event_type = T.meta_var(self.tile_attr[tile])
        self.smem_manager.enter_tile_runtime(tile)
        with T.cta():
            lane_id = T.thread_id([32], parent="warp")
            if self.profiler_on:
                self.profiler.start(event_type, lane_id == 0)
            tile.run(*args)
            if self.profiler_on:
                self.profiler.end(event_type, lane_id == 0)

    @T.macro
    def run_tile_prefetch(self, tile, *args):
        self.smem_manager.enter_tile_runtime(tile)
        with T.cta():
            lane_id = T.thread_id([32], parent="warp")
            if self.profiler_on:
                self.profiler.start(ProfileEventType.PREFETCH, lane_id == 0)
            tile.prefetch(*args)
            if self.profiler_on:
                self.profiler.end(ProfileEventType.PREFETCH, lane_id == 0)

    def _add_tile(self, tile, profiler_event_type):
        self.tile_attr[tile] = profiler_event_type
        subclass = GroupGEMMTile if isinstance(tile, GemmTile) else tile.__class__
        self.class_list.add(subclass)
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
        for tile in self.tile_attr.keys():
            smem_manager.set_tile(tile)
            tile.init(smem_manager)

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
        etensor_gating_global,
        etensor_topk_softmax_global,
        etensor_moe_align_global,
        etensor_count_and_sort_global,
        etensor_group_gemm_gate_up_global,
        etensor_silu_mul_global,
        etensor_group_gemm_down_global,
        etensor_end,
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        low_batch,
        Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]],
        Scheduler: Type[Union[static_scheduler.StaticTileScheduler, dynamic_scheduler.DynamicTileScheduler]],
    ):
        A_tensor_map_gate: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map_gate: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_gate: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_gate_up_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_gate_up_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_gate_up_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map_grp_gate_up: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_gate_up_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_gate_up_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_gate_up_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_down_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_down_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        A_tensor_map_grp_down_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map_grp_down: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_down_128: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_down_64: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        D_tensor_map_grp_down_32: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

        # initialize tile
        self.set_tiles(batch_size, low_batch)

        self.gate.set_tensor_map(A_tensor_map_gate, B_tensor_map_gate, D_tensor_map_gate, hidden_state_global, gate_weight_global, gating_output_global)
        self.group_gemm_gate_up.set_tensor_map([A_tensor_map_grp_gate_up_128, A_tensor_map_grp_gate_up_64, A_tensor_map_grp_gate_up_32], B_tensor_map_grp_gate_up, [D_tensor_map_grp_gate_up_128, D_tensor_map_grp_gate_up_64, D_tensor_map_grp_gate_up_32], reordered_hidden_state_global, grp_gate_up_weight_global, gate_up_output_global)
        self.group_gemm_down.set_tensor_map([A_tensor_map_grp_down_128, A_tensor_map_grp_down_64, A_tensor_map_grp_down_32], B_tensor_map_grp_down, [D_tensor_map_grp_down_128, D_tensor_map_grp_down_64, D_tensor_map_grp_down_32], silu_mul_output_global, grp_down_weight_global, topk_reduce_output_global)

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
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data))
                self.device_init_all(smem_manager)
                self.class_init_all(smem_manager)


                # initialize event tensors
                evt_gating = T.meta_var(Semaphore(-1, etensor_gating_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_topk_softmax = T.meta_var(Semaphore(-1, etensor_topk_softmax_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_moe_align = T.meta_var(Semaphore(-1, etensor_moe_align_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_count_and_sort = T.meta_var(Semaphore(-1, etensor_count_and_sort_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_group_gemm_gate_up = T.meta_var(Semaphore(-1, etensor_group_gemm_gate_up_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_silu_mul = T.meta_var(Semaphore(-1, etensor_silu_mul_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_group_gemm_down = T.meta_var(Semaphore(-1, etensor_group_gemm_down_global, decrement=True, use_nvshmem=self.world_size > 1))
                evt_end = T.meta_var(Semaphore(-1, etensor_end, decrement=True, use_nvshmem=self.world_size>1))

                # initialize tile scheduler and smem_manager
                self.init_tile_scheduler(issubclass(Scheduler, StaticTileScheduler), smem_manager, exec_queue, exec_task, exec_head, exec_tail)
                smem_manager.init()
                topk_ids_flattened = topk_indices_global.view(-1)
                topk_weights_flattened = topk_weights_global.view(-1)
                while self.tile_scheduler.valid():
                    if self.tile_scheduler.task_type == JobType.MOE_GATING.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_gating.semaphore_notify(0, pre_notify=True)
                            self.tile_scheduler.push_task(
                                evt_gating, 1,
                                lambda trigger_idx: (
                                    -1, self.topk_softmax.PERSISTENT_SM_NUMBER,
                                    lambda push_idx: (JobType.MOE_TOPK_SOFTMAX.value, push_idx, 0, 0)
                                ), "cta", "cta"
                            )
                        self.run_tile(self.gate, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, self.profiler)
                        if wg_id == 0:
                            T.cuda.warpgroup_sync(1)
                            if tid == 0:
                                evt_gating.semaphore_notify(0)
                    elif self.tile_scheduler.task_type == JobType.MOE_TOPK_SOFTMAX.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_topk_softmax.semaphore_notify(0, pre_notify=True)
                                self.tile_scheduler.push_task(
                                    evt_topk_softmax, 1,
                                    lambda trigger_idx: (
                                        -1, 1,
                                        lambda push_idx: (JobType.MOE_ALIGN.value, 0, 0, 0)
                                    ), "thread", "thread"
                                )
                        evt_gating.semaphore_wait(0)
                        self.run_tile(self.topk_softmax, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, gating_output_global, topk_weights_global, topk_indices_global)
                        T.cuda.cta_sync()
                        if tid == 0:
                            evt_topk_softmax.semaphore_notify(0)
                    elif self.tile_scheduler.task_type == JobType.MOE_ALIGN.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_moe_align.semaphore_notify(0, pre_notify=True)
                            self.tile_scheduler.push_task(
                                evt_moe_align, 1,
                                lambda trigger_idx: (
                                    -1, KernelConfig.SM_NUMBER,
                                    lambda push_idx: (JobType.MOE_COUNT_AND_SORT.value, push_idx, 0, 0)
                                ), "cta", "cta"
                            )
                        evt_topk_softmax.semaphore_wait(0)
                        self.run_tile(self.align, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, topk_ids_flattened, sorted_token_ids_global, expert_ids_global, num_tokens_post_pad_global, cumsum_buffer_global, num_valid_tokens_global)
                        T.cuda.cta_sync()
                        if tid == 0:
                            if issubclass(Scheduler, DynamicTileScheduler):
                                etensor_end[0] = (evt_end.base + 1) * (num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE) * (self.HIDDEN_SIZE // GroupGEMMTile.BLK_N)
                            evt_moe_align.semaphore_notify(0)
                    elif self.tile_scheduler.task_type == JobType.MOE_COUNT_AND_SORT.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_count_and_sort.semaphore_notify(0, pre_notify=True)
                            n_axis_len = T.meta_var(self.INTERMEDIATE_SIZE * 2 // GroupGEMMTile.BLK_N)
                            self.tile_scheduler.push_task(
                                evt_count_and_sort, 1,
                                lambda trigger_idx: (
                                    -1, num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE * n_axis_len,
                                    lambda push_idx: (JobType.MOE_GROUP_GEMM_GATE_UP.value, push_idx // n_axis_len, push_idx % n_axis_len, 0)
                                ), "cta", "cta"
                            )
                        evt_moe_align.semaphore_wait(0)
                        self.run_tile(self.count_and_sort_expert_tokens, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, topk_ids_flattened, sorted_token_ids_global, cumsum_buffer_global, hidden_state_global, reordered_hidden_state_global)
                        T.cuda.cta_sync()
                        if tid == 0:
                            evt_count_and_sort.semaphore_notify(0)
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_GATE_UP.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_group_gemm_gate_up.semaphore_notify(self.tile_scheduler.m_idx, pre_notify=True)
                            self.tile_scheduler.push_task(
                                evt_group_gemm_gate_up, 1,
                                lambda trigger_idx: (
                                    -1, self.INTERMEDIATE_SIZE // SiluMultiplyMOETile.TILE_SIZE,
                                    lambda push_idx: (JobType.MOE_SILU_MULTIPLY.value, self.tile_scheduler.m_idx, push_idx, 0)
                                ), "warp", "warp"
                            )
                        evt_count_and_sort.semaphore_wait_warp(0)
                        if issubclass(Scheduler, DynamicTileScheduler) or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE:
                            self.run_tile(self.group_gemm_gate_up, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, self.profiler)
                        if wg_id == 0:
                            T.cuda.warpgroup_sync(1)
                            if tid == 0:
                                evt_group_gemm_gate_up.semaphore_notify(self.tile_scheduler.m_idx)
                    elif self.tile_scheduler.task_type == JobType.MOE_SILU_MULTIPLY.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_silu_mul.semaphore_notify(self.tile_scheduler.m_idx, pre_notify=True)
                            self.tile_scheduler.push_task(
                                evt_silu_mul, 1,
                                lambda trigger_idx: (
                                    -1, self.HIDDEN_SIZE // GroupGEMMTile.BLK_N,
                                    lambda push_idx: (JobType.MOE_GROUP_GEMM_DOWN.value, self.tile_scheduler.m_idx, push_idx, 0)
                                ), "warp", "warp"
                            )
                        evt_group_gemm_gate_up.semaphore_wait(self.tile_scheduler.m_idx)
                        if issubclass(Scheduler, DynamicTileScheduler) or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE:
                            self.run_tile(self.silu_mul, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, gate_up_output_global, silu_mul_output_global, sorted_token_ids_global)
                        T.cuda.cta_sync()
                        if tid == 0:
                            evt_silu_mul.semaphore_notify(self.tile_scheduler.m_idx)
                    elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_DOWN.value:
                        if issubclass(Scheduler, DynamicTileScheduler):
                            if tid == 0:
                                evt_end.semaphore_notify(0, pre_notify=True)
                            self.tile_scheduler.push_task(
                                evt_end, 1,
                                lambda trigger_idx: (
                                    -1, KernelConfig.SM_NUMBER,
                                    lambda push_idx: (JobType.END.value, 0, 0, 0)
                                ), "warp", "warp"
                            )
                        evt_silu_mul.semaphore_wait_warp(self.tile_scheduler.m_idx)
                        if issubclass(Scheduler, DynamicTileScheduler) or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE:
                            self.run_tile(self.group_gemm_down, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, self.profiler)
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
            etensor_gating_ptr: T.handle,
            etensor_topk_softmax_ptr: T.handle,
            etensor_moe_align_ptr: T.handle,
            etensor_count_and_sort_ptr: T.handle,
            etensor_group_gemm_gate_up_ptr: T.handle,
            etensor_silu_mul_ptr: T.handle,
            etensor_group_gemm_down_ptr: T.handle,

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
            etensor_gating_global = T.match_buffer(etensor_gating_ptr, [1], "int32", scope="global")
            etensor_topk_softmax_global = T.match_buffer(etensor_topk_softmax_ptr, [1], "int32", scope="global")
            etensor_moe_align_global = T.match_buffer(etensor_moe_align_ptr, [1], "int32", scope="global")
            etensor_count_and_sort_global = T.match_buffer(etensor_count_and_sort_ptr, [1], "int32", scope="global")
            max_blocks_padded = T.int32()
            etensor_group_gemm_gate_up_global = T.match_buffer(etensor_group_gemm_gate_up_ptr, [max_blocks_padded], "int32", scope="global")
            etensor_silu_mul_global = T.match_buffer(etensor_silu_mul_ptr, [max_blocks_padded], "int32", scope="global")
            etensor_group_gemm_down_global = T.match_buffer(etensor_group_gemm_down_ptr, [1], "int32", scope="global")

            # exec queue
            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")

            @T.macro
            def run(low_batch, dynamic_gemm_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_gating_global, etensor_topk_softmax_global, etensor_moe_align_global,
                    etensor_count_and_sort_global, etensor_group_gemm_gate_up_global, etensor_silu_mul_global, etensor_group_gemm_down_global,
                    None, profiler_buffer, exec_queue, None, None, None, low_batch,
                    static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
                )

            if batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True)
            elif batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True)
            else:
                run(low_batch=True, dynamic_gemm_size=False)
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
            etensor_gating_ptr: T.handle,
            etensor_topk_softmax_ptr: T.handle,
            etensor_moe_align_ptr: T.handle,
            etensor_count_and_sort_ptr: T.handle,
            etensor_group_gemm_gate_up_ptr: T.handle,
            etensor_silu_mul_ptr: T.handle,
            etensor_group_gemm_down_ptr: T.handle,
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
            etensor_gating_global = T.match_buffer(etensor_gating_ptr, [1], "int32", scope="global")
            etensor_topk_softmax_global = T.match_buffer(etensor_topk_softmax_ptr, [1], "int32", scope="global")
            etensor_moe_align_global = T.match_buffer(etensor_moe_align_ptr, [1], "int32", scope="global")
            etensor_count_and_sort_global = T.match_buffer(etensor_count_and_sort_ptr, [1], "int32", scope="global")
            max_blocks_padded = T.int32()
            etensor_group_gemm_gate_up_global = T.match_buffer(etensor_group_gemm_gate_up_ptr, [max_blocks_padded], "int32", scope="global")
            etensor_silu_mul_global = T.match_buffer(etensor_silu_mul_ptr, [max_blocks_padded], "int32", scope="global")
            etensor_group_gemm_down_global = T.match_buffer(etensor_group_gemm_down_ptr, [1], "int32", scope="global")
            etensor_end_global = T.match_buffer(etensor_end_ptr, [1], "int32", scope="global")


            # exec queue
            queue_tasks_global = T.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = T.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = T.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)

            @T.macro
            def run(low_batch, dynamic_gemm_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_gating_global, etensor_topk_softmax_global, etensor_moe_align_global,
                    etensor_count_and_sort_global, etensor_group_gemm_gate_up_global, etensor_silu_mul_global, etensor_group_gemm_down_global,
                    etensor_end_global, profiler_buffer, None, queue_tasks_global, queue_head_global, queue_tail_global, low_batch,
                    dynamic_scheduler.Semaphore, dynamic_scheduler.DynamicTileScheduler
                )

            if batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True)
            elif batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True)
            else:
                run(low_batch=True, dynamic_gemm_size=False)
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


def prepare_data(batch_size, mk: MegaKernel):
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
    max_num_tokens_padded = batch_size * mk.NUM_EXPERTS_PER_TOK + mk.NUM_EXPERTS * (
        mk.MOE_M_PAD_SIZE - 1
    )
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
def test(batch_size, mega_kernel_static, mega_kernel_dynamic, mega_kernel_wrapper, sess):
    arg_dict = prepare_data(batch_size, mega_kernel_wrapper)

    def tir(arg_dict, mk: MegaKernel, scheduler: Literal["static", "dynamic"]):
        REPEAT = 100
        DEV = tvm.cuda(0)
        tvm_arg_dict = {}
        target = tvm.target.Target("cuda")
        if scheduler == "static":
            # static schedule
            exec_queue = generate_exec_queue_moe(batch_size, "static")
            tvm_arg_dict[f"exec_queue"] = tvm.runtime.tensor(exec_queue, DEV)
        else:
            exec_queue = generate_exec_queue_moe(batch_size, "dynamic")
            for i in range(REPEAT):
                tvm_arg_dict[f"queue_tasks_{i}"] = tvm.runtime.tensor(exec_queue.tasks, DEV)
                tvm_arg_dict[f"queue_head_{i}"] = tvm.runtime.tensor(exec_queue.head, DEV)
                tvm_arg_dict[f"queue_tail_{i}"] = tvm.runtime.tensor(exec_queue.tail, DEV)

        for key, value in arg_dict.items():
            tvm_arg_dict[key] = tvm.runtime.tensor(value, device=DEV)

        tvm_arg_dict["output"] = tvm.runtime.tensor(
            np.zeros((batch_size, mk.HIDDEN_SIZE), dtype=np.float16), device=DEV
        )

        for i in range(REPEAT):
            tvm_arg_dict[f"residual_{i}"] = tvm.runtime.tensor(arg_dict["residual"], device=DEV)
            # initial tensor must be 0
            tvm_arg_dict[f"gating_output_{i}"] = tvm.runtime.tensor(
                arg_dict["gating_output"], device=DEV
            )
            tvm_arg_dict[f"topk_reduce_output_{i}"] = tvm.runtime.tensor(
                arg_dict["topk_reduce_output"], device=DEV
            )
            # generate event tensor
            (
                tvm_arg_dict[f"etensor_gating_{i}"],
                tvm_arg_dict[f"etensor_topk_softmax_{i}"],
                tvm_arg_dict[f"etensor_moe_align_{i}"],
                tvm_arg_dict[f"etensor_count_and_sort_{i}"],
                tvm_arg_dict[f"etensor_group_gemm_gate_up_{i}"],
                tvm_arg_dict[f"etensor_silu_mul_{i}"],
                tvm_arg_dict[f"etensor_group_gemm_down_{i}"],
                tvm_arg_dict[f"etensor_end_{i}"],
            ) = generate_event_tensor_moe(batch_size, mk.world_size)
        tvm_arg_dict[f"profiler_buffer"] = tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), device=DEV
        )

        if mk.world_size > 1:
            raise ValueError(f"Unsupported world size: {mk.world_size}")
        with target:
            iter = 0

            if scheduler == "static":
                kernel = mega_kernel_static["main"]
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
                        work_arg_dict["grp_gate_up_weight"],
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
                        work_arg_dict[f"etensor_gating_{iter}"],
                        work_arg_dict[f"etensor_topk_softmax_{iter}"],
                        work_arg_dict[f"etensor_moe_align_{iter}"],
                        work_arg_dict[f"etensor_count_and_sort_{iter}"],
                        work_arg_dict[f"etensor_group_gemm_gate_up_{iter}"],
                        work_arg_dict[f"etensor_silu_mul_{iter}"],
                        work_arg_dict[f"etensor_group_gemm_down_{iter}"],
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
                        work_arg_dict["grp_gate_up_weight"],
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
                        work_arg_dict[f"etensor_gating_{iter}"],
                        work_arg_dict[f"etensor_topk_softmax_{iter}"],
                        work_arg_dict[f"etensor_moe_align_{iter}"],
                        work_arg_dict[f"etensor_count_and_sort_{iter}"],
                        work_arg_dict[f"etensor_group_gemm_gate_up_{iter}"],
                        work_arg_dict[f"etensor_silu_mul_{iter}"],
                        work_arg_dict[f"etensor_group_gemm_down_{iter}"],
                        work_arg_dict[f"etensor_end_{iter}"],
                        # exec queue
                        work_arg_dict[f"queue_tasks_{iter}"],
                        work_arg_dict[f"queue_head_{iter}"],
                        work_arg_dict[f"queue_tail_{iter}"],
                        work_arg_dict[f"profiler_buffer"],
                    )
                    iter += 1

            if mk.world_size == 1:
                ms = bench(func, warmup=1, repeat=7, proton_name=f"tir-{scheduler}")
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

    def std(arg_dict, mk: MegaKernel):
        import flashinfer
        import torch

        FULL_INTERMEDIATE_SIZE = mk.INTERMEDIATE_SIZE * mk.world_size
        FULL_NUM_ATTENTION_HEADS = mk.NUM_ATTENTION_HEADS * mk.world_size
        FULL_NUM_KEY_VALUE_HEADS = mk.NUM_KEY_VALUE_HEADS * mk.world_size

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

        def func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
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

        ms = bench(func, warmup=10, repeat=30, proton_name=f"flashinfer")
        print(f"flashinfer time: {ms:.3f} ms")
        return output.cpu().numpy()

    def sglang_fused(arg_dict):
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.clone().to(torch_dev)
        output = torch.zeros_like(std_arg_dict["hidden_state"])

        def func():
            gating_output = std_arg_dict["hidden_state"] @ std_arg_dict["gate_weight"].T
            topk_softmax(
                gating_output=gating_output,
                topk_weights=std_arg_dict["topk_weights"],
                topk_ids=std_arg_dict["topk_indices"],
            )
            return fused_moe_sglang(
                std_arg_dict["hidden_state"],
                std_arg_dict["grp_gate_up_weight"],
                std_arg_dict["grp_down_weight"],
                gating_output,
                std_arg_dict["topk_weights"],
                std_arg_dict["topk_indices"].to(torch.int),
            )

        output = func()
        ms = bench(func, warmup=10, repeat=30, proton_name=f"sglang_fused")
        print(f"sglang_fused time: {ms:.3f} ms")
        return output.cpu().numpy()

    def run():
        if mega_kernel_static["main"] is not None:
            output1_tir_static = tir(arg_dict, mega_kernel_wrapper, "static")
            print("static tir finish", flush=True)
        if mega_kernel_dynamic["main"] is not None:
            output1_tir_dynamic = tir(arg_dict, mega_kernel_wrapper, "dynamic")
            print("dynamic tir finish", flush=True)
        output1_std = std(arg_dict, mk=mega_kernel_wrapper)
        output1_flashinfer = flashinfer(arg_dict)
        np.testing.assert_allclose(output1_flashinfer, output1_std, rtol=1e-3, atol=1e-2)
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
        default=["static", "dynamic"],
        choices=["static", "dynamic", "none"],
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
    mega_kernel_wrapper = MegaKernel(world_size=args.world_size, profiler_on=args.profiler_on)
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
        test(batch_size, lib_static, lib_dynamic, mega_kernel_wrapper, sess)
