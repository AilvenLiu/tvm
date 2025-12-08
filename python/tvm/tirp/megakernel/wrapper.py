import argparse
import math
import tempfile
import functools

import flashinfer
import numpy as np
import pytest
from typing import Dict

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import tirp as Tp
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace, CudaProfiler
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile
from tvm.tirp.megakernel.allreduce import AllreduceTile
from tvm.tirp.megakernel.common import *
from tvm.tirp.megakernel.batch_attn import BatchAttnTile
from tvm.tirp.megakernel.batch_merge import BatchMergeTile
from tvm.tirp.megakernel.init_etensor import InitETensorTile, f_init_const
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
from tvm.tirp.megakernel.support import (
    generate_exec_queue,
    get_inverse_plan_info,
)

class MegaKernelWrapper:

    MAX_PAGE_NUM = 8192
    PAGE_SIZE = 16
    NUM_TASK_ARGS = 10
    MAX_TOTAL_NUM_WORKERS = 1025
    MAX_NUM_KV_SPLITS = 4 * KernelConfig.SM_NUMBER * 2 * 16
    ETENSOR_WORKSPACE_SIZE = 1024 * 1024 # 4MB

    NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
    PROFILER_BUFFER_SIZE = int(1e7)
    PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS

    def __init__(self, config: Dict, tp_size, profiler_on):
        self.tp_size = tp_size
        self.MODEL_NAME = config.get("MODEL_NAME", None)
        self.TIE_WORD_EMBEDDINGS = config.get("TIE_WORD_EMBEDDINGS", None)
        self.NUM_HIDDEN_LAYERS = config.get("NUM_HIDDEN_LAYERS", None)
        self.HIDDEN_SIZE = config.get("HIDDEN_SIZE", None)
        self.VOCAB_SIZE = config.get("VOCAB_SIZE", None)
        self.INTERMEDIATE_SIZE_TP1 = config.get("INTERMEDIATE_SIZE", None)
        self.NUM_ATTENTION_HEADS_TP1 = config.get("NUM_ATTENTION_HEADS", None)
        self.NUM_KEY_VALUE_HEADS_TP1 = config.get("NUM_KEY_VALUE_HEADS", None)
        self.HEAD_DIM = config.get("HEAD_DIM", None)
        self.RMS_NORM_EPS = config.get("RMS_NORM_EPS", None)
        self.ROPE_THETA = config.get("ROPE_THETA", None)
        self.ROPE_SCALING = config.get("ROPE_SCALING", None)
        self.NUM_EXPERTS = config.get("NUM_EXPERTS", None)
        self.NUM_EXPERTS_PER_TOK = config.get("NUM_EXPERTS_PER_TOK", None)
        self.GATING_SPLIT_K_FACTOR = config.get("GATING_SPLIT_K_FACTOR", None)
        self.SPLIT_QKV_PROJECT_DICT = config.get("SPLIT_QKV_PROJECT_DICT", None)
        self.SPLIT_O_PROJECT_DICT = config.get("SPLIT_O_PROJECT_DICT", None)
        self.GATE_UP_PROJ_SPLIT_K_FACTOR_DICT = config.get("GATE_UP_PROJ_SPLIT_K_FACTOR_DICT", None)
        self.DOWN_PROJ_SPLIT_K_FACTOR_DICT = config.get("DOWN_PROJ_SPLIT_K_FACTOR_DICT", None)
        self.SPLIT_QKV_PROJECT = self.SPLIT_QKV_PROJECT_DICT[tp_size] if self.SPLIT_QKV_PROJECT_DICT is not None else None
        self.SPLIT_O_PROJECT = self.SPLIT_O_PROJECT_DICT[tp_size] if self.SPLIT_O_PROJECT_DICT is not None else None
        self.GATE_UP_PROJ_SPLIT_K_FACTOR = self.GATE_UP_PROJ_SPLIT_K_FACTOR_DICT[tp_size] if self.GATE_UP_PROJ_SPLIT_K_FACTOR_DICT is not None else None
        self.DOWN_PROJ_SPLIT_K_FACTOR = self.DOWN_PROJ_SPLIT_K_FACTOR_DICT[tp_size] if self.DOWN_PROJ_SPLIT_K_FACTOR_DICT is not None else None
        self.INTERMEDIATE_SIZE = self.INTERMEDIATE_SIZE_TP1 // tp_size if self.INTERMEDIATE_SIZE_TP1 is not None else None
        self.NUM_ATTENTION_HEADS = self.NUM_ATTENTION_HEADS_TP1 // tp_size if self.NUM_ATTENTION_HEADS_TP1 is not None else None
        self.NUM_KEY_VALUE_HEADS = self.NUM_KEY_VALUE_HEADS_TP1 // tp_size if self.NUM_KEY_VALUE_HEADS_TP1 is not None else None
        self.config = config
        self.tile_attr = {}
        self.class_list = set()
        self.etensor_and_f_init_pairs = []
        self.num_etensors = {}
        self.etensor_workspace_offset = 0
        self.profiler_on = profiler_on

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
            self.tile_scheduler.init()
            if not is_static_scheduler:
                self.tile_scheduler.next_tile()

    @T.macro
    def run_tile(self, tile, *args, **kwargs):
        event_type = T.meta_var(self.tile_attr[tile][0])
        self.smem_manager.enter_tile_runtime(tile)
        with T.cta():
            lane_id = T.thread_id([32], parent="warp")
            if self.profiler_on:
                self.profiler.start(event_type, lane_id == 0)
            tile.run(*args, **kwargs)
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

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        self.class_list.add(tile.__class__)
        return tile

    def add_etensor(self, sem_class, etensor_workspace, shape, f_init=None):
        size = functools.reduce(lambda x, y: x * y, shape, 1)
        etensor_buffer = T.decl_buffer(shape, "int32", etensor_workspace.data, elem_offset=self.etensor_workspace_offset, name="etensor")
        self.etensor_workspace_offset += size
        etensor = sem_class(etensor_buffer)
        self.etensor_and_f_init_pairs.append((etensor_buffer, f_init))
        return etensor

    def set_events_complete(self, Semaphore: Type[Union[static_scheduler.Semaphore, dynamic_scheduler.Semaphore]], etensor_workspace_global):
        if issubclass(Semaphore, static_scheduler.Semaphore):
            l = len(self.etensor_and_f_init_pairs)
            self.evt_etensor_init_complete = self.add_etensor(Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(l+1+KernelConfig.SM_NUMBER))
        else:
            self.evt_etensor_init_complete = None
        self.init_etensor_tile = self._add_tile(InitETensorTile(self.etensor_and_f_init_pairs), ProfileEventType.INIT_ETENSOR)
        if issubclass(Semaphore, static_scheduler.Semaphore):
            self.num_etensors["static"] = len(self.etensor_and_f_init_pairs)
            self.num_etensors["unfused"] = len(self.etensor_and_f_init_pairs)
        else:
            self.num_etensors["dynamic"] = len(self.etensor_and_f_init_pairs)

    @T.macro
    def task_impl_init_etensor(self, is_dynamic_sch):
        # TODO: add wait, notify and push
        self.run_tile(self.init_etensor_tile, self.tile_scheduler.m_idx, self.tile_scheduler.n_idx, self.tile_scheduler.k_idx)
        if self.evt_etensor_init_complete is not None:
            if self.tile_scheduler.m_idx < len(self.etensor_and_f_init_pairs):
                self.tile_scheduler.notify(
                    self.evt_etensor_init_complete, lambda notify_idx: (1, -1, 0), scope="cta", release=True
                )

    @T.macro
    def task_impl_wait_etensor_init_complete(self, is_dynamic_sch):
        if not is_dynamic_sch:
            with T.thread():
                warp_id = T.warp_id(
                    [KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta"
                )
                lane_id = T.thread_id([32], parent="warp")
                if self.profiler_on:
                    self.profiler.start(ProfileEventType.WAIT_ETENSOR_INIT, lane_id == 0)
                state = T.alloc_local([1], "int32")
                state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(
                            state[0],
                            self.evt_etensor_init_complete.sem.ptr_to([0]),
                        )
                    if any_sync(0xFFFFFFFF, state[0] <= KernelConfig.SM_NUMBER * (SemaphoreBase.base + 1) and state[0] > 0):
                        if lane_id == 0 and warp_id == 0:
                            T.cuda.atomic_add(self.evt_etensor_init_complete.sem.ptr_to([0]), -(SemaphoreBase.base + 1))
                        break
                    T.cuda.nano_sleep(40)
                if self.profiler_on:
                    self.profiler.end(ProfileEventType.WAIT_ETENSOR_INIT, lane_id == 0)

    def reset(self):
        self.etensor_and_f_init_pairs = []
        self.etensor_workspace_offset = 0
        self.tile_attr = {}
        self.class_list = set()

    def host_init_all(self):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                tile.host_init()

    def class_init_all(self, smem_manager: SmemManager):
        for cls in self.class_list:
            if cls.need_init:
                smem_manager.set_tile(cls)
                cls.class_init(smem_manager)

    def class_finalize_all(self):
        for cls in self.class_list:
            if cls.need_init:
                cls.class_finalize()

    def device_init_all(self, smem_manager: SmemManager):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                smem_manager.set_tile(tile)
                tile.init(smem_manager)

    def _get_func_static(self, unfused=False):
        return None

    def _get_func_dynamic(self):
        return None

    def get_func(self, scheduler: Literal["static", "dynamic", "unfused"]):
        if scheduler == "static" or scheduler == "unfused":
            return self._get_func_static(unfused=scheduler == "unfused")
        elif scheduler == "dynamic":
            return self._get_func_dynamic()
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    def get_module(self, scheduler: Literal["static", "dynamic", "unfused"]):

        @I.ir_module(tirp=True)
        class Module:

            @T.prim_func(tirp=True)
            def main():
                pass

        module: tvm.IRModule = Module
        if scheduler == "static" or scheduler == "unfused":
            module.update_func(
                module.get_global_var("main"), self._get_func_static(unfused=scheduler == "unfused")
            )
        elif scheduler == "dynamic":
            module.update_func(module.get_global_var("main"), self._get_func_dynamic())
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        return module
