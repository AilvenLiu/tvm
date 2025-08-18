import math
import ml_dtypes
import numpy as np
from enum import Enum
import pytest

import tvm
from tvm.script import tir as T, tirp as Tp
import tvm.testing
from tvm.script.ir_builder import IRBuilder
import flashinfer
from tvm.tirp.bench.utils import ProtonContext, bench, export_to_perfetto_trace
from tvm.tirp.megakernel.common import KernelConfig, ceildiv, JobType, get_source, ProfileEventType, event_type_names
from tvm.tirp.megakernel.gemm import GemmTile
from tvm.tirp.megakernel.split_silu_multiply import SiluMultiplyTile
from tvm.tirp.megakernel.gemm_splitk_reduce import SplitKReduceTile
from tvm.tirp.megakernel.add_rmsnorm import AddRMSNormTile
from tvm.tirp.megakernel.static_scheduler import StaticTileScheduler
from tvm.tirp.megakernel.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost, stg_v4

INTERMEDIATE_SIZE = 25600
HIDDEN_SIZE = 5120
dtype = "float16"
DOWN_PROJ_SPLIT_K_FACTOR = 10

# profiling
NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
PROFILER_BUFFER_SIZE = int(1e6)
PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS
PROFILER_ON = False
# fmt: off
class MegaKernel:
    def __init__(self):
        self.tile_list = []
        self.class_list = set()

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

    def add_tile(self, tile):
        self.tile_list.append(tile)
        self.class_list.add(tile.__class__)

    def get_static_scheduler_func(self):
        from .static_scheduler import Semaphore

        @T.prim_func(tirp=True)
        def mega_kernel_static(
            input_ptr: T.handle,
            W_gate_up: T.Buffer((INTERMEDIATE_SIZE*2, HIDDEN_SIZE), dtype, layout="default"),
            out_gate_up_proj_ptr: T.handle,
            out_silu_multiply_ptr: T.handle,
            W_down_proj: T.Buffer((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype, layout="default"),
            partial_sum_down_proj_ptr: T.handle,
            out_down_proj_ptr: T.handle,
            residual_ptr: T.handle,
            rmsnorm_weight: T.Buffer((HIDDEN_SIZE, ), dtype, layout="default"),
            etensor_gate_up_proj: T.Buffer((INTERMEDIATE_SIZE//GemmTile.BLK_N, ), dtype="int32"),
            etensor_down_proj: T.Buffer((DOWN_PROJ_SPLIT_K_FACTOR, ), dtype="int32"),
            etensor_down_proj_reduce: T.Buffer((HIDDEN_SIZE//GemmTile.BLK_N, ), dtype="int32"),
            etensor_add_rms_norm_ptr: T.handle,
            exec_queue: T.Buffer((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4), "int32"),
            profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
        ):
            batch_size = T.int32()
            input = T.match_buffer(input_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            out_gate_up_proj = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE*2], dtype, layout="default")
            out_silu_multiply = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], dtype, layout="default")
            partial_sum_down_proj = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], "float32", layout="default")
            out_down_proj = T.match_buffer(out_down_proj_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            residual = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            etensor_add_rms_norm = T.match_buffer(etensor_add_rms_norm_ptr, [batch_size], dtype="int32")
            A_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(input, W_gate_up, out_gate_up_proj, split_k_factor=1))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj, out_silu_multiply))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply, W_down_proj, partial_sum_down_proj, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj, out_down_proj))
            add_rms_norm_tile = T.meta_var(AddRMSNormTile(out_down_proj, residual, rmsnorm_weight))
            self.add_tile(gemm_gate_up_proj_tile)
            self.add_tile(silu_multiply_tile)
            self.add_tile(gemm_down_proj_tile)
            self.add_tile(down_proj_reduce_tile)
            self.add_tile(add_rms_norm_tile)
            gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
            gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)
            self.host_init_all()
            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)

                with T.cta():
                    wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                    warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                    tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                    lane_id = T.thread_id([32], parent="warp")
                    buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    if PROFILER_ON:
                        T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, warp_id)
                    self.class_init_all(pool)
                    self.device_init_all(pool)
                    # initialize event tensors
                    evt_gate_up_proj = T.meta_var(Semaphore(2, etensor_gate_up_proj))
                    evt_down_proj = T.meta_var(Semaphore(INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR, etensor_down_proj))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), etensor_down_proj_reduce))
                    evt_add_rms_norm = T.meta_var(
                        Semaphore(HIDDEN_SIZE // down_proj_reduce_tile.N_TILE, etensor_add_rms_norm)
                    )
                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(StaticTileScheduler(prefix="gemm", exec_queue=exec_queue, pool_allocator=pool))
                    tile_scheduler.init(bx, tid)
                    while tile_scheduler.valid():
                        if tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx-INTERMEDIATE_SIZE//GemmTile.BLK_N)
                                    else:
                                        evt_gate_up_proj.semaphore_notify(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            evt_gate_up_proj.semaphore_wait(tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_down_proj.semaphore_notify(tile_scheduler.m_idx // (INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        elif tile_scheduler.task_type == JobType.GEMM_DOWN_PROJ.value:
                            evt_down_proj.semaphore_wait(tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            gemm_down_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if wg_id == 0:
                                T.ptx.bar.sync(1, 128)
                                if tid == 0:
                                    evt_down_proj_reduce.semaphore_notify(tile_scheduler.n_idx // (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N))
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_DOWN_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            evt_down_proj_reduce.semaphore_wait(tile_scheduler.n_idx)
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            T.tvm_storage_sync("shared")
                            if tid == 0:
                                evt_add_rms_norm.semaphore_notify(tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        elif tile_scheduler.task_type == JobType.ADD_RMS_NORM.value:
                            evt_add_rms_norm.semaphore_wait(tile_scheduler.m_idx//(batch_size//down_proj_reduce_tile.M_split))
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        tile_scheduler.next_tile()
                    self.class_finalize_all()
        return mega_kernel_static

    def get_dynamic_scheduler_func(self):
        from .dynamic_scheduler import Semaphore

        @T.prim_func(tirp=True)
        def mega_kernel_dynamic(
            input_ptr: T.handle,
            W_gate_up: T.Buffer((INTERMEDIATE_SIZE*2, HIDDEN_SIZE), dtype, layout="default"),
            out_gate_up_proj_ptr: T.handle,
            out_silu_multiply_ptr: T.handle,
            W_down_proj: T.Buffer((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype, layout="default"),
            partial_sum_down_proj_ptr: T.handle,
            out_down_proj_ptr: T.handle,
            residual_ptr: T.handle,
            rmsnorm_weight: T.Buffer((HIDDEN_SIZE, ), dtype, layout="default"),
            etensor_gate_up_proj: T.Buffer((INTERMEDIATE_SIZE//GemmTile.BLK_N, ), dtype="int32"),
            etensor_down_proj: T.Buffer((DOWN_PROJ_SPLIT_K_FACTOR, ), dtype="int32"),
            etensor_down_proj_reduce: T.Buffer((HIDDEN_SIZE//GemmTile.BLK_N, ), dtype="int32"),
            etensor_add_rms_norm_ptr: T.handle,
            etensor_end: T.Buffer((1, ), "int32"),
            queue_tasks: T.Buffer((DynamicTileScheduler.MAX_TASKS, 4), "int32"),
            queue_head: T.Buffer((1,), "int32"),
            queue_tail: T.Buffer((1,), "int32"),
            actual_order: T.Buffer((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4), "int32"),
            profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
        ):
            batch_size = T.int32()
            input = T.match_buffer(input_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            out_gate_up_proj = T.match_buffer(out_gate_up_proj_ptr, [batch_size, INTERMEDIATE_SIZE*2], dtype, layout="default")
            out_silu_multiply = T.match_buffer(out_silu_multiply_ptr, [batch_size, INTERMEDIATE_SIZE], dtype, layout="default")
            partial_sum_down_proj = T.match_buffer(partial_sum_down_proj_ptr, [DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE], "float32", layout="default")
            out_down_proj = T.match_buffer(out_down_proj_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            residual = T.match_buffer(residual_ptr, [batch_size, HIDDEN_SIZE], dtype, layout="default")
            etensor_add_rms_norm = T.match_buffer(etensor_add_rms_norm_ptr, [batch_size], dtype="int32")
            A_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_up_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            A_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            B_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
            D_tensor_map_down_proj: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)

            gemm_gate_up_proj_tile = T.meta_var(GemmTile(input, W_gate_up, out_gate_up_proj, split_k_factor=1))
            silu_multiply_tile = T.meta_var(SiluMultiplyTile(out_gate_up_proj, out_silu_multiply))
            gemm_down_proj_tile = T.meta_var(GemmTile(out_silu_multiply, W_down_proj, partial_sum_down_proj, split_k_factor=DOWN_PROJ_SPLIT_K_FACTOR))
            down_proj_reduce_tile = T.meta_var(SplitKReduceTile(partial_sum_down_proj, out_down_proj))
            add_rms_norm_tile = T.meta_var(AddRMSNormTile(out_down_proj, residual, rmsnorm_weight))
            self.add_tile(gemm_gate_up_proj_tile)
            self.add_tile(silu_multiply_tile)
            self.add_tile(gemm_down_proj_tile)
            self.add_tile(down_proj_reduce_tile)
            self.add_tile(add_rms_norm_tile)
            gemm_gate_up_proj_tile.set_tensor_map(A_tensor_map_up_proj, B_tensor_map_up_proj, D_tensor_map_up_proj)
            gemm_down_proj_tile.set_tensor_map(A_tensor_map_down_proj, B_tensor_map_down_proj, D_tensor_map_down_proj)
            self.host_init_all()
            with T.kernel():
                bx = T.cta_id([KernelConfig.SM_NUMBER], parent="kernel")
                profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)
                profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
                with T.cta():
                    wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER], parent="cta")
                    warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
                    tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
                    lane_id = T.thread_id([32], parent="warp")
                    buf = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
                    pool = T.meta_var(Tp.PoolAllocator(buf.data))
                    if PROFILER_ON:
                        T.timer_init_cuda(profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, NUM_GROUPS, warp_id)
                    self.class_init_all(pool)
                    self.device_init_all(pool)
                    # initialize event tensors
                    evt_gate_up_proj = T.meta_var(Semaphore(2, etensor_gate_up_proj))
                    evt_down_proj = T.meta_var(Semaphore(INTERMEDIATE_SIZE//SiluMultiplyTile.TILE_SIZE // DOWN_PROJ_SPLIT_K_FACTOR, etensor_down_proj))
                    evt_down_proj_reduce = T.meta_var(Semaphore(DOWN_PROJ_SPLIT_K_FACTOR * (down_proj_reduce_tile.N_TILE // GemmTile.BLK_N), etensor_down_proj_reduce))
                    evt_add_rms_norm = T.meta_var(
                        Semaphore(HIDDEN_SIZE // down_proj_reduce_tile.N_TILE, etensor_add_rms_norm)
                    )
                    evt_end = T.meta_var(
                        Semaphore(batch_size, etensor_end)
                    )
                    # initialize tile scheduler
                    tile_scheduler = T.meta_var(DynamicTileScheduler(queue_tasks, queue_head, queue_tail, pool_allocator=pool))
                    tile_scheduler.init(warp_id)
                    # tile_idx = T.alloc_local([1], "int32")
                    while tile_scheduler.valid():
                        # if tid == 0:
                        #     T.cuda.func_call("stg_v4", tile_scheduler.task_type, tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, T.address_of(actual_order[bx, tile_idx[0], 0]), source_code=stg_v4)
                        #     tile_idx[0] += 1
                        if tile_scheduler.task_type == JobType.GEMM_GATE_UP_PROJ.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            gemm_gate_up_proj_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            if tile_scheduler.n_idx >= INTERMEDIATE_SIZE // GemmTile.BLK_N:
                                offset = T.meta_var(tile_scheduler.n_idx-INTERMEDIATE_SIZE//GemmTile.BLK_N)
                                tile_scheduler.push_task(JobType.SPLIT_SILU_MULTIPLY.value, offset, 0, 0, warp_id, evt_gate_up_proj, offset, use_barrier=False)
                            else:
                                tile_scheduler.push_task(JobType.SPLIT_SILU_MULTIPLY.value, tile_scheduler.n_idx, 0, 0, warp_id, evt_gate_up_proj, tile_scheduler.n_idx, use_barrier=False)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.GEMM_GATE_UP_PROJ, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        elif tile_scheduler.task_type == JobType.SPLIT_SILU_MULTIPLY.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                            silu_multiply_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx, tile_scheduler)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.SPLIT_SILU_MULTIPLY, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
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
                                    lane_id == 0,
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
                                    lane_id == 0,
                                )
                        elif tile_scheduler.task_type == JobType.DOWN_PROJ_REDUCE.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            down_proj_reduce_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(JobType.ADD_RMS_NORM.value, tile_scheduler.m_idx * (batch_size //down_proj_reduce_tile.M_split), 0, 0, batch_size //down_proj_reduce_tile.M_split, 0, warp_id, lane_id, evt_add_rms_norm, tile_scheduler.m_idx)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.DOWN_PROJ_REDUCE, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        elif tile_scheduler.task_type == JobType.ADD_RMS_NORM.value:
                            if PROFILER_ON:
                                T.timer_start_cuda(ProfileEventType.ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                            add_rms_norm_tile.run(tile_scheduler.m_idx, tile_scheduler.n_idx, tile_scheduler.k_idx)
                            tile_scheduler.push_tasks_along_dim(JobType.END.value, 0, 0, 0, KernelConfig.SM_NUMBER, 0, warp_id, lane_id, evt_end, 0)
                            if PROFILER_ON:
                                T.timer_end_cuda(ProfileEventType.ADD_RMS_NORM, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, tid == 0)
                        if PROFILER_ON:
                            T.timer_start_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                        tile_scheduler.next_tile(warp_id)
                        if PROFILER_ON:
                            T.timer_end_cuda(ProfileEventType.FETCH, profiler_buffer.data, profiler_tag.data, profiler_write_offset.data, PROFILER_WRITE_STRIDE, lane_id == 0)
                    self.class_finalize_all()

        return mega_kernel_dynamic
# fmt: on
@tvm.testing.requires_cuda_compute_version(10, exact=True)
@pytest.mark.skip
def test(batch_size, mod_mlp_static, mod_mlp_dynamic):

    def generate_exec_queue_static():
        exec_queue = np.zeros((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4), dtype=np.int32)
        central_queue = []
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
            central_queue.append((m, 0, 0, JobType.ADD_RMS_NORM.value))
        tile_idx = 0
        while len(central_queue) > 0:
            for bx in range(KernelConfig.SM_NUMBER):
                if len(central_queue) > 0:
                    m, n, k, task_type = central_queue.pop(0)
                    exec_queue[bx, tile_idx, 0] = m
                    exec_queue[bx, tile_idx, 1] = n
                    exec_queue[bx, tile_idx, 2] = k
                    exec_queue[bx, tile_idx, 3] = task_type
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
        # queue = np.load("actual_order.npy")
        # queue=np.roll(queue, shift=-1, axis=2)
        # return queue

    def generate_exec_queue_dynamic():
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        # gate_up_proj
        for n in range(INTERMEDIATE_SIZE // GemmTile.BLK_N):
            exec_queue.enqueue(JobType.GEMM_GATE_UP_PROJ.value, 0, n, 0)
            exec_queue.enqueue(JobType.GEMM_GATE_UP_PROJ.value, 0, n+INTERMEDIATE_SIZE//GemmTile.BLK_N, 0)
        return exec_queue

    def prepare_data():
        import torch
        arg_dict = {}
        arg_dict["input"] = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["W_gate_up"] = torch.zeros((INTERMEDIATE_SIZE*2, HIDDEN_SIZE), dtype=torch.float16)
        torch.nn.init.xavier_normal_(arg_dict["W_gate_up"], gain=1.0)
        arg_dict["out_gate_up_proj"] = torch.zeros((batch_size, INTERMEDIATE_SIZE*2), dtype=torch.float16)
        arg_dict["out_silu_multiply"] = torch.zeros((batch_size, INTERMEDIATE_SIZE), dtype=torch.float16)
        arg_dict["W_down_proj"] = torch.zeros((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.float16) * 0.01
        torch.nn.init.xavier_normal_(arg_dict["W_down_proj"], gain=1.0)
        arg_dict["partial_sum_down_proj"] = torch.zeros((DOWN_PROJ_SPLIT_K_FACTOR, batch_size, HIDDEN_SIZE), dtype=torch.float32)
        arg_dict["out_down_proj"] = torch.zeros((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        arg_dict["residual"] = torch.zeros((batch_size, HIDDEN_SIZE), dtype=torch.float16)
        torch.nn.init.xavier_normal_(arg_dict["residual"], gain=1.0)
        arg_dict["rmsnorm_weight"] = torch.randn((HIDDEN_SIZE, ), dtype=torch.float16)
        arg_dict["etensor_gate_up_proj"] = torch.zeros((INTERMEDIATE_SIZE//GemmTile.BLK_N, ), dtype=torch.int32)
        arg_dict["etensor_down_proj"] = torch.zeros((DOWN_PROJ_SPLIT_K_FACTOR, ), dtype=torch.int32)
        arg_dict["etensor_down_proj_reduce"] = torch.zeros((HIDDEN_SIZE//GemmTile.BLK_N, ), dtype=torch.int32)
        arg_dict["etensor_add_rms_norm"] = torch.zeros((batch_size, ), dtype=torch.int32)
        arg_dict["etensor_end"] = torch.zeros((1, ), dtype=torch.int32)
        arg_dict["actual_order"] = torch.full((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS, 4), JobType.END.value, dtype=torch.int32)
        arg_dict["profiler_buffer"] = torch.zeros((PROFILER_BUFFER_SIZE, ), dtype=torch.uint64)
        return arg_dict

    arg_dict = prepare_data()
    tvm_dev = tvm.cuda(0)
    tvm_arg_dict = {}
    for key, value in arg_dict.items():
        tvm_arg_dict[key] = tvm.nd.array(value, device=tvm_dev)

    def tir_mlp_static():
        target = tvm.target.Target("cuda")
        exec_queue = generate_exec_queue_static()
        exec_queue_tvm = tvm.nd.array(exec_queue, tvm_dev)
        with target:
            def func():
                tvm_arg_dict["etensor_gate_up_proj"] = tvm.nd.array(arg_dict["etensor_gate_up_proj"], device=tvm_dev)
                tvm_arg_dict["etensor_down_proj"] = tvm.nd.array(arg_dict["etensor_down_proj"], device=tvm_dev)
                tvm_arg_dict["etensor_down_proj_reduce"] = tvm.nd.array(arg_dict["etensor_down_proj_reduce"], device=tvm_dev)
                tvm_arg_dict["etensor_add_rms_norm"] = tvm.nd.array(arg_dict["etensor_add_rms_norm"], device=tvm_dev)
                tvm_arg_dict["residual"] = tvm.nd.array(arg_dict["residual"], device=tvm_dev)
                if PROFILER_ON:
                    tvm_arg_dict["profiler_buffer"] = tvm.nd.array(arg_dict["profiler_buffer"], device=tvm_dev)
                mod_mlp_static(
                    tvm_arg_dict["input"],
                    tvm_arg_dict["W_gate_up"],
                    tvm_arg_dict["out_gate_up_proj"],
                    tvm_arg_dict["out_silu_multiply"],
                    tvm_arg_dict["W_down_proj"],
                    tvm_arg_dict["partial_sum_down_proj"],
                    tvm_arg_dict["out_down_proj"],
                    tvm_arg_dict["residual"],
                    tvm_arg_dict["rmsnorm_weight"],
                    tvm_arg_dict["etensor_gate_up_proj"],
                    tvm_arg_dict["etensor_down_proj"],
                    tvm_arg_dict["etensor_down_proj_reduce"],
                    tvm_arg_dict["etensor_add_rms_norm"],
                    exec_queue_tvm,
                    tvm_arg_dict["profiler_buffer"],
                )
            ms = bench(func, warmup=10, repeat=30, proton_name="tir_static")
            print(f"TIR time: {ms:.3f} ms")
            if PROFILER_ON:
                export_to_perfetto_trace(
                    tvm_arg_dict["profiler_buffer"].numpy(),
                    f"megakernel-mlp-static.perfetto-trace",
                    event_type_names,
                )
        return tvm_arg_dict["out_gate_up_proj"].numpy(), tvm_arg_dict["out_silu_multiply"].numpy(), tvm_arg_dict["out_down_proj"].numpy(), tvm_arg_dict["residual"].numpy()

    def tir_mlp_dynamic():
        target = tvm.target.Target("cuda")
        exec_queue = generate_exec_queue_dynamic()
        with target:
            def func():
                tvm_arg_dict["etensor_gate_up_proj"] = tvm.nd.array(
                    arg_dict["etensor_gate_up_proj"], device=tvm_dev
                )
                tvm_arg_dict["etensor_down_proj"] = tvm.nd.array(
                    arg_dict["etensor_down_proj"], device=tvm_dev
                )
                tvm_arg_dict["etensor_down_proj_reduce"] = tvm.nd.array(
                    arg_dict["etensor_down_proj_reduce"], device=tvm_dev
                )
                tvm_arg_dict["etensor_add_rms_norm"] = tvm.nd.array(
                    arg_dict["etensor_add_rms_norm"], device=tvm_dev
                )
                tvm_arg_dict["etensor_end"] = tvm.nd.array(
                    arg_dict["etensor_end"], device=tvm_dev
                )
                tvm_arg_dict["queue_tasks"] = tvm.nd.array(exec_queue.tasks, tvm_dev)   
                tvm_arg_dict["queue_head"] = tvm.nd.array(exec_queue.head, tvm_dev)
                tvm_arg_dict["queue_tail"] = tvm.nd.array(exec_queue.tail, tvm_dev)
                tvm_arg_dict["residual"] = tvm.nd.array(arg_dict["residual"], device=tvm_dev)
                if PROFILER_ON:
                    tvm_arg_dict["profiler_buffer"] = tvm.nd.array(arg_dict["profiler_buffer"], device=tvm_dev)
                tvm_arg_dict["actual_order"] = tvm.nd.array(arg_dict["actual_order"], device=tvm_dev)
                mod_mlp_dynamic(
                    tvm_arg_dict["input"],
                    tvm_arg_dict["W_gate_up"],
                    tvm_arg_dict["out_gate_up_proj"],
                    tvm_arg_dict["out_silu_multiply"],
                    tvm_arg_dict["W_down_proj"],
                    tvm_arg_dict["partial_sum_down_proj"],
                    tvm_arg_dict["out_down_proj"],
                    tvm_arg_dict["residual"],
                    tvm_arg_dict["rmsnorm_weight"],
                    tvm_arg_dict["etensor_gate_up_proj"],
                    tvm_arg_dict["etensor_down_proj"],
                    tvm_arg_dict["etensor_down_proj_reduce"],
                    tvm_arg_dict["etensor_add_rms_norm"],
                    tvm_arg_dict["etensor_end"],
                    tvm_arg_dict["queue_tasks"],
                    tvm_arg_dict["queue_head"],
                    tvm_arg_dict["queue_tail"],
                    tvm_arg_dict["actual_order"],
                    tvm_arg_dict["profiler_buffer"],
                )
            ms = bench(func, warmup=10, repeat=30, proton_name="tir_dynamic")
            print(f"TIR time: {ms:.3f} ms")
            if PROFILER_ON:
                export_to_perfetto_trace(
                    tvm_arg_dict["profiler_buffer"].numpy(),
                    f"megakernel-mlp-dynamic.perfetto-trace",
                    event_type_names,
                )
        return tvm_arg_dict["out_gate_up_proj"].numpy(), tvm_arg_dict["out_silu_multiply"].numpy(), tvm_arg_dict["out_down_proj"].numpy(), tvm_arg_dict["residual"].numpy()

    def std_mlp():
        import torch
        torch_dev = torch.device("cuda")
        std_arg_dict = {}
        for key, value in arg_dict.items():
            std_arg_dict[key] = value.to(torch_dev)
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
            out_gate_up_proj = torch.matmul(std_arg_dict["input"], std_arg_dict["W_gate_up"].T)
            out_gate_up_proj_tvm = tvm.nd.array(out_gate_up_proj.cpu(), device=tvm.cuda(0))
            out_silu_multiply_tvm = tvm.nd.array(torch.zeros((batch_size, INTERMEDIATE_SIZE), dtype=torch.float16), device=tvm.cuda(0))
            mod_fused_split_silu_multiply(out_gate_up_proj_tvm, out_silu_multiply_tvm)
            out_silu_multiply = torch.from_numpy(out_silu_multiply_tvm.numpy()).to(torch_dev)
            out_down_proj = torch.matmul(out_silu_multiply, std_arg_dict["W_down_proj"].T)
            residual_clone = std_arg_dict["residual"].clone()
            flashinfer.norm.fused_add_rmsnorm(
                out_down_proj, residual_clone, std_arg_dict["rmsnorm_weight"], AddRMSNormTile.EPS, enable_pdl=False
            )
            return out_gate_up_proj, out_silu_multiply, out_down_proj, residual_clone
        out_gate_up_proj, out_silu_multiply, out_down_proj, residual = func()
        ms = bench(func, warmup=10, repeat=30, proton_name="std")
        print(f"CUBLAS time: {ms:.3f} ms")
        return out_gate_up_proj.cpu().numpy(), out_silu_multiply.cpu().numpy(), out_down_proj.cpu().numpy(), residual.cpu().numpy()

    with ProtonContext("blackwell_mlp"):
        out_gate_up_proj_tir_static, out_silu_multiply_tir_static, out_add_rmsnorm_tir_static, residual_tir_static = tir_mlp_static()
        print("static pass", flush=True)
        out_gate_up_proj_tir_dynamic, out_silu_multiply_tir_dynamic, out_add_rmsnorm_tir_dynamic, residual_tir_dynamic = tir_mlp_dynamic()
        print("dynamic pass", flush=True)
        out_gate_up_proj_std, out_silu_multiply_std, out_add_rmsnorm_std, residual_std = std_mlp()

    np.testing.assert_allclose(out_gate_up_proj_tir_static, out_gate_up_proj_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(out_silu_multiply_tir_static, out_silu_multiply_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(out_add_rmsnorm_tir_static, out_add_rmsnorm_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_tir_static, residual_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(out_gate_up_proj_tir_dynamic, out_gate_up_proj_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(out_silu_multiply_tir_dynamic, out_silu_multiply_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(out_add_rmsnorm_tir_dynamic, out_add_rmsnorm_std, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(residual_tir_dynamic, residual_std, rtol=1e-3, atol=1e-2)

if __name__ == "__main__":
    mega_kernel_static = MegaKernel().get_static_scheduler_func()
    mega_kernel_dynamic = MegaKernel().get_dynamic_scheduler_func()
    src_mlp_static, mod_mlp_static = get_source(mega_kernel_static)
    src_mlp_dynamic, mod_mlp_dynamic = get_source(mega_kernel_dynamic)
    
    for batch_size in [128, 64, 32, 16, 8, 4, 2, 1]:
        print("testing batch size: ", batch_size, flush=True)
        test(batch_size, mod_mlp_static, mod_mlp_dynamic)
