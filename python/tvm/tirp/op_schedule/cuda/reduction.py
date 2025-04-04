# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Implementation of reduce operator schedules."""

from typing import Any, Dict, Optional
import functools
import operator

from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext
from tvm.arith.analyzer import Analyzer

from ..common import ReduceOpType, register_unary_binary_schedule
from .common import target_cuda


reduce_op_table = {
    ReduceOpType.SUM: lambda a, b: a + b,
    ReduceOpType.MAX: T.max,
    ReduceOpType.MIN: T.min,
}

reduce_default_value_table = lambda dtype: {
    ReduceOpType.SUM: 0.0,
    ReduceOpType.MAX: T.min_value(dtype),
    ReduceOpType.MIN: T.max_value(dtype),
}


def reduction_cuda_shared_nd_sync_cta_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.

    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """

    # Basic validation checks
    if sctx.exec_scope.name != "cta":
        return None

    thread_cnt = sctx.launch_params["threadIdx.x"]
    threads_per_warp = 32
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if not (thread_cnt >= threads_per_warp and thread_cnt % threads_per_warp == 0):
        return None

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    dtype = src.dtype

    # Check dst is first m dimensions of src (and the rest dimensions, if exist, are all 1s)
    if not all(
        [
            src.scope().startswith("shared"),
            dst.scope().startswith("shared"),
            len(src_region) >= len(dst_region),
        ]
    ):
        return None

    analyzer = Analyzer()
    spatial_dims = -1

    if len(src_extent) == len(dst_extent):
        for i in range(len(dst_extent)):
            if src_extent[i] != dst_extent[i]:
                if dst_extent[i] != 1:
                    return None
                if not functools.reduce(operator.mul, dst_extent[i:], 1) == 1:
                    return None
                else:
                    spatial_dims = i
                    break
        if spatial_dims == -1:
            return None

    else:
        spatial_dims = len(dst_extent)
        if not all(
            analyzer.can_prove_equal(s, d) for s, d in zip(src_extent[:spatial_dims], dst_extent)
        ):
            return None

    assert spatial_dims > 0 and spatial_dims < len(src_extent)

    spatial_len = functools.reduce(operator.mul, src_extent[:spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, src_extent[spatial_dims:], 1)

    # get reduce op
    op_func = reduce_op_table.get(reduce_op)
    if op_func is None:
        return None

    # get init value if not accum
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    def get_indices(nth, st, extent):
        # example: st: [0, 1], extent: [32, 16], nth: 10
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        warp_cnt = T.meta_var(T.ceildiv(thread_cnt, threads_per_warp))
        for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
            thread_buffer = T.allocate([1], dtype=dtype, scope="local")
            thread_data = T.Buffer(1, data=thread_buffer, dtype=dtype, scope="local")
            for step in T.serial(T.ceildiv(spatial_len, warp_cnt)):
                # reduction on dst_indices
                spa_fused = T.meta_var(step * warp_cnt + T.floordiv(tid_x, threads_per_warp))
                if spa_fused < spatial_len:
                    src_indices_1 = T.meta_var(get_indices(spa_fused, src_st[:spatial_dims], src_extent[:spatial_dims]))
                    thread_data[0] = init_value
                    # load from src
                    for t in T.serial(T.ceildiv(reduction_len, threads_per_warp)):
                        red_fused = T.meta_var(t * threads_per_warp + tid_x % threads_per_warp)
                        if red_fused < reduction_len:
                            src_indices_2 = T.meta_var(get_indices(red_fused, src_st[spatial_dims:], src_extent[spatial_dims:]))
                            thread_data[0] = op_func(thread_data[0], src[*(src_indices_1 + src_indices_2)])
                    # warp reduce
                    mask = T.tvm_warp_activemask()
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 1, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 2, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 4, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 8, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 16, 32, 32))

                    # write result to dst_indices
                    if tid_x % threads_per_warp == 0:
                        dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                        dst[*dst_indices] = T.if_then_else(T.bool(accum), op_func(dst[*dst_indices], thread_data[0]), thread_data[0])

        T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def reduction_cuda_warp_logical_view_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    schedule_config: Dict[str, Any],
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule reduction operation on logical tensor of local memory on CUDA.

    If 'thread_reduce' is set to True in schedule_config, perform a reduction
    across threads based on the src buffer layout.

    Currently, only WGMMA layout is supported for this feature.

    Note: for now, is_tile_inner only support warp-level buffer view verification.
    Warpgroup-level buffer view and cta-level buffer view need sharding on the
    outermost level, thus not supported to be checked and verified at the moment.
    User should pass in warp-level buffer view for src and dst whenever possible.
    """

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    # basic validation checks
    if not all(
        [
            src.scope() == "local",
            dst.scope() == "local",
            src.layout is not None,
            dst.layout is not None,
            src.logical_scope() == "warp",
            dst.logical_scope() == "warp",
            src.dtype == dst.dtype,
            sctx.is_cuda(),
            sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"],
        ]
    ):
        return None

    # no slicing allowed
    if not all(
        [
            len(src_region) == 2 and len(dst_region) == 2,
            src_region[0].min == 0 and src_region[1].min == 0,
            dst_region[0].min == 0 and dst_region[1].min == 0,
            src_region[0].extent == src.shape[0] and src_region[1].extent == src.shape[1],
            dst_region[0].extent == dst.shape[0] and dst_region[1].extent == dst.shape[1],
        ]
    ):
        return None

    # check for WGMMA layout
    if any(
        [
            len(src.shape) != 2,
            len(dst.shape) != 2,
            src.shape[0] != 16,
            not (src.shape[1] % 8 == 0 or src.shape[1] == 4),
            dst.shape[0] != 16,
            dst.shape[1] != 4,
        ]
    ):
        return None

    if src.layout.is_swizzle() or dst.layout.is_swizzle():
        return None

    atom = T.TileLayout.from_tuple((1, 2), (2, 1))
    warp_atom = T.TileLayout.shard((8, 8), (8, 4), "S0S1", inner=atom, from_to=("thread", "warp"))
    red_atom = T.TileLayout.from_tuple(1, 1)
    red_warp_atom = T.TileLayout.shard(
        (32,), (32,), "S0", inner=red_atom, from_to=("thread", "warp")
    )

    shuffle = T.bool(schedule_config.get("thread_reduce", False))

    # get reduce op
    op_func = reduce_op_table.get(reduce_op)
    if op_func is None:
        return None

    # get init value if not accum
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    # case 0. shuffle-only, no within-thread reduction
    # provided that src and dst are same-shaped buffer
    if src.shape[1] == 4:
        if red_warp_atom.is_tile_inner(src.layout.normalize(), (64,), (32,)) is None:
            return None
        if red_warp_atom.is_tile_inner(dst.layout.normalize(), (64,), (32,)) is None:
            return None
        if shuffle is False:
            return None

        num_rows = 2
        local_shape = (num_rows,)
        is_same_buffer = src.same_as(dst)

        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl_shuffle_only():
            with T.thread():
                src_local = T.get(src, local_shape)
                dst_local = T.get(dst, local_shape)
                for i in T.serial(num_rows):
                    if not is_same_buffer:
                        dst_local[i] = src_local[i]
                    row_var = T.meta_var(dst_local[i])
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 2, 32, 32))
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 1, 32, 32))
        # fmt: on

        return impl_shuffle_only

    # case 1. normal reduction
    # src and dst are different-shaped buffer
    src_tile_outer = warp_atom.is_tile_inner(src.layout, src.shape, [8, 8])
    if src_tile_outer is None:
        return None
    if red_warp_atom.is_tile_inner(dst.layout.normalize(), (64,), (32,)) is None:
        return None

    num_rows = 2
    src_local_shape = (num_rows, src_tile_outer.size)
    dst_local_shape = (num_rows,)

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        with T.thread():
            src_local = T.get(src, src_local_shape)
            dst_local = T.get(dst, dst_local_shape)
            for i in T.serial(num_rows):
                # reduce within threads
                if not accum:
                    dst_local[i] = init_value
                row_var = T.meta_var(dst_local[i])
                for j in T.serial(src_local_shape[1]):
                    dst_local[i] = op_func(row_var, src_local[i, j])
                # if shuffle is True, perform shuffling among threads of 4
                if shuffle:
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 2, 32, 32))
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 1, 32, 32))
    # fmt: on

    return impl


def reduction_cuda_impl(
    op: OpCall,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Dispatch to shared memory scheduler or logical tensor of local memory scheduler
    based on the storage scope of buffers.
    """

    # FIXME: correctly handle the axes field
    dst_buffer_region, src_buffer_region, axes, accum = op.args
    schedule_config = op.schedule_config

    if src_buffer_region.buffer.scope().startswith("shared"):
        return reduction_cuda_shared_nd_sync_cta_impl(
            dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
        )
    elif (
        src_buffer_region.buffer.scope() == "local"
        and src_buffer_region.buffer.logical_scope() == "warp"
    ):
        return reduction_cuda_warp_logical_view_impl(
            dst_buffer_region, src_buffer_region, accum, reduce_op, schedule_config, sctx
        )
    return None


for op_name_, op_type_ in {
    "sum": ReduceOpType.SUM,
    "max": ReduceOpType.MAX,
    "min": ReduceOpType.MIN,
}.items():
    register_unary_binary_schedule(
        op_name_,
        op_type_,
        "cuda",
        target_cuda,
        [reduction_cuda_impl],
    )
