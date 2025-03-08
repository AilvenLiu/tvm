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

from typing import Optional
import functools
import operator

from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext
from tvm.arith.analyzer import Analyzer

from ..common import ReduceOpType, register_unary_binary_schedule
from .common import target_cuda


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

    if reduce_op not in [ReduceOpType.SUM]:
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
                    thread_data[0] = 0.0
                    # load from src
                    for t in T.serial(T.ceildiv(reduction_len, threads_per_warp)):
                        red_fused = T.meta_var(t * threads_per_warp + tid_x % threads_per_warp)
                        if red_fused < reduction_len:
                            src_indices_2 = T.meta_var(get_indices(red_fused, src_st[spatial_dims:], src_extent[spatial_dims:]))
                            thread_data[0] += src[*(src_indices_1 + src_indices_2)]
                    # warp reduce
                    mask = T.tvm_warp_activemask()
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 1, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 2, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 4, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 8, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 16, 32, 32)

                    # write result to dst_indices
                    if tid_x % threads_per_warp == 0:
                        dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                        dst[*dst_indices] = T.if_then_else(T.bool(accum), dst[*dst_indices] + thread_data[0], thread_data[0])

        T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def reduction_cuda_shared_nd_sync_cta(
    op: OpCall,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.

    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """

    # FIXME: correctly handle the axes field
    dst_buffer_region, src_buffer_region, axes, accum = op.args
    return reduction_cuda_shared_nd_sync_cta_impl(
        dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
    )


register_unary_binary_schedule(
    "sum",
    ReduceOpType.SUM,
    "cuda",
    target_cuda,
    [reduction_cuda_shared_nd_sync_cta],
)
