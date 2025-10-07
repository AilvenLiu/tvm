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

"""Implementation of unary operator schedules."""

import functools
import operator
from typing import Optional, Union

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, OpCall, PrimFunc
from tvm.tir.expr import FloatImm
from tvm.tirp.op_schedule import ScheduleContext, fail

from ..common import MapOpType
from .common import get_indices

unary_op_table = {
    MapOpType.ZERO: lambda x: 0.0,
    MapOpType.FILL: lambda x: x,
    MapOpType.SQRT: lambda x: T.sqrt(x),
    MapOpType.RECIPROCAL: lambda x: 1.0 / x,
    MapOpType.EXP: T.exp2,
}


def unary_map_cuda_shared_nd_sync_cta_impl(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """
    Schedule unary map operation on CUDA in shared memory.
    The destination and source regions must be of the same shape.
    """
    _dst: BufferRegion = op.args[0]
    _src: BufferRegion = op.args[1]

    # Check CUDA and CTA context, and supported op types.
    if sctx.exec_scope.name != "cta":
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    dst, src = _dst.buffer, _src.buffer
    dst_region, src_region = _dst.region, _src.region
    dtype = dst.dtype

    dst_start = [r.min for r in dst_region]
    src_start = [r.min for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    src_extent = [r.extent for r in src_region]

    if not (
        dst.layout
        and src.layout
        and dst.layout.is_trivial()
        and src.layout.is_trivial()
        and dst.scope().startswith("shared")
        and src.scope().startswith("shared")
        and src.dtype == dtype
    ):
        fail("unsupported layout/scope/dtype for shared-memory unary map")

    analyzer = Analyzer()
    num_elements = functools.reduce(operator.mul, src_extent, 1)
    if not (
        len(src_extent) == len(dst_extent)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent, dst_extent))
    ):
        fail("shape mismatch between src and dst for unary map")

    thread_cnt = sctx.launch_params["threadIdx.x"].dom.extent
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    # Define operation lambda.
    op_func = unary_op_table.get(unary_op)
    if op_func is None:
        fail(f"unsupported unary op: {unary_op}")

    @T.prim_func(tirp=True)
    def impl():
        for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
            for itr in T.serial(T.ceildiv(num_elements, thread_cnt)):
                fused_idx = T.meta_var(itr * thread_cnt + tid_x)
                if fused_idx < num_elements:
                    idx_dst = T.meta_var(get_indices(fused_idx, dst_start, dst_extent))
                    idx_src = T.meta_var(get_indices(fused_idx, src_start, src_extent))
                    dst[*idx_dst] = T.Cast(dtype, op_func(src[*idx_src]))
        T.tvm_storage_sync("shared")

    return impl


def unary_map_cuda_shared_nd_sync_cta_impl_with_bias_scale(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary map operation on CUDA in shared memory with bias and scale."""

    _bias: Optional[Union[BufferRegion, FloatImm]] = op.args[2]
    _scale: Optional[FloatImm] = op.args[3]

    if _bias is not None or _scale is not None:
        fail("bias/scale not supported for shared-memory unary map")
    return unary_map_cuda_shared_nd_sync_cta_impl(op, unary_op, sctx)


def unary_map_cuda_warp_logical_view_nd_impl(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """
    Schedule unary map operation on CUDA on warp-level logical tensor.
    The destination and source regions must be of the same layout
    to ensure correctness.
    """

    _dst: BufferRegion = op.args[0]
    _src: BufferRegion = op.args[1]

    dst, src = _dst.buffer, _src.buffer
    src_region, dst_region = _src.region, _dst.region
    dtype = src.dtype

    dst_start = [r.min for r in dst_region]
    src_start = [r.min for r in src_region]
    ref_start = [0] * len(src_start)
    dst_extent = [r.extent for r in dst_region]
    src_extent = [r.extent for r in src_region]

    # basic validation checks
    if not all(
        [
            src.scope() == "local",
            dst.scope() == "local",
            src.layout is not None,
            dst.layout is not None,
            src.dtype == dst.dtype,
            sctx.is_cuda(),
            sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"],
        ]
    ):
        fail("unsupported layout/scope or exec_scope for local tensor unary map")

    # get unary op
    op_func = unary_op_table.get(unary_op)
    if op_func is None:
        fail(f"unsupported unary op: {unary_op}")

    # no slicing allowed, since op is on local tensor
    analyzer = Analyzer()
    if not all(
        [
            len(src.shape) == len(dst.shape),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src.shape, dst.shape)),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent, src.shape)),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(dst_extent, dst.shape)),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src_start, ref_start)),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(dst_start, ref_start)),
        ]
    ):
        fail("slicing not supported for local tensor unary map; expect full-shape buffers")

    # layout check
    if any(
        [
            src.layout != dst.layout,
            src.layout.size() != dst.layout.size(),
            src.layout.cosize() != dst.layout.cosize(),
            src.layout.is_swizzle(),
            dst.layout.is_swizzle(),
        ]
    ):
        fail("layout mismatch or swizzle not supported for local tensor unary map")

    LOCAL_LEN = src.layout.size()
    src_local_shape = dst_local_shape = (LOCAL_LEN,)

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        with T.thread():
            src_local = src.storage(*src_local_shape)
            dst_local = dst.storage(*dst_local_shape)
            for idx in T.serial(LOCAL_LEN):
                dst_local[idx] = T.Cast(dtype, op_func(src_local[idx]))
    # fmt: on

    return impl


def unary_map_cuda_warp_logical_view_nd_impl_with_bias_scale(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary map operation on CUDA on warp-level logical tensor with bias and scale."""

    _bias: Optional[Union[BufferRegion, FloatImm]] = op.args[2]
    _scale: Optional[FloatImm] = op.args[3]

    if _bias is not None or _scale is not None:
        fail("bias/scale not supported for local tensor unary map")
    return unary_map_cuda_warp_logical_view_nd_impl(op, unary_op, sctx)


def unary_cuda_impl(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Dispatch to shared memory scheduler or logical tensor of local memory scheduler
    based on the storage scope of buffers.
    """

    dst_buffer_region = op.args[0]
    if dst_buffer_region.buffer.scope().startswith("shared"):
        if unary_op in {MapOpType.SQRT, MapOpType.EXP}:
            return unary_map_cuda_shared_nd_sync_cta_impl_with_bias_scale(op, unary_op, sctx)
        return unary_map_cuda_shared_nd_sync_cta_impl(op, unary_op, sctx)
    elif dst_buffer_region.buffer.scope() == "local":
        if unary_op in {MapOpType.SQRT, MapOpType.EXP}:
            return unary_map_cuda_warp_logical_view_nd_impl_with_bias_scale(op, unary_op, sctx)
        return unary_map_cuda_warp_logical_view_nd_impl(op, unary_op, sctx)

    fail("unsupported buffer scope for unary op")
