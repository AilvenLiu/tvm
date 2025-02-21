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

"""Implementation of copy operator schedules."""

from typing import Optional
import operator
from functools import reduce

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
)


@register_schedule("copy")
def copy_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    src, dst = src_buffer_region.buffer, dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.scope() in ["global", "trn.sbuf", "trn.psum"],
            dst.scope() in ["global", "trn.sbuf", "trn.psum"],
            src.scope() != "global" or dst.scope() != "global",
            (src.scope() == "global" and isinstance(src.layout, T.TileLayout))
            or (
                src.scope() in ["trn.sbuf", "trn.psum"] and isinstance(src.layout, T.TrainiumLayout)
            ),
            (dst.scope() == "global" and isinstance(dst.layout, T.TileLayout))
            or (
                dst.scope() in ["trn.sbuf", "trn.psum"] and isinstance(dst.layout, T.TrainiumLayout)
            ),
        ]
    ):
        raise ValueError("Invalid buffer layout/scope for copy operation.")

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    inst_size = None
    inst_stride = None
    inst_data_iters = None
    src_to_dst = None
    if isinstance(src.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            src_buffer_region, dst_buffer_region, analyzer
        )
        src_to_dst = True
    if inst_size is None and isinstance(dst.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            dst_buffer_region, src_buffer_region, analyzer
        )
        src_to_dst = False
    assert src_to_dst is not None
    if src_to_dst:
        p_size = src_buffer_region.buffer.layout.partition_size
        f_gen_axes = generate_axes_in_region(
            src_buffer_region, inst_stride, inst_data_iters, analyzer
        )
        dim_map = get_ewise_dim_map(src_buffer_region, dst_buffer_region, analyzer)
    else:
        p_size = dst_buffer_region.buffer.layout.partition_size
        f_gen_axes = generate_axes_in_region(
            dst_buffer_region, inst_stride, inst_data_iters, analyzer
        )
        dim_map = get_ewise_dim_map(dst_buffer_region, src_buffer_region, analyzer)

    def f_gen_src_idx(b_loop, b_extent, f_loop, p_loop):
        indices = [src_buffer_region.region[i].min for i in range(len(src_buffer_region.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        if not src_to_dst:
            for i, j in dim_map.items():
                indices[j] += axes[i]
        else:
            assert len(indices) == len(axes)
            for i, axis in enumerate(axes):
                indices[i] += axis
        return indices

    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        indices = [dst_buffer_region.region[i].min for i in range(len(dst_buffer_region.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        if src_to_dst:
            for i, j in dim_map.items():
                indices[j] += axes[i]
        else:
            assert len(indices) == len(axes)
            for i, axis in enumerate(axes):
                indices[i] += axis
        return indices

    b_extent = reduce(operator.mul, src_extent, 1) // p_size // inst_size
    # fmt: off
    if src.scope() == "global":
        func = T.nki_load
    elif dst.scope() == "global":
        func = T.nki_store
    else:
        func = T.nki_tensor_copy
    
    inst_size_limit = get_hardware_inst_size_limit(func!=T.nki_tensor_copy)
    actual_inst_size, additional_b_size = bound_inst_with_limit(inst_size, inst_size_limit, analyzer)
    
    @T.prim_func(tirp=True)
    def impl():
        # the additional b loop is to satisfy hardware instuction size limit
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, actual_inst_size):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        src_indices = T.meta_var(f_gen_src_idx(b_loop, b_extent, f_loop_wo_limit, p_loop))
                        dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop_wo_limit, p_loop))
                        func(dst[*dst_indices], src[*src_indices])
    # fmt: on

    return impl
