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

from typing import Optional
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext

from functools import reduce
from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
)
from ..common import MapOpType

unary_map_ops = {
    MapOpType.SQRT: (T.nki_activation, "sqrt"),
    MapOpType.RECIPROCAL: (T.nki_reciprocal, None),
}


def unary_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary operation on Trainium."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    src, dst = src_buffer_region.buffer, dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            src.scope() == "trn.sbuf" or src.scope() == "trn.psum",
            dst.scope() == "trn.sbuf",
            isinstance(src.layout, T.TrainiumLayout),
            isinstance(dst.layout, T.TrainiumLayout),
        ]
    ):
        return None
    assert unary_op in unary_map_ops, f"Unsupported unary operation {unary_op}"
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
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
        src_buffer_region, dst_buffer_region, analyzer
    )
    p_size = src_buffer_region.buffer.layout.partition_size
    f_gen_axes = generate_axes_in_region(src_buffer_region, inst_stride, inst_data_iters, analyzer)

    def f_gen_src_idx(b_loop, b_extent, f_loop, p_loop):
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [src_buffer_region.region[i].min + axes[i] for i in range(len(axes))]

    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        dim_map = get_ewise_dim_map(src_buffer_region, dst_buffer_region, analyzer)
        indices = [dst_buffer_region.region[i].min for i in range(len(dst_buffer_region.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        for i, j in dim_map.items():
            indices[j] += axes[i]
        return indices

    b_extent = reduce(operator.mul, src_extent, 1) // p_size // inst_size

    func, opcode = unary_map_ops[unary_op]

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, inst_size):
                        src_indices = T.meta_var(f_gen_src_idx(b_loop, b_extent, f_loop, p_loop))
                        dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop))
                        if opcode is None:
                            T.evaluate(func(dst[*dst_indices], src[*src_indices]))
                        else:
                            T.evaluate(func(dst[*dst_indices], src[*src_indices], opcode))
    # fmt: on

    return impl
