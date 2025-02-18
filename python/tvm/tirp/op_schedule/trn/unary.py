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

from typing import Optional, Union
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, FloatImm
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
    MapOpType.MEMSET: (T.nki_memset, None),
}

const_input_ops = [MapOpType.MEMSET]

def unary_trn(
    dst_buffer_region: BufferRegion,
    _src: Union[BufferRegion, FloatImm],
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary operation on Trainium."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None
    CONST = None
    if isinstance(_src, FloatImm):
        assert unary_op in const_input_ops, f"Unsupported unary operation {unary_op} taking const as input"
        CONST = _src
    else:
        src_buffer_region = _src    

    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    assert unary_op in unary_map_ops, f"Unsupported unary operation {unary_op}"
    dst = dst_buffer_region.buffer
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if CONST is None:
        src = src_buffer_region.buffer
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
        # Extract regions and validate dimensions
        src_region = src_buffer_region.region
        src_extent = [r.extent for r in src_region]

        # Validate non-unit dimensions match
        src_extent_ = [e for e in src_extent if e != 1]
        if not (
            len(src_extent_) == len(dst_extent_)
            and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
        ):
            return None
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            dst_buffer_region, src_buffer_region, analyzer
        )
    else:
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            dst_buffer_region, dst_buffer_region, analyzer
        )
    p_size = dst_buffer_region.buffer.layout.partition_size
    f_gen_axes = generate_axes_in_region(dst_buffer_region, inst_stride, inst_data_iters, analyzer)

    def f_gen_src_idx(b_loop, b_extent, f_loop, p_loop):
        dim_map = get_ewise_dim_map(dst_buffer_region, src_buffer_region, analyzer)
        indices = [src_buffer_region.region[i].min for i in range(len(src_buffer_region.region))]
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        for i, j in dim_map.items():
            indices[j] += axes[i]
        return indices

    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [dst_buffer_region.region[i].min + axes[i] for i in range(len(axes))]

    b_extent = reduce(operator.mul, dst_extent, 1) // p_size // inst_size

    func, opcode = unary_map_ops[unary_op]
    # fmt: off
    
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, inst_size):
                        dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop))
                        if CONST is not None:
                            if unary_op == MapOpType.MEMSET:
                                T.evaluate(func(dst[*dst_indices], CONST))
                        else:
                            src_indices = T.meta_var(f_gen_src_idx(b_loop, b_extent, f_loop, p_loop) if CONST is None else [])
                            if opcode is None:
                                T.evaluate(func(dst[*dst_indices], src[*src_indices]))
                            else:
                                #todo: if we use direct allocation, nki activation should take zero bias tensor
                                T.evaluate(func(dst[*dst_indices], src[*src_indices], opcode))
    # fmt: on

    return impl
