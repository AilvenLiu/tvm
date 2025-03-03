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

"""Implementation of reduction schedules."""


from typing import Optional, Tuple
import operator
from functools import reduce

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T, tirp as Tp
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext

from .common import find_max_inst_size_from_one_region, generate_axes_in_region, init_analyzer, get_reduction_dim_map, f_gen_idx_anchor, f_gen_idx_mapped
from ..registry import register_schedule
from ..common import _make_schedule, ReduceOpType


reduce_ops = {
    ReduceOpType.SUM: "add",
    ReduceOpType.MAX: "max",
    ReduceOpType.MIN: "min",
}

def generate_intermediate_buffer(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    axes: Tuple[int],
    inst_size,
    analyzer: Analyzer
):
    reduction_size = reduce(operator.mul, [src_buffer_region.region[i].extent for i in axes], 1)
    if analyzer.can_prove(reduction_size == inst_size):
        # No need to split into 2 stages
        return None, None, None
    assert analyzer.can_prove(reduction_size % inst_size == 0), f"Reduction size {reduction_size} must be divisible by instruction size {inst_size}"
    rfactor_size = reduction_size // inst_size
    dst_layout = dst_buffer_region.buffer.layout
    intermediate_shape = [dst_layout.partition_size, rfactor_size]
    intermediate_dimension_types = [T.TrainiumLayout.Partition, T.TrainiumLayout.Free]
    intermediate_layout = T.TrainiumLayout(intermediate_dimension_types, T.TileLayout.from_tuple((dst_layout.partition_size, rfactor_size), (1, 1)))
    return intermediate_shape, intermediate_layout, rfactor_size


def reduction_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    axes: Tuple[int],
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None
    assert not accum, "Accumulation is not supported for reduction on Trainium"
    analyzer = init_analyzer(sctx)
    assert reduce_op in reduce_ops, f"Unsupported reduce operation {reduce_op}"

    dst = dst_buffer_region.buffer
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    src = src_buffer_region.buffer
    axes = [i if i >= 0 else len(src.shape) + i for i in axes]
    dim_map = get_reduction_dim_map(src_buffer_region, dst_buffer_region, axes)
    # layout checks
    assert all(
        [
            src.layout and dst.layout,
            src.scope() == "trn.sbuf" or src.scope() == "trn.psum",
            dst.scope() == "trn.sbuf",
            isinstance(src.layout, T.TrainiumLayout),
            isinstance(dst.layout, T.TrainiumLayout),
            src.layout.partition_size == dst.layout.partition_size,
        ]
    ), "Invalid layout"

    # reduction axes must be f dim
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_from_one_region(
        src_buffer_region, analyzer, axes
    )
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"
    p_size = dst.layout.partition_size
    b_extent = reduce(operator.mul, dst_extent, 1) // p_size
    opcode = reduce_ops[reduce_op]
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(dst_buffer_region, src_buffer_region, axes, inst_size, analyzer)
    f_gen_axes = generate_axes_in_region(src_buffer_region, inst_stride, inst_data_iters, analyzer)
    f_gen_src_idx = f_gen_idx_anchor(src_buffer_region, f_gen_axes)
    f_gen_dst_idx = f_gen_idx_mapped(dst_buffer_region, f_gen_axes, dim_map)
    if intermediate_shape is None:
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop, f_loop in T.grid(p_size, inst_size):
                        src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent),), f_loop, p_loop))
                        dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        T.evaluate(T.nki_tensorreduce(dst[dst_indices], src[src_indices], opcode, -1))
        # fmt: on
        return impl
    else:
        dim2block_var = {dim: 1 if dim in axes else 0 for dim in range(len(src.shape))}
        # fmt: off
        @T.prim_func(tirp=True)
        def two_stage_reduction():
            with T.kernel():
                intermediate_buffer = T.alloc_buffer(intermediate_shape, dtype=dst.dtype, layout=intermediate_layout, scope="trn.sbuf")
                for b_loop in T.serial(0, b_extent):
                    for reduction_b_loop in T.serial(0, reduction_b_extent):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, f_loop in T.grid(p_size, inst_size):
                                src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                T.evaluate(T.nki_tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], src[src_indices], opcode, -1))
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                            T.evaluate(T.nki_tensorreduce(dst[dst_indices], intermediate_buffer[p_loop, f_loop], opcode, -1))
        # fmt: on
        return two_stage_reduction


# Register unary mapping schedules.
for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:
    custom_name = f"reduction_{op_name}_trn_impl"
    func = _make_schedule(op_type, 3, [reduction_trn])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule reduction {op_name}."
    register_schedule(op_name)(func)
