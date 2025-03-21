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
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir.stmt import OpCall
from .common import (
    find_max_inst_size_from_one_region,
    generate_axes_in_region,
    init_analyzer,
    get_reduction_dim_map,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    bound_buffer_region,
    make_guard,
    nki_dim,
    bound_inst_data_iter_with_limit,
    check_workspace_buffer,
)
from ..common import register_unary_binary_schedule, ReduceOpType
from .common import target_trn

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
    workspace,
    sctx: ScheduleContext,
    analyzer: Analyzer,
):
    """Generate an intermediate buffer for two-stage reduction if needed.

    Returns:
        Tuple[Optional[buffer], int]: The intermediate buffer and reduction factor size.
    """
    reduction_size = reduce(operator.mul, [src_buffer_region.region[i].extent for i in axes], 1)
    # No need to split into 2 stages
    if analyzer.can_prove(reduction_size <= inst_size):
        return None, 1

    assert analyzer.can_prove(
        reduction_size % inst_size == 0
    ), f"Reduction size {reduction_size} must be divisible by instruction size {inst_size}"

    rfactor_size = reduction_size // inst_size
    dst_layout = dst_buffer_region.buffer.layout
    intermediate_shape = [dst_layout.partition_size, rfactor_size]

    if "partial_reduce" in workspace:
        intermediate_buffer = workspace["partial_reduce"]
        check_workspace_buffer(intermediate_buffer, intermediate_shape, "trn.sbuf")
    else:
        intermediate_buffer = T.buffer(
            intermediate_shape,
            dtype=dst_buffer_region.buffer.dtype,
            scope="trn.sbuf",
            buffer_name="partial_reduce",
        )
        sctx.add_alloc_buffer(intermediate_buffer)

    return intermediate_buffer, rfactor_size


def reduction_trn(
    op: OpCall,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
    negate: bool = False,
) -> Optional[PrimFunc]:
    """Schedule reduction operation on Trainium.

    Args:
        op: The operation call.
        reduce_op: The reduction operation type.
        sctx: The schedule context.
        negate: Whether to negate the result.

    Returns:
        Optional[PrimFunc]: The scheduled function, or None if not applicable.
    """
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    dst_buffer_region, src_buffer_region, axes, accum = op.args[:4]
    assert not accum, "Accumulation is not supported for reduction on Trainium"
    analyzer = init_analyzer(sctx)
    assert reduce_op in reduce_ops, f"Unsupported reduce operation {reduce_op}"

    # Extract buffers
    dst = dst_buffer_region.buffer
    src = src_buffer_region.buffer
    axes = [i if i >= 0 else len(src.shape) + i for i in axes]
    dim_map = get_reduction_dim_map(src_buffer_region, dst_buffer_region, axes, analyzer)

    # Layout validation
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

    # Get bound buffer regions
    bound_src = bound_buffer_region(src_buffer_region, analyzer)
    bound_dst = bound_buffer_region(dst_buffer_region, analyzer)

    # Find maximum instruction size
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_from_one_region(
        bound_src, analyzer, axes
    )
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"

    # Get partition size and extents
    p_size = dst.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // p_size

    # Handle instruction size limit from config
    inst_size_limit = op.schedule_config.get("max_inst_size", None)
    inst_size, inst_data_iters = bound_inst_data_iter_with_limit(
        bound_src, inst_data_iters, inst_size_limit, analyzer
    )

    # Get reduction operation code
    opcode = reduce_ops[reduce_op]

    # Set up index generation functions
    f_gen_axes = generate_axes_in_region(bound_src, inst_stride, inst_data_iters, analyzer)
    f_gen_src_idx = f_gen_idx_anchor(src_buffer_region, f_gen_axes)
    f_gen_dst_idx = f_gen_idx_mapped(dst_buffer_region, f_gen_axes, dim_map)
    f_src_guard = make_guard(src_buffer_region, analyzer)

    # Generate intermediate buffer if needed
    intermediate_buffer, reduction_b_extent = generate_intermediate_buffer(
        bound_dst, bound_src, axes, inst_size, op.workspace, sctx, analyzer
    )

    # fmt: off
    # Define reduction instruction macro
    @T.macro
    def reduction_inst_macro(b_loop, p_loop, f_loop, dst_buffer, src_buffer):
        if f_src_guard(f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)):
            src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent),), f_loop, p_loop))
            dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
            T.evaluate(T.nki_tensorreduce(dst_buffer[dst_indices], src_buffer[src_indices], opcode, negate, -1))

    # Define two-stage reduction macro
    @T.macro
    def two_stage_reduction_macro(b_loop, reduction_b_loop, p_loop, f_loop, intermediate_buffer, dim2block_var):
        if f_src_guard(f_gen_axes(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var)):
            src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
            T.evaluate(T.nki_tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], src[src_indices], opcode, False, -1))

    @T.macro
    def two_stage_final_reduction_macro(b_loop, p_loop, f_loop, intermediate_buffer, dim2block_var):
        if f_src_guard(f_gen_axes(((b_loop, b_extent), (f_loop, reduction_b_extent)), 0, p_loop, dim2block_var)):
            dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
            T.evaluate(T.nki_tensorreduce(dst[dst_indices], intermediate_buffer[p_loop, f_loop], opcode, negate, -1))    
    # Single-stage reduction implementation
    if intermediate_buffer is None:
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, inst_size, annotations={nki_dim: "F"}):
                            reduction_inst_macro(b_loop, p_loop, f_loop, dst, src)
        return impl
    # Two-stage reduction implementation
    else:
        dim2block_var = {dim: 1 if dim in axes else 0 for dim in range(len(src.shape))}
        @T.prim_func(tirp=True)
        def two_stage_reduction():
            for b_loop in T.serial(0, b_extent):
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in T.serial(0, inst_size, annotations={nki_dim: "F"}):
                                two_stage_reduction_macro(b_loop, reduction_b_loop, p_loop, f_loop, intermediate_buffer, dim2block_var)
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            two_stage_final_reduction_macro(b_loop, p_loop, f_loop, intermediate_buffer, dim2block_var)
        return two_stage_reduction
    # fmt: on


# Register schedules for supported reduction operations
for op_name_, op_type_ in {
    "sum": ReduceOpType.SUM,
    "max": ReduceOpType.MAX,
    "min": ReduceOpType.MIN,
}.items():
    register_unary_binary_schedule(op_name_, op_type_, "trn", target_trn, [reduction_trn])
