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
from tvm.tirp.op_schedule import ScheduleContext, fail
from tvm.tir.stmt import OpCall
from .common import (
    init_analyzer,
    get_reduction_dim_map,
    nki_dim,
    check_workspace_buffer,
    InstructionGenerator,
)
from ..common import register_unary_binary_schedule, ReduceOpType
from .common import target_trn

reduce_ops = {
    ReduceOpType.SUM: "add",
    ReduceOpType.MAX: "max",
    ReduceOpType.MIN: "min",
}


def generate_intermediate_buffer(
    dst_buffer_region: int,
    rfactor_size: int,
    workspace,
    sctx: ScheduleContext,
):
    """Generate an intermediate buffer for two-stage reduction if needed.

    Returns:
        Tuple[Optional[buffer], int]: The intermediate buffer and reduction factor size.
    """
    intermediate_shape = [dst_buffer_region.buffer.layout.size("P"), rfactor_size]

    if "partial_reduce" in workspace:
        intermediate_buffer = workspace["partial_reduce"]
        check_workspace_buffer(intermediate_buffer, intermediate_shape, "trn.sbuf")
    else:
        assert (
            sctx.alloc_only
        ), "Partial reduce buffer must be specified in workspace. Run tvm.tirp.transform.PrivateBufferAlloc first."
        intermediate_buffer = T.buffer(
            intermediate_shape,
            dtype=dst_buffer_region.buffer.dtype,
            scope="trn.sbuf",
            buffer_name="partial_reduce",
        )
        sctx.add_alloc_buffer(intermediate_buffer)

    return intermediate_buffer


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
        fail("requires Trainium target and kernel exec_scope")

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
            src.layout.is_trainium(),
            dst.layout.is_trainium(),
            src.layout.size("P") == dst.layout.size("P"),
        ]
    ), "Invalid layout"

    # Find maximum instruction size
    inst_gen = InstructionGenerator([src_buffer_region, dst_buffer_region], analyzer)
    inst_gen.link_buffer_regions(src_buffer_region, dst_buffer_region, dim_map)
    inst_repr = inst_gen.find_max_inst_size_from_one_region(src_buffer_region, axes)
    inst_size_limit = op.config.get("max_inst_size", None)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)
    assert analyzer.can_prove(inst_repr.size > 1), "Instruction size must be greater than 1"

    # Get partition size and extents
    p_size = src.layout.size("P")
    f_var = T.Var("F", "int32")
    p_var = T.Var("P", "int32")
    spatial_b_var = T.Var("sB", "int32")
    reduction_b_var = T.Var("rB", "int32")
    inst_gen.bind_inst_iter(src_buffer_region, f_var, inst_repr.size, inst_repr.stride, True)
    inst_gen.bind_inst_iter(src_buffer_region, p_var, p_size, 1, False)
    reduction_b_extent = inst_gen.fill_in_block_dim(src_buffer_region, reduction_b_var, axes)
    spatial_b_extent = inst_gen.fill_in_block_dim(src_buffer_region, spatial_b_var)
    # Get reduction operation code
    opcode = reduce_ops[reduce_op]

    # Generate intermediate buffer if needed
    if reduction_b_extent != 1:
        intermediate_buffer = generate_intermediate_buffer(
            dst_buffer_region,
            reduction_b_extent,
            op.workspace,
            sctx,
        )

    # fmt: off
    # Single-stage reduction implementation
    if reduction_b_extent == 1:
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(src_buffer_region):
                                src_indices = T.meta_var(inst_gen.generate_indices(src_buffer_region))
                                dst_indices = T.meta_var(inst_gen.generate_indices(dst_buffer_region))
                                T.evaluate(T.nki.tensorreduce(dst[*dst_indices], src[*src_indices], opcode, negate, -1))
        return impl
    # Two-stage reduction implementation
    else:
        @T.prim_func(tirp=True)
        def two_stage_reduction():
            for b_loop in T.serial(0, spatial_b_extent):
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                                inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop, reduction_b_var: reduction_b_loop})
                                if inst_gen.make_guard(src_buffer_region):
                                    src_indices = T.meta_var(inst_gen.generate_indices(src_buffer_region))
                                    T.evaluate(T.nki.tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], src[src_indices], opcode, False, -1))
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map(src_buffer_region, {p_var: p_loop, f_var: 0, spatial_b_var: b_loop, reduction_b_var: f_loop})
                            inst_gen.set_bind_map(dst_buffer_region, {p_var: p_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(src_buffer_region):
                                dst_indices = T.meta_var(inst_gen.generate_indices(dst_buffer_region))
                                T.evaluate(T.nki.tensorreduce(dst[dst_indices], intermediate_buffer[p_loop, f_loop], opcode, negate, -1))
        return two_stage_reduction
    # fmt: on


# Register schedules for supported reduction operations
for op_name_, op_type_ in {
    "sum": ReduceOpType.SUM,
    "max": ReduceOpType.MAX,
    "min": ReduceOpType.MIN,
}.items():
    register_unary_binary_schedule(op_name_, op_type_, "trn", target_trn, [reduction_trn])
