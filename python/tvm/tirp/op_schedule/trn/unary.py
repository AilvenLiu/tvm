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

from typing import Optional, Union, Tuple
import operator
from functools import reduce

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T, tirp as Tp
from tvm.tir import BufferRegion, PrimFunc, FloatImm
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir.stmt import OpCall
from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
    bound_inst_with_limit,
    init_analyzer,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    bound_buffer_region,
    make_guard,
    check_workspace_buffer,
    nki_dim,
)
from ..common import MapOpType
from .binary import try_find_inst_binary

# Operation type classifications
non_activation_unary_map_ops = [MapOpType.RECIPROCAL, MapOpType.MEMSET]
activation_map_ops = [MapOpType.SQRT, MapOpType.EXP]

# Operation code table for instructions
opcode_table = {
    MapOpType.SQRT: "sqrt",
    MapOpType.EXP: "exp",
}

# Operations that take constants as input
const_input_ops = [MapOpType.MEMSET]


def try_find_inst_unary(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    analyzer: Analyzer,
    allowed_f_dim_dst: Optional[Tuple[int]] = None,
    allowed_f_dim_src: Optional[Tuple[int]] = None,
):
    """Find instruction parameters for a unary operation."""
    dst = dst_buffer_region.buffer
    src = src_buffer_region.buffer

    # Validate buffer layouts and scopes
    valid_layout_scope = all(
        [
            src.layout and dst.layout,
            src.scope() in ("trn.sbuf", "trn.psum"),
            dst.scope() == "trn.sbuf",
            isinstance(src.layout, T.TrainiumLayout),
            isinstance(dst.layout, T.TrainiumLayout),
        ]
    )

    if not valid_layout_scope:
        assert (
            False
        ), f"scope or layout mismatch, src: {src_buffer_region}, dst: {dst_buffer_region}"

    # Extract and validate dimensions
    dst_region = dst_buffer_region.region
    src_region = src_buffer_region.region

    dst_extent = [r.extent for r in dst_region]
    src_extent = [r.extent for r in src_region]

    dst_extent_nonunit = [e for e in dst_extent if e != 1]
    src_extent_nonunit = [e for e in src_extent if e != 1]

    # Verify dimensions match
    dims_match = len(src_extent_nonunit) == len(dst_extent_nonunit) and all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_nonunit, dst_extent_nonunit)
    )

    if not dims_match:
        assert (
            False
        ), f"shape or dimension mismatch, src: {src_buffer_region}, dst: {dst_buffer_region}"

    # Find optimal instruction parameters
    return find_max_inst_size_unary(
        dst_buffer_region, src_buffer_region, analyzer, allowed_f_dim_dst, allowed_f_dim_src
    )


def get_const_bias_tensor(bias, shape, dtype, workspace, sctx):
    """Create or retrieve a constant bias tensor."""
    if "const_bias" not in workspace:
        # Create new bias buffer
        bias_buffer = T.buffer(shape, dtype, scope="trn.sbuf", buffer_name="const_bias")
        sctx.add_alloc_buffer(bias_buffer)

        @T.prim_func(tirp=True)
        def const_bias_init():
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, shape[0], annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, shape[1], annotations={nki_dim: "F"}):
                        T.evaluate(T.nki_memset(bias_buffer[p_loop, f_loop], bias))
            Tp.tvm_kernel_replace_point()

        sctx.add_init_stmt(const_bias_init.body)
    else:
        # Use existing bias buffer
        bias_buffer = workspace["const_bias"]
        check_workspace_buffer(bias_buffer, shape, "trn.sbuf")

    return bias_buffer


def generate_unary_func(
    dst_buffer_region,
    _src,
    inst_size,
    f_gen_axes,
    f_gen_dst_idx,
    f_gen_src_idx,
    f_gen_bias_idx,
    unary_op,
    bias,
    scale,
    analyzer,
    workspace,
    schedule_config,
    sctx,
):
    """Generate a function that implements a unary operation."""
    # Prepare parameters
    bound_dst = bound_buffer_region(dst_buffer_region, analyzer)
    p_size = dst_buffer_region.buffer.layout.partition_size

    # Calculate extents
    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // p_size // inst_size

    # Apply instruction size limits if specified
    inst_size_limit = schedule_config.get("max_inst_size", 512)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )

    # Get operation code if available
    opcode = opcode_table.get(unary_op, None)

    # Extract buffers
    dst = dst_buffer_region.buffer
    src = _src.buffer if isinstance(_src, BufferRegion) else None

    # Create guard function for destination indices
    f_dst_guard = make_guard(dst_buffer_region, analyzer)

    # Handle bias tensor
    if isinstance(bias, (FloatImm, float)):
        bias_buffer = get_const_bias_tensor(
            bias, (p_size, actual_inst_size), dst.dtype, workspace, sctx
        )
    elif isinstance(bias, BufferRegion):
        bias_buffer = bias.buffer

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, actual_inst_size, annotations={nki_dim: "F"}):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                        if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                            if unary_op == MapOpType.MEMSET:
                                T.evaluate(T.nki_memset(dst[*dst_indices], _src))
                            else:
                                src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop) )
                                if unary_op == MapOpType.RECIPROCAL:
                                    T.evaluate(T.nki_reciprocal(dst[*dst_indices], src[*src_indices]))
                                elif isinstance(bias, BufferRegion):
                                    bias_indices = T.meta_var(f_gen_bias_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                                    T.evaluate(T.nki_activation(dst[*dst_indices], src[*src_indices], opcode, scale=scale, bias=bias_buffer[*bias_indices]))
                                else:
                                    T.evaluate(T.nki_activation(dst[*dst_indices], src[*src_indices], opcode, scale=scale, bias=bias_buffer[p_loop, f_loop]))
    # fmt: on

    return impl


def unary_trn(
    op: OpCall,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary operation on Trainium."""
    # Check execution environment
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    # Extract operation arguments
    dst_buffer_region, _src = op.args

    # Handle constant or buffer source
    if isinstance(_src, FloatImm):
        if unary_op not in const_input_ops:
            assert False, f"Unsupported unary operation {unary_op} taking const as input"
        CONST = _src
        src_buffer_region = None
    else:
        CONST = None
        src_buffer_region = _src

    # Initialize analyzer and validate operation type
    analyzer = init_analyzer(sctx)
    assert unary_op in non_activation_unary_map_ops, f"Unsupported unary operation {unary_op}"

    # Prepare bound regions
    bound_dst = bound_buffer_region(dst_buffer_region, analyzer)

    # Find instruction parameters
    if CONST is None:
        bound_src = bound_buffer_region(src_buffer_region, analyzer)
        inst_size, inst_stride, inst_data_iters = try_find_inst_unary(
            bound_dst, bound_src, analyzer
        )
    else:
        inst_size, inst_stride, inst_data_iters = try_find_inst_unary(
            bound_dst, bound_dst, analyzer
        )

    # Generate index functions
    f_gen_axes = generate_axes_in_region(dst_buffer_region, inst_stride, inst_data_iters, analyzer)
    f_gen_dst_idx = f_gen_idx_anchor(dst_buffer_region, f_gen_axes)

    f_gen_src_idx = None
    if CONST is None:
        dim_map = get_ewise_dim_map(dst_buffer_region, src_buffer_region, analyzer)
        f_gen_src_idx = f_gen_idx_mapped(src_buffer_region, f_gen_axes, dim_map)

    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        _src,
        inst_size,
        f_gen_axes,
        f_gen_dst_idx,
        f_gen_src_idx,
        None,  # No bias indices
        unary_op,
        None,  # No bias
        None,  # No scale
        analyzer,
        op.workspace,
        op.schedule_config,
        sctx,
    )


def unary_with_bias_scale_trn(
    op: OpCall,
    unary_op: MapOpType = MapOpType.SQRT,
    sctx: ScheduleContext = None,
) -> Optional[PrimFunc]:
    """Schedule unary operation with bias and scale on Trainium."""
    # Check execution environment
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    # Extract operation arguments with defaults
    dst_buffer_region, src_buffer_region, _bias, scale = op.args
    scale = 1.0 if scale is None else scale
    _bias = 0.0 if _bias is None else _bias

    # Initialize analyzer and validate operation type
    analyzer = init_analyzer(sctx)
    assert unary_op in activation_map_ops, f"Unsupported activation operation {unary_op}"

    # Prepare bound regions
    bound_dst = bound_buffer_region(dst_buffer_region, analyzer)
    bound_src = bound_buffer_region(src_buffer_region, analyzer)

    # Find instruction parameters
    if isinstance(_bias, BufferRegion):
        # Handle buffer bias
        bound_bias = bound_buffer_region(_bias, analyzer)
        result = try_find_inst_binary(
            bound_dst,
            bound_src,
            bound_bias,
            analyzer,
            allow_tensortensor=False,
            allow_reverse=False,
        )
        inst_size, inst_stride, inst_data_iters, inst_type, reverse, broadcast_dims = result
        assert inst_size is not None, f"Failed to find a valid instruction: {op}"
    else:
        # Handle scalar bias
        inst_size, inst_stride, inst_data_iters = try_find_inst_unary(
            bound_dst, bound_src, analyzer
        )

    # Generate index functions
    f_gen_axes = generate_axes_in_region(bound_dst, inst_stride, inst_data_iters, analyzer)
    f_gen_dst_idx = f_gen_idx_anchor(dst_buffer_region, f_gen_axes)

    # Map dimensions from destination to source
    dst_to_src_dim_map = get_ewise_dim_map(dst_buffer_region, src_buffer_region, analyzer)
    f_gen_src_idx = f_gen_idx_mapped(src_buffer_region, f_gen_axes, dst_to_src_dim_map)

    # Handle bias indices if needed
    f_gen_bias_idx = None
    if isinstance(_bias, BufferRegion):
        offset = len(src_buffer_region.region) - len(_bias.region)
        dst_to_bias_dim_map = {
            d: s - offset for d, s in dst_to_src_dim_map.items() if s not in broadcast_dims
        }
        f_gen_bias_idx = f_gen_idx_mapped(_bias, f_gen_axes, dst_to_bias_dim_map)

    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        src_buffer_region,
        inst_size,
        f_gen_axes,
        f_gen_dst_idx,
        f_gen_src_idx,
        f_gen_bias_idx,
        unary_op,
        _bias,
        scale,
        analyzer,
        op.workspace,
        op.schedule_config,
        sctx,
    )
