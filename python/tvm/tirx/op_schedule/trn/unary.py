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
from tvm.script import tir as T, tirx as Tx
from tvm.tir import BufferRegion, PrimFunc, FloatImm
from tvm.tirx.op_schedule import ScheduleContext, fail
from tvm.tir.stmt import OpCall
from .common import (
    init_analyzer,
    check_workspace_buffer,
    InstructionGenerator,
    get_ewise_dim_map,
    nki_dim,
)
from ..common import MapOpType
from .binary import try_find_inst_nary

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
    inst_gen: InstructionGenerator,
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
            src.layout.is_trainium(),
            dst.layout.is_trainium(),
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
    dim_map = get_ewise_dim_map(src_buffer_region, dst_buffer_region, analyzer)
    inst_gen.link_buffer_regions(src_buffer_region, dst_buffer_region, dim_map)
    # Find optimal instruction parameters
    inst_repr = inst_gen.find_max_inst_size_from_one_region(dst_buffer_region, allowed_f_dim_dst)
    inst_repr = inst_gen.fit_inst_tile_to_region(inst_repr, src_buffer_region, allowed_f_dim_src)
    return inst_repr


def get_const_bias_tensor(bias, shape, dtype, workspace, sctx):
    """Create or retrieve a constant bias tensor."""
    if "const_bias" not in workspace:
        assert (
            sctx.alloc_only
        ), "Constant bias tensor must be specified in workspace. Run tvm.tirx.transform.PrivateBufferAlloc first."
        # Create new bias buffer
        bias_buffer = T.buffer(shape, dtype, scope="trn.sbuf", buffer_name="const_bias")
        sctx.add_alloc_buffer(bias_buffer)

        @T.prim_func(tirx=True)
        def const_bias_init():
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, shape[0], annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, shape[1], annotations={nki_dim: "F"}):
                        T.evaluate(T.nki.memset(bias_buffer[p_loop, f_loop], bias))
            Tx.tvm_kernel_replace_point()

        sctx.add_init_stmt(const_bias_init.body)
    else:
        # Use existing bias buffer
        bias_buffer = workspace["const_bias"]
        check_workspace_buffer(bias_buffer, shape, "trn.sbuf")

    return bias_buffer


def generate_unary_func(
    dst_buffer_region,
    _src,
    inst_gen: InstructionGenerator,
    inst_repr,
    unary_op,
    bias,
    scale,
    analyzer,
    workspace,
    config,
    sctx,
):
    """Generate a function that implements a unary operation."""
    # Prepare parameters
    p_size = dst_buffer_region.buffer.layout.size("P")

    # Apply instruction size limits if specified
    inst_size_limit = config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    f_var = T.Var("F", "int32")
    p_var = T.Var("P", "int32")
    b_var = T.Var("B", "int32")
    inst_gen.bind_inst_iter(dst_buffer_region, f_var, inst_repr.size, inst_repr.stride, True)
    inst_gen.bind_inst_iter(dst_buffer_region, p_var, p_size, 1, False)
    b_extent = inst_gen.fill_in_block_dim(dst_buffer_region, b_var)

    # Get operation code if available
    opcode = opcode_table.get(unary_op, None)

    # Extract buffers
    dst = dst_buffer_region.buffer
    src = _src.buffer if isinstance(_src, BufferRegion) else None

    # Handle bias tensor
    if isinstance(bias, (FloatImm, float)):
        bias_buffer = get_const_bias_tensor(
            bias, (p_size, inst_repr.size), dst.dtype, workspace, sctx
        )
    elif isinstance(bias, BufferRegion):
        bias_buffer = bias.buffer

    # fmt: off
    @T.prim_func(tirx=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})
                        dst_indices = T.meta_var(inst_gen.generate_indices(dst_buffer_region))
                        if inst_gen.make_guard(dst_buffer_region):
                            if unary_op == MapOpType.MEMSET:
                                T.evaluate(T.nki.memset(dst[*dst_indices], _src))
                            else:
                                src_indices = T.meta_var(inst_gen.generate_indices(_src))
                                if unary_op == MapOpType.RECIPROCAL:
                                    T.evaluate(T.nki.reciprocal(dst[*dst_indices], src[*src_indices]))
                                elif isinstance(bias, BufferRegion):
                                    bias_indices = T.meta_var(inst_gen.generate_indices(bias))
                                    T.evaluate(T.nki.activation(dst[*dst_indices], src[*src_indices], opcode, scale=scale, bias=bias_buffer[*bias_indices]))
                                else:
                                    T.evaluate(T.nki.activation(dst[*dst_indices], src[*src_indices], opcode, scale=scale, bias=bias_buffer[p_loop, f_loop]))
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
        fail("requires Trainium target and kernel exec_scope")

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

    inst_gen = InstructionGenerator([dst_buffer_region, _src], analyzer)
    # Find instruction parameters
    if CONST is None:
        inst_repr = try_find_inst_unary(dst_buffer_region, src_buffer_region, analyzer, inst_gen)
    else:
        inst_repr = try_find_inst_unary(dst_buffer_region, dst_buffer_region, analyzer, inst_gen)
    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        _src,
        inst_gen,
        inst_repr,
        unary_op,
        None,  # No bias
        None,  # No scale
        analyzer,
        op.workspace,
        op.config,
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
        fail("requires Trainium target and kernel exec_scope")

    # Extract operation arguments with defaults
    dst_buffer_region, src_buffer_region, _bias, scale = op.args
    scale = 1.0 if scale is None else scale
    _bias = 0.0 if _bias is None else _bias

    # Initialize analyzer and validate operation type
    analyzer = init_analyzer(sctx)
    assert unary_op in activation_map_ops, f"Unsupported activation operation {unary_op}"

    # Find instruction parameters
    inst_gen = InstructionGenerator([dst_buffer_region, src_buffer_region, _bias], analyzer)
    if isinstance(_bias, BufferRegion):
        inst_repr, _, _ = try_find_inst_nary(
            dst_buffer_region,
            [src_buffer_region, _bias],
            analyzer,
            inst_gen,
            allow_first_op_tensortensor=False,
        )
    else:
        # Handle scalar bias
        inst_repr = try_find_inst_unary(dst_buffer_region, src_buffer_region, analyzer, inst_gen)

    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        src_buffer_region,
        inst_gen,
        inst_repr,
        unary_op,
        _bias,
        scale,
        analyzer,
        op.workspace,
        op.config,
        sctx,
    )
