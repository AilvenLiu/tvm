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

"""Implementation of compose operator schedules."""

from typing import Optional, List
import operator
from functools import reduce

from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, OpCall
from tvm.tirp.op_schedule import (
    ScheduleContext,
    register_dispatch,
    predicate,
)
from tvm.ir import Op
from tvm.tirp.operator.op import BinaryReduce, UnaryReduce, BinaryChain, ReduceNegate
from .common import (
    init_analyzer,
    get_reduction_dim_map,
    InstructionGenerator,
    nki_dim,
)
from .unary import try_find_inst_unary, get_const_bias_tensor
from .binary import InstType, try_find_inst_nary
from .reduction import generate_intermediate_buffer, reduction_trn
from ..common import ReduceOpType

# Operation code mappings
opcode_table = {
    Op.get("tirp.add"): "add",
    Op.get("tirp.sub"): "sub",
    Op.get("tirp.mul"): "mul",
    Op.get("tirp.maximum"): "max",
    Op.get("tirp.minimum"): "min",
    Op.get("tirp.sqrt"): "sqrt",
    Op.get("tirp.sum"): "add",
    Op.get("tirp.max"): "max",
    Op.get("tirp.min"): "min",
    Op.get("tirp.exp"): "exp",
}

optype_table = {
    Op.get("tirp.sum"): ReduceOpType.SUM,
    Op.get("tirp.max"): ReduceOpType.MAX,
    Op.get("tirp.min"): ReduceOpType.MIN,
}


def binary_reduce_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate a TRN schedule for binary reduction operations."""
    op = OpCall.downcast(op)
    assert isinstance(op, BinaryReduce), f"invalid operator downcast: {op}"

    # Extract operation components
    binary_output, reduce_output = op.dsts
    binary_input1, binary_input2 = op.srcs
    reduce_axes = op.reduce_axes
    analyzer = init_analyzer(sctx)

    # Normalize negative axes
    reduce_axes = [i if i >= 0 else len(binary_output.buffer.shape) + i for i in reduce_axes]

    # Find instruction patterns
    inst_gen = InstructionGenerator(
        [binary_output, binary_input1, binary_input2, reduce_output], analyzer
    )
    reduce_dim_map = get_reduction_dim_map(binary_output, reduce_output, reduce_axes, analyzer)
    inst_gen.link_buffer_regions(binary_output, reduce_output, reduce_dim_map)
    inst_repr, inst_type, reverse = try_find_inst_nary(
        binary_output,
        [binary_input1, binary_input2],
        analyzer,
        inst_gen,
        allowed_f_dim_dst=reduce_axes,
        allow_first_op_tensortensor=False,
    )

    # Apply instruction size limits
    inst_size_limit = op.config.get("max_inst_size", None)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    # Generate axes and validate
    assert (
        inst_type[0] == InstType.TENSOR_SCALAR
    ), f"TensorTensor is not supported for vector reduce: {op}"

    # Handle input reversal if needed
    if reverse[0]:
        binary_input1, binary_input2 = binary_input2, binary_input1

    # Generate intermediate buffer for reduction if needed
    p_var = T.Var("P", "int32")
    f_var = T.Var("F", "int32")
    reduction_b_var = T.Var("rB", "int32")
    spatial_b_var = T.Var("sB", "int32")
    p_size = binary_output.buffer.layout.size("P")
    inst_gen.bind_inst_iter(binary_output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(binary_output, f_var, inst_repr.size, inst_repr.stride, True)
    reduction_b_extent = inst_gen.fill_in_block_dim(binary_output, reduction_b_var, reduce_axes)
    spatial_b_extent = inst_gen.fill_in_block_dim(binary_output, spatial_b_var)
    if reduction_b_extent != 1:
        intermediate_buffer = generate_intermediate_buffer(
            reduce_output,
            reduction_b_extent,
            op.workspace,
            sctx,
        )

    # Handle source 2 (either buffer region or constant)
    CONST = binary_input2 if not isinstance(binary_input2, BufferRegion) else None
    # Extract buffers and opcodes
    src1, src2 = binary_input1.buffer, (
        binary_input2.buffer if isinstance(binary_input2, BufferRegion) else None
    )
    dst1, dst2 = binary_output.buffer, reduce_output.buffer
    binary_opcode, reduce_opcode = opcode_table[op.binary_op], opcode_table[op.reduce_op]
    # Create appropriate implementation based on intermediate buffer requirement
    if reduction_b_extent == 1:
        # Direct implementation without intermediate buffer
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop})
                            src_1_indices = T.meta_var(inst_gen.generate_indices(binary_input1))
                            vec_dst_idx = T.meta_var(inst_gen.generate_indices(binary_output))
                            reduce_dst_idx = T.meta_var(inst_gen.generate_indices(reduce_output))
                            if inst_gen.make_guard(binary_output):
                                if CONST is None:
                                    src_2_indices = T.meta_var(inst_gen.generate_indices(binary_input2))
                                    T.nki.tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], binary_opcode, reduce_opcode, reverse[0])
                                else:
                                    T.nki.tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, binary_opcode, reduce_opcode, reverse[0])
        # fmt: on
    else:
        # Implementation with intermediate buffer
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                                inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop, reduction_b_var: reduction_b_loop})
                                if inst_gen.make_guard(binary_output):
                                    src_1_indices = T.meta_var(inst_gen.generate_indices(binary_input1))
                                    vec_dst_idx = T.meta_var(inst_gen.generate_indices(binary_output))
                                    if CONST is None:
                                        src_2_indices = T.meta_var(inst_gen.generate_indices(binary_input2))
                                        T.nki.tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], binary_opcode, reduce_opcode, reverse[0])
                                    else:
                                        T.nki.tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, binary_opcode, reduce_opcode, reverse[0])
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(reduce_output):
                                dst_2_indices = T.meta_var(inst_gen.generate_indices(reduce_output))
                                T.nki.tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1)
        # fmt: on

    return impl


def unary_reduce_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate a TRN schedule for unary reduction operations."""
    op = OpCall.downcast(op)
    assert isinstance(op, UnaryReduce), f"invalid operator downcast: {op}"

    # Extract operation components
    unary_output, reduce_output = op.dsts
    unary_input, bias, scale = op.srcs
    analyzer = init_analyzer(sctx)

    # Normalize axes and default values
    reduce_axes = [i if i >= 0 else len(unary_output.buffer.shape) + i for i in op.reduce_axes]
    scale = 1.0 if scale is None else scale
    bias = 0.0 if bias is None else bias

    inst_gen = InstructionGenerator([unary_output, unary_input, bias, reduce_output], analyzer)
    reduce_dim_map = get_reduction_dim_map(unary_output, reduce_output, reduce_axes, analyzer)
    inst_gen.link_buffer_regions(unary_output, reduce_output, reduce_dim_map)
    # Find instruction patterns based on bias type
    if isinstance(bias, BufferRegion):
        inst_repr, _, _ = try_find_inst_nary(
            unary_output,
            [unary_input, bias],
            analyzer,
            inst_gen,
            allow_first_op_tensortensor=False,
            allowed_f_dim_dst=reduce_axes,
        )
    else:
        inst_repr = try_find_inst_unary(
            unary_output, unary_input, analyzer, inst_gen, allowed_f_dim_dst=reduce_axes
        )

    # Apply instruction size limits
    inst_size_limit = op.config.get("max_inst_size", None)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    p_var = T.Var("P", "int32")
    f_var = T.Var("F", "int32")
    reduction_b_var = T.Var("rB", "int32")
    spatial_b_var = T.Var("sB", "int32")
    p_size = unary_output.buffer.layout.size("P")
    inst_gen.bind_inst_iter(unary_output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(unary_output, f_var, inst_repr.size, inst_repr.stride, True)
    reduction_b_extent = inst_gen.fill_in_block_dim(unary_output, reduction_b_var, reduce_axes)
    spatial_b_extent = inst_gen.fill_in_block_dim(unary_output, spatial_b_var)
    if reduction_b_extent != 1:
        intermediate_buffer = generate_intermediate_buffer(
            reduce_output,
            reduction_b_extent,
            op.workspace,
            sctx,
        )
    # Extract buffers and opcodes
    src, dst1, dst2 = unary_input.buffer, unary_output.buffer, reduce_output.buffer
    unary_opcode = opcode_table[op.unary_op]
    reduce_opcode = opcode_table[op.reduce_op]

    # Handle bias buffer
    bias_buffer = (
        bias.buffer
        if isinstance(bias, BufferRegion)
        else get_const_bias_tensor(bias, (p_size, inst_repr.size), dst1.dtype, op.workspace, sctx)
    )

    # Create appropriate implementation based on intermediate buffer requirement
    if reduction_b_extent == 1:
        # Direct implementation without intermediate buffer
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop})
                            src_1_indices = T.meta_var(inst_gen.generate_indices(unary_input))
                            dst_1_indices = T.meta_var(inst_gen.generate_indices(unary_output))
                            dst_2_indices = T.meta_var(inst_gen.generate_indices(reduce_output))
                            if inst_gen.make_guard(unary_output):
                                if isinstance(bias, BufferRegion):
                                    src_bias_indices = T.meta_var(inst_gen.generate_indices(bias))
                                    T.evaluate(T.nki.activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                                else:
                                    T.evaluate(T.nki.activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[p_loop, f_loop], scale))
        # fmt: on

        import tvm

        mod = tvm.IRModule({"main": impl})
        mod = tvm.tir.transform.Simplify()(mod)
        return mod["main"]
    else:
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                                inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop, reduction_b_var: reduction_b_loop})
                                src_1_indices = T.meta_var(inst_gen.generate_indices(unary_input))
                                dst_1_indices = T.meta_var(inst_gen.generate_indices(unary_output))
                                if inst_gen.make_guard(unary_output):
                                    if isinstance(bias, BufferRegion):
                                        src_bias_indices = T.meta_var(inst_gen.generate_indices(bias))
                                        T.evaluate(T.nki.activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                                    else:
                                        T.evaluate(T.nki.activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[p_loop, f_loop], scale))
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(reduce_output):
                                dst_2_indices = T.meta_var(inst_gen.generate_indices(reduce_output))
                                # TODO: we should use nki.activation_reduce as second stage reduction
                                T.evaluate(T.nki.tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1))
        # fmt: on

        return impl


def binary_chain_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate a TRN schedule for binary chain operations."""
    op = OpCall.downcast(op)
    assert isinstance(op, BinaryChain), f"invalid operator downcast: {op}"

    # Extract operation components
    output = op.dsts[0]
    srcs = op.srcs
    reverse = [False, op.reverse1]
    analyzer = init_analyzer(sctx)

    # Find instruction patterns
    inst_gen = InstructionGenerator([output] + srcs, analyzer)
    inst_result = try_find_inst_nary(
        output, srcs, analyzer, inst_gen, allow_first_op_tensortensor=False
    )
    inst_repr, inst_types, _reverse = inst_result

    # Generate axes and validate
    assert (
        inst_types[0] == InstType.TENSOR_SCALAR
    ), "The first operator must be a tensor scalar operator"

    # Handle input reversal if needed
    reverse[0] = _reverse[0]
    if reverse[0]:
        srcs[0], srcs[1] = srcs[1], srcs[0]

    p_var = T.Var("P", "int32")
    b_var = T.Var("B", "int32")
    f_var = T.Var("F", "int32")
    p_size = output.buffer.layout.size("P")
    inst_size_limit = op.config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)
    inst_gen.bind_inst_iter(output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(output, f_var, inst_repr.size, inst_repr.stride, True)
    b_extent = inst_gen.fill_in_block_dim(output, b_var)

    # Extract buffers and opcodes
    src, dst = srcs[0].buffer, output.buffer
    opcode0, opcode1 = opcode_table[op.op0], opcode_table[op.op1]

    # Determine operation function based on instruction type
    func = (
        T.nki.scalar_tensor_scalar
        if inst_types[1] == InstType.TENSOR_SCALAR
        else T.nki.scalar_tensor_tensor
    )

    # Helper function to get source indices
    def get_srcs(inst_gen):
        return [
            (
                srcs[i].buffer[inst_gen.generate_indices(srcs[i])]
                if isinstance(srcs[i], BufferRegion)
                else srcs[i]
            )
            for i in range(len(srcs))
        ]

    # Create implementation
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})
                        dst_indices = T.meta_var(inst_gen.generate_indices(output))
                        srcs = T.meta_var(get_srcs(inst_gen))
                        if inst_gen.make_guard(output):
                            T.evaluate(func(dst[*dst_indices], *srcs, opcode0, opcode1, reverse[0], reverse[1]))
    # fmt: on

    return impl


def reduce_negate_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate a TRN schedule for reduce negate operations."""
    op = OpCall.downcast(op)
    assert isinstance(op, ReduceNegate), f"invalid operator downcast: {op}"
    return reduction_trn(op, optype_table[op.reduce_op], sctx, negate=True)


def compose_op_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate a TRN schedule for compose operations."""
    raise NotImplementedError(
        "Generic compose_op must be lowered to specific compose ops before operator-level passes"
    )


# Rich dispatcher variants for TRN compose ops
@register_dispatch(
    "binary_reduce",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "kernel",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        )
    ],
)
def binary_reduce_trn_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return binary_reduce_trn(op, sctx)


@register_dispatch(
    "unary_reduce",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "kernel",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        )
    ],
)
def unary_reduce_trn_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return unary_reduce_trn(op, sctx)


@register_dispatch(
    "binary_chain",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "kernel",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        )
    ],
)
def binary_chain_trn_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return binary_chain_trn(op, sctx)


@register_dispatch(
    "reduce_negate",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "kernel",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        )
    ],
)
def reduce_negate_trn_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return reduce_negate_trn(op, sctx)


@register_dispatch(
    "compose_op",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "kernel",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        )
    ],
)
def compose_op_trn_dispatch(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    return compose_op_trn(op, sctx)
