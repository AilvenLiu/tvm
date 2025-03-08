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


from typing import Optional, Union, List
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, FloatImm, OpCall
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from tvm.ir import Op, assert_structural_equal, structural_equal

from functools import reduce
from .common import (
    get_ewise_dim_map,
    init_analyzer,
    get_reduction_dim_map,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
)
from .unary import try_find_inst_unary
from .binary import try_find_inst_binary, InstType, try_find_inst_nary
from .reduction import generate_intermediate_buffer, reduction_trn
from ..common import ReduceOpType
from ..registry import f_op_scheduler


vector_ops = [
    Op.get("tirp.add"),
    Op.get("tirp.sub"),
    Op.get("tirp.mul"),
    Op.get("tirp.maximum"),
    Op.get("tirp.minimum"),
]
act_ops = [Op.get("tirp.sqrt"), Op.get("tirp.exp")]
reduce_ops = [Op.get("tirp.sum"), Op.get("tirp.max"), Op.get("tirp.min")]
reduce_ops_after_act = [Op.get("tirp.sum")]
reduce_ops_after_vector = [Op.get("tirp.sum"), Op.get("tirp.max"), Op.get("tirp.min")]

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


def validate_single_op(op_calls: List[OpCall], sctx: ScheduleContext):
    for op_call in op_calls:
        try:
            f_op_scheduler(op_call, sctx)
        except (ValueError, NotImplementedError) as e:
            raise ValueError(f"Invalid operator {op_call}, error: {e}")


def compose_vector_reduce(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    vec_output, vec_input1, vec_input2 = op_call_1.args
    reduce_output, reduce_input, reduce_axes, accum = op_call_2.args

    # validate single operator
    validate_single_op([op_call_1, op_call_2], sctx)

    # start compose
    assert_structural_equal(
        vec_output, reduce_input
    ), "Vector output and reduce input must be the same"
    analyzer = init_analyzer(sctx)
    reduce_axes = [i if i >= 0 else len(reduce_input.buffer.shape) + i for i in reduce_axes]
    inst_size, f_gen_axes, inst_type, reverse, broadcast_dims = try_find_inst_binary(
        vec_output,
        vec_input1,
        vec_input2,
        analyzer,
        allowed_f_dim_dst=reduce_axes,
        allow_tensortensor=False,
    )
    assert inst_size is not None, "Failed to find a valid instruction"
    assert inst_type == InstType.TENSOR_SCALAR, "TensorTensor is not supported for vector reduce"
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"
    if reverse:
        vec_input1, vec_input2 = vec_input2, vec_input1
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(
        reduce_output, reduce_input, reduce_axes, inst_size, analyzer
    )
    f_gen_vec_dst_idx = f_gen_idx_anchor(vec_output, f_gen_axes)
    vec_dst_to_src1_dim_map = get_ewise_dim_map(vec_output, vec_input1, analyzer)
    f_gen_src1_idx = f_gen_idx_mapped(vec_input1, f_gen_axes, vec_dst_to_src1_dim_map)
    if isinstance(vec_input2, BufferRegion):
        src1_src2_offset = len(vec_input1.region) - len(vec_input2.region)
        vec_dst_to_src2_dim_map = {
            d: s - src1_src2_offset
            for d, s in vec_dst_to_src1_dim_map.items()
            if s not in broadcast_dims
        }
        f_gen_src2_idx = f_gen_idx_mapped(vec_input2, f_gen_axes, vec_dst_to_src2_dim_map)
        CONST = None
    else:
        CONST = vec_input2
    f_gen_reduce_dst_idx = f_gen_idx_mapped(
        reduce_output, f_gen_axes, get_reduction_dim_map(reduce_input, reduce_output, reduce_axes)
    )
    p_size = vec_output.buffer.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in vec_output.region], 1) // p_size // inst_size
    src1, src2, dst1, dst2 = (
        vec_input1.buffer,
        vec_input2.buffer if isinstance(vec_input2, BufferRegion) else None,
        vec_output.buffer,
        reduce_output.buffer,
    )
    opcode, reduce_opcode = opcode_table[op_call_1.op], opcode_table[op_call_2.op]
    if intermediate_shape is None:
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop, f_loop in T.grid(p_size, inst_size):
                        src_1_indices = T.meta_var(f_gen_src1_idx(((b_loop, b_extent),), f_loop, p_loop))
                        vec_dst_idx = T.meta_var(f_gen_vec_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        reduce_dst_idx = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        if CONST is None:
                            src_2_indices = T.meta_var(f_gen_src2_idx(((b_loop, b_extent),), f_loop, p_loop))
                            T.nki_tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], opcode, reduce_opcode, reverse)
                        else:
                            T.nki_tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, opcode, reduce_opcode, reverse)
        # fmt: on
        return impl
    else:
        dim2block_var = {dim: 1 if dim in reduce_axes else 0 for dim in range(len(dst1.shape))}
        b_extent //= reduction_b_extent
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            with T.kernel():
                intermediate_buffer = T.alloc_buffer(intermediate_shape, dtype=dst2.dtype, layout=intermediate_layout, scope="trn.sbuf")
                for b_loop in T.serial(0, b_extent):
                    for reduction_b_loop in T.serial(0, reduction_b_extent):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, f_loop in T.grid(p_size, inst_size):
                                src_1_indices = T.meta_var(f_gen_src1_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                vec_dst_idx = T.meta_var(f_gen_vec_dst_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                if CONST is None:
                                    src_2_indices = T.meta_var(f_gen_src2_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                    T.nki_tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], opcode, reduce_opcode, reverse)
                                else:
                                    T.nki_tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, opcode, reduce_opcode, reverse)
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            dst_2_indices = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                            T.nki_tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1)
        # fmt: on
        return impl


def compose_act_reduce(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    act_output, act_input, bias, scale = op_call_1.args
    reduce_output, reduce_input, reduce_axes, accum = op_call_2.args

    # validate single operator
    validate_single_op([op_call_1, op_call_2], sctx)

    # start compose
    assert_structural_equal(
        act_output, reduce_input
    ), "Act output and reduce input must be the same"
    analyzer = init_analyzer(sctx)
    reduce_axes = [i if i >= 0 else len(reduce_input.buffer.shape) + i for i in reduce_axes]
    scale = 1.0 if scale is None else scale
    if bias is not None:
        inst_size, f_gen_axes, inst_type, reverse, broadcast_dims = try_find_inst_binary(
            act_output,
            act_input,
            bias,
            analyzer,
            allow_tensortensor=False,
            allow_reverse=False,
            allowed_f_dim_dst=reduce_axes,
        )
    else:
        inst_size, f_gen_axes = try_find_inst_unary(
            act_output, act_input, analyzer, allowed_f_dim_dst=reduce_axes
        )
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(
        reduce_output, reduce_input, reduce_axes, inst_size, analyzer
    )
    f_gen_act_dst_idx = f_gen_idx_anchor(act_output, f_gen_axes)
    dst_to_src_dim_map = get_ewise_dim_map(act_output, act_input, analyzer)
    f_gen_act_src_idx = f_gen_idx_mapped(act_input, f_gen_axes, dst_to_src_dim_map)
    f_gen_reduce_dst_idx = f_gen_idx_mapped(
        reduce_output, f_gen_axes, get_reduction_dim_map(reduce_input, reduce_output, reduce_axes)
    )
    if bias is not None and isinstance(bias, BufferRegion):
        offset = len(act_input.region) - len(bias.region)
        dst_to_bias_dim_map = {
            d: s - offset for d, s in dst_to_src_dim_map.items() if s not in broadcast_dims
        }
        f_gen_bias_idx = f_gen_idx_mapped(bias, f_gen_axes, dst_to_bias_dim_map)
    else:
        f_gen_bias_idx = None
    p_size = act_output.buffer.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in act_output.region], 1) // p_size // inst_size
    src, dst1, dst2 = act_input.buffer, act_output.buffer, reduce_output.buffer
    opcode, reduce_opcode = opcode_table[op_call_1.op], opcode_table[op_call_2.op]
    bias_buffer = bias.buffer if isinstance(bias, BufferRegion) else None

    if intermediate_shape is None:
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop, f_loop in T.grid(p_size, inst_size):
                        src_1_indices = T.meta_var(f_gen_act_src_idx(((b_loop, b_extent),), f_loop, p_loop))
                        dst_1_indices = T.meta_var(f_gen_act_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        dst_2_indices = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        if bias is None:
                            T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode))
                        elif isinstance(bias, BufferRegion):
                            src_bias_indices = T.meta_var(f_gen_bias_idx(((b_loop, b_extent),), f_loop, p_loop))
                            T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                        else:
                            T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode, bias, scale))
        # fmt: on
        import tvm

        mod = tvm.IRModule({"main": impl})
        mod = tvm.tir.transform.Simplify()(mod)
        return mod["main"]
    else:
        dim2block_var = {dim: 1 if dim in reduce_axes else 0 for dim in range(len(dst1.shape))}
        b_extent //= reduction_b_extent
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            with T.kernel():
                intermediate_buffer = T.alloc_buffer(intermediate_shape, dtype=dst2.dtype, layout=intermediate_layout, scope="trn.sbuf")
                for b_loop in T.serial(0, b_extent):
                    for reduction_b_loop in T.serial(0, reduction_b_extent):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, f_loop in T.grid(p_size, inst_size):
                                src_1_indices = T.meta_var(f_gen_act_src_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                dst_1_indices = T.meta_var(f_gen_act_dst_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                if bias is None:
                                    T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode))
                                elif isinstance(bias, BufferRegion):
                                    src_bias_indices = T.meta_var(f_gen_bias_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                    T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                                else:
                                    T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode, bias, scale))
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            dst_2_indices = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                            T.evaluate(T.nki_tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1))
        # fmt: on
        return impl


def compose_vector_chain(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    first_output, first_input_1, first_input_2 = op_call_1.args
    second_output, second_input_1, second_input_2 = op_call_2.args

    # validate single operator
    validate_single_op([op_call_1, op_call_2], sctx)

    # validate composition
    if (
        isinstance(second_input_1, BufferRegion)
        and structural_equal(first_output, second_input_1)
        and first_output.buffer == second_input_1.buffer
    ):
        reverse = [False, False]
        srcs = [first_input_1, first_input_2, second_input_2]
    elif (
        isinstance(second_input_2, BufferRegion)
        and structural_equal(first_output, second_input_2)
        and first_output.buffer == second_input_2.buffer
    ):
        reverse = [False, True]
        srcs = [first_input_1, first_input_2, second_input_1]
    else:
        raise ValueError(f"The output of {op_call_1} is not the input of {op_call_2}")

    # FIXME: we need to check whether the intermediate buffer is used by other ops
    analyzer = init_analyzer(sctx)
    inst_size, f_gen_axes, inst_types, _reverse, broadcast_dims = try_find_inst_nary(
        second_output, srcs, analyzer, allow_first_op_tensortensor=False
    )
    assert inst_size is not None, "Failed to find a valid instruction"
    assert (
        inst_types[0] == InstType.TENSOR_SCALAR
    ), "The first operator must be a tensor scalar operator"
    reverse[0] = _reverse[0]
    if reverse[0]:
        srcs[0], srcs[1] = srcs[1], srcs[0]
    f_gen_dst_idx = f_gen_idx_anchor(second_output, f_gen_axes)
    dst_to_src0_dim_map = get_ewise_dim_map(second_output, srcs[0], analyzer)
    f_gen_src_idx_array = [f_gen_idx_mapped(srcs[0], f_gen_axes, dst_to_src0_dim_map)]
    for i, src in enumerate(srcs[1:]):
        if isinstance(src, BufferRegion):
            src_src0_offset = len(srcs[0].region) - len(src.region)
            dst_to_src_dim_map = {
                d: s - src_src0_offset
                for d, s in dst_to_src0_dim_map.items()
                if s not in broadcast_dims[i]
            }
            f_gen_src_idx_array.append(f_gen_idx_mapped(src, f_gen_axes, dst_to_src_dim_map))
        else:
            f_gen_src_idx_array.append(None)
    p_size = second_output.buffer.layout.partition_size
    b_extent = (
        reduce(operator.mul, [r.extent for r in second_output.region], 1) // p_size // inst_size
    )
    src, dst = srcs[0].buffer, second_output.buffer
    opcode1, opcode2 = opcode_table[op_call_1.op], opcode_table[op_call_2.op]
    func = (
        T.nki_scalar_tensor_scalar
        if inst_types[1] == InstType.TENSOR_SCALAR
        else T.nki_scalar_tensor_tensor
    )
    inst_size_limit = get_hardware_inst_size_limit(is_dma=False)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )

    def get_srcs(b_loop, f_loop, p_loop):
        return [
            (
                srcs[i].buffer[f_gen_src_idx_array[i](((b_loop, b_extent),), f_loop, p_loop)]
                if isinstance(srcs[i], BufferRegion)
                else srcs[i]
            )
            for i in range(len(srcs))
        ]

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, f_loop in T.grid(p_size, actual_inst_size):
                    f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                    dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                    srcs = T.meta_var(get_srcs(b_loop, f_loop_wo_limit, p_loop))
                    T.evaluate(func(dst[*dst_indices], *srcs, opcode1, opcode2, reverse[0], reverse[1]))
    # fmt: on
    return impl


def is_negate_op(op_call: OpCall) -> bool:
    if op_call.op != Op.get("tirp.mul"):
        return False
    if isinstance(op_call.args[1], FloatImm):
        return op_call.args[1].value == -1.0
    if isinstance(op_call.args[2], FloatImm):
        return op_call.args[2].value == -1.0
    return False


def compose_reduce_negate(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    reduce_output, reduce_input, reduce_axes, accum = op_call_1.args
    negate_output, _negate_input_1, _negate_input_2 = op_call_2.args
    # validate single operator
    validate_single_op([op_call_1, op_call_2], sctx)

    # start compose
    if isinstance(_negate_input_1, FloatImm):
        negate_input = _negate_input_2
    elif isinstance(_negate_input_2, FloatImm):
        negate_input = _negate_input_1
    else:
        raise ValueError(f"The second operator of {op_call_2} is not a negate operator")
    assert_structural_equal(
        reduce_output, negate_input
    ), "Reduce output and negate input must be the same"

    return reduction_trn(op_call_1, optype_table[op_call_1.op], sctx, negate=True)


@register_schedule("compose_op", "trn")
def compose_op_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    op_calls = op.args
    assert len(op_calls) == 2, "Currently only support composing two TIRp op calls"
    assert all(isinstance(op_call, OpCall) for op_call in op_calls), "All arguments must be OpCall"
    op1 = op_calls[0].op
    op2 = op_calls[1].op
    if op1 in vector_ops and op2 in reduce_ops_after_vector:
        return compose_vector_reduce(op_calls[0], op_calls[1], sctx)
    if op1 in act_ops and op2 in reduce_ops_after_act:
        return compose_act_reduce(op_calls[0], op_calls[1], sctx)
    if op1 in vector_ops and op2 in vector_ops:
        return compose_vector_chain(op_calls[0], op_calls[1], sctx)
    if op1 in reduce_ops and is_negate_op(op_calls[1]):
        return compose_reduce_negate(op_calls[0], op_calls[1], sctx)
    raise ValueError(f"No composition rule for {op1} and {op2}")
