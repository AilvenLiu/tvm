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
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from tvm.ir import Op
from tvm.tirp.operator.op import BinaryReduce, UnaryReduce, BinaryChain, ReduceNegate
from .common import (
    get_ewise_dim_map,
    init_analyzer,
    get_reduction_dim_map,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
    bound_buffer_region,
    make_guard,
)
from .unary import try_find_inst_unary
from .binary import try_find_inst_binary, InstType, try_find_inst_nary
from .reduction import generate_intermediate_buffer, reduction_trn
from ..common import ReduceOpType
from ..registry import f_op_scheduler


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


@register_schedule("binary_reduce", "trn")
def binary_reduce_trn(
    op: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    op = OpCall.downcast(op)
    assert isinstance(op, BinaryReduce), f"invalid operator downcast: {op}"
    binary_output, reduce_output = op.dsts
    binary_input1, binary_input2 = op.srcs
    reduce_axes = op.reduce_axes
    analyzer = init_analyzer(sctx)
    reduce_axes = [i if i >= 0 else len(binary_output.buffer.shape) + i for i in reduce_axes]
    bound_binary_output = bound_buffer_region(binary_output, analyzer)
    bound_binary_input1 = bound_buffer_region(binary_input1, analyzer)
    bound_binary_input2 = bound_buffer_region(binary_input2, analyzer)
    bound_reduce_output = bound_buffer_region(reduce_output, analyzer)
    inst_size, f_gen_axes, inst_type, reverse, broadcast_dims = try_find_inst_binary(
        bound_binary_output,
        bound_binary_input1,
        bound_binary_input2,
        analyzer,
        allowed_f_dim_dst=reduce_axes,
        allow_tensortensor=False,
    )
    assert inst_size is not None, f"Failed to find a valid instruction: {op}"
    assert (
        inst_type == InstType.TENSOR_SCALAR
    ), f"TensorTensor is not supported for vector reduce: {op}"
    assert analyzer.can_prove(inst_size > 1), f"Instruction size must be greater than 1: {op}"
    if reverse:
        binary_input1, binary_input2 = binary_input2, binary_input1
        bound_binary_input1, bound_binary_input2 = bound_binary_input2, bound_binary_input1
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(
        bound_reduce_output, bound_binary_output, reduce_axes, inst_size, analyzer
    )
    f_gen_vec_dst_idx = f_gen_idx_anchor(binary_output, f_gen_axes)
    vec_dst_to_src1_dim_map = get_ewise_dim_map(binary_output, binary_input1, analyzer)
    f_gen_src1_idx = f_gen_idx_mapped(binary_input1, f_gen_axes, vec_dst_to_src1_dim_map)
    if isinstance(binary_input2, BufferRegion):
        src1_src2_offset = len(binary_input1.region) - len(binary_input2.region)
        vec_dst_to_src2_dim_map = {
            d: s - src1_src2_offset
            for d, s in vec_dst_to_src1_dim_map.items()
            if s not in broadcast_dims
        }
        f_gen_src2_idx = f_gen_idx_mapped(binary_input2, f_gen_axes, vec_dst_to_src2_dim_map)
        CONST = None
    else:
        CONST = binary_input2
    f_gen_reduce_dst_idx = f_gen_idx_mapped(
        reduce_output,
        f_gen_axes,
        get_reduction_dim_map(binary_output, reduce_output, reduce_axes, analyzer),
    )
    p_size = binary_output.buffer.layout.partition_size
    b_extent = (
        reduce(operator.mul, [r.extent for r in bound_binary_output.region], 1)
        // p_size
        // inst_size
    )
    src1, src2, dst1, dst2 = (
        binary_input1.buffer,
        binary_input2.buffer if isinstance(binary_input2, BufferRegion) else None,
        binary_output.buffer,
        reduce_output.buffer,
    )
    binary_opcode, reduce_opcode = opcode_table[op.binary_op], opcode_table[op.reduce_op]
    f_dst_guard = make_guard(binary_output, analyzer)
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
                        if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)):
                            if CONST is None:
                                src_2_indices = T.meta_var(f_gen_src2_idx(((b_loop, b_extent),), f_loop, p_loop))
                                T.nki_tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], binary_opcode, reduce_opcode, reverse)
                            else:
                                T.nki_tensorscalar_reduce(dst2[*reduce_dst_idx], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, binary_opcode, reduce_opcode, reverse)
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
                                if f_dst_guard(f_gen_axes(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var)):
                                    src_1_indices = T.meta_var(f_gen_src1_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                    vec_dst_idx = T.meta_var(f_gen_vec_dst_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                    if CONST is None:
                                        src_2_indices = T.meta_var(f_gen_src2_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                        T.nki_tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], src2[*src_2_indices], binary_opcode, reduce_opcode, reverse)
                                    else:
                                        T.nki_tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*vec_dst_idx], src1[*src_1_indices], CONST, binary_opcode, reduce_opcode, reverse)
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            if f_dst_guard(f_gen_axes(((b_loop, b_extent), (f_loop, reduction_b_extent)), 0, p_loop, dim2block_var)):
                                dst_2_indices = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                T.nki_tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1)
        # fmt: on
        return impl


@register_schedule("unary_reduce", "trn")
def unary_reduce_trn(
    op: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    op = OpCall.downcast(op)
    assert isinstance(op, UnaryReduce), f"invalid operator downcast: {op}"
    unary_output, reduce_output = op.dsts
    unary_input, bias, scale = op.srcs
    analyzer = init_analyzer(sctx)
    reduce_axes = [i if i >= 0 else len(unary_output.buffer.shape) + i for i in op.reduce_axes]
    scale = 1.0 if scale is None else scale
    bound_unary_output = bound_buffer_region(unary_output, analyzer)
    bound_unary_input = bound_buffer_region(unary_input, analyzer)
    bound_bias = bound_buffer_region(bias, analyzer)
    bound_reduce_output = bound_buffer_region(reduce_output, analyzer)
    if bias is not None:
        inst_size, f_gen_axes, inst_type, reverse, broadcast_dims = try_find_inst_binary(
            bound_unary_output,
            bound_unary_input,
            bound_bias,
            analyzer,
            allow_tensortensor=False,
            allow_reverse=False,
            allowed_f_dim_dst=reduce_axes,
        )
    else:
        inst_size, f_gen_axes = try_find_inst_unary(
            bound_unary_output, bound_unary_input, analyzer, allowed_f_dim_dst=reduce_axes
        )
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(
        bound_reduce_output, bound_unary_output, reduce_axes, inst_size, analyzer
    )
    f_gen_act_dst_idx = f_gen_idx_anchor(unary_output, f_gen_axes)
    dst_to_src_dim_map = get_ewise_dim_map(unary_output, unary_input, analyzer)
    f_gen_act_src_idx = f_gen_idx_mapped(unary_input, f_gen_axes, dst_to_src_dim_map)
    f_gen_reduce_dst_idx = f_gen_idx_mapped(
        reduce_output,
        f_gen_axes,
        get_reduction_dim_map(unary_output, reduce_output, reduce_axes, analyzer),
    )
    if bias is not None and isinstance(bias, BufferRegion):
        offset = len(unary_input.region) - len(bias.region)
        dst_to_bias_dim_map = {
            d: s - offset for d, s in dst_to_src_dim_map.items() if s not in broadcast_dims
        }
        f_gen_bias_idx = f_gen_idx_mapped(bias, f_gen_axes, dst_to_bias_dim_map)
    else:
        f_gen_bias_idx = None
    p_size = unary_output.buffer.layout.partition_size
    b_extent = (
        reduce(operator.mul, [r.extent for r in bound_unary_output.region], 1)
        // p_size
        // inst_size
    )
    src, dst1, dst2 = unary_input.buffer, unary_output.buffer, reduce_output.buffer
    unary_opcode, reduce_opcode = opcode_table[op.unary_op], opcode_table[op.reduce_op]
    bias_buffer = bias.buffer if isinstance(bias, BufferRegion) else None
    f_dst_guard = make_guard(unary_output, analyzer)
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
                        if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)):
                            if bias is None:
                                T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode))
                            elif isinstance(bias, BufferRegion):
                                src_bias_indices = T.meta_var(f_gen_bias_idx(((b_loop, b_extent),), f_loop, p_loop))
                                T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                            else:
                                T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias, scale))
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
                                if f_dst_guard(f_gen_axes(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var)):
                                    if bias is None:
                                        T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode))
                                    elif isinstance(bias, BufferRegion):
                                        src_bias_indices = T.meta_var(f_gen_bias_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                        T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias_buffer[*src_bias_indices], scale))
                                    else:
                                        T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], unary_opcode, reduce_opcode, bias, scale))
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            if f_dst_guard(f_gen_axes(((b_loop, b_extent), (f_loop, reduction_b_extent)), 0, p_loop, dim2block_var)):
                                dst_2_indices = T.meta_var(f_gen_reduce_dst_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                T.evaluate(T.nki_tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1))
        # fmt: on
        return impl


@register_schedule("binary_chain", "trn")
def binary_chain_trn(
    op: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    op = OpCall.downcast(op)
    assert isinstance(op, BinaryChain), f"invalid operator downcast: {op}"
    output = op.dsts[0]
    srcs = op.srcs
    reverse = [False, op.reverse1]
    analyzer = init_analyzer(sctx)
    bound_output = bound_buffer_region(output, analyzer)
    bound_srcs = [bound_buffer_region(src, analyzer) for src in srcs]
    inst_size, f_gen_axes, inst_types, _reverse, broadcast_dims = try_find_inst_nary(
        bound_output, bound_srcs, analyzer, allow_first_op_tensortensor=False
    )
    assert inst_size is not None, "Failed to find a valid instruction"
    assert (
        inst_types[0] == InstType.TENSOR_SCALAR
    ), "The first operator must be a tensor scalar operator"
    reverse[0] = _reverse[0]
    if reverse[0]:
        srcs[0], srcs[1] = srcs[1], srcs[0]
        bound_srcs[0], bound_srcs[1] = bound_srcs[1], bound_srcs[0]
    f_gen_dst_idx = f_gen_idx_anchor(output, f_gen_axes)
    dst_to_src0_dim_map = get_ewise_dim_map(output, srcs[0], analyzer)
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
    p_size = output.buffer.layout.partition_size
    b_extent = (
        reduce(operator.mul, [r.extent for r in bound_output.region], 1) // p_size // inst_size
    )
    src, dst = srcs[0].buffer, output.buffer
    opcode0, opcode1 = opcode_table[op.op0], opcode_table[op.op1]
    func = (
        T.nki_scalar_tensor_scalar
        if inst_types[1] == InstType.TENSOR_SCALAR
        else T.nki_scalar_tensor_tensor
    )
    inst_size_limit = get_hardware_inst_size_limit(is_dma=False)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )
    f_dst_guard = make_guard(output, analyzer)

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
                    if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                        T.evaluate(func(dst[*dst_indices], *srcs, opcode0, opcode1, reverse[0], reverse[1]))
    # fmt: on
    return impl


@register_schedule("reduce_negate", "trn")
def reduce_negate_trn(
    op: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    op = OpCall.downcast(op)
    assert isinstance(op, ReduceNegate), f"invalid operator downcast: {op}"
    return reduction_trn(op, optype_table[op.reduce_op], sctx, negate=True)


@register_schedule("compose_op", "trn")
def compose_op_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    raise NotImplementedError(
        "Generic compose_op must be lowered to specific compose ops before operator-level passes"
    )
