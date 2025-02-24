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
from tvm.ir import Op, assert_structural_equal

from functools import reduce
from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
    init_analyzer,
    get_reduction_dim_map,
    f_gen_idx_anchor,
    f_gen_idx_mapped
)
from .unary import try_find_inst_unary, unary_trn
from .reduction import generate_intermediate_buffer, reduction_trn
from ..common import MapOpType, ReduceOpType
from ..registry import f_op_scheduler


vector_ops = {Op.get("tirp.add") : (MapOpType.ADD, "add"),
             Op.get("tirp.sub") : (MapOpType.SUB, "sub"),
             Op.get("tirp.mul") : (MapOpType.MUL, "mul")}
act_ops = {Op.get("tirp.sqrt") : (MapOpType.SQRT, "sqrt")}
reduce_ops_after_act = {Op.get("tirp.sum") : (ReduceOpType.SUM, "sum")}
reduce_ops_after_vector = {Op.get("tirp.sum") : (ReduceOpType.SUM, "sum"),
                           Op.get("tirp.max") : (ReduceOpType.MAX, "max"),
                           Op.get("tirp.min") : (ReduceOpType.MIN, "min")}

def compose_vector_reduce(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    raise NotImplementedError("Vector reduce composition is not implemented")

def validate_single_op(op_calls: List[OpCall], sctx: ScheduleContext):
    for op_call in op_calls:
        try:
            f_op_scheduler(op_call.op, op_call.args, sctx)
        except (ValueError, NotImplementedError) as e:
            raise ValueError(f"Invalid operator {op_call.op}, error: {e}")


def compose_act_reduce(
    op_call_1: OpCall,
    op_call_2: OpCall,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:    
    act_output, act_input = op_call_1.args
    reduce_output, reduce_input, reduce_axes, accum = op_call_2.args
    
    # validate single operator
    validate_single_op([op_call_1, op_call_2], sctx)
    
    # start compose
    assert_structural_equal(act_output, reduce_input),  "Act output and reduce input must be the same"
    analyzer = init_analyzer(sctx)
    reduce_axes = [i if i >= 0 else len(reduce_input.buffer.shape) + i for i in reduce_axes]
    inst_size, inst_stride, inst_data_iters, f_gen_axes_1 = try_find_inst_unary(act_output, act_input, analyzer, allowed_f_dim_dst=reduce_axes)
    assert analyzer.can_prove(inst_size > 1), "Instruction size must be greater than 1"
    intermediate_shape, intermediate_layout, reduction_b_extent = generate_intermediate_buffer(reduce_output, reduce_input, reduce_axes, inst_size, analyzer)
    f_gen_dst_1_idx = f_gen_idx_anchor(act_output, f_gen_axes_1)
    f_gen_src_1_idx = f_gen_idx_mapped(act_input, f_gen_axes_1, get_ewise_dim_map(act_output, act_input, analyzer))
    f_gen_dst_2_idx = f_gen_idx_mapped(reduce_output, f_gen_axes_1, get_reduction_dim_map(reduce_input, reduce_output, reduce_axes))
    p_size = act_output.buffer.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in act_output.region], 1) // p_size // inst_size
    src, dst1, dst2 = act_input.buffer, act_output.buffer, reduce_output.buffer
    opcode, reduce_opcode = act_ops[op_call_1.op][1], reduce_ops_after_act[op_call_2.op][1]
    if intermediate_shape is None:
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop, f_loop in T.grid(p_size, inst_size):
                        src_1_indices = T.meta_var(f_gen_src_1_idx(((b_loop, b_extent),), f_loop, p_loop))
                        dst_1_indices = T.meta_var(f_gen_dst_1_idx(((b_loop, b_extent),), f_loop, p_loop))
                        dst_2_indices = T.meta_var(f_gen_dst_2_idx(((b_loop, b_extent),), f_loop, p_loop))
                        T.evaluate(T.nki_activation_reduce(dst2[*dst_2_indices], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode))
        # fmt: on
        import tvm
        mod = tvm.IRModule({"main": impl})
        mod = tvm.tir.transform.Simplify()(mod)
        return mod["main"]
    else:
        dim2block_var = {dim: 1 if dim in reduce_axes else 0 for dim in range(len(dst1.shape))}
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            with T.kernel():
                intermediate_buffer = T.alloc_buffer(intermediate_shape, dtype=dst2.dtype, layout=intermediate_layout, scope="trn.sbuf")
                for b_loop in T.serial(0, b_extent):
                    for reduction_b_loop in T.serial(0, reduction_b_extent):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, f_loop in T.grid(p_size, inst_size):
                                src_1_indices = T.meta_var(f_gen_src_1_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                dst_1_indices = T.meta_var(f_gen_dst_1_idx(((b_loop, b_extent), (reduction_b_loop, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                                T.evaluate(T.nki_activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[*dst_1_indices], src[*src_1_indices], opcode, reduce_opcode))
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop, f_loop in T.grid(p_size, reduction_b_extent):
                            dst_2_indices = T.meta_var(f_gen_dst_2_idx(((b_loop, b_extent), (0, reduction_b_extent)), f_loop, p_loop, dim2block_var))
                            T.evaluate(T.nki_tensorreduce(dst2[*dst_2_indices], intermediate_buffer[p_loop, f_loop], reduce_opcode, -1))
        # fmt: on
        return impl
    
@register_schedule("compose_op")
def compose_op_trn(
    *op_calls,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    assert len(op_calls) == 2, "Currently only support composing two TIRp op calls"
    assert all(isinstance(op_call, OpCall) for op_call in op_calls), "All arguments must be OpCall"
    op1  = op_calls[0].op
    op2 = op_calls[1].op
    if op1 in vector_ops and op2 in reduce_ops_after_vector:
        return compose_vector_reduce(op_calls[0], op_calls[1], sctx)
    if op1 in act_ops and op2 in reduce_ops_after_act:
        return compose_act_reduce(op_calls[0], op_calls[1], sctx)
    raise ValueError(f"No composition rule for {op1} and {op2}")
