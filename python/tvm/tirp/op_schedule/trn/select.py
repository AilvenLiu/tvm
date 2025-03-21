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

"""Implementation of select schedules."""

from functools import reduce
from typing import Optional
import operator

from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, OpCall, FloatImm
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from tvm.tirp.operator.op import Select
from .common import (
    init_analyzer,
    find_max_inst_size_unary,
    infer_range_info,
    generate_axes_in_region,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    get_ewise_dim_map,
    bound_inst_with_limit,
    bound_buffer_region,
    make_guard,
    nki_dim,
)


@register_schedule("select", "trn")
def select_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Generate schedule for select operation on Trainium."""
    if sctx.exec_scope.name != "kernel":
        return None

    op = OpCall.downcast(op)
    assert isinstance(op, Select), f"{op} is not a Select"

    # Unpack operands
    dst, true_value, false_value = *op.dsts, *op.srcs
    pred = op.predicate

    # Check that one of the sources is a float immediate
    assert isinstance(true_value, FloatImm) or isinstance(
        false_value, FloatImm
    ), f"{op} expects one of the source to be a float"

    # Ensure true_value is the buffer and false_value is the float immediate
    if isinstance(true_value, FloatImm):
        pred = not pred
        true_value, false_value = false_value, true_value

    assert isinstance(true_value, BufferRegion), f"{op} expects one of the source to be a buffer"

    # Initialize analyzer and validate buffers
    analyzer = init_analyzer(sctx)

    # Validate buffer layout and scope
    buffer_conditions = [
        dst.buffer.layout and true_value.buffer.layout,
        dst.buffer.scope() == "trn.sbuf" and true_value.buffer.scope() == "trn.sbuf",
        isinstance(true_value.buffer.layout, T.TrainiumLayout),
        isinstance(dst.buffer.layout, T.TrainiumLayout),
    ]

    if not all(buffer_conditions):
        assert False, f"scope or layout mismatch, {dst} vs {true_value}"

    # Extract regions and validate dimensions
    dst_extent = [r.extent for r in dst.region]
    dst_extent_non_unit = [e for e in dst_extent if e != 1]
    true_value_extent = [r.extent for r in true_value.region]
    true_value_extent_non_unit = [e for e in true_value_extent if e != 1]

    # Validate non-unit dimensions match
    dims_match = len(true_value_extent_non_unit) == len(dst_extent_non_unit) and all(
        analyzer.can_prove_equal(s, d)
        for s, d in zip(true_value_extent_non_unit, dst_extent_non_unit)
    )

    if not dims_match:
        assert False, f"shape or dimension mismatch, {dst} vs {true_value}"

    # Bound buffer regions and find instruction size
    bound_dst = bound_buffer_region(dst, analyzer)
    bound_true_value = bound_buffer_region(true_value, analyzer)
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
        bound_dst, bound_true_value, analyzer
    )

    # affine_select only takes linear expression, so we can only extract instruction from one data iter
    # TODO: other operators should also restrict free dim if free dimensions have guard
    _, layout, _ = infer_range_info(dst, analyzer)
    data_iter_array = layout.combined_1d_layout.data_iter_array

    # Find iteration with minimum stride
    min_stride_iter = min(
        ((i, data_iter_array[i].stride) for i in inst_data_iters), key=lambda tup: tup[1]
    )[0]

    # Update instruction size and data iterators
    inst_size = inst_data_iters[min_stride_iter]
    inst_data_iters = {min_stride_iter: inst_data_iters[min_stride_iter]}

    # Generate index functions
    f_gen_axes = generate_axes_in_region(bound_dst, inst_stride, inst_data_iters, analyzer)
    f_gen_dst_idx = f_gen_idx_anchor(dst, f_gen_axes)
    f_gen_true_value_idx = f_gen_idx_mapped(
        true_value, f_gen_axes, get_ewise_dim_map(dst, true_value, analyzer)
    )

    # Calculate loop bounds
    p_size = dst.buffer.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // inst_size // p_size

    # Handle instruction size limit
    inst_size_limit = op.schedule_config.get("max_inst_size", None)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )

    # Get buffer references and guard function
    dst_buffer = dst.buffer
    true_value_buffer = true_value.buffer
    f_dst_guard = make_guard(dst, analyzer)

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, actual_inst_size, annotations={nki_dim: "F"}):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                            dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            true_value_indices = T.meta_var(f_gen_true_value_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            pred = T.meta_var(analyzer.simplify(op.predicate.apply(f_gen_axes(((b_loop, b_extent),),f_loop_wo_limit, p_loop))))
                            T.evaluate(T.nki_affine_select(dst_buffer[*dst_indices], pred, true_value_buffer[*true_value_indices], false_value))
    # fmt: on

    return impl
