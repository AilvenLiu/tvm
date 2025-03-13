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
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
    bound_buffer_region,
    make_guard,
)


@register_schedule("select", "trn")
def select_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    if sctx.exec_scope.name != "kernel":
        return None
    op = OpCall.downcast(op)
    assert isinstance(op, Select), f"{op} is not a Select"
    dst, true_value, false_value = *op.dsts, *op.srcs
    pred = op.predicate
    assert isinstance(true_value, FloatImm) or isinstance(
        false_value, FloatImm
    ), f"{op} expects one of the source to be a float"
    if isinstance(true_value, FloatImm):
        pred = not pred
        true_value, false_value = false_value, true_value
    assert isinstance(true_value, BufferRegion), f"{op} expects one of the source to be a buffer"
    analyzer = init_analyzer(sctx)
    if not all(
        [
            dst.buffer.layout and true_value.buffer.layout,
            dst.buffer.scope() == "trn.sbuf" and true_value.buffer.scope() == "trn.sbuf",
            isinstance(true_value.buffer.layout, T.TrainiumLayout),
            isinstance(dst.buffer.layout, T.TrainiumLayout),
        ]
    ):
        assert False, f"scope or layout mismatch, {dst} vs {true_value}"
    # Extract regions and validate dimensions
    dst_extent = [r.extent for r in dst.region]
    dst_extent_ = [e for e in dst_extent if e != 1]
    true_value_extent = [r.extent for r in true_value.region]
    true_value_extent_ = [e for e in true_value_extent if e != 1]
    # Validate non-unit dimensions match
    if not (
        len(true_value_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(true_value_extent_, dst_extent_))
    ):
        assert False, f"shape or dimension mismatch, {dst} vs {true_value}"
    bound_dst = bound_buffer_region(dst, analyzer)
    bound_true_value = bound_buffer_region(true_value, analyzer)
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
        bound_dst, bound_true_value, analyzer
    )
    # affine_select only takes linear expression, so we can only extract instruction from one data iter
    _, layout, _ = infer_range_info(dst, analyzer)
    data_iter_array = layout.combined_1d_layout.data_iter_array
    min_stride_iter = min(
        ((i, data_iter_array[i].stride) for i in inst_data_iters), key=lambda tup: tup[1]
    )[0]
    inst_size = inst_data_iters[min_stride_iter]
    inst_data_iters = {min_stride_iter: inst_data_iters[min_stride_iter]}
    f_gen_axes = generate_axes_in_region(bound_dst, inst_stride, inst_data_iters, analyzer)
    f_gen_dst_idx = f_gen_idx_anchor(dst, f_gen_axes)
    f_gen_true_value_idx = f_gen_idx_mapped(
        true_value, f_gen_axes, get_ewise_dim_map(dst, true_value, analyzer)
    )
    p_size = dst.buffer.layout.partition_size
    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // inst_size // p_size
    inst_size_limit = get_hardware_inst_size_limit(is_dma=False)
    actual_inst_size, additional_b_size = bound_inst_with_limit(
        inst_size, inst_size_limit, analyzer
    )
    dst_buffer = dst.buffer
    true_value_buffer = true_value.buffer
    f_dst_guard = make_guard(dst, analyzer)
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, actual_inst_size):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        if f_dst_guard(f_gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                            dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            true_value_indices = T.meta_var(f_gen_true_value_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            pred = T.meta_var(analyzer.simplify(op.predicate.apply(f_gen_axes(((b_loop, b_extent),),f_loop_wo_limit, p_loop))))
                            T.evaluate(T.nki_affine_select(dst_buffer[*dst_indices], pred, true_value_buffer[*true_value_indices], false_value))
    # fmt: on
    return impl
