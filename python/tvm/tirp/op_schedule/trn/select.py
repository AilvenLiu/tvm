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
    get_ewise_dim_map,
    InstructionGenerator,
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
    inst_gen = InstructionGenerator([dst, true_value], analyzer)
    dim_map = get_ewise_dim_map(dst, true_value, analyzer)
    inst_gen.link_buffer_regions(dst, true_value, dim_map)
    inst_repr = inst_gen.find_max_inst_size_from_one_region(dst)
    inst_repr = inst_gen.fit_inst_tile_to_region(inst_repr, true_value)
    inst_repr = inst_gen.restrict_inst_to_one_dim(inst_repr)
    inst_repr.bound_inst_size(op.schedule_config.get("max_inst_size", 512), analyzer)

    p_var = T.var("int32", name="p")
    b_var = T.var("int32", name="b")
    f_var = T.var("int32", name="f")
    p_size = dst.buffer.layout.partition_size
    inst_gen.bind_inst_iter(dst, f_var, inst_repr.size, inst_repr.stride, True)
    inst_gen.bind_inst_iter(dst, p_var, p_size, 1, False)
    b_extent = inst_gen.fill_in_block_dim(dst, b_var)
    
    # Get buffer references and guard function
    dst_buffer = dst.buffer
    true_value_buffer = true_value.buffer
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({f_var: f_loop, p_var: p_loop, b_var: b_loop})
                        if inst_gen.make_guard(dst):
                            dst_indices = T.meta_var(inst_gen.generate_indices(dst))
                            true_value_indices = T.meta_var(inst_gen.generate_indices(true_value))
                            pred = T.meta_var(analyzer.simplify(op.predicate.apply(inst_gen.generate_axes(dst))))
                            T.evaluate(T.nki.affine_select(dst_buffer[*dst_indices], pred, true_value_buffer[*true_value_indices], false_value))
    # fmt: on

    return impl
