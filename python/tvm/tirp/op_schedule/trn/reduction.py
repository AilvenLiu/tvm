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
from tvm.tirp.op_schedule import ScheduleContext

from .common import find_max_inst_size_from_one_region, generate_axes_in_region
from ..registry import register_schedule
from ..common import _make_schedule, ReduceOpType


reduce_ops = {
    ReduceOpType.SUM: "add",
    ReduceOpType.MAX: "max",
    ReduceOpType.MIN: "min",
}


def reduction_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    axes: Tuple[int],
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None
    assert not accum, "Accumulation is not supported for reduction on Trainium"
    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    assert reduce_op in reduce_ops, f"Unsupported reduce operation {reduce_op}"

    dst = dst_buffer_region.buffer
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    dst_non_unit_extent_ = [(i, e) for i, e in enumerate(dst_extent) if e != 1]
    src = src_buffer_region.buffer
    src_region = src_buffer_region.region
    src_extent = [r.extent for r in src_region]
    src_non_unit_extent_ = [(i, e) for i, e in enumerate(src_extent) if e != 1]
    axes = [i if i >= 0 else len(src_non_unit_extent_) + i for i in axes]
    src_non_reduction_extents = [(i, e) for i, e in src_non_unit_extent_ if i not in axes]
    assert len(src_non_reduction_extents) == len(
        dst_non_unit_extent_
    ), "Source and destination must have the same number of non-reduction extents"
    for i in range(len(src_non_reduction_extents)):
        assert (
            src_non_reduction_extents[i][1] == dst_non_unit_extent_[i][1]
        ), "Source and destination must have the same extent for non-reduction axes"
    dim_map = {s[0]: d[0] for s, d in zip(src_non_reduction_extents, dst_non_unit_extent_)}
    # layout checks
    assert all(
        [
            src.layout and dst.layout,
            src.scope() == "trn.sbuf" or src.scope() == "trn.psum",
            dst.scope() == "trn.sbuf",
            isinstance(src.layout, T.TrainiumLayout),
            isinstance(dst.layout, T.TrainiumLayout),
            src.layout.partition_size == dst.layout.partition_size,
        ]
    ), "Invalid layout"

    # reduction axes must be f dim
    inst_size, inst_stride, inst_data_iters = find_max_inst_size_from_one_region(
        src_buffer_region, axes, analyzer
    )
    reduce_size = reduce(operator.mul, [src_buffer_region.region[i].extent for i in axes], 1)

    # TODO: split into 2 stages if cannot find an instruction that covers the whole reduction axes
    assert analyzer.can_prove(
        reduce_size == inst_size
    ), "Cannot find an instruction that covers the whole reduction axes"
    f_gen_axes = generate_axes_in_region(src_buffer_region, inst_stride, inst_data_iters, analyzer)

    def f_gen_src_idx(b_loop, b_extent, f_loop, p_loop):
        region_axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        return [src_buffer_region.region[i].min + region_axes[i] for i in range(len(region_axes))]

    def f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop):
        indices = [dst_buffer_region.region[i].min for i in range(len(dst_buffer_region.region))]
        region_axes = f_gen_axes(((b_loop, b_extent),), f_loop, p_loop)
        for i, j in dim_map.items():
            indices[j] += region_axes[i]
        return indices

    p_size = dst.layout.partition_size
    b_extent = reduce(operator.mul, [e for i, e in dst_non_unit_extent_], 1) // p_size
    opcode = reduce_ops[reduce_op]
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, f_loop in T.grid(p_size, inst_size):
                    src_indices = T.meta_var(f_gen_src_idx(b_loop, b_extent, f_loop, p_loop))
                    dst_indices = T.meta_var(f_gen_dst_idx(b_loop, b_extent, f_loop, p_loop))
                    T.evaluate(T.nki_tensorreduce(dst[dst_indices], src[src_indices], opcode, -1))
    # fmt: on
    return impl


# Register unary mapping schedules.
for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:
    custom_name = f"reduction_{op_name}_trn_impl"
    func = _make_schedule(op_type, 3, [reduction_trn])
    func.__name__ = custom_name
    func.__doc__ = f"Schedule reduction {op_name}."
    register_schedule(op_name)(func)
