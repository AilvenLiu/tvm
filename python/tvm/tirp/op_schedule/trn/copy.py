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

"""Implementation of copy operator schedules."""

from typing import Optional
import operator
from functools import reduce

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
    get_hardware_inst_size_limit,
    bound_inst_with_limit,
    init_analyzer,
    check_partition_dim_match,
    find_inst_transpose,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    get_max_psum_elements
)


def transpose_schedule(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    analyzer: Analyzer,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    
    assert not src_buffer_region.buffer.scope() == "trn.psum", "Transpose on psum buffer is not supported"
    
    first_inst_stride, first_inst_data_iters, second_inst_stride, second_inst_data_iters = find_inst_transpose(
        dst_buffer_region, src_buffer_region, analyzer
    )
    f_gen_axes_dst = generate_axes_in_region(
        dst_buffer_region, first_inst_stride, first_inst_data_iters, analyzer
    )
    f_gen_axes_src = generate_axes_in_region(
        src_buffer_region, second_inst_stride, second_inst_data_iters, analyzer
    )
    f_gen_dst_idx = f_gen_idx_anchor(dst_buffer_region, f_gen_axes_dst)
    f_gen_src_idx = f_gen_idx_anchor(src_buffer_region, f_gen_axes_src)

    p_size = src_buffer_region.buffer.layout.partition_size
    lhs_f_size = dst_buffer_region.buffer.layout.partition_size
    rhs_f_size = p_size
    b_extent = reduce(operator.mul, [r.extent for r in dst_buffer_region.region], 1) // p_size // lhs_f_size
    identity_tensor = T.buffer((p_size, rhs_f_size), src_buffer_region.buffer.dtype, scope="trn.sbuf")
    sctx.add_alloc_buffer(identity_tensor)
    @T.prim_func(tirp=True)
    def identity_init():
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop, rhs_f_loop in T.grid(p_size, rhs_f_size):
                T.evaluate(T.nki_identity(identity_tensor[p_loop, rhs_f_loop], p_size))
    sctx.add_init_stmt(identity_init.body)
    
    dst_buffer = dst_buffer_region.buffer
    src_buffer = src_buffer_region.buffer
    #fmt: off
    
    max_rhs_f_size = 128
    max_psum_slots = get_max_psum_elements() // max_rhs_f_size
    
    @T.macro
    def matmul_inst_macro(b_loop, psum_slot, dst, use_dst_indices):
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop, lhs_f_loop, rhs_f_loop in T.grid(p_size, lhs_f_size, rhs_f_size):
                src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent),), lhs_f_loop, p_loop))
                if use_dst_indices:
                    dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), rhs_f_loop, lhs_f_loop))
                    T.evaluate(T.nki_matmul(dst[*dst_indices], src_buffer[*src_indices], identity_tensor[p_loop, rhs_f_loop]))
                else:
                    T.evaluate(T.nki_matmul(dst[psum_slot, lhs_f_loop, rhs_f_loop], src_buffer[*src_indices], identity_tensor[p_loop, rhs_f_loop]))
    
    @T.prim_func(tirp=True)
    def transpose_psum_output():
        for b_loop in T.serial(0, b_extent):
            matmul_inst_macro(b_loop, None, dst_buffer, True)
            
    @T.prim_func(tirp=True)
    def transpose_sbuf_output():
        with T.kernel():
            dst_psum_shape = T.meta_var((max_psum_slots, p_size, max_rhs_f_size))
            dst_psum = T.alloc_buffer(
                (max_psum_slots, p_size, max_rhs_f_size),
                "float32",
                logical_scope="trn.psum",
                layout=T.TrainiumPSUMLayout(
                    "FPF",
                    T.TileLayout.from_tuple(dst_psum_shape, (max_rhs_f_size, 1, 1)),
                ),
            )
            for b_loop in T.serial(0, b_extent):
                psum_slot = T.meta_var(b_loop % max_psum_slots)
                matmul_inst_macro(b_loop, psum_slot, dst_psum, False)
                
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop, f_loop in T.grid(lhs_f_size, rhs_f_size):
                        dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                        T.evaluate(T.nki_tensor_copy(dst_buffer[*dst_indices], dst_psum[psum_slot, p_loop, f_loop]))
                            
    if dst_buffer.scope() == "trn.psum":
        return transpose_psum_output
    return transpose_sbuf_output

    #fmt: on
@register_schedule("copy")
def copy_trn(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    src, dst = src_buffer_region.buffer, dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.scope() in ["global", "trn.sbuf", "trn.psum"],
            dst.scope() in ["global", "trn.sbuf", "trn.psum"],
            src.scope() != "global" or dst.scope() != "global",
            (src.scope() == "global" and isinstance(src.layout, T.TileLayout))
            or (
                src.scope() in ["trn.sbuf", "trn.psum"] and isinstance(src.layout, T.TrainiumLayout)
            ),
            (dst.scope() == "global" and isinstance(dst.layout, T.TileLayout))
            or (
                dst.scope() in ["trn.sbuf", "trn.psum"] and isinstance(dst.layout, T.TrainiumLayout)
            ),
        ]
    ):
        raise ValueError("Invalid buffer layout/scope for copy operation.")

    # Extract regions and validate dimensions
    analyzer = init_analyzer(sctx)
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    if not check_partition_dim_match(src_buffer_region, dst_buffer_region, analyzer):
        return transpose_schedule(dst_buffer_region, src_buffer_region, analyzer, sctx)
    inst_size = None
    inst_stride = None
    inst_data_iters = None
    src_to_dst = None
    if isinstance(src.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            src_buffer_region, dst_buffer_region, analyzer
        )
        src_to_dst = True
    if inst_size is None and isinstance(dst.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            dst_buffer_region, src_buffer_region, analyzer
        )
        src_to_dst = False
    assert src_to_dst is not None
    if src_to_dst:
        p_size = src_buffer_region.buffer.layout.partition_size
        f_gen_axes = generate_axes_in_region(
            src_buffer_region, inst_stride, inst_data_iters, analyzer
        )
        dim_map = get_ewise_dim_map(src_buffer_region, dst_buffer_region, analyzer)
        f_gen_src_idx = f_gen_idx_anchor(src_buffer_region, f_gen_axes)
        f_gen_dst_idx = f_gen_idx_mapped(dst_buffer_region, f_gen_axes, dim_map)
    else:
        p_size = dst_buffer_region.buffer.layout.partition_size
        f_gen_axes = generate_axes_in_region(
            dst_buffer_region, inst_stride, inst_data_iters, analyzer
        )
        dim_map = get_ewise_dim_map(dst_buffer_region, src_buffer_region, analyzer)
        f_gen_dst_idx = f_gen_idx_anchor(dst_buffer_region, f_gen_axes)
        f_gen_src_idx = f_gen_idx_mapped(src_buffer_region, f_gen_axes, dim_map)

    b_extent = reduce(operator.mul, src_extent, 1) // p_size // inst_size
    # fmt: off
    if src.scope() == "global":
        func = T.nki_load
    elif dst.scope() == "global":
        func = T.nki_store
    else:
        func = T.nki_tensor_copy
    
    inst_size_limit = get_hardware_inst_size_limit(func!=T.nki_tensor_copy)
    actual_inst_size, additional_b_size = bound_inst_with_limit(inst_size, inst_size_limit, analyzer)
    
    @T.prim_func(tirp=True)
    def impl():
        # the additional b loop is to satisfy hardware instuction size limit
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size):
                    for f_loop in T.serial(0, actual_inst_size):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        src_indices = T.meta_var(f_gen_src_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                        dst_indices = T.meta_var(f_gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                        func(dst[*dst_indices], src[*src_indices])
    # fmt: on

    return impl
