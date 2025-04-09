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
from tvm.script import tirp as Tp
from tvm.tir import PrimFunc
from tvm.tir.stmt import OpCall
from tvm.tirp.op_schedule import ScheduleContext, register_schedule

from .common import (
    generate_axes_in_region,
    get_ewise_dim_map,
    find_max_inst_size_unary,
    bound_inst_with_limit,
    init_analyzer,
    check_partition_dim_match,
    find_max_inst_size_transpose,
    f_gen_idx_anchor,
    f_gen_idx_mapped,
    get_max_psum_banks,
    check_workspace_buffer,
    get_largest_psum_per_bank,
    target_trn,
    bound_buffer_region,
    make_guard,
    nki_dim,
)


def transpose_schedule(
    op: OpCall,
    analyzer: Analyzer,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    dst_region, src_region = op.args
    assert src_region.buffer.scope() != "trn.psum", "Transpose on psum buffer is not supported"

    bound_src = bound_buffer_region(src_region, analyzer)
    bound_dst = bound_buffer_region(dst_region, analyzer)

    inst1_stride, inst1_iters, inst2_stride, inst2_iters = find_max_inst_size_transpose(
        bound_dst, bound_src, analyzer
    )

    gen_axes_dst = generate_axes_in_region(bound_dst, inst1_stride, inst1_iters, analyzer)
    gen_axes_src = generate_axes_in_region(bound_src, inst2_stride, inst2_iters, analyzer)
    gen_dst_idx = f_gen_idx_anchor(dst_region, gen_axes_dst)
    gen_src_idx = f_gen_idx_anchor(src_region, gen_axes_src)

    p_size = src_region.buffer.layout.partition_size
    lhs_f_size = dst_region.buffer.layout.partition_size
    rhs_f_size = p_size
    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // p_size // lhs_f_size

    if "identity" not in op.workspace:
        assert sctx.alloc_only, "Identity tensor must be specified in workspace. Run tvm.tirp.transform.PrivateBufferAlloc first."
        identity_tensor = T.buffer(
            (p_size, rhs_f_size),
            src_region.buffer.dtype,
            scope="trn.sbuf",
            buffer_name="identity",
        )
        sctx.add_alloc_buffer(identity_tensor)

        @T.prim_func(tirp=True)
        def identity_init():
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for rhs_f_loop in T.serial(0, rhs_f_size, annotations={nki_dim: "F"}):
                        T.evaluate(T.nki.identity(identity_tensor[p_loop, rhs_f_loop], p_size))
            Tp.tvm_kernel_replace_point()

        sctx.add_init_stmt(identity_init.body)
    else:
        identity_tensor = op.workspace["identity"]
        check_workspace_buffer(identity_tensor, (p_size, rhs_f_size), "trn.sbuf")

    dst_buffer = dst_region.buffer
    src_buffer = src_region.buffer
    max_psum_slots = get_max_psum_banks()
    largest_psum_per_bank = get_largest_psum_per_bank()
    inst_num_per_slot = largest_psum_per_bank // rhs_f_size
    f_src_guard = make_guard(src_region, analyzer)
    f_dst_guard = make_guard(dst_region, analyzer)

    # fmt: off
    @T.macro
    def matmul_inst_macro(b_loop, dst, use_dst_indices, max_psum_slots):
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                for lhs_f_loop in T.serial(0, lhs_f_size, annotations={nki_dim: "lhs_F"}):
                    for rhs_f_loop in T.serial(0, rhs_f_size, annotations={nki_dim: "rhs_F"}):
                        src_indices = T.meta_var(gen_src_idx(((b_loop, b_extent),), lhs_f_loop, p_loop))
                        src_guard = T.meta_var(f_src_guard(gen_axes_src(((b_loop, b_extent),), lhs_f_loop, p_loop)))
                        dst_guard = T.meta_var(f_dst_guard(gen_axes_dst(((b_loop, b_extent),), rhs_f_loop, lhs_f_loop)))
                        if src_guard and dst_guard:
                            if use_dst_indices:
                                dst_indices = T.meta_var(gen_dst_idx(((b_loop, b_extent),), rhs_f_loop, lhs_f_loop))
                                T.evaluate(T.nki.matmul(dst[*dst_indices], src_buffer[*src_indices], identity_tensor[p_loop, rhs_f_loop]))
                            else:
                                if src_guard:
                                    T.evaluate(T.nki.matmul(dst[b_loop // inst_num_per_slot % max_psum_slots, lhs_f_loop, b_loop % inst_num_per_slot * rhs_f_size + rhs_f_loop], src_buffer[*src_indices], identity_tensor[p_loop, rhs_f_loop]))
    # fmt: on

    if dst_buffer.scope() == "trn.psum":

        @T.prim_func(tirp=True)
        def transpose_psum_output():
            for b_loop in T.serial(0, b_extent):
                matmul_inst_macro(b_loop, dst_buffer, True, max_psum_slots)

        return transpose_psum_output

    if "acc_psum" not in op.workspace:
        assert sctx.alloc_only, "Accumulation psum buffer must be specified in workspace. Run tvm.tirp.transform.PrivateBufferAlloc first."
        acc_psum = T.buffer(
            (max_psum_slots, p_size, largest_psum_per_bank),
            "float32",
            scope="trn.psum",
            allocated_addr=(0, 0),
            buffer_name="acc_psum",
        )
        sctx.add_alloc_buffer(acc_psum)
    else:
        acc_psum = op.workspace["acc_psum"]
        check_workspace_buffer(acc_psum, (p_size, largest_psum_per_bank), "trn.psum")
        max_psum_slots = acc_psum.shape[0]

    # fmt: off
    @T.prim_func(tirp=True)
    def transpose_sbuf_output():
        with T.kernel():
            for b_loop in T.serial(0, b_extent):
                matmul_inst_macro(b_loop, acc_psum, False, max_psum_slots)
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, rhs_f_size, annotations={nki_dim: "F"}):
                            dst_guard = T.meta_var(f_dst_guard(gen_axes_dst(((b_loop, b_extent),), f_loop, p_loop)))
                            dst_indices = T.meta_var(gen_dst_idx(((b_loop, b_extent),), f_loop, p_loop))
                            if dst_guard:
                                T.evaluate(T.nki.tensor_copy(dst_buffer[*dst_indices], acc_psum[b_loop // inst_num_per_slot % max_psum_slots, p_loop, b_loop % inst_num_per_slot * rhs_f_size + f_loop]))
    # fmt: on

    return transpose_sbuf_output


@register_schedule("copy", "trn")
@target_trn
def copy_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if sctx.exec_scope.name != "kernel":
        return None

    dst_region, src_region = op.args
    src, dst = src_region.buffer, dst_region.buffer

    # Check for valid buffer configurations
    valid_config = all(
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
    )

    if not valid_config:
        raise ValueError("Invalid buffer layout/scope for copy operation.")

    analyzer = init_analyzer(sctx)
    src_extent = [r.extent for r in src_region.region]
    dst_extent = [r.extent for r in dst_region.region]

    # Validate non-unit dimensions match
    src_non_unit = [e for e in src_extent if e != 1]
    dst_non_unit = [e for e in dst_extent if e != 1]
    dims_match = len(src_non_unit) == len(dst_non_unit) and all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_non_unit, dst_non_unit)
    )

    if not dims_match:
        return None

    bound_src = bound_buffer_region(src_region, analyzer)
    bound_dst = bound_buffer_region(dst_region, analyzer)
    dim_map = get_ewise_dim_map(src_region, dst_region, analyzer)

    if not check_partition_dim_match(bound_src, bound_dst, dim_map, analyzer):
        return transpose_schedule(op, analyzer, sctx)

    inst_size, inst_stride, inst_data_iters = None, None, None
    src_to_dst = None

    if isinstance(src.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            bound_src, bound_dst, analyzer
        )
        src_to_dst = True

    if inst_size is None and isinstance(dst.layout, T.TrainiumLayout):
        inst_size, inst_stride, inst_data_iters = find_max_inst_size_unary(
            bound_dst, bound_src, analyzer
        )
        src_to_dst = False

    assert src_to_dst is not None

    if src_to_dst:
        p_size = src_region.buffer.layout.partition_size
        gen_axes = generate_axes_in_region(bound_src, inst_stride, inst_data_iters, analyzer)
        dim_map = get_ewise_dim_map(src_region, dst_region, analyzer)
        gen_src_idx = f_gen_idx_anchor(src_region, gen_axes)
        gen_dst_idx = f_gen_idx_mapped(dst_region, gen_axes, dim_map)
    else:
        p_size = dst_region.buffer.layout.partition_size
        gen_axes = generate_axes_in_region(bound_dst, inst_stride, inst_data_iters, analyzer)
        dim_map = get_ewise_dim_map(dst_region, src_region, analyzer)
        gen_dst_idx = f_gen_idx_anchor(dst_region, gen_axes)
        gen_src_idx = f_gen_idx_mapped(src_region, gen_axes, dim_map)

    b_extent = reduce(operator.mul, [r.extent for r in bound_dst.region], 1) // p_size // inst_size

    # fmt: off
    if src.scope() == "global":
        func = T.nki.load
    elif dst.scope() == "global":
        func = T.nki.store
    else:
        func = T.nki.tensor_copy
    
    if func == T.nki.tensor_copy:
        inst_size_limit = op.schedule_config.get("max_inst_size", 512)
        actual_inst_size, additional_b_size = bound_inst_with_limit(inst_size, inst_size_limit, analyzer)
    else:
        assert "max_inst_size" not in op.schedule_config, "max_inst_size is not supported for load/store"
        actual_inst_size, additional_b_size = inst_size, 1
        
    f_guard = make_guard(dst_region, analyzer)

    @T.prim_func(tirp=True)
    def impl():
        # the additional b loop is to satisfy hardware instuction size limit
        for b_loop, additional_b_loop in T.grid(b_extent, additional_b_size):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, actual_inst_size, annotations={nki_dim: "F"}):
                        f_loop_wo_limit = T.meta_var(f_loop + additional_b_loop * actual_inst_size)
                        if f_guard(gen_axes(((b_loop, b_extent),), f_loop_wo_limit, p_loop)):
                            src_indices = T.meta_var(gen_src_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            dst_indices = T.meta_var(gen_dst_idx(((b_loop, b_extent),), f_loop_wo_limit, p_loop))
                            func(dst[*dst_indices], src[*src_indices])
    # fmt: on
    return impl
