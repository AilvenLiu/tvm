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
import functools
from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.ir import assert_structural_equal
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from tvm.tir.stmt import OpCall
from .common import (
    infer_range_info,
    generate_axes_in_region,
    find_max_inst_size_from_one_region,
    find_max_inst_size_matmul,
    bound_inst_with_limit,
    init_analyzer,
    get_max_psum_banks,
    f_gen_idx_anchor,
    check_workspace_buffer,
    get_largest_psum_per_bank,
    target_trn,
    bound_buffer_region,
    make_guard,
)


class OperatorKind:
    A = 0
    B = 1
    C = 2


def get_pf_dim_from_buffer_region(
    buffer_region: BufferRegion,
    analyzer: Analyzer,
    operator_kind: OperatorKind,
    transposed: bool = False,
):
    """Extract partition and free dimensions from buffer region."""
    # Find non-unit dimensions
    non_unit_dims = [
        i
        for i in range(len(buffer_region.buffer.shape))
        if not analyzer.can_prove_equal(buffer_region.region[i].extent, 1)
    ]
    assert len(non_unit_dims) == 2, "Only 2D matrix is supported for gemm"

    _, layout, seps = infer_range_info(buffer_region, analyzer)

    # Determine partition and free dimensions based on operator kind
    if operator_kind == OperatorKind.A:
        p_dim, f_dim = non_unit_dims[1], non_unit_dims[0]
    elif operator_kind == OperatorKind.B:
        p_dim, f_dim = non_unit_dims[0], non_unit_dims[1]
    else:
        assert (
            not transposed
        ), "Transposed C is implemented by swapping lhs and rhs. No need to specify by user."
        # For C, determine dimensions based on layout
        has_partition = any(
            layout.dimension_types[i] == T.TrainiumLayout.Partition
            for i in range(seps[non_unit_dims[0]], seps[non_unit_dims[0] + 1])
        )
        p_dim, f_dim = (
            (non_unit_dims[0], non_unit_dims[1])
            if has_partition
            else (non_unit_dims[1], non_unit_dims[0])
        )

    # Swap dimensions if transposed
    if transposed:
        p_dim, f_dim = f_dim, p_dim

    # Validate partition dimension
    p_exts = [
        layout.combined_1d_layout.data_iter_array[i].extent
        for i in range(seps[p_dim], seps[p_dim + 1])
        if layout.dimension_types[i] == T.TrainiumLayout.Partition
    ]

    assert functools.reduce(operator.mul, p_exts, 1) == layout.partition_size, (
        f"Accumulation dimension and output non-streaming dimension must contain whole P dimension. "
        f"However, the {p_dim} dimension of {buffer_region} does not."
    )

    # Validate free dimension
    assert all(
        layout.dimension_types[i] == T.TrainiumLayout.Free
        or layout.combined_1d_layout.data_iter_array[i].extent == 1
        for i in range(seps[f_dim], seps[f_dim + 1])
    ), f"Spatial dimension must not contain P. However, the {f_dim} dimension of {buffer_region} does."

    return p_dim, f_dim


@register_schedule("gemm", "trn")
@target_trn
def matmul_trn(op: OpCall, sctx: ScheduleContext) -> Optional[PrimFunc]:
    """Schedule GEMM operation on Trainium."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    # Extract arguments
    (
        D_buffer_region,
        A_buffer_region,
        B_buffer_region,
        C_buffer_region,
        transpose_A,
        transpose_B,
        alpha,
        beta,
    ) = op.args
    analyzer = init_analyzer(sctx)
    A, B, C, D = (
        A_buffer_region.buffer,
        B_buffer_region.buffer,
        C_buffer_region.buffer,
        D_buffer_region.buffer,
    )

    # Validate alpha, beta
    assert analyzer.can_prove_equal(alpha, 1) and analyzer.can_prove_equal(
        beta, 0
    ), "Only alpha=1 and beta=0 are supported"

    # D and C must be the same buffer region
    assert_structural_equal(D_buffer_region, C_buffer_region)

    # Validate buffer properties
    assert all(
        [
            A.layout and B.layout and C.layout,
            A.dtype == B.dtype,
            A.scope() == "trn.sbuf" and B.scope() == "trn.sbuf",
            C.scope() == "trn.psum" or C.scope() == "trn.sbuf",
            isinstance(A.layout, T.TrainiumLayout),
            isinstance(B.layout, T.TrainiumLayout),
            isinstance(C.layout, T.TrainiumLayout),
            A.layout.partition_size == B.layout.partition_size,
        ]
    ), "Invalid buffer layout and scope"

    p_size = A.layout.partition_size
    assert p_size == B.layout.partition_size, "Partition size mismatch"

    # Bound buffer regions
    bound_A = bound_buffer_region(A_buffer_region, analyzer)
    bound_B = bound_buffer_region(B_buffer_region, analyzer)
    bound_C = bound_buffer_region(C_buffer_region, analyzer)

    # Get partition and free dimensions
    lhs_p_dim, lhs_f_dim = get_pf_dim_from_buffer_region(
        bound_A, analyzer, OperatorKind.A, transpose_A
    )
    rhs_p_dim, rhs_f_dim = get_pf_dim_from_buffer_region(
        bound_B, analyzer, OperatorKind.B, transpose_B
    )
    acc_p_dim, acc_f_dim = get_pf_dim_from_buffer_region(bound_C, analyzer, OperatorKind.C)

    # Swap LHS and RHS if needed based on accumulator dimensions
    swap_lhs_rhs = acc_p_dim > acc_f_dim
    if swap_lhs_rhs:
        lhs_p_dim, rhs_p_dim = rhs_p_dim, lhs_p_dim
        lhs_f_dim, rhs_f_dim = rhs_f_dim, lhs_f_dim
        A, B = B, A
        A_buffer_region, B_buffer_region = B_buffer_region, A_buffer_region
        bound_A, bound_B = bound_B, bound_A

    # Validate dimension compatibility
    assert analyzer.can_prove(
        A_buffer_region.region[lhs_p_dim].extent == B_buffer_region.region[rhs_p_dim].extent
    ), f"Reduction dimension must match, but the {lhs_p_dim} dimension of {A_buffer_region} != the {rhs_p_dim} dimension of {B_buffer_region}"

    assert analyzer.can_prove(
        A_buffer_region.region[lhs_f_dim].extent == C_buffer_region.region[acc_p_dim].extent
    ), f"Spatial dimension must match, but the {lhs_f_dim} dimension of {A_buffer_region} != the {acc_p_dim} dimension of {C_buffer_region}"

    assert analyzer.can_prove(
        B_buffer_region.region[rhs_f_dim].extent == C_buffer_region.region[acc_f_dim].extent
    ), f"Spatial dimension must match, but the {rhs_f_dim} dimension of {B_buffer_region} != the {acc_f_dim} dimension of {C_buffer_region}"

    # Find instruction sizes
    lhs_f_size, lhs_f_stride, lhs_f_data_iters = find_max_inst_size_from_one_region(
        bound_A, analyzer, [lhs_f_dim]
    )

    rhs_f_size, rhs_f_stride, acc_f_stride, rhs_f_data_iters = find_max_inst_size_matmul(
        bound_B,
        bound_C,
        analyzer,
        allowed_f_dim_rhs=[rhs_f_dim],
        allowed_f_dim_output=[acc_f_dim],
        f_dim_map={rhs_f_dim: acc_f_dim},
    )

    if C.scope() == "trn.psum":
        assert acc_f_stride == 1, "psum_f_stride must be 1"

    # Set up dimension mapping and access generation
    dim2block_var_lhs = {lhs_f_dim: 0, lhs_p_dim: 1}
    dim2block_var_rhs = {rhs_f_dim: 0, rhs_p_dim: 1}
    f_gen_lhs_axes = generate_axes_in_region(bound_A, lhs_f_stride, lhs_f_data_iters, analyzer)
    f_gen_rhs_axes = generate_axes_in_region(bound_B, rhs_f_stride, rhs_f_data_iters, analyzer)
    f_gen_lhs_indices = f_gen_idx_anchor(A_buffer_region, f_gen_lhs_axes)
    f_gen_rhs_indices = f_gen_idx_anchor(B_buffer_region, f_gen_rhs_axes)

    def f_gen_acc_indices(
        lhs_b_loop,
        lhs_b_extent,
        rhs_b_loop,
        rhs_b_extent,
        reduction_b_extent,
        lhs_f_loop,
        rhs_f_loop,
    ):
        lhs_axes = f_gen_lhs_axes(
            [(lhs_b_loop, lhs_b_extent), (0, reduction_b_extent)], lhs_f_loop, 0, dim2block_var_lhs
        )
        rhs_axes = f_gen_rhs_axes(
            [(rhs_b_loop, rhs_b_extent), (0, reduction_b_extent)], rhs_f_loop, 0, dim2block_var_rhs
        )

        acc_indices = [C_buffer_region.region[i].min for i in range(len(C_buffer_region.region))]
        acc_indices[acc_p_dim] += lhs_axes[lhs_f_dim]
        acc_indices[acc_f_dim] += rhs_axes[rhs_f_dim]
        return acc_indices

    # Validate extents
    assert analyzer.can_prove(
        bound_A.region[lhs_f_dim].extent % lhs_f_size == 0
    ) and analyzer.can_prove(bound_B.region[rhs_f_dim].extent % rhs_f_size == 0), "Invalid extent"

    # Calculate block extents
    lhs_b_extent = bound_A.region[lhs_f_dim].extent // lhs_f_size
    rhs_b_extent = bound_B.region[rhs_f_dim].extent // rhs_f_size
    reduction_b_extent = bound_A.region[lhs_p_dim].extent // p_size

    # Apply size limits
    max_lhs_size, max_rhs_size = 128, 512
    actual_lhs_f_size, additional_lhs_b_size = bound_inst_with_limit(
        lhs_f_size, max_lhs_size, analyzer
    )
    actual_rhs_f_size, additional_rhs_b_size = bound_inst_with_limit(
        rhs_f_size, max_rhs_size, analyzer
    )

    # Get psum configuration
    max_psum_slots = get_max_psum_banks()
    largest_psum_per_bank = get_largest_psum_per_bank()
    inst_num_per_slot = largest_psum_per_bank // rhs_f_size

    # FIXME: we need to lower the guard to things like matmul(lhs[...][lhs_guard], rhs[...][rhs_guard], mask=p_guard)
    # so we need to separate the guard for lhs_f, rhs_f and p
    f_lhs_guard = make_guard(A_buffer_region, analyzer)
    f_rhs_guard = make_guard(B_buffer_region, analyzer)

    # fmt: off
    @T.macro
    def matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop, b_idx, acc, C_as_output, max_psum_slots):
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in T.serial(0, p_size, annotations={"nki_dim": "P"}):
                for lhs_f_loop in T.serial(0, actual_lhs_f_size, annotations={"nki_dim": "lhs_F"}):
                    for rhs_f_loop in T.serial(0, actual_rhs_f_size, annotations={"nki_dim": "rhs_F"}):
                        lhs_f_loop_wo_limit = T.meta_var(lhs_f_loop + additional_lhs_b_loop * actual_lhs_f_size)
                        rhs_f_loop_wo_limit = T.meta_var(rhs_f_loop + additional_rhs_b_loop * actual_rhs_f_size)
                        lhs_indices = T.meta_var(f_gen_lhs_indices(((lhs_b_loop, lhs_b_extent), (reduction_b_loop, reduction_b_extent)), lhs_f_loop_wo_limit, p_loop, dim2block_var_lhs))
                        rhs_indices = T.meta_var(f_gen_rhs_indices(((rhs_b_loop, rhs_b_extent), (reduction_b_loop, reduction_b_extent)), rhs_f_loop_wo_limit, p_loop, dim2block_var_rhs))
                        C_indices = T.meta_var(f_gen_acc_indices(lhs_b_loop, lhs_b_extent, rhs_b_loop, rhs_b_extent, reduction_b_extent, lhs_f_loop_wo_limit, rhs_f_loop_wo_limit))
                        if f_lhs_guard(f_gen_lhs_axes(((lhs_b_loop, lhs_b_extent), (reduction_b_loop, reduction_b_extent)), lhs_f_loop_wo_limit, p_loop, dim2block_var_lhs)) and \
                            f_rhs_guard(f_gen_rhs_axes(((rhs_b_loop, rhs_b_extent), (reduction_b_loop, reduction_b_extent)), rhs_f_loop_wo_limit, p_loop, dim2block_var_rhs)):
                            if C_as_output:
                                T.evaluate(T.nki_matmul(acc[C_indices], A[lhs_indices], B[rhs_indices]))
                            else:
                                T.evaluate(T.nki_matmul(acc[b_idx // inst_num_per_slot % max_psum_slots, lhs_f_loop, b_idx % inst_num_per_slot * rhs_f_size + rhs_f_loop], A[lhs_indices], B[rhs_indices]))

    if C.scope() == "trn.psum":
        @T.prim_func(tirp=True)
        def impl_C_psum():
            for lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(lhs_b_extent, rhs_b_extent, reduction_b_extent, additional_lhs_b_size, additional_rhs_b_size):
                matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop, None, C, True, None)
        return impl_C_psum

    # todo: generalize the process of generating composite matmul + another_op pattern
    # by generating TIR op and reusing existing dispatch rule
    
    # we will support matmul + epilogue as a user-specified pattern
    # and a matmul fusion pass can help infer the pattern
    
    acc_psum_shape = (max_psum_slots, p_size, largest_psum_per_bank)
    if "acc_psum" not in op.workspace:
        acc_psum = T.buffer(
                acc_psum_shape,
                "float32",
                scope="trn.psum",
                allocated_addr=(0, 0),
                buffer_name="acc_psum"
            )
        sctx.add_alloc_buffer(acc_psum)
    else:
        acc_psum = op.workspace["acc_psum"]
        check_workspace_buffer(acc_psum, (p_size, largest_psum_per_bank), "trn.psum")
        max_psum_slots = acc_psum.shape[0]

    @T.prim_func(tirp=True)
    def impl_C_sbuf():
        for lhs_b_loop, rhs_b_loop, additional_lhs_b_loop, additional_rhs_b_loop in T.grid(lhs_b_extent, rhs_b_extent, additional_lhs_b_size, additional_rhs_b_size):
                b_idx = T.meta_var(lhs_b_loop * additional_lhs_b_size * additional_rhs_b_size * rhs_b_extent + additional_lhs_b_loop * additional_rhs_b_size * rhs_b_extent + rhs_b_loop * additional_rhs_b_size + additional_rhs_b_loop) 
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, additional_lhs_b_loop, additional_rhs_b_loop, b_idx, acc_psum, False, max_psum_slots)
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for lhs_f_loop in T.serial(0, actual_lhs_f_size, annotations={"nki_dim": "P"}):
                        for rhs_f_loop in T.serial(0, actual_rhs_f_size, annotations={"nki_dim": "F"}):
                            lhs_f_loop_wo_limit = T.meta_var(lhs_f_loop + additional_lhs_b_loop * actual_lhs_f_size)
                            rhs_f_loop_wo_limit = T.meta_var(rhs_f_loop + additional_rhs_b_loop * actual_rhs_f_size)
                            if f_lhs_guard(f_gen_lhs_axes([(lhs_b_loop, lhs_b_extent), (0, reduction_b_extent)], lhs_f_loop_wo_limit, 0, dim2block_var_lhs)) and \
                                f_rhs_guard(f_gen_rhs_axes([(rhs_b_loop, rhs_b_extent), (0, reduction_b_extent)], rhs_f_loop_wo_limit, 0, dim2block_var_rhs)):
                                acc_indices = T.meta_var(f_gen_acc_indices(lhs_b_loop, lhs_b_extent, rhs_b_loop, rhs_b_extent, reduction_b_extent, lhs_f_loop_wo_limit, rhs_f_loop_wo_limit))
                                T.evaluate(T.nki_tensor_copy(C[acc_indices], acc_psum[b_idx // inst_num_per_slot % max_psum_slots, lhs_f_loop, b_idx % inst_num_per_slot * rhs_f_size + rhs_f_loop]))
    # fmt: on
    return impl_C_sbuf
