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

from typing import Optional, Set, List, Dict
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc, PrimExpr, Var
from tvm.ir import Range, assert_structural_equal
from tvm.tirp.op_schedule import ScheduleContext, register_schedule
from .common import (
    get_layout_data_iters,
    infer_range_info,
    generate_axes_in_region,
)

max_inst_size = 128
max_psum_banks = 8 * 4


class OperatorKind:
    A = 0
    B = 1
    C = 2


def get_pf_dim_from_buffer_region(
    buffer_region: BufferRegion, analyzer: Analyzer, operator_kind: OperatorKind
):
    non_unit_dims = []
    for i in range(len(buffer_region.buffer.shape)):
        r = buffer_region.region[i]
        if not analyzer.can_prove_equal(r.extent, 1):
            non_unit_dims.append(i)
    assert len(non_unit_dims) == 2, "Only 2D matrix is supported for gemm"
    _, layout, seps = infer_range_info(buffer_region, analyzer)
    if operator_kind == OperatorKind.A:
        p_dim = non_unit_dims[1]
        f_dim = non_unit_dims[0]
    elif operator_kind == OperatorKind.B:
        p_dim = non_unit_dims[0]
        f_dim = non_unit_dims[1]
    else:
        if any(
            layout.dimension_types[i] == T.TrainiumLayout.Partition
            for i in range(seps[non_unit_dims[0]], seps[non_unit_dims[0] + 1])
        ):
            p_dim = non_unit_dims[0]
            f_dim = non_unit_dims[1]
        else:
            # need swap lhs and rhs
            p_dim = non_unit_dims[1]
            f_dim = non_unit_dims[0]
    return p_dim, f_dim


def get_inst_size(
    buffer_region: BufferRegion,
    partition_size: PrimExpr,
    analyzer: Analyzer,
    operator_kind: OperatorKind,
):
    p_dim, f_dim = get_pf_dim_from_buffer_region(buffer_region, analyzer, operator_kind)
    tiled_range_infos_per_dim, layout, seps = infer_range_info(buffer_region, analyzer)
    # check p_dim covers whole partition
    prod = 1
    data_iters = get_layout_data_iters(layout)
    for i in range(seps[p_dim], seps[p_dim + 1]):
        if layout.dimension_types[i] == T.TrainiumLayout.Partition:
            prod *= data_iters[i].extent
    assert analyzer.can_prove_equal(
        prod, partition_size
    ), "Reduction dimension must cover whole partition"

    # check largest inst size
    inst_size = 1
    inst_stride = 1
    inst_data_iters = {}
    while len(tiled_range_infos_per_dim) > 0:
        range_info = tiled_range_infos_per_dim[0]
        tiled_range_infos_per_dim = tiled_range_infos_per_dim[1:]
        st, ext, dim_in_data_iter, dim_in_shape, dim_type = range_info
        if dim_type == T.TrainiumLayout.Partition:
            continue
        if dim_in_shape != f_dim:
            break
        if inst_size != 1 and not analyzer.can_prove(
            inst_stride * inst_size == data_iters[dim_in_data_iter].stride
        ):
            # the stride of the found data iter is not compatible with previous data iters
            break
        if inst_size == 1:
            inst_stride = data_iters[dim_in_data_iter].stride
        if operator_kind != OperatorKind.C and analyzer.can_prove(inst_size * ext > max_inst_size):
            if analyzer.can_prove(max_inst_size % inst_size == 0) and analyzer.can_prove(
                ext % (max_inst_size // inst_size) == 0
            ):
                inst_data_iters[dim_in_data_iter] = max_inst_size // inst_size
                inst_size *= max_inst_size // inst_size
            break
        inst_data_iters[dim_in_data_iter] = ext
        inst_size *= ext

    return inst_size, inst_stride, inst_data_iters, p_dim, f_dim


@register_schedule("gemm")
def matmul_trn(
    D_buffer_region: BufferRegion,
    A_buffer_region: BufferRegion,
    B_buffer_region: BufferRegion,
    C_buffer_region: BufferRegion,
    alpha: PrimExpr,
    beta: PrimExpr,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.exec_scope.name == "kernel"):
        return None

    analyzer = Analyzer()
    for v, r in sctx.var_range_map.items():
        analyzer.bind(v, r)
    A, B, C, D = (
        A_buffer_region.buffer,
        B_buffer_region.buffer,
        C_buffer_region.buffer,
        D_buffer_region.buffer,
    )
    assert analyzer.can_prove_equal(alpha, 1) and analyzer.can_prove_equal(
        beta, 0
    ), "Only alpha=1 and beta=0 are supported"
    # D and C must be the same buffer region
    assert_structural_equal(D_buffer_region, C_buffer_region)
    assert all(
        [
            A.layout and B.layout and C.layout,
            A.dtype == B.dtype,
            A.scope() == "trn.sbuf" and B.scope() == "trn.sbuf",
            C.scope() == "trn.psum" or C.scope() == "trn.sbuf",
            isinstance(A.layout, T.TrainiumLayout),
            isinstance(B.layout, T.TrainiumLayout),
            isinstance(C.layout, T.TrainiumLayout),
        ]
    ), "Invalid buffer layout and scope"

    p_size = A.layout.partition_size
    assert p_size == B.layout.partition_size, "Partition size mismatch"

    lhs_f_size, lhs_f_stride, lhs_f_data_iters, lhs_p_dim, lhs_f_dim = get_inst_size(
        A_buffer_region, p_size, analyzer, OperatorKind.A
    )
    rhs_f_size, rhs_f_stride, rhs_f_data_iters, rhs_p_dim, rhs_f_dim = get_inst_size(
        B_buffer_region, p_size, analyzer, OperatorKind.B
    )
    acc_p_dim, acc_f_dim = get_pf_dim_from_buffer_region(C_buffer_region, analyzer, OperatorKind.C)
    swap_lhs_rhs = acc_p_dim > acc_f_dim

    if C.scope() == "trn.psum":
        acc_f_size, acc_f_stride, _, _, _ = get_inst_size(
            C_buffer_region, p_size, analyzer, OperatorKind.C
        )
        assert acc_f_stride == 1, "psum_f_stride must be 1"
        assert acc_f_size >= rhs_f_size
        assert lhs_f_size == C.layout.partition_size

    dim2block_var_lhs = {lhs_f_dim: 0, lhs_p_dim: 1}
    dim2block_var_rhs = {rhs_f_dim: 0, rhs_p_dim: 1}
    f_gen_lhs_axes = generate_axes_in_region(
        A_buffer_region, lhs_f_stride, lhs_f_data_iters, analyzer, dim2block_var_lhs
    )
    f_gen_rhs_axes = generate_axes_in_region(
        B_buffer_region, rhs_f_stride, rhs_f_data_iters, analyzer, dim2block_var_rhs
    )

    def f_gen_lhs_indices(
        lhs_b_loop, lhs_b_extent, reduction_b_loop, reduction_b_extent, f_loop, p_loop
    ):
        lhs_axes = f_gen_lhs_axes(
            [(lhs_b_loop, lhs_b_extent), (reduction_b_loop, reduction_b_extent)], f_loop, p_loop
        )
        return [A_buffer_region.region[i].min + lhs_axes[i] for i in range(len(lhs_axes))]

    def f_gen_rhs_indices(
        rhs_b_loop, lhs_b_extent, reduction_b_loop, reduction_b_extent, f_loop, p_loop
    ):
        rhs_axes = f_gen_rhs_axes(
            [(rhs_b_loop, lhs_b_extent), (reduction_b_loop, reduction_b_extent)], f_loop, p_loop
        )
        return [B_buffer_region.region[i].min + rhs_axes[i] for i in range(len(rhs_axes))]

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
            [(lhs_b_loop, lhs_b_extent), (0, reduction_b_extent)], lhs_f_loop, 0
        )
        rhs_axes = f_gen_rhs_axes(
            [(rhs_b_loop, rhs_b_extent), (0, reduction_b_extent)], rhs_f_loop, 0
        )

        acc_indices = [C_buffer_region.region[i].min for i in range(len(C_buffer_region.region))]
        if swap_lhs_rhs:
            acc_indices[acc_p_dim] += rhs_axes[rhs_f_dim]
            acc_indices[acc_f_dim] += lhs_axes[lhs_f_dim]
        else:
            acc_indices[acc_p_dim] += lhs_axes[lhs_f_dim]
            acc_indices[acc_f_dim] += rhs_axes[rhs_f_dim]
        return acc_indices

    assert analyzer.can_prove(
        A_buffer_region.region[lhs_f_dim].extent % lhs_f_size == 0
    ) and analyzer.can_prove(
        B_buffer_region.region[rhs_f_dim].extent % rhs_f_size == 0
    ), "Invalid extent"
    lhs_b_extent = A_buffer_region.region[lhs_f_dim].extent // lhs_f_size
    rhs_b_extent = B_buffer_region.region[rhs_f_dim].extent // rhs_f_size
    reduction_b_extent = A_buffer_region.region[lhs_p_dim].extent // p_size
    # fmt: off
    @T.prim_func(tirp=True)
    def impl_C_psum():
        for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(lhs_b_extent, rhs_b_extent, reduction_b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop, lhs_f_loop, rhs_f_loop in T.grid(p_size, lhs_f_size, rhs_f_size):
                    lhs_indices = T.meta_var(f_gen_lhs_indices(lhs_b_loop, lhs_b_extent, reduction_b_loop, reduction_b_extent, lhs_f_loop, p_loop))
                    rhs_indices = T.meta_var(f_gen_rhs_indices(rhs_b_loop, rhs_b_extent,  reduction_b_loop, reduction_b_extent,rhs_f_loop, p_loop))
                    acc_indices = T.meta_var(f_gen_acc_indices(lhs_b_loop, lhs_b_extent, rhs_b_loop, rhs_b_extent, reduction_b_extent, lhs_f_loop, rhs_f_loop))
                    if swap_lhs_rhs:
                        T.evaluate(
                            T.nki_matmul(
                                res=C[acc_indices],
                                lhs=B[rhs_indices],
                                rhs=A[lhs_indices],
                                accum=True,
                            )
                        )
                    else:
                        T.evaluate(
                            T.nki_matmul(
                                res=C[acc_indices],
                                lhs=A[lhs_indices],
                                rhs=B[rhs_indices],
                                accum=True,
                            )
                        )

    # todo: generalize the process of generating composite matmul + another_op pattern
    # by generating TIR op and reusing existing dispatch rule
    @T.prim_func(tirp=True)
    def impl_C_sbuf():
        with T.kernel():
            dimension_types = T.meta_var("FFP" if swap_lhs_rhs else "FPF")
            C_psum_shape = T.meta_var((max_psum_banks, max_inst_size, p_size) if swap_lhs_rhs else (max_psum_banks, p_size, max_inst_size))
            C_psum = T.alloc_buffer(
                C_psum_shape,
                "float32",
                logical_scope="trn.psum",
                layout=T.TrainiumPSUMLayout(
                    dimension_types,
                    T.TileLayout.from_tuple(C_psum_shape, (max_inst_size, 1, 1)),
                ),
            )
            for lhs_b_loop, rhs_b_loop in T.grid(lhs_b_extent, rhs_b_extent):
                    psum_bank = T.meta_var((lhs_b_loop * rhs_b_extent + rhs_b_loop)%max_psum_banks) 
                    for reduction_b_loop in T.serial(0, reduction_b_extent):
                        with T.attr(0, "tensorized_nki_instruction", 1):
                            for p_loop, lhs_f_loop, rhs_f_loop in T.grid(p_size, lhs_f_size, rhs_f_size):
                                lhs_indices = T.meta_var(f_gen_lhs_indices(lhs_b_loop, lhs_b_extent, reduction_b_loop, reduction_b_extent, lhs_f_loop, p_loop))
                                rhs_indices = T.meta_var(f_gen_rhs_indices(rhs_b_loop, rhs_b_extent,  reduction_b_loop, reduction_b_extent,rhs_f_loop, p_loop))
                                if swap_lhs_rhs:
                                    T.evaluate(
                                        T.nki_matmul(
                                            res=C_psum[psum_bank, lhs_f_loop, rhs_f_loop],
                                            lhs=B[rhs_indices],
                                            rhs=A[lhs_indices],
                                            accum=True,
                                        )
                                    )
                                else:
                                    T.evaluate(
                                        T.nki_matmul(
                                            res=C_psum[psum_bank, lhs_f_loop, rhs_f_loop],
                                            lhs=A[lhs_indices],
                                            rhs=B[rhs_indices],
                                            accum=True,
                                        )
                                            )
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for lhs_f_loop, rhs_f_loop in T.grid(lhs_f_size, rhs_f_size):
                            acc_indices = T.meta_var(f_gen_acc_indices(lhs_b_loop, lhs_b_extent, rhs_b_loop, rhs_b_extent, reduction_b_extent, lhs_f_loop, rhs_f_loop))
                            C[acc_indices] = C_psum[psum_bank, lhs_f_loop, rhs_f_loop]
    # fmt: on
    if C.scope() == "trn.psum":
        return impl_C_psum
    return impl_C_sbuf
