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

import functools
import operator
import copy

import tvm
from tvm.script import tir as T
from tvm.tir import PrimFunc
from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.tir.stmt import OpCall
from tvm.tir.layout import TileLayout
from tvm.tirp.op_schedule import ScheduleContext, register_dispatch, predicate
from .common import single_thread, validate_gemm_op, get_st_extent
from .copy_async import SwizzleMode, tma_atom_layout, tma_atom_shape, tma_atom_compatible


def gemm_async_tcgen05_impl(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    """Schedule an asynchronous gemm operation using tcgen05.mma"""

    C_buffer_region, A_buffer_region, B_buffer_region = op_call.args[:3]
    C_buffer, A_buffer, B_buffer = (
        C_buffer_region.buffer,
        A_buffer_region.buffer,
        B_buffer_region.buffer,
    )
    C_scope, A_scope, B_scope = C_buffer.scope(), A_buffer.scope(), B_buffer.scope()
    if not (C_scope == "tmem" and A_scope.startswith("shared") and B_scope.startswith("shared")):
        raise ValueError(
            f"tcgen05 schedule expected C_scope=tmem, A_scope=shared, B_scope=shared, got C_scope={C_scope}, A_scope={A_scope}, B_scope={B_scope}"
        )

    analyzer = Analyzer()

    C_type, A_type, B_type = C_buffer.dtype, A_buffer.dtype, B_buffer.dtype
    assert C_type == "float32", f"tcgen05 schedule expected C_type=float32, got {C_type}"
    assert A_type == "float16", f"tcgen05 schedule expected A_type=float16, got {A_type}"
    assert B_type == "float16", f"tcgen05 schedule expected B_type=float16, got {B_type}"
    C_elem_size = DataType(C_type).bits
    C_elem_per_32b = 32 // C_elem_size

    C_st, C_extent = get_st_extent(C_buffer_region)
    A_st, A_extent = get_st_extent(A_buffer_region)
    B_st, B_extent = get_st_extent(B_buffer_region)

    transA, transB, accum = op_call.args[3:6]
    assert (
        len(C_extent) == 2 and len(A_extent) >= 2 and len(B_extent) >= 2
    ), "Only 2D C, A, B are supported for gemm"
    for buf_name, extent in [("C", C_extent), ("A", A_extent), ("B", B_extent)]:
        assert all(
            analyzer.can_prove_equal(ext, 1) for ext in extent[:-2]
        ), f"tcgen05 schedule expected {buf_name}_extent to be 1 before the last two dimensions"

    M = C_extent[-2]
    N = C_extent[-1]
    K = A_extent[-2] if transA else A_extent[-1]
    cta_group = op_call.config.get("cta_group", 1)
    descI = op_call.config.get("descI", None)

    assert M == 128, f"tcgen05 schedule expected M=128, got {M}"
    assert analyzer.can_prove(N >= 16), f"tcgen05 schedule expected N >= 16, got {N}"
    assert analyzer.can_prove(N <= 256), f"tcgen05 schedule expected N <= 256, got {N}"
    assert analyzer.can_prove_equal(
        tvm.tir.floormod(N, 16), 0
    ), f"tcgen05 schedule expected N % 16 == 0, got {N}"
    assert analyzer.can_prove_equal(
        tvm.tir.floormod(K, 16), 0
    ), f"tcgen05 schedule expected K % 16 == 0, got {K}"

    MMA_K = 16

    # Check C's region [0:128, st:st+MMA_N] and layout (128, NCOLS):(1@TLane, 1@TCol)
    assert analyzer.can_prove_equal(C_buffer.shape[0], 128)
    tmem_layout = TileLayout(([128, C_buffer.shape[1]], [(1, "TLane"), (1, "TCol")])).normalize()
    tvm.ir.assert_structural_equal(C_buffer.layout.normalize(), tmem_layout)
    assert analyzer.can_prove_equal(C_st[0], 0)
    assert analyzer.can_prove_equal(C_extent[0], 128)
    tmem_offset = C_st[1]
    assert analyzer.can_prove_equal(tvm.tir.floormod(tmem_offset, C_elem_per_32b), 0)
    tmem_addr = C_buffer.allocated_addr[0]
    tmem_offset_32b = tvm.tir.floordiv(tmem_offset, C_elem_per_32b)

    # Check A and B's region and layouts
    def swizzle_check(buffer, st, extent, shape, dtype):
        for mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            swizzle_atom = tma_atom_layout(dtype, mode)
            atom_shape = tma_atom_shape(dtype, mode, shape)
            atom_size = functools.reduce(operator.mul, atom_shape, 1)
            tiler = swizzle_atom.is_tile_inner(buffer.layout, shape, atom_shape)
            if tiler is not None and tma_atom_compatible(shape, st, extent, atom_shape):
                tiler_shape = [s // a for s, a in zip(shape, atom_shape)]
                tiler_grouped, seps = tiler.normalize().group_by_shape(tiler_shape)
                assert seps[-3] == seps[-1] - 2
                assert seps[-2] == seps[-1] - 1
                ldo = (tiler_grouped.shard[-1].stride * atom_size) // (
                    128 // tvm.DataType(dtype).bits
                )
                sdo = (tiler_grouped.shard[-2].stride * atom_size) // (
                    128 // tvm.DataType(dtype).bits
                )
                return mode, ldo, sdo

    A_swizzle_mode, A_ldo, A_sdo = swizzle_check(A_buffer, A_st, A_extent, A_buffer.shape, A_type)
    B_swizzle_mode, B_ldo, B_sdo = swizzle_check(B_buffer, B_st, B_extent, B_buffer.shape, B_type)

    def get_st_coord(st, delta, trans):
        st_ki = copy.copy(st)
        if trans:
            st_ki[-2] += delta
        else:
            st_ki[-1] += delta
        return st_ki

    assert C_buffer.allocated_addr is not None
    tmem_addr = C_buffer.allocated_addr[0]

    # fmt: off
    @T.macro
    def main_impl(descA_in, descB_in, descI_in):
        for ki in T.serial(tvm.tir.floordiv(K, MMA_K)):
            A_st_ki = T.meta_var(get_st_coord(A_st, ki * MMA_K, transA))
            B_st_ki = T.meta_var(get_st_coord(B_st, ki * MMA_K, transB))
            T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA_in), A_buffer.ptr_to(A_st_ki), ldo=A_ldo, sdo=A_sdo, swizzle=A_swizzle_mode.value)
            T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB_in), B_buffer.ptr_to(B_st_ki), ldo=B_ldo, sdo=B_sdo, swizzle=B_swizzle_mode.value)
            if ki == 0 and not accum:
                T.ptx.tcgen05.mma("float32", A_type, B_type, T.cuda.get_tmem_addr(tmem_addr, 0, tmem_offset_32b), 
                                  descA_in, descB_in, descI_in, False, cta_group, False)
            else:
                T.ptx.tcgen05.mma("float32", A_type, B_type, T.cuda.get_tmem_addr(tmem_addr, 0, tmem_offset_32b), 
                                  descA_in, descB_in, descI_in, False, cta_group, True)
    # fmt: on

    if descI is not None:
        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl():
            descA = T.local_cell("uint64")
            descB = T.local_cell("uint64")

            main_impl(descA, descB, descI)
        # fmt: on
    else:
        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl():
            descA = T.local_cell("uint64")
            descB = T.local_cell("uint64")
            descI_local = T.local_cell("uint32")

            T.ptx.tcgen05.encode_instr_descriptor(T.address_of(descI_local), C_type, A_type, B_type, 
                                                  M * cta_group, N * cta_group, MMA_K, transA, transB, cta_group)
            main_impl(descA, descB, descI_local)
        # fmt: on

    return impl


@register_dispatch(
    "gemm_async",
    "cuda",
    variant="tcgen05",
    priority=10,
    when=[
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
        predicate(
            "validate_gemm_op", lambda op, sctx: (validate_gemm_op(op, sctx), "not a valid gemm op")
        ),
    ],
)
def gemm_async_dispatch_tcgen05(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    return gemm_async_tcgen05_impl(op_call, sctx)
