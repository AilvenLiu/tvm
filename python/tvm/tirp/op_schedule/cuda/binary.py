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

"""Implementation of binary operator schedules."""

from typing import Optional, Union
import functools
import operator

from tvm.error import InternalError
from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext
from tvm.tir import BufferRegion, PrimFunc, OpCall
from tvm.tir.expr import FloatImm
from tvm.arith.analyzer import Analyzer

from ..common import MapOpType
from .common import get_indices


binary_op_table = {
    MapOpType.ADD: lambda a, b: a + b,
    MapOpType.SUB: lambda a, b: a - b,
    MapOpType.MUL: lambda a, b: a * b,
    MapOpType.FDIV: lambda a, b: a / b,
}

def get_indices_zero_out(indices, src1_start, src1_extent, src2_start, src2_extent):
    """Compute src2 indices for broadcasting based on src1 indices."""
    len_diff = len(src1_extent) - len(src2_extent)
    return [
        (
            (indices[i + len_diff] - src1_start[i + len_diff]) + src2_start[i]
            if src2_extent[i] != 1
            else src2_start[i]
        )
        for i in range(len(src2_extent))
    ]


def binary_map_cuda_shared_nd_sync_cta_impl(
    op: OpCall,
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """
    Schedule binary map operation on CUDA in shared memory.

    For commutative ops (ADD, MUL), at most one of _src1 and _src2 can be a FloatImm,
    and if both are buffers, numpy-style broadcasting is supported.
    For non-commutative ops (SUB, FDIV), only _src2 can be a FloatImm and if both are buffers,
    only broadcasting of _src2 to _src1 is supported.
    """
    _dst: BufferRegion = op.args[0]
    _src1: Union[BufferRegion, FloatImm] = op.args[1]
    _src2: Union[BufferRegion, FloatImm] = op.args[2]

    if sctx.exec_scope.name != "cta":
        return None

    CONST = None
    # Ensure at least one source is not a constant.
    if isinstance(_src1, FloatImm) and isinstance(_src2, FloatImm):
        return None
    # If src1 is constant, swap (only allowed for ADD and MUL).
    if isinstance(_src1, FloatImm):
        if binary_op not in (MapOpType.ADD, MapOpType.MUL):
            return None
        _src1, _src2 = _src2, _src1
    if isinstance(_src2, FloatImm):
        CONST = _src2

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = None if CONST is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if CONST is not None else _src2.region
    dtype = dst.dtype

    dst_start = [r.min for r in dst_region]
    src1_start = [r.min for r in src1_region]
    src1_extent = [r.extent for r in src1_region]
    dst_extent = [r.extent for r in dst_region]
    if src2_region is not None:
        src2_start = [r.min for r in src2_region]
        src2_extent = [r.extent for r in src2_region]
    else:
        src2_start = src2_extent = None

    # Check layout, scope, and dtype.
    if not (
        dst.layout
        and src1.layout
        and (src2.layout if src2 else True)
        and dst.layout.is_trivial()
        and src1.layout.is_trivial()
        and (src2.layout.is_trivial() if src2 else True)
        and dst.scope().startswith("shared")
        and src1.scope().startswith("shared")
        and (src2.scope().startswith("shared") if src2 else True)
        and src1.dtype == dtype
        and ((src2.dtype == dtype) if src2 else (CONST.dtype == dtype))
    ):
        return None

    analyzer = Analyzer()
    num_elements = functools.reduce(operator.mul, dst_extent, 1)

    # For non-constant second source, switch broadcasting if needed.
    if CONST is None:
        src2_num = functools.reduce(operator.mul, src2_extent, 1)
        if num_elements < src2_num:
            if binary_op not in (MapOpType.ADD, MapOpType.MUL):
                return None
            # Swap src1 and src2.
            _src1, _src2 = _src2, _src1
            src1, src2 = src2, src1
            src1_region, src2_region = src2_region, src1_region
            src1_start, src2_start = src2_start, src1_start
            src1_extent, src2_extent = src2_extent, src1_extent

    # Check that non-singleton dimensions of dst and src1 match.
    dst_non1 = [e for e in dst_extent if e != 1]
    src1_non1 = [e for e in src1_extent if e != 1]
    if not (
        len(dst_non1) == len(src1_non1)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src1_non1, dst_non1))
    ):
        return None

    # For buffer src2, ensure it is broadcastable to src1.
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src2_extent[-i] not in (1, src1_extent[-i]):
                return None

    thread_cnt = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    op_func = binary_op_table.get(binary_op)
    if op_func is None:
        return None

    if CONST is not None:

        @T.prim_func(tirp=True)
        def impl():
            for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
                for itr in T.serial(T.ceildiv(num_elements, thread_cnt)):
                    fused_idx = T.meta_var(itr * thread_cnt + tid_x)
                    if fused_idx < num_elements:
                        idx_dst = T.meta_var(get_indices(fused_idx, dst_start, dst_extent))
                        idx_src1 = T.meta_var(get_indices(fused_idx, src1_start, src1_extent))
                        dst[*idx_dst] = op_func(src1[*idx_src1], CONST)
            T.tvm_storage_sync("shared")

        return impl

    @T.prim_func(tirp=True)
    def impl():  # pylint: disable=function-redefined
        for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
            for itr in T.serial(T.ceildiv(num_elements, thread_cnt)):
                fused_idx = T.meta_var(itr * thread_cnt + tid_x)
                if fused_idx < num_elements:
                    idx_dst = T.meta_var(get_indices(fused_idx, dst_start, dst_extent))
                    idx_src1 = T.meta_var(get_indices(fused_idx, src1_start, src1_extent))
                    idx_src2 = T.meta_var(
                        get_indices_zero_out(
                            idx_src1, src1_start, src1_extent, src2_start, src2_extent
                        )
                    )
                    dst[*idx_dst] = op_func(src1[*idx_src1], src2[*idx_src2])
        T.tvm_storage_sync("shared")

    return impl


def binary_map_cuda_warp_logical_view_nd_impl(
    op: OpCall,
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """
    Schedule binary map operation on CUDA on warp-level logical tensor.

    Note: for now, is_tile_inner only support warp-level buffer view verification.
    Warpgroup-level buffer view and cta-level buffer view need sharding on the
    outermost level, thus not supported to be checked and verified at the moment.
    User should pass in warp-level buffer view for src and dst whenever possible.

    Since broadcast checking for arbitrary layout is complicated, for now,
    we only support two kinds of layouts: WGMMA and ROW_RED.
    ROW_RED layout is the row-reduced form of WGMMA, which is:

                            T0 T1 T2 T3
                            T4 T5 T6 T7
                                ...
                            T28 T29 T30 T31
                            T0 T1 T2 T3
                            T4 T5 T6 T7
                                ...
                            T28 T29 T30 T31

    More layouts can be supported in the future.
    """

    _dst: BufferRegion = op.args[0]
    _src1: Union[BufferRegion, FloatImm] = op.args[1]
    _src2: Union[BufferRegion, FloatImm] = op.args[2]

    # Ensure at least one source is not a constant.
    CONST = None
    if isinstance(_src1, FloatImm) and isinstance(_src2, FloatImm):
        return None
    # If src1 is constant, swap (only allowed for ADD and MUL).
    if isinstance(_src1, FloatImm):
        if binary_op not in (MapOpType.ADD, MapOpType.MUL):
            return None
        _src1, _src2 = _src2, _src1
    if isinstance(_src2, FloatImm):
        CONST = _src2

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = None if CONST is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if CONST is not None else _src2.region
    dtype = dst.dtype

    dst_start = [r.min for r in dst_region]
    src1_start = [r.min for r in src1_region]
    src1_extent = [r.extent for r in src1_region]
    dst_extent = [r.extent for r in dst_region]
    if src2_region is not None:
        src2_start = [r.min for r in src2_region]
        src2_extent = [r.extent for r in src2_region]
    else:
        src2_start = src2_extent = None

    # basic validation checks
    if not all(
        [
            src1.scope() == "local",
            src2.scope() == "local" if src2 else True,
            dst.scope() == "local",
            src1.layout is not None,
            src2.layout is not None if src2 else True,
            dst.layout is not None,
            src1.logical_scope() == "warp",
            src2.logical_scope() == "warp" if src2 else True,
            dst.logical_scope() == "warp",
            src1.dtype == dtype,
            src2.dtype == dtype if src2 else CONST.dtype == dtype,
            sctx.is_cuda(),
            sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"],
        ]
    ):
        return None

    # get binary op
    op_func = binary_op_table.get(binary_op)
    if op_func is None:
        return None

    # no slicing allowed, since op is on local tensor
    analyzer = Analyzer()
    if not all(
        [
            len(src1_region) == 2 and len(dst_region) == 2,
            len(src2_region) == 2 if src2 else True,
            src1_region[0].min == 0 and src1_region[1].min == 0,
            (src2_region[0].min == 0 and src2_region[1].min == 0) if src2 else True,
            dst_region[0].min == 0 and dst_region[1].min == 0,
            src1_region[0].extent == src1.shape[0] and src1_region[1].extent == src1.shape[1],
            (src2_region[0].extent == src2.shape[0] and src2_region[1].extent == src2.shape[1]) if src2 else True,
            dst_region[0].extent == dst.shape[0] and dst_region[1].extent == dst.shape[1],
        ]
    ):
        return None

    # For non-constant second source, switch broadcasting if needed.
    analyzer = Analyzer()
    if CONST is None:
        src1_num = functools.reduce(operator.mul, src1_extent, 1)
        src2_num = functools.reduce(operator.mul, src2_extent, 1)
        if src1_num < src2_num:
            if binary_op not in (MapOpType.ADD, MapOpType.MUL):
                return None
            # Swap src1 and src2.
            _src1, _src2 = _src2, _src1
            src1, src2 = src2, src1
            src1_region, src2_region = src2_region, src1_region
            src1_start, src2_start = src2_start, src1_start
            src1_extent, src2_extent = src2_extent, src1_extent

    # For buffer src2, ensure it is broadcastable to src1,
    # and non-broadcasting dimensions match.
    BROADCAST = False
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src1_extent[-i] != dst_extent[-i]:
                return None
            if src2_extent[-i] not in (4, src1_extent[-i]):
                return None
            if src2_extent[-i] == 4 and src1_extent[-i] != 4:
                BROADCAST = True

    # basic shape check
    if any(
        [
            len(src1.shape) != 2,
            len(src2.shape) != 2 if src2 else False,
            len(dst.shape) != 2,
            src1.shape[0] != 16,
            not (src1.shape[1] % 8 == 0 or src1.shape[1] == 4),
            src2.shape[0] != 16 if src2 else False,
            not (src2.shape[1] % 8 == 0 or src2.shape[1] == 4) if src2 else False,
            dst.shape[0] != 16,
            not (dst.shape[1] % 8 == 0 or dst.shape[1] == 4),
            src1.layout.is_swizzle(),
            (src2.layout.is_swizzle() if src2 else False),
            dst.layout.is_swizzle(),
        ]
    ):
        return None

    # layout check:
    # (dst, src1, src2) layout must adhere to one of the five cases below:
    # 1. (WGMMA, WGMMA, WGMMA)
    # 2. (WGMMA, WGMMA, ROW_RED)
    # 3. (ROW_RED, ROW_RED, ROW_RED)
    # 4. (WGMMA, WGMMA, const)
    # 5. (ROW_RED, ROW_RED, const)

    # WGMMA layout check
    atom = T.TileLayout.from_tuple((1, 2), (2, 1))
    warp_atom = T.TileLayout.shard(
        (8, 8), (8, 4), "S0S1", inner=atom, from_to=("thread", "warp")
    )
    def check_wgmma(buf):
        try:
            return warp_atom.is_tile_inner(buf.layout, buf.shape, [8, 8])
        except InternalError:
            return None

    # ROW_RED layout check
    red_atom = T.TileLayout.from_tuple(1, 1)
    red_warp_atom = T.TileLayout.shard(
        (32,), (32,), "S0", inner=red_atom, from_to=("thread", "warp")
    )
    def check_row_red(buf):
        try:
            return red_warp_atom.is_tile_inner(buf.layout.normalize(), (64,), (32,))
        except InternalError:
            return None

    if CONST is not None:
        # check for case 4 and 5
        num_rows = 2
        if check_wgmma(dst) and check_wgmma(src1):
            num_cols = check_wgmma(src1).size
        elif check_row_red(dst) and check_row_red(src1):
            num_cols = 1
        else:
            return None

        src1_local_shape = dst_local_shape = (num_rows, num_cols)

        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl_const():
            with T.thread():
                src1_local = T.get(src1, src1_local_shape)
                dst_local = T.get(dst, dst_local_shape)
                for i in T.serial(num_rows):
                    for j in T.serial(num_cols):
                        dst_local[i, j] = op_func(src1_local[i, j], CONST)
        # fmt: on

        return impl_const

    if BROADCAST:
        # check for case 2
        if not (check_wgmma(dst) and check_wgmma(src1) and check_row_red(src2)):
            return None

        num_rows = 2
        src1_local_shape = dst_local_shape = (num_rows, check_wgmma(src1).size)
        src2_local_shape = (num_rows, 1)

        # fmt: off
        @T.prim_func(tirp=True, check_well_formed=False)
        def impl_broadcast():
            with T.thread():
                src1_local = T.get(src1, src1_local_shape)
                src2_local = T.get(src2, src2_local_shape)
                dst_local = T.get(dst, dst_local_shape)
                for i in T.serial(num_rows):
                    for j in T.serial(dst_local_shape[1]):
                        dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, 0])
        # fmt: on

        return impl_broadcast

    # check for case 1 and 3
    num_rows = 2
    if check_wgmma(dst) and check_wgmma(src1) and check_wgmma(src2):
        num_cols = check_wgmma(src1).size
    elif check_row_red(dst) and check_row_red(src1) and check_row_red(src2):
        num_cols = 1
    else:
        return None

    src1_local_shape = src2_local_shape = dst_local_shape = (num_rows, num_cols)

    # fmt: off
    @T.prim_func(tirp=True, check_well_formed=False)
    def impl():
        with T.thread():
            src1_local = T.get(src1, src1_local_shape)
            src2_local = T.get(src2, src2_local_shape)
            dst_local = T.get(dst, dst_local_shape)
            for i in T.serial(num_rows):
                for j in T.serial(num_cols):
                    dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, j])
    # fmt: on

    return impl


def binary_cuda_impl(
    op: OpCall,
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Dispatch to shared memory scheduler or logical tensor of local memory scheduler
    based on the storage scope of buffers.
    """

    dst_buffer_region = op.args[0]
    if dst_buffer_region.buffer.scope().startswith("shared"):
        return binary_map_cuda_shared_nd_sync_cta_impl(op, binary_op, sctx)
    elif dst_buffer_region.buffer.scope() == "local" and dst_buffer_region.buffer.logical_scope() == "warp":
        return binary_map_cuda_warp_logical_view_nd_impl(op, binary_op, sctx)

    return None
