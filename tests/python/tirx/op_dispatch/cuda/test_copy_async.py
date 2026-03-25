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
# pylint: disable=invalid-name, missing-function-docstring
import functools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.ir.type import TensorMapType
from tvm.script import tirx as Tx
from tvm.tir import IntImm, StringImm, Var
from tvm.tir.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tir.layout import tid_in_wg as axis_tid_in_wg
from tvm.tir.stmt import DeclBuffer, OpCall
from tvm.tir.stmt_functor import StmtExprVisitor
from tvm.tirx.op_dispatch.cuda.common import next_power_of_2
from tvm.tirx.op_dispatch.cuda.tma_utils import (
    SwizzleMode,
    tma_atom_layout,
    tma_atom_shape,
    tma_shared_layout,
)

# ===========================================================================
# Helpers
# ===========================================================================


class TMACounter(StmtExprVisitor):
    """Visitor to count total TMA operations including loop iterations.

    This verifies that TMA copy operations are optimized correctly,
    resulting in minimal TMA instructions instead of multiple iterations.
    """

    def __init__(self):
        super().__init__()
        self.loop_extents = []  # Stack of loop extents
        self.total_tma_ops = 0

    def visit_for_(self, op):
        extent = op.extent
        self.loop_extents.append(extent)
        self.visit_stmt(op.body)
        self.loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.tir.Call):
            if op.value.op.name in (
                "tir.ptx_cp_async_bulk_tensor_global_to_cluster",
                "tir.ptx_cp_async_bulk_tensor_shared_to_global",
                "tir.ptx_cp_async_bulk_tensor_shared_to_global_reduce",
            ):
                # Multiply all enclosing loop extents
                iters = 1
                for ext in self.loop_extents:
                    iters *= ext
                self.total_tma_ops += iters


def _make_tma_call(
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype="float16",
    direction="g2s",
):
    """Construct OpCall + DispatchContext and call copy_tma_impl.

    Returns (impl, host_init_stmts) on success, raises DispatchFail on failure.
    impl is the device-side PrimFunc, host_init_stmts is a list of Stmt
    for host-side tensor map creation.
    """
    from tvm.ir import Range
    from tvm.tir import Var
    from tvm.tir.exec_scope import ExecScope
    from tvm.tir.stmt import BufferRegion
    from tvm.tirx.op_dispatch.cuda.copy_async import copy_tma_impl
    from tvm.tirx.op_dispatch.dispatch_context import DispatchContext
    from tvm.tirx.operator.op import CopyAsync

    g_buf = tvm.tir.decl_buffer(g_shape, dtype, "A", layout=gmem_layout)
    s_buf = tvm.tir.decl_buffer(s_shape, dtype, "A_smem", scope="shared.dyn", layout=smem_layout)

    g_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in g_region]
    s_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in s_region]

    config = {}
    if direction == "g2s":
        mbar_ptr = Var("mbar_ptr", "handle")
        config["mbar"] = mbar_ptr
        config["cta_group"] = 1
        dst_br = BufferRegion(s_buf, s_ranges)
        src_br = BufferRegion(g_buf, g_ranges)
    else:  # s2g
        config["cta_group"] = 1
        dst_br = BufferRegion(g_buf, g_ranges)
        src_br = BufferRegion(s_buf, s_ranges)

    op_call = CopyAsync(dst_br, src_br, config=config)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope.create("thread"), {}, {})

    impl = copy_tma_impl(op_call, sctx)
    host_init_stmts = list(sctx.callbacks.get("host_init_stmt", []))
    return impl, host_init_stmts


def _count_tma_ops(impl):
    """Count total TMA ops in a PrimFunc (including loop multiplier)."""
    counter = TMACounter()
    counter.visit_stmt(impl.body)
    return counter.total_tma_ops


def _build_expected_host_init(dtype, encode_args):
    """Build expected host_init LetStmt for cuTensorMapEncodeTiled.

    encode_args is a list of ints: the numeric arguments to cuTensorMapEncodeTiled
    after (tensormap, dtype_str, ndim, A_ptr). The full call is:
        runtime.cuTensorMapEncodeTiled(tensormap, dtype_str, ndim, A_ptr, *encode_args)
    where ndim = encode_args[0] and the rest are the tensor map parameters.
    """
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))
    stack_alloca = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tir.tvm_stack_alloca"),
        [StringImm("tensormap"), IntImm("int32", 1)],
    )
    A_var = Var("A", PointerType(PrimType(dtype), "global"))
    call_args = (
        [
            StringImm("runtime.cuTensorMapEncodeTiled"),
            A_tensormap,
            StringImm(dtype),
            IntImm("int32", encode_args[0]),  # ndim
            A_var,
        ]
        + [IntImm("int32", v) for v in encode_args[1:]]
    )
    encode_call = tvm.tir.Call("int32", tvm.ir.Op.get("tir.tvm_call_packed"), call_args)
    replace_point = OpCall(op=tvm.ir.Op.get("tirx.tvm_kernel_replace_point"))
    return tvm.tir.LetStmt(
        A_tensormap,
        stack_alloca,
        tvm.tir.SeqStmt([tvm.tir.Evaluate(encode_call), replace_point]),
    )


def _build_expected_impl(direction, dtype, s_shape, s_layout, impl_spec):
    """Build expected impl PrimFunc.

    impl_spec is a dict with:
        loop_extents: list[int]  — e.g. [1], [2, 2], [8]
        dim: int  — TMA rank (number of coordinates, also the dim arg to PTX call)
        elem_offset_fn: callable(loop_vars) -> PrimExpr  (or None for 0)
        coord_fn: callable(loop_vars) -> list[PrimExpr]  (dim coordinate args)
        s_start: optional list[int]  — starting index for address_of (default all zeros)
    """
    from tvm.tir.layout import ComposeLayout, SwizzleLayout

    loop_extents = impl_spec["loop_extents"]
    dim = impl_spec["dim"]
    elem_offset_fn = impl_spec.get("elem_offset_fn")
    coord_fn = impl_spec["coord_fn"]

    # Mirror to_tile_layout() in copy_async.py:
    #   ComposeLayout → tile_layout
    #   SwizzleLayout → identity TileLayout(S[shape])
    #   TileLayout    → as-is
    if isinstance(s_layout, ComposeLayout):
        buf_layout = s_layout.tile_layout
    elif isinstance(s_layout, SwizzleLayout):
        buf_layout = TileLayout(S[tuple(s_shape)])
    else:
        buf_layout = s_layout

    # Create loop vars
    n_loops = len(loop_extents)
    if n_loops == 1:
        loop_vars = [Var("loop_vars", "int32")]
    else:
        loop_vars = [Var(f"loop_vars_{i}", "int32") for i in range(n_loops)]

    # Buffer
    s_buf_ptr = Var(
        "s_buf_w_offset_ptr",
        PointerType(PrimType(dtype), "shared.dyn"),
    )
    elem_offset = elem_offset_fn(loop_vars) if elem_offset_fn else IntImm("int32", 0)
    s_buf = tvm.tir.decl_buffer(
        s_shape,
        dtype,
        "s_buf_w_offset",
        data=s_buf_ptr,
        elem_offset=elem_offset,
        scope="shared.dyn",
        layout=buf_layout,
    )

    # Free variables
    mbar_ptr = Var("mbar_ptr", "handle")
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))

    # address_of(s_buf[s_start...])
    s_start = impl_spec.get("s_start")
    if s_start:
        buf_indices = [IntImm("int32", v) for v in s_start]
    else:
        buf_indices = [IntImm("int32", 0)] * len(s_shape)
    addr_of = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tir.address_of"),
        [tvm.tir.BufferLoad(s_buf, buf_indices)],
    )

    # Coordinate args (must have exactly `dim` entries)
    coords = coord_fn(loop_vars)

    # Build PTX call based on direction
    if direction == "g2s":
        # g2c(dim, addr, mbar, tensormap, cta_mask, cta_group, cache_hint, *coords)
        ptx_op = tvm.ir.Op.get("tir.ptx_cp_async_bulk_tensor_global_to_cluster")
        ptx_args = [
            IntImm("int32", dim),
            addr_of,
            mbar_ptr,
            A_tensormap,
            IntImm("int32", 0),
            IntImm("int32", 1),
            StringImm(""),
            *coords,
        ]
    else:  # s2g
        # s2g(dim, addr, tensormap, cache_hint, *coords)
        ptx_op = tvm.ir.Op.get("tir.ptx_cp_async_bulk_tensor_shared_to_global")
        ptx_args = [IntImm("int32", dim), addr_of, A_tensormap, StringImm(""), *coords]

    eval_stmt = tvm.tir.Evaluate(tvm.tir.Call("", ptx_op, ptx_args))

    # Wrap: DeclBuffer -> nested For loops
    body = DeclBuffer(s_buf, eval_stmt)
    for i in range(n_loops - 1, -1, -1):
        body = tvm.tir.For(
            loop_vars[i],
            IntImm("int32", 0),
            IntImm("int32", loop_extents[i]),
            tvm.tir.ForKind.SERIAL,
            body,
        )

    func = tvm.tir.PrimFunc([], body, ret_type=None, buffer_map={})
    func = func.with_attr("global_symbol", "impl")
    func = func.with_attr("is_tirx", IntImm("bool", 1))
    return func


def _zeros(n):
    """Return n zero IntImm coords."""
    return [IntImm("int32", 0)] * n


def _atom_elem_offset(lvs):
    """elem_offset for 2x2 atom tile: lv0 * 8192 + lv1 * 4096."""
    return lvs[0] * 8192 + lvs[1] * 4096


def _atom_coords(lvs):
    """coord_fn for 2x2 atom tile: [0, 0, lv1*4, lv0*2]."""
    return [IntImm("int32", 0), IntImm("int32", 0), lvs[1] * 4, lvs[0] * 2]


def _stride_gap_elem_offset(lvs):
    """elem_offset for stride-gap-outer: lv * 4096."""
    return lvs[0] * 4096


def _stride_gap_coords(lvs):
    """coord_fn for stride-gap-outer: [0, lv*32]."""
    return [IntImm("int32", 0), lvs[0] * 32]


# fmt: off
# Expected parameters for each TMA test case.
# Each entry maps case_id -> (impl_spec_dict, encode_args_list).
#
# impl_spec keys:
#   loop_extents: list[int] — iteration counts for nested loops
#   dim: int — TMA rank = number of coordinates = dim arg to PTX call
#   coord_fn: callable(loop_vars) -> list[PrimExpr] — coordinate arguments (len == dim)
#   elem_offset_fn: optional callable(loop_vars) -> PrimExpr — buffer offset
#
# encode_args: list[int] — all numeric args to cuTensorMapEncodeTiled
#   [ndim, global_strides..., global_dims..., box_dims..., elem_strides...,
#    interleave, swizzle_mode, l2_promotion, oob_fill]
GOLDEN_PARAMS = {
    # ---- G2S cases ----
    "g2s-2d-8x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-3d-shared-64x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3), s_start=[1, 0, 0]),
        [3, 64, 64, 4, 512, 128, 64, 64, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-32x512-atom": (
        dict(loop_extents=[2, 2], dim=4, coord_fn=_atom_coords, elem_offset_fn=_atom_elem_offset),
        [4, 64, 8, 8, 4, 1024, 128, 8192, 64, 8, 4, 2, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-partial-8192": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 8192, 8192, 16384, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-swizzle2": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 32, 8, 8, 512, 64, 32, 8, 8, 1, 1, 1, 0, 2, 2, 0],
    ),
    "g2s-2d-8x256-swizzle1": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 16, 8, 16, 512, 32, 16, 8, 16, 1, 1, 1, 0, 1, 2, 0],
    ),
    "g2s-2d-8x256-swizzle0": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 256, 8, 512, 256, 8, 1, 1, 0, 0, 2, 0],
    ),
    "g2s-2d-8x256-int8": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-bf16": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-fp32": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 32, 8, 8, 1024, 128, 32, 8, 8, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-uint8": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-fp8e4m3": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-2d-8x256-fp8e5m2": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-multiphase-3x8x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 24, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-multiphase-5x64x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 320, 4, 512, 128, 64, 64, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-multiphase-7x32x512-atom": (
        dict(loop_extents=[2, 2], dim=4, coord_fn=_atom_coords, elem_offset_fn=_atom_elem_offset),
        [4, 64, 8, 8, 28, 1024, 128, 8192, 64, 8, 4, 2, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-edge-4d-shared-128x64": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 64, 128, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-edge-partial-offset": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 64)]),  # noqa: E501
        [2, 64, 128, 128, 64, 24, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-edge-large-region": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 128)]),  # noqa: E501
        [2, 64, 256, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-4d-reorder-a": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 512, 256, 1024, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-4d-reorder-b": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 256, 8, 1024, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-partial-3d-shared-a": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 256, 128, 512, 64, 32, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-partial-3d-shared-b": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2), s_start=[1, 0, 0]),
        [2, 512, 256, 1024, 64, 64, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-multidim-4d-a": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 64, 512, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-multidim-4d-b": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 256, 8, 1024, 128, 64, 64, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "g2s-3d-full-contiguous": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 32, 4, 128, 4096, 64, 32, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    "g2s-3d-partial-contiguous": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 16, 8, 256, 4096, 128, 16, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    "g2s-3d-stride-gap-outer": (
        dict(loop_extents=[8], dim=2, coord_fn=_stride_gap_coords, elem_offset_fn=_stride_gap_elem_offset),  # noqa: E501
        [2, 64, 256, 128, 64, 32, 1, 1, 0, 0, 2, 0],
    ),
    # ---- S2G cases ----
    "s2g-multiphase-3x8x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 24, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "s2g-multiphase-5x64x256": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 64, 320, 4, 512, 128, 64, 64, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    "s2g-multiphase-7x32x512-atom": (
        dict(loop_extents=[2, 2], dim=4, coord_fn=_atom_coords, elem_offset_fn=_atom_elem_offset),
        [4, 64, 8, 8, 28, 1024, 128, 8192, 64, 8, 4, 2, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    "s2g-multiphase-3x8x256-swizzle2": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 32, 24, 8, 512, 64, 32, 8, 8, 1, 1, 1, 0, 2, 2, 0],
    ),
    "s2g-multiphase-3x8x256-swizzle0": (
        dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        [2, 256, 24, 512, 256, 8, 1, 1, 0, 0, 2, 0],
    ),
    "s2g-multiphase-3x8x256-int8": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 128, 24, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    "s2g-multiphase-3x8x256-fp32": (
        dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        [3, 32, 24, 8, 1024, 128, 32, 8, 8, 1, 1, 1, 0, 3, 2, 0],
    ),
}
# fmt: on


def _assert_golden(golden_key, direction, dtype, s_shape, smem_layout, impl, host_init_stmts):
    """Build expected impl + host_init from GOLDEN_PARAMS and assert structural equality."""
    impl_spec, encode_args = GOLDEN_PARAMS[golden_key]
    expected_impl = _build_expected_impl(direction, dtype, s_shape, smem_layout, impl_spec)
    expected_host_init = _build_expected_host_init(dtype, encode_args)
    tvm.ir.assert_structural_equal(impl, expected_impl, map_free_vars=True)
    assert len(host_init_stmts) == 1, f"Expected 1 host_init_stmt, got {len(host_init_stmts)}"
    tvm.ir.assert_structural_equal(host_init_stmts[0], expected_host_init, map_free_vars=True)


# ===========================================================================
# Section 1: Non-TMA tests (unchanged)
# ===========================================================================


@pytest.mark.parametrize(
    "task",
    [
        ################ A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8] ################
        (
            (16, 16),  # g_shape
            (8, 8),  # s_shape
            (0, 0),  # g_st
            (8, 8),  # g_extent
            8,  # thread_cnt
            TileLayout(S[16, 16]),  # layoutA
            TileLayout(S[16, 16]),  # layoutB
            TileLayout(S[8, 8]),  # layoutS
        ),
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[128, 32]),  # layoutA
            TileLayout(S[128, 32]),  # layoutB
            TileLayout(S[128, 32]),  # layoutS
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[64, 64]),  # layoutA
            TileLayout(S[64, 64]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################  # noqa: E501
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[4, 32, 32]),  # layoutA
            TileLayout(S[4, 32, 32]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2s_s2g_cta_vec_load(task, dtype):
    g_shape, s_shape, g_st, g_extent, thread_cnt, layoutA, layoutB, layoutS = task
    dev = tvm.cuda(0)

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_st[i], g_st[i] + g_extent[i]) for i in range(len(g_shape)))

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.cta():
                A_smem = Tx.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

                Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], dispatch="non-bulk-copy")
                Tx.ptx.cp_async.commit_group()
                Tx.ptx.cp_async.wait_group()
                Tx.cuda.cta_sync()
                Tx.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.tir.transform.LowerTIRx()(mod)
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
def test_copy_tmem2reg_async(dtype, width_32b):
    """Test async tmem<->local copy using copy_async instead of copy.

    This tests the new copy_async dispatch for tmem<->local that doesn't
    immediately wait after the operation, allowing for pipelining.
    """

    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async_test(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.warpgroup_id([1], parent="cta")
            Tx.warp_id([4], parent="warpgroup")
            Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="cta")

            tmem_addr = Tx.alloc_shared([1], "uint32")

            with Tx.warpgroup()[0:1]:
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                     layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

                A_reg = Tx.alloc_local((WIDTH), dtype)
                B_reg = Tx.alloc_local((WIDTH), dtype)
                A_local = A_reg.view(128, WIDTH, layout=local_view)
                B_local = B_reg.view(128, WIDTH, layout=local_view)

                # A -> A_local
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
                    for i in range(WIDTH):
                        B_reg[i] = Tx.cast(0, dtype)
                Tx.cuda.cta_sync()

                # A_local -> tmem (async)
                Tx.copy_async(tmem[:, :], A_local[:, :])
                Tx.ptx.tcgen05.wait.st()  # explicit wait
                Tx.cuda.cta_sync()

                # tmem -> B_local (async)
                Tx.copy_async(B_local[:, :], tmem[:, :])
                Tx.ptx.tcgen05.wait.ld()  # explicit wait
                Tx.cuda.cta_sync()

                # B_local -> B
                with Tx.thread():
                    for i in range(WIDTH // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                    Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


# ===========================================================================
# Section 2: TMA unit tests — directly call copy_tma_impl
# ===========================================================================


# fmt: off
# ---- DispatchFail cases (transpose layouts) ----
TMA_FAIL_CASES = [
    pytest.param(
        (32, 64), ((0, 32), (0, 64)), (32, 64), ((0, 32), (0, 64)),
        TileLayout(S[32, 64]), TileLayout(S[(32, 64):(1, 32)]),
        id="transpose-32x64",
    ),
    pytest.param(
        (64, 32), ((0, 64), (0, 32)), (64, 32), ((0, 64), (0, 32)),
        TileLayout(S[64, 32]), TileLayout(S[(64, 32):(1, 64)]),
        id="transpose-64x32",
    ),
    pytest.param(
        (128, 64), ((0, 64), (0, 64)), (64, 64), ((0, 64), (0, 64)),
        TileLayout(S[128, 64]), TileLayout(S[(64, 64):(1, 64)]),
        id="transpose-partial-region",
    ),
    pytest.param(
        (128, 64), ((64, 128), (0, 32)), (64, 32), ((0, 64), (0, 32)),
        TileLayout(S[128, 64]), TileLayout(S[(64, 32):(1, 64)]),
        id="transpose-partial-offset",
    ),
]

# ---- G2S success cases ----
# Each entry: (g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout, dtype, golden_key)
TMA_G2S_CASES = [
    # --- From test_copy_g2s_cta_tma_load ---
    # Task 0: (8,256) full region, swizzle_len=3, fp16
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float16", 3, (8, 256)),
        "float16", "g2s-2d-8x256", id="g2s-2d-8x256",
    ),
    # Task 1: (64,256) → 3D shared (3,64,256), region s=(1:2,...), swizzle_len=3, fp16
    pytest.param(
        (64, 256), ((0, 64), (0, 256)),
        (3, 64, 256), ((1, 2), (0, 64), (0, 256)),
        TileLayout(S[64, 256]),
        tma_shared_layout("float16", 3, (3, 64, 256)),
        "float16", "g2s-3d-shared-64x256", id="g2s-3d-shared-64x256",
    ),
    # Task 2: (32,512) custom atom tile layout, swizzle_len=3, fp16
    pytest.param(
        (32, 512), ((0, 32), (0, 512)),
        (32, 512), ((0, 32), (0, 512)),
        TileLayout(S[32, 512]),
        (
            tma_atom_layout("float16", 3)
            .tile_to((16, 256), tma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        "float16", "g2s-2d-32x512-atom", id="g2s-2d-32x512-atom",
    ),
    # Task 3: (8192,8192), partial region → (128,64) shared, swizzle_len=3, fp16
    pytest.param(
        (8192, 8192), ((0, 128), (0, 64)),
        (128, 64), ((0, 128), (0, 64)),
        TileLayout(S[8192, 8192]),
        tma_shared_layout("float16", 3, (128, 64)),
        "float16", "g2s-2d-partial-8192", id="g2s-2d-partial-8192",
    ),
    # --- From test_copy_g2s_cta_tma_load with other swizzle_len ---
    # swizzle_len=2
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float16", 2, (8, 256)),
        "float16", "g2s-2d-8x256-swizzle2", id="g2s-2d-8x256-swizzle2",
    ),
    # swizzle_len=1
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float16", 1, (8, 256)),
        "float16", "g2s-2d-8x256-swizzle1", id="g2s-2d-8x256-swizzle1",
    ),
    # swizzle_len=0
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float16", 0, (8, 256)),
        "float16", "g2s-2d-8x256-swizzle0", id="g2s-2d-8x256-swizzle0",
    ),
    # --- From test_copy_g2s_cta_tma_load with other dtypes ---
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("int8", 3, (8, 256)),
        "int8", "g2s-2d-8x256-int8", id="g2s-2d-8x256-int8",
    ),
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("bfloat16", 3, (8, 256)),
        "bfloat16", "g2s-2d-8x256-bf16", id="g2s-2d-8x256-bf16",
    ),
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float32", 3, (8, 256)),
        "float32", "g2s-2d-8x256-fp32", id="g2s-2d-8x256-fp32",
    ),
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("uint8", 3, (8, 256)),
        "uint8", "g2s-2d-8x256-uint8", id="g2s-2d-8x256-uint8",
    ),
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float8_e4m3fn", 3, (8, 256)),
        "float8_e4m3fn", "g2s-2d-8x256-fp8e4m3", id="g2s-2d-8x256-fp8e4m3",
    ),
    pytest.param(
        (8, 256), ((0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[8, 256]),
        tma_shared_layout("float8_e5m2", 3, (8, 256)),
        "float8_e5m2", "g2s-2d-8x256-fp8e5m2", id="g2s-2d-8x256-fp8e5m2",
    ),
    # --- From test_copy_g2s_cta_tma_load_multi_phase (per-phase slice) ---
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("float16", 3, (8, 256)),
        "float16", "g2s-multiphase-3x8x256", id="g2s-multiphase-3x8x256",
    ),
    pytest.param(
        (5, 64, 256), ((0, 1), (0, 64), (0, 256)),
        (64, 256), ((0, 64), (0, 256)),
        TileLayout(S[5, 64, 256]),
        tma_shared_layout("float16", 3, (64, 256)),
        "float16", "g2s-multiphase-5x64x256", id="g2s-multiphase-5x64x256",
    ),
    pytest.param(
        (7, 32, 512), ((0, 1), (0, 32), (0, 512)),
        (32, 512), ((0, 32), (0, 512)),
        TileLayout(S[7, 32, 512]),
        (
            tma_atom_layout("float16", 3)
            .tile_to((16, 256), tma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        "float16", "g2s-multiphase-7x32x512-atom", id="g2s-multiphase-7x32x512-atom",
    ),
    # --- From test_copy_g2s_cta_tma_load_edge_case ---
    # (128,64) g → (2,2,128,64) 4D shared
    pytest.param(
        (128, 64), ((0, 128), (0, 64)),
        (2, 2, 128, 64), ((0, 1), (0, 1), (0, 128), (0, 64)),
        TileLayout(S[128, 64]).canonicalize(),
        tma_shared_layout("float16", 3, (2, 2, 128, 64)).canonicalize(),
        "float16", "g2s-edge-4d-shared-128x64", id="g2s-edge-4d-shared-128x64",
    ),
    # (128,64) g, partial g=(64:88, 0:64) → (2,2,24,64) s
    pytest.param(
        (128, 64), ((64, 64 + 24), (0, 64)),
        (2, 2, 24, 64), ((0, 1), (0, 1), (0, 24), (0, 64)),
        TileLayout(S[128, 64]).canonicalize(),
        tma_shared_layout("float16", 3, (2, 2, 24, 64)).canonicalize(),
        "float16", "g2s-edge-partial-offset", id="g2s-edge-partial-offset",
    ),
    # (256,64) g, region g=(128:256,...) → (256,64) s, region s=(0:128,...)
    pytest.param(
        (256, 64), ((128, 256), (0, 64)),
        (256, 64), ((0, 128), (0, 64)),
        TileLayout(S[256, 64]).canonicalize(),
        tma_shared_layout("float16", 3, (256, 64)).canonicalize(),
        "float16", "g2s-edge-large-region", id="g2s-edge-large-region",
    ),
    # --- From test_copy_g2s_tma_4d_axis_reorder ---
    pytest.param(
        (2, 128, 8, 64), ((0, 1), (0, 128), (0, 1), (0, 64)),
        (1, 1, 128, 64), ((0, 1), (0, 1), (0, 128), (0, 64)),
        TileLayout(S[2, 128, 8, 64]).canonicalize(),
        tma_shared_layout("float16", 3, (1, 1, 128, 64)).canonicalize(),
        "float16", "g2s-4d-reorder-a", id="g2s-4d-reorder-a",
    ),
    pytest.param(
        (4, 64, 4, 128), ((0, 1), (0, 64), (0, 1), (0, 128)),
        (1, 1, 64, 128), ((0, 1), (0, 1), (0, 64), (0, 128)),
        TileLayout(S[4, 64, 4, 128]).canonicalize(),
        tma_shared_layout("float16", 3, (1, 1, 64, 128)).canonicalize(),
        "float16", "g2s-4d-reorder-b", id="g2s-4d-reorder-b",
    ),
    # --- From test_copy_g2s_tma_partial_region_3d_shared ---
    pytest.param(
        (128, 256), ((0, 32), (0, 64)),
        (6, 128, 64), ((0, 1), (0, 32), (0, 64)),
        TileLayout(S[128, 256]).canonicalize(),
        tma_shared_layout("float16", 3, (6, 128, 64)).canonicalize(),
        "float16", "g2s-partial-3d-shared-a", id="g2s-partial-3d-shared-a",
    ),
    pytest.param(
        (256, 512), ((0, 64), (0, 64)),
        (4, 256, 64), ((1, 2), (0, 64), (0, 64)),
        TileLayout(S[256, 512]).canonicalize(),
        tma_shared_layout("float16", 3, (4, 256, 64)).canonicalize(),
        "float16", "g2s-partial-3d-shared-b", id="g2s-partial-3d-shared-b",
    ),
    # --- From test_copy_tma_multidim_partition ---
    pytest.param(
        (2, 2, 128, 64), ((0, 1), (0, 1), (0, 128), (0, 64)),
        (128, 64), ((0, 128), (0, 64)),
        TileLayout(S[2, 2, 128, 64]).canonicalize(),
        tma_shared_layout("float16", 3, (128, 64)),
        "float16", "g2s-multidim-4d-a", id="g2s-multidim-4d-a",
    ),
    pytest.param(
        (4, 64, 4, 128), ((0, 1), (0, 64), (0, 1), (0, 128)),
        (64, 128), ((0, 64), (0, 128)),
        TileLayout(S[4, 64, 4, 128]).canonicalize(),
        tma_shared_layout("float16", 3, (64, 128)),
        "float16", "g2s-multidim-4d-b", id="g2s-multidim-4d-b",
    ),
    # --- From test_copy_tma_3d_contiguous ---
    pytest.param(
        (4, 32, 64), ((0, 4), (0, 32), (0, 64)),
        (4, 32, 64), ((0, 4), (0, 32), (0, 64)),
        TileLayout(S[4, 32, 64]),
        TileLayout(S[4, 32, 64]),
        "float16", "g2s-3d-full-contiguous", id="g2s-3d-full-contiguous",
    ),
    pytest.param(
        (8, 16, 128), ((0, 4), (0, 16), (0, 128)),
        (4, 16, 128), ((0, 4), (0, 16), (0, 128)),
        TileLayout(S[8, 16, 128]),
        TileLayout(S[4, 16, 128]),
        "float16", "g2s-3d-partial-contiguous", id="g2s-3d-partial-contiguous",
    ),
    # --- From test_copy_tma_partial_contiguous ---
    pytest.param(
        (8, 32, 64), ((0, 8), (0, 32), (0, 64)),
        (8, 32, 64), ((0, 8), (0, 32), (0, 64)),
        TileLayout(S[8, 32, 64]),
        TileLayout(S[(8, 32, 64):(4096, 64, 1)]),
        "float16", "g2s-3d-stride-gap-outer", id="g2s-3d-stride-gap-outer",
    ),
]

# ---- S2G success cases ----
TMA_S2G_CASES = [
    # From test_copy_s2g_tma_store (per-phase slice, same shapes as multi_phase)
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("float16", 3, (8, 256)),
        "float16", "s2g-multiphase-3x8x256", id="s2g-multiphase-3x8x256",
    ),
    pytest.param(
        (5, 64, 256), ((0, 1), (0, 64), (0, 256)),
        (64, 256), ((0, 64), (0, 256)),
        TileLayout(S[5, 64, 256]),
        tma_shared_layout("float16", 3, (64, 256)),
        "float16", "s2g-multiphase-5x64x256", id="s2g-multiphase-5x64x256",
    ),
    pytest.param(
        (7, 32, 512), ((0, 1), (0, 32), (0, 512)),
        (32, 512), ((0, 32), (0, 512)),
        TileLayout(S[7, 32, 512]),
        (
            tma_atom_layout("float16", 3)
            .tile_to((16, 256), tma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        "float16", "s2g-multiphase-7x32x512-atom", id="s2g-multiphase-7x32x512-atom",
    ),
    # Other swizzle lengths for s2g
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("float16", 2, (8, 256)),
        "float16", "s2g-multiphase-3x8x256-swizzle2", id="s2g-multiphase-3x8x256-swizzle2",
    ),
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("float16", 0, (8, 256)),
        "float16", "s2g-multiphase-3x8x256-swizzle0", id="s2g-multiphase-3x8x256-swizzle0",
    ),
    # S2G with different dtypes
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("int8", 3, (8, 256)),
        "int8", "s2g-multiphase-3x8x256-int8", id="s2g-multiphase-3x8x256-int8",
    ),
    pytest.param(
        (3, 8, 256), ((0, 1), (0, 8), (0, 256)),
        (8, 256), ((0, 8), (0, 256)),
        TileLayout(S[3, 8, 256]),
        tma_shared_layout("float32", 3, (8, 256)),
        "float32", "s2g-multiphase-3x8x256-fp32", id="s2g-multiphase-3x8x256-fp32",
    ),
]
# fmt: on


@pytest.mark.parametrize(
    "g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout",
    TMA_FAIL_CASES,
)
def test_copy_tma_codegen_fail(g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout):
    """Verify that copy_tma_impl raises DispatchFail for unsupported layouts."""
    from tvm.tirx.op_dispatch.dispatcher import DispatchFail

    with pytest.raises(DispatchFail, match="stride.*1"):
        _make_tma_call(g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout)


@pytest.mark.parametrize(
    "g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout, dtype, golden_key",
    TMA_G2S_CASES,
)
def test_copy_tma_codegen_g2s(
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype,
    golden_key,
):
    """Unit test: call copy_tma_impl for G2S and assert structural equality with golden."""
    impl, host_init_stmts = _make_tma_call(
        g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout, dtype=dtype
    )
    _assert_golden(golden_key, "g2s", dtype, s_shape, smem_layout, impl, host_init_stmts)


@pytest.mark.parametrize(
    "g_shape, g_region, s_shape, s_region, gmem_layout, smem_layout, dtype, golden_key",
    TMA_S2G_CASES,
)
def test_copy_tma_codegen_s2g(
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype,
    golden_key,
):
    """Unit test: call copy_tma_impl for S2G and assert structural equality with golden."""
    impl, host_init_stmts = _make_tma_call(
        g_shape,
        g_region,
        s_shape,
        s_region,
        gmem_layout,
        smem_layout,
        dtype=dtype,
        direction="s2g",
    )
    _assert_golden(golden_key, "s2g", dtype, s_shape, smem_layout, impl, host_init_stmts)


# ===========================================================================
# Section 3: TMA special cases (symbolic dimension, buffer view)
# ===========================================================================


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_symbolic_dimension(dtype, swizzle_len):
    """Test TMA copy with symbolic dimension in global buffer (like hgemm pattern).

    This tests the pattern:
        Tx.copy_async(A_smem[ks, :, :], A[m_st : m_st + BLK_M, k_start : k_start + BLK_K], **tma_copy)  # noqa: E501

    Where M is a symbolic dimension in the global buffer.
    """  # noqa: E501
    # Fixed dimensions
    K = 256
    BLK_M = 64
    BLK_K = 64
    SMEM_PIPE_DEPTH = 2
    M_CONCRETE = 128  # Concrete value for testing
    thread_cnt = 128

    dev = tvm.cuda(0)

    # Shared memory layout with swizzle
    shared_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
    )

    # Compute bytes for mbarrier
    smem_bytes = SMEM_PIPE_DEPTH * BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        M = Tx.int32()
        A = Tx.match_buffer(A_ptr, [M, K], dtype)
        B = Tx.match_buffer(B_ptr, [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(
                    [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype, dyn.data, elem_offset=0, layout=shared_layout  # noqa: E501
                )
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()

                # Copy with pipeline index (like hgemm pattern)
                for ks in range(SMEM_PIPE_DEPTH):
                    with Tx.thread()[0:1]:
                        Tx.copy_async(
                            A_smem[ks, :, :],
                            A[0:BLK_M, ks * BLK_K:(ks + 1) * BLK_K],
                            dispatch="tma",
                            mbar=mbar_ptr
                        )
                        Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

                    Tx.ptx.mbarrier.try_wait(mbar_ptr, ks % 2)

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()

                # Copy back to global for verification
                with Tx.cta():
                    for ks in range(SMEM_PIPE_DEPTH):
                        Tx.copy(
                            B[ks, :, :],
                            A_smem[ks, :, :]
                        )
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, (M_CONCRETE, K))
        B_np = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # Verify: B[ks, :, :] should equal A[0:BLK_M, ks*BLK_K:(ks+1)*BLK_K]
        B_ref = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)
        for ks in range(SMEM_PIPE_DEPTH):
            B_ref[ks, :, :] = A_np[0:BLK_M, ks * BLK_K : (ks + 1) * BLK_K]
        np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_3d_with_view(dtype, swizzle_len):
    """Test 3D TMA copy using buffer view and swizzle layout (like flash attention pattern).

    This tests the pattern from FA4:
        Q_smem allocated as 4D: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH, NUM_BLK_K, SEQ_TILE, GQA_RATIO, BLK_K)
        Tx.copy_async(Q_smem_3d[pipe_idx, blk_k_idx, :, :, :],
                      Q[batch, seq_start:seq_end, head_start:head_end, k_start:k_end], ...)
    """
    dev = tvm.cuda(0)
    smem_bytes = 2 * 2 * 128 * 64 * tvm.DataType(dtype).bits // 8
    copy_bytes_per_blk = 32 * 4 * 64 * tvm.DataType(dtype).bits // 8

    # Shared memory layout with swizzle
    shared_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[(2, 128, 128) : (128 * 128, 128, 1)]),
    )

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(Q_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        Q = Tx.match_buffer(Q_ptr, (2, 128, 8, 128), dtype)
        B = Tx.match_buffer(B_ptr, (32, 4, 64), dtype)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([128], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                # Allocate as 4D like FA4: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
                Q_smem = Tx.decl_buffer(
                    (2, 2, 128, 64),
                    dtype, dyn.data, elem_offset=0, layout=shared_layout
                )
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                # Create 5D view for 3D copy pattern
                Q_smem_5d = Q_smem.view(2, 2, 32, 4, 64)

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    # 3D copy: [SEQ_Q_PER_TILE, GQA_RATIO, BLK_K]
                    Tx.copy_async(
                        Q_smem_5d[0, 0, :, :, :],
                        Q[0, 0:32, 0:4, 0:64],
                        dispatch="tma",
                        mbar=mbar_ptr
                    )
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes_per_blk)

                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()

                # Copy back to global for verification
                with Tx.cta():
                    Tx.copy(
                        B[:, :, :],
                        Q_smem_5d[0, 0, :, :, :]
                    )
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tir.transform.LowerTIRx()(mod)
        counter = TMACounter()
        counter.visit_stmt(lowered["main"].body)

        assert counter.total_tma_ops == 1, (
            f"Expected exactly 1 TMA operation, got {counter.total_tma_ops}. "
            "This indicates the 3D TMA copy with view is not generating optimal code."
        )

        # Now compile and verify correctness
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        Q_np = tvm.testing.generate_random_array(dtype, (2, 128, 8, 128))
        B_np = np.zeros((32, 4, 64), dtype=np_dtype)

        Q = tvm.runtime.tensor(Q_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(Q, B)

        B_ref = np.zeros((32, 4, 64), dtype=np_dtype)
        B_ref[:, :, :] = Q_np[0, 0:32, 0:4, 0:64]
        np.testing.assert_allclose(B_ref, B.numpy())


# ===========================================================================
# Section 4: TMA GPU smoke tests (end-to-end compilation + correctness)
# ===========================================================================


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize(
    "task",
    [
        # (a) Basic 2D G2S: (8,256) full region
        pytest.param(
            (
                (8, 256),  # g_shape
                ((0, 8), (0, 256)),  # g_region
                (8, 256),  # s_shape
                ((0, 8), (0, 256)),  # s_region
                8,  # thread count per CTA
                TileLayout(S[8, 256]),  # A_layout
                TileLayout(S[8, 256]),  # B_layout
                lambda dtype: tma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-2d-basic",
        ),
        # (b) 3D pipeline G2S: (3,8,256) → (8,256) per-phase
        pytest.param(
            (
                (3, 8, 256),
                None,  # multi-phase: region computed per-phase
                (8, 256),
                None,  # multi-phase
                8,
                TileLayout(S[3, 8, 256]),
                TileLayout(S[3, 8, 256]),
                lambda dtype: tma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-3d-pipeline",
        ),
        # (c) 4D with unit dims: (2,2,128,64), copy (1,1,128,64) → 2D shared (128,64)
        pytest.param(
            (
                (2, 2, 128, 64),
                ((0, 1), (0, 1), (0, 128), (0, 64)),
                (128, 64),
                ((0, 128), (0, 64)),
                128,
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                lambda dtype: tma_shared_layout(dtype, 3, (128, 64)),
            ),
            id="g2s-4d-unit-dims",
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_g2s(task, dtype):
    """Smoke test: compile and run TMA G2S copy on GPU to verify end-to-end correctness."""
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    shared_layout = layoutS_fn(dtype)
    is_pipeline = g_region is None

    if is_pipeline:
        n = g_shape[0]
        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(0, s) for s in s_shape]

        def r_gmem(stage):
            return [
                slice(stage, stage + 1),
                *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
            ]

        # fmt: off
        @Tx.prim_func(tirx=True)
        def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            with Tx.kernel():
                Tx.cta_id([1], parent="kernel")
                Tx.thread_id([thread_cnt], parent="cta")

                with Tx.thread():
                    dyn = Tx.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
                    A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)  # noqa: E501
                    mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                    phase: Tx.int32

                    phase = 0
                    with Tx.thread()[0:1]:
                        Tx.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.cta_sync()

                    for stage in range(n):
                        with Tx.thread()[0:1]:
                            Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))], dispatch="tma", mbar=mbarrier.ptr_to([0]))  # noqa: E501
                            Tx.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)

                        Tx.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
                        phase = phase ^ 1

                        Tx.ptx.fence.proxy_async("shared::cta")
                        Tx.cuda.cta_sync()
                        with Tx.cta():
                            Tx.copy(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)])
        # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(A_np, B.numpy())
    else:
        total_bytes = functools.reduce(
            lambda acc, region: acc * (region[1] - region[0]), s_region, 1
        )
        total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
        r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

        # fmt: off
        @Tx.prim_func(tirx=True)
        def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            with Tx.kernel():
                Tx.cta_id([1], parent="kernel")
                Tx.thread_id([thread_cnt], parent="cta")

                with Tx.thread():
                    dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                    A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)  # noqa: E501
                    mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                    mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                    with Tx.thread()[0:1]:
                        Tx.ptx.mbarrier.init(mbar_ptr, 1)
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.cta_sync()

                    with Tx.thread()[0:1]:
                        Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], dispatch="tma", mbar=mbar_ptr)  # noqa: E501
                        Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
                    Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)
                    Tx.cuda.cta_sync()

                    with Tx.cta():
                        Tx.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
        # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)

            B_ref = np.zeros(g_shape, dtype=np_dtype)
            B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
            np.testing.assert_allclose(B_ref, B.numpy())


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_s2g(dtype):
    """Smoke test: compile and run TMA S2G store on GPU."""
    g_shape = (3, 8, 256)
    s_shape = (8, 256)
    thread_cnt = 8
    n = g_shape[0]

    shared_layout = tma_shared_layout(dtype, 3, s_shape)

    smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [
            slice(stage, stage + 1),
            *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
        ]

    layoutA = TileLayout(S[3, 8, 256])
    layoutB = TileLayout(S[3, 8, 256])

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        with Tx.kernel():
            Tx.cta_id([1], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)  # noqa: E501

                for stage in range(n):
                    Tx.copy(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))])
                    Tx.ptx.fence.proxy_async("shared::cta")
                    with Tx.thread()[0:1]:
                        Tx.copy_async(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)], dispatch="tma")  # noqa: E501
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group()
                    Tx.cuda.cta_sync()
    # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


################################################################################
# SMEM->TMEM async copy tests (consolidated)
################################################################################


def _smem2tmem_async_task(
    dtype, width_32b, *, swizzle_mode=0, smem_rows=32, cta_group=1, partial=None, multicast=""
):
    """Create a task dict for smem2tmem async tests."""
    return dict(
        dtype=dtype,
        width_32b=width_32b,
        swizzle_mode=swizzle_mode,
        smem_rows=smem_rows,
        cta_group=cta_group,
        partial=partial,
        multicast=multicast,
    )


SMEM2TMEM_ASYNC_TASKS = [
    # --- basic (no swizzle) ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w), id=f"basic-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [2, 4, 8, 16, 32, 64, 128]
    ],
    # --- swizzle modes ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w, swizzle_mode=sm), id=f"swizzle{sm}-{dt}-{w}")
        for dt in ["uint8", "float16", "float32"]
        for w in [4, 8, 16, 128]
        for sm in [1, 2, 3]
    ],
    # --- cta_group=2 (cluster copy) ---
    *[
        pytest.param(_smem2tmem_async_task(dt, w, cta_group=2), id=f"cta_group2-{dt}-{w}")
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
    ],
    # --- partial region ---
    *[
        pytest.param(
            _smem2tmem_async_task(dt, w, partial=(sr, sc, tc)),
            id=f"partial-{dt}-{w}-{sr}-{sc}-{tc}",
        )
        for dt in ["float16", "float32"]
        for w in [8, 16]
        for sr, sc, tc in [(32, 0, 4), (0, 4, 16), (64, 8, 12)]
    ],
    # --- shape / multicast ---
    *[
        pytest.param(
            _smem2tmem_async_task(dt, w, smem_rows=int(sh.split("x")[0]), multicast=mc),
            id=f"shape-{sh}-mc{mc or 'none'}-{dt}-{w}",
        )
        for dt in ["float16", "float32"]
        for w in [4, 8, 16, 128]
        for sh, mc in [("128x256b", ""), ("128x128b", ""), ("32x128b", "warpx4")]
    ],
]


@pytest.mark.parametrize("task", SMEM2TMEM_ASYNC_TASKS)
def test_smem2tmem_async(task):
    """Consolidated SMEM->TMEM async copy test covering basic, swizzle, cta_group,
    partial region, and shape/multicast variants."""
    dtype = task["dtype"]
    width_32b = task["width_32b"]
    swizzle_mode = task["swizzle_mode"]
    smem_rows = task["smem_rows"]
    cta_group = task["cta_group"]
    partial = task["partial"]
    multicast = task["multicast"]

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")
    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    # Swizzle compatibility check
    if swizzle_mode > 0:
        atom_shape = tma_atom_shape(dtype, SwizzleMode(swizzle_mode))
        atom_cols = atom_shape[-1]
        if WIDTH < atom_cols or WIDTH % atom_cols != 0:
            pytest.skip(f"WIDTH {WIDTH} not compatible with swizzle mode {swizzle_mode}")

    # Compute dimensions and layouts
    if partial:
        smem_row_start, smem_col_start_32b, tmem_col_offset_32b = partial
        SMEM_COL_START = smem_col_start_32b * (32 // bits)
        TMEM_COL_OFFSET = tmem_col_offset_32b * (32 // bits)
        A_ROWS = 128
        A_WIDTH = ((WIDTH + SMEM_COL_START + VEC_LEN - 1) // VEC_LEN) * VEC_LEN
        TMEM_WIDTH = TMEM_COL_OFFSET + WIDTH
        smem_shape = (A_ROWS, A_WIDTH)
        a_shape = smem_shape
        tmem_n_cols = max(32, next_power_of_2(tmem_col_offset_32b + width_32b))
    else:
        SMEM_COL_START = 0
        TMEM_COL_OFFSET = 0
        A_ROWS = smem_rows
        A_WIDTH = WIDTH
        TMEM_WIDTH = WIDTH
        smem_shape = (smem_rows, WIDTH)
        a_shape = (128, WIDTH)
        tmem_n_cols = max(32, next_power_of_2(width_32b))

    if swizzle_mode > 0:
        smem_layout = tma_shared_layout(dtype, SwizzleMode(swizzle_mode), smem_shape)
    else:
        smem_layout = TileLayout(S[smem_rows, VEC_LEN]).tile_to(smem_shape, (smem_rows, VEC_LEN))
    local_view = TileLayout(S[(128, WIDTH) : (1 @ tid_in_wg, 1)])
    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])

    is_partial = partial is not None
    is_cluster = cta_group == 2
    B_ROWS = 256 if is_cluster else 128
    CTA_STRIDE = 128 * WIDTH

    if is_cluster:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def smem2tmem_async_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                cbx, cby = Tx.cta_id([2, 1], parent="cluster")
                bx = Tx.cta_id([2], parent="kernel")  # noqa: F841
                wg_id = Tx.warpgroup_id([1], parent="cta")  # noqa: F841
                warp_id = Tx.warp_id([4], parent="warpgroup")  # noqa: F841
                lane_id = Tx.thread_id([32], parent="warp")  # noqa: F841
                tid_in_wg = Tx.thread_id([128], parent="cta")
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
                Tx.cuda.cta_sync()
                Tx.cuda.cluster_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.mbarrier_init()
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.warpgroup()[0:1]:
                    with Tx.cta():
                        Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()
                    Tx.cuda.cluster_sync()
                    if cbx == 0:
                        with Tx.thread()[0:1]:
                            Tx.copy_async(tmem[:, 0:WIDTH], A_smem[:, :],
                                    mbar=cp_mbar.ptr_to([0]), cta_group=cta_group, cta_mask=3)
                    Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    Tx.copy_async(B_local[:, :], tmem[:, 0:WIDTH])
                    Tx.ptx.tcgen05.wait.ld()
                    Tx.cuda.cta_sync()
                    cta_row_offset = Tx.meta_var(cbx * CTA_STRIDE)
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[cta_row_offset + g_offset: cta_row_offset + g_offset + VEC_LEN],  # noqa: E501
                                    B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=cta_group)  # noqa: E501
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True)
        def smem2tmem_async_kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, a_shape, dtype)
            B = Tx.match_buffer(B_ptr, (B_ROWS, WIDTH), dtype)
            B_flat = B.view(-1)
            with Tx.kernel():
                bx = Tx.cta_id([1], parent="kernel")  # noqa: F841
                wg_id = Tx.warpgroup_id([1], parent="cta")  # noqa: F841
                warp_id = Tx.warp_id([4], parent="warpgroup")  # noqa: F841
                lane_id = Tx.thread_id([32], parent="warp")  # noqa: F841
                tid_in_wg = Tx.thread_id([128], parent="cta")
                A_smem = Tx.alloc_buffer(smem_shape, dtype, scope="shared", layout=smem_layout, align=1024)  # noqa: E501
                B_reg = Tx.alloc_local((WIDTH), dtype)
                B_local = B_reg.view(128, WIDTH, layout=local_view)
                tmem_addr = Tx.alloc_shared([1], "uint32")
                cp_mbar = Tx.alloc_shared([1], "uint64")
                with Tx.warp()[0:1]:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=tmem_n_cols, cta_group=1)
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer((128, TMEM_WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                            layout=TileLayout(S[(128, TMEM_WIDTH) : (1@TLane, 1@TCol)]))
                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.warpgroup()[0:1]:
                    with Tx.cta():
                        if is_partial:
                            Tx.copy(A_smem[:, :], A[:, :])
                        else:
                            Tx.copy(A_smem[0:smem_rows, 0:WIDTH], A[0:smem_rows, 0:WIDTH])
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()
                    with Tx.thread()[0:1]:
                        if is_partial:
                            Tx.copy_async(tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH],
                                    A_smem[smem_row_start:smem_row_start + 32, SMEM_COL_START:SMEM_COL_START + WIDTH],  # noqa: E501
                                    mbar=cp_mbar.ptr_to([0]), cta_group=1)
                        else:
                            Tx.copy_async(tmem[:, 0:WIDTH], A_smem[:, :],
                                    mbar=cp_mbar.ptr_to([0]), cta_group=1)
                    Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    Tx.copy_async(B_local[:, :], tmem[:, TMEM_COL_OFFSET:TMEM_COL_OFFSET + WIDTH])
                    Tx.ptx.tcgen05.wait.ld()
                    Tx.cuda.cta_sync()
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[g_offset: g_offset + VEC_LEN],
                                    B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])
                    with Tx.warp()[0:1]:
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_n_cols, cta_group=1)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": smem2tmem_async_kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, a_shape)
        B_np = np.zeros((B_ROWS, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)

        B_result = B.numpy()
        if partial:
            expected = A_np[
                smem_row_start : smem_row_start + 32, SMEM_COL_START : SMEM_COL_START + WIDTH
            ]
        elif multicast == "warpx4":
            expected = A_np[0:32, :]
        elif smem_rows < 128:
            expected = A_np[0:32, :]
        else:
            expected = A_np[0:32, :]

        n_warps = 8 if is_cluster else 4
        if not partial and multicast != "warpx4" and smem_rows >= 128:
            np.testing.assert_allclose(B_result, A_np[:B_ROWS, :], err_msg="data mismatch")
        else:
            for warp_idx in range(n_warps):
                np.testing.assert_allclose(
                    B_result[warp_idx * 32 : (warp_idx + 1) * 32, :],
                    expected,
                    err_msg=f"Warp {warp_idx} data mismatch",
                )


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_dynamic_cta_mask(dtype):
    """Regression test for B00004: dynamic cta_mask expression in TMA multicast.

    Verifies that a TIR expression (depending on Tx.cta_id) used as cta_mask in
    copy_async compiles through the full TIRX pipeline without crashing.
    Previously, lower_tirx_scope_ids replaced scope-ID vars via Substitute,
    but Substitute didn't visit OpCall.config values, leaving stale var
    references that caused MakePackedAPI to fail with:
        "variables [...] are used, but are not passed in as API arguments"
    """
    CLUSTER_SIZE = 4
    CTA_GROUP = 2
    BLK_M = 64
    BLK_K = 64
    thread_cnt = 128

    smem_shape = (BLK_M, BLK_K)
    shared_layout = Tx.ComposeLayout(
        Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        Tx.TileLayout(Tx.S[smem_shape : (BLK_K, 1)]),
    )
    smem_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = smem_bytes

    # fmt: off
    @Tx.prim_func(tirx=True)
    def copy_async_dynamic_mask(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [BLK_M, BLK_K], dtype)

        with Tx.kernel():
            cbx = Tx.cta_id([CLUSTER_SIZE], parent="cluster")
            Tx.cta_id([CLUSTER_SIZE], parent="kernel")
            Tx.thread_id([thread_cnt], parent="cta")

            # Dynamic cta_mask: exact expression from B00004 bug report
            cta_mask = Tx.meta_var(5 + 5 * cbx)

            with Tx.thread():
                dyn = Tx.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                A_smem = Tx.decl_buffer(
                    smem_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout,
                )
                mbarrier = Tx.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
                mbar_ptr = Tx.meta_var(mbarrier.ptr_to([0]))

                with Tx.thread()[0:1]:
                    Tx.ptx.mbarrier.init(mbar_ptr, 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()

                with Tx.thread()[0:1]:
                    Tx.copy_async(
                        A_smem[:, :],
                        A[:, :],
                        dispatch="tma",
                        mbar=mbar_ptr,
                        cta_mask=cta_mask,
                        cta_group=CTA_GROUP,
                    )
                    Tx.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

                Tx.ptx.mbarrier.try_wait(mbar_ptr, 0)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_dynamic_mask})
        # This compilation crashed before the B00004 fix with:
        #   "variables [...] are used, but are not passed in as API arguments"
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    # Verify multicast instruction was generated
    src = mod.mod.imports[0].inspect_source()
    assert "multicast" in src, "Expected multicast TMA instruction in generated code"


if __name__ == "__main__":
    tvm.testing.main()
