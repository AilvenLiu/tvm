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

"""Implementation of binary operator dispatchs for CUDA targets.

Registered ops: add, sub, mul, fdiv.
Each op gets two dispatch variants: "shared" and "local" (both priority=10).
See the registration block at the bottom of this file for detailed dispatch
documentation with before/after IR examples.
"""

import functools
import operator
import re
from typing import Literal, NamedTuple

from tvm.arith.analyzer import Analyzer
from tvm.error import InternalError
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, OpCall, PrimExpr, PrimFunc
from tvm.tirx.expr import FloatImm
from tvm.tirx.layout import TileLayout, laneid
from tvm.tirx.op_dispatch import DispatchContext, fail
from tvm.tirx.op_dispatch.dispatcher import predicate

from ..common import MapOpType
from .common import get_indices, get_st_extent, get_vec_len
from .layout_utils import (
    get_local_region,
    get_sublayout_from_region,
    layout_signature,
    resolve_thread_var,
    sig_equal,
)

binary_op_table = {
    MapOpType.ADD: lambda a, b: a + b,
    MapOpType.SUB: lambda a, b: a - b,
    MapOpType.MUL: lambda a, b: a * b,
    MapOpType.FDIV: lambda a, b: a / b,
}

binary_op_f32x2_table = {
    MapOpType.ADD: Tx.ptx.add_packed_f32x2,
    MapOpType.SUB: Tx.ptx.sub_packed_f32x2,
    MapOpType.MUL: Tx.ptx.mul_packed_f32x2,
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


def _match_storage_scope(
    op_call: OpCall,
    sctx: DispatchContext,
    expected_scope: list[Literal["global", "shared*", "local"]],
) -> tuple[bool, str | None]:
    dst_scope = op_call.args[0].buffer.scope()
    if isinstance(op_call.args[1], BufferRegion):
        src1_scope = op_call.args[1].buffer.scope()
    else:
        src1_scope = None
    if isinstance(op_call.args[2], BufferRegion):
        src2_scope = op_call.args[2].buffer.scope()
    else:
        src2_scope = None

    def _check_scope(scope: str | None, pattern: str) -> bool:
        """Glob-lite: 'shared*' => prefix match; otherwise exact."""
        if scope is None:
            return True
        if pattern.endswith("*"):
            return scope.startswith(pattern[:-1])
        return scope == pattern

    ok = any(
        _check_scope(dst_scope, scope)
        and _check_scope(src1_scope, scope)
        and _check_scope(src2_scope, scope)
        for scope in expected_scope
    )
    return (
        ok,
        None
        if ok
        else f"storage scope mismatch: dst {dst_scope}, src1 {src1_scope}, "
        f"src2 {src2_scope}; expected {expected_scope}",
    )


def _dtype_ok(op: OpCall, sctx: DispatchContext, expected_dtype: str):
    """Check if src buffer dtype matches."""
    dst, src1, src2 = op.args[:3]
    if dst.buffer.dtype != expected_dtype:
        return (False, f"dst dtype {dst.buffer.dtype} != {expected_dtype}")
    for i, src in enumerate([src1, src2], 1):
        if isinstance(src, BufferRegion) and src.buffer.dtype != expected_dtype:
            return (False, f"src{i} dtype {src.buffer.dtype} != {expected_dtype}")
        elif isinstance(src, PrimExpr) and src.dtype != expected_dtype:
            return (False, f"src{i} dtype {src.dtype} != {expected_dtype}")
    return (True, None)


def _sm_version_ok(op: OpCall, sctx: DispatchContext, min_version: int):
    """Check if SM version >= min_version."""
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    ok = sm_version >= min_version
    return (ok, None if ok else f"sm_version {sm_version} < {min_version}")


def _get_thread_cnt(sctx: DispatchContext) -> int | None:
    scope_name = sctx.exec_scope.name
    if scope_name == "cta":
        return sctx.launch_params["threadIdx.x"].dom.extent
    if scope_name == "warpgroup":
        return 128
    if scope_name == "warp":
        return 32
    if scope_name == "thread":
        return 1
    return None


def _validate_binary(
    op_call: OpCall,
    sctx: DispatchContext,
    op_type: MapOpType,
) -> tuple[bool, str | None]:
    """Basic validation for binary ops."""
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    # Check that op_type is supported.
    if binary_op_table.get(op_type) is None:
        return (False, f"unsupported binary op: {op_type}")
    _, _, _, msg = _normalize_binary_args(_src1, _src2, op_type)
    return (msg is None, msg)


class _BinaryMapInfo(NamedTuple):
    dst_br: BufferRegion
    src1_br: BufferRegion
    src2_br: BufferRegion | None
    const: FloatImm | None
    dst_start: list
    dst_extent: list
    src1_start: list
    src1_extent: list
    src2_start: list | None
    src2_extent: list | None


def _is_scalar(x) -> bool:
    """Check if x is a scalar (non-BufferRegion PrimExpr, including FloatImm)."""
    return not isinstance(x, BufferRegion)


def _normalize_binary_args(
    _src1: BufferRegion | PrimExpr,
    _src2: BufferRegion | PrimExpr,
    op_type: MapOpType,
) -> tuple[BufferRegion | PrimExpr, BufferRegion | PrimExpr, PrimExpr | None, str | None]:
    """Normalize binary args: move const to rhs and ensure rhs is broadcastable to lhs if needed."""
    # Ensure at least one source is not a scalar constant.
    if _is_scalar(_src1) and _is_scalar(_src2):
        return _src1, _src2, None, "both inputs are constants; unsupported for binary map"

    # If src1 is scalar, swap (only allowed for commutative ops).
    if _is_scalar(_src1):
        if op_type not in (MapOpType.ADD, MapOpType.MUL):
            return _src1, _src2, None, "commutativity required to swap constant as lhs"
        _src1, _src2 = _src2, _src1

    const = _src2 if _is_scalar(_src2) else None
    if const is not None:
        return _src1, _src2, const, None

    # For non-constant rhs, switch broadcasting direction if needed.
    src1_num = functools.reduce(operator.mul, [r.extent for r in _src1.region], 1)
    src2_num = functools.reduce(operator.mul, [r.extent for r in _src2.region], 1)
    if src1_num < src2_num:
        if op_type not in (MapOpType.ADD, MapOpType.MUL):
            return _src1, _src2, None, "non-commutative op cannot broadcast second source"
        _src1, _src2 = _src2, _src1
    return _src1, _src2, None, None


def _try_prepare_binary_map(
    op_call: OpCall,
    op_type: MapOpType,
    require_trivial_layout: bool,
    allow_no_layout: bool = False,
) -> tuple[_BinaryMapInfo | None, str | None]:
    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return None, msg

    dst = _dst.buffer
    src1 = _src1.buffer
    src2_br = _src2 if const is None else None
    src2 = src2_br.buffer if src2_br is not None else None
    dtype = dst.dtype
    has_layout = dst.layout and src1.layout and (src2.layout if src2 else True)
    if not has_layout and not allow_no_layout:
        return None, "unsupported layout/dtype for binary map"
    if not (src1.dtype == dtype and ((src2.dtype == dtype) if src2 else (const.dtype == dtype))):
        return None, "unsupported dtype for binary map"
    if (
        has_layout
        and require_trivial_layout
        and not (
            dst.layout.is_trivial()
            and src1.layout.is_trivial()
            and (src2.layout.is_trivial() if src2 else True)
        )
    ):
        return None, "unsupported non-trivial layout for binary map"

    analyzer = Analyzer()
    dst_extent = [r.extent for r in _dst.region]
    src1_extent = [r.extent for r in _src1.region]
    dst_non1 = [e for e in dst_extent if e != 1]
    src1_non1 = [e for e in src1_extent if e != 1]
    if not (
        len(dst_non1) == len(src1_non1)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src1_non1, dst_non1))
    ):
        return None, "shape mismatch between dst and src1 for binary map"

    src2_extent = [r.extent for r in src2_br.region] if src2_br is not None else None
    if src2_br is not None:
        for i in range(1, len(src2_extent) + 1):
            if src2_extent[-i] not in (1, src1_extent[-i]):
                return None, "src2 not broadcastable to src1 for binary map"

    info = _BinaryMapInfo(
        dst_br=_dst,
        src1_br=_src1,
        src2_br=src2_br,
        const=const,
        dst_start=[r.min for r in _dst.region],
        dst_extent=dst_extent,
        src1_start=[r.min for r in _src1.region],
        src1_extent=src1_extent,
        src2_start=[r.min for r in src2_br.region] if src2_br is not None else None,
        src2_extent=src2_extent,
    )
    return info, None


def _infer_binary_vec_len(
    op_call: OpCall,
    sctx: DispatchContext,
    _dst: BufferRegion,
    _src1: BufferRegion,
    _src2: BufferRegion | None,
) -> tuple[int | None, str | None]:
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is not None:
        return vec_len, None
    tx = _get_thread_cnt(sctx)
    if tx is None:
        return None, f"unsupported exec_scope {sctx.exec_scope.name} for vec_len"

    elem_size = DataType(_dst.buffer.dtype).bits  # in bits
    possible_vec_len = [128 // elem_size, 64 // elem_size, 32 // elem_size, 1]
    vec_len = get_vec_len(_dst, _src1, possible_vec_len, thread_cnt=tx)
    if vec_len is None:
        return None, "no valid vector length; check alignment/extents/thread-count"
    possible_vec_len = [vl for vl in possible_vec_len if vl <= vec_len]
    if _src2 is not None:
        vec_len = get_vec_len(_dst, _src2, possible_vec_len, thread_cnt=tx)
    if vec_len is None:
        return None, "no valid vector length; check alignment/extents/thread-count"
    return vec_len, None


def _is_binary_local_packed_f32x2_case(
    op_call: OpCall,
    op_type: MapOpType,
    sctx: DispatchContext,
) -> bool:
    """Check whether local-thread trivial layout can use packed f32x2 path."""
    if sctx.exec_scope.name != "thread":
        return False
    if op_type not in binary_op_f32x2_table:
        return False
    if not _sm_version_ok(op_call, sctx, min_version=100)[0]:
        return False
    if not _dtype_ok(op_call, sctx, expected_dtype="float32")[0]:
        return False

    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]
    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return False
    if get_vec_len(_dst, _src1, [2], thread_cnt=1) != 2:
        return False
    if const is None:
        if not isinstance(_src2, BufferRegion) or get_vec_len(_dst, _src2, [2], thread_cnt=1) != 2:
            return False
    return True


def _is_binary_local_wgmma_row_red_view_case(
    op_call: OpCall,
    op_type: MapOpType,
    sctx: DispatchContext,
) -> bool:
    """Check whether op fits the local WGMMA/ROW_RED view implementation."""
    if sctx.exec_scope.name not in ["warp", "warpgroup", "cta"]:
        return False
    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return False

    dst, src1 = _dst.buffer, _src1.buffer
    src2 = None if const is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if const is not None else _src2.region

    # no slicing allowed, since op is on local tensor
    if not (
        len(src1_region) == 2
        and len(dst_region) == 2
        and (len(src2_region) == 2 if src2 else True)
        and len(src1.shape) == 2
        and (len(src2.shape) == 2 if src2 else True)
        and len(dst.shape) == 2
        and src1_region[0].min == 0
        and src1_region[1].min == 0
        and ((src2_region[0].min == 0 and src2_region[1].min == 0) if src2 else True)
        and dst_region[0].min == 0
        and dst_region[1].min == 0
        and src1_region[0].extent == src1.shape[0]
        and src1_region[1].extent == src1.shape[1]
        and (
            (src2_region[0].extent == src2.shape[0] and src2_region[1].extent == src2.shape[1])
            if src2
            else True
        )
        and dst_region[0].extent == dst.shape[0]
        and dst_region[1].extent == dst.shape[1]
    ):
        return False

    # basic shape/layout check
    if (
        len(src1.shape) != 2
        or (len(src2.shape) != 2 if src2 else False)
        or len(dst.shape) != 2
        or src1.shape[0] != 16
        or not (src1.shape[1] % 8 == 0 or src1.shape[1] == 4)
        or (src2.shape[0] != 16 if src2 else False)
        or (not (src2.shape[1] % 8 == 0 or src2.shape[1] == 4) if src2 else False)
        or dst.shape[0] != 16
        or not (dst.shape[1] % 8 == 0 or dst.shape[1] == 4)
        or src1.layout is None
        or (src2.layout is None if src2 else False)
        or dst.layout is None
        or src1.layout.is_swizzle()
        or (src2.layout.is_swizzle() if src2 else False)
        or dst.layout.is_swizzle()
    ):
        return False

    src1_extent = [r.extent for r in src1_region]
    dst_extent = [r.extent for r in dst_region]
    src2_extent = [r.extent for r in src2_region] if src2 else None
    broadcast = False
    if src2 is not None:
        for i in range(1, len(src2_extent) + 1):
            if src1_extent[-i] != dst_extent[-i]:
                return False
            if src2_extent[-i] not in (4, src1_extent[-i]):
                return False
            if src2_extent[-i] == 4 and src1_extent[-i] != 4:
                broadcast = True

    atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
    warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
    warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
    red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
    red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))

    def _kind(buf) -> str | None:
        try:
            if warp_atom.is_tile_inner(buf.layout, buf.shape, [8, 8]):
                return "wgmma"
        except InternalError:
            pass
        try:
            if red_warp_atom.is_tile_inner(buf.layout.canonicalize(), (64,), (32,)):
                return "row_red"
        except InternalError:
            pass
        return None

    kind_dst = _kind(dst)
    kind_src1 = _kind(src1)
    if kind_dst is None or kind_src1 is None:
        return False
    if const is not None:
        return kind_dst == kind_src1

    kind_src2 = _kind(src2)
    if kind_src2 is None:
        return False
    if broadcast:
        return kind_dst == "wgmma" and kind_src1 == "wgmma" and kind_src2 == "row_red"
    return kind_dst == kind_src1 == kind_src2


def _validate_binary_local_view_case(
    op_call: OpCall,
    sctx: DispatchContext,
    op_type: MapOpType,
) -> tuple[bool, str | None]:
    if sctx.exec_scope.name not in ["cta", "warpgroup", "warp"]:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for local-view binary op"

    info, msg = _try_prepare_binary_map(op_call, op_type, require_trivial_layout=False)
    if msg is not None:
        return False, msg
    assert info is not None

    _dst, _src1, _src2 = info.dst_br, info.src1_br, info.src2_br
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None

    analyzer = Analyzer()
    dst_st, dst_extent = get_st_extent(_dst)
    src1_st, src1_extent = get_st_extent(_src1)
    src2_st, src2_extent = get_st_extent(_src2) if _src2 is not None else (None, None)

    check_regions = [(dst, dst_st, dst_extent), (src1, src1_st, src1_extent)]
    if src2 is not None:
        check_regions.append((src2, src2_st, src2_extent))

    for buf, st, ext in check_regions:
        layout = buf.layout
        if layout is None:
            return False, "missing layout for local-view binary op"
        if not isinstance(layout, TileLayout) or not getattr(layout, "shard", None):
            return False, "non-TileLayout is not supported for local-view binary op"
        if layout.is_swizzle():
            return False, "swizzle layout is not supported for local-view binary op"
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return False, "thread-shared dimension with zero stride is not supported"
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False, "thread-shared dimension with replica is not supported"
        if get_local_region(layout, buf.shape, st, ext) is None:
            return False, "invalid region for local-view binary op"

    # src1 and dst should represent the same sliced thread/local partition.
    dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)
    src1_sliced = get_sublayout_from_region(src1.layout, src1.shape, src1_st, src1_extent)
    dst_sig = layout_signature(
        dst_sliced.canonicalize() if hasattr(dst_sliced, "canonicalize") else dst_sliced
    )
    src1_sig = layout_signature(
        src1_sliced.canonicalize() if hasattr(src1_sliced, "canonicalize") else src1_sliced
    )
    if not sig_equal(analyzer, src1_sig, dst_sig):
        return False, "cannot validate src1 and dst layout signatures for local-view binary op"

    thread_vars_list = []
    thr_extents = []
    for it in dst_sliced.shard:
        if it.axis.is_thread():
            var = resolve_thread_var(it.axis, sctx)
            if var is None:
                return False, "cannot resolve thread variable"
            thread_vars_list.append(var)
            thr_extents.append(it.extent)

    if thread_vars_list and "threadIdx.x" in sctx.launch_params:
        expected = functools.reduce(operator.mul, thr_extents, 1)
        actual = _get_thread_cnt(sctx)
        if len(set(id(v) for v in thread_vars_list)) == 1:
            if thread_vars_list[0] is sctx.launch_params["threadIdx.x"].var:
                if not analyzer.can_prove_equal(actual, expected):
                    return False, f"thread count mismatch; expected {expected} but got {actual}"

    return True, None


def validate_binary_shared(
    op_call: OpCall,
    sctx: DispatchContext,
    op_type: MapOpType,
) -> tuple[bool, str | None]:
    if sctx.exec_scope.name not in ["cta", "warpgroup", "warp", "thread"]:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for shared binary op"
    _, msg = _try_prepare_binary_map(op_call, op_type, require_trivial_layout=False)
    return (msg is None, msg)


_BINARY_LOCAL_CASE_SUBCTA = "subcta_view"
_BINARY_LOCAL_CASE_THREAD = "thread"


def _classify_binary_local_case(
    op_call: OpCall,
    op_type: MapOpType,
    sctx: DispatchContext,
) -> tuple[str | None, str | None]:
    """Classify local binary path by layout capability.

    For non-thread scopes (cta/warpgroup/warp): WGMMA/ROW_RED-view and
    trivial-layout local-view are supported.
    For thread scope: trivial layout path is supported, with optional packed_f32x2 optimization.
    """
    scope = sctx.exec_scope.name
    if scope not in ["cta", "warpgroup", "warp", "thread"]:
        return None, f"unsupported exec_scope {sctx.exec_scope.name}"

    if scope in ["cta", "warpgroup", "warp"]:
        if _is_binary_local_wgmma_row_red_view_case(op_call, op_type, sctx):
            return _BINARY_LOCAL_CASE_SUBCTA, None
        _, msg = _try_prepare_binary_map(op_call, op_type, require_trivial_layout=True)
        if msg is not None:
            return None, msg
        ok, msg = _validate_binary_local_view_case(op_call, sctx, op_type)
        if ok:
            return _BINARY_LOCAL_CASE_SUBCTA, None
        return None, msg
    elif scope == "thread":
        _, msg = _try_prepare_binary_map(
            op_call, op_type, require_trivial_layout=True, allow_no_layout=True
        )
        if msg is not None:
            return None, msg
        return _BINARY_LOCAL_CASE_THREAD, None
    else:
        return None, f"unsupported exec_scope {sctx.exec_scope.name}"


def validate_binary_local(
    op_call: OpCall,
    sctx: DispatchContext,
    op_type: MapOpType,
) -> tuple[bool, str | None]:
    local_case, err = _classify_binary_local_case(op_call, op_type, sctx)
    return (local_case is not None, err)


def _emit_binary_shared(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc:
    """Emit shared-memory binary map for cta/warpgroup/warp/thread scope."""
    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=False)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, const = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent

    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1  # fallback to scalar
    assert vec_len is not None and vec_len >= 1
    thread_cnt = _get_thread_cnt(sctx)
    if thread_cnt is None:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for shared binary map impl")
    exec_scope_name = sctx.exec_scope.name

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        if exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        if exec_scope_name == "thread":
            return 0
        fail(f"unsupported exec_scope {exec_scope_name} for shared binary map impl")

    @Tx.inline
    def sync():
        if exec_scope_name == "cta":
            Tx.cuda.cta_sync()
        elif exec_scope_name == "warpgroup":
            Tx.cuda.warpgroup_sync(8)  # TODO: derive from launch config
        elif exec_scope_name == "warp":
            Tx.cuda.warp_sync()
        elif exec_scope_name == "thread":
            pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        tid = get_tid_in_scope()
        for s in Tx.serial(0, Tx.ceildiv(n_elements, vec_len * thread_cnt)):
            for vec in Tx.vectorized(vec_len):
                fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                if fused < n_elements:
                    dst_indices = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                    src1_indices = Tx.meta_var(get_indices(fused, src1_start, src1_extent))
                    if const is not None:
                        dst[tuple(dst_indices)] = op_func(src1[tuple(src1_indices)], const)
                    else:
                        src2_indices = Tx.meta_var(
                            get_indices_zero_out(
                                src1_indices, src1_start, src1_extent, src2_start, src2_extent
                            )
                        )
                        dst[tuple(dst_indices)] = op_func(
                            src1[tuple(src1_indices)], src2[tuple(src2_indices)]
                        )
        sync()

    return impl


def _emit_binary_local_trivial_layout(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    """Emit local trivial-layout binary map."""
    if sctx.exec_scope.name != "thread":
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for local thread-trivial binary op")

    info, msg = _try_prepare_binary_map(
        op, binary_op, require_trivial_layout=True, allow_no_layout=True
    )
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, CONST = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent
    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1
    assert vec_len is not None

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for s in Tx.serial(0, Tx.ceildiv(n_elements, vec_len)):
            for vec in Tx.vectorized(vec_len):
                fused = Tx.meta_var(s * vec_len + vec)
                if fused < n_elements:
                    dst_indices = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                    src1_indices = Tx.meta_var(get_indices(fused, src1_start, src1_extent))
                    if CONST is not None:
                        dst[tuple(dst_indices)] = op_func(src1[tuple(src1_indices)], CONST)
                    else:
                        src2_indices = Tx.meta_var(
                            get_indices_zero_out(
                                src1_indices, src1_start, src1_extent, src2_start, src2_extent
                            )
                        )
                        dst[tuple(dst_indices)] = op_func(
                            src1[tuple(src1_indices)], src2[tuple(src2_indices)]
                        )

    return impl


def _emit_binary_local_view(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=False)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, const = info.dst_br, info.src1_br, info.src2_br, info.const

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_st, dst_extent = get_st_extent(_dst)
    src1_st, src1_extent = get_st_extent(_src1)
    src2_st, src2_extent = get_st_extent(_src2) if _src2 is not None else (None, None)

    dst_local_info = get_local_region(dst.layout, dst.shape, dst_st, dst_extent)
    src1_local_info = get_local_region(src1.layout, src1.shape, src1_st, src1_extent)
    if not dst_local_info or not src1_local_info:
        fail("dst/src1 layout is not supported for local-view binary op")
    src2_local_info = (
        get_local_region(src2.layout, src2.shape, src2_st, src2_extent)
        if src2 is not None
        else None
    )
    if src2 is not None and not src2_local_info:
        fail("src2 layout is not supported for local-view binary op")

    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    src1_local_shape, src1_local_st, src1_local_ext = src1_local_info
    if src2_local_info is not None:
        src2_local_shape, src2_local_st, src2_local_ext = src2_local_info
    else:
        src2_local_shape, src2_local_st, src2_local_ext = None, None, None

    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1
    assert vec_len is not None

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    if const is None:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                src1_local = src1.local(*src1_local_shape)
                src2_local = src2.local(*src2_local_shape)

                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src1_indices = Tx.meta_var(
                                get_indices(fused, src1_local_st, src1_local_ext)
                            )
                            src2_indices = Tx.meta_var(
                                get_indices_zero_out(
                                    src1_indices,
                                    src1_local_st,
                                    src1_local_ext,
                                    src2_local_st,
                                    src2_local_ext,
                                )
                            )
                            dst_local[tuple(dst_indices)] = op_func(
                                src1_local[tuple(src1_indices)], src2_local[tuple(src2_indices)]
                            )
    else:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                src1_local = src1.local(*src1_local_shape)

                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src1_indices = Tx.meta_var(
                                get_indices(fused, src1_local_st, src1_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = op_func(
                                src1_local[tuple(src1_indices)], const
                            )

    return impl


def _emit_binary_local_wgmma_row_red_view(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
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
    op = OpCall.downcast(op)
    _dst: BufferRegion = op.output
    _src1: BufferRegion | FloatImm = op.lhs
    _src2: BufferRegion | FloatImm = op.rhs
    _src1, _src2, CONST, msg = _normalize_binary_args(_src1, _src2, binary_op)
    if msg is not None:
        fail(msg)
    if not isinstance(_src1, BufferRegion):
        fail("normalized src1 is not a BufferRegion for local WGMMA/ROW_RED-view binary map")

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = None if CONST is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if CONST is not None else _src2.region
    dtype = dst.dtype

    _src1_start = [r.min for r in src1_region]
    src1_extent = [r.extent for r in src1_region]
    _dst_start = [r.min for r in dst_region]
    dst_extent = [r.extent for r in dst_region]
    if src2_region is not None:
        _src2_start = [r.min for r in src2_region]
        src2_extent = [r.extent for r in src2_region]
    else:
        _src2_start = src2_extent = None

    # basic validation checks
    if not all(
        [
            src1.layout is not None,
            src2.layout is not None if src2 else True,
            dst.layout is not None,
            src1.dtype == dtype,
            src2.dtype == dtype if src2 else CONST.dtype == dtype,
        ]
    ):
        fail("unsupported layout/dtype or exec_scope for local WGMMA/ROW_RED-view binary map")

    # get binary op
    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    # no slicing allowed, since op is on local tensor
    Analyzer()
    if not (
        len(src1_region) == 2
        and len(dst_region) == 2
        and (len(src2_region) == 2 if src2 else True)
        and len(src1.shape) == 2
        and len(dst.shape) == 2
        and (len(src2.shape) == 2 if src2 else True)
        and src1_region[0].min == 0
        and src1_region[1].min == 0
        and ((src2_region[0].min == 0 and src2_region[1].min == 0) if src2 else True)
        and dst_region[0].min == 0
        and dst_region[1].min == 0
        and src1_region[0].extent == src1.shape[0]
        and src1_region[1].extent == src1.shape[1]
        and (
            (src2_region[0].extent == src2.shape[0] and src2_region[1].extent == src2.shape[1])
            if src2
            else True
        )
        and dst_region[0].extent == dst.shape[0]
        and dst_region[1].extent == dst.shape[1]
    ):
        fail("unsupported layout/dtype or exec_scope for local WGMMA/ROW_RED-view binary map")

    # For buffer src2, ensure it is broadcastable to src1,
    # and non-broadcasting dimensions match.
    BROADCAST = False
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src1_extent[-i] != dst_extent[-i]:
                fail("src1 does not match dst extent in binary map")
            if src2_extent[-i] not in (4, src1_extent[-i]):
                fail("src2 not broadcastable to src1 for binary map")
            if src2_extent[-i] == 4 and src1_extent[-i] != 4:
                BROADCAST = True

    # basic shape check
    if (
        len(src1.shape) != 2
        or (len(src2.shape) != 2 if src2 else False)
        or len(dst.shape) != 2
        or src1.shape[0] != 16
        or not (src1.shape[1] % 8 == 0 or src1.shape[1] == 4)
        or (src2.shape[0] != 16 if src2 else False)
        or (not (src2.shape[1] % 8 == 0 or src2.shape[1] == 4) if src2 else False)
        or dst.shape[0] != 16
        or not (dst.shape[1] % 8 == 0 or dst.shape[1] == 4)
        or src1.layout.is_swizzle()
        or (src2.layout.is_swizzle() if src2 else False)
        or dst.layout.is_swizzle()
    ):
        fail("basic shape/layout check failed for WGMMA/ROW_RED-view binary map")

    # layout check:
    # (dst, src1, src2) layout must adhere to one of the five cases below:
    # 1. (WGMMA, WGMMA, WGMMA)
    # 2. (WGMMA, WGMMA, ROW_RED)
    # 3. (ROW_RED, ROW_RED, ROW_RED)
    # 4. (WGMMA, WGMMA, const)
    # 5. (ROW_RED, ROW_RED, const)

    # WGMMA layout check
    atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
    warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
    warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))

    def check_wgmma(buf):
        try:
            return warp_atom.is_tile_inner(buf.layout, buf.shape, [8, 8])
        except InternalError:
            return None

    # ROW_RED layout check
    red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
    red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))

    def check_row_red(buf):
        try:
            return red_warp_atom.is_tile_inner(buf.layout.canonicalize(), (64,), (32,))
        except InternalError:
            return None

    if CONST is not None:
        # check for case 4 and 5
        num_rows = 2
        if check_wgmma(dst) and check_wgmma(src1):
            num_cols = check_wgmma(src1).size()
        elif check_row_red(dst) and check_row_red(src1):
            num_cols = 1
        else:
            fail("layout check failed for const binary map case")

        src1_local_shape = dst_local_shape = (num_rows, num_cols)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl_const():
            with Tx.thread():
                src1_local = src1.local(*src1_local_shape)
                dst_local = dst.local(*dst_local_shape)
                for i in Tx.serial(num_rows):
                    for j in Tx.serial(num_cols):
                        dst_local[i, j] = op_func(src1_local[i, j], CONST)
        # fmt: on

        return impl_const

    if BROADCAST:
        # check for case 2
        if not (check_wgmma(dst) and check_wgmma(src1) and check_row_red(src2)):
            fail("layout check failed for broadcast binary map case")

        num_rows = 2
        src1_local_shape = dst_local_shape = (num_rows, check_wgmma(src1).size())
        src2_local_shape = (num_rows, 1)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl_broadcast():
            with Tx.thread():
                src1_local = src1.local(*src1_local_shape)
                src2_local = src2.local(*src2_local_shape)
                dst_local = dst.local(*dst_local_shape)
                for i in Tx.serial(num_rows):
                    for j in Tx.serial(dst_local_shape[1]):
                        dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, 0])
        # fmt: on

        return impl_broadcast

    # check for case 1 and 3
    num_rows = 2
    if check_wgmma(dst) and check_wgmma(src1) and check_wgmma(src2):
        num_cols = check_wgmma(src1).size()
    elif check_row_red(dst) and check_row_red(src1) and check_row_red(src2):
        num_cols = 1
    else:
        fail("layout check failed for binary map (WGMMA/ROW_RED)")

    src1_local_shape = src2_local_shape = dst_local_shape = (num_rows, num_cols)

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            src1_local = src1.local(*src1_local_shape)
            src2_local = src2.local(*src2_local_shape)
            dst_local = dst.local(*dst_local_shape)
            for i in Tx.serial(num_rows):
                for j in Tx.serial(num_cols):
                    dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, j])
    # fmt: on

    return impl


def _emit_binary_local_packed_f32x2(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    _dst: BufferRegion = op.args[0]

    op_func_f32x2 = binary_op_f32x2_table.get(binary_op)
    if op_func_f32x2 is None:
        fail(f"binary op {binary_op} does not support f32x2 vectorization")

    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=True)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, CONST = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent

    # f32x2 check
    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len = op.config.get("vec_len", None)
    if vec_len is None:
        if get_vec_len(_dst, _src1, [2], thread_cnt=1) is None:
            fail("src1/dst cannot be accessed as float32x2")
        if _src2 is not None and get_vec_len(_dst, _src2, [2], thread_cnt=1) is None:
            fail("src2/dst cannot be accessed as float32x2")
    else:
        if vec_len != 2:
            fail("vec_len must be 2 for f32x2 vectorization")

    rounding_mode = op.config.get("rounding_mode", "rz")

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for s in Tx.serial(0, n_elements // 2):
            dst_indices = Tx.meta_var(get_indices(2 * s, dst_start, dst_extent))
            src1_indices_1 = Tx.meta_var(get_indices(2 * s, src1_start, src1_extent))
            src1_indices_2 = Tx.meta_var(get_indices(2 * s + 1, src1_start, src1_extent))
            if CONST is not None:
                op_func_f32x2(
                    src1[tuple(src1_indices_1)],
                    src1[tuple(src1_indices_2)],
                    CONST,
                    CONST,
                    Tx.address_of(dst[tuple(dst_indices)]),
                    rounding_mode=rounding_mode,
                )
            else:
                src2_indices_1 = Tx.meta_var(
                    get_indices_zero_out(
                        src1_indices_1, src1_start, src1_extent, src2_start, src2_extent
                    )
                )
                src2_indices_2 = Tx.meta_var(
                    get_indices_zero_out(
                        src1_indices_2, src1_start, src1_extent, src2_start, src2_extent
                    )
                )
                op_func_f32x2(
                    src1[tuple(src1_indices_1)],
                    src1[tuple(src1_indices_2)],
                    src2[tuple(src2_indices_1)],
                    src2[tuple(src2_indices_2)],
                    Tx.address_of(dst[tuple(dst_indices)]),
                    rounding_mode=rounding_mode,
                )

    return impl


def binary_shared_impl(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    return _emit_binary_shared(op, binary_op, sctx)


def binary_local_impl(
    op: OpCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    local_case, err = _classify_binary_local_case(op, binary_op, sctx)
    if local_case is None:
        fail(err if err is not None else "unknown error in classifying local binary case")
    if local_case == _BINARY_LOCAL_CASE_SUBCTA:
        if _is_binary_local_wgmma_row_red_view_case(op, binary_op, sctx):
            return _emit_binary_local_wgmma_row_red_view(op, binary_op, sctx)
        return _emit_binary_local_view(op, binary_op, sctx)
    if local_case == _BINARY_LOCAL_CASE_THREAD:
        if _is_binary_local_packed_f32x2_case(op, binary_op, sctx):
            return _emit_binary_local_packed_f32x2(op, binary_op, sctx)
        return _emit_binary_local_trivial_layout(op, binary_op, sctx)
    fail(f"unsupported local case {local_case} for local binary map impl")


# ---------------------------------------------------------------------------
# Registration: bind each binary op name to its CUDA schedule candidates.
# ---------------------------------------------------------------------------
#
# === Variant: "shared" (priority=10) ===
#
# When: dst, src1, src2 are all shared-memory TileLayout buffers (or one src
# is a FloatImm constant — but only for commutative ops like add/mul).
# Non-commutative ops (sub, fdiv) reject constant LHS.
#
# Before (OpCall):
#     with Tx.cta():
#         A_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
#         B_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
#         C_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
#         Tx.add(C_smem[0:32, 0:32], A_smem[0:32, 0:32], B_smem[0:32, 0:32])
#
# After (scheduled PrimFunc, thread_cnt=64, vec_len=8):
#     for s in Tx.serial(ceildiv(1024, 8 * 64)):
#         for vec in Tx.vectorized(8):
#             fused = s * 512 + threadIdx.x * 8 + vec
#             if fused < 1024:
#                 idx = [fused // 32, fused % 32]
#                 C_smem[idx] = A_smem[idx] + B_smem[idx]
#     Tx.cuda.cta_sync()
#
# With constant RHS: Tx.mul(C_smem, A_smem, Tx.float16(2.0))
#     → C_smem[idx] = A_smem[idx] * float16(2.0)
#
# === Variant: "local" (priority=10) ===
#
# When: dst, src1, src2 are all local-scope TileLayout buffers with valid
# thread-partition (shard, no swizzle).
#
# Four sub-paths within binary_local_impl:
#
# (A) wgmma_row_red_view (warp/warpgroup/cta scope, layout has laneid shard
#     matching WGMMA accumulator pattern):
#     Decomposes the logical layout into per-warp iteration with physical
#     indices computed from layout decomposition. Handles the common pattern
#     of binary ops on WGMMA output registers.
#
# Before:
#     with Tx.warp():
#         Tx.add(C_view[0:16, 0:128], A_view[0:16, 0:128], B_view[0:16, 0:128])
#         # A_view, B_view, C_view: local bufs with WGMMA layout
#
# After (per-warp decomposition):
#     with Tx.thread():
#         for outer in Tx.serial(N_COLS // 8):
#             for inner in Tx.unroll(2):
#                 for vec in Tx.vectorized(2):
#                     C[inner, outer * 2 + vec] = A[...] + B[...]
#
# (B) generic view (warp/warpgroup/cta scope, non-WGMMA layout):
#     Flat local view like unary view_full.
#
# (C) packed_f32x2 (thread scope, SM100+, float32, vec_len=2):
#
# Before:
#     with Tx.thread():
#         Tx.add(dst_local, a_local, b_local)  # all float32 local bufs
#
# After (uses PTX add.f32x2):
#     with Tx.thread():
#         for s in Tx.serial(n // 2):
#             Tx.cuda.func_call("add_f32x2", &dst[s*2], &a[s*2], &b[s*2])
#
# (D) trivial_layout (thread scope, generic fallback):
#     Simple per-element loop: serial(n/vec) x vectorized(vec).
#
from tvm.tirx.op_dispatch import register_dispatch  # noqa: E402

for _op_name, _op_type in {
    "add": MapOpType.ADD,
    "sub": MapOpType.SUB,
    "mul": MapOpType.MUL,
    "fdiv": MapOpType.FDIV,
}.items():

    @register_dispatch(
        _op_name,
        "cuda",
        variant="shared",
        priority=10,
        when=[
            predicate("validate_binary", _validate_binary, op_type=_op_type),
            predicate("storage_scope", _match_storage_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_binary_shared, op_type=_op_type),
        ],
    )
    def _shared_dispatch(op: OpCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return binary_shared_impl(op, _ty, sctx)

    @register_dispatch(
        _op_name,
        "cuda",
        variant="local",
        priority=10,
        when=[
            predicate("validate_binary", _validate_binary, op_type=_op_type),
            predicate("storage_scope", _match_storage_scope, expected_scope=["local"]),
            predicate("local_valid", validate_binary_local, op_type=_op_type),
        ],
    )
    def _local_dispatch(op: OpCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return binary_local_impl(op, _ty, sctx)
