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

"""Implementation of unary operator schedules."""

import functools
import operator
from typing import Literal

from tvm.arith.analyzer import Analyzer
from tvm.ir.expr import PrimExpr
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tir import BufferRegion, OpCall, PrimFunc
from tvm.tir.expr import FloatImm
from tvm.tir.layout import TileLayout
from tvm.tirx.op_schedule import ScheduleContext, fail
from tvm.tirx.op_schedule.dispatcher import predicate

from ..common import MapOpType, UnaryBinaryScheduleCandidate
from .cast import (
    _get_local_region,
    _get_sublayout_from_region,
    _layout_signature,
    _resolve_thread_var,
    _sig_equal,
)
from .common import get_indices, get_st_extent, get_vec_len

unary_op_table = {
    MapOpType.ZERO: lambda x, s, b: 0.0,
    MapOpType.FILL: lambda x, s, b: x,
    MapOpType.SQRT: lambda x, s, b: Tx.sqrt(x * s + b) if b is not None else Tx.sqrt(x * s),
    MapOpType.RECIPROCAL: lambda x, s, b: Tx.FloatImm(x.dtype, 1.0) / x,
    MapOpType.EXP: lambda x, s, b: Tx.exp(x * s + b) if b is not None else Tx.exp(x * s),
    MapOpType.EXP2: lambda x, s, b: Tx.exp2(x * s + b) if b is not None else Tx.exp2(x * s),
}


def _match_storage_scope(
    op_call: OpCall,
    sctx: ScheduleContext,
    expected_scope: list[Literal["global", "shared*", "local"]],
) -> tuple[bool, str | None]:
    dst_scope = op_call.args[0].buffer.scope()
    if isinstance(op_call.args[1], BufferRegion):
        src_scope = op_call.args[1].buffer.scope()
    else:
        src_scope = None
    if len(op_call.args) > 2 and isinstance(op_call.args[2], BufferRegion):
        bias_scope = op_call.args[2].buffer.scope()
    else:
        bias_scope = None

    def _check_scope(scope: str | None, pattern: str) -> bool:
        """Glob-lite: 'shared*' => prefix match; otherwise exact."""
        if scope is None:
            return True
        if pattern.endswith("*"):
            return scope.startswith(pattern[:-1])
        return scope == pattern

    ok = any(
        _check_scope(dst_scope, scope)
        and _check_scope(src_scope, scope)
        and _check_scope(bias_scope, scope)
        for scope in expected_scope
    )
    return (
        ok,
        None
        if ok
        else (
            f"storage scope mismatch: dst {dst_scope}, src {src_scope},"
            f" bias {bias_scope}; expected {expected_scope}"
        ),
    )


def get_thread_cnt(sctx: ScheduleContext) -> int | None:
    scope_name = sctx.exec_scope.name
    if scope_name == "cta":
        return sctx.launch_params["threadIdx.x"].dom.extent
    elif scope_name == "warpgroup":
        return 128
    elif scope_name == "warp":
        return 32
    elif scope_name == "thread":
        return 1
    return None


def _unary_args(
    op: OpCall,
) -> tuple[
    BufferRegion,
    BufferRegion | PrimExpr,
    BufferRegion | FloatImm | None,
    FloatImm | None,
]:
    """Parse unary op-call args as (dst, src, bias, scale)."""
    _dst: BufferRegion = op.args[0]
    _src: BufferRegion | PrimExpr = op.args[1]
    _bias: BufferRegion | FloatImm | None = op.args[2] if len(op.args) > 2 else None
    _scale: FloatImm | None = op.args[3] if len(op.args) > 2 else None
    return _dst, _src, _bias, _scale


def _slice_and_layout_signature(buf_region: BufferRegion):
    """Slice a layout by region and return (start, extent, sliced_layout, canonical_signature)."""
    st, ext = get_st_extent(buf_region)
    sliced = _get_sublayout_from_region(buf_region.buffer.layout, buf_region.buffer.shape, st, ext)
    canonical = sliced.canonicalize() if hasattr(sliced, "canonicalize") else sliced
    return st, ext, sliced, _layout_signature(canonical)


def _basic_shape_layout_dtype_checks(
    cur_buf_region: BufferRegion,
    ref_buf_region: BufferRegion,
    analyzer: Analyzer,
    *,
    disallow_swizzle: bool,
) -> bool:
    cur_buf, ref_buf = cur_buf_region.buffer, ref_buf_region.buffer
    cur_region = [r.extent for r in cur_buf_region.region]
    ref_region = [r.extent for r in ref_buf_region.region]
    return (
        len(cur_region) == len(ref_region)
        and all(analyzer.can_prove_equal(r, rr) for r, rr in zip(cur_region, ref_region))
        and (cur_buf.layout is not None and ref_buf.layout is not None)
        and isinstance(cur_buf.layout, TileLayout)
        and isinstance(ref_buf.layout, TileLayout)
        and getattr(cur_buf.layout, "shard", None)
        and getattr(ref_buf.layout, "shard", None)
        and not (disallow_swizzle and (cur_buf.layout.is_swizzle() or ref_buf.layout.is_swizzle()))
    )


def _infer_unary_vec_len(
    op: OpCall,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    thread_cnt: int,
    *,
    fallback_to_scalar: bool,
) -> int | None:
    vec_len = op.config.get("vec_len", None)
    if vec_len is not None:
        return vec_len

    ele_size = DataType(_dst.buffer.dtype).bits  # in bits
    if isinstance(_src, BufferRegion):
        ele_size = max(ele_size, DataType(_src.buffer.dtype).bits)
    possible_vec_lens = [128 // ele_size, 64 // ele_size, 32 // ele_size, 1]
    if isinstance(_src, BufferRegion):
        vec_len = get_vec_len(_src, _dst, possible_vec_lens, thread_cnt)
        if vec_len is None:
            return 1 if fallback_to_scalar else None
        possible_vec_lens = [vl for vl in possible_vec_lens if vl <= vec_len]
        if isinstance(_bias, BufferRegion):
            vec_len = get_vec_len(_bias, _dst, possible_vec_lens, thread_cnt)
    else:
        vec_len = get_vec_len(_dst, _dst, possible_vec_lens, thread_cnt)

    if vec_len is None and fallback_to_scalar:
        return 1
    return vec_len


_LOCAL_CASE_VIEW_FULL = "view_full"
_LOCAL_CASE_VIEW_SLICED = "view_sliced"
_LOCAL_CASE_THREAD_WISE = "thread_wise"


def _classify_unary_local_case(
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    sctx: ScheduleContext,
) -> str | None:
    """Classify local unary implementation path without changing public registration."""

    def _full_region(buf_region: BufferRegion | PrimExpr | None) -> bool:
        if not isinstance(buf_region, BufferRegion):
            return True
        st, ext = get_st_extent(buf_region)
        analyzer = Analyzer()
        zero_st = [0] * len(st)
        return all(
            analyzer.can_prove_equal(e, s) for e, s in zip(ext, buf_region.buffer.shape)
        ) and all(analyzer.can_prove_equal(s, z) for s, z in zip(st, zero_st))

    if sctx.exec_scope.name == "thread":
        return _LOCAL_CASE_THREAD_WISE
    if sctx.exec_scope.name in ["warp", "warpgroup", "cta"]:
        if _full_region(_dst) and _full_region(_src) and _full_region(_bias):
            return _LOCAL_CASE_VIEW_FULL
        return _LOCAL_CASE_VIEW_SLICED
    return None


def validate_unary_shared(
    op: OpCall,
    sctx: ScheduleContext,
) -> tuple[bool, str | None]:
    _dst, _src, _bias, _scale = _unary_args(op)

    # support local-view and local-thread-wise unary ops
    if sctx.exec_scope.name not in ["thread", "warp", "warpgroup", "cta"]:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for shared unary op"

    if not (
        _dst.buffer.scope().startswith("shared")
        and _dst.buffer.layout is not None
        and (_src.buffer.scope().startswith("shared") if isinstance(_src, BufferRegion) else True)
        and (_src.buffer.layout is not None if isinstance(_src, BufferRegion) else True)
        and (_bias.buffer.scope().startswith("shared") if isinstance(_bias, BufferRegion) else True)
        and (_bias.buffer.layout is not None if isinstance(_bias, BufferRegion) else True)
    ):
        return (
            False,
            "invalid storage scope or missing layout for shared unary op;"
            " expected shared scope and valid layout for src/dst/bias if applicable",
        )

    compute_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
    if _scale is not None and _scale.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for scale in shared unary op;"
            f" expected {compute_dtype} but got {_scale.dtype}",
        )
    if isinstance(_bias, BufferRegion) and _bias.buffer.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for bias in shared unary op;"
            f" expected {compute_dtype} but got {_bias.buffer.dtype}",
        )

    analyzer = Analyzer()
    if isinstance(_src, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_src, _dst, analyzer, disallow_swizzle=False):
            return False, "shape or layout mismatch between src and dst for shared unary op"
    if isinstance(_bias, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_bias, _dst, analyzer, disallow_swizzle=False):
            return False, "shape or layout mismatch between bias and dst for shared unary op"

    dst_sig = _slice_and_layout_signature(_dst)[3]
    src_sig = _slice_and_layout_signature(_src)[3] if isinstance(_src, BufferRegion) else None
    bias_sig = _slice_and_layout_signature(_bias)[3] if isinstance(_bias, BufferRegion) else None

    # Here check the canonicalized layouts are semantically equal.
    if src_sig and not _sig_equal(analyzer, src_sig, dst_sig):
        return False, "cannot validate src and dst layout signatures for shared unary op"
    if bias_sig and not _sig_equal(analyzer, bias_sig, dst_sig):
        return False, "cannot validate bias and dst layout signatures for shared unary op"

    return True, None


def unary_shared_impl(
    op: OpCall,
    op_type: MapOpType,
    sctx: ScheduleContext,
) -> PrimFunc | None:
    _dst, _src, _bias, _scale = _unary_args(op)

    dst_start, dst_extent = get_st_extent(_dst)
    num_elements = functools.reduce(operator.mul, dst_extent, 1)
    thread_cnt = get_thread_cnt(sctx)
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
    vec_len = _infer_unary_vec_len(
        op, _dst, _src, _bias, thread_cnt=thread_cnt, fallback_to_scalar=True
    )
    assert vec_len is not None

    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    exec_scope_name = sctx.exec_scope.name

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        elif exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        elif exec_scope_name == "thread":
            return 0

    @Tx.inline
    def sync():
        if exec_scope_name == "cta":
            Tx.cuda.cta_sync()
        elif exec_scope_name == "warpgroup":
            Tx.cuda.warpgroup_sync(8)  # TODO: fix this hardcoded value
        elif exec_scope_name == "warp":
            Tx.cuda.warp_sync()
        elif exec_scope_name == "thread":
            pass

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            tid = get_tid_in_scope()
            for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                # for tid in Tx.thread_binding(thread_st, thread_st + thread_cnt, "threadIdx.x"):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                    if fused < num_elements:
                        idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                        dst[tuple(idx_dst)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
            sync()
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_start, src_extent = get_st_extent(_src)
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, PrimExpr):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                tid = get_tid_in_scope()
                for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                    # for tid in Tx.thread_binding(
                    #     thread_st, thread_st + thread_cnt, "threadIdx.x"):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                        if fused < num_elements:
                            idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                            idx_src = Tx.meta_var(get_indices(fused, src_start, src_extent))
                            dst[tuple(idx_dst)] = Tx.cast(
                                op_func(src[tuple(idx_src)], _scale, _bias),
                                dst.dtype,
                            )
                sync()
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_start, bias_extent = get_st_extent(_bias)

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                tid = get_tid_in_scope()
                for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                    # for tid in Tx.thread_binding(
                    #     thread_st, thread_st + thread_cnt, "threadIdx.x"):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                        if fused < num_elements:
                            idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                            idx_src = Tx.meta_var(get_indices(fused, src_start, src_extent))
                            idx_bias = Tx.meta_var(get_indices(fused, bias_start, bias_extent))
                            dst[tuple(idx_dst)] = Tx.cast(
                                op_func(
                                    src[tuple(idx_src)],
                                    _scale,
                                    bias[tuple(idx_bias)],
                                ),
                                dst.dtype,
                            )
                sync()
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")

    return impl


def validate_unary_local(
    op: OpCall,
    sctx: ScheduleContext,
) -> tuple[bool, str | None]:
    _dst, _src, _bias, _scale = _unary_args(op)
    local_case = _classify_unary_local_case(_dst, _src, _bias, sctx)
    if local_case is None:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for local unary op"

    if not (
        _dst.buffer.scope() == "local"
        and _dst.buffer.layout is not None
        and (_src.buffer.scope() == "local" if isinstance(_src, BufferRegion) else True)
        and (_src.buffer.layout is not None if isinstance(_src, BufferRegion) else True)
        and (_bias.buffer.scope() == "local" if isinstance(_bias, BufferRegion) else True)
        and (_bias.buffer.layout is not None if isinstance(_bias, BufferRegion) else True)
    ):
        return (
            False,
            "invalid storage scope or missing layout for local unary op; expected local scope and "
            "valid layout for src/dst/bias if applicable",
        )

    compute_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
    if _scale is not None and _scale.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for scale in local unary op;"
            f" expected {compute_dtype} but got {_scale.dtype}",
        )
    if isinstance(_bias, BufferRegion) and _bias.buffer.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for bias in local unary op;"
            f" expected {compute_dtype} but got {_bias.buffer.dtype}",
        )

    analyzer = Analyzer()
    if isinstance(_src, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_src, _dst, analyzer, disallow_swizzle=True):
            return False, "shape or layout mismatch between src and dst for local unary op"
    if isinstance(_bias, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_bias, _dst, analyzer, disallow_swizzle=True):
            return False, "shape or layout mismatch between bias and dst for local unary op"

    dst_st, dst_extent, dst_sliced, dst_sig = _slice_and_layout_signature(_dst)
    src_st, src_extent = get_st_extent(_src) if isinstance(_src, BufferRegion) else (None, None)
    bias_st, bias_extent = get_st_extent(_bias) if isinstance(_bias, BufferRegion) else (None, None)

    check_regions = [(_dst.buffer, dst_st, dst_extent)]
    if isinstance(_src, BufferRegion):
        check_regions.append((_src.buffer, src_st, src_extent))
    if isinstance(_bias, BufferRegion):
        check_regions.append((_bias.buffer, bias_st, bias_extent))
    for buf, st, ext in check_regions:
        layout = buf.layout
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return (
                    False,
                    "thread-shared dimension with zero stride is not supported for local unary op",
                )
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False, "thread-shared dimension with replica is not supported for local unary op"
        if _get_local_region(layout, buf.shape, st, ext) is None:
            return False, "invalid region for local-view unary op"

    src_sig = _slice_and_layout_signature(_src)[3] if isinstance(_src, BufferRegion) else None
    bias_sig = _slice_and_layout_signature(_bias)[3] if isinstance(_bias, BufferRegion) else None

    # Here check the canonicalized layouts are semantically equal.
    if src_sig and not _sig_equal(analyzer, src_sig, dst_sig):
        return False, "cannot validate src and dst layout signatures for local unary op"
    if bias_sig and not _sig_equal(analyzer, bias_sig, dst_sig):
        return False, "cannot validate bias and dst layout signatures for local unary op"

    # Validate launch-thread consistency against dst layout thread partition.
    thread_vars_list = []
    thr_extents = []
    for it in dst_sliced.shard:
        if it.axis.is_thread():
            var = _resolve_thread_var(it.axis, sctx)
            if var is None:
                return False, "cannot resolve thread variable"
            thread_vars_list.append(var)
            thr_extents.append(it.extent)

    if thread_vars_list and "threadIdx.x" in sctx.launch_params:
        expected = functools.reduce(operator.mul, thr_extents, 1)
        actual = get_thread_cnt(sctx)
        if len(set(id(v) for v in thread_vars_list)) == 1:
            if thread_vars_list[0] is sctx.launch_params["threadIdx.x"].var:
                if not analyzer.can_prove_equal(actual, expected):
                    return (
                        False,
                        f"thread count mismatch for local unary op;"
                        f" expected {expected} but got {actual}",
                    )
    return True, None


def _emit_unary_local_view_full(
    op: OpCall,
    op_type: MapOpType,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
    dst_local_info: tuple[list, list, list],
) -> PrimFunc:
    dst_local_shape, _, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    vec_len = op.config.get("vec_len", None)
    if vec_len is None:
        analyzer = Analyzer()
        ele_size = DataType(_dst.buffer.dtype).bits  # in bits
        src_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
        ele_size = max(ele_size, DataType(src_dtype).bits)
        for v in [128 // ele_size, 64 // ele_size, 32 // ele_size, 1]:
            if v > 0 and analyzer.can_prove_equal(local_total % v, 0):
                vec_len = v
                break
    assert vec_len is not None

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                base_dst = Tx.decl_buffer((local_total,), dst.dtype, dst.data, scope=dst.scope())
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        local_idx = Tx.meta_var(s * vec_len + vec)
                        base_dst[local_idx] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    base_src = Tx.decl_buffer(
                        (local_total,), src.dtype, src.data, scope=src.scope()
                    )
                    base_dst = Tx.decl_buffer(
                        (local_total,), dst.dtype, dst.data, scope=dst.scope()
                    )
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            local_idx = Tx.meta_var(s * vec_len + vec)
                            base_dst[local_idx] = Tx.cast(
                                op_func(base_src[local_idx], _scale, _bias), dst.dtype
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    base_src = Tx.decl_buffer(
                        (local_total,), src.dtype, src.data, scope=src.scope()
                    )
                    base_dst = Tx.decl_buffer(
                        (local_total,), dst.dtype, dst.data, scope=dst.scope()
                    )
                    base_bias = Tx.decl_buffer(
                        (local_total,), bias.dtype, bias.data, scope=bias.scope()
                    )
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            local_idx = Tx.meta_var(s * vec_len + vec)
                            base_dst[local_idx] = Tx.cast(
                                op_func(base_src[local_idx], _scale, base_bias[local_idx]),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def _emit_unary_local_view_sliced(
    op: OpCall,
    op_type: MapOpType,
    sctx: ScheduleContext,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
    dst_local_info: tuple[list, list, list],
) -> PrimFunc:
    thread_cnt = get_thread_cnt(sctx)
    assert thread_cnt is not None
    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    vec_len = _infer_unary_vec_len(
        op, _dst, _src, _bias, thread_cnt=thread_cnt, fallback_to_scalar=False
    )
    assert vec_len is not None

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                        dst_local[tuple(dst_indices)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_st, src_extent = get_st_extent(_src)
        src_local_info = _get_local_region(src.layout, src.shape, src_st, src_extent)
        if not src_local_info:
            fail("src layout is not supported for local-view cast")
        src_local_shape, src_local_st, src_local_ext = src_local_info
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    src_local = src.local(*src_local_shape)
                    dst_local = dst.local(*dst_local_shape)
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src_indices = Tx.meta_var(
                                get_indices(fused, src_local_st, src_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = Tx.cast(
                                op_func(
                                    src_local[tuple(src_indices)],
                                    _scale,
                                    _bias,
                                ),
                                dst.dtype,
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_st, bias_extent = get_st_extent(_bias)
            bias_local_info = _get_local_region(bias.layout, bias.shape, bias_st, bias_extent)
            if not bias_local_info:
                fail("bias layout is not supported for local-view cast")
            bias_local_shape, bias_local_st, bias_local_ext = bias_local_info

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    src_local = src.local(*src_local_shape)
                    dst_local = dst.local(*dst_local_shape)
                    bias_local = bias.local(*bias_local_shape)
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src_indices = Tx.meta_var(
                                get_indices(fused, src_local_st, src_local_ext)
                            )
                            bias_indices = Tx.meta_var(
                                get_indices(fused, bias_local_st, bias_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = Tx.cast(
                                op_func(
                                    src_local[tuple(src_indices)],
                                    _scale,
                                    bias_local[tuple(bias_indices)],
                                ),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def _emit_unary_local_thread_wise(
    op: OpCall,
    op_type: MapOpType,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
) -> PrimFunc:
    dst_st, dst_extent = get_st_extent(_dst)
    local_total = functools.reduce(operator.mul, dst_extent, 1)
    vec_len = _infer_unary_vec_len(op, _dst, _src, _bias, thread_cnt=1, fallback_to_scalar=False)
    assert vec_len is not None

    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        dst[tuple(dst_indices)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_st, src_extent = get_st_extent(_src)
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            dst[tuple(dst_indices)] = Tx.cast(
                                op_func(
                                    src[tuple(src_indices)],
                                    _scale,
                                    None,
                                ),
                                dst.dtype,
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_st, bias_extent = get_st_extent(_bias)

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            bias_indices = Tx.meta_var(get_indices(fused, bias_st, bias_extent))
                            dst[tuple(dst_indices)] = Tx.cast(
                                op_func(
                                    src[tuple(src_indices)],
                                    _scale,
                                    bias[tuple(bias_indices)],
                                ),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def unary_local_impl(
    op: OpCall,
    op_type: MapOpType,
    sctx: ScheduleContext,
) -> PrimFunc | None:
    _dst, _src, _bias, _scale = _unary_args(op)
    local_case = _classify_unary_local_case(_dst, _src, _bias, sctx)
    if local_case is None:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for unary map impl")

    if local_case in (_LOCAL_CASE_VIEW_FULL, _LOCAL_CASE_VIEW_SLICED):
        dst_st, dst_extent = get_st_extent(_dst)
        dst_local_info = _get_local_region(
            _dst.buffer.layout, list(_dst.buffer.shape), dst_st, dst_extent
        )
        if not dst_local_info:
            fail("dst layout is not supported for local unary op")
        if local_case == _LOCAL_CASE_VIEW_FULL:
            return _emit_unary_local_view_full(
                op, op_type, _dst, _src, _bias, _scale, dst_local_info
            )
        return _emit_unary_local_view_sliced(
            op, op_type, sctx, _dst, _src, _bias, _scale, dst_local_info
        )

    if local_case == _LOCAL_CASE_THREAD_WISE:
        return _emit_unary_local_thread_wise(op, op_type, _dst, _src, _bias, _scale)
    fail(f"unsupported local case {local_case} for unary map impl")


def get_unary_cuda_candidate(unary_op: MapOpType) -> list[UnaryBinaryScheduleCandidate]:
    """Get the appropriate unary schedule candidates for CUDA."""
    candidates = [
        UnaryBinaryScheduleCandidate(
            impl=unary_shared_impl,
            variant="shared",
            priority=10,
            preds=[
                predicate(
                    "storage_scope",
                    _match_storage_scope,
                    expected_scope=["shared*"],
                ),
                predicate("shared_valid", validate_unary_shared),
            ],
        ),
        UnaryBinaryScheduleCandidate(
            impl=unary_local_impl,
            variant="local",
            priority=10,
            preds=[
                predicate("storage_scope", _match_storage_scope, expected_scope=["local"]),
                predicate("local_valid", validate_unary_local),
            ],
        ),
    ]
    return candidates
