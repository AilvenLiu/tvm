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

"""Implementation of cast operator dispatch for CUDA targets.

Registered op: cast.
Two dispatch variants: "local_view" (priority=15) and "thread_wise" (priority=10).
See the @register_dispatch blocks below for detailed documentation with
before/after IR examples.

"""

import functools
import operator

from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, BufferRegion, IntImm, PrimFunc
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.scope_op_dispatch import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import ScopeOpCall

from .common import get_indices, get_st_extent, get_vec_len
from .layout_utils import (
    get_local_region,
    get_sublayout_from_region,
    layout_signature,
    resolve_thread_var,
    sig_equal,
)


def validate_cast_op(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: DispatchContext,  # pylint: disable=unused-argument
) -> bool:
    """Sanity check for cast op"""
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (src.layout and dst.layout):
        return False
    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    # Extract extents and validate non-unit dimensions match
    src_extent_ = [r.extent for r in src_region if r.extent != 1]
    dst_extent_ = [r.extent for r in dst_region if r.extent != 1]
    if len(src_extent_) != len(dst_extent_) or not all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_)
    ):
        return False
    return True


vec2_cast_cuda_intrinsic_dict = {
    ("float32", "float16"): "__float22half2_rn",
    ("float16", "float32"): "__half22float2",
    ("bfloat16", "float32"): "__bfloat1622float2",
    ("float32", "bfloat16"): "__float22bfloat162_rn",
}


_LOCAL_CASE_VIEW_FULL = "view_full"
_LOCAL_CASE_VIEW_SLICED = "view_sliced"
_LOCAL_CASE_THREAD_WISE = "thread_wise"


def _classify_cast_local_case(
    dst_region: BufferRegion,
    src_region: BufferRegion,
    sctx: DispatchContext,
) -> str | None:
    """Classify local cast implementation path."""

    def _full_region(buf_region: BufferRegion) -> bool:
        st, ext = get_st_extent(buf_region)
        analyzer = Analyzer()
        return all(
            analyzer.can_prove_equal(e, s) for e, s in zip(ext, buf_region.buffer.shape)
        ) and all(analyzer.can_prove_equal(s, 0) for s in st)

    if sctx.exec_scope.name == "thread":
        return _LOCAL_CASE_THREAD_WISE
    if sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"]:
        if _full_region(dst_region) and _full_region(src_region):
            return _LOCAL_CASE_VIEW_FULL
        return _LOCAL_CASE_VIEW_SLICED
    return None


def validate_cast_local_view(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: DispatchContext,
) -> bool:
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

    if not (
        src.scope() == "local"
        and dst.scope() == "local"
        and src.layout
        and dst.layout
        and sctx.is_cuda()
        and sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"]
    ):
        return False

    analyzer = Analyzer()

    src_region_extents = [r.extent for r in src_buffer_region.region]
    dst_region_extents = [r.extent for r in dst_buffer_region.region]
    if len(src_region_extents) != len(dst_region_extents):
        return False
    if not all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_region_extents, dst_region_extents)
    ):
        return False
    if (
        (src.layout.size() != dst.layout.size())
        or src.layout.is_swizzle()
        or dst.layout.is_swizzle()
    ):
        return False
    if not isinstance(src.layout, TileLayout) or not isinstance(dst.layout, TileLayout):
        return False
    if not getattr(src.layout, "shard", None) or not getattr(dst.layout, "shard", None):
        return False

    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    for layout, buf, st, ext in [
        (src.layout, src, src_st, src_extent),
        (dst.layout, dst, dst_st, dst_extent),
    ]:
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return False
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False
        if get_local_region(layout, list(buf.shape), st, ext) is None:
            return False

    # Slice → Canonicalize
    src_sliced = get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
    dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)

    src_can = src_sliced.canonicalize() if hasattr(src_sliced, "canonicalize") else src_sliced
    dst_can = dst_sliced.canonicalize() if hasattr(dst_sliced, "canonicalize") else dst_sliced

    src_sig = layout_signature(src_can)
    dst_sig = layout_signature(dst_can)

    # Here check the canonicalized layouts are semantically equal.
    if not sig_equal(analyzer, src_sig, dst_sig):
        return False

    # Resolve thread vars once; reuse for launch count check
    thread_vars_list = []
    thr_extents = []
    for it in src_sliced.shard:
        if it.axis.is_thread():
            var = resolve_thread_var(it.axis, sctx)
            if var is None:
                return False
            thread_vars_list.append(var)
            thr_extents.append(it.extent)

    if thread_vars_list and "threadIdx.x" in sctx.launch_params:
        expected = functools.reduce(operator.mul, thr_extents, 1)
        actual = sctx.launch_params["threadIdx.x"].dom.extent
        if len(set(id(v) for v in thread_vars_list)) == 1:
            if thread_vars_list[0] is sctx.launch_params["threadIdx.x"].var:
                if not analyzer.can_prove_equal(actual, expected):
                    return False

    return True


def _emit_cast_local_view_full(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    dst_local_info,
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    _, _, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # Decide vec2 availability (offset is 0 for full region, so alignment is guaranteed)
    use_vec2 = (src.dtype, dst.dtype) in vec2_cast_cuda_intrinsic_dict
    local_total_imm = int(local_total.value) if isinstance(local_total, IntImm) else None

    # Prepare vec2 intrinsic codegen if needed
    if use_vec2:
        intrinsic_name = vec2_cast_cuda_intrinsic_dict[(src.dtype, dst.dtype)]
        src_dtypex2 = dtypex2_dict[src.dtype]
        dst_dtypex2 = dtypex2_dict[dst.dtype]
        func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""
    else:
        func_name = None
        source_code = None

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            base_src = Tx.decl_buffer((local_total,), src.dtype, src.data, scope=src.scope())
            base_dst = Tx.decl_buffer((local_total,), dst.dtype, dst.data, scope=dst.scope())

            if use_vec2 and (local_total_imm is not None):
                # vec2 loop + tail scalar (odd case)
                n2 = Tx.meta_var(local_total_imm // 2)
                tail = Tx.meta_var(local_total_imm % 2)

                for s in Tx.serial(0, n2):
                    local_idx = Tx.meta_var(s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[local_idx]),
                        Tx.address_of(base_src[local_idx]),
                        source_code=source_code,
                    )

                if tail == 1:
                    local_idx = Tx.meta_var(n2 * 2)
                    base_dst[local_idx] = Tx.cast(base_src[local_idx], dst.dtype)
            else:
                for s in Tx.serial(0, local_total):
                    local_idx = Tx.meta_var(s)
                    base_dst[local_idx] = Tx.cast(base_src[local_idx], dst.dtype)
    # fmt: on

    return impl


def _emit_cast_local_view_sliced(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    dst_local_info,
    src_local_info,
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    src_local_shape, src_local_st, src_local_ext = src_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            src_local = src.local(*src_local_shape)
            dst_local = dst.local(*dst_local_shape)
            for s in Tx.serial(0, local_total):
                fused = Tx.meta_var(s)
                dst_indices = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                src_indices = Tx.meta_var(get_indices(fused, src_local_st, src_local_ext))
                dst_local[tuple(dst_indices)] = Tx.cast(src_local[tuple(src_indices)], dst.dtype)
    # fmt: on

    return impl


def cast_local_view_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: DispatchContext,
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if sctx.exec_scope.name not in ["warp", "warpgroup", "cta", "cluster"]:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for local-view cast")

    local_case = _classify_cast_local_case(dst_buffer_region, src_buffer_region, sctx)

    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
    if not dst_local_info:
        fail("dst layout is not supported for local-view cast")

    if local_case == _LOCAL_CASE_VIEW_FULL:
        return _emit_cast_local_view_full(dst_buffer_region, src_buffer_region, dst_local_info)

    # view_sliced
    src_st, src_extent = get_st_extent(src_buffer_region)
    src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
    if not src_local_info:
        fail("src layout is not supported for local-view cast")

    return _emit_cast_local_view_sliced(
        dst_buffer_region, src_buffer_region, dst_local_info, src_local_info
    )


dtypex2_dict = {
    "float32": "float2",
    "float16": "half2",
    "bfloat16": "nv_bfloat162",
}


def cast_thread_wise_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: DispatchContext,
) -> PrimFunc | None:
    if sctx.exec_scope.name != "thread":
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    # Extract regions and validate dimensions
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    vec_len = get_vec_len(dst_buffer_region, src_buffer_region, [2, 1])

    intrinsic_name = vec2_cast_cuda_intrinsic_dict.get((src.dtype, dst.dtype), None)
    if intrinsic_name is None:
        fail(f"unsupported CUDA cast intrinsic for {src.dtype} -> {dst.dtype}")

    src_dtypex2 = dtypex2_dict[src.dtype]
    dst_dtypex2 = dtypex2_dict[dst.dtype]

    func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
    source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for s in Tx.serial(0, n_elements // (vec_len)):
            fused = Tx.meta_var(s * vec_len)
            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
            if vec_len == 2:
                Tx.cuda.func_call(
                    func_name,
                    dst.ptr_to(dst_indices),
                    src.ptr_to(src_indices),
                    source_code=source_code,
                )
            else:
                dst[tuple(dst_indices)] = Tx.cast(src[tuple(src_indices)], dst.dtype)
    # fmt: on
    return impl


# === Variant: cast/local_view (priority=15) ===
#
# When: both dst and src are local-scope TileLayout buffers with matching
# canonical layout signatures, at warp/warpgroup/cta/cluster scope.
# Higher priority than thread_wise — preferred when layout partition is valid.
#
# Before (ScopeOpCall):
#     with Tx.warp():
#         Tx.cast(dst_view[0:16, 0:128], src_view[0:16, 0:128])
#         # src: local float16, dst: local float32, both WGMMA layout
#
# After — view_full path (local_total=64, fp16→fp32 uses vec2 intrinsic):
#     with Tx.thread():
#         base_dst = Tx.decl_buffer((64,), "float32", dst.data, scope="local")
#         base_src = Tx.decl_buffer((64,), "float16", src.data, scope="local")
#         for s in Tx.serial(64 // 2):
#             # __half22float2: converts 2 fp16 values to 2 fp32 at once
#             Tx.cuda.func_call("__half22float2", &base_dst[s*2], &base_src[s*2])
#         # If dtype pair has no vec2 intrinsic, falls back to:
#         # base_dst[idx] = Tx.cast(base_src[idx], "float32")
#
# After — view_sliced path (partial region, e.g. [4:12, 0:128]):
#     with Tx.thread():
#         src_local = src.local(*src_local_shape)
#         dst_local = dst.local(*dst_local_shape)
#         for s in Tx.serial(local_total):
#             dst_indices = get_indices(s, dst_local_st, dst_local_ext)
#             src_indices = get_indices(s, src_local_st, src_local_ext)
#             dst_local[dst_indices] = Tx.cast(src_local[src_indices], dst.dtype)
#
@register_dispatch(
    "cast",
    "cuda",
    variant="local_view",
    priority=15,
    when=[
        predicate(
            "validate_cast_local_view",
            lambda op, sctx: (
                validate_cast_local_view(
                    ScopeOpCall.downcast(op).output, ScopeOpCall.downcast(op).input, sctx
                ),
                "validate_cast_local_view failed",
            ),
        ),
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"],
                f"unsupported exec_scope {sctx.exec_scope.name} for local cast",
            ),
        ),
    ],
)
def cast_dispatch_local_view(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc:
    op_call = ScopeOpCall.downcast(op_call)
    return cast_local_view_impl(op_call.output, op_call.input, sctx)


# === Variant: cast/thread_wise (priority=10) ===
#
# When: both dst and src are local-scope TileLayout buffers, at thread scope.
# Fallback when local_view is not applicable (wrong scope or layout).
#
# Before (ScopeOpCall):
#     with Tx.thread():
#         Tx.cast(dst_local[0:16, 0:16], src_local[0:16, 0:16])
#         # src: local float16 (16,16), dst: local float32 (16,16)
#
# After (n_elems=256, fp16→fp32 uses vec2 intrinsic):
#     with Tx.thread():
#         for s in Tx.serial(256 // 2):
#             Tx.cuda.func_call("__half22float2", &dst[s*2], &src[s*2])
#         # Without vec2: dst[s] = Tx.cast(src[s], "float32")
#
@register_dispatch(
    "cast",
    "cuda",
    variant="thread_wise",
    priority=10,
    when=[
        predicate(
            "validate_cast_op",
            lambda op, sctx: (
                validate_cast_op(
                    ScopeOpCall.downcast(op).output, ScopeOpCall.downcast(op).input, sctx
                ),
                "validate_cast_op failed",
            ),
        ),
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.exec_scope.name == "thread",
                f"unsupported exec_scope {sctx.exec_scope.name}",
            ),
        ),
    ],
)
def cast_dispatch_thread_wise(op_call: ScopeOpCall, sctx: DispatchContext) -> PrimFunc:
    op_call = ScopeOpCall.downcast(op_call)
    return cast_thread_wise_impl(op_call.output, op_call.input, sctx)
