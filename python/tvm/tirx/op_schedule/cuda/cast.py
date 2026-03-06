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

"""Implementation of cast operator on CUDA."""

import functools
import operator

from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tir import Buffer, BufferRegion, IntImm, PrimFunc
from tvm.tir.layout import TileLayout
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule import (
    ScheduleContext,
    fail,
    predicate,
    register_dispatch,
)

from .common import get_indices, get_st_extent, get_vec_len


def _get_sublayout_from_region(layout, buffer_shape, region_st, region_extent):
    """Get sublayout by slicing the layout with the buffer region.

    Args:
        layout: The buffer's TileLayout.
        buffer_shape: The buffer's shape.
        region_st: Region start indices.
        region_extent: Region extents.

    Returns:
        Sublayout if slicing succeeds, otherwise the original layout.
    """
    if not layout:
        return layout
    region = [(region_st[i], region_st[i] + region_extent[i]) for i in range(len(region_st))]
    sliced = layout.slice(list(buffer_shape), region)
    return sliced if sliced is not None else layout


def _get_layout_thread_local_partition(layout):
    """Extract thread and local dimension info from layout.

    Returns:
        tuple | None: On success, (thread_groups, local_dim_indices, local_extents).
            - thread_groups: dict {axis: (dim_indices, extents)} for each thread axis
            - local_dim_indices: list of dimension indices for local (memory) axes
            - local_extents: list of extents for local dimensions
            Returns None if layout is not supported.

    Validates:
        - No stride==0 on thread dims (broadcast/overlap = cross-thread semantics)
        - Local dims may have arbitrary strides (alignment uses actual layout strides)
        - No thread axes in replica

    Example:
        Layout (2, 8, 4, 2):(2@warpid, 4@laneid, 1@laneid, 1@m) returns:
        - thread_groups = {warpid: ([0], [2]), laneid: ([1, 2], [8, 4])}
        - local_dim_indices = [3], local_extents = [2]
    """
    if not isinstance(layout, TileLayout):
        return None

    shard = getattr(layout, "shard", None)
    if not shard:
        return None

    # Partition dimensions into thread and local (memory) axes
    thread_dim_indices = [i for i, it in enumerate(shard) if it.axis.is_thread()]
    local_dim_indices = [i for i, it in enumerate(shard) if not it.axis.is_thread()]

    if not thread_dim_indices or not local_dim_indices:
        return None

    analyzer = Analyzer()
    for idx in thread_dim_indices:
        if analyzer.can_prove_equal(shard[idx].stride, 0):
            return None

    # Replica must not contain thread axes
    replica = getattr(layout, "replica", None)
    if replica and any(it.axis.is_thread() for it in replica):
        return None

    # Group thread dimensions by axis
    from collections import defaultdict

    thread_groups_dict = defaultdict(list)
    for idx in thread_dim_indices:
        thread_groups_dict[shard[idx].axis].append(idx)

    thread_groups = {}

    for axis, dim_indices in thread_groups_dict.items():
        dim_indices = sorted(dim_indices)
        extents = [shard[i].extent for i in dim_indices]
        thread_groups[axis] = (dim_indices, extents)

    local_extents = [shard[i].extent for i in local_dim_indices]
    return (thread_groups, local_dim_indices, local_extents)


def _get_local_region(orig_layout: TileLayout, buffer_shape, region_st, region_extent):
    """Compute local storage shape, iteration starts, and extents with validation of region.

    Args:
        orig_layout: The original (unsliced) TileLayout.
        buffer_shape: The buffer shape.
        region_st: Region start in shape space.
        region_extent: Region extent in shape space.

    Returns:
        (local_shape, local_st, local_ext) or None if no local dims / invalid region.
        - local_shape: full storage extents per local dim.
        - local_st: region start per local dim.
        - local_ext: region extent per local dim.

    Example:
        Layout (2, 8, 4, 2):(8@m, 2@laneid, 2@m, 1@m), Shape [16, 8], Region [8:16, :] returns:
        - local_shape = [2, 8], local_st = [1, 0], local_ext = [1, 8]
    """
    grouped, seps = orig_layout.group(list(buffer_shape))

    local_shape = []
    local_st = []
    local_ext = []
    analyzer = Analyzer()

    for d in range(len(buffer_shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        has_local = any(not grouped.shard[s].axis.is_thread() for s in shard_range)
        if not has_local:
            continue

        has_thread = any(grouped.shard[s].axis.is_thread() for s in shard_range)

        if not has_thread:
            # Pure local shape dim: use shape-level values directly.
            local_shape.append(buffer_shape[d])
            local_st.append(region_st[d])
            local_ext.append(region_extent[d])
        else:
            # Decompose start element
            remaining_st = region_st[d]
            st_coords = []
            for i, s_idx in enumerate(shard_range):
                sub_prod = 1
                for j in range(i + 1, len(shard_range)):
                    sub_prod = sub_prod * grouped.shard[shard_range[j]].extent
                st_coords.append(remaining_st // sub_prod)
                remaining_st = remaining_st % sub_prod

            # Decompose end element
            remaining_end = region_st[d] + region_extent[d] - 1
            end_coords = []
            for i, s_idx in enumerate(shard_range):
                sub_prod = 1
                for j in range(i + 1, len(shard_range)):
                    sub_prod = sub_prod * grouped.shard[shard_range[j]].extent
                end_coords.append(remaining_end // sub_prod)
                remaining_end = remaining_end % sub_prod

            # check the rectangularity and contiguity of the sliced region
            cur_local_shape, cur_local_st, cur_local_end = 1, 0, 0
            for k in reversed(range(len(st_coords))):
                if grouped.shard[seps[d] + k].axis.is_thread():
                    # for thread dims, region must be contiguous and span full extent
                    if not (
                        analyzer.can_prove_equal(st_coords[k], 0)
                        and analyzer.can_prove_equal(
                            end_coords[k], grouped.shard[seps[d] + k].extent - 1
                        )
                    ):
                        return None
                else:
                    if not analyzer.can_prove_equal(end_coords[k] - st_coords[k], 1) and not (
                        analyzer.can_prove_equal(st_coords[k], 0)
                        and analyzer.can_prove_equal(
                            end_coords[k], grouped.shard[seps[d] + k].extent - 1
                        )
                    ):
                        # to ensure contiguity, if the region spans multiple values
                        # in this dim, it must span the full extent
                        return None
                    cur_local_shape *= grouped.shard[seps[d] + k].extent
                    cur_local_st = cur_local_st * grouped.shard[seps[d] + k].extent + st_coords[k]
                    cur_local_end = (
                        cur_local_end * grouped.shard[seps[d] + k].extent + end_coords[k]
                    )

            # double check the validity of the sliced region
            assert region_extent[d] == functools.reduce(
                operator.mul, [end - st + 1 for st, end in zip(st_coords, end_coords)], 1
            )

            # append the local info without thread dims
            local_shape.append(cur_local_shape)
            local_st.append(cur_local_st)
            local_ext.append(cur_local_end - cur_local_st + 1)

    if not local_shape:
        return None
    return local_shape, local_st, local_ext


def validate_cast_op(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,  # pylint: disable=unused-argument
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


def _cast_layout_supported_for_local(layout) -> bool:
    """Check that layout is valid for local cast (warp/warpgroup/cta/cluster): filter out cross-thread semantics.

    Args:
        layout: TileLayout to check.

    Returns:
        True if layout is valid for local cast, False otherwise.
    """  # noqa: E501
    return _get_layout_thread_local_partition(layout) is not None


def _compute_linear_offset(region_st, local_dims, layout):
    """Compute linear offset using layout's actual strides.

    Physical offset = sum(region_st[dim] * layout.shard[dim].stride) for all local dims.
    """
    offset = 0
    for dim_idx in local_dims:
        offset = offset + region_st[dim_idx] * layout.shard[dim_idx].stride
    return offset


def _axis_key(axis):
    if hasattr(axis, "name") and axis.name:
        return str(axis.name)
    return str(axis)


def _layout_signature(layout):
    """Return semantic signature from canonicalized TileLayout.

    Returns (thread_sig, local_sig, replica_sig).
    Each sig is a list of (axis_key, extent, stride) in shard/replica order.
    """
    if not isinstance(layout, TileLayout):
        return None
    shard = getattr(layout, "shard", None)
    if not shard:
        return None

    thread_sig = []
    local_sig = []
    for it in shard:
        item = (_axis_key(it.axis), it.extent, it.stride)
        if it.axis.is_thread():
            thread_sig.append(item)
        else:
            local_sig.append(item)

    replica_sig = []
    replica = getattr(layout, "replica", None) or []
    for it in replica:
        replica_sig.append((_axis_key(it.axis), it.extent, it.stride))
    return (thread_sig, local_sig, replica_sig)


def _sig_equal(analyzer: Analyzer, src_sig, dst_sig) -> bool:
    """Compare two layout signatures with semantic equality (Analyzer).

    Signatures come from _layout_signature(layout) and are:
      (thread_sig, local_sig, replica_sig)
    Each sig element is (axis_key, extent, stride).
    """
    if src_sig is None or dst_sig is None:
        return False

    src_thread_sig, src_local_sig, src_replica_sig = src_sig
    dst_thread_sig, dst_local_sig, dst_replica_sig = dst_sig

    if len(src_thread_sig) != len(dst_thread_sig):
        return False
    if len(src_local_sig) != len(dst_local_sig):
        return False
    if len(src_replica_sig) != len(dst_replica_sig):
        return False

    def _list_equal(a_list, b_list) -> bool:
        for (a_key, a_ext, a_str), (b_key, b_ext, b_str) in zip(a_list, b_list):
            if a_key != b_key:
                return False
            if not analyzer.can_prove_equal(a_ext, b_ext):
                return False
            if not analyzer.can_prove_equal(a_str, b_str):
                return False
        return True

    return (
        _list_equal(src_thread_sig, dst_thread_sig)
        and _list_equal(src_local_sig, dst_local_sig)
        and _list_equal(src_replica_sig, dst_replica_sig)
    )


_LOCAL_CASE_VIEW_FULL = "view_full"
_LOCAL_CASE_VIEW_SLICED = "view_sliced"
_LOCAL_CASE_THREAD_WISE = "thread_wise"


def _classify_cast_local_case(
    dst_region: BufferRegion,
    src_region: BufferRegion,
    sctx: ScheduleContext,
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
    sctx: ScheduleContext,
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
        if _get_local_region(layout, list(buf.shape), st, ext) is None:
            return False

    # Slice → Canonicalize
    src_sliced = _get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
    dst_sliced = _get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)

    src_can = src_sliced.canonicalize() if hasattr(src_sliced, "canonicalize") else src_sliced
    dst_can = dst_sliced.canonicalize() if hasattr(dst_sliced, "canonicalize") else dst_sliced

    src_sig = _layout_signature(src_can)
    dst_sig = _layout_signature(dst_can)

    # Here check the canonicalized layouts are semantically equal.
    if not _sig_equal(analyzer, src_sig, dst_sig):
        return False

    # Resolve thread vars once; reuse for launch count check
    thread_vars_list = []
    thr_extents = []
    for it in src_sliced.shard:
        if it.axis.is_thread():
            var = _resolve_thread_var(it.axis, sctx)
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


def _resolve_thread_var(axis, sctx):
    """Map the axis to the corresponding thread variable."""
    axis_name = getattr(axis, "name", None)
    if not axis_name:
        try:
            axis_name = str(axis)
        except Exception:
            axis_name = ""

    for key, itervar in sctx.launch_params.items():
        if getattr(itervar.var, "name", "") == axis_name:
            return itervar.var

    if axis_name:
        axis_name_lower = axis_name.lower()
        for key in sctx.launch_params:
            if axis_name_lower in key.lower() or (axis_name == "tx" and "threadIdx.x" in key):
                return sctx.launch_params[key].var

    if "threadIdx.x" in sctx.launch_params:
        return sctx.launch_params["threadIdx.x"].var

    return None


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
    sctx: ScheduleContext,
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if sctx.exec_scope.name not in ["warp", "warpgroup", "cta", "cluster"]:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for local-view cast")

    local_case = _classify_cast_local_case(dst_buffer_region, src_buffer_region, sctx)

    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    dst_local_info = _get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
    if not dst_local_info:
        fail("dst layout is not supported for local-view cast")

    if local_case == _LOCAL_CASE_VIEW_FULL:
        return _emit_cast_local_view_full(dst_buffer_region, src_buffer_region, dst_local_info)

    # view_sliced
    src_st, src_extent = get_st_extent(src_buffer_region)
    src_local_info = _get_local_region(src.layout, list(src.shape), src_st, src_extent)
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
    sctx: ScheduleContext,
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
                    OpCall.downcast(op).output, OpCall.downcast(op).input, sctx
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
def cast_dispatch_local_view(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    op_call = OpCall.downcast(op_call)
    return cast_local_view_impl(op_call.output, op_call.input, sctx)


@register_dispatch(
    "cast",
    "cuda",
    variant="thread_wise",
    priority=10,
    when=[
        predicate(
            "validate_cast_op",
            lambda op, sctx: (
                validate_cast_op(OpCall.downcast(op).output, OpCall.downcast(op).input, sctx),
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
def cast_dispatch_thread_wise(op_call: OpCall, sctx: ScheduleContext) -> PrimFunc:
    op_call = OpCall.downcast(op_call)
    return cast_thread_wise_impl(op_call.output, op_call.input, sctx)
