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

"""Implementation of reduction operator dispatchs for CUDA targets.

Registered ops: sum, max, min.
Each op gets three dispatch variants: optimized SM100+ (priority=20),
shared-memory (priority=10), and local-memory (priority=10).
See the registration block at the bottom of this file for detailed dispatch
documentation with before/after IR examples.
"""

import functools
import math
import operator
import re
from typing import Any

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as Tx
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.layout import TileLayout, laneid
from tvm.tir.stmt import OpCall
from tvm.tirx.op_dispatch import DispatchContext, fail
from tvm.tirx.op_dispatch.dispatcher import predicate, register_dispatch

from ..common import ReduceOpType
from .common import get_indices, get_st_extent, next_power_of_2
from .layout_utils import (
    get_local_region,
    get_sublayout_from_region,
)

reduce_op_table = {
    ReduceOpType.SUM: lambda a, b: a + b,
    ReduceOpType.MAX: Tx.max,
    ReduceOpType.MIN: Tx.min,
}


def reduce_default_value_table(dtype):
    return {
        ReduceOpType.SUM: 0.0,
        ReduceOpType.MAX: Tx.min_value(dtype),
        ReduceOpType.MIN: Tx.max_value(dtype),
    }


def _reduction_args(op: OpCall) -> tuple[BufferRegion, BufferRegion, tuple[int, ...], bool, dict]:
    """Parse ReduceOp -> (dst, src, reduce_axes, accum, config)."""
    op = OpCall.downcast(op)
    dst = op.output
    src = op.input
    reduce_axes = tuple(int(a) for a in op.reduce_axes)
    accum = op.accum
    config = op.config
    return dst, src, reduce_axes, accum, config


def _match_reduction_storage_scope(
    op: OpCall,
    sctx: DispatchContext,
    expected_scope: list[str],
) -> tuple[bool, str | None]:
    """Check that dst and src scopes match one of the expected patterns."""
    op = OpCall.downcast(op)
    dst_scope = op.output.buffer.scope()
    src_scope = op.input.buffer.scope()

    def _check(scope: str, pattern: str) -> bool:
        if pattern.endswith("*"):
            return scope.startswith(pattern[:-1])
        return scope == pattern

    ok = any(_check(dst_scope, p) and _check(src_scope, p) for p in expected_scope)
    msg = f"storage scope mismatch: dst {dst_scope}, src {src_scope}; expected {expected_scope}"
    return (ok, None if ok else msg)


def _analyze_axes(src_ndim: int, reduce_axes: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Normalize negative axes -> (reduce_dim_set, spatial_dim_list)."""
    reduce_dims = set()
    for ax in reduce_axes:
        a = ax if ax >= 0 else ax + src_ndim
        assert 0 <= a < src_ndim, f"reduce axis {ax} out of range for ndim={src_ndim}"
        reduce_dims.add(a)
    spatial_dims = [d for d in range(src_ndim) if d not in reduce_dims]
    return sorted(reduce_dims), spatial_dims


def _analyze_layout_dims(layout, shape):
    """layout.group(shape) -> decompose each dim into thread/local iters.

    Returns list of per-dim (thread_extent, local_extent, thread_strides):
        thread_extent = product of thread iter extents in this dim
        local_extent  = product of local iter extents in this dim
        thread_strides = list of (stride, extent) for thread iters in this dim
    """
    grouped, seps = layout.group(list(shape))
    result = []
    for d in range(len(shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        thread_extent = 1
        local_extent = 1
        thread_strides = []
        for s_idx in shard_range:
            it = grouped.shard[s_idx]
            if it.axis.is_thread():
                thread_extent *= it.extent
                thread_strides.append((it.stride, it.extent))
            else:
                local_extent *= it.extent
        result.append((thread_extent, local_extent, thread_strides))
    return result


def _compute_shuffle_masks(dim_info, reduce_dims: set[int]) -> list[int]:
    """From reduction dims' thread iter (stride, extent) pairs, compute XOR masks.

    For each thread iter in a reduction dim:
        masks += [stride * 2^i for i in range(log2(extent))]
    Sorted ascending.
    """
    masks = []
    for d in reduce_dims:
        _, _, thread_strides = dim_info[d]
        for stride, extent in thread_strides:
            ext_int = int(extent) if hasattr(extent, "__int__") else extent
            n_bits = int(math.log2(ext_int))
            for i in range(n_bits):
                stride_int = int(stride) if hasattr(stride, "__int__") else stride
                masks.append(stride_int * (1 << i))
    masks.sort()
    return masks


def _build_local_dim_map(layout, buffer_shape):
    """Map original dim index to position in get_local_region output (None if pure-thread)."""
    grouped, seps = layout.group(list(buffer_shape))
    dim_map = {}
    local_pos = 0
    for d in range(len(buffer_shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        has_local = any(not grouped.shard[s].axis.is_thread() for s in shard_range)
        if has_local:
            dim_map[d] = local_pos
            local_pos += 1
        else:
            dim_map[d] = None
    return dim_map


def _validate_reduction_layout(
    src_layout,
    dst_layout,
    src_shape,
    dst_shape,
    reduce_dims: list[int],
) -> tuple[bool, str | None]:
    """Validate that spatial dims of src/dst have matching thread+local structure,
    and that reduction dims in dst have local_extent == 1.
    """
    src_dim_info = _analyze_layout_dims(src_layout, src_shape)
    dst_dim_info = _analyze_layout_dims(dst_layout, dst_shape)
    analyzer = Analyzer()

    # Spatial dims: src/dst must match in both thread and local extents.
    # Reduce dims: src/dst thread extent must match, and dst local extent must be 1.

    # get expected simplified dst layout
    expected_dst_dim = []
    for src_idx in range(len(src_shape)):
        if analyzer.can_prove_equal(src_dim_info[src_idx][0], 1) and analyzer.can_prove_equal(
            src_dim_info[src_idx][1], 1
        ):
            continue  # skip if extent=1
        if src_idx in reduce_dims:  # reduce dims
            if not analyzer.can_prove_equal(src_dim_info[src_idx][0], 1):
                expected_dst_dim.append((src_dim_info[src_idx][0], 1))
        else:  # spatial dims
            expected_dst_dim.append((src_dim_info[src_idx][0], src_dim_info[src_idx][1]))

    # check dst layout
    check_idx = 0
    for dst_idx in range(len(dst_shape)):
        if analyzer.can_prove_equal(dst_dim_info[dst_idx][0], 1) and analyzer.can_prove_equal(
            dst_dim_info[dst_idx][1], 1
        ):
            continue
        if not (
            analyzer.can_prove_equal(dst_dim_info[dst_idx][0], expected_dst_dim[check_idx][0])
            and analyzer.can_prove_equal(dst_dim_info[dst_idx][1], expected_dst_dim[check_idx][1])
        ):
            return False, "mismatch dst/src layout for reduction"
        check_idx += 1
    if check_idx != len(expected_dst_dim):
        return False, "mismatch dst/src layout for reduction"
    return True, None


def validate_reduction_shared(
    op: OpCall,
    sctx: DispatchContext,
) -> tuple[bool, str | None]:
    """Validate reduction in shared memory."""
    if sctx.exec_scope.name not in ["cta", "warpgroup", "warp", "thread"]:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for shared reduction"

    op = OpCall.downcast(op)
    dst, src = op.output.buffer, op.input.buffer
    if not (src.scope().startswith("shared") and dst.scope().startswith("shared")):
        return False, "expected shared scope for both src and dst"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"

    if "threadIdx.x" not in sctx.launch_params:
        return False, "threadIdx.x not in launch_params"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return False, "multi-dimensional thread binding not supported for shared reduction"

    reduce_axes = tuple(int(a) for a in op.reduce_axes)
    src_region = op.input.region
    dst_region = op.output.region
    src_ndim = len(src_region)
    try:
        reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)
    except AssertionError as e:
        return False, str(e)

    # Validate dst shape matches spatial dims of src
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    expected_dst_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    actual_dst_len = functools.reduce(operator.mul, dst_extent, 1)
    analyzer = Analyzer()
    if not analyzer.can_prove_equal(expected_dst_len, actual_dst_len):
        return (
            False,
            f"dst size {actual_dst_len} != expected spatial size {expected_dst_len}",
        )

    return True, None


def _analyze_shuffle_reduce(src_layout, dst_layout):
    """Analyze src/dst layouts for laneid shard→replica reduce pattern.

    Returns (reduce_width, local_elems) if the pattern matches, or None.
    - reduce_width: number of lanes participating in each group's reduction
    - local_elems: per-thread element count (product of non-laneid shard extents)
    """
    if src_layout.is_swizzle() or dst_layout.is_swizzle():
        return None

    src_canon = src_layout.canonicalize()
    dst_canon = dst_layout.canonicalize()

    # Extract laneid iters from shard and replica
    src_laneid_shard = [it for it in src_canon.shard if it.axis == laneid]
    dst_laneid_replica = [it for it in dst_canon.replica if it.axis == laneid]

    # src shard must contain laneid (data distributed across lanes)
    if not src_laneid_shard:
        return None
    # dst replica must contain laneid (result broadcast to lanes)
    if not dst_laneid_replica:
        return None

    # laneid span must be 32 (full warp)
    src_laneid_span = 1 + sum(abs(int(it.stride)) * (int(it.extent) - 1) for it in src_laneid_shard)
    if src_laneid_span != 32:
        return None

    reduce_width = functools.reduce(operator.mul, [int(it.extent) for it in dst_laneid_replica], 1)
    if reduce_width <= 0 or reduce_width > 32 or (reduce_width & (reduce_width - 1)) != 0:
        return None  # must be power of 2

    # local_elems = product of non-laneid shard extents in src
    src_non_laneid = [it for it in src_canon.shard if it.axis != laneid]
    local_elems = functools.reduce(operator.mul, [int(it.extent) for it in src_non_laneid], 1)

    return reduce_width, local_elems


def _gen_warp_shuffle_reduce(src, dst, reduce_width, local_elems, accum, op_func, init_value):
    """Generate warp shuffle reduce codegen for laneid shard→replica pattern.

    Unified for both full warp (reduce_width=32) and partial warp (e.g. reduce_width=8).
    """
    num_steps = int(math.log2(reduce_width))
    is_same_buffer = src.same_as(dst)

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            src_local = src.local(local_elems)
            dst_local = dst.local(local_elems)
            for k in Tx.serial(local_elems):
                if not is_same_buffer:
                    dst_local[k] = src_local[k]
                row_var = Tx.meta_var(dst_local[k])
                for step in Tx.unroll(num_steps):
                    xor_mask = Tx.meta_var(reduce_width >> (step + 1))
                    dst_local[k] = op_func(row_var, Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, xor_mask, 32, 32))  # noqa: E501
    # fmt: on

    return impl


def validate_reduction_local(
    op: OpCall,
    sctx: DispatchContext,
) -> tuple[bool, str | None]:
    """Validate reduction in local memory."""
    op = OpCall.downcast(op)
    dst_br, src_br = op.output, op.input
    dst, src = dst_br.buffer, src_br.buffer

    if not (src.scope() == "local" and dst.scope() == "local" and sctx.is_cuda()):
        return False, "expected local scope and CUDA target"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"

    if sctx.exec_scope.name == "thread":
        return True, None  # thread-wise reduction
    elif sctx.exec_scope.name == "warp":
        # VIEW: need layouts and layout analysis
        if not (src.layout and dst.layout):
            return False, "layouts required for view-based local reduction"
        if not (isinstance(src.layout, TileLayout) and isinstance(dst.layout, TileLayout)):
            return False, "TileLayout required for view-based local reduction"
        if src.layout.is_swizzle() or dst.layout.is_swizzle():
            return False, "swizzle layout unsupported for local reduction"

        analyzer = Analyzer()

        # Validate get_local_region succeeds for both
        src_st, src_extent = get_st_extent(src_br)
        dst_st, dst_extent = get_st_extent(dst_br)

        # Check for laneid shard→replica shuffle reduce pattern first.
        # This pattern has laneid in dst replica (broadcast), which the
        # general validation below would reject.
        shuffle_info = _analyze_shuffle_reduce(src.layout, dst.layout)
        if shuffle_info is not None:
            return True, None

        for layout, buf, st, ext, name in [
            (src.layout, src, src_st, src_extent, "src"),
            (dst.layout, dst, dst_st, dst_extent, "dst"),
        ]:
            for it in layout.shard:
                if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                    return False, f"thread dim with zero stride in {name}"
            replica = getattr(layout, "replica", None) or []
            if any(it.axis.is_thread() for it in replica):
                return False, f"thread axis in replica for {name}"
            if get_local_region(layout, list(buf.shape), st, ext) is None:
                return False, f"get_local_region failed for {name}"

        # Validate layout compatibility
        # Spatial dims match, reduce dims in dst have local_extent==1
        reduce_axes = tuple(int(a) for a in op.reduce_axes)
        src_ndim = len(src_br.region)
        try:
            reduce_dims, _ = _analyze_axes(src_ndim, reduce_axes)
        except AssertionError as e:
            return False, str(e)
        src_sliced = get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
        dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)
        ok, msg = _validate_reduction_layout(
            src_sliced, dst_sliced, list(src_extent), list(dst_extent), reduce_dims
        )
        return ok, msg
    else:
        return False, f"unsupported exec_scope {sctx.exec_scope.name} for local reduction"


def build_src_indices(spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st):
    """Combine spatial and reduction indices into full src index tuple."""

    # Build index helpers that work with the explicit axis split
    def get_spatial_or_reduction_src_indices(spa_or_red_fused, is_spatial):
        dims = spatial_dims if is_spatial else reduce_dims
        spa_extents = [src_extent[d] for d in dims]
        indices = []
        rem = spa_or_red_fused
        for e in reversed(spa_extents):
            indices.append(rem % e)
            rem //= e
        indices.reverse()
        return [idx + src_st[d] for idx, d in zip(indices, dims)]

    spa_vals = get_spatial_or_reduction_src_indices(spa_fused, is_spatial=True)
    red_vals = get_spatial_or_reduction_src_indices(red_fused, is_spatial=False)
    full = [None] * len(src_extent)
    for i, d in enumerate(spatial_dims):
        full[d] = spa_vals[i]
    for i, d in enumerate(reduce_dims):
        full[d] = red_vals[i]
    return full


def _emit_reduction_shared_cta(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    exec_scope_name = sctx.exec_scope.name

    def get_thread_cnt():
        if exec_scope_name == "cta":
            return sctx.launch_params["threadIdx.x"].dom.extent
        elif exec_scope_name == "warpgroup":
            return 128
        elif exec_scope_name == "warp":
            return 32
        elif exec_scope_name == "thread":
            return 1

    thread_cnt = get_thread_cnt()
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    dtype = src.dtype

    # Compute spatial/reduction from the explicit axes
    spatial_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[d] for d in reduce_dims], 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    # Adaptive group size: nearest power-of-2 for reduction length, capped at warp size and thread count. # noqa: E501
    group_size = min(next_power_of_2(int(reduction_len)), 32, int(thread_cnt))
    group_size = max(group_size, 1)  # ensure at least 1
    n_shuffles = int(math.log2(group_size)) if group_size > 1 else 0
    spatial_par = int(thread_cnt) // group_size

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        elif exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        elif exec_scope_name == "thread":
            return 0

    def shuffle_data(thread_data):
        @Tx.inline
        def inner_shuffle(mask, v, shuffle_mask):
            v[0] = op_func(v[0], Tx.tvm_warp_shuffle_xor(mask, v[0], shuffle_mask, group_size, 32))

        if n_shuffles > 0:
            mask = Tx.tvm_warp_activemask()
            for i in range(n_shuffles):
                inner_shuffle(mask, thread_data, 1 << i)

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

    # fmt: off
    @Tx.prim_func(tirx=True)
    def impl():
        tid_in_scope = get_tid_in_scope()
        thread_data = Tx.alloc_buffer([1], dtype=dtype, scope="local")
        group_id = Tx.meta_var(Tx.floordiv(tid_in_scope, group_size))
        lane_in_grp = Tx.meta_var(tid_in_scope % group_size)
        for step in Tx.serial(Tx.ceildiv(spatial_len, spatial_par)):
            spa_fused = Tx.meta_var(step * spatial_par + group_id)
            if spa_fused < spatial_len:
                thread_data[0] = init_value
                for t in Tx.serial(Tx.ceildiv(reduction_len, group_size)):
                    red_fused = Tx.meta_var(t * group_size + lane_in_grp)
                    if red_fused < reduction_len:
                        src_indices = Tx.meta_var(build_src_indices(spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st))  # noqa: E501
                        thread_data[0] = op_func(thread_data[0], src[tuple(src_indices)])
                shuffle_data(thread_data)
                if lane_in_grp == 0:
                    dst_indices = Tx.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                    dst[tuple(dst_indices)] = Tx.if_then_else(Tx.bool(accum), op_func(dst[tuple(dst_indices)], thread_data[0]), thread_data[0])  # noqa: E501

        sync()
    # fmt: on

    return impl


def _emit_reduction_shared_thread(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    dtype = src.dtype

    # Compute spatial/reduction from the explicit axes
    spatial_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[d] for d in reduce_dims], 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    @Tx.prim_func(tirx=True)
    def impl():
        for spa_fused in Tx.serial(spatial_len):
            dst_indices = Tx.meta_var(get_indices(spa_fused, dst_st, dst_extent))
            if not accum:
                dst[tuple(dst_indices)] = init_value
            for red_fused in Tx.serial(reduction_len):
                src_indices = Tx.meta_var(
                    build_src_indices(
                        spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st
                    )
                )
                dst[tuple(dst_indices)] = op_func(dst[tuple(dst_indices)], src[tuple(src_indices)])

    return impl


def _emit_reduction_local_thread_wise(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    dtype = src.dtype
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    src_ndim = len(src_extent)
    spa_extents = [src_extent[d] for d in spatial_dims]
    red_extents = [src_extent[d] for d in reduce_dims]
    spatial_len = functools.reduce(operator.mul, spa_extents, 1)
    reduction_len = functools.reduce(operator.mul, red_extents, 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    def get_src_indices(spa_fused, red_fused):
        spa_indices = []
        rem = spa_fused
        for e in reversed(spa_extents):
            spa_indices.append(rem % e)
            rem //= e
        spa_indices.reverse()

        red_indices = []
        rem = red_fused
        for e in reversed(red_extents):
            red_indices.append(rem % e)
            rem //= e
        red_indices.reverse()

        full = [None] * src_ndim
        for i, d in enumerate(spatial_dims):
            full[d] = spa_indices[i] + src_st[d]
        for i, d in enumerate(reduce_dims):
            full[d] = red_indices[i] + src_st[d]
        return full

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            for spa in Tx.serial(spatial_len):
                dst_idx = Tx.meta_var(get_indices(spa, dst_st, dst_extent))
                if not accum:
                    dst[tuple(dst_idx)] = init_value
                for red in Tx.serial(reduction_len):
                    src_idx = Tx.meta_var(get_src_indices(spa, red))
                    dst[tuple(dst_idx)] = op_func(dst[tuple(dst_idx)], src[tuple(src_idx)])
    # fmt: on

    return impl


def _emit_reduction_local_view(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    config: dict[str, Any],
    reduce_dims: set[int],
    spatial_dims: list[int],
    src_local_info,
    dst_local_info,
    shuffle_masks: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    dtype = src.dtype

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    src_local_shape, src_local_st, src_local_ext = src_local_info
    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info

    # Build maps from original dim index to position in get_local_region output
    src_dim_map = _build_local_dim_map(src.layout, list(src.shape))
    dst_dim_map = _build_local_dim_map(dst.layout, list(dst.shape))

    # Only include reduction dims that have local parts in src
    src_ndim = len(src_br.region)
    reduce_local_dims = [d for d in reduce_dims if src_dim_map[d] is not None]
    reduction_local_ext = [src_local_ext[src_dim_map[d]] for d in reduce_local_dims]
    reduction_local_st = [src_local_st[src_dim_map[d]] for d in reduce_local_dims]

    reduction_local_total = functools.reduce(operator.mul, reduction_local_ext, 1)
    dst_local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    def _get_src_local_index(dst_fused, red_fused):
        """Compute src local multi-dim index from dst fused index and reduction fused index."""
        dst_indices = get_indices(dst_fused, dst_local_st, dst_local_ext)
        red_indices = get_indices(red_fused, reduction_local_st, reduction_local_ext)

        # Interleave into src local indices (skipping pure-thread dims)
        src_local = []
        ri = 0
        for d in range(src_ndim):
            if src_dim_map[d] is None:
                continue  # pure-thread in src, not in src.local()
            if d in reduce_dims:
                src_local.append(red_indices[ri])
                ri += 1
            else:
                # Spatial dim: use corresponding dst local position
                src_local.append(dst_indices[dst_dim_map[d]])

        return src_local

    # is_same_buffer = src.same_as(dst)
    shuffle = bool(config.get("thread_reduce", False))
    in_place = dst.same_as(src)

    def shuffle_data(mask, dst_local, dst_idx):
        @Tx.inline
        def inner_shuffle(v, shuffle_mask):
            dst_local[tuple(dst_idx)] = op_func(
                v, Tx.tvm_warp_shuffle_xor(mask, v, shuffle_mask, 32, 32)
            )

        for i in range(len(shuffle_masks)):
            inner_shuffle(dst_local[tuple(dst_idx)], shuffle_masks[i])

    need_save_accum = accum and shuffle

    # fmt: off
    if need_save_accum:
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                src_local = src.local(*src_local_shape)
                dst_local = dst.local(*dst_local_shape)
                old_val = Tx.alloc_buffer([1], dtype, scope="local")

                for spa in Tx.serial(dst_local_total):
                    dst_idx = Tx.meta_var(get_indices(spa, dst_local_st, dst_local_ext))
                    old_val[0] = dst_local[tuple(dst_idx)]
                    if not in_place:
                        dst_local[tuple(dst_idx)] = init_value
                        for red in Tx.serial(reduction_local_total):
                            src_idx = Tx.meta_var(_get_src_local_index(spa, red))
                            dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], src_local[tuple(src_idx)])  # noqa: E501
                    if shuffle:
                        mask = Tx.tvm_warp_activemask()
                        shuffle_data(mask, dst_local, dst_idx)
                    dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], old_val[0])
    else:
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                src_local = src.local(*src_local_shape)
                dst_local = dst.local(*dst_local_shape)

                for spa in Tx.serial(dst_local_total):
                    dst_idx = Tx.meta_var(get_indices(spa, dst_local_st, dst_local_ext))
                    if not in_place:
                        if not accum:
                            dst_local[tuple(dst_idx)] = init_value
                        for red in Tx.serial(reduction_local_total):
                            src_idx = Tx.meta_var(_get_src_local_index(spa, red))
                            dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], src_local[tuple(src_idx)])  # noqa: E501
                    if shuffle:
                        mask = Tx.tvm_warp_activemask()
                        shuffle_data(mask, dst_local, dst_idx)
    # fmt: on

    return impl


def _emit_reduction_local_thread_packed_add_sum(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc:
    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    [r.extent for r in dst_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            local_sum = Tx.alloc_buffer([8], dtype, scope="local")
            # First pass: copy first 8 elements (with optional accumulator)
            for i in Tx.unroll(8):
                if accum and i == 0:
                    local_sum[i] = src[src_base + i] + dst[tuple(dst_st)]
                else:
                    local_sum[i] = src[src_base + i]

            # Process remaining full chunks of 8
            for outer in Tx.serial(num_full_chunks - 1):
                for j in Tx.unroll(4):
                    Tx.ptx.add_packed_f32x2(
                        local_sum[2 * j],
                        local_sum[2 * j + 1],
                        src[src_base + 8 * (outer + 1) + 2 * j],
                        src[src_base + 8 * (outer + 1) + 2 * j + 1],
                        Tx.address_of(local_sum[2 * j]),
                    )

            # Handle remainder elements (0 to 7)
            for i in Tx.serial(remainder):
                local_sum[0] = local_sum[0] + src[src_base + remainder_base + i]

            # Final packed add sum: 8 -> 4 -> 2 -> 1
            Tx.ptx.add_packed_f32x2(
                local_sum[0], local_sum[1],
                local_sum[2], local_sum[3],
                Tx.address_of(local_sum[0]),
            )
            Tx.ptx.add_packed_f32x2(
                local_sum[4], local_sum[5],
                local_sum[6], local_sum[7],
                Tx.address_of(local_sum[4]),
            )
            Tx.ptx.add_packed_f32x2(
                local_sum[0], local_sum[1],
                local_sum[4], local_sum[5],
                Tx.address_of(local_sum[0]),
            )
            dst[tuple(dst_st)] = local_sum[0] + local_sum[1]
    # fmt: on

    return impl


def _emit_reduction_local_thread_3input_maxmin(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc:
    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    op_func = reduce_op_table[reduce_op]
    reduce3_func = (
        Tx.ptx.reduce3_max_f32 if reduce_op == ReduceOpType.MAX else Tx.ptx.reduce3_min_f32
    )

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            temp = Tx.alloc_buffer([4], dtype, scope="local")
            # First pass: process first 8 elements into 4 temps
            for i in Tx.unroll(4):
                if accum and i == 0:
                    temp[i] = reduce3_func(src[src_base + 2 * i], src[src_base + 2 * i + 1], dst[tuple(dst_st)])  # noqa: E501
                else:
                    temp[i] = op_func(src[src_base + 2 * i], src[src_base + 2 * i + 1])

            # Process remaining full chunks of 8
            for outer in Tx.serial(num_full_chunks - 1):
                for i in Tx.unroll(4):
                    temp[i] = reduce3_func(
                        temp[i],
                        src[src_base + 8 * (outer + 1) + 2 * i],
                        src[src_base + 8 * (outer + 1) + 2 * i + 1],
                    )

            # Process remainder elements (0 to 7 elements)
            for i in Tx.serial(remainder):
                temp[0] = op_func(temp[0], src[src_base + remainder_base + i])

            # Final merge: combine 4 temps into result
            dst[tuple(dst_st)] = op_func(temp[0], temp[1])
            dst[tuple(dst_st)] = reduce3_func(dst[tuple(dst_st)], temp[2], temp[3])
    # fmt: on

    return impl


def reduction_shared_impl(
    op: OpCall,
    op_type: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    dst_br, src_br, reduce_axes, accum, config = _reduction_args(op)
    src_ndim = len(src_br.region)
    reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)
    if sctx.exec_scope.name in ["cta", "warpgroup", "warp"]:
        return _emit_reduction_shared_cta(
            dst_br, src_br, accum, op_type, sctx, reduce_dims, spatial_dims
        )
    elif sctx.exec_scope.name == "thread":
        return _emit_reduction_shared_thread(
            dst_br, src_br, accum, op_type, sctx, reduce_dims, spatial_dims
        )
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for reduction_shared_impl")


def reduction_local_impl(
    op: OpCall,
    op_type: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc | None:
    dst_br, src_br, reduce_axes, accum, config = _reduction_args(op)
    src_ndim = len(src_br.region)
    reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)

    if sctx.exec_scope.name == "thread":
        return _emit_reduction_local_thread_wise(
            dst_br, src_br, accum, op_type, reduce_dims, spatial_dims
        )
    elif sctx.exec_scope.name == "warp":
        src = src_br.buffer
        dst = dst_br.buffer

        # --- Try laneid shard→replica shuffle reduce ---
        shuffle_info = _analyze_shuffle_reduce(src.layout, dst.layout)
        if shuffle_info is not None:
            reduce_width, local_elems = shuffle_info
            op_func = reduce_op_table.get(op_type)
            if op_func is None:
                fail(f"unsupported reduce op: {op_type}")
            dtype = src.dtype
            init_value = reduce_default_value_table(dtype).get(op_type)
            return _gen_warp_shuffle_reduce(
                src, dst, reduce_width, local_elems, accum, op_func, init_value
            )

        # --- Existing WGMMA layout path below ---
        src_st, src_extent = get_st_extent(src_br)
        dst_st, dst_extent = get_st_extent(dst_br)

        src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
        dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
        assert src_local_info is not None and dst_local_info is not None

        src_dim_info = _analyze_layout_dims(src.layout, list(src.shape))
        shuffle_masks = (
            _compute_shuffle_masks(src_dim_info, reduce_dims)
            if config.get("thread_reduce", False)
            else []
        )

        return _emit_reduction_local_view(
            dst_br,
            src_br,
            accum,
            op_type,
            config,
            reduce_dims,
            spatial_dims,
            src_local_info,
            dst_local_info,
            shuffle_masks,
        )
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for reduction_local_impl")


def _exec_scope_ok(op: OpCall, sctx: DispatchContext, expected_scopes: list[str]):
    ok = sctx.exec_scope.name in expected_scopes
    return (ok, None if ok else f"exec_scope {sctx.exec_scope.name} not in {expected_scopes}")


def _dtype_ok(op: OpCall, sctx: DispatchContext, expected_dtype: str):
    op = OpCall.downcast(op)
    dtype = op.input.buffer.dtype
    ok = dtype == expected_dtype
    return (ok, None if ok else f"dtype {dtype} != {expected_dtype}")


def _sm_version_ok(op: OpCall, sctx: DispatchContext, min_version: int):
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    ok = sm_version >= min_version
    return (ok, None if ok else f"sm_version {sm_version} < {min_version}")


def _reduction_len_ok(op: OpCall, sctx: DispatchContext, min_len: int):
    op = OpCall.downcast(op)
    src_extent = [r.extent for r in op.input.region]
    reduction_len = functools.reduce(operator.mul, src_extent, 1)
    ok = reduction_len >= min_len
    return (ok, None if ok else f"reduction_len {reduction_len} < {min_len}")


def _dst_len_ok(op: OpCall, sctx: DispatchContext, expected_len: int):
    op = OpCall.downcast(op)
    dst_extent = [r.extent for r in op.output.region]
    dst_len = functools.reduce(operator.mul, dst_extent, 1)
    ok = dst_len == expected_len
    return (ok, None if ok else f"dst_len {dst_len} != {expected_len}")


def _src_ndim_ok(op: OpCall, sctx: DispatchContext, expected_ndim: int):
    op = OpCall.downcast(op)
    src_extent = [r.extent for r in op.input.region]
    ok = len(src_extent) == expected_ndim
    return (ok, None if ok else f"src ndim {len(src_extent)} != {expected_ndim}")


def _local_scope_match(op: OpCall, sctx: DispatchContext):
    op = OpCall.downcast(op)
    src, dst = op.input.buffer, op.output.buffer
    ok = all(
        [src.scope() == "local", dst.scope() == "local", src.dtype == dst.dtype, sctx.is_cuda()]
    )
    if not ok:
        return (False, "src/dst must be local scope with matching dtype on CUDA")
    return (True, None)


_optimized_local_reduction_predicates = [
    predicate("exec_scope", _exec_scope_ok, expected_scopes=["thread"]),
    predicate("local_scope", _local_scope_match),
    predicate("dst_len", _dst_len_ok, expected_len=1),
    predicate("src_ndim", _src_ndim_ok, expected_ndim=1),
    predicate("dtype", _dtype_ok, expected_dtype="float32"),
    predicate("sm_version", _sm_version_ok, min_version=100),
    predicate("reduction_len", _reduction_len_ok, min_len=8),
]


def _sm100_packed_add_sum_impl(op: OpCall, op_type: ReduceOpType, sctx: DispatchContext):
    op = OpCall.downcast(op)
    return _emit_reduction_local_thread_packed_add_sum(op.output, op.input, op.accum, op_type, sctx)


def _sm100_3input_maxmin_impl(op: OpCall, op_type: ReduceOpType, sctx: DispatchContext):
    op = OpCall.downcast(op)
    return _emit_reduction_local_thread_3input_maxmin(op.output, op.input, op.accum, op_type, sctx)


_optimized_impl_table = {
    ReduceOpType.SUM: ("packed_add_sum", _sm100_packed_add_sum_impl),
    ReduceOpType.MAX: ("3input_maxmin", _sm100_3input_maxmin_impl),
    ReduceOpType.MIN: ("3input_maxmin", _sm100_3input_maxmin_impl),
}


# ---------------------------------------------------------------------------
# Registration: bind each reduction op name to its CUDA schedule candidates.
# ---------------------------------------------------------------------------
#
# === Variant: {sum→"packed_add_sum", max/min→"3input_maxmin"} (priority=20) ===
#
# When: thread scope, all local buffers, float32, 1D src with len >= 8,
# SM100+ (uses packed PTX instructions not available on older GPUs).
#
# Before (OpCall — sum example):
#     with Tx.thread():
#         Tx.sum(dst_local[0:1], src_local[0:32])   # float32, reduce 32 → 1
#
# After — packed_add_sum (uses add.f32x2 to reduce pairs):
#     with Tx.thread():
#         # Iteratively reduce: 32 → 16 → 8 → 4 → 2 → 1
#         # Each step: add.f32x2 combines adjacent pairs
#         for i in Tx.serial(16):
#             Tx.cuda.func_call("add_f32x2", &buf[i*2], &buf[i*2], &buf[i*2+2])
#         # ... repeat halving until scalar result
#         dst_local[0] = buf[0]
#
# After — 3input_maxmin (uses 3-input PTX max/min):
#     with Tx.thread():
#         # Tree reduction with 3-input instructions:
#         # max(a, b, c) in one PTX instruction
#         for i in Tx.serial(n // 3):
#             Tx.cuda.func_call("max3_f32", &buf[i*3], &buf[i*3+1], &buf[i*3+2])
#
# With accum=True: accumulator folded into first element/pair of the reduction.
#
# === Variant: "shared" (priority=10) ===
#
# When: dst and src are both shared-memory buffers, exec scope is one of
# {cta, warpgroup, warp, thread}, threadIdx.x bound, reduce axes valid.
#
# (A) CTA/warpgroup/warp scope — adaptive-group shuffle tree
#     (_emit_reduction_shared_cta):
#     group_size = min(next_power_of_2(reduction_len), 32).
#     Each group of threads reduces one spatial position via shfl_xor.
#
# Before:
#     with Tx.cta():
#         Tx.sum(B_smem[0:4], A_smem[0:4, 0:8], [-1], False)
#
# After (scheduled PrimFunc, group_size=8, spatial_par=4):
#     thread_data[0] = Tx.float32(0.0)
#     thread_data[0] = thread_data[0] + A_smem[tid_in_scope]  # gather
#     # log2(8) = 3 shuffle-xor steps with width=8
#     thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 1, 8, 32)
#     thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 2, 8, 32)
#     thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 4, 8, 32)
#     if tid_in_scope % 8 == 0:
#         B_smem[tid_in_scope // 8] = thread_data[0]
#
# (B) Thread scope — sequential loop (_emit_reduction_shared_thread):
#
# Before:
#     with Tx.thread()[65:66]:
#         Tx.sum(B_smem[0:4], A_smem[0:4, 0:8], [-1], False)
#
# After (scheduled PrimFunc):
#     for spa in range(4):
#         B_smem[spa] = Tx.float32(0.0)                       # init (skipped if accum)
#         for red in range(8):
#             B_smem[spa] = B_smem[spa] + A_smem[spa * 8 + red]
#
# === Variant: "local" (priority=10) ===
#
# When: dst and src are both local-scope buffers with matching dtype, on CUDA.
#
# (A) Thread scope — sequential per-element reduction
#     (_emit_reduction_local_thread_wise):
#
# Before:
#     with Tx.thread():
#         Tx.sum(B_local[0:2, 0:3], A_local[0:2, 0:3, 0:4], [-1], False)
#
# After (scheduled PrimFunc, spatial_len=6, reduction_len=4):
#     for spa in range(6):
#         B_local[spa] = Tx.float32(0.0)                      # init (skipped if accum)
#         for red in range(4):
#             B_local[spa] = B_local[spa] + A_local[spa * 4 + red]
#
# (B) Warp scope — layout-driven reduction with optional shfl_xor
#     (_emit_reduction_local_view):
#     Requires TileLayout with valid thread-partition. Decomposes layout to
#     identify thread-local elements, then optionally shuffles partial sums.
#
#     thread_reduce=False: local-only, no shuffle.
#     thread_reduce=True: local reduction + cross-thread shfl_xor steps.
#     accum=True + shuffle: saves old dst before reduce+shuffle, combines after.
#
# Before:
#     with Tx.warp():
#         Tx.sum(red_view[0:16, 0:4], acc_view[0:16, 0:128], [-1], False,
#                thread_reduce=True)
#
# After (scheduled PrimFunc, local_total=2, local_red=32, 2 shuffle steps):
#     src_local = acc_view.view(64)
#     dst_local = red_view.view(2)
#     for spa in range(2):
#         dst_local[spa] = Tx.float32(0.0)
#         for red in range(32):
#             dst_local[spa] = dst_local[spa] + src_local[...]
#         dst_local[spa] = dst_local[spa] + shfl_xor(..., 1, 32, 32)
#         dst_local[spa] = dst_local[spa] + shfl_xor(..., 2, 32, 32)


for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:
    variant_name, optimized_impl = _optimized_impl_table[op_type]

    # Register optimized dispatch (sm_100a+, float32, thread-level local reduction)
    @register_dispatch(
        op_name,
        "cuda",
        variant=variant_name,
        priority=20,
        when=_optimized_local_reduction_predicates,
    )
    def _optimized_dispatch(
        op: OpCall, sctx: DispatchContext, _impl=optimized_impl, _op_type=op_type
    ) -> PrimFunc:
        op = OpCall.downcast(op)
        return _impl(op, _op_type, sctx)

    # Register shared memory dispatch
    @register_dispatch(
        op_name,
        "cuda",
        variant="shared",
        priority=10,
        when=[
            predicate(
                "storage_scope",
                _match_reduction_storage_scope,
                expected_scope=["shared*"],
            ),
            predicate("shared_valid", validate_reduction_shared),
        ],
    )
    def _shared_dispatch(op: OpCall, sctx: DispatchContext, _op_type=op_type) -> PrimFunc:
        op = OpCall.downcast(op)
        return reduction_shared_impl(op, _op_type, sctx)

    # Register local memory dispatch
    @register_dispatch(
        op_name,
        "cuda",
        variant="local",
        priority=10,
        when=[
            predicate(
                "storage_scope",
                _match_reduction_storage_scope,
                expected_scope=["local"],
            ),
            predicate("local_valid", validate_reduction_local),
        ],
    )
    def _local_dispatch(op: OpCall, sctx: DispatchContext, _op_type=op_type) -> PrimFunc:
        op = OpCall.downcast(op)
        return reduction_local_impl(op, _op_type, sctx)
