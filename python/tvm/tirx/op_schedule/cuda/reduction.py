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

"""Implementation of reduce operator schedules."""

import functools
import operator
import re
from typing import Any, Dict, Optional

from tvm.arith.analyzer import Analyzer
from tvm.script import tir as T
from tvm.tir import BufferRegion, PrimFunc
from tvm.tir.layout import laneid
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule import ScheduleContext, fail, register_dispatch, predicate

from ..common import ReduceOpType

reduce_op_table = {
    ReduceOpType.SUM: lambda a, b: a + b,
    ReduceOpType.MAX: T.max,
    ReduceOpType.MIN: T.min,
}

reduce_default_value_table = lambda dtype: {
    ReduceOpType.SUM: 0.0,
    ReduceOpType.MAX: T.min_value(dtype),
    ReduceOpType.MIN: T.max_value(dtype),
}


# ---------------------------------------------------------------------------
# Predicate functions for dispatch
# ---------------------------------------------------------------------------


def _exec_scope_ok(op: OpCall, sctx: ScheduleContext, expected_scopes: list[str]):
    """Check if exec_scope is in the allowed list."""
    ok = sctx.exec_scope.name in expected_scopes
    return (ok, None if ok else f"exec_scope {sctx.exec_scope.name} not in {expected_scopes}")


def _dtype_ok(op: OpCall, sctx: ScheduleContext, expected_dtype: str):
    """Check if src buffer dtype matches."""
    _, src_buffer_region = op.args[:2]
    dtype = src_buffer_region.buffer.dtype
    ok = dtype == expected_dtype
    return (ok, None if ok else f"dtype {dtype} != {expected_dtype}")


def _sm_version_ok(op: OpCall, sctx: ScheduleContext, min_version: int):
    """Check if SM version >= min_version."""
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    ok = sm_version >= min_version
    return (ok, None if ok else f"sm_version {sm_version} < {min_version}")


def _reduction_len_ok(op: OpCall, sctx: ScheduleContext, min_len: int):
    """Check if reduction_len >= min_len."""
    _, src_buffer_region = op.args[:2]
    src_extent = [r.extent for r in src_buffer_region.region]
    reduction_len = functools.reduce(operator.mul, src_extent, 1)
    ok = reduction_len >= min_len
    return (ok, None if ok else f"reduction_len {reduction_len} < {min_len}")


def _dst_len_ok(op: OpCall, sctx: ScheduleContext, expected_len: int):
    """Check if dst_len == expected_len."""
    dst_buffer_region = op.args[0]
    dst_extent = [r.extent for r in dst_buffer_region.region]
    dst_len = functools.reduce(operator.mul, dst_extent, 1)
    ok = dst_len == expected_len
    return (ok, None if ok else f"dst_len {dst_len} != {expected_len}")


def _src_ndim_ok(op: OpCall, sctx: ScheduleContext, expected_ndim: int):
    """Check if src buffer is expected_ndim-dimensional."""
    _, src_buffer_region = op.args[:2]
    src_extent = [r.extent for r in src_buffer_region.region]
    ok = len(src_extent) == expected_ndim
    return (ok, None if ok else f"src ndim {len(src_extent)} != {expected_ndim}")


def _local_scope_match(op: OpCall, sctx: ScheduleContext):
    """Check if both src and dst are local scope with matching dtype."""
    dst_buffer_region, src_buffer_region = op.args[:2]
    src, dst = src_buffer_region.buffer, dst_buffer_region.buffer
    ok = all([
        src.scope() == "local",
        dst.scope() == "local",
        src.dtype == dst.dtype,
        sctx.is_cuda(),
    ])
    if not ok:
        return (False, "src/dst must be local scope with matching dtype on CUDA")
    return (True, None)


def reduction_cuda_shared_nd_sync_cta_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.

    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """

    # Basic validation checks
    if sctx.exec_scope.name != "cta":
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    thread_cnt = sctx.launch_params["threadIdx.x"].dom.extent
    threads_per_warp = 32
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if not (thread_cnt >= threads_per_warp and thread_cnt % threads_per_warp == 0):
        fail("threadIdx.x must be >=32 and multiple of warp size")

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    dtype = src.dtype

    # Check dst is first m dimensions of src (and the rest dimensions, if exist, are all 1s)
    if not all(
        [
            src.scope().startswith("shared"),
            dst.scope().startswith("shared"),
            len(src_region) >= len(dst_region),
        ]
    ):
        fail("unsupported buffer scopes or region ranks for shared reduction")

    analyzer = Analyzer()
    spatial_dims = -1

    if len(src_extent) == len(dst_extent):
        for i in range(len(dst_extent)):
            if src_extent[i] != dst_extent[i]:
                if dst_extent[i] != 1:
                    fail("dst trailing dims must be 1s beyond spatial dims")
                if not functools.reduce(operator.mul, dst_extent[i:], 1) == 1:
                    fail("dst trailing dims beyond spatial must be 1s")
                else:
                    spatial_dims = i
                    break
        if spatial_dims == -1:
            fail("no reduction dims detected; not a reduction")

    else:
        spatial_dims = len(dst_extent)
        if not all(
            analyzer.can_prove_equal(s, d) for s, d in zip(src_extent[:spatial_dims], dst_extent)
        ):
            fail("dst must match src prefix dims for reduction")

    assert spatial_dims > 0 and spatial_dims < len(src_extent)

    spatial_len = functools.reduce(operator.mul, src_extent[:spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, src_extent[spatial_dims:], 1)

    # get reduce op
    op_func = reduce_op_table.get(reduce_op)
    if op_func is None:
        fail(f"unsupported reduce op: {reduce_op}")

    # get init value if not accum
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    def get_indices(nth, st, extent):
        # example: st: [0, 1], extent: [32, 16], nth: 10
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    # fmt: off
    @T.prim_func(tirx=True)
    def impl():
        warp_cnt = T.meta_var(T.ceildiv(thread_cnt, threads_per_warp))
        for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
            thread_buffer = T.allocate([1], dtype=dtype, scope="local")
            thread_data = T.Buffer(1, data=thread_buffer, dtype=dtype, scope="local")
            for step in T.serial(T.ceildiv(spatial_len, warp_cnt)):
                # reduction on dst_indices
                spa_fused = T.meta_var(step * warp_cnt + T.floordiv(tid_x, threads_per_warp))
                if spa_fused < spatial_len:
                    src_indices_1 = T.meta_var(get_indices(spa_fused, src_st[:spatial_dims], src_extent[:spatial_dims]))
                    thread_data[0] = init_value
                    # load from src
                    for t in T.serial(T.ceildiv(reduction_len, threads_per_warp)):
                        red_fused = T.meta_var(t * threads_per_warp + tid_x % threads_per_warp)
                        if red_fused < reduction_len:
                            src_indices_2 = T.meta_var(get_indices(red_fused, src_st[spatial_dims:], src_extent[spatial_dims:]))
                            thread_data[0] = op_func(thread_data[0], src[*(src_indices_1 + src_indices_2)])
                    # warp reduce
                    mask = T.tvm_warp_activemask()
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 1, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 2, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 4, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 8, 32, 32))
                    thread_data[0] = op_func(thread_data[0], T.tvm_warp_shuffle_xor(mask, thread_data[0], 16, 32, 32))

                    # write result to dst_indices
                    if tid_x % threads_per_warp == 0:
                        dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                        dst[*dst_indices] = T.if_then_else(T.bool(accum), op_func(dst[*dst_indices], thread_data[0]), thread_data[0])

        T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def reduction_cuda_local_thread_packed_add_sum_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule thread-level sum reduction using packed add sum with add.f32x2 PTX instruction.

    This implementation uses packed add sum leveraging the PTX `add.rz.ftz.f32x2`
    instruction which can add two pairs of floats in parallel.

    The algorithm:
    1. Copy first 8 elements to local_sum buffer (with optional accumulator)
    2. For remaining full chunks of 8, use add_packed_f32x2 to add them to local_sum
    3. Handle remainder elements (0-7) with sequential addition
    4. Final packed add sum: reduce 8 values down to 1

    Note: Requirements are checked via predicates in register_dispatch:
    - exec_scope == "thread", src/dst scope == "local", dtype == "float32"
    - sm_version >= 100, reduction_len >= 8, dst_len == 1, src_ndim == 1
    """

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with T.thread():
            local_sum = T.alloc_buffer([8], dtype, scope="local")
            # First pass: copy first 8 elements (with optional accumulator)
            for i in T.unroll(8):
                if accum and i == 0:
                    # Include accumulator in first element
                    local_sum[i] = src[src_base + i] + dst[*dst_st]
                else:
                    local_sum[i] = src[src_base + i]

            # Process remaining full chunks of 8
            for outer in T.serial(num_full_chunks - 1):
                for j in T.unroll(4):
                    T.cuda.add_packed_f32x2(
                        local_sum[2 * j],
                        local_sum[2 * j + 1],
                        src[src_base + 8 * (outer + 1) + 2 * j],
                        src[src_base + 8 * (outer + 1) + 2 * j + 1],
                        T.address_of(local_sum[2 * j]),
                    )

            # Handle remainder elements (0 to 7)
            for i in T.serial(remainder):
                local_sum[0] = local_sum[0] + src[src_base + remainder_base + i]

            # Final packed add sum: 8 -> 4 -> 2 -> 1
            T.cuda.add_packed_f32x2(
                local_sum[0], local_sum[1],
                local_sum[2], local_sum[3],
                T.address_of(local_sum[0]),
            )
            T.cuda.add_packed_f32x2(
                local_sum[4], local_sum[5],
                local_sum[6], local_sum[7],
                T.address_of(local_sum[4]),
            )
            T.cuda.add_packed_f32x2(
                local_sum[0], local_sum[1],
                local_sum[4], local_sum[5],
                T.address_of(local_sum[0]),
            )
            dst[*dst_st] = local_sum[0] + local_sum[1]
    # fmt: on

    return impl


def reduction_cuda_local_thread_3input_maxmin_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule thread-level max/min reduction using 3-input PTX intrinsics.

    This implementation uses the PTX `max.f32`/`min.f32` instruction which can
    compare 3 values at once for better performance.

    The algorithm:
    1. Use 4 temp variables to process elements in parallel
    2. First pass: temp[i] = op(src[2*i], src[2*i+1]) for i in 0..3
    3. Loop: temp[i] = op3(temp[i], src[...], src[...]) for remaining elements
    4. Final merge: dst = op3(op(temp[0], temp[1]), temp[2], temp[3])

    Note: Requirements are checked via predicates in register_dispatch:
    - exec_scope == "thread", src/dst scope == "local", dtype == "float32"
    - sm_version >= 100, reduction_len >= 8, dst_len == 1, src_ndim == 1
    """

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    op_func = reduce_op_table[reduce_op]
    reduce3_func = T.cuda.reduce3_max_f32 if reduce_op == ReduceOpType.MAX else T.cuda.reduce3_min_f32

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with T.thread():
            temp = T.alloc_buffer([4], dtype, scope="local")
            # First pass: process first 8 elements into 4 temps
            for i in T.unroll(4):
                if accum and i == 0:
                    # Include accumulator in first temp
                    temp[i] = reduce3_func(src[src_base + 2 * i], src[src_base + 2 * i + 1], dst[*dst_st])
                else:
                    temp[i] = op_func(src[src_base + 2 * i], src[src_base + 2 * i + 1])

            # Process remaining full chunks of 8
            for outer in T.serial(num_full_chunks - 1):
                for i in T.unroll(4):
                    temp[i] = reduce3_func(
                        temp[i],
                        src[src_base + 8 * (outer + 1) + 2 * i],
                        src[src_base + 8 * (outer + 1) + 2 * i + 1],
                    )

            # Process remainder elements (0 to 7 elements)
            for i in T.serial(remainder):
                temp[0] = op_func(temp[0], src[src_base + remainder_base + i])

            # Final merge: combine 4 temps into result
            dst[*dst_st] = op_func(temp[0], temp[1])
            dst[*dst_st] = reduce3_func(dst[*dst_st], temp[2], temp[3])
    # fmt: on

    return impl


def reduction_cuda_local_thread_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule thread-level reduction operation on local memory on CUDA.

    This is the fallback implementation using simple sequential reduction.
    """

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    # basic validation checks
    if sctx.exec_scope.name != "thread":
        fail(f"unsupported exec_scope {sctx.exec_scope.name} for thread-level reduction")

    if not all(
        [
            src.scope() == "local",
            dst.scope() == "local",
            src.dtype == dst.dtype,
            sctx.is_cuda(),
        ]
    ):
        fail("unsupported scope or dtype for thread-level local reduction")

    # get region extents
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    # compute reduction length (product of all src extents)
    reduction_len = functools.reduce(operator.mul, src_extent, 1)
    dst_len = functools.reduce(operator.mul, dst_extent, 1)

    # currently only support reducing to a single element
    if dst_len != 1:
        fail("thread-level reduction currently only supports reducing to a single element")

    # only support 1D src buffer for now
    if len(src_extent) != 1:
        fail("thread-level reduction currently only supports 1D source buffer")

    # get reduce op
    op_func = reduce_op_table.get(reduce_op)
    if op_func is None:
        fail(f"unsupported reduce op: {reduce_op}")

    # get init value if not accum
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    def get_src_indices(nth, st, extent):
        """Convert linear index to multi-dimensional indices."""
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl_simple():
        with T.thread():
            if not accum:
                dst[*dst_st] = init_value
            for i in T.serial(reduction_len):
                src_indices = T.meta_var(get_src_indices(i, src_st, src_extent))
                dst[*dst_st] = op_func(dst[*dst_st], src[*src_indices])
    # fmt: on

    return impl_simple


def reduction_cuda_warp_logical_view_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    config: Dict[str, Any],
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule reduction operation on logical tensor of local memory on CUDA.

    If 'thread_reduce' is set to True in config, perform a reduction
    across threads based on the src buffer layout.

    Currently, only WGMMA layout is supported for this feature.

    Note: for now, is_tile_inner only support warp-level buffer view verification.
    Warpgroup-level buffer view and cta-level buffer view need sharding on the
    outermost level, thus not supported to be checked and verified at the moment.
    User should pass in warp-level buffer view for src and dst whenever possible.
    """

    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    # basic validation checks
    if not all(
        [
            src.scope() == "local",
            dst.scope() == "local",
            src.layout is not None,
            dst.layout is not None,
            src.dtype == dst.dtype,
            sctx.is_cuda(),
            sctx.exec_scope.name in ["warp", "warpgroup", "cta", "cluster"],
        ]
    ):
        fail("unsupported layout/scope or exec_scope for local reduction")

    # no slicing allowed
    if not all(
        [
            len(src_region) == 2 and len(dst_region) == 2,
            src_region[0].min == 0 and src_region[1].min == 0,
            dst_region[0].min == 0 and dst_region[1].min == 0,
            src_region[0].extent == src.shape[0] and src_region[1].extent == src.shape[1],
            dst_region[0].extent == dst.shape[0] and dst_region[1].extent == dst.shape[1],
        ]
    ):
        fail("slicing not supported for local reduction; expect full buffers")

    # check for WGMMA layout
    if any(
        [
            len(src.shape) != 2,
            len(dst.shape) != 2,
            src.shape[0] != 16,
            not (src.shape[1] % 8 == 0 or src.shape[1] == 4),
            dst.shape[0] != 16,
            dst.shape[1] != 4,
        ]
    ):
        fail("shape/layout unsupported for local reduction (expect 16xN to 16x4)")

    if src.layout.is_swizzle() or dst.layout.is_swizzle():
        fail("swizzle layout unsupported for local reduction")

    atom = T.TileLayout(shard=([1, 2], [2, 1]))
    warp_layout = T.TileLayout(shard=([8, 4], [4@laneid, 1@laneid]))
    warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
    red_atom = T.TileLayout(shard=([1, 1], [1, 1]))
    red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))

    shuffle = T.bool(config.get("thread_reduce", False))

    # get reduce op
    op_func = reduce_op_table.get(reduce_op)
    if op_func is None:
        fail(f"unsupported reduce op: {reduce_op}")

    # get init value if not accum
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    # case 0. shuffle-only, no within-thread reduction
    # provided that src and dst are same-shaped buffer
    if src.shape[1] == 4:
        if red_warp_atom.is_tile_inner(src.layout.canonicalize(), (64,), (32,)) is None:
            fail("src layout not compatible with ROW_RED tile for shuffle-only reduction")
        if red_warp_atom.is_tile_inner(dst.layout.canonicalize(), (64,), (32,)) is None:
            fail("dst layout not compatible with ROW_RED tile for shuffle-only reduction")
        if shuffle is False:
            fail("thread_reduce (shuffle) must be enabled for this reduction case")

        num_rows = 2
        local_shape = (num_rows,)
        is_same_buffer = src.same_as(dst)

        # fmt: off
        @T.prim_func(tirx=True, check_well_formed=False)
        def impl_shuffle_only():
            with T.thread():
                src_local = src.storage(*local_shape)
                dst_local = dst.storage(*local_shape)
                for i in T.serial(num_rows):
                    if not is_same_buffer:
                        dst_local[i] = src_local[i]
                    row_var = T.meta_var(dst_local[i])
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 2, 32, 32))
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 1, 32, 32))
        # fmt: on

        return impl_shuffle_only

    # case 1. normal reduction
    # src and dst are different-shaped buffer
    src_tile_outer = warp_atom.is_tile_inner(src.layout, src.shape, [8, 8])
    if src_tile_outer is None:
        fail("src layout not compatible with WGMMA tile for reduction")
    if red_warp_atom.is_tile_inner(dst.layout.canonicalize(), (64,), (32,)) is None:
        fail("dst layout not compatible with ROW_RED tile for reduction")

    num_rows = 2
    src_local_shape = (num_rows, src_tile_outer.size())
    dst_local_shape = (num_rows,)

    # fmt: off
    @T.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with T.thread():
            src_local = src.storage(*src_local_shape)
            dst_local = dst.storage(*dst_local_shape)
            for i in T.serial(num_rows):
                # reduce within threads
                if not accum:
                    dst_local[i] = init_value
                row_var = T.meta_var(dst_local[i])
                for j in T.serial(src_local_shape[1]):
                    dst_local[i] = op_func(row_var, src_local[i, j])
                # if shuffle is True, perform shuffling among threads of 4
                if shuffle:
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 2, 32, 32))
                    dst_local[i] = op_func(row_var, T.tvm_warp_shuffle_xor(0xFFFFFFFF, row_var, 1, 32, 32))
    # fmt: on

    return impl


def reduction_cuda_impl(
    op: OpCall,
    reduce_op: ReduceOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Dispatch to shared memory scheduler or logical tensor of local memory scheduler
    based on the storage scope of buffers.
    """

    # FIXME: correctly handle the axes field
    dst_buffer_region, src_buffer_region, axes, accum = op.args
    config = op.config

    if src_buffer_region.buffer.scope().startswith("shared"):
        return reduction_cuda_shared_nd_sync_cta_impl(
            dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
        )
    elif src_buffer_region.buffer.scope() == "local":
        if sctx.exec_scope.name == "thread":
            return reduction_cuda_local_thread_impl(
                dst_buffer_region, src_buffer_region, accum, reduce_op, sctx
            )
        else:
            return reduction_cuda_warp_logical_view_impl(
                dst_buffer_region, src_buffer_region, accum, reduce_op, config, sctx
            )
    fail("unsupported buffer scope for reduction")


# ---------------------------------------------------------------------------
# Common predicates for optimized thread-level local reduction (sm_100a+)
# ---------------------------------------------------------------------------

_optimized_local_reduction_predicates = [
    predicate("exec_scope", _exec_scope_ok, expected_scopes=["thread"]),
    predicate("local_scope", _local_scope_match),
    predicate("dst_len", _dst_len_ok, expected_len=1),
    predicate("src_ndim", _src_ndim_ok, expected_ndim=1),
    predicate("dtype", _dtype_ok, expected_dtype="float32"),
    predicate("sm_version", _sm_version_ok, min_version=100),
    predicate("reduction_len", _reduction_len_ok, min_len=8),
]


# ---------------------------------------------------------------------------
# Register reduction schedules (sum, max, min)
# ---------------------------------------------------------------------------

# Optimized implementations for sm_100a+ (packed_add_sum for SUM, 3-input for MAX/MIN)
_optimized_impl_table = {
    ReduceOpType.SUM: ("packed_add_sum", reduction_cuda_local_thread_packed_add_sum_impl),
    ReduceOpType.MAX: ("3input_maxmin", reduction_cuda_local_thread_3input_maxmin_impl),
    ReduceOpType.MIN: ("3input_maxmin", reduction_cuda_local_thread_3input_maxmin_impl),
}

for op_name, op_type in [("sum", ReduceOpType.SUM), ("max", ReduceOpType.MAX), ("min", ReduceOpType.MIN)]:
    variant_name, optimized_impl = _optimized_impl_table[op_type]

    # Register optimized dispatch (sm_100a+, float32, thread-level local reduction)
    @register_dispatch(
        op_name, "cuda", variant=variant_name, priority=10,
        when=_optimized_local_reduction_predicates,
    )
    def _optimized_dispatch(
        op: OpCall, sctx: ScheduleContext, _impl=optimized_impl, _op_type=op_type
    ) -> PrimFunc:
        dst_buffer_region, src_buffer_region, _, accum = op.args
        return _impl(dst_buffer_region, src_buffer_region, accum, _op_type, sctx)

    # Register default fallback dispatch
    @register_dispatch(op_name, "cuda", variant="default", priority=0)
    def _default_dispatch(
        op: OpCall, sctx: ScheduleContext, _op_type=op_type
    ) -> PrimFunc:
        return reduction_cuda_impl(op, _op_type, sctx)
