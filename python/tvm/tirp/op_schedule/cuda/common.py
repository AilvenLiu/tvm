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

"""Common utilities for operator scheduling."""

from enum import Enum
from typing import Optional, Union
import functools
import operator

from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext
from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.tir import BufferRegion, PrimFunc, Buffer
from tvm.tir.expr import FloatImm
from ..common import MapOpType


class InstType(Enum):
    """Enumeration of instruction types for memory operations."""

    NORMAL = 0
    CP_ASYNC = 1


def copy_cuda_g2s_s2g_2d_cta_vec_load_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
    inst_type: InstType,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not all(
        [
            src.layout and dst.layout,
            src.dtype == dst.dtype,
            src.layout.is_trivial() or dst.layout.is_trivial(),
            (src.scope() == "global" and dst.scope().startswith("shared"))
            or (src.scope().startswith("shared") and dst.scope() == "global"),
        ]
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]

    # Validate non-unit dimensions match
    src_extent_ = [e for e in src_extent if e != 1]
    dst_extent_ = [e for e in dst_extent if e != 1]
    if not (
        len(src_extent_) == len(dst_extent_)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_))
    ):
        return None

    # Thread and vectorization setup
    tx = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    elem_size = DataType(src.dtype).bits  # in bits
    n_elements = functools.reduce(operator.mul, src_extent, 1)

    # Find valid vector length
    if n_elements % tx != 0:
        return None
    for vec_len in [128 // elem_size, 64 // elem_size, 32 // elem_size, 1]:
        if vec_len > 0 and all(
            analyzer.can_prove_equal(x % vec_len, 0)
            for x in [
                src_st[-1],
                dst_st[-1],
                src.shape[-1],
                dst.shape[-1],
                src_extent[-1],
                dst_extent[-1],
                n_elements // tx,
            ]
        ):
            break
    else:
        return None
    # cp-size (the size of data in bytes) can only be 4, 8 and 16 for cp.async
    if inst_type == InstType.CP_ASYNC:
        cp_size = vec_len * elem_size // 8  # in bytes
        if cp_size not in [4, 8, 16]:
            return None

    def get_indices(st, extent, fused):
        indices = []
        product = 1
        for i in reversed(range(len(extent))):
            indices.append(st[i] + (fused // product) % extent[i])
            product *= extent[i]
        return reversed(indices)

    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        """Implement copy operation with vectorized loads/stores."""
        for s in T.serial(0, n_elements // (tx * vec_len)):
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                if inst_type == InstType.NORMAL:
                    for vec in T.vectorized(vec_len):
                        fused = T.meta_var((s * tx + tid_x) * vec_len + vec)
                        dst_indices = T.meta_var(get_indices(dst_st, dst_extent, fused))
                        src_indices = T.meta_var(get_indices(src_st, src_extent, fused))
                        dst[*dst_indices] = src[*src_indices]
                elif inst_type == InstType.CP_ASYNC:
                    fused = T.meta_var((s * tx + tid_x) * vec_len)
                    dst_indices = T.meta_var(get_indices(dst_st, dst_extent, fused))
                    src_indices = T.meta_var(get_indices(src_st, src_extent, fused))
                    T.evaluate(T.ptx_cp_async(dst.dtype, dst.data, dst.offset_of_p([*dst_indices]),
                                              src.data, src.offset_of_p([*src_indices]), cp_size))
        if dst.scope().startswith("shared") and inst_type == InstType.NORMAL:
            T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def reduction_cuda_shared_nd_sync_cta_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: str,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule warp-level tree-reduction operation in shared memory on CUDA.

    Support reduction along the last D dimensions.
    Warp partition follows the rule below:
        For src tensor [s1, s2, ..., r1, r2, ...], where si are spatial axes and ri are reduction axes.
        Use one warp (32 threads) for each si for reduction.
    """
    # Basic validation checks
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None

    if reduce_op not in ["add"]:
        return None

    thread_cnt = sctx.launch_params["threadIdx.x"]
    threads_per_warp = 32
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if not (thread_cnt >= threads_per_warp and thread_cnt % threads_per_warp == 0):
        return None

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
        return None

    analyzer = Analyzer()
    spatial_dims = -1

    if len(src_extent) == len(dst_extent):
        for i in range(len(dst_extent)):
            if src_extent[i] != dst_extent[i]:
                if dst_extent[i] != 1:
                    return None
                if not functools.reduce(operator.mul, dst_extent[i:], 1) == 1:
                    return None
                else:
                    spatial_dims = i
                    break
        if spatial_dims == -1:
            return None

    else:
        spatial_dims = len(dst_extent)
        if not all(
            analyzer.can_prove_equal(s, d) for s, d in zip(src_extent[:spatial_dims], dst_extent)
        ):
            return None

    assert spatial_dims > 0 and spatial_dims < len(src_extent)

    spatial_len = functools.reduce(operator.mul, src_extent[:spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, src_extent[spatial_dims:], 1)

    def get_indices(nth, st, extent):
        # example: st: [0, 1], extent: [32, 16], nth: 10
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    # fmt: off
    @T.prim_func(tirp=True)
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
                    thread_data[0] = 0.0
                    # load from src
                    for t in T.serial(T.ceildiv(reduction_len, threads_per_warp)):
                        red_fused = T.meta_var(t * threads_per_warp + tid_x % threads_per_warp)
                        if red_fused < reduction_len:
                            src_indices_2 = T.meta_var(get_indices(red_fused, src_st[spatial_dims:], src_extent[spatial_dims:]))
                            thread_data[0] += src[*(src_indices_1 + src_indices_2)]
                    # warp reduce
                    mask = T.tvm_warp_activemask()
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 1, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 2, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 4, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 8, 32, 32)
                    thread_data[0] += T.tvm_warp_shuffle_xor(mask, thread_data[0], 16, 32, 32)

                    # write result to dst_indices
                    if tid_x % threads_per_warp == 0:
                        dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                        dst[*dst_indices] = T.if_then_else(T.bool(accum), dst[*dst_indices] + thread_data[0], thread_data[0])

        T.tvm_storage_sync("shared")
    # fmt: on

    return impl


def unary_map_cuda_shared_nd_sync_cta_impl(
    _dst: BufferRegion,
    _src: BufferRegion,
    unary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule unary map operation on CUDA in shared memory.

    Dst and src regions must be the same shape.
    """

    # Basic validation checks
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None
    if not (isinstance(unary_op, MapOpType) and unary_op in [MapOpType.ZERO, MapOpType.SQRT]):
        return None

    dst, src = _dst.buffer, _src.buffer
    dst_region, src_region = _dst.region, _src.region
    dtype = dst.dtype

    dst_st = [r.min for r in dst_region]
    src_st = [r.min for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    src_extent = [r.extent for r in src_region]

    if not all(
        [
            dst.layout and src.layout,
            dst.layout.is_trivial() and src.layout.is_trivial(),
            dst.scope().startswith("shared"),
            src.scope().startswith("shared"),
            src.dtype == dtype,
        ]
    ):
        return None

    # Shape checks
    analyzer = Analyzer()
    NUM_ELEMENTS = functools.reduce(operator.mul, src_extent, 1)
    if not all(
        [
            len(src_extent) == len(dst_extent),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src_extent, dst_extent)),
        ]
    ):
        return None

    # hardware parameters
    thread_cnt = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    def get_lambda_func(op):
        if op == MapOpType.ZERO:
            return lambda x: 0.0
        if op == MapOpType.SQRT:
            return lambda x: T.sqrt(x)
        raise NotImplementedError(f"Unsupported unary op: {op}")

    f = get_lambda_func(unary_op)

    def get_indices(nth, st, extent):
        # example: st: [0, 1], extent: [32, 16], nth: 10
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    def get_impl():
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
                for itr in T.serial(T.ceildiv(NUM_ELEMENTS, thread_cnt)):
                    idx_fused = T.meta_var(itr * thread_cnt + tid_x)
                    if idx_fused < NUM_ELEMENTS:
                        idx_dst = T.meta_var(get_indices(idx_fused, dst_st, dst_extent))
                        idx_src = T.meta_var(get_indices(idx_fused, src_st, src_extent))
                        dst[*idx_dst] = T.Cast(dtype, f(src[*idx_src]))
            T.tvm_storage_sync("shared")
        # fmt: on
        return impl

    return get_impl()


def binary_map_cuda_shared_nd_sync_cta_impl(
    _dst: BufferRegion,
    _src1: Union[BufferRegion, FloatImm],
    _src2: Union[BufferRegion, FloatImm],
    binary_op: MapOpType,
    sctx: ScheduleContext,
) -> Optional[PrimFunc]:
    """Schedule binary map operation on CUDA in shared memory.

    For commutative ops like ADD and MUL, support at most one of _src1 and _src2 to be FloatImm,
      and if both are buffer regions support numpy-style broadcasting.
    For non-commutative ops like SUB and FDIV, only support _src2 to be FloatImm, and if both
      are buffer regions only support broadcasting _src2 to _src1.
    """

    # Basic validation checks
    if not (sctx.is_cuda() and sctx.exec_scope.name == "cta"):
        return None
    if not (
        isinstance(binary_op, MapOpType)
        and binary_op in [MapOpType.ADD, MapOpType.SUB, MapOpType.MUL, MapOpType.FDIV]
    ):
        return None

    CONST = None

    # Type checks
    if isinstance(_src1, FloatImm) and isinstance(_src2, FloatImm):
        return None
    if isinstance(_src1, FloatImm):
        if binary_op not in [MapOpType.ADD, MapOpType.MUL]:
            return None
        _src1, _src2 = _src2, _src1
    if isinstance(_src2, FloatImm):
        CONST = _src2

    # Basic region checks
    dst, src1, src2 = _dst.buffer, _src1.buffer, None if CONST is not None else _src2.buffer
    dst_region, src1_region, src2_region = (
        _dst.region,
        _src1.region,
        None if CONST is not None else _src2.region,
    )
    dtype = dst.dtype

    dst_st = [r.min for r in dst_region]
    src1_st = [r.min for r in src1_region]
    src2_st = [r.min for r in src2_region] if src2_region else None
    dst_extent = [r.extent for r in dst_region]
    src1_extent = [r.extent for r in src1_region]
    src2_extent = [r.extent for r in src2_region] if src2_region else None

    if not all(
        [
            dst.layout and src1.layout and (src2.layout if src2 else True),
            dst.layout.is_trivial()
            and src1.layout.is_trivial()
            and (src2.layout.is_trivial() if src2 else True),
            dst.scope().startswith("shared"),
            src1.scope().startswith("shared"),
            (src2.scope().startswith("shared") if src2 else True),
            src1.dtype == dtype and (src2.dtype == dtype if src2 else CONST.dtype == dtype),
        ]
    ):
        return None

    # Switch broadcasting
    analyzer = Analyzer()
    NUM_ELEMENTS = functools.reduce(operator.mul, dst_extent, 1)

    if CONST is None:
        src2_num = functools.reduce(operator.mul, src2_extent, 1)
        if NUM_ELEMENTS < src2_num:
            if binary_op not in [MapOpType.ADD, MapOpType.MUL]:
                return None
            (
                _src1,
                _src2,
                src1,
                src2,
                src1_region,
                src2_region,
                src1_st,
                src2_st,
                src1_extent,
                src2_extent,
            ) = (
                _src2,
                _src1,
                src2,
                src1,
                src2_region,
                src1_region,
                src2_st,
                src1_st,
                src2_extent,
                src1_extent,
            )

    # Check dst and src1 have the same shape
    dst_extent_ = [e for e in dst_extent if e != 1]
    src1_extent_ = [e for e in src1_extent if e != 1]
    if not all(
        [
            len(src1_extent_) == len(dst_extent_),
            all(analyzer.can_prove_equal(s, d) for s, d in zip(src1_extent_, dst_extent_)),
        ]
    ):
        return None

    # Check src2 is broadcastable to src1
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src2_extent[-i] != 1 and src2_extent[-i] != src1_extent[-i]:
                return None

    # hardware parameters
    thread_cnt = sctx.launch_params["threadIdx.x"]
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    def get_lambda_func(op):
        if op == MapOpType.ADD:
            return lambda a, b: a + b
        if op == MapOpType.SUB:
            return lambda a, b: a - b
        if op == MapOpType.MUL:
            return lambda a, b: a * b
        if op == MapOpType.FDIV:
            return lambda a, b: a / b
        raise NotImplementedError(f"Unsupported binary op: {op}")

    f = get_lambda_func(binary_op)

    def get_indices(nth, st, extent):
        # example: st: [0, 1], extent: [32, 16], nth: 10
        relative_idx = []
        for e in reversed(extent):
            relative_idx.append(nth % e)
            nth //= e
        return [r + s for r, s in zip(reversed(relative_idx), st)]

    def get_const_impl():
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
                for itr in T.serial(T.ceildiv(NUM_ELEMENTS, thread_cnt)):
                    idx_fused = T.meta_var(itr * thread_cnt + tid_x)
                    if idx_fused < NUM_ELEMENTS:
                        idx_dst = T.meta_var(get_indices(idx_fused, dst_st, dst_extent))
                        idx_src1 = T.meta_var(get_indices(idx_fused, src1_st, src1_extent))
                        dst[*idx_dst] = f(src1[*idx_src1], CONST)
            T.tvm_storage_sync("shared")
        # fmt: on
        return impl

    if CONST is not None:
        return get_const_impl()

    def get_indices_zero_out(indices):
        # comp src2 indices based on src1 indices
        src2_indices = []
        len_diff = len(src1_extent) - len(src2_extent)
        for i in range(len(src2_extent)):
            offset = indices[i + len_diff] - src1_st[i + len_diff]
            src2_indices.append(offset + src2_st[i] if src2_extent[i] != 1 else src2_st[i])
        return src2_indices

    def get_broadcast_impl():
        # fmt: off
        @T.prim_func(tirp=True)
        def impl():
            for tid_x in T.thread_binding(thread_cnt, "threadIdx.x"):
                for itr in T.serial(T.ceildiv(NUM_ELEMENTS, thread_cnt)):
                    idx_fused = T.meta_var(itr * thread_cnt + tid_x)
                    if idx_fused < NUM_ELEMENTS:
                        idx_dst = T.meta_var(get_indices(idx_fused, dst_st, dst_extent))
                        idx_src1 = T.meta_var(get_indices(idx_fused, src1_st, src1_extent))
                        idx_src2 = T.meta_var(get_indices_zero_out(idx_src1))
                        dst[*idx_dst] = f(src1[*idx_src1], src2[*idx_src2])
            T.tvm_storage_sync("shared")
        # fmt: on
        return impl

    return get_broadcast_impl()
