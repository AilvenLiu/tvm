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
from typing import Optional, Callable, List
import functools
import operator
from functools import wraps
from tvm.script import tir as T
from tvm.tirp.op_schedule import ScheduleContext
from tvm.runtime import DataType
from tvm.arith.analyzer import Analyzer
from tvm.tir import BufferRegion, PrimFunc, Buffer, PrimExpr
from tvm.tir.exec_scope import ExecScopeSlice
from tvm.tir.stmt import OpCall


def get_st_extent(buffer_region: BufferRegion):
    """Get the start and extent of a buffer region."""
    region = buffer_region.region
    return [r.min for r in region], [r.extent for r in region]


def get_indices(nth, start, extent):
    """Convert a fused index into multi-dimensional indices."""
    relative = []
    for e in reversed(extent):
        relative.append(nth % e)
        nth //= e
    return [r + s for r, s in zip(reversed(relative), start)]


def target_cuda(fn: Callable[[OpCall, ScheduleContext], Optional[PrimFunc]]):
    """Decorator that ensures the function is only executed for CUDA targets."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sctx = kwargs.get("sctx", None)
        if sctx is None:
            assert len(args) == 2 and isinstance(
                args[1], ScheduleContext
            ), "The target_cuda() needs to annotate a function with signature (op_call, sctx)"
            sctx = args[1]
        if not sctx.is_cuda():
            return None
        return fn(*args, **kwargs)

    return wrapper


################################################################################
# Execscope utilities
################################################################################


def macro_or_prim_func(macro: Callable, need_macro: bool = False) -> Callable:
    """Convert a macro to a prim_func."""

    if need_macro:
        return macro

    @T.prim_func(tirp=True, check_well_formed=False)
    def func():
        macro()

    return func


def thread_selector(sctx: ScheduleContext, inner_impl, macro=False) -> Callable:
    """Select a single thread from the given exec scope.

    For a certain scope, it should return a deterministic thread index, i.e. the
    same thread is elected every time. This is vital for the correctness of many
    synchronization primitives. PTX's elect_sync() is one example.

    Parameters
    ----------
    sctx : ScheduleContext
        The schedule context.

    inner_impl : T.macro
        The inner implementation.

    macro : bool
        Whether return a macro or a prim_func.

    Returns
    -------
    thread_selector : a macro or a prim_func
        The inner implementation wrapped by a thread selector in the given exec scope.
    """
    assert not isinstance(
        inner_impl, PrimFunc
    ), "inner_impl should be a macro rather than a PrimFunc"

    exec_scope = sctx.exec_scope
    tx = sctx.launch_params["threadIdx.x"]
    # currently don't support multi-dimensional thread binding
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if exec_scope.name == "cta":
        assert not isinstance(exec_scope, ExecScopeSlice)

        @T.macro()
        def impl():
            with T.thread()[0:1]:
                inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)

    elif exec_scope.name == "warp":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warps
            warp_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @T.macro()
            def impl():
                with T.warp()[*warp_selector]:
                    with T.thread(parent="warp")[T.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warp
            @T.macro()
            def impl():
                with T.thread(parent="warp")[T.ptx.elect_sync()]:
                    inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    elif exec_scope.name == "warpgroup":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warpgroups
            warpgroup_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @T.macro()
            def impl():
                with T.warpgroup()[*warpgroup_selector]:
                    with T.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                        with T.thread(parent="warp")[T.ptx.elect_sync()]:
                            inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warpgroup
            @T.macro()
            def impl():
                with T.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                    with T.thread(parent="warp")[T.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    else:
        raise ValueError(
            f"Currently exec scope {exec_scope} is not supported to select a single thread within"
        )


################################################################################
# Copy operations related utilities
# Reused by sync and async copy pipelines
################################################################################


class CopyInstType(Enum):
    """Enumeration of instruction types for memory operations."""

    NORMAL = 0
    CP_ASYNC = 1


def validate_copy_op(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,  # pylint: disable=unused-argument
) -> bool:
    """Sanity check for copy op"""
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (src.layout and dst.layout and src.dtype == dst.dtype):
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


def get_vec_len(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    vec_candidates: List[int],
    thread_cnt=1,
) -> Optional[int]:
    """Get the vector length for the copy operation."""

    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer
    if not (src.layout.is_trivial() and dst.layout.is_trivial()):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    elem_size = DataType(src.dtype).bits  # in bits
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    if n_elements % thread_cnt != 0:
        return None

    # Find valid vector length
    for vec_len in vec_candidates:
        if vec_len > 0 and all(
            analyzer.can_prove_equal(x % vec_len, 0)
            for x in [
                src_st[-1],
                dst_st[-1],
                src.shape[-1],
                dst.shape[-1],
                src_extent[-1],
                dst_extent[-1],
                n_elements // thread_cnt,
            ]
        ):
            return vec_len
    else:
        return None


def copy_g2s_s2g_cta_vec_load_impl(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    sctx: ScheduleContext,
    inst_type: CopyInstType,
) -> Optional[PrimFunc]:
    """Schedule copy operation between global and shared memory on CUDA across a CTA.
    The implementation tries to vectorize the copy operation and parallelize over
    threads in a CTA.
    """

    # Sanity checks
    if sctx.exec_scope.name != "cta":
        return None

    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (
        (src.scope() == "global" and dst.scope().startswith("shared"))
        or (src.scope().startswith("shared") and dst.scope() == "global")
    ):
        return None

    # Thread and vectorization setup
    tx = sctx.launch_params["threadIdx.x"].dom.extent
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    elem_size = DataType(src.dtype).bits  # in bits
    vec_len = get_vec_len(
        dst_buffer_region,
        src_buffer_region,
        [128 // elem_size, 64 // elem_size, 32 // elem_size, 1],
        thread_cnt=tx,
    )
    if vec_len is None:
        return None

    # cp-size (the size of data in bytes) can only be 4, 8 and 16 for cp.async
    if inst_type == CopyInstType.CP_ASYNC:
        cp_size = vec_len * elem_size // 8  # in bytes
        if cp_size not in [4, 8, 16]:
            return None

    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    # fmt: off
    @T.prim_func(tirp=True)
    def impl():
        """Implement copy operation with vectorized loads/stores."""
        for s in T.serial(0, n_elements // (tx * vec_len)):
            for tid_x in T.thread_binding(tx, "threadIdx.x"):
                if inst_type == CopyInstType.NORMAL:
                    for vec in T.vectorized(vec_len):
                        fused = T.meta_var((s * tx + tid_x) * vec_len + vec)
                        dst_indices = T.meta_var(get_indices(fused, dst_st, dst_extent))
                        src_indices = T.meta_var(get_indices(fused, src_st, src_extent))
                        dst[*dst_indices] = src[*src_indices]
                elif inst_type == CopyInstType.CP_ASYNC:
                    fused = T.meta_var((s * tx + tid_x) * vec_len)
                    dst_indices = T.meta_var(get_indices(fused, dst_st, dst_extent))
                    src_indices = T.meta_var(get_indices(fused, src_st, src_extent))
                    T.evaluate(T.ptx.cp_async(dst.dtype, dst.data, dst.offset_of_p([*dst_indices]),
                                              src.data, src.offset_of_p([*src_indices]), cp_size))
        if dst.scope().startswith("shared") and inst_type == CopyInstType.NORMAL:
            T.tvm_storage_sync("shared")
    # fmt: on

    return impl
