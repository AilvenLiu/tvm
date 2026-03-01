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

import functools
import operator
from collections.abc import Callable
from enum import Enum
from functools import wraps

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tir import Buffer, BufferRegion, PrimExpr, PrimFunc
from tvm.tir.exec_scope import ExecScopeSlice
from tvm.tir.stmt import OpCall
from tvm.tirx.op_schedule import ScheduleContext, fail


def get_st_extent(buffer_region: BufferRegion):
    """Get the start and extent of a buffer region."""
    region = buffer_region.region
    return [r.min for r in region], [r.extent for r in region]


def get_indices(nth, start, extent):
    """Convert a fused index into multi-dimensional indices."""
    assert len(start) == len(extent)
    if len(start) == 1:
        return [start[0] + nth]
    relative = []
    for e in reversed(extent):
        relative.append(nth % e)
        nth //= e
    return [r + s for r, s in zip(reversed(relative), start)]


def target_cuda(fn: Callable[[OpCall, ScheduleContext], PrimFunc | None]):
    """Decorator that ensures the function is only executed for CUDA targets."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sctx = kwargs.get("sctx", None)
        if sctx is None:
            assert len(args) == 2 and isinstance(args[1], ScheduleContext), (
                "The target_cuda() needs to annotate a function with signature (op_call, sctx)"
            )
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

    @Tx.prim_func(tirx=True, check_well_formed=False)
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

    inner_impl : Tx.inline
        The inner implementation.

    macro : bool
        Whether return a macro or a prim_func.

    Returns
    -------
    thread_selector : a macro or a prim_func
        The inner implementation wrapped by a thread selector in the given exec scope.
    """
    assert not isinstance(inner_impl, PrimFunc), (
        "inner_impl should be a macro rather than a PrimFunc"
    )

    exec_scope = sctx.exec_scope
    tx = sctx.launch_params["threadIdx.x"]
    # currently don't support multi-dimensional thread binding
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    if exec_scope.name == "cta":
        assert not isinstance(exec_scope, ExecScopeSlice)

        @Tx.inline()
        def impl():
            with Tx.thread()[0:1]:
                inner_impl()

        return macro_or_prim_func(impl, need_macro=macro)

    elif exec_scope.name == "warp":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warps
            warp_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @Tx.inline()
            def impl():
                with Tx.warp()[tuple(warp_selector)]:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warp
            @Tx.inline()
            def impl():
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    elif exec_scope.name == "warpgroup":
        if isinstance(exec_scope, ExecScopeSlice) and not isinstance(exec_scope.slices, PrimExpr):
            # slice of multiple warpgroups
            warpgroup_selector = [slice(r.min, r.min + 1) for r in exec_scope.slices]

            @Tx.inline()
            def impl():
                with Tx.warpgroup()[tuple(warpgroup_selector)]:
                    with Tx.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
        else:
            # a single warpgroup
            @Tx.inline()
            def impl():
                with Tx.warp(parent="warpgroup")[(tx // 32) % 4 == 0]:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        inner_impl()

            return macro_or_prim_func(impl, need_macro=macro)
    elif exec_scope.name == "thread":
        # already a single thread, just return the inner_impl
        return macro_or_prim_func(inner_impl, need_macro=macro)
    else:
        raise ValueError(
            f"Currently exec scope {exec_scope} is not supported to select a single thread within"
        )


def single_thread(op_call: OpCall, sctx: ScheduleContext) -> bool:
    return (
        sctx.exec_scope.name == "thread"
        and isinstance(sctx.exec_scope, ExecScopeSlice)
        and (
            isinstance(sctx.exec_scope.slices, PrimExpr)
            or functools.reduce(operator.mul, [s.extent for s in sctx.exec_scope.slices], 1) == 1
        )
    )


################################################################################
# Gemm operations related utilities
# Reused by sync and async gemm pipelines
################################################################################


def validate_gemm_op(op_call: OpCall, sctx: ScheduleContext) -> bool:
    """Sanity check for gemm op"""
    C_buffer_region, A_buffer_region, B_buffer_region = op_call.args[:3]
    C: Buffer = C_buffer_region.buffer
    A: Buffer = A_buffer_region.buffer
    B: Buffer = B_buffer_region.buffer
    if not (C.layout and A.layout and B.layout and A.dtype == B.dtype):
        return False
    # Extract regions and validate dimensions
    analyzer = Analyzer()
    C_region, A_region, B_region = (
        C_buffer_region.region,
        A_buffer_region.region,
        B_buffer_region.region,
    )
    # Extract extents and validate non-unit dimensions match
    transA, transB = op_call.args[3:5]
    C_extent_ = [r.extent for r in C_region if r.extent != 1]
    A_extent_ = [r.extent for r in A_region if r.extent != 1]
    B_extent_ = [r.extent for r in B_region if r.extent != 1]
    assert len(C_extent_) == len(A_extent_) == len(B_extent_) == 2, (
        "Only 2D C, A, B are supported for gemm"
    )
    if transA:
        A_extent_ = [A_extent_[1], A_extent_[0]]
    if transB:
        B_extent_ = [B_extent_[1], B_extent_[0]]
    # C: MxN, A: MxK, B: NxK
    if not all(
        [
            analyzer.can_prove_equal(C_extent_[0], A_extent_[0]),
            analyzer.can_prove_equal(C_extent_[1], B_extent_[0]),
            analyzer.can_prove_equal(A_extent_[1], B_extent_[1]),
        ]
    ):
        return False
    return True


################################################################################
# Copy operations related utilities
# Reused by sync and async copy pipelines
################################################################################


class CopyInstType(Enum):
    """Enumeration of instruction types for memory operations."""

    NORMAL = 0
    CP_ASYNC = 1


def validate_copy_op(
    op_call: OpCall,
    sctx: ScheduleContext,  # pylint: disable=unused-argument
) -> bool:
    """Sanity check for copy op"""
    dst_buffer_region, src_buffer_region = op_call.args[:2]
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
    vec_candidates: list[int],
    thread_cnt=1,
) -> int | None:
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
    DataType(src.dtype).bits  # in bits
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
                src.shape[-1] if len(src.shape) > 1 else 0,
                dst.shape[-1] if len(dst.shape) > 1 else 0,
                src_extent[-1],
                dst_extent[-1],
                n_elements // thread_cnt,
            ]
        ):
            return vec_len
    else:
        return None


def copy_vec_load_impl(
    op_call: OpCall,
    sctx: ScheduleContext,
    inst_type: CopyInstType,
) -> PrimFunc | None:
    """Schedule copy operation between global and local/shared memory on CUDA across a CTA/thread.
    The implementation tries to vectorize the copy operation and parallelize over
    threads in a CTA/using a single thread.
    """
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (
        (src.scope() == "global" and dst.scope().startswith("shared"))
        or (src.scope().startswith("shared") and dst.scope() == "global")
        or (src.scope() == "global" and dst.scope() == "local")
        or (src.scope() == "local" and dst.scope() == "global")
        or (src.scope().startswith("shared") and dst.scope() == "local")
        or (dst.scope().startswith("shared") and src.scope() == "local")
    ):
        fail(f"unsupported memory scopes src={src.scope()} dst={dst.scope()}")

    # Thread and vectorization setup
    if sctx.exec_scope.name == "cta":
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
    elif sctx.exec_scope.name == "thread":
        tx = 1
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")

    elem_size = DataType(src.dtype).bits  # in bits
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
        vec_len = get_vec_len(
            dst_buffer_region,
            src_buffer_region,
            [128 // elem_size, 64 // elem_size, 32 // elem_size, 1],
            thread_cnt=tx,
        )
    if vec_len is None:
        fail("no valid vector length; check alignment/extents/thread-count")

    # cp-size (the size of data in bytes) can only be 4, 8 and 16 for cp.async
    if inst_type == CopyInstType.CP_ASYNC:
        cp_size = vec_len * elem_size // 8  # in bytes
        if cp_size not in [4, 8, 16]:
            fail("invalid cp.async cp_size; expected 4, 8 or 16 bytes")

    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    n_elements = functools.reduce(operator.mul, src_extent, 1)

    if sctx.exec_scope.name == "cta":
        # fmt: off
        @Tx.prim_func(tirx=True)
        def impl():
            """Implement copy operation with vectorized loads/stores."""
            for s in Tx.serial(0, n_elements // (tx * vec_len)):
                for tid_x in Tx.thread_binding(tx, "threadIdx.x"):
                    if inst_type == CopyInstType.NORMAL:
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var((s * tx + tid_x) * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            dst[tuple(dst_indices)] = src[tuple(src_indices)]
                    elif inst_type == CopyInstType.CP_ASYNC:
                        fused = Tx.meta_var((s * tx + tid_x) * vec_len)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                        Tx.evaluate(Tx.ptx.cp_async(dst.ptr_to(dst_indices), src.ptr_to(src_indices), cp_size))  # noqa: E501
            if dst.scope().startswith("shared") and inst_type == CopyInstType.NORMAL:
                Tx.tvm_storage_sync("shared")
        # fmt: on
    elif sctx.exec_scope.name == "thread":
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // (vec_len)):
                if inst_type == CopyInstType.NORMAL:
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                        dst[tuple(dst_indices)] = src[tuple(src_indices)]
                elif inst_type == CopyInstType.CP_ASYNC:
                    fused = Tx.meta_var(s * vec_len)
                    dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                    Tx.evaluate(Tx.ptx.cp_async(dst.ptr_to(dst_indices), src.ptr_to(src_indices), cp_size))  # noqa: E501
        # fmt: on
    else:
        fail(f"unsupported exec_scope {sctx.exec_scope.name}")
    return impl
